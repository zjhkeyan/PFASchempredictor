from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from rdkit import Chem, RDLogger

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
TABPFN_SRC = Path(os.environ.get("TABPFN_SRC", PACKAGE_ROOT / "code" / "TabPFN" / "src")).resolve()
if str(TABPFN_SRC) not in sys.path:
    sys.path.insert(0, str(TABPFN_SRC))

from tabpfn.finetuning.finetuned_regressor import FinetunedTabPFNRegressor  # noqa: E402

from src.build_oecd2021_pfas_6endpoint_wide import oecd2021_pfas  # noqa: E402
from src.chemistry import standardize_smiles  # noqa: E402
from src.run_pfas_random_tabpfn import (  # noqa: E402
    calculate_rdkit_descriptors,
    prepare_features,
    predict_in_chunks,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = Path(
    os.environ.get(
        "TABPFN_CHECKPOINT",
        PACKAGE_ROOT / "models" / "tabpfn-v3.5-20260909.safetensors",
    )
).resolve()
DEFAULT_OUTDIR = REPO_ROOT / "runs" / "oecd2021_pfas_nonpfas_finetuned_tabpfn_20260616"

TASKS = ["TBP_C", "TMP_C", "logSW", "logPLV", "logKOW", "logKH"]
LOGKOW_CONFLICT_THRESHOLD = 0.5


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune TabPFN on global non-OECD-PFAS molecules and perform "
            "in-context inference on local OECD PFAS train/test molecules."
        )
    )
    parser.add_argument("--main-wide", type=Path, required=True)
    parser.add_argument("--hlc-wide", type=Path, required=True)
    parser.add_argument("--tl-logkow-xlsx", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--pfas-test-fraction", type=float, default=0.30)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--n-finetune-ctx-plus-query-samples", type=int, default=2048)
    parser.add_argument("--finetune-query-ratio", type=float, default=0.20)
    parser.add_argument("--n-estimators-finetune", type=int, default=1)
    parser.add_argument("--n-estimators-validation", type=int, default=1)
    parser.add_argument("--n-estimators-final", type=int, default=1)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--max-validation-labels", type=int, default=512)
    parser.add_argument("--n-inference-subsample-samples", type=int, default=10_000)
    parser.add_argument("--prediction-chunk-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        raw = float(value)
        return None if math.isnan(raw) or math.isinf(raw) else raw
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def standardize_value_smiles(smiles: object) -> tuple[str | None, str | None, str]:
    result = standardize_smiles("" if pd.isna(smiles) else str(smiles))
    reasons = ";".join(result.exclusion_reasons)
    return result.smiles, result.inchikey, reasons


def canonical_inchikey(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def first_non_null(values: pd.Series) -> Any:
    non_null = values.dropna()
    if len(non_null) == 0:
        return pd.NA
    return non_null.iloc[0]


def split_group_from_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"acyclic::{smiles}"
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold

        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        scaffold = ""
    return f"scaffold::{scaffold}" if scaffold else f"acyclic::{smiles}"


def filter_logkow_conflicts(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    kept: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for smiles, group in records.groupby("qsar_smiles", sort=False):
        values = group["logKOW"].astype(float).to_numpy()
        value_range = float(np.nanmax(values) - np.nanmin(values)) if len(values) else np.nan
        action = "kept_clean"
        kept_values = values

        if len(values) > 1 and value_range > LOGKOW_CONFLICT_THRESHOLD:
            if len(values) >= 3:
                median = float(np.nanmedian(values))
                mask = np.abs(values - median) <= LOGKOW_CONFLICT_THRESHOLD
                candidate = values[mask]
                candidate_range = float(np.nanmax(candidate) - np.nanmin(candidate)) if len(candidate) else np.nan
                if len(candidate) and candidate_range <= LOGKOW_CONFLICT_THRESHOLD:
                    action = "removed_outlier_rows"
                    kept_values = candidate
                else:
                    action = "discarded_ambiguous_conflict"
                    kept_values = np.asarray([], dtype=float)
            else:
                action = "discarded_ambiguous_conflict"
                kept_values = np.asarray([], dtype=float)
        elif len(values) > 1:
            action = "kept_within_threshold"

        if len(kept_values):
            kept.append(
                {
                    "qsar_smiles": smiles,
                    "logKOW": float(np.nanmedian(kept_values)),
                    "logKOW_source_tables": ";".join(sorted(set(group["source_table"].astype(str)))),
                    "logKOW_n_contributors": int(len(kept_values)),
                    "logKOW_filter_action": action,
                }
            )
        audit.append(
            {
                "qsar_smiles": smiles,
                "n_initial_rows": int(len(values)),
                "initial_min": float(np.nanmin(values)) if len(values) else None,
                "initial_max": float(np.nanmax(values)) if len(values) else None,
                "initial_range": value_range,
                "action": action,
                "n_kept_rows": int(len(kept_values)),
                "source_tables": ";".join(sorted(set(group["source_table"].astype(str)))),
            }
        )
    return pd.DataFrame(kept), pd.DataFrame(audit)


def load_tl_logkow_s7(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_excel(path, sheet_name="S7", header=1)
    raw = raw.rename(columns={"smiles": "raw_smiles", "Spilt": "source_split"})
    raw = raw.loc[raw["raw_smiles"].notna() & raw["logKOW"].notna()].copy()
    standardized = raw["raw_smiles"].map(standardize_value_smiles)
    raw["qsar_smiles"] = [item[0] for item in standardized]
    raw["inchikey"] = [item[1] for item in standardized]
    raw["standardization_reasons"] = [item[2] for item in standardized]
    raw["logKOW"] = pd.to_numeric(raw["logKOW"], errors="coerce")
    clean = raw.loc[
        raw["qsar_smiles"].notna()
        & raw["logKOW"].notna()
        & raw["standardization_reasons"].eq("")
    ].copy()
    clean["source_table"] = "tl_mt_bio_s7_logkow"
    kept, audit = filter_logkow_conflicts(clean[["qsar_smiles", "logKOW", "source_table"]])
    meta = (
        clean.sort_values("qsar_smiles")
        .groupby("qsar_smiles", as_index=False)
        .agg(
            inchikey=("inchikey", first_non_null),
            tl_s7_source_split=("source_split", first_non_null),
            tl_s7_molecular_weight=("Molecular Weight", first_non_null),
        )
    )
    kept = kept.merge(meta, on="qsar_smiles", how="left")
    summary = {
        "tl_s7_raw_rows": int(len(raw)),
        "tl_s7_after_est_jcim_filters": int(len(clean)),
        "tl_s7_unique_after_filters": int(clean["qsar_smiles"].nunique()),
        "tl_s7_conflict_kept_unique": int(len(kept)),
        "tl_s7_invalid_or_excluded_rows": int(len(raw) - len(clean)),
        "tl_s7_duplicate_conflict_audit_rows": int(len(audit)),
        "tl_s7_ambiguous_conflicts_removed": int(audit["action"].eq("discarded_ambiguous_conflict").sum())
        if len(audit)
        else 0,
    }
    return kept, summary


def build_merged_wide(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    main = pd.read_csv(args.main_wide)
    for task in TASKS:
        if task not in main.columns:
            main[task] = np.nan
    main["source_tables"] = main.get("source_tables", "clean_merge_no_hlc")
    main["source_tables"] = main["source_tables"].fillna("clean_merge_no_hlc").astype(str)
    main["inchikey"] = main.get("inchikey", pd.Series(pd.NA, index=main.index))

    tl_logkow, tl_summary = load_tl_logkow_s7(args.tl_logkow_xlsx)
    current_logkow = main.loc[main["logKOW"].notna(), ["qsar_smiles", "logKOW"]].copy()
    current_logkow["source_table"] = "current_clean_merge_logkow"
    combined_logkow_records = pd.concat(
        [
            current_logkow,
            tl_logkow[["qsar_smiles", "logKOW"]].assign(source_table="tl_mt_bio_s7_logkow"),
        ],
        ignore_index=True,
    )
    final_logkow, logkow_audit = filter_logkow_conflicts(combined_logkow_records)

    main = main.drop(columns=["logKOW"])
    main = main.merge(final_logkow[["qsar_smiles", "logKOW"]], on="qsar_smiles", how="left")

    existing_smiles = set(main["qsar_smiles"].astype(str))
    new_tl = final_logkow.loc[~final_logkow["qsar_smiles"].isin(existing_smiles)].copy()
    if len(new_tl):
        new_tl = new_tl.merge(tl_logkow.drop(columns=["logKOW"]), on="qsar_smiles", how="left")
        rows = []
        for _, row in new_tl.iterrows():
            rows.append(
                {
                    "qsar_smiles": row["qsar_smiles"],
                    "inchikey": row.get("inchikey") or canonical_inchikey(row["qsar_smiles"]),
                    "chemical_name": pd.NA,
                    "casrn": pd.NA,
                    "is_pfas": False,
                    "is_fluorinated_local": False,
                    "subset_label": "non_pfas",
                    "is_paper_core": False,
                    "possible_source_overlap_any": False,
                    "n_total_labels": 1,
                    "n_unique_endpoints": 1,
                    "mean_sample_weight": 1.0,
                    "TBP_C": np.nan,
                    "TMP_C": np.nan,
                    "logSW": np.nan,
                    "logPLV": np.nan,
                    "logKOW": float(row["logKOW"]),
                    "logKH": np.nan,
                    "split_group": split_group_from_smiles(row["qsar_smiles"]),
                    "split": str(row.get("tl_s7_source_split", "training")).replace("training", "train"),
                    "source_tables": "tl_mt_bio_s7_logkow",
                }
            )
        main = pd.concat([main, pd.DataFrame(rows)], ignore_index=True, sort=False)

    hlc = pd.read_csv(args.hlc_wide)
    hlc = hlc.loc[hlc["qsar_smiles"].notna() & hlc["logKH"].notna()].copy()
    hlc = hlc[["qsar_smiles", "logKH", "inchikey", "is_pfas", "is_fluorinated_local", "split"]].copy()
    hlc = hlc.rename(columns={"split": "hlc_source_split"})
    main = main.merge(
        hlc[["qsar_smiles", "logKH", "hlc_source_split"]].rename(columns={"logKH": "_hlc_logKH"}),
        on="qsar_smiles",
        how="outer",
    )
    main["logKH"] = main["logKH"].where(main["logKH"].notna(), main["_hlc_logKH"])
    main = main.drop(columns=["_hlc_logKH"])
    new_hlc_mask = main["source_tables"].isna() & main["logKH"].notna()
    if new_hlc_mask.any():
        main.loc[new_hlc_mask, "source_tables"] = "est_kh_random_tabpfn_20260608"
        main.loc[new_hlc_mask, "split_group"] = main.loc[new_hlc_mask, "qsar_smiles"].map(split_group_from_smiles)
        main.loc[new_hlc_mask, "split"] = main.loc[new_hlc_mask, "hlc_source_split"].fillna("train")
        main.loc[new_hlc_mask, "is_pfas"] = False
        main.loc[new_hlc_mask, "is_fluorinated_local"] = False
        main.loc[new_hlc_mask, "inchikey"] = main.loc[new_hlc_mask, "qsar_smiles"].map(canonical_inchikey)

    main["oecd2021_pfas"] = main["qsar_smiles"].map(oecd2021_pfas)
    main["endpoint_count_6"] = main[TASKS].notna().sum(axis=1).astype(int)
    main = main.loc[main["endpoint_count_6"] > 0].copy().reset_index(drop=True)

    summary = {
        **tl_summary,
        "current_logKOW_before_tl": int(current_logkow["qsar_smiles"].nunique()),
        "logKOW_after_tl_est_jcim_conflict_filter": int(main["logKOW"].notna().sum()),
        "logKOW_ambiguous_conflicts_removed_after_merge": int(
            logkow_audit["action"].eq("discarded_ambiguous_conflict").sum()
        )
        if len(logkow_audit)
        else 0,
        "six_endpoint_molecules": int(len(main)),
        "six_endpoint_labels": int(main[TASKS].notna().sum().sum()),
        "endpoint_counts": {task: int(main[task].notna().sum()) for task in TASKS},
        "oecd2021_pfas_molecules": int(main["oecd2021_pfas"].sum()),
        "oecd2021_nonpfas_molecules": int((~main["oecd2021_pfas"]).sum()),
    }
    return main, summary, logkow_audit


def assign_oecd_pfas_context_split(wide: pd.DataFrame, *, seed: int, test_fraction: float) -> pd.DataFrame:
    pfas_idx = np.flatnonzero(wide["oecd2021_pfas"].to_numpy(dtype=bool))
    if len(pfas_idx) < 2:
        raise ValueError("Need at least two OECD PFAS molecules for context/test split.")
    rng = np.random.default_rng(seed)
    shuffled = pfas_idx.copy()
    rng.shuffle(shuffled)
    test_n = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * test_fraction))))
    roles = pd.Series("nonpfas", index=wide.index, dtype=object)
    roles.iloc[pfas_idx] = "context_train"
    roles.iloc[shuffled[:test_n]] = "test"
    out = wide.copy()
    out["oecd_pfas_context_split"] = roles
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() == 0:
        return {"n_eval": 0, "mae": None, "rmse": None, "r2": None, "spearman": None}
    yt = y_true[valid].astype(float)
    yp = y_pred[valid].astype(float)
    base = baseline[valid].astype(float)
    spearman = float(pd.Series(yt).corr(pd.Series(yp), method="spearman"))
    baseline_mae = float(mean_absolute_error(yt, base))
    mae = float(mean_absolute_error(yt, yp))
    return {
        "n_eval": int(valid.sum()),
        "mae": mae,
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "r2": float(r2_score(yt, yp)) if len(yt) >= 2 else None,
        "spearman": None if not np.isfinite(spearman) else spearman,
        "baseline_context_mean_mae": baseline_mae,
        "mae_improvement_vs_context_mean": float(baseline_mae - mae),
        "relative_mae_improvement_vs_context_mean": float((baseline_mae - mae) / baseline_mae)
        if baseline_mae
        else None,
    }


def endpoint_seed(seed: int, endpoint: str) -> int:
    return int(seed + sum((i + 1) * ord(ch) for i, ch in enumerate(endpoint)) % 100_000)


def fit_predict_endpoint(
    *,
    args: argparse.Namespace,
    endpoint: str,
    wide: pd.DataFrame,
    descriptors: np.ndarray,
    desc_names: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    endpoint_outdir = args.outdir / "endpoint_finetune" / endpoint
    endpoint_outdir.mkdir(parents=True, exist_ok=True)

    fit_mask = wide[endpoint].notna() & ~wide["oecd2021_pfas"].astype(bool)
    context_mask = (
        wide[endpoint].notna()
        & wide["oecd2021_pfas"].astype(bool)
        & wide["oecd_pfas_context_split"].eq("context_train")
    )
    test_mask = (
        wide[endpoint].notna()
        & wide["oecd2021_pfas"].astype(bool)
        & wide["oecd_pfas_context_split"].eq("test")
    )
    fit_idx = np.flatnonzero(fit_mask.to_numpy())
    context_idx = np.flatnonzero(context_mask.to_numpy())
    test_idx = np.flatnonzero(test_mask.to_numpy())

    if len(fit_idx) < 20 or len(context_idx) < 2 or len(test_idx) < 2:
        metrics = {
            "endpoint": endpoint,
            "status": "skipped_insufficient_labels",
            "n_finetune_nonpfas": int(len(fit_idx)),
            "n_context_oecd_pfas": int(len(context_idx)),
            "n_test_oecd_pfas": int(len(test_idx)),
        }
        return metrics, pd.DataFrame()

    x_fit_all, x_eval, feature_meta = prepare_features(
        descriptors,
        train_idx=fit_idx,
        eval_idx=np.concatenate([context_idx, test_idx]),
        feature_names=desc_names,
    )
    x_context = x_eval[: len(context_idx)]
    x_test = x_eval[len(context_idx) :]
    y_fit_all = wide.iloc[fit_idx][endpoint].astype(float).to_numpy(dtype=np.float32)
    y_context = wide.iloc[context_idx][endpoint].astype(float).to_numpy(dtype=np.float32)
    y_test = wide.iloc[test_idx][endpoint].astype(float).to_numpy(dtype=np.float32)

    rng = np.random.default_rng(endpoint_seed(args.seed, endpoint))
    order = np.arange(len(y_fit_all))
    rng.shuffle(order)
    val_n = min(
        int(args.max_validation_labels),
        max(1, int(round(float(args.validation_fraction) * len(order)))),
    )
    val_n = min(val_n, max(1, len(order) - 20))
    val_order = order[:val_n]
    train_order = order[val_n:]
    x_ft_train = x_fit_all[train_order]
    y_ft_train = y_fit_all[train_order]
    x_ft_val = x_fit_all[val_order]
    y_ft_val = y_fit_all[val_order]

    reg = FinetunedTabPFNRegressor(
        device=args.device,
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        validation_split_ratio=float(args.validation_fraction),
        n_finetune_ctx_plus_query_samples=int(args.n_finetune_ctx_plus_query_samples),
        finetune_ctx_query_split_ratio=float(args.finetune_query_ratio),
        n_inference_subsample_samples=int(args.n_inference_subsample_samples),
        random_state=endpoint_seed(args.seed, endpoint),
        early_stopping=False,
        n_estimators_finetune=int(args.n_estimators_finetune),
        n_estimators_validation=int(args.n_estimators_validation),
        n_estimators_final_inference=int(args.n_estimators_final),
        save_checkpoint_interval=None,
        extra_regressor_kwargs={
            "model_path": args.checkpoint.resolve(),
            "inference_config": {"SUBSAMPLE_SAMPLES": int(args.n_inference_subsample_samples)},
            "show_progress_bar": False,
        },
    )
    print(
        f"[{utc_now()}] {endpoint}: finetune nonPFAS train={len(y_ft_train)} "
        f"val={len(y_ft_val)} contextPFAS={len(y_context)} testPFAS={len(y_test)} "
        f"features={x_fit_all.shape[1]}",
        flush=True,
    )
    reg.fit(x_ft_train, y_ft_train, X_val=x_ft_val, y_val=y_ft_val, output_dir=endpoint_outdir)

    inference_model = reg.finetuned_inference_regressor_
    inference_model.fit(x_context, y_context)
    y_pred = predict_in_chunks(inference_model, x_test, int(args.prediction_chunk_size))
    baseline = np.full_like(y_test, float(np.mean(y_context)), dtype=np.float32)
    metric_values = regression_metrics(y_test, y_pred, baseline)
    metrics = {
        "endpoint": endpoint,
        "status": "ok",
        "n_finetune_nonpfas": int(len(fit_idx)),
        "n_finetune_train_used": int(len(y_ft_train)),
        "n_finetune_validation_used": int(len(y_ft_val)),
        "n_context_oecd_pfas": int(len(context_idx)),
        "n_test_oecd_pfas": int(len(test_idx)),
        "selected_features": int(feature_meta["selected_features"]),
        **metric_values,
    }

    pred_frame = wide.iloc[test_idx][
        ["qsar_smiles", "inchikey", "oecd_pfas_context_split", "source_tables", endpoint]
    ].copy()
    pred_frame = pred_frame.rename(columns={endpoint: "target"})
    pred_frame["endpoint"] = endpoint
    pred_frame["prediction"] = y_pred
    pred_frame["context_mean_baseline"] = baseline
    pred_frame["absolute_error"] = np.abs(pred_frame["prediction"].to_numpy(float) - pred_frame["target"].to_numpy(float))

    # Release GPU memory before the next endpoint.
    del reg, inference_model
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

    write_json(endpoint_outdir / "feature_metadata.json", feature_meta)
    return metrics, pred_frame


def make_report(summary: dict[str, Any], metrics: pd.DataFrame) -> str:
    lines = [
        "# OECD PFAS Non-PFAS Fine-Tuned TabPFN",
        "",
        "Single-run experiment: gradient fine-tuning uses global non-OECD-PFAS molecules; final inference refits the fine-tuned TabPFN context on OECD PFAS context-train molecules and predicts held-out OECD PFAS molecules.",
        "",
        "## Dataset",
        "",
        f"- Six-endpoint molecules after TL-MT-Bio S7 logKOW merge: `{summary['dataset']['six_endpoint_molecules']}`",
        f"- Six-endpoint labels after merge: `{summary['dataset']['six_endpoint_labels']}`",
        f"- OECD PFAS molecules: `{summary['dataset']['oecd2021_pfas_molecules']}`",
        "",
        "| endpoint | labels |",
        "|---|---:|",
    ]
    for endpoint, count in summary["dataset"]["endpoint_counts"].items():
        lines.append(f"| {endpoint} | {count} |")
    lines.extend(["", "## Metrics", "", "| endpoint | n fine-tune nonPFAS | n context PFAS | n test PFAS | R2 | RMSE | MAE | Spearman |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for _, row in metrics.iterrows():
        r2 = "" if pd.isna(row.get("r2")) else f"{row['r2']:.4f}"
        rmse = "" if pd.isna(row.get("rmse")) else f"{row['rmse']:.4f}"
        mae = "" if pd.isna(row.get("mae")) else f"{row['mae']:.4f}"
        spear = "" if pd.isna(row.get("spearman")) else f"{row['spearman']:.4f}"
        lines.append(
            f"| {row['endpoint']} | {int(row.get('n_finetune_nonpfas', 0))} | "
            f"{int(row.get('n_context_oecd_pfas', 0))} | {int(row.get('n_test_oecd_pfas', 0))} | "
            f"{r2} | {rmse} | {mae} | {spear} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.*")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    wide, dataset_summary, logkow_audit = build_merged_wide(args)
    wide = assign_oecd_pfas_context_split(wide, seed=args.seed, test_fraction=args.pfas_test_fraction)
    wide.to_csv(args.outdir / "merged_6endpoint_wide_with_tl_logkow.csv", index=False)
    wide.loc[wide["oecd2021_pfas"]].to_csv(args.outdir / "oecd2021_pfas_context_split_wide.csv", index=False)
    logkow_audit.to_csv(args.outdir / "logkow_tl_s7_merge_conflict_audit.csv", index=False)

    descriptor_cache = args.outdir / "rdkit2d_descriptor_cache.npz"
    if descriptor_cache.exists():
        cached = np.load(descriptor_cache, allow_pickle=True)
        descriptors = cached["descriptors"].astype(np.float32, copy=False)
        desc_names = [str(item) for item in cached["descriptor_names"].tolist()]
        invalid_indices = [int(item) for item in cached["invalid_indices"].tolist()]
    else:
        print(f"[{utc_now()}] calculating RDKit descriptors for {len(wide)} molecules", flush=True)
        descriptors, desc_names, invalid_indices = calculate_rdkit_descriptors(wide["qsar_smiles"])
        np.savez_compressed(
            descriptor_cache,
            descriptors=descriptors.astype(np.float32, copy=False),
            descriptor_names=np.asarray(desc_names, dtype=str),
            invalid_indices=np.asarray(invalid_indices, dtype=int),
        )

    metrics_rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for endpoint in TASKS:
        metrics, pred_frame = fit_predict_endpoint(
            args=args,
            endpoint=endpoint,
            wide=wide,
            descriptors=descriptors,
            desc_names=desc_names,
        )
        metrics_rows.append(metrics)
        if len(pred_frame):
            predictions.append(pred_frame)
        pd.DataFrame(metrics_rows).to_csv(args.outdir / "metrics_partial.csv", index=False)

    metrics_df = pd.DataFrame(metrics_rows)
    pred_df = pd.concat(predictions, ignore_index=True, sort=False) if predictions else pd.DataFrame()
    metrics_df.to_csv(args.outdir / "metrics.csv", index=False)
    pred_df.to_csv(args.outdir / "predictions.csv", index=False)

    split_summary: dict[str, Any] = {}
    for split_name in ["context_train", "test"]:
        frame = wide.loc[wide["oecd_pfas_context_split"].eq(split_name)]
        split_summary[split_name] = {
            "molecules": int(len(frame)),
            "endpoint_counts": {task: int(frame[task].notna().sum()) for task in TASKS},
            "labels": int(frame[TASKS].notna().sum().sum()),
        }
    summary = {
        "created_at": utc_now(),
        "argv": sys.argv,
        "checkpoint": str(args.checkpoint),
        "tabpfn_source": str(TABPFN_SRC),
        "official_docs": "https://docs.priorlabs.ai/capabilities/fine-tuning#combine-with-feature-engineering",
        "dataset": dataset_summary,
        "oecd_pfas_split": split_summary,
        "descriptor_metadata": {
            "input_features": int(descriptors.shape[1]),
            "invalid_smiles_count": int(len(invalid_indices)),
        },
        "finetuning_config": {
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "n_finetune_ctx_plus_query_samples": int(args.n_finetune_ctx_plus_query_samples),
            "n_estimators_finetune": int(args.n_estimators_finetune),
            "n_estimators_validation": int(args.n_estimators_validation),
            "n_estimators_final": int(args.n_estimators_final),
            "n_inference_subsample_samples": int(args.n_inference_subsample_samples),
            "feature_engineering": "RDKit 2D descriptors with endpoint-specific nonPFAS-train imputation and nonconstant filtering",
            "gradient_finetune_pool": "global non-OECD-PFAS molecules only",
            "inference_context": "OECD PFAS context_train molecules only",
            "test_pool": "held-out OECD PFAS molecules",
        },
        "metrics": metrics_df.to_dict(orient="records"),
    }
    write_json(args.outdir / "summary.json", summary)
    (args.outdir / "REPORT.md").write_text(make_report(summary, metrics_df), encoding="utf-8")
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
