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


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
TABPFN_SRC = Path(os.environ.get("TABPFN_SRC", PACKAGE_ROOT / "code" / "TabPFN" / "src")).resolve()
if str(TABPFN_SRC) not in sys.path:
    sys.path.insert(0, str(TABPFN_SRC))

from tabpfn import TabPFNRegressor  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = REPO_ROOT / "runs" / "oecd2021_pfas_context_only_tabpfn_same_split_20260616"
DEFAULT_CHECKPOINT = Path(
    os.environ.get(
        "TABPFN_CHECKPOINT",
        PACKAGE_ROOT / "models" / "tabpfn-v3.5-20260909.safetensors",
    )
).resolve()
TASKS = ["TBP_C", "TMP_C", "logSW", "logPLV", "logKOW", "logKH"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a no-gradient-finetuning TabPFN baseline on the exact OECD PFAS "
            "context/test split produced by the non-PFAS fine-tuning experiment."
        )
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--subsample-samples", type=int, default=10_000)
    parser.add_argument("--prediction-chunk-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fit-mode", default="fit_preprocessors")
    parser.add_argument("--memory-saving-mode", default="auto")
    parser.add_argument(
        "--feature-reference",
        choices=["nonpfas", "context"],
        default="nonpfas",
        help=(
            "Which rows define RDKit2D imputation and nonconstant feature filtering. "
            "`nonpfas` matches the fine-tuned run's feature preprocessing and best "
            "isolates the effect of gradient fine-tuning."
        ),
    )
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


def endpoint_seed(seed: int, endpoint: str) -> int:
    return int(seed + sum((i + 1) * ord(ch) for i, ch in enumerate(endpoint)) % 100_000)


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


def prepare_features(
    descriptors: np.ndarray,
    *,
    reference_idx: np.ndarray,
    eval_idx: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    ref_raw = descriptors[reference_idx].astype(np.float32, copy=False)
    finite_ref = np.isfinite(ref_raw)
    has_any_value = finite_ref.any(axis=0)
    if not has_any_value.any():
        raise ValueError("No finite RDKit descriptors are available for this endpoint.")

    medians = np.zeros(ref_raw.shape[1], dtype=np.float32)
    with np.errstate(all="ignore"):
        median_values = np.nanmedian(np.where(finite_ref, ref_raw, np.nan)[:, has_any_value], axis=0)
    median_values = np.where(np.isfinite(median_values), median_values, 0.0)
    medians[has_any_value] = median_values.astype(np.float32)

    ref_imputed = np.where(np.isfinite(ref_raw), ref_raw, medians)
    variance = np.zeros(ref_raw.shape[1], dtype=np.float32)
    variance[has_any_value] = np.var(ref_imputed[:, has_any_value], axis=0)
    non_constant = variance > 0.0
    selected = has_any_value & non_constant
    if not selected.any():
        selected = has_any_value

    matrix = descriptors[eval_idx].astype(np.float32, copy=False)
    imputed = np.where(np.isfinite(matrix), matrix, medians)
    transformed = imputed[:, selected].astype(np.float32, copy=False)
    meta = {
        "input_features": int(descriptors.shape[1]),
        "selected_features": int(selected.sum()),
        "dropped_all_missing_features": int((~has_any_value).sum()),
        "dropped_constant_features": int((has_any_value & ~non_constant).sum()),
        "selected_feature_names": [feature_names[i] for i in np.flatnonzero(selected)],
    }
    return transformed, meta


def predict_in_chunks(model: TabPFNRegressor, x: np.ndarray, chunk_size: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(x), chunk_size):
        stop = min(start + chunk_size, len(x))
        chunks.append(np.asarray(model.predict(x[start:stop]), dtype=np.float32))
    return np.concatenate(chunks) if chunks else np.asarray([], dtype=np.float32)


def make_model(args: argparse.Namespace, endpoint: str) -> TabPFNRegressor:
    inference_config: dict[str, Any] = {}
    if args.subsample_samples and args.subsample_samples > 0:
        inference_config["SUBSAMPLE_SAMPLES"] = int(args.subsample_samples)
    return TabPFNRegressor(
        model_path=args.checkpoint.resolve(),
        n_estimators=int(args.n_estimators),
        device=args.device,
        ignore_pretraining_limits=True,
        fit_mode=args.fit_mode,
        memory_saving_mode=args.memory_saving_mode,
        random_state=endpoint_seed(args.seed, endpoint),
        inference_config=inference_config or None,
        show_progress_bar=False,
    )


def fit_predict_endpoint(
    *,
    args: argparse.Namespace,
    endpoint: str,
    wide: pd.DataFrame,
    descriptors: np.ndarray,
    desc_names: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    nonpfas_mask = wide[endpoint].notna() & ~wide["oecd2021_pfas"].astype(bool)
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
    nonpfas_idx = np.flatnonzero(nonpfas_mask.to_numpy())
    context_idx = np.flatnonzero(context_mask.to_numpy())
    test_idx = np.flatnonzero(test_mask.to_numpy())
    if len(context_idx) < 2 or len(test_idx) < 2:
        return (
            {
                "endpoint": endpoint,
                "status": "skipped_insufficient_labels",
                "n_reference_nonpfas": int(len(nonpfas_idx)),
                "n_context_oecd_pfas": int(len(context_idx)),
                "n_test_oecd_pfas": int(len(test_idx)),
            },
            pd.DataFrame(),
        )

    reference_idx = nonpfas_idx if args.feature_reference == "nonpfas" else context_idx
    eval_idx = np.concatenate([context_idx, test_idx])
    x_eval, feature_meta = prepare_features(
        descriptors,
        reference_idx=reference_idx,
        eval_idx=eval_idx,
        feature_names=desc_names,
    )
    x_context = x_eval[: len(context_idx)]
    x_test = x_eval[len(context_idx) :]
    y_context = wide.iloc[context_idx][endpoint].astype(float).to_numpy(dtype=np.float32)
    y_test = wide.iloc[test_idx][endpoint].astype(float).to_numpy(dtype=np.float32)

    print(
        f"[{utc_now()}] {endpoint}: contextPFAS={len(y_context)} testPFAS={len(y_test)} "
        f"feature_reference={args.feature_reference} features={x_eval.shape[1]}",
        flush=True,
    )
    model = make_model(args, endpoint)
    model.fit(x_context, y_context)
    y_pred = predict_in_chunks(model, x_test, int(args.prediction_chunk_size))
    baseline = np.full_like(y_test, float(np.mean(y_context)), dtype=np.float32)
    metric_values = regression_metrics(y_test, y_pred, baseline)
    metrics = {
        "endpoint": endpoint,
        "status": "ok",
        "n_reference_nonpfas": int(len(nonpfas_idx)),
        "n_context_oecd_pfas": int(len(context_idx)),
        "n_test_oecd_pfas": int(len(test_idx)),
        "feature_reference": args.feature_reference,
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

    del model
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics, pred_frame


def make_report(summary: dict[str, Any], metrics: pd.DataFrame, comparison: pd.DataFrame) -> str:
    lines = [
        "# OECD PFAS Context-Only TabPFN Same-Split Baseline",
        "",
        "No-gradient-finetuning baseline on the exact OECD PFAS context/test split from the non-PFAS fine-tuned TabPFN experiment.",
        "",
        "## Protocol",
        "",
        f"- Source split run: `{summary['source_run']}`",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- TabPFN source: `{summary['tabpfn_source']}`",
        f"- Feature reference: `{summary['feature_reference']}`",
        f"- n_estimators: `{summary['n_estimators']}`",
        "",
        "## Context-Only Metrics",
        "",
        "| endpoint | n context PFAS | n test PFAS | R2 | RMSE | MAE | Spearman |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['endpoint']} | {int(row['n_context_oecd_pfas'])} | {int(row['n_test_oecd_pfas'])} | "
            f"{row['r2']:.4f} | {row['rmse']:.4f} | {row['mae']:.4f} | {row['spearman']:.4f} |"
        )
    if len(comparison):
        lines.extend(
            [
                "",
                "## Comparison With Non-PFAS Fine-Tuning",
                "",
                "| endpoint | context-only R2 | fine-tuned R2 | delta R2 | context-only MAE | fine-tuned MAE | delta MAE |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in comparison.iterrows():
            lines.append(
                f"| {row['endpoint']} | {row['context_only_r2']:.4f} | {row['finetuned_r2']:.4f} | "
                f"{row['delta_r2_finetuned_minus_context_only']:.4f} | {row['context_only_mae']:.4f} | "
                f"{row['finetuned_mae']:.4f} | {row['delta_mae_finetuned_minus_context_only']:.4f} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    args.outdir.mkdir(parents=True, exist_ok=True)
    wide_path = args.source_run / "merged_6endpoint_wide_with_tl_logkow.csv"
    descriptor_path = args.source_run / "rdkit2d_descriptor_cache.npz"
    if not wide_path.exists():
        raise FileNotFoundError(wide_path)
    if not descriptor_path.exists():
        raise FileNotFoundError(descriptor_path)
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    wide = pd.read_csv(wide_path)
    cached = np.load(descriptor_path, allow_pickle=True)
    descriptors = cached["descriptors"].astype(np.float32, copy=False)
    desc_names = [str(item) for item in cached["descriptor_names"].tolist()]

    metrics_rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for endpoint in args.tasks:
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

    comparison = pd.DataFrame()
    finetuned_metrics_path = args.source_run / "metrics.csv"
    if finetuned_metrics_path.exists():
        finetuned = pd.read_csv(finetuned_metrics_path)
        comparison = metrics_df.merge(
            finetuned[["endpoint", "r2", "rmse", "mae", "spearman"]].rename(
                columns={
                    "r2": "finetuned_r2",
                    "rmse": "finetuned_rmse",
                    "mae": "finetuned_mae",
                    "spearman": "finetuned_spearman",
                }
            ),
            on="endpoint",
            how="left",
        )
        comparison = comparison.rename(
            columns={
                "r2": "context_only_r2",
                "rmse": "context_only_rmse",
                "mae": "context_only_mae",
                "spearman": "context_only_spearman",
            }
        )
        comparison["delta_r2_finetuned_minus_context_only"] = (
            comparison["finetuned_r2"] - comparison["context_only_r2"]
        )
        comparison["delta_mae_finetuned_minus_context_only"] = (
            comparison["finetuned_mae"] - comparison["context_only_mae"]
        )
        comparison["delta_rmse_finetuned_minus_context_only"] = (
            comparison["finetuned_rmse"] - comparison["context_only_rmse"]
        )
        comparison.to_csv(args.outdir / "comparison_with_nonpfas_finetuned.csv", index=False)

    summary = {
        "created_at": utc_now(),
        "source_run": str(args.source_run),
        "checkpoint": str(args.checkpoint),
        "tabpfn_source": str(TABPFN_SRC),
        "feature_reference": args.feature_reference,
        "n_estimators": int(args.n_estimators),
        "fit_mode": args.fit_mode,
        "metrics": metrics_df.to_dict(orient="records"),
        "comparison_with_nonpfas_finetuned": comparison.to_dict(orient="records")
        if len(comparison)
        else [],
    }
    write_json(args.outdir / "summary.json", summary)
    (args.outdir / "REPORT.md").write_text(make_report(summary, metrics_df, comparison), encoding="utf-8")
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
