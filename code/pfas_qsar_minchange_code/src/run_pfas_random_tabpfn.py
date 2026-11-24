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

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    from rdkit.ML.Descriptors import MoleculeDescriptors
except Exception as exc:  # pragma: no cover
    raise ImportError("RDKit is required for PFAS TabPFN evaluation.") from exc


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
TABPFN_SRC = Path(os.environ.get("TABPFN_SRC", PACKAGE_ROOT / "code" / "TabPFN" / "src")).resolve()
if str(TABPFN_SRC) not in sys.path:
    sys.path.insert(0, str(TABPFN_SRC))

from tabpfn import TabPFNRegressor  # noqa: E402


DEFAULT_TASKS = ["TBP_C", "TMP_C", "logSW", "logPLV", "logKOW"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single TabPFN RDKit2D experiment on PFAS-only filtered 5-endpoint data."
    )
    parser.add_argument("--wide-csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fit-mode", default="fit_preprocessors")
    parser.add_argument("--memory-saving-mode", default="auto")
    parser.add_argument("--prediction-chunk-size", type=int, default=512)
    parser.add_argument(
        "--subsample-samples",
        type=int,
        default=10_000,
        help="Optional TabPFN SUBSAMPLE_SAMPLES inference config; harmless when train n is smaller.",
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


def descriptor_names() -> list[str]:
    return [name for name, _ in Descriptors.descList]


def _coerce_descriptor_row(row: tuple[Any, ...], n_features: int) -> list[float]:
    cleaned: list[float] = []
    for item in row:
        try:
            raw = float(item)
        except (TypeError, ValueError):
            raw = np.nan
        if not np.isfinite(raw) or abs(raw) > np.finfo(np.float32).max:
            raw = np.nan
        cleaned.append(raw)
    if len(cleaned) != n_features:
        return [float("nan")] * n_features
    return cleaned


def calculate_rdkit_descriptors(smiles_values: pd.Series) -> tuple[np.ndarray, list[str], list[int]]:
    RDLogger.DisableLog("rdApp.*")
    names = descriptor_names()
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(names)
    values = np.empty((len(smiles_values), len(names)), dtype=np.float32)
    invalid_indices: list[int] = []
    for idx, smiles in enumerate(smiles_values.astype(str)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            values[idx, :] = np.nan
            invalid_indices.append(idx)
            continue
        try:
            row = calc.CalcDescriptors(mol)
        except Exception:
            values[idx, :] = np.nan
            invalid_indices.append(idx)
            continue
        values[idx, :] = np.asarray(_coerce_descriptor_row(row, len(names)), dtype=np.float32)
    return values, names, invalid_indices


def assign_random_split(wide: pd.DataFrame, *, test_fraction: float, seed: int) -> pd.DataFrame:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(wide))
    rng.shuffle(indices)
    test_n = max(1, min(len(wide) - 1, int(round(len(wide) * test_fraction))))
    roles = np.full(len(wide), "train", dtype=object)
    roles[indices[:test_n]] = "test"

    out = wide.copy()
    if "split" in out.columns:
        out["source_split"] = out["split"].astype(str)
    out["split"] = roles
    return out


def split_summary(wide: pd.DataFrame, tasks: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split_name in ["train", "test"]:
        frame = wide[wide["split"].eq(split_name)]
        summary[split_name] = {
            "molecules": int(len(frame)),
            "endpoint_counts": {task: int(frame[task].notna().sum()) for task in tasks},
            "labels": int(frame[tasks].notna().sum().sum()),
        }
    return summary


def prepare_features(
    descriptors: np.ndarray,
    *,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    fit_raw = descriptors[train_idx].astype(np.float32, copy=False)
    finite_fit = np.isfinite(fit_raw)
    has_any_value = finite_fit.any(axis=0)
    if not has_any_value.any():
        raise ValueError("No finite RDKit descriptors are available for this endpoint.")

    medians = np.zeros(fit_raw.shape[1], dtype=np.float32)
    with np.errstate(all="ignore"):
        median_values = np.nanmedian(np.where(finite_fit, fit_raw, np.nan)[:, has_any_value], axis=0)
    median_values = np.where(np.isfinite(median_values), median_values, 0.0)
    medians[has_any_value] = median_values.astype(np.float32)

    fit_imputed = np.where(np.isfinite(fit_raw), fit_raw, medians)
    variance = np.zeros(fit_raw.shape[1], dtype=np.float32)
    variance[has_any_value] = np.var(fit_imputed[:, has_any_value], axis=0)
    non_constant = variance > 0.0
    selected = has_any_value & non_constant
    if not selected.any():
        selected = has_any_value

    def transform(rows: np.ndarray) -> np.ndarray:
        matrix = descriptors[rows].astype(np.float32, copy=False)
        imputed = np.where(np.isfinite(matrix), matrix, medians)
        return imputed[:, selected].astype(np.float32, copy=False)

    meta = {
        "input_features": int(descriptors.shape[1]),
        "selected_features": int(selected.sum()),
        "dropped_all_missing_features": int((~has_any_value).sum()),
        "dropped_constant_features": int((has_any_value & ~non_constant).sum()),
        "selected_feature_names": [feature_names[i] for i in np.flatnonzero(selected)],
    }
    return transform(train_idx), transform(eval_idx), meta


def make_model(args: argparse.Namespace) -> TabPFNRegressor:
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
        random_state=int(args.seed),
        inference_config=inference_config or None,
        show_progress_bar=False,
    )


def predict_in_chunks(model: TabPFNRegressor, x: np.ndarray, chunk_size: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(x), chunk_size):
        stop = min(start + chunk_size, len(x))
        chunks.append(np.asarray(model.predict(x[start:stop]), dtype=np.float32).reshape(-1))
    return np.concatenate(chunks, axis=0)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() == 0:
        return {
            "n_eval": 0,
            "mae": None,
            "rmse": None,
            "r2": None,
            "spearman": None,
            "baseline_mae": None,
            "mae_improvement_vs_baseline": None,
            "relative_mae_improvement": None,
        }
    yt = y_true[valid].astype(float)
    yp = y_pred[valid].astype(float)
    base = baseline[valid].astype(float)
    mae = float(mean_absolute_error(yt, yp))
    baseline_mae = float(mean_absolute_error(yt, base))
    spearman = float(pd.Series(yt).corr(pd.Series(yp), method="spearman"))
    return {
        "n_eval": int(valid.sum()),
        "mae": mae,
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "r2": float(r2_score(yt, yp)) if len(yt) >= 2 else None,
        "spearman": None if not np.isfinite(spearman) else spearman,
        "baseline_mae": baseline_mae,
        "mae_improvement_vs_baseline": float(baseline_mae - mae),
        "relative_mae_improvement": float((baseline_mae - mae) / baseline_mae) if baseline_mae else None,
    }


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    tasks: list[str],
    wide: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    rows = [
        "| Endpoint | train n | test n | MAE | RMSE | R2 | Spearman | baseline MAE | rel MAE gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics.iterrows():
        rows.append(
            "| {endpoint} | {n_train:d} | {n_eval:d} | {mae:.6g} | {rmse:.6g} | {r2:.6g} | "
            "{spearman:.6g} | {baseline_mae:.6g} | {relative_mae_improvement:.4f} |".format(
                endpoint=row["endpoint"],
                n_train=int(row["n_train"]),
                n_eval=int(row["n_eval"]),
                mae=float(row["mae"]),
                rmse=float(row["rmse"]),
                r2=float(row["r2"]) if pd.notna(row["r2"]) else float("nan"),
                spearman=float(row["spearman"]) if pd.notna(row["spearman"]) else float("nan"),
                baseline_mae=float(row["baseline_mae"]),
                relative_mae_improvement=float(row["relative_mae_improvement"]),
            )
        )

    split_lines = [
        "| Split | molecules | labels | " + " | ".join(tasks) + " |",
        "|---|---:|---:" + "|---:" * len(tasks) + "|",
    ]
    for split_name in ["train", "test"]:
        item = summary["split_summary"][split_name]
        endpoint_cells = " | ".join(str(item["endpoint_counts"].get(task, 0)) for task in tasks)
        split_lines.append(
            f"| {split_name} | {item['molecules']} | {item['labels']} | {endpoint_cells} |"
        )

    lines = [
        "# PFAS Random-Split TabPFN Single Run",
        "",
        f"- Wide CSV: `{args.wide_csv}`",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Tasks: {', '.join(tasks)}",
        f"- Split: molecule-random train/test with seed `{args.seed}` and test fraction `{args.test_fraction}`",
        f"- Model: `TabPFNRegressor`, RDKit 2D descriptors, n_estimators `{args.n_estimators}`",
        f"- Total PFAS molecules: {len(wide)}",
        "",
        "## Split Summary",
        "",
        *split_lines,
        "",
        "## Test Metrics",
        "",
        *rows,
        "",
        "## Summary",
        "",
        f"- Test endpoints beating train-median baseline: {summary['test_endpoints_beating_baseline']} / {len(metrics)}",
        f"- Mean relative MAE improvement vs train-median baseline: {summary['test_relative_mae_improvement_mean']:.4f}",
        "",
        "This is one fixed random split experiment, not a scaffold split, not hyperparameter tuning, and not a multi-seed stability estimate.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    wide_raw = pd.read_csv(args.wide_csv)
    tasks = [task for task in args.tasks if task in wide_raw.columns]
    if not tasks:
        raise ValueError("No requested endpoint columns were found.")
    if "qsar_smiles" not in wide_raw.columns:
        raise ValueError("wide CSV must contain `qsar_smiles`.")

    wide = assign_random_split(wide_raw, test_fraction=args.test_fraction, seed=args.seed)
    print(f"[{utc_now()}] PFAS random split: {split_summary(wide, tasks)}", flush=True)
    descriptors, desc_names, invalid_indices = calculate_rdkit_descriptors(wide["qsar_smiles"])

    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    feature_meta: dict[str, Any] = {}

    for task in tasks:
        train_mask = wide["split"].eq("train") & wide[task].notna()
        test_mask = wide["split"].eq("test") & wide[task].notna()
        train_idx = np.flatnonzero(train_mask.to_numpy())
        test_idx = np.flatnonzero(test_mask.to_numpy())
        if len(train_idx) < 10 or len(test_idx) == 0:
            continue

        print(
            f"[{utc_now()}] endpoint={task} train={len(train_idx)} test={len(test_idx)}",
            flush=True,
        )
        x_train, x_test, meta = prepare_features(
            descriptors,
            train_idx=train_idx,
            eval_idx=test_idx,
            feature_names=desc_names,
        )
        feature_meta[task] = meta
        y_train = wide.loc[train_mask, task].astype(float).to_numpy()
        y_true = wide.loc[test_mask, task].astype(float).to_numpy()

        model = make_model(args)
        model.fit(x_train, y_train)
        y_pred = predict_in_chunks(model, x_test, args.prediction_chunk_size)
        baseline = np.full_like(y_true, float(np.median(y_train)), dtype=float)
        row = {
            "endpoint": task,
            "split": "test",
            "n_train": int(len(train_idx)),
            "target_median_train": float(np.median(y_train)),
            "target_mean_train": float(np.mean(y_train)),
            "target_std_train": float(np.std(y_train) or 1.0),
            "model": f"tabpfn_rdkit2d_nest{args.n_estimators}_random{args.test_fraction:g}",
            "selected_features": int(meta["selected_features"]),
            **regression_metrics(y_true, y_pred, baseline),
        }
        metrics_rows.append(row)

        pred_frame = wide.loc[test_mask, ["qsar_smiles", "split", task]].copy()
        if "source_split" in wide.columns:
            pred_frame["source_split"] = wide.loc[test_mask, "source_split"].to_numpy()
        pred_frame = pred_frame.rename(columns={task: "target"})
        pred_frame["endpoint"] = task
        pred_frame["prediction"] = y_pred
        pred_frame["baseline_prediction"] = baseline
        prediction_frames.append(pred_frame)

        del model
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    test_metrics = metrics[metrics["split"].eq("test")]
    summary = {
        "wide_csv": str(args.wide_csv),
        "checkpoint": str(args.checkpoint),
        "tasks": tasks,
        "n_molecules": int(len(wide)),
        "split_policy": {
            "type": "PFAS-only molecule-random train/test split",
            "seed": int(args.seed),
            "test_fraction": float(args.test_fraction),
            "uses_pfas_scaffold": False,
            "uses_original_source_split": False,
        },
        "split_summary": split_summary(wide, tasks),
        "model": {
            "name": "TabPFNRegressor",
            "feature_type": "RDKit 2D descriptors",
            "n_estimators": int(args.n_estimators),
            "device": args.device,
            "fit_mode": args.fit_mode,
            "memory_saving_mode": args.memory_saving_mode,
            "subsample_samples": int(args.subsample_samples),
        },
        "descriptor_metadata": {
            "input_features": int(descriptors.shape[1]),
            "invalid_smiles_count": int(len(invalid_indices)),
            "invalid_smiles_indices": invalid_indices,
            "per_endpoint": feature_meta,
        },
        "test_endpoints_beating_baseline": int((test_metrics["mae_improvement_vs_baseline"] > 0).sum())
        if len(test_metrics)
        else 0,
        "test_relative_mae_improvement_mean": float(test_metrics["relative_mae_improvement"].mean())
        if len(test_metrics)
        else None,
        "metrics": metrics.to_dict(orient="records"),
        "created_at": utc_now(),
        "argv": sys.argv,
    }

    wide.to_csv(args.outdir / "pfas_random_wide.csv", index=False)
    wide[["qsar_smiles", "split"] + (["source_split"] if "source_split" in wide.columns else [])].to_csv(
        args.outdir / "random_split_assignments.csv",
        index=False,
    )
    metrics.to_csv(args.outdir / "metrics.csv", index=False)
    predictions.to_csv(args.outdir / "predictions.csv", index=False)
    write_json(args.outdir / "summary.json", summary)
    write_report(args.outdir / "report.md", args=args, tasks=tasks, wide=wide, metrics=metrics, summary=summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
