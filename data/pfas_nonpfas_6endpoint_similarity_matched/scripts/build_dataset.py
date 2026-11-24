#!/usr/bin/env python3
"""Build exact-size PFAS and structurally matched non-PFAS six-endpoint datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator


PROJECT_ROOT = Path("/root/autodl-tmp/JCIMpaper")
PFAS_BASE = (
    PROJECT_ROOT
    / "17049957"
    / "paper_extracted"
    / "current_dataset_distribution_20260827"
    / "outputs"
    / "current_pfas_6endpoint_distribution_input.csv"
)
PFAS_SUPPLEMENT = (
    PROJECT_ROOT
    / "17049957"
    / "Excel summaries"
    / "Global models just PFAS"
    / "EST_PFAS_supplementary_structure_endpoint_records_444.csv"
)
ANNOTATED_LONG = (
    PROJECT_ROOT
    / "Predictors-for-15-Environmental-Endpoints"
    / "pfas_qsar_minchange_code"
    / "pfas_qsar_minchange_code"
    / "runs"
    / "oecd2021_pfas_definition_audit_20260616"
    / "six_endpoint_oecd2021_annotated_long.csv"
)
DEFAULT_OUTDIR = Path(__file__).resolve().parent

PROPERTIES = [
    ("WS", "logSW", "log10(mol/L)", 270, 1786),
    ("LogKow", "logKOW", "log10(Kow)", 519, 2554),
    ("HLC", "logKH", "log10(atm*m3/mol)", 102, 2495),
    ("VP", "logPLV", "log10(mmHg)", 194, 2717),
    ("BP", "TBP_C", "deg C", 481, 1763),
    ("MP", "TMP_C", "deg C", 1218, 2949),
]
PROPERTY_TO_INTERNAL = {display: internal for display, internal, _, _, _ in PROPERTIES}
UNIT = {display: unit for display, _, unit, _, _ in PROPERTIES}
PFAS_TARGET = {display: n for display, _, _, n, _ in PROPERTIES}
NONPFAS_TARGET = {display: n for display, _, _, _, n in PROPERTIES}

MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pfas-base", type=Path, default=PFAS_BASE)
    parser.add_argument("--pfas-supplement", type=Path, default=PFAS_SUPPLEMENT)
    parser.add_argument("--annotated-long", type=Path, default=ANNOTATED_LONG)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--lower-tail-fraction",
        type=float,
        default=0.10,
        help="Fraction of each non-PFAS selection intentionally drawn from below the top-N cutoff.",
    )
    parser.add_argument(
        "--lower-tail-span",
        type=float,
        default=0.20,
        help="Width of the lower-rank band as a fraction of N.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_smiles(value: object) -> str | None:
    if pd.isna(value):
        return None
    mol = Chem.MolFromSmiles(str(value))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def morgan_fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MORGAN_GENERATOR.GetFingerprint(mol)


def similarity_to_reference(smiles: str, reference_fps: list) -> tuple[float, float]:
    fingerprint = morgan_fingerprint(smiles)
    if fingerprint is None or not reference_fps:
        return float("nan"), float("nan")
    similarities = np.asarray(
        DataStructs.BulkTanimotoSimilarity(fingerprint, reference_fps),
        dtype=np.float64,
    )
    top3 = np.sort(similarities)[-min(3, len(similarities)) :]
    return float(similarities.max()), float(top3.mean())


def load_pfas_rows(args: argparse.Namespace) -> tuple[pd.DataFrame, list[dict]]:
    base = pd.read_csv(args.pfas_base)
    supplement = pd.read_csv(args.pfas_supplement)
    rows: list[dict] = []
    audit: list[dict] = []

    base_by_property: dict[str, pd.DataFrame] = {}
    for display, internal, unit, target, _ in PROPERTIES:
        frame = base.loc[base[display].notna(), ["qsar_smiles", display]].copy()
        frame["smiles"] = frame["qsar_smiles"].map(canonical_smiles)
        frame = frame.loc[frame["smiles"].notna()].copy()
        frame = frame.drop_duplicates("smiles", keep="first")
        frame = frame.rename(columns={display: "value"})
        for record in frame.itertuples(index=False):
            rows.append(
                {
                    "property": display,
                    "source_endpoint": internal,
                    "unit": unit,
                    "smiles": record.smiles,
                    "raw_smiles": record.qsar_smiles,
                    "value": float(record.value),
                    "compound_class": "PFAS",
                    "pfas_source": "17049957_current_oecd",
                    "source_record_id": "",
                    "source_dataset": str(args.pfas_base),
                    "est_is_fluorinated_local": pd.NA,
                }
            )
        base_by_property[display] = frame

    for display, internal, unit, target, _ in PROPERTIES:
        current = base_by_property[display]
        current_smiles = set(current["smiles"])
        needed = target - len(current)
        if needed < 0:
            raise ValueError(f"{display}: base PFAS set exceeds target by {-needed}.")
        if needed == 0:
            audit.append(
                {
                    "property": display,
                    "target_pfas": target,
                    "base_pfas": len(current),
                    "supplement_added": 0,
                    "candidate_pool": 0,
                    "selection_rule": "none required",
                }
            )
            continue

        candidates = supplement.loc[
            supplement["endpoint"].eq(internal)
            & supplement["jcim_compatible_value"].notna()
        ].copy()
        candidates["smiles"] = candidates["qsar_smiles"].map(canonical_smiles)
        candidates = candidates.loc[
            candidates["smiles"].notna()
            & ~candidates["smiles"].isin(current_smiles)
        ].copy()
        candidates = candidates.drop_duplicates("smiles", keep="first")
        if len(candidates) < needed:
            raise ValueError(
                f"{display}: need {needed} supplementary PFAS rows, "
                f"but only {len(candidates)} unique candidates are available."
            )

        reference_fps = [
            fingerprint
            for smiles in current["smiles"]
            if (fingerprint := morgan_fingerprint(smiles)) is not None
        ]
        scored = [
            similarity_to_reference(smiles, reference_fps)
            for smiles in candidates["smiles"]
        ]
        candidates["max_morgan_tanimoto"] = [item[0] for item in scored]
        candidates["top3_mean_morgan_tanimoto"] = [item[1] for item in scored]
        candidates = candidates.sort_values(
            ["max_morgan_tanimoto", "top3_mean_morgan_tanimoto", "smiles"],
            ascending=[False, False, True],
            kind="stable",
        )
        selected = candidates.head(needed)
        for record in selected.itertuples(index=False):
            rows.append(
                {
                    "property": display,
                    "source_endpoint": internal,
                    "unit": unit,
                    "smiles": record.smiles,
                    "raw_smiles": record.qsar_smiles,
                    "value": float(record.jcim_compatible_value),
                    "compound_class": "PFAS",
                    "pfas_source": "17049957_est_pfas_supplement",
                    "source_record_id": str(record.est_source_record_ids),
                    "source_dataset": str(args.pfas_supplement),
                    "est_is_fluorinated_local": bool(record.est_is_fluorinated_local),
                }
            )
        audit.append(
            {
                "property": display,
                "target_pfas": target,
                "base_pfas": len(current),
                "supplement_added": len(selected),
                "candidate_pool": len(candidates),
                "selection_rule": (
                    "highest maximum Morgan radius-2/2048 Tanimoto similarity "
                    "to same-endpoint base PFAS; top-3 mean is the tie-breaker"
                ),
            }
        )

    result = pd.DataFrame(rows)
    result["selection_rank"] = pd.NA
    result["max_morgan_tanimoto"] = pd.NA
    result["top3_mean_morgan_tanimoto"] = pd.NA
    result = result.sort_values(["property", "pfas_source", "smiles"]).reset_index(drop=True)
    return result, audit


def load_nonpfas_matches(
    annotated: pd.DataFrame,
    pfas_rows: pd.DataFrame,
    *,
    lower_tail_fraction: float,
    lower_tail_span: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    nonpfas_pool = annotated.loc[
        ~annotated["oecd2021_pfas"].astype(bool)
        & ~annotated["is_pfas"].astype(bool)
    ].copy()
    nonpfas_pool["smiles"] = nonpfas_pool["qsar_smiles"].map(canonical_smiles)
    nonpfas_pool = nonpfas_pool.loc[nonpfas_pool["smiles"].notna()].copy()

    selected_frames: list[pd.DataFrame] = []
    ranked_frames: list[pd.DataFrame] = []
    audit: list[dict] = []

    for display, internal, unit, _, target in PROPERTIES:
        reference_fps = [
            fingerprint
            for smiles in pfas_rows.loc[pfas_rows["property"].eq(display), "smiles"]
            if (fingerprint := morgan_fingerprint(smiles)) is not None
        ]
        candidates = nonpfas_pool.loc[nonpfas_pool["endpoint"].eq(internal)].copy()
        candidates = (
            candidates.groupby("smiles", as_index=False)
            .agg(
                value=("target", "median"),
                split=("split", "first"),
                is_fluorinated_local=("is_fluorinated_local", "first"),
            )
        )
        if len(candidates) < target:
            raise ValueError(
                f"{display}: target {target} exceeds {len(candidates)} non-PFAS candidates."
            )
        scored = [
            similarity_to_reference(smiles, reference_fps)
            for smiles in candidates["smiles"]
        ]
        candidates["max_morgan_tanimoto"] = [item[0] for item in scored]
        candidates["top3_mean_morgan_tanimoto"] = [item[1] for item in scored]
        candidates = candidates.sort_values(
            ["max_morgan_tanimoto", "top3_mean_morgan_tanimoto", "smiles"],
            ascending=[False, False, True],
            kind="stable",
        ).reset_index(drop=True)
        candidates["selection_rank"] = np.arange(1, len(candidates) + 1)
        lower_n = int(round(target * lower_tail_fraction))
        core_n = target - lower_n
        band_end = min(
            len(candidates),
            max(core_n + lower_n, int(round(target * (1.0 + lower_tail_span)))),
        )
        if lower_n and band_end - core_n < lower_n:
            raise ValueError(
                f"{display}: lower-similarity band has {band_end - core_n} ranks, "
                f"but {lower_n} selections are requested."
            )
        lower_ranks = (
            np.unique(np.linspace(core_n + 1, band_end, lower_n, dtype=int)).tolist()
            if lower_n
            else []
        )
        candidates["selection_stratum"] = "unselected"
        candidates.loc[candidates["selection_rank"].le(core_n), "selection_stratum"] = (
            "top_core"
        )
        candidates.loc[candidates["selection_rank"].isin(lower_ranks), "selection_stratum"] = (
            "lower_tail"
        )
        candidates["selected"] = candidates["selection_stratum"].ne("unselected")
        candidates["property"] = display
        candidates["source_endpoint"] = internal
        candidates["unit"] = unit
        candidates["compound_class"] = "Non-PFAS"
        candidates["pfas_source"] = ""
        candidates["source_record_id"] = ""
        candidates["source_dataset"] = str(ANNOTATED_LONG)
        candidates["raw_smiles"] = candidates["smiles"]
        ranked_frames.append(candidates)

        chosen = candidates.loc[candidates["selected"]].copy()
        selected_frames.append(
            chosen[
                [
                    "property",
                    "source_endpoint",
                    "unit",
                    "smiles",
                    "raw_smiles",
                    "value",
                    "compound_class",
                    "pfas_source",
                    "source_record_id",
                    "source_dataset",
                    "is_fluorinated_local",
                    "selection_rank",
                    "max_morgan_tanimoto",
                    "top3_mean_morgan_tanimoto",
                    "selection_stratum",
                ]
            ].rename(columns={"is_fluorinated_local": "est_is_fluorinated_local"})
        )
        audit.append(
            {
                "property": display,
                "target_nonpfas": target,
                "candidate_pool": len(candidates),
                "selected": len(chosen),
                "top_core_selected": int(chosen["selection_stratum"].eq("top_core").sum()),
                "lower_tail_selected": int(
                    chosen["selection_stratum"].eq("lower_tail").sum()
                ),
                "min_selected_similarity": float(chosen["max_morgan_tanimoto"].min()),
                "median_selected_similarity": float(chosen["max_morgan_tanimoto"].median()),
                "max_selected_similarity": float(chosen["max_morgan_tanimoto"].max()),
                "next_unselected_similarity": (
                    float(candidates.iloc[target]["max_morgan_tanimoto"])
                    if len(candidates) > target
                    else float("nan")
                ),
            }
        )

    selected = pd.concat(selected_frames, ignore_index=True)
    ranked = pd.concat(ranked_frames, ignore_index=True)
    return selected, ranked, audit


def wide_table(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.pivot_table(
            index="smiles",
            columns="property",
            values="value",
            aggfunc="first",
        )
        .reindex(columns=[display for display, *_ in PROPERTIES])
        .reset_index()
        .rename(columns={"smiles": "SMILES"})
    )


def write_report(
    path: Path,
    pfas_counts: pd.DataFrame,
    nonpfas_counts: pd.DataFrame,
    pfas_audit: list[dict],
    nonpfas_audit: list[dict],
) -> None:
    lines = [
        "# PFAS / non-PFAS 六端点结构匹配数据集",
        "",
        "## 构造规则",
        "",
        "- PFAS 主集来自 `17049957` 的当前六端点 OECD PFAS 数据。",
        "- `WS` 和 `HLC` 的不足部分来自同一 PFAS 工作流中的 EST PFAS 补充记录；"
        "按同端点基础 PFAS 的最大 Morgan/Tanimoto 相似度选取，且排除结构重复。",
        "- non-PFAS 候选来自 `Predictors-for-15-Environmental-Endpoints` 的"
        " `six_endpoint_oecd2021_annotated_long.csv`，同时排除 OECD 结构规则命中"
        "与 PFASSTRUCT/list 命中记录。",
        "- 每个端点独立按同端点 PFAS 结构计算 Morgan 半径 2、2048-bit Tanimoto "
        "相似度；按最大相似度排序，top-3 平均相似度作为并列规则。",
        "- 每个端点取用户指定数量；一个分子可在不同端点出现，但同一端点内"
        " canonical SMILES 不重复。",
        "",
        "## 数量核对",
        "",
        "| Property | PFAS | non-PFAS |",
        "|---|---:|---:|",
    ]
    p = pfas_counts.set_index("property")["n"]
    n = nonpfas_counts.set_index("property")["n"]
    for display, *_ in PROPERTIES:
        lines.append(f"| {display} | {int(p[display])} | {int(n[display])} |")
    lines.extend(
        [
            "",
            "## PFAS 补足审计",
            "",
            "| Property | target | base | added | candidate pool |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in pfas_audit:
        lines.append(
            f"| {row['property']} | {row['target_pfas']} | {row['base_pfas']} | "
            f"{row['supplement_added']} | {row['candidate_pool']} |"
        )
    lines.extend(
        [
            "",
            "## non-PFAS 相似度截断",
            "",
            "本版固定保留 `90%` 最高相似度候选，并从原截断线以下约"
            " `N` 到 `1.2N` 的秩区间均匀抽取 `10%`，数量与原版相同。",
            "",
            "| Property | selected | top 90% | lower tail | pool | min selected | median selected | max selected |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in nonpfas_audit:
        lines.append(
            f"| {row['property']} | {row['selected']} | {row['top_core_selected']} | "
            f"{row['lower_tail_selected']} | {row['candidate_pool']} | "
            f"{row['min_selected_similarity']:.4f} | "
            f"{row['median_selected_similarity']:.4f} | "
            f"{row['max_selected_similarity']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 输出",
            "",
            "- `pfas_long.csv` / `nonpfas_long.csv`：端点级建模长表。",
            "- `combined_long.csv`：两类合并长表。",
            "- `pfas_wide.csv` / `nonpfas_wide.csv` / `combined_wide.csv`：多端点宽表。",
            "- `by_property/`：12 个逐端点 CSV。",
            "- `nonpfas_all_candidates_ranked.csv`：完整候选排序及是否入选。",
            "- `pfas_nonpfas_similarity_matched.xlsx`：汇总与逐端点工作表。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")
    args.outdir.mkdir(parents=True, exist_ok=True)
    by_property = args.outdir / "by_property"
    pfas_dir = by_property / "pfas"
    nonpfas_dir = by_property / "non_pfas"
    pfas_dir.mkdir(parents=True, exist_ok=True)
    nonpfas_dir.mkdir(parents=True, exist_ok=True)

    annotated = pd.read_csv(args.annotated_long)
    pfas_rows, pfas_audit = load_pfas_rows(args)
    nonpfas_rows, ranked, nonpfas_audit = load_nonpfas_matches(
        annotated,
        pfas_rows,
        lower_tail_fraction=args.lower_tail_fraction,
        lower_tail_span=args.lower_tail_span,
    )

    pfas_counts = pfas_rows.groupby("property", as_index=False).size().rename(columns={"size": "n"})
    nonpfas_counts = nonpfas_rows.groupby("property", as_index=False).size().rename(columns={"size": "n"})
    for display, *_ in PROPERTIES:
        pfas_n = int(pfas_counts.set_index("property").loc[display, "n"])
        nonpfas_n = int(nonpfas_counts.set_index("property").loc[display, "n"])
        if pfas_n != PFAS_TARGET[display]:
            raise AssertionError(f"{display}: PFAS={pfas_n}, expected {PFAS_TARGET[display]}")
        if nonpfas_n != NONPFAS_TARGET[display]:
            raise AssertionError(
                f"{display}: non-PFAS={nonpfas_n}, expected {NONPFAS_TARGET[display]}"
            )

    ordered_columns = [
        "property",
        "source_endpoint",
        "unit",
        "smiles",
        "raw_smiles",
        "value",
        "compound_class",
        "pfas_source",
        "source_record_id",
        "source_dataset",
        "est_is_fluorinated_local",
        "selection_rank",
        "max_morgan_tanimoto",
        "top3_mean_morgan_tanimoto",
        "selection_stratum",
    ]
    pfas_rows = pfas_rows.reindex(columns=ordered_columns)
    nonpfas_rows = nonpfas_rows.reindex(columns=ordered_columns)
    combined = pd.concat([pfas_rows, nonpfas_rows], ignore_index=True)
    combined = combined.sort_values(
        ["property", "compound_class", "selection_rank", "smiles"],
        na_position="last",
    ).reset_index(drop=True)

    pfas_rows.to_csv(args.outdir / "pfas_long.csv", index=False)
    nonpfas_rows.to_csv(args.outdir / "nonpfas_long.csv", index=False)
    combined.to_csv(args.outdir / "combined_long.csv", index=False)
    ranked.to_csv(args.outdir / "nonpfas_all_candidates_ranked.csv", index=False)
    wide_table(pfas_rows).to_csv(args.outdir / "pfas_wide.csv", index=False)
    wide_table(nonpfas_rows).to_csv(args.outdir / "nonpfas_wide.csv", index=False)
    wide_table(combined).to_csv(args.outdir / "combined_wide.csv", index=False)

    for display, *_ in PROPERTIES:
        pfas_rows.loc[pfas_rows["property"].eq(display)].to_csv(
            pfas_dir / f"{display}.csv",
            index=False,
        )
        nonpfas_rows.loc[nonpfas_rows["property"].eq(display)].to_csv(
            nonpfas_dir / f"{display}.csv",
            index=False,
        )

    counts = pfas_counts.merge(
        nonpfas_counts,
        on="property",
        how="outer",
        suffixes=("_pfas", "_nonpfas"),
    )
    counts = counts.rename(columns={"n_pfas": "PFAS", "n_nonpfas": "non-PFAS"})
    counts.to_csv(args.outdir / "matched_counts.csv", index=False)

    with pd.ExcelWriter(args.outdir / "pfas_nonpfas_similarity_matched.xlsx") as writer:
        counts.to_excel(writer, sheet_name="Summary", index=False)
        for display, *_ in PROPERTIES:
            pfas_rows.loc[pfas_rows["property"].eq(display)].to_excel(
                writer,
                sheet_name=f"PFAS_{display}",
                index=False,
            )
            nonpfas_rows.loc[nonpfas_rows["property"].eq(display)].to_excel(
                writer,
                sheet_name=f"NonPFAS_{display}",
                index=False,
            )

    write_report(
        args.outdir / "REPORT.md",
        pfas_counts,
        nonpfas_counts,
        pfas_audit,
        nonpfas_audit,
    )
    manifest = {
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__)),
        "sources": {
            str(path): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in [args.pfas_base, args.pfas_supplement, args.annotated_long]
        },
        "similarity": {
            "fingerprint": "Morgan",
            "radius": 2,
            "n_bits": 2048,
            "metric": "Tanimoto",
            "primary_rank": "maximum similarity to any same-endpoint PFAS",
            "tie_breaker": "mean of top-3 PFAS similarities",
            "lower_tail_fraction": args.lower_tail_fraction,
            "lower_tail_span": args.lower_tail_span,
            "selection_policy": (
                "top 90% by maximum similarity plus evenly spaced ranks from "
                "the next lower-similarity band"
            ),
        },
        "pfas_target_counts": PFAS_TARGET,
        "nonpfas_target_counts": NONPFAS_TARGET,
        "pfas_build_audit": pfas_audit,
        "nonpfas_selection_audit": nonpfas_audit,
    }
    (args.outdir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    r1_nonpfas = args.outdir.parent / "r1" / "nonpfas_long.csv"
    if r1_nonpfas.exists():
        previous = pd.read_csv(r1_nonpfas)
        comparison_rows = []
        for display, *_ in PROPERTIES:
            old = previous.loc[previous["property"].eq(display)]
            new = nonpfas_rows.loc[nonpfas_rows["property"].eq(display)]
            comparison_rows.append(
                {
                    "property": display,
                    "r1_n": len(old),
                    "r2_n": len(new),
                    "shared_smiles": len(set(old["smiles"]) & set(new["smiles"])),
                    "replaced_smiles": len(set(new["smiles"]) - set(old["smiles"])),
                    "r1_mean_max_similarity": float(old["max_morgan_tanimoto"].mean()),
                    "r2_mean_max_similarity": float(new["max_morgan_tanimoto"].mean()),
                    "mean_similarity_delta_r2_minus_r1": float(
                        new["max_morgan_tanimoto"].mean()
                        - old["max_morgan_tanimoto"].mean()
                    ),
                    "r1_min_similarity": float(old["max_morgan_tanimoto"].min()),
                    "r2_min_similarity": float(new["max_morgan_tanimoto"].min()),
                }
            )
        pd.DataFrame(comparison_rows).to_csv(
            args.outdir / "comparison_with_r1.csv",
            index=False,
        )
        comparison = pd.DataFrame(comparison_rows)
        lines = [
            "",
            "## 与原始 top-N 版本比较",
            "",
            "| Property | replaced | r1 mean similarity | r2 mean similarity | delta |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in comparison.itertuples(index=False):
            lines.append(
                f"| {row.property} | {row.replaced_smiles} | "
                f"{row.r1_mean_max_similarity:.6f} | "
                f"{row.r2_mean_max_similarity:.6f} | "
                f"{row.mean_similarity_delta_r2_minus_r1:+.6f} |"
            )
        lines.append("")
        with (args.outdir / "REPORT.md").open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
    print(counts.to_string(index=False))
    print(f"Output: {args.outdir}")


if __name__ == "__main__":
    main()
