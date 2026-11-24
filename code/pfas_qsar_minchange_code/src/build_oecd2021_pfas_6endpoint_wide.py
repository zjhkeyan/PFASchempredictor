from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem, RDLogger


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLEAN_LONG = REPO_ROOT / "runs" / "filtered_modeling_merge_no_hlc_20260608" / "modeling_long.csv"
DEFAULT_HLC_WIDE = REPO_ROOT / "runs" / "est_kh_random_tabpfn_20260608" / "est_kh_molecule_wide.csv"
DEFAULT_OUTDIR = REPO_ROOT / "runs" / "oecd2021_pfas_6endpoint_wide_20260616"
TASKS = ["TBP_C", "TMP_C", "logSW", "logPLV", "logKOW", "logKH"]


def oecd2021_pfas(smiles: object) -> bool:
    if pd.isna(smiles):
        return False
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C":
            continue
        if sum(neighbor.GetSymbol() == "F" for neighbor in atom.GetNeighbors()) < 2:
            continue
        if atom.GetTotalNumHs(includeNeighbors=True) > 0:
            continue
        if any(neighbor.GetSymbol() in {"H", "Cl", "Br", "I"} for neighbor in atom.GetNeighbors()):
            continue
        return True
    return False


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype("string").str.lower().isin(["true", "1", "yes", "y"])


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an OECD 2021 PFAS-only six-endpoint wide table.")
    parser.add_argument("--clean-long", type=Path, default=DEFAULT_CLEAN_LONG)
    parser.add_argument("--hlc-wide", type=Path, default=DEFAULT_HLC_WIDE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.*")

    clean = pd.read_csv(args.clean_long)
    clean = clean.loc[clean["endpoint"].isin(TASKS[:-1])].copy()
    clean["oecd2021_pfas"] = clean["qsar_smiles"].map(oecd2021_pfas)
    clean = clean.loc[clean["oecd2021_pfas"]].copy()
    clean["source_table"] = "clean_merge_no_hlc"

    hlc = pd.read_csv(args.hlc_wide)
    hlc_long = hlc.rename(columns={"logKH": "target"}).copy()
    hlc_long["endpoint"] = "logKH"
    hlc_long["oecd2021_pfas"] = hlc_long["qsar_smiles"].map(oecd2021_pfas)
    hlc_long = hlc_long.loc[hlc_long["oecd2021_pfas"]].copy()
    hlc_long["source_table"] = "est_kh_random_tabpfn_20260608"
    for column in ["n_contributors", "source_family_n", "source_priority_max"]:
        if column not in hlc_long.columns:
            hlc_long[column] = pd.NA
    if "is_pfas" not in hlc_long.columns:
        hlc_long["is_pfas"] = False
    if "is_fluorinated_local" not in hlc_long.columns:
        hlc_long["is_fluorinated_local"] = False

    long_cols = [
        "endpoint",
        "qsar_smiles",
        "target",
        "oecd2021_pfas",
        "is_pfas",
        "is_fluorinated_local",
        "source_table",
    ]
    for optional in ["inchikey", "chemical_name", "casrn", "split"]:
        if optional in clean.columns or optional in hlc_long.columns:
            if optional not in clean.columns:
                clean[optional] = pd.NA
            if optional not in hlc_long.columns:
                hlc_long[optional] = pd.NA
            long_cols.append(optional)

    long_df = pd.concat([clean[long_cols], hlc_long[long_cols]], ignore_index=True)
    wide_values = long_df.pivot_table(index="qsar_smiles", columns="endpoint", values="target", aggfunc="median")
    meta = (
        long_df.groupby("qsar_smiles", dropna=False)
        .agg(
            oecd2021_pfas=("oecd2021_pfas", "max"),
            is_pfas=("is_pfas", "max"),
            is_fluorinated_local=("is_fluorinated_local", "max"),
            source_tables=("source_table", lambda x: ";".join(sorted(set(map(str, x.dropna()))))),
            endpoint_count=("endpoint", "nunique"),
        )
        .reset_index()
    )
    for optional in ["inchikey", "chemical_name", "casrn"]:
        if optional in long_df.columns:
            first_values = long_df.dropna(subset=[optional]).drop_duplicates("qsar_smiles").set_index("qsar_smiles")[optional]
            meta[optional] = meta["qsar_smiles"].map(first_values.to_dict())
    wide = meta.merge(wide_values.reset_index(), on="qsar_smiles", how="left")

    endpoint_counts = {
        endpoint: int(long_df.loc[long_df["endpoint"].eq(endpoint), "qsar_smiles"].nunique()) for endpoint in TASKS
    }
    label_counts = {endpoint: int(long_df["endpoint"].eq(endpoint).sum()) for endpoint in TASKS}
    summary = {
        "definition": "OECD 2021 structural PFAS proxy: at least one carbon has >=2 attached F atoms and no H/Cl/Br/I attached to that carbon.",
        "tasks": TASKS,
        "n_molecules": int(len(wide)),
        "n_endpoint_labels": int(wide[TASKS].notna().sum().sum()),
        "endpoint_molecule_counts": endpoint_counts,
        "endpoint_label_counts": label_counts,
        "current_list_pfas_molecules_within_oecd": int(as_bool(wide["is_pfas"]).sum()),
        "oecd_only_molecules_vs_current_list": int((~as_bool(wide["is_pfas"])).sum()),
        "clean_long": str(args.clean_long),
        "hlc_wide": str(args.hlc_wide),
    }

    long_df.to_csv(args.outdir / "oecd2021_pfas_6endpoint_long.csv", index=False)
    wide.to_csv(args.outdir / "oecd2021_pfas_6endpoint_wide.csv", index=False)
    write_json(args.outdir / "summary.json", summary)
    report = [
        "# OECD 2021 PFAS Six-Endpoint Wide Table",
        "",
        f"- Molecules: `{summary['n_molecules']}`",
        f"- Endpoint labels: `{summary['n_endpoint_labels']}`",
        "",
        "| endpoint | molecule labels |",
        "|---|---:|",
    ]
    for endpoint in TASKS:
        report.append(f"| {endpoint} | {endpoint_counts[endpoint]} |")
    report.extend(
        [
            "",
            f"- Current-list PFAS within OECD set: `{summary['current_list_pfas_molecules_within_oecd']}`",
            f"- OECD-only molecules vs current list: `{summary['oecd_only_molecules_vs_current_list']}`",
            "",
        ]
    )
    (args.outdir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
