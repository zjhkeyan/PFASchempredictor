# PFASchempredictor

**PFASchempredictor** is a reproducible QSAR modeling and dataset repository for predicting six key physicochemical properties of **per- and polyfluoroalkyl substances (PFAS)** via **TabPFN v3.5** in-context learning and **non-PFAS → PFAS transfer fine-tuning**.

It provides:
1. **Six-Endpoint Structurally Matched PFAS / non-PFAS Dataset** (`17,048` endpoint records across `12,420` unique molecules: `1,927` OECD-2021 PFAS and `10,493` structurally matched non-PFAS selected via Morgan ECFP4 2048-bit Tanimoto similarity with a 90% top-similarity + 10% lower-tail policy).
2. **TabPFN v3.5 Transfer Fine-Tuning & Baseline Pipelines** covering global non-PFAS gradient fine-tuning with local PFAS in-context inference, exact-split context-only TabPFN ablation, and PFAS-only random-split evaluation on RDKit 2D descriptors.
3. **Vendored TabPFN v3.5 Source & Verified Weight Downloader** (`download_tabpfn35_weight.sh`) with SHA-256 checkpoint verification.

---

## Repository Structure

```text
PFASchempredictor/
├── README.md                                  # Project documentation & reproduction guide
├── .gitignore                                 # Excludes downloaded checkpoints, runs/, and Python caches
├── download_tabpfn35_weight.sh                # Downloads & SHA-256-verifies TabPFN-v3.5 weights from Hugging Face
├── code/
│   ├── pfas_qsar_minchange_code/              # Core PFAS QSAR modeling & preprocessing codebase
│   │   ├── requirements.txt                   # Python dependencies
│   │   └── src/
│   │       ├── __init__.py
│   │       ├── chemistry.py                   # SMILES standardization, neutralization, InChIKey & Murcko scaffolds
│   │       ├── build_oecd2021_pfas_6endpoint_wide.py
│   │       │                                  # OECD-2021 structural PFAS classifier & 6-endpoint wide-table builder
│   │       ├── run_pfas_random_tabpfn.py      # PFAS-only random-split TabPFN + RDKit2D baseline
│   │       ├── run_oecd_pfas_nonpfas_finetuned_tabpfn.py
│   │       │                                  # Global non-PFAS fine-tuning + local OECD PFAS in-context inference
│   │       └── run_oecd_pfas_context_only_tabpfn_same_split.py
│   │                                          # No-gradient context-only TabPFN ablation on exact identical splits
│   └── TabPFN/                                # Vendored TabPFN v3.5 library (pyproject.toml, LICENSE, src/tabpfn/)
├── data/
│   └── pfas_nonpfas_6endpoint_similarity_matched/
│       ├── scripts/
│       │   └── build_dataset.py               # Exact-size Morgan-Tanimoto similarity-matched dataset builder
│       ├── tables/
│       │   ├── combined_long.csv              # 17,048 PFAS + non-PFAS endpoint records (long format)
│       │   ├── combined_wide.csv              # 12,420 unique molecules × 6 endpoints (wide format)
│       │   ├── pfas_long.csv                  # 2,784 PFAS endpoint records (long format)
│       │   ├── pfas_wide.csv                  # 1,927 unique PFAS molecules (wide format)
│       │   ├── nonpfas_long.csv               # 14,264 matched non-PFAS records (long format)
│       │   ├── nonpfas_wide.csv               # 10,493 unique non-PFAS molecules (wide format)
│       │   ├── nonpfas_all_candidates_ranked.csv
│       │   │                                  # Full 64,948-row non-PFAS candidate pool with similarity ranks & flags
│       │   └── pfas_nonpfas_similarity_matched.xlsx
│       │                                      # Multi-sheet Excel workbook (summary + per-endpoint sheets)
│       ├── by_property/
│       │   ├── pfas/                          # Endpoint-specific PFAS CSVs: WS, LogKow, HLC, VP, BP, MP
│       │   └── non_pfas/                      # Endpoint-specific matched non-PFAS CSVs: WS, LogKow, HLC, VP, BP, MP
│       └── metadata/
│           ├── DATASET_REPORT.md              # Detailed dataset construction & audit report (Chinese)
│           ├── matched_counts.csv             # Exact PFAS vs. non-PFAS counts per endpoint
│           ├── comparison_with_r1.csv         # Comparison between r2 (90%+10% tail) and r1 (strict top-N)
│           └── run_manifest.json              # Full provenance, source SHA-256 digests, and build audit
└── models/
    ├── config.json                            # TabPFN-v3.5 configuration stub
    ├── WEIGHT_DOWNLOAD.txt                    # Official Hugging Face URL & SHA-256 checksum
    └── TABPFN-3.5-LICENSE.txt                 # Prior Labs TabPFN-3.5 license terms
```

---

## Six-Endpoint Dataset Summary

All six physicochemical endpoints were curated with strict per-endpoint canonical SMILES deduplication and OECD-2021 PFAS structural filtering. For each endpoint independently, non-PFAS molecules (excluding both OECD-2021 structural PFAS hits and PFASSTRUCT/list matches) were ranked by **maximum Morgan fingerprint (radius 2, 2048-bit) Tanimoto similarity** to any same-endpoint PFAS (with top-3 mean similarity as tie-breaker), keeping the **top 90%** by similarity plus **10%** uniformly sampled across the lower-similarity rank band `[N, 1.2N]`.

| Property | Internal Task Name | Unit | PFAS (`n`) | Matched non-PFAS (`n`) | Candidate Pool | Median Selected $S_{\max}$ | Range $[S_{\min}, S_{\max}]$ |
|---|---|---|---:|---:|---:|---:|---|
| **WS** (Aqueous Solubility) | `logSW` | `log10(mol/L)` | 270 | 1,786 | 9,061 | 0.3519 | `[0.2812, 0.8421]` |
| **LogKow** (Octanol–Water Partition) | `logKOW` | `log10(Kow)` | 519 | 2,554 | 15,033 | 0.5000 | `[0.3725, 0.8421]` |
| **HLC** (Henry's Law Constant) | `logKH` | `log10(atm*m3/mol)` | 102 | 2,495 | 3,480 | 0.1957 | `[0.1304, 0.8421]` |
| **VP** (Vapor Pressure) | `logPLV` | `log10(mmHg)` | 194 | 2,717 | 3,188 | 0.2157 | `[0.0328, 0.8421]` |
| **BP** (Boiling Point) | `TBP_C` | `deg C` | 481 | 1,763 | 6,350 | 0.3824 | `[0.2973, 0.6667]` |
| **MP** (Melting Point) | `TMP_C` | `deg C` | 1,218 | 2,949 | 27,836 | 0.5294 | `[0.4524, 0.8421]` |
| **Total Records / Unique SMILES** | — | — | **2,784 / 1,927** | **14,264 / 10,493** | **64,948** | — | — |

- **Long tables** (`tables/{pfas,nonpfas,combined}_long.csv`): 15 columns including `property`, `source_endpoint`, `unit`, `smiles`, `raw_smiles`, `value`, `compound_class`, `pfas_source`, `selection_rank`, `max_morgan_tanimoto`, `top3_mean_morgan_tanimoto`, and `selection_stratum`.
- **Wide tables** (`tables/{pfas,nonpfas,combined}_wide.csv`): columns `SMILES, WS, LogKow, HLC, VP, BP, MP`.
- Full construction audit and similarity comparisons are documented in [`data/pfas_nonpfas_6endpoint_similarity_matched/metadata/DATASET_REPORT.md`](data/pfas_nonpfas_6endpoint_similarity_matched/metadata/DATASET_REPORT.md) and [`run_manifest.json`](data/pfas_nonpfas_6endpoint_similarity_matched/metadata/run_manifest.json).

---

## Installation & Setup

### 1. Environment & Python Dependencies

Python `>= 3.10` with CUDA-enabled PyTorch is recommended for TabPFN v3.5 fine-tuning.

```bash
conda create -n pfaschempredictor python=3.11 -y
conda activate pfaschempredictor

# Install modeling dependencies
pip install -r code/pfas_qsar_minchange_code/requirements.txt
pip install huggingface_hub

# Optional: install vendored TabPFN in editable mode (scripts also auto-load code/TabPFN/src)
pip install -e code/TabPFN --no-deps
```

### 2. Download TabPFN v3.5 Checkpoint

The official TabPFN v3.5 checkpoint is not stored directly in Git due to size and license terms. Accept the terms at [`Prior-Labs/tabpfn_3_5`](https://huggingface.co/Prior-Labs/tabpfn_3_5), authenticate via Hugging Face CLI, and run the verified download script:

```bash
huggingface-cli login
./download_tabpfn35_weight.sh
```

This saves the checkpoint to `models/tabpfn-v3.5-20260909.safetensors` and verifies its SHA-256 digest (`ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3`). See [`models/WEIGHT_DOWNLOAD.txt`](models/WEIGHT_DOWNLOAD.txt) for direct links.

---

## Usage & Reproducibility

All modeling scripts automatically resolve the vendored `code/TabPFN/src` package and `models/tabpfn-v3.5-20260909.safetensors` relative to the repository root (overridable via `TABPFN_SRC` and `TABPFN_CHECKPOINT` environment variables). Run commands from `code/pfas_qsar_minchange_code/` or with `PYTHONPATH=code/pfas_qsar_minchange_code`.

### 1. Global non-PFAS Gradient Fine-Tuning + Local OECD PFAS Inference

Fine-tunes TabPFN v3.5 on global non-OECD-PFAS molecules (`--main-wide`, `--hlc-wide`, `--tl-logkow-xlsx`) and evaluates in-context inference on held-out OECD-2021 PFAS splits across all six endpoints (`TBP_C`, `TMP_C`, `logSW`, `logPLV`, `logKOW`, `logKH`):

```bash
PYTHONPATH=code/pfas_qsar_minchange_code \
python code/pfas_qsar_minchange_code/src/run_oecd_pfas_nonpfas_finetuned_tabpfn.py \
  --main-wide <path/to/main_6endpoint_wide.csv> \
  --hlc-wide <path/to/hlc_wide.csv> \
  --tl-logkow-xlsx <path/to/tl_logkow.xlsx> \
  --checkpoint models/tabpfn-v3.5-20260909.safetensors \
  --outdir runs/oecd2021_pfas_nonpfas_finetuned_tabpfn \
  --device cuda
```

### 2. Exact-Split No-Gradient Context-Only TabPFN Ablation

Runs a zero-gradient TabPFN v3.5 baseline on the exact OECD PFAS context/test splits and RDKit 2D preprocessing produced by Step 1 (`--source-run`), cleanly isolating the gain from non-PFAS gradient fine-tuning:

```bash
PYTHONPATH=code/pfas_qsar_minchange_code \
python code/pfas_qsar_minchange_code/src/run_oecd_pfas_context_only_tabpfn_same_split.py \
  --source-run runs/oecd2021_pfas_nonpfas_finetuned_tabpfn \
  --checkpoint models/tabpfn-v3.5-20260909.safetensors \
  --outdir runs/oecd2021_pfas_context_only_baseline \
  --feature-reference nonpfas \
  --device cuda
```

### 3. PFAS-Only Random-Split TabPFN Baseline

Evaluates TabPFN v3.5 directly on a multi-endpoint PFAS wide table containing a `qsar_smiles` column (e.g., generated by `build_oecd2021_pfas_6endpoint_wide.py`, or after renaming `SMILES` → `qsar_smiles` in `tables/pfas_wide.csv`):

```bash
PYTHONPATH=code/pfas_qsar_minchange_code \
python code/pfas_qsar_minchange_code/src/run_pfas_random_tabpfn.py \
  --wide-csv <path/to/pfas_wide_with_qsar_smiles.csv> \
  --checkpoint models/tabpfn-v3.5-20260909.safetensors \
  --outdir runs/pfas_random_tabpfn \
  --tasks TBP_C TMP_C logSW logPLV logKOW \
  --device cuda
```

### 4. Rebuilding the Similarity-Matched Dataset (Optional)

The pre-built CSV and Excel files are ready to use under `data/pfas_nonpfas_6endpoint_similarity_matched/`. To regenerate them from raw upstream curation files:

```bash
python data/pfas_nonpfas_6endpoint_similarity_matched/scripts/build_dataset.py \
  --pfas-base <path/to/current_pfas_6endpoint_distribution_input.csv> \
  --pfas-supplement <path/to/EST_PFAS_supplementary_structure_endpoint_records_444.csv> \
  --annotated-long <path/to/six_endpoint_oecd2021_annotated_long.csv> \
  --outdir data/pfas_nonpfas_6endpoint_similarity_matched_rebuilt
```

---

## Licensing & Attribution

- **TabPFN v3.5 Code**: Vendored under `code/TabPFN/` per [`code/TabPFN/LICENSE`](code/TabPFN/LICENSE) and [`code/TabPFN/NOTICE`](code/TabPFN/NOTICE).
- **TabPFN v3.5 Weights**: Governed by the Prior Labs TabPFN-3.5 license ([`models/TABPFN-3.5-LICENSE.txt`](models/TABPFN-3.5-LICENSE.txt)).
- **PFAS QSAR Code & Curated Datasets**: Provided for reproducible environmental chemistry and cheminformatics research.
