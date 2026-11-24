# PFAS / non-PFAS 六端点结构匹配数据集

## 构造规则

- PFAS 主集来自 `17049957` 的当前六端点 OECD PFAS 数据。
- `WS` 和 `HLC` 的不足部分来自同一 PFAS 工作流中的 EST PFAS 补充记录；按同端点基础 PFAS 的最大 Morgan/Tanimoto 相似度选取，且排除结构重复。
- non-PFAS 候选来自 `Predictors-for-15-Environmental-Endpoints` 的 `six_endpoint_oecd2021_annotated_long.csv`，同时排除 OECD 结构规则命中与 PFASSTRUCT/list 命中记录。
- 每个端点独立按同端点 PFAS 结构计算 Morgan 半径 2、2048-bit Tanimoto 相似度；按最大相似度排序，top-3 平均相似度作为并列规则。
- 每个端点取用户指定数量；一个分子可在不同端点出现，但同一端点内 canonical SMILES 不重复。

## 数量核对

| Property | PFAS | non-PFAS |
|---|---:|---:|
| WS | 270 | 1786 |
| LogKow | 519 | 2554 |
| HLC | 102 | 2495 |
| VP | 194 | 2717 |
| BP | 481 | 1763 |
| MP | 1218 | 2949 |

## PFAS 补足审计

| Property | target | base | added | candidate pool |
|---|---:|---:|---:|---:|
| WS | 270 | 269 | 1 | 15 |
| LogKow | 519 | 519 | 0 | 0 |
| HLC | 102 | 88 | 14 | 14 |
| VP | 194 | 194 | 0 | 0 |
| BP | 481 | 480 | 1 | 76 |
| MP | 1218 | 1217 | 1 | 67 |

## non-PFAS 相似度截断

本版固定保留 `90%` 最高相似度候选，并从原截断线以下约 `N` 到 `1.2N` 的秩区间均匀抽取 `10%`，数量与原版相同。

| Property | selected | top 90% | lower tail | pool | min selected | median selected | max selected |
|---|---:|---:|---:|---:|---:|---:|---:|
| WS | 1786 | 1607 | 179 | 9061 | 0.2812 | 0.3519 | 0.8421 |
| LogKow | 2554 | 2299 | 255 | 15033 | 0.3725 | 0.5000 | 0.8421 |
| HLC | 2495 | 2245 | 250 | 3480 | 0.1304 | 0.1957 | 0.8421 |
| VP | 2717 | 2445 | 272 | 3188 | 0.0328 | 0.2157 | 0.8421 |
| BP | 1763 | 1587 | 176 | 6350 | 0.2973 | 0.3824 | 0.6667 |
| MP | 2949 | 2654 | 295 | 27836 | 0.4524 | 0.5294 | 0.8421 |

## 输出

- `pfas_long.csv` / `nonpfas_long.csv`：端点级建模长表。
- `combined_long.csv`：两类合并长表。
- `pfas_wide.csv` / `nonpfas_wide.csv` / `combined_wide.csv`：多端点宽表。
- `by_property/`：12 个逐端点 CSV。
- `nonpfas_all_candidates_ranked.csv`：完整候选排序及是否入选。
- `pfas_nonpfas_similarity_matched.xlsx`：汇总与逐端点工作表。

## 与原始 top-N 版本比较

| Property | replaced | r1 mean similarity | r2 mean similarity | delta |
|---|---:|---:|---:|---:|
| WS | 119 | 0.378408 | 0.377670 | -0.000738 |
| LogKow | 170 | 0.523880 | 0.522497 | -0.001383 |
| HLC | 166 | 0.218766 | 0.217834 | -0.000932 |
| VP | 172 | 0.236734 | 0.234342 | -0.002392 |
| BP | 117 | 0.402803 | 0.401631 | -0.001173 |
| MP | 196 | 0.540610 | 0.539678 | -0.000933 |
