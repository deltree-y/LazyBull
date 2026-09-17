# stk_holdertrade（股东增减持）因子体检与诊断结论（Phase 3）

> 日期：2026-09-17　版本：v0.124.0
> 扫描口径：2020-01-01 ~ 2026-07-02，每 3 交易日采样（525 个分区），主板股票池 3,482 只
> 特征清单：生产基线 `v24123_features.json`（158 列，剔除 4 个运行时派生 availability marker）+ 10 个本族列 = **164 列**
> 标签：`neu_y_ret_20`；市场波动列：`mkt_vol_20`；规模代理：`zscore_size`
> 复现命令：`python scripts/materialize_factor_health_snapshot.py --start 20200101 --end 20260702 --every 3 --with-holdertrade --feature-file data/models/stock_selection/v24123_features.json --skip-usage --out-root temp/ht_health_root_20260917`
> 产物归档：`data/reports/factor_health/holdertrade_20260917/`、`data/reports/factor_diagnosis/holdertrade_20260917/`

## 0. 为什么需要"快照物化"

本族列按契约**运行时派生**（不写入生产 `features/cs_train`），而体检/诊断工具只读分区。
`scripts/materialize_factor_health_snapshot.py` 把「特征清单 + 本族列」物化到独立临时数据根
（只保留工具实际读取的列，2.29 GB）、以目录联接复用生产 `clean/raw/models`，再把工具的
`data.root` 指向该根运行——**不触碰任何生产数据**，工具口径与生产完全一致。

## 1. 覆盖率（全期与逐年）

| 列 | 全期覆盖 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| 8 个因子列（含 `ht_net_ratio_*` / `ht_*_count_*` / `accel`） | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | ✅ 设计性全覆盖（窗口外显式 0） |
| `ht_freshness_days` | 0.200 | 0.248 | 0.230 | 0.215 | 0.176 | 0.144 | 0.190 | 0.196 | ⚠️ 设计性缺失（无事件 ⇒ NaN），非数据缺口 |
| `holdertrade_schema_v1` | 1.000 | — | — | — | — | — | — | — | 哨兵（常数，判为 market_level，不参与截面 IC） |

> `ht_freshness_days` 被诊断工具归入 **A 覆盖缺口**（证据：覆盖率 0.200、有值/缺失标签中位差 −0.0031、
> 规模分位差 −0.006、根因判定"正常结构性"、预期效应量级 列级 +2~5pp）。**按本族契约这是设计语义**
> （窗口无事件即无新鲜度），**不做"补数据"**；若日后要显式化"是否存在近期公告"，
> 应走样本域/标签载体而非再加 0/1 列（参见 v0.121.0 可用性标记实测结论）。

## 2. 截面 IC（RankIC vs `neu_y_ret_20`，525 日）

| 列 | IC 均值 | t | IR | \|IC\| 分位* | 逐年符号 | 定级 |
|---|---|---|---|---|---|---|
| `ht_net_ratio_90d` | **+0.0213** | +15.2 | 0.66 | 66% | 7/7 正 | 强 |
| `ht_net_count_90d` | **+0.0207** | +15.0 | 0.65 | 64% | 7/7 正 | 强（与上者同簇） |
| `ht_sell_count_30d` | −0.0120 | −10.9 | −0.48 | 41% | 6/7 负 | 中·**负向** |
| `ht_net_ratio_30d` | +0.0105 | +8.8 | 0.38 | 38% | 6/7 正 | 中 |
| `ht_net_ratio_30d_other` | +0.0095 | +8.4 | 0.37 | 33% | 6/7 正 | 中（与上者同簇） |
| `ht_net_ratio_accel` | −0.0087 | −8.4 | −0.37 | 29% | **7/7 负** | 中·负向（见 §4 警告） |
| `ht_freshness_days` | +0.0066 | +3.6 | 0.16 | 23% | 符号翻转 | 弱/不稳 |
| `ht_net_ratio_30d_exec` | +0.0053 | +5.0 | 0.22 | 19% | 6/7 正 | 弱-中 |
| `ht_buy_count_30d` | −0.0003 | −0.32 | −0.01 | **0.7%** | 翻号 | **弱（唯一建议删除列）** |

\* 146 个可判 IC 列的 \|IC\| 分位：中位 0.0145、P75 0.0308、P90 0.0539、最高 0.0887
（Top 多为 `zscore_amount_ma20` / 波动率 / 换手率等负向流动性-波动因子）。

**读法**：本族最强列（+0.021）≈ 全样本 66 分位，属"中上但非头部"；符号 7/7 一致说明
不是单年噪声；`ht_sell_count_30d` 的负号（持续减持 ⇒ 后续下跌）是经济学最可解释的一列。

## 3. 与既有因子的关系（冗余 vs 新颖）

- **家族内**：`ht_net_ratio_90d` ~ `ht_net_count_90d`（簇 6）、`ht_net_ratio_30d` ~ `ht_net_ratio_30d_other`（簇 7）
  各构成一对孪生 → 体检标记为 **dedup 候选**。
- **但与外部**：每列与 154 个既有列的 **最大 |ρ| 仅 0.05~0.15**（最高为 `list_days` 0.149、
  `cons_eps_yield_fy0` 0.117）⇒ **本族与现有家族基本正交**，没有任何一列被并入既有簇。
- **偏 IC（控制簇代表后的秩残差）**：两个"孪生"仍保留增量信息——
  `ht_net_ratio_30d_other` 对 `ht_net_ratio_30d` 偏 IC **+0.0060（t=6.2）**、
  `ht_net_count_90d` 对 `ht_net_ratio_90d` 偏 IC **+0.0063（t=6.9）**。
  ⇒ **仅凭高相关就删列会丢信息**（正是 v0.120.0 因子诊断契约的硬定义：偏 IC 才是冗余判据）。

## 4. 使用度（模型是否真的用）

体检/诊断工具只加载 `_model.joblib`，而冒烟折模型落盘为 `_model.json`（Booster），
因此工具侧使用度为"缺失"（`gain_present_ratio = 0`，**不可读作"未被使用"**）。
补充统计（`ht_usage_manual.csv`，读取 JSON booster 的 gain/weight）：

| 列 | v24124 gain 份额 | v24125 gain 份额 | 分裂次数（v24124） |
|---|---|---|---|
| `ht_net_count_90d` | 1.33% | 0.86% | 38 |
| `ht_net_ratio_90d` | 1.12% | 1.05% | 118 |
| `ht_net_ratio_accel` | 0.90% | 0.61% | 53 |
| `ht_net_ratio_30d_exec` | 0.76% | 0.54% | 19 |
| `ht_net_ratio_30d_other` | 0.61% | 0.36% | 12 |
| `ht_net_ratio_30d` | 0.44% | 0.45% | 22 |
| `ht_buy_count_30d` | 0.36% | 0.48% | 5 |
| `ht_sell_count_30d` | 0.21% | 0.40% | 2 |
| **合计** | **≈5.7%** | **≈4.8%** | — |

对照：可用性标记 4 列在 14 折实验里的 gain 份额合计仅 ≈0.78%（`has_fund_holding` 0 次分裂）。
本族在**60 列**的冒烟模型里拿到 ≈5% 份额，说明模型确实在使用这些列（但该模型是单折单种子、
且 `best_iteration` 偏低，**只作方向性证据**，不可作为收益证据）。

⚠️ `ht_net_ratio_accel = net30 − net90/3` 是净额的线性组合，**其负 IC 与正净额因子负相关
（镜像）**，不能按符号解释为"加速买入 ⇒ 下跌"；其价值只能由偏 IC / A-B 判定。

## 5. 结论与下一步

1. **保留本族**：无任何列被诊断为"无增量信息"（B）、"口径退化"（F）或"使用不稳定"（D）；
   与既有因子正交（max |ρ| ≤ 0.15）；最强两列 IC t≈15、7/7 年同号。
2. **`ht_buy_count_30d` 是唯一弱列**（t=−0.32，|IC| 分位 0.7%）→ Phase 4 应作为**独立消融位**
   或先剔除后再 A/B（与 `ht_sell_count_30d` 形成"只留负向计数"的对照组）。
3. **孪生列不要按相关删**（偏 IC 仍有 0.6pp 量级信息，t≈6~7）。
4. **`ht_freshness_days` 属设计性缺失**，不进"补数据"清单；若模型不使用可随列集实验一并裁剪。
5. **效应量级仍属列级（0~1pp）** ⇒ 按 v0.120.0 契约**不可判定**，不得用净值口径直接追；
   合法路径 = **Phase 4 先过 v0.122.0 信号层尺子**（噪声带 ±10 bps ≈ ±9% 相对），
   相对降幅 ≥9% 才值得跑全量净值 A/B（约 2 h/臂）。

## 6. 复现与产物

| 环节 | 命令 / 文件 |
|---|---|
| 快照物化 + 体检 + 诊断 | `python scripts/materialize_factor_health_snapshot.py --start 20200101 --end 20260702 --every 3 --with-holdertrade --feature-file data/models/stock_selection/v24123_features.json --skip-usage --usage-model-count 2 --out-root temp/ht_health_root_20260917` |
| 台账（164 列） | `data/reports/factor_health/holdertrade_20260917/factor_register.csv` |
| IC 定位 | 同上目录 `ht_ic_positioning.csv` |
| 使用度（补偿统计） | 同上目录 `ht_usage_manual.csv` |
| 诊断候选清单 | `data/reports/factor_diagnosis/holdertrade_20260917/数据改良候选清单.csv` |
| 特征清单 | 同上目录 `feature_file_with_holdertrade.json`（164 列） |
