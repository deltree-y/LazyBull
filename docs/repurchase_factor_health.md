# repurchase（股票回购）因子体检与诊断结论（Phase 3）

> 2026-09-18 · 口径定稿见 `docs/repurchase_pit_audit.md` §6.1 · 快照数据根 `temp/rp_health_root_20260918`（临时）
> 产物：`data/reports/factor_health/repurchase_20260918/`、`data/reports/factor_diagnosis/repurchase_20260918/`

## 0. 为什么需要"快照物化"

repurchase 家族按契约**运行时派生**，不写入生产 `features/cs_train`；而体检/诊断工具只读分区。
因此走与 stk_holdertrade 同一入口 `scripts/materialize_factor_health_snapshot.py`（本次已扩展
`--with-repurchase` + 家族登记表 `_runtime_family_specs`）。

**本次新增的处理**：repurchase 派生需要当日 `circ_mv` / `amount` / `vol`（VWAP = `amount × 10 ÷ vol`），
它们**不在**体检读取列里 ⇒ 物化时把这些**派生支撑列**读出、派生后从快照中裁掉（`read_cols` 与
`write_cols` 分离）。踩坑记录：首版把运行时列混进 `read_cols` 导致 `ArrowInvalid`（分区里没有这些列），
先修脚本再跑。

**基线清单**：`data/models/stock_selection/v24123_features.json`（与 stk_holdertrade Phase 3 同一基线，
保证百分位可比）+ 6 个 repurchase 列 = **160 列**、525 个采样交易日（20200102~20260702，每 3 个交易日取 1 天）。

## 1. 覆盖率（全期与逐年）

| 列 | 全期覆盖 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| `rp_amount_to_mv_90d` | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | ✅ 设计性全覆盖（窗口外显式 0） |
| `rp_amount_to_mv_180d` | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | ✅ 同上 |
| `rp_exec_flag_90d` | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | ✅ 同上 |
| `rp_price_headroom` | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | ✅ 同上 |
| `rp_freshness_days` | 0.232 | 0.233 | 0.242 | 0.223 | 0.184 | 0.190 | 0.181 | 0.171 | ⚠️ 设计性缺失（无公告 ⇒ NaN）；**入模时被 >0.6 缺失率门禁删除** |
| `repurchase_schema_v1` | 1.000 | — | — | — | — | — | — | — | 哨兵（常数，判为 market_level，不参与截面 IC；入模时被常数门禁删除） |

逐年覆盖**无收缩**（4 个值列全 7 年 100%）；`rp_freshness_days` 的 23.2% 是「180 日内有公告的股票占比」的
设计性结果，诊断判定根因 **正常（结构性）**，不做补数据。

## 2. 截面 IC（RankIC vs `neu_y_ret_20`，525 日）

| 列 | IC 均值 | t | IR | 逐年符号 | 全表 \|t\| 百分位* | 定级 |
|---|---|---|---|---|---|---|
| `rp_price_headroom` | **+0.0104** | **+7.90** | 0.34 | **7/7 正** | **59%**（160 列中位 t=5.9） | 中（**家族唯一可用列**） |
| `rp_amount_to_mv_90d` | +0.0024 | +1.50 | 0.07 | 2021–2023 负、其余正 | 18% | 弱·符号不稳 |
| `rp_exec_flag_90d` | +0.0016 | +0.97 | 0.04 | 2021–2023 负、其余正 | 10% | **弱**（同簇冗余） |
| `rp_amount_to_mv_180d` | +0.0010 | +0.56 | 0.02 | 半数年负 | **6%** | **弱（全表垫底区）** |
| `rp_freshness_days` | −0.0049 | −2.37 | −0.10 | 符号翻转 | 23%（覆盖 23%） | 弱·不稳·低覆盖（入模即被门禁删除） |

\* 百分位口径：与同一次扫描的 160 列 `|ic_t|` 分布对比（中位 5.89、75 分位 10.68）。
**参照系**：stk_holdertrade 最强列 `ht_net_ratio_90d`（t≈+15.2、7/7 正）在本表约 90 分位——即
**repurchase 家族最强列的强度不到已判定失败的持增族最强列的一半**。

逐年 IC 明细：

| 列 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|
| `rp_amount_to_mv_90d` | +0.0119 | −0.0096 | −0.0033 | −0.0200 | +0.0206 | +0.0032 | +0.0264 |
| `rp_amount_to_mv_180d` | +0.0077 | −0.0055 | −0.0053 | −0.0215 | +0.0137 | +0.0035 | +0.0282 |
| `rp_exec_flag_90d` | +0.0133 | −0.0099 | −0.0044 | −0.0222 | +0.0181 | +0.0023 | +0.0273 |
| `rp_price_headroom` | +0.0069 | +0.0056 | +0.0077 | +0.0210 | +0.0147 | +0.0028 | +0.0183 |
| `rp_freshness_days` | −0.0132 | +0.0224 | −0.0031 | +0.0107 | −0.0232 | −0.0089 | −0.0335 |

`rp_price_headroom` 是**唯一逐月为正**的列；3 个金额/状态列在 2021–2023（含 2022 熊市与 2023）
持续反向，属「跨期翻号」而非单年噪声。

## 3. 与既有因子的关系（冗余 vs 新颖）

| 列 | cluster_id | cluster_size | 簇代表 | 解读 |
|---|---|---|---|---|
| `rp_amount_to_mv_90d` | 6 | 2 | 自己 | 与 `rp_exec_flag_90d` 同簇（家族内共线） |
| `rp_exec_flag_90d` | 6 | 2 | `rp_amount_to_mv_90d` | **被判 flag_dup**（非代表成员） |
| `rp_amount_to_mv_180d` | 7 | 1 | 自己 | **不与任何既有特征同簇**（新颖轴，但很弱） |
| `rp_price_headroom` | 9 | 1 | 自己 | **不与任何既有特征同簇**（新颖轴，且最强） |
| `rp_freshness_days` | 8 | 1 | 自己 | 独立轴（低覆盖） |

⇒ repurchase 的金额轴与价格上限轴**不是既有因子的线性重复**（单元素簇）；问题不在"重复"，在"弱"。

## 4. 使用度（模型是否真的用）

**本轮未展开**：截至目前**尚无任何模型启用 `--enable-repurchase-features`**，所以
`usage_stability.csv` 中 4 列的 `gain_present_ratio = 0.0`、`gain_mean = NaN`——
**这是"尚未观测"，不得读作"模型不用"**（与 stk_holdertrade 同一条口径约束）。
使用度必须在 A 臂（启用开关）训出模型后另行统计。

## 5. 偏 IC 与诊断命中

《数据改良候选清单》（16 条）中**只有 1 条命中本族**：

| 类型 | 因子 | 证据 | 建议 |
|---|---|---|---|
| A 覆盖缺口 | `rp_freshness_days` | 覆盖 0.232；有值/缺失标签中位差 +0.0004（正差占比 49%）；规模分位差 +0.273；根因=正常结构性 | 勿默认补数据（且入模即被缺失率门禁删除） |

**偏 IC（控制簇代表后的秩残差）**：

| 因子 | 控制 | 偏 IC 均值 | 偏 IC t | 解读 |
|---|---|---|---|---|
| `rp_exec_flag_90d` | `rp_amount_to_mv_90d` | −0.0031 | **−2.79** | 控制金额后**显著负增量信息** ⇒ 冗余且方向为负 |

⇒ 与 §2/§3 一致：`rp_exec_flag_90d` 应在任何实验中**剔除**（弱 + 翻号 + 同簇 + 偏 IC 负）。

## 6. 结论与下一步建议

**事实（可复现）**：
1. 4 个值列设计性全覆盖（1.000）、无逐年收缩；口径无退化（`collapse_ratio` 0.64~1.03，根因全部"正常"）。
2. 家族内**只有 `rp_price_headroom` 有跨期稳定的截面信息**（t=+7.90、7/7 年正），但其强度仅居
   全表 **59 百分位**（中位水平）。
3. 3 个金额/状态列强度在 **6~18 百分位**，且 2021–2023 持续反向；`rp_exec_flag_90d` 偏 IC 为 −2.79。
4. 家族各轴**不与既有特征共线**（除家族内 90d/exec 同簇）⇒ 弱的原因不是"重复"。

**建议（按项目判据纪律）**：
- 依「列级（0~1pp）候选不可判定、不得据此开 WF 实验」与「加列默认带稀释成本」两条契约，
  repurchase 当前证据**不足以支持完整 14 折 A/B**：预期效应低于噪声带，实验结果只会落进
  「不可判定」区间，且家族最强列弱于已失败的 stk_holdertrade 族。
- 若仍要耗一次实验额度，**A 臂必须是单列**（`rp_price_headroom`，不含 90d/180d/exec/freshness）：
  这是唯一未被任何候选清单命中、且 7/7 年同号的列，1 列扰动也最接近「稀释成本 ≈ 0」的干净对照；
  同时**预登记**其解释风险：`high_limit ÷ 现价 − 1` 在价格下跌后变大，可能与**反转/超跌**同向，
  即使通过也只说明"该代理有效"，须再做"剔除价格分子只留公告"的对照才能归因到回购本身。
- 不做的事：**禁止**用 `--feature-set` 式消融位搜索去挑"哪几列组合最好"（等同事后选择）。

## 7. 复现与产物

| 环节 | 命令 / 文件 |
|---|---|
| 快照物化 + 体检 + 诊断 | `python scripts/materialize_factor_health_snapshot.py --start 20200101 --end 20260702 --every 3 --with-repurchase --skip-usage --feature-file data/models/stock_selection/v24123_features.json --out-root temp/rp_health_root_20260918 --health-out data/reports/factor_health/repurchase_20260918 --diagnosis-out data/reports/factor_diagnosis/repurchase_20260918` |
| 台账（160 列） | `data/reports/factor_health/repurchase_20260918/factor_register.csv` |
| 体检报告 | 同上目录 `factor_health_report.md`、`candidates_*.csv`、`clusters.csv`、`corr_matrix.csv` |
| 诊断报告 | `data/reports/factor_diagnosis/repurchase_20260918/因子诊断报告.md`、`数据改良候选清单.csv`、`partial_ic.csv`、`usage_stability.csv` |
| 运行日志 | `logs/rp_health_20260918.log`（临时）、`logs/health_snapshot_*.log` |
| 快照数据根 | `temp/rp_health_root_20260918`（**临时，跑完即删**；重建约 45 秒） |
