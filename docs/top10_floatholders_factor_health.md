# top10_floatholders（十大流通股东）因子体检与诊断结论（Phase 3）

> 2026-09-19 · 口径定稿见 `docs/top10_floatholders_pit_audit.md` §6 · 快照数据根 `temp/top10fh_health_root_20260919`（临时，跑完即删）
> 产物：`data/reports/factor_health/top10fh_20260919/`、`data/reports/factor_diagnosis/top10fh_20260919/`
> 运行：`logs/top10fh_health_20260919.log`（物化 525 分区 8.5 分钟，2.33 GB / 162 列）

## 0. 为什么需要"快照物化"

本族按契约**运行时派生**，不写入生产 `features/cs_train`；而体检/诊断工具只读分区。
因此走既有入口 `scripts/materialize_factor_health_snapshot.py --with-top10fh`（Phase 2 已登记家族）。

**本次与 repurchase 的差异**：repurchase 的容器是 `{交易日: DataFrame}` 字典（稀疏，只需窗口内活跃股票），
本族容器是**报告期面板**（DataFrame，247,006 行）——因为本族逐日全市场稠密（~5,500 只/日），
逐日字典在 525 天区间上会膨胀到百万行级。物化器通过家族登记表的 `build_lookup` 回调天然兼容两种容器，
脚本主体无家族分支（本次唯一的工程改动就是登记 `top10fh` 一项）。

**基线清单**：`data/models/stock_selection/v24123_features.json`（与 stk_holdertrade / repurchase Phase 3
同一基线，保证百分位可比）+ 8 个 top10fh 列（6 值列 + freshness + 哨兵）= **162 列**、
525 个采样交易日（20200102~20260702，每 3 个交易日取 1 天）。

## 1. 覆盖率（全期与逐年）

| 列 | 全期覆盖 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| `tfh_top10_ratio` | **1.000** | 1.000 | 1.000 | 0.9997 | 0.9998 | 0.9996 | 0.9994 | 0.9998 | ✅ 设计性全覆盖（未披露股票极少） |
| `tfh_top1_ratio` | **1.000** | 1.000 | 1.000 | 0.9997 | 0.9998 | 0.9996 | 0.9994 | 0.9998 | ✅ 同上 |
| `tfh_inst_ratio` | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | 0.9998 | 0.9996 | 1.000 | ✅ 同上 |
| `tfh_inst_count` | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | 0.9998 | 0.9996 | 1.000 | ✅ 同上 |
| `tfh_social_security_flag` | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | 0.9998 | 0.9996 | 1.000 | ✅ 同上 |
| `tfh_concentration_chg` | **1.000** | 1.000 | 1.000 | 0.9997 | 0.9998 | 0.9994 | 0.9994 | 0.9998 | ✅ 仅首期无上一期（NaN 0.06%） |
| `tfh_freshness_days` | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | 0.9998 | 0.9996 | 1.000 | ✅ 同上 |
| `tfh_schema_v1` | 1.000 | — | — | — | — | — | — | — | 哨兵（常数，判为 market_level，不参与截面 IC） |

**无低覆盖候选**（`candidates_low_coverage.csv` 命中 0 条）——与事件族的稀疏风险不同，
本族天然全市场覆盖，因此**入模不会被 0.6 缺失率门禁整删**（Phase 2 实测 NaN 0.02%~0.06%）。

口径退化检查（`column_year_summary.csv`）：7 列全部"正常"，无历史幅度坍塌
（`collapse_ratio` 0.63~1.13；`tfh_top10_ratio` 0.97、`tfh_top1_ratio` 0.98 最稳）。

## 2. 逐日截面 RankIC

| 列 | IC 均值 | IC-IR | t | |IC| 全表分位 | 逐年 IC（2020→2026） | 同号年数 | 判定 |
|---|---|---|---|---|---|---|---|---|
| `tfh_concentration_chg` | +0.0156 | **0.573** | **+13.12** | 45.1%（**\|t\| 76.5%**） | +.030/+.010/+.015/+.018/+.016/+.015/**−.006** | 6/7 | ⭐ 家族最强、跨期最稳 |
| `tfh_top1_ratio` | +0.0172 | 0.324 | +7.43 | 48.1% | +.023/+.009/+.012/+.028/+.024/+.003/+.027 | **7/7** | ✅ 方向上稳定，强度中位 |
| `tfh_top10_ratio` | +0.0161 | 0.240 | +5.49 | 46.3% | +.032/+.002/+.003/+.019/+.020/+.005/+.047 | **7/7** | ✅ 同上（但与 top1 同向同源） |
| `tfh_freshness_days` | +0.0057 | 0.174 | +3.99 | 20.4% | +.014/+.001/+.004/+.005/+.009/+.002/+.004 | 7/7 | ⚠️ 弱；且与既有 freshness 近乎同一列（见 §3） |
| `tfh_inst_ratio` | +0.0001 | 0.001 | +0.02 | **0.0%** | +.041/−.013/−.011/−.032/+.007/−.015/+.049 | 4 次翻转 | ❌ 弱信息 + 翻号 |
| `tfh_inst_count` | +0.0006 | 0.007 | +0.15 | **1.9%** | +.042/−.015/−.011/−.030/+.007/−.016/+.052 | 4 次翻转 | ❌ 同上 |
| `tfh_social_security_flag` | −0.0024 | −0.029 | −0.66 | 8.6% | +.045/−.019/−.012/−.035/+.006/−.015/+.030 | 2 次翻转 | ❌ 同上（且与 inst 轴 ρ≈0.71） |

**分层（市场波动高/低半区）**：`top10_ratio` 0.0085/0.0236、`top1_ratio` 0.0098/0.0246、
`concentration_chg` 0.0135/0.0176 —— **两半区同号**（低波动区更强）；
`inst_*`/`social_flag` 在高波动区翻负 ⇒ 又一处不稳定证据。

**2026 为部分年**（40 个采样日）：`concentration_chg` 的 2026 由正转负更可能是样本量问题，登记待观察。

## 3. 相关与聚类（正交性）

**家族内（|ρ|）**：
- `inst_ratio ↔ inst_count` **0.990** ⇒ 实质同列（`inst_ratio` 被判 `flag_dup`）；
- `social_flag ↔ inst_*` 0.71 ——同源（社保是长线机构子集）；
- `top10_ratio ↔ top1_ratio` **0.786** ——同源但未达去重阈值（不同簇：11 / 12）；
- `concentration_chg` 对家族内各列 |ρ| ≤ 0.080；`freshness` 对值列 |ρ| ≤ 0.063 ⇒ 独立轴。

**对既有特征的最强相关**（每列 Top5）：
| 列 | 最强既有相关 |
|---|---|
| `tfh_top10_ratio` | `zscore_turnover_rate` −0.381、`cost_concentration` +0.333、`zscore_size` +0.304 |
| `tfh_top1_ratio` | `zscore_turnover_rate` −0.280、`zscore_size` +0.246 |
| `tfh_inst_ratio` / `inst_count` | `fund_count` **+0.42**、`fund_hold_ratio` **+0.42**、`zscore_size` +0.31 |
| `tfh_social_security_flag` | `fund_hold_ratio` +0.357、`fund_count` +0.349 |
| `tfh_concentration_chg` | `holder_num_chg_2q` −0.197、`holder_num_chg` −0.190 |
| `tfh_freshness_days` | `fundamental_freshness_days` **+0.999**、`cashflow_freshness_days` +0.993 |

**解读**：
1. **集中度轴（top10/top1）与既有特征不强相关**（|ρ| ≤ 0.38）⇒ 是新的信息轴（筹码集中度，
   与 `cost_concentration`（筹码分布）方向一致但不同口径）；
2. **机构轴与 `fund_portfolio` 家族 ρ≈0.42** ⇒ 部分重叠（公募与保险同属机构资金），
   **不得**当作完全正交；加之自身 IC/翻号不合格（§2），本轴的三列整体判定为噪声；
3. **`concentration_chg` 与股东户数变化 ρ≈−0.20** ⇒ 经济含义一致（集中度上升 ≈ 户数下降），
   但相关性弱（−0.2 远低于去重阈值）⇒ 是**同向但增量的口径**（户数变化看总量、集中度变化看前 10 大结构）；
4. **`freshness` 与既有 freshness 列 ρ≈0.999** ⇒ 完全重复（同一"报告新鲜度"轴）；
   若做 Phase 4，**该列无预期增量**。

**簇结构**：`top10_ratio` / `top1_ratio` / `concentration_chg` / `social_flag` 各自成为**单列簇**（未与任何既有特征同簇）；
`inst_ratio`/`inst_count` 自成一簇；`freshness` 并入既有 freshness 簇（3 列）。

## 4. 诊断（偏 IC 与候选项）

- **偏 IC**（控制簇代表后的增量信息）：只对本族两处共簇列计算——
  `tfh_inst_ratio` 控制 `tfh_inst_count` 后 **偏 IC t = −5.58**（显著**负**增量）⇒ 该列应剔除
  （被同簇代表完全替代且方向相反）；另有 `fundamental_freshness_days` / `cashflow_freshness_days`
  控制 `tfh_freshness_days` 后偏 IC t=+3.57 / +2.74（说明本族的 freshness 作为簇代表"吃掉"了部分基线列的增量——
  反过来印证 §3 第 4 条的重复性）。
- **《数据改良候选清单》14 条中 0 条涉及 `tfh_*`** ⇒ 无覆盖缺口、无口径退化、无需补数据的条目。
- **使用度本轮不可观测**：尚无模型启用 `--enable-top10fh-features`，
  `gain_present_ratio = 0.0` 属"**尚未观测**"，**不得**读作"模型不用"（与 stk_holdertrade / repurchase 同一口径约束）。

## 5. 结论与 Phase 4 建议

**家族体检画像**：一个"**一强 + 两中 + 四弱**"的家族——

- ⭐ 一强：`tfh_concentration_chg`（集中度环比变化）——t=+13.1（|t| 全表 76.5 百分位）、
  6/7 年同号、两 regime 同号、家族内外均正交（最大 |ρ| 0.20），且经济含义与"筹码集中"一致。
  **它是本族唯一按既有标准"值得考虑"的列**。
- ✅ 两中：`tfh_top10_ratio` / `tfh_top1_ratio`——7/7 年同号但强度中位（|IC| ≈0.016~0.017，
  全表 46~48 百分位），两者 ρ=0.79 同源，**同时入模属重复投注**。
- ❌ 四弱：`tfh_inst_ratio`（负偏 IC）、`tfh_inst_count`（|IC| 1.9 百分位）、
  `tfh_social_security_flag`（翻号）、`tfh_freshness_days`（与既有 freshness ρ=0.999 重复）。

**建议（按契约与既往证据）**：

1. **不跑 6/7 列的全量 A/B**：列级预期效应落噪声带内（14 折 MDE≈5pp），
   且家族内 4 列已判定为噪声/重复，全列臂只会把稀释成本（既往实测：+4 列 ≈ −9.5% 信号）
   与弱列噪声混在一起，**结论必然是"不通过"且无法归因**；
2. **若要做 Phase 4，唯一合理的 A 臂 = 单列 `tfh_concentration_chg`**（同时是"加 1 列"的最干净稀释对照）；
   需先加列集开关 `--top10fh-feature-set {full,concentration}`（超参签名维度，沿 repurchase 先例）；
   预登记判据沿用既有标准（信号层 ±10 bps 尺子 + 净值 ΔMaxDD 主判据 + 逐折同向数 ≥9/14），
   并**预登记解释风险**：集中度变化与"户数下降/超跌后集中"同向（ρ≈−0.2），
   即使通过也只能证明"该代理有效"，需另做对照才能归因到股东结构本身；
   成本约 1.5~2 h（14 折 × 1 臂，B0 复用既有基线 batch）；
3. **若接受登记**：把本族列整体登记为"数据层已体检、单列强度 |t|=13 但按噪声带判据不予追"，
   省下 2 h；日后只有在出现**新信息**（例如把集中度变化做成组合级 regime 信号、
   或与股东户数变化构造合成因子）时才重新预登记；
4. 无论哪种选择：**本轮不把本族列设为默认开启**（开关保持默认关），
   且**禁止**用 `--top10fh-feature-set` 之外的消融位搜索挑列（等同事后选择）。

## 6. 复现与产物

| 环节 | 命令 / 文件 |
|---|---|
| 快照物化 + 体检 + 诊断 | `python scripts/materialize_factor_health_snapshot.py --start 20200101 --end 20260702 --every 3 --with-top10fh --skip-usage --feature-file data/models/stock_selection/v24123_features.json --out-root temp/top10fh_health_root_20260919 --health-out data/reports/factor_health/top10fh_20260919 --diagnosis-out data/reports/factor_diagnosis/top10fh_20260919` |
| 台账（162 列） | `data/reports/factor_health/top10fh_20260919/factor_register.csv` |
| 体检报告 | 同上目录 `factor_health_report.md`、`candidates_*.csv`、`clusters.csv`、`corr_matrix.csv` |
| 诊断报告 | `data/reports/factor_diagnosis/top10fh_20260919/因子诊断报告.md`、`数据改良候选清单.csv`、`partial_ic.csv`、`column_year_summary.csv`、`usage_stability.csv` |
| 运行日志 | `logs/top10fh_health_20260919.log`（8.5 分钟：物化 525 分区 ≈6 min + 体检 ≈1 min + 诊断 ≈1.5 min） |
| 快照数据根 | `temp/top10fh_health_root_20260919`（**临时，跑完即删**；含指向生产 `data/{clean,raw,models}` 的目录联接，删除时先删联接再删根） |
