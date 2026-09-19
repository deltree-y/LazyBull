# top10_floatholders 单列 A 臂 A/B 结论（Phase 4）

> 2026-09-19 · 预登记 `docs/plans/top10fh_ab_prereg.md`（**跑前写定**，本文件不得修改其判据）
> 产物：`data/reports/wf_signal_compare/phase4_tfh_conc_20260919/`、`data/reports/wf_fold_subset/phase4_tfh_conc_20260919/`
> 日志：`logs/phase4_tfh_conc_20260919.log`（训练）、`logs/tfh_ab_signal_20260919.log`、`logs/tfh_ab_fold_subset_20260919.log`

## 1. 结论：**不通过 ⇒ 不采纳，家族终结**

| 判据 | 结果 | 判定 |
|---|---|---|
| **判据 1 信号层**（筛选尺子）：Δ平均持有期收益 ≥ −10 bps | **−18.08 bps**（Top20）/ −16.36 bps（Top30），相对基线 **−19.6% / −17.8%**；95% 区间 **[−34.5, −0.9] / [−31.2, −1.6]**（**整体为负**） | ❌ **不通过**（超门槛，且非"带内不可判定"） |
| **判据 2 净值层**：ΔMaxDD ≥ 0 且逐折改善 ≥ 9/14 | **ΔMaxDD −5.24pp**（回撤**变大**）；逐折改善 **8/14** | ❌ **不通过**（两项均未达） |
| 辅助：ΔCAGR / Δ夏普 | −3.45pp（[−10.3, +2.4]）/ −0.133（[−0.42, +0.11]） | 方向为负（落在 −5.8pp / −0.15 噪声带内，不作否决依据） |

**⇒ 两判据均不通过，且方向一致为负（信号层区间整体为负、逐折 MaxDD 12/14 同向下行）**：
把 `tfh_concentration_chg` 加进选股模型不会带来可测收益，反而**系统性降低**Top-K 选择质量
（≈ −18 bps ≈ −20% 相对）与回撤控制。

## 2. 臂与可比性

| 臂 | batch | 折数 | 特征列 | 运行耗时 |
|---|---|---|---|---|
| B0 基线（复用，未重跑） | `data/walk_forward/batches/phase4_ht_ab_20260917_base/` | 14/14 | 52 | （2026-09-17 完成） |
| A 单列（`tfh_concentration_chg`） | `data/walk_forward/batches/phase4_tfh_conc_20260919/` | 14/14 | **53** | **1h58m**（08:24→10:22，含部署模型训练） |

- 逐折窗口与 B0 **逐字一致**（`split_index` / `test_start` / `test_end` 全 14 折相等，已校验）；
- 每折模型 `*_features.json`：B0 全 14 折 **52 列且无 `tfh_*`**；A 臂全 14 折 **53 列且 `tfh_*` 恰为
  `{tfh_concentration_chg}`**（哨兵被常数门禁剔除，符合预登记）；
- 其余超参与 B0 逐字相同；链式净值覆盖同一 **1,700 个交易日**（两臂 `chain_nav` 均为 1,714 行 = 1,700 + 14 个折起点），
  两臂汇总均 14/14 折成功。

## 3. 判据 1：信号层（同日配对 + 交易日分块自举）

| TopK | Δ平均持有期收益 | 相对基线水平 | 95% 区间 | 正差值概率 | 折内同向折数 |
|---|---|---|---|---|---|
| 20 | **−18.08 bps** | **−19.6%** | [−34.53, −0.90]（块长 20） | 0.020 | 7/14 |
| 30 | **−16.36 bps** | −17.8% | [−31.24, −1.58]（块长 20） | 0.017 | 7/14 |

- Δ命中率：−0.64pp（Top20）/ −0.52pp（Top30）（噪声带 ±0.6pp，属边界；方向同负）。
- **关键读数**：三个块长（20/40/60）的区间**上界全部为负**（−0.90 ~ −2.48 bps）⇒
  不是"加列稀释落在噪声带内"，而是**可检出的负效应**（≈ −20% 相对）。
- **参照系**：holdertrade A2（4 列）−11.25 bps、repurchase 单列 −12.19 bps；
  本臂单列 **−18.08 bps 是三轮里最差的信号层结果**——说明问题不在"列数稀释"，
  而是该列把排序往低效方向拽。

## 4. 判据 2：净值层

**链式（同一折子集，归一化到子集起点）**：

| 指标 | B0 基线 | A 单列 | Δ |
|---|---|---|---|
| CAGR | 17.58% | **14.13%** | **−3.45pp** |
| 最大回撤 | −22.79% | **−28.02%** | **−5.24pp** |
| 夏普 | 0.731 | 0.598 | −0.133 |

**逐折明细**（ΔMaxDD = A − B0，正 = 回撤变小 = 改善；Δ收益 = A − B0）：

| 折 | Δ收益(pp) | ΔMaxDD(pp) | 折 | Δ收益(pp) | ΔMaxDD(pp) |
|---|---|---|---|---|---|
| 0 | +0.12 | **+0.79** | 7 | −2.74 | +0.97 |
| 1 | +0.51 | **+1.86** | 8 | −1.45 | −0.50 |
| 2 | +1.87 | +0.30 | 9 | −1.33 | −1.07 |
| 3 | −4.04 | +1.05 | 10 | **+5.87** | **+2.56** |
| 4 | −0.67 | +0.91 | 11 | −1.08 | −0.79 |
| 5 | **−17.96** | −1.78 | 12 | +0.73 | +1.55 |
| 6 | **−7.20** | **−4.64** | 13 | **+5.61** | −2.16 |

- 逐折 MaxDD 改善 **8/14**（要求 ≥ 9/14）；逐折收益为正仅 6/14；ΔMaxDD 中位 +0.54pp，
  但**点估计被折 6（−4.64pp）与折 5 拖成 −5.24pp** ⇒ 逐折方向 12/14 同向为**恶化**方向。

## 5. 数据态放行登记（必须与结论一起读）

| 项 | 值 |
|---|---|
| B0 | `git_commit=c47069d`、`data_state_id=2a732925`、summary **不含**任何 top10fh 列（本族当时尚不存在） |
| A 臂 | `git_commit=b0a4215`、`data_state_id=11498114`（与 2026-09-18 repurchase 臂相同 ⇒ 数据态自那时未变） |

放行理由（三点，沿 repurchase 先例）：
1. **逐折窗口与 B0 逐字一致**（防止"换窗口"型不可比）；
2. **B0 端开关天然惰性**：B0 summary 连 `enable_top10fh_features` 列都没有 ⇒ 本族开关不可能影响 B0；
3. **列集已逐折核验**：B0 52 列无 `tfh_*`、A 臂 53 列恰为 `{tfh_concentration_chg}`。
   ⚠️ 注意 `data_state_id` 相同**不构成**代码同一性证据（其间已新增本族开关代码），
   本表的可比性只由上述 1~3 条支撑。

## 6. 解释与归因约束（预登记要求）

- **方向风险成立**：预登记时已登记"该列与股东户数下降同向（ρ≈−0.20）、与筹码集中类指标同源"，
  结果是**负向**——即该代理在当前特征集里起的是"把排序推向低效方向"的作用，
  与 repurchase 单列（`high_limit ÷ 现价 − 1` 与反转/超跌同向）同一模式。
- **不得**把本结果读作"股东结构信息无用"：本臂只测了**该单一代理**在**当前标签口径
  （neu_y_ret_20）与当前模型配置**下的表现；集中度的其它载体（组合级 regime 信号、
  与股东户数变化构造合成因子、事件域标签）未经测试，属**未观测**、非"已否定"。
- **不得**用 `--top10fh-feature-set` 之外的任何变体继续试列（例如改试 `top1_ratio`、
  两列组合、改窗口）——那属事后选择（消融位搜索），预登记已明令禁止。

## 7. 家族处置

**⇒ 不采纳；`top10_floatholders` 家族终结**（与 stk_holdertrade / repurchase 同处置）：

- 开关保持**默认关**；生产不启用本族任何列；
- **不再开新臂、禁止消融位搜索**；
- 本轮**不改动生产特征分区**（本族仍为运行时派生，cs_train / cs_infer 未写入任何 `tfh_*` 列）；
- 三轮数据层家族（holdertrade 8 列 / repurchase 单列 / top10fh 单列）**全部不通过**，
  且"单列最好列"也一致为负 ⇒ 进一步支持既有契约「**加列默认带稀释成本**」，
  后续新增因子/数据源必须先过信号层尺子（相对降幅 ≥9% 才值得跑全量 A/B），
  且**禁止**再用单臂净值口径追列级效应；
- 日后仅当出现**新信息**（新数据源 / 新标签目标域 / 组合级载体）才允许重开，且需重新预登记。

## 8. 复现

```bash
# A 臂（1h58m；与 B0 逐字同参，仅多 top10fh 开关）
python scripts/walk_forward.py --algorithm xgboost --split-count 14 --final-date 20260105 \
  --train-window-years 6 --test-window-months 6 --val-ratio 0.2 --label neu_y_ret_20 \
  --neutral-label-blend-weight 0 --task regression --label-transform cs_zscore --objective mse \
  --n-estimators 500 --max-depth 5 --num-leaves 63 --learning-rate 0.03 --subsample 0.8 \
  --colsample-bytree 0.3 --min-child-weight 200 --reg-alpha 0.03 --reg-lambda 7.0 --gamma 0.5 \
  --rank-weight-topk 120 --rank-weight 100 --rank-weight-topk-weight-mode linear_decay \
  --early-stopping-rounds 50 --early-stopping-metric rank_ic_daily --min-best-iteration 30 \
  --time-decay-half-life 0 --freshness-strategy state_keep_event_decay \
  --event-freshness-half-life-days 120 --ensemble-seeds 42,61,82 \
  --ensemble-seed-keep-top-ratio 0.3 --ensemble-seed-keep-min-models 3 \
  --oos-backtest --oos-backtest-months 6 --bt-top-n 20 --bt-initial-capital 1000000 \
  --bt-sell-timing open --bt-min-list-days 365 --bt-max-weight-per-stock 0.15 \
  --position-sizing kelly --kelly-vol-window 60 --kelly-max-leverage 0.2 --stagger-tranches 2 \
  --data-root ./data --enable-top10fh-features --top10fh-feature-set concentration \
  --batch-run-id phase4_tfh_conc_20260919 --batch-period-label tfh_ab_conc \
  --wf-summary-csv data/walk_forward/batches/phase4_tfh_conc_20260919/raw/walk_forward_summary_tfh_conc_0001.csv

# 判据 1 / 判据 2
python scripts/compare_wf_signal_metrics.py --baseline data/walk_forward/batches/phase4_ht_ab_20260917_base \
  --arm data/walk_forward/batches/phase4_tfh_conc_20260919 --topk 20 30 --block-days 20 40 60 \
  --allow-state-mismatch --out data/reports/wf_signal_compare/phase4_tfh_conc_20260919
python scripts/compare_wf_fold_subset.py --baseline data/walk_forward/batches/phase4_ht_ab_20260917_base \
  --arm data/walk_forward/batches/phase4_tfh_conc_20260919 --bootstrap 2000 \
  --allow-state-mismatch --out data/reports/wf_fold_subset/phase4_tfh_conc_20260919
```
