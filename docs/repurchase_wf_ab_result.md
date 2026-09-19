# repurchase 家族 WF A/B 结果（Phase 4，`rp_price_headroom` 单列臂）

> 2026-09-18 · 预登记 `docs/plans/repurchase_ab_prereg.md`（跑前写定）· 结论：**不通过 ⇒ 不采纳，家族终结**
> 产物：`data/reports/wf_signal_compare/phase4_rp_headroom_20260918/`、`data/reports/wf_fold_subset/phase4_rp_headroom_20260918/`

## 1. 两臂

| 臂 | batch | 折数 | 特征列 | 运行耗时 |
|---|---|---|---|---|
| B0 基线（复用，未重跑） | `data/walk_forward/batches/phase4_ht_ab_20260917_base/` | 14/14 | 52 | （2026-09-17 完成） |
| A 单列（`rp_price_headroom`） | `data/walk_forward/batches/phase4_rp_headroom_20260918/` | 14/14 | **53** | **1h32m**（14:32→16:04） |

- 逐折窗口与 B0 **逐字一致**（校验通过）；每折模型 `*_features.json` 中 rp 列集合 = `{rp_price_headroom}`（
  14 折全部只有 1 列，已核）。
- 各折链式净值：**B0 CAGR 17.58% / MaxDD −22.79% / 夏普 0.731**；
  **A CAGR 14.35% / MaxDD −20.21% / 夏普 0.605**（14 折、1,700 交易日）。

## 2. 判据 1｜信号层（筛选尺子，v0.122.0）——**不通过**

| Top-K | Δ平均持有期收益 | 相对基线 | 95% 区间（20/40/60 日块） | 折内同向折数 | Δ命中率 |
|---|---|---|---|---|---|
| Top20 | **−12.19 bps** | **−13.2%** | [−28.5, +5.5] / [−30.2, +7.3] / [−32.0, +8.1] | 6/14 | −0.50 pp |
| Top30 | **−12.98 bps** | **−14.1%** | [−27.6, +1.9] / [−30.2, +2.9] / [−31.6, +4.3] | 5/14 | −0.56 pp |

- 基线水平 92.3 / 91.7 bps；噪声带 ±10 bps（≈±9% 相对）。
- **触发预登记门槛**：Δ ≤ −10 bps（相对降幅 ≥9%）⇒ 判「加列稀释」，**不通过**。

**参照（同判据、同基线）**：holdertrade A1（8 列）**+5.03 / +7.00 bps**（带内不可判定）；
A2（4 列）**−11.25 bps**；本次 repurchase **单列 −12.19 bps**——**1 列比 4 列还差**，
说明不是单纯"列数稀释"，而是该列本身把排序往低效方向拽（见 §5 解释）。

## 3. 判据 2｜净值层（链式指标 + 折级自举 2000 次，种子 42）——**不通过**

| 指标 | B0 基线 | A 单列 | Δ | 自举 95% 区间 |
|---|---|---|---|---|
| CAGR | 17.58% | 14.35% | **−3.23 pp** | [−9.70, +1.42] |
| **最大回撤（主判据）** | −22.79% | −20.21% | **+2.58 pp（改善）** | [−3.51, +4.56] |
| 夏普 | 0.731 | 0.605 | −0.126 | [−0.400, +0.065] |
| 逐折收益为正 | 9/14 | 9/14 | — | — |

**预登记的两条并列条款**：

1. ΔMaxDD ≥ 0 ⇒ **满足**（+2.58 pp）；
2. 逐折 MaxDD 改善折数 ≥ **9/14** ⇒ **仅 7/14，不满足**。

⇒ 主判据整体**不满足**（并列条件未同时成立）；辅助 ΔCAGR/ΔSharpe 虽落在噪声带内
（−5.8 pp / −0.15 阈值），但方向为负。工具独立判定同为 **不通过**。

**逐折明细**（ΔMaxDD = A − B0，正 = 回撤变小 = 改善）：

| 折 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 折收益差（pp） | −2.2 | +0.4 | −0.4 | −0.2 | −4.0 | **−16.5** | +2.4 | −1.5 | +4.3 | −3.0 | +2.9 | −1.5 | +2.5 | −4.8 |
| ΔMaxDD（pp） | −0.3 | +2.7 | −0.8 | +0.4 | −0.3 | −2.3 | **+2.8** | +1.6 | +2.2 | −0.6 | +1.1 | −1.4 | +1.9 | **−2.5** |

- 收益变差折数 **9/14**（变好 5/14）；链式 MaxDD 的改善集中在 2 折（f6 +2.8pp、f1 +2.7pp），
  而 f13/f5 反向恶化 −2.5/−2.3pp ⇒ **不是稳健的回撤结构改善**。

## 4. 判据 3｜逐折一致性

- ΔCAGR 与 ΔMaxDD 的折内同向数：收益 12/14（12 折比基线差）、MaxDD 7/14（改善偏少）；
- 结论必须连同比值一起读：单看链式 ΔMaxDD +2.58pp 会给出偏乐观印象。

## 5. 结论：**不通过 ⇒ 不采纳（家族终结）**

| 判据 | 观测 | 结论 |
|---|---|---|
| 判据 1 信号层 | Δ −12.19 / −12.98 bps（相对 −13.2% / −14.1%） | **不通过**（超 −10 bps 门槛） |
| 判据 2 主判据 | ΔMaxDD +2.58 pp 但**逐折改善仅 7/14**（要求 ≥9/14） | **不通过** |
| 判据 2 辅助 | ΔCAGR −3.23 pp、Δ夏普 −0.126（均在噪声带内） | 不可判定（方向为负） |

**裁定**：`--enable-repurchase-features` **保持默认关**（含 `headroom` 列集）。
按预登记 §4 的禁止项，**不再开新臂、禁止消融位搜索**（不允许再试 `headroom + 180d` 等组合）；
若日后重开，必须具备**新信息**（新数据源 / 新标签目标域 / 语汇口径变更）并重新预登记。

**归因约束（预登记 §4.1 已声明，此处据实登记）**：`rp_price_headroom = high_limit ÷ VWAP − 1`
在股价下跌后变大，与**反转/超跌**同向。本臂观测到的"回撤略降、收益略降"与该代理的经济含义一致
（把组合推向跌后标的），**不能**归因为"回购信号有效"；即便它通过也必须先补"剔除价格分子"的对照。

## 6. 已知代价与局限（必须随结论一起报告）

- **家族内弱列未进入任何一臂**（`rp_amount_to_mv_90d/180d`、`rp_exec_flag_90d` 于 Phase 3 判定为
  全表 6~18 百分位强度、2021–2023 持续反向、`rp_exec_flag_90d` 偏 IC t=−2.79）；
  因此本结论覆盖的是**家族最优列**，可外推为"该家族在当前 52 列基线下无采纳价值"。
- `rp_freshness_days` 覆盖 23%，入模即被 >0.6 缺失率门禁删除 ⇒ 从未参与任何臂。
- 单列臂的 ΔCAGR/ΔSharpe 落噪声带内 ⇒ 净值维度的**方向**可读、**幅度不可读**（±3pp 级）。
- 链式 ΔMaxDD 与逐折 MaxDD 改善折数**不一致**（+2.58pp vs 7/14）再次证明：只看链式点估计会误导，
  逐折散布必须并列报告。

## 7. 数据态放行说明（重要）

B0 运行在 `git_commit=c47069d`（data_state_id `2a732925`），A 臂运行在 `b0a4215`（`11498114`）⇒ ID 不同，
比较工具默认拒绝。经逐字段核对（预登记 §5 的三条条件全部成立）：

1. `raw_latest_partitions`（daily/adj_factor/daily_basic/moneyflow/stk_limit/suspend/stock_st/margin_detail）
   两臂均 = **2026-07-31**；`features_cs_train_latest` 均 = **20260702**；
2. `dividend` 覆盖两臂均 = `data=5880, empty=10, failed=0, pending=0, total=5890`；
3. 期间代码差异对本臂**惰性**：新增的 `--repurchase-feature-set` 是显式开关（B0 未传），
   且 B0 的 `enable_repurchase_features=False`（已从 B0 summary 读回确认）；
   其余代码改动为新增因子模块与测试，不触碰既有特征/训练路径。

⇒ 以 `--allow-state-mismatch` **显式放行**并在此登记。

## 8. 复现

```bash
# A 臂（约 1.5 h；与 B0 逐字同参，仅多 repurchase 开关）
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
  --data-root ./data --enable-repurchase-features --repurchase-feature-set headroom \
  --batch-run-id phase4_rp_headroom_20260918 --batch-period-label rp_ab_headroom \
  --wf-summary-csv data/walk_forward/batches/phase4_rp_headroom_20260918/raw/walk_forward_summary_rp_headroom_0001.csv

# 判据 1 / 判据 2
python scripts/compare_wf_signal_metrics.py --baseline data/walk_forward/batches/phase4_ht_ab_20260917_base \
  --arm data/walk_forward/batches/phase4_rp_headroom_20260918 --topk 20 30 --block-days 20 40 60 \
  --allow-state-mismatch --out data/reports/wf_signal_compare/phase4_rp_headroom_20260918
python scripts/compare_wf_fold_subset.py --baseline data/walk_forward/batches/phase4_ht_ab_20260917_base \
  --arm data/walk_forward/batches/phase4_rp_headroom_20260918 --bootstrap 2000 \
  --allow-state-mismatch --out data/reports/wf_fold_subset/phase4_rp_headroom_20260918
```

日志：`logs/phase4_rp_headroom_20260918.log`（训练）、`logs/rp_ab_signal_20260918.log`、
`logs/rp_ab_fold_subset_20260918.log`（两条判据）。
