# v2 MVP 基线冻结登记（B0 / B1 双臂）

> P0 交付物：现产线默认配置的双臂基线（配置指纹 + 逐折收益路径）。
> 冻结时点 = 启动日（2026-09-30）。裁决前 P2a 不开工。

## 1. 双臂定义

| 臂 | 含义 | 配置要点 |
|---|---|---|
| **B0** | 无暴露政策基线（neutral） | 不启用 exposure-policy / exposure-table / replenish；其余与 B1 逐字一致 |
| **B1** | 默认臂 e2online_r（含政策） | E2-滚动250、λ=0.5、在线现算、对称回补开、不裁剪覆盖（20181126） |

**迁移目标 = P1.5 政策层裁决的胜者**（多判据组合：配对 ΔCAGR 分布 + 亏损日频率/幅度 + §4.8 结构成本）。**（2026-10-01 已裁决：退役 ⇒ 迁移目标 = B0，方案 F8；裁决报告 `docs/reports/p15_policy_layer_adjudication_20261001.md`）**

## 2. 公共配置指纹（两臂共享）

> 来源：`scripts/batch/batch_walk_forward.ps1` 配置区（启动日 2026-09-30 当前值）。
> **教训登记**：B0 首次复跑误用手拼的 walk_forward.py 默认参数（漏掉分批调仓 / 半凯利 / 资金量 /
> 上市天数 / 单股上限），导致 MaxDD 虚高；正确做法 = 直接调批脚本 `-NoExposure`（读配置区当前值，
> 保证与产线逐字一致）。已废弃错误批次并更正。

| 参数 | 值 |
|---|---|
| split_count / final_date | 14 / 20260105 |
| train_window_years / test_window_months | 6 / 6 |
| val_ratio | 0.2 |
| label | neu_y_ret_20（中性化），blend_weight=0 |
| task / label_transform / objective | regression / cs_zscore / mse |
| n_estimators / max_depth / learning_rate | 500 / 5 / 0.03 |
| subsample / colsample_bytree / min_child_weight | 0.8 / 0.3 / 200 |
| reg_alpha / reg_lambda / gamma | 0.03 / 7 / 0.5 |
| rank_weight | topk=120, weight=100, linear_decay |
| ensemble_seeds | 42,61,82（keep_top_ratio=1, keep_min=3） |
| early_stopping | rounds=50, metric=rank_ic_daily, min_best_iteration=30 |
| freshness | strategy=state_keep_event_decay, half_life=120 |
| 因子开关 | fundamental / alt / margin / cyq / fund / express / north / lhb / consensus / enhanced / cashflow_quality = true；consensus_revision / dividend_policy / feature_stability_filter / factor_prune = false |
| skip-training | 复用折模型 StartModelVersion=24008（v24008~v24021） |
| **stagger_tranches** | **2（分批调仓）** |
| oos_backtest | months=0（自动对齐 6）, top_n=20, rebalance_freq=20, sell_timing=open |
| **bt_initial_capital** | **1000000（100 万）** |
| bt_exclude_st / **bt_min_list_days** | true / **365** |
| **bt_max_weight_per_stock** | **0.15（单股上限）**；bt_max_per_industry=null |
| **position_sizing** | **half_kelly**（kelly_vol_window=60, kelly_max_leverage=0.2） |
| enable_early_rebalance_on_empty | true |
| stock_domain | main |
| deploy_train | true |

## 3. 数据态（B1 实测，B0 复跑前核对）

| 字段 | B1（wf_batch_20260929_153922） |
|---|---|
| data_state_id | `96572479` |
| git_commit | `3a37184`（dirty=true） |
| raw 最新分区 | daily / adj_factor / daily_basic / moneyflow / stk_limit / suspend / stock_st / margin_detail = 2026-07-31 |
| features_cs_train_latest | 20260702 |
| dividend_coverage | data 5880 / empty 10 / failed 0 / pending 0 |

## 4. 数据态（双臂最终一致）

最终双臂均在**同一数据态与代码态**下复跑：data_state_id = `9d0408ee`、git_commit = `b9d866e`、
raw 最新分区 2026-07-31、cs_train 最新分区 20260702。**无漂移**。

> 历史登记：B1 最初拟复用 0929 扫描批次（git `3a37184` / data_state `96572479`），后发现其
> top_n=10 与生产默认 top_n=20 不符，已改用 2026-09-30 重跑批次（172037），与 B0 同数据态。

结论：B0/B1 同数据态同代码态，配置差异**仅限政策臂开关**（B1=e2online_r，B0=neutral）。
满足折子集对比契约的可比性前提。

| 臂 | 链式净值 | 逐折成交 | 持仓快照 | 状态 |
|---|---|---|---|---|
| B1 | `wf_batch_20260930_172037\raw\chain_nav_wf_20260930_172039_dff2a95a.csv` | 同目录 split00~13 | 同目录 | ✅ 已冻结（批脚本默认臂 e2online_r，top_n=20） |
| B0 | `wf_batch_20260930_171221\raw\chain_nav_wf_20260930_171223_f71a780c.csv` | 同目录 split00~13 | 同目录 | ✅ 已冻结（批脚本 `-NoExposure`，top_n=20） |

> 废弃记录：① `p0_baseline_B0_neutral`（手拼默认参数的首次误跑）已删除；
> ② ~~B1 = `wf_batch_20260929_153922`~~（0929 扫描批次，**top_n=10**，与 B0 的 top_n=20
> 不可比）已弃用——B1 重跑为 172037。
> **教训登记**：双臂基线的 top_n / 分批 / 仓位口径必须逐项核对一致后才可比；参数扫描批
> 不能直接当默认配置基线。

## 5. 基线实测值（双臂，14 折 / 1,700 交易日）

| 指标 | B0（neutral 无政策） | B1（e2online_r 含政策） | Δ（B1−B0） |
|---|---|---|---|
| CAGR | 22.41% | **24.3%** | **+1.9pp** |
| MaxDD | −25.43% | **−19.4%** | **+6.0pp（B1 更优）** |
| 夏普 | 1.049 | **1.09** | **+0.04** |
| 总收益 | 295.6% | **335.0%** | **+39.4pp** |

**P1.5 裁决关键观察（事实登记，非裁决结论）**：在正确的同配置对照下，政策层 B1
**三项指标全胜**——MaxDD 改善 6.0pp（−25.43% → −19.4%；1,698 日中降暴露 283 日）、
CAGR +1.9pp、夏普 +0.04。**与 R-004 在线复核"ΔMaxDD −0.15pp 不达标"方向相反**——
差异源于 R-004 用同覆盖窗口（2024-01-02 起）+ 不同配置（当时 top_n / 仓位口径不同）。
**这正是 P1.5 必须按多判据组合 + 配对制度重排裁决、禁止单点读数定去留的原因**：
政策层的价值对配置与窗口高度敏感，单批单窗口读数不足以裁决。
P1.5 必须按多判据组合（配对 ΔCAGR 分布 + 亏损日频率/幅度 + §4.8 结构成本）裁决，
禁止用本表单点读数直接定去留。**（裁决已执行：2026-10-01 退役，方案 F8；本段读数存为事实登记。）**

**Δ（B1−B0）= +6.0pp MaxDD 的构成归因（P1.5 附录任务登记，2026-10-01 P0 评审）**：
现有证据链上该差值 ≈ 冻结表口径回撤改善（+1.02pp，R-004）+ 回补增量（+1.63pp，R-007 §6）
+ 窗口（4 折 vs 14 折）/ 配置（top_n=10→20 等）差异（剩余约 3pp 未分解）。
注意 **R-004 时代的对照组无回补组件**——本节早前把方向相反仅归因于「窗口 + 配置」两项，
漏了回补构成项。P1.5 裁决报告附录必须做 +6.0pp 的**显式分解归因**（政策门控 / 回补 /
窗口与配置各占多少），禁止把混合差值整体记在单一组件名下。（**已履行**，2026-10-01：
分解归因见裁决报告 `docs/reports/p15_policy_layer_adjudication_20261001.md` §8 附录）

## 附录 A：127 键配置指纹快照（F3 修订——runs 契约附录 A 的逐键对账基准）

> 口径 = **全键入指纹 − 显式排除清单**（fail-safe 方向）；排除清单见本附录末尾。
> **唯一权威源 = 代码模块 `src/lazybull/v2/evidence/fingerprint_keys.py`**（F3 起）；
> 本快照为 127 键逐键对账基准（`tests/test_v2_fingerprint_keys.py`），文档清单与代码
> 不一致时以代码为准并回写本文档。

**快照（127 键，自 B0/B1 批次 summary 表头提取）**：

```
说明, Top20_list, Top30_list, Top20_hit_rate, Top20_avg_return_median, Top20_lift_mean,
Top30_hit_rate, Top30_avg_return_median, Top30_lift_mean, split_index,
train_start, train_end, test_start, test_end, model_version,
consensus_revision_cols_live, cashflow_quality_cols_live, train_samples, val_samples, test_samples,
best_iteration, best_iteration_floor_triggered, val_rankic_ir,
bt_total_return, bt_annual_return, bt_max_drawdown, bt_volatility, bt_sharpe, bt_calmar,
bt_trading_days, bt_start, bt_end, bt_top_n, wf_run_id, batch_run_id, batch_period_label,
split_count, final_date, wf_start_date, wf_end_date, algorithm,
train_window_years, test_window_months, val_ratio, label_column, neutral_label_blend_weight,
task, label_transform, n_estimators, max_depth, num_leaves,
learning_rate, subsample, colsample_bytree, min_child_weight, gamma,
reg_alpha, reg_lambda, early_stopping_rounds, early_stopping_metric, rank_weight_enabled,
rank_weight_topk, rank_weight, rank_weight_topk_weight_mode, time_decay_half_life, freshness_strategy,
event_freshness_half_life_days, objective, enable_fundamental, enable_alt, enable_margin,
enable_cyq, enable_fund, enable_express, feature_stability_filter, factor_prune,
factor_exclude_file, ensemble_offsets, ensemble_seeds, ensemble_seed_keep_top_ratio, ensemble_seed_keep_min_models,
enable_enhanced_features, enable_north_features, enable_lhb_features, enable_consensus_features, enable_cashflow_quality_features,
enable_consensus_revision_features, enable_dividend_policy_features, enable_availability_markers, enable_holdertrade_features, holdertrade_feature_set,
enable_repurchase_features, repurchase_feature_set, enable_top10fh_features, top10fh_feature_set, enable_top_inst_features,
stock_domain, oos_backtest, oos_backtest_months, bt_rebalance_freq, bt_initial_capital,
bt_sell_timing, bt_exclude_st, bt_min_list_days, bt_max_weight_per_stock, bt_max_per_industry,
bt_stop_loss_enabled, bt_stop_loss_drawdown_pct, bt_stop_loss_consecutive_limit_down, position_sizing, kelly_vol_window,
kelly_max_leverage, stagger_tranches, enable_early_rebalance_on_empty, no_deploy_train, skip_training,
skip_training_eval, start_model_version, selected_split_indices, downside_penalty, downside_penalty_column,
data_state_id, git_commit, git_dirty, data_daily_latest, data_cs_train_latest,
data_dividend_coverage
```

**排除清单（fail-safe 唯一例外；F3 勘误——补登 `registered_at` 与 `key_*`，与代码模块对齐）**：
- 运行标识类（4 键）：`wf_run_id, batch_run_id, batch_period_label, registered_at`（F3 补登；
  快照 127 键中无 `registered_at`——注册时元数据，不入 summary 表头）；
- 数据/代码态（6 键）：`data_state_id, git_commit, git_dirty, data_daily_latest, data_cs_train_latest, data_dividend_coverage`；
- 统计输出类：逐折回测指标 9 列（`bt_total_return, bt_annual_return, bt_max_drawdown, bt_volatility, bt_sharpe, bt_calmar, bt_trading_days,`
  `bt_start, bt_end`）+ 训练产出（`train_samples, val_samples, test_samples, best_iteration, best_iteration_floor_triggered`）
  + 信号层统计前缀 `key_*`（F3 补登：KEY_* 映射列，即快照的 `Top20_hit_rate` 等 9 键的转换后形态）。
注：`bt_top_n` 等 `bt_*` 前缀中**仅统计输出**排除；`bt_top_n, bt_rebalance_freq, bt_initial_capital, bt_sell_timing,`
`bt_exclude_st, bt_min_list_days, bt_max_weight_per_stock, bt_max_per_industry, bt_stop_loss_*` 等**配置键一律入指纹**。

**Δ（B1−B0）= +6.0pp MaxDD 的构成归因（P1.5 附录任务登记，2026-10-01 P0 评审）**：
现有证据链上该差值 ≈ 冻结表口径回撤改善（+1.02pp，R-004）+ 回补增量（+1.63pp，R-007 §6）
+ 窗口（4 折 vs 14 折）/ 配置（top_n=10→20 等）差异（剩余约 3pp 未分解）。
注意 **R-004 时代的对照组无回补组件**——本节早前把方向相反仅归因于「窗口 + 配置」两项，
漏了回补构成项。P1.5 裁决报告附录必须做 +6.0pp 的**显式分解归因**（政策门控 / 回补 /
窗口与配置各占多少），禁止把混合差值整体记在单一组件名下。（**已履行**，2026-10-01：
分解归因见裁决报告 `docs/reports/p15_policy_layer_adjudication_20261001.md` §8 附录）
