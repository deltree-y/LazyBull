# v2 MVP 基线冻结登记（B0 / B1 双臂）

> P0 交付物：现产线默认配置的双臂基线（配置指纹 + 逐折收益路径）。
> 冻结时点 = 启动日（2026-09-30）。裁决前 P2a 不开工。

## 1. 双臂定义

| 臂 | 含义 | 配置要点 |
|---|---|---|
| **B0** | 无暴露政策基线（neutral） | 不启用 exposure-policy / exposure-table / replenish；其余与 B1 逐字一致 |
| **B1** | 默认臂 e2online_r（含政策） | E2-滚动250、λ=0.5、在线现算、对称回补开、不裁剪覆盖（20181126） |

**迁移目标 = P1.5 政策层裁决的胜者**（多判据组合：配对 ΔCAGR 分布 + 亏损日频率/幅度 + §4.8 结构成本）。

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
禁止用本表单点读数直接定去留。

**Δ（B1−B0）= +6.0pp MaxDD 的构成归因（P1.5 附录任务登记，2026-10-01 P0 评审）**：
现有证据链上该差值 ≈ 冻结表口径回撤改善（+1.02pp，R-004）+ 回补增量（+1.63pp，R-007 §6）
+ 窗口（4 折 vs 14 折）/ 配置（top_n=10→20 等）差异（剩余约 3pp 未分解）。
注意 **R-004 时代的对照组无回补组件**——本节早前把方向相反仅归因于「窗口 + 配置」两项，
漏了回补构成项。P1.5 裁决报告附录必须做 +6.0pp 的**显式分解归因**（政策门控 / 回补 /
窗口与配置各占多少），禁止把混合差值整体记在单一组件名下。
