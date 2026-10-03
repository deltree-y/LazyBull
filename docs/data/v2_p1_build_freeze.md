# v2 P1 数据底座·构建冻结物（开工前冻结）

> 版本：v1（2026-10-02）　状态：**已生效**（P1 开工闸门，随计划批准生效）
> 依据：`docs/contracts/v2_architecture_plan.md` §8-P1 行「开工前冻结构建窗口口径」+ §4.1 数据存储契约 + P1 实施计划（2026-10-02 批准）。
> 冻结纪律：本文 commit（git `d729f82` 之后首个 P1 提交）后口径即冻结；此后任何改变构建行为的代码改动触发**重冻结程序**（对应段基线重冻结 + 差异台账回溯标注，沿方案 §9.1 基线稳定条款）。

## 1. 构建窗口口径（回填构建器必须逐位遵守）

| 项 | 冻结值 | 依据 |
|---|---|---|
| 回填区间 | **20120104 ~ 20260702**（= cs_train 现产物覆盖区间，3,518 交易日） | 现产物实测 |
| 预热窗口 | 加载窗口 = 起点**前 7 个自然月** + 终点**后 1 个自然月**；fund_portfolio 单独回溯 18 个月 | `features/pipeline.py:383-385,490-495` |
| 标签口径 | T 日信号，**T+1 收盘（close_adj）买入，T+1+N 开盘（open_adj）卖出**；`y_ret_N = open_adj(T+1+N)/close_adj(T+1) − 1` | `features/labels.py:6-8,98-101` |
| horizons | **{5, 10, 20}**（多值模式 ⇒ `label_filter_mode="all"`，行须三标签全非空才保留） | `scripts/build_clean_features.py:263-270` |
| require_label | True（cs_train 语义；无标签拒绝落盘沿 `storage.py:592-597`） | 现产物契约 |
| 样本过滤 | `is_st==0 & list_days>=365 & is_suspended==0`（min_list_days=365） | `builder/static_extra.py:450` |
| 中性化 | **行业（申万）+ 规模双开**（现产物同时含 `zscore_*` 57 列与 `zscore_*_sz` 56 列） | 现产物实测 |
| 可选因子组 | 以现产物实际列集为准（哨兵 `cashflow_quality_schema_v2` / `cons_revision_schema_v2` / `dividend_schema_v1` 在列；holdertrade/repurchase/top10fh/top_inst 四族**不**在 cs_train 列集，见 §6 物化拍板） | 现产物实测 |

**截面口径（核心，0.201.1 探测教训登记）**：构建编排顺序冻结为——合并标签 → 过滤标记 → **行过滤（步骤 11）→ 中性化（步骤 12）→ 个股特征 → 市场状态**（`features/builder/orchestration.py:204-229`）。即 **`zscore_*` / `neu_*` / `mkt_*` 全部在「过滤后截面」上计算**（过滤后截面 = 上述样本过滤 + 标签全非空后的同日行集）。回填构建器必须保持同一顺序与同一截面；单日增量构建（影子通路）必须使用与批量回填**完全一致**的过滤口径（全市场统一截面），禁止以当日局部截面重排。

**panel 行口径拍板**：v2 panel 与 cs_train **逐行一致**（同一过滤后截面）——等价复刻优先；全截面分母需求（pct_* 母截面、M1.5 覆盖率剖面）由 normalized + 母截面构建器承担，不从 panel 取（沿 `risk/terminal_loss/mother_section.py` 契约不变）。

## 2. 列族分组表（383 列 → panel 8 族 + labels 3 表；逐列映射冻结）

校验：383 列全覆盖、无重复、无遗漏、无越界（生成脚本 `temp/p1_gen_column_groups_20261002.py`，产物 `temp/p1_column_groups_20261002.json`；单元 2 将代码化为 `v2/store/column_groups.py` 单一来源）。

| 族 | 列数 | 内容 |
|---|---|---|
| `core` | 52 | 键（trade_date/ts_code）+ clean 复用列（vol/amount/is_st/is_suspended/is_limit_*/list_days/tradable）+ 基础量价（ret_*/vol_ratio_*/ma_deviation_*/amount_ma*/opening_strength/intraday_vol_structure）+ daily_basic 直通与价值红利派生（pb/pe_ttm/ps_ttm/dv_ttm/total_mv/circ_mv/turnover_rate/volume_ratio/ep_ttm/bp/log_*/is_loss/*_missing）+ 申万行业参考（sw_*/in_date） |
| `fundamental` | 69 | fina_indicator 19 + forecast 3 + express 5 + 一致预期/修订 23 + 现金流 9 + 分红 11 |
| `moneyflow` | 44 | 资金流向 14 + 融资融券 6 + 北向 13 + 龙虎榜 11 |
| `technical` | 43 | 筹码 5 + 基金持仓 5 + 高级量价 30 + 个股特征 3 |
| `announcement` | 10（+物化） | 股东户数 3 + 质押 3 + 解禁 2 + 大宗 2；**运行时派生四族 ht_*/rp_*/tfh_*/ti_* 与 has_* 标记物化后入此族（§6）** |
| `risk` | 29 | 步骤 9.5 `_add_risk_factors` 产出块（@register_risk_factor 注册族的 panel 侧部分；注册族另 4 列 short_balance_change_5/unlock_ratio/block_discount_*/short_sell_vol_change_5 物理上随 moneyflow/announcement 族，注册表为逻辑视图不作存储分组依据） |
| `market_state` | 13 | mkt_* 市场状态广播列 |
| `neutralized` | 117 | zscore_* 57 + zscore_*_sz 56 + zscore_size + neu_ret_1/5/10/20 |
| **labels**（独立成表） | 6 | y_ret_5/10/20 + neu_y_ret_5/10/20（训练默认标签 = `neu_y_ret_20`，`ml/train_core/prepare.py:63`） |

逐列清单见附录 A（与 `temp/p1_column_groups_20261002.json` 逐项一致）。

## 3. 冷热分层参数（P1 一次性定型）

- **热区**：最近 **12 个自然月**按日分区 `data/features/panel/YYYYMMDD/<group>.parquet`（N=12 依据：覆盖 7 个月预热窗口 + 影子对账窗口 + 单日增量构建期）。
- **冷区**：封存月压实 `data/features/panel_archive/YYYY-MM/<group>.parquet`，范围 2012-01 ~ 2025-06（162 月 × 8 族 ≈ 1,300 文件，贴合方案 ~1,000 量级估计）。
- **单文件规则**：软上限 256MB（超限按列子族拆分）、软下限 32MB（欠限触发压实评审）；回填时逐文件校验并登记。
- **labels**：`data/labels/y_ret_{5,10,20}/YYYYMMDD.parquet`（日分区，不分冷热；单表列 = `ts_code, trade_date, y_ret_N, neu_y_ret_N, maturity_status`）。
- **labels 成熟度生命周期**：封存时点 = T + max(h)=T+20 端点可算日（T+1 起算第 20 个交易日存在）；封存前幂等可重写（`maturity_status=forming`），封存后拒绝改写（`maturity_status=sealed`）；回填区间内全部标签已成熟（区间末端 20260702 的 T+20 ≈ 20260730 ≤ raw 末端 2026-07-31）。

## 4. manifest 规格要点（单元 1 实现的输入）

- 列级：`列名 → {group, source, definition_version, available_from, backfilled_at, status}`；`available_from` 初值 = 列族实际可用起点（单元 5 剖面报告回填精化；回填的新因子只允许在其可用起点后参与训练）。
- per-分区**内容指纹**（sha256-16，封存/压实/备份/恢复复用同一指纹）；**依赖声明**（raw 清单 + factors 派生函数标识）；**两阶段提交**（临时名写入 → 原子纳入 manifest → 孤儿 GC + 单机写锁）。
- 拼接键固定 `(trade_date, ts_code)`；core 族为左表 left join；族缺日整族 NaN + 计数登记（§4.1 补丁 1）。

## 5. raw 快照（水位 + 指纹 = 快照；raw 只追加不可变，不物理复制）

- **数据态锚点**：与 B0/B1 基线冻结同数据态 **`9d0408ee`**——8 个数据态数据集（daily/adj_factor/daily_basic/moneyflow/stk_limit/suspend/stock_st/margin_detail）水位全部 = 2026-07-31 ✓（2026-10-02 复核一致），cs_train 最新 = 20260702 ✓。
- **全量水位**（34 数据集 + 4 单文件，2026-10-02 采集）：日频类 2026-07-31（top_list 2026-08-07 / block_trade 2026-08-05）；季频类 2026-06-30；年分区类 2026-12-31（dividend/share_float/report_rc/repurchase/stk_holdertrade/top10_floatholders/top_inst）；pledge_stat 2025-12-31；cyq_perf/moneyflow_hsgt 2026-07-31；单文件 trade_cal/stock_basic/shenwan_industry/stk_holdernumber/hot_rank。
- **抽验指纹**（sha256-16，附录 B 全表）：daily `2783617a92ab2ce9`、adj_factor `43c16bd328e4e29f`、cs_train(20260702) `54678be3decbd5d5`、clean/daily(2026-07-31) `d6a6f7857393896f` 等 10 项。
- **代码态**：git `d729f82`（v0.203.5，F9），工作树干净。
- 回填期间纸面软生产继续运行（raw 只追加，水位 ≤2026-07-31 的冻结段不受影响）。

## 6. 运行时派生四族处置（拍板）

按方案 §4.3「运行时派生退役：历史区间一律物化到列族」——holdertrade / repurchase / top10_floatholders / top_inst 四族及 availability 标记（has_*）在 v2 panel 中**物化进 `announcement` 族**，经 `build_*_feature_frame` 单一实现接线（与现四侧运行时派生同一数值路径）。manifest 登记 `status=deprecated`（已终结家族，默认链路不消费；物化是数据资产合规，不代表重开实验）。对账口径 = **列交集**（现 cs_train 无此四族列，panel 多出部分为预期差异，先登记后比对）。

## 7. 验收闸门参数（单元 4 执行）

- 对账门：母截面三段式（块级逐值 atol=1e-6 → 合并 → 判定）；实现漂移判据 = 超容差行占比 > 1e-4 或任一列 max|Δ| > 0.05 ⇒ raise；极稀疏差异 ⇒ 数据态漂移告警 + 离群登记（沿 `MOTHER_OUTLIER_SHARE_LIMIT` / `MOTHER_HARD_ATOL`）。
- 出口 = 残差 100% 归因（差异逐列归因：截面口径 / 派生物化 / 数据态）。
- 性能闸门：全历史跨列族加载 ≤ 现状 1.5 倍（双口径：抽 5 列 + 全列）。
- 新增列只写新文件（历史分区指纹零改动校验）。

## 附录 A：列族分组逐列清单

（与 `temp/p1_column_groups_20261002.json` 一致；单元 2 起以 `v2/store/column_groups.py` 为代码单一来源，本附录留档备查。）

- **core（52）**：amount, amount_ma10, amount_ma20, amount_ma5, bp, circ_mv, dv_ttm, dv_ttm_missing, ep_ttm, in_date, intraday_vol_structure, is_limit_down, is_limit_up, is_loss, is_st, is_suspended, list_days, log_circ_mv, log_total_mv, ma_deviation_10, ma_deviation_20, ma_deviation_5, opening_strength, pb, pe_ttm, pe_ttm_missing, ps_ttm, ret_1, ret_10, ret_20, ret_5, sw_industry, sw_industry_code, sw_industry_id, sw_l1, sw_l1_code, sw_l1_id, sw_l2, sw_l2_code, sw_l2_id, sw_l3, sw_l3_code, total_mv, tradable, trade_date, ts_code, turnover_rate, vol, vol_ratio_10, vol_ratio_20, vol_ratio_5, volume_ratio
- **fundamental（69）**：assets_turn, capex_to_ocf, cashflow_freshness_days, cashflow_quality_schema_v2, cf_nm, cf_sales, cons_analyst_count_30d, cons_analyst_count_chg, cons_eps_dispersion, cons_eps_dispersion_chg, cons_eps_mean_fy0, cons_eps_mean_fy1, cons_eps_mean_fy2, cons_eps_mean_fym1, cons_eps_revision_30d, cons_eps_revision_accel, cons_eps_yield_fy0, cons_eps_yield_fy1, cons_eps_yield_fy2, cons_eps_yield_fym1, cons_rating_score, cons_rating_upgrade_ratio, cons_revision_freshness_days, cons_revision_schema_v2, cons_target_price_mid, cons_target_upside, cons_target_upside_chg, consensus_freshness_days, current_ratio, debt_to_assets, dividend_continuity_5y, dividend_days_to_ex_date, dividend_freshness_days, dividend_growth_3y, dividend_growth_5y, dividend_hist_missing, dividend_payout_ratio, dividend_recent_imp_ann_10d, dividend_schema_v1, dividend_stability_5y, dividend_yield_hist_12m, equity_yoy, express_freshness_days, express_profit_yoy, express_revenue_yoy, express_roe, express_surprise, fcf, fcf_yield, forecast_chg_mid, forecast_freshness_days, forecast_type_score, fundamental_freshness_days, grossprofit_margin, int_to_talcap, inv_turn, netprofit_margin, netprofit_yoy, ocf, ocf_to_profit, ocf_to_revenue, or_yoy, profit_dedt, q_gr_yoy, q_ocf_to_sales, quick_ratio, roa, roe_dt, roe_waa
- **moneyflow（44）**：elg_net_amount, elg_net_amount_sum_20, elg_net_amount_sum_5, lg_net_amount, lg_net_amount_sum_20, lg_net_amount_sum_5, lhb_amount_rate, lhb_cont_on_list, lhb_cont_up_days_20, lhb_cont_up_days_5, lhb_net_amount, lhb_net_rate, lhb_net_sum_20, lhb_net_sum_5, lhb_on_list, lhb_reason_count, lhb_up_days_20, margin_net_buy, net_mf_amount, net_mf_amount_mean_20, net_mf_amount_mean_5, net_mf_amount_sum_20, net_mf_amount_sum_5, north_net_buy, north_net_buy_ma20, north_net_buy_ma5, north_net_buy_sign_streak, north_net_buy_sum5, north_net_buy_z20, north_turnover, north_turnover_change_streak, north_turnover_flag, north_turnover_ma20, north_turnover_ma5, north_turnover_sum5, north_turnover_z20, order_imbalance, order_imbalance_mean_20, order_imbalance_mean_5, rqye_rzye_ratio, rzye_chg_20, rzye_chg_5, short_balance_change_5, short_sell_vol_change_5
- **technical（43）**：acceleration, alpha_industry_10, alpha_industry_20, alpha_industry_5, amplitude, atr_14, atr_pct_14, bb_lower, bb_middle, bb_pct, bb_upper, bb_width, body_length, cost_concentration, fund_count, fund_count_chg, fund_hold_ratio, fund_hold_ratio_chg, fund_portfolio_freshness_days, ind_momentum_rank, ind_ret_avg, is_new_stock, kdj_d, kdj_j, kdj_k, lower_shadow, macd_dea, macd_dif, macd_hist, rsi_14, size, spec_score, upper_shadow, vol_burst_10, vol_burst_20, vol_burst_5, volatility_10, volatility_20, volatility_5, weight_avg_bias, winner_rate, winner_rate_chg_20, winner_rate_chg_5
- **announcement（10 + 物化四族）**：block_discount_avg_10d, block_discount_days_10d, days_to_unlock, holder_freshness_days, holder_num_chg, holder_num_chg_2q, pledge_freshness_days, pledge_ratio, pledge_ratio_prev, unlock_ratio（物化追加：ht_*/rp_*/tfh_*/ti_*/has_* 全列，以 manifest 登记为准）
- **risk（29）**：amihud_illiq_20, amount_cv_20, cvar_95_20, downside_corr_20, downside_vol_20, drawdown_duration, earnings_yield, gap_risk, garch_persistence, high_low_range_ratio, kurtosis_20, max_drawdown_20, momentum_decay, parkinson_vol_20, pledge_delta, pledge_high_flag, pledge_ratio_decayed, ret_volatility_ratio, skewness_20, turnover_cv_20, turnover_percentile, unlock_risk_flag, up_down_vol_ratio, var_95_20, vol_of_vol_20, vol_ratio_5_20, vol_regime_percentile, volume_climax_days, volume_price_divergence
- **market_state（13）**：mkt_adv_dec_ratio, mkt_atr_pct, mkt_atr_pct_ma250, mkt_drawdown_20, mkt_ma250_ratio, mkt_ma_trend, mkt_ret_avg_20, mkt_ret_avg_60, mkt_ret_vol_20, mkt_turnover_ratio, mkt_turnover_std, mkt_vol_20, mkt_vol_cnt
- **neutralized（117）**：neu_ret_1, neu_ret_10, neu_ret_20, neu_ret_5, zscore_acceleration, zscore_acceleration_sz, zscore_amount_ma20, zscore_amount_ma20_sz, zscore_assets_turn, zscore_assets_turn_sz, zscore_bb_width, zscore_bb_width_sz, zscore_bp, zscore_bp_sz, zscore_capex_to_ocf, zscore_capex_to_ocf_sz, zscore_cf_nm, zscore_cf_nm_sz, zscore_cf_sales, zscore_cf_sales_sz, zscore_cons_analyst_count_chg, zscore_cons_analyst_count_chg_sz, zscore_cons_eps_dispersion, zscore_cons_eps_dispersion_chg, zscore_cons_eps_dispersion_chg_sz, zscore_cons_eps_dispersion_sz, zscore_cons_eps_revision_accel, zscore_cons_eps_revision_accel_sz, zscore_cons_rating_upgrade_ratio, zscore_cons_rating_upgrade_ratio_sz, zscore_cons_target_upside, zscore_cons_target_upside_chg, zscore_cons_target_upside_chg_sz, zscore_cons_target_upside_sz, zscore_current_ratio, zscore_current_ratio_sz, zscore_debt_to_assets, zscore_debt_to_assets_sz, zscore_dividend_continuity_5y, zscore_dividend_continuity_5y_sz, zscore_dividend_days_to_ex_date, zscore_dividend_days_to_ex_date_sz, zscore_dividend_growth_3y, zscore_dividend_growth_3y_sz, zscore_dividend_growth_5y, zscore_dividend_growth_5y_sz, zscore_dividend_payout_ratio, zscore_dividend_payout_ratio_sz, zscore_dividend_recent_imp_ann_10d, zscore_dividend_recent_imp_ann_10d_sz, zscore_dividend_stability_5y, zscore_dividend_stability_5y_sz, zscore_dividend_yield_hist_12m, zscore_dividend_yield_hist_12m_sz, zscore_dv_ttm, zscore_dv_ttm_sz, zscore_elg_net_amount_sum_20, zscore_elg_net_amount_sum_20_sz, zscore_equity_yoy, zscore_equity_yoy_sz, zscore_fcf_yield, zscore_fcf_yield_sz, zscore_grossprofit_margin, zscore_grossprofit_margin_sz, zscore_int_to_talcap, zscore_int_to_talcap_sz, zscore_intraday_vol_structure, zscore_intraday_vol_structure_sz, zscore_inv_turn, zscore_inv_turn_sz, zscore_log_total_mv, zscore_log_total_mv_sz, zscore_ma_deviation_20, zscore_ma_deviation_20_sz, zscore_macd_hist, zscore_macd_hist_sz, zscore_net_mf_amount, zscore_net_mf_amount_sz, zscore_netprofit_margin, zscore_netprofit_margin_sz, zscore_netprofit_yoy, zscore_netprofit_yoy_sz, zscore_ocf_to_profit, zscore_ocf_to_profit_sz, zscore_ocf_to_revenue, zscore_ocf_to_revenue_sz, zscore_opening_strength, zscore_opening_strength_sz, zscore_or_yoy, zscore_or_yoy_sz, zscore_order_imbalance, zscore_order_imbalance_sz, zscore_pb, zscore_pb_sz, zscore_pe_ttm, zscore_pe_ttm_sz, zscore_profit_dedt, zscore_profit_dedt_sz, zscore_q_gr_yoy, zscore_q_gr_yoy_sz, zscore_quick_ratio, zscore_quick_ratio_sz, zscore_roa, zscore_roa_sz, zscore_roe_dt, zscore_roe_dt_sz, zscore_roe_waa, zscore_roe_waa_sz, zscore_size, zscore_turnover_rate, zscore_turnover_rate_sz, zscore_volatility_10, zscore_volatility_10_sz, zscore_volatility_20, zscore_volatility_20_sz, zscore_volatility_5, zscore_volatility_5_sz
- **labels（6，独立成表）**：y_ret_5, y_ret_10, y_ret_20, neu_y_ret_5, neu_y_ret_10, neu_y_ret_20

## 附录 B：raw 快照抽验指纹（sha256 前 16 位，2026-10-02）

| 数据集（分区） | 指纹 |
|---|---|
| daily（2026-07-31） | 2783617a92ab2ce9 |
| adj_factor（2026-07-31） | 43c16bd328e4e29f |
| daily_basic（2026-07-31） | 48ad81b4621f63ba |
| moneyflow（2026-07-31） | 9071cc45a5946ebf |
| stk_limit（2026-07-31） | f79334470d421dc8 |
| suspend（2026-07-31） | a4db2d44e1e1e9f2 |
| stock_st（2026-07-31） | 31124dcea2facd25 |
| margin_detail（2026-07-31） | 2bfdbeb9ee41de34 |
| cs_train（20260702） | 54678be3decbd5d5 |
| clean/daily（2026-07-31） | d6a6f7857393896f |
