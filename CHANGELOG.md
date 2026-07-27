# Changelog

All notable changes to this project will be documented in this file.

## [0.85.22] - 2026-07-27

### Removed

- **移除盈亏动态持仓功能**：删除 `src/lazybull/backtest/holding_strength.py` 及所有相关代码、配置开关、测试。包括：
  - `enable_profit_based_holding` 总开关及所有子功能（亏损提前换出、盈利延续持有、ATR 动态止损、时间止损、strength_veto 二次确认）
  - 所有相关 dataclass 字段、CLI 参数、backtest engine 初始化参数
  - 纸面交易中的 `evaluate_profit_extension`、`evaluate_early_exit`、`_check_early_exit` 等方法
  - `batch_walk_forward.ps1` 中的全部分段配置
  - `compare_walk_forward.py` 中的中文标签映射和参数列表
  - `test_holding_strength.py` 及所有测试文件中的相关引用

- **移除整体持仓止盈功能**：删除 `take_profit_threshold`、`take_profit_refill` 及相关逻辑（整体止盈检查、止盈补位、元数据处理）

## [0.85.21] - 2026-07-27

### Removed

- **移除风险惩罚(Bad-Pick)功能**：删除 `src/lazybull/risk/bad_pick.py` 及所有相关代码、配置开关、测试。该功能在实际使用中效果不佳，简化核心架构。移除内容包括：
  - `BadPickConfig`、`RegimeBadPickConfig`、`apply_conditional_penalty`、`detect_market_regime` 等核心类/函数
  - `learn_risk_penalty_config` 训练函数及 `_apply_risk_penalty` / `_apply_risk_penalty_scores` 推理函数
  - `BAD_PICK_CLASSIFIER_FEATURES`、`MARKET_STATE_FEATURES`、`RISK_PENALTY_DEFAULT_LAMBDA_GRID` 等常量
  - `model_registry.py` 中的分类器内嵌逻辑
  - `walk_forward.py` 和 `train_ml_model.py` 中的风险惩罚学习/评估/参数定义
  - `batch_walk_forward.ps1` 中的风险惩罚配置/参数拼接/扫描循环
  - `compare_walk_forward.py` 中的风险惩罚指标列
  - `test_bad_pick_conditional.py` 及 `test_ml_signal.py`/`test_train_core_val_embargo.py`/`test_walk_forward.py` 中的相关测试

## [0.85.20] - 2026-07-26

### Changed

- **Bad-Pick 分类器特征恢复**：将 `BAD_PICK_CLASSIFIER_FEATURES` 恢复为此前 21 因子版本（波动/量价、成交额/振幅/布林、技术形态、动量/反转、开盘/资金、估值/行为），替换 v0.85.18–v0.85.19 引入的 20 因子重构版本。同步更新测试中 `kdj_d` → `kdj_j`。

## [0.85.19] - 2026-07-25

### Changed

- **Bad-Pick 去变体优化**：移除 3 个与主模型 zscore 版信息重复的原始值变体（`pb`/`zscore_bp`、`dv_ttm`/`zscore_dv_ttm`、`turnover_rate`/`zscore_turnover_rate`），新增 3 个真正独立的维度：`macd_dea`（补全 MACD 金叉/死叉判断）、`amount_ma5`（5日流动性枯竭预警，主模型只用20日）、`vol_ratio_5`（5日原始量比，区别于主模型 vol_ratio_20）。因子总数保持 20，信号维度从 6 类扩展为 7 类（新增流动性维度）。

## [0.85.18] - 2026-07-25

### Changed

- **Bad-Pick 因子重构**：从 18 个调整为 20 个，按 6 个独立信号维度重新组织。移除 5 个同质化/稀疏因子（`alpha_industry_10`、`ind_momentum_rank`、`margin_net_buy_ratio`、`lg_net_amount`、`vol_burst_10`），新增 7 个覆盖新维度的因子：`ret_5`（超短期反转）、`ma_deviation_5`/`ma_deviation_10`（均值回归）、`vol_burst_5`（5日量能异动）、`pb`/`dv_ttm`/`ep_ttm`（绝对估值锚定，区别于主模型的行业中性版本）。

## [0.85.17] - 2026-07-25

### Added

- **Bad-Pick AUC 阈值可配置**：新增 `--risk-penalty-clf-auc-threshold` 参数（walk_forward.py）和 `$risk_penalty_clf_auc_threshold_list` 批量变量（batch_walk_forward.ps1），默认 0.55。`learn_risk_penalty_config()` 新增 `clf_auc_threshold` 参数替代硬编码。

## [0.85.16] - 2026-07-25

### Fixed

- **Bad-Pick regime 样本门槛过高导致搜索被跳过**：`min_regime_samples` 从 `max(200, min_total_samples // 4)` 降为 `max(50, min_total_samples // 8)`，并新增全局兜底——当所有 regime 都不满足门槛时，回退到全量校准集做单网格搜索。修复小校准集（如 260 样本）下 threshold/lambda 恒为 (1.0, 0.0) 的问题。

## [0.85.15] - 2026-07-25

### Changed

- **Bad-Pick 分类器因子调优**：移除 `fund_hold_ratio`、`fund_hold_ratio_chg`（在 factor_exclude_list 中，ICIR 或覆盖率不达标）；新增 `margin_net_buy_ratio`（融资行为）、`weight_avg_bias`（筹码成本）、`turnover_rate`（原始换手率）、`vol_burst_10`（10日爆量），均不在主模型且不在排除列表。因子总数 16→18，非主模型因子 10→12。

## [0.85.14] - 2026-07-25

### Changed

- **Bad-Pick 因子按候选池实证重新筛选**：复现最新 split 的 Top150 候选池，并按覆盖率、单因子坏票分离度和三个时间段方向稳定性筛选。分类器特征由 27 个调整为 16 个，其中 10 个不在最新主模型中；移除在近期模型中始终未被使用的 3 个一致预期因子和 `zscore_fcf_yield`，以及方向翻转、低覆盖或与主模型重复度较高的因子。
- **Bad-Pick 改为严格样本外校准**：候选日期按 70%/10%/20% 拆分为训练、early-stop、calibration；AUC、regime 和惩罚参数只在最后 20% 日期上计算，随后按选定树数使用全部候选样本重训部署分类器。样本外 AUC 启用门槛调整为 0.55，并继续要求 TopK 中位数或 RankIC IR 改善。

### Fixed

- 修复旧版在分类器训练全量样本上计算 AUC、并在同批样本搜索惩罚参数导致的校准过拟合。近期记录中训练内 AUC 可达 0.80，但测试集 `swap_alpha` 仍为负，现改为严格时间留出评估。

## [0.85.13] - 2026-07-25

### Changed

- **Bad-Pick 特征回退至 32 因子版**：从 v0.85.12 的 20 特征精简版回退到 v0.85.10 的 27+5=32 特征版，保留全部候选因子供重新评估。

## [0.85.12] - 2026-07-25

### Changed

- **Bad-Pick 特征精简回退**：v0.85.8 过度追求占比目标，引入了多个有问题的因子导致分类器效果退化。
  - 移除 7 个：`mkt_ma250_ratio`/`mkt_turnover_ratio`（同日全市场相同值，截面零区分力）、`kdj_d`（与 kdj_j 相关系数>0.95）、`ps_ttm`/`net_mf_amount_mean_5`（原始量纲与 zscore 混合）、`zscore_cons_eps_dispersion_chg`/`zscore_cons_analyst_count_chg`（短窗口内稀疏）。
  - 保留 14 个核心 + 6 个新增 = 20 特征（+5 MARKET_STATE = 25）。
  - 设计原则改为"每个因子必须有明确的截面区分力"。

## [0.85.11] - 2026-07-25

### Fixed

- **Bad-Pick 空切片 NaN 填充警告消除**：新增因子在部分截面可能全列为 NaN，`Series.median()` 在空切片上触发 `RuntimeWarning`。修复：填充前先 `dropna()` 判断有效值数量，全 NaN 列直接填 0.0。覆盖 `prepare_classifier_features` 和 `learn_conditional_bad_pick_config` 两处。

## [0.85.10] - 2026-07-25

### Fixed

- **Bad-Pick 特征列表与 MARKET_STATE_FEATURES 去重**：`mkt_drawdown_20` 同时出现在 `BAD_PICK_CLASSIFIER_FEATURES`（v0.85.8 新增）和 `MARKET_STATE_FEATURES`（原有）中，导致训练时两列表拼接后列名重复，`X_clf[col]` 返回 DataFrame 而非 Series，触发 `ValueError: The truth value of a Series is ambiguous`。修复：从 `BAD_PICK_CLASSIFIER_FEATURES` 移除 `mkt_drawdown_20`（仍通过 MARKET_STATE 传入），并在 `learn_conditional_bad_pick_config` 加入 `dict.fromkeys` 去重防护。

## [0.85.9] - 2026-07-25

### Fixed

- **Bad-Pick 分类器实际可用特征数修复**：`prepare_training_data` 的内存优化裁剪掉了不在主模型 `feature_columns` 中的列，导致坏票分类器的 14 个新因子（`atr_pct_14`、`body_length`、`ps_ttm` 等）在训练时被丢弃，实际仅剩 15 个。修复后在 `needed_cols` 中显式保留 `BAD_PICK_CLASSIFIER_FEATURES` + `MARKET_STATE_FEATURES`。

## [0.85.8] - 2026-07-24

### Changed

- **Bad-Pick 分类器特征 v2 优化**：实现 50% 因子不在主模型中，确保惩罚信号与排序信号正交。
  - 移除 8 个冗余/低增量因子：`vol_ratio_20`（与 vol_burst_20 冗余）、`spec_score`（vol×size 衍生）、`rsi_14`（标准指标）、`zscore_acceleration`（动量饱和）、`zscore_opening_strength`（正交性弱）、`winner_rate`（主模型权重 0.007）、`zscore_or_yoy`（netprofit_yoy 更直接）、`zscore_quick_ratio`（debt_to_assets 已覆盖）。
  - 新增 13 个主模型未使用的因子：
    - 日内风险：`atr_pct_14`（ATR日内波幅）、`body_length`（K线实体，主模型明确排除）
    - 技术确认：`kdj_d`（KDJ慢线，主模型只用J值）、`net_mf_amount_mean_5`（5日平均资金流，主模型排除）
    - 市场环境：`mkt_ma250_ratio`（长期牛熊）、`mkt_drawdown_20`（回撤深度）、`mkt_turnover_ratio`（市场拥挤度）
    - 机构博弈：`fund_count_chg`（基金撤离）、`fund_hold_ratio_chg`（机构减持）
    - 价值陷阱：`ps_ttm`（市销率，主模型排除。高P/S+低P/E=暂时性盈利膨胀）
    - 一致预期修正：`zscore_cons_eps_dispersion_chg`（分歧度恶化）、`zscore_cons_analyst_count_chg`（分析师撤退）
    - 现金质量：`zscore_fcf_yield`（自由现金流，价值陷阱识别）
  - 注：`weight_avg_bias` 因数据中缺少 `close_adj` 列而不可用，替换为 `ps_ttm`。
  - 净变化：23 → 28 个特征，主模型内/外比例 14:14 = 50%:50%。

## [0.85.7] - 2026-07-24

### Changed

- **Bad-Pick 分类器特征优化**：减少与主模型高权重因子的重叠，增强风险维度的正交性。
  - 移除 7 个主模型高权重因子（`zscore_ma_deviation_20` 0.031、`lg_net_amount_sum_5` 0.022、`zscore_turnover_rate` 0.018、`zscore_amount_ma20` 0.015、`zscore_bb_width` 0.012、`amplitude` 0.011、`zscore_volatility_5` 0.010），这些因子与主模型排序信号高度相关，导致惩罚冗余。
  - 新增 8 个主模型低权重/未覆盖的风险维度因子：基本面恶化（`zscore_roe_dt`、`zscore_or_yoy`、`zscore_netprofit_yoy`）、杠杆与流动性（`zscore_debt_to_assets`、`zscore_quick_ratio`）、现金质量（`zscore_cf_nm`）、资金博弈（`zscore_order_imbalance`、`holder_num_chg`）。
  - 净变化：22 → 23 个特征，覆盖更全面的"质量陷阱"与"尾部风险"信号。

## [0.85.6] - 2026-07-24

### Fixed

- **批量汇总年化收益率与 broker 日志完全对齐**：`scripts/batch/batch_paper_trade.ps1` 的 `Get-NavSummary` 不再从 `nav.parquet` 重新计算（其价格来源与 broker 不同），改为直接读取 `config.yaml`（`initial_capital`/`account_start_date`）+ `account.json`（`cash`/`positions`）+ clean daily 收盘价，与 broker 的 `print_positions_summary` → `_calculate_annualized_return` 使用完全相同的数据源和 CAGR 公式。

## [0.85.5] - 2026-07-23

### Added

- **风险惩罚训练参数接入 batch_walk_forward.ps1 并打印**：
  - `scripts/walk_forward.py` 新增 10 个 CLI 参数：
    - 校准行为：`--risk-penalty-candidate-topk`、`--risk-penalty-bad-bottom-pct`、`--risk-penalty-min-bad-samples`、`--risk-penalty-min-total-samples`
    - 分类器超参：`--risk-penalty-clf-max-depth`、`--risk-penalty-clf-n-estimators`、`--risk-penalty-clf-learning-rate`、`--risk-penalty-clf-subsample`、`--risk-penalty-clf-colsample-bytree`、`--risk-penalty-clf-early-stopping-rounds`
  - 在 split 训练与部署训练两处 `learn_risk_penalty_config()` 调用中透传上述 10 个参数。
  - `learn_risk_penalty_config()` 签名新增 6 个分类器超参（`clf_max_depth`/`clf_n_estimators`/`clf_learning_rate`/`clf_subsample`/`clf_colsample_bytree`/`clf_early_stopping_rounds`），替代原硬编码常量。
  - `_log_risk_penalty_params()` 扩展打印，含分类器超参。
  - walk-forward 汇总 CSV 新增对应 10 列。
- **batch_walk_forward.ps1 接入**：
  - 配置区新增 6 个分类器超参列表变量。
  - 自动加入笛卡尔积遍历（+6 层 foreach）与命令行构建，参与总任务数统计。

## [0.85.4] - 2026-07-23

### Changed

- **walk-forward 风险惩罚强度支持可配调节（lambda scale/grid）**：
  - `scripts/walk_forward.py` 新增参数 `--risk-penalty-lambda-scale` 与 `--risk-penalty-lambda-grid`。
  - 在 split 训练与部署训练路径中，`learn_risk_penalty_config()` 现透传 `lambda_grid`，支持更温和/更激进的惩罚强度探索。
  - 当未显式提供 grid 且 `scale != 1.0` 时，自动按默认网格缩放生成候选，保持原默认行为可兼容。

### Added

- **walk-forward 汇总新增风险惩罚效果诊断列**：
  - `risk_penalty_penalized_ratio`（惩罚覆盖率）
  - `risk_penalty_penalty_mean`（惩罚均值）
  - `risk_penalty_topk_changed_days_ratio`（TopN变更日占比）
  - `risk_penalty_swap_alpha`（替换收益贡献）
- **批量对比报表接入风险惩罚效果指标**：
  - `scripts/compare_walk_forward.py` 已支持上述诊断列的聚合、中文映射与主表展示。
- **单元测试补充**：
  - `tests/test_walk_forward.py` 新增 lambda 网格解析与惩罚效果指标计算测试。

## [0.85.3] - 2026-07-23

### Fixed

- **修复 conditional bad-pick 在 walk-forward 评估侧的特征名不一致导致惩罚静默失效**：
  - `scripts/walk_forward.py` 的 `_apply_risk_penalty_scores()` 在 v2 路径中，现会按分类器 `feature_names_in_` 对 `X_clf` 做列对齐（补缺失列、剔除多余列、重排列顺序）。
  - 修复了分类器训练列不含 `mkt_drawdown_20` 但评估输入包含该列时 `predict_proba` 抛错并被静默回退到 `pred_score` 的问题。
  - v2 预测异常日志补充了 `bad_pick_model_version` 与具体异常信息，便于定位线上回退原因。
  - 新增 `tests/test_walk_forward.py` 回归测试，覆盖“输入含额外市场特征列仍能成功应用惩罚”的场景。

## [0.85.2] - 2026-07-23

### Fixed

- **修复 conditional bad-pick 在 walk-forward 评估侧的分类器兜底加载不可达分支**：
  - `scripts/walk_forward.py` 的 `_apply_risk_penalty_scores()` 原先在 `_clf_model` 缺失时会提前返回，导致后续按 `bad_pick_model_version` 兜底加载分类器的逻辑永远无法执行。
  - 现已调整为：先尝试按模型版本恢复分类器，兜底失败后再跳过惩罚，并输出明确失败原因日志。
  - 新增 `tests/test_walk_forward.py` 回归测试，覆盖“_clf_model 缺失 + bad_pick_model_version 兜底”路径。

## [0.85.1] - 2026-07-23

### Fixed

- **年化收益率统一为 CAGR 公式**：`src/lazybull/paper/broker.py` 的 `_calculate_annualized_return()` 从简单线性年化改为复合年化（CAGR），与批量汇总脚本一致。
- **批量纸面交易汇总结束日期修复**：`scripts/batch/batch_paper_trade.ps1` 的 `Get-NavSummary` 现在使用实际最终交易日（`$finalTradeDate`）而非 `nav.parquet` 末行日期，确保年化收益率计算覆盖完整的实际交易区间。

## [0.85.0] - 2026-07-23

### Changed

- **条件式 Bad-Pick 模型（完全替换线性惩罚）**：
  - src/lazybull/risk/bad_pick.py（新建）：BadPickConfig/RegimeBadPickConfig, detect_market_regime() 三层OR判断, apply_conditional_penalty() 阈值门控扣分。
  - src/lazybull/ml/train_core.py：learn_risk_penalty_config() 重写为训练XGB二分类器+分位数网格校准regime阈值+per-regime二维网格搜索。
  - src/lazybull/signals/ml_signal.py：_apply_risk_penalty() 支持v1/v2双模式，_load_model() 自动加载坏票分类器。
  - src/lazybull/ml/model_registry.py：register_model() 自动注册坏票分类器并回填版本号。
  - scripts/walk_forward.py 和 scripts/train_ml_model.py：日志与评估逻辑同步适配v2。
- **分类器特征从15个扩展到22个**：新增主力资金、动量衰竭、日内形态、极端估值、分析师分歧等维度。
- **所有阈值用分位数定义**：每个walk-forward split独立校准，自动适应市场数据分布进化。

### Added

- 	ests/test_bad_pick_conditional.py：21个测试覆盖配置序列化、regime检测、特征提取、门控逻辑、边界条件。

## [0.84.0] - 2026-07-22
