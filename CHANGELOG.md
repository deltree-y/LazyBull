# Changelog

All notable changes to this project will be documented in this file.

## [0.8.3] - 2026-02-14

### Added

- **训练运行日志CSV追加记录功能**
  - 新增 `src/lazybull/ml/run_logger.py` 模块，提供训练运行记录的结构化存储与CSV追加功能
  - `TrainingRunRecord` 数据类：记录每次训练的完整信息
    - 基本信息：时间戳、版本号、训练日期区间、标签、任务类型
    - 训练配置：label_transform、winsorize_p、分类任务参数（pos_quantile/pos_topk/scale_pos_weight及模式）
    - XGBoost超参数：n_estimators、max_depth、learning_rate、subsample、colsample_bytree、gamma、reg_alpha、reg_lambda、early_stopping_rounds、tree_method、random_state、n_jobs
    - 数据统计：交易日数、总样本数、过滤后样本数、训练集/验证集样本数、验证集日期范围
    - 训练结果：best_iteration
    - 评估指标：训练集/验证集的MSE、RMSE、R2、IC、RankIC、ACC、AUC、Precision、Recall
    - 逐日评估：RankIC均值/标准差/IR、TopK收益统计
    - 诊断统计：全市场收益、样本数分布、TopK提升和分位数
  - `write_training_run_to_csv()` 函数：支持追加模式写入CSV，自动创建文件和表头
  - `create_training_run_record_from_training_session()` 函数：从训练会话信息创建记录对象
  - **动态列扩展**：新增字段时自动扩展表头，旧行缺失字段留空（向前兼容）

- **训练脚本集成日志记录**
  - `scripts/train_ml_model.py` 新增 `--run-log-csv` 参数，支持自定义日志文件路径（默认 `data/ml_train_runs.csv`）
  - 修改 `load_features_data()` 返回交易日数量
  - 修改 `prepare_training_data()` 返回数据统计（samples_after_filter、val_start_date、val_end_date）
  - 修改 `train_xgboost_model()` 在 train_params 中记录 best_iteration
  - 训练完成后自动记录运行日志到CSV（失败不影响模型保存）

### Documentation

- 新增 `docs/PR/training_run_logging.md` - 本 PR 详细说明
  - 功能介绍：CSV日志结构、字段说明、使用方法
  - 示例命令：如何使用 --run-log-csv 参数
  - 分析建议：如何利用CSV进行模型对比与超参数调优

### Tests

- 新增 `tests/test_ml_run_logger.py` - 训练运行日志模块完整测试套件（9个测试用例）
  - 测试CSV创建和首次写入
  - 测试追加记录功能
  - 测试自定义路径
  - 测试列扩展兼容性
  - 测试回归和分类任务记录
  - 测试完整工作流

## [0.8.2] - 2026-02-13

### Added

- **纸面交易与回测适配新模型/新特征（moneyflow + daily_basic）**
  - `DataLoader.load_clean_moneyflow()` - 新增资金流向数据加载方法，支持日期范围分区加载
  - `FeatureBuilder` 现在强制依赖 moneyflow 数据，缺失时会明确报错并提示补齐步骤
  - `MLSignal` 适配 classification 模型：自动使用 `predict_proba` 获取正类概率作为分数
  - `ensure_features_for_date()` 增强错误提示：moneyflow 缺失时提供详细的补数据命令

- **逐日评估诊断增强（排查 TopK/RankIC 不一致风险）**
  - `eval_utils.py` 新增 `compute_diagnostic_statistics()` 函数：
    - 全市场收益逐日均值/标准差统计
    - TopK 相对全市场提升计算（TopK - UniverseMean）
    - 每日样本数分布（min/median/max）
    - TopK 收益分位数（25%/50%/75%）
  - `eval_utils.py` 新增 `print_diagnostic_report()` 函数：格式化输出诊断报告
  - `train_ml_model.py` 集成诊断输出到验证集逐日评估流程

### Fixed

- **消除 Pandas FutureWarning（groupby.apply）**
  - `feature_utils.py` - `cross_sectional_zscore()` 改用矢量化 `transform` 方法，避免 `groupby.apply` 触发 FutureWarning
  - `train_ml_model.py` - `generate_classification_labels()` 改用矢量化方式计算百分比阈值，避免 `groupby.apply`

- **特征列一致性检查增强**
  - `ensure_features_for_date()` 增加 moneyflow 数据日志输出，记录加载的条数
  - moneyflow 缺失时的报错信息更友好，包含推荐的补数据命令

### Documentation

- 新增 `docs/PR/fix_paper_trade_moneyflow_v0.8.2.md` - 本 PR 详细说明
  - 说明纸面交易/回测为何缺特征列、如何修复、如何补齐数据
  - 说明 moneyflow 强制依赖的行为与补数据命令
  - 说明逐日评估新增诊断项的意义
  - 说明 FutureWarning 修复点

## [0.8.1] - 2026-02-13

### Fixed

- **修复 cs_zscore 的"重复 winsorize"问题**
  - `feature_utils.py` - 修复 `cross_sectional_zscore` 函数在按组标准化时重复 winsorize 的 bug
  - `train_ml_model.py` - 当 `label_transform=cs_zscore` 时，训练阶段不再对标签进行 winsorize（避免重复处理）
  - 新增 `skip_label_winsorize` 参数控制训练阶段是否跳过标签 winsorize
  - 目标行为：`label_transform=cs_zscore` 时仅在 cs_zscore 步骤做 winsorize；`label_transform=raw` 时仍保留训练阶段 winsorize 并在日志中说明

- **修复 Pandas FutureWarning（groupby.apply）**
  - `feature_utils.py` - 优化 `cross_sectional_zscore` 的 groupby 逻辑，避免 FutureWarning
  - `train_ml_model.py` - 改用 `rank(method='first')` 矢量化方式生成 pos_topk 标签，替代 groupby.apply
  - pos_topk 标签生成规则明确：正类数量严格等于 topk（使用 rank(method='first') 打散并列）

### Added

- **Classification 训练增强**
  - 新增 `--scale-pos-weight` CLI 参数，支持用户指定或自动计算（neg/pos）正类权重
  - 自动计算时在日志打印详细信息（负类数、正类数、计算值）
  - 新增 `src/lazybull/ml/eval_utils.py` 模块：提供可复用的逐日评估函数
    - `compute_daily_rankic()` - 计算单日 RankIC（Spearman）
    - `compute_daily_topk_returns()` - 计算单日 TopK 平均收益
    - `evaluate_predictions_by_date()` - 对多日预测进行逐日评估
    - `summarize_daily_metrics()` - 汇总逐日指标（均值、标准差、IR）
  - 分类任务训练后增加**验证集逐日评估**（贴近交易场景）
    - 逐日 RankIC（Spearman）：按每个 `trade_date` 计算预测概率与真实收益的秩相关，输出均值/标准差/IR
    - 逐日 TopK 收益评估：按每个 `trade_date` 以预测概率排序，计算 TopK（K=30/100/300）对应原始真实收益的均值，输出跨日均值/标准差
  - 统一 RankIC 计算口径：训练脚本中的 RankIC 改为"逐日计算后取均值"（与回测 eval panel 一致）

- **回测与纸面交易脚本适配新模型**
  - `model_registry.py` - 新增 `strict_version_check` 参数（默认 True），严格检查模型元数据
    - 检查必需字段：`feature_columns`、`train_params`、`model_type`
    - 缺少字段时明确报错并提示重新训练
  - `model_registry.py` - 新增 `check_feature_consistency()` 方法，检查推理数据特征列一致性
    - 验证推理数据是否包含模型训练时使用的所有特征列
    - 缺失特征时抛出详细错误（列出前 20 个缺失列）
  - `ml_signal.py` - 集成旧模型拒绝和特征列一致性检查
    - `_load_model` 方法调用 `strict_version_check=True` 拒绝旧模型
    - `generate` 和 `generate_ranked` 方法在预测前调用特征列一致性检查
  - **不兼容声明**：本版本明确不兼容旧模型（v1~v5），需重新训练

### Documentation

- 新增 `docs/PR/fix_cs_zscore_classification_enhancements_v0.8.1.md` - 本 PR 详细说明
  - 修复点说明
  - Classification 增强功能说明
  - 不兼容旧模型的决定与迁移方式
- 新增 `docs/guide/classification_evaluation_guide.md` - 分类模型评估指标指南
  - 说明应重点关注的指标（逐日 RankIC、TopK 收益）
  - 不要过度解读 Accuracy/Recall
  - 与回测结果对比的最佳实践

### Version

- 版本号从 0.8.0 升级到 0.8.1

---

## [0.8.0] - 2026-02-12

### Added

- **新增资金流数据源（moneyflow）**：提升模型在"价值红利"方向的选股能力
  - Raw/Ensure 层：新增 `TushareClient.get_moneyflow()` 方法，支持从 TuShare 获取个股资金流向数据
  - 在 `ensure_raw_data_for_date()` 中新增 moneyflow 下载逻辑，设为强制依赖（缺失时报错提示）
  - 在 `download_raw.py` 脚本中集成 moneyflow 下载
  - Clean 层：新增 `DataCleaner.clean_moneyflow()` 清洗方法
  - 在 `build_clean_features.py` 脚本中集成 moneyflow 清洗流程
  - 更新 `docs/data_contract.md` 补充 moneyflow 数据契约（主键、字段说明）
  
- **新增价值红利和资金流特征**：丰富因子库，支持价值投资和资金流分析
  - 新增 `feature_utils.py` 工具模块：提供 winsorize、log1p、zscore、cross_sectional_zscore 等通用特征处理函数
  - FeatureBuilder 新增 `_add_value_dividend_features()` 方法：
    - 基础因子：pb, pe_ttm, ps_ttm, dv_ttm, total_mv, circ_mv, turnover_rate, volume_ratio
    - 派生因子：ep_ttm (1/pe_ttm)、bp (1/pb)、log_total_mv、log_circ_mv
    - 亏损标记：is_loss（pe_ttm 为负或 NaN）
    - 处理 pe_ttm/pb 缺失和为0的情况
  - FeatureBuilder 新增 `_add_moneyflow_features()` 方法：
    - 当日净流入：net_mf_amount
    - 大单/特大单净流入：lg_net_amount、elg_net_amount
    - Rolling 特征（窗口 5/20）：net_mf_amount_sum/mean、lg_net_amount_sum、elg_net_amount_sum
    - 对重尾列自动应用 winsorize 处理
  - 更新 `build_features.py` 和 `build_clean_features.py` 加载并传递 daily_basic 和 moneyflow 数据

- **训练标签变换：cs_zscore（截面标准化）**：更稳定的回归标签，减少极端值影响
  - 新增 `transform_labels_cs_zscore()` 函数：对每个 trade_date 的标签进行截面 winsorize + zscore 变换
  - 变换后每个交易日标签均值≈0，标准差≈1
  - 新增 CLI 参数：`--label-transform {raw,cs_zscore}`（默认 raw）
  - 新增 CLI 参数：`--winsorize-p FLOAT`（默认 0.01，截断上下1%极端值）
  - 在模型元数据（model_registry.json）中记录 label_transform 和 winsorize_p

- **新增训练任务：classification（Top 分位分类）**：更贴近 TopN 选股的实际交易场景
  - 新增 `generate_classification_labels()` 函数：按每个交易日截面将标签转为 0/1 二分类标签
  - 支持两种模式（二选一，pos_topk 优先级更高）：
    - 百分比模式：`--pos-quantile FLOAT`（例如 0.2 表示 Top20% 为正类）
    - 数量模式：`--pos-topk INT`（例如 300 表示每日收益最高的 300 只为正类）
  - 新增 CLI 参数：`--task {regression,classification}`（默认 regression）
  - 新增 CLI 参数：`--pos-quantile FLOAT` 和 `--pos-topk INT`
  - 支持 XGBoost 分类器训练，目标函数自动切换为 `binary:logistic`
  - 分类任务评估指标：Accuracy、AUC、Precision、Recall
  - 在模型元数据中记录 task、pos_quantile、pos_topk
  - 模型类型标记为 `xgboost_classification` 以区分回归模型

### Changed

- **train_xgboost_model 函数增强**：统一支持回归和分类任务
  - 新增 `task` 参数，根据任务类型选择 XGBRegressor 或 XGBClassifier
  - 回归任务：保留 winsorize 处理和 IC/RankIC 评估
  - 分类任务：跳过 winsorize，使用 AUC/Precision/Recall 评估
  - 早停机制对两种任务均生效

### Documentation

- 更新 `docs/data_contract.md`：补充 moneyflow（资金流向）数据源的字段说明
- 新增 feature_utils.py 模块文档字符串：详细说明各工具函数的用法和示例

### Version

- 版本号从 0.7.0 升级到 0.8.0

---

## [0.7.0] - 2026-02-12

### Fixed
- **修复 `weight_method=score` 未生效问题**：修复了在回测引擎中 `score` 权重方法被等权覆盖的bug
  - 问题原因：`BacktestEngine._generate_signal()` 在信号生成阶段强制重新归一化权重，导致 MLSignal 已计算的按分数加权结果被覆盖
  - 修复方案：正确处理 `weight_method` 属性，当使用 `score` 时按预测分数归一化权重，而不是强制等权
  - 新增日志：明确显示当前使用的权重方法和前几只股票的权重示例，便于验证权重方法是否生效

### Added
- **权重后处理功能（限权/归一化）**：新增可复用的权重约束管理模块
  - 新增 `src/lazybull/portfolio/weight_processor.py` 模块
  - 实现 `cap_and_normalize_weights()` 函数：对权重进行限制并重新归一化
    - 支持设置单个股票最大权重 `max_weight_per_stock`（0-1之间）
    - 迭代式限权确保最终所有权重都不超过上限
    - 自动处理边界情况：空权重、全0、NaN、负数（过滤 <= 0 的权重）
  - BacktestEngine 新增 `max_weight_per_stock` 参数
  - CLI 新增 `--max-weight-per-stock` 参数（示例：`0.2` 表示单票最大 20%）
  - 单元测试：26 个测试全部通过，覆盖各种边界情况

- **行业持仓数量约束**：新增基于行业的持仓数量约束功能
  - 新增 `src/lazybull/portfolio/industry_constraint.py` 模块
  - 实现 `load_industry_mapping()` 函数：从 `stock_basic` 数据加载行业映射
    - 自动将行业缺失的股票归为"未知行业"
  - 实现 `apply_industry_constraint()` 函数：应用行业数量约束
    - 按分数排序选股，跳过已达到行业上限的股票并顺延
    - "未知行业"同样受约束限制
  - BacktestEngine 新增 `max_per_industry` 和 `stock_basic` 参数
  - CLI 新增 `--max-per-industry` 参数（示例：`3` 表示每个行业最多 3 只）
  - 单元测试：14 个测试全部通过，覆盖各种场景

### Documentation
- 新增 `docs/PR/portfolio_construction_enhancements.md`：详细说明三项改进的背景、实现和使用方法
- 新增测试文件：
  - `tests/test_weight_processor.py`：权重后处理模块测试（12个测试）
  - `tests/test_industry_constraint.py`：行业约束模块测试（14个测试）
- 扩展 `tests/test_ml_signal.py`：验证 score 权重方法产生非等权结果

### Technical Details
- 权重限权采用迭代算法：限权 → 归一化 → 检查收敛 → 重复（最多100次）
  - 确保最终所有权重都不超过设定上限
  - 处理多只股票同时被限权的情况
- 行业约束在信号生成阶段应用，在选择候选股票之前进行过滤
- 权重限权在权重归一化之后应用，确保最终权重满足约束

## [0.6.0] - 2026-02-12

### Added
- **统一评估面板（CSV输出）**：在回测运行时按日评估 MLSignal 的截面排序质量
  - 新增 `--export-eval` 参数：是否导出评估面板 CSV（默认开启）
  - 新增 `--eval-groups` 参数：分组数量（默认 10）
  - 新增 `--eval-topk` 参数：TopK 指标的 K（默认使用 --top-n）
  - 输出三个 CSV 文件：
    - `{output_name}_eval_daily.csv`：日度评估指标（RankIC、TopK收益、多空收益等）
    - `{output_name}_eval_groups.csv`：分组收益明细（每日每组的平均真实收益）
    - `{output_name}_eval_summary.csv`：汇总指标（参数配置和聚合统计）
  - 评估口径：
    - 真实收益标签直接使用 features 文件中的 label 列（y_ret_5/y_ret_10/y_ret_20）
    - 分组方式：按预测分数排序后等数量分组（默认 10 组）
    - RankIC 使用 Spearman 相关系数
  - CSV 统一使用 utf-8-sig 编码（Excel 兼容）

### Documentation
- 新增 `docs/PR/unified_eval_panel_csv_output.md`：详细说明评估面板功能的背景、实现和使用方法
- 新增 `docs/guide/ml_eval_panel_guide.md`：评估面板使用指南

## [0.5.0] - 2026-02-11

### Added
- **多 horizon 标签支持**：特征构建同时生成 `y_ret_5`, `y_ret_10`, `y_ret_20` 三个标签
  - 标签定义：未来 N 个交易日的后复权收益率，公式：`(close_adj(t+N) / close_adj(t)) - 1`
  - `FeatureBuilder` 新增 `horizons` 参数（默认 `[5, 10, 20]`），同时生成多个预测窗口的标签
  - `scripts/build_features.py` 新增 `--horizons` CLI 参数，支持自定义预测窗口列表

- **训练脚本支持选择标签**：`scripts/train_ml_model.py` 支持选择不同 horizon 的标签进行训练
  - 新增 `--label` CLI 参数（可选 `y_ret_5|y_ret_10|y_ret_20`，默认 `y_ret_5`）
  - 训练元数据自动记录所用标签到 `model_registry.json` 的 `label_column` 字段

- **回测脚本支持标签选择与自动调仓频率**：`scripts/run_ml_backtest.py` 增强标签和调仓频率管理
  - 新增 `--label` CLI 参数（可选 `y_ret_5|y_ret_10|y_ret_20`）
  - 当未显式指定 `--rebalance-freq` 时，根据标签自动设置默认值：
    - `y_ret_5` → 调仓频率 5 个交易日
    - `y_ret_10` → 调仓频率 10 个交易日
    - `y_ret_20` → 调仓频率 20 个交易日
  - 若同时指定 `--model-version` 和 `--label`，自动校验模型元数据中的标签一致性，不一致时给出清晰的中文报错提示

### Changed
- **`FeatureBuilder` 向后兼容**：保留 `horizon` 参数（已废弃），新参数 `horizons` 优先级更高
- **过滤逻辑优化**：`_apply_filters` 方法改为要求至少一个标签非空（而非所有标签都非空），更加灵活

### Documentation
- 新增 `docs/PR/multi_horizon_labels_and_selectable_training.md`：详细说明本次功能的背景、实现方案和使用方法
- 新增 `docs/guide/ml_label_horizon_guide.md`：完整的使用指南，包含标签定义、特征构建、训练和回测的详细说明

### Version
- 版本号从 0.4.2 升级到 0.5.0

## [0.4.2] - 2026-02-10

### Added
- **补位买入股数估算口径统一**：提示信息与实际执行逻辑完全一致
  - **问题背景**：生成 pending_buys（补位计划）时，提示的"预计购买数量"与实际执行时计算逻辑不一致，导致用户困惑
    - 提示逻辑：简单使用 `available_cash / len(targets)` 平均分配
    - 执行逻辑：考虑现金保留比例、成本预估、可用现金上限约束
  - **新增方法**：`PaperTradingRunner._estimate_pending_buy_shares()`
    - 封装统一的补位买入股数估算逻辑
    - 参数：`ts_code`, `price`, `target_weight`, `total_pending_count`, `pendding_capital_retention_ratio`
    - 计算逻辑与 `_execute_pending_buys()` 完全一致：
      1. `total_cash = account.cash * (1 - retention_ratio)` - 扣除保留比例
      2. `available_cash = total_cash / pending_count` - 平均分配
      3. `target_value = total_cash * target_weight` - 按权重计算
      4. 预估成本并检查是否超出可用现金上限
      5. `buy_shares = floor(target_value / price / 100) * 100` - 100股取整
  - **测试覆盖**：新增 `tests/test_pending_buy_estimation.py`，8个测试用例全部通过
    - 正常情况、现金受限、不足一手、异常价格、多目标分配、高保留比例、取整验证

### Changed
- **重构 `_execute_pending_buys()` 方法**：使用统一的 `_estimate_pending_buy_shares()` 计算股数
  - 简化原有逻辑约40行代码
  - 消除了重复的计算代码
  - 确保执行逻辑与估算逻辑完全一致

- **重构 `_print_replacement_targets()` 方法**：使用统一的估算逻辑并增加说明
  - 表头改为"估算股数"而非"建议股数"
  - 新增提示信息：
    - "注意：以下股数为估算值，基于当前价格与现金（保留比例 X%）"
    - "实际执行时会受到执行日价格变化、补位队列长度变化等因素影响，但计算规则一致"
  - 不足一手时显示为 "0 (不足一手)" 而非简单的 "0"

### Documentation
- 新增 `docs/PR/pending_buy_estimation_alignment.md` 详细说明本次改进的背景、方案和影响
- 新增 `docs/guide/pending_buy_estimation_guide.md` 说明补位买入股数的估算逻辑和影响因素

### Version
- 版本号从 0.4.1 升级到 0.4.2

## [0.3.15] - 2026-02-09

### Added
- **新增停牌判断统一工具类 SuspendCalendar**：基于 raw/suspend 数据提供统一的停牌判断接口
  - **问题背景**：停牌信息不在 daily/clean daily 中，停牌股票在 daily 中可能缺行或价格缺失/为0，导致：
    - 纸面交易止损检查可能误触发（0价造成大回撤）
    - 调仓/执行卖出可能因 sell_prices 缺失而直接跳过，既不卖出也不进入延迟卖出队列
    - 回测同样可能出现止损误触发或卖出静默失败
  - **新增模块**：`src/lazybull/common/suspend_calendar.py`
    - 实现 `SuspendCalendar` 工具类，提供 `is_suspended()`, `get_status_reason()`, `batch_is_suspended()` 方法
    - 判定规则基于 raw/suspend 数据：suspend_type='S' => 停牌，suspend_type='R' => 复牌，无记录 => 非停牌
    - 严格模式：suspend 数据文件缺失时抛出 FileNotFoundError 异常
    - 按 trade_date 缓存机制，提高查询效率
  - **测试覆盖**：新增 `tests/test_suspend_calendar.py`，8个测试用例全部通过

### Changed
- **纸面交易集成 SuspendCalendar**：
  - 修改 `scripts/paper_trade.py` 的 `_check_stop_loss()` 方法：
    - 使用 SuspendCalendar 判断停牌（而非依赖 daily 中的 is_suspended 列）
    - 停牌股票跳过止损检查，输出中文日志"停牌，跳过止损检查"
    - 无行情数据股票跳过止损检查，输出中文日志"无行情数据，跳过止损检查"
  - 修改 `src/lazybull/paper/broker.py` 的卖出流程：
    - `generate_orders()` 和 `execute_instructions()` 方法：停牌/无价格时创建 PendingSell 并持久化
    - reason 文案按优先级：停牌优先，否则无价格数据
    - 更新 `_check_can_sell()` 方法支持通过 trade_date 参数使用 SuspendCalendar
  - 修改 `src/lazybull/paper/runner.py`：传递 data_storage 给 broker，确保使用相同的数据根路径

- **回测引擎集成 SuspendCalendar**：
  - 修改 `src/lazybull/backtest/engine.py` 的止损检查：
    - `_check_stop_loss()` 方法使用 SuspendCalendar 判断停牌（而非依赖 price_data 中的 is_suspended 列）
    - 停牌时跳过止损检查，输出中文日志"股票 {stock} 停牌，跳过止损检查"
  - 修改回测引擎的卖出流程：
    - `_sell_stock_with_status_check()` 方法：停牌/无价格时进入延迟卖出队列
    - reason 文案按优先级：停牌优先，否则无价格数据或跌停
  - 新增 `data_storage` 参数支持传入 Storage 实例

### Documentation
- 新增 `docs/PR/suspend_detection_unified.md` 详细说明本次功能的问题、方案、影响范围和验证步骤

## [0.3.11] - 2026-02-09

### Fixed
- **修复 paper_trade positions 命令股票名称显示问题**：持仓明细现在能正确显示股票名称
  - **问题描述**：运行 `python scripts/paper_trade.py positions --trade-date YYYYMMDD` 时，所有股票名称都显示为 `(na)` 而非实际名称
  - **问题根因**：`print_positions()` 函数试图从 `daily_data` 的 `name` 列构建 `stock_names` 字典，但 clean daily 数据不包含 `name` 列
  - **解决方案**：从 `stock_basic` 表读取股票名称（包含 `ts_code` 和 `name` 列）
    - 新增 `build_stock_names_dict()` 辅助函数
    - 优先使用 `DataLoader.load_clean_stock_basic()`
    - 回退使用 `DataLoader.load_stock_basic()`
    - 若无法加载 stock_basic，输出清晰的中文提示日志，建议运行 `python scripts/update_basic_data.py`
  - **核心修改**：
    - `scripts/paper_trade.py`：新增 `build_stock_names_dict()` 函数，修改 `print_positions()` 函数
  - **验收测试**：新增 `tests/test_stock_names_display.py`，5个测试用例全部通过
    - 测试当提供股票名称字典时，持仓明细能正确显示股票名称
    - 测试当不提供股票名称字典时，持仓明细回退显示 `(na)`
    - 测试从 clean/raw stock_basic 加载

### Documentation
- 新增 `docs/PR/fix_stock_names_display.md` 详细说明本次修复的问题、方案和验证方法

## [0.3.9] - 2026-02-09

### Fixed
- **修复纸面交易日志/原因文案不清晰的问题**：卖出订单的 reason 文案现在能准确反映实际交易行为
  - **问题描述**：当目标权重为0时，所有卖出订单统一使用"退出持仓"，但实际可能只是减仓（部分卖出），容易误导用户
  - **解决方案**：根据实际卖出股数和持仓股数判断 reason 文案
    - 完全清仓（`sell_shares == pos.shares` 且 `target_weight == 0`）→ "退出持仓"
    - 部分清仓（`sell_shares < pos.shares` 且 `target_weight == 0`）→ "减仓(退出持仓未完全清仓)"
    - 普通减仓（`target_weight > 0`）→ "减仓"
  - **核心修改**：
    - `src/lazybull/paper/broker.py` 中的 `generate_orders()` 方法：重新组织卖出订单生成逻辑，在计算 sell_shares 后根据实际情况确定 reason
    - 同步更新 PendingSell 延迟卖出订单的 reason 逻辑
  - **验收测试**：新增 `tests/test_sell_order_reason.py`，7个测试用例全部通过

### Improved
- **增强执行日志统计**：在订单执行完成后增加详细的交易类型统计
  - 新增 `_calculate_execution_stats()` 方法，统计以下信息：
    - 买入：新建持仓笔数、加仓笔数
    - 卖出：清仓笔数、减仓笔数
  - 统计基于执行前的持仓快照，避免卖出/买入顺序影响判断
  - 日志格式示例：
    ```
    执行完成: 27 买，26 卖
      - 买入: 新建持仓 15 笔，加仓 12 笔
      - 卖出: 清仓 10 笔，减仓 16 笔
    ```
  - 这些统计以"成交 fill"为准，帮助用户更直观地了解交易结果

### Documentation
- 新增 `docs/PR/fix_sell_order_reason_clarity.md` 详细说明本次修复的动机与变更点
- 更新 CHANGELOG.md 记录版本变更

## [0.3.8] - 2026-02-09

### Fixed
- **修复补位机制导致的清仓问题**：补位目标不再覆盖全量组合目标，避免触发"退出持仓"卖出订单
  - **问题根因**：补位目标直接保存到 `pending/{next_date}.parquet`，T1执行时当作全量目标，导致现有持仓被清仓
  - **解决方案**：引入独立的 `pending_buys` 队列存储补位计划（增量买入），与 `pending_weights`（全量调仓）分离
  - **核心修改**：
    - `scripts/paper_trade.py`：将补位目标保存到 `pending_buys` 队列
    - `src/lazybull/paper/runner.py`：新增 `_execute_pending_buys()` 方法专门处理补位买入（仅买入，不触发卖出）
    - `run_t1()` 方法分别处理 pending_weights 和 pending_buys
    - 重构 `_execute_t1_if_pending()` 和新增 `_handle_failed_buys()` 辅助函数
  - **验收测试**：新增 `tests/test_replenishment_no_sell.py` 验证修复
    - 场景1：持有27只股票 + 3只补位计划 → 不生成卖出订单 ✓
    - 场景2：错误使用（3只补位作为全量目标） → 生成27个卖出订单（清仓）✓
    - 场景3：正确的补位流程 → 仅买入，不影响持仓 ✓

### Improved
- 补位机制更加健壮，与现有持仓管理解耦
- 补位执行不再影响全量调仓逻辑
- 更贴近真实交易场景：补位仅用于增量买入，不触发减仓/清仓

### Documentation
- 新增修复说明文档（详见本次提交的PR描述）
- 补位机制的生命周期和数据格式说明

## [0.3.6] - 2026-02-08

### Added
- **买入失败补位机制**：当 T1 买入因涨停/停牌/不可交易失败时，系统自动生成补位计划，在下一交易日继续买入
  - 新增 `PendingBuy` 数据模型，对称于现有的 `PendingSell`
  - 新增 `PaperStorage.save_pending_buys()` 和 `load_pending_buys()` 持久化方法
  - 新增 `PaperBroker.retry_pending_buys()` 重试补位订单
  - 新增 `PaperTradingRunner.generate_replacement_targets()` 生成补位目标
  - 在 `paper_trade.py run` 中新增步骤 3：处理延迟买入队列（补位计划）
- **补位重试机制**：最多重试 5 次，同日不重复推进 attempts 计数
- **一手可买约束**：补位目标必须满足至少能买入 100 股（1 手）的约束
- **补位输出格式化**：补位目标输出表格与 T0 输出格式保持一致
- **测试覆盖**：新增 `tests/test_buy_replacement.py` 测试文件，覆盖核心功能

### Changed
- **PaperBroker.generate_orders()**：增强买入失败检测，记录失败原因（涨停、停牌、无价格、现金不足、不足一手等）
- **T1 执行流程**：自动检测买入失败并生成补位计划，无需手工干预
- **手工操作指令汇总**：新增"延迟买入清单（补位计划）"部分

### Improved
- 提升资金使用效率，避免买入失败导致的资金长期闲置
- 增强纸面交易与回测的一致性，引入"候选顺延"机制
- 更贴近真实交易场景，自动处理买入受限情况

### Documentation
- 新增 `docs/PR/buy_replacement.md`：详细说明补位机制的设计与实现
- 说明纸面交易与回测在补位处理上的一致性与合理差异

## [0.3.5] - 2026-02-08

### Added
- 纸面交易T0等权策略的"一手可买约束 + 顺延补足"功能
  - 在等权模式下（`weight_method=="equal"`），对每只候选股票检查按资金分配是否能买入至少1手（100股）
  - 不足1手的股票将被跳过，并从排序候选中顺延选择下一只
  - 确保最终保存到pending的目标都是可有效购买的股票
  - 添加详细日志：原始候选数、跳过数、最终目标数、跳过示例

### Changed
- `PaperTradingRunner._generate_signals` 方法新增 `buy_price_type` 参数，用于确定一手判断价格
- 复用 `MLSignal.generate_ranked()` 方法获取完整排序候选列表，支持顺延补足

### Technical Details
- 新增 `PaperTradingRunner._generate_equal_weight_with_lot_constraint` 方法实现核心逻辑
- 等权策略下使用T1的买入价格类型（open/close）进行一手可买性判断
- Score加权策略暂不启用此约束，保持原有行为

## [0.3.4] - Previous Version
- 之前版本的功能
