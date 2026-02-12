# Changelog

All notable changes to this project will be documented in this file.

## [0.6.2] - 2026-02-12

### Fixed
- **修复多 horizon 标签计算错误**：修复 `FeatureBuilder._calculate_forward_returns()` 中 `y_ret_10` / `y_ret_20` 计算错误的问题
  - **根因**：`_get_trading_dates()` 方法未对交易日列表去重，导致当 `trade_cal` 包含重复日期时（如多个交易所数据），`current_idx + horizon` 索引计算错误
  - **修复内容**：
    1. 在 `_get_trading_dates()` 中使用 `.unique()` 对交易日列表去重，确保每个交易日只出现一次
    2. 在 `build_features_for_day()` 中构建 `date_to_idx` 字典映射，避免重复使用 `list.index()` 带来的 O(n) 查找开销，优化为 O(1) 查找
    3. 新增 `tests/test_multi_horizon_fix.py` 测试文件，包含6个测试用例：
       - 重复交易日期处理测试
       - 真实场景模拟测试（600036.SH 案例）
       - 所有 horizon (5/10/20) 同时正确性测试
       - 日期格式一致性测试
       - 乱序交易日历处理测试
       - 日期索引映射性能测试
  - **影响**：使用 `build_features` / `build_clean_features` 脚本重新生成 `cs_train` 数据后，多 horizon 标签将正确反映未来 N 个交易日的收益率
  - **验证**：所有现有测试（15个 `test_features.py` + 12个 `test_multi_horizon_labels.py` + 6个新测试）均通过

### Documentation
- 新增 `docs/PR/fix_multi_horizon_label_calculation.md`：详细说明bug根因、修复方式、对已有数据的影响

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
