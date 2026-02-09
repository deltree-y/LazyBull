# Changelog

All notable changes to this project will be documented in this file.

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
