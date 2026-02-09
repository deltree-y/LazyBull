# PR: 停牌判断统一：基于 raw/suspend 的 SuspendCalendar 工具类，并对齐 paper/backtest 行为

## 问题描述

停牌信息不在 daily/clean daily 中，停牌股票在 daily 中可能缺行或价格缺失/为0，导致：

1. **纸面交易止损检查可能误触发**：当停牌股票的价格为0或缺失时，计算出的回撤幅度会异常巨大，可能导致止损误触发
2. **调仓/执行卖出可能因 sell_prices 缺失而直接跳过**：当停牌股票需要卖出时，如果 sell_prices 字典中缺少该股票，原有代码会直接 `continue` 跳过，既不卖出也不进入延迟卖出队列（PendingSell），导致调仓失败但无追踪
3. **回测同样可能出现止损误触发或卖出静默失败**：与纸面交易相同的问题

### 根本原因

项目中停牌判断逻辑分散且不统一：
- 部分地方依赖 daily 数据中的 `is_suspended` 列
- 但停牌信息的权威来源是 `raw/suspend` 数据
- daily 数据中的 `is_suspended` 列可能不准确或缺失
- 停牌股票的 daily 记录可能完全缺失

## 解决方案

### 核心设计

新增统一的停牌判断工具类 `SuspendCalendar`，基于 `raw/suspend` 数据提供停牌判断接口：

1. **数据来源**：使用 Storage 读取 `raw/suspend` 按日期分区数据
2. **判定规则**：
   - 当日存在记录且 `suspend_type == 'S'` => 停牌 True
   - 当日存在记录且 `suspend_type == 'R'` => 非停牌 False
   - 当日无记录 => 非停牌 False
3. **严格模式**：suspend 数据文件缺失时抛出 FileNotFoundError 异常（不静默降级）
4. **缓存机制**：按 trade_date 缓存已加载的数据，提高查询效率

### 实现要点

#### A. SuspendCalendar 工具类

**位置**：`src/lazybull/common/suspend_calendar.py`

**主要方法**：
- `is_suspended(ts_code, trade_date)`: 判断单个股票是否停牌
- `get_status_reason(ts_code, trade_date)`: 获取停牌状态描述（用于日志）
- `batch_is_suspended(ts_codes, trade_date)`: 批量判断多个股票是否停牌

**关键特性**：
- 复用现有 Storage 的 `load_raw_by_date()` 方法读取 suspend 数据
- 按日期缓存数据，避免重复读取
- 严格模式：数据缺失时抛异常而非静默降级

#### B. 纸面交易集成

**修改文件**：
- `scripts/paper_trade.py`
- `src/lazybull/paper/broker.py`
- `src/lazybull/paper/runner.py`

**主要改动**：

1. **止损检查**（`_check_stop_loss()`）：
   - 使用 SuspendCalendar 判断停牌（不再依赖 daily 中的 is_suspended 列）
   - 停牌股票跳过止损检查，输出中文日志："停牌，跳过止损检查"
   - 无行情数据股票跳过止损检查，输出中文日志："无行情数据，跳过止损检查"
   - suspend 数据缺失时，记录错误日志并跳过所有止损检查

2. **卖出流程**（`generate_orders()` 和 `execute_instructions()`）：
   - 当需要卖出但 `ts_code not in sell_prices` 时：
     - 调用 SuspendCalendar 判断停牌
     - 创建 PendingSell 并持久化（复用现有 `save_pending_sells`）
     - reason 文案优先级：停牌优先，否则无价格数据
   - 不再静默 `continue` 跳过

3. **依赖注入**：
   - runner 传递 data_storage 给 broker
   - broker 使用延迟创建的 SuspendCalendar 实例
   - 确保 broker 与 runner 使用相同的数据根路径

#### C. 回测引擎集成

**修改文件**：
- `src/lazybull/backtest/engine.py`

**主要改动**：

1. **止损检查**（`_check_stop_loss()`）：
   - 使用 SuspendCalendar 判断停牌（不再依赖 price_data 中的 is_suspended 列）
   - 停牌时跳过止损检查，输出中文日志
   - 检查价格有效性，无效价格跳过止损检查

2. **卖出流程**（`_sell_stock_with_status_check()`）：
   - 优先使用 SuspendCalendar 检查停牌状态
   - 停牌/无价格/跌停时进入延迟卖出队列（pending_order_manager）
   - reason 文案按优先级：停牌 > 无行情数据 > 跌停

3. **参数扩展**：
   - 新增 `data_storage` 参数支持传入 Storage 实例
   - 延迟创建 SuspendCalendar 实例

## 影响范围

### 代码层面

**新增文件**：
- `src/lazybull/common/suspend_calendar.py`（停牌日历工具类）
- `tests/test_suspend_calendar.py`（单元测试）
- `docs/PR/suspend_detection_unified.md`（本文档）

**修改文件**：
- `scripts/paper_trade.py`（纸面交易止损检查）
- `src/lazybull/paper/broker.py`（纸面交易卖出流程）
- `src/lazybull/paper/runner.py`（传递 data_storage）
- `src/lazybull/backtest/engine.py`（回测止损和卖出）
- `pyproject.toml`（版本号 0.3.14 -> 0.3.15）
- `CHANGELOG.md`（更新日志）

### 功能层面

**纸面交易**：
- 止损检查更准确：停牌股票不会误触发止损
- 卖出逻辑更健壮：停牌/无价格卖出会进入 PendingSell 并持久化，不再静默跳过
- 日志更清晰：明确区分停牌、无行情数据、止损触发等情况

**回测**：
- 止损检查更准确：停牌股票不会误触发止损
- 卖出逻辑更健壮：停牌/无价格卖出进入延迟队列并后续可重试
- 行为与纸面交易对齐

## 验证步骤

### 1. 单元测试

运行 SuspendCalendar 的单元测试：
```bash
pytest tests/test_suspend_calendar.py -v
```

预期结果：8个测试用例全部通过
- 测试 S/R/无记录三种情况
- 测试 suspend 文件缺失时抛异常
- 测试批量查询
- 测试缓存机制
- 测试日期格式（YYYYMMDD 和 YYYY-MM-DD）

### 2. 集成测试（纸面交易）

**前置条件**：
- 确保 `data/raw/suspend/` 目录下有测试日期的 suspend 数据
- 确保有持仓和配置

**测试场景1：停牌股票不触发止损**
```bash
# 持仓中有停牌股票
python scripts/paper_trade.py run --trade-date YYYYMMDD
```
预期：
- 日志中显示"股票 XXX 停牌，跳过止损检查"
- 停牌股票不会触发止损

**测试场景2：停牌股票卖出进入 PendingSell**
```bash
# 需要卖出的股票停牌或无价格数据
python scripts/paper_trade.py run --trade-date YYYYMMDD
```
预期：
- 日志中显示"股票 XXX 停牌，无法卖出，加入延迟卖出队列"
- pending_sells.json 中有该股票的记录，reason 包含"停牌"

### 3. 集成测试（回测）

**测试场景：停牌日止损不触发，卖出进入延迟队列**

运行回测脚本，观察日志：
```python
from src.lazybull.backtest.engine import BacktestEngine
from src.lazybull.data import Storage
# ... 初始化 engine 时传入 data_storage
engine = BacktestEngine(..., data_storage=Storage())
# ... 运行回测
```

预期：
- 停牌日日志显示"股票 XXX 停牌，跳过止损检查"
- 停牌卖出日志显示"卖出延迟: ... 原因: 停牌"

### 4. 边界测试

**测试suspend 数据缺失场景**：
- 删除某个交易日的 suspend 数据文件
- 运行纸面交易或回测
- 预期：抛出 FileNotFoundError 异常，日志显示"停牌数据文件缺失"

## 风险评估

### 低风险

1. **SuspendCalendar 是新增模块**：不影响现有逻辑
2. **有完整的单元测试覆盖**：8个测试用例验证各种场景
3. **降级机制**：suspend 数据加载失败时，纸面交易和回测都有降级处理

### 注意事项

1. **严格模式要求 suspend 数据完整**：
   - 如果某个交易日的 suspend 数据缺失，会抛出异常
   - 需要确保 `scripts/build_features.py` 或数据下载流程正确保存 suspend 数据

2. **性能影响很小**：
   - SuspendCalendar 有按日期缓存机制
   - 每个交易日的 suspend 数据只加载一次

3. **向后兼容性**：
   - 不影响现有的 is_suspended 列（如果存在）
   - 仅在新增的停牌判断逻辑中使用 SuspendCalendar

## 后续优化建议

1. **统一 is_tradeable 函数**：考虑在 `trade_status.py` 中添加基于 SuspendCalendar 的版本
2. **扩展批量接口**：纸面交易止损检查可以使用 `batch_is_suspended` 优化性能
3. **监控 suspend 数据完整性**：添加数据质量检查，确保 suspend 数据按日更新

## 总结

本 PR 通过引入统一的 SuspendCalendar 工具类，解决了停牌判断逻辑分散、不准确的问题。主要改进：

1. **准确性提升**：基于权威的 raw/suspend 数据，而非可能缺失的 daily 列
2. **健壮性提升**：停牌/无价格卖出不再静默跳过，而是进入延迟队列
3. **一致性提升**：纸面交易和回测使用相同的停牌判断逻辑
4. **可维护性提升**：停牌判断逻辑集中在 SuspendCalendar 类中

验收标准：
- ✅ 停牌日纸面交易止损不触发
- ✅ 停牌/无价卖出进入 PendingSell 并持久化
- ✅ 回测停牌日止损不触发
- ✅ 回测卖出停牌/无价进入延迟队列
- ✅ suspend 数据缺失时严格报错
- ✅ 测试通过、版本号递增、文档齐全、全中文日志/注释
