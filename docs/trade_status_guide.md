# 涨跌停与停牌状态处理指南

## 概述

LazyBull 从 v0.3.0 开始支持对股票涨跌停与停牌状态的自动处理，包括：
- **选股阶段过滤**：自动过滤停牌、涨跌停股票
- **交易延迟机制**：遇到涨跌停/停牌时延迟订单而非直接失败

这些功能可以通过配置参数灵活控制，确保策略在面对交易限制时更加稳健。

## 功能说明

### 1. 选股阶段过滤

在生成选股信号时，系统会自动过滤以下股票：
- **停牌股票**：成交量为0或有停牌标记的股票
- **涨停股票**：涨幅≥9.9%（非ST）或≥4.9%（ST），或收盘价达到涨停价
- **跌停股票**：跌幅≤-9.9%（非ST）或≤-4.9%（ST），或收盘价达到跌停价

**过滤逻辑**：
- 买入时：过滤停牌和涨停股票（难以买入）
- 卖出时：过滤停牌和跌停股票（难以卖出）

### 2. 延迟订单机制

当股票因涨跌停或停牌无法立即交易时，订单会被加入延迟队列：
- **自动重试**：在每个交易日检查延迟订单，条件解除后自动执行
- **最大重试次数**：默认5次（可配置）
- **最大延迟天数**：默认10个交易日（可配置）
- **超时处理**：超过限制后自动放弃并记录日志

## 使用方法

### 基本用法

```python
from src.lazybull.backtest import BacktestEngine
from src.lazybull.universe import BasicUniverse
from src.lazybull.signals.base import EqualWeightSignal

# 1. 创建支持过滤的股票池
universe = BasicUniverse(
    stock_basic=stock_basic_df,
    exclude_st=True,           # 排除ST股票
    filter_suspended=True,     # 过滤停牌股票（默认True）
    filter_limit_stocks=True   # 过滤涨跌停股票（默认True）
)

# 2. 创建支持延迟订单的回测引擎
engine = BacktestEngine(
    universe=universe,
    signal=signal,
    initial_capital=1000000,
    enable_pending_order=True,  # 启用延迟订单（默认True）
    max_retry_count=5,          # 最大重试次数（默认5）
    max_retry_days=10           # 最大延迟天数（默认10）
)

# 3. 运行回测
nav_curve = engine.run(
    start_date=start_date,
    end_date=end_date,
    trading_dates=trading_dates,
    price_data=price_data  # 需包含交易状态字段
)

# 4. 查看延迟订单统计
if engine.pending_order_manager:
    stats = engine.pending_order_manager.get_statistics()
    print(f"累计添加: {stats['total_added']}")
    print(f"成功执行: {stats['total_succeeded']}")
    print(f"过期放弃: {stats['total_expired']}")
```

### 配置选项

#### 股票池过滤配置

```python
universe = BasicUniverse(
    stock_basic=stock_basic_df,
    filter_suspended=True,      # 是否过滤停牌股票
    filter_limit_stocks=True    # 是否过滤涨跌停股票
)
```

#### 延迟订单配置

```python
engine = BacktestEngine(
    universe=universe,
    signal=signal,
    enable_pending_order=True,  # 是否启用延迟订单
    max_retry_count=5,          # 最大重试次数
    max_retry_days=10           # 最大延迟天数
)
```

### 关闭功能

如果不需要这些功能，可以关闭：

```python
# 关闭选股过滤
universe = BasicUniverse(
    stock_basic=stock_basic_df,
    filter_suspended=False,
    filter_limit_stocks=False
)

# 关闭延迟订单
engine = BacktestEngine(
    universe=universe,
    signal=signal,
    enable_pending_order=False
)
```

## 数据要求

要使用交易状态检查功能，价格数据需要包含以下字段：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `ts_code` | str | 股票代码 |
| `trade_date` | str | 交易日期（YYYYMMDD格式） |
| `close` | float | 收盘价 |
| `filter_is_suspended` | int | 停牌标记（0=否，1=是） |
| `is_limit_up` | int | 涨停标记（0=否，1=是） |
| `is_limit_down` | int | 跌停标记（0=否，1=是） |
| `vol` | float | 成交量（可选，用于辅助判断停牌） |
| `pct_chg` | float | 涨跌幅（可选，用于辅助判断涨跌停） |

**注意**：这些字段已在 `DataCleaner.add_tradable_universe_flag()` 中自动添加，使用 `clean` 数据层即可。

## 日志输出

### 选股过滤日志

```
INFO  | 选股过滤 2023-01-10: 原始 100 只，过滤停牌 5 只，过滤涨停 3 只，过滤跌停 2 只，最终 90 只
```

### 延迟订单日志

```
INFO  | 买入延迟: 2023-01-10 000001.SZ, 原因: 涨停, 目标市值: 100000.00
INFO  | 卖出延迟: 2023-01-11 000002.SZ, 原因: 跌停
INFO  | 延迟订单执行成功: 000001.SZ buy (重试次数: 2, 延迟天数: 1)
INFO  | 延迟订单超过最大重试次数，放弃: 000003.SZ sell (重试次数: 6, 最大重试: 5)
```

### 回测结束统计

```
INFO  | 延迟订单统计: 累计添加 10, 成功执行 7, 过期放弃 3, 剩余待处理 0
```

## 工具函数

可以独立使用交易状态检查工具：

```python
from src.lazybull.common.trade_status import (
    is_suspended,
    is_limit_up,
    is_limit_down,
    is_tradeable,
    get_trade_status_info
)

# 检查是否停牌
if is_suspended('000001.SZ', '20230110', quote_data):
    print("股票停牌")

# 检查是否可交易
tradeable, reason = is_tradeable('000001.SZ', '20230110', quote_data, action='buy')
if not tradeable:
    print(f"不可交易，原因: {reason}")

# 获取完整状态信息
info = get_trade_status_info('000001.SZ', '20230110', quote_data)
print(f"停牌: {info['is_suspended']}")
print(f"涨停: {info['is_limit_up']}")
print(f"跌停: {info['is_limit_down']}")
print(f"可买入: {info['can_buy']}")
print(f"可卖出: {info['can_sell']}")
```

## 延迟订单管理器

可以独立使用延迟订单管理器：

```python
from src.lazybull.execution.pending_order import PendingOrderManager

# 创建管理器
manager = PendingOrderManager(
    max_retry_count=5,
    max_retry_days=10
)

# 添加延迟订单
manager.add_order(
    stock='000001.SZ',
    action='buy',
    current_date=pd.Timestamp('2023-01-10'),
    signal_date=pd.Timestamp('2023-01-09'),
    target_value=100000.0,
    reason='涨停'
)

# 获取可重试订单
orders = manager.get_orders_to_retry(pd.Timestamp('2023-01-11'))

# 标记成功
manager.mark_success('000001.SZ', 'buy')

# 查看统计
stats = manager.get_statistics()
print(stats)
```

## 最佳实践

1. **启用所有功能（推荐）**：
   ```python
   # 选股过滤 + 延迟订单
   universe = BasicUniverse(..., filter_suspended=True, filter_limit_stocks=True)
   engine = BacktestEngine(..., enable_pending_order=True)
   ```

2. **合理设置延迟参数**：
   - 短期策略：`max_retry_count=3, max_retry_days=5`
   - 中长期策略：`max_retry_count=5, max_retry_days=10`（默认）
   - 高频策略：建议关闭延迟功能或设置更短的限制

3. **关注日志输出**：
   - 定期查看延迟订单统计
   - 如果过期放弃的订单较多，可能需要调整策略或参数

4. **测试验证**：
   - 使用集成测试验证功能正确性
   - 对比开关功能前后的回测结果

## 性能影响

- **选股过滤**：轻微影响（每次选股时额外的DataFrame操作）
- **延迟订单**：轻微影响（每个tick检查延迟队列）
- **整体影响**：可忽略不计（<1%回测时间增加）

## 向后兼容性

- 所有新功能默认启用，但不破坏现有API
- 未提供 `quote_data` 参数时自动跳过过滤
- 未启用延迟订单时行为与之前完全一致
- 所有现有测试保持通过

## 常见问题

### Q1: 为什么选股时过滤了一些股票但日志显示"未找到行情数据"？

A: 这是正常行为。当某只股票在当日没有行情数据时，系统会假定它未停牌/未涨跌停，并输出debug级别的日志。这通常发生在：
- 股票在该日期尚未上市
- 数据缺失
- 测试数据不完整

### Q2: 延迟订单一直无法执行怎么办？

A: 检查以下几点：
1. 是否超过了最大重试次数或延迟天数
2. 股票是否持续停牌或连续涨/跌停
3. 查看日志中的"过期放弃"数量
4. 考虑增加 `max_retry_count` 或 `max_retry_days`

### Q3: 可以只过滤停牌不过滤涨跌停吗？

A: 可以，分别配置即可：
```python
universe = BasicUniverse(
    ...,
    filter_suspended=True,      # 过滤停牌
    filter_limit_stocks=False   # 不过滤涨跌停
)
```

### Q4: 延迟订单会影响资金使用率吗？

A: 会有轻微影响。延迟的买入订单会占用计划资金，延迟的卖出订单会延迟资金回笼。但这更接近真实交易情况。

## 更新历史

- **v0.3.0** (2026-01-18): 首次发布涨跌停与停牌处理功能

## 相关文档

- [回测假设](backtest_assumptions.md)
- [数据契约](data_contract.md)
- [API 文档](../README.md)
