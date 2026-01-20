# 卖出时机配置指南

## 概述

从 v0.5.0 开始，LazyBull 支持配置卖出时机，允许用户选择在 T+n 日使用**开盘价卖出**或**收盘价卖出**。

## 功能特性

### 支持的卖出时机

| 卖出时机 | 参数值 | 说明 | 默认 |
|---------|--------|------|------|
| T+n 日收盘价卖出 | `'close'` | 在持有期结束当天的收盘价卖出 | ✅ 是 |
| T+n 日开盘价卖出 | `'open'` | 在持有期结束当天的开盘价卖出 | ❌ 否 |

### 价格降级策略

当使用 `sell_timing='open'` 但数据中缺少开盘价时，系统会自动降级：

1. **优先使用开盘价**：如果数据包含 `open` 或 `open_adj` 字段
2. **降级到收盘价**：如果开盘价缺失或为 NaN，自动使用收盘价
3. **日志警告**：降级时会在日志中输出警告信息

## 使用方法

### 基础用法

```python
from src.lazybull.backtest import BacktestEngine
from src.lazybull.signals import EqualWeightSignal
from src.lazybull.universe import BasicUniverse
from src.lazybull.common.cost import CostModel

# 创建回测引擎，默认使用收盘价卖出
engine = BacktestEngine(
    universe=universe,
    signal=signal,
    initial_capital=1000000,
    cost_model=CostModel(),
    rebalance_freq=5,
    holding_period=10
    # sell_timing='close'  # 默认值，可省略
)
```

### 配置为开盘价卖出

```python
# 使用开盘价卖出
engine = BacktestEngine(
    universe=universe,
    signal=signal,
    initial_capital=1000000,
    cost_model=CostModel(),
    rebalance_freq=5,
    holding_period=10,
    sell_timing='open'  # 在 T+n 日开盘价卖出
)
```

### 运行回测

```python
import pandas as pd

# 准备数据（需包含 open 和 close 价格）
# price_data 应包含以下字段：
# - ts_code: 股票代码
# - trade_date: 交易日期
# - close: 收盘价（必需）
# - open: 开盘价（使用 sell_timing='open' 时需要）
# - close_adj: 后复权收盘价（推荐）
# - open_adj: 后复权开盘价（推荐）

nav_curve = engine.run(
    start_date=pd.Timestamp('2023-01-01'),
    end_date=pd.Timestamp('2023-12-31'),
    trading_dates=trading_dates,
    price_data=price_data
)

# 获取交易记录
trades = engine.get_trades()

# 检查卖出时机
sell_trades = trades[trades['action'] == 'sell']
print(sell_trades[['date', 'stock', 'price', 'sell_timing']])
```

## 数据要求

### 必需字段

无论使用哪种卖出时机，以下字段都是必需的：

| 字段 | 类型 | 说明 |
|------|------|------|
| `ts_code` | str | 股票代码 |
| `trade_date` | str/datetime | 交易日期 |
| `close` | float | 收盘价（不复权） |

### 可选字段（推荐）

| 字段 | 类型 | 说明 | 用于 |
|------|------|------|------|
| `open` | float | 开盘价（不复权） | 开盘价卖出 |
| `close_adj` | float | 收盘价（后复权） | 绩效计算 |
| `open_adj` | float | 开盘价（后复权） | 绩效计算（开盘卖出） |

### 数据准备示例

```python
from src.lazybull.data import TushareClient, Storage, DataCleaner

client = TushareClient()
storage = Storage()
cleaner = DataCleaner()

# 下载数据（包含开盘价）
trade_date = "20230110"
daily_data = client.get_daily(trade_date=trade_date)
adj_factor = client.get_adj_factor(trade_date=trade_date)

# 清洗数据（会自动计算 open_adj、close_adj 等）
daily_clean = cleaner.clean_daily(daily_data, adj_factor)

# daily_clean 现在包含以下字段：
# - close, open, high, low (原始价格)
# - close_adj, open_adj, high_adj, low_adj (后复权价格)
# - tradable, is_st, is_suspended 等过滤标记

# 保存清洗后的数据
storage.save_clean_by_date(daily_clean, "daily", trade_date)
```

## 应用场景

### 使用收盘价卖出的场景（默认）

- **常规回测**：适合大多数量化策略
- **长期持仓**：持有期较长（如周频、月频调仓）
- **容错性高**：不依赖开盘价数据，数据缺失风险低

### 使用开盘价卖出的场景

- **快速止损**：需要在开盘后快速卖出止损
- **捕捉跳空**：利用隔夜跳空机会
- **高频策略**：持有期较短，需要快速变现
- **实盘模拟**：更接近实际交易（集合竞价后开盘卖出）

### 性能对比

一般情况下：

- **开盘价卖出**：可能获得更早的退出时机，但也可能错过当日上涨
- **收盘价卖出**：捕捉全天涨幅，但可能无法及时止损

具体差异取决于市场行情和持仓股票特性。

## 技术细节

### 双价格体系

LazyBull 使用双价格体系分离现金流和绩效计算：

| 价格类型 | 字段 | 用途 |
|---------|------|------|
| **成交价格** | `close` / `open` | 计算成交金额、持仓市值、可买入数量 |
| **绩效价格** | `close_adj` / `open_adj` | 计算收益率、夏普比率等绩效指标 |

这种设计避免了复权调整导致的虚增收益，同时保证绩效指标的准确性。

### 价格选择逻辑

卖出价格的选择遵循以下逻辑：

```
如果 sell_timing == 'open':
    尝试获取开盘成交价格 (open)
    如果开盘价不存在或为NaN:
        降级到收盘成交价格 (close)
        输出警告日志
    
    尝试获取开盘绩效价格 (open_adj)
    如果开盘绩效价格不存在或为NaN:
        降级到收盘绩效价格 (close_adj)
        
否则 (sell_timing == 'close'):
    使用收盘成交价格 (close)
    使用收盘绩效价格 (close_adj)
```

### 交易记录

每笔卖出交易会在交易记录中包含 `sell_timing` 字段：

```python
{
    'date': pd.Timestamp('2023-01-10'),
    'stock': '000001.SZ',
    'action': 'sell',
    'price': 10.05,  # 实际卖出价格
    'shares': 1000,
    'amount': 10050.0,
    'cost': 10.05,
    'sell_timing': 'open',  # 或 'close'
    'sell_type': 'holding_period',  # 或 'stop_loss'
    # ... 其他字段
}
```

## 限制和注意事项

### 1. 数据依赖

- 使用 `sell_timing='open'` 需要数据源包含开盘价
- TuShare Pro 提供的日线数据默认包含 `open` 字段
- 缺少开盘价时会自动降级到收盘价，但会输出警告

### 2. 回测假设

- **开盘价成交假设**：假设可以在开盘价成交，实际可能有滑点
- **集合竞价**：未模拟集合竞价过程，直接使用开盘价
- **流动性**：未考虑开盘时的流动性问题

### 3. 停牌和涨跌停

- 停牌股票：开盘价和收盘价都为 0 或 NaN，会被延迟订单系统处理
- 涨跌停：一字板情况下开盘价=收盘价，两种模式结果相同

### 4. 成本影响

- 卖出时机不影响手续费计算（按成交金额固定比例）
- 开盘价低于收盘价时，开盘卖出的收益可能略低

## 常见问题

### Q1: 默认卖出时机是什么？

**A**: 默认为 `sell_timing='close'`，即在 T+n 日收盘价卖出。这是最常用的设置，也是向后兼容的选择。

### Q2: 如果数据没有开盘价怎么办？

**A**: 系统会自动降级到收盘价卖出，并在日志中输出警告。不会导致回测失败。

### Q3: 开盘价卖出一定比收盘价卖出好吗？

**A**: 不一定。这取决于具体的市场行情和股票特性。建议通过回测对比两种模式的实际效果。

### Q4: 可以买入也用开盘价吗？

**A**: 当前版本仅支持卖出时机配置。买入固定在 T+1 日收盘价。未来版本可能会扩展买入时机配置。

### Q5: 如何验证卖出时机配置生效？

**A**: 检查交易记录中的 `sell_timing` 字段和 `price` 字段，对比开盘价和收盘价数据。

## 版本历史

- **v0.5.0** (2026-01-20): 首次发布卖出时机配置功能
  - 支持 `sell_timing` 参数
  - 实现开盘价/收盘价双价格体系
  - 添加完整的降级策略和错误处理
  - 包含完整的单元测试覆盖

## 相关文档

- [回测假设文档](backtest_assumptions.md)
- [数据契约文档](data_contract.md)
- [交易状态处理指南](trade_status_guide.md)
