# 止损触发功能说明

## 概述

止损功能提供基于回撤、连续跌停等条件的触发式卖出，实现风险控制。不必等到调仓日，当触发条件满足时立即卖出（或在下一交易日开盘卖出）。

## 功能说明

### 支持的止损类型

1. **回撤止损（Drawdown Stop Loss）**
   - 从买入成本价回撤超过设定百分比时触发
   - 适用于控制单笔交易最大损失

2. **移动止损（Trailing Stop Loss）**
   - 从持仓期间最高价回撤超过设定百分比时触发
   - 适用于保护已有盈利

3. **连续跌停止损**
   - 连续N天跌停时触发
   - 适用于识别极端风险信号

### 触发后处理策略

- **hold_cash**: 触发后卖出并持有现金，等待下一个调仓日
- **buy_alternative**: 触发后卖出并从备选池补买其他标的（待实现）

## 配置说明

在 `configs/base.yaml` 中配置：

```yaml
stop_loss:
  enabled: false  # 是否启用止损功能
  # 回撤止损
  drawdown_pct: 20.0  # 从买入成本回撤超过N%触发止损
  # 移动止损
  trailing_enabled: false  # 是否启用移动止损
  trailing_pct: 15.0  # 从最高点回撤超过N%触发止损
  # 连续跌停止损
  consecutive_limit_down: 2  # 连续N天跌停触发止损
  # 触发后操作
  post_action: "hold_cash"  # 触发后操作
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `enabled` | bool | false | 是否启用止损功能 |
| `drawdown_pct` | float | 20.0 | 回撤止损阈值（%） |
| `trailing_enabled` | bool | false | 是否启用移动止损 |
| `trailing_pct` | float | 15.0 | 移动止损阈值（%） |
| `consecutive_limit_down` | int | 2 | 连续跌停天数阈值 |
| `post_action` | str | hold_cash | 触发后操作 |

## 使用示例

### 示例1：仅启用回撤止损

```yaml
stop_loss:
  enabled: true
  drawdown_pct: 15.0  # 亏损15%止损
  trailing_enabled: false
  consecutive_limit_down: 2
  post_action: "hold_cash"
```

### 示例2：回撤止损 + 移动止损

```yaml
stop_loss:
  enabled: true
  drawdown_pct: 20.0  # 亏损20%止损
  trailing_enabled: true
  trailing_pct: 10.0  # 从最高点回撤10%止损
  consecutive_limit_down: 2
  post_action: "hold_cash"
```

### 示例3：完整风控配置

```yaml
stop_loss:
  enabled: true
  drawdown_pct: 20.0
  trailing_enabled: true
  trailing_pct: 15.0
  consecutive_limit_down: 2  # 连续2天跌停止损
  post_action: "hold_cash"
```

### 示例4：Python 代码使用

```python
from src.lazybull.risk import StopLossConfig, StopLossMonitor
from src.lazybull.backtest import BacktestEngine

# 创建止损配置
stop_loss_config = StopLossConfig(
    enabled=True,
    drawdown_pct=15.0,
    trailing_stop_enabled=True,
    trailing_stop_pct=10.0,
    consecutive_limit_down_days=2,
    post_trigger_action='hold_cash'
)

# 创建止损监控器
stop_loss_monitor = StopLossMonitor(stop_loss_config)

# 在回测引擎中使用（集成到引擎后）
engine = BacktestEngine(
    universe=universe,
    signal=signal,
    initial_capital=500000,
    # stop_loss_config=stop_loss_config  # 待集成
)
```

## 止损触发流程

### 1. 每日监控

回测引擎在每个交易日会：
1. 遍历所有持仓
2. 获取当前价格和交易状态
3. 调用 `stop_loss_monitor.check_stop_loss()` 检查止损条件
4. 如果触发止损，立即生成卖出信号

### 2. 止损执行

触发止损后：
1. 记录止损原因和触发时间
2. 生成卖出订单
3. 在下一可交易日执行卖出
4. 根据 `post_action` 决定后续操作

### 3. 日志记录

系统会记录详细的止损日志：
```
[WARNING] 600000.SH 触发止损: 回撤止损: 从买入价10.50下跌至8.19，跌幅22.00%
[INFO] 止损卖出: 600000.SH, 买入价=10.50, 当前价=8.19, 触发类型=drawdown
```

## 边界情况处理

### 1. 跌停无法卖出

如果触发止损当天股票跌停：
- 订单进入延迟队列
- 在后续交易日继续尝试卖出
- 直到成功卖出或达到最大重试次数

### 2. 停牌

如果触发止损后股票停牌：
- 订单进入延迟队列
- 等待复牌后卖出

### 3. 移动止损的最高价记录

- 买入时初始化为买入价
- 每日更新为历史最高价
- 卖出后清除记录

### 4. 连续跌停计数

- 每日更新跌停状态
- 连续跌停计数累加
- 非跌停日重置计数为0

## 注意事项

### 1. 止损与正常调仓的关系

- 止损是主动卖出，不等待调仓日
- 触发止损后，该标的从持仓中移除
- 下一个调仓日会按正常逻辑重新选股

### 2. 止损参数设置建议

- **回撤止损**: 建议设置在 15%-25% 之间
  - 过小：容易误触发，增加交易成本
  - 过大：风险保护不足

- **移动止损**: 建议设置在 10%-20% 之间
  - 用于保护已有盈利
  - 可以比回撤止损更宽松

- **连续跌停**: 建议设置 2-3 天
  - 2天：灵敏度高，但可能误触发
  - 3天：更稳健，但可能损失已扩大

### 3. 性能影响

止损检查的计算开销很小：
- 每个持仓每天执行一次检查
- 主要是简单的数值比较
- 对回测速度影响可忽略不计

### 4. 历史回测局限性

- 回测中假设止损触发日可以成交
- 实盘中可能因流动性不足或跌停无法成交
- 建议保守设置止损参数

## 扩展功能（待实现）

1. **补买备选标的**
   - 止损后从备选池选择新标的补仓
   - 保持持仓数量稳定

2. **动态止损**
   - 根据市场波动率动态调整止损阈值
   - 牛市放宽，熊市收紧

3. **止盈功能**
   - 与止损对称的止盈触发器
   - 达到目标收益率后自动卖出

4. **分档止损**
   - 按盈亏程度分段设置不同止损比例
   - 亏损越大，止损阈值越宽松

## 示例：回测报告中的止损信息

启用止损后，回测报告中会包含止损统计：

```
============================================================
回测报告摘要
============================================================
总收益率      : 15.23%
年化收益率    : 15.50%
最大回撤      : -8.45%
波动率        : 12.30%
夏普比率      : 1.25
交易次数      : 24
  正常调仓    : 18
  止损触发    : 6  <-- 新增统计
止损触发详情:
  回撤止损    : 4次
  移动止损    : 1次
  连续跌停    : 1次
总交易成本    : 12345.67元
============================================================
```

## 相关文档

- [回测假设说明](backtest_assumptions.md)
- [交易状态处理指南](trade_status_guide.md)
- [涨跌停与停牌处理](trade_status_guide.md)
