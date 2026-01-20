# 止损触发功能说明

## 概述

止损功能提供基于回撤、连续跌停等条件的触发式卖出，实现风险控制。**止损触发后不会在当日立即卖出，而是生成 T+1 卖出信号，在下一交易日收盘价执行卖出**，与回测引擎的 T 日出信号、T+1 日执行的交易规则保持一致。

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

### 止损执行时序（T+1 卖出）

止损功能遵循回测引擎的 T+1 执行规则：

1. **T 日**：检查持仓是否触发止损条件
   - 如果触发，记录止损信号到待卖出队列
   - 不会在当日立即卖出
   
2. **T+1 日**：执行止损卖出
   - 在下一交易日以收盘价卖出
   - 与正常卖出一样，会进行交易状态检查（跌停/停牌等）
   - 如果 T+1 日不可卖出（如跌停），订单进入延迟队列继续重试

这与回测引擎的正常交易流程一致：
- 正常调仓：T 日生成信号 → T+1 日执行买入
- 持有期卖出：T 日达到持有期 → T 日执行卖出
- **止损卖出：T 日触发止损 → T+1 日执行卖出**

## 配置说明

### 方式一：命令行参数配置（推荐用于 run_ml_backtest.py）

使用 `run_ml_backtest.py` 时，可通过命令行参数配置止损：

```bash
# 启用回撤止损（20%）
python scripts/run_ml_backtest.py \
    --start-date 20230101 \
    --end-date 20231231 \
    --stop-loss-enabled \
    --stop-loss-drawdown-pct 20.0

# 启用回撤止损 + 移动止损
python scripts/run_ml_backtest.py \
    --start-date 20230101 \
    --end-date 20231231 \
    --stop-loss-enabled \
    --stop-loss-drawdown-pct 20.0 \
    --stop-loss-trailing-enabled \
    --stop-loss-trailing-pct 15.0 \
    --stop-loss-consecutive-limit-down 2
```

#### 命令行参数说明

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `--stop-loss-enabled` | flag | False | 启用止损功能 |
| `--stop-loss-drawdown-pct` | float | 20.0 | 回撤止损阈值（%） |
| `--stop-loss-trailing-enabled` | flag | False | 启用移动止损 |
| `--stop-loss-trailing-pct` | float | 15.0 | 移动止损阈值（%） |
| `--stop-loss-consecutive-limit-down` | int | 2 | 连续跌停天数阈值 |

### 方式二：YAML 配置文件（用于自定义回测脚本）

在 `configs/base.yaml` 或自定义配置文件中：

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

### 示例1：使用 run_ml_backtest.py（推荐）

```bash
# 仅启用回撤止损（15%）
python scripts/run_ml_backtest.py \
    --start-date 20230101 \
    --end-date 20231231 \
    --top-n 5 \
    --rebalance-freq 10 \
    --stop-loss-enabled \
    --stop-loss-drawdown-pct 15.0

# 启用回撤止损 + 移动止损
python scripts/run_ml_backtest.py \
    --start-date 20230101 \
    --end-date 20231231 \
    --top-n 5 \
    --rebalance-freq 10 \
    --stop-loss-enabled \
    --stop-loss-drawdown-pct 20.0 \
    --stop-loss-trailing-enabled \
    --stop-loss-trailing-pct 10.0 \
    --stop-loss-consecutive-limit-down 2

# 完整示例（包含其他参数）
python scripts/run_ml_backtest.py \
    --start-date 20230101 \
    --end-date 20231231 \
    --initial-capital 500000 \
    --rebalance-freq 10 \
    --top-n 5 \
    --weight-method equal \
    --exclude-st \
    --min-list-days 60 \
    --stop-loss-enabled \
    --stop-loss-drawdown-pct 20.0 \
    --stop-loss-trailing-enabled \
    --stop-loss-trailing-pct 15.0 \
    --stop-loss-consecutive-limit-down 2
```

### 示例2：在自定义 Python 代码中使用

```python
from src.lazybull.risk import StopLossConfig, StopLossMonitor
from src.lazybull.backtest import BacktestEngineML

# 方式1：直接创建配置对象
stop_loss_config = StopLossConfig(
    enabled=True,
    drawdown_pct=15.0,
    trailing_stop_enabled=True,
    trailing_stop_pct=10.0,
    consecutive_limit_down_days=2,
    post_trigger_action='hold_cash'
)

# 方式2：从字典创建配置（例如从 YAML 加载）
from src.lazybull.risk.stop_loss import create_stop_loss_config_from_dict

config_dict = {
    'stop_loss_enabled': True,
    'stop_loss_drawdown_pct': 15.0,
    'stop_loss_trailing_enabled': True,
    'stop_loss_trailing_pct': 10.0,
    'stop_loss_consecutive_limit_down': 2,
    'stop_loss_post_action': 'hold_cash'
}
stop_loss_config = create_stop_loss_config_from_dict(config_dict)

# 在回测引擎中使用
engine = BacktestEngineML(
    universe=universe,
    signal=signal,
    features_by_date=features_by_date,
    initial_capital=500000,
    rebalance_freq=10,
    stop_loss_config=stop_loss_config  # 传入止损配置
)
```

## 止损触发流程

### 1. 每日监控

回测引擎在每个交易日会：
1. 遍历所有持仓
2. 获取当前价格和交易状态
3. 调用 `stop_loss_monitor.check_stop_loss()` 检查止损条件
4. **如果触发止损，记录到待卖出队列，不会立即卖出**

### 2. 止损执行（T+1）

触发止损后：
1. **T 日**：记录止损原因和触发时间，加入待卖出队列
2. **T+1 日**：在下一交易日执行卖出
   - 检查交易状态（是否跌停/停牌）
   - 如果可交易，以收盘价卖出
   - 如果不可交易，加入延迟队列继续重试
3. 卖出完成后，清理止损监控器中的持仓状态
4. 根据 `post_action` 决定后续操作

### 3. 日志记录

系统会记录详细的止损日志：
```
[WARNING] 600000.SH 触发止损: 回撤止损: 从买入价10.50下跌至8.19，跌幅22.00%, 将在下一交易日执行卖出
[INFO] 止损卖出执行: 2023-06-15, 卖出 1 只股票（触发日: 2023-06-14）
```

### 4. 交易记录区分

在交易记录（trades）中，止损卖出会被标记：
- `sell_type`: `'stop_loss'`（止损卖出）或 `'holding_period'`（持有期卖出）
- `sell_reason`: 止损触发原因（仅止损卖出）
- `trigger_type`: 止损触发类型（仅止损卖出，如 `'drawdown'`, `'trailing_stop'`, `'consecutive_limit_down'`）

## 边界情况处理

### 1. 跌停无法卖出

如果 T+1 日股票跌停无法卖出：
- 订单进入延迟队列
- 在后续交易日继续尝试卖出
- 直到成功卖出或达到最大重试次数

### 2. 停牌

如果 T+1 日股票停牌：
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

### 5. 避免重复触发

- 止损触发后，股票加入待卖出队列
- 后续日期检查时，如果股票已在队列中，跳过检查
- 卖出完成后，从队列中移除

### 6. 与正常调仓的协同

- 如果某股票止损触发生成 T+1 卖出信号，但在 T 日晚上被正常调仓卖出
- T+1 日执行止损卖出时会检查持仓是否存在
- 如果已被卖出，则跳过该止损订单

## 注意事项

### 1. 止损与正常调仓的关系

- 止损是主动卖出，不等待调仓日
- 止损触发后生成 T+1 卖出信号
- 该标的从持仓中移除后，下一个调仓日会按正常逻辑重新选股

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

- **T+1 执行的现实性**：回测中假设 T+1 日可以成交，但实盘中可能因流动性不足或连续跌停无法成交
- **价格假设**：回测使用收盘价执行，实盘可能有滑点
- 建议保守设置止损参数

### 5. 默认行为（向后兼容）

- 止损功能默认关闭（`enabled=false`）
- 不使用 `--stop-loss-enabled` 参数时，回测行为与之前完全一致
- 不会影响现有脚本和配置

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
