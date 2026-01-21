# 纸面交易（Paper Trading）使用指南

## 概述

纸面交易模块实现了一个完整的日频工作流，可用于模拟实盘交易，而不实际提交真实订单。适合用于：
- 验证策略在实盘环境下的表现
- 生成每日调仓指令供手工执行
- 跟踪纸面账户的净值表现

## 核心特性

- **统一运行入口（v0.7.0）**：`run` 命令自动编排执行所有动作（止损、延迟卖出、T1、T0）
- **全局配置管理（v0.7.0）**：`config` 命令持久化账户级配置
- **止损功能（v0.7.0）**：支持回撤止损、移动止损、连续跌停止损
- **智能调度（v0.7.0）**：非调仓日允许仅执行卖出和T1，跳过T0
- **交易日自动校正**：输入非交易日自动滚动到下一交易日
- **调仓频率控制**：可设置调仓频率，仅在调仓日允许生成新目标
- **100股整手交易**：买卖严格按100股操作，清仓自动处理零股
- **幂等性保障**：同日 T0/T1 只能执行一次，防止重复操作
- **延迟卖出机制**：跌停/停牌股票自动延迟，自动重试
- **完整的持久化**：账户状态、交易记录、净值曲线均可持久化和恢复
- **详细的打印输出**：包含股票、方向、权重、价格、成本、原因等完整信息
- **灵活的价格配置**：买入可选开盘价/收盘价，卖出可选开盘价/收盘价
- **主板股票池**：仅包含沪深主板股票，排除科创板、创业板、北交所
- **成本计算**：包括佣金、印花税、滑点

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境

创建 `.env` 文件并设置 TuShare token：

```bash
TS_TOKEN=your_tushare_token_here
```

### 3. 设置全局配置（首次使用）

在开始纸面交易之前，需要先使用 `config` 命令设置全局配置：

```bash
python scripts/paper_trade.py config \
    --buy-price close \
    --sell-price close \
    --top-n 5 \
    --initial-capital 500000 \
    --rebalance-freq 5 \
    --weight-method equal \
    --universe mainboard \
    --stop-loss-enabled \
    --stop-loss-drawdown-pct 20.0 \
    --stop-loss-trailing-enabled \
    --stop-loss-trailing-pct 15.0 \
    --stop-loss-consecutive-limit-down 2
```

**配置参数说明：**
- `--buy-price`: 买入价格类型（`open` 或 `close`，默认 `close`）
- `--sell-price`: 卖出价格类型（`open` 或 `close`，默认 `close`）
- `--top-n`: 持仓股票数（默认：5）
- `--initial-capital`: 初始资金（默认：500000）
- `--rebalance-freq`: 调仓频率（交易日数，默认：5）
- `--weight-method`: 权重分配方法（`equal` 等权，`score` 按分数加权，默认：equal）
- `--model-version`: ML模型版本（可选）
- `--universe`: 股票池类型（`mainboard` 仅主板，`all` 全市场，默认：mainboard）
- `--stop-loss-enabled`: 启用止损功能（标志位）
- `--stop-loss-drawdown-pct`: 回撤止损百分比（默认：20.0）
- `--stop-loss-trailing-enabled`: 启用移动止损（标志位）
- `--stop-loss-trailing-pct`: 移动止损百分比（默认：15.0）
- `--stop-loss-consecutive-limit-down`: 连续跌停触发天数（默认：2）

**配置存储：**
- 配置保存在 `data/paper/config.json`
- 可随时运行 `config` 命令更新配置

### 4. 每日运行纸面交易

配置完成后，每天只需运行一次 `run` 命令：

```bash
python scripts/paper_trade.py run --trade-date 20260121
```

**参数说明：**
- `--trade-date`: 交易日期（格式：YYYYMMDD）。**注意**：如果输入非交易日，会自动校正到下一交易日
- `--model-version`: ML模型版本（可选，覆盖配置中的默认值）
- `--weight-method`: 权重分配方法（可选，覆盖配置中的默认值）

**`run` 命令自动执行流程：**

1. **交易日校正**：非交易日自动滚动到下一交易日
2. **执行止损检查**：
   - 检查所有持仓是否触发止损
   - 触发止损的标的生成卖出指令
   - 如果跌停不可卖出，自动加入延迟卖出队列
3. **处理延迟卖出队列**：
   - 尝试卖出之前因跌停/停牌延迟的订单
   - 成功卖出的订单从队列移除
   - 仍不可卖出的订单保留在队列中
4. **执行 T1（如有待执行目标）**：
   - 检查是否存在 `pending_weights/{trade_date}.parquet`
   - 如存在，生成并执行调仓订单
   - 更新账户状态和净值
   - 同日只执行一次（幂等保护）
5. **判断是否调仓日并执行 T0**：
   - 检查距离上次调仓是否满足 `rebalance_freq`
   - 如果是调仓日，拉取数据并生成信号
   - 生成的目标保存到下一交易日的 `pending_weights`
   - 同日只执行一次（幂等保护）
   - **非调仓日自动跳过 T0，不会报错**

**输出示例：**

```
================================================================================
纸面交易自动运行
================================================================================
交易日期: 20260121
使用配置：
  买入价格类型: close
  卖出价格类型: close
  持仓数: 5
  调仓频率: 5 个交易日
  权重方法: equal
  止损开关: True
================================================================================

--------------------------------------------------------------------------------
步骤1: 检查止损触发
--------------------------------------------------------------------------------
止损检查完成：触发 1 个止损信号

--------------------------------------------------------------------------------
步骤2: 处理延迟卖出队列
--------------------------------------------------------------------------------
延迟卖出处理完成：成交 0 笔，剩余 1 笔

--------------------------------------------------------------------------------
步骤3: 检查并执行 T1
--------------------------------------------------------------------------------
找到 5 个待执行目标，执行 T1
... （执行明细）
T1 执行完成：5 个订单

--------------------------------------------------------------------------------
步骤4: 检查是否调仓日并执行 T0
--------------------------------------------------------------------------------
当前是调仓日，执行 T0
... （执行流程）
T0 执行完成：生成 5 个目标

============================================================================
手工操作指令汇总
============================================================================

【止损卖出清单】
----------------------------------------------------------------------------
股票代码         建议股数    是否可执行     原因
----------------------------------------------------------------------------
600000.SH       2000       否(跌停)      回撤止损: 从买入价15.00下跌至12.00，跌幅20.00%

【延迟卖出清单】
----------------------------------------------------------------------------
股票代码         待卖股数    状态          原因
----------------------------------------------------------------------------
600000.SH       2000       不可卖出      止损-回撤止损: 从买入价15.00下跌至12.00，跌幅20.00%

【T1 调仓订单清单】
----------------------------------------------------------------------------
股票代码         方向      股数          原因
----------------------------------------------------------------------------
000001.SZ       buy       2000         新建仓位
000002.SZ       buy       1000         新建仓位

【T0 生成目标清单】
----------------------------------------------------------------------------
股票代码         目标权重      原因/评分
----------------------------------------------------------------------------
000003.SZ       0.2000       信号生成
000004.SZ       0.2000       信号生成

============================================================================
运行完成 - 20260121
============================================================================
```

### 5. 查看持仓明细

使用 `positions` 子命令查看当前持仓状态和收益情况：

```bash
python scripts/paper_trade.py positions --trade-date 20260122
```

**输出示例：**

```
================================================================================
持仓明细
================================================================================
股票代码       股数     买入均价     买入成本     买入日期     持有天数  当前价格     当前市值       浮盈         收益率(%)    状态    
--------------------------------------------------------------------------------
000001.SZ    1000     12.50      25.00      20260115   7       13.20      13200.00     450.00     3.53       持有    
000002.SZ    800      15.30      20.40      20260116   6       15.80      12640.00     379.60     3.09       持有    
--------------------------------------------------------------------------------
合计         1800                45.40                         25840.00     829.60     3.32      
================================================================================
账户现金: 475,134.00
持仓市值: 25,840.00
总资产: 500,974.00
================================================================================
```

## 工作流示例

### 连续多日运行

```bash
# 首次配置（仅需一次）
python scripts/paper_trade.py config \
    --buy-price close \
    --sell-price close \
    --top-n 5 \
    --initial-capital 500000 \
    --rebalance-freq 5 \
    --weight-method equal \
    --stop-loss-enabled

# Day 1 (2026-01-21)
python scripts/paper_trade.py run --trade-date 20260121

# Day 2 (2026-01-22)
python scripts/paper_trade.py run --trade-date 20260122

# Day 3 (2026-01-23)
python scripts/paper_trade.py run --trade-date 20260123

# 查看持仓
python scripts/paper_trade.py positions --trade-date 20260123
```

### 使用 ML 模型

```bash
# 配置时指定模型版本
python scripts/paper_trade.py config \
    --buy-price close \
    --sell-price close \
    --top-n 10 \
    --model-version 1 \
    --weight-method score

# 或在运行时覆盖
python scripts/paper_trade.py run \
    --trade-date 20260121 \
    --model-version 2
```

## 止损功能详解

### 回撤止损

当持仓从买入成本价下跌超过设定百分比时触发。

**配置：**
```bash
--stop-loss-enabled \
--stop-loss-drawdown-pct 20.0
```

**触发条件：** 当前价格 ≤ 买入价格 × (1 - drawdown_pct/100)

### 移动止损

当持仓从最高价下跌超过设定百分比时触发。

**配置：**
```bash
--stop-loss-enabled \
--stop-loss-trailing-enabled \
--stop-loss-trailing-pct 15.0
```

**触发条件：** 当前价格 ≤ 最高价格 × (1 - trailing_pct/100)

**注意：** 系统会自动跟踪每个持仓的历史最高价

### 连续跌停止损

当持仓连续跌停达到设定天数时触发。

**配置：**
```bash
--stop-loss-enabled \
--stop-loss-consecutive-limit-down 2
```

**触发条件：** 连续跌停天数 ≥ consecutive_limit_down

**注意：** 如果某天不跌停，计数器会重置为0

### 止损状态持久化

止损监控器的运行状态（最高价、连续跌停计数）会自动持久化到 `data/paper/state/stop_loss_state.json`，确保重启后状态不丢失。

## 数据结构

### 持久化目录

```
data/paper/
├── config.json           # 【新增 v0.7.0】全局配置
├── pending/              # 待执行目标权重
│   └── {YYYYMMDD}.parquet
├── state/                # 账户状态
│   ├── account.json
│   └── stop_loss_state.json  # 【新增 v0.7.0】止损状态
├── trades/               # 交易记录
│   └── trades.parquet
├── nav/                  # 净值曲线
│   └── nav.parquet
├── runs/                 # 执行记录（幂等性）
│   ├── t0_{YYYYMMDD}.json
│   ├── t1_{YYYYMMDD}.json
│   └── rebalance_state.json
└── pending_sells/        # 延迟卖出队列
    └── pending_sells.json
```

### 全局配置 (config.json) - 新增

```json
{
  "buy_price": "close",
  "sell_price": "close",
  "top_n": 5,
  "initial_capital": 500000.0,
  "rebalance_freq": 5,
  "weight_method": "equal",
  "model_version": null,
  "stop_loss_enabled": true,
  "stop_loss_drawdown_pct": 20.0,
  "stop_loss_trailing_enabled": true,
  "stop_loss_trailing_pct": 15.0,
  "stop_loss_consecutive_limit_down": 2,
  "universe": "mainboard"
}
```

### 止损状态 (stop_loss_state.json) - 新增

```json
{
  "position_high_prices": {
    "000001.SZ": 15.8,
    "000002.SZ": 22.5
  },
  "consecutive_limit_down_days": {
    "000001.SZ": 0,
    "000002.SZ": 1
  }
}
```

## 命令对比（v0.6.0 vs v0.7.0）

### v0.6.0（旧版本）

需要分别运行多个命令：

```bash
# T0 生成信号
python scripts/paper_trade.py t0 --trade-date 20260121 --buy-price close --top-n 5 --rebalance-freq 5

# T1 执行调仓
python scripts/paper_trade.py t1 --trade-date 20260122 --buy-price close --sell-price close

# 重试延迟卖出
python scripts/paper_trade.py retry --trade-date 20260122 --sell-price close
```

### v0.7.0（新版本）

一条命令自动执行所有动作：

```bash
# 首次配置
python scripts/paper_trade.py config --buy-price close --sell-price close --top-n 5 --rebalance-freq 5

# 每日运行
python scripts/paper_trade.py run --trade-date 20260122
```

**优势：**
- 更少的命令，更简单的流程
- 自动处理止损、延迟卖出、T1、T0
- 非调仓日自动跳过 T0，无需手动判断
- 统一的输出格式，便于手工实盘

## 注意事项

1. **必须先配置**：首次使用前必须运行 `config` 命令设置配置
2. **数据依赖**：确保已下载必要的基础数据（trade_cal、stock_basic）
3. **TuShare 限制**：免费版 TuShare 有访问频率限制，请合理安排拉取频率
4. **状态恢复**：每次运行后会自动保存状态，可随时中断和恢复
5. **交易日自动校正**：输入非交易日会自动滚动到下一交易日
6. **幂等性保护**：T0 和 T1 命令同一日期只能执行一次，重复执行会报错
7. **100股整手规则**：所有买卖都按100股操作，清仓时零股会保留并标记
8. **延迟卖出自动重试**：延迟卖出在 `run` 命令中自动重试，无需手动执行
9. **止损状态持久化**：止损监控器的状态会自动保存和恢复
10. **非调仓日允许卖出**：即使不是调仓日，止损卖出和延迟卖出仍会执行

## 高级用法

### 动态调整配置

配置可以随时更新：

```bash
# 调整止损参数
python scripts/paper_trade.py config \
    --buy-price close \
    --sell-price close \
    --top-n 5 \
    --initial-capital 500000 \
    --rebalance-freq 5 \
    --weight-method equal \
    --stop-loss-enabled \
    --stop-loss-drawdown-pct 15.0 \
    --stop-loss-trailing-pct 10.0
```

### 临时覆盖配置

运行时可以临时覆盖部分配置：

```bash
# 临时使用不同的模型版本
python scripts/paper_trade.py run \
    --trade-date 20260121 \
    --model-version 2 \
    --weight-method score
```

### 查看账户状态

```bash
# 使用 Python 读取配置
python -c "
import json
with open('data/paper/config.json') as f:
    config = json.load(f)
    for k, v in config.items():
        print(f'{k}: {v}')
"

# 查看账户状态
python -c "
import json
with open('data/paper/state/account.json') as f:
    state = json.load(f)
    print(f'现金: {state[\"cash\"]:,.2f}')
    print(f'持仓数: {len(state[\"positions\"])}')
"

# 查看止损状态
python -c "
import json
with open('data/paper/state/stop_loss_state.json') as f:
    state = json.load(f)
    print('最高价记录:', state['position_high_prices'])
    print('连续跌停计数:', state['consecutive_limit_down_days'])
"
```

### 查看交易记录

```bash
# 使用 pandas 读取
python -c "
import pandas as pd
trades = pd.read_parquet('data/paper/trades/trades.parquet')
print(trades.tail(10))
"
```

### 查看净值曲线

```bash
# 使用 pandas 读取并绘图
python -c "
import pandas as pd
import matplotlib.pyplot as plt
nav = pd.read_parquet('data/paper/nav/nav.parquet')
nav.plot(x='trade_date', y='nav')
plt.show()
"
```

## 故障排查

### 问题：提示未找到配置文件

**原因**：未运行 `config` 命令

**解决**：
```bash
python scripts/paper_trade.py config \
    --buy-price close \
    --sell-price close \
    --top-n 5 \
    --initial-capital 500000 \
    --rebalance-freq 5
```

### 问题：未找到待执行目标

**原因**：T0 未成功运行或日期不匹配

**解决**：
1. 检查 `data/paper/pending/` 目录下是否有对应日期的文件
2. 确认 T0 在调仓日成功执行
3. 使用 `run` 命令会自动处理这种情况

### 问题：无法加载数据

**原因**：缺少基础数据或 clean 数据

**解决**：
```bash
# 下载基础数据
python scripts/update_basic_data.py

# 下载并清洗日线数据
python scripts/download_raw.py --start-date 20260101 --end-date 20260131
python scripts/build_clean_features.py --start-date 20260101 --end-date 20260131 --only-clean
```

### 问题：止损未触发

**原因**：
1. 止损功能未启用
2. 止损条件未达到
3. 止损状态未正确加载

**解决**：
1. 检查配置：`stop_loss_enabled` 是否为 `true`
2. 检查止损参数设置是否合理
3. 查看 `data/paper/state/stop_loss_state.json` 确认状态

## 相关文档

- [回测引擎说明](../docs/backtest_assumptions.md)
- [数据契约](../docs/data_contract.md)
- [ML 模型训练](../README.md#机器学习模型训练与回测)

## 支持

如有问题，请访问：
- [GitHub Issues](https://github.com/deltree-y/LazyBull/issues)
- [项目主页](https://github.com/deltree-y/LazyBull)

## 更新日志

### v0.7.0（当前版本）

- **重大重构**：统一命令接口
  - 新增 `config` 命令：持久化全局配置
  - 新增 `run` 命令：自动编排执行所有动作
  - 保留 `positions` 命令：查看持仓
  - 移除 `t0`、`t1`、`retry` 命令（旧版本已备份）

- **止损功能**：
  - 回撤止损：从买入成本价触发
  - 移动止损：从历史最高价触发
  - 连续跌停止损：连续N天跌停触发
  - 状态持久化：自动保存和恢复止损状态

- **智能调度**：
  - 非调仓日允许执行止损、延迟卖出、T1
  - 自动跳过非调仓日的 T0
  - 延迟卖出自动重试

- **更好的输出**：
  - 统一的手工操作指令汇总
  - 清晰的表格格式
  - 完整的动作记录

### v0.6.0

- T0/T1 分离工作流
- 交易日自动校正
- 调仓频率控制
- 100股整手交易
- 幂等性保障
- 延迟卖出机制
