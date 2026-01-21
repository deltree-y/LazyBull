# 纸面交易（Paper Trading）使用指南

## 概述

纸面交易模块实现了一个完整的日频工作流，可用于模拟实盘交易，而不实际提交真实订单。适合用于：
- 验证策略在实盘环境下的表现
- 生成每日调仓指令供手工执行
- 跟踪纸面账户的净值表现

## 核心特性

- **T0/T1 分离工作流**：T0 收盘后生成信号，T1 执行调仓
- **完整的持久化**：账户状态、交易记录、净值曲线均可持久化和恢复
- **详细的打印输出**：包含股票、方向、权重、价格、成本、原因等完整信息
- **灵活的价格配置**：买入可选开盘价/收盘价，卖出固定收盘价
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

### 3. 运行 T0 工作流

T0 工作流在交易日收盘后运行，拉取数据并生成次日待执行目标：

```bash
python scripts/paper_trade.py t0 \
    --trade-date 20260121 \
    --buy-price close \
    --universe mainboard \
    --top-n 5
```

**参数说明：**
- `--trade-date`: T0 交易日期（格式：YYYYMMDD）
- `--buy-price`: T1 买入价格类型（`open` 或 `close`，默认 `close`）
- `--universe`: 股票池类型（`mainboard` 仅主板，`all` 全市场，默认 `mainboard`）
- `--top-n`: 持仓股票数（默认：5）
- `--model-version`: ML模型版本（可选，指定则使用ML信号）
- `--initial-capital`: 初始资金（默认：500000）
- `--weight-method`: **新增** 权重分配方法（`equal` 等权，`score` 按分数加权，默认：equal）

**数据下载机制变更（v0.5.0）：**
- 不再在paper模块中自建下载逻辑
- 复用仓库既有的TushareClient和DataCleaner能力
- 自动下载raw数据：日线行情、复权因子、停复牌信息、涨跌停信息
- 自动构建clean数据：数据清洗 + 可交易性标记（`is_suspended`、`is_limit_up`、`is_limit_down`等）
- 数据保存到标准的partitioned存储路径（`data/raw` 和 `data/clean`）
- `--top-n`: 持仓股票数（默认 5）
- `--model-version`: ML 模型版本（可选，如果有训练好的模型）
- `--initial-capital`: 初始资金（默认 500000）

**输出：**
- 拉取 T0 日的行情数据
- 生成 T1 日的待执行目标权重
- 保存到 `data/paper/pending/{T1日期}.parquet`

### 4. 运行 T1 工作流

T1 工作流在次日运行，读取待执行目标并生成订单：

```bash
python scripts/paper_trade.py t1 \
    --trade-date 20260122 \
    --buy-price close \
    --sell-price close
```

**参数说明：**
- `--trade-date`: T1 交易日期（格式：YYYYMMDD）
- `--buy-price`: **增强** 买入价格类型（`open` 或 `close`，默认 `close`）
- `--sell-price`: **增强** 卖出价格类型（`open` 或 `close`，默认 `close`）

**价格口径支持（v0.5.0）：**
- 支持灵活的价格组合：T1开盘卖出+收盘买入、T1开盘买入+收盘卖出等
- open价格缺失时自动降级到close价格并打印warning
- 订单打印中明确显示使用的价格口径

**可交易性检查（v0.5.0）：**
- T1执行订单时自动检查涨跌停、停牌状态
- 停牌股票：不可买入或卖出，订单跳过
- 涨停股票：不可买入，订单跳过
- 跌停股票：不可卖出，订单延迟
- 打印详细的不可交易原因

**输出示例：**

```
========================================================================================================================
纸面交易执行明细 - 20260122
========================================================================================================================
股票代码       方向     目标权重     当前权重     股数      价格类型   参考价格     成交金额       佣金        印花税      滑点        总成本      原因            
------------------------------------------------------------------------------------------------------------------------
600000.SH    sell     0.0000     0.2000     2000     close      15.00      30000.00      5.85      15.00      15.00      35.85      退出持仓       
000001.SZ    buy      0.2000     0.0000     2000     close      10.00      20000.00      5.00      0.00       10.00      15.00      新建仓位       
000002.SZ    buy      0.2000     0.0000     1000     close      20.00      20000.00      5.00      0.00       10.00      15.00      新建仓位       
========================================================================================================================
执行完成: 2 买，1 卖
账户现金: 460065.85
持仓数量: 2
========================================================================================================================
```

## 数据结构

### 持久化目录

```
data/paper/
├── pending/              # 待执行目标权重
│   └── {YYYYMMDD}.parquet
├── state/                # 账户状态
│   └── account.json
├── trades/               # 交易记录
│   └── trades.parquet
└── nav/                  # 净值曲线
    └── nav.parquet
```

### 账户状态 (account.json)

```json
{
  "cash": 500000.0,
  "last_update": "20260122",
  "positions": {
    "000001.SZ": {
      "ts_code": "000001.SZ",
      "shares": 2000,
      "buy_price": 10.0,
      "buy_cost": 15.0,
      "buy_date": "20260122",
      "status": "持有",
      "notes": ""
    }
  }
}
```

### 交易记录 (trades.parquet)

| trade_date | ts_code   | action | shares | price | amount   | commission | stamp_tax | slippage | total_cost | reason   |
|------------|-----------|--------|--------|-------|----------|------------|-----------|----------|------------|----------|
| 20260122   | 000001.SZ | buy    | 2000   | 10.0  | 20000.00 | 5.00       | 0.00      | 10.00    | 15.00      | 新建仓位 |

### 净值曲线 (nav.parquet)

| trade_date | cash      | position_value | total_value | nav    |
|------------|-----------|----------------|-------------|--------|
| 20260122   | 460065.85 | 40000.00       | 500065.85   | 1.0001 |

## 工作流示例

### 连续多日运行

```bash
# Day 1: T0 (2026-01-21)
python scripts/paper_trade.py t0 --trade-date 20260121

# Day 2: T1 (2026-01-22)
python scripts/paper_trade.py t1 --trade-date 20260122

# Day 2: T0 (2026-01-22)
python scripts/paper_trade.py t0 --trade-date 20260122

# Day 3: T1 (2026-01-23)
python scripts/paper_trade.py t1 --trade-date 20260123

# ... 依此类推
```

### 使用 ML 模型

如果已经训练了 ML 模型（参见 `scripts/train_ml_model.py`）：

```bash
# T0: 使用模型版本 1 生成信号
python scripts/paper_trade.py t0 \
    --trade-date 20260121 \
    --model-version 1 \
    --top-n 10
```

## 股票池说明

### 主板股票池 (mainboard)

仅包含沪深主板股票：
- 过滤条件：`market == "主板"`
- 自动排除：科创板、创业板、北交所
- 自动排除：ST 股票
- 自动排除：上市不足 60 天的股票

### 全市场股票池 (all)

包含所有上市股票（仍会排除 ST 和上市不足 60 天的股票）。

## 成本计算

使用 `CostModel` 计算交易成本：

- **佣金**：0.01954%，最低 5 元（买卖双向）
- **印花税**：0.05%（仅卖出）
- **滑点**：0.05%（买卖双向）

可以通过修改 `src/lazybull/common/cost.py` 中的默认值来调整。

## 注意事项

1. **数据依赖**：确保已下载必要的基础数据（trade_cal、stock_basic）
2. **TuShare 限制**：免费版 TuShare 有访问频率限制，请合理安排拉取频率
3. **状态恢复**：每次运行 T1 后会自动保存状态，可随时中断和恢复
4. **买入价格**：开盘价买入适合盘前下单，收盘价买入更保守
5. **卖出价格**：当前固定为收盘价，后续可扩展

## 高级用法

### 查看持仓明细（v0.5.0新增）

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

**字段说明：**
- **股票代码**：持仓股票的代码
- **持仓股数**：当前持有股数
- **买入均价**：平均买入价格
- **买入成本**：总交易成本（佣金+滑点）
- **买入日期**：最后一次买入日期
- **持有天数**：从买入日期到当前日期的自然日天数
- **当前价格**：参考日期的收盘价
- **当前市值**：持仓股数 × 当前价格
- **浮动盈亏**：当前市值 - （买入成本 + 买入均价 × 股数）
- **收益率(%)**：浮动盈亏 / 成本 × 100
- **状态**：持仓状态（持有、延迟卖出等）

### 查看账户状态

```bash
# 使用 Python 读取
python -c "
import json
with open('data/paper/state/account.json') as f:
    state = json.load(f)
    print(f'现金: {state[\"cash\"]:,.2f}')
    print(f'持仓数: {len(state[\"positions\"])}')
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

### 问题：未找到待执行目标

**原因**：T0 未成功运行或日期不匹配

**解决**：
1. 检查 `data/paper/pending/` 目录下是否有对应日期的文件
2. 确认 T0 和 T1 的日期是连续的交易日

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

### 问题：股票池为空

**原因**：主板股票过滤过于严格或数据缺失

**解决**：
1. 检查 `stock_basic.parquet` 是否存在且包含 `market` 字段
2. 尝试使用 `--universe all` 查看全市场股票数量
3. 检查日线数据是否完整

## 扩展开发

### 自定义信号生成器

参考 `src/lazybull/signals/base.py`，实现自己的信号生成器：

```python
from src.lazybull.signals.base import Signal

class MySignal(Signal):
    def generate(self, date, universe, data):
        # 自定义信号逻辑
        signals = {}
        # ...
        return signals
```

### 自定义成本模型

参考 `src/lazybull/common/cost.py`，调整成本参数：

```python
from src.lazybull.common.cost import CostModel

custom_cost = CostModel(
    commission_rate=0.0003,  # 万3佣金
    min_commission=5.0,
    stamp_tax=0.001,         # 千1印花税
    slippage=0.001           # 0.1%滑点
)
```

## 相关文档

- [回测引擎说明](../docs/backtest_assumptions.md)
- [数据契约](../docs/data_contract.md)
- [ML 模型训练](../README.md#机器学习模型训练与回测)

## 支持

如有问题，请访问：
- [GitHub Issues](https://github.com/deltree-y/LazyBull/issues)
- [项目主页](https://github.com/deltree-y/LazyBull)
