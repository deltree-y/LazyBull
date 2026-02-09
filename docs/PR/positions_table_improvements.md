# 纸面交易持仓表格与汇总统计改进

## 动机

当前纸面交易的 `positions --trade-date` 命令打印持仓表格时存在以下不足：

1. 股票代码只显示 ts_code，缺少股票名称，可读性不强
2. 表格中包含"买入成本"列，但这主要是内部计算用的手续费，对用户查看意义不大
3. "当前价格"列位于"买入均价"之后，不够直观（通常希望先看当前价格再看买入均价）
4. 汇总统计缺少关键指标：总盈亏百分比和年化收益率

## 改动点

### 1. 股票代码显示增强

**文件：** `src/lazybull/paper/broker.py`

- 修改 `get_positions_detail()` 方法，新增 `stock_names` 参数
- 股票代码显示格式改为：`ts_code(股票名称)`
- 如果股票名称未获取到，显示：`ts_code(na)`

**文件：** `scripts/paper_trade.py`

- 在 `print_positions()` 函数中，从 `daily_data` 提取股票名称（如果存在 `name` 列）
- 构建 `stock_names` 字典并传递给 `broker.print_positions_summary()`

### 2. 持仓表列调整

**删除列：**
- ~~买入成本~~ （内部仍用于计算总成本，但不再展示）

**列顺序调整：**

调整前：
```
股票代码 | 股数 | 买入均价 | 买入成本 | 买入日期 | 持有天数 | 当前价格 | 当前市值 | 浮盈 | 收益率(%) | 状态
```

调整后：
```
股票代码(名称) | 股数 | 当前价格 | 买入均价 | 买入日期 | 持有天数 | 当前市值 | 浮盈 | 收益率(%) | 状态
```

**更改：**
- `src/lazybull/paper/broker.py` 中更新 `positions_table_widths` 从 11 列改为 10 列
- 股票代码列宽度从 12 增加到 18（以容纳名称）

### 3. 汇总统计新增指标

**文件：** `src/lazybull/paper/broker.py`

在 `print_positions_summary()` 方法末尾新增以下输出：

#### 总盈亏百分比
- 计算公式：`总浮盈 / 总成本 * 100`
- 显示格式：`总盈亏百分比: X.XX%`

#### 年化收益率
- 新增 `_calculate_annualized_return()` 私有方法
- 起始资金：从配置读取 `initial_capital`（优先使用持久化配置，如果不存在则使用账户的 `initial_capital` 属性）
- 起始日期：
  - 优先从配置 `config.json` 读取 `account_start_date` 字段（如果存在）
  - 如果配置中没有该字段，则从 NAV 记录（`nav.parquet`）获取最早的交易日期
  - 如果两者都不存在，显示"无法计算（缺少账户起始日期）"
- 计算公式：`(当前总资产 / 初始资金) ** (365 / 持有天数) - 1`
- 显示格式：`年化收益率: X.XX%`

**注意：**
- 本 PR 不新增 `account_start_date` 字段到配置，仅实现从 NAV 记录推断的逻辑
- 如果用户需要更精确的年化收益率计算，可以在配置中手动添加 `account_start_date` 字段

### 4. 测试更新

**文件：** `tests/test_paper_trading.py`

新增以下测试用例：

1. `test_broker_positions_with_stock_names()` - 测试股票名称正确显示
2. `test_broker_positions_without_stock_names()` - 测试缺少名称时显示 `na`
3. `test_broker_positions_column_order()` - 测试列顺序（当前价格在买入均价前）
4. `test_broker_calculate_annualized_return_with_nav()` - 测试通过 NAV 记录计算年化收益率
5. `test_broker_calculate_annualized_return_without_start_date()` - 测试无起始日期时返回 None
6. `test_broker_calculate_annualized_return_zero_days()` - 测试零天时返回 0

### 5. 版本更新

**文件：** `pyproject.toml`

- 版本号从 `0.3.9` 更新到 `0.3.10`

## 如何验证

### 1. 运行单元测试

```bash
pytest tests/test_paper_trading.py::test_broker_positions_with_stock_names -v
pytest tests/test_paper_trading.py::test_broker_positions_without_stock_names -v
pytest tests/test_paper_trading.py::test_broker_positions_column_order -v
pytest tests/test_paper_trading.py::test_broker_calculate_annualized_return_with_nav -v
```

### 2. 手工验证持仓表格

执行以下命令查看持仓情况（需要有实际持仓数据）：

```bash
python scripts/paper_trade.py positions --trade-date YYYYMMDD
```

预期输出示例：

```
================================================================================
[YYYYMMDD]持仓情况
================================================================================
股票代码           股数     当前价格   买入均价   买入日期     持有天数 当前市值     浮盈         收益率(%)    状态
--------------------------------------------------------------------------------
000001.SZ(平安银行)  1000     12.50      10.00      20260115     30       12500.00     2485.00      24.85        持有
600000.SH(浦发银行)  500      8.30       8.00       20260120     25       4150.00      135.00       3.36         持有
--------------------------------------------------------------------------------
合计               1500                                                     16650.00     2620.00      18.69
================================================================================
账户现金: 83,350.00
持仓市值: 16,650.00
总资产: 100,000.00
总盈亏百分比: 18.69%
年化收益率: 256.34%
================================================================================
```

### 3. 验证年化收益率计算逻辑

**场景 1：有 NAV 记录**

如果纸面账户已经运行一段时间并有 NAV 记录，年化收益率会自动从最早的 NAV 记录日期开始计算。

**场景 2：手动设置起始日期**

可以在配置文件 `data/paper/config.json` 中手动添加 `account_start_date` 字段：

```json
{
  "initial_capital": 500000,
  "account_start_date": "20260101",
  ...
}
```

**场景 3：新账户无历史数据**

如果是全新账户且没有 NAV 记录，年化收益率会显示"无法计算（缺少账户起始日期）"。

## 影响范围

本 PR 仅修改持仓表格的显示逻辑和汇总统计，不影响：
- 订单生成逻辑
- 交易执行逻辑
- 持仓数据存储格式
- 回测功能
- 其他命令（如 `run`, `backtest` 等）

## 后续建议

1. 如果用户希望更精确控制账户起始日期，可以在首次配置时自动记录 `account_start_date`
2. 可以考虑在 `config` 命令中增加 `--account-start-date` 参数
3. 对于已运行一段时间的老账户，建议用户手动在配置中添加准确的起始日期

## 相关文件

- `src/lazybull/paper/broker.py` - 核心逻辑修改
- `scripts/paper_trade.py` - CLI 命令修改
- `tests/test_paper_trading.py` - 测试用例新增
- `pyproject.toml` - 版本号更新
- `docs/PR/positions_table_improvements.md` - 本文档
