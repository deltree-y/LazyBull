# 纸面交易「持仓情况」表单与汇总统计改进

## 版本信息
- **版本号**: 0.3.10
- **提交日期**: 2026-02-09

## 改进动机

当前纸面交易的持仓情况显示存在以下问题：
1. 股票代码列仅显示代码，不显示股票名称，可读性较差
2. "买入成本"列对用户价值不大（主要用于内部计算），增加表格冗余
3. "当前价格"列在"买入均价"之后，不符合从价格到成本的阅读习惯
4. 汇总统计缺少"总盈亏百分比"和"年化收益率"两个重要指标

## 改动内容

### 1. 持仓表单列调整

#### 1.1 股票代码显示增加名称
- **改动前**: 仅显示 `ts_code`（如 `000001.SZ`）
- **改动后**: 显示 `ts_code(股票名称)`（如 `000001.SZ(平安银行)`）
- 股票名称从 `daily_data` 的 `name` 列获取（已在数据清洗流程中合并）
- 若名称未获取到，显示 `ts_code(na)`

#### 1.2 删除"买入成本"列
- "买入成本"列从显示中移除，但数据模型中保留用于内部计算
- 减少表格宽度，提高可读性

#### 1.3 调整列顺序
- **改动前**: 股票代码、股数、买入均价、买入成本、买入日期、持有天数、当前价格、当前市值、浮盈、收益率(%)、状态
- **改动后**: 股票代码、股数、**当前价格**、买入均价、买入日期、持有天数、当前市值、浮盈、收益率(%)、状态
- 将"当前价格"移至"买入均价"前，符合"当前→历史"的阅读习惯

### 2. 汇总统计新增指标

#### 2.1 总盈亏百分比
- **公式**: `(总资产 - 初始资金) / 初始资金 × 100`
- 显示账户整体盈亏情况

#### 2.2 年化收益率
- **公式**: `(总资产 / 初始资金) ^ (365 / 持有天数) - 1`
- 持有天数 = 账户起始日期到当前日期的自然日天数（至少为1天）
- 空仓时年化收益率为 0

### 3. 数据模型更新

#### 3.1 AccountState 新增字段
- `initial_capital`: 初始资金（用于计算年化收益率）
- `start_date`: 账户起始日期 YYYYMMDD（用于计算年化收益率）

#### 3.2 持久化逻辑
- 新账户创建时自动记录 `initial_capital` 和 `start_date`
- `start_date` 默认为创建账户的当天日期
- 保存和加载账户状态时包含这两个字段
- 旧账户加载时：
  - 如果缺少 `initial_capital`，使用构造函数的 `initial_capital` 参数
  - 如果缺少 `start_date`，使用 `last_update` 作为近似值

## 技术实现

### 修改的文件

1. **src/lazybull/paper/models.py**
   - `AccountState` 增加 `initial_capital` 和 `start_date` 字段

2. **src/lazybull/paper/account.py**
   - 初始化账户时设置 `initial_capital` 和 `start_date`
   - 兼容旧账户状态的加载

3. **src/lazybull/paper/storage.py**
   - `save_account_state()` 保存新字段
   - `load_account_state()` 加载新字段（支持旧版本兼容）

4. **src/lazybull/paper/broker.py**
   - `get_positions_detail()` 增加 `stock_names` 参数，支持股票名称显示
   - `print_positions_summary()` 更新表头、列顺序、汇总统计

5. **scripts/paper_trade.py**
   - `print_positions()` 从 `daily_data` 提取股票名称，传递给 broker

6. **tests/test_paper_trading.py**
   - 新增测试：股票名称显示、列顺序、年化收益率计算

7. **pyproject.toml**
   - 版本号从 0.3.9 升级到 0.3.10

## 如何验证

### 1. 单元测试
```bash
pytest tests/test_paper_trading.py::test_broker_get_positions_detail_with_stock_names -v
pytest tests/test_paper_trading.py::test_positions_detail_column_order -v
pytest tests/test_paper_trading.py::test_account_state_with_initial_capital_and_start_date -v
pytest tests/test_paper_trading.py::test_annual_return_calculation -v
pytest tests/test_paper_trading.py::test_empty_positions_annual_return -v
```

### 2. 命令行验证
```bash
# 查看持仓情况
python scripts/paper_trade.py positions --trade-date YYYYMMDD
```

**预期输出**:
- 股票代码列显示为 `ts_code(股票名称)` 或 `ts_code(na)`
- 不显示"买入成本"列
- "当前价格"在"买入均价"之前
- 汇总统计包含"总盈亏百分比"和"年化收益率"

### 3. 验证示例

假设账户有以下持仓：
- 000001.SZ(平安银行), 1000股, 买入均价10元, 当前价格12元

期望输出：
```
股票代码               股数      当前价格   买入均价   买入日期      持有天数   当前市值     浮盈         收益率(%)   状态
000001.SZ(平安银行)   1000    12.00    10.00    20260101    31       12000.00   1985.00    19.70     持有
合计                  1000                                            12000.00   1985.00    19.70
================================================================================
账户现金: 89,985.00
持仓市值: 12,000.00
总资产: 101,985.00
总盈亏百分比: 1.99%
年化收益率: 25.23%
================================================================================
```

## 影响范围

### 数据兼容性
- **新账户**: 自动记录 `initial_capital` 和 `start_date`
- **旧账户**: 
  - 加载时自动兼容，缺失字段使用默认值
  - 首次保存后自动升级到新格式
  - 不需要手动迁移数据

### API 变更
- `PaperBroker.get_positions_detail()` 增加可选参数 `stock_names`（向后兼容）
- `PaperBroker.print_positions_summary()` 增加可选参数 `stock_names`（向后兼容）

### 显示变更
- 持仓表格宽度略有变化（股票代码列加宽）
- 总分隔线宽度从140字符调整为120字符

## 注意事项

1. **股票名称缺失**: 如果 `daily_data` 中没有 `name` 列或某些股票没有名称，会显示为 `(na)`
2. **年化收益率精度**: 基于自然日计算，不考虑实际交易日
3. **旧账户起始日期**: 旧账户的 `start_date` 使用 `last_update` 作为近似值，可能不够精确
4. **最小持有天数**: 年化收益率计算中，持有天数至少为1天，避免除零错误

## 后续优化建议

1. 考虑在配置文件中允许用户自定义 `start_date`（用于手动校准）
2. 支持多个基准日期的年化收益率计算（如：30天、90天、365天）
3. 增加更多风险指标（如：最大回撤、夏普比率等）
