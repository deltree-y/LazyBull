# 纸面交易持仓表格与汇总统计改进 - 实现总结

## PR 概述

本 PR 实现了纸面交易 `positions --trade-date` 命令的持仓表格显示改进和汇总统计增强。所有代码注释、日志输出和文档均使用中文。

## 主要改进

### 1. 股票代码显示增强
- **改进前**: 只显示 `ts_code`（如 `000001.SZ`）
- **改进后**: 显示 `ts_code(股票名称)`（如 `000001.SZ(平安银行)`）
- **缺失处理**: 名称未获取到时显示 `ts_code(na)`

### 2. 持仓表列调整
**删除列**: 
- ~~买入成本~~（内部仍用于计算，但不再展示）

**列顺序优化**:
```
旧顺序: 股票代码 | 股数 | 买入均价 | 买入成本 | 买入日期 | 持有天数 | 当前价格 | 当前市值 | 浮盈 | 收益率(%) | 状态
新顺序: 股票代码(名称) | 股数 | 当前价格 | 买入均价 | 买入日期 | 持有天数 | 当前市值 | 浮盈 | 收益率(%) | 状态
```

### 3. 汇总统计新增指标

#### 总盈亏百分比
- 计算公式: `总浮盈 / 总成本 * 100`
- 显示格式: `总盈亏百分比: X.XX%`

#### 年化收益率
- 起始资金: 从配置读取 `initial_capital`（持久化配置优先，否则使用账户属性）
- 起始日期:
  1. 优先从配置 `config.json` 读取 `account_start_date`
  2. 若不存在，从 NAV 记录获取最早交易日期
  3. 若仍无法获取，显示"无法计算（缺少账户起始日期）"
- 计算公式: `(当前总资产 / 初始资金) ** (365 / 持有天数) - 1`
- 显示格式: `年化收益率: X.XX%`

## 技术实现细节

### 代码修改

#### src/lazybull/paper/broker.py
1. `get_positions_detail()`: 新增 `stock_names` 参数，生成包含名称的股票代码
2. `print_positions_summary()`: 
   - 新增 `stock_names` 参数
   - 调整表头和列顺序
   - 新增总盈亏百分比和年化收益率输出
3. `_calculate_annualized_return()`: 新增私有方法，实现年化收益率计算
4. 更新 `positions_table_widths`: 从 11 列改为 10 列（删除买入成本）

#### scripts/paper_trade.py
在 `print_positions()` 函数中:
1. 从 `daily_data` 提取 `name` 列（如果存在）
2. 构建 `stock_names` 字典
3. 传递给 `broker.print_positions_summary()`

### 测试覆盖

新增 6 个测试用例，100% 通过:
1. `test_broker_positions_with_stock_names` - 验证股票名称正确显示
2. `test_broker_positions_without_stock_names` - 验证缺失名称时显示 `na`
3. `test_broker_positions_column_order` - 验证列顺序（当前价格在买入均价前）
4. `test_broker_calculate_annualized_return_with_nav` - 验证通过 NAV 记录计算年化收益率
5. `test_broker_calculate_annualized_return_without_start_date` - 验证无起始日期返回 None
6. `test_broker_calculate_annualized_return_zero_days` - 验证零天返回 0

更新 1 个已有测试:
- `test_broker_get_positions_detail` - 更新断言以匹配新的股票代码格式

## 示例输出

运行演示脚本 `examples/demo_positions_improvements.py` 可查看效果:

```
股票代码           股数     当前价格   买入均价   买入日期     持有天数 当前市值     浮盈         收益率(%)    状态
----------------------------------------------------------------------------
000001.SZ(平安银行) 10000    12.50      10.50      20260115     16       125000.00    19895.00     18.93        持有
600000.SH(浦发银行) 5000     8.80       8.30       20260120     11       44000.00     2458.50      5.92         持有
601398.SH(工商银行) 20000    6.20       5.50       20260125     6        124000.00    13890.00     12.61        持有
----------------------------------------------------------------------------
合计                 35000                                                293000.00    36243.50     14.12
============================================================================
账户现金: 243,243.50
持仓市值: 293,000.00
总资产: 536,243.50
总盈亏百分比: 14.12%
年化收益率: 134.30%
============================================================================
```

## 使用说明

### 基本使用
```bash
python scripts/paper_trade.py positions --trade-date YYYYMMDD
```

### 年化收益率配置

**方式 1: 自动从 NAV 记录推断**
- 系统会自动使用 NAV 记录中最早的交易日期作为起始日期

**方式 2: 手动配置起始日期**
编辑 `data/paper/config.json`:
```json
{
  "initial_capital": 500000,
  "account_start_date": "20260101",
  ...
}
```

## 影响范围

✅ **改动范围**:
- 持仓表格显示逻辑
- 汇总统计输出
- 测试用例

❌ **不影响**:
- 订单生成
- 交易执行
- 持仓数据存储
- 回测功能
- 其他命令

## 版本更新

- 版本号: `0.3.9` → `0.3.10`
- 变更类型: 小版本（功能增强）

## 文件清单

### 修改文件
- `src/lazybull/paper/broker.py` - 核心功能实现
- `scripts/paper_trade.py` - CLI 集成
- `tests/test_paper_trading.py` - 测试用例
- `pyproject.toml` - 版本号更新

### 新增文件
- `docs/PR/positions_table_improvements.md` - PR 说明文档
- `examples/demo_positions_improvements.py` - 演示脚本

## 验证结果

✅ 所有新增测试通过（6/6）
✅ 所有更新测试通过（1/1）
✅ 演示脚本正常运行
✅ 代码风格一致
✅ 中文注释和日志完整

## 后续建议

1. 考虑在 `config` 命令中增加 `--account-start-date` 参数，便于首次配置
2. 对于老账户，建议在文档中说明如何手动设置准确的起始日期
3. 可以考虑在账户首次创建时自动记录 `account_start_date`

## 相关链接

- PR 详细说明: `docs/PR/positions_table_improvements.md`
- 演示脚本: `examples/demo_positions_improvements.py`
- 主要变更: `src/lazybull/paper/broker.py`
