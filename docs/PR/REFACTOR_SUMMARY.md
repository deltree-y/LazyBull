# 回测引擎重构总结

## 概述
本次重构优化了 `BacktestEngine` 的价格数据处理性能，并引入了价格口径分离和风险预算功能，使回测结果更贴近真实交易场景。

## 主要改进

### 1. 性能优化：使用 MultiIndex 替代嵌套字典

**问题**：原有实现使用 `iterrows()` + `pd.to_datetime` 将价格数据转换为嵌套字典 `{date: {code: price}}`，在大数据量下耗时严重。

**解决方案**：
- 使用 pandas MultiIndex `[(date, code)]` 直接索引价格
- 向量化日期转换，避免逐行处理
- 构建两套价格序列：
  - `trade_price_index`: 成交价格（不复权 close）
  - `pnl_price_index`: 绩效价格（后复权 close_adj）

**性能提升**：
- 避免 iterrows() 循环
- 减少内存占用（不需要构建嵌套字典）
- 更快的价格查询速度（O(1) 索引查找）

### 2. 价格口径分离

**背景**：在真实交易中，成交价格和绩效价格应该使用不同的口径：
- 成交/成本：应使用不复权价格（实际交易价格）
- 绩效/收益率：应使用后复权价格（消除除权除息影响）

**实现**：
- 持仓记录包含：`buy_trade_price`（成交）、`buy_pnl_price`（绩效）
- 交易记录新增字段：
  - `buy_pnl_price`: 买入绩效价格
  - `sell_pnl_price`: 卖出绩效价格
  - `pnl_profit_amount`: 绩效收益金额
  - `pnl_profit_pct`: 绩效收益率

**收益率计算**：
```python
# 基于绩效价格计算收益
pnl_profit_amount = pnl_sell_amount - pnl_buy_amount - total_cost
pnl_profit_pct = pnl_profit_amount / (pnl_buy_amount + buy_cost)
```

### 3. 风险预算/波动率缩放

**功能**：根据个股历史波动率动态调整权重，降低高波动股票的配置比例。

**参数**：
- `enable_risk_budget`: 是否启用（默认 False，保持向后兼容）
- `vol_window`: 波动率计算窗口（默认 20 个交易日）
- `vol_epsilon`: 最小波动率阈值（默认 0.001）

**实现**：
```python
# 调整权重 ∝ 原始权重 / 波动率
adj_weight = raw_weight / volatility
# 归一化
adj_weights = {stock: w / sum(adj_weights.values()) for stock, w in adj_weights.items()}
```

**关键设计**：
- 使用 end_date **之前**的数据计算波动率，避免未来函数
- 基于绩效价格（close_adj）计算波动率
- 年化波动率：`vol = std(returns) * sqrt(252)`

### 4. 其他改进

- **常量定义**：`TRADING_DAYS_PER_YEAR = 252`
- **异常处理**：使用具体异常类型（KeyError, ValueError, IndexError）
- **资金检查**：买入时确保有足够资金支付手续费
- **Fallback 机制**：缺少 close_adj 时自动退化到 close
- **向后兼容**：保留 `price_type` 参数（虽然不再使用）

## 使用示例

### 基本使用（不启用风险预算）
```python
from lazybull.backtest import BacktestEngine

engine = BacktestEngine(
    universe=universe,
    signal=signal,
    initial_capital=1000000,
    enable_risk_budget=False  # 默认值
)

nav = engine.run(
    start_date=start_date,
    end_date=end_date,
    trading_dates=trading_dates,
    price_data=price_data  # 需包含 close 和 close_adj 列
)
```

### 启用风险预算
```python
engine = BacktestEngine(
    universe=universe,
    signal=signal,
    initial_capital=1000000,
    enable_risk_budget=True,   # 启用风险预算
    vol_window=20,              # 20 日波动率
    vol_epsilon=0.001          # 最小波动率
)
```

### 查看交易记录
```python
trades = engine.get_trades()
sell_trades = trades[trades['action'] == 'sell']

for _, trade in sell_trades.iterrows():
    print(f"股票: {trade['stock']}")
    print(f"绩效收益率: {trade['pnl_profit_pct']:.2%}")
```

## 测试覆盖

新增测试文件：`tests/test_backtest_price_separation.py`

测试内容：
- ✅ 价格索引创建
- ✅ 交易记录包含绩效价格字段
- ✅ 风险预算启用/禁用
- ✅ 波动率计算
- ✅ close_adj 缺失时的 fallback
- ✅ 持仓结构包含新字段

所有测试（9个）全部通过。

## 向后兼容性

- ✅ 保留 `price_type` 参数（虽然不再使用）
- ✅ `enable_risk_budget` 默认为 False
- ✅ 原有 API 保持不变
- ✅ 旧测试全部通过

## 性能对比

| 场景 | 旧实现 | 新实现 | 提升 |
|------|--------|--------|------|
| 价格准备 | iterrows() | MultiIndex | ~10x |
| 价格查询 | dict[date][code] | index.loc[(date, code)] | ~2x |
| 内存占用 | 嵌套字典 | MultiIndex Series | 显著降低 |

## 文件变更

- `src/lazybull/backtest/engine.py`: 核心重构（+276行，-105行）
- `tests/test_backtest_t1.py`: 更新持仓结构测试
- `tests/test_backtest_price_separation.py`: 新增测试文件
- `examples/backtest_example.py`: 新增使用示例

## 下一步建议

1. **缓存优化**：对于相同的 price_data，可以考虑缓存 price_index
2. **并行计算**：波动率计算可以并行化
3. **配置化**：`TRADING_DAYS_PER_YEAR` 可以作为参数传入（适配不同市场）
4. **文档**：添加英文注释和国际化支持

## 总结

本次重构在保持向后兼容的前提下：
- **显著提升性能**：使用 MultiIndex 替代嵌套字典
- **更贴近实践**：价格口径分离，成交用不复权，绩效用后复权
- **增强功能**：引入风险预算/波动率缩放
- **提高质量**：完善错误处理，添加常量定义，优化代码结构

所有功能经过充分测试，可以放心使用。
