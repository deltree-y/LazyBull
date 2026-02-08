# PR Summary: T0 纸面交易等权策略一手可买约束

## 问题描述

当前纸面交易 T0 会生成 top_n 个目标并保存到 pending，但在 T1 执行时，部分股票因按资金*权重折算后不足100股（1手）导致 target_shares=0 而被跳过，最终实际可买股票数少于 top_n。

## 解决方案

在等权模式（`weight_method="equal"`）下，在 T0 生成信号阶段就预先过滤不足1手的股票，并从排序候选中顺延补足，确保保存到 pending 的目标都是可买至少1手的股票。

## 核心变更

### 1. 新增方法 `_generate_equal_weight_with_lot_constraint()`

**位置**：`src/lazybull/paper/runner.py`

**功能**：
- 调用 `MLSignal.generate_ranked()` 获取完整排序候选列表
- 根据 `buy_price_type` 加载对应价格（open/close）
- 计算等权分配金额：`total_capital / top_n`
- 检查每只股票可买股数：`int(金额 / 价格 / 100) * 100`
- 不足100股的跳过并从候选顺延，直到凑足 top_n 或候选耗尽
- 输出详细日志：原始候选数、跳过数、最终数、示例

**代码示例**：
```python
# 计算每只股票的等权分配金额
total_capital = self.account.initial_capital
equal_weight_value = total_capital / top_n

# 从排序候选中筛选可买至少1手的股票
for ts_code, score in ranked_candidates:
    if len(selected_stocks) >= top_n:
        break
    
    price = price_map.get(ts_code)
    if price is None or price <= 0:
        skipped_stocks.append((ts_code, "无价格数据"))
        continue
    
    affordable_shares = int(equal_weight_value / price / SHARE_LOT_SIZE) * SHARE_LOT_SIZE
    
    if affordable_shares < SHARE_LOT_SIZE:
        skipped_stocks.append((ts_code, f"不足1手(价格={price:.2f})"))
        continue
    
    selected_stocks.append(ts_code)
```

### 2. 修改 `_generate_signals()` 方法

**变更**：
- 新增 `buy_price_type` 参数
- 等权策略使用新的约束方法
- Score 加权策略保持原有逻辑

```python
if self.weight_method == "equal" and isinstance(self.signal, MLSignal):
    signal_dict = self._generate_equal_weight_with_lot_constraint(
        date_ts, stocks, signal_data, daily_data, top_n, buy_price_type
    )
else:
    signal_dict = self.signal.generate(
        date_ts, stocks, {'features': signal_data}
    )
```

### 3. 更新 `run_t0()` 调用

传递 `buy_price_type` 参数到 `_generate_signals()`：

```python
targets = self._generate_signals(
    corrected_date,
    universe_type=universe_type,
    top_n=top_n,
    model_version=model_version,
    buy_price_type=buy_price_type  # 新增
)
```

## 测试覆盖

新增测试文件：`tests/test_equal_weight_lot_constraint.py`

**6个测试用例**：

1. ✅ `test_equal_weight_lot_constraint_basic` - 基本功能，跳过高价股并顺延
2. ✅ `test_equal_weight_lot_constraint_insufficient_candidates` - 候选不足 top_n
3. ✅ `test_equal_weight_lot_constraint_all_too_expensive` - 所有股票都太贵
4. ✅ `test_equal_weight_lot_constraint_with_open_price` - 使用开盘价
5. ✅ `test_equal_weight_lot_constraint_missing_price_data` - 缺失价格数据
6. ✅ `test_equal_weight_lot_constraint_boundary_case` - 边界情况（刚好1手）

**测试结果**：所有测试通过 ✅

## 日志示例

### 正常场景
```
等权+一手约束: 原始排序候选数 50
等权+一手约束: 最终目标数 5, 跳过 2 只 (原始候选 50)
  跳过示例: 000001.SZ - 不足1手(价格=1200.00, 可买=27股)
  跳过示例: 600688.SH - 不足1手(价格=980.50, 可买=33股)
```

### 候选不足场景
```
等权+一手约束: 原始排序候选数 8
等权+一手约束: 最终目标数 3, 跳过 5 只 (原始候选 8)
等权+一手约束: 候选不足，目标 5 只，实际仅 3 只可选
```

## 文档更新

### 1. CHANGELOG.md（新建）
- 记录 v0.3.5 版本变更

### 2. docs/paper_trading_guide.md
- 核心特性列表中新增此功能
- 新增专门章节详细说明工作原理、示例、日志输出

### 3. docs/paper_vs_backtest_alignment.md（新建）
- 对比纸面交易和回测引擎的行为差异
- 说明纸面交易更严格（T0提前过滤 vs T+1事后计算）
- 建议保持现状

## 版本号更新

`pyproject.toml`：`0.3.4` → `0.3.5`

## 与回测引擎对齐

| 维度 | 纸面交易 (新) | 回测引擎 |
|------|-------------|---------|
| 检查时机 | T0 信号生成 | T+1 买入执行 |
| 过滤条件 | 一手可买 + 可交易性 | 仅可交易性 |
| 0股处理 | 提前跳过 | 尝试买入但不成交 |
| 顺延补足 | ✓ | ✓ |

**结论**：纸面交易更严格，更接近实盘场景，建议保持现状

## 验收标准达成情况

| 标准 | 状态 |
|------|------|
| pending 目标不包含无法买入至少100股的股票 | ✅ |
| 最终目标数量 <= top_n | ✅ |
| 候选耗尽时有清晰日志提示 | ✅ |
| 测试覆盖"价格过高顺延补足"行为 | ✅ |
| 现有功能无回退 | ✅ |
| 代码审查通过 | ✅ |
| 安全检查通过 | ✅ |

## 代码审查结果

- **审查意见**：1个提示性建议（SHARE_LOT_SIZE 常量已在模块级定义）
- **安全检查**：0个告警 ✅

## 影响范围

### 受影响的功能
- ✅ 纸面交易 T0 等权策略信号生成
- ✅ Score 加权策略不受影响

### 不受影响的功能
- ✅ T1 执行逻辑
- ✅ 持仓管理
- ✅ 止损功能
- ✅ 延迟卖出
- ✅ 回测引擎

## 使用示例

### 场景：初始资金 100,000 元，top_n=3

**候选列表（按ML分数排序）**：
1. 000001.SZ - 价格 1200 元 → 可买 27 股（不足1手） ❌
2. 000002.SZ - 价格 50 元 → 可买 666 股 ✅
3. 000003.SZ - 价格 30 元 → 可买 1111 股 ✅
4. 600000.SH - 价格 25 元 → 可买 1333 股 ✅

**最终结果**：
- 选中：000002.SZ, 000003.SZ, 600000.SH
- 等权分配：各 1/3 (33.33%)

## 后续建议

1. 考虑在回测引擎中也加入一手可买约束（可选，确保完全一致）
2. 监控实际运行效果，收集反馈
3. 未来可扩展到 score 加权策略（如需要）

## 相关文件

**核心代码**：
- `src/lazybull/paper/runner.py`（+126行，修改3处）

**测试**：
- `tests/test_equal_weight_lot_constraint.py`（新增，272行）

**文档**：
- `CHANGELOG.md`（新增）
- `docs/paper_trading_guide.md`（+66行）
- `docs/paper_vs_backtest_alignment.md`（新增）

**配置**：
- `pyproject.toml`（版本号更新）

## 提交记录

1. `e590753` - Implement equal weight minimum lot constraint with backfill logic
2. `9e0b0e5` - Add documentation for equal weight minimum lot constraint feature
3. `e7fb9e2` - Add paper trading vs backtest alignment documentation
