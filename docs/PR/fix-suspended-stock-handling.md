# PR: 修复停牌股票处理逻辑（纸面交易 + 回测）

## 版本
v0.3.14

## 问题背景

### 1. 止损检查问题
持有过程中若股票停牌，止损检查不应将其纳入止损考量。当前实现存在以下问题：

**纸面交易 (`scripts/paper_trade.py::_check_stop_loss()`)**
- 仅使用 `close` 构建价格字典，未检查 `is_suspended` 字段
- 若停牌导致 close 缺失或为 0，可能触发错误的回撤止损
- 停牌期间价格异常可能被误判为大幅下跌

**回测 (`src/lazybull/backtest/engine.py::_check_stop_loss()`)**
- 未检查股票是否停牌即进行止损判断
- 停牌导致的价格 None/0 可能触发不正确的止损逻辑

### 2. 调仓日买卖问题
调仓日买卖时，停牌股票应被适当处理：

**卖出侧问题**
- 需要卖出但因停牌/无价格数据导致 `sell_prices` 缺失
- 当前实现直接"跳过卖出"，不进入延迟卖出队列
- 造成应卖未卖且不会重试的情况

**策略选择**
- 用户选择"策略2"：对于"无卖出价格数据"的情况
- 应优先判断 `tradability` 的 `is_suspended==1` 则原因写"停牌"
- 否则写"无价格数据"

## 原因分析

### 根本原因
1. **止损检查缺少停牌过滤**：止损逻辑假设所有持仓都有有效价格，未考虑停牌情况
2. **缺失价格处理不一致**：纸面交易中，无价格数据时直接跳过而非加入延迟队列
3. **停牌与无数据未区分**：未按策略2要求区分停牌和无价格数据两种情况

### 影响范围
- **纸面交易**：可能在停牌期间误触发止损，或应卖未卖导致策略偏离
- **回测**：回测结果不准确，与实盘表现不一致
- **用户体验**：实盘操作者无法清晰了解停牌导致的延迟

## 修复方案

### A. 纸面交易修复

#### A1. 止损检查（`scripts/paper_trade.py`）

**修改内容**：
```python
# 构建价格字典、跌停信息和停牌信息
suspended_info = {}
for _, row in daily_data.iterrows():
    ts_code = row['ts_code']
    prices[ts_code] = row.get('close', 0.0)
    limit_down_info[ts_code] = row.get('is_limit_down', 0) == 1
    suspended_info[ts_code] = row.get('is_suspended', 0) == 1  # 新增

# 检查每个持仓
for ts_code, pos in positions.items():
    # 检查是否停牌
    is_suspended = suspended_info.get(ts_code, False)
    if is_suspended:
        logger.info(f"股票 {ts_code} 停牌，跳过止损检查")  # 新增
        continue
```

**效果**：
- 停牌股票不会进入止损检查逻辑
- 避免停牌期间价格异常触发误报
- 清晰的中文日志提示

#### A2. 调仓/指令卖出（`src/lazybull/paper/broker.py`）

**`generate_orders()` 修改**：
```python
if ts_code not in sell_prices:
    # 判断原因（策略2：停牌优先）
    reason_suffix = ""
    if ts_code in tradability and tradability[ts_code].get('is_suspended', 0) == 1:
        reason_suffix = "（停牌）"
        logger.warning(f"股票 {ts_code} 停牌，无法卖出，加入延迟卖出队列")
    else:
        reason_suffix = "（无价格数据）"
        logger.warning(f"股票 {ts_code} 无卖出价格数据，加入延迟卖出队列")
    
    # 加入延迟卖出队列（使用当前持仓股数）
    pending_sell = PendingSell(
        ts_code=ts_code,
        shares=pos.shares,
        target_weight=target_weight,
        reason=f"{sell_reason}{reason_suffix}",
        create_date=trade_date,
        attempts=0
    )
    self.pending_sells.append(pending_sell)
    continue
```

**`execute_instructions()` 同样修改**：
- 检查 `ts_code not in sell_prices` 时不再跳过
- 根据停牌状态决定原因文案
- 创建 `PendingSell` 并加入队列

**效果**：
- 应卖未卖的情况不再发生
- 延迟卖出会在后续交易日重试
- 原因清晰（停牌 vs 无价格数据）

### B. 回测修复

#### B1. 止损检查（`src/lazybull/backtest/engine.py`）

**修改内容**：
```python
# 获取当日行情数据判断停牌状态
trade_date_str = to_trade_date_str(date)
date_quote = self.price_data_cache[self.price_data_cache['trade_date'] == trade_date_str]

# 检查是否停牌
is_suspended = False
if not date_quote.empty:
    stock_quote = date_quote[date_quote['ts_code'] == stock]
    if not stock_quote.empty and 'is_suspended' in stock_quote.columns:
        is_suspended = bool(stock_quote['is_suspended'].iloc[0] == 1)

# 停牌时跳过止损检查
if is_suspended:
    if self.verbose:
        logger.info(f"股票 {stock} 停牌，跳过止损检查 ({date.date()})")
    continue
```

**效果**：
- 回测中停牌股票不会触发止损
- 与纸面交易行为一致

#### B2. 卖出延迟机制

**现状验证**：
回测引擎已有完善的延迟卖出机制：
- `_sell_stock()` → `_sell_stock_with_status_check()` 
- 使用 `is_tradeable()` 检查停牌状态
- 停牌时自动加入 `pending_order_manager` 延迟队列
- 后续交易日自动重试

**结论**：回测卖出延迟机制已符合要求，无需额外修改。

## Paper/Backtest 对齐点

| 场景 | 纸面交易行为 | 回测行为 | 对齐状态 |
|------|------------|---------|---------|
| 持仓停牌时止损检查 | 跳过止损检查 | 跳过止损检查 | ✅ 已对齐 |
| 调仓卖出遇停牌 | 进入 pending_sells 队列 | 进入 pending_order_manager 队列 | ✅ 已对齐 |
| 卖出无价格数据 | 进入 pending_sells 队列（注明原因） | 进入 pending_order_manager 队列 | ✅ 已对齐 |
| 延迟卖出重试 | 使用 `retry_pending_sells()` | 使用 `pending_order_manager` | ✅ 已对齐 |

## 验证步骤

### 1. 代码审查
- ✅ 检查 `scripts/paper_trade.py::_check_stop_loss()` 是否正确跳过停牌股票
- ✅ 检查 `src/lazybull/paper/broker.py` 是否正确处理无价格/停牌情况
- ✅ 检查 `src/lazybull/backtest/engine.py` 是否正确跳过停牌股票止损

### 2. 单元测试
新增测试文件 `tests/test_suspended_stock_handling.py`：

**纸面交易测试**
- `test_suspended_stock_skipped_in_stop_loss_check`: 验证停牌股票止损检查被跳过
- `test_sell_suspended_stock_added_to_pending_sells`: 验证停牌股票加入延迟队列，原因包含"停牌"
- `test_sell_no_price_data_added_to_pending_sells`: 验证无价格数据加入延迟队列，原因包含"无价格数据"
- `test_execute_instructions_suspended_stock_added_to_pending_sells`: 验证指令执行时停牌处理

**回测测试**
- `test_suspended_stock_skipped_in_backtest_stop_loss`: 验证回测止损检查跳过停牌股票
- `test_suspended_stock_sell_deferred_in_backtest`: 验证回测卖出延迟机制

### 3. 集成测试建议
1. 使用历史数据中有停牌记录的股票进行回测
2. 观察日志输出，确认停牌股票被正确标识和跳过
3. 检查 pending_sells 队列是否正确记录延迟卖出
4. 验证延迟卖出在复牌后是否成功执行

### 4. 日志验证
运行纸面交易时，应看到类似日志：
```
[INFO] 股票 000001.SZ 停牌，跳过止损检查
[WARNING] 股票 000002.SZ 停牌，无法卖出，加入延迟卖出队列
[WARNING] 股票 000003.SZ 无卖出价格数据，加入延迟卖出队列
```

## 技术细节

### 复用现有能力
1. **trade_status 模块**：使用 `is_suspended()`, `is_tradeable()` 等现有函数
2. **pending 队列**：复用现有 `PendingSell` 模型和 `retry_pending_sells()` 逻辑
3. **tradability 信息**：复用 `_load_tradability_info()` 加载机制

### 不引入历史兼容
- 不添加配置开关（直接修复，成为默认行为）
- 不保留旧逻辑分支
- 仅做必要的最小改动

### 版本递增
- 从 `0.3.13` 递增至 `0.3.14`
- 属于 bugfix 类型的小版本更新

## 影响评估

### 正面影响
1. **准确性提升**：止损逻辑更准确，避免停牌误报
2. **一致性改善**：纸面交易与回测行为对齐
3. **可靠性增强**：应卖未卖的情况得到修复
4. **可观测性提升**：清晰的中文日志便于实盘操作

### 潜在风险
1. **行为变化**：之前可能触发的止损现在不会触发（停牌期间）
   - **评估**：这是正确行为，风险可控
2. **队列增长**：pending_sells 可能有更多记录
   - **评估**：符合预期，会在复牌后自动处理

### 兼容性
- **向前兼容**：新版本可以读取旧版本数据
- **向后兼容**：不支持（旧版本不能正确处理新行为）
- **建议**：升级后不要降级回旧版本

## 后续建议

1. **监控**：上线后关注 pending_sells 队列大小和重试成功率
2. **日志分析**：定期分析停牌导致的延迟卖出情况
3. **文档更新**：更新用户手册，说明停牌处理机制
4. **配置优化**：考虑添加停牌最大等待天数配置（可选）

## 参考文档
- 中国A股交易规则：停牌股票不可交易
- `src/lazybull/common/trade_status.py`：交易状态检查模块
- `src/lazybull/paper/models.py`：PendingSell 数据模型
- `src/lazybull/execution/pending_order.py`：延迟订单管理器

---

**修复人员**：GitHub Copilot  
**审核人员**：待审核  
**发布日期**：2026-02-09
