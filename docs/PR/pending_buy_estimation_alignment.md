# 补位买入股数估算口径统一

## 概述

本 PR 实现了"补位买入 pending_buys 的预计买入数量提示口径与实际执行口径一致"的改进。

## 背景

当前在未成功买入后生成 pending_buys（补位计划）时，提示给用户的"预计购买数量/金额"策略与实际执行补位买入时（`_execute_pending_buys`）计算买入股数的策略不一致，造成使用困惑。

### 原有问题

1. **提示逻辑（`_print_replacement_targets`）**：
   - 简单使用 `available_cash / len(targets)` 平均分配
   - 未考虑现金保留比例 `pendding_capital_retention_ratio`
   - 未考虑成本预估
   - 未考虑可用现金上限约束

2. **实际执行逻辑（`_execute_pending_buys`）**：
   - `total_cash = account.cash * (1 - pendding_capital_retention_ratio)`
   - `available_cash = total_cash / len(pending_buys)`
   - `target_value = total_cash * target_weight`
   - 若 `target_value + estimated_cost > available_cash`，调整 `target_value`
   - `buy_shares = floor(target_value / price / 100) * 100`

这种不一致导致用户看到的"估算股数"与实际买入股数有较大差异。

## 解决方案

### 1. 抽取统一的估算方法

新增 `PaperTradingRunner._estimate_pending_buy_shares()` 方法，封装补位买入股数的计算逻辑：

```python
def _estimate_pending_buy_shares(
    self,
    ts_code: str,
    price: float,
    target_weight: float,
    total_pending_count: int,
    pendding_capital_retention_ratio: float
) -> int:
    """估算补位买入股数（与_execute_pending_buys的实际执行口径一致）"""
```

**计算逻辑**：
1. `total_cash = account.cash * (1 - pendding_capital_retention_ratio)` - 扣除保留比例
2. `available_cash = total_cash / total_pending_count` - 每个补位目标平均分配
3. `target_value = total_cash * target_weight` - 根据目标权重计算买入金额
4. `estimated_cost = cost_model.calculate_buy_cost(target_value)` - 预估成本
5. 若 `target_value + estimated_cost > available_cash`，则 `target_value = available_cash - estimated_cost` - 调整到可用现金上限
6. `buy_shares = floor(target_value / price / 100) * 100` - 按100股取整

### 2. 更新执行逻辑

`_execute_pending_buys` 方法现在调用统一的估算方法：

```python
# 使用统一的估算方法计算买入股数
buy_shares = self._estimate_pending_buy_shares(
    ts_code=ts_code,
    price=buy_prices[ts_code],
    target_weight=pending_buy.target_weight,
    total_pending_count=len(pending_buys),
    pendding_capital_retention_ratio=cfg['costs']['pendding_capital_retention_ratio']
)
```

### 3. 更新提示逻辑

`_print_replacement_targets` 方法现在也使用统一的估算方法，并添加了明确的说明：

```python
logger.info(f"注意：以下股数为估算值，基于当前价格与现金（保留比例 {pendding_capital_retention_ratio:.1%}）")
logger.info(f"实际执行时会受到执行日价格变化、补位队列长度变化等因素影响，但计算规则一致")
```

同时，表头改为"估算股数"而非"建议股数"，更明确地表达这是估算值。

## 测试

新增测试文件 `tests/test_pending_buy_estimation.py`，包含8个测试用例：

1. **test_estimate_pending_buy_shares_normal** - 正常情况
2. **test_estimate_pending_buy_shares_cash_limit** - 现金受限情况
3. **test_estimate_pending_buy_shares_less_than_one_lot** - 不足一手
4. **test_estimate_pending_buy_shares_zero_price** - 价格为0的异常情况
5. **test_estimate_pending_buy_shares_zero_pending_count** - 补位数量为0
6. **test_estimate_pending_buy_shares_multiple_targets** - 多个补位目标
7. **test_estimate_pending_buy_shares_high_retention_ratio** - 高保留比例
8. **test_estimate_pending_buy_shares_rounding** - 100股取整验证

所有测试用例均已通过，覆盖了关键场景：
- 价格、现金、权重的不同组合
- pending_buys 数量变化
- 成本预估影响
- 100股取整
- 不足一手等边界情况

## 影响范围

### 代码变更

1. **src/lazybull/paper/runner.py**
   - 新增 `_estimate_pending_buy_shares()` 方法（约60行）
   - 重构 `_execute_pending_buys()` 方法，使用统一估算逻辑（简化约40行代码）
   - 重构 `_print_replacement_targets()` 方法，使用统一估算逻辑并增加说明（约15行）

2. **tests/test_pending_buy_estimation.py**
   - 新增测试文件（约260行）

3. **pyproject.toml**
   - 版本号从 0.4.1 升级到 0.4.2

### 用户体验改进

1. **提示更准确**：用户看到的估算股数与实际执行更接近
2. **说明更清晰**：明确标注为"估算值"，并说明可能的影响因素
3. **行为一致**：消除了提示与执行之间的逻辑差异

## 注意事项

1. **估算值说明**：虽然计算逻辑一致，但实际执行时仍可能因以下因素产生差异：
   - 执行日价格变化（使用当日价格而非生成日价格）
   - 补位队列长度变化（如有新的失败买入加入队列）
   - 账户现金变化（如有其他操作影响现金）

2. **不足一手显示**：当估算股数为0时，显示为 "0 (不足一手)"，更清晰地说明原因

3. **向后兼容**：本改动不涉及数据格式或接口变更，完全向后兼容

## 相关配置

估算逻辑依赖的配置项（在 `configs/base.yaml`）：

```yaml
costs:
  pendding_capital_retention_ratio: 0.3  # 补位买入时保留的资金比例，默认30%
```

## 版本信息

- **版本号**：0.4.1 → 0.4.2
- **发布日期**：2026-02-10
- **向后兼容**：是

## 后续改进建议

1. 考虑将 `pendding_capital_retention_ratio` 作为参数传入，而不是每次从配置文件读取
2. 可以在日志中输出更详细的估算过程（调试模式下）
3. 考虑在 Web 界面中也显示估算股数（如有 Web 界面）
