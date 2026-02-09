# 修复纸面交易补位机制导致的清仓问题

## 问题描述

### 复现场景（来自用户日志）

**背景**：
- 账户已有 27 只持仓股票
- 调仓频率 `rebalance_freq=20`
- 运行日期：20260107（非调仓日）

**现象**：
1. 读取 `pending/20260107.parquet` 仅 3 条记录（source=replenishment，补位目标）
2. `PaperBroker.generate_orders()` 生成 `0 买，27 卖`
3. 全部持仓按"退出持仓"逻辑卖出（被清仓）
4. 日志显示 3 个不可买入的目标被记录为补位候选
5. 随后生成下一交易日 20260108 的补位目标 3 条，保存为 `pending/20260108.parquet`

**结论**：
- 补位目标（仅 3 只）直接写入 `pending/{next_date}.parquet`
- T1 执行时把 pending 当作"全量目标权重"
- 导致当前持仓（27 只）不在目标（3 只）中时，被当作 `target_weight=0` 清仓

## 问题根因

### 架构层面

当前补位机制的实现存在设计缺陷：

```
T0（调仓日）：生成 30 只目标 → 保存为 pending/T1.parquet
T1：读取 pending → 买入成功 27 只，失败 3 只
T1：基于当日数据生成 3 只补位目标 → 覆盖保存为 pending/T2.parquet  # ❌ 问题
T2：读取 pending（仅 3 只补位目标） → 当作全量目标
T2：生成订单：0 买（3 只补位目标仍不可买），27 卖（持仓不在目标中）
T2：结果：全部持仓被清仓！
```

### 代码层面

**旧实现**（问题代码）：
```python
# scripts/paper_trade.py (_execute_t1_if_pending)
if failed_buy_targets:
    replacement_targets = runner.generate_replacement_targets(...)
    
    # ❌ 直接保存为 pending_weights，覆盖全量目标
    runner.paper_storage.save_pending_weights(
        next_trade_date, 
        replacement_targets,
        metadata={'source': 'replenishment', ...}
    )
```

**问题点**：
1. 补位目标与全量调仓目标共用同一个存储通道（`pending_weights`）
2. T1 执行时无法区分"全量调仓"和"增量补位"
3. `broker.generate_orders()` 把 3 只补位目标当作全量目标，导致 27 只持仓被当作 `target_weight=0` 生成卖出订单

## 解决方案

### 设计原则

1. **分离存储通道**：补位目标使用独立的 `pending_buys` 队列，不覆盖 `pending_weights`
2. **分离执行逻辑**：T1 分别处理全量调仓和补位买入
3. **仅买入不卖出**：补位执行不触发任何卖出订单
4. **向后兼容**：与现有 T0/T1 架构兼容

### 数据结构

**新增存储**：
```
data/paper/
├── pending/           # 全量调仓目标（来自T0）
│   ├── 20260121.parquet
│   └── 20260121_meta.json
└── pending_buys/      # 补位买入计划（来自T1失败买入）
    └── pending_buys.json  # ✅ 新增：独立存储补位计划
```

**PendingBuy 数据模型**（已存在于 models.py）：
```python
@dataclass
class PendingBuy:
    ts_code: str                  # 股票代码
    target_weight: float          # 目标权重
    reason: str                   # 原因（如：补位-涨停）
    create_date: str              # 创建日期
    attempts: int                 # 尝试次数
    last_attempt_date: str        # 最后尝试日期
    original_signal_date: str     # 原始信号日期
```

### 执行流程

**新流程**：
```
T0（调仓日）：生成 30 只目标 → 保存为 pending/T1.parquet
T1：读取 pending_weights → 全量调仓（买入 27 只，失败 3 只）
T1：生成补位计划 → 保存到 pending_buys（独立队列，不覆盖 pending_weights）
T2：读取 pending_buys（3 只补位计划） → 仅尝试买入
T2：不读取 pending_weights（因为不存在）
T2：结果：仅生成买入订单（如果可买），不触发卖出  # ✅ 修复
```

### 核心代码修改

#### 1. scripts/paper_trade.py

**旧代码**：
```python
# ❌ 错误：保存为 pending_weights
runner.paper_storage.save_pending_weights(
    next_trade_date, 
    replacement_targets,
    metadata={'source': 'replenishment', ...}
)
```

**新代码**：
```python
# ✅ 正确：保存为 pending_buys
from src.lazybull.paper.models import PendingBuy

pending_buys = []
for target in replacement_targets:
    pending_buys.append(PendingBuy(
        ts_code=target.ts_code,
        target_weight=target.target_weight,
        reason=target.reason,
        create_date=trade_date,
        attempts=next_attempt,
        last_attempt_date="",
        original_signal_date=trade_date
    ))

# 保存到独立队列
runner.paper_storage.save_pending_buys(pending_buys)
```

#### 2. src/lazybull/paper/runner.py

**新增方法**：`_execute_pending_buys()`
```python
def _execute_pending_buys(
    self,
    pending_buys: List[PendingBuy],
    buy_prices: Dict[str, float],
    trade_date: str,
    buy_price_type: str = 'close'
) -> List[Fill]:
    """执行补位买入计划（仅买入，不触发卖出）
    
    核心逻辑：
    1. 遍历 pending_buys 队列
    2. 检查可交易性（涨跌停、停牌）
    3. 仅生成买入订单（不调用 broker.generate_orders）
    4. 失败的继续保留在队列中，尝试次数+1
    5. 成功的从队列中移除
    """
    # 详见代码实现
```

**修改方法**：`run_t1()`
```python
def run_t1(self, trade_date: str, ...):
    # 1. 读取全量调仓目标
    targets = self.paper_storage.load_pending_weights(corrected_date)
    
    # 2. 读取补位买入计划（新增）
    pending_buys = self.paper_storage.load_pending_buys()
    
    # 3. 执行全量调仓（如果有 targets）
    if targets:
        orders = self.broker.generate_orders(targets, ...)
        fills = self.broker.execute_orders(orders, ...)
    
    # 4. 执行补位买入（如果有 pending_buys）（新增）
    if pending_buys:
        replenishment_fills = self._execute_pending_buys(
            pending_buys, buy_prices, corrected_date, buy_price_type
        )
```

## 验收测试

### 测试场景

新增 `tests/test_replenishment_no_sell.py`，包含 3 个测试用例：

#### 测试 1：存在持仓 + 仅补位计划 → 不生成卖出订单 ✓

```python
def test_replenishment_with_existing_positions_no_sell():
    """
    设置：
    - 账户持有 27 只股票
    - pending_buys 有 3 条补位计划
    - pending_weights 不存在
    
    验证：
    - 不生成任何卖出订单
    - 27 只持仓保持不变
    
    反向验证：
    - 如果错误地将 3 只补位作为 pending_weights
    - 会生成 27 个卖出订单（清仓）
    """
```

**结果**：✅ PASSED

#### 测试 2：正确的补位流程 ✓

```python
def test_replenishment_correct_flow():
    """
    设置：
    - 账户持有 27 只股票
    - pending_buys 有 1 条补位计划（可买入）
    
    验证：
    - 仅生成买入订单（如果可买）
    - 不触发任何卖出订单
    - 27 只持仓不受影响
    """
```

**结果**：✅ PASSED

#### 测试 3：全量调仓 vs 补位对比 ✓

```python
def test_full_rebalance_vs_replenishment():
    """
    对比：
    - 全量调仓（pending_weights）：会生成卖出订单
    - 补位（pending_buys）：仅生成买入订单
    
    验证：
    - 全量调仓会卖出原有 5 只，买入 3 只新的
    - 补位不会调用 broker.generate_orders
    """
```

**结果**：✅ PASSED

### 测试运行结果

```bash
$ pytest tests/test_replenishment_no_sell.py -v
============================= test session starts ==============================
collected 3 items

tests/test_replenishment_no_sell.py::test_replenishment_with_existing_positions_no_sell PASSED [ 33%]
tests/test_replenishment_no_sell.py::test_replenishment_correct_flow PASSED [ 66%]
tests/test_replenishment_no_sell.py::test_full_rebalance_vs_replenishment PASSED [100%]

============================== 3 passed in 6.56s ===============================
```

### 现有测试

所有现有的补位相关测试仍然通过：

```bash
$ pytest tests/test_paper_replenishment.py -v
============================= test session starts ==============================
collected 9 items

tests/test_paper_replenishment.py::test_pending_weights_metadata_save_and_load PASSED
tests/test_paper_replenishment.py::test_pending_weights_without_metadata PASSED
tests/test_paper_replenishment.py::test_t0_metadata_save PASSED
tests/test_paper_replenishment.py::test_replenishment_metadata_increment PASSED
tests/test_paper_replenishment.py::test_replenishment_max_attempts_logic PASSED
tests/test_paper_replenishment.py::test_pending_weights_file_structure PASSED
tests/test_paper_replenishment.py::test_amount_filter_concept PASSED
tests/test_paper_replenishment.py::test_replenishment_targets_reason_format PASSED
tests/test_paper_replenishment.py::test_metadata_overwrite_warning_scenario PASSED

============================== 9 passed in 0.57s ===============================
```

## 补位机制生命周期说明

### T0/T1 架构

```
T0（调仓日，如周一）：
  输入：当日数据（特征、信号）
  输出：pending/T1.parquet（全量目标权重，30只）
  元数据：source=t0_signal, attempt_count=0

T1（买入日，如周二）：
  输入：pending/T1.parquet（全量目标）
  执行：
    1. 生成订单（买入 + 卖出）
    2. 执行订单
    3. 买入成功 27 只，失败 3 只（涨停、停牌等）
  输出：
    - pending_buys/pending_buys.json（3只补位计划）
    - attempts=1

T2（补位日，如周三）：
  输入：pending_buys（3只补位计划）
  执行：
    1. 读取 pending_buys
    2. 尝试买入（仅买入，不卖出）
    3. 成功：从队列移除
    4. 失败：attempts+1，保留在队列
  输出：
    - 更新后的 pending_buys（失败的继续保留）
    - 如果失败，生成新的补位计划（基于当日数据）

T3-T6：
  重复 T2 的逻辑，直到：
    - 补位成功（从队列移除）
    - 或达到最大尝试次数（5次，放弃）
```

### 数据格式

**pending_buys/pending_buys.json**：
```json
[
  {
    "ts_code": "600100.SH",
    "target_weight": 0.033,
    "reason": "补位-信号生成（涨停）",
    "create_date": "20260107",
    "attempts": 2,
    "last_attempt_date": "20260108",
    "original_signal_date": "20260106"
  }
]
```

## 与旧版本的兼容性

### 不兼容变更

**旧格式（已废弃）**：
- 补位目标保存在 `pending/{next_date}.parquet` + `{next_date}_meta.json`
- 元数据：`source=replenishment, attempt_count=N`

**新格式（0.3.8+）**：
- 补位目标保存在 `pending_buys/pending_buys.json`
- 使用 `PendingBuy` 数据模型

### 迁移建议

如果您的环境中存在旧格式的补位 pending：

1. **手工清理**：删除 `pending/` 目录下标记为 `source=replenishment` 的文件
2. **重新生成**：在下次买入失败时，系统会自动使用新格式
3. **验证**：检查 `pending_buys/pending_buys.json` 是否正常生成

**重要提示**：本次修复不向后兼容旧格式的补位 pending。如需兼容，请联系维护者。

## 总结

### 问题修复

- ✅ 补位目标不再覆盖全量组合目标
- ✅ 补位执行不会触发"退出持仓"卖出订单
- ✅ 账户持仓不会因补位而被清仓

### 核心改进

1. **架构解耦**：补位与全量调仓使用独立存储和执行通道
2. **语义清晰**：pending_weights=全量目标，pending_buys=增量买入
3. **执行安全**：补位仅生成买入订单，不触发卖出
4. **测试覆盖**：新增回归测试，确保不再出现清仓问题

### 版本号

- **0.3.8** (2026-02-09)

### 相关文档

- `tests/test_replenishment_no_sell.py`：回归测试
- `CHANGELOG.md`：版本变更记录
- `docs/PR/fix_replenishment_clearance_issue.md`（本文档）
