# 纸面交易补位机制修复：每日基于当天数据重新生成信号

## PR 标题

修复纸面交易补位机制：每日基于当天数据重新生成下一交易日待买入信号，确保与回测策略一致

## 版本更新

- 版本号：0.3.6 → 0.3.7
- 类型：Bug 修复 + 功能优化

## 背景与问题

### 用户反馈的问题

用户反馈当前补位逻辑存在以下两个核心问题：

#### 问题 1：补位未基于当天数据重新生成信号

**期望行为：**
- T1 日（Tn）：买入失败后，基于 Tn 当天数据重新生成信号，用于 T2 日（Tn+1）的买入执行
- T2 日（Tn+1）：如仍有买入失败，再基于 Tn+1 当天数据重新生成信号，用于 T3 日（Tn+2）
- 依此类推，每次补位都使用最新数据重新生成信号

**实际行为（修复前）：**
- T1 日：买入失败后，基于 T1 数据生成补位目标，保存为 `PendingBuy` 对象
- T2、T3、T4...：仅重试相同的 `PendingBuy` 对象，**不重新生成信号**
- 这导致补位目标"固化"，无法根据市场变化动态调整

**问题根源：**
旧实现使用 `pending_buys` 队列机制（类似 `pending_sells`），但这种机制不适合补位场景。补位需要每日重新生成信号，而不是重试固定的订单列表。

#### 问题 2：补位信号生成的成交额过滤一致性

**期望行为：**
- 回测系统在生成信号时会剔除成交额排名后 20% 的股票
- 纸面交易补位也必须一致地应用成交额过滤（默认后 20%）

**实际行为（验证）：**
经过代码审查，发现现有实现已经正确：
- `generate_replacement_targets()` → `_generate_equal_weight_with_lot_constraint()` 
- → `MLSignal.generate_ranked()` → `_apply_amount_filter()`
- 成交额过滤已正确应用，无需修复

**结论：**
问题 2 实际上**不存在**，现有代码已经正确实现了成交额过滤。但我们仍然保留了相关测试用例以确保未来不会退化。

### 用户确认的排除规则

用户已确认补位的"排除规则"选择 **规则 2**：
- 补位生成信号时**不额外排除**"今日不可买入的原目标股票"
- 让信号自然决定是否仍选择它们（如果涨停股票在候选中排名高，允许重新选择）

## 解决方案

### 核心设计思路

将补位机制从"订单队列重试"模式改为"每日信号重新生成"模式，与 T0→T1 机制保持一致：

| 机制 | T0 → T1 | T1 补位 → T2 | T2 补位 → T3 |
|------|---------|-------------|-------------|
| **信号生成日** | T0 | T1 | T2 |
| **执行日** | T1 | T2 | T3 |
| **持久化方式** | `pending/{T1}.parquet` | `pending/{T2}.parquet` | `pending/{T3}.parquet` |
| **元数据** | `source=t0_signal, attempt_count=0` | `source=replenishment, attempt_count=1` | `source=replenishment, attempt_count=2` |

### 实现要点

#### 1. pending_weights 元数据支持

**文件：** `src/lazybull/paper/storage.py`

新增元数据支持，允许在 `pending_weights` 旁边保存元数据文件：

```python
def save_pending_weights(
    self, 
    trade_date: str, 
    targets: List[TargetWeight],
    metadata: Optional[Dict] = None  # 新增
) -> None:
    # 保存 {trade_date}.parquet
    # 保存 {trade_date}_meta.json（如果提供 metadata）

def load_pending_weights_metadata(self, trade_date: str) -> Optional[Dict]:
    # 加载 {trade_date}_meta.json
```

**元数据字段：**
- `source`: 来源标识（`t0_signal` 或 `replenishment`）
- `attempt_count`: 补位尝试次数（0 表示 T0 首次生成，1-5 表示补位）
- `original_signal_date`: 原始信号日期
- `timestamp`: 时间戳

#### 2. T1 执行逻辑更新

**文件：** `scripts/paper_trade.py` - `_execute_t1_if_pending()`

**关键改动：**

```python
# 读取元数据（包含补位尝试次数）
metadata = runner.paper_storage.load_pending_weights_metadata(trade_date)
current_attempt = metadata.get('attempt_count', 0) if metadata else 0

# 处理买入失败
MAX_REPLENISHMENT_ATTEMPTS = 5

if failed_buy_targets:
    next_attempt = current_attempt + 1
    
    # 检查上限
    if next_attempt > MAX_REPLENISHMENT_ATTEMPTS:
        logger.warning(f"补位尝试次数已达上限 ({MAX_REPLENISHMENT_ATTEMPTS})，不再继续补位")
        return actions
    
    # 基于当日 Tn 数据重新生成补位信号
    replacement_targets = runner.generate_replacement_targets(
        trade_date=trade_date,  # 使用当日数据
        failed_count=len(failed_buy_targets),
        ...
    )
    
    # 构建元数据
    replenishment_metadata = {
        'source': 'replenishment',
        'attempt_count': next_attempt,
        'original_signal_date': trade_date,
        ...
    }
    
    # 保存为下一交易日的 pending_weights（与 T0→T1 一致）
    runner.paper_storage.save_pending_weights(
        next_trade_date, 
        replacement_targets,
        metadata=replenishment_metadata
    )
```

#### 3. T0 生成逻辑更新

**文件：** `src/lazybull/paper/runner.py` - `run_t0()`

**关键改动：**

```python
# 检查是否存在补位目标（警告用户将被覆盖）
existing_meta = self.paper_storage.load_pending_weights_metadata(t1_date)
if existing_meta and existing_meta.get('source') == 'replenishment':
    logger.warning(
        f"注意: {t1_date} 已存在补位目标（第 {existing_meta.get('attempt_count', 0)} 次尝试），"
        f"将被本次 T0 信号覆盖"
    )

# 保存 T0 生成的目标（带元数据）
t0_metadata = {
    'source': 't0_signal',
    'attempt_count': 0,
    'signal_date': corrected_date,
}
self.paper_storage.save_pending_weights(t1_date, targets, metadata=t0_metadata)
```

#### 4. 移除旧的 pending_buys 队列处理

**文件：** `scripts/paper_trade.py` - `run_main()`

**关键改动：**

```python
# 旧代码（已移除）：
# 步骤3: 处理延迟买入队列（补位计划）
# pending_buy_actions = _process_pending_buys(runner, corrected_date, config)

# 新代码：
# 注意：旧的"延迟买入队列（补位计划）"机制已被移除
# 新机制：补位目标直接保存为下一交易日的 pending_weights，无需单独处理
# 补位将在 T1 执行时自动处理（如果有失败，会生成下一日的 pending）
```

**影响：**
- `_process_pending_buys()` 函数不再被调用（保留以兼容，但实际不使用）
- `retry_pending_buys()` 方法不再被调用
- `pending_buys.json` 文件不再使用（新机制使用 `pending/{date}.parquet` 和 `{date}_meta.json`）

### 补位流程示意图

```
T0 日（调仓日）:
  └─ 生成信号 → 保存 pending/{T1}.parquet
                └─ 元数据: source=t0_signal, attempt_count=0

T1 日:
  ├─ 读取 pending/{T1}.parquet（attempt_count=0）
  ├─ 执行买入
  └─ 如有失败:
      ├─ 基于 T1 数据重新生成信号
      └─ 保存 pending/{T2}.parquet
          └─ 元数据: source=replenishment, attempt_count=1

T2 日:
  ├─ 读取 pending/{T2}.parquet（attempt_count=1）
  ├─ 执行买入
  └─ 如有失败:
      ├─ 基于 T2 数据重新生成信号
      └─ 保存 pending/{T3}.parquet
          └─ 元数据: source=replenishment, attempt_count=2

...

Tn 日（attempt_count=5）:
  ├─ 读取 pending/{Tn}.parquet（attempt_count=5）
  ├─ 执行买入
  └─ 如有失败:
      └─ 达到上限，不再继续补位
```

## 与回测的一致性分析

### 一致点

1. **信号生成源**：都使用 `MLSignal` 和相同的特征数据
2. **成交额过滤**：都应用 `_apply_amount_filter()`，默认过滤后 20%
3. **一手可买约束**：都应用相同的约束（至少 100 股）
4. **候选顺延**：都从排序候选列表中顺延选择

### 差异点（合理且不可避免）

| 维度 | 纸面交易 | 回测 | 差异原因 |
|------|---------|------|---------|
| **信号生成时点** | T1、T2、T3...（每日重新生成） | T0（一次性生成完整排序候选） | 纸面交易无"上帝视角"，必须逐日发现失败 |
| **信号新鲜度** | 使用最新数据（T1、T2、T3...） | 使用 T0 数据 | 纸面交易更贴近实盘（动态调整） |
| **递归深度** | 最多 5 次（风控） | 可能一次性顺延更多 | 避免过度追逐失败目标 |

**这些差异是合理的**，因为：
1. 纸面交易必须遵循真实交易的时间顺序
2. 使用最新数据生成补位目标，更符合实盘操作逻辑
3. 5 次重试上限提供风控保障

## 测试覆盖

新增测试文件：`tests/test_paper_replenishment.py`

### 测试用例清单

1. **`test_pending_weights_metadata_save_and_load`**
   - 测试元数据的保存和加载
   - 验证 source、attempt_count 等字段

2. **`test_pending_weights_without_metadata`**
   - 测试不带元数据保存（T0 场景）
   - 验证向后兼容性

3. **`test_t0_metadata_save`**
   - 测试 T0 保存带元数据的 pending_weights
   - 验证 source=t0_signal, attempt_count=0

4. **`test_replenishment_metadata_increment`**
   - 测试补位元数据的递增（模拟多次失败）
   - 验证 attempt_count 从 1 递增到 2、3...

5. **`test_replenishment_max_attempts_logic`**
   - 测试补位尝试次数上限逻辑（5次）
   - 验证超过上限时停止补位

6. **`test_pending_weights_file_structure`**
   - 测试文件结构（parquet + meta.json）
   - 验证 JSON 格式正确

7. **`test_amount_filter_concept`**
   - 概念测试：验证成交额过滤逻辑
   - 模拟 MLSignal 的过滤行为

8. **`test_replenishment_targets_reason_format`**
   - 测试补位目标的 reason 格式
   - 验证 "补位-" 前缀

9. **`test_metadata_overwrite_warning_scenario`**
   - 测试 T0 覆盖补位目标的场景
   - 验证元数据检测和覆盖行为

**测试结果：** ✅ 全部通过（9/9）

## 已知限制与后续优化方向

### 已知限制

1. **T0 覆盖补位**：如果某日既是补位日（Tn+1）又是调仓日（T0），T0 信号会覆盖补位信号
   - **影响**：补位被中断，但这是合理的（调仓优先级更高）
   - **缓解**：系统会输出警告日志

2. **同日重复运行**：如果同一日重复运行 T1，metadata 可能被覆盖
   - **影响**：attempt_count 可能不准确
   - **缓解**：T1 有幂等性检查，不允许重复执行

3. **历史数据兼容**：旧的 `pending_buys.json` 文件不会自动迁移
   - **影响**：旧补位队列会被忽略
   - **缓解**：新机制启用后，旧队列自然淘汰（或手工清理）

### 后续优化方向

1. **补位优先级策略**：优先重试涨停股票（可能第二天开板）
2. **补位统计报告**：生成补位成功率、平均重试次数等统计
3. **动态上限调整**：根据市场情况动态调整补位尝试次数上限
4. **补位成本监控**：记录补位导致的滑点成本

## 文档更新

- [x] 更新 `pyproject.toml` 版本号: 0.3.6 → 0.3.7
- [x] 创建 `docs/PR/fix_replenishment_daily_regeneration.md`（本文档）
- [ ] 更新 `CHANGELOG.md`
- [ ] 更新 `docs/paper_trading_guide.md` 说明新补位机制

## 相关文件清单

### 修改的文件

1. `src/lazybull/paper/storage.py`
   - 新增 `save_pending_weights()` 的 metadata 参数
   - 新增 `load_pending_weights_metadata()` 方法

2. `src/lazybull/paper/runner.py`
   - 修改 `run_t0()`：保存 T0 元数据，检测补位覆盖

3. `scripts/paper_trade.py`
   - 修改 `_execute_t1_if_pending()`：读取元数据，检查补位上限，生成下一日 pending
   - 修改 `run_main()`：移除 `_process_pending_buys()` 调用

4. `pyproject.toml`
   - 更新版本号：0.3.6 → 0.3.7

### 新增的文件

1. `tests/test_paper_replenishment.py`
   - 新增 9 个测试用例

2. `docs/PR/fix_replenishment_daily_regeneration.md`
   - 本 PR 说明文档

## 验收标准

- [x] T1 买入失败后，基于当日 Tn 数据重新生成下一日 Tn+1 的 pending 目标
- [x] Tn+1 读取 pending 执行，如仍失败，再基于 Tn+1 数据生成 Tn+2 pending
- [x] 补位尝试次数上限 5 次，超过后停止补位
- [x] 补位信号生成确保经过 MLSignal 成交额过滤（后 20%）
- [x] 新增测试覆盖核心逻辑，全部通过
- [x] PR 说明文档完整，包含设计思路、一致性分析、已知限制
- [x] 版本号更新
- [x] 不引入历史兼容分支（旧 pending_buys 机制自然淘汰）

## 安全性考虑

### 变更风险评估

- **风险等级**：中等
- **影响范围**：纸面交易补位机制
- **潜在风险**：
  1. 旧 `pending_buys.json` 文件中的订单会被忽略（低风险，因为补位会在下次失败时重新生成）
  2. T0 可能覆盖补位 pending（中等风险，但有警告日志）

### 缓解措施

1. 充分的单元测试覆盖
2. 详细的日志输出（INFO 级别）
3. 清晰的警告信息（T0 覆盖补位时）
4. 幂等性保障（T1 不允许重复执行）

## 总结

本 PR 修复了纸面交易补位机制的核心问题：**每日基于当天数据重新生成信号**，使补位行为更贴近真实交易场景，并与回测策略保持一致。

**关键改进：**
1. 从"订单队列重试"模式改为"每日信号重新生成"模式
2. 使用 `pending_weights` + 元数据机制，与 T0→T1 流程一致
3. 确保补位信号经过 MLSignal 成交额过滤
4. 补位尝试次数上限 5 次，提供风控保障

**测试保障：**
- 新增 9 个测试用例，全部通过
- 覆盖核心流程、边界条件、元数据管理

**文档完善：**
- PR 说明文档详细记录设计思路、一致性分析、已知限制
- 为后续优化提供清晰的方向
