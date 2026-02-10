# 彻底清理 pending_weights 残留逻辑与测试

## 问题背景

此前虽有 PR 声称移除 `pending_weights`，但代码中仍存在多处残留逻辑：

### 残留问题盘点

1. **运行时调用不存在的方法**
   - `scripts/paper_trade.py` 调用 `save_pending_weights()` / `load_pending_weights()` （第446、752、763行）
   - 但这些方法在 `src/lazybull/paper/storage.py` 中并不存在
   - 运行时会抛出 `AttributeError`

2. **兼容模式分支混乱**
   - T1 执行逻辑中仍保留"指令驱动模式 vs 兼容模式"分支
   - 日志输出"将忽略 pending_weights"、"将按 pending_weights 生成订单"
   - 实际上 pending_weights 已不可用

3. **测试文件大量引用**
   - `tests/test_paper_replenishment.py`：整个文件测试 pending_weights（9个测试）
   - `tests/test_paper_trading.py`：3个测试调用 pending_weights 方法
   - `tests/test_replenishment_no_sell.py`：多处断言和注释涉及 pending_weights

4. **文档描述过时**
   - 文档仍提及 pending_weights 作为"已废弃"机制
   - 升级建议中要求"完成所有待执行的 pending_weights"

## 解决方案

### 1. 代码清理（scripts/paper_trade.py）

#### 1.1 清理 T1 执行逻辑

**变更前**：
```python
# 检查是否有待执行目标（全量调仓，兼容模式）
targets = runner.paper_storage.load_pending_weights(trade_date)

if not instructions and not targets and not pending_buys:
    logger.info(f"未找到交易指令、待执行目标或补位买入计划，跳过 T1")

if instructions:
    logger.info("【T1 指令驱动模式】")
    logger.info("将忽略 pending_weights，严格按指令执行")
elif targets:
    logger.info("【T1 兼容模式】")
    logger.info("将按 pending_weights 生成订单执行")
```

**变更后**：
```python
# 仅检查交易指令和补位买入
instructions = runner.paper_storage.load_instructions(trade_date)
pending_buys = runner.paper_storage.load_pending_buys()

if not instructions and not pending_buys:
    logger.info(f"未找到交易指令或补位买入计划，跳过 T1")

if instructions:
    logger.info("【T1 指令驱动】读取到 {len(instructions)} 条交易指令")
```

**影响**：
- 移除"兼容模式"分支及相关日志
- T1 执行仅支持 instructions 和 pending_buys

#### 1.2 清理 ECT 应用逻辑

**变更前**：
```python
# 读取生成的目标
targets = runner.paper_storage.load_pending_weights(t1_date)
if targets:
    # 应用 ECT 系数到目标权重
    for target in targets:
        target.target_weight = original_weight * ect_exposure
    runner.paper_storage.save_pending_weights(t1_date, targets)
```

**变更后**：
```python
# 读取生成的交易指令
instructions = runner.paper_storage.load_instructions(t1_date)
if instructions:
    # 应用 ECT 系数到买入指令（调整股数）
    for inst in instructions:
        if inst.action == 'buy':
            inst.shares = int(inst.shares * ect_exposure)
            inst.shares = (inst.shares // 100) * 100  # 确保是100的倍数
    runner.paper_storage.save_instructions(t1_date, instructions)
```

**影响**：
- ECT 系数应用从"调整权重"改为"调整买入股数"
- 符合指令驱动的设计理念

#### 1.3 清理注释和 docstring

**删除/更新的内容**：
- "旧的延迟买入队列机制已被移除"（误导性注释）
- "保存到独立的 pending_buys 队列（不覆盖 pending_weights）"
- "优先检查并执行 instructions，若不存在则使用 pending_weights（兼容模式）"

### 2. 测试清理

#### 2.1 删除整个文件
- `tests/test_paper_replenishment.py`（266行，9个测试）
- 原因：整个文件测试 pending_weights 元数据保存/加载，已无意义

#### 2.2 删除 test_paper_trading.py 中的测试
- `test_storage_save_and_load_pending()`
- `test_storage_pending_weights_not_exist()`
- `test_instruction_priority_over_targets()`

#### 2.3 清理 test_replenishment_no_sell.py
- 移除 `assert storage.load_pending_weights(...) is None` 断言
- 更新 docstring 中的 pending_weights 描述
- 保留核心测试逻辑（验证 pending_buys 不触发卖出）

### 3. 文档更新

#### 3.1 更新 `docs/guide/paper_trading_adjust_guide.md`

**变更前**：
```markdown
- ✅ 仅使用 instructions（废弃 pending_weights）
- ⚠️ 使用 pending_weights（已废弃）
- 在升级前完成所有待执行的 pending_weights
```

**变更后**：
```markdown
- ✅ 仅使用 instructions（指令驱动）
- ✅ pending_buys 队列处理补位买入
```

#### 3.2 创建本文档
- 说明删改点、影响范围、验证方式

### 4. 版本号递增

- `pyproject.toml`：`0.4.0` → `0.4.1`
- `src/lazybull/__init__.py`：`0.3.5` → `0.4.1`

**递增理由**：
- 这是修复性变更（让实现与既定破坏性变更一致）
- 小版本号 +1 表示 bug 修复

## 影响范围

### 向后不兼容

- **已经不兼容**：v0.4.0 声称移除 pending_weights，但代码实际未移除导致运行时错误
- **本次修复**：彻底移除残留引用，确保代码与文档一致

### 数据迁移

- **无需迁移**：`data/paper/pending/` 目录保留（选择 A：不管它）
- **代码不再读写**：确保代码完全不使用该目录

### 测试影响

- 删除 266 行测试代码（1个文件 + 3个测试函数）
- 保留的测试覆盖 instructions 和 pending_buys 核心功能

## 验证方式

### 1. 代码搜索验证

```bash
# 确认无 pending_weights 残留
grep -r "pending_weights" scripts/ src/ tests/
grep -r "save_pending_weights" scripts/ src/ tests/
grep -r "load_pending_weights" scripts/ src/ tests/
```

**预期结果**：仅在文档和 CHANGELOG 中出现

### 2. 测试验证

```bash
pytest tests/ -v
```

**预期结果**：所有测试通过

### 3. 运行时验证

```bash
# T0 生成指令
python scripts/paper_trade.py run --trade-date 20260210

# T1 执行指令
python scripts/paper_trade.py run --trade-date 20260211
```

**预期行为**：
- T0 保存 instructions 到 `data/paper/instructions/20260211.parquet`
- T1 读取并执行 instructions
- 日志显示"【T1 指令驱动】读取到 X 条交易指令"
- 无"兼容模式"或 pending_weights 相关日志

## 总结

### 删改统计

| 类别 | 删除 | 修改 | 新增 |
|------|------|------|------|
| 代码（scripts） | 70行 | 39行 | 0 |
| 测试文件 | 380行 | 6行 | 0 |
| 文档 | 12行 | 8行 | 1个新文件 |

### 核心改进

1. **运行时稳定性**：移除对不存在方法的调用
2. **代码一致性**：T1 执行逻辑单一化（仅 instructions + pending_buys）
3. **文档准确性**：移除过时描述，避免用户困惑
4. **测试有效性**：删除无效测试，保留核心功能验证

### 后续建议

1. 可考虑在未来版本中删除 `data/paper/pending/` 目录的创建逻辑
2. CHANGELOG 中明确记录此次修复，避免用户疑惑
3. 如需支持旧数据迁移，可提供独立的迁移脚本（当前不提供）
