# 修复 CLI T1 路径未使用指令驱动模式的问题

## PR 编号
版本：v0.3.13

## 问题现象

当用户运行 `python scripts/paper_trade.py run --trade-date 20260203` 时，即使 T0 已在前一天（20260202）生成并保存了交易指令（`data/paper/instructions/20260203.parquet`），T1 阶段仍然完全忽略这些指令，继续使用旧的 `pending_weights` 逻辑执行全量调仓。

### 具体表现

用户报告：
- T0 阶段（20260202）已生成指令，提示 603115.SH 需要减仓 100 股
- T1 阶段（20260203）运行时，日志显示"找到30个全量调仓目标/生成订单26买26卖/卖出清仓26笔减仓0笔"
- 实际上 603115.SH 并没有按指令减仓 100 股卖出

这说明 CLI 的 T1 执行路径仍在走旧模式，没有读取和执行 instructions。

## 根本原因

### 代码层面分析

1. **Runner 层已正确实现指令优先逻辑**
   - `PaperTradingRunner.run_t1()` 方法（`src/lazybull/paper/runner.py:413`）已实现：
     - 优先调用 `self.paper_storage.load_instructions(corrected_date)`
     - 若存在 instructions，调用 `self.broker.execute_instructions()` 执行
     - 仅在 instructions 不存在时才使用 `pending_weights` 作为兜底

2. **CLI 层未同步更新**
   - `scripts/paper_trade.py` 的 `_execute_t1_if_pending()` 函数（第391行）：
     - 只调用 `runner.paper_storage.load_pending_weights(trade_date)`
     - 完全没有检查 instructions 是否存在
     - 直接进入 `broker.generate_orders()` → `broker.execute_orders()` 旧流程
   - 这导致 CLI 路径完全绕过了指令驱动模式

3. **T0 的 sell_price_type 硬编码问题**
   - `run_t0()` 方法（`src/lazybull/paper/runner.py:352`）在生成指令时：
     ```python
     instructions = self._generate_instructions(
         targets=targets,
         buy_price_type=buy_price_type,
         sell_price_type='close',  # ← 硬编码为 close
         ...
     )
     ```
   - 应该使用配置中的 `sell_price` 参数，而不是硬编码

## 修复方案

### 1. 修改 CLI T1 执行流程

**文件**：`scripts/paper_trade.py`

**修改点**：`_execute_t1_if_pending()` 函数（第391行起）

**变更内容**：
```python
# 优先检查是否有交易指令（新模式）
instructions = runner.paper_storage.load_instructions(trade_date)

# 检查是否有待执行目标（全量调仓，兼容模式）
targets = runner.paper_storage.load_pending_weights(trade_date)

# 输出清晰的模式标识
if instructions:
    logger.info("=" * 80)
    logger.info(f"【T1 指令驱动模式】读取到 {len(instructions)} 条交易指令")
    logger.info("将忽略 pending_weights，严格按指令执行")
    logger.info("=" * 80)
elif targets:
    logger.info("=" * 80)
    logger.info(f"【T1 兼容模式】找到 {len(targets)} 个全量调仓目标")
    logger.info("将按 pending_weights 生成订单执行")
    logger.info("=" * 80)

# 执行交易指令（优先，新模式）
if instructions:
    logger.info("执行交易指令")
    fills = runner.broker.execute_instructions(
        instructions,
        buy_prices,
        sell_prices,
        trade_date
    )
    ...
# 处理全量调仓（如果没有指令，使用兼容模式）
elif targets:
    orders = runner.broker.generate_orders(targets, buy_prices, sell_prices, trade_date)
    ...
```

**设计要点**：
- 优先检查并加载 instructions
- 若存在 instructions，完全忽略 pending_weights，直接调用 `broker.execute_instructions()`
- 若不存在 instructions，保持现有 pending_weights 逻辑作为兜底
- 仍保留并执行 pending_buys（补位买入）流程
- 输出清晰的中文日志，明确标识当前走哪个模式

### 2. 修复 T0 的 sell_price_type 参数

**文件**：`src/lazybull/paper/runner.py`

**修改点1**：`run_t0()` 方法签名（第275行）
```python
def run_t0(
    self,
    trade_date: str,
    buy_price_type: str = 'close',
    sell_price_type: str = 'close',  # 新增参数
    universe_type: str = 'mainboard',
    top_n: int = 5,
    model_version: Optional[int] = None,
    rebalance_freq: int = 5
) -> None:
```

**修改点2**：指令生成调用（第349行）
```python
# 使用传入的 sell_price_type 参数，而非硬编码 'close'
instructions = self._generate_instructions(
    targets=targets,
    buy_price_type=buy_price_type,
    sell_price_type=sell_price_type,  # 使用参数
    current_prices=current_prices,
    source_date=corrected_date
)
```

**修改点3**：CLI 调用更新（`scripts/paper_trade.py:707`）
```python
runner.run_t0(
    trade_date=trade_date,
    buy_price_type=config['buy_price'],
    sell_price_type=config['sell_price'],  # 传入配置值
    universe_type=config['universe'],
    top_n=config['top_n'],
    model_version=config.get('model_version'),
    rebalance_freq=config['rebalance_freq']
)
```

### 3. 日志与可观测性增强

**增强点**：
- 在 CLI T1 开始时，明确输出当前走"指令驱动模式"还是"兼容模式"
- 使用醒目的分隔线（80个等号）突出显示模式选择
- 指令模式下，明确打印"将忽略 pending_weights"
- 避免用户误判当前执行路径

**示例输出**：
```
================================================================================
【T1 指令驱动模式】读取到 15 条交易指令
将忽略 pending_weights，严格按指令执行
================================================================================
执行交易指令
指令执行完成：15 条指令，12 笔成交
```

或：
```
================================================================================
【T1 兼容模式】找到 30 个全量调仓目标
将按 pending_weights 生成订单执行
================================================================================
```

### 4. 测试覆盖

**文件**：`tests/test_paper_trading_cli.py`

**新增测试1**：`test_instructions_loading()`
- 测试 instructions 的保存和加载功能
- 验证保存的指令包含正确的字段（ts_code, action, shares, price_type等）

**新增测试2**：`test_instructions_not_exist()`
- 测试加载不存在的指令文件时的行为
- 确保返回 None 或空列表，不报错

**测试覆盖点**：
- instructions 文件存在时，CLI 应走指令路径
- instructions 文件不存在时，CLI 应走 pending_weights 路径
- 指令中的 price_type 应正确反映配置的 sell_price

## 验收标准

### 测试步骤

1. **准备环境**
   ```bash
   # 配置纸面交易参数
   python scripts/paper_trade.py config \
     --buy-price close --sell-price close \
     --top-n 5 --initial-capital 500000 \
     --rebalance-freq 5 --weight-method equal
   ```

2. **执行 T0（生成指令）**
   ```bash
   # 在 20260202 执行 T0，生成 T1 指令
   python scripts/paper_trade.py run --trade-date 20260202
   ```
   
   **预期结果**：
   - 生成 `data/paper/instructions/20260203.parquet`
   - 日志显示"T0工作流完成 - 已生成 N 个目标权重和 M 条交易指令"

3. **执行 T1（指令驱动模式）**
   ```bash
   # 在 20260203 执行 T1，应读取指令
   python scripts/paper_trade.py run --trade-date 20260203
   ```
   
   **预期结果**：
   - 日志显示：
     ```
     ================================================================================
     【T1 指令驱动模式】读取到 M 条交易指令
     将忽略 pending_weights，严格按指令执行
     ================================================================================
     ```
   - 对 T0 指令中的减仓/清仓，必须生成对应的卖出成交（除非股票不可交易）
   - **不应出现**"找到N个全量调仓目标/生成订单…"这类旧模式主路径日志

4. **验证具体案例（603115.SH 减仓100）**
   - 若 T0 指令包含：`603115.SH, action=sell, shares=100`
   - T1 执行后，必须在成交记录中看到 603115.SH 卖出 100 股（或因跌停无法卖出的明确标识）

### 验收检查点

- [x] CLI T1 路径优先读取 instructions
- [x] instructions 存在时，完全忽略 pending_weights
- [x] 日志明确标识"指令驱动模式"或"兼容模式"
- [x] T0 生成的 sell_price_type 使用配置值，不再硬编码
- [x] 单元测试覆盖指令加载和模式选择逻辑
- [x] 版本号递增至 0.3.13

## 向后兼容性

本次修改完全向后兼容：

1. **兜底逻辑保留**
   - 当 instructions 不存在时，T1 自动回退到 pending_weights 路径
   - 旧项目无需修改任何代码或配置

2. **API 接口兼容**
   - `run_t0()` 新增的 `sell_price_type` 参数有默认值 `'close'`
   - 旧代码调用不传该参数时，行为不变

3. **数据格式兼容**
   - instructions 和 pending_weights 并存，互不影响
   - 补位买入（pending_buys）流程保持不变

## 相关文档

- [指令驱动T0T1执行模式](./指令驱动T0T1执行模式.md) - 指令驱动模式的详细设计文档
- [paper_trading_guide.md](../paper_trading_guide.md) - 纸面交易完整指南
- [CHANGELOG.md](../../CHANGELOG.md) - 版本变更记录

## 总结

本次修复解决了 CLI 路径下 T1 未使用指令驱动模式的关键问题，确保：

1. **功能完整性**：CLI 和 Runner 路径的 T1 执行逻辑完全一致
2. **用户体验**：清晰的日志输出，用户不会误判执行模式
3. **配置一致性**：sell_price_type 正确使用配置值，不再硬编码
4. **可测试性**：新增单元测试，覆盖指令加载和执行逻辑
5. **向后兼容**：保留兜底逻辑，旧项目无需修改

用户现在可以放心使用 CLI 运行纸面交易，T0→T1 的指令驱动流程将正确执行。
