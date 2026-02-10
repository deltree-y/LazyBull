# 纸面交易手工修正/回滚使用指南

## 概述

在纸面交易运行过程中，可能因各种原因需要手工修正账户状态（持仓或现金），并从指定日期重新推导后续数据。`adjust` 命令提供了一套完整的工具来实现这一需求。

## 核心概念

### Cut-off 日期
修正的 **cut-off 日期**是指"修正生效的日期"，修正发生在该日期的 `run` 之前。

- **修正后**：从 cut-off 日期开始可以重新运行
- **数据清理**：所有日期 >= cut-off 的数据将被清理
- **一致性保障**：重新运行后，数据链路完整一致

### 自动清理机制
每次执行 `adjust` 命令后，系统会自动：
1. 将 `account.json` 的 `last_update` 设置为 cut-off 日期
2. 清理所有 >= cut-off 日期的数据：
   - 成交记录（trades.parquet）
   - 净值记录（nav.parquet）
   - 运行记录（runs/t0_*.json, t1_*.json）
   - 交易指令（instructions/）
   - 延迟队列（pending_buys.json, pending_sells.json）
3. 智能回滚调仓状态（rebalance_state.json）

## 使用场景

### 场景1：错误买入/卖出，需要撤销重来
**问题**：昨天（20260208）错误买入了某只股票，需要撤销并重新运行。

**解决方案**：
```bash
# 1. 删除错误的持仓（释放资金）
python scripts/paper_trade.py adjust delete-position \
  --trade-date 20260208 \
  --ts-code 000001.SZ

# 2. 查看当前持仓确认修正
python scripts/paper_trade.py positions --trade-date 20260208

# 3. 从该日期重新运行 T0
python scripts/paper_trade.py run --trade-date 20260208
```

**效果**：
- 删除 000001.SZ 持仓
- 按买入价格释放资金回账户
- 清理 20260208 及之后的所有数据
- 可重新执行 20260208 的 T0/T1

---

### 场景2：手工调整持仓股数和成本
**问题**：实盘账户实际持仓与纸面交易不一致，需要调整为实际值。

**解决方案**：
```bash
# 修正持仓为实际股数和成本
python scripts/paper_trade.py adjust update-position \
  --trade-date 20260209 \
  --ts-code 600519.SH \
  --shares 1200 \
  --buy-price 1800.00
```

**效果**：
- 持仓更新为 1200 股，买入价 1800.00
- 自动计算现金变动：
  - `delta_cash = old_shares*old_price - 1200*1800.00`
  - `cash += delta_cash`
- 清理 20260209 及之后的数据

---

### 场景3：对已有持仓加仓
**问题**：手工买入了额外的股票，需要同步到纸面账户。

**解决方案**：
```bash
# 加仓 500 股
python scripts/paper_trade.py adjust add-shares \
  --trade-date 20260209 \
  --ts-code 600519.SH \
  --shares 500 \
  --price 1850.00
```

**效果**：
- 扣减现金：`cash -= 500 * 1850.00`
- 加权更新买入价：
  - `new_price = (old_price*old_shares + 1850*500) / (old_shares+500)`
- 持仓股数增加 500

**注意**：`add-shares` 仅允许对已存在持仓加仓。若持仓不存在，会报错：
```
持仓 600519.SH 不存在，无法加仓
提示：add-shares 仅允许对已存在持仓加仓
      如需新建持仓，请使用 update-position 命令
```

---

### 场景4：追加或调整资金
**问题**：账户实际可用资金发生变化（追加投资、提取资金等）。

**解决方案**：
```bash
# 设置账户现金为 60 万
python scripts/paper_trade.py adjust cash \
  --trade-date 20260209 \
  --set 600000.00
```

**效果**：
- 现金直接设置为 600,000.00
- 显示旧现金、新现金及变动金额

---

## 命令参考

### `adjust delete-position`
删除持仓并按买入价格释放资金。

**参数**：
- `--trade-date` (必需): Cut-off 日期，格式 YYYYMMDD
- `--ts-code` (必需): 股票代码

**示例**：
```bash
python scripts/paper_trade.py adjust delete-position \
  --trade-date 20260210 \
  --ts-code 600519.SH
```

---

### `adjust update-position`
更新持仓股数和买入价格，自动计算现金变动。

**参数**：
- `--trade-date` (必需): Cut-off 日期，格式 YYYYMMDD
- `--ts-code` (必需): 股票代码
- `--shares` (必需): 新持仓股数
- `--buy-price` (必需): 新买入价格

**示例**：
```bash
python scripts/paper_trade.py adjust update-position \
  --trade-date 20260210 \
  --ts-code 600519.SH \
  --shares 1000 \
  --buy-price 1800.50
```

---

### `adjust add-shares`
对已有持仓加仓，加权更新买入价格。

**参数**：
- `--trade-date` (必需): Cut-off 日期，格式 YYYYMMDD
- `--ts-code` (必需): 股票代码
- `--shares` (必需): 加仓股数
- `--price` (必需): 加仓价格

**示例**：
```bash
python scripts/paper_trade.py adjust add-shares \
  --trade-date 20260210 \
  --ts-code 600519.SH \
  --shares 500 \
  --price 1850.00
```

**限制**：
- 仅允许对已存在持仓加仓
- 若持仓不存在，会报错并提示使用 `update-position`

---

### `adjust cash`
直接设置账户现金金额。

**参数**：
- `--trade-date` (必需): Cut-off 日期，格式 YYYYMMDD
- `--set` (必需): 新现金金额

**示例**：
```bash
python scripts/paper_trade.py adjust cash \
  --trade-date 20260210 \
  --set 500000.00
```

---

## 操作流程

### 标准修正流程
1. **查看当前状态**（可选）
   ```bash
   python scripts/paper_trade.py positions --trade-date 20260210
   ```

2. **执行修正**
   ```bash
   python scripts/paper_trade.py adjust <子命令> [参数...]
   ```

3. **确认修正结果**
   - 查看日志输出
   - 检查账户文件 `data/paper/state/account.json`
   - 确认 `last_update` 已更新为 cut-off 日期

4. **重新运行**
   ```bash
   python scripts/paper_trade.py run --trade-date 20260210
   ```

### 批量修正
若需要批量修正多个持仓，按顺序执行多个 `adjust` 命令：
```bash
# 修正持仓1
python scripts/paper_trade.py adjust update-position \
  --trade-date 20260210 --ts-code 600519.SH --shares 1000 --buy-price 1800.00

# 修正持仓2
python scripts/paper_trade.py adjust update-position \
  --trade-date 20260210 --ts-code 000001.SZ --shares 500 --buy-price 12.50

# 调整现金
python scripts/paper_trade.py adjust cash \
  --trade-date 20260210 --set 550000.00

# 只需执行一次 truncate_since（最后一个 adjust 会自动执行）
```

**注意**：每个 `adjust` 命令都会调用 `truncate_since`，但多次调用幂等无副作用。

---

## 数据清理详解

### 清理范围
执行 `adjust` 后，以下数据会被清理（>= cut-off 日期）：

| 数据类型 | 文件路径 | 清理方式 |
|---------|---------|---------|
| 成交记录 | `data/paper/trades/trades.parquet` | 删除行（`trade_date >= cut_off`） |
| 净值记录 | `data/paper/nav/nav.parquet` | 删除行（`trade_date >= cut_off`） |
| T0 运行记录 | `data/paper/runs/t0_YYYYMMDD.json` | 删除文件 |
| T1 运行记录 | `data/paper/runs/t1_YYYYMMDD.json` | 删除文件 |
| 交易指令 | `data/paper/instructions/YYYYMMDD.parquet` | 删除文件 |
| 延迟买入 | `data/paper/pending_buys/pending_buys.json` | 清空（写入 `[]`） |
| 延迟卖出 | `data/paper/pending_sells/pending_sells.json` | 清空（写入 `[]`） |
| 调仓状态 | `data/paper/runs/rebalance_state.json` | 智能回滚 |

### rebalance_state 回滚逻辑
调仓状态的回滚逻辑较复杂：

**条件1**：`last_rebalance_date >= cut_off_date`（需要回滚）
- 查找 cut_off 之前最近的 `t0_YYYYMMDD.json` 记录
- 将 `last_rebalance_date` 回滚到该日期
- 若找不到，则删除 `rebalance_state.json`（视为首次运行）

**条件2**：`last_rebalance_date < cut_off_date`（无需回滚）
- 保持不变

**示例**：
```
场景：cut_off_date = 20260210
现有 t0 记录：20260205, 20260208, 20260210, 20260212
当前 last_rebalance_date = 20260212

回滚后：
- 删除 t0_20260210.json, t0_20260212.json
- last_rebalance_date 回滚到 20260208
```

---

## 注意事项

### 1. 修正顺序
建议按以下顺序修正：
1. 删除/更新持仓（`delete-position`, `update-position`）
2. 加仓（`add-shares`）
3. 调整现金（`cash`）

**原因**：避免因现金不足导致加仓失败。

### 2. Cut-off 日期选择
- **建议**：选择需要重新运行的第一个日期
- **避免**：选择过早的日期（会清理大量数据）
- **最佳实践**：选择最近一次出错的日期

### 3. 幂等性保障
- `adjust` 命令会清理幂等性记录（`runs/` 目录）
- 重新运行 `run` 命令不会因"已执行"而阻塞
- 可多次运行 `adjust` 进行反复调试

### 4. 备份建议
⚠️ **重要**：修正操作不可逆，建议修正前备份数据：
```bash
# 备份整个 paper 目录
cp -r data/paper data/paper.backup.$(date +%Y%m%d%H%M%S)
```

### 5. 验证修正结果
修正后，建议：
1. 查看日志输出确认操作成功
2. 检查 `account.json` 的 `last_update` 字段
3. 使用 `positions` 命令查看持仓
4. 重新运行 `run` 命令验证数据一致性

---

## 故障排查

### 问题1：提示持仓不存在
**错误**：
```
持仓 600519.SH 不存在，无法删除/更新/加仓
```

**解决**：
- `delete-position` / `update-position` / `add-shares`：检查 `ts_code` 是否正确
- `add-shares`：若需新建持仓，改用 `update-position`

### 问题2：现金不足
**错误**：
```
现金不足：需要 100000.00，可用 50000.00
```

**解决**：
- 先使用 `cash` 命令调整现金
- 或减少加仓股数

### 问题3：清理后无法重新运行
**症状**：执行 `run` 命令后提示"已执行过"

**排查**：
- 检查 `data/paper/runs/` 目录是否仍有 >= cut_off 日期的文件
- 检查 `account.json` 的 `last_update` 是否为 cut_off 日期
- 手工删除残留的 runs 文件

### 问题4：调仓状态异常
**症状**：重新运行后调仓逻辑不符合预期

**排查**：
- 检查 `rebalance_state.json` 的 `last_rebalance_date`
- 确认是否正确回滚到 cut_off 之前的日期
- 若有问题，手工编辑 `rebalance_state.json` 或删除该文件

---

## 版本兼容性

### v0.4.0+ (当前版本)
- ✅ 完全支持 `adjust` 命令
- ✅ 仅使用 instructions（废弃 pending_weights）
- ✅ 智能清理和回滚

### v0.3.x (旧版本)
- ❌ 不支持 `adjust` 命令
- ⚠️ 使用 pending_weights（已废弃）
- ⚠️ 升级到 v0.4.0 后，旧的 pending 数据不再被读取

**升级建议**：
- 在升级前完成所有待执行的 pending_weights
- 或从最后一次成功的 T1 日期重新运行

---

## 最佳实践

1. **定期备份**：每天运行前备份 `data/paper/` 目录
2. **小步快跑**：修正后立即验证，避免连环错误
3. **日志审查**：仔细阅读 `adjust` 命令的日志输出
4. **版本管理**：将 `data/paper/` 目录加入 `.gitignore`，避免误提交
5. **文档记录**：记录每次修正的原因和操作，便于后续审计

---

## 相关文档
- [纸面交易主指南](../paper_trading_guide.md)
- [指令驱动 T0/T1 执行模式](../PR/指令驱动T0T1执行模式.md)
- [PR: 手工修正回滚功能](../PR/手工修正回滚功能.md)

---

**文档版本**: v1.0  
**更新日期**: 2026-02-10
