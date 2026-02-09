# PR: 修复停牌信息判断，避免止损/交易误判

## 问题背景

### 问题描述
当前系统在止损检查和交易执行时，从 `clean/daily` 数据中获取停牌信息，存在以下问题：

1. **停牌信息不可靠**：`daily` 数据中的 `is_suspended` 字段可能不准确或缺失
2. **停牌股票误触止损**：当停牌股票在 daily 中缺行或价格为 0 时，可能被错误触发止损
3. **卖出静默失败**：调仓/执行卖出时，若 `sell_prices` 缺失（停牌导致），既不卖出也不进入 `PendingSell` 延迟队列
4. **回测与纸面交易不一致**：回测引擎也存在类似问题，需要与纸面交易策略对齐

### 停牌数据来源
根据 Tushare 数据源，停牌信息应从 `raw/suspend` 数据获取：
- 字段：`ts_code`, `trade_date`, `suspend_type`, `suspend_timing`
- `suspend_type`：
  - `'S'`：停牌（Suspend）
  - `'R'`：复牌/恢复交易（Resume）
- `trade_date`：YYYYMMDD 格式

## 解决方案

### A. 统一停牌判断能力

**文件：`src/lazybull/common/trade_status.py`**

1. 新增函数 `is_suspended_by_suspend_df()`
   - 基于 `suspend_df` 数据判断停牌状态
   - 规则：
     - `suspend_type == 'S'` → 停牌 `True`
     - `suspend_type == 'R'` → 复牌 `False`
     - 无记录 → 未停牌 `False`

2. 扩展 `is_suspended()` 函数
   - 新增可选参数 `suspend_df`
   - 优先级：`suspend_df` > `quote_data.is_suspended` > 成交量判断
   - 最后降级：quote_data 缺行时假定停牌（保守策略）

3. 更新 `is_tradeable()` 和 `get_trade_status_info()`
   - 支持传入 `suspend_df` 参数
   - 确保停牌判断使用统一逻辑

**文件：`src/lazybull/data/loader.py`**

4. 新增方法 `load_suspend_by_date()`
   - 从 `raw/suspend` 加载指定日期的停牌数据
   - 确保日期格式统一为 YYYYMMDD

### B. 纸面交易行为修复

**文件：`scripts/paper_trade.py`**

1. 修改 `_check_stop_loss()` 函数
   - 加载当日 `suspend_data`（通过 `DataLoader.load_suspend_by_date()`）
   - 对每个持仓：
     - 若当日停牌 → 记录日志并跳过止损检查
     - 若非停牌但无行情（daily 缺行/无 close）→ 跳过止损检查并输出"无行情数据"日志
     - 若价格为 0 或 NaN → 跳过止损检查

**文件：`src/lazybull/paper/broker.py`**

2. 修改 `_load_tradability_info()` 方法
   - 加载当日 `suspend_data`
   - 使用统一的 `is_suspended()` 函数判断停牌状态
   - 确保 `tradability` 字典中的 `is_suspended` 字段基于 suspend 数据

3. 现有卖出逻辑已确保进入 PendingSell
   - `generate_orders()` 和 `execute_instructions()` 中的卖出逻辑：
     - 已有"无卖出价格数据时创建 PendingSell"的机制
     - 已根据 `tradability` 判断停牌原因并添加相应标记

### C. 回测引擎策略对齐

**文件：`src/lazybull/backtest/engine.py`**

1. 扩展 `run()` 方法
   - 新增可选参数 `suspend_data`
   - 缓存 `suspend_data` 用于后续止损和可交易性检查

2. 修改 `_check_stop_loss()` 方法
   - 从缓存的 `suspend_data` 中筛选当日数据
   - 使用统一的 `is_suspended()` 函数判断停牌
   - 停牌股票跳过止损检查
   - 价格为 None 或 <= 0 时跳过止损检查

3. 延迟卖出机制
   - 现有 `pending_stop_loss_sells` 队列已处理延迟卖出
   - 止损触发但跌停/停牌时，进入延迟队列
   - 后续交易日重试执行

### D. 测试覆盖

**文件：`tests/test_trade_status.py`**

新增以下测试用例：
1. `test_is_suspended_by_suspend_df_suspended`：测试 suspend_type='S' 判断为停牌
2. `test_is_suspended_by_suspend_df_resumed`：测试 suspend_type='R' 判断为复牌
3. `test_is_suspended_by_suspend_df_no_record`：测试无记录判断为未停牌
4. `test_is_suspended_with_suspend_df_priority`：测试优先使用 suspend_df
5. `test_is_suspended_fallback_to_quote_data`：测试回退到 quote_data
6. `test_is_tradeable_with_suspend_df`：测试可交易性判断
7. `test_get_trade_status_info_with_suspend_df`：测试交易状态信息获取

更新现有测试：
- `test_is_suspended_missing_data`：调整为保守策略（缺失数据假定停牌）
- `test_get_trade_status_info_missing`：同上

### E. 版本号与文档

1. 版本号递增至 `0.3.15`（`pyproject.toml`）
2. 新增本 PR 说明文档（`docs/PR/fix_stop_loss_suspend_check.md`）

## 验证步骤

### 1. 运行单元测试
```bash
cd /path/to/LazyBull
python -m pytest tests/test_trade_status.py -v
```

预期结果：所有 27 个测试通过

### 2. 验证纸面交易止损
创建测试场景：
- 持仓股票在某日停牌（`raw/suspend` 中 `suspend_type='S'`）
- 该股票当日 `daily` 数据缺行或 close 为 0
- 运行纸面交易止损检查

预期行为：
- 输出日志："股票 XXX 停牌，跳过止损检查"
- 不触发止损

### 3. 验证纸面交易卖出
创建测试场景：
- 调仓需要卖出某股票
- 该股票停牌或 `sell_prices` 中缺失

预期行为：
- 创建 `PendingSell` 记录
- reason 包含"（停牌）"或"（无价格数据）"
- 复牌后通过 `retry_pending_sells()` 重试

### 4. 验证回测引擎止损
创建测试场景：
- 回测时传入 `suspend_data`
- 持仓股票在某日停牌

预期行为：
- 输出日志："股票 XXX 停牌，跳过止损检查"
- 不触发止损

## 技术细节

### 停牌判断优先级
```
1. suspend_df 存在 → 使用 suspend_type 判断
   - 'S' → 停牌
   - 'R' → 复牌（未停牌）
   - 无记录 → 未停牌

2. suspend_df 不存在 → 回退到 quote_data
   - 检查 is_suspended 字段
   - 检查成交量（vol <= 0 视为停牌）
   - 缺行 → 假定停牌（保守策略）
```

### 数据流程
```
纸面交易：
  DataLoader.load_suspend_by_date(trade_date)
    → Storage.load_raw_by_date("suspend", date_str)
    → is_suspended(ts_code, trade_date, daily_data, suspend_df)

回测：
  engine.run(..., suspend_data=suspend_df)
    → self.suspend_data_cache = suspend_data
    → _check_stop_loss() 使用 suspend_data_cache
    → is_suspended(stock, trade_date_str, date_quote, suspend_df_today)
```

### 向后兼容性
- `is_suspended()`, `is_tradeable()`, `get_trade_status_info()` 的 `suspend_df` 参数为可选
- 不传入时行为与之前一致（从 quote_data 判断）
- 回测引擎的 `suspend_data` 参数为可选，不传入时行为与之前一致

### 日志输出
所有关键路径均输出中文日志：
- 停牌跳过止损：`"股票 {ts_code} 停牌，跳过止损检查"`
- 无价格跳过止损：`"股票 {ts_code} 价格数据缺失或为0，跳过止损检查"`
- 停牌无法卖出：`"股票 {ts_code} 停牌，无法卖出，加入延迟卖出队列"`
- 无价格无法卖出：`"股票 {ts_code} 无卖出价格数据，加入延迟卖出队列"`

## 注意事项

1. **数据准备**：确保 `raw/suspend` 数据已下载
   - 可能需要运行 `scripts/download_raw.py` 下载停牌数据

2. **历史数据**：对于历史回测，需要提供完整的 `suspend_data`
   - 建议在回测脚本中加载完整的停牌数据范围

3. **保守策略**：当既无 suspend_df 又无 quote_data 时，假定停牌
   - 这是为了避免对停牌股票的错误操作
   - 若影响正常交易，请确保数据完整性

4. **性能考虑**：suspend_data 相对较小，对性能影响可忽略
   - 纸面交易每次加载当日数据，IO 开销很小
   - 回测一次性加载全量数据并缓存，无重复 IO

## 相关文档

- [停牌数据说明](../data_contract.md#停牌数据)
- [纸面交易指南](../paper_trading_guide.md)
- [回测假设说明](../backtest_assumptions.md)
- [交易状态检查指南](../trade_status_guide.md)
