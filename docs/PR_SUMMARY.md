# 纸面交易增强功能实现总结

## 概述

本PR完成了LazyBull纸面交易模块的全面增强，针对问题陈述中的5个关键需求进行了逐项实现和验证。

## 实现的功能

### 1. Runner数据下载重构 ✅

**问题**：现有paper runner中存在独立实现的download逻辑，重复造轮子。

**解决方案**：
- 移除了`_download_data()`中的自建下载逻辑
- 改为调用仓库既有的`TushareClient`进行raw数据下载
- 改为调用仓库既有的`DataCleaner`进行clean数据构建
- 按项目标准的partitioned存储格式生成raw与clean文件
- 新增下载停复牌（suspend）、涨跌停（stk_limit）信息
- 自动添加可交易性标记（`is_suspended`、`is_limit_up`、`is_limit_down`、`tradable`等）

**修改文件**：
- `src/lazybull/paper/runner.py`: 重构`_download_data()`方法

**关键代码**：
```python
# 1. 下载raw数据（复用TushareClient）
daily_data = self.client.get_daily(trade_date=trade_date)
suspend = self.client.get_suspend_d(trade_date=trade_date)
limit_up_down = self.client.get_stk_limit(trade_date=trade_date)

# 2. 构建clean数据（复用DataCleaner）
cleaner = DataCleaner()
daily_clean = cleaner.clean_daily(daily_raw, adj_factor_raw)
daily_clean = cleaner.add_tradable_universe_flag(
    daily_clean, stock_basic, suspend_info_df, limit_info_df
)
```

### 2. weight_method参数支持 ✅

**问题**：MLSignal未接收weight_method参数，永远使用默认equal权重。

**解决方案**：
- 在`PaperTradingRunner`中添加`weight_method`参数
- 在`run_t0()`中传递`weight_method`到`MLSignal`
- 更新CLI添加`--weight-method`参数（equal/score可选）
- 更新帮助文本说明参数用途

**修改文件**：
- `src/lazybull/paper/runner.py`: 添加weight_method参数和传递逻辑
- `scripts/paper_trade.py`: 添加CLI参数

**使用示例**：
```bash
# 使用等权分配（默认）
python scripts/paper_trade.py t0 --trade-date 20260121 --weight-method equal

# 使用按分数加权
python scripts/paper_trade.py t0 --trade-date 20260121 --weight-method score
```

### 3. T1执行涨跌停/停牌判断 ✅

**问题**：T1执行订单时未判断涨跌停、停牌等可交易性。

**解决方案**：
- 在`broker.generate_orders()`中添加可交易性检查
- 从clean层daily数据读取可交易性标记
- 实现交易规则：
  - 停牌：不可买入或卖出，订单跳过
  - 涨停：不可买入，订单跳过
  - 跌停：不可卖出，订单延迟
- 打印详细的不可交易原因
- 新增辅助方法：`_load_tradability_info()`、`_check_can_buy()`、`_check_can_sell()`

**修改文件**：
- `src/lazybull/paper/broker.py`: 添加可交易性检查逻辑

**关键代码**：
```python
# 检查买入可交易性
can_buy, buy_reason = self._check_can_buy(ts_code, tradability)
if not can_buy:
    logger.warning(f"股票 {ts_code} 不可买入: {buy_reason}，跳过订单")
    continue

# 检查卖出可交易性
can_sell, sell_reason = self._check_can_sell(ts_code, tradability)
if not can_sell:
    logger.warning(f"股票 {ts_code} 不可卖出: {sell_reason}，订单延迟")
    continue
```

**日志示例**：
```
WARNING  股票 000001.SZ 不可买入: 涨停，跳过订单
WARNING  股票 000002.SZ 不可卖出: 跌停，订单延迟
WARNING  股票 000003.SZ 不可买入: 停牌，跳过订单
```

### 4. _load_prices()分开盘/收盘价 ✅

**问题**：现有`_load_prices()`只返回单一价格口径，不支持open/close分开。

**解决方案**：
- 改造`_load_prices()`支持返回(buy_prices, sell_prices)元组
- 支持open和close价格分开加载
- 在`run_t1()`中分别传递buy_price和sell_price字典
- 更新`broker.generate_orders()`接收分别的买卖价格字典
- 添加open价格缺失时的降级策略（fallback到close并打印warning）
- 新增`DataLoader.load_clean_daily_by_date()`方法

**修改文件**：
- `src/lazybull/paper/runner.py`: 重构`_load_prices()`方法
- `src/lazybull/paper/broker.py`: 更新`generate_orders()`签名
- `src/lazybull/data/loader.py`: 新增`load_clean_daily_by_date()`方法
- `scripts/paper_trade.py`: 支持--sell-price参数

**支持的价格组合**：
```bash
# T1开盘卖出，收盘买入（推荐）
python scripts/paper_trade.py t1 --trade-date 20260122 --buy-price close --sell-price open

# T1开盘买入，收盘卖出
python scripts/paper_trade.py t1 --trade-date 20260122 --buy-price open --sell-price close

# 全部使用收盘价（默认）
python scripts/paper_trade.py t1 --trade-date 20260122 --buy-price close --sell-price close
```

### 5. Broker持仓/收益查询增强 ✅

**问题**：缺乏便于查看的持仓明细、收益和持仓状态能力。

**解决方案**：
- 在`PaperBroker`中添加`get_positions_detail()`方法
- 显示详细持仓信息：股票、股数、成本、当前价、浮盈、收益率
- 新增`print_positions_summary()`方法输出格式化的持仓明细
- 在CLI添加positions子命令查看持仓
- 在Position模型中添加status和notes字段
- 新增`get_holding_days()`方法计算持有天数
- 更新`PaperAccount`以支持新的Position字段
- 确保持仓信息可从持久化状态恢复

**修改文件**：
- `src/lazybull/paper/broker.py`: 添加持仓查询方法
- `src/lazybull/paper/models.py`: 增强Position模型
- `src/lazybull/paper/account.py`: 更新add_position()和reduce_position()
- `scripts/paper_trade.py`: 添加positions子命令

**使用示例**：
```bash
python scripts/paper_trade.py positions --trade-date 20260122
```

**输出示例**：
```
================================================================================
持仓明细
================================================================================
股票代码       股数     买入均价     买入成本     买入日期     持有天数  当前价格     当前市值       浮盈         收益率(%)    状态    
--------------------------------------------------------------------------------
000001.SZ    1000     12.50      25.00      20260115   7       13.20      13200.00     450.00     3.53       持有    
000002.SZ    800      15.30      20.40      20260116   6       15.80      12640.00     379.60     3.09       持有    
--------------------------------------------------------------------------------
合计         1800                45.40                         25840.00     829.60     3.32      
================================================================================
账户现金: 475,134.00
持仓市值: 25,840.00
总资产: 500,974.00
================================================================================
```

## 测试与验证

### 单元测试

新增了以下单元测试（位于`tests/test_paper_trading.py`）：

1. `test_position_with_status()`: 测试Position模型的status和notes字段
2. `test_position_holding_days()`: 测试持有天数计算
3. `test_broker_get_positions_detail()`: 测试获取持仓明细
4. `test_broker_generate_orders_with_separate_prices()`: 测试使用分开的买卖价格生成订单
5. `test_broker_check_can_buy()`: 测试买入可交易性检查
6. `test_broker_check_can_sell()`: 测试卖出可交易性检查

同时修复了旧测试以适配新的`broker.generate_orders()`签名。

### 语法验证

所有Python文件已通过py_compile语法检查：
```bash
python -m py_compile src/lazybull/paper/*.py scripts/paper_trade.py tests/test_paper_trading.py
```

## 文档更新

更新了`docs/paper_trading_guide.md`，新增或修改了以下章节：

1. **T0工作流参数说明**：添加了weight_method参数和数据下载机制变更说明
2. **T1工作流参数说明**：添加了价格口径支持和可交易性检查说明
3. **持仓查询**：新增完整的positions子命令使用说明
4. **账户状态JSON示例**：更新了Position字段

## 代码统计

总体变更：
```
8 files changed, 725 insertions(+), 68 deletions(-)
```

具体变更：
- `docs/paper_trading_guide.md`: +71/-2（文档更新）
- `scripts/paper_trade.py`: +56/-1（CLI增强）
- `src/lazybull/data/loader.py`: +24/0（新增方法）
- `src/lazybull/paper/account.py`: +18/-2（支持新字段）
- `src/lazybull/paper/broker.py`: +232/-6（核心增强）
- `src/lazybull/paper/models.py`: +19/0（模型增强）
- `src/lazybull/paper/runner.py`: +200/-15（重构优化）
- `tests/test_paper_trading.py`: +173/-2（测试增强）

## 技术亮点

1. **复用既有能力**：完全复用了仓库的TushareClient、DataCleaner、DataLoader等既有模块，避免重复造轮子
2. **标准化存储**：数据按项目标准的partitioned格式保存，与既有数据架构完美兼容
3. **灵活配置**：支持多种权重方法和价格口径组合，满足不同交易策略需求
4. **健壮性保障**：完善的可交易性检查和价格降级策略，提高系统健壮性
5. **可观测性增强**：详细的日志输出和持仓查询能力，便于监控和诊断
6. **向后兼容**：通过默认参数保持与旧代码的兼容性

## 使用建议

### 完整的交易周期示例

```bash
# Day 1: T0工作流 - 生成信号（使用按分数加权）
python scripts/paper_trade.py t0 \
  --trade-date 20260121 \
  --buy-price close \
  --weight-method score \
  --model-version 1

# Day 2: T1工作流 - 执行订单（开盘卖出，收盘买入）
python scripts/paper_trade.py t1 \
  --trade-date 20260122 \
  --buy-price close \
  --sell-price open

# Day 2: 查看持仓
python scripts/paper_trade.py positions \
  --trade-date 20260122
```

### 注意事项

1. **数据完整性**：确保已下载必要的基础数据（trade_cal、stock_basic）
2. **可交易性标记**：系统依赖clean层的可交易性标记，首次运行会自动构建
3. **价格降级**：当open价格缺失时会自动降级到close，注意查看warning日志
4. **订单延迟**：跌停导致的卖出订单会被延迟，可在后续交易日重试

## 后续优化方向

虽然当前实现已满足所有需求，但以下功能可在未来版本中进一步优化：

1. **延迟订单队列**：持久化跌停导致的延迟卖出订单，实现自动重试机制
2. **持仓状态扩展**：添加距离计划卖出时间、止损触发状态等高级字段
3. **风控增强**：添加最大持仓限制、单日最大交易次数等风控规则
4. **报表增强**：生成持仓收益曲线、交易明细报表、策略表现分析等

## 参考资料

- [纸面交易使用指南](docs/paper_trading_guide.md)
- [数据清洗模块](src/lazybull/data/cleaner.py)
- [信号生成模块](src/lazybull/signals/ml_signal.py)
- [单元测试](tests/test_paper_trading.py)
