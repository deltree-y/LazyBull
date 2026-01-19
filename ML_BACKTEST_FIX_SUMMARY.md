# ML 回测信号生成逻辑修复说明

## 问题描述

原先 `scripts/run_ml_backtest.py` 中的 `BacktestEngineML` 类完全重写了 `_generate_signal` 方法，导致父类 `BacktestEngine` 中的以下增强逻辑全部失效：

- ❌ T+1 可交易性检查（停牌、涨停等）
- ❌ 候选股票回填到 top_n（当前面的候选不可交易时）
- ❌ 过滤原因统计（停牌数、涨停数、跌停数）
- ❌ 权重归一化和等权兜底
- ❌ 详细的 verbose 日志输出

## 解决方案

采用 **扩展点（Hook）模式**，让 ML 回测引擎复用父类的完整信号生成逻辑，只注入特征数据。

### 1. 在 BacktestEngine 中添加扩展点

```python
def _build_signal_data(self, date: pd.Timestamp) -> Optional[Dict]:
    """构建传递给信号生成器的额外数据（扩展点）
    
    子类可以重写此方法以注入特定数据（如 ML 特征）。
    返回 None 表示该日期无可用数据，将跳过信号生成。
    """
    return {}
```

### 2. 修改 _generate_signal 调用扩展点

```python
def _generate_signal(self, ...):
    # 调用扩展点获取额外数据（如 ML 特征）
    extra_data = self._build_signal_data(date)
    if extra_data is None:
        # None 表示该日期无可用数据，跳过信号生成
        logger.warning(f"信号日 {date.date()} 无可用数据，跳过")
        return
    
    # 合并默认数据和额外数据
    signal_data = {}
    signal_data.update(extra_data)
    
    # 生成排序后的候选列表
    ranked_candidates = self.signal.generate_ranked(date, stock_universe, signal_data)
    
    # ... 后续过滤、回填、归一化逻辑保持不变 ...
```

### 3. 创建独立的 BacktestEngineML 模块

**新文件：`src/lazybull/backtest/engine_ml.py`**

```python
class BacktestEngineML(BacktestEngine):
    """支持 ML 信号的回测引擎
    
    通过重写 _build_signal_data 方法注入特征数据，
    其他回测逻辑复用父类实现。
    """
    
    def __init__(self, features_by_date: Dict[str, pd.DataFrame], **kwargs):
        super().__init__(**kwargs)
        self.features_by_date = features_by_date
    
    def _build_signal_data(self, date: pd.Timestamp) -> Optional[Dict]:
        """构建信号数据（注入 ML 特征）"""
        date_str = date.strftime('%Y%m%d')
        features_df = self.features_by_date.get(date_str)
        
        if features_df is None or len(features_df) == 0:
            logger.warning(f"信号日 {date.date()} 没有特征数据，跳过")
            return None
        
        return {"features": features_df}
```

**关键点**：
- 仅重写 `_build_signal_data` 方法（最小化修改）
- 不再重写 `_generate_signal`，复用父类逻辑
- 特征缺失时返回 None，父类会自动跳过

### 4. 更新 run_ml_backtest.py

```python
# 旧代码：脚本内定义 BacktestEngineML（52行，完全重写 _generate_signal）
class BacktestEngineML(BacktestEngine):
    def _generate_signal(self, ...):
        # ... 52 行代码 ...

# 新代码：从模块导入（1行）
from src.lazybull.backtest import BacktestEngineML
```

## 效果验证

### 日志输出对比

**修复前**（只有简单的信号生成日志）：
```
信号生成: 2023-06-01, 信号数 3
```

**修复后**（显示完整的过滤统计）：
```
ML 排序候选生成完成: 2023-06-01, 候选数 5, 平均预测分数=1.020000
信号生成: 2023-06-01, 信号数 3/3, 检查候选 3 个, 过滤: 停牌 0, 涨停 0, 跌停 0
买入执行: 2023-06-02, 买入 3 只股票（信号日: 2023-06-01）
```

### 测试结果

所有测试通过（40个测试）：
- ✅ ML 回测引擎测试：3/3
- ✅ ML 信号测试：10/10
- ✅ 回测 T1 测试：3/3
- ✅ 价格分离测试：6/6
- ✅ 调仓频率测试：8/8
- ✅ 延迟订单测试：10/10

## 使用方式

无需修改调用代码，使用方式完全不变：

```python
from src.lazybull.backtest import BacktestEngineML

# 创建 ML 回测引擎
engine = BacktestEngineML(
    features_by_date=features_by_date,  # 按日期组织的特征数据
    universe=universe,
    signal=ml_signal,
    initial_capital=100000.0,
    rebalance_freq=10,
    verbose=True  # 开启详细日志，能看到过滤统计
)

# 运行回测
nav_curve = engine.run(
    start_date=start_date,
    end_date=end_date,
    trading_dates=trading_dates,
    price_data=price_data
)
```

## 文件变更清单

1. **src/lazybull/backtest/engine.py**
   - 添加 `_build_signal_data()` 扩展点方法
   - 修改 `_generate_signal()` 调用扩展点

2. **src/lazybull/backtest/engine_ml.py**（新增）
   - 定义 `BacktestEngineML` 类
   - 仅重写 `_build_signal_data()` 注入特征

3. **src/lazybull/backtest/__init__.py**
   - 导出 `BacktestEngineML`

4. **scripts/run_ml_backtest.py**
   - 删除本地 `BacktestEngineML` 类定义（-52 行）
   - 从模块导入 `BacktestEngineML`（+1 行）

5. **tests/test_ml_backtest_engine.py**（新增）
   - 添加 ML 回测引擎测试

6. **tests/test_backtest_price_separation.py**
   - 修复测试 mock 兼容性

## 总结

通过引入扩展点模式，实现了：

✅ **最小化修改**：子类只需重写 1 个方法（_build_signal_data）  
✅ **逻辑复用**：完全复用父类的过滤/回填/归一化逻辑  
✅ **代码组织**：将 ML 引擎移到独立模块，脚本更简洁  
✅ **向后兼容**：调用方式不变，无需修改现有代码  
✅ **可扩展性**：未来其他类型的回测引擎也可用同样方式扩展  

问题已完全解决！
