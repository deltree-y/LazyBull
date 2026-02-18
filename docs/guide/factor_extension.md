# 如何扩展因子与技术指标

本指南说明如何在 LazyBull 中添加自定义因子和技术指标。

---

## 架构概述

LazyBull 的因子计算采用模块化设计：

```
src/lazybull/
├── factors/                    # 因子库（可复用）
│   ├── __init__.py
│   ├── technical_indicators.py # 技术指标
│   ├── candlestick.py          # K线形态
│   ├── volatility.py           # 波动率
│   ├── industry.py             # 行业相关
│   ├── momentum.py             # 动量
│   └── volume.py               # 量能
└── features/
    └── builder.py              # 特征构建器（调用因子库）
```

### 设计原则

1. **职责分离**: `factors/` 负责因子计算，`features/builder.py` 负责数据流编排
2. **纯函数**: 因子计算函数接受 DataFrame 输入，返回 DataFrame 输出
3. **可复用**: 因子函数可在任何场景使用（特征生成、因子分析、回测等）
4. **易测试**: 每个因子函数可独立单元测试

---

## 添加新因子的步骤

### 步骤1: 确定因子分类

选择合适的模块文件或创建新模块：

- 技术指标（MA, RSI, MACD等） → `technical_indicators.py`
- K线形态（上影线、十字星等） → `candlestick.py`
- 波动率（ATR, 历史波动率等） → `volatility.py`
- 行业相关（行业轮动、行业强度等） → `industry.py`
- 动量（惯性、反转等） → `momentum.py`
- 量能（放量、缩量等） → `volume.py`
- 其他类别 → 创建新模块

### 步骤2: 实现因子计算函数

在对应模块中添加函数，遵循以下规范：

```python
def calculate_your_factor(df: pd.DataFrame, param1: int, param2: float) -> pd.DataFrame:
    """计算您的因子
    
    Args:
        df: DataFrame，需包含 ts_code, trade_date, 其他必需列
        param1: 参数1的说明
        param2: 参数2的说明
        
    Returns:
        DataFrame，包含 ts_code, trade_date, your_factor_col
    """
    result = df[['ts_code', 'trade_date']].copy()
    
    # 因子计算逻辑
    # ...
    
    return result
```

#### 示例1: 简单因子（当日计算）

```python
def calculate_price_position(df: pd.DataFrame) -> pd.DataFrame:
    """计算价格位置
    
    价格位置 = (收盘价 - 最低价) / (最高价 - 最低价)
    
    Args:
        df: DataFrame，需包含 ts_code, trade_date, open_adj, high_adj, low_adj, close_adj
        
    Returns:
        DataFrame，包含 ts_code, trade_date, price_position
    """
    result = df[['ts_code', 'trade_date']].copy()
    
    # 计算价格位置
    result['price_position'] = np.where(
        df['high_adj'] - df['low_adj'] > 1e-6,
        (df['close_adj'] - df['low_adj']) / (df['high_adj'] - df['low_adj']),
        np.nan
    )
    
    return result
```

#### 示例2: 需要历史数据的因子

```python
def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """计算ATR（平均真实波幅）
    
    Args:
        df: DataFrame，需包含 ts_code, trade_date, high_adj, low_adj, close_adj
        window: ATR 窗口，默认14
        
    Returns:
        DataFrame，包含 ts_code, trade_date, atr_{window}
    """
    result = df[['ts_code', 'trade_date']].copy()
    
    # 按股票分组计算
    grouped = df.sort_values(['ts_code', 'trade_date']).groupby('ts_code')
    
    atr_values = []
    for ts_code, group in grouped:
        group = group.sort_values('trade_date').copy()
        
        # 计算真实波幅 TR
        high_low = group['high_adj'] - group['low_adj']
        high_close = np.abs(group['high_adj'] - group['close_adj'].shift(1))
        low_close = np.abs(group['low_adj'] - group['close_adj'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # 计算 ATR（TR 的移动平均）
        atr = tr.rolling(window=window, min_periods=window).mean()
        
        temp_df = pd.DataFrame({
            'ts_code': ts_code,
            'trade_date': group['trade_date'].values,
            f'atr_{window}': atr.values
        })
        atr_values.append(temp_df)
    
    if atr_values:
        result = pd.concat(atr_values, ignore_index=True)
    else:
        result[f'atr_{window}'] = np.nan
    
    return result
```

### 步骤3: 在 `__init__.py` 中导出

编辑 `src/lazybull/factors/__init__.py`，添加导出：

```python
from .your_module import (
    calculate_your_factor,
)

__all__ = [
    # ... 其他导出
    'calculate_your_factor',
]
```

### 步骤4: 在 FeatureBuilder 中注册

编辑 `src/lazybull/features/builder.py`：

#### 4.1 添加导入

```python
from ..factors import (
    # ... 其他导入
    calculate_your_factor,
)
```

#### 4.2 在 `_add_advanced_factors` 中调用

```python
def _add_advanced_factors(self, ...):
    result = features.copy()
    
    # ... 其他因子计算
    
    # 添加您的因子
    if all(col in current_data.columns for col in ['必需列1', '必需列2']):
        your_factor_df = calculate_your_factor(current_data, param1=value1)
        result = result.merge(your_factor_df, on=['ts_code', 'trade_date'], how='left')
    
    return result
```

### 步骤5: 编写单元测试

在 `tests/` 目录创建或编辑测试文件：

```python
def test_your_factor_exists(
    mock_daily_data_with_ohlc,
    mock_adj_factor_extended,
    mock_trade_cal_extended,
    mock_stock_basic_with_industry
):
    """测试您的因子存在"""
    builder = FeatureBuilder(min_list_days=0, require_label=False)
    
    trading_dates = builder._get_trading_dates(mock_trade_cal_extended)
    trade_date = trading_dates[30]
    
    result = builder.build_features_for_day(
        trade_date=trade_date,
        trade_cal=mock_trade_cal_extended,
        daily_data=mock_daily_data_with_ohlc,
        adj_factor=mock_adj_factor_extended,
        stock_basic=mock_stock_basic_with_industry
    )
    
    assert 'your_factor_col' in result.columns
```

运行测试：
```bash
python -m pytest tests/test_new_features.py -v
```

---

## 最佳实践

### 1. 向量化计算

尽量使用 pandas/numpy 的向量化操作，避免循环：

❌ **不推荐**（逐行循环）:
```python
for i, row in df.iterrows():
    result.at[i, 'factor'] = row['a'] / row['b']
```

✅ **推荐**（向量化）:
```python
result['factor'] = df['a'] / df['b']
```

### 2. 处理缺失值和除零

```python
# 除零保护
result['factor'] = np.where(
    df['denominator'] > 1e-6,
    df['numerator'] / df['denominator'],
    np.nan
)
```

### 3. 按股票分组计算

对于需要历史数据的因子，按 `ts_code` 分组：

```python
grouped = df.sort_values(['ts_code', 'trade_date']).groupby('ts_code')

for ts_code, group in grouped:
    group = group.sort_values('trade_date').copy()
    # 在 group 上计算
```

### 4. 保持输出格式一致

始终返回包含 `ts_code` 和 `trade_date` 的 DataFrame。

### 5. 添加类型注解和文档字符串

```python
def calculate_factor(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """简洁的一句话描述
    
    详细说明（可选）
    
    Args:
        df: 输入DataFrame及其必需列
        window: 参数说明
        
    Returns:
        输出DataFrame及其包含的列
    """
    pass
```

---

## 常见场景

### 场景1: 需要行业信息的因子

如果因子需要行业信息，使用 `industry.py` 中的工具：

```python
from ..factors.industry import add_industry_features

def calculate_industry_momentum(df: pd.DataFrame, stock_basic: pd.DataFrame) -> pd.DataFrame:
    # 先添加行业信息
    df_with_industry = add_industry_features(df, stock_basic, ret_col='ret_1')
    
    # 然后计算行业级别的因子
    # ...
```

### 场景2: 需要历史窗口数据的因子

在 `_add_advanced_factors` 中，使用 `daily_adj` 获取历史数据：

```python
lookback = 50
hist_start_date = trading_dates[max(0, current_idx - lookback)]
hist_dates = [d for d in trading_dates if hist_start_date <= d <= trade_date]
hist_data = daily_adj[daily_adj['trade_date'].isin(hist_dates)].copy()

your_factor_df = calculate_your_factor(hist_data, window=20)
# 只保留当日
your_factor_today = your_factor_df[your_factor_df['trade_date'] == trade_date]
result = result.merge(your_factor_today, on=['ts_code', 'trade_date'], how='left')
```

### 场景3: 需要截面标准化的因子

```python
def calculate_cross_sectional_factor(df: pd.DataFrame) -> pd.DataFrame:
    result = df[['ts_code', 'trade_date']].copy()
    
    # 按交易日分组，做截面 zscore
    result['factor_zscore'] = df.groupby('trade_date')['raw_factor'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    
    return result
```

---

## 调试技巧

### 1. 单独测试因子函数

```python
from src.lazybull.factors import calculate_your_factor
import pandas as pd

# 构造小型测试数据
test_df = pd.DataFrame({
    'ts_code': ['000001.SZ'] * 10,
    'trade_date': ['2023010' + str(i) for i in range(1, 11)],
    'close_adj': [10 + i * 0.1 for i in range(10)],
    # ... 其他必需列
})

result = calculate_your_factor(test_df, param=value)
print(result)
```

### 2. 检查中间结果

在因子函数中添加打印语句：

```python
def calculate_factor(df: pd.DataFrame) -> pd.DataFrame:
    result = df[['ts_code', 'trade_date']].copy()
    
    # 中间步骤
    intermediate = df['a'] / df['b']
    print(f"Intermediate stats: min={intermediate.min()}, max={intermediate.max()}, mean={intermediate.mean()}")
    
    result['factor'] = intermediate
    return result
```

### 3. 逐步验证

先在小范围日期测试，确认无误后再扩展到全部历史：

```bash
# 只生成一天的特征
python scripts/ensure_features.py --start-date 20231201 --end-date 20231201
```

---

## 示例：添加威廉指标 (%R)

完整示例，从零开始添加威廉指标。

### 1. 在 `technical_indicators.py` 中实现

```python
def calculate_williams_r(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """计算威廉指标 (%R)
    
    %R = (最高价(N) - 收盘价) / (最高价(N) - 最低价(N)) * 100
    
    Args:
        df: DataFrame，需包含 ts_code, trade_date, high_adj, low_adj, close_adj
        window: 回看窗口，默认14
        
    Returns:
        DataFrame，包含 ts_code, trade_date, williams_r_{window}
    """
    result = df[['ts_code', 'trade_date']].copy()
    
    grouped = df.sort_values(['ts_code', 'trade_date']).groupby('ts_code')
    
    wr_values = []
    for ts_code, group in grouped:
        group = group.sort_values('trade_date').copy()
        
        # 计算 N 日最高和最低
        high_n = group['high_adj'].rolling(window=window, min_periods=window).max()
        low_n = group['low_adj'].rolling(window=window, min_periods=window).min()
        
        # 计算 %R
        wr = -100 * (high_n - group['close_adj']) / (high_n - low_n).replace(0, np.nan)
        
        temp_df = pd.DataFrame({
            'ts_code': ts_code,
            'trade_date': group['trade_date'].values,
            f'williams_r_{window}': wr.values
        })
        wr_values.append(temp_df)
    
    if wr_values:
        result = pd.concat(wr_values, ignore_index=True)
    else:
        result[f'williams_r_{window}'] = np.nan
    
    return result
```

### 2. 在 `__init__.py` 中导出

```python
from .technical_indicators import (
    # ... 其他
    calculate_williams_r,
)

__all__ = [
    # ... 其他
    'calculate_williams_r',
]
```

### 3. 在 `builder.py` 中调用

```python
# 在 imports 中添加
from ..factors import (
    # ... 其他
    calculate_williams_r,
)

# 在 _add_advanced_factors 的技术指标部分添加
if current_idx >= 30:
    # ...
    
    # Williams %R(14)
    if all(col in tech_hist_data.columns for col in ['high_adj', 'low_adj', 'close_adj']):
        wr_df = calculate_williams_r(tech_hist_data, window=14)
        wr_today = wr_df[wr_df['trade_date'] == trade_date]
        if len(wr_today) > 0:
            result = result.merge(wr_today, on=['ts_code', 'trade_date'], how='left')
```

### 4. 添加测试

```python
def test_williams_r_exists(...):
    builder = FeatureBuilder(min_list_days=0, require_label=False)
    # ... 构建特征
    assert 'williams_r_14' in result.columns
```

### 5. 更新文档

在 `docs/features_schema.md` 中添加字段说明。

---

## 参考资料

- [pandas 文档](https://pandas.pydata.org/docs/)
- [numpy 文档](https://numpy.org/doc/)
- [TA-Lib 技术指标库](https://github.com/mrjbq7/ta-lib)（参考实现）
- 项目源码：`src/lazybull/factors/`
