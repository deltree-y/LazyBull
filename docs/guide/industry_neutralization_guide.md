# 行业中性化使用指南

本指南介绍如何使用行业中性化功能，以及如何扩展和验证中性化效果。

## 目录

1. [快速开始](#快速开始)
2. [核心概念](#核心概念)
3. [扩展白名单](#扩展白名单)
4. [验证中性化效果](#验证中性化效果)
5. [常见问题](#常见问题)

## 快速开始

### 步骤1: 下载申万行业数据

```bash
python scripts/update_basic_data.py --only-shenwan --force
```

这会从TuShare下载申万一级行业分类数据并保存到 `data/raw/shenwan_industry.parquet`。

### 步骤2: 重新构建Features（如果需要）

如果你已有features数据，需要重新构建以包含行业信息和中性化特征：

```bash
# 指定日期范围重建
python scripts/build_features.py \
  --start-date 20230101 \
  --end-date 20231231 \
  --force
```

### 步骤3: 在代码中使用

```python
from src.lazybull.data import DataLoader
from src.lazybull.features import FeatureBuilder

# 加载数据
loader = DataLoader()
builder = FeatureBuilder()

# 加载申万行业数据
shenwan_industry = loader.load_shenwan_industry()

# 构建特征（启用行业中性化）
features = builder.build_features_for_day(
    trade_date='20240101',
    # ... 其他参数 ...
    shenwan_industry=shenwan_industry,
    apply_industry_neutralization=True
)

# 使用中性化后的特征
neu_features = [col for col in features.columns if col.startswith('neu_')]
print(f"中性化特征列表: {neu_features}")
```

## 核心概念

### 什么是行业中性化？

行业中性化（Industry Neutralization）是消除行业间系统性差异的技术。通过对特征进行行业内标准化，可以：

1. **消除行业效应**: 不同行业的估值水平、波动率等天然不同
2. **提高因子纯净度**: 聚焦于个股在行业内的相对表现
3. **改善模型泛化**: 避免模型过度依赖行业分类信息

### 数学原理

对于特征 `x`，行业中性化定义为：

```
neu_x = (x - mean_industry(x)) / std_industry(x)
```

这是一个行业内的Z-Score标准化，将特征转换为行业内的相对值。

### 实现细节

1. **截面处理**: 每个交易日独立计算，不涉及跨日数据
2. **可交易筛选**: 只使用 `tradable == 1` 的样本计算统计量
3. **小样本回退**: 行业样本数 < 5 时，使用全市场统计量
4. **保留原特征**: 原始特征列保留，新增 `neu_*` 前缀的中性化列

## 扩展白名单

默认白名单包含13个特征列。如果需要对其他列进行中性化，可以修改白名单。

### 方法1: 修改FeatureBuilder源码

编辑 `src/lazybull/features/builder.py` 中的 `_apply_industry_neutralization` 方法：

```python
def _apply_industry_neutralization(self, features: pd.DataFrame) -> pd.DataFrame:
    # 定义需要行业中性化的列白名单
    neutralization_columns = [
        'pe_ttm',
        'pb',
        # ... 现有列 ...
        
        # 新增你的自定义列
        'my_custom_feature',
        'another_feature',
    ]
    
    # ... 其余代码不变 ...
```

### 方法2: 直接调用中性化函数

如果不想修改源码，可以在构建features后手动调用：

```python
from src.lazybull.factors.normalization import industry_neutralization

# 构建features（不启用中性化）
features = builder.build_features_for_day(
    trade_date='20240101',
    # ... 参数 ...
    apply_industry_neutralization=False
)

# 手动对自定义列进行中性化
features = industry_neutralization(
    features,
    columns=['my_custom_feature', 'another_feature'],
    industry_col='sw_name',
    tradable_col='tradable',
    min_group_size=5,
    prefix='neu_'
)
```

### 建议

- **连续数值特征**: 适合中性化（如估值、规模、波动率、收益率）
- **离散特征/标签**: 不适合中性化
- **已标准化的特征**: 如果特征已经是Z-Score，可能不需要再次中性化

## 验证中性化效果

### 简单示例：检查行业内分布

```python
import pandas as pd
import matplotlib.pyplot as plt

# 加载某日features
features = loader.load_features_for_date('20240101')

# 检查某个特征的原始分布 vs 中性化后的分布
feature_name = 'pe_ttm'

# 按行业查看原始特征的均值和标准差
original_stats = features.groupby('sw_name')[feature_name].agg(['mean', 'std'])
print("原始特征 - 各行业统计:")
print(original_stats)

# 按行业查看中性化后特征的均值和标准差
neu_stats = features.groupby('sw_name')[f'neu_{feature_name}'].agg(['mean', 'std'])
print(f"\n中性化后 - 各行业统计:")
print(neu_stats)

# 中性化后，每个行业的均值应该接近0，标准差接近1
```

### 进阶验证：检查因子IC

```python
import numpy as np

# 计算原始特征的IC
def calculate_ic(features, factor_col, label_col='y_ret_5'):
    valid_data = features[[factor_col, label_col]].dropna()
    if len(valid_data) < 10:
        return np.nan
    return valid_data[factor_col].corr(valid_data[label_col])

# 对比原始特征和中性化特征的IC
ic_original = calculate_ic(features, 'pe_ttm')
ic_neutralized = calculate_ic(features, 'neu_pe_ttm')

print(f"原始特征 IC: {ic_original:.4f}")
print(f"中性化特征 IC: {ic_neutralized:.4f}")

# 期望: 中性化后IC的绝对值应该更高（因子更纯净）
```

### 批量验证脚本

创建一个脚本 `scripts/validate_neutralization.py`：

```python
"""验证行业中性化效果"""

import pandas as pd
import numpy as np
from src.lazybull.data import DataLoader

def validate_neutralization_for_date(trade_date: str):
    """验证单日行业中性化效果"""
    loader = DataLoader()
    features = loader.load_features_for_date(trade_date)
    
    if features is None:
        print(f"{trade_date}: 无数据")
        return
    
    # 找出所有中性化特征
    neu_features = [col for col in features.columns if col.startswith('neu_')]
    
    results = []
    for neu_col in neu_features:
        original_col = neu_col.replace('neu_', '')
        
        # 按行业统计
        industry_stats = features.groupby('sw_name')[neu_col].agg(['mean', 'std'])
        
        # 检查是否符合预期（均值接近0，标准差接近1）
        mean_abs_mean = industry_stats['mean'].abs().mean()
        mean_std = industry_stats['std'].mean()
        
        results.append({
            'feature': original_col,
            'mean_abs_mean': mean_abs_mean,  # 应该接近0
            'mean_std': mean_std,  # 应该接近1
            'valid': (mean_abs_mean < 0.1) and (0.8 < mean_std < 1.2)
        })
    
    df_results = pd.DataFrame(results)
    print(f"\n{trade_date} 验证结果:")
    print(df_results)
    
    # 统计
    valid_count = df_results['valid'].sum()
    total_count = len(df_results)
    print(f"\n通过验证: {valid_count}/{total_count} 个特征")

if __name__ == '__main__':
    # 验证多个日期
    dates = ['20240101', '20240108', '20240115']
    for date in dates:
        validate_neutralization_for_date(date)
```

## 常见问题

### Q1: 为什么有些股票没有行业信息？

**A**: 可能原因：
1. 新上市股票，申万行业还未分类
2. ST股票可能被剔除出行业指数
3. TuShare数据更新延迟

**解决方案**: 
- 这些股票的 `sw_name` 为 NaN，会被跳过中性化
- 如果样本数较多，建议重新下载行业数据

### Q2: 中性化后特征变成NaN？

**A**: 检查以下几点：
1. 原始特征本身是否有NaN
2. 行业列 `sw_name` 是否为NaN
3. 该交易日是否有足够的可交易样本（`tradable == 1`）

### Q3: 小样本行业如何处理？

**A**: 当行业样本数 < 5 时，自动回退到全市场统计：
- 使用全市场（`tradable == 1`）的均值和标准差
- 这样可以确保所有样本都能得到有效的中性化值
- `min_group_size=5` 是默认值，可以调整

### Q4: 中性化后模型效果反而变差？

**A**: 可能原因：
1. 行业信息本身对预测很重要，中性化反而丢失了信息
2. 某些特征不适合行业中性化（例如技术指标）
3. 白名单列选择不当

**建议**:
- 不要对所有特征都中性化
- 保留原始特征和中性化特征，让模型自动选择
- 对估值类、规模类特征进行中性化，技术指标可以不中性化

### Q5: 如何在模型中使用中性化特征？

**A**: 有两种策略：

**策略1: 只使用中性化特征**
```python
# 特征列
feature_cols = [col for col in df.columns if col.startswith('neu_')]
X = df[feature_cols]
```

**策略2: 同时使用原始和中性化特征**
```python
# 原始特征
original_features = ['pe_ttm', 'pb', 'log_total_mv', ...]
# 中性化特征
neu_features = [f'neu_{f}' for f in original_features]
# 合并
feature_cols = original_features + neu_features
X = df[feature_cols]
```

推荐使用策略2，让模型自动选择最有效的特征。

### Q6: 行业分类会变化吗？

**A**: 会，但不频繁：
- 申万行业分类每年可能调整1-2次
- 个股的行业归属也可能变化
- 建议每季度更新一次行业数据

### Q7: 是否需要对标签进行中性化？

**A**: 通常不需要：
- 标签（未来收益率）已经是相对值
- 对标签中性化可能降低预测难度，但也可能丢失信息
- 可以实验对比效果

## 最佳实践

1. **数据更新频率**:
   - 基础数据（trade_cal, stock_basic）: 每季度
   - 申万行业数据: 每季度
   - 日线数据: 每日
   - Features: 按需（通常是训练前）

2. **中性化时机**:
   - 在特征构建阶段进行（推荐）
   - 不要在数据清洗阶段进行
   - 确保训练/回测/实盘使用相同逻辑

3. **特征选择**:
   - 估值类: 一定要中性化（PE, PB, PS等）
   - 规模类: 建议中性化（市值、成交额等）
   - 技术指标: 根据情况（RSI, MACD可以不中性化）
   - 动量类: 根据情况（收益率可以中性化）

4. **监控**:
   - 定期验证中性化效果（均值接近0，标准差接近1）
   - 对比原始特征和中性化特征的IC
   - 观察模型特征重要性

## 相关资源

- [PR说明文档](../PR/industry_neutralization.md)
- [特征字段说明](../features_schema.md)
- [TuShare申万行业分类API文档](https://tushare.pro/document/2?doc_id=335)
