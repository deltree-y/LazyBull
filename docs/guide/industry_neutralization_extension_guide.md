# 行业中性化扩展与验证指南

本指南说明如何扩展行业中性化白名单、调整参数、以及验证中性化效果。

## 一、扩展中性化白名单

### 1.1 添加 Z-Score 中性化列

**适用场景**：对新的估值、流动性、风格指标进行行业内标准化

**操作步骤**：

1. 打开 `src/lazybull/features/builder.py`
2. 找到 `_apply_industry_neutralization` 方法
3. 修改 `zscore_columns` 列表

**示例**：

```python
# 在 _apply_industry_neutralization 方法中
zscore_columns = [
    # === 原有白名单 ===
    'pe_ttm',           # 市盈率
    'pb',               # 市净率
    'bp',               # 市净率倒数
    'dv_ttm',          # 股息率
    'log_total_mv',    # 对数总市值
    'amount_ma20',     # 20日均成交额
    'turnover_rate',   # 换手率
    'net_mf_amount',   # 净资金流入
    'ma_deviation_20', # 20日均线偏离度
    
    # === 新增列 ===
    'ps_ttm',          # 市销率
    'pcf_ocf_ttm',     # 市现率（经营现金流）
    'roe_ttm',         # ROE
    'roa_ttm',         # ROA
    'circ_mv',         # 流通市值
    # 添加更多列...
]
```

**注意事项**：
- 确保新增列在特征构建时已生成
- 如果列来自 `daily_basic`，需确保该数据已下载
- 列名必须与特征文件中的列名完全一致

### 1.2 添加去均值（Demean）列

**适用场景**：对新的收益率窗口进行行业去均值

**操作步骤**：

1. 打开 `src/lazybull/features/builder.py`
2. 找到 `_apply_industry_neutralization` 方法中的去均值部分
3. 扩展收益率窗口列表

**示例**：

```python
# 在 _apply_industry_neutralization方法中
# 标签列（保持不变）
for horizon in self.horizons:
    label_col = f'y_ret_{horizon}'
    if label_col in result.columns:
        demean_columns.append(label_col)

# 历史收益列（扩展窗口）
for window in [3, 5, 7, 10, 15, 20, 30]:  # 扩展窗口
    ret_col = f'ret_{window}'
    if ret_col in result.columns:
        demean_columns.append(ret_col)
```

**注意事项**：
- 去均值主要用于收益率类特征
- 不建议对非收益率特征使用去均值（应使用 Z-Score）
- 确保相应的 `ret_N` 列已在特征构建时生成

## 二、验证中性化效果

### 2.1 验证行业内均值为0（去均值）

**目标**：检查每个行业内的中性化列均值是否接近0

**脚本**：

```python
import pandas as pd
from src.lazybull.data import DataLoader

loader = DataLoader()

# 加载某日特征
features = loader.load_features("20231201")

# 验证去均值效果
demean_columns = ['neu_y_ret_20', 'neu_ret_20', 'neu_ret_10']

print("=== 去均值验证 ===")
for col in demean_columns:
    if col not in features.columns:
        print(f"{col}: 列不存在")
        continue
    
    print(f"\n{col}:")
    for industry in features['sw_name'].unique():
        industry_data = features[
            (features['sw_name'] == industry) & 
            (features['tradable'] == 1)
        ]
        
        if len(industry_data) < 5:
            continue  # 跳过小样本行业
        
        mean_val = industry_data[col].mean()
        std_val = industry_data[col].std()
        
        # 均值应该接近0（允许1e-6的数值误差）
        if abs(mean_val) > 1e-3:
            print(f"  ⚠️  {industry}: 均值={mean_val:.6f} (应接近0)")
        else:
            print(f"  ✓  {industry}: 均值={mean_val:.6f}, 标准差={std_val:.4f}")
```

### 2.2 验证行业内标准化（Z-Score）

**目标**：检查每个行业内的 Z-Score 列均值为0、标准差为1

**脚本**：

```python
# 验证 Z-Score 效果
zscore_columns = ['pe_ttm_zscore', 'pb_zscore', 'log_total_mv_zscore']

print("\n=== Z-Score 验证 ===")
for col in zscore_columns:
    if col not in features.columns:
        print(f"{col}: 列不存在")
        continue
    
    print(f"\n{col}:")
    for industry in features['sw_name'].unique():
        industry_data = features[
            (features['sw_name'] == industry) & 
            (features['tradable'] == 1)
        ]
        
        if len(industry_data) < 5:
            continue
        
        mean_val = industry_data[col].mean()
        std_val = industry_data[col].std()
        
        # 均值应该接近0，标准差应该接近1
        mean_ok = abs(mean_val) < 1e-2
        std_ok = abs(std_val - 1.0) < 0.1
        
        if mean_ok and std_ok:
            print(f"  ✓  {industry}: 均值={mean_val:.4f}, 标准差={std_val:.4f}")
        else:
            print(f"  ⚠️  {industry}: 均值={mean_val:.4f}, 标准差={std_val:.4f}")
```

### 2.3 IC分析（预测能力验证）

**目标**：比较原始特征和中性化特征的预测能力（IC）

**脚本**：

```python
import numpy as np

# 加载多日特征
dates = ['20231201', '20231204', '20231205', '20231206', '20231207']
all_features = []

for date in dates:
    features = loader.load_features(date)
    if features is not None:
        all_features.append(features)

combined = pd.concat(all_features, ignore_index=True)

# 只使用可交易样本
combined = combined[combined['tradable'] == 1]

# 计算IC（信息系数）
def calc_ic(feature_col, label_col='y_ret_5'):
    """计算特征与未来收益的相关性"""
    valid_data = combined[[feature_col, label_col]].dropna()
    if len(valid_data) < 100:
        return np.nan
    return valid_data.corr().iloc[0, 1]

# 对比原始特征和中性化特征
feature_pairs = [
    ('ret_20', 'neu_ret_20', '20日收益'),
    ('pe_ttm', 'pe_ttm_zscore', '市盈率'),
    ('pb', 'pb_zscore', '市净率'),
    ('log_total_mv', 'log_total_mv_zscore', '对数市值'),
]

print("\n=== IC分析（特征预测能力）===")
print(f"{'特征':<20} {'原始IC':<12} {'中性化IC':<12} {'差异':<10}")
print("-" * 60)

for orig_col, neu_col, name in feature_pairs:
    if orig_col in combined.columns and neu_col in combined.columns:
        ic_orig = calc_ic(orig_col)
        ic_neu = calc_ic(neu_col)
        diff = ic_neu - ic_orig
        
        print(f"{name:<20} {ic_orig:>11.4f} {ic_neu:>11.4f} {diff:>+9.4f}")
```

## 三、最佳实践

1. **白名单管理**：
   - 只对有明确业务含义的特征进行中性化
   - 去均值用于收益率，Z-Score 用于估值/流动性指标
   - 定期review白名单，移除无效特征

2. **参数选择**：
   - `min_group_size=5` 对大部分场景适用
   - 如果行业数量多（>50），可提高到10
   - 如果行业数量少（<20），可降低到3

3. **效果验证**：
   - 每次修改白名单后运行验证脚本
   - 定期（每月）进行IC分析，评估中性化效果
   - 对比中性化前后的回测结果

4. **版本管理**：
   - 记录每次修改白名单的日期和原因
   - 保留不同版本的特征文件用于对比
   - 在模型元数据中记录中性化参数
