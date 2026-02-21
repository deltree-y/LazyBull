# 分层回退中性化验证指南

本指南说明如何验证申万三级行业分层回退中性化是否正确生效。

---

## 一、验证分层回退是否生效

### 1. 检查行业层级字段是否正确输出

```python
import pandas as pd
from src.lazybull.data import Storage

storage = Storage()
# 加载某个特征文件（假设已运行 build_features.py）
df = pd.read_parquet("data/features/cs_train/20240101.parquet")

# 检查 L3/L2/L1 字段是否存在
required_cols = ['sw_industry', 'sw_industry_code', 'sw_industry_id',
                 'sw_l2', 'sw_l2_code', 'sw_l2_id',
                 'sw_l1', 'sw_l1_code', 'sw_l1_id']
for col in required_cols:
    present = col in df.columns
    print(f"  {'✅' if present else '❌'} {col}")

# 查看行业层级样本分布
print("\nL3 行业数:", df['sw_industry'].nunique())
print("L2 行业数:", df['sw_l2'].nunique())
print("L1 行业数:", df['sw_l1'].nunique())
print("\nL3 行业内样本数分布（前10小的行业）:")
l3_counts = df.groupby('sw_industry')['ts_code'].count().sort_values()
print(l3_counts.head(10))
```

### 2. 检查中性化后列名前缀是否正确

```python
# neu_ 前缀列（去均值，用于收益率/标签）
neu_cols = [c for c in df.columns if c.startswith('neu_')]
print("neu_ 前缀列：", neu_cols)

# zscore_ 前缀列（Z-Score，用于指标）
zscore_cols = [c for c in df.columns if c.startswith('zscore_')]
print("zscore_ 前缀列：", zscore_cols)
```

### 3. 验证 L3 行业内去均值效果

```python
# 选择一个 L3 行业，检查 neu_y_ret_20 是否均值接近 0
if 'neu_y_ret_20' in df.columns and 'sw_industry' in df.columns:
    for ind in df['sw_industry'].value_counts().head(5).index:
        grp = df[df['sw_industry'] == ind]['neu_y_ret_20'].dropna()
        if len(grp) >= 5:
            print(f"  {ind} (n={len(grp)}): mean={grp.mean():.6f} （应接近 0）")
```

### 4. 验证分层回退实际触发的样本

```python
from src.lazybull.factors.hierarchical_industry_neutralization import (
    hierarchical_zscore,
)

# 加载当日特征（合并行业字段后的 DataFrame）
# 假设 df 含有 sw_industry_code, sw_l2_code, sw_l1_code, tradable 列
small_l3_industries = (
    df[df['tradable'] == 1]
    .groupby('sw_industry_code')['ts_code']
    .count()
    .sort_values()
)
print("可交易样本数 < 5 的 L3 行业（将触发回退）：")
print(small_l3_industries[small_l3_industries < 5])

# 手动触发一次分层 zscore 并检查结果
test_df = df[['ts_code', 'sw_industry_code', 'sw_l2_code', 'sw_l1_code',
              'pe_ttm', 'tradable']].dropna(subset=['pe_ttm']).copy()

result = hierarchical_zscore(
    test_df,
    columns=['pe_ttm'],
    l3_col='sw_industry_code',
    l2_col='sw_l2_code',
    l1_col='sw_l1_code',
    tradable_col='tradable',
    min_group_size=5,
)

print("\nzscore_pe_ttm 基本统计：")
print(result['zscore_pe_ttm'].describe())
print("\nNaN 比例：", result['zscore_pe_ttm'].isna().mean())
```

---

## 二、验证回退链路（数值精确检验）

以下代码手动构造一个 L3 样本数不足（=3）的场景，验证是否正确使用 L2 统计量：

```python
import numpy as np
import pandas as pd
from src.lazybull.factors.hierarchical_industry_neutralization import hierarchical_zscore

# 构造测试数据
# L3A: 5个可交易，L3B: 3个可交易（同属 L2=M）
rows = []
for i in range(5):  # L3A: 5 tradable
    rows.append({'ts_code': f'{i:06d}.SZ', 'value': float(i * 10 + 5),
                 'sw_industry_code': 'L3A', 'sw_l2_code': 'M', 'sw_l1_code': 'T', 'tradable': 1})
for i in range(5, 8):  # L3B: 3 tradable（不足 5）
    rows.append({'ts_code': f'{i:06d}.SZ', 'value': float(i * 10 + 5),
                 'sw_industry_code': 'L3B', 'sw_l2_code': 'M', 'sw_l1_code': 'T', 'tradable': 1})

df = pd.DataFrame(rows)

result = hierarchical_zscore(
    df, columns=['value'],
    l3_col='sw_industry_code', l2_col='sw_l2_code', l1_col='sw_l1_code',
    tradable_col='tradable', min_group_size=5,
)

# 手动计算 L2 统计量
l2_vals = df[df['tradable'] == 1]['value']
l2_mean, l2_std = l2_vals.mean(), l2_vals.std()
print(f"L2 统计量：mean={l2_mean:.4f}, std={l2_std:.4f}")

# L3B 应使用 L2 统计量
l3b_results = result[result['sw_industry_code'] == 'L3B']
for _, row in l3b_results.iterrows():
    expected = (row['value'] - l2_mean) / l2_std
    actual = row['zscore_value']
    match = abs(actual - expected) < 1e-9
    print(f"  {row['ts_code']}: expected={expected:.6f}, actual={actual:.6f}, {'✅' if match else '❌'}")
```

---

## 三、验证市场状态特征（继承自 v0.12.1）

```python
# 市场状态特征应为当日广播标量，每行相同
mkt_cols = ['mkt_vol_cnt', 'mkt_vol_20', 'mkt_turnover_ratio',
            'mkt_ret_avg_20', 'mkt_turnover_std', 'mkt_adv_dec_ratio']

for col in mkt_cols:
    if col in df.columns:
        n_unique = df[col].nunique()
        print(f"  {col}: 唯一值数={n_unique} （正常应为 1）")
    else:
        print(f"  ❌ {col} 不存在")
```

---

## 四、常见问题排查

### 问题：分层字段不存在（`sw_l2_code` 等缺失）

**原因**：`shenwan_industry.parquet` 仍为旧式 L2 格式（只有 `sw_code`/`sw_name`）

**解决**：
```bash
python scripts/update_basic_data.py --only-shenwan --force
```

### 问题：所有 L3 行业都使用全市场统计（没有回退到 L2/L1）

**原因**：可能 `tradable` 列全为 NaN 或不存在

**检查**：
```python
print(df['tradable'].value_counts())
print(df[df['sw_industry_code'] == 'xxx']['tradable'].sum())
```

### 问题：`zscore_` 列全为 NaN

**原因**：可能特征列本身（如 `pe_ttm`）全为 NaN，或全市场样本数 < 2

**检查**：
```python
print(df['pe_ttm'].notna().sum())
```
