# 市场状态特征验证指南

本指南说明如何验证 v0.12.1 新增的市场状态特征是否计算正确。

## 快速验证

### 前提条件

已构建至少一天的 features 文件（`data/features/cs_train/*.parquet`）。

### 1. 加载单日特征并检查市场状态列

```python
import pandas as pd
from src.lazybull.data import Storage

storage = Storage()
df = storage.load_cs_train_day('20230110')

# 检查市场状态特征是否存在
mkt_cols = ['mkt_vol_cnt', 'mkt_vol_20', 'mkt_turnover_ratio',
            'mkt_ret_avg_20', 'mkt_turnover_std', 'mkt_adv_dec_ratio']
print("市场状态特征：")
print(df[mkt_cols].iloc[0])  # 同一天所有股票值相同

# 市场状态特征应该在每天所有行中相同（广播值）
assert df[mkt_cols].nunique().max() == 1, "市场状态特征应为常数列（同日相同）"
print("✅ 市场状态特征广播正确")
```

### 2. 验证 mkt_vol_cnt（截面波动率）

```python
import numpy as np

# mkt_vol_cnt 应等于当日 tradable 股票 ret_1 的截面标准差
expected_vol_cnt = df[df['tradable'] == 1]['ret_1'].std()
actual_vol_cnt = df['mkt_vol_cnt'].iloc[0]
print(f"期望 mkt_vol_cnt: {expected_vol_cnt:.6f}")
print(f"实际 mkt_vol_cnt: {actual_vol_cnt:.6f}")
assert abs(expected_vol_cnt - actual_vol_cnt) < 1e-4, "mkt_vol_cnt 计算有误"
print("✅ mkt_vol_cnt 验证通过")
```

### 3. 验证 mkt_turnover_ratio

```python
# mkt_turnover_ratio = sum(amount) / sum(circ_mv)（tradable==1）
tradable_df = df[df['tradable'] == 1]
expected_ratio = tradable_df['amount'].sum() / tradable_df['circ_mv'].sum()
actual_ratio = df['mkt_turnover_ratio'].iloc[0]
print(f"期望 mkt_turnover_ratio: {expected_ratio:.6f}")
print(f"实际 mkt_turnover_ratio: {actual_ratio:.6f}")
assert abs(expected_ratio - actual_ratio) < 1e-6, "mkt_turnover_ratio 计算有误"
print("✅ mkt_turnover_ratio 验证通过")
```

### 4. 验证多日滚动特征（mkt_vol_20）

```python
# 加载连续30天的数据，验证 mkt_vol_20 趋势合理
dates = ['20230105', '20230106', '20230109', '20230110', '20230111']
mkt_vols = []
for d in dates:
    day_df = storage.load_cs_train_day(d)
    if day_df is not None:
        mkt_vols.append({
            'date': d,
            'mkt_vol_cnt': day_df['mkt_vol_cnt'].iloc[0],
            'mkt_vol_20': day_df['mkt_vol_20'].iloc[0],
        })

mkt_df = pd.DataFrame(mkt_vols)
print(mkt_df)
# mkt_vol_20 应比 mkt_vol_cnt 更平滑
print("✅ 多日滚动特征验证完成")
```

### 5. 验证新增个股特征

```python
# 验证 is_new_stock
new_stocks = df[df['is_new_stock'] == 1]
print(f"新股数量（上市<365天）：{len(new_stocks)}")

# 验证 zscore_size 行业内均值接近0
for industry in df['sw_industry'].dropna().unique()[:3]:
    ind_df = df[(df['sw_industry'] == industry) & (df['tradable'] == 1)]
    if len(ind_df) >= 5:
        mean_z = ind_df['zscore_size'].mean()
        print(f"行业 {industry}: zscore_size 均值 = {mean_z:.4f}（应接近0）")
```

## 注意事项

- 市场状态特征由 `src/lazybull/factors/market_state.py` 计算，依赖当日及历史 `daily_data`
- `mkt_turnover_ratio` 和 `mkt_turnover_std` 需要 `daily_basic` 数据（`circ_mv`、`turnover_rate_f`）
- 历史数据不足时（如前 60 日数据不足），滚动特征使用现有数据计算（min_periods=1）
- `spec_score` 需要启用 `apply_industry_neutralization=True` 才会非 NaN
