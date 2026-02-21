# 市场状态特征验证指南

## 背景

`v0.13.1` 对 `compute_market_state_features` 的批量构建性能进行了优化，
新增了 `precompute_market_state_features()` 函数及 `FeatureBuilder` 实例级缓存。
本指南说明如何在实际数据上验证新旧实现的输出一致性。

---

## 快速验证（抽样对比）

以下脚本可在已有 clean 数据的环境中运行，对任意 N 个交易日逐一对比：

```python
import numpy as np
import pandas as pd
from src.lazybull.factors.market_state import (
    compute_market_state_features,
    precompute_market_state_features,
)
from src.lazybull.data import DataLoader, Storage

# 加载数据
storage = Storage()
loader = DataLoader(storage)

start_date = "20230101"
end_date = "20231231"
daily_data = loader.load_clean_daily(start_date, end_date)
daily_basic = loader.load_clean_daily_basic(start_date, end_date)
trading_dates = loader.get_trading_dates(
    "2023-01-01", "2023-12-31"
)
trading_dates_str = [
    d.strftime('%Y%m%d') if isinstance(d, pd.Timestamp) else d
    for d in trading_dates
]

# 批量预计算
batch = precompute_market_state_features(daily_data, trading_dates_str, daily_basic)

# 抽样对比（随机取 10 个交易日）
import random
sample_dates = random.sample(trading_dates_str, min(10, len(trading_dates_str)))

all_ok = True
for d in sorted(sample_dates):
    idx = trading_dates_str.index(d)
    single = compute_market_state_features(daily_data, d, trading_dates_str, idx, daily_basic)
    for col in single:
        b_val = float(batch.loc[d, col])
        s_val = float(single[col])
        if np.isnan(s_val) and np.isnan(b_val):
            continue
        diff = abs(b_val - s_val)
        if diff > 1e-9:
            print(f"[FAIL] [{d}] {col}: 批量={b_val:.12f} 逐日={s_val:.12f} 差异={diff:.2e}")
            all_ok = False

if all_ok:
    print(f"验证通过：{len(sample_dates)} 个交易日 × 6 个特征，批量与逐日结果完全一致")
```

---

## 字段说明

| 字段 | 计算方式 | 口径 |
|------|----------|------|
| `mkt_vol_cnt` | tradable（vol>0）股票当日收益率截面标准差 | ddof=1 |
| `mkt_vol_20` | `mkt_vol_cnt` 的 20 日滚动均值 | min_periods=1，NaN 跳过 |
| `mkt_ret_avg_20` | tradable 股票日均收益率的 20 日累加 | min_periods=1，NaN 跳过 |
| `mkt_adv_dec_ratio` | `(adv+1)/(dec+1)` 的 60 日滚动均值 | min_periods=1，NaN 跳过 |
| `mkt_turnover_ratio` | tradable 股票 `sum(amount)/sum(circ_mv)`（当日） | 需 daily_basic |
| `mkt_turnover_std` | tradable 股票换手率截面标准差（当日） | 优先 `turnover_rate_f`，ddof=1 |

> **tradable 代理**：`vol > 0`（历史日期停牌判断方式，与 v0.12.1 保持一致）

---

## 注意事项

1. **NaN 来源**：交易日期无 `daily_data` 记录（如非全量数据），对应日期的截面特征为 NaN；
   rolling 计算使用 `min_periods=1`，不会因为窗口不足而返回 NaN。

2. **换手特征为 NaN**：若未提供 `daily_basic_data`，`mkt_turnover_ratio` 和
   `mkt_turnover_std` 均为 NaN，此为正常行为。

3. **缓存重置**：若需在同一进程中对不同数据集构建特征，请使用不同的 `FeatureBuilder` 实例，
   或手动将 `builder._market_state_cache = None` 重置缓存。
