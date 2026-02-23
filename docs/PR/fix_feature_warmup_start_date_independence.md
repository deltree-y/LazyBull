# 修复特征构建对 `--start-date` 敏感问题（v0.13.6）

## 背景与问题复现

用户发现使用 `scripts/build_clean_features.py` 生成特征时，改变 `--start-date` 会导致相同
目标日期（如 `20251231`、`20260105`）的特征值出现显著差异。差异集中在：
- `mkt_adv_dec_ratio`（市场涨跌比率）
- KDJ 指标（`kdj_k`、`kdj_d`、`kdj_j`）
- MACD 指标（`macd_dif`、`macd_dea`、`macd_hist`）
- 及衍生的 `zscore_macd_hist`、`spec_score` 等

**复现示例**：
- `start_date=20251228` → 加载历史数据从 `2025-11-28`（1 个月前）
- `start_date=20251229` → 加载历史数据从 `2025-12-01`（1 个月前）
- 两次构建 `trade_date=20251231` 的 `mkt_adv_dec_ratio`：1.45947 vs 1.37279（差异显著）

## 根因分析

### 问题一：数据加载起点不固定

`build_clean_features.py` 使用以下逻辑确定历史数据加载起点：

```python
start_dt = pd.to_datetime(start_date, format='%Y%m%d') - pd.DateOffset(months=1)
```

不同的 `--start-date` 导致加载的历史起点不同，从而传入 `FeatureBuilder` 的 `daily_adj` 覆盖范围不同。

### 问题二：滚动/递推因子对输入历史截断点敏感

- **市场状态特征**（`precompute_market_state_features`）：`mkt_adv_dec_ratio` 使用 60 日
  rolling 均值，`mkt_vol_20` 使用 20 日 rolling 均值。当输入只有 20-22 个交易日的历史时，
  rolling 窗口覆盖的天数不同（如 22 天 vs 20 天），导致均值计算结果不同。
- **技术指标**（`precompute_technical_factors`）：KDJ 和 MACD 使用 EWM（指数加权平均）。
  EWM 的数值与计算序列的起点强相关：序列越长，EWM 越充分"预热"，起点不同导致
  相同目标日的 EWM 值不同。

### 问题三：v0.13.5 修复不完整

v0.13.5 在 `_get_tech_factor_today()` 中将 `daily_adj` 过滤到全量 `trading_dates` 集合，
但这并未修复输入数据历史起点的差异——全量 `trading_dates` 远比 `daily_adj` 的日期范围宽，
过滤只是去掉了非交易日，并未统一历史起点。

## 修复方案

### 方案选择：固定 warmup = 120 个交易日（方案 A）

确保预计算输入始终从「目标批量区间首个 `trade_date` 往前 120 个交易日」处开始，
120 个交易日足以使 EWM（MACD slow=26、KDJ n=9）充分收敛，也覆盖 mkt_adv_dec_ratio
的 60 日滚动窗口需要的历史。

### 核心改动

#### 1. `src/lazybull/features/builder.py`

**新增模块常量**：
```python
_WARMUP_TRADING_DAYS = 120
```

**新增 `_slice_by_trading_days()` 辅助方法**：

```python
def _slice_by_trading_days(self, daily_df, trading_dates, anchor_trade_date, warmup_days=120):
```

以 `anchor_trade_date` 在全量 `trading_dates` 中的位置为锚点，向前回溯 `warmup_days`
个交易日，返回该起点（含）之后属于全量交易日历的所有数据。两次构建使用相同的
`trading_dates` 和相同的 `anchor_trade_date` 时，切片结果完全一致。

**修改 `_add_market_state_features()`**：

首次建立 `_market_state_cache` 时，先对 `daily_adj` 和 `daily_basic_data` 应用
`_slice_by_trading_days`，再传入 `precompute_market_state_features`：

```python
sliced_daily_adj = self._slice_by_trading_days(daily_adj, trading_dates, trade_date)
sliced_daily_basic = (
    self._slice_by_trading_days(daily_basic_data, trading_dates, trade_date)
    if daily_basic_data is not None else None
)
self._market_state_cache = precompute_market_state_features(
    daily_data=sliced_daily_adj,
    trading_dates=trading_dates,
    daily_basic_data=sliced_daily_basic,
)
```

**修改 `_get_tech_factor_today()`**：

首次建立 `_tech_factor_cache` 时，使用 `_slice_by_trading_days` 替换原有全量过滤：

```python
daily_adj_for_cache = self._slice_by_trading_days(daily_adj, trading_dates, trade_date)
self._tech_factor_cache = precompute_technical_factors(
    daily_adj=daily_adj_for_cache, vol_windows=self.lookback_windows
)
```

#### 2. `scripts/build_clean_features.py`

将数据加载起点从 1 个月扩展为 7 个月，确保有足够历史覆盖 120 个交易日 warmup：

```python
# 修改前：
start_dt = pd.to_datetime(start_date, format='%Y%m%d') - pd.DateOffset(months=1)
# 修改后：
start_dt = pd.to_datetime(start_date, format='%Y%m%d') - pd.DateOffset(months=7)
```

### 一致性保证

设 `target_date = T`，`trading_dates` 为全量交易日历：
- `anchor_idx = trading_dates.index(T)`
- `warmup_start_idx = max(0, anchor_idx - 120)`
- `warmup_start_date = trading_dates[warmup_start_idx]`

无论 `--start-date` 如何设置，只要 `daily_adj` 中包含 `warmup_start_date` 之后的数据，
`_slice_by_trading_days` 的输出对两次构建完全相同 → 预计算结果完全相同。

## 边界情况

| 场景 | 行为 |
|------|------|
| `anchor_trade_date` 不在 `trading_dates` 中 | 原样返回 `daily_df`（安全回退） |
| 历史 < 120 个交易日 | 从 `trading_dates[0]` 开始，`min_periods=1` 行为不变 |
| `daily_df` 为空或 None | 原样返回 |
| 批量构建多个 `trade_date` | 缓存基于首次 `trade_date` 建立，后续日期复用同一缓存 |

## 新增单元测试（`tests/test_market_and_new_features.py`）

新增 `TestWarmupStartDateIndependence` 测试类，包含 7 个测试用例：

| 测试方法 | 验证内容 |
|----------|----------|
| `test_slice_by_trading_days_basic` | warmup 切片起点正确（anchor-120 个交易日） |
| `test_slice_by_trading_days_insufficient_history` | 历史不足时从第一天开始，不抛异常 |
| `test_slice_by_trading_days_unknown_anchor` | anchor 不在列表时原样返回 |
| `test_market_state_independent_of_start_date` | 同一 `trade_date` 市场状态特征不受历史起点影响 |
| `test_market_state_with_basic_independent_of_start_date` | 含 `daily_basic` 时同样不受影响 |
| `test_tech_factor_independent_of_start_date` | KDJ/MACD/RSI/布林带不受历史起点影响 |
| `test_insufficient_warmup_produces_different_values` | warmup 不足时特征值确实不同（预期行为） |

## 版本信息

- **版本**：0.13.5 → **0.13.6**
- **类型**：一致性 bugfix（patch bump）
- **变更文件**：
  - `src/lazybull/features/builder.py`
  - `scripts/build_clean_features.py`
  - `tests/test_market_and_new_features.py`
  - `pyproject.toml`
  - `README.md`
