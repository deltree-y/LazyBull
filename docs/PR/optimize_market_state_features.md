# 性能优化：市场状态特征批量预计算 + 缓存

## 版本

`v0.13.0` → `v0.13.1`（patch bump，纯性能优化，输出口径不变）

---

## 问题背景

`src/lazybull/factors/market_state.py` 的 `compute_market_state_features()` 在计算滚动特征时，
对最近 60 个交易日逐日循环，并在全量 `daily_data` 上反复执行 `daily_data[daily_data['trade_date']==d]`
行过滤，导致批量构建时：

- 每个交易日调用一次 `compute_market_state_features()`
- 每次调用内部再循环 60 次，每次做全量 DataFrame 过滤
- 全量数据（4000+ 支股票 × 若干年）下，单日耗时约 **10 秒**
- 1 年 ≈ 250 个交易日，总耗时约 **40 分钟**

`FeatureBuilder._add_market_state_features()` 每个交易日都调用一次，无任何复用机制。

---

## 优化方案（方案A：批量预计算 + 缓存）

### 1. 新增批量预计算函数

在 `src/lazybull/factors/market_state.py` 中新增：

```python
precompute_market_state_features(
    daily_data: pd.DataFrame,
    trading_dates: list,
    daily_basic_data: pd.DataFrame = None,
) -> pd.DataFrame
```

实现要点：
- **截面统计**：对全量 `daily_data` 按 `vol>0` 过滤后，以 `trade_date` 为 key 做一次 `groupby`，
  向量化计算每日的 `vol_cnt`（std）、`mean_ret`（mean）、`adv_dec_ratio`（涨跌比）。
- **rolling 窗口**：将日级序列 `reindex(trading_dates)` 对齐（缺失补 NaN），
  再用 pandas `rolling(window, min_periods=1)` 计算 `mkt_vol_20`（.mean）、
  `mkt_ret_avg_20`（.sum）、`mkt_adv_dec_ratio`（.mean）。
- **换手特征**：`daily_basic_data` 与 `daily_data.amount` 以 `(trade_date, ts_code)` 向量化 merge，
  `tradable_merged.groupby('trade_date')` 计算 `sum(amount)/sum(circ_mv)` 和 `std(turnover_rate_f)`。
- **输出**：以 `trade_date` 为索引的 DataFrame，包含全部 6 个市场状态列。

### 2. FeatureBuilder 实例级缓存

在 `FeatureBuilder.__init__` 中添加：

```python
self._market_state_cache: Optional[pd.DataFrame] = None
```

在 `_add_market_state_features()` 中：
- 若 `self._market_state_cache is None`，触发 `precompute_market_state_features(...)` 并缓存。
- 否则直接 `self._market_state_cache.loc[trade_date]` 取值（O(1)）。
- 若 `trade_date` 不在缓存中（异常情况），安全回退到原 `compute_market_state_features()`。

### 3. 保留原函数

`compute_market_state_features()` 保持不变，作为：
- 单日调用的兼容入口
- 缓存未命中时的安全回退

---

## 影响范围

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `src/lazybull/factors/market_state.py` | 新增函数 | `precompute_market_state_features()` |
| `src/lazybull/factors/__init__.py` | 新增导出 | `precompute_market_state_features` |
| `src/lazybull/features/builder.py` | 逻辑修改 | `__init__` 新增缓存字段；`_add_market_state_features` 使用缓存 |
| `tests/test_market_and_new_features.py` | 新增测试 | `TestPrecomputeMarketStateFeatures`（6 个用例） |
| `docs/PR/optimize_market_state_features.md` | 新增文档 | 本文件 |
| `docs/guide/market_state_features.md` | 新增文档 | 验证一致性指南 |
| `pyproject.toml` | 版本更新 | `0.13.0` → `0.13.1` |
| `src/lazybull/__init__.py` | 版本更新 | 同上 |
| `README.md` | 更新说明 | 当前版本描述 |
| `CHANGELOG.md` | 新增条目 | `[0.13.1]` 变更记录 |

---

## 口径验证

以下口径与旧实现完全一致（测试精度 < 1e-9）：

| 特征 | 口径说明 |
|------|----------|
| `mkt_vol_cnt` | tradable（vol>0）股票收益率截面标准差（ddof=1） |
| `mkt_vol_20` | `mkt_vol_cnt` 的 20 日滚动均值（min_periods=1，NaN 跳过） |
| `mkt_ret_avg_20` | tradable 股票日均收益率的 20 日滚动累加（min_periods=1） |
| `mkt_adv_dec_ratio` | `(adv+1)/(dec+1)` 的 60 日滚动均值（min_periods=1） |
| `mkt_turnover_ratio` | tradable 股票 `sum(amount)/sum(circ_mv)`（当日） |
| `mkt_turnover_std` | tradable 股票 `turnover_rate_f`（或 `turnover_rate`）的截面标准差（ddof=1） |

---

## 两条构建链路均生效

- `build_features.py`：通过 `builder.build_features_for_day(...)` 循环调用，
  第一次进入时触发 `precompute_market_state_features`，后续 O(1)。
- `build_clean_features.py`：同上（均通过 `FeatureBuilder` 实例，缓存共享整个构建批次）。

---

## 测试

共新增 6 个测试用例，均通过：

1. `test_output_shape`：输出 DataFrame 形状符合预期
2. `test_parity_with_single_day_no_basic`：无 daily_basic 时与逐日结果完全一致
3. `test_parity_with_single_day_with_basic`：含 daily_basic 时与逐日结果完全一致
4. `test_rolling_min_periods_1`：min_periods=1 行为符合预期，不提前返回 NaN
5. `test_empty_data_returns_nan`：空数据不抛异常
6. `test_no_duplicate_compute_with_cache`：多次调用只触发一次批量预计算
