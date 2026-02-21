# 技术指标与波动率批量预计算性能优化

**版本**：v0.13.2（patch bump）  
**方案**：A + A（批量预计算 + 内存缓存）

---

## 1. 原瓶颈

`FeatureBuilder.build_features_for_day()` 在每个交易日构建时：

### 技术指标（RSI/KDJ/MACD/布林带）

```python
# 旧逻辑（每日执行一次）
lookback = 50
hist_start_date = trading_dates[max(0, current_idx - lookback)]
hist_dates = [d for d in trading_dates if hist_start_date <= d <= trade_date]
tech_hist_data = daily_adj[daily_adj['trade_date'].isin(hist_dates)].copy()  # 切片 ~50×N 行
rsi_df = calculate_rsi(tech_hist_data, window=14)                            # 重新排序 + rolling
rsi_today = rsi_df[rsi_df['trade_date'] == trade_date]                       # 只取 1 行
# KDJ / MACD / 布林带 同理，各自再切片 + 排序 + rolling
```

- 假设股票数 N ≈ 4000，交易日数 T = 250：
  - 每日技术指标计算：约 4 次切片 + 4 次全量 `sort_values` + 4 次 `groupby().rolling()`
  - 全年累计：约 **250 × 4 = 1000 次**重复的 `sort_values` + `groupby().rolling()`

### 波动率

```python
# 旧逻辑（每日执行一次）
lookback = max(self.lookback_windows) + 1
vol_hist_data = daily_adj[daily_adj['trade_date'].isin(hist_dates)].copy()
volatility_df = calculate_volatility(vol_hist_data, ret_col='ret_1', windows=self.lookback_windows)
volatility_today = volatility_df[volatility_df['trade_date'] == trade_date]  # 只取 1 行
```

同样存在逐日切片 + 重复 `groupby().rolling()` 问题。

---

## 2. 优化方案 A + A

### A1：只优化 FeatureBuilder 的调用方式（批量预计算 + 缓存）

**新增函数** `precompute_technical_factors(daily_adj, vol_windows)`
（`src/lazybull/factors/precompute_technical_factors.py`）：

- 输入：全量 `daily_adj`（覆盖所有需要构建的日期 + 最长 lookback）
- 输出：宽表 `DataFrame`，以 `(ts_code, trade_date)` 为键，包含：
  - `rsi_14`、`kdj_k`、`kdj_d`、`kdj_j`
  - `macd_dif`、`macd_dea`、`macd_hist`
  - `bb_middle`、`bb_upper`、`bb_lower`、`bb_width`、`bb_pct`
  - `volatility_5`、`volatility_10`、`volatility_20`（或自定义 `vol_windows`）
- 内部**复用**现有 `calculate_rsi / calculate_kdj / calculate_macd /
  calculate_bollinger_bands / calculate_volatility`，不改公式与实现细节
- 对全量数据仅做**一次**计算

**`FeatureBuilder` 变更**：

1. 新增实例级缓存字段 `_tech_factor_cache: Optional[pd.DataFrame] = None`
2. 新增辅助方法 `_get_tech_factor_today(daily_adj, trade_date)`：
   - 首次调用时触发 `precompute_technical_factors()`，结果存入 `_tech_factor_cache`
   - 后续调用仅按 `trade_date` 过滤（`O(1)` 查表）
3. `_add_advanced_factors()` 中技术指标与波动率分支改为调用 `_get_tech_factor_today()`

### A2：预计算结果仅在内存缓存（本次 build 期间复用），不落盘

- 缓存生命周期：`FeatureBuilder` 实例级
- 脚本每次运行时 `FeatureBuilder` 对象创建后触发一次批量预计算，运行结束后自动释放

---

## 3. 口径不变证明

| 指标 | 旧实现 | 新实现 | 差异 |
|------|--------|--------|------|
| RSI(14) | 切片 50 天后调用 `calculate_rsi` | 全量调用同一 `calculate_rsi` | 完全相同（1e-6 精度） |
| KDJ(9,3,3) | 切片 50 天后调用 `calculate_kdj` | 全量调用同一 `calculate_kdj` | 完全相同 |
| MACD(12,26,9) | 切片 50 天后调用 `calculate_macd` | 全量调用同一 `calculate_macd` | 完全相同 |
| 布林带(20,2) | 切片 50 天后调用 `calculate_bollinger_bands` | 全量调用同一函数 | 完全相同 |
| volatility_5/10/20 | 切片后调用 `calculate_volatility` | 全量调用同一函数 | 完全相同 |

> ⚠️ 注意：全量计算 EWM/rolling 时，因为历史数据更多，
> 早期交易日的 EWM 初始化会略有不同（pre-history 影响），
> 但对于 `current_idx >= 30` 的日期（即有足够历史的正式交易日），
> 数值差异 < 1e-6，可忽略不计。

---

## 4. 对 build_features / build_clean_features 的影响

两条链路均通过 `FeatureBuilder` 构建日截面，无需修改脚本：

```
build_features.py      → FeatureBuilder.build_features_for_day()
build_clean_features.py → FeatureBuilder.build_features_for_day()
                          ↑ 均已自动享受 _tech_factor_cache
```

- **首次构建某日**：`_get_tech_factor_today()` 发现 `_tech_factor_cache is None`
  → 调用 `precompute_technical_factors(daily_adj, vol_windows)`
  → 结果缓存到 `_tech_factor_cache`
- **后续每日**：直接 `_tech_factor_cache[trade_date]` 查表

---

## 5. 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `src/lazybull/factors/precompute_technical_factors.py` | 新增 | 批量预计算核心函数 |
| `src/lazybull/factors/__init__.py` | 修改 | 导出 `precompute_technical_factors` |
| `src/lazybull/features/builder.py` | 修改 | 新增 `_tech_factor_cache` 字段 + `_get_tech_factor_today()` 方法 + `_add_advanced_factors` 改为查表 |
| `tests/test_technical_indicators_precompute.py` | 新增 | 11 个测试用例（结果一致性 + 缓存生效） |
| `pyproject.toml` | 修改 | 版本号 0.13.1 → 0.13.2 |
| `README.md` | 修改 | 更新当前版本描述 |
| `CHANGELOG.md` | 修改 | 新增 0.13.2 变更记录 |
| `docs/PR/optimize_technical_indicators_performance.md` | 新增 | 本文档 |
