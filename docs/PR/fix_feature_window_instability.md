# PR: 修复 --start-date 变化导致同一 trade_date 特征值不稳定

## 问题描述

使用 `scripts/build_clean_features.py` 构建特征时，仅修改 `--start-date` 会导致相同
`trade_date` 的特征值出现差异。

典型案例：
- 运行 A：`--start-date 20251228 --end-date 20260109`
- 运行 B：`--start-date 20251229 --end-date 20260109`

两次运行在 `trade_date=20260105` 的特征值存在明显差异，包括：
`rsi_14`、`kdj_*`、`macd_*`、`bb_*`、`zscore_*`、`mkt_adv_dec_ratio`、
`ret_N`、`vol_ratio_N`、`ma_deviation_N` 等窗口/滚动指标。

## 根因分析

**双重根因：**

### 根因一：窗口日期选择依赖 `trading_dates` 下标切片

在 `_calculate_features()` 和 `_add_moneyflow_features()` 中，历史窗口通过：
```python
hist_start_date = trading_dates[current_idx - window]
hist_end_date   = trading_dates[current_idx - 1]
hist_dates = [d for d in trading_dates if hist_start_date <= d <= hist_end_date]
```
计算。若 `trading_dates` 不是全量交易日历（例如被截断为从 `start_date` 起的日期列表），
`current_idx` 变小，导致 `current_idx - window` 可能为负或指向错误的历史位置，
产生窗口错位。

### 根因二：`daily_adj` 起始日期不同导致 EWM 指标差异

`precompute_technical_factors` 在 `daily_adj` 上批量计算滚动/EWM 指标（KDJ、MACD 等）。
当 `daily_adj` 来自不同 `start_date` 运行时，数据起点不同，EWM 初始值不同，
导致相同 `trade_date` 的 KDJ/MACD 值存在差异（MACD span=26 时，
EWM 初始值的影响权重约为 `(1-2/27)^N`，N 较小时不可忽略）。

## 解决方案

### 变更 1：新增 `_get_lookback_dates` 私有方法

```python
def _get_lookback_dates(self, trade_date, n, trading_dates) -> List[str]:
    """从全量交易日序列中，以 trade_date 为锚点向前回溯恰好 n 个交易日"""
```

以 `trade_date` 在**全量** `trading_dates`（由 `trade_cal` 提取）中的位置为锚点，
向前回溯恰好 `n` 个交易日。历史不足时返回空列表，对应特征置 NaN。

### 变更 2：`_calculate_features()` 使用新方法

替换旧的 `current_idx - window` 切片 + 区间筛选：
```python
# 旧实现（受 trading_dates 截断影响）
hist_start_date = trading_dates[current_idx - window]
hist_dates = [d for d in trading_dates if hist_start_date <= d <= hist_end_date]

# 新实现（只由全量 trade_cal 决定）
hist_dates = self._get_lookback_dates(trade_date, window, trading_dates)
if not hist_dates:
    features[...] = np.nan; continue
```

### 变更 3：`_add_moneyflow_features()` 同步修复

资金流 rolling 窗口（5/20 日）同步改用 `_get_lookback_dates`。

### 变更 4：`_get_tech_factor_today()` 过滤 `daily_adj`

新增 `trading_dates` 参数；预计算技术指标前，先将 `daily_adj` 过滤到
全量 `trading_dates` 集合，确保：
- 非交易日数据不参与滚动计算
- 两次运行使用相同的日期子集（只要 `trading_dates` 一致）

```python
if trading_dates is not None:
    trading_dates_set = set(trading_dates)
    daily_adj_for_cache = daily_adj[daily_adj['trade_date'].isin(trading_dates_set)]
```

## 影响范围

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `src/lazybull/features/builder.py` | 修复 | 新增方法 + 修改 3 处窗口逻辑 |
| `tests/test_features.py` | 新增测试 | 3 个新测试用例 |
| `pyproject.toml` | 版本 | 0.13.4 → 0.13.5 |
| `src/lazybull/__init__.py` | 版本 | 0.13.1 → 0.13.5 |
| `README.md` | 文档 | 更新版本描述 |
| `CHANGELOG.md` | 文档 | 新增 0.13.5 条目 |

## 验证方法

### 单元测试

```bash
python -m pytest tests/test_features.py -v
```

新增的测试用例：
1. `test_get_lookback_dates_basic`：验证基础回溯逻辑与边界情况
2. `test_window_features_stable_across_start_dates`：
   相同全量 `trade_cal`、不同 `daily_data` 起始截断，同一 `trade_date` 的
   `ret_N`/`vol_ratio_N`/`ma_deviation_N` 完全一致（精度 < 1e-9）
3. `test_window_features_nan_when_insufficient_history`：
   历史不足时窗口特征全部为 NaN

### 实盘验证（需真实数据）

```bash
# 两次运行，对比 trade_date=20260105 的特征
python scripts/build_clean_features.py --start-date 20251228 --end-date 20260109 --only-features --force
python scripts/build_clean_features.py --start-date 20251229 --end-date 20260109 --only-features --force
# 对比两次输出中 20260105 的特征，diff 应为零（EWM 指标在历史充足时误差 < 1e-6）
```

## 注意事项

- EWM 类指标（KDJ、MACD）的稳定性还依赖于 `daily_adj` 包含充足的暖启动历史
  （建议 `start_date` 向前扩展 3~6 个月，以确保 EWM 充分收敛）。
  本次修复通过过滤到全量 `trading_dates` 消除了非交易日干扰；
  建议同步将 `build_features_data` 中的历史扩展从 1 个月提升至 3 个月以上。
- 不破坏现有接口：`_get_tech_factor_today` 新增的 `trading_dates` 参数默认为 `None`，
  向后兼容旧调用方式。
