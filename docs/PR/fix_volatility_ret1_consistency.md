# 修复 volatility_20 / zscore_volatility_20 / spec_score 数值不一致问题

**版本**：v0.13.3  
**类型**：Bug 修复（补丁版本）  
**涉及文件**：
- `src/lazybull/factors/returns.py`（新增）
- `src/lazybull/factors/precompute_technical_factors.py`（修改）
- `src/lazybull/factors/__init__.py`（修改）
- `tests/test_technical_indicators_precompute.py`（新增测试）

---

## 一、问题背景

v0.13.2 引入技术指标与波动率批量预计算（`precompute_technical_factors`）后，
发现在部分样本上 `volatility_20` 及其衍生指标 `zscore_volatility_20`、`spec_score`
与优化前结果存在数值不一致。

### 根因：`ret_1` 构造口径不统一

预计算函数中，当 `daily_adj` 缺少 `ret_1` 列时，直接使用 `pct_chg / 100` 作为
收益率来源：

```python
# 旧代码（存在问题）
if 'ret_1' in daily_adj.columns:
    vol_input['ret_1'] = daily_adj['ret_1'].values
else:
    vol_input['ret_1'] = daily_adj['pct_chg'].values / 100.0  # ← 口径问题
```

而 `pct_chg` 字段来自 Tushare 原始涨跌幅（可能基于未复权或前复权价格计算），
与通过 `close_adj`（后复权价）逐日 `pct_change()` 得到的复权收益率存在微小差异，
导致滚动标准差（波动率）产生偏差。

---

## 二、修复方案

### 2.1 新增共用函数 `compute_ret_1`

新增 `src/lazybull/factors/returns.py`，提供统一的 `ret_1` 构造逻辑：

| 优先级 | 条件 | 处理方式 |
|--------|------|---------|
| 1 | `daily_adj` 已含 `ret_1` 列 | 直接返回，不重新计算 |
| 2 | 含 `close_adj` 列 | 按 `ts_code` 分组、`trade_date` 升序，调用 `pct_change()`（无前瞻、无跨股票边界差分） |
| 3 | 含 `pct_chg` 列 | 使用 `pct_chg / 100`（fallback，记录 WARNING） |
| 4 | 以上均无 | 返回全 NaN，记录 WARNING |

```python
from src.lazybull.factors.returns import compute_ret_1

ret_1 = compute_ret_1(daily_adj)
```

### 2.2 修改预计算函数调用

`precompute_technical_factors.py` 步骤 6（波动率计算）改为调用 `compute_ret_1`：

```python
# 修复后
ret_1_series = compute_ret_1(daily_adj)
ret_col_available = not ret_1_series.isna().all()
if ret_col_available:
    vol_input['ret_1'] = ret_1_series.values
    vol_df = calculate_volatility(vol_input, ...)
```

---

## 三、不变的内容

- `calculate_volatility` 函数的公式、`ddof`、`min_periods` **完全不变**
- 预计算宽表的输出列名 **完全不变**（`volatility_5`、`volatility_10`、`volatility_20`）
- FeatureBuilder 的缓存机制 **不变**
- 模型特征列表 **不变**
- 如果用户数据中已存在 `ret_1` 列，行为与之前 **完全一致**

---

## 四、影响范围

- **需要重新生成 features**：如果历史 features 文件中 `volatility_20` 等指标
  是通过旧版本（v0.13.2）计算的，且当时 `daily_adj` 缺少 `ret_1` 列但含有
  `close_adj` 列，则新旧结果会有微小差异（来自 `pct_chg` 与复权 pct_change
  之间的口径差）。建议重新构建 features。
- **衍生指标自动修复**：`zscore_volatility_20` 与 `spec_score` 均由
  `volatility_20` 派生，随 `volatility_20` 修复后自动一致，无需额外处理。

---

## 五、测试覆盖

新增测试类 `TestComputeRet1` 与 `TestVolatilityRet1Consistency`：

- `test_priority1_uses_existing_ret_1`：已有 `ret_1` 时直接返回
- `test_priority2_uses_close_adj_pct_change`：`close_adj` 路径结果与参考实现一致
- `test_priority2_no_cross_stock_leakage`：每支股票第一日为 NaN，不产生跨股票差分
- `test_priority3_fallback_pct_chg_with_warning`：fallback 行为可控且有 warning
- `test_priority4_all_nan_with_warning`：全缺失时返回 NaN 并有 warning
- `test_result_aligned_to_original_index`：结果索引与输入对齐
- `test_volatility_20_consistent_with_close_adj_pct_change`：端到端一致性验证
- `test_volatility_differs_from_pct_chg_path`：确认修复确实改变了旧口径
- `test_zscore_volatility_20_stable_with_close_adj`：zscore_volatility_20 一致性
