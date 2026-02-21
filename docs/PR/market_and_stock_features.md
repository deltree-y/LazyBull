# PR 说明：新增个股特征与市场状态特征（v0.12.1）

## 概述

本 PR 在 `FeatureBuilder` 构建流程中新增以下特征：

**个股特征（4个）**：
- `is_new_stock`、`size`、`zscore_size`、`spec_score`

**市场状态特征（6个）**：
- `mkt_vol_cnt`、`mkt_vol_20`、`mkt_turnover_ratio`、`mkt_ret_avg_20`、`mkt_turnover_std`、`mkt_adv_dec_ratio`

---

## 新特征列表与公式

### 个股特征

| 特征名 | 公式/来源 | 依赖字段 |
|--------|-----------|----------|
| is_new_stock | list_days < 365 → 1，否则 → 0 | list_days（来自 stock_basic.list_date） |
| size | = circ_mv | daily_basic.circ_mv |
| zscore_size | 行业内 Z-Score(log1p(size))，sw_industry 分组，min_group_size=5 | size, sw_industry, tradable |
| spec_score | zscore_volatility_20 × (−zscore_size) | zscore_volatility_20, zscore_size |

**注意**：`spec_score` 依赖 `zscore_volatility_20`，需在 `build_features_for_day` 中启用 `apply_industry_neutralization=True`。

### 市场状态特征

| 特征名 | 公式 | 说明 |
|--------|------|------|
| mkt_vol_cnt | std(ret_1) | 当日截面，仅 tradable==1 |
| mkt_vol_20 | mean(mkt_vol_cnt(t-19:t)) | 20日滚动均值 |
| mkt_turnover_ratio | sum(amount) / sum(circ_mv) | 当日，仅 tradable==1 |
| mkt_ret_avg_20 | sum_{i=0}^{19} mean_cs(ret_1(t-i)) | 20日收益均值之和 |
| mkt_turnover_std | std(turnover_rate_f) | 当日，优先 turnover_rate_f，fallback turnover_rate |
| mkt_adv_dec_ratio | mean_{i=0}^{59} [(adv+1)/(dec+1)] | 60日滚动，+1 为 Laplace 平滑 |

---

## 依赖字段与缺失时的补齐方式

### 字段依赖链

```
is_new_stock
  └── list_days
        └── list_date (stock_basic)

size / zscore_size
  └── circ_mv (daily_basic)
  └── sw_industry (shenwan_industry, 申万二级)
  └── tradable (clean daily 或 vol>0 代理)

spec_score
  └── zscore_volatility_20 (需 apply_industry_neutralization=True)
  └── zscore_size (见上)

mkt_turnover_ratio
  └── amount (daily_data)
  └── circ_mv (daily_basic)

mkt_turnover_std
  └── turnover_rate_f (daily_basic，优先)
  └── turnover_rate (daily_basic，fallback)
```

### 补齐优先级

1. **先从 clean daily / clean daily_basic merge 补齐**：若数据层已有这些字段，直接使用
2. **若仍缺失，从 TuShare ensure 能力拉取**：使用现有 `TushareClient`/`ensure_*` 函数下载 `daily_basic`（包含 `circ_mv`、`turnover_rate_f`）

---

## 重建 features 与重训模型命令

### 重新生成 features

```bash
# 指定日期范围重建特征（启用行业中性化）
python scripts/build_features.py --start-date 20230101 --end-date 20231231 --apply-neutralization

# 或重建全量
python scripts/build_features.py --apply-neutralization
```

### 重新训练模型

```bash
python scripts/train_ml_model.py
```

---

## 实现位置

- 市场状态特征：`src/lazybull/factors/market_state.py`（新增）
- 个股特征：`src/lazybull/features/builder.py` → `_add_new_individual_features()`
- 市场状态特征接入：`src/lazybull/features/builder.py` → `_add_market_state_features()`

---

## 版本

- 版本号：`0.12.0` → `0.12.1`（patch 增量，仅新增特征，无 Breaking Change）
