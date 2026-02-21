# 申万行业三级升级（L3）与分层回退中性化 PR 说明

> 版本：v0.13.0（minor 升级）

## 一、背景与目标

- **现状**：申万行业口径为二级（L2，约 100 个子行业），中性化使用单层 `sw_industry` 分组
- **目标**：升级到三级（L3，约 200+ 子行业），并实现 **L3→L2→L1→全市场** 分层回退中性化，
  确保小样本子行业不因样本不足而丢失中性化效果

---

## 二、申万三级行业下载与存储结构

### 下载接口

```python
# 1. 获取 L3 指数列表
index_classify = client.get_index_classify(level="L3", src="SW2021")

# 2. 逐行业获取成分股（含 L1/L2/L3 完整层级）
members = client.get_index_member(l3_code=index_code)
# 返回字段：ts_code, l1_code, l1_name, l2_code, l2_name, l3_code, l3_name,
#           in_date, out_date, is_new
```

### 存储结构

单文件 `data/raw/shenwan_industry.parquet`，字段：

| 字段名 | 说明 |
|--------|------|
| ts_code | 股票代码 |
| sw_l1_code | 申万一级行业代码 |
| sw_l1 | 申万一级行业名称 |
| sw_l2_code | 申万二级行业代码 |
| sw_l2 | 申万二级行业名称 |
| sw_l3_code | 申万三级行业代码 |
| sw_l3 | 申万三级行业名称 |
| in_date | 纳入日期（若可得） |

---

## 三、features 输出字段

### 主字段（映射到 L3）

| 字段名 | 说明 |
|--------|------|
| sw_industry | 申万三级行业名称（中性化分组键） |
| sw_industry_code | 申万三级行业代码 |
| sw_industry_id | 三级行业稳定整数编码 |

### 辅助字段

| 字段名 | 说明 |
|--------|------|
| sw_l2 | 申万二级行业名称 |
| sw_l2_code | 申万二级行业代码 |
| sw_l2_id | 二级行业稳定整数编码 |
| sw_l1 | 申万一级行业名称 |
| sw_l1_code | 申万一级行业代码 |
| sw_l1_id | 一级行业稳定整数编码 |

---

## 四、分层回退中性化规则

### 触发条件

对每列、每支股票独立执行，按以下顺序确定使用哪层统计量：

```
L3 行业内 tradable==1 样本数 >= min_group_size(=5) → 使用 L3 统计
  否则 L2 行业内 tradable==1 样本数 >= 5 → 使用 L2 统计
    否则 L1 行业内 tradable==1 样本数 >= 5 → 使用 L1 统计
      否则 → 使用全市场（所有 tradable==1）统计
```

### 实现模块

`src/lazybull/factors/hierarchical_industry_neutralization.py`

- `hierarchical_zscore(df, columns, l3_col, l2_col, l1_col, ...)` → 指标 Z-Score（`zscore_` 前缀）
- `hierarchical_demean(df, columns, l3_col, l2_col, l1_col, ...)` → 收益率/标签去均值（`neu_` 前缀）

### 自动检测路径

`FeatureBuilder._apply_industry_neutralization()` 自动检测：
- 若 `sw_industry_code`、`sw_l2_code`、`sw_l1_code` 均存在 → 启用分层回退路径
- 否则 → 退化为单层 `sw_industry` 中性化（向后兼容）

---

## 五、新增特征定义（v0.12.1 已实现，本 PR 不新增）

本 PR 不新增特征，继承 v0.12.1 的个股特征与市场状态特征：
- `is_new_stock`, `size`, `zscore_size`, `spec_score`
- `mkt_vol_cnt`, `mkt_vol_20`, `mkt_turnover_ratio`, `mkt_ret_avg_20`, `mkt_turnover_std`, `mkt_adv_dec_ratio`

---

## 六、重建 features 与重训模型命令

```bash
# 步骤1：重新下载申万三级行业数据
python scripts/update_basic_data.py --only-shenwan --force

# 步骤2：验证下载结果
python -c "
from src.lazybull.data import Storage
s = Storage()
sw = s.load_raw('shenwan_industry')
print(sw.columns.tolist())
print(sw.head())
print('L1 行业数:', sw['sw_l1'].nunique())
print('L2 行业数:', sw['sw_l2'].nunique())
print('L3 行业数:', sw['sw_l3'].nunique())
"

# 步骤3：重建特征
python scripts/build_features.py --start-date 20240101 --end-date 20241231

# 步骤4：重训模型
python scripts/train_ml_model.py
```

---

## 七、Breaking Changes

1. `DataCleaner.clean_shenwan_industry()` 默认 `level_str` 从 `'l2'` 改为 `'l3'`
2. 旧式 `shenwan_industry.parquet`（含 `sw_code`/`sw_name` 字段）仍可使用，
   `_merge_shenwan_industry()` 会自动降级为单层处理
3. 若需完整分层回退功能，**必须重新下载** L3 格式的行业数据

---

## 八、测试覆盖

| 测试文件 | 覆盖点 |
|---------|--------|
| `tests/test_sw_industry_l2.py` | L3 格式合并、L2 向后兼容、中性化字段命名 |
| `tests/test_sw_industry_l3.py` | L3/L2/L1/全市场各层回退精确断言 |
| `tests/test_market_and_new_features.py` | 个股特征与市场状态特征（继承） |
