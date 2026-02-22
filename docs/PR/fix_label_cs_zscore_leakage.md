# 修复 `label_transform=cs_zscore` 场景下的数据泄露问题

**版本**: v0.13.4  
**日期**: 2026-02-22  
**影响文件**: `src/lazybull/ml/train_core.py`、`scripts/train_ml_model.py`、`scripts/walk_forward.py`

---

## 一、问题描述

### 1.1 背景

`cs_zscore`（截面 z-score）标签变换是将每个交易日内所有样本的标签按截面进行 winsorize + 标准化（均值=0，标准差=1）。此变换的统计量（均值和标准差）依赖当日所有参与计算的样本。

### 1.2 原问题

**旧逻辑**（修复前）：

```
1. 对全量 df 做 cs_zscore 变换（transform_labels_cs_zscore）
   → 此时每日截面统计量包含了该日的所有样本（包括将落入验证集的样本）
2. 调用 prepare_training_data 按行数比例（iloc）切分 train/val
   → 部分交易日的样本可能被拆到 train 侧，另一部分被拆到 val 侧
```

这产生了两个问题：

| 问题 | 描述 |
|------|------|
| **数据泄露** | 训练集某边界交易日的 label 是用包含验证集样本的截面统计量计算的，导致验证集信息混入训练集 |
| **评估口径不一致** | 验证集的 label 是用包含训练集样本的截面统计量计算的，不是验证期真实的独立截面统计 |

### 1.3 影响范围

- `scripts/train_ml_model.py`：当 `--label-transform cs_zscore` + `--task regression` 时
- `scripts/walk_forward.py`：当 `--label-transform cs_zscore` + `--task regression` 时

---

## 二、修复方案

### 2.1 新增 `split_train_val_by_date()`

在 `src/lazybull/ml/train_core.py` 中新增共用切分函数：

```python
def split_train_val_by_date(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
    date_col: str = 'trade_date'
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    ...
```

**切分规则**：
- 获取排序后的唯一交易日列表（`all_dates`）
- 最后 `ceil(n_dates * val_ratio)` 个日期作为验证集
- 剩余日期作为训练集
- 使用集合成员判断（`isin`），保证同一交易日的所有样本不被拆分到两侧

### 2.2 修改 `prepare_training_data()`

- 将旧的 `iloc` 按行切分替换为调用 `split_train_val_by_date`
- 新增可选参数 `label_transform_fn`：若提供，则在切分后分别对 `df_train_split` 和 `df_val_split` 独立调用，不共享截面统计量

### 2.3 修改 `train_ml_model.py`

**修复前**：
```python
if args.label_transform == "cs_zscore":
    df = transform_labels_cs_zscore(df, ...)  # 先对全量变换
# 再切分
X_train, y_train, X_val, y_val, ... = prepare_training_data(df, ...)
```

**修复后**：
```python
# 通过 label_transform_fn 在切分后各自独立变换
label_transform_fn = None
if args.task == "regression" and args.label_transform == "cs_zscore":
    label_transform_fn = lambda d: transform_labels_cs_zscore(d, ...)
X_train, y_train, X_val, y_val, ... = prepare_training_data(
    df, actual_label_column, label_transform_fn=label_transform_fn
)
```

### 2.4 修改 `walk_forward.py`

与 `train_ml_model.py` 相同的修复逻辑，应用于每个 split 的内部训练/验证切分。

---

## 三、新逻辑流程

```
加载全量数据 df
↓
（分类任务）生成二分类标签（按日截面，无泄露）
↓
调用 prepare_training_data(df, label_column, label_transform_fn=fn)
  ├─ 过滤样本（is_st/is_suspended/is_limit_up/is_limit_down）
  ├─ 移除标签 NaN
  ├─ split_train_val_by_date()  ← 按 trade_date 粒度切分
  │    → train_dates 与 val_dates 不相交
  ├─ （若 cs_zscore）对 df_train_split 独立变换
  ├─ （若 cs_zscore）对 df_val_split 独立变换
  └─ 构建 X_train / y_train / X_val / y_val
↓
训练模型
```

---

## 四、影响评估

| 项目 | 影响 |
|------|------|
| **模型训练结果** | cs_zscore 模式下，训练集和验证集的 label 分布略有差异（各自独立标准化），模型效果可能小幅变化 |
| **`raw` 模式** | 不受影响，`label_transform_fn=None` 时行为与旧逻辑相同（除日期切分外） |
| **分类任务** | 不受影响，`generate_classification_labels` 每日独立，无跨集合污染 |
| **接口兼容性** | `prepare_training_data` 新增参数为可选，旧有调用方无需修改 |
| **测试** | 新增 12 个单元测试，全部通过 |
