# cs_zscore 标签变换：训练/验证口径指南

## 概述

`cs_zscore`（截面 z-score）是一种针对回归标签的标准化方法：
对每个交易日内的所有样本，先截断极端值（winsorize），再进行截面标准化（均值=0，标准差=1）。

使用方式：
```bash
python scripts/train_ml_model.py --label-transform cs_zscore --task regression ...
python scripts/walk_forward.py --label-transform cs_zscore --task regression ...
```

---

## 为什么需要 cs_zscore？

原始收益率标签（如 `neu_y_ret_20`，20日行业中性化收益）在不同时间段的分布差异很大（牛市均值高，熊市均值低），直接作为回归目标会导致模型难以收敛，且不同时期的 label 不可比。

`cs_zscore` 将每日截面内的收益率标准化为均值=0、标准差=1 的分布，使模型学习的是相对强弱（排序），而不是绝对收益水平。

---

## 训练/验证切分的口径要求

### 关键原则：先按日期切分，再各自独立变换

`cs_zscore` 的截面统计量（均值、标准差）依赖当日参与计算的样本集合。  
若不同集合（训练集 vs 验证集）共享统计量，会产生以下问题：

| 问题 | 描述 |
|------|------|
| **数据泄露** | 训练集某日的 label 值依赖了验证集样本，破坏了时序隔离 |
| **评估失真** | 验证集 label 依赖训练集样本，评估指标偏乐观 |

### 正确做法（v0.13.4 起）

```
1. 按 trade_date 粒度切分 → 保证同日样本全部在同一集合
2. 对训练集独立做 cs_zscore（统计量仅基于训练集样本）
3. 对验证集独立做 cs_zscore（统计量仅基于验证集样本）
```

### 错误做法（已修复）

```
❌ 先对全量数据做 cs_zscore → 再按行数比例切分
   → 边界交易日的 label 依赖了另一侧的样本
```

---

## 样本外测试集（walk_forward OOS）

测试集不参与任何 `cs_zscore` 变换。  
OOS 评估使用原始标签列（`args.label_column`，如 `y_ret_5`），
通过 `evaluate_validation_daily` 计算 RankIC、TopK 平均收益等指标。

---

## 相关代码

| 功能 | 位置 |
|------|------|
| cs_zscore 变换函数 | `src/lazybull/ml/train_core.py::transform_labels_cs_zscore` |
| 按日期切分函数 | `src/lazybull/ml/train_core.py::split_train_val_by_date` |
| 训练数据准备 | `src/lazybull/ml/train_core.py::prepare_training_data` |
| 截面 z-score 底层实现 | `src/lazybull/common/feature_utils.py::cross_sectional_zscore` |

---

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--label-transform` | `raw` | 标签变换方式，`cs_zscore` 启用截面标准化 |
| `--winsorize-p` | `0.01` | 截断比例，截断上下 1% 极端值 |
| `--task` | `regression` | 仅 `regression` 任务时 cs_zscore 生效 |
