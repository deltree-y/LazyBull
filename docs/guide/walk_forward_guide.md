# Walk-forward 滚动训练使用指南

## 什么是 Walk-forward？

Walk-forward（滚动训练）是一种模型评估方法，用于模拟实盘场景中"定期刷新模型"的情况。通过按固定频率（如每季度）重新训练模型，并在紧随其后的样本外窗口进行评估，可以更真实地了解模型在实际应用中的表现。

### 为什么需要 Walk-forward？

1. **避免过拟合**：传统的单次训练可能过度拟合特定时期的数据
2. **模拟实盘**：实盘中模型会定期更新，walk-forward 模拟了这一过程
3. **评估稳定性**：通过多段 OOS 表现序列，可以评估模型的稳定性和鲁棒性
4. **发现时效性**：观察模型表现随时间的变化趋势

### Walk-forward 的基本原理

```
时间轴：  |-------- 训练窗口 --------|--- OOS 测试 ---|-------- 训练窗口 --------|--- OOS 测试 ---|
          [  5年历史数据  ] [  6个月  ]   [  5年历史数据  ] [  6个月  ]
                                     ↓ 向前滚动（例如：每季度）
```

- **训练窗口**：固定长度（默认 5 年），用于训练模型
- **OOS 测试窗口**：紧随训练窗口（默认 6 个月），用于评估模型
- **滚动步长**：每次向前推进的时间（默认每季度）

## 快速开始

### 1. 最简单的用法

```bash
python scripts/walk_forward.py \
    --split-count 12 \
    --final-date 20231231
```

这将使用默认配置：
- 滚动频率：每季度
- 训练窗口：5 年
- 测试窗口：6 个月
- 任务类型：回归（y_ret_5）

### 2. 查看帮助

```bash
python scripts/walk_forward.py --help
```

## 配置参数详解

### Walk-forward 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--split-count` | int | 必填 | 切分数量（正整数） |
| `--final-date` | str | 必填 | 最终日期（YYYYMMDD）。启用部署训练时表示部署训练数据最后一天；禁用部署训练时表示最后一个 split 测试结束日 |
| `--step` | str | quarterly | 滚动频率：monthly（月度）、quarterly（季度）、semiannual（半年） |
| `--train-window-years` | int | 5 | 训练窗口长度（年） |
| `--test-window-months` | int | 6 | 测试窗口长度（月） |
| `--val-ratio` | float | 0.2 | 训练数据内部验证集比例 |

### 训练参数（与 train_ml_model.py 一致）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--task` | str | regression | 任务类型：regression（回归）、classification（分类） |
| `--label` | str | y_ret_5 | 标签列：y_ret_5、y_ret_10、y_ret_20 |
| `--label-transform` | str | raw | 标签变换：raw（原始）、cs_zscore（截面标准化） |
| `--pos-topk` | int | None | 分类任务正类数量阈值（例如 300） |
| `--pos-quantile` | float | None | 分类任务正类百分比阈值（例如 0.2） |
| `--n-estimators` | int | 200 | XGBoost 树的数量 |
| `--max-depth` | int | 8 | XGBoost 树的最大深度 |
| `--learning-rate` | float | 0.05 | XGBoost 学习率 |

更多参数请参考 `--help` 或 [ML 使用指南](../ml_usage_guide.md)。

## 使用示例

### 示例 1：按月度滚动（更频繁的模型更新）

```bash
python scripts/walk_forward.py \
    --split-count 24 \
    --final-date 20231231 \
    --step monthly \
    --train-window-years 3 \
    --test-window-months 1
```

**适用场景**：
- 需要更频繁地更新模型
- 评估模型在短期内的表现
- 数据量充足，可以支持更多次训练

### 示例 2：分类任务 + Top300

```bash
python scripts/walk_forward.py \
    --split-count 12 \
    --final-date 20231231 \
    --task classification \
    --pos-topk 300 \
    --label y_ret_20
```

**适用场景**：
- 选股策略（每次选出 Top300）
- 评估分类模型的稳定性

### 示例 3：回归任务 + cs_zscore 标签变换

```bash
python scripts/walk_forward.py \
    --split-count 12 \
    --final-date 20231231 \
    --task regression \
    --label-transform cs_zscore \
    --label y_ret_20
```

**适用场景**：
- 需要标签标准化，减少异常值影响
- 回归模型训练

### 示例 4：自定义 XGBoost 超参数

```bash
python scripts/walk_forward.py \
    --split-count 12 \
    --final-date 20231231 \
    --n-estimators 300 \
    --max-depth 10 \
    --learning-rate 0.03 \
    --subsample 0.7 \
    --colsample-bytree 0.7
```

**适用场景**：
- 调优超参数
- 实验不同的模型配置

## 输出文件说明

Walk-forward 运行后会生成以下文件：

### 1. ml_train_runs.csv

位置：`data/models/ml_train_runs.csv`

每个 split 的训练记录都会追加到这个文件，包含：
- 训练配置（日期区间、标签、任务类型）
- XGBoost 超参数
- 训练集/验证集/测试集评估指标
- walk-forward 相关字段（wf_run_id、split_index、step_frequency、test_start_date、test_end_date）

### 2. walk_forward_summary.csv

位置：`data/walk_forward/walk_forward_summary_{wf_run_id}.csv`

汇总所有 split 的关键指标，包含：
- 每个 split 的基本信息（日期区间、样本数）
- 测试集（OOS）关键指标：
  - `daily_rankic_mean` - 逐日 RankIC 均值
  - `daily_rankic_std` - 逐日 RankIC 标准差
  - `daily_rankic_ir` - 逐日 RankIC IR
  - `top30_return_mean` - Top30 平均收益
  - `top100_return_mean` - Top100 平均收益
  - `top300_return_mean` - Top300 平均收益
  - 诊断统计（全市场收益、提升、分位数等）

### 3. 模型文件

位置：`data/models/v{XX}_model.joblib`

每个 split 都会注册一个独立的模型版本，保存在 models 目录。

## 结果分析

### 1. 读取汇总文件

```python
import pandas as pd

# 读取汇总文件
df = pd.read_csv('data/walk_forward/walk_forward_summary_wf_20240101_120000_abc123.csv')

print(df.head())
```

### 2. 可视化 OOS 表现趋势

```python
import matplotlib.pyplot as plt

# RankIC 趋势
plt.figure(figsize=(12, 6))
plt.plot(df['split_index'], df['daily_rankic_mean'], marker='o')
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.xlabel('Split Index')
plt.ylabel('Daily RankIC Mean')
plt.title('Walk-forward OOS RankIC Trend')
plt.grid(True, alpha=0.3)
plt.show()

# Top30 收益趋势
plt.figure(figsize=(12, 6))
plt.plot(df['split_index'], df['top30_return_mean'], marker='o', label='Top30')
plt.plot(df['split_index'], df['top100_return_mean'], marker='s', label='Top100')
plt.plot(df['split_index'], df['top300_return_mean'], marker='^', label='Top300')
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.xlabel('Split Index')
plt.ylabel('Average Return')
plt.title('Walk-forward OOS TopK Returns Trend')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 3. 统计分析

```python
# 计算 OOS 指标的均值和标准差
print("OOS 表现统计：")
print(f"RankIC 均值: {df['daily_rankic_mean'].mean():.4f} ± {df['daily_rankic_mean'].std():.4f}")
print(f"Top30 收益均值: {df['top30_return_mean'].mean():.4f} ± {df['top30_return_mean'].std():.4f}")
print(f"Top100 收益均值: {df['top100_return_mean'].mean():.4f} ± {df['top100_return_mean'].std():.4f}")

# 计算稳定性（RankIC > 0 的比例）
stability = (df['daily_rankic_mean'] > 0).mean()
print(f"RankIC > 0 的比例: {stability:.2%}")
```

### 4. 按时间段分析

```python
# 添加时间列
df['test_start_dt'] = pd.to_datetime(df['test_start'], format='%Y%m%d')
df['year'] = df['test_start_dt'].dt.year

# 按年度汇总
yearly_stats = df.groupby('year').agg({
    'daily_rankic_mean': ['mean', 'std'],
    'top30_return_mean': ['mean', 'std']
})

print("\n按年度统计：")
print(yearly_stats)
```

## 常见问题

### Q1: Walk-forward 需要多长时间？

**答**：取决于以下因素：
- split 数量（由 step 频率和时间跨度决定）
- 每个 split 的样本数
- XGBoost 超参数（n_estimators、max_depth 等）

例如：
- 5 年数据，季度滚动 → 约 20 个 splits
- 每个 split 训练约 5-10 分钟 → 总计 2-3 小时

### Q2: 如何选择 step 频率？

**答**：根据实际需求选择：
- **monthly**：适合高频策略，需要频繁更新模型
- **quarterly**（推荐）：平衡了更新频率和计算成本
- **semiannual**：适合长周期策略，或数据量较少的情况

### Q3: 训练窗口和测试窗口应该设多大？

**答**：
- **训练窗口**：通常 3-5 年，确保有足够的历史数据
- **测试窗口**：通常 3-6 个月，模拟实盘中模型使用的时长

### Q4: Walk-forward 结果不稳定怎么办？

**答**：可能的原因和解决方案：
1. **数据质量问题**：检查特征数据是否完整、是否有异常值
2. **模型过拟合**：尝试增加正则化（gamma、reg_alpha、reg_lambda）
3. **特征选择**：移除不稳定的特征
4. **超参数调优**：使用更保守的超参数（降低 max_depth、增加 min_child_weight）

### Q5: 如何对比不同配置的 Walk-forward 结果？

**答**：
1. 使用不同的 `wf_run_id` 区分不同运行
2. 读取各自的 `walk_forward_summary.csv` 进行对比
3. 关注关键指标：RankIC 均值、IR、Top30 收益均值、稳定性

```python
# 对比两次运行
df1 = pd.read_csv('data/walk_forward/walk_forward_summary_run1.csv')
df2 = pd.read_csv('data/walk_forward/walk_forward_summary_run2.csv')

print("配置 1 - RankIC 均值:", df1['daily_rankic_mean'].mean())
print("配置 2 - RankIC 均值:", df2['daily_rankic_mean'].mean())
```

## 最佳实践

1. **先用小范围测试**：首次使用时，先用短时间范围（如 1-2 年）测试，确保流程正常
2. **选择合适的频率**：不要过度频繁（如每日），会导致计算成本过高且意义不大
3. **关注 OOS 指标**：重点看测试集（test_*）的指标，这是 walk-forward 的核心价值
4. **记录 wf_run_id**：方便后续追溯和对比不同实验
5. **定期清理旧模型**：walk-forward 会产生大量模型文件，定期清理不需要的版本

## 相关文档

- [Walk-forward 实现说明](../PR/walk_forward_implementation.md) - 技术实现细节
- [ML 使用指南](../ml_usage_guide.md) - 机器学习模型训练指南
- [训练脚本文档](../../scripts/train_ml_model.py) - 单次训练脚本说明
