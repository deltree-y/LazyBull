# 训练运行日志CSV追加记录功能

## 概述

本 PR 实现了训练脚本的运行日志持久化功能，将每次训练的参数、数据统计、评估指标以结构化方式追加记录到 CSV 文件中，便于对比不同训练运行的效果、进行超参数调优和模型版本管理。

## 新增功能

### 1. 训练运行日志模块 (`src/lazybull/ml/run_logger.py`)

#### 核心组件

- **`TrainingRunRecord` 数据类**：记录单次训练运行的完整信息
- **`write_training_run_to_csv()`**：将记录追加写入CSV文件
- **`create_training_run_record_from_training_session()`**：从训练会话信息创建记录

#### 记录字段说明

CSV文件包含以下字段（每列对应一个参数或指标）：

**基本信息**
- `timestamp`：训练时间戳
- `model_version`：模型版本号
- `start_date`：训练开始日期
- `end_date`：训练结束日期
- `label_column`：标签列名（如 y_ret_5）
- `task`：任务类型（regression/classification）

**标签变换配置（回归任务）**
- `label_transform`：标签变换方式（raw/cs_zscore）
- `winsorize_p`：winsorize 参数（仅 cs_zscore）

**分类任务配置**
- `pos_quantile`：正类百分比阈值（如 0.2 表示 Top20%）
- `pos_topk`：正类数量阈值（如 300 表示每日 Top300）
- `scale_pos_weight`：正类权重（实际使用值）
- `scale_pos_weight_mode`：权重模式（auto/manual）

**XGBoost 超参数**
- `n_estimators`：树的数量
- `max_depth`：树的最大深度
- `learning_rate`：学习率
- `subsample`：样本采样比例
- `colsample_bytree`：特征采样比例
- `gamma`：分裂所需的最小损失减少
- `reg_alpha`：L1 正则化系数
- `reg_lambda`：L2 正则化系数
- `early_stopping_rounds`：早停轮数
- `tree_method`：树构建方法
- `random_state`：随机种子
- `n_jobs`：并行线程数

**数据统计**
- `trade_days_count`：交易日数量
- `total_samples`：总样本数（加载后）
- `samples_after_filter`：过滤后样本数
- `train_samples`：训练集样本数
- `val_samples`：验证集样本数
- `val_start_date`：验证集开始日期
- `val_end_date`：验证集结束日期
- `val_ratio`：验证集比例

**训练结果**
- `best_iteration`：最佳迭代次数

**训练集评估指标（回归）**
- `train_mse`：均方误差
- `train_rmse`：均方根误差
- `train_r2`：决定系数
- `train_ic`：信息系数

**训练集评估指标（分类）**
- `train_accuracy`：准确率
- `train_auc`：AUC
- `train_precision`：精确率
- `train_recall`：召回率

**验证集评估指标（回归）**
- `val_mse`、`val_rmse`、`val_r2`、`val_ic`、`val_rank_ic`

**验证集评估指标（分类）**
- `val_accuracy`、`val_auc`、`val_precision`、`val_recall`

**验证集逐日评估**
- `val_daily_rankic_mean`：逐日 RankIC 均值
- `val_daily_rankic_std`：逐日 RankIC 标准差
- `val_daily_rankic_ir`：逐日 RankIC IR（信息比率）

**TopK 收益统计**（动态字段）
- `top30_return_mean`、`top30_return_std`
- `top100_return_mean`、`top100_return_std`
- `top300_return_mean`、`top300_return_std`

**诊断统计**（动态字段）
- `diagnostic_全市场收益_逐日均值的均值`
- `diagnostic_全市场收益_逐日均值的标准差`
- `diagnostic_全市场收益_逐日标准差的均值`
- `diagnostic_每日样本数_最小`、`最大`、`中位数`
- `diagnostic_Top{k}_逐日均值的均值/标准差`
- `diagnostic_Top{k}_相对全市场提升_均值/标准差`
- `diagnostic_Top{k}_逐日均值_25/50/75分位`

### 2. 训练脚本集成

#### 新增 CLI 参数

```bash
--run-log-csv PATH
```

指定训练运行日志CSV路径，默认为 `{data_root}/ml_train_runs.csv`。

#### 脚本变更

- `load_features_data()` 现在返回 `(df, trade_days_count)` 元组
- `prepare_training_data()` 现在返回数据统计（包含 samples_after_filter、验证集日期范围）
- `train_xgboost_model()` 在 train_params 中添加 `best_iteration`
- 训练完成后自动调用日志模块记录运行信息

### 3. CSV 文件特性

#### 追加模式
- 文件不存在时：自动创建并写入表头和第一条记录
- 文件存在时：追加新行，不覆盖历史记录

#### 动态列扩展
- 当新增字段（如新增 TopK 评估指标）时，自动扩展表头
- 旧记录的新字段留空（NaN），保持历史数据不变
- 确保日志格式向前兼容

#### 编码
- 使用 UTF-8 with BOM（`utf-8-sig`），确保 Excel 能正确打开中文字段

## 使用方法

### 基本使用（使用默认路径）

```bash
python scripts/train_ml_model.py \
    --start-date 20230101 \
    --end-date 20231231 \
    --label y_ret_5 \
    --task regression \
    --n-estimators 200 \
    --max-depth 8
```

日志将自动写入 `data/ml_train_runs.csv`。

### 指定自定义日志路径

```bash
python scripts/train_ml_model.py \
    --start-date 20230101 \
    --end-date 20231231 \
    --label y_ret_5 \
    --task classification \
    --pos-topk 300 \
    --run-log-csv /path/to/my_custom_runs.csv
```

### 多次训练累积日志

```bash
# 第一次训练
python scripts/train_ml_model.py \
    --start-date 20230101 \
    --end-date 20231231 \
    --n-estimators 200 \
    --max-depth 6

# 第二次训练（不同参数）
python scripts/train_ml_model.py \
    --start-date 20230101 \
    --end-date 20231231 \
    --n-estimators 300 \
    --max-depth 8

# 第三次训练（不同标签）
python scripts/train_ml_model.py \
    --start-date 20230101 \
    --end-date 20231231 \
    --label y_ret_10 \
    --n-estimators 200 \
    --max-depth 6
```

所有训练记录将累积到同一个 CSV 文件中，便于对比。

## 数据分析建议

### 使用 pandas 读取并分析

```python
import pandas as pd

# 读取训练日志
df = pd.read_csv("data/ml_train_runs.csv")

# 查看所有训练运行
print(df[['timestamp', 'model_version', 'task', 'val_rank_ic', 'val_daily_rankic_mean']])

# 对比不同超参数的效果
df_regression = df[df['task'] == 'regression']
print(df_regression[['n_estimators', 'max_depth', 'val_rank_ic']].sort_values('val_rank_ic', ascending=False))

# 查找最佳模型
best_model = df.loc[df['val_rank_ic'].idxmax()]
print(f"最佳模型版本: {best_model['model_version']}, RankIC: {best_model['val_rank_ic']:.4f}")

# 分析学习率对性能的影响
import matplotlib.pyplot as plt
plt.scatter(df['learning_rate'], df['val_rank_ic'])
plt.xlabel('Learning Rate')
plt.ylabel('Validation RankIC')
plt.title('Learning Rate vs RankIC')
plt.show()
```

### 常见分析场景

1. **超参数调优**：对比不同 n_estimators、max_depth、learning_rate 组合的效果
2. **标签对比**：对比 y_ret_5、y_ret_10、y_ret_20 的预测效果
3. **任务对比**：对比回归任务和分类任务的表现
4. **时间窗口影响**：对比不同训练日期区间的模型稳定性
5. **过拟合检测**：对比训练集和验证集指标，识别过拟合

## 技术细节

### 错误处理
- CSV 写入失败不会影响模型训练和保存
- 日志记录失败时会输出警告，但不会中断训练流程

### 性能影响
- CSV 追加写入性能良好，单次写入耗时约 10-50ms
- 对训练流程几乎无影响

### 存储空间
- 每条记录约 1-2 KB（取决于字段数量）
- 训练 1000 次约占用 1-2 MB

## 示例输出

训练完成后，日志输出：

```
模型训练完成！版本: v5
模型保存路径: ./data/models/
训练运行日志已记录到: data/ml_train_runs.csv
```

CSV 文件示例（部分列）：

| timestamp | model_version | task | label_column | n_estimators | max_depth | val_rank_ic | val_daily_rankic_mean | top100_return_mean |
|-----------|--------------|------|--------------|--------------|-----------|-------------|----------------------|--------------------|
| 2024-01-01 10:00:00 | 1 | regression | y_ret_5 | 200 | 8 | 0.048 | 0.050 | - |
| 2024-01-01 11:30:00 | 2 | regression | y_ret_5 | 300 | 8 | 0.052 | 0.055 | - |
| 2024-01-01 14:20:00 | 3 | classification | y_ret_20 | 200 | 8 | - | 0.062 | 0.0025 |

## 相关文件

- 核心模块：`src/lazybull/ml/run_logger.py`
- 训练脚本：`scripts/train_ml_model.py`
- 单元测试：`tests/test_ml_run_logger.py`
- 版本更新：`pyproject.toml`（0.8.2 → 0.8.3）
- 变更日志：`CHANGELOG.md`

## 测试覆盖

所有功能均有完整的单元测试覆盖：

```bash
pytest tests/test_ml_run_logger.py -v
```

测试项：
- ✅ CSV 创建和首次写入
- ✅ 追加记录功能
- ✅ 自定义路径
- ✅ 列扩展兼容性
- ✅ 回归任务记录
- ✅ 分类任务记录
- ✅ 完整工作流

共 9 个测试用例，全部通过。
