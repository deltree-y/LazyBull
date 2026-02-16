# Walk-forward（滚动训练）能力实现

## 概述

本 PR 实现了 Walk-forward 滚动训练能力，用于模拟实盘"定期刷新模型"的场景。通过按固定频率（季度/月度/半年度）滚动训练模型，并在紧随其后的样本外窗口进行评估，形成多段 OOS（Out-of-Sample）表现序列，更真实地评估模型在实际应用中的表现。

## 功能特性

### 1. 核心功能

- **滚动训练**：按指定频率（monthly/quarterly/semiannual）滚动生成训练/测试切分
- **固定窗口**：每次训练使用固定长度的历史数据（默认 5 年）
- **样本外评估**：在紧随训练窗口的样本外区间进行评估（默认 6 个月）
- **完整能力复用**：复用 `train_ml_model.py` 的所有能力（训练、评估、模型注册、日志记录等）

### 2. Walk-forward 切分口径

每个 split 包含：
- **训练区间**：`[train_start, train_end]`，长度由 `--train-window-years` 控制（默认 5 年）
- **测试区间**：`[test_start, test_end]`，长度由 `--test-window-months` 控制（默认 6 个月）
- **滚动步长**：由 `--step` 控制（monthly/quarterly/semiannual）

**日期对齐规则**：
- 所有日期自动对齐到交易日（向后查找最近的交易日）
- `train_end` 每次向前推进一个 step 周期
- `test_start` 是 `train_end` 的下一个交易日
- 不引入 gap（训练集和测试集紧密相连）

## 实现细节

### 1. 代码结构

#### 新增模块

1. **`src/lazybull/ml/train_core.py`**
   - 从 `scripts/train_ml_model.py` 抽取的核心训练函数
   - 提供可复用的训练逻辑：
     - `load_features_data()` - 加载特征数据
     - `prepare_training_data()` - 准备训练数据
     - `transform_labels_cs_zscore()` - 标签变换
     - `generate_classification_labels()` - 分类标签生成
     - `train_xgboost_model()` - 训练模型
     - `evaluate_validation_daily()` - 逐日评估

2. **`src/lazybull/ml/walk_forward_utils.py`**
   - Walk-forward 切分工具
   - `generate_walk_forward_splits()` - 生成训练/测试区间切分
   - `WalkForwardSplit` - 切分数据结构
   - `print_splits_summary()` - 打印切分汇总

3. **`scripts/walk_forward.py`**
   - Walk-forward 主脚本
   - 生成切分、执行训练、记录日志、生成汇总

#### 扩展模块

1. **`src/lazybull/ml/run_logger.py`**
   - 扩展 `TrainingRunRecord` 支持 walk-forward 字段：
     - `wf_run_id` - walk-forward 运行 ID
     - `split_index` - 切分索引
     - `step_frequency` - 滚动频率
     - `test_start_date` / `test_end_date` - 测试区间日期

### 2. 输出与记录

#### ml_train_runs.csv

每个 split 作为一次完整的训练运行，追加写入到 `ml_train_runs.csv`，包含：
- 所有原有字段（训练配置、超参数、评估指标等）
- 新增 walk-forward 字段（wf_run_id、split_index、step_frequency、test_start_date、test_end_date）
- 测试集（OOS）的逐日评估指标（`test_*` 前缀）

#### walk_forward_summary.csv

专门为 walk-forward 生成的汇总文件，包含：
- 每个 split 的基本信息（split_index、训练/测试日期区间、样本数）
- 关键 OOS 指标（RankIC 均值/标准差/IR、TopK 收益统计、诊断统计）
- 便于快速画曲线和分析

文件位置：`data/walk_forward/walk_forward_summary_{wf_run_id}.csv`

### 3. 模型注册

每个 split 注册一个独立的模型版本（vXX），metadata 包含：
- 训练/测试日期区间
- walk-forward 运行 ID 和 split 索引
- 完整的性能指标（训练集、验证集、测试集）

## 使用方法

### 基础用法

```bash
# 使用默认参数（季度滚动，5年训练窗口，6个月测试窗口）
python scripts/walk_forward.py --wf-start-date 20180101 --wf-end-date 20231231
```

### 自定义滚动频率

```bash
# 按月度滚动
python scripts/walk_forward.py --wf-start-date 20180101 --wf-end-date 20231231 --step monthly

# 按季度滚动（默认）
python scripts/walk_forward.py --wf-start-date 20180101 --wf-end-date 20231231 --step quarterly

# 按半年度滚动
python scripts/walk_forward.py --wf-start-date 20180101 --wf-end-date 20231231 --step semiannual
```

### 自定义窗口大小

```bash
# 3年训练窗口，3个月测试窗口
python scripts/walk_forward.py --wf-start-date 20180101 --wf-end-date 20231231 \
    --train-window-years 3 --test-window-months 3
```

### 透传训练参数

```bash
# 分类任务 + Top300 + y_ret_20
python scripts/walk_forward.py --wf-start-date 20180101 --wf-end-date 20231231 \
    --task classification --pos-topk 300 --label y_ret_20

# 回归任务 + cs_zscore 标签变换
python scripts/walk_forward.py --wf-start-date 20180101 --wf-end-date 20231231 \
    --task regression --label-transform cs_zscore --label y_ret_20

# 自定义 XGBoost 超参数
python scripts/walk_forward.py --wf-start-date 20180101 --wf-end-date 20231231 \
    --n-estimators 300 --max-depth 10 --learning-rate 0.03
```

## 输出文件说明

### ml_train_runs.csv 新增字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| wf_run_id | str | walk-forward 运行 ID（例如 `wf_20240101_120000_abc123`） |
| split_index | int | 切分索引（在一次 walk-forward 运行中的序号，从 0 开始） |
| step_frequency | str | 滚动频率（monthly/quarterly/semiannual） |
| test_start_date | str | 样本外测试开始日期（YYYYMMDD） |
| test_end_date | str | 样本外测试结束日期（YYYYMMDD） |
| test_* | float | 测试集逐日评估指标（例如 `test_daily_rankic_mean`） |

### walk_forward_summary.csv 字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| split_index | int | 切分索引 |
| train_start | str | 训练开始日期 |
| train_end | str | 训练结束日期 |
| test_start | str | 测试开始日期 |
| test_end | str | 测试结束日期 |
| model_version | int | 模型版本号 |
| train_samples | int | 训练集样本数 |
| val_samples | int | 验证集样本数 |
| test_samples | int | 测试集样本数 |
| daily_rankic_mean | float | 测试集逐日 RankIC 均值 |
| daily_rankic_std | float | 测试集逐日 RankIC 标准差 |
| daily_rankic_ir | float | 测试集逐日 RankIC IR |
| top30_return_mean | float | Top30 平均收益（测试集） |
| top100_return_mean | float | Top100 平均收益（测试集） |
| top300_return_mean | float | Top300 平均收益（测试集） |
| diagnostic_* | float | 诊断统计（全市场收益、提升、分位数等） |

## 与 train_ml_model.py 的关系

### 复用能力

Walk-forward 脚本完全复用以下能力：
- ✅ 训练集/验证集评估
- ✅ 验证集逐日评估（贴近交易场景）
- ✅ 模型注册（ModelRegistry）
- ✅ 训练运行日志 CSV 追加写入（ml_train_runs.csv）
- ✅ 诊断报告（诊断统计）
- ✅ 所有训练参数透传（task、label、pos_topk/pos_quantile、XGBoost 超参数、label_transform 等）

### 区别

| 特性 | train_ml_model.py | walk_forward.py |
|------|------------------|----------------|
| 训练次数 | 1 次 | 多次（按 step 滚动） |
| 测试集 | 无独立测试集 | 有样本外测试集（OOS） |
| 输出 | 1 个模型版本 | 多个模型版本 |
| 汇总文件 | 无 | walk_forward_summary.csv |
| wf_run_id | 无 | 有 |

## 测试

新增单元测试 `tests/test_walk_forward.py`，覆盖：
- ✅ split 生成逻辑（季度/月度/半年度）
- ✅ 边界条件（窗口过大、日期范围不足）
- ✅ 切分验证（日期推进、无重叠）
- ✅ 汇总 CSV 生成
- ✅ 与 run_logger 集成（wf 字段写入、动态列扩展）

运行测试：
```bash
python -m pytest tests/test_walk_forward.py -v
```

## 兼容性

- ✅ 向后兼容：旧的 `ml_train_runs.csv` 仍然可用，新字段对旧记录自动留空
- ✅ 不影响现有训练脚本：`train_ml_model.py` 仍然可以独立使用
- ✅ 不引入新的依赖

## 注意事项

1. **数据要求**：确保 walk-forward 时间区间内有完整的特征数据
2. **计算资源**：walk-forward 会训练多个模型，需要更多时间和计算资源
3. **日期对齐**：所有日期自动对齐到交易日，可能与指定日期略有偏差
4. **测试集评估**：重点关注测试集（OOS）的逐日评估指标，这是 walk-forward 的核心价值

## 相关文档

- [Walk-forward 使用指南](../guide/walk_forward_guide.md) - 详细的使用教程
- [ML 使用指南](../ml_usage_guide.md) - 机器学习模型训练指南
- [训练运行日志](../implementation_summary.md) - 日志记录系统说明
