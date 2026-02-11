# ML 标签 Horizon 使用指南

## 概述

本指南介绍如何使用 LazyBull 框架的多 horizon 标签功能，包括标签定义、特征构建、模型训练和回测的完整流程。

## 标签定义

### 什么是 Horizon？

Horizon（预测窗口）是指从当前时间点到未来某个时间点的交易日数量。不同的 horizon 对应不同的投资周期：

- **短期（5日）**：适合高频交易，捕捉短期市场波动
- **中期（10日）**：平衡收益和交易成本，适合中等频率策略
- **长期（20日）**：降低交易频率，更注重趋势把握

### 标签计算公式

LazyBull 支持三个预设 horizon 的标签：

| 标签列 | Horizon | 计算公式 | 说明 |
|--------|---------|----------|------|
| `y_ret_5` | 5 个交易日 | `(close_adj(t+5) / close_adj(t)) - 1` | 未来 5 日后复权收益率 |
| `y_ret_10` | 10 个交易日 | `(close_adj(t+10) / close_adj(t)) - 1` | 未来 10 日后复权收益率 |
| `y_ret_20` | 20 个交易日 | `(close_adj(t+20) / close_adj(t)) - 1` | 未来 20 日后复权收益率 |

**公式说明**：
- `close_adj(t)`: 当前交易日的后复权收盘价
- `close_adj(t+N)`: N 个交易日后的后复权收盘价
- 收益率为小数形式（例如 0.05 表示 5% 收益）

**示例**：
```
假设某股票：
- 2023-01-03（t）：后复权收盘价 = 10.0 元
- 2023-01-10（t+5）：后复权收盘价 = 10.5 元
- 2023-01-17（t+10）：后复权收盘价 = 11.0 元
- 2023-02-01（t+20）：后复权收盘价 = 10.8 元

则标签为：
- y_ret_5 = (10.5 / 10.0) - 1 = 0.05 (5%)
- y_ret_10 = (11.0 / 10.0) - 1 = 0.10 (10%)
- y_ret_20 = (10.8 / 10.0) - 1 = 0.08 (8%)
```

### 标签特点

- **后复权**：使用后复权价格，消除分红送股的影响
- **相对收益**：计算相对收益率，而非绝对价格变化
- **交易日**：基于实际交易日计算，而非自然日
- **前视偏差**：标签使用未来数据，仅用于训练，回测时不可见

## 特征构建

### 基本用法

使用 `build_features.py` 脚本构建特征，默认同时生成三个 horizon 的标签：

```bash
python scripts/build_features.py \
  --start-date 20230101 \
  --end-date 20231231
```

这将生成包含 `y_ret_5`、`y_ret_10`、`y_ret_20` 三列标签的特征文件。

### 自定义 Horizon

如果只需要特定的 horizon，可以使用 `--horizons` 参数：

```bash
# 只生成 5 日和 10 日标签
python scripts/build_features.py \
  --start-date 20230101 \
  --end-date 20231231 \
  --horizons 5 10

# 只生成 20 日标签
python scripts/build_features.py \
  --start-date 20230101 \
  --end-date 20231231 \
  --horizons 20
```

### 特征文件结构

生成的特征文件位于 `data/features/cs_train/{YYYYMMDD}.parquet`，包含：

```python
import pandas as pd

# 加载特征文件
df = pd.read_parquet('data/features/cs_train/20230103.parquet')

# 查看列名
print(df.columns.tolist())
# ['ts_code', 'trade_date', 'name', 
#  'y_ret_5', 'y_ret_10', 'y_ret_20',  # 标签列
#  'ret_1', 'ret_5', 'ret_10', 'ret_20',  # 特征列
#  'vol_ratio_5', 'vol_ratio_10', 'vol_ratio_20',
#  ...]

# 查看标签分布
print(df[['y_ret_5', 'y_ret_10', 'y_ret_20']].describe())
```

### 注意事项

#### 1. 数据完整性
- 生成长 horizon 标签需要足够的未来数据
- 数据末尾的样本可能缺失长 horizon 标签
- 示例：如果数据到 2023-12-29，则 12-10 之后的样本缺失 `y_ret_20`

#### 2. 过滤逻辑
- 特征构建会过滤 ST 股票、新股（上市<60天）、停牌股票
- 标签过滤：要求**至少一个** horizon 的标签非空（而非全部非空）
- 好处：即使长 horizon 数据不足，短 horizon 的样本仍可保留

#### 3. 构建时间
- 同时生成 3 个 horizon 标签比单个标签慢约 10-15%
- 但显著快于分 3 次构建（避免重复计算特征）
- 建议：一次性构建所有需要的 horizon

## 模型训练

### 选择标签训练

使用 `train_ml_model.py` 脚本训练模型，通过 `--label` 参数选择标签：

```bash
# 训练 5 日预测模型（默认）
python scripts/train_ml_model.py \
  --start-date 20230101 \
  --end-date 20231231 \
  --label y_ret_5

# 训练 10 日预测模型
python scripts/train_ml_model.py \
  --start-date 20230101 \
  --end-date 20231231 \
  --label y_ret_10

# 训练 20 日预测模型
python scripts/train_ml_model.py \
  --start-date 20230101 \
  --end-date 20231231 \
  --label y_ret_20
```

### 模型元数据

训练完成后，模型元数据保存在 `data/models/model_registry.json`：

```json
{
  "models": [
    {
      "version": 1,
      "version_str": "v1",
      "label_column": "y_ret_5",
      "train_start_date": "20230101",
      "train_end_date": "20231231",
      ...
    },
    {
      "version": 2,
      "version_str": "v2",
      "label_column": "y_ret_10",
      "train_start_date": "20230101",
      "train_end_date": "20231231",
      ...
    }
  ]
}
```

**重要字段**：
- `label_column`：模型训练时使用的标签列
- `version`：模型版本号
- 回测时会读取此字段进行一致性校验

### 训练建议

#### 1. 样本数量
- 短 horizon（5日）：样本多，噪音大，需要更强的正则化
- 长 horizon（20日）：样本少（数据末尾缺失），但信号更稳定
- 建议根据验证集 IC/RankIC 选择最优 horizon

#### 2. 超参数
- 短 horizon：降低 `max_depth`（如 6），增加 `reg_alpha`（如 0.3）
- 长 horizon：可适当增加 `max_depth`（如 10），减少 `reg_alpha`（如 0.1）
- 建议针对不同 horizon 进行超参数调优

#### 3. 特征选择
- 所有 horizon 使用相同的特征集
- 如需优化，可针对不同 horizon 使用不同的 `lookback_windows`
- 示例：5 日模型使用 `[3, 5, 10]`，20 日模型使用 `[10, 20, 30]`

## 回测

### 基本用法

使用 `run_ml_backtest.py` 脚本运行回测：

```bash
# 使用最新模型，自动选择标签和调仓频率
python scripts/run_ml_backtest.py \
  --start-date 20240101 \
  --end-date 20241231

# 指定标签（会加载对应标签的最新模型）
python scripts/run_ml_backtest.py \
  --start-date 20240101 \
  --end-date 20241231 \
  --label y_ret_10
```

### 自动调仓频率

当未显式指定 `--rebalance-freq` 时，系统会根据标签自动设置调仓频率：

| 标签 | 自动调仓频率 | 说明 |
|------|--------------|------|
| `y_ret_5` | 5 个交易日 | 每 5 天调仓一次 |
| `y_ret_10` | 10 个交易日 | 每 10 天调仓一次 |
| `y_ret_20` | 20 个交易日 | 每 20 天调仓一次 |

**设计原理**：
- 调仓频率应与模型预测窗口对齐
- 避免过度交易（频率高于预测窗口）或信息浪费（频率低于预测窗口）
- 提供合理的默认值，减少配置错误

**示例输出**：
```
2026-02-11 10:00:00 | INFO | 未指定 --rebalance-freq 参数，根据标签 y_ret_10 自动设置为: 10
```

### 显式调仓频率

如需覆盖自动设置，可显式指定 `--rebalance-freq`：

```bash
# 使用 10 日模型，但每 5 日调仓（更激进）
python scripts/run_ml_backtest.py \
  --start-date 20240101 \
  --end-date 20241231 \
  --label y_ret_10 \
  --rebalance-freq 5

# 使用 10 日模型，但每 20 日调仓（更保守）
python scripts/run_ml_backtest.py \
  --start-date 20240101 \
  --end-date 20241231 \
  --label y_ret_10 \
  --rebalance-freq 20
```

**应用场景**：
- 研究调仓频率对策略表现的影响
- 降低交易成本（使用较低频率）
- 捕捉更多信号（使用较高频率）

### 模型-标签一致性校验

如果同时指定 `--model-version` 和 `--label`，系统会自动校验一致性：

```bash
# 场景 1：一致 - 正常运行
python scripts/run_ml_backtest.py \
  --start-date 20240101 \
  --end-date 20241231 \
  --model-version 2 \
  --label y_ret_10

# 场景 2：不一致 - 报错
python scripts/run_ml_backtest.py \
  --start-date 20240101 \
  --end-date 20241231 \
  --model-version 1 \
  --label y_ret_10
```

**错误提示**（不一致时）：
```
============================================================
参数错误：模型版本与标签不一致
============================================================
模型版本 v1 训练时使用的标签: y_ret_5
您指定的标签: y_ret_10

解决方案：
1. 移除 --label 参数，使用模型训练时的标签（y_ret_5）
2. 移除 --model-version 参数，自动加载使用 y_ret_10 训练的最新模型
3. 使用正确的 --model-version，该模型应使用 y_ret_10 标签训练
============================================================
```

### 回测结果对比

建议同时回测不同 horizon 的模型，对比表现：

```bash
# 回测 5 日模型
python scripts/run_ml_backtest.py \
  --start-date 20240101 \
  --end-date 20241231 \
  --label y_ret_5 \
  --output-name ml_backtest_5d

# 回测 10 日模型
python scripts/run_ml_backtest.py \
  --start-date 20240101 \
  --end-date 20241231 \
  --label y_ret_10 \
  --output-name ml_backtest_10d

# 回测 20 日模型
python scripts/run_ml_backtest.py \
  --start-date 20240101 \
  --end-date 20241231 \
  --label y_ret_20 \
  --output-name ml_backtest_20d
```

**对比指标**：
- 总收益率：长 horizon 通常更高（更少交易成本）
- 夏普比率：中等 horizon 可能最优（平衡收益和波动）
- 最大回撤：长 horizon 可能更大（更长持有期）
- 交易次数：短 horizon 更多，成本更高

## 最佳实践

### 1. 研究流程

**第一步：探索性分析**
```bash
# 构建特征（一次生成所有 horizon）
python scripts/build_features.py \
  --start-date 20200101 \
  --end-date 20231231 \
  --horizons 5 10 20

# 训练多个模型
for label in y_ret_5 y_ret_10 y_ret_20; do
  python scripts/train_ml_model.py \
    --start-date 20200101 \
    --end-date 20231231 \
    --label $label
done

# 回测对比
for label in y_ret_5 y_ret_10 y_ret_20; do
  python scripts/run_ml_backtest.py \
    --start-date 20240101 \
    --end-date 20241231 \
    --label $label \
    --output-name ml_backtest_${label}
done
```

**第二步：选择最优 horizon**
- 查看回测报告 `data/reports/ml_backtest_{label}/*.csv`
- 对比夏普比率、收益率、回撤等指标
- 考虑交易成本和执行难度
- 选择综合表现最好的 horizon

**第三步：生产部署**
- 仅部署最优 horizon 的模型
- 使用默认调仓频率（与 horizon 一致）
- 定期重新训练和回测，验证稳定性

### 2. 参数选择建议

#### Horizon 选择
- **高频交易者**：使用 `y_ret_5`，快速响应市场变化
- **中频交易者**：使用 `y_ret_10`，平衡收益和成本
- **低频交易者**：使用 `y_ret_20`，降低交易频率

#### 调仓频率建议
- **默认**：与 horizon 保持一致（5/10/20）
- **成本敏感**：使用略低于 horizon 的频率（如 horizon=10 时用 15）
- **信号充分利用**：使用略高于 horizon 的频率（如 horizon=10 时用 7）
- **避免极端**：不建议频率远离 horizon（如 horizon=20 但频率=3）

### 3. 常见问题

#### 问题 1：标签缺失过多
**原因**：数据末尾不足 N 个交易日
**解决**：
- 扩展数据获取范围（`--end-date` 后延 1-2 个月）
- 或使用较短的 horizon

#### 问题 2：模型表现不佳
**原因**：Horizon 与市场特征不匹配
**解决**：
- 尝试不同 horizon，找到最适合当前市场的周期
- 检查验证集 IC/RankIC，确认模型有预测能力
- 调整超参数或特征工程

#### 问题 3：回测与实盘不一致
**原因**：调仓频率设置不当
**解决**：
- 确保回测调仓频率与实盘一致
- 使用自动调仓频率（移除 `--rebalance-freq`）
- 模拟实盘约束（交易成本、滑点等）

## 高级主题

### 自定义 Horizon

如需使用非标准 horizon（如 15 日），需修改代码：

```python
# 在 build_features.py 中
builder = FeatureBuilder(horizons=[5, 10, 15, 20])

# 在 train_ml_model.py 中
parser.add_argument(
    "--label",
    choices=["y_ret_5", "y_ret_10", "y_ret_15", "y_ret_20"],
    ...
)
```

### 多模型集成

未来版本可能支持多 horizon 模型的集成学习：

```python
# 伪代码示例
ensemble = EnsembleModel([
    ('model_5d', model_5, weight=0.3),
    ('model_10d', model_10, weight=0.4),
    ('model_20d', model_20, weight=0.3),
])
predictions = ensemble.predict(features)
```

### 自适应 Horizon

未来版本可能支持根据市场状态动态切换 horizon：

```python
# 伪代码示例
if market_volatility > threshold:
    active_horizon = 5  # 高波动用短期
else:
    active_horizon = 20  # 低波动用长期
```

## 总结

本指南介绍了 LazyBull 多 horizon 标签功能的完整使用方法：

1. **标签定义**：3 个预设 horizon（5/10/20日），基于后复权收益率
2. **特征构建**：一次生成所有 horizon 标签，提高效率
3. **模型训练**：灵活选择标签，元数据自动记录
4. **回测评估**：自动调仓频率，严格一致性校验
5. **最佳实践**：探索-选择-部署的完整工作流

通过合理使用不同 horizon，可以显著提升策略研究的效率和质量。建议从对比不同 horizon 的回测表现开始，找到最适合您的投资风格和市场环境的预测周期。
