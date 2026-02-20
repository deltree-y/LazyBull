# rank-weight 使用指南

本文档说明如何启用、配置和验证 Top/Bottom K 样本权重增强（rank-weight）功能。

## 什么是 rank-weight？

在训练 XGBoost 选股模型时，**预测精度最重要的位置是每日截面的头部（Top30）和尾部（Bottom30）**。  
这些样本直接决定买入和规避的标的。

rank-weight 的核心思路：对每个交易日，按标签值排序，将 Top K 和 Bottom K 样本的训练权重  
提升为 `top_weight`（默认 5.0），其余样本权重保持 1.0。

## CLI 参数说明

```bash
python scripts/train_ml_model.py \
    --start-date 20230101 --end-date 20231231 \
    [--rank-weight-enabled]        # 默认已开启，可省略
    [--no-rank-weight]             # 禁用 rank-weight
    [--rank-weight-topk 30]        # 每日 Top/Bottom K 数量，默认 30
    [--rank-weight-weight 5.0]     # Top/Bottom K 样本权重，默认 5.0
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--rank-weight-enabled` | flag | `True` | 启用（默认开启，无需显式指定） |
| `--no-rank-weight` | flag | — | 禁用，用于对比实验 |
| `--rank-weight-topk` | int | 30 | 每日 Top/Bottom K 样本数 |
| `--rank-weight-weight` | float | 5.0 | Top/Bottom K 样本的权重倍数 |

## 使用示例

### 默认设置（推荐）

```bash
python scripts/train_ml_model.py \
    --start-date 20230101 --end-date 20231231
```

日志中可以看到：

```
rank-weight: 已启用（topk=30, weight=5.0）
样本权重构造完成: Top/Bottom 30 增强，加权样本数=18000，权重=5.0，普通样本数=...
```

### 自定义参数

```bash
# 增加 topk 范围、降低权重倍数
python scripts/train_ml_model.py \
    --start-date 20230101 --end-date 20231231 \
    --rank-weight-topk 50 \
    --rank-weight-weight 3.0
```

### 关闭 rank-weight（对比基线）

```bash
python scripts/train_ml_model.py \
    --start-date 20230101 --end-date 20231231 \
    --no-rank-weight
```

## 参数对比实验建议

| 实验名 | 参数 | 说明 |
|---|---|---|
| 基线 | `--no-rank-weight` | 等权训练 |
| 默认 rank-weight | 默认 | topk=30, weight=5.0 |
| 激进 | `--rank-weight-topk 30 --rank-weight-weight 10.0` | 更强的头部强化 |
| 宽松 | `--rank-weight-topk 50 --rank-weight-weight 2.0` | 覆盖更多样本但权重更低 |

通过比较 `ml_train_runs.csv` 中的 `daily_rankic_mean` 和 `top30_return_mean` 来判断哪组参数效果更好。

## 回溯与审计

每次训练运行会将 rank-weight 配置记录到：

1. **训练日志**：`[INFO] rank-weight: 已启用（topk=30, weight=5.0）`
2. **`ml_train_runs.csv`**：新增列 `rank_weight_enabled`、`rank_weight_topk`、`rank_weight_weight`

## 边界处理说明

| 情形 | 处理方式 |
|---|---|
| 某日样本数 `n > 2*topk` | 正常：Top K 和 Bottom K 各赋 top_weight |
| 某日样本数 `n <= 2*topk` | 退化：整日全部样本赋 top_weight |
| 标签列含 NaN | NaN 样本不参与排名，也不被赋予 top_weight |
| 标签列不存在 | 返回全 1 权重（等效禁用），打印警告日志 |

## 验证 rank-weight 是否生效

### 方法一：查看日志

训练时日志会打印：
```
rank-weight: 已启用（topk=30, weight=5.0）
样本权重构造完成: Top/Bottom 30 增强，加权样本数=18000，权重=5.0
使用样本权重（rank-weight），加权样本数=18000
```

### 方法二：对比 ml_train_runs.csv

```python
import pandas as pd
df = pd.read_csv('data/models/ml_train_runs.csv')
print(df[['model_version', 'rank_weight_enabled', 'rank_weight_topk',
          'rank_weight_weight', 'daily_rankic_mean']].tail(5))
```

### 方法三：单元测试验证

```bash
python -m pytest tests/test_rank_sample_weight.py -v
```
