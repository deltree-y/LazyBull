# 成交量过滤功能说明

## 概述

成交量过滤功能用于在选股过程中剔除成交量较低的股票，提高持仓的流动性和可交易性。

## 功能说明

### 过滤逻辑

在特征构建阶段，系统会：
1. 计算当日所有候选股票的成交量分位数
2. 剔除成交量排名后 N% 的股票（N 可配置）
3. 成交量缺失或为 0 的股票已在停牌过滤中处理

### 配置参数

在 `configs/base.yaml` 中配置：

```yaml
backtest:
  # 成交量过滤配置
  volume_filter_enabled: true  # 是否启用成交量过滤，默认true
  volume_filter_pct: 20  # 过滤成交量后N%的股票，默认20%
```

### 参数说明

- **volume_filter_enabled**: 布尔值，是否启用成交量过滤
  - `true`: 启用过滤
  - `false`: 禁用过滤

- **volume_filter_pct**: 浮点数（0-100），过滤阈值百分比
  - 默认值：20（过滤成交量后20%的股票）
  - 取值范围：0-100
  - 0：不过滤任何股票
  - 100：过滤所有股票（慎用）

## 使用示例

### 示例1：使用默认配置（过滤后20%）

```python
from src.lazybull.features import FeatureBuilder

builder = FeatureBuilder(
    min_list_days=60,
    horizon=5,
    volume_filter_pct=20,  # 默认值
    volume_filter_enabled=True  # 默认启用
)
```

### 示例2：自定义过滤比例

```python
# 过滤后30%的股票
builder = FeatureBuilder(
    min_list_days=60,
    horizon=5,
    volume_filter_pct=30,
    volume_filter_enabled=True
)
```

### 示例3：禁用成交量过滤

```python
builder = FeatureBuilder(
    min_list_days=60,
    horizon=5,
    volume_filter_enabled=False
)
```

### 示例4：不过滤（设置为0%）

```python
builder = FeatureBuilder(
    min_list_days=60,
    horizon=5,
    volume_filter_pct=0,
    volume_filter_enabled=True
)
```

## 边界情况处理

### 1. 成交量缺失

成交量为 NaN 或缺失的股票会被视为成交量为0，在停牌过滤阶段已被过滤。

### 2. 成交量为0

成交量为0的股票被视为停牌，在停牌过滤阶段已被过滤，不会参与成交量分位数计算。

### 3. 候选股票数量不足

当候选股票数量较少时，成交量过滤可能导致最终可选股票过少。建议根据实际情况调整 `volume_filter_pct` 参数。

### 4. 过滤比例为100%

设置为100%会过滤掉所有股票，导致无可选标的。系统会记录警告日志。

## 日志输出

过滤过程会输出详细日志：

```
过滤前样本数: 4500, ST: 120, 上市<60天: 230, 停牌: 45, 标签缺失: 180, 成交量过滤: 816
过滤后样本数: 3109
```

## 注意事项

1. **流动性考虑**：成交量过滤可以提高持仓流动性，但过度过滤可能错失优质小盘股
2. **参数调优**：建议根据回测结果和实际交易情况调整过滤比例
3. **与其他过滤的关系**：成交量过滤在 ST、停牌、上市天数过滤之后执行
4. **数据质量**：成交量数据来自日线行情，需要确保数据质量

## 性能影响

成交量过滤的计算开销很小，主要是：
- 计算分位数：O(n log n)
- 过滤操作：O(n)

对整体特征构建性能影响可忽略不计。
