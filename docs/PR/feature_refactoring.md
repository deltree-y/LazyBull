# 特征工程重构 - factors 模块与新增特征

**版本**: v0.5.0  
**日期**: 2026-02-18  
**类型**: 重大更新（Breaking Changes）

---

## 概述

本次PR对特征工程进行了系统性重构，将特征计算逻辑从 `FeatureBuilder` 拆分到独立的 `factors` 模块，实现了可复用的因子库。同时删除了冗余特征，新增了多个高质量的技术指标和行业特征。

### 核心目标

1. **模块化重构**: 将特征计算拆分到 `src/lazybull/factors` 模块，提升可维护性和可扩展性
2. **精简特征**: 删除 `amount_ratio_*` 和 `vol_ma*` 等冗余特征
3. **增强特征**: 新增技术指标（RSI/KDJ/MACD/布林带）、K线形态、波动率、行业 alpha 等特征
4. **行业支持**: 基于 stock_basic 的 industry 字段实现行业分类与 alpha 计算

---

## 为什么要重构到 factors 模块？

### 原有架构的问题

- 所有特征计算逻辑集中在 `FeatureBuilder` 类中，代码超过 1000 行，难以维护
- 添加新特征需要修改 `FeatureBuilder`，容易引入 bug
- 特征计算逻辑无法复用，无法在其他场景（如因子分析、特征选择）中独立使用

### 新架构的优势

- **模块化**: 每类因子独立模块，职责清晰
- **可复用**: 因子计算函数可在任何场景使用，接受 DataFrame 输入返回 DataFrame
- **易扩展**: 添加新因子只需在对应模块中新增函数，无需修改 `FeatureBuilder`
- **易测试**: 每个因子函数可独立单元测试

---

## 新增特征列表

### 1. 技术指标 (technical_indicators.py)

| 特征名 | 说明 | 参数 | 公式/方法 |
|--------|------|------|----------|
| rsi_14 | 相对强弱指标 | 窗口=14 | RSI = 100 - 100/(1+RS), RS = 平均涨幅/平均跌幅 |
| kdj_k | KDJ指标K值 | n=9, m1=3, m2=3 | 基于 RSV 的指数移动平均 |
| kdj_d | KDJ指标D值 | 同上 | K 的指数移动平均 |
| kdj_j | KDJ指标J值 | 同上 | J = 3*K - 2*D |
| macd_dif | MACD的DIF线 | fast=12, slow=26, signal=9 | DIF = EMA(12) - EMA(26) |
| macd_dea | MACD的DEA线 | 同上 | DEA = EMA(DIF, 9) |
| macd_hist | MACD柱 | 同上 | HIST = (DIF - DEA) * 2 |
| bb_middle | 布林带中轨 | 窗口=20, std=2 | MA(20) |
| bb_upper | 布林带上轨 | 同上 | MA(20) + 2*STD(20) |
| bb_lower | 布林带下轨 | 同上 | MA(20) - 2*STD(20) |
| bb_width | 布林带宽度 | 同上 | (上轨 - 下轨) / 中轨 |
| bb_pct | 布林带%B | 同上 | (价格 - 下轨) / (上轨 - 下轨) |

### 2. K线形态 (candlestick.py)

| 特征名 | 说明 | 公式 |
|--------|------|------|
| amplitude | 振幅 | (high_adj - low_adj) / pre_close_adj |
| upper_shadow | 上影线比例 | (high_adj - max(open_adj, close_adj)) / close_adj |
| lower_shadow | 下影线比例 | (min(open_adj, close_adj) - low_adj) / close_adj |
| body_length | 实体长度比例 | abs(close_adj - open_adj) / close_adj |

### 3. 波动率 (volatility.py)

| 特征名 | 说明 | 公式 |
|--------|------|------|
| volatility_5 | 5日波动率 | ret_1 的5日滚动标准差 |
| volatility_10 | 10日波动率 | ret_1 的10日滚动标准差 |
| volatility_20 | 20日波动率 | ret_1 的20日滚动标准差 |

### 4. 行业特征 (industry.py)

| 特征名 | 说明 | 公式 |
|--------|------|------|
| industry | 行业名称（字符串） | 从 stock_basic 获取 |
| industry_id | 行业编码（整数） | 稳定映射，同一 industry 映射到同一 id |
| alpha_industry | 当日行业 alpha | ret_1 - 行业平均 ret_1 |
| alpha_industry_5 | 5日行业 alpha | ret_5 - 行业平均 ret_5 |
| alpha_industry_10 | 10日行业 alpha | ret_10 - 行业平均 ret_10 |
| alpha_industry_20 | 20日行业 alpha | ret_20 - 行业平均 ret_20 |

### 5. 动量加速度 (momentum.py)

| 特征名 | 说明 | 公式 |
|--------|------|------|
| acceleration | 动量加速度 | ret_5 - ret_10 (短期相对中期的加速) |

### 6. 量能突变 (volume.py)

| 特征名 | 说明 | 公式 |
|--------|------|------|
| vol_burst_5 | 5日量能突变 | vol_ratio_5 的截面 zscore |
| vol_burst_10 | 10日量能突变 | vol_ratio_10 的截面 zscore |
| vol_burst_20 | 20日量能突变 | vol_ratio_20 的截面 zscore |

---

## 删除的特征列表

### 已删除

- **amount_ratio_5**: 当日成交额 / 5日平均成交额
- **amount_ratio_10**: 当日成交额 / 10日平均成交额
- **amount_ratio_20**: 当日成交额 / 20日平均成交额
- **vol_ma5**: 5日平均成交量
- **vol_ma10**: 10日平均成交量
- **vol_ma20**: 20日平均成交量

### 保留（未删除）

- **amount_ma5/10/20**: 过去N日平均成交额（保留）
- **vol_ratio_5/10/20**: 当日成交量 / 过去N日平均成交量（保留）

---

## 实现细节

### factors 模块结构

```
src/lazybull/factors/
├── __init__.py              # 导出所有因子函数
├── technical_indicators.py  # RSI, KDJ, MACD, 布林带
├── candlestick.py           # 振幅, 上下影线
├── volatility.py            # 波动率
├── industry.py              # 行业 alpha, industry_id
├── momentum.py              # 加速度
└── volume.py                # 量能突变
```

### FeatureBuilder 集成

在 `FeatureBuilder.build_features_for_day()` 流程中新增步骤：

```python
# 5.5 添加新增因子（技术指标、K线形态、波动率、行业等）
features = self._add_advanced_factors(
    features,
    current_data,
    daily_adj,
    trade_date,
    trading_dates,
    current_idx,
    stock_basic
)
```

`_add_advanced_factors()` 方法按顺序调用各因子模块的函数。

### 行业数据处理

- 从 `stock_basic` 的 `industry` 字段获取行业信息
- 如果 `stock_basic` 缺少 `industry` 字段，抛出 `ValueError` 并给出中文提示
- 生成稳定的 `industry_id` 编码（字母序排序后编号）
- 按 `trade_date` + `industry` 分组计算行业平均收益

### 复权价格计算

扩展 `_calculate_adj_close()` 方法，在计算 `close_adj` 的同时计算：
- `open_adj = open * adj_factor`
- `high_adj = high * adj_factor`
- `low_adj = low * adj_factor`

用于支持 K线形态、技术指标等需要 OHLC 的因子。

---

## 如何重建 features 数据与重训模型

由于特征列发生变化，旧的特征数据和模型**不兼容**新版本，需要重新生成。

### 步骤1: 确保基础数据包含 industry

检查 stock_basic 数据：
```python
from src.lazybull.data import Storage
storage = Storage()
stock_basic = storage.load_stock_basic()
print('industry' in stock_basic.columns)  # 应该为 True
```

如果 `industry` 列不存在，重新拉取 stock_basic：
```bash
python scripts/pull_data.py --table stock_basic
```

### 步骤2: 重新生成 features

删除旧的 features 数据（可选，也可以覆盖）：
```bash
rm -rf data/features/cs_train/*
```

重新生成 features（根据训练日期范围）：
```bash
python scripts/ensure_features.py --start-date 20200101 --end-date 20231231
```

### 步骤3: 重新训练模型

使用新的特征数据训练模型：
```bash
python scripts/train_ml_model.py \
  --task regression \
  --label cs_zscore \
  --train-start 20200101 \
  --train-end 20221231 \
  --val-start 20230101 \
  --val-end 20231231
```

### 步骤4: 验证新特征

检查生成的特征文件是否包含新字段：
```python
from src.lazybull.data import Storage
storage = Storage()
df = storage.load_cs_train_day('20231201')

# 验证删除的特征不存在
assert 'amount_ratio_5' not in df.columns
assert 'vol_ma5' not in df.columns

# 验证新增的特征存在
assert 'amplitude' in df.columns
assert 'rsi_14' in df.columns
assert 'industry_id' in df.columns
```

---

## 注意事项

### 1. 历史数据依赖

部分技术指标（如 MACD、布林带、RSI）需要足够的历史数据才能计算：
- RSI(14): 至少需要14天历史
- MACD(12,26,9): 至少需要26天历史
- 布林带(20): 至少需要20天历史

当历史数据不足时，对应特征值为 `NaN`。

### 2. 行业数据必需

本版本要求 stock_basic 必须包含 `industry` 字段，否则无法生成特征。确保使用最新的数据拉取脚本。

### 3. 特征列数量增加

新版本特征列数量显著增加（新增约20+列），可能影响：
- 特征文件大小（约增加20-30%）
- 模型训练时间（可通过特征选择缓解）

### 4. 模型不兼容

旧模型期望的特征列与新版本不同，直接使用会报错。必须重新训练。

---

## 测试覆盖

新增 `tests/test_new_features.py`，包含12个测试用例：

1. **test_amount_ratio_removed**: 验证 amount_ratio_* 已删除
2. **test_vol_ma_removed**: 验证 vol_ma* 已删除
3. **test_amount_ma_preserved**: 验证 amount_ma* 保留
4. **test_amplitude_feature_exists**: 验证振幅特征存在
5. **test_shadow_features_exist**: 验证上下影线特征存在
6. **test_volatility_features_exist**: 验证波动率特征存在
7. **test_industry_features_exist**: 验证行业特征存在
8. **test_industry_id_encoding_stable**: 验证 industry_id 编码稳定
9. **test_missing_industry_raises_error**: 验证缺失行业时报错
10. **test_acceleration_feature_exists**: 验证加速度特征存在
11. **test_volume_burst_features_exist**: 验证量能突变特征存在
12. **test_technical_indicators_exist**: 验证技术指标特征存在

运行测试：
```bash
python -m pytest tests/test_new_features.py -v
```

所有测试通过。

---

## 后续扩展建议

本次重构为后续特征扩展奠定了基础。建议后续可以添加：

1. **更多技术指标**: ATR, CCI, Williams %R 等
2. **高级形态识别**: 锤子线, 十字星, 吞没形态等
3. **量价关系**: OBV, MFI 等
4. **市场宽度**: 涨跌家数比, 涨停数量等
5. **因子组合**: 多因子复合特征

所有新因子只需在 `factors/` 模块中添加函数，并在 `_add_advanced_factors()` 中调用即可。

---

## 相关文档

- [因子扩展开发指南](../guide/factor_extension.md)
- [特征与标签定义](../features_schema.md)
- [CHANGELOG](../../CHANGELOG.md)
