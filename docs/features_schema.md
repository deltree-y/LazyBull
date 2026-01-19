# 特征与标签定义

本文档说明日频截面特征构建的详细规范，包括特征定义、标签计算、数据schema等。

## 概述

- **特征类型**: 日频截面特征（每个交易日一个文件，包含当日全市场可交易股票）
- **标签类型**: 未来5个交易日的后复权收益率
- **存储格式**: Parquet
- **存储路径**: `data/features/cs_train/{YYYYMMDD}.parquet`

## 数据Schema

每个特征文件包含以下字段：

### 基础字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| trade_date | str | 交易日期，格式YYYYMMDD |
| ts_code | str | 股票代码，如'000001.SZ' |
| name | str | 股票名称 |

### 标签字段

| 字段名 | 类型 | 说明 | 计算方式 |
|--------|------|------|----------|
| y_ret_5 | float | 未来5日后复权收益率 | (close_adj(t+5) / close_adj(t)) - 1 |

**说明**：
- t+5 表示当前交易日之后的第5个交易日
- 使用后复权收盘价计算，消除分红送股的影响
- 标签缺失的样本（如未来停牌、退市）会被自动剔除

### 特征字段

#### 1. 收益率特征

| 字段名 | 类型 | 说明 | 计算方式 |
|--------|------|------|----------|
| ret_1 | float | 当日收益率 | (close(t) / close(t-1)) - 1 |
| ret_5 | float | 过去5日累计收益率 | (close_adj(t-1) / close_adj(t-6)) - 1 |
| ret_10 | float | 过去10日累计收益率 | (close_adj(t-1) / close_adj(t-11)) - 1 |
| ret_20 | float | 过去20日累计收益率 | (close_adj(t-1) / close_adj(t-21)) - 1 |

**注意**: 
- 特征只使用 <= t 的数据，避免未来信息泄露
- 回看窗口不包含当日（t-1往前看）

#### 2. 成交量特征

| 字段名 | 类型 | 说明 | 计算方式 |
|--------|------|------|----------|
| vol_ratio_5 | float | 当日成交量 / 过去5日平均成交量 | vol(t) / mean(vol(t-5:t-1)) |
| vol_ratio_10 | float | 当日成交量 / 过去10日平均成交量 | vol(t) / mean(vol(t-10:t-1)) |
| vol_ratio_20 | float | 当日成交量 / 过去20日平均成交量 | vol(t) / mean(vol(t-20:t-1)) |

#### 3. 成交额特征

| 字段名 | 类型 | 说明 | 计算方式 |
|--------|------|------|----------|
| amount_ratio_5 | float | 当日成交额 / 过去5日平均成交额 | amount(t) / mean(amount(t-5:t-1)) |
| amount_ratio_10 | float | 当日成交额 / 过去10日平均成交额 | amount(t) / mean(amount(t-10:t-1)) |
| amount_ratio_20 | float | 当日成交额 / 过去20日平均成交额 | amount(t) / mean(amount(t-20:t-1)) |

#### 4. 均线偏离特征

| 字段名 | 类型 | 说明 | 计算方式 |
|--------|------|------|----------|
| ma_deviation_5 | float | 收盘价偏离5日均线 | (close_adj(t) - MA5(t-5:t-1)) / MA5(t-5:t-1) |
| ma_deviation_10 | float | 收盘价偏离10日均线 | (close_adj(t) - MA10(t-10:t-1)) / MA10(t-10:t-1) |
| ma_deviation_20 | float | 收盘价偏离20日均线 | (close_adj(t) - MA20(t-20:t-1)) / MA20(t-20:t-1) |

### 过滤与标记字段

**注意**：从 v0.2.0 开始，filter 列定义已更新：
- `filter_list_days` 字段不再作为 filter 列，改为 `list_days` 字段
- `filter_` 前缀已从输出列中移除
- 列名已统一为与 clean 层一致的命名

| 字段名 | 类型 | 说明 | 用途 |
|--------|------|------|------|
| is_st | int | ST股票标记，1=是，0=否 | 过滤标记（已过滤样本，保留字段用于审计） |
| is_suspended | int | 停牌标记，1=停牌，0=正常 | 过滤标记（已过滤样本，保留字段用于审计） |
| list_days | int | 上市天数（自然日） | 信息字段，不作为过滤标记 |
| is_limit_up | int | 涨停标记，1=涨停，0=非涨停 | 标记字段，不过滤，但需注意流动性 |
| is_limit_down | int | 跌停标记，1=跌停，0=非跌停 | 标记字段，不过滤，但需注意流动性 |

**clean 层复用**：
当使用 clean daily 数据构建特征时，这些标记列会直接从 clean 层复用，确保数据一致性。

## 后复权计算方式

### 什么是后复权？

后复权是一种复权方式，保持最新价格不变，向前调整历史价格。适用于预测未来收益的场景。

### 计算公式

```
close_adj = close × adj_factor
```

其中：
- `close`: 原始收盘价
- `adj_factor`: 复权因子（TuShare提供）
- `close_adj`: 后复权收盘价

### 为什么使用后复权？

1. **一致性**: 保证当前时刻的价格就是真实可交易价格
2. **可比性**: 消除分红、送股等事件对收益率计算的影响
3. **实用性**: 适合用于预测未来收益，与实际投资场景一致

## 股票池过滤规则

### 必须剔除的股票

1. **ST股票** (`is_st=1`)
   - 判断标准：股票名称包含"ST"、"*ST"、"S*ST"、"退"等
   - 剔除原因：风险高、流动性差、涨跌停限制不同

2. **上市不满60天** (`list_days < 60`)
   - 判断标准：从上市日期到当前交易日的自然日天数 < 60
   - 剔除原因：历史数据不足、价格波动大、缺乏稳定性

3. **停牌股票** (`is_suspended=1`)
   - 判断标准：当日成交量为0，或在停复牌记录中
   - 剔除原因：无法交易

4. **标签缺失** (`y_ret_5 is NaN`)
   - 判断标准：未来5个交易日的收盘价缺失
   - 剔除原因：无法计算标签，无法用于训练

### 不剔除但标记的情况

1. **涨停** (`is_limit_up=1`)
   - 判断标准：
     - 非ST股票：涨幅 >= 9.9%
     - ST股票：涨幅 >= 4.9%
   - 不剔除原因：仍可持有，但需注意卖出困难

2. **跌停** (`is_limit_down=1`)
   - 判断标准：
     - 非ST股票：跌幅 <= -9.9%
     - ST股票：跌幅 <= -4.9%
   - 不剔除原因：仍可持有，但需注意买入困难

## 涨跌停判断规则

### 简化规则（当前实现）

基于当日涨跌幅 `pct_chg` 判断：

- **非ST股票**：
  - 涨停：`pct_chg >= 9.9%`
  - 跌停：`pct_chg <= -9.9%`

- **ST股票**：
  - 涨停：`pct_chg >= 4.9%`
  - 跌停：`pct_chg <= -4.9%`

**说明**：使用9.9%和4.9%作为阈值是考虑到浮点精度和一字板的情况。

### 精确规则（如有涨跌停价格数据）

如果通过 TuShare `stk_limit` 接口获取到涨跌停价格，则使用价格对比：

```python
limit_up = (close >= up_limit * 0.999)
limit_down = (close <= down_limit * 1.001)
```

## 特征缺失处理

### 历史数据不足

对于回看窗口（5、10、20日）特征，如果历史数据不足：

- 相应特征填充为 `NaN`
- 样本仍保留（不剔除）
- 模型训练时需处理缺失值（如使用XGBoost可自动处理）

### 极端值处理

特征可能包含极端值：

- **成交量/成交额比率**: 如果历史均值接近0，比率可能非常大或无穷
- **处理方式**: 保留原始值，由模型训练时的缺失值处理或异常值处理机制解决

## 数据使用示例

### 加载单日特征

```python
from src.lazybull.data import Storage

storage = Storage()
features = storage.load_cs_train_day('20230110')

print(f"样本数: {len(features)}")
print(f"特征列: {features.columns.tolist()}")
print(features.head())
```

### 加载多日特征并合并

```python
import pandas as pd
from src.lazybull.data import Storage

storage = Storage()
trading_dates = ['20230109', '20230110', '20230111']

all_features = []
for date in trading_dates:
    df = storage.load_cs_train_day(date)
    if df is not None:
        all_features.append(df)

features = pd.concat(all_features, ignore_index=True)
print(f"总样本数: {len(features)}")
```

### 训练XGBoost模型

```python
import xgboost as xgb

# 选择特征列
feature_cols = [
    'ret_1', 'ret_5', 'ret_10', 'ret_20',
    'vol_ratio_5', 'vol_ratio_10', 'vol_ratio_20',
    'amount_ratio_5', 'amount_ratio_10', 'amount_ratio_20',
    'ma_deviation_5', 'ma_deviation_10', 'ma_deviation_20'
]

X = features[feature_cols]
y = features['y_ret_5']

# 训练模型
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1
)

model.fit(X, y)
```

## 注意事项

### 1. 数据时效性

- 复权因子可能在分红送股后更新，需确保使用最新数据
- 建议定期重新拉取数据并重新生成特征

### 2. 未来函数陷阱

- 所有特征必须只使用 <= t 的数据
- 标签使用未来数据是正常的（预测目标）
- 回测时需注意信号生成的时点

### 3. 幸存者偏差

- 当前实现使用的是某时刻仍上市的股票列表
- 未考虑已退市股票，存在幸存者偏差
- 实际使用时需注意这一局限性

### 4. 涨跌停流动性

- 涨停股票难以买入
- 跌停股票难以卖出
- 回测时需要模拟流动性限制

## 扩展方向

### 短期扩展

1. 添加更多技术指标特征（RSI、MACD、布林带等）
2. 添加基本面特征（PE、PB、ROE等）
3. 支持不同的预测时间窗口（horizon=1, 3, 10等）
4. 添加行业、市场分类特征

### 长期扩展

1. 支持分钟级高频特征
2. 添加另类数据特征（舆情、资金流向等）
3. 支持因子标准化、去极值处理
4. 支持在线特征更新（实盘场景）

## 参考资料

- [TuShare Pro API文档](https://tushare.pro/document/2)
- [XGBoost官方文档](https://xgboost.readthedocs.io/)
- 项目README: `README.md`
- 数据契约: `docs/data_contract.md`
