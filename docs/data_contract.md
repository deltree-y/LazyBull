# 数据契约文档

本文档定义LazyBull项目中各数据层的字段规范与主键约定。

## 数据分层

```
data/
├── raw/        # 原始数据层（从数据源直接获取）
│   ├── {name}.parquet              # 非分区数据（旧格式，向后兼容）
│   └── {name}/                     # 按日分区数据（新格式）
│       ├── YYYY-MM-DD.parquet      # 每日数据文件
│       └── ...
├── clean/      # 清洗数据层（标准化、去重、填充）
│   ├── {name}.parquet              # 非分区数据（旧格式，向后兼容）
│   └── {name}/                     # 按日分区数据（新格式）
│       ├── YYYY-MM-DD.parquet      # 每日数据文件
│       └── ...
├── features/   # 特征数据层（因子计算结果）
└── reports/    # 报告数据层（回测结果）
```

## 数据存储策略

### 按日分区存储

从v0.2.0开始，raw和clean层支持按交易日分区存储：

- **目录结构**: `data/{layer}/{name}/{YYYY-MM-DD}.parquet`
- **优势**:
  - 减少单文件大小，提高读写效率
  - 支持增量更新，避免重复拉取全量数据
  - 便于按日期范围查询和清理历史数据
- **向后兼容**: 系统自动兼容旧的非分区数据格式

### 使用示例

```python
from src.lazybull.data import Storage

storage = Storage(enable_partitioning=True)

# 保存按日分区的原始数据
storage.save_raw_by_date(daily_df, "daily", "20230101")

# 加载单日数据
df = storage.load_raw_by_date("daily", "20230101")

# 加载日期范围数据
df = storage.load_raw_by_date_range("daily", "20230101", "20230131")

# 列出所有分区日期
dates = storage.list_partitions("raw", "daily")
```

## Raw 层数据契约

### 1. trade_cal - 交易日历

**来源**: TuShare `trade_cal`  
**主键**: `(exchange, cal_date)`  
**字段说明**:

| 字段名 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| exchange | str | 交易所代码 | SSE/SZSE |
| cal_date | str/datetime | 日历日期 | 20230101 |
| is_open | int | 是否交易 | 0=休市 1=交易 |
| pretrade_date | str/datetime | 上一交易日 | 20221230 |

### 2. stock_basic - 股票基本信息

**来源**: TuShare `stock_basic`  
**主键**: `ts_code`  
**字段说明**:

| 字段名 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| ts_code | str | 股票代码 | 000001.SZ |
| symbol | str | 股票代码（不含交易所） | 000001 |
| name | str | 股票名称 | 平安银行 |
| area | str | 地域 | 深圳 |
| industry | str | 行业 | 银行 |
| market | str | 市场类型 | 主板/创业板/科创板 |
| list_date | str/datetime | 上市日期 | 19910403 |

### 3. daily - 日线行情

**来源**: TuShare `daily`  
**主键**: `(ts_code, trade_date)`  
**字段说明**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| ts_code | str | 股票代码 |
| trade_date | str/datetime | 交易日期 |
| open | float | 开盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| close | float | 收盘价 |
| pre_close | float | 昨收价 |
| change | float | 涨跌额 |
| pct_chg | float | 涨跌幅(%) |
| vol | float | 成交量(手) |
| amount | float | 成交额(千元) |

### 4. daily_basic - 每日指标

**来源**: TuShare `daily_basic`  
**主键**: `(ts_code, trade_date)`  
**字段说明**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| ts_code | str | 股票代码 |
| trade_date | str/datetime | 交易日期 |
| close | float | 收盘价 |
| turnover_rate | float | 换手率(%) |
| turnover_rate_f | float | 换手率(自由流通股) |
| volume_ratio | float | 量比 |
| pe | float | 市盈率 |
| pe_ttm | float | 市盈率TTM |
| pb | float | 市净率 |
| ps | float | 市销率 |
| ps_ttm | float | 市销率TTM |
| dv_ratio | float | 股息率(%) |
| dv_ttm | float | 股息率TTM(%) |
| total_share | float | 总股本(万股) |
| float_share | float | 流通股本(万股) |
| free_share | float | 自由流通股本(万股) |
| total_mv | float | 总市值(万元) |
| circ_mv | float | 流通市值(万元) |

### 5. suspend_d - 停复牌信息

**来源**: TuShare `suspend_d`  
**主键**: `(ts_code, trade_date)`  
**API变更**: 从v0.2.0开始更新为新版API参数

**字段说明**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| ts_code | str | 股票代码 |
| trade_date | str/datetime | 停复牌日期 |
| suspend_type | str | 停复牌类型：S=停牌，R=复牌 |
| suspend_timing | str | 盘中停复牌时段（可选） |

**API参数变更说明**:

- **旧版API（已弃用）**: `suspend_date`, `resume_date`
- **新版API**: `trade_date`, `start_date`, `end_date`, `suspend_type`

```python
# 新版调用示例
client = TushareClient()

# 获取某日所有停牌股票
suspend_df = client.get_suspend_d(trade_date='20230315', suspend_type='S')

# 获取某段时间某只股票的停复牌记录
suspend_df = client.get_suspend_d(
    ts_code='000001.SZ',
    start_date='20230101',
    end_date='20230331'
)
```

## Clean 层数据契约

Clean层数据基于Raw层处理：
- 日期统一转换为datetime类型
- 缺失值按规则填充或删除
- 异常值处理（涨跌停、停牌标记）
- 数据类型标准化

**主键保持与Raw层一致**

## Features 层数据契约

### 因子数据标准格式

**主键**: `(ts_code, trade_date)`  
**基础字段**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| ts_code | str | 股票代码 |
| trade_date | datetime | 交易日期 |
| factor_name | float | 因子值 |

**因子命名规范**:
- 红利因子: `dividend_*`（如 dividend_yield）
- 价值因子: `value_*`（如 value_pb, value_pe）
- 技术因子: `tech_*`（如 tech_ma20, tech_rsi）
- 质量因子: `quality_*`（如 quality_roe, quality_debt_ratio）

## Reports 层数据契约

### 1. 净值曲线

**文件名**: `{backtest_name}_nav.csv`  
**主键**: `date`  
**字段**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| date | datetime | 日期 |
| portfolio_value | float | 组合总市值 |
| capital | float | 现金 |
| market_value | float | 持仓市值 |
| nav | float | 净值 |
| return | float | 累计收益率 |

### 2. 交易记录

**文件名**: `{backtest_name}_trades.csv`  
**主键**: `(date, stock, action)`  
**字段**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| date | datetime | 交易日期 |
| stock | str | 股票代码 |
| action | str | 交易方向(buy/sell) |
| price | float | 成交价格 |
| shares | int | 成交数量 |
| amount | float | 成交金额 |
| cost | float | 交易成本 |

## 数据质量要求

1. **完整性**: 主键字段不允许为空
2. **唯一性**: 主键组合保证唯一
3. **一致性**: 日期格式统一，代码格式统一
4. **准确性**: 数值范围合理（价格>0，涨跌幅在合理范围）
5. **时效性**: 数据更新及时，避免使用过期数据

## 版本历史

- v0.1.0 (2024-01): 初始版本，定义基础数据契约
