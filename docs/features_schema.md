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
| amount_ma5 | float | 过去5日平均成交额 | mean(amount(t-5:t-1)) |
| amount_ma10 | float | 过去10日平均成交额 | mean(amount(t-10:t-1)) |
| amount_ma20 | float | 过去20日平均成交额 | mean(amount(t-20:t-1)) |

**注意**: v0.9.0 删除了 `amount_ratio_5/10/20` 特征，保留 `amount_ma*` 特征。

#### 4. 均线偏离特征

| 字段名 | 类型 | 说明 | 计算方式 |
|--------|------|------|----------|
| ma_deviation_5 | float | 收盘价偏离5日均线 | (close_adj(t) - MA5(t-5:t-1)) / MA5(t-5:t-1) |
| ma_deviation_10 | float | 收盘价偏离10日均线 | (close_adj(t) - MA10(t-10:t-1)) / MA10(t-10:t-1) |
| ma_deviation_20 | float | 收盘价偏离20日均线 | (close_adj(t) - MA20(t-20:t-1)) / MA20(t-20:t-1) |

#### 5. K线形态特征 (v0.9.0新增)

| 字段名 | 类型 | 说明 | 计算方式 |
|--------|------|------|----------|
| amplitude | float | 振幅 | (high_adj - low_adj) / pre_close_adj |
| upper_shadow | float | 上影线比例 | (high_adj - max(open_adj, close_adj)) / close_adj |
| lower_shadow | float | 下影线比例 | (min(open_adj, close_adj) - low_adj) / close_adj |
| body_length | float | K线实体长度比例 | abs(close_adj - open_adj) / close_adj |

#### 6. 波动率特征 (v0.9.0新增)

| 字段名 | 类型 | 说明 | 计算方式 |
|--------|------|------|----------|
| volatility_5 | float | 5日波动率 | std(ret_1(t-5:t-1)) |
| volatility_10 | float | 10日波动率 | std(ret_1(t-10:t-1)) |
| volatility_20 | float | 20日波动率 | std(ret_1(t-20:t-1)) |

#### 7. 行业特征 (v0.9.0新增)

| 字段名 | 类型 | 说明 | 计算方式 |
|--------|------|------|----------|
| industry | str | 行业名称 | 从 stock_basic 获取 |
| industry_id | int | 行业编码 | 稳定的整数映射 |
| alpha_industry | float | 当日行业alpha | ret_1(t) - mean_by_industry(ret_1(t)) |
| alpha_industry_5 | float | 5日行业alpha | ret_5(t) - mean_by_industry(ret_5(t)) |
| alpha_industry_10 | float | 10日行业alpha | ret_10(t) - mean_by_industry(ret_10(t)) |
| alpha_industry_20 | float | 20日行业alpha | ret_20(t) - mean_by_industry(ret_20(t)) |

**申万行业分类字段（v0.55.2 起支持项目级主口径配置，默认二级）**：

| 字段名 | 类型 | 说明 | 数据来源 |
|--------|------|------|----------|
| sw_industry | str | 申万主行业名称（由 `industry.shenwan_level` 决定，默认二级） | TuShare index_classify（L3 数据源）+ index_member |
| sw_industry_code | str | 申万主行业指数代码（由 `industry.shenwan_level` 决定） | TuShare index_classify（L3 数据源） |
| sw_industry_id | int | 申万主行业整数编码（稳定映射） | 基于 sw_industry 排序生成 |

**字段命名变更历史**：

| 版本 | 旧字段名 | 新字段名 | 行业层级 |
|---|---|---|---|
| v0.10.0-v0.11.0 | sw_code / sw_name / industry_id | — | 一级（~30个） |
| v0.12.0-v0.55.1 | — | sw_industry_code / sw_industry / sw_industry_id | 固定二级（~100个） |
| v0.55.2+ | — | sw_industry_code / sw_industry / sw_industry_id | 项目配置决定（默认二级） |

**说明**：
- v0.55.2 起通过 `configs/base.yaml` 的 `industry.shenwan_level` 统一指定主行业层级，支持 `l1` / `l2` / `l3`，默认 `l2`
- v0.11.0 及更早版本的旧字段（`sw_code`/`sw_name`/`industry_id`）不再出现在 FeatureBuilder 输出中
- 申万行业数据通过 `scripts/update_basic_data.py --only-shenwan` 更新，当前主流程使用 L3 数据源并在特征阶段映射到统一主口径
- sw_industry_id 编码稳定：同一 sw_industry 名称始终映射到相同的整数ID
- 中性化分组字段：`sw_industry`（原为 `sw_name`）

#### 8. 动量加速度特征 (v0.9.0新增)

| 字段名 | 类型 | 说明 | 计算方式 |
|--------|------|------|----------|
| acceleration | float | 动量加速度 | ret_5 - ret_10 |

#### 9. 量能突变特征 (v0.9.0新增)

| 字段名 | 类型 | 说明 | 计算方式 |
|--------|------|------|----------|
| vol_burst_5 | float | 5日量能突变 | zscore_cross_section(vol_ratio_5) |
| vol_burst_10 | float | 10日量能突变 | zscore_cross_section(vol_ratio_10) |
| vol_burst_20 | float | 20日量能突变 | zscore_cross_section(vol_ratio_20) |

#### 10. 技术指标 (v0.9.0新增)

**RSI（相对强弱指标）**:

| 字段名 | 类型 | 说明 | 参数 |
|--------|------|------|------|
| rsi_14 | float | RSI(14) | 窗口=14 |

**KDJ（随机指标）**:

| 字段名 | 类型 | 说明 | 参数 |
|--------|------|------|------|
| kdj_k | float | KDJ的K值 | n=9, m1=3, m2=3 |
| kdj_d | float | KDJ的D值 | n=9, m1=3, m2=3 |
| kdj_j | float | KDJ的J值 | n=9, m1=3, m2=3 |

**MACD（指数平滑移动平均线）**:

| 字段名 | 类型 | 说明 | 参数 |
|--------|------|------|------|
| macd_dif | float | MACD的DIF线 | fast=12, slow=26, signal=9 |
| macd_dea | float | MACD的DEA线 | fast=12, slow=26, signal=9 |
| macd_hist | float | MACD柱 | fast=12, slow=26, signal=9 |

**布林带（Bollinger Bands）**:

| 字段名 | 类型 | 说明 | 参数 |
|--------|------|------|------|
| bb_middle | float | 布林带中轨 | 窗口=20, std=2 |
| bb_upper | float | 布林带上轨 | 窗口=20, std=2 |
| bb_lower | float | 布林带下轨 | 窗口=20, std=2 |
| bb_width | float | 布林带宽度 | 窗口=20, std=2 |
| bb_pct | float | 布林带%B | 窗口=20, std=2 |

#### 11. 行业中性化字段 (v0.11.0新增)

LazyBull 支持两类行业中性化，分别适用于不同类型的数据：

**类型1：行业去均值（Demean）- 收益率/标签列**

适用于收益率和标签列，目的是消除行业间收益差异。

| 字段名 | 类型 | 说明 | 计算方式 |
|--------|------|------|----------|
| neu_y_ret_5 | float | 行业中性化后的5日标签 | y_ret_5 - mean_industry(y_ret_5) |
| neu_y_ret_10 | float | 行业中性化后的10日标签 | y_ret_10 - mean_industry(y_ret_10) |
| neu_y_ret_20 | float | 行业中性化后的20日标签 | y_ret_20 - mean_industry(y_ret_20) |
| neu_ret_5 | float | 行业中性化后的5日收益 | ret_5 - mean_industry(ret_5) |
| neu_ret_10 | float | 行业中性化后的10日收益 | ret_10 - mean_industry(ret_10) |
| neu_ret_20 | float | 行业中性化后的20日收益 | ret_20 - mean_industry(ret_20) |

**命名规则**：`neu_` 前缀

**类型2：行业内Z-Score - 指标/特征列**

适用于估值、市值、流动性等指标，目的是标准化行业内相对水平。

| 字段名 | 类型 | 说明 | 计算方式 |
|--------|------|------|----------|
| zscore_pe_ttm | float | 市盈率行业内Z-Score | (pe_ttm - mean_industry) / std_industry |
| zscore_pb | float | 市净率行业内Z-Score | (pb - mean_industry) / std_industry |
| zscore_bp | float | 市净率倒数行业内Z-Score | (bp - mean_industry) / std_industry |
| zscore_dv_ttm | float | 股息率行业内Z-Score | (dv_ttm - mean_industry) / std_industry |
| zscore_log_total_mv | float | 对数总市值行业内Z-Score | (log_total_mv - mean_industry) / std_industry |
| zscore_amount_ma20 | float | 20日均成交额行业内Z-Score | (amount_ma20 - mean_industry) / std_industry |
| zscore_turnover_rate | float | 换手率行业内Z-Score | (turnover_rate - mean_industry) / std_industry |
| zscore_volatility_5 | float | 5日波动率行业内Z-Score | (volatility_5 - mean_industry) / std_industry |
| zscore_volatility_10 | float | 10日波动率行业内Z-Score | (volatility_10 - mean_industry) / std_industry |
| zscore_volatility_20 | float | 20日波动率行业内Z-Score | (volatility_20 - mean_industry) / std_industry |
| zscore_net_mf_amount | float | 净资金流入行业内Z-Score | (net_mf_amount - mean_industry) / std_industry |
| zscore_ma_deviation_20 | float | 20日均线偏离度行业内Z-Score | (ma_deviation_20 - mean_industry) / std_industry |

**命名规则**：`zscore_` 前缀

**通用规则**：
- **统计范围**：仅使用 `tradable==1` 的样本计算行业统计量（均值/标准差）
- **小样本处理**：当行业内可交易样本数 < 5 时，回退使用全市场统计量
- **行业列**：基于 `sw_industry`（项目统一主行业名称，默认申万二级）进行分组
- **启用方式**：在 FeatureBuilder 中设置 `apply_industry_neutralization=True`

**使用建议**：
- **训练标签**：推荐使用 `neu_y_ret_20` 作为训练标签（v0.11.0默认）
- **特征选择**：可同时使用原始特征和中性化特征，让模型自主学习
- **行业轮动**：使用原始收益率特征可捕捉行业轮动效应
- **个股选择**：使用中性化特征可专注行业内个股选择能力

#### 12. 新增个股特征 (v0.12.1新增)

| 字段名 | 类型 | 说明 | 计算方式 |
|--------|------|------|----------|
| is_new_stock | int | 新股标记 | list_days < 365 则为 1，否则为 0（自然日） |
| size | float | 流通市值 | circ_mv（来自 daily_basic） |
| zscore_size | float | 行业内流通市值 Z-Score | 对 log1p(size) 按 sw_industry 行业内 Z-Score（仅 tradable==1，min_group_size=5 回退） |
| spec_score | float | 个股特质得分 | zscore_volatility_20 × (−zscore_size) |

**依赖字段**：
- `list_days`：来自 `_add_filter_flags` 计算（基于 stock_basic.list_date）
- `circ_mv`：来自 daily_basic；若缺失则 size/zscore_size/spec_score 为 NaN
- `zscore_volatility_20`：需启用行业中性化（`apply_industry_neutralization=True`）

#### 13. 市场状态特征 (v0.12.1新增)

每日一个标量值，广播到当日所有股票。仅 tradable==1（历史日期以 vol>0 近似）参与截面统计。

| 字段名 | 类型 | 说明 | 计算方式 |
|--------|------|------|----------|
| mkt_vol_cnt | float | 当日市场收益率截面标准差 | std(ret_1(t), tradable==1) |
| mkt_vol_20 | float | mkt_vol_cnt 过去 20 日滚动均值 | mean(mkt_vol_cnt(t-19:t))（无前瞻） |
| mkt_turnover_ratio | float | 市场拥挤度因子 | sum(amount(t)) / sum(circ_mv(t))（tradable==1） |
| mkt_ret_avg_20 | float | 过去 20 日市场平均收益率之和 | sum_{i=0}^{19} mean_cs(ret_1(t-i))（tradable==1） |
| mkt_turnover_std | float | 市场换手率截面标准差 | std(turnover_rate_f(t), tradable==1)（fallback: turnover_rate） |
| mkt_adv_dec_ratio | float | 过去 60 日涨跌比滚动均值 | mean_{i=0}^{59} [(adv(t-i)+1)/(dec(t-i)+1)]（tradable==1） |

**说明**：
- adv = 当日 ret_1 > 0 的股票数；dec = 当日 ret_1 < 0 的股票数（不含 ret_1==0）
- 历史日期以 `vol > 0` 近似 tradable，当日以 `tradable==1` 列（若存在）为准
- 滚动窗口不足时使用现有数据计算（min_periods=1）
- `turnover_rate_f`：优先使用 daily_basic 中的自由流通换手率；若不存在则回退至 `turnover_rate`

#### 14. 行业分层信息字段 (v0.13.0新增)

当 `shenwan_industry` 数据为 L3 格式时，FeatureBuilder 输出以下三层行业字段：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| sw_industry | str | 申万主行业名称（主字段，由 `industry.shenwan_level` 决定，默认 L2） |
| sw_industry_code | str | 申万主行业指数代码 |
| sw_industry_id | int | 主行业稳定整数编码（基于名称排序） |
| sw_l2 | str | 申万二级行业名称（显式保留字段） |
| sw_l2_code | str | 申万二级行业指数代码（显式保留字段） |
| sw_l2_id | int | 二级行业稳定整数编码 |
| sw_l3 | str | 申万三级行业名称（保留更细粒度信息供分析/调试） |
| sw_l3_code | str | 申万三级行业指数代码 |
| sw_l1 | str | 申万一级行业名称 |
| sw_l1_code | str | 申万一级行业指数代码 |
| sw_l1_id | int | 一级行业稳定整数编码 |

**字段映射关系**：`sw_industry*` 始终对应系统统一主口径（由 `industry.shenwan_level` 决定，默认 = L2），`sw_l3*` 保留更细粒度行业，`sw_l1*` 用于更粗粒度回退与解释。

#### 15. 分层回退中性化规则 (v0.13.0新增)

中性化（`apply_industry_neutralization=True`）执行策略：

1. **检测主口径所需层级信息**：根据 `industry.shenwan_level` 选择分层路径；若缺少所需 code 列，则退化为单层 `sw_industry` 分组。
2. **分组逻辑（每列独立判断）**：
   - `l3`：三级行业内 `tradable==1` 样本数 ≥ `min_group_size(=5)` → 使用三级统计量；否则回退到二级，再回退到一级，最后回退到全市场
   - `l2`：二级行业内 `tradable==1` 样本数 ≥ `min_group_size(=5)` → 使用二级统计量；否则回退到一级，再回退到全市场
   - `l1`：一级行业内 `tradable==1` 样本数 ≥ `min_group_size(=5)` → 使用一级统计量；否则回退到全市场
3. **统计过程无前瞻**：仅使用当日截面数据。
4. **命名规范**：去均值输出 `neu_` 前缀；Z-Score 输出 `zscore_` 前缀。


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

### v0.9.0 已实现 ✅

1. ✅ 添加更多技术指标特征（RSI、MACD、布林带、KDJ）
2. ✅ 添加行业分类特征（industry_id, alpha_industry）
3. ✅ K线形态特征（振幅、上下影线）
4. ✅ 波动率特征
5. ✅ 量能突变特征
6. ✅ 动量加速度特征

### 短期扩展

1. 添加更多技术指标（ATR、CCI、Williams %R等）
2. 增强基本面特征（PE、PB、ROE等）
3. 支持不同的预测时间窗口（horizon=1, 3, 10等）
4. 市场宽度特征（涨跌家数比、涨停数量等）

### 长期扩展

1. 支持分钟级高频特征
2. 添加另类数据特征（舆情、资金流向等）
3. 支持因子标准化、去极值处理
4. 支持在线特征更新（实盘场景）

## 版本变更历史

### v0.12.1 (2026-02-21) - 新增个股特征与市场状态特征

**新增特征**:
- 个股：is_new_stock, size, zscore_size, spec_score
- 市场状态：mkt_vol_cnt, mkt_vol_20, mkt_turnover_ratio, mkt_ret_avg_20, mkt_turnover_std, mkt_adv_dec_ratio

**命名规范**：zscore 列使用 `zscore_` 前缀（v0.12.0 已切换，本文件同步更新）

详见：[docs/PR/market_and_stock_features.md](./PR/market_and_stock_features.md)

### v0.9.0 (2026-02-18) - 特征工程重构

**Breaking Changes**: 
- 删除 `amount_ratio_5/10/20` 特征
- 删除 `vol_ma5/10/20` 特征
- 需要重新生成 features 并重训模型

**新增特征**:
- K线形态: amplitude, upper_shadow, lower_shadow, body_length
- 波动率: volatility_5/10/20
- 行业: industry, industry_id, alpha_industry, alpha_industry_5/10/20
- 动量: acceleration
- 量能: vol_burst_5/10/20
- 技术指标: rsi_14, kdj_k/d/j, macd_dif/dea/hist, bb_*

**架构变更**:
- 新增 `src/lazybull/factors` 模块，提供可复用的因子库
- FeatureBuilder 重构，委托 factors 模块进行因子计算

详见: [docs/PR/feature_refactoring.md](./PR/feature_refactoring.md)

## 现金流质量因子（v0.96.3 / schema v3：版本化 PIT + 事件驱动 TTM）

### 字段定义

| 字段名 | 类型 | 说明 |
|--------|------|------|
| ocf | float | TTM 经营活动现金流净额（滚动四个季度） |
| ocf_to_revenue | float | TTM OCF / TTM 销售商品、提供劳务收到的现金（`c_fr_sale_sg`，非营业收入）；分母经济尺度下限 1000 万元，比值裁剪 ±10 |
| ocf_to_profit | float | TTM OCF / TTM 净利润；Q1/Q3 期 net_profit 覆盖率约 2%，训练侧按缺失率门禁剔除；分母下限 1000 万元，比值裁剪 ±50 |
| fcf | float | TTM 自由现金流（TuShare `free_cashflow` 供应商口径） |
| fcf_yield | float | fcf / 总市值（handler 层计算；总市值分母下限 1 亿元，裁剪 ±1） |
| capex_to_ocf | float | TTM 资本支出（`c_pay_acq_const_fiolta`）/ TTM OCF；分母下限 1000 万元，比值裁剪 ±50 |
| cashflow_freshness_days | float | 最近一次现金流公告（f_ann_date）距当日天数 |
| cashflow_quality_schema_v2 | int | 稳定命名的语义哨兵列（当前恒写版本号 3）；旧语义分区缺失或值不符时自动重建，训练入口校验失败 |

训练列：`zscore_ocf_to_revenue` / `zscore_ocf_to_profit` / `zscore_fcf_yield` / `zscore_capex_to_ocf`（含 `_sz` 市值中性化变体）+ `cashflow_freshness_days`。

### 核心语义（v3）

- **版本化 PIT**：数据可用时间取 `f_ann_date`（实际公告日，缺失回退 `ann_date`）；去重键 `(ts_code, end_date, f_ann_date)` 保留同报告期多次修订；同键冲突优先按 TuShare 官方 `update_flag=1` 最新记录选择，标志相同或缺失时以全行内容哈希稳定决胜，不再依赖输入行序。
- **事件驱动 TTM**：`TTM(q_y) = cum(q_y) − cum(q_{y−1}) + cum(Q4_{y−1})`。当前期、去年同季度、去年 Q4 任一版本可用时都会生成当前期新快照，依赖期晚到修订会在其可用日重算 TTM 并重置 freshness；Q4 年报退化为当年累计。
- **数值稳定性**：分母经济尺度下限 + 比值有界裁剪（见上表），负 OCF/负利润的方向信息保留。
- **数据链路**：cashflow_vip 的季度与公告日查询均按 6400 分页；默认加载两年历史满足 TTM 依赖；近 8 季度每日刷新，首次升级及之后每 90 天全历史复查，水位仅在对应范围全部成功后推进。

### 迁移说明

- 启用 `--enable-cashflow-quality-features` 的区间必须重建 `cs_train` / `cs_infer`；哨兵列值不是 3 时训练入口会明确失败。
- 离线构建前应对所用历史范围执行 `python scripts/download_raw.py ... --download cashflow --force`；起点至少早于训练起点两年。纸面 ensure 首次运行会自动全历史补齐版本，之后每 90 天复查。
- schema v2 及更早模型与新分区同名异义，普通加载和 skip-training 均会告警，必须以重建后的特征重新训练。

## 参考资料

- [TuShare Pro API文档](https://tushare.pro/document/2)
- [XGBoost官方文档](https://xgboost.readthedocs.io/)
- [因子扩展开发指南](./guide/factor_extension.md)
- 项目README: `README.md`
- 数据契约: `docs/data_contract.md`
