# PR: 行业中性化与申万行业分类接入

**版本**: v0.10.0  
**创建日期**: 2026-02-19  
**作者**: deltree-y

## 概述

本PR实现了行业中性化（行业内Z-Score）特征工程，并新增TuShare申万一级行业分类数据源接入。这是量化策略中常用的技术，用于消除行业间的系统性差异，提高因子的纯净度。

## 核心功能

### 1. 申万行业分类数据源接入

- **数据来源**: TuShare Pro申万行业分类（Shenwan 2021版）一级行业
- **数据结构**:
  - `ts_code`: 股票代码
  - `sw_code`: 申万行业指数代码
  - `sw_name`: 申万行业名称（一级，如"农林牧渔"、"化工"等）
  - `in_date`: 纳入日期

- **下载命令**:
  ```bash
  # 首次下载或更新
  python scripts/update_basic_data.py --only-shenwan --force
  
  # 与其他基础数据一起更新（推荐）
  python scripts/update_basic_data.py --force
  ```

- **数据存储**: `data/raw/shenwan_industry.parquet`（单文件，非分区）
- **更新频率**: 建议每季度更新一次（行业调整不频繁）

### 2. 行业中性化（行业内Z-Score）

#### 数学定义

对于特征 `x`，行业中性化定义为：

```
neu_x = (x - mean_industry(x)) / std_industry(x)
```

其中 `mean_industry(x)` 和 `std_industry(x)` 分别是该行业内（申万一级）的均值和标准差。

#### 实现口径

1. **截面范围**: 仅使用当日 `tradable == 1` 的样本参与统计
   - ST股票、停牌股票、上市未满60天的股票已在 `tradable` 标记中被过滤

2. **小样本处理**:
   - 参数: `min_group_size = 5`
   - 当行业样本数 < 5 时，回退使用全市场（同一日、`tradable==1`）的均值和标准差
   - 确保所有样本都能得到有效的中性化值

3. **执行位置**: 在 `FeatureBuilder.build_features_for_day` 中执行
   - 确保训练、回测、纸面交易使用相同的中性化逻辑
   - 避免前瞻偏差

#### 白名单列

以下特征列会被自动进行行业中性化（如果存在）：

| 原始列 | 中性化列 | 说明 |
|--------|----------|------|
| `pe_ttm` | `neu_pe_ttm` | 市盈率（TTM） |
| `pb` | `neu_pb` | 市净率 |
| `bp` | `neu_bp` | 市净率倒数 |
| `dv_ttm` | `neu_dv_ttm` | 股息率（TTM） |
| `log_total_mv` | `neu_log_total_mv` | 对数总市值 |
| `amount_ma20` | `neu_amount_ma20` | 20日均成交额 |
| `turnover_rate` | `neu_turnover_rate` | 换手率 |
| `net_mf_amount` | `neu_net_mf_amount` | 净资金流入 |
| `ret_20` | `neu_ret_20` | 20日收益率 |
| `ma_deviation_20` | `neu_ma_deviation_20` | 20日均线偏离度 |
| `volatility_5` | `neu_volatility_5` | 5日波动率 |
| `volatility_10` | `neu_volatility_10` | 10日波动率 |
| `volatility_20` | `neu_volatility_20` | 20日波动率 |

**注意**: 
- 原始列保留，新增 `neu_*` 前缀的中性化列
- 如果某列不存在（例如缺少 `daily_basic` 或 `moneyflow` 数据），会跳过该列的中性化
- 行业中性化默认**不启用**，需要在特征构建时显式指定

### 3. 使用方式

#### 在FeatureBuilder中使用

```python
from src.lazybull.data import DataLoader
from src.lazybull.features import FeatureBuilder

# 初始化
loader = DataLoader()
builder = FeatureBuilder()

# 加载申万行业数据
shenwan_industry = loader.load_shenwan_industry()

# 构建特征（启用行业中性化）
features = builder.build_features_for_day(
    trade_date='20240101',
    trade_cal=trade_cal,
    daily_data=daily_data,
    adj_factor=adj_factor,
    stock_basic=stock_basic,
    daily_basic_data=daily_basic_data,
    moneyflow_data=moneyflow_data,
    shenwan_industry=shenwan_industry,  # 传递行业数据
    apply_industry_neutralization=True  # 启用中性化
)

# 查看中性化后的特征
print(features[['ts_code', 'sw_name', 'pe_ttm', 'neu_pe_ttm']].head())
```

#### 在特征构建脚本中使用

特征构建脚本 `scripts/build_features.py` 和 `scripts/build_clean_features.py` 会自动：
1. 检测是否存在申万行业数据
2. 如果存在，自动启用行业中性化
3. 如果不存在，给出警告并跳过中性化

#### 单独使用中性化模块

```python
from src.lazybull.factors.normalization import industry_neutralization
import pandas as pd

# 准备数据（必须包含行业列和tradable列）
df = pd.DataFrame({
    'ts_code': ['000001.SZ', '000002.SZ', ...],
    'pe_ttm': [10.0, 20.0, ...],
    'pb': [1.0, 2.0, ...],
    'sw_name': ['农林牧渔', '化工', ...],  # 行业列
    'tradable': [1, 1, ...]  # 可交易标记
})

# 应用行业中性化
result = industry_neutralization(
    df,
    columns=['pe_ttm', 'pb'],  # 需要中性化的列
    industry_col='sw_name',     # 行业列名
    tradable_col='tradable',    # 可交易标记列
    min_group_size=5,           # 最小组内样本数
    prefix='neu_',              # 输出列前缀
    inplace=False               # 不覆盖原列
)

# 查看结果
print(result[['ts_code', 'sw_name', 'pe_ttm', 'neu_pe_ttm']].head())
```

## 数据接口说明

### TushareClient 新增方法

```python
# 获取申万指数列表（一级行业）
index_basic = client.get_index_basic(market='SW')

# 获取申万行业分类
index_classify = client.get_index_classify(level='L1', src='SW2021')

# 获取指数成分股
members = client.get_index_member(index_code='801010.SI')
```

### DataCleaner 新增方法

```python
# 清洗申万行业数据，生成 ts_code -> 行业映射
clean_df = cleaner.clean_shenwan_industry(
    raw_index_basic=index_basic_df,
    raw_index_members={
        '801010.SI': members_df_1,
        '801020.SI': members_df_2,
        ...
    }
)
```

### DataLoader 新增方法

```python
# 加载申万行业分类数据
shenwan_industry = loader.load_shenwan_industry()
# 返回 DataFrame: ts_code, sw_code, sw_name, in_date
```

## 新增字段说明

### 特征表新增字段

构建features后，DataFrame会包含以下新增字段：

| 字段名 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| `sw_code` | str | 申万行业指数代码 | '801010.SI' |
| `sw_name` | str | 申万行业名称（一级） | '农林牧渔' |
| `industry_id` | int | 行业整数编码（用于模型） | 0, 1, 2, ... |
| `neu_pe_ttm` | float | 行业中性化后的市盈率 | 0.5, -1.2, ... |
| `neu_pb` | float | 行业中性化后的市净率 | -0.3, 0.8, ... |
| `neu_log_total_mv` | float | 行业中性化后的对数市值 | 1.2, -0.5, ... |
| ... | ... | 其他中性化特征 | ... |

## 重建Features说明

如果你已有features数据，需要重新构建以包含行业中性化特征：

```bash
# 1. 先下载申万行业数据
python scripts/update_basic_data.py --only-shenwan --force

# 2. 重新构建features（指定日期范围）
python scripts/build_features.py --start-date 20230101 --end-date 20231231 --force

# 3. 或使用clean数据构建（更快）
python scripts/build_clean_features.py --start-date 20230101 --end-date 20231231 --force
```

**注意**: 
- 重建features会覆盖原有数据
- 如果不想立即使用行业中性化，可以不重建
- 行业中性化特征主要用于机器学习模型训练

## 测试

新增测试文件 `tests/test_industry_neutralization.py`，包含：

1. ✅ 全市场Z-Score测试
2. ✅ 只使用tradable==1样本的测试
3. ✅ 行业内Z-Score（组内样本数>=5）测试
4. ✅ 小样本回退（组内样本数<5）测试
5. ✅ 行业中性化函数基本功能测试
6. ✅ 缺少列/行业列时的错误处理测试
7. ✅ log_total_mv中性化测试
8. ✅ 白名单列集成测试

运行测试：
```bash
pytest tests/test_industry_neutralization.py -v
```

## 兼容性说明

- **向后兼容**: 
  - 如果不提供 `shenwan_industry` 参数，或 `apply_industry_neutralization=False`，则不进行行业中性化
  - 原有特征构建流程完全兼容，不受影响
  
- **数据依赖**:
  - 申万行业数据是可选的，不是必需的
  - 如果缺少行业数据，会给出清晰的警告信息和下载指引

- **历史features**:
  - 不会自动更新历史features
  - 如需使用行业中性化，需要重新构建features

## 注意事项

1. **TuShare权限**:
   - `index_basic`, `index_classify`, `index_member` 接口需要TuShare Pro权限
   - 普通积分可能无法访问，请确认权限后使用

2. **数据完整性**:
   - 部分股票可能没有申万行业分类（例如新股、ST股）
   - 这些股票的 `sw_name` 字段会为 NaN
   - 中性化时会自动跳过这些股票

3. **性能考虑**:
   - 行业中性化在单日截面数据上执行，性能影响很小
   - 申万行业数据为单文件（约5000条记录），加载速度快

4. **模型训练**:
   - 建议使用中性化后的特征（`neu_*`）进行模型训练
   - 可以同时保留原始特征和中性化特征，让模型自动选择

## 相关文档

- [行业中性化使用指南](../guide/industry_neutralization_guide.md)
- [特征字段说明](../features_schema.md)
- [数据更新指南](../data_contract.md)

## 变更日志

### v0.10.0 (2026-02-19)

**新增**:
- 申万行业分类数据源接入（TuShare）
- 行业中性化（行业内Z-Score）功能
- `industry_neutralization` 和 `cross_sectional_zscore` 函数
- `src/lazybull/factors/normalization.py` 模块
- 行业数据下载和清洗逻辑
- 完整的单元测试覆盖

**修改**:
- `FeatureBuilder.build_features_for_day` 新增参数：
  - `shenwan_industry`: 申万行业数据
  - `apply_industry_neutralization`: 是否启用中性化
- `update_basic_data.py` 新增 `--only-shenwan` 选项
- 版本号更新至 v0.10.0

**文档**:
- 更新 README 当前版本说明
- 新增 PR 说明文档
- 新增行业中性化使用指南
- 更新特征字段说明文档
