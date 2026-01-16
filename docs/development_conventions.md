# LazyBull 开发约定

本文档说明 LazyBull 项目的开发规范和约定。

## 语言约定

### 文档语言

**本仓库对外文档默认使用中文**

所有面向用户的文档必须使用中文编写，包括但不限于：

- `README.md` - 项目说明
- `docs/` 目录下所有文档
- PR 描述和标题
- Issue 描述和标题
- 更新日志（CHANGELOG）
- 迁移指南
- 使用示例

### 代码注释

**代码注释和文档字符串统一使用中文**

```python
# ✅ 推荐：中文注释
def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """计算收益率
    
    Args:
        prices: 价格数据，包含 close 列
        
    Returns:
        包含 returns 列的 DataFrame
    """
    # 计算日收益率
    returns = prices['close'].pct_change()
    return returns


# ❌ 不推荐：英文注释
def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate returns
    
    Args:
        prices: Price data with close column
        
    Returns:
        DataFrame with returns column
    """
    # Calculate daily returns
    returns = prices['close'].pct_change()
    return returns
```

**例外情况**：
- 变量名、函数名、类名使用英文（遵循 Python 命名规范）
- 第三方库名、技术术语保留英文
- 代码中的字符串常量可根据具体场景决定

### 日志消息

日志消息统一使用中文：

```python
# ✅ 推荐
logger.info("开始拉取数据...")
logger.warning(f"股票 {ts_code} 数据缺失")
logger.error(f"保存文件失败: {e}")

# ❌ 不推荐
logger.info("Start fetching data...")
logger.warning(f"Missing data for {ts_code}")
logger.error(f"Failed to save file: {e}")
```

### 命令行帮助

argparse 的 help 参数使用中文：

```python
# ✅ 推荐
parser.add_argument(
    "--start_date",
    type=str,
    help="开始日期，格式YYYYMMDD"
)

# ❌ 不推荐
parser.add_argument(
    "--start_date",
    type=str,
    help="Start date in YYYYMMDD format"
)
```

## 代码规范

### 命名规范

遵循 PEP 8 规范：

```python
# 模块和包：小写+下划线
from lazybull.data import storage
from lazybull.features.builder import FeatureBuilder

# 类名：大驼峰
class TushareClient:
    pass

# 函数和变量：小写+下划线
def load_daily_data():
    trade_date = "20230101"
    stock_list = []

# 常量：大写+下划线
MAX_RETRIES = 3
DEFAULT_HORIZON = 5

# 私有成员：前缀单下划线
def _internal_helper():
    pass

self._private_var = 0
```

### 类型标注

推荐使用类型标注，提升代码可读性：

```python
from typing import List, Optional, Dict
import pandas as pd

def merge_data(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: List[str]
) -> pd.DataFrame:
    """合并数据
    
    Args:
        left: 左表
        right: 右表
        on: 连接键列表
        
    Returns:
        合并后的数据
    """
    return pd.merge(left, right, on=on, how='left')
```

### 文档字符串

使用 Google 风格的文档字符串：

```python
def build_features(
    trade_date: str,
    daily_data: pd.DataFrame,
    stock_basic: pd.DataFrame
) -> pd.DataFrame:
    """构建单日特征
    
    从日线数据和股票基本信息构建特征。
    
    Args:
        trade_date: 交易日期，格式 YYYYMMDD
        daily_data: 日线数据
        stock_basic: 股票基本信息
        
    Returns:
        特征 DataFrame，包含以下列：
        - trade_date: 交易日期
        - ts_code: 股票代码
        - ret_5: 5日收益率
        - vol_ratio_5: 5日成交量比率
        
    Raises:
        ValueError: 当日期格式不正确时
        KeyError: 当必需列缺失时
        
    Examples:
        >>> features = build_features(
        ...     trade_date='20230110',
        ...     daily_data=daily_df,
        ...     stock_basic=stock_df
        ... )
        >>> print(len(features))
        4523
    """
    pass
```

### 异常处理

明确捕获特定异常，避免空 except：

```python
# ✅ 推荐
try:
    df = pd.read_parquet(file_path)
except FileNotFoundError:
    logger.warning(f"文件不存在: {file_path}")
    return None
except Exception as e:
    logger.error(f"读取文件失败: {e}")
    raise

# ❌ 不推荐
try:
    df = pd.read_parquet(file_path)
except:  # 避免空 except
    return None
```

### 日志记录

使用合适的日志级别：

```python
from loguru import logger

# DEBUG: 调试信息，详细的执行过程
logger.debug(f"处理股票 {ts_code}，当前价格 {price}")

# INFO: 关键流程节点
logger.info("开始构建特征...")
logger.info(f"特征构建完成，样本数: {len(features)}")

# WARNING: 警告信息，不影响继续执行
logger.warning(f"股票 {ts_code} 数据缺失，已跳过")

# ERROR: 错误信息，需要人工介入
logger.error(f"保存文件失败: {e}")

# CRITICAL: 严重错误，程序无法继续
logger.critical("数据库连接失败，程序退出")
```

## 数据规范

### 数据分层

```
data/
├── raw/        # 原始层：未处理的数据
├── clean/      # 清洗层：标准化、去重、补全
├── features/   # 特征层：工程特征、可训练
└── reports/    # 报告层：回测结果
```

### 主键约定

- `trade_cal`: trade_date (交易日期)
- `stock_basic`: ts_code (股票代码)
- `daily`: (trade_date, ts_code)
- `features`: (trade_date, ts_code)

### 日期格式

- **TuShare 接口**：YYYYMMDD（如 "20230101"）
- **文件命名**：YYYY-MM-DD（如 "2023-01-01"）
- **内部处理**：pandas.Timestamp

工具函数统一转换：

```python
# YYYYMMDD -> YYYY-MM-DD
formatted = pd.to_datetime(date_str, format='%Y%m%d').strftime('%Y-%m-%d')

# YYYY-MM-DD -> YYYYMMDD
formatted = pd.to_datetime(date_str).strftime('%Y%m%d')
```

### 字段命名

- 使用小写+下划线：`trade_date`, `ts_code`, `close_adj`
- 布尔字段：`is_` 前缀（`is_st`, `is_open`）
- 比率字段：`_ratio` 后缀（`vol_ratio_5`, `turnover_ratio`）
- 收益率：`ret_` 前缀（`ret_1`, `ret_5`, `ret_20`）
- 标签：`y_` 前缀（`y_ret_5`, `y_direction`）

## 测试规范

### 测试组织

```
tests/
├── conftest.py              # pytest 配置和 fixture
├── test_storage.py          # 存储测试
├── test_features.py         # 特征测试
├── test_calendar.py         # 日历测试
└── test_cost.py             # 成本测试
```

### 测试命名

```python
# 测试函数：test_<功能>_<场景>
def test_save_raw_creates_file():
    """测试 save_raw 创建文件"""
    pass

def test_load_raw_missing_file_returns_none():
    """测试 load_raw 在文件缺失时返回 None"""
    pass

# 测试类：Test<模块名>
class TestFeatureBuilder:
    def test_calculate_returns(self):
        pass
    
    def test_filter_st_stocks(self):
        pass
```

### Fixture 使用

```python
import pytest
import pandas as pd

@pytest.fixture
def sample_daily_data():
    """样例日线数据"""
    return pd.DataFrame({
        'trade_date': ['20230101', '20230102'],
        'ts_code': ['000001.SZ', '000002.SZ'],
        'close': [10.0, 20.0]
    })

def test_merge_with_sample_data(sample_daily_data):
    """使用 fixture 测试数据合并"""
    assert len(sample_daily_data) == 2
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定文件
pytest tests/test_features.py

# 运行特定测试
pytest tests/test_features.py::test_calculate_returns

# 查看覆盖率
pytest --cov=src/lazybull --cov-report=html

# 显示详细输出
pytest -v

# 显示打印输出
pytest -s
```

## Git 规范

### 分支命名

```
main                    # 主分支
copilot/<功能描述>      # Copilot 开发分支
feature/<功能名>        # 功能分支
bugfix/<问题描述>       # 修复分支
hotfix/<紧急修复>       # 紧急修复
```

### 提交消息

使用中文，简洁描述变更：

```
# ✅ 推荐
新增 clean 数据层功能
修复停牌数据过滤逻辑
优化特征构建性能
更新文档：数据契约说明

# ❌ 不推荐
Add clean data layer
Fix bug
Update
WIP
```

### PR 标题和描述

- 标题：简洁描述主要功能（中文）
- 描述：详细说明变更内容、测试情况（中文）
- 避免 Copilot 自动附加的英文 suffix

```markdown
# ✅ 推荐
标题：新增 clean 数据层并打通全流程

描述：
## 目的
实现 clean 数据层，支持数据清洗和质量校验...

## 变更内容
1. 新增 Storage 类，支持 raw/clean/features 三层存储
2. 实现 FeatureBuilder，包含数据清洗和特征构建
...
```

## 配置管理

### 配置文件层级

```
configs/
├── base.yaml                    # 基础配置
├── strategy_dividend_value.yaml # 策略配置
├── runtime_local.yaml           # 本地运行
└── runtime_cloud.yaml           # 云端运行
```

### 配置加载

```python
from src.lazybull.common.config import get_config

# 加载配置（按顺序覆盖）
config = get_config([
    "configs/base.yaml",
    "configs/strategy_dividend_value.yaml",
    "configs/runtime_local.yaml"
])

# 访问配置
data_root = config['data']['root']
min_list_days = config['features']['min_list_days']
```

### 敏感信息

- 使用 `.env` 文件管理敏感信息
- `.env.example` 提供模板
- `.gitignore` 忽略 `.env`

```bash
# .env.example
TS_TOKEN=your_tushare_token_here

# .env（不提交）
TS_TOKEN=1234567890abcdef
```

## 依赖管理

### 使用 Poetry（推荐）

```bash
# 添加依赖
poetry add pandas

# 添加开发依赖
poetry add --dev pytest

# 更新依赖
poetry update

# 导出 requirements.txt
poetry export -f requirements.txt -o requirements.txt --without-hashes
```

### 使用 pip

```bash
# 安装依赖
pip install -r requirements.txt

# 冻结依赖
pip freeze > requirements.txt
```

### 版本固定

- 生产依赖：固定主版本（如 `pandas>=1.5.0,<2.0.0`）
- 开发依赖：可灵活（如 `pytest>=7.2.0`）
- 特殊依赖：严格固定（如 `tensorflow==2.10`）

## 文档规范

### 文档结构

```
docs/
├── data_contract.md              # 数据契约
├── features_schema.md            # 特征定义
├── backtest_assumptions.md       # 回测假设
├── roadmap.md                    # 项目路线图
├── development_conventions.md    # 开发约定（本文档）
└── migration_partitioned_storage.md  # 迁移指南
```

### Markdown 规范

- 使用三级标题层次（#, ##, ###）
- 代码块标注语言（```python, ```bash）
- 表格对齐
- 适当使用列表和引用

### 示例代码

文档中的示例代码应：
- 可直接运行或易于修改后运行
- 包含必要的导入语句
- 添加注释说明关键步骤
- 提供预期输出示例

## 代码审查

### 审查清单

提交 PR 前自查：

- [ ] 代码符合 PEP 8 规范
- [ ] 注释和文档字符串为中文
- [ ] 添加了必要的单元测试
- [ ] 所有测试通过
- [ ] 更新了相关文档
- [ ] PR 标题和描述使用中文
- [ ] 没有提交敏感信息
- [ ] 代码有适当的日志记录
- [ ] 异常处理合理

### 代码格式化

```bash
# 使用 black 格式化
black src/ tests/

# 使用 isort 排序导入
isort src/ tests/

# 使用 flake8 检查
flake8 src/ tests/
```

## 性能优化

### Pandas 优化

```python
# ✅ 向量化操作
df['ret'] = df['close'].pct_change()

# ❌ 避免循环
for i in range(len(df)):
    df.loc[i, 'ret'] = (df.loc[i, 'close'] / df.loc[i-1, 'close']) - 1

# ✅ 使用 query
filtered = df.query("close > 10 and volume > 1000000")

# ❌ 多次布尔索引
filtered = df[df['close'] > 10]
filtered = filtered[filtered['volume'] > 1000000]
```

### 内存优化

```python
# 指定 dtype 减少内存
df = pd.read_parquet(
    file_path,
    columns=['trade_date', 'ts_code', 'close']  # 只读取需要的列
)

# 使用分类类型
df['ts_code'] = df['ts_code'].astype('category')
```

## 总结

本文档提供了 LazyBull 项目的开发规范和最佳实践。遵循这些约定可以确保代码质量、提升团队协作效率、降低维护成本。

**核心原则**：
1. **中文优先**：文档、注释、日志统一中文
2. **代码规范**：遵循 PEP 8 和 Python 最佳实践
3. **数据规范**：清晰的数据分层和字段命名
4. **测试完整**：充分的单元测试覆盖
5. **文档完善**：详细的使用说明和示例

如有疑问或建议，欢迎提 Issue 讨论。
