# LazyBull - A股量化研究与回测框架

<div align="center">

**专注价值红利策略的量化投资框架**

[![Python](https://img.shields.io/badge/Python-3.9.13-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[功能特性](#功能特性) • [快速开始](#快速开始) • [项目结构](#项目结构) • [文档](#文档) • [路线图](#路线图)

</div>

---

## 📖 项目简介

LazyBull 是一个轻量级的A股量化研究与回测框架，专注于**价值红利**方向的策略研究。项目支持从本地开发到云端自动化运行的完整生命周期，强调**可复现性**和**可迁移性**。

### 核心理念
- 🎯 **专注价值红利**: 聚焦高股息率、低估值策略
- 📊 **数据驱动**: 基于TuShare Pro接口获取全面数据
- 🔄 **周频/月频**: 适合中长期持仓，降低交易成本
- ☁️ **云端友好**: 易于部署到云端定时任务
- 🇨🇳 **中文优先**: 代码注释、文档均使用中文

---

## ✨ 功能特性

### 当前版本 (v0.4.0 - 功能增强与重构版)

- ✅ **完整的项目骨架**: 模块化设计，易于扩展
- ✅ **TuShare数据接入**: 自动拉取交易日历、股票列表、日线行情、财务指标
- ✅ **Parquet存储**: 高效的列式存储，加速数据读取
- ✅ **回测引擎**: 支持日/周/月频调仓，**支持自定义天数调仓**（如每5天、10天）
- ✅ **T+1 交易规则**: T 日生成信号，T+1 日收盘价买入，T+n 日收盘价卖出
- ✅ **涨跌停与停牌处理**: **信号生成时基于T+1数据过滤并回填，确保top N可交易**（优化）
- ✅ **实时进度显示**: 回测时使用 tqdm 进度条实时显示当前日期、净值、耗时，**支持详细日志开关**
- ✅ **价格口径配置**: 统一使用不复权价格计算成本，后复权价格计算收益
- ✅ **收益明细跟踪**: 每笔卖出交易自动计算收益金额和收益率（已扣除成本）
- ✅ **信号生成**: 提供等权、因子打分等多种方法
- ✅ **报告生成**: 自动计算收益率、夏普、最大回撤等指标，支持中文列名
- ✅ **单元测试**: 基于pytest的测试框架，**测试数据隔离，不污染工作区**
- ✅ **ML 模型训练**: 支持 XGBoost 模型训练，自动验证集评估
- ✅ **模型优化**: 早停机制、标签 winsorize、正则化、IC/RankIC 评估
- ✅ **特征优化**: 向量化计算提升特征生成效率
- ✅ **IC优化指南**: 提供系统性的 IC/RankIC 提升方案和诊断工具
- ✅ **默认参数优化**: Top N=5, 初始资金=50万, 周频调仓, 默认排除ST
- ✅ **成交额过滤**: 在信号生成（选股）阶段过滤成交额后N%的股票，提高持仓流动性（新增）
- ✅ **分批调仓**: 支持将完整调仓分多批执行，降低冲击成本（新增）
- ✅ **止损触发**: 支持回撤止损、移动止损、连续跌停止损（新增）

### v0.4.0 更新内容（2026-01-19）

**功能增强与重构** - 新增三大核心功能，优化数据架构：

#### 新增功能
- **成交额过滤**: 在信号生成（选股）阶段剔除成交额后N%的股票（默认20%），提高持仓流动性
  - **重要**: 仅在选股时过滤，模型训练时保留所有股票数据以保证学习效果
- **分批调仓**: 支持将完整调仓分多批执行，例如20只股票分4周调仓，每周5只
- **止损触发**: 支持回撤止损、移动止损、连续跌停止损，实现风险管理

#### 重构变更 ⚠️ Breaking Changes
- **删除 filter_ 前缀**: clean 数据列名简化
  - `filter_is_st` → `is_st`
  - `filter_is_suspended` → `is_suspended`
  - `filter_list_days` → `list_days`
- **删除 price_type 参数**: 统一价格口径，简化配置
- **文档重组**: PR 相关文档移至 `docs/PR/`

详见 [重大变更说明](docs/BREAKING_CHANGES.md)

### v0.3.1 更新内容（2026-01-19）

**涨跌停与停牌处理优化** - 重要设计变更：
- **新逻辑**: 信号生成时基于 T+1 日数据检查涨跌停/停牌，从候选池自动回填，确保 top N 全部可交易
- **为什么变更**: T 日涨跌停不代表 T+1 日也涨跌停，延迟一天会引入新的市场变化，不应使用旧预测
- **Universe过滤**: 仅过滤停牌股票，涨跌停不在此过滤（留给信号生成阶段基于T+1数据处理）
- **延迟订单**: 买入不再使用延迟订单（已在信号生成时过滤），仅用于卖出跌停情况
- **信号接口**: 新增 `generate_ranked()` 方法支持回填候选
- **详细日志**: 显示检查候选数、过滤数量、回填情况
- **测试验证**: 37个测试全部通过，确保功能正确
- 详见 [涨跌停与停牌处理指南](docs/trade_status_guide.md)

### v0.3.0 更新内容（2026-01-18）

**涨跌停与停牌处理** - 初版实现（已在v0.3.1优化）:
- 选股阶段过滤、延迟订单机制
- 详见v0.3.1的设计优化

### v0.2.1 更新内容

**回测进度优化**:
- 进度条实时刷新显示，不再缓存到最后输出
- 添加 `verbose` 参数控制详细日志输出
- 日志输出到 stderr，进度条输出到 stdout，避免混乱
- 优化进度条配置（固定宽度、加快刷新频率）

**调仓频率增强**:
- **支持自定义天数**：`rebalance_freq=5` 表示每5个交易日调仓一次
- 保持向后兼容：`D`/`W`/`M` 仍然有效
- 添加参数校验，提供清晰的中文错误提示
- 持有期自动匹配调仓频率

**价格口径配置**:
- 新增 `price_type` 参数，支持选择 `close`（不复权）、`close_adj`（后复权）、`close_hfq`（前复权）
- **默认使用不复权价格**（`close`），更符合实际交易
- 提供详细的[价格口径说明文档](docs/price_type_guide.md)
- 包含迁移指南和结果对比说明

**测试数据隔离**:
- 所有测试使用临时目录（`tempfile.TemporaryDirectory`）
- 测试运行不会修改 `data/` 目录中的文件
- 确保测试的独立性和可重复性

**IC/RankIC 优化指南**:
- 新增[IC优化指南文档](docs/ic_optimization_guide.md)
- 涵盖特征工程、标签定义、样本选择、模型训练等全方位优化
- 提供可执行的代码示例和评估工具
- 包含短期、中期、长期分阶段优化建议

**特征数据优化**:
- 移除 `filter_list_days` 作为 filter 列，改为信息列 `list_days`
- filter 列现在只包含 `is_st` 和 `suspend`
- 所有输出列去掉 `filter_` 前缀，列名更简洁

**模型训练增强**:
- 训练时自动按时间切分验证集（默认 20%）
- 训练结束打印验证集评估结果（MSE、RMSE、R2）
- 随机种子固定，保证可复现性

**回测体验优化**:
- 实时打印回测进度（当前日期、完成度、耗时、ETA）
- 报告列名改为中文（日期、组合总值、净值、收益率等）
- 交易记录列名改为中文（交易日期、股票代码、操作、成交价格等）

**交易规则更新**:
- 实现 T+1 买入、T+n 卖出逻辑
- T 日生成信号，T+1 日收盘价买入
- 持有 n 天后（T+n 日）收盘价卖出
- 持有期可自定义或根据调仓频率自动设置

**性能优化**:
- 特征生成使用向量化计算，提升效率
- 使用 pandas groupby + agg 替代循环
- 使用 np.where 替代条件赋值

### 计划功能 (Roadmap)

- 🔲 完整的价值红利因子库
- 🔲 组合优化与风险管理
- 🔲 云端定时任务
- 🔲 实盘接口（长期）

详见 [项目路线图](docs/roadmap.md)

---

## 🚀 快速开始

### 环境要求

- Python: 3.9.13
- TensorFlow: 2.10 (不可升级)
- 操作系统: Linux/macOS/Windows

### 方式一: 使用Poetry安装（推荐）

```bash
# 1. 克隆仓库
git clone https://github.com/deltree-y/LazyBull.git
cd LazyBull

# 2. 安装Poetry (如未安装)
curl -sSL https://install.python-poetry.org | python3 -

# 3. 安装依赖
poetry install

# 4. 激活虚拟环境
poetry shell
```

### 方式二: 使用pip安装

```bash
# 1. 克隆仓库
git clone https://github.com/deltree-y/LazyBull.git
cd LazyBull

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt
```

### 配置TuShare Token

```bash
# 1. 复制环境变量模板
cp .env.example .env

# 2. 编辑.env文件，填入你的TuShare token
# TS_TOKEN=your_tushare_token_here

# 获取token: https://tushare.pro/register
```

### 运行示例

LazyBull 提供三种数据处理模式，适应不同使用场景：

#### 模式一：快速开始（推荐）- 一键构建特征

最简单的方式，自动补齐所有依赖：

```bash
# 直接构建特征，自动下载raw、构建clean（如缺失）
python scripts/build_features.py --start-date 20230101 --end-date 20231231

# 强制重新构建所有数据
python scripts/build_features.py --start-date 20230101 --end-date 20231231 --force
```

#### 模式二：分步构建 - 精细控制

适合需要分步骤、精细控制的场景：

```bash
# 步骤1: 仅下载raw数据（不构建clean/features）
python scripts/download_raw.py --start-date 20230101 --end-date 20231231

# 步骤2: 构建clean和features（假设raw已存在）
python scripts/build_clean_features.py --start-date 20230101 --end-date 20231231

# 或者只构建clean
python scripts/build_clean_features.py --start-date 20230101 --end-date 20231231 --only-clean

# 或者只构建features（假设clean已存在）
python scripts/build_clean_features.py --start-date 20230101 --end-date 20231231 --only-features

# 强制重新构建
python scripts/build_clean_features.py --start-date 20230101 --end-date 20231231 --force
```

#### 模式三：仅更新基础数据

更新trade_cal和stock_basic（用于定时任务）：

```bash
# 更新交易日历和股票列表
python scripts/update_basic_data.py

# 仅更新交易日历
python scripts/update_basic_data.py --only-trade-cal

# 仅更新股票列表
python scripts/update_basic_data.py --only-stock-basic

# 强制更新（即使已是最新）
python scripts/update_basic_data.py --force
```

#### 运行回测

```bash
# 运行回测 (如无数据会使用mock数据演示)
python scripts/run_backtest.py
```

#### 机器学习模型训练与回测

LazyBull 支持基于机器学习模型的量化策略：

```bash
# 步骤1: 训练 XGBoost 模型（使用已构建的特征数据）
python scripts/train_ml_model.py --start-date 20230101 --end-date 20231231

# 自定义超参数训练
python scripts/train_ml_model.py --start-date 20230101 --end-date 20231231 \
    --n-estimators 200 --max-depth 5 --learning-rate 0.05

# 步骤2: 使用 ML 模型运行回测（使用新的默认值）
# 默认：Top N=5, 初始资金=50万, 周频调仓, 排除ST
python scripts/run_ml_backtest.py --start-date 20230101 --end-date 20231231

# 自定义参数示例
python scripts/run_ml_backtest.py --start-date 20230101 --end-date 20231231 \
    --model-version 1 --top-n 10 --initial-capital 1000000

# 指定调仓频率（M=月度，W=周度，D=日度）
python scripts/run_ml_backtest.py --start-date 20230101 --end-date 20231231 \
    --rebalance-freq M --top-n 5

# 使用自定义天数调仓（每10个交易日调仓一次）
python scripts/run_ml_backtest.py --start-date 20230101 --end-date 20231231 \
    --rebalance-freq 10 --top-n 5

# 包含ST股票（默认排除）
python scripts/run_ml_backtest.py --start-date 20230101 --end-date 20231231 \
    --include-st
```

**ML 模型特点：**
- 使用全量特征列训练 XGBoost 回归模型
- 标签为 `y_ret_5`（未来 5 日收益率）
- **训练时自动切分验证集**（默认最后 20% 时间作为验证集）
- **训练结束后打印验证集评估结果**（MSE、RMSE、R2、IC、RankIC）
- **使用早停机制**（early_stopping_rounds=30）防止过拟合
- **标签 winsorize 处理**减少极端值影响
- **增加正则化参数**（L1/L2）提升泛化能力
- 模型自动保存到 `data/models` 目录
- 版本号自动递增（v1, v2, v3...）
- 元数据记录在 `model_registry.json`
- 支持排序选股 Top N 策略
- 随机种子固定（random_state=42），保证可复现

**默认回测参数：**
- Top N: 5（选择前5只股票）
- 初始资金: 500,000（50万）
- 调仓频率: W（周频）
- 排除ST: 是（默认过滤ST股票）

**查看模型文件：**
```bash
ls data/models/              # ML 模型目录
  ├── model_registry.json    # 模型版本注册表
  ├── v1_model.joblib        # 模型文件
  ├── v1_features.json       # 特征列表
  ├── v2_model.joblib
  └── v2_features.json
```

#### 查看数据

```bash
ls data/raw/              # 原始数据
  ├── trade_cal.parquet        # 交易日历（单文件）
  ├── stock_basic.parquet      # 股票列表（单文件）
  ├── daily/                   # 日线行情（按日分区）
  │   └── YYYY-MM-DD.parquet
  ├── daily_basic/             # 每日指标（按日分区）
  └── ...

ls data/clean/            # 清洗后数据（包含复权价格和可交易标记）
  ├── trade_cal.parquet        # 清洗后交易日历
  ├── stock_basic.parquet      # 清洗后股票列表
  └── daily/                   # 清洗后日线（按日分区）
      └── YYYY-MM-DD.parquet

ls data/features/         # 特征数据
  └── cs_train/                # 截面训练特征（按日分区）
      └── YYYYMMDD.parquet

ls data/reports/          # 回测报告
```

### 数据架构说明

LazyBull 采用三层数据架构，统一使用 **partitioned 存储**：

- **raw 层**: 从 TuShare 直接拉取的原始数据
  - `trade_cal`、`stock_basic`: 单文件存储（不分区）
  - 其他数据（daily、daily_basic等）: 按日期分区存储 `{YYYY-MM-DD}.parquet`
  
- **clean 层**: 经过清洗和标准化的数据
  - 去重（按主键 ts_code+trade_date）
  - 类型统一（trade_date 统一为 YYYYMMDD 字符串）
  - 复权价格（close_adj, open_adj, high_adj, low_adj）
  - 可交易标记（tradable, is_st, is_suspended, is_limit_up, is_limit_down）
  - 数据校验和排序
  - 存储方式同raw层
  
- **features 层**: 基于 clean 数据计算的特征和标签
  - 按交易日分区存储: `{YYYYMMDD}.parquet`

### force 参数说明

所有脚本均支持 `--force` 参数：

- 默认行为：存在即跳过（节省时间）
- 使用 `--force`：强制重新下载/构建并覆盖已有文件
- 适用场景：数据更正、重新计算、完整性检查

### trade_cal 和 stock_basic 更新策略

这两个基础数据采用"智能更新"策略：

1. **判断逻辑**：
   - `trade_cal`: 检查本地最新日期是否覆盖所需范围
   - `stock_basic`: 简化为检查文件是否存在（建议每季度手动更新）

2. **更新方式**：
   - 每次更新都是全量更新（不是增量patch）
   - 保证数据完整性和一致性

3. **推荐频率**：
   - `trade_cal`: 每年年初更新一次（新增当年全部数据）
   - `stock_basic`: 每季度更新一次
   - 或在 cron 中定期运行 `update_basic_data.py`

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_cost.py
pytest tests/test_features.py
pytest tests/test_cleaner.py

# 查看覆盖率
pytest --cov=src/lazybull --cov-report=html
```

---

## 📁 项目结构

```
LazyBull/
├── configs/                    # 配置文件
│   ├── base.yaml              # 基础配置
│   ├── strategy_dividend_value.yaml  # 红利价值策略配置
│   ├── runtime_local.yaml     # 本地运行配置
│   └── runtime_cloud.yaml     # 云端运行配置
├── data/                       # 数据目录
│   ├── raw/                   # 原始数据（支持按日分区）
│   │   └── {name}/            # 按日分区: YYYY-MM-DD.parquet
│   ├── clean/                 # 清洗后数据（支持按日分区）
│   │   └── {name}/            # 按日分区: YYYY-MM-DD.parquet
│   ├── features/              # 特征数据
│   ├── models/                # ML 模型目录
│   │   ├── model_registry.json  # 模型版本注册表
│   │   └── v*_model.joblib    # 训练好的模型文件
│   └── reports/               # 回测报告
├── docs/                       # 文档
│   ├── data_contract.md       # 数据契约
│   ├── backtest_assumptions.md # 回测假设
│   └── roadmap.md             # 路线图
├── scripts/                    # 脚本
│   ├── download_raw.py        # 下载raw数据
│   ├── build_clean_features.py # 构建clean和features
│   ├── build_features.py      # 直接构建features（自动补齐依赖）
│   ├── update_basic_data.py   # 更新trade_cal和stock_basic
│   ├── train_ml_model.py      # 训练 ML 模型
│   ├── run_backtest.py        # 运行回测
│   └── run_ml_backtest.py     # 运行 ML 信号回测
├── src/lazybull/              # 源代码
│   ├── common/                # 通用模块
│   │   ├── config.py          # 配置管理
│   │   ├── logger.py          # 日志工具
│   │   └── cost.py            # 成本模型
│   ├── data/                  # 数据模块
│   │   ├── tushare_client.py  # TuShare客户端
│   │   ├── storage.py         # 数据存储
│   │   └── loader.py          # 数据加载
│   ├── universe/              # 股票池模块
│   │   └── base.py            # 股票池基类
│   ├── factors/               # 因子模块 (TODO)
│   ├── signals/               # 信号模块
│   │   ├── base.py            # 信号基类
│   │   └── ml_signal.py       # ML 信号生成器
│   ├── ml/                    # 机器学习模块
│   │   └── model_registry.py  # 模型版本管理
│   ├── portfolio/             # 组合管理 (TODO)
│   ├── execution/             # 执行模块 (TODO)
│   ├── backtest/              # 回测模块
│   │   ├── engine.py          # 回测引擎
│   │   └── reporter.py        # 报告生成
│   └── live/                  # 实盘模块 (TODO)
├── tests/                      # 测试
│   ├── conftest.py            # pytest配置
│   ├── test_config.py         # 配置测试
│   ├── test_cost.py           # 成本模型测试
│   └── test_calendar.py       # 日历测试
├── .env.example               # 环境变量模板
├── .gitignore                 # Git忽略文件
├── pyproject.toml             # Poetry配置
├── requirements.txt           # pip依赖
└── README.md                  # 本文件
```

---

## 📚 文档

- [数据契约](docs/data_contract.md): 各数据层的字段规范与主键约定
- [回测假设](docs/backtest_assumptions.md): 回测系统的假设、简化与局限性
- [特征与标签定义](docs/features_schema.md): 日频特征构建、标签计算、过滤规则说明
- [涨跌停与停牌处理指南](docs/trade_status_guide.md): 涨跌停与停牌状态的自动处理机制
- [项目路线图](docs/roadmap.md): 分阶段开发计划
- [IC与RankIC优化指南](docs/ic_optimization_guide.md): 提升模型预测能力的系统性优化方案
- [成交额过滤指南](docs/amount_filter_guide.md): 成交额过滤功能说明与配置 ⭐ 新增
- [分批调仓指南](docs/batch_rebalance_guide.md): 分批调仓功能说明与配置 ⭐ 新增
- [止损触发指南](docs/stop_loss_guide.md): 止损触发功能说明与配置 ⭐ 新增
- [重大变更说明](docs/BREAKING_CHANGES.md): v0.4.0 版本的 Breaking Changes ⚠️ 重要
- [项目更新记录](docs/PR/UPDATES.md): 历史版本更新说明
- [重构总结](docs/PR/REFACTOR_SUMMARY.md): 代码重构文档

---

## 🎯 使用示例

### 1. 命令行使用（推荐）

#### 快速开始 - 一键构建

```bash
# 最简单方式：直接构建特征，自动补齐所有依赖
python scripts/build_features.py --start-date 20230101 --end-date 20231231
```

#### 分步构建 - 精细控制

```bash
# 第一步：下载raw数据
python scripts/download_raw.py --start-date 20230101 --end-date 20231231

# 第二步：构建clean和features
python scripts/build_clean_features.py --start-date 20230101 --end-date 20231231
```

#### 定期更新基础数据

```bash
# 在cron或定时任务中运行
python scripts/update_basic_data.py
```

### 2. Python API 使用

#### 下载和清洗数据

```python
from src.lazybull.data import TushareClient, Storage, DataCleaner

# 初始化（Storage现在默认使用partitioned存储）
client = TushareClient()  # 从环境变量读取TS_TOKEN
storage = Storage()  # 统一使用partitioned存储
cleaner = DataCleaner()

# 下载基础数据（单文件存储）
trade_cal = client.get_trade_cal("20230101", "20231231")
storage.save_raw(trade_cal, "trade_cal", is_force=True)

stock_basic = client.get_stock_basic()
storage.save_raw(stock_basic, "stock_basic", is_force=True)

# 下载日线数据（按日期分区存储）
trade_date = "20230110"
daily_data = client.get_daily(trade_date=trade_date)
storage.save_raw_by_date(daily_data, "daily", trade_date)

# 清洗数据
trade_cal_clean = cleaner.clean_trade_cal(trade_cal)
storage.save_clean(trade_cal_clean, "trade_cal", is_force=True)

stock_basic_clean = cleaner.clean_stock_basic(stock_basic)
storage.save_clean(stock_basic_clean, "stock_basic", is_force=True)

# 清洗日线数据（按日期分区）
adj_factor = client.get_adj_factor(trade_date=trade_date)
daily_clean = cleaner.clean_daily(daily_data, adj_factor)
storage.save_clean_by_date(daily_clean, "daily", trade_date)
```

#### 使用 clean 数据构建特征

```python
from src.lazybull.data import DataLoader, Storage
from src.lazybull.features import FeatureBuilder

# 初始化
storage = Storage()
loader = DataLoader(storage)
builder = FeatureBuilder(
    min_list_days=60,  # 最小上市60天
    horizon=5          # 预测未来5个交易日
)

# 加载 clean 数据（优先使用，已包含复权价格）
trade_cal = loader.load_clean_trade_cal()
stock_basic = loader.load_clean_stock_basic()
daily_clean = loader.load_clean_daily("20230101", "20231231")

# clean 数据已包含复权价格列：close_adj, open_adj, high_adj, low_adj
# 以及可交易标记：tradable, is_st, is_suspended, is_limit_up, is_limit_down
print(daily_clean.columns)

# 构建单日特征（clean 数据自动跳过复权计算）
features = builder.build_features_for_day(
    trade_date='20230110',
    trade_cal=trade_cal,
    daily_data=daily_clean,
    adj_factor=pd.DataFrame(),  # clean 数据已含复权价格，无需提供
    stock_basic=stock_basic
)

# 保存特征
storage.save_cs_train_day(features, '20230110')
```

### 3. 传统方式：使用 raw 数据

```python
from src.lazybull.data import DataLoader, Storage
from src.lazybull.features import FeatureBuilder

# 初始化
storage = Storage()
loader = DataLoader(storage)
builder = FeatureBuilder(
    min_list_days=60,  # 最小上市60天
    horizon=5          # 预测未来5个交易日
)

# 加载数据
trade_cal = loader.load_trade_cal()
stock_basic = loader.load_stock_basic()
daily_data = storage.load_raw("daily")
adj_factor = storage.load_raw("adj_factor")

# 构建单日特征
features = builder.build_features_for_day(
    trade_date='20230110',
    trade_cal=trade_cal,
    daily_data=daily_data,
    adj_factor=adj_factor,
    stock_basic=stock_basic
)

# 保存特征
storage.save_cs_train_day(features, '20230110')

# 加载特征
features = storage.load_cs_train_day('20230110')
print(f"样本数: {len(features)}")
print(f"特征列: {features.columns.tolist()}")
```

### 4. 构建股票池

```python
from src.lazybull.universe import BasicUniverse
from src.lazybull.data import DataLoader
import pandas as pd

# 加载数据
loader = DataLoader()
stock_basic = loader.load_stock_basic()

# 创建股票池
universe = BasicUniverse(
    stock_basic=stock_basic,
    exclude_st=True,          # 排除ST
    min_list_days=252,        # 至少上市1年
    markets=['主板', '创业板']  # 限定市场
)

# 获取某日股票池
stocks = universe.get_stocks(pd.Timestamp('2023-12-31'))
print(f"股票池大小: {len(stocks)}")
```

### 5. 运行回测

```python
from src.lazybull.backtest import BacktestEngine, Reporter
from src.lazybull.signals import EqualWeightSignal
from src.lazybull.common.cost import CostModel

# 初始化组件
signal = EqualWeightSignal(top_n=30)  # 等权30只
cost_model = CostModel()

# 示例1：基础回测（月度调仓）
engine = BacktestEngine(
    universe=universe,
    signal=signal,
    initial_capital=1000000,
    cost_model=cost_model,
    rebalance_freq="M"  # 月度调仓
)

# 示例2：自定义天数调仓
engine = BacktestEngine(
    universe=universe,
    signal=signal,
    initial_capital=1000000,
    cost_model=cost_model,
    rebalance_freq=10,  # 每10个交易日调仓
    verbose=False  # 关闭详细日志，保持输出整洁
)

# 示例3：指定价格类型（推荐使用不复权价格）
engine = BacktestEngine(
    universe=universe,
    signal=signal,
    initial_capital=1000000,
    cost_model=cost_model,
    rebalance_freq="W",
    price_type='close',  # 使用不复权价格（默认，推荐）
    verbose=True  # 输出详细交易日志
)

# 运行回测
nav_curve = engine.run(
    start_date=pd.Timestamp('2023-01-01'),
    end_date=pd.Timestamp('2023-12-31'),
    trading_dates=trading_dates,
    price_data=daily_data
)

# 生成报告
reporter = Reporter()
trades = engine.get_trades()
stats = reporter.generate_report(nav_curve, trades)
```

---

## 🔧 配置说明

### 配置文件层级

配置采用继承机制，后加载的配置会覆盖先加载的：

```
base.yaml (基础配置)
  ↓
strategy_dividend_value.yaml (策略配置)
  ↓
runtime_local.yaml 或 runtime_cloud.yaml (运行时配置)
```

### 主要配置项

```yaml
# configs/base.yaml
data:
  root: "./data"

backtest:
  start_date: "20200101"
  end_date: "20231231"
  initial_capital: 1000000
  rebalance_frequency: "M"

costs:
  commission_rate: 0.0003    # 万3佣金
  min_commission: 5          # 最低5元
  stamp_tax: 0.001           # 千1印花税
  slippage: 0.001            # 0.1%滑点
```

---

## 🧪 开发指南

### 添加新因子

```python
# 在 src/lazybull/factors/ 中创建新文件
class MyFactor:
    def calculate(self, data):
        # 实现因子计算逻辑
        pass
```

### 添加新策略

```python
# 继承 Signal 基类
from src.lazybull.signals.base import Signal

class MyStrategy(Signal):
    def generate(self, date, universe, data):
        # 实现信号生成逻辑
        return {stock: weight for stock, weight in ...}
```

### 代码风格

- 使用 Black 格式化: `black src/ tests/`
- 使用 isort 排序导入: `isort src/ tests/`
- 使用 flake8 检查: `flake8 src/ tests/`

---

## 📊 回测示例输出

```
============================================================
回测报告摘要
============================================================
总收益率      : 15.23%
年化收益率    : 15.50%
最大回撤      : -8.45%
波动率        : 12.30%
夏普比率      : 1.25
交易次数      : 24
总交易成本    : 12345.67元
回测天数      : 252
起始净值      : 1.0000
结束净值      : 1.1523
============================================================
```

---

## ⚠️ 风险提示

1. **历史回测不代表未来**: 过去的表现不预示未来收益
2. **数据质量**: TuShare数据可能存在错误或延迟
3. **简化假设**: 当前版本存在多项简化（详见 [回测假设](docs/backtest_assumptions.md)）
4. **仅供研究**: 本项目仅用于量化研究学习，不构成投资建议

---

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 贡献流程

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

### 开发规范

- 所有代码需通过测试
- 保持测试覆盖率 > 80%
- 遵循现有代码风格
- 更新相关文档

---

## 📄 License

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 📮 联系方式

- 作者: deltree-y
- 项目地址: [https://github.com/deltree-y/LazyBull](https://github.com/deltree-y/LazyBull)
- Issue反馈: [https://github.com/deltree-y/LazyBull/issues](https://github.com/deltree-y/LazyBull/issues)

---

## 🙏 致谢

- [TuShare](https://tushare.pro/): 优秀的财经数据接口
- [Backtrader](https://www.backtrader.com/): 回测框架设计参考
- 所有开源社区的贡献者

---

<div align="center">

**⭐ 如果这个项目对你有帮助，欢迎Star支持！**

Made with ❤️ by deltree-y

</div>
