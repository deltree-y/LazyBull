# LazyBull - A股量化研究与回测框架

<div align="center">

**专注价值红利策略的量化投资框架**

[![Python](https://img.shields.io/badge/Python-3.9.13-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[功能特性](#功能特性) • [快速开始](#快速开始) • [项目结构](#项目结构) • [文档](#文档) • [计划功能](#计划功能-roadmap)

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

### 核心功能模块

- ✅ **完整的项目骨架**: 模块化设计，易于扩展
- ✅ **TuShare数据接入**: 自动拉取交易日历、股票列表、日线行情、财务指标
- ✅ **Parquet存储**: 高效的列式存储，加速数据读取
- ✅ **回测引擎**: 支持日/周/月频调仓，**支持自定义天数调仓**（如每5天、10天）
- ✅ **T+1 交易规则**: T 日生成信号，T+1 日收盘价买入，T+n 日卖出（收盘价或开盘价可配置）
- ✅ **可配置卖出时机**: **支持 T+n 日开盘价卖出或收盘价卖出**，默认收盘价卖出
- ✅ **涨跌停与停牌处理**: **信号生成时基于T+1数据过滤并回填，确保top N可交易**
- ✅ **实时进度显示**: 回测时使用 tqdm 进度条实时显示当前日期、净值、耗时，**支持详细日志开关**
- ✅ **仓位补齐机制**: **调仓后未满仓时自动在补齐窗口期内尝试补齐，确保回测实盘一致**
- ✅ **止损机制**: 回撤止损、移动止损、连续跌停止损（可选）
- ✅ **价格口径配置**: 统一使用不复权价格计算成本，后复权价格计算收益
- ✅ **收益明细跟踪**: 每笔卖出交易自动计算收益金额和收益率（已扣除成本）
- ✅ **信号生成**: 提供等权、因子打分等多种方法
- ✅ **报告生成**: 自动计算收益率、夏普、最大回撤等指标，支持中文列名
- ✅ **单元测试**: 基于pytest的测试框架，**测试数据隔离，不污染工作区**
- ✅ **ML 模型训练**: 支持 XGBoost 模型训练，自动验证集评估
- ✅ **模型优化**: 早停机制、标签 winsorize、正则化、IC/RankIC 评估
- ✅ **特征优化**: 向量化计算提升特征生成效率
- ✅ **现金流质量因子**: 基于 `f_ann_date` 的版本化 PIT、依赖修订事件驱动 TTM 与供应商自由现金流口径
- 🧪 **分红政策质量因子（待 WF 验证）**: 分红稳定性/增长率、归母净利润支付率 + 双日期稠密事件因子，每股调整口径 PIT 截断、`ex_date` 防前视
- ✅ **IC优化指南**: 提供系统性的 IC/RankIC 提升方案和诊断工具
- ✅ **数据质量看板**: 按本地数据截止日扫描 raw/clean/features 的覆盖率、区间加权缺失率、异常值、schema 版本和同步水位，输出离线 HTML 报告与 Parquet 快照
- ✅ **默认参数优化**: Top N=5, 初始资金=50万, 周频调仓, 默认排除ST
- ✅ **成交额过滤**: 在信号生成（选股）阶段过滤成交额后N%的股票，提高持仓流动性
- ✅ **分批调仓**: 总 Top-N 槽位与资金按周期拆批，支持漏批追赶、配置迁移及 T1 失败同日顺位补买
- ✅ **止损触发**: 支持回撤止损、移动止损、连续跌停止损

### 计划功能 (Roadmap)

- ✅ **纸面交易（Paper Trading）**: 日频工作流，T0 生成信号，T1 执行打印，支持状态持久化
- 🔲 完整的价值红利因子库（分红政策质量因子 v0.98.3 已实现，待 WF 验证）
- 🔲 组合优化与风险管理
- 🔲 云端定时任务
- 🔲 实盘接口（长期）

详见 [项目路线图](docs/roadmap.md)

---

## 🚀 快速开始

### 环境要求

- Python: 3.9.13
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

#### 模式一：分步构建（推荐）

适合需要分步骤、精细控制的场景：

```bash
# 步骤1: 仅下载raw数据（不构建clean/features）
python scripts/download_raw.py --start-date 20230101 --end-date 20231231

# 风控公告类数据（质押/解禁/大宗，供风控模型专用因子使用）
python scripts/download_raw.py --start-date 20230101 --end-date 20231231 --download pledge_stat share_float block_trade

# 现金流质量因子首次启用或升级 schema v3：起点至少早于训练起点两年，并强制重建版本化 raw
python scripts/download_raw.py --start-date 20210101 --end-date 20231231 --download cashflow --force

# 分红送股数据（分红政策质量因子，按股全历史查询 + ann_date 年分区）
# 必须显式请求 base_share 基准股本用于支付率；存量数据缺该列会自动触发重下，也可手动 --force
python scripts/download_raw.py --start-date 20210101 --end-date 20231231 --download dividend

# 利润表归母净利润（分红支付率，首次接入需强制建立 f_ann_date 版本化季度分区）
python scripts/download_raw.py --start-date 20170101 --end-date 20231231 --download income --force

# 步骤2: 构建clean和features（假设raw已存在）
# --horizon / --horizons 二选一必填：
#   --horizon 20         : 单值模式，仅按主 horizon 对应的 y_ret_20 非空过滤（推荐，保留停牌导致的辅助标签缺失样本）
#   --horizons 5 10 20   : 多值模式，AND 过滤，要求所有 horizons 对应 y_ret_N 同时非空
# 两种模式下生成的特征文件都包含 y_ret_5/10/20 三列，schema 一致
python scripts/build_clean_features.py --start-date 20230101 --end-date 20231231 --horizon 20

# 启用风控公告类因子（质押/解禁/大宗，需先下载 pledge_stat/share_float/block_trade）
python scripts/build_clean_features.py --start-date 20230101 --end-date 20231231 --horizon 20 --enable-announcement-risk-features

# 启用现金流质量因子（默认加载两年 TTM 预热；schema v3 会拦截旧缓存）
python scripts/build_clean_features.py --start-date 20230101 --end-date 20231231 --horizon 20 --enable-cashflow-quality-features

# 启用分红政策质量因子（需先下载 dividend + 有效 income 合并年报，缺失时构建会直接失败）
python scripts/build_clean_features.py --start-date 20230101 --end-date 20231231 --horizon 20 --enable-dividend-policy-features

# 或者只构建clean
python scripts/build_clean_features.py --start-date 20230101 --end-date 20231231 --only-clean --horizon 20

# 或者只构建features（假设clean已存在）
python scripts/build_clean_features.py --start-date 20230101 --end-date 20231231 --only-features --horizon 20

# 强制重新构建
python scripts/build_clean_features.py --start-date 20230101 --end-date 20231231 --force --horizon 20
```

#### 模式二：仅更新基础数据

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

#### 数据质量看板

全历史扫描本地 raw、clean、`cs_train`、`cs_infer` 分区，生成可离线打开的 HTML 报告和 Parquet 指标快照：

```bash
# 默认读取 configs/base.yaml 的 quality 阈值，输出至 data/reports/quality/
python scripts/quality_dashboard.py

# 对指定数据目录和日期区间执行诊断（end-date 即本次权威截止日）
python scripts/quality_dashboard.py --data-root ./data --start-date 20250101 --end-date 20260731
```

未传 `--end-date` 时，覆盖率截止日取本地 `raw/daily` 最新分区，不使用当前日期或交易日历外推。各层有效起点和允许尾差由 `quality.coverage_start_dates`、`quality.coverage_tail_lag_trading_days` 配置；`cs_train` 默认允许标签所需的 21 个交易日尾窗。逐分区缺失率保留作明细，错误门禁使用全扫描区间的行数加权缺失率，特征全空列仍会失败。

退出码 `0` 表示无错误，`1` 表示扫描完成但发现质量错误，`2` 表示扫描执行失败。扫描时会输出当前分区、累计进度、耗时和预计剩余时间；进度心跳间隔在 `configs/base.yaml` 的 `quality.progress_interval_seconds` 中统一配置。HTML 使用状态卡片、数据集摘要和紧凑表格呈现结果，仅展示前 `quality.html_max_detail_rows` 条异常和快照变化（默认 100），完整明细保存在同目录 `latest_metrics.parquet`。

##### 纸面交易（Paper Trading）

LazyBull 支持纸面交易工作流，用于模拟实盘交易：

```bash
# T0 工作流：拉取数据 + 生成T1待执行目标
python scripts/paper_trade.py t0 --trade-date 20260121 --buy-price close --universe mainboard --top-n 5

# T1 工作流：读取待执行目标 + 执行订单 + 打印明细
python scripts/paper_trade.py t1 --trade-date 20260122 --buy-price close --sell-price close

# 实时行情：查看持仓当前价格及盈亏（--trade-date 默认当日）
python scripts/paper_trade.py real

# 精简模式：仅输出单行收益统计（适合定时任务/屏幕显示）
python scripts/paper_trade.py real --ret-profit-only
```

**纸面交易特点：**
- T0/T1 分离工作流（T0 收盘后生成信号，T1 执行调仓）
- 完整的持久化（账户状态、交易记录、净值曲线）
- 详细的打印输出（股票、方向、权重、价格、成本、原因）
- 灵活的价格配置（买入可选开盘价/收盘价）
- 主板股票池（仅沪深主板，排除科创板、创业板、北交所）
- 成本计算（佣金、印花税、滑点）
- **实时行情查看**（`real` 子命令，基于 Tushare `realtime_quote` 接口）

详见 [纸面交易使用指南](docs/paper_trading_guide.md)

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

# 使用现金流质量因子；旧 schema 模型必须重新训练
python scripts/train_ml_model.py --start-date 20230101 --end-date 20231231 \
  --enable-cashflow-quality-features

# 步骤2: 使用 ML 模型运行回测（使用新的默认值）
# 注意：scripts/run_ml_backtest.py 已删除，回测已并入 walk_forward 滚动回测，
# 或经 src.lazybull.common.backtest_runtime 工厂驱动 BacktestEngineML

# 批量运行最小因子实验（共同基线、历史股息率、两对现金流）
powershell -ExecutionPolicy Bypass -File .\scripts\batch\batch_walk_forward.ps1
```

批量脚本的 `factor_experiment_configs` 默认使用相同参数运行三组方案：不启用候选因子的
基线、仅保留 `dividend_yield_hist_12m` 的分红方案，以及仅保留 `fcf_yield` 和
`ocf_to_revenue`（均含 `_sz` 版）的现金流方案。三组方案复用相同的 OOS split 配置，
并通过因子开关、精简清单列及独立汇总文件区分；候选清单位于 `configs/factor_exclude_dividend_yield_only_v1.json` 与
`configs/factor_exclude_cashflow_keep_2pairs_v1.json`，不会覆盖生产排除清单。

Walk-forward 对比表的全周期 CAGR 按有效日收益区间进行几何年化；每个 split 重复的
起始净值点不计入交易日数，链式 Sharpe 同样只使用 split 内日收益。

**ML 模型特点：**
- 使用全量特征列训练 XGBoost 回归模型
- 标签为 `y_ret_5`（未来 5 日收益率，T+1 收盘买入 / T+1+5 开盘卖出口径）
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
│   ├── update_basic_data.py   # 更新trade_cal和stock_basic
│   ├── train_ml_model.py      # 训练 ML 模型
│   ├── run_backtest.py        # 运行回测
│   ├── run_ml_backtest.py     # 运行 ML 信号回测
│   ├── compare_walk_forward.py # 实验对比与稳定性汇总（薄入口）
│   ├── compare/               # 实验对比分析子包（constants/loading/aggregate/scoring/...）
│   └── ana/
│       ├── analyze_factor_importance.py # 因子重要性分析
│       └── analyze_factor_stability.py  # 集成模型因子使用稳定性分析
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
│   ├── factors/               # 因子库模块 ✅ v0.9.0
│   │   ├── technical_indicators.py  # 技术指标（RSI/KDJ/MACD/布林带）
│   │   ├── candlestick.py          # K线形态（振幅/上下影线）
│   │   ├── volatility.py           # 波动率
│   │   ├── industry.py             # 行业相关（alpha/偏离）
│   │   ├── momentum.py             # 动量加速度
│   │   ├── volume.py               # 量能突变
│   │   └── risk/                   # 风控模型专用因子子包 ✅ v0.92.4
│   │       ├── factor_registry.py      # 因子注册表 + compute_all_risk_factors()
│   │       ├── downside_factors.py     # 下行风险（VaR/CVaR/偏度/峰度）
│   │       ├── volatility_factors.py   # 波动结构（Parkinson/GARCH 等）
│   │       ├── liquidity_factors.py    # 流动性风险（Amihud/量价背离等）
│   │       ├── announcement_factors.py # 公告类（质押/解禁/大宗/融券）
│   │       ├── announcement_lookup.py  # 公告类 PIT 日频查询表 ✅ v0.94.0
│   │       ├── derived_factors.py      # 衍生（momentum_decay/earnings_yield）
│   │       └── position_features.py    # 持仓上下文特征
│   ├── features/              # 特征构建模块
│   │   ├── builder.py         # 特征构建器（调用 factors 模块）
│   │   └── handlers_announcement.py # 风控公告类因子处理器 ✅ v0.94.0
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
- [纸面交易使用指南](docs/paper_trading_guide.md): 纸面交易（Paper Trading）完整使用指南 ⭐ 新增
- [项目路线图](docs/roadmap.md): 分阶段开发计划
- [IC与RankIC优化指南](docs/ic_optimization_guide.md): 提升模型预测能力的系统性优化方案
- [成交额过滤指南](docs/amount_filter_guide.md): 成交额过滤功能说明与配置
- [分批调仓指南](docs/batch_rebalance_guide.md): 分批调仓功能说明与配置
- [止损触发指南](docs/stop_loss_guide.md): 止损触发功能说明与配置
- [重大变更说明](docs/BREAKING_CHANGES.md): v0.4.0 版本的 Breaking Changes ⚠️ 重要
- [项目更新记录](docs/PR/UPDATES.md): 历史版本更新说明
- [重构总结](docs/PR/REFACTOR_SUMMARY.md): 代码重构文档

---

## 🎯 使用示例

### 1. 命令行使用（推荐）

#### 分步构建

```bash
# 第一步：下载raw数据
python scripts/download_raw.py --start-date 20230101 --end-date 20231231

# 第二步：构建clean和features（--horizon 20 走单值过滤；如需 AND 过滤用 --horizons 5 10 20）
python scripts/build_clean_features.py --start-date 20230101 --end-date 20231231 --horizon 20
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

# 示例1：基础回测（每20个交易日调仓，约1个月）
engine = BacktestEngine(
    universe=universe,
    signal=signal,
    initial_capital=1000000,
    cost_model=cost_model,
    rebalance_freq=20  # 每20个交易日调仓
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

# 示例3：周频调仓（每5个交易日，约1周）
engine = BacktestEngine(
    universe=universe,
    signal=signal,
    initial_capital=1000000,
    cost_model=cost_model,
    rebalance_freq=5,  # 每5个交易日调仓
    verbose=True  # 输出详细交易日志
)

# 示例4：配置卖出时机为开盘价（默认为收盘价）
engine = BacktestEngine(
    universe=universe,
    signal=signal,
    initial_capital=1000000,
    cost_model=cost_model,
    rebalance_freq=5,
    sell_timing='open'  # T+n日开盘价卖出，默认为'close'（收盘价卖出）
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

默认运行时仅自动加载 `configs/base.yaml`。如需使用其他 YAML 覆盖，需要在代码中显式调用 `merge_config()` 或自行指定配置文件。

```
base.yaml (默认自动加载)

runtime_local.yaml / runtime_cloud.yaml
  └─ 手工覆盖示例，不会被默认流程自动合并
```

### 主要配置项

```yaml
# configs/base.yaml
data:
  root: "./data"
  raw: "./data/raw"
  clean: "./data/clean"
  features: "./data/features"
  reports: "./data/reports"

tushare:
  max_retries: 3
  retry_delay: 1
  rate_limit: 200

industry:
  shenwan_level: "l2"  # 支持 l1 / l2 / l3

costs:
  commission_rate: 0.0001954
  min_commission: 5          # 最低5元
  stamp_tax: 0.0005
  slippage: 0.0005
```

说明：
- 未显式传入 `--data-root` 时，训练、回测、walk-forward、因子分析、纸面交易与树莓派显示脚本都会使用 `configs/base.yaml` 中的项目默认路径。
- `data.root`、`data.raw`、`data.clean`、`data.features`、`data.reports` 为当前真实接线的数据目录配置；模型目录与纸面交易目录默认分别派生为 `data.root/models` 与 `data.root/paper`。
- 命令行显式指定路径或参数时，仍优先于项目配置。

---

## 🧪 开发指南

### 添加新因子（v0.5.0）

LazyBull 使用模块化的因子库架构。添加新因子只需在 `src/lazybull/factors/` 中添加函数，无需修改核心代码。

```python
# 在 src/lazybull/factors/technical_indicators.py 中添加
import numpy as np
import pandas as pd

def calculate_your_indicator(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """计算您的技术指标
    
    Args:
        df: DataFrame，需包含 ts_code, trade_date, close_adj
        window: 窗口参数
        
    Returns:
        DataFrame，包含 ts_code, trade_date, your_indicator
    """
    result = df[['ts_code', 'trade_date']].copy()
    
    # 按股票分组计算
    grouped = df.sort_values(['ts_code', 'trade_date']).groupby('ts_code')
    
    indicator_values = []
    for ts_code, group in grouped:
        group = group.sort_values('trade_date').copy()
        
        # 实现您的计算逻辑
        indicator = group['close_adj'].rolling(window=window).mean()
        
        temp_df = pd.DataFrame({
            'ts_code': ts_code,
            'trade_date': group['trade_date'].values,
            'your_indicator': indicator.values
        })
        indicator_values.append(temp_df)
    
    if indicator_values:
        result = pd.concat(indicator_values, ignore_index=True)
    else:
        result['your_indicator'] = np.nan
    
    return result
```

然后在 `FeatureBuilder._add_advanced_factors()` 中调用该函数。

**因子归属约定**（v0.92.4）：
- 通用选股因子 → `src/lazybull/factors/` 根目录（如 `technical_indicators.py`）
- **风控模型专用因子** → `src/lazybull/factors/risk/` 子包，通过 `@register_risk_factor`
  装饰器注册到 `factor_registry.py`，由 builder 统一计算；`risk/` 仅保留风控逻辑
- 公告类原始数据（质押/解禁/大宗）→ `factors/risk/announcement_lookup.py` 构建
  PIT 日频查询表，经 `features/handlers_announcement.py` 处理器合并进 features，
  再交由 `announcement_factors.py` 三层加工（v0.94.0）
  （PositionRiskModel/label_builder/precompute 调度），不放因子本体
- 新增任何因子都必须遵循此归属，避免因子逻辑散落在脚本或 risk 模块中

**详细指南**: 参见 [docs/guide/factor_extension.md](docs/guide/factor_extension.md)

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
