# LazyBull - A股量化研究与回测框架

<div align="center">

**专注价值红利策略的量化投资框架**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
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
- 🧪 **股东增减持因子（stk_holdertrade，待 WF 验证）**: 30/90 日自然日窗口的净增持比例（含高管/非高管拆分）、披露日计数与强度加速度，`ann_date` 唯一 PIT 锚点，窗口外显式填 0 保证全市场覆盖
- 🧪 **股票回购因子（repurchase，待 WF 验证）**: 90/180 日自然日窗口的**已回购金额增量 / 流通市值**（执行类公告为累计口径 ⇒ 取增量，禁止直接求和）、已执行标志、回购价上限相对现价空间，`ann_date` 唯一 PIT 锚点，窗口外显式填 0 保证全市场覆盖
- 🧪 **十大流通股东因子（top10_floatholders，待 WF 验证）**: 报告期状态保留 + freshness（`ann_date` 唯一 PIT 锚点）；前 10 集中度 / 第一大占比 / **长线机构**（社保·养老·年金·保险）合计占比与户数 / 社保标志 / 集中度环比变化；未披露股票为 NaN（**禁止 0 填充**）
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

- Python: 3.12
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

# 股东增减持数据（stk_holdertrade，按月窗口分页读满 + ann_date 年分区）
# 接口单页上限 3000 行且超限不报错，必须按月窗口 + offset 翻页；水位=分区内最大 ann_date，自动增量续传
python scripts/download_raw.py --start-date 20200101 --end-date 20231231 --download stk_holdertrade

# 股票回购数据（repurchase，按月窗口分页读满 + ann_date 年分区）
# 接口单页上限 2000 行且超限不报错（2022 全年真实 6034 行，不翻页丢 67%）；水位=分区内最大 ann_date
python scripts/download_raw.py --start-date 20100101 --end-date 20260917 --download repurchase

# 十大流通股东数据（top10_floatholders，按报告期批量拉取 + end_date 年分区）
# 接口单页上限 6000 行且超限不报错（单期 5.5 万行 / 10 页）；水位=分区内最大 end_date，
# 增量 = 回拉最近 2 个已存报告期 + 补齐区间内尚无数据的报告期（缺一期就是永久空洞）
python scripts/download_raw.py --start-date 20070101 --end-date 20260918 --download top10_floatholders

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

# 启用股东增减持因子（物化到特征分区；窗口外显式填 0，9 列全覆盖）
# 注意：训练/OOS 评估默认走**运行时派生**（不写分区），无需此步
python scripts/build_clean_features.py --start-date 20230101 --end-date 20231231 --horizon 20 --enable-holdertrade-features

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

#### 因子体检（Factor Health Check）

对当前生产模型的特征清单做一次体检：覆盖率、逐日截面 RankIC（分年 + 高低波动分层）、
平均截面相关矩阵聚类（识别孪生与冗余簇）、既有模型的因子使用度（gain 份额 / 分裂使用率）：

```bash
# 默认：全历史（2020 起）每 3 个交易日采样，特征清单取最新注册模型，产物落 data/reports/factor_health/<时间戳>/
python scripts/analyze_factor_health.py

# 指定区间与步长；跳过模型加载（只看 IC/覆盖/聚类）
python scripts/analyze_factor_health.py --start 20200101 --end 20260702 --every 3 --skip-models

# 指定特征清单与模型版本范围（默认取最近 15 个注册版本）
python scripts/analyze_factor_health.py --feature-file data/models/stock_selection/v24052_features.json \
    --model-versions 24050-24052
```

产物：`factor_health_report.md`（结论 + 家族画像 + 候选清单）、`factor_register.csv`（逐因子台账）、
`daily_ic.csv.gz`、`corr_matrix.csv`、`clusters.csv`、`candidates_*.csv`，
以及 **三份可直接喂给 WF 实验的排除清单**（`exclude_weak_v1.json` / `exclude_dedup_v1.json` /
`exclude_weak_dedup_v1.json`）：

```bash
# 用于单变量消融：弱信息+未用（B）/ 同簇去重（A）分开做，每次只动一个变量
python scripts/walk_forward.py --factor-prune \
    --factor-exclude-file data/reports/factor_health/<时间戳>/exclude_dedup_v1.json
```

口径提示：市场级截面常数（截面 IC 无定义）单独识别、不参与截面 IC 筛选；
负 IC **不**等于坏因子（模型自行学习方向）；候选清单只是**实验输入**，
采纳必须走 WF A/B 对照（预登记判据 + 噪声带）判定。

另生成一份**口径交换清单** `exclude_dedup_plain_v1.json`（同一簇划分，代表改为优先保留非 `_sz`
口径，与默认去重清单同规模）：孪生对里多数 `_sz`（市值中性化）口径 |IC-IR| 更高，直接用默认去重清单会
同时削减 size 暴露，因此两条清单必须成对做单变量对照。

#### 运行时派生家族的体检扫描（快照物化）

股东增减持（`stk_holdertrade`）等**运行时派生**家族列不写入 `cs_train`，而体检/诊断工具只读分区。
`scripts/materialize_factor_health_snapshot.py` 把「特征清单 + 运行时派生列」物化成**独立临时数据根**
（只保留工具实际读取的列，2 GB 量级），以目录联接复用生产 `clean/raw/models`，再把工具的
`data.root` 指向该根，**不改动任何生产数据、口径与生产一致**：

```bash
python scripts/materialize_factor_health_snapshot.py \
  --start 20200101 --end 20260702 --every 3 \
  --with-holdertrade \
  --feature-file data/models/stock_selection/v24123_features.json \
  --skip-usage --usage-model-count 2 \
  --out-root temp/ht_health_root_20260917
```

家族开关：`--with-holdertrade` / `--with-repurchase` / `--with-top10fh`（可在一次运行中并列）。

产物：`<out-root>/reports/factor_health|factor_diagnosis/`（台账、候选清单、报告、特征清单）。
`--skip-usage` 用于模型以 `_model.json`（而非 `_model.joblib`）落盘时——此时工具侧使用度为“缺失”，
**不可读作“未被使用”**，需另行统计。结论示例见 `docs/reports/holdertrade_factor_health.md`。

#### 因子诊断 v2（增量信息 / 覆盖显著性 / 使用度稳定性）

在体检台账基础上回答“**哪条候选值不值得改数据**”，只读 cs_train 与模型文件：

```bash
# 默认取 data/reports/factor_health 最新台账；产物落 data/reports/factor_diagnosis/<时间戳>/
python scripts/analyze_factor_diagnosis.py --start 20200101 --every 3
```

三项诊断：**偏 IC**（控制簇代表后的秩残差 IC——“冗余”的硬定义，仅 `|ρ|` 高不足以判定）、
**覆盖-标签差**（有值/缺失子样本的标签中位差 + 规模分位差，判定缺失是否只是规模代理）、
**使用度稳定性**（跨模型版本 gain 份额的变异系数）；另有**逐年覆盖/幅度剖面**
（每年一个代表分区）用于判根因：**源头起点**（如 cyq 2018+、north 2014-11+）、**资格/覆盖扩张**（如两融标的扩容）、
**覆盖收缩**（如研报披露率下降）、**口径退化**（近年截面幅度不足历史峰值 10%，如转融券暂停后的 `rqye_rzye_ratio`）。
输出《数据改良候选清单》（A 覆盖缺口 / B 无增量信息 / C 有信号未被使用 / D 使用不稳定 / E 缺失携带信息 /
F 口径退化），每行标注**预期效应量级**与**可检出性**（以 14 折 MDE≈5pp 为标尺：列级 0~1pp 的改动不可判定；标尺数值已被 v2 方案 F7 §3.5 实测定稿取代——净值层 MDE=12pp）。
⚠️ **覆盖缺口不得默认“补数据”**：本地 raw 已含 2005+ 数据，缺口多来自 TuShare 起点、标的资格或披露口径。

#### 运行时可用性标记因子（结构性缺失显式化）

诊断 v2 的结论是「一致预期 / 两融 / 基金持仓 / 业绩快报的缺失是**结构性的**（资格名单、覆盖、披露口径），
且只是规模代理」。这类信息不必改数据就能显式喂给模型：

```bash
# 训练与滚动训练：启用运行时可用性标记（默认关闭；不改任何特征分区）
python scripts/train_ml_model.py --start-date 20200101 --end-date 20231231 --enable-availability-markers
python scripts/walk_forward.py --split-count 14 --enable-availability-markers
```

- 标记：`has_cons_coverage` / `has_margin_balance` / `has_fund_holding` / `has_express_data`，
  语义为「该股当日**至少一个**来源列有值」；
- 派生位置：训练侧 `prepare.py` 组装训练矩阵时、推理侧 `signals/ml_signal.py` 预测前
  （**按模型 `feature_columns` 驱动**，所以 train/serve 天然一致，不会静默补成 NaN）；
- `cs_train` / `cs_infer` 分区**完全不改动**；来源列缺失时跳过并告警，不产出常量零列。

#### 下行风险惩罚（A5 路由型，默认关；WF A/B 未通过）

对候选排序做一次**后处理**（不改训练列集、不碰 colsample 稀释通道）：
`score' = 分位(score) − λ×风险分位`，λ 冻结网格 {0, 0.25, 0.5}（0=关闭且与基线逐位一致）：

```bash
# 滚动训练（惩罚在推理侧生效，可配 --skip-training --skip-training-eval 复用旧折模型，零训练成本）
python scripts/walk_forward.py --split-count 14 --skip-training --start-model-version 24126 \
  --skip-training-eval --downside-penalty 0.25 --downside-penalty-column downside_vol_20
# 纸面交易配置
python scripts/paper_trade.py config --downside-penalty 0.25
```

- 口径：风险方向白名单（`downside_vol_20` 越大越危险；`cvar_95_20` 越负越危险）；风险分位缺失按截面中位；缺列必报错；
- `--skip-training-eval`：skip 模式下补跑 OOS 评估（逐日 Top-K 明细 + 测试指标，**不注册模型、不写训练台账**）；
- 预登记 `docs/plans/stock_selection/downside_penalty_prereg.md`；**结论（2026-09-21）：不通过 ⇒ 不采纳**——
  信号层 Δ −43.4 bps（Top20，相对 −47%；三块长区间全负）、净值 ΔCAGR −5.62pp / ΔMaxDD +0.89pp / Δ夏普 −0.188；
  机制根因 = rank 空间相减在 λ=0.25 即强倾斜（Top20 重叠 26.8%、换入 14.6 只/日）。
  详见 `docs/reports/downside_penalty_wf_ab_result.md` ⇒ **生产默认关闭，不再开新臂、禁止消融位搜索**。

#### 模型列集审计（列集漂移 / 配置分组）

列集本身就是模型契约：线上推理的特征矩阵与训练矩阵若不是同一套列，模型默认行为就变了。
本工具只读各版本 `*_features.json`，回答「**哪一列掉了、是掉列还是换配置**」：

```bash
# 最近 40 个版本（默认 source: deploy=data/models/stock_selection）
python scripts/audit_model_columns.py --last 40

# 多来源对照（部署目录 vs 训练折目录）；--last 0 = 全部版本
python scripts/audit_model_columns.py --source deploy=data/models/stock_selection \
    --source fold0=data/walk_forward/batches/<batch>/fold_0 --last 0
```

产物（中文表头，落 `data/reports/model_column_audit/<时间戳>/`）：
`列集配置分组.csv`（按列集完全一致聚类——同一目录会累积多套实验配置，逐版本 diff 会把「换配置」误读成「掉列」）、
`配置差异对比.csv`（各配置相对最新配置的缺失列/多出列）、`列集漂移台账.csv`（逐版本新增/移除明细）、
`列集出现频次.csv`（常驻/新增/已移除/间断出现 + 家族归属）、`跨来源列集差异.csv`（部署 vs 折目录两两差异）、
`模型列集审计.md`。

口径提示：`express_*` 一类稀疏列在不同训练窗口下缺失率不同，会被训练入口缺失率门禁（默认 0.6）
逐折丢列——表现为**间断出现**，属可解释漂移但必须登记；开关驱动的家族列（cyq/margin/north_turnover/lhb）
差异属**配置差异**，应在 `配置差异对比.csv` 里核对而非视为数据缺口。

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

v2 P4 纸面宿主已提供显式引擎选择。**生产已于 2026-10-31 由 legacy 切换为 v2**（`data/paper/config.yaml` 的 `extra.engine: v2`；切换证据 `data/frozen_reference/v2_p4/t5_release_final_20261031/switch_result.json`，备份根 `data/paper_backup_pre_v2_20261031`，旧链保活根 `data/paper_legacy_pre_switch_20261031`）。键缺失时的缺省值仍为 legacy（供隔离/回滚环境）；legacy 接回须经 D11 三步程序与 `state/rollback_manifest.json` 双向校验，禁止陈旧账户直连。下列示例用于隔离环境或显式覆盖：

```bash
python scripts/paper_trade.py run --trade-date 20261030 --engine legacy
python scripts/paper_trade.py positions --trade-date 20261030 --engine v2
python scripts/paper_trade.py model-info --engine v2
python scripts/paper_trade.py adjust --engine v2 cash --trade-date 20261030 --set 500000
```

`v2` 的 adjust 修改权威 `engine_state.bin` 并重建旧布局投影，不直接改 `account.json`；从 v2 回到 legacy 必须经过 D11 回滚清单与后续运行标记校验。树莓派/SMB 只读仅发布完整代。真实 `config --engine v2` 还须 `--switch-evidence` 指向门 1/5/6/7 全过的收口证据，并在当时显式给出 `--authorize-production-switch`；**生产切换已于 2026-10-31 完成（commit `db6d03b`），该入口仅用于回滚后再次切换或隔离环境，勿在生产上重复执行。**

T5 工具默认 dry-run（只打印计划）。真实执行需逐日显式授权分钟数，并登记窗口级总授权：

```bash
# 0) 首跑前逐源缺口扫描（只读，不写数据、不驱动任何臂）
python scripts/v2_p4/t5_daily_driver.py --mode preflight \
  --start-date <扫描出的首缺日> --target-date <首跑日> \
  --evidence-dir data/frozen_reference/v2_p4/<RUN_ID> --execute

# 1) Q11-A：逐真实交易日恢复现役旧链（D4 → paper_trade.py run）
python scripts/v2_p4/t5_daily_driver.py --mode restart --trade-date <真实交易日> \
  --run-id <RUN_ID> --evidence-dir data/frozen_reference/v2_p4/<RUN_ID> \
  --execute --authorized-minutes <本次授权> --authorized-total-minutes <窗口总授权>

# 2) 第 10 个有效日后冻结门 1（必须显式给恢复台账；台账 >10 条时另给 --restart-segment-end）
python scripts/v2_p4/t5_gate1_identity.py \
  --evidence-dir data/frozen_reference/v2_p4/<RUN_ID> --model-version <现役模型> \
  --restart-ledger data/frozen_reference/v2_p4/<RUN_ID>/q11a_restart_ledger.json --freeze

# 2b) Q11-SIM 等效（仅用户明确批准时；历史 10 日、隔离账户、逐日独立进程）
python scripts/v2_p4/t5_q11_sim.py --mode prepare --source-paper data/paper \
  --root data/paper_q11sim/<RUN_ID>/paper \
  --evidence-dir data/frozen_reference/v2_p4/<RUN_ID> --start-date <历史起始日> --days 10
python scripts/v2_p4/t5_q11_sim.py --mode run-all --authorized-by-user \
  --root data/paper_q11sim/<RUN_ID>/paper \
  --evidence-dir data/frozen_reference/v2_p4/<RUN_ID> --start-date <历史起始日> --days 10
python scripts/v2_p4/t5_q11_sim.py --mode finalize --root data/paper_q11sim/<RUN_ID>/paper \
  --evidence-dir data/frozen_reference/v2_p4/<RUN_ID> --start-date <历史起始日> --days 10
python scripts/v2_p4/t5_gate1_identity.py --evidence-dir data/frozen_reference/v2_p4/<RUN_ID> \
  --paper-root data/paper_q11sim/<RUN_ID>/paper --model-version <现役模型> \
  --restart-ledger data/frozen_reference/v2_p4/<RUN_ID>/q11a_sim_ledger.json \
  --q11-mode sim \
  --q11-sim-authorization data/frozen_reference/v2_p4/<RUN_ID>/q11a_sim_authorization.json \
  --freeze

# 2c) SHADOW_SIM/SLA_SIM（仅用户明确批准时；10 个历史交易日三臂 + 实测墙钟）
python scripts/v2_p4/t5_shadow_sim.py --mode prepare \
  --gate1 <门1 verification_summary.json> --frozen-paper <Q11-SIM 冻结纸面根> \
  --work-root data/paper_shadow/<RUN_ID>/arms \
  --evidence-dir data/frozen_reference/v2_p4/<RUN_ID> \
  --start-date <历史起始日> --days 10
python scripts/v2_p4/t5_shadow_sim.py --mode run-all --authorized-by-user \
  --gate1 <门1 verification_summary.json> --frozen-paper <Q11-SIM 冻结纸面根> \
  --work-root data/paper_shadow/<RUN_ID>/arms \
  --evidence-dir data/frozen_reference/v2_p4/<RUN_ID> \
  --start-date <历史起始日> --days 10
python scripts/v2_p4/t5_shadow_sim.py --mode finalize \
  --gate1 <门1 verification_summary.json> --frozen-paper <Q11-SIM 冻结纸面根> \
  --work-root data/paper_shadow/<RUN_ID>/arms \
  --evidence-dir data/frozen_reference/v2_p4/<RUN_ID> \
  --start-date <历史起始日> --days 10

# 3) 影子期逐真实交易日：三臂 + 现役账户同频推进（Q11-d）
python scripts/v2_p4/t5_daily_driver.py --mode shadow --trade-date <真实交易日> \
  --run-id <RUN_ID> --evidence-dir data/frozen_reference/v2_p4/<RUN_ID> \
  --work-root data/paper_shadow/<RUN_ID>/arms \
  --gate1 data/frozen_reference/v2_p4/<RUN_ID>/verification_summary.json \
  --active-production --execute \
  --authorized-minutes <本次授权> --authorized-total-minutes <窗口总授权>
```

口径与边界：

- `restart` 只把「D4 未降级 + 账户已推进 + 真实日历日 + run_completed」的日子计入累计台账；失败日
  写入 `sla/restart/<日>.json` 与 `q11a_restart_attempts.json` 保留现场，不计数、不补填。
- 影子模式未加 `--active-production` 时按**演练**处理：记录写入 `sla/drill/`，不计入门 5/6 窗口。
- 同一真实交易日只允许由本驱动执行一次（D4 → 现役臂 → neutral → v2 → 比较/登记）；不得用计划任务、
  Bot 或 CLI 并发补数据或驱动同一账户。现役臂已成功而影子失败的当日保留事实并按系统原因重计窗口，
  不回滚已发生成交。
- D4 步骤落盘为 SLA 字段（`ensure_raw/ensure_clean/cs_infer/panel_label_refresh/readiness/arms`），
  `snapshot` 单列前置步骤；重试/缺口/降级取自协调者真实产物，产物缺失按降级处理。
- 数据水位只到**前一交易日**：当日数据不得采集（例如 20261030 允许的最高水位为 20261029）；D4 的
  `--target-date` 必须按此规则设置，盘中未发布分区不得强行写入。
- Q11-SIM / SHADOW_SIM / SLA-SIM 只在用户明确批准时使用：分别替代旧链连续 10 日重启、
  10 日三臂影子与逐日墙钟，均不覆盖 live 网络/实时钟证据，也不能替代门 7 真实演练、
  门 8 全量/收口或生产切换授权。
- TuShare/AKShare 抓数时必须临时关闭代理；`download_raw` 与 `daily_data_coordinator` 默认清除
  `HTTP(S)_PROXY/ALL_PROXY` 并设置 `NO_PROXY=*`，如需保留代理可显式设 `LAZYBULL_DOWNLOAD_BYPASS_PROXY=0`。
- 三臂首次初始化由 `t5_three_arm.prepare_three_arms` 完成：legacy 两臂复制冻结源，v2 臂复用
  `t5_migration.migrate_isolated`（staging → 校验 → 发布），初始化记录见
  `data/paper_shadow/<RUN_ID>/arms/arms_initialized.json`。已初始化后只核对身份与完整代，不重迁/不重锚。
- gate 7 真实演练取证：`python scripts/v2_p4/t5_gate7_rehearsal.py --mode real --checklist <清单> --evidence-dir <证据根>`，
  操作清单与逐项要求见 [门 7 真实演练手册](docs/guide/t5_gate7_real_rehearsal_manual.md)；合成结果仍为 `SYNTHETIC_PASS`，
  不能升级为正式通过。
- 收口汇总只读本次实测证据，缺证据标 `MISSING` 并计入阻塞原因：

```bash
python scripts/v2_p4/t5_closeout.py --evidence-dir data/frozen_reference/v2_p4/<RUN_ID> \
  --run-id <RUN_ID> --runs-batch data/runs/<paper_shadow 批次> \
  --focused-tests-report <本次聚焦测试 JSON> --full-tests-report <本次全量 pytest JSON> \
  --complexity-report <本次复杂度 JSON>
```

当前 T5 工具链已修通（R1~R6），D4 已追平到允许水位 20261029；20261029 就绪门因 Q3
`free_cashflow` 全空导致 FCF 覆盖降至 31.8% 而失败。Q11-A、真实影子/SLA 与门 7 真实演练
仍未产生真实窗口；在这些门与用户当时显式授权齐备前，不得把生产配置切换为 v2。执行状态见
P4 实施规划 §24.10/§24.11/§24.12。

详见 [纸面交易使用指南](docs/guide/paper_trading_guide.md)

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

# 使用股东增减持因子（**运行时派生**，无需重建 cs_train；需先下载 stk_holdertrade 年分区）
python scripts/train_ml_model.py --start-date 20230101 --end-date 20231231 \
  --enable-holdertrade-features

# 使用股票回购因子（**运行时派生**，无需重建 cs_train；需先下载 repurchase 年分区）
python scripts/train_ml_model.py --start-date 20230101 --end-date 20231231 \
  --enable-repurchase-features

# 使用十大流通股东因子（**运行时派生**，无需重建 cs_train；需先下载 top10_floatholders 年分区）
python scripts/train_ml_model.py --start-date 20230101 --end-date 20231231 \
  --enable-top10fh-features

# 步骤2: 使用 ML 模型运行回测（使用新的默认值）
# 注意：scripts/run_ml_backtest.py 已删除，回测已并入 walk_forward 滚动回测，
# 或经 src.lazybull.common.backtest_runtime 工厂驱动 BacktestEngineML

# 选股域（单一来源 universe/domains.py；--stock-domain 为超参签名维度）：
#   main=主板50-1500亿（默认，生产现状）| main_small=主板25-1500亿（训练池同 main 可 skip 复用）
#   main_gem=主板+创业板≥50亿（训练池扩展，必须全折重训）
python scripts/walk_forward.py --split-count 14 --final-date 20260105 --stock-domain main_small

# 批量运行最小因子实验（共同基线、历史股息率、两对现金流）
powershell -ExecutionPolicy Bypass -File .\scripts\batch\batch_walk_forward.ps1
```

#### 期末异常亏损风控模型（terminal_loss）

预测持仓在剩余持有期期末发生波动标准化异常亏损的概率，用于到期前风险退出
（方案见 `docs/plans/terminal_loss/terminal_loss_risk_model_plan.md`）：

```bash
# 第一阶段：标签与离线模型（独立标签，不写回 cs_train/cs_infer）
# 三段：Train（拟合）| Val（早停）| ES（评估/门禁），逐段按 label_end_date 隔离；
# 早停段与评估段必须分离（v0.109.0）：早停只用 Val 段，ES 段只用于概率质量报告
# 与门禁评估——把门禁指标算在早停选择段上会带乐观偏差（旧产物签名 valm=0，
# 与新结果不可并组比较）
python scripts/train_terminal_risk_model.py \
    --start-date 20210104 --end-date 20260731 \
    --train-start 20210104 --train-end 20241231 \
    --val-start 20250102 --val-end 20250630 \
    --es-start 20250701 --es-end 20251231

# 输出：始终经 ModelRegistry 版本化保存（每次训练注册新版本 v{N}，不覆盖历史）：
# 模型 artifact、v{N}_report.json 概率质量报告、v{N}_calibration_by_h_sigma.csv
# 校准表、v{N}_label_coverage.csv 覆盖分布、v{N}_es_predictions.parquet ES 逐行预测
#（供门禁区间重采样，--no-es-predictions 可跳过）；--fixed-name 只额外写一套固定名
# 别名供既有工具读取（别名可被覆盖，版本历史才是唯一副本）
# 报告段：v{N}_report.json 同时含 es（评估段，门禁 lift 口径）与 val（早停段，
# 旁路审计；不参与 lift/pred_bias），train 段用于过拟合差距诊断
# pct 母截面：pct_* 分母取自 clean/daily 全量化重建的标签过滤前完整同日截面
#（cs_train 行已按 y_ret 标签有效性过滤，分母窄约 10%），四个基列与特征流水线
# 同一实现，并在交集上与 cs_train 同名列逐值校验（超容差即报错）
# 报告门禁：各 label_status 占比、缺失组 vs valid 组的代理画像与当日截面分位
# 条件事件率、endpoint_delayed 敏感性，按预登记阈值（缺失占比 >= 1%）判定
# 是否必须补做敏感性
# 内存与抽样：全网格标签按 50 交易日分块构建，每组 (股票,日) 抽 2 个期限、
# 每 3 个交易日取 1（--h-per-group / --every-n-days 可调，参数落入元数据）
# 训练超参：--max-depth / --learning-rate / --n-estimators / --early-stopping-rounds
#           / --subsample / --colsample-bytree / --reg-lambda 可透传
#（min_child_weight 与 scale_pos_weight 为正则尺度策略 A 设计不变量，不暴露）
# 训练设备默认 cuda（与主模型一致），--device cpu 可切换
# 特征集（--feature-set，v0.111.0/v0.113.0）：full 33 列冻结清单（默认）；core =
# 波动率/尺度状态 + 期限 10 列（标签已按 σ√h 归一化，可学截面信息集中于此；
# 单折实测 daynorm 1.119→1.392）；core_state = core + 风格/状态轴 12 列
#（v0.113.0：log_total_mv + pledge_ratio_decayed）——**已于 2026-09-14 8 折实测
# 否定，不得默认启用**（三判据点估计 7/8 折下降、2024H1 区间缺口从 0.0006
# 扩到 0.0444；机制是 2 列扰动改变早停轨迹），保留仅供复现；ic_admit = 训练
# 段逐日截面 Spearman IC 降序 top-k（--ic-top-k 默认 8，只抽日不抽行，必含
# 期限与 σ；选中列与 IC 值随折落盘）
# 训练目标（--objective，v0.111.0）：binary 输出即概率（默认）；rank_pairwise =
# rank:pairwise + qid=trade_date（只学当日截面排序），再用 Val 段 isotonic 校准
# 映射回概率（校准器随 artifact 落盘，缺校准器预测直接报错）；校准档位内按原始
# 分数恢复严格序（v0.112.2，否则 isotonic 的大样本压缩会把日内排序压成并列值）；训练完成后模型
# 物理裁剪到停点树（v0.112.1），默认预测即停点模型
# 早停指标（--eval-metric，默认按目标解析：binary→logloss、rank→auc）：
# logloss（概率校准）/ rank_ic_daily（逐日截面 Spearman，仅 binary）/ auc
#（池化 AUC，仅 rank；自实现回调逐树增量评估，v0.112.1 起不再用 XGBoost
# 内置 ranking auc——其按 query group 做 O(n²) 成对展开会超 int32 上限；与
# 门禁第三判据 auc_lift 同向且基准率不变）/ ndcg（仅 rank，列表口径且 Val 上
# 极易饱和，仅供对照）。非法组合由训练入口报错
# 特征集与目标、早停口径都是签名维度（fs= / obj= / em=），禁止混组比较

# 滚动 Walk-forward（14 折研究型 WF：与选股模型 OOS 折**逐一对齐**，覆盖 2018-11-26~2025-12-12）
powershell -ExecutionPolicy Bypass -File .\scripts\batch\batch_terminal_risk_wf.ps1
# 完成后自动汇总：data/walk_forward/terminal_risk_wf_oos14/summary.csv
# 折表：OOS00_201811 … OOS13_202506（ES = 选股 WF 的 test 窗，末端取 min(bt_end, 下一窗首日−1)；
#   Train = ValStart−6 年、Val = ES 前 6 个月；旧 8 折日历半年集仍在 terminal_risk_wf）
# 首轮实测（2026-09-20）：14/14 折有 lift，tuning_score 1.233，但 lift_min = 1.077（OOS00）
#   < 1.1 ⇒ 门禁不通过；该折处理见 R-006（政策默认转正后为不裁剪 20181126，
#   如需跳过弱点折可显式 --policy-coverage-start 20190527）
# 门禁：跨折 lift = ES PR-AUC / ES 事件率，组内最小值 >= 1.1 才建议进入第二阶段
# 调参指标（超参消融对比用）：折目录按 _d*/_lr* 后缀 × 折 meta 超参签名分组，
# 每超参组一行 → tuning_scores.csv；台账 tuning_history.csv 按组追加（跨 batch
# 纵向对比，--no-history 跳过）
# 调参分 tuning_score = 0.5×组内lift几何均值 + 0.5×组内lift最小值（绝对量纲，
# 跨 batch 直接可比：几何均值代表平均排序能力，最小值代表最差折稳健性）
# 历史最优自动比较：汇总末尾按超参签名聚合台账（含当次），醒目打印历史最优
# 超参签名与当次是否刷新；超参身份以折 meta（train/label/sampling 配置）为
# 权威——同后缀不同批次超参（如 baseline 从 depth=3 改跑 depth=4）自动拆组

# 门禁区间重判（点估计余量薄时必看：判断"通过"是否只是超参选择的结果）
python scripts/analyze_terminal_risk_gate.py --wf-root data\walk_forward\terminal_risk_wf
# 双判据并列（v0.110.0，--score-mode {both,raw,daynorm,dayauc,all}，默认 both）：
# raw = p_loss（跨日期水平对齐 + 当日截面排序混合），daynorm = p_loss_daypct
#（当日截面百分位，只反映截面排序）；两口径共用阈值 1.1 且都必须"逐折区间下限 >
# 阈值"才能算通过，只报一个口径属口径选择偏差（实测失败折不重叠）
# 第三判据 dayauc（v0.112.0）：daynorm 分数上的 auc_lift = 2×AUC（随机 = 1.0，共用
# 阈值）。lift = PR-AUC/事件率 带基准率压缩（完美排序上限 = 1/事件率），实测
# 2024H1 事件率 21.7%（最高）而日内 RankIC 中游、daynorm lift 全场最低；dayauc
# 只作并列交叉验证，不得单独宣布通过或绕过两个 lift 判据；--score-mode all 出三判据
# 折级口径：8 折 = 8 个独立制度，输出折间分布（min/median/max/std）、达标折占比、
# 均值 lift 的 90% 自举区间；产物 gate_ci.csv（每行 = 判据，含 metric 列）
# 注意折级自举抽不到比观测最小值更差的折，"最差折是否真高于阈值"须看逐折分块区间
# 逐折口径（需 ES 逐行预测，训练已默认落盘）：按连续交易日分块做 moving-block
# bootstrap，块长默认 5/10/20 日（方案第 6 节的 40 日块是组合级多年 OOS 口径，
# ES 段只有几十个交易日，用 40 日会使块数不足甚至退化为原样本）；
# 产物 gate_ci_block.csv（每行 = 折 × 块长 × 判据）
# 提速（v0.112.0）：--block-jobs N 按（折 × 判据）并行，与串行逐位等价
#（实测单臂串行 ≈30 分钟 → 并行几分钟），如：--score-mode all --block-jobs 8

# 政策旁路（P2-1，v0.114.0）：持仓快照 + 离线打分（全部中文表头）
# 1) 持仓快照：跑一次带 OOS 回测的 walk_forward 即自动导出（每 split 一份）
#    data/walk_forward/raw/walk_forward_持仓快照_{wf_run_id}_splitNN.csv
#    字段：运行标识/折序号/模型版本 + 日期/股票代码/持仓股数/持仓市值/持仓权重/
#    组合总值/买入日/信号日/持有交易日数/到期执行日/剩余持有交易日
#    只读旁路：开关对成交与净值逐位一致；到期执行日按标准持有期推算，
#    超出回测窗口记空值（动态延期不在本表口径内）
# 2) 离线打分（按"该日期时 Train/Val 都已结束"的折模型，无前视）
#    推荐用批量入口：按 WF batch（data/walk_forward/batches/*）选择快照打分
powershell -ExecutionPolicy Bypass -File .\scripts\batch\batch_policy_sidecar.ps1 -ListBatches
powershell -ExecutionPolicy Bypass -File .\scripts\batch\batch_policy_sidecar.ps1 `
    -Batch wf_batch_20260914_081944 -Arms _d5_v6m_fscore -PHi 0.90,0.95,0.99 -PAbs 0.05,0.10,0.15,0.20
#    -Batch latest（默认）= 自动选最新含快照的 batch；产物落 <batch>\policy_sidecar<arm>\
#    也可直接用 Python 入口（默认扫 data/walk_forward/raw 下全部快照）：
python scripts/analyze_policy_sidecar.py --arm _d5_v6m_fscore
# 产物（中文表头，utf-8-sig 可直接 Excel 打开）：
#   {risk-root}/archives/policy_sidecar{arm}/风险台账.csv
#     （风险概率 / 当日截面分位 / 日波动率 / 持有期预期波动 / 市场波动状态 /
#       事后实际收益 / 是否异常亏损 / 事后可避免损失 / 主策略最后持仓日）
#   触发清单.csv（双条件命中：正确拦截 / 误杀；可选漏报）
#   阈值扫描.csv（截面分位阈值 × 绝对概率阈值 → 拦截率/误杀率/漏报率/收益代价）
# 常用参数：--p-hi 0.90 0.95 0.99 --p-abs 0.05 0.10 0.15 0.20
#           --no-daypct（跳过截面分位，阈值扫描随之不可用）
# 口径硬约束：标签经 build_terminal_loss_labels 按折的 k/h_max 关联；
# σ 用 compute_sigma_daily_panel；当日截面分位在完整同日截面内计算；
# 剩余持有交易日超出 [1, h_max] 直接报错；full 特征集的 pct_* 不在 cs_train
# 内 → 明确报错（请用 core / core_state）

# 暴露门控 E2（P2-2 一阶筛选，v0.116.0）：条件化"降暴露"而非个股退出
# 三臂并列：E2(regime×分数) / 对照A(纯regime) / 对照B(纯模型分数)
python scripts/calibrate_exposure_gate.py `
    --batch wf_batch_20260914_084247 `
    --calibration-start 20220701 --calibration-end 20231229 `
    --eval-start 20240101 --eval-end 20251204 `
    --arms combined score regime --de-exposure 0.5 --cost-bps 15
# 滚动分位口径（阈值随模型水平漂移自归一，只用 [t-W, t-1] 窗口；两种口径不得混组比较）
python scripts/calibrate_exposure_gate.py `
    --batch wf_batch_20260914_084247 --threshold-mode rolling --window-days 250 `
    --eval-start 20240101 --eval-end 20251204 --arms combined score regime
# 产物（中文表头，落 <batch>\暴露门控E2[_滚动{W}日]\）：
#   暴露门控校准.csv     每个臂的冻结阈值（固定口径）或口径说明行（滚动口径）
#   暴露门控逐日判定.csv 一行 = 某臂某交易日（阈值口径/窗口/市场波动阈值/得分阈值/是否触发/暴露系数）
#   暴露门控评估.csv     一行 = 臂 × 口径 × 分组（收益项/成本项/净增量，含两条对照臂）
#   暴露门控评估_按折.csv 同指标按折拆开（稳健性）
# 口径硬约束：阈值只在"校准段"拟合，与评估段重叠直接报错；非 valid 标签行剔除并告警；
# 同日"市场波动状态"不唯一直接报错；层内校准日数低于下限拒绝校准。
# 判据：只看"降低平均亏损日频率与幅度"，不得以 MaxDD/尾部损失为判据；
# 结果是一阶代理（非 NAV），组合级净值必须由 P2-3 shadow 模式回测给出。
```

### 暴露门控 E2 的三批实测（2026-09-14，`_d5_v6m_fscore`）

| 批次（形态） | 臂 | 触发占比 | 触发日平均加权收益 | 未触发日 | 触发日事件率 | 未触发日 | 净增量（日均口径） |
|---|---|---|---|---|---|---|---|
| top_n=10 | **E2** | 33.8% | **−1.39%** | +2.11% | 26.6% | 12.8% | **+0.00231** |
| top_n=10 | 对照B 纯分数 | 40.8% | +0.15% | +1.46% | 23.7% | 13.2% | −0.00037 |
| top_n=10 | 对照A 纯 regime | 75.9% | +0.68% | +1.71% | 16.3% | 21.1% | −0.00261 |
| top_n=15 | **E2** | 34.3% | **−0.42%** | +1.58% | 20.8% | 15.8% | **+0.00067** |
| top_n=15 | 对照B 纯分数 | 38.8% | +0.79% | +0.96% | 19.5% | 16.3% | −0.00157 |
| top_n=15 | 对照A 纯 regime | 75.8% | +0.78% | +1.26% | 15.6% | 23.6% | −0.00299 |
| top_n=20 | **E2** | 33.1% | **−0.62%** | +1.90% | 22.5% | 14.1% | **+0.00096** |
| top_n=20 | 对照B 纯分数 | 41.7% | +0.83% | +1.24% | 20.0% | 14.6% | −0.00179 |
| top_n=20 | 对照A 纯 regime | 75.8% | +0.87% | +1.70% | 14.9% | 23.0% | −0.00332 |

要点：**只有 E2 三批净增量全为正，两条对照臂全为负**（纯 regime 甚至触发在更安全的日子）；
保守的"首触日"口径下 E2 三批同样为正（+0.00008 / +0.00021 / +0.00015）。成本敏感：
单次首触事件的毛收益 ≈ 0.5×触发日损失，往返成本按 2×15bp×λ 计，二者同量级 → P2-3 影子
运行必须先做成本敏感性。风险登记见 `docs/terminal_loss_risk_register.md` 的 R-002 / R-003。

**滚动分位口径（`--threshold-mode rolling`）**：触发占比 33%→19–21%，触发日事件率 21–27%→**30–36%**，
触发日平均加权收益 −0.4~−1.4%→**−1.8~−2.3%**（信号更锐利，两条对照臂仍为负）。

**净值层面（近似，非影子回测）**：固定口径 Δ收益 +0.6 / −4.7 / +1.4pp、ΔMaxDD 全为 0；
滚动口径 Δ收益 +21.2 / +24.1 / +22.1pp（乐观 T+1）或 **+9.8 / +12.0 / +10.3pp**（保守 T+2，
承担开盘跳空）、ΔMaxDD 0~+1.3pp。乐观口径含**跳空伪影**（降暴露覆盖了 2025-04-07 −8.8% 等最差日，
而 T+1 开盘卖出躲不过当日跳空）→ **幅度不可当预期收益**，只能读方向；最终结论必须由 P2-3
影子回测给出。见 R-004。

### 暴露覆盖影子回测（P2-3，引擎侧真实成交）

把 E2 的逐日暴露系数表（`--export-table` 产物，两列 CSV：`日期, 暴露系数`；现已同时写 `<table>.csv.meta.json` 记录策略口径与导出源）接入回测引擎，
即可在同批模型、同期间的 14 折 OOS 上真实对比（约 10 分钟/臂）：

```powershell
# 批量滚动训练脚本支持命令行覆盖系数表（留空 = 生产默认，不启用且逐位一致）
powershell -ExecutionPolicy Bypass -File .\scripts\batch\batch_walk_forward.ps1 `
    -ExposureTable "temp\exposure_e2_rolling250.csv" -SkipCompare
```

引擎侧只做两件事（`src/lazybull/backtest/`）：

1. **主动减仓**（`exposure_trim.py`）：**每日**判定 `持仓市值 > λ×组合总值 + 3% 容差`，
   超配则 T0 生成卖单、**T+1 开盘**按同一比例对**全部持仓**部分卖出（不做个股选择；
   剩余不足一手整仓卖出）；已在到期/止损整仓队列的股票跳过；停牌/跌停跳过并由次日重判。
2. **暂停加仓**（`exposure_override.py`）：把调仓与补齐两条买入路径的预算基数乘上 λ，
   避免减仓后在缺口日买回。

配套两条回补（均默认关、逐位一致）：

- **对称回补**（`exposure_replenish.py`，P2-4）：λ 回满且存在未回补减仓额时，
  T+1 按市值比例把持仓买回（`--exposure-replenish`）。
- **建仓折扣回补**（A3 v2，v0.127.14）：λ<1 信号日的买入预算折扣按实际成交额记入回补释放额
  （`成交额 × (1/λ − 1)`），λ 恢复后由同一回补机制买回（`--exposure-budget-discount-replenish`，
  必须与回补同用）——否则建仓恰逢政策期的批次会半额挂到下一次调仓轮换。

未设置系数表时两条路径都不生效（成交与净值与改动前逐位一致）。

**四臂实测（2026-09-14，14 折 OOS，窗口 20240102~20251204）**

| 臂 | 窗口累计收益 | MaxDD | Sharpe | Δ收益 | ΔMaxDD | ΔSharpe | 减仓笔数/金额 |
|---|---|---|---|---|---|---|---|
| 基线（不启用，全 1 表逐位一致） | 0.3325 | −0.2695 | 0.6094 | — | — | — | 0 |
| **E2-滚动250** | **0.5055** | −0.2593 | **0.9853** | **+0.1730** | **+0.0102** | **+0.3759** | 156 笔 / 384.1 万 |
| 对照A 纯 regime-滚动250 | 0.3764 | **−0.2505** | 0.8359 | +0.0440 | +0.0190 | +0.2265 | 177 笔 / 401.3 万 |
| 对照B 纯分数-滚动250 | 0.2931 | −0.2106 | 0.6766 | −0.0394 | +0.0588 | +0.0672 | 135 笔 / 325.3 万 |

预登记判据（Δ收益>0 且逐折 MaxDD 改善 ≥2/3 且 成本后 Sharpe 不下降）**通过**，
且窗口外 split 0~9 三臂与基线**逐位一致**。**但必须一起读**：① 68% 的收益来自"信号期外滞留"
（减仓现金等下一次调仓才回补，+7.14pp / 总 +10.47pp）；② 收益 ≈95% 集中在 2024-01/02 与 2025-04/10
（2024-11 反为 −3.32pp）；③ 逐折收益仅 2/4 为正，跨折稳健的是**回撤与波动**。
三条未验证项（参数敏感性 / 对称回补 / 事件依赖）与完整归因见 `docs/terminal_loss_risk_register.md` R-004 §8。

### 暴露政策在线现算（P2-5，v0.127.4：不再需要导出系数表）

系数表与持仓路径绑定（换任何影响持仓的配置都要重算），所以现在**直接把策略写成参数**，
由引擎逐日现算 λ_t（`f(折模型, 判定日持仓, 当日截面)`，与离线判定**共用同一套实现**）：

```powershell
# 默认臂已就位（脚本配置区 $exposure_arm_list = e2online_r，2026-09-20 转正：回补开），直接跑即可：
# （P1.5 已裁决：该臂于 v2 切换日退役，迁移期默认不变——方案 F8 / docs/reports/p15_policy_layer_adjudication_20261001.md）
#   $policy_model_root = "data\walk_forward\terminal_risk_wf_oos14"   # 与选股 OOS 对齐的 14 折集
#   $policy_arm_suffix = "_v6m_fscore"
#   $exposure_arm_list = @( [PSCustomObject]@{ Name = "e2online_r";
#       Policy = "arm=combined,mode=rolling,window=250,regime_q=0.75,score_q=0.5,lambda=0.5";
#       Table = ""; Replenish = $true; Tolerance = 0; StopLoss = $false } )
powershell -ExecutionPolicy Bypass -File .\scripts\batch\batch_walk_forward.ps1 -SkipCompare

# 需要基线对照（不启用政策、与历史基线逐位一致）时：命令行一键切回 neutral 臂
powershell -ExecutionPolicy Bypass -File .\scripts\batch\batch_walk_forward.ps1 -NoExposure -SkipCompare

# 多臂对照：在配置区清单里并列多条 [PSCustomObject]（每条跑一遍，臂名自动拼进 batch label）
```

- `Policy`（在线）与 `Table`（原系数表，已退化为缓存/回放）**互斥**；命令行同名参数退化为
  “单臂覆盖”（给出任一暴露参数即忽略清单，不叠加）；`-NoExposure` = 命令行一键基线臂；
  只给 `-ExposureReplenish`/`-TrimTolerance` 而不给政策源会**直接报错**（禁止静默忽略）；
- λ_t 用**判定时点持仓**（当日执行前），逐日归档 `raw/policy_lambda_<run_id>.csv`（含日级面板列）
  + `.json`（策略口径/模型来源/指纹/统计），便于审计与回放；
- 默认覆盖起点 `20181126`（= 不裁剪全 14 折；只抑制动作、不抑制阈值历史的语义不变）；
  跳过门禁未达标的 OOS00 折（R-006）显式填 `20190527`；与历史扫描同覆盖用 `--policy-coverage-start 20240102`。

**在线口径复核（2026-09-19）**：崩盘月贡献逐值可复现（2024-01 +6.00pp / 2024-02 +4.04pp 与冻结表臂相同），
但**主判据不可复现**——冻结表臂 ΔMaxDD +1.02pp vs 在线臂（同覆盖）**−0.15pp**，
差异归因于 5/446 日 λ 翻转改写的 2024-06/07 路径 ⇒ **“以回撤改善为由启用”不被支持**，
政策的默认启用改由下述**回补转正**承载；详情与两个在线通路缺陷的修复见 `docs/reports/terminal_loss_policy_online_result.md`。

**对称回补转正（2026-09-20，v0.127.7）**：默认臂改为 `e2online_r`（E2-滚动250、λ=0.5、在线、**回补开**、
不裁剪覆盖）。依据 = 语义自洽（λ 回满后应回满仓；现金拖累是 P2-4 已登记缺陷）+ 单臂实测
（R vs A：ΔCAGR +1.20pp / ΔMaxDD +1.63pp / Δ总收益 +26.66pp，逐折年化改善 8/14；
16 个恢复窗口 11 正 / 5 负，方向与窗口内市场涨跌一致，最大负项 split2 2020-03 −3.42pp）。
**可外推性声明**：收益增量“方向偏正、量级不可承诺”（机制期望 ≈0.5pp/年；单窗口 ±3pp 事件会再现）；
**回撤改善不可外推**（−20.25%→−18.62% 由 split6 单折驱动）；**转正 ≠ 收益承诺**。
**P1.5 迁移前裁决（2026-10-01，方案 F8）：e2online_r 于 v2 切换日退役**——迁移期默认臂不变、
直接跑即可；裁决报告 `docs/reports/p15_policy_layer_adjudication_20261001.md`。
详见 `docs/terminal_loss_risk_register.md` R-007 §6/§7。

**纸面交易接线（v0.127.10）**：`paper_trade.py` / `bot_service.py`（共享 `execute_trade_workflow`）
已接同一在线判定链路——判定日 λ 在 **T1 执行前**评估，减仓/回补指令**追加到既有 T+1 指令文件**，
复用 T0/T1 链路（不新增执行路径）；回补加仓保留原买入日；状态落
`data/paper/state/exposure_policy.json`（换策略串/折后缀自动清空重算；**指纹机器无关、不含模型路径**，
模型内容一致性由预热文件的折模型摘要校验）。配置入口：
`data/paper/config.yaml` 的 `exposure` 区块（默认 `exposure_policy: null` = 关闭且零副作用；
即从零生成配置文件时该区块也会自动出现，v0.127.11 起）。

```powershell
# 启用示例（在线现算 + 对称回补；与回测默认臂同口径）
python scripts\paper_trade.py config `
  --exposure-policy "arm=combined,mode=rolling,window=250,regime_q=0.75,score_q=0.5,lambda=0.5" `
  --policy-model-root "data\walk_forward\terminal_risk_wf_oos14" `
  --policy-arm-suffix "_v6m_fscore" `
  --exposure-replenish

# 关闭（恢复默认；回补开关一并写回 false）
python scripts\paper_trade.py config --no-exposure-replenish --exposure-policy ""
```

已知差异（登记）：补齐（pending_buys）路径的买入缩放取**执行日** λ（回测取信号日，≤1 个交易日）；
数据类异常（当日特征构建失败/打分输入缺失）告警 + 降级顺延，非静默。

**实盘模式与预热面板（v0.127.12）**：纸面以 `serving` 模式运行——终损折模型只要
“训练/早停段已结束”（`date > val_end`）就可持续使用，**不再受折 ES 窗口上界限制**
（2026 年及以后用最新可用折，如 `OOS13_202506`）；也可在 `exposure` 区块用 `policy_fold`
固定指定折（默认 `null` = 自动最新可用折）。

为避免启动初期“阈值窗口不足（< 60 交易日）不判定”，用**回测持仓快照**生成**预热面板**——
预热文件是**自包含的 1 MB 级 JSON**（不引用任何快照文件），**纸面机不需要任何回测产物**，
在生成端产出后拷贝即可：

```powershell
# ① 生成端（有 data\walk_forward\batches\* 回测产物的机器）——先列候选，显式选与纸面策略参数一致的批
py -3 scripts\prepare_paper_exposure_warmup.py --list-batches
py -3 scripts\prepare_paper_exposure_warmup.py `
  --snapshot-dir "data\walk_forward\batches\<所选批次>\raw"
# 默认输出 <policy_model_root>/paper_warmup/state.json；默认回看 270 交易日；
# 策略串/模型根/后缀默认从 data/paper/config.yaml 的 exposure 配置读取

# ② 纸面机：把 state.json 拷到 <policy_model_root>/paper_warmup/ 下
#    （或任意路径 + exposure.policy_warmup_file 指向）；纸面启动时仅当面板为空才恢复；
#    折模型重训后必须重新生成（文件内含折模型内容摘要，失配时纸面告警忽略）
```

批量脚本的 `factor_experiment_configs` 默认使用相同参数运行三组方案：不启用候选因子的
基线、仅保留 `dividend_yield_hist_12m` 的分红方案，以及仅保留 `fcf_yield` 和
`ocf_to_revenue`（均含 `_sz` 版）的现金流方案。三组方案复用相同的 OOS split 配置，
并通过因子开关、精简清单列及独立汇总文件区分；候选清单位于 `configs/factor_exclude_dividend_yield_only_v1.json` 与
`configs/factor_exclude_cashflow_keep_2pairs_v1.json`，不会覆盖生产排除清单。

Walk-forward 对比表的全周期 CAGR 按有效日收益区间进行几何年化；每个 split 重复的
起始净值点不计入交易日数，链式 Sharpe 同样只使用 split 内日收益。

**数据态血缘与跨数据态告警：** 每次 walk-forward 运行会采集当前 git 版本（含工作区
脏标记）与关键数据水位（raw/daily 等最新分区、cs_train 最新分区、dividend 覆盖状态
摘要），完整快照落盘为汇总同目录的 `data_state_{wf_run_id}.json`，摘要列
（数据态ID、Git版本、raw/daily水位等）并入 summary CSV 与对比表。同一配置在不同
数据态下复跑的结果不可比；`compare_walk_forward.py` 检测到对比表内存在多个数据态时
会输出显式告警，配置差异只在同一数据态内比较。

**折子集（省时消融）对比：** 消融实验若用 `walk_forward.py --selected-split-indices` 只跑部分折
以省时，必须与基线在**同一折子集**上比较：

```bash
python scripts/compare_wf_fold_subset.py \
    --baseline data/walk_forward/batches/wf_batch_<基线> \
    --arm data/walk_forward/batches/wf_batch_<臂> --splits 8-13
```

该工具只读既有 `chain_nav` / summary / 数据态快照，按折子集重算链式指标（子集净值先
归一化到子集起点，再复用 `v2/common/chain_metrics.py` 口径；`ml/walk_forward/chain_metrics.py` 为 re-export shim），输出 `折子集对比.csv`
（含 ΔCAGR / Δ最大回撤 / Δ夏普 与逐折同向数）与 `逐折对比.csv`（中文表头，默认落
`data/reports/wf_fold_subset/<时间戳>/`）。折集合、逐折窗口或数据态 ID 不一致会直接报错终止。

**日频信号级配对对比（筛选尺子）：** 链式净值口径的噪声带极宽——同配置**仅换随机种子**即
ΔCAGR −3.6pp，列集增删更到 −5.8pp 量级，而列级改动的预期效应只有 0~1pp，**在净值口径上判不出来**。
`walk_forward_topk_details_*.csv` 保留了逐日 Top-K 明细（`pred_score` + 真实持有期收益），
可把评估单位从「14 个折净值读数」提升到「14 折 × 约 1700 个交易日的**逐日配对读数**」：

```bash
# 标定噪声带：同配置仅换种子
python scripts/compare_wf_signal_metrics.py \
    --baseline data/walk_forward/batches/<基线> --arm data/walk_forward/batches/<换种子臂> \
    --baseline-label 基线 --arm-label 换种子B0

# 多个候选臂一次过尺子
python scripts/compare_wf_signal_metrics.py --baseline <基线> \
    --arm <臂A> --arm <臂B> --arm-label A --arm-label B
```

做法：两臂在**同一天**相减（市场共同冲击自动抵消），差值序列按**交易日分块**自举（块长默认
20/40/60 日，复用 `risk/terminal_loss/block_stats.py::paired_day_mean_ci`），输出
`日频信号对比.csv`（点估计 / 相对基线水平 / 95% 区间 / 正差值概率 / 折内同向折数）、
`逐折信号对比.csv` 与 `口径说明.md`，落 `data/reports/wf_signal_compare/<时间戳>/`。

**口径边界（不得省略）**：该口径是**信号级代理**，不含交易成本、调仓节奏、Kelly 仓位与路径效应，
**只能用于筛选**（"这个改动是否真的动了信号"）；最终裁决仍以链式净值判据为准（**ΔMaxDD 为主判据**），
且只对通过筛选的改动花 2 小时跑全量。首次标定（2026-09-17）见 `CHANGELOG.md` 0.122.0。

**v2 证据机器通路（P5a-1，v0.203.x）：** 回测/实验产物以 runs 契约 schema 为准
（`docs/contracts/runs_artifact_contract.md`），旧 walk-forward 批次经一次性转换器对齐，
读入桥对契约 §9 不变量做硬校验（列集合/指纹重算/数据态/跨折衔接/daily-trades 对账等），
信号层尺子固化为 `src/lazybull/v2/evidence/signal_ruler.py`（判据口径沿历史四轮：
点估计破 ±10bps 噪声带或相对降幅 ≥9% ⇒ 不过尺）。A3 验收（3 个已登记历史实验
转换 → 读入 → 链式重算与既有报表逐项一致）：

```bash
python scripts/v2_p5a1/verify_runs_recalc.py
```

产物 `data/reports/p5a1_a3_recalc_<日期>.{csv,json}`（校验表 + 报告）；转换器审计产物
`<out_root>/<batch_id>.convert_report.json`（行数校验 / 字段映射表 / 丢弃列清单 /
缺列登记 / 抽样 md5）。

**v2 证据机器·记忆（P5a-2，v0.203.5）：** 假设台账 `data/ledger/hypotheses.jsonl`
（10 族历史结论回填 + 占位 prereg，append-only）；入场点敏感度报告（§3.3 爬坡立项门输入；
最差 K 起点 + 中位/离散度口径，分位数精读禁止）：

```bash
python scripts/v2_p5a2/run_entry_sensitivity.py            # 入场点敏感度（B0/B1 冻结批）
python scripts/v2_p5a1/run_power_calibration.py --shift-grid 2 5 8 10 12 15 --out-name <名>  # 功效标定
#  --sections both|curve|consistency 分节输出；--out-name 防同日覆盖（台账引用产物）
```

**v2 数据底座（P1，v0.204.x）：** 单表 features 全历史回填 + labels + manifest 已落地并验收
（总验收报告 `docs/reports/v2_p1_final_report_20261006.md`，构建冻结
`docs/data/v2_p1_build_freeze.md`）。数据资产：`data/features/panel/`（热区
20250701~20260702，244 日 × 8 族）+ `data/features/panel_archive/`（冷区 2012-01~2025-06，
162 月 × 8 族）+ `data/labels/y_ret_{5,10,20}/`（3 表 × 3,518 日，全 sealed）+
`data/features/manifest.json`（411 列登记 + 分区指纹 + 依赖声明 + repairs 审计轨迹）；
冻结验收基准 `data/frozen_reference/v2_p1/`（3,518 分区；上述标签 3,518 日全 sealed 为
P1 验收时点口径，日更段追加后现状见下文 P3 前段落）。命令入口：

```bash
python scripts/v2_p1/backfill_panel.py               # 全历史回填（冷热分层）
python scripts/v2_p1/reconcile_panel.py              # 严格门 + 归因门对账（vs 冻结参照 / cs_train）
python scripts/v2_p1/verify_immutable_partitions.py  # 不可变分区指纹校验
python scripts/v2_p1/bench_panel_perf.py             # 性能闸门（全历史跨列族加载）
python scripts/v2_p1/shadow_path_acceptance.py       # 影子通路验收
```

**注意：P1 未切换生产链路**——现役训练/回测/纸面仍走 cs_train/cs_infer，panel 底座待
后续阶段裁决接入。

**v2 内核搬运（P2a，v0.205.0）：** 旧回测内核已纯搬运落位 `src/lazybull/v2/`
（`common/rules` 共享纯函数 + `core/{decision,accounting,execution,signal}` +
`hosts/backtest`；数值路径零重新设计、行为冻结），B0 冻结基线 14 折重放八道门全 PASS
（成交 3,214 行 9 列逐位一致、链式净值 1,714 行 max|Δ|=0.0；退役 exposure 政策族与
downside_penalty 挂载按裁决摘除，λ>0 显式 fail-fast）。实施规划
`docs/plans/v2/p2a_implementation_plan_20261006.md`（本地）。**注意：P2a 未切换生产链路**——
现役训练/回测/纸面仍走旧内核，v2 内核经 P2b/P3/P4 推进接入。

**v2 单袖合并器（P2b 交付，P4 T3 原生链接通）：** `core/decision/portfolio_merger.py`
提供 A1 单袖目标直通与 S6 单袖指令透传；混合袖输入显式 fail-fast。P4 T3 起由
`hosts/native_engine.py` 在宿主边界构造 `TargetPortfolio` / `Account` 并直接调用合并器，
旧 dict 适配器与 `merge_port` 已删除；`legacy_bypass` 仅保留给下列冻结回归边：

```bash
# 关闭臂（14 折）；预检可追加 --splits 0,1,2
python scripts/v2_p2b/replay_merger.py --mode legacy_bypass --batch-id <关闭批ID>

# 开启臂（独立运行态，baseline_ref 指向当前关闭批）
python scripts/v2_p2b/replay_merger.py --mode single_sleeve_passthrough \
  --baseline-ref <关闭批ID> --batch-id <开启批ID>

# 三条比较边；子集预检比对才追加 --allow-subset
python scripts/v2_p2b/compare_merger_batches.py \
  --closed-batch data/runs/<关闭批ID> \
  --enabled-batch data/runs/<开启批ID> \
  --input-identity-verified
```

`--input-identity-verified` 只允许在两臂均已由 `replay_merger` 完成运行前后完整
preflight 后使用；它只是向比对器显式声明门 1 已完成，**不是跳过或替代门 1 的豁免**。

**边界：仍未切换生产链路**——多袖净额化、虚拟账本、三恒等式与订单级真实多袖接线
归 P6.5；P4 只交付单袖原生边界，不宣称收益改善。

**v2 合成多袖状态信封与同代恢复（P6.5 T5）：** `v2/core/accounting/state_envelope.py`
提供**独立同代状态信封**（五项所有权：P4 基础 blob / 物理 lot 伴随视图 / 虚拟账 /
结算与估值身份 / schema 与同代标记），`state_owners.py` 负责五项投影与还原，
`envelope_store.py` 负责原子落盘与同代链（临时名 + 原子替换、禁止覆盖、崩溃现场隔离到
`crashed/`）；`v2/hosts/recovery_runtime.py` 与 `recovery_cases.py` 提供可恢复运行时与
五类恢复场景（多来源 lot / 中途提交点 / 恢复后再次内部转账 / 成交后水位 / 半提交代崩溃 /
恒等式破缺终态），比较口径为 float64 位模式 + 整数股 + 键序（零容差）。信封**不改** P4
`EngineState` 键集，且不进入生产默认路径：

```bash
# 恢复矩阵 + G5~G7 证据包（全量 pytest 未跑时包内标记为"复审态"，收口须传真实日志）
python scripts/v2_p65/run_t5_acceptance.py \
  --pytest-targeted logs/p65_t5_targeted.log \
  --complexity logs/p65_t5_complexity_src_scripts.log
```

**v2 P7 T0.5 治理落档（预检 + 预登记，状态 draft_pending_review）：** P7（首袖子立项 +
历史多袖复核）的 T0.5 交付三个只读/确定性入口，均不联网、不写生产根、不运行训练或推理：

```bash
# 1) 输入完整性预检（约 2~4 分钟）：14 折 test/bt 日期核对、模型与清单 SHA、逐折面板覆盖、
#    V20/V10 相位日历与末折 runoff、三源水位（fina 只读扫描推导）、三源事件分布与有效块计数；
#    交易日历先截断到共同数据截止，日历尾部不参与成熟判定
python scripts/v2_p7/precheck_t05_inputs.py \
  --out-dir data/paper_shadow/p7_testbed/p7_t05_20261111_r5/precheck

# 2) 崩盘窗规格冻结（沪深300 年内峰谷规则；首次联网取指数快照，规格已存在即拒绝覆盖）
python scripts/v2_p7/crash_spec.py \
  --basis market_index --rule year_peak_trough --index-code 000300.SH \
  --out data/paper_shadow/p7_testbed/p7_t05_20261111_r5/specs/crash_windows_v5.json

# 3) 治理工件生成（确定性，重复运行逐字节一致）：复合 prereg / R3 授权快照 / 预算台账 /
#    写白名单与禁改面 / H 报告 schema 与 kill criteria / 治理快照
python scripts/v2_p7/build_t05_governance.py \
  --root data/paper_shadow/p7_testbed/p7_t05_20261111_r5
```

退出码：`0` 通过 / `2` 输入验收门失败（含附件缺件或字节不符）/ `3` 核心格有效块不足
（Q5=A 回用户重裁）/ `4` 写域拒绝 / `5` 输入不可用；非零退出不写出任何工件。
本次结果：timeline 3637 日、块总数 121、有效块 forecast 121 / express 72 / fina 121，
右删失 75 / 0 / 1468；崩盘窗 2015-06-08~08-26、2018-01-24~12-27、2022-01-04~10-31。口径与契约文本见 `docs/plans/v2/p7_t05_contract_revisions_20261111.md`
与运行手册 `docs/plans/v2/p7_t05_runbook_20261111.md`；**边界：T0.5 未生效、T1~T6 未获授权，
本阶段不产生任何前向/日历结论，也不构成 B 区转正**。

**v2 P7 T1 · P-1 同域静态可执行性（候选线冻结判定）：** 用 T 日公开信息构造「1 事件股 + 5 对照股」
六股匹配组，判定六股能否按冻结的 T+1 开盘入场 / 到期收盘离场规则共同成交并完整持有：

```bash
# 正式全量（约 9 分钟）：全历史面板 + 三源全部事件 + 6 格 × 3 条互斥随机流 × 10000 次内层实现
# 产物原子发布到 data/frozen_reference/v2_p7/<run_id>/，禁止覆盖；exit 2 输入不可用 / 4 写域拒绝或已存在 / 5 计数不守恒
python scripts/v2_p7/t1/run_p1.py --run-id p7_t1_20261111_r1

# 冒烟降配（自动标 diagnostic_only，不产生三向结论）
python scripts/v2_p7/t1/run_p1.py --run-id smoke_x --out-root temp/p7_t1_smoke \
  --smoke-panel-start 20200101 --smoke-event-start 20200401 --limit-signal-days 40 --implementations 200
```

本次结果（`data/frozen_reference/v2_p7/p7_t1_20261111_r2/`，首轮评审 13 条整改后重取，
旧证据 `_r1` 只读保留）：意图域 246,257 事件 → 54,331 / 14,913 / 106,019 个六股组；
已解析 49,122 / 13,450 / 97,503，不可解析 4,255 / 1,174 / 7,349（7.83% / 7.87% / 6.93%，
主因是长期停牌导致事件协变量不可算，其次到期窗内无共同离场日），尾端右删失
140 / 0 / 1,479 单独登记不记 0；计数全格守恒、有效块 117 / 71 / 90、MDE = 4.0σ、
假通过率 1.43%~1.58% 均在 CP 上界内。按 prereg §6.2.4「任一核心格 `unresolved_count>0` ⇒ 冻结」，
裁决为 **`inconclusive_frozen`（不可估计/研究冻结）**：B0 复合条目不得进入 T2/T3，
冻结期间不得改口径重跑。详见 `docs/reports/p7_t1_p1_result_20261111.md`、
运行手册 `docs/plans/v2/p7_t1_runbook_20261111.md`、评审处置
`docs/review/v2_p7_t1/p7_t1_author_response_20261111.md`；**边界：不构成 B 区转正、生产多袖能力、
前向或日历验证通过，也不是「B0 无效」的经济结论**。

**v2 P7 T2.5 · A 逐日信号再生 / 相位目标 / 原始意图预检（机械线输入层）：** 14 折逐日分数与 Top-N、
20 日持期目标、V20/V10 相位日历、跨折连续持仓账与 runoff 输入清单，另做**原始意图级**预检
（S1~S5 分字段计数 + 额度缺口/召回/重叠/相关性，标 `estimated_intent_coverage=true`）：

```bash
# 正式全窗（约 8 分钟）：硬门全过才落盘；run_id 已存在即拒绝覆盖
python scripts/v2_p7/t25/run_t25.py --run-id <NEW_RUN_ID> \
  --counterpart-top-n 10 --counterpart-capital 350000 --verify-splits 0 6 13

# 分段验证（新 run）/ 续跑（仅限**未完成**的 run）/ 跳过锚批次计划复算
python scripts/v2_p7/t25/run_t25.py --splits 0 1 --run-id <SMOKE_ID>
python scripts/v2_p7/t25/run_t25.py --splits 2 --resume --run-id <UNFINISHED_RUN_ID>
python scripts/v2_p7/t25/run_t25.py --run-id <SMOKE_2> --no-anchor-plans

# 显式接受输入漂移（形如 path:sha256；也可用 P7_T25_ACCEPT_DRIFT 或 accept_input_drift.json）
python scripts/v2_p7/t25/run_t25.py --run-id <NEW_RUN_ID> \
  --accept-input-drift data/clean/stock_basic.parquet:<sha256>
```

`--resume` **只对未完成运行有效**：目标根一旦写出 `manifest.json` 即只读，续跑会被拒绝（改用新 `run_id`）。
`--counterpart-top-n` / `--counterpart-capital` 只改**构造对手袖**（v10）的深度与初资，基准袖（v20）恒为冻结值；
偏离项写进 manifest 的 `params.design_deviation`。

**状态：2026-11-12 复审 4 通过并收口**（四轮评审全部整改）。收口证据
`data/paper_shadow/p7_testbed/p7_t25_hk_20261112_r6/`（56 件、484.2s）——基准袖按冻结 **half_kelly**
计权重（信号日定权、执行日只用开盘已知量），原始意图 S1 **0** / S2 20 / S3 56、S4 695 / S5 695、S6 2,218；
**S1 无自然输入，按用户 2026-11-12 裁决登记为缺口（`observed_with_registered_gap`），S2~S5 缺位仍阻断**；
锚分数 1,747/1,747、计划 175/175（未解释 0）、分数与目标双重重放零差、输入 3,472 件比对（1 件显式接受）；
工作区 `source_digest` 与测试 SHA 与复审记录逐字节一致。
**T4 / T5 / T6 均已收口**（见后续 P7 T5、T6 章节；P7 出口由用户裁决）；**历史代际**（`p7_t25_c7_20261111_r1` 等）为等权代理时期产物，
已被同 ID 重取覆盖或仅作历史登记、不可复算，不得与现行结果并列当作"本次结果"。
**边界：只推理不训练（`alpha_consumed=false`），不产生 pass/fail 经济判定，不构成 B 区转正。**
详见 `docs/reports/p7_t25_closure_20261112.md`（收口）、`p7_t25_result_20261111.md`（结果）与
`docs/review/v2_p7_t25/`（四轮评审与作者回应）。

**v2 P7 T5 · 原始需求 / 配额 / 日内净额 / 借用会计与死局（机械复核第二段）：** 在 T4 生命周期之上补齐
「授权额度 + 借用会计 + 召回/死局」，把逐袖 `cash_i + credit_i ≥ 0` 从 T4 的登记项升级为**硬门**，
并交付 VV-H（同配额、禁 credit 参照）与 VV-S（允许 credit）**共用的机械实现**与逐日序列：

```bash
# 全窗三臂（T4 互证 / VV-H / VV-S + VV-H 参照重跑；约 6~8 分钟）
python scripts/v2_p7/t5/run_t5.py --run-id <NEW_RUN_ID>

# 分段验证（子窗口）/ 单臂（调试）/ 续跑（完成根仍拒绝）
python scripts/v2_p7/t5/run_t5.py --run-id <SMOKE_ID> --window 20190101 20191231 --arms parity vvs
python scripts/v2_p7/t5/run_t5.py --run-id <SMOKE_ID> --arms vvs
python scripts/v2_p7/t5/run_t5.py --run-id <UNFINISHED_ID> --resume
```

日内顺序写死：原始目标 `G` → 额度 `ρG`（不改写逐袖归一化权重）→ 约束/整手 → **预净额计划（零副作用）**
→ `cash_open + 本日 credit_granted` 对原始买入负债背书 → 授权后最终净额 → 仅净市场卖出 →
内部转账与净市场买入 → 一次结算。`G` 取 T2.5 `targets/daily_{arm}.parquet` 的 `target_notional`
（账户级 = 各相位之和，`(ρ_V20, ρ_V10) = (0.7, 0.3)` 为资本份额），**不改写**选股、权重与 20 日到期口径。

机械口径：`D=max(0,G−MV)`、`R=max(0,ρG−MV)`、`A ≤ min(D, R+L)` 且 `A+Fee(A) ≤ cash_open+L`；
`credit_outstanding` 不参与授权、`L` 只按本日转账一次入账；`creditsign` 借入为正、出借为负，
`E_i = cash_i + MV_i − credit_i`，本金转移零 P&L；T+1 召回未还即永久 `deadlocked`，
自然卖出只记 `late_recovery`（无被动卖出路径）。`credit=off & quota=off` 与 T4 完成根**逐位一致**
（逐日账本投影行 / 交易 / 守恒行 / 费用逐键 / 实际与独立 NAV / 终态摘要）。
**边界：只交付机械量与逐日序列（`alpha_consumed=false`、`verdict=null`）；两臂绩效、
三连指标经济裁定与阶段收口归 T6，不构成 B 区转正或前向/日历验证通过。**
证据落 `data/paper_shadow/p7_testbed/<t5_run_id>/`（九件套 + 逐日帧），详见
`docs/reports/p7_t5_closure_20261113.md` 与 `docs/review/v2_p7_t5/`（双评审闭环记录）。

**v2 P7 T6 · 两臂运行、绩效与三连指标经济裁定（机械复核第三段）：** 在同一初资 / 同一目标 / 同一费用模型 /
同一交易日历 / 同一执行语义下跑 VV-H 与 VV-S（唯一差异 = `credit` 开关），按**运行前预登记**判据给出 `verdict`：

```bash
# 全窗两臂 + 确定性重跑 + 指标/噪声带 + 裁定 + 黄金例（约 4.5 分钟）
python scripts/v2_p7/t6/run_t6.py --run-id <NEW_RUN_ID>

# 只读复算（不写盘）：从落盘 parquet 重算 NAV/链式/年化/费用/换手/配对差/分月分年/噪声带/预登记摘要
python scripts/v2_p7/t6/verify.py --run-root data/paper_shadow/p7_testbed/<run_id>
```

预登记口径（`calibration.py`，运行后不得改）：δ_TR=0.50pp、δ_MDD=0.50pp、δ_cagr=0.10pp、δ_vol=0.50pp；
配对分块 bootstrap（block=20=冻结持期、B=4,000、seed=42）；**越带 ≡ |Δ| ≥ δ 且 95% 区间不含 0**；
判据 C1 收益 / C2 MaxDD / C3 链式三项 / C4 费用-换手归因闭合 / C5 稳健性（集中度 + 留出 + 分年同号）；
`verdict`：C1–C5 全满足 → `pass`；C1/C2/C3 任一「方向相反且越带」→ `fail`；其余 → `inconclusive`。

最新证据：`p7_t6_20261113_r6`（1,745 日；ΔTR=+0.2248pp、ΔMaxDD=+0.0160pp、Δfee=−200.28 元；
**`verdict = inconclusive`**，C1 的统计可判下限 2.81pp 为 δ 的 5.63 倍）。
抽样为预登记同名的 stationary bootstrap（几何块长、两臂共用索引）；回撤费用占比按峰谷损失金额计
（4.8733%/4.8739%），费用占毛利 9.3308%/9.3115%；G5 已把宿主 `replay`/`restart` 自检纳入硬门。
**边界：`verdict` 只对本窗口、本机械与费用口径成立；不构成 P7 出口、B 区转正、生产多袖、
前向/日历验证或实盘可用结论。** 详见 `docs/reports/p7_t6_closure_20261113.md` 与 `docs/review/v2_p7_t6/`。

**v2 日更双通路与标签专用刷新（P3 前工程义务，v0.207.0）：** 冻结段 `label_filtered`
（≤20260702）与日更段 `structural`（≥20260703）双口径落地：日更段只做结构过滤、不因
未来标签不可算而拒绝落盘，结构性初建三表一律 forming；forming→sealed 只能由标签专用
刷新路径（E20=idx(T)+21 到达实际加载水位后逐日追赶）产生。行域与刷新审计随 manifest
迁移登记（13,802 个既有分区仅元数据迁移、parquet 零改动；repairs 707 条逐项保留）。
命令入口（涉及生产写入的 migrate/update 均默认 dry-run，须单独授权后 `--execute`；
verify 只读生产输入、只写 scratch 与正式报告）：

```bash
# manifest 行域迁移（一次性，已完成并审计）
python scripts/v2_p3_pre/migrate_store_manifest.py --data-root data \
  --backup-root data/backup/v2_p3_pre/<RUN_ID>

# 日更追加 / 标签追赶（逐日 build_daily；--refresh-only 仅标签追赶；--recover-event 受控恢复）
python scripts/v2_p3_pre/update_daily_store.py --start-date 20260703 --data-root data \
  --backup-root data/backup/v2_p3_pre/<RUN_ID>

# store 迭代验收（真实日更 forming + 历史影子 sealed/性能 + 幂等/故障门）
python scripts/v2_p3_pre/verify_store_iteration.py --dates 20260703,20260731 \
  --historical-shadow-count 20 --scratch-root temp/<SCRATCH> --out data/reports/<报告>.json

# manifest 损坏恢复（dry-run 默认；--execute 另案授权）
python scripts/v2_p1/recover_manifest.py --data-root data
```

**边界：本期仍未切换生产链路**——现役训练/回测/纸面仍走 cs_train/cs_infer 与旧内核；
v2 panel/labels 供 P3 只读消费，P3 开工硬门与训练适配契约见实施规划（本地
`docs/plans/v2/p3_pre_label_refresh_store_t0_plan_20261017.md`）§6。

**P3 迁移验收工具（内部使用）：** `scripts/v2_p3/` 的信号辅证复用回放的股票池与当日
clean 行情，应用相同的 ST、上市天数和停牌/无行情筛选；评估 Top-K 明细必须覆盖全部预期折。
归因报告会核验实际交易日历及特征文件的消费身份，读取前后或报告结束时发现漂移即失败，
缓存命中也执行身份检查。北向（Stock Connect）闭市日仅在 raw `moneyflow_hsgt` 的
`hgt/sgt/north_money` 全空且源侧 `north_net_buy` 全空时登记为 `excluded_days`，
不参与日频指纹评分；其余缺失或被拒日仍直接失败。袖子身份对非 `V2Trainer` 调用方
保持现状（视为未携带）；宿主携带身份时必须由 trainer 实现 `bind_sleeve_identity`
同步到模型 metadata，否则明确报错。工具仍按阶段分次授权运行，生产链路保持上述现状。

**v2 训练迁移验收（P3 T5，v0.208.0）：** 三臂验收全部通过——臂① legacy×cs_train、
臂② v2×cs_train、臂③ v2×panel。**G-code**（臂① vs 臂②）：14/14 逐折指纹逐位一致、
成交逐笔精确、链式净值 max|Δ|=0、信号/评估 top-k 零互换；**G-data**（臂③ vs B0'）：
ΔCAGR +0.445pp（±8pp 带内）、ΔMaxDD −3.77pp 经 F11 归因闭合后放行登记、ΔSharpe
−0.0096；**G-data PASS 的语义是“差异可归因”，不是“panel ≈ cs_train”**：实测成交键集
对称差逐折 >80%、预测同键改分近全量（max|Δ| 0.3~1.3）、max|Δnav| 5.689e-01、
ΔMaxDD −3.77pp，P4 消费必须带此量级登记；**F11 归因**：136 个差异列 100% 词表分类、
拒绝 0、结构破坏 0（CYQ5 结构白名单登记 40 项）、输出级一致性通过；**回归边**：冻结旧模型 + P2b `legacy_bypass`
复跑 vs `p2b_t5_closed_20261017_203508` 成对精确 PASS。三臂证据根
`data/frozen_reference/v2_p3/`。**注意：P3 仍未切换生产链路**——v2 panel/labels 供
P4 消费，现役训练/回测/纸面继续走 cs_train/cs_infer 与旧内核。

**v2 纸面宿主（P4，已切换生产）：** `v2/hosts/paper/factory.py::create_paper_host`
提供隔离根装配与单日驱动能力，状态以引擎快照为权威、旧纸面账本文件为投影；整日写锁
限制单写入者，崩溃恢复在同日或次日启动时补齐投影与发布标记。数据就绪门覆盖 panel、
clean 行情、事件 freshness 与 derived 实际产出；cs_infer freshness 回退会按股票键合入
最终模型输入。**就绪门门 4（B′，v0.208.35）**：以模型 metadata `feature_missing_rate`
（schema v2：逐列训练期**逐日**缺失率分布 p50/p90/p95/p99/max/mean/n_days）为基线，
当日率 ≤p95 通过、(p95,p99] 警告、>p99 错误——**三档均不阻断日跑**（信号层仍对全空列
硬拒预测，故不等于当日必有信号；阈值含
2% 绝对下限，避免近零基线列噪音）；唯一阻断是「聚合升级」（error 列数 ≥ max(15, 20%×
判定列数)，防大面积源故障）；结构性漂移列按 metadata 登记豁免（error 封顶 warning、
全空不豁免）。训练与当日**同域**（主板域），基线可用近 60 日滚动分位自校准
（初始种子 = 76 日对照域内率）。逐日证据落 `runs/readiness_<date>.json`
（通过/失败日均有）、滚动历史 `runs/coverage_history.json`，失败日另有 `runs/gap_<date>.json`；
推送文本附 `数据质量: error N / warning M`。**生产纸面入口与默认链路已于 2026-10-31 切至 v2**（**F24-A2 起观察期概念取消**：原「观察期 ≥4 自然周且 ≥一个大版本」与真实日历 SLA 判据已删除，M1 出口改历史等效证据 + 上线后监控；禁止宣称「真实日历验证通过」）。

**纸面估值口径（v0.209.9）：** `engine_state.bin` 的 `portfolio_values` 是**逐业务日收盘估值**
的历史序列（NAV 曲线源），其末日行必须与 `nav.parquet` 同业务日行同值；「某一行情日的当前
市值」只落只读旁路 `state/valuation_preview.json`（显式带 `market_date` 与
`authority_last_date`），**不得**写进 `portfolio_values` 或 `nav.parquet`。迁移估值三来源按
优先级收敛且全部绑定账户最后处理日：旧 `nav.parquet` 同日行逐位沿用 → 显式冻结输入（须为
业务日）→ 按业务日 clean 截面复算（缺分区直接报错，禁止回退最新截面）。启动一致性校验含
**第三腿**：`nav.parquet` 末行 ↔ 权威同日行的 `cash`/`position_value`/`total_value`（容差
1e-6），数值不一致返回 `"nav_divergence"`（记 warning、**不自动改写任一侧**），同日行缺失
fail-fast。出现该状态时先按授权程序修正，再推进日跑：

```bash
# 只读规划（打印 row_before / row_after / 差异），零写入
python scripts/v2_p4/t5_nav_row_repair.py --evidence-dir <证据目录> \
    --expect-total-value <业务日复算总值> --expect-market-value <业务日复算市值>
# 落盘（三重校验：当日分区复算 + nav 同日行交叉 + 操作者期望值；先备份原字节）
python scripts/v2_p4/t5_nav_row_repair.py --evidence-dir <证据目录> \
    --expect-total-value <业务日复算总值> --expect-market-value <业务日复算市值> --apply
```

该工具只改权威末日行三数值（日期/持仓/现金/账本/`nav.parquet` 均不动），原始字节备份在
`<证据目录>/engine_state.before.bin` 可整字节回滚；`20260603` 行已于 2026-11-05 按此程序
修正（证据 `data/frozen_reference/v2_p4/t5_nav_row_repair_20261105/`）。

**v2 到期义务清偿（P4 T3，已闭合）：** 新宿主 normalize/Kelly 统一回测侧语义；行业映射、
停牌日历与行情读取由回测/纸面宿主显式注入，core/common 不反向依赖 `v2/store`。
正常 v2 回测缺省经原生 `TargetPortfolio` 调用单袖 merge，显式 `legacy_bypass` 仅用于
冻结回归边；纸面 S6 在提交前消费同一指令序列，回测 S6 仅作 T3 证明性证据，其成交
反投影不得复用于 paper_replay。P2b dict 过渡件已删除；`cs_train_verify` 仅作为保活至
旧链路物理删除的过渡验收工具。T4/T5 已验收闭合（门 1/5/6/7/8 PASS），生产纸面默认链路已于 2026-10-31 切至 v2。

**v2 历史回放：** `schedule_mode="replay"` 的 `replay_calendar` 只声明驱动窗口，
`trading_calendar` 可包含窗口后的已知交易日。调仓以窗口首日为锚，末日可在没有
未来行情时生成次交易日计划；成交与 NAV 只记录窗口内日期。连续参照可显式传入
`run(..., trading_dates_full=完整交易日历)` 使用相同口径，省略该参数保持原回测行为。

**ML 模型特点：**
- 使用全量特征列训练 XGBoost 回归模型
- 标签为 `y_ret_5`（未来 5 日收益率，T+1 收盘买入 / T+1+5 开盘卖出口径）
- **训练时自动切分验证集**（默认最后 20% 时间作为验证集）
- **训练结束后打印验证集评估结果**（MSE、RMSE、R2、IC、RankIC）
- **使用早停机制**（early_stopping_rounds）防止过拟合，早停指标可选整段 `rank_ic` 或逐日截面 `rank_ic_daily`（与 daily_rankic 评估口径一致），并支持 `--min-best-iteration` 下限监控告警
- **标签 winsorize 处理**减少极端值影响
- **增加正则化参数**（L1/L2）提升泛化能力
- 模型自动保存到 `data/models/stock_selection` 目录（与 risk、terminal_loss 平级）
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
ls data/models/                     # ML 模型根目录（三平级模型家族）
  ├── stock_selection/              # 选股模型
  │   ├── model_registry.json       # 模型版本注册表
  │   ├── v1_model.joblib           # 模型文件
  │   ├── v1_features.json          # 特征列表
  │   ├── v2_model.joblib
  │   └── v2_features.json
  ├── risk/                         # 持仓风控模型（独立 registry）
  └── terminal_loss/                # 期末异常亏损模型（独立 registry，版本化保存）
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

### 申万行业分类历史（PIT，v0.209.68）

申万行业分类的**历史归属**取自申万宏源研究公开下载中心（0 元，无需授权采购），
TuShare 现有接口只能给当前快照。契约见 `docs/data/sw_industry_pit_audit.md`。

```bash
python scripts/download_sw_industry_history.py        # 拉取官网历史 + 码表 + 北交所映射
python scripts/download_sw_industry_history.py --skip-tushare   # 离线降级（仅官网数据）
```

产出：

- `data/raw/sw_industry_history.parquet`（原始四列快照）+ `sw_industry_history_meta.json`
- `data/clean/sw_industry_code_table.parquet`（SW2014/SW2021 码表，L1~L3）
- `data/clean/sw_industry_pit.parquet`（区间表：`code6 / in_date / out_date / ind_code / sw_version`）

使用（PIT：`in_date <= T < out_date`）：

```python
from src.lazybull.data import Storage
from src.lazybull.data.sw_industry_pit import attach_pit, build_exclusion_ledger, lookup_industry

pit = Storage().load_clean("sw_industry_pit")
lookup_industry(pit, "601888.SH", "20150105")   # 单点查询
attach_pit(pit, daily_df)                       # 附加行业列 + industry_known
build_exclusion_ledger(pit, daily_df)           # 未命中行台账（reason/fixable）
```

注意：该数据层为新增能力，**不写回 cs_train/cs_infer**，默认不改变既有生产行为；
接入需新 prereg（判据按 `CLAUDE.md` §5「数据完备性判据契约」的三件套写）。

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

单元测试使用合成数据与模拟客户端；纸面交易展示与补位测试也应模拟 TuShare 客户端，
提供合成配置并显式指定临时数据目录，避免依赖真实令牌或写入用户目录的令牌缓存。

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_cost.py
pytest tests/test_features.py
pytest tests/test_cleaner.py

# P4 回放与宿主接线回归（全部使用隔离合成数据）
pytest tests/test_v2_p4_t4_replay.py tests/test_v2_p4_t4_compare.py tests/test_v2_p4_t3_native_merge.py

# 默认临时目录不可访问时，在仓库 temp/ 下指定本次运行的独立目录
pytest tests/test_v2_p3_t5_arm_runner.py --basetemp=temp/pytest_t5_arm_runner

# 查看覆盖率
pytest --cov=src/lazybull --cov-report=html
```

全量回归约 5,400+ 用例，单次 30 分钟以上属正常（耗时集中在 P5b 试车场与训练/回测端到端
真实运行）。测试夹具批量写入 `PanelDataStore` 时必须走
`with store.defer_manifest_save():` —— 逐次 append 会触发 manifest 全量落盘，
160 日 × 14 次写入实测慢 5.2 倍（见 `tests/test_v2_train_wf_orchestrator_e2e.py`）。

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
│   ├── raw/                   # 原始数据（按日分区，部分数据集按年分区见下）
│   │   └── {name}/            # 按日分区: YYYY-MM-DD.parquet
│   │                          # 按年分区: YYYY-12-31.parquet（dividend/stk_holdertrade/repurchase/top10_floatholders）
│   ├── clean/                 # 清洗后数据（支持按日分区）
│   │   └── {name}/            # 按日分区: YYYY-MM-DD.parquet
│   ├── features/              # 特征数据
│   ├── models/                # ML 模型根目录
│   │   ├── stock_selection/   # 选股模型（model_registry.json + v*_model.joblib）
│   │   ├── risk/              # 持仓风控模型
│   │   └── terminal_loss/     # 期末异常亏损模型（版本化保存）
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
│   ├── train_terminal_risk_model.py # 期末异常亏损风控模型训练
│   ├── run_backtest.py        # 运行回测
│   ├── run_ml_backtest.py     # 运行 ML 信号回测
│   ├── compare_walk_forward.py # 实验对比与稳定性汇总（薄入口）
│   ├── compare/               # 实验对比分析子包（constants/loading/aggregate/scoring/fold_subset/signal_metrics/...）
│   ├── compare_wf_fold_subset.py # 折子集链式对比（薄入口；省时消融实验的可比对照）
│   ├── compare_wf_signal_metrics.py # 日频信号级配对对比（薄入口；筛选尺子+噪声带标定） ✅ v0.122.0
│   ├── analyze_factor_health.py # 因子体检（薄入口）
│   ├── factor_health/         # 因子体检子包（constants/scan/analysis/report/diagnose）
│   ├── analyze_factor_diagnosis.py # 因子诊断 v2（薄入口；偏 IC/覆盖显著性/使用度稳定性）
│   ├── materialize_factor_health_snapshot.py # 体检快照物化（运行时派生列 → 临时数据根 + 驱动体检/诊断） ✅ v0.124.0
│   ├── audit_model_columns.py # 模型列集审计（薄入口；列集漂移/配置分组/跨来源差异） ✅ v0.121.0
│   ├── model_audit/           # 模型列集审计子包（constants/columns/report） ✅ v0.121.0
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
│   │   ├── availability.py         # 运行时可用性标记（has_*，不改特征分区） ✅ v0.121.0
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
├── logs/                       # 临时日志目录（项目共识：临时 log 统一写这里，已 gitignore）
├── temp/                       # 临时文件目录（项目共识：临时脚本/中间产物，已 gitignore）
├── .env.example               # 环境变量模板
├── .gitignore                 # Git忽略文件
├── pyproject.toml             # Poetry配置
├── requirements.txt           # pip依赖
└── README.md                  # 本文件
```

---

## 📚 文档

- [数据契约](docs/data/data_contract.md): 各数据层的字段规范与主键约定
- [回测假设](docs/guide/backtest_assumptions.md): 回测系统的假设、简化与局限性
- [特征与标签定义](docs/data/features_schema.md): 日频特征构建、标签计算、过滤规则说明
- [涨跌停与停牌处理指南](docs/guide/trade_status_guide.md): 涨跌停与停牌状态的自动处理机制
- [纸面交易使用指南](docs/guide/paper_trading_guide.md): 纸面交易（Paper Trading）完整使用指南 ⭐ 新增
- [项目路线图](docs/roadmap.md): 分阶段开发计划
- [IC与RankIC优化指南](docs/guide/ic_optimization_guide.md): 提升模型预测能力的系统性优化方案
- [成交额过滤指南](docs/guide/amount_filter_guide.md): 成交额过滤功能说明与配置
- [分批调仓指南](docs/guide/batch_rebalance_guide.md): 分批调仓功能说明与配置
- [止损触发指南](docs/guide/stop_loss_guide.md): 止损触发功能说明与配置
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

### 树莓派 LCD35

运行 `python scripts/respi/lcd35_display.py`。入口自动合并
`configs/runtime_respi.yaml`，通过 `data.paper_remote` 读取 NAS 上的纸面账户；
认证由 `LAZYBULL_SMB_USER`、`LAZYBULL_SMB_PASS` 环境变量提供，系统需安装 `smbclient`。

- 远端文件缓存有效期为 180 秒；亮屏期间（06:00 至 23:00）持续按需同步，盘中摘要约每 3 分钟刷新，周期图及盘外同步沿用 10 分钟节奏。读取失败保留最近有效快照。
- “下次调仓”展示下一未履行的 T0 计划日，复用交易层分批排期；漏批时显示待补执行日期及剩余 0 天，T1 实际成交仍以纸面交易指令为准。
- 周期图的“账户”曲线使用 `nav/nav.parquet` 中的真实总资产，包含现金、成交成本和已实现盈亏；普通模式从最近调仓信号日开始，分批模式从批次锚定日开始，并与指数对齐到共同有效日期。
- 净值缺失不回填、不倒推；起点记录缺失时从首个共同有效日期起算并记录告警，完全无记录时不生成新图。日内“持仓”图仍展示当前持仓的日内涨跌，与账户周期收益区别使用。
- 纸面交易和 LCD 年化统一调用 `src/lazybull/paper/performance.py`，按实际自然日计算复合年化；起始日优先取 `account_start_date`，未配置时取最早有效净值日期。空仓后的已实现收益仍计入年化。

---

## 🧪 开发指南

### P5b 回撤裁判校准入口（治理/验收用途，非生产工具）

回撤裁判（`DD-W` 恶化否决 / `DD-I` 改善检出）的校准只在历史试车场里跑，产物落
`data/paper_shadow/p5b_testbed/<campaign_id>/reports/`，**不是前向证据、不构成 SLA 或生产裁决权**
（闸门由用户在合并验收时确认）：

```bash
# 1) 冻结回撤仪器 manifest（只读 B0 盲态选窗；上游收益仪器哈希必须一致）
python scripts/v2_p5b/drawdown_calibration.py --campaign-root <campaign> --freeze --apply
# 2) 训练两个方向的临界值 c^W / c^I
python scripts/v2_p5b/drawdown_calibration.py --campaign-root <campaign> --train --apply
# 3) holdout 逐单元 Wilson 门 + 功效/MDE + 判定表/报告
python scripts/v2_p5b/drawdown_holdout.py --campaign-root <campaign> --apply
# 4) 只读导出判定表（可选）
python scripts/v2_p5b/drawdown_export_table.py --campaign-root <campaign>
```

### P5b T6 历史端到端演练入口（治理/验收用途，非生产工具）

「多臂运行器 → 配对账本 → 裁判 → 报告」全链路的历史演练（五臂：基准 / 机械 A/A / 收益 −5δ /
DD-I `c^I_anchor+2pp` / DD-W `c^W_anchor−2pp`——臂角色名保留 `12pp` 后缀，实际目标由冻结阈值
推导，当前为 +28.09pp / −30.88pp），同样只落 `data/paper_shadow/p5b_testbed/<campaign_id>/reports/`，
`evidence_class=calibration_only`，**不是前向证据、不产生频率/MDE、闸门由用户在合并验收时确认**：

```bash
# 0) 完整步骤、退出码语义、验证与回滚见运行手册
#    docs/guide/p5b_testbed_runbook.md
# 1) 冻结 T6 演练 manifest（盲态预登记；上游 T4/T5 哈希必须一致）
python scripts/v2_p5b/rehearsal.py --freeze --campaign-id <campaign> --apply
# 2) 登记 campaign + 五臂 + D0 fresh
python scripts/v2_p5b/rehearsal.py --register --campaign-id <campaign>
# 3) 单日墙钟探针（>20 分钟外推需先取得用户授权再整批运行）
python scripts/v2_p5b/rehearsal.py --run --campaign-id <campaign> --days 1 --json
python scripts/v2_p5b/rehearsal.py --run --campaign-id <campaign> --authorization <授权记录.json>
# 4) 读数 + 报告 + 判定表（作者侧检查；不含闸门结论）
python scripts/v2_p5b/rehearsal.py --report --campaign-id <campaign> --apply
# 5) 只读校验（三环哈希/上游链接/A-A/判定边界）
python scripts/v2_p5b/rehearsal.py --check --campaign-id <campaign>
```

### P6 报告系统 T0.5 验收入口（治理/验收用途，非生产工具）

P6 报告系统当前只完成 T0.5（依赖、合成语料、双性能入口与审计基线），**投影/渲染实现尚未开始**。
固定合成语料与参考生成器只验证计时/体积/离线协议，不代表 P6 报告已验收：

```bash
# 一次性执行完整 T0.5 验收（语料双跑 + 参考生成 + 5 冷/5 热打开 + 5 次生成 + 契约/基线）
python scripts/v2_p6/run_t05_acceptance.py --validation-id <new_id> --json
# 不传 --validation-id 时自动生成全小写默认 id（p6_t05_acceptance_<UTC时间数字>），亦合法可用
# 产物（原始 JSON 与 SHA）：data/reports/p6/validations/research/<new_id>/
# 契约冻结件：docs/contracts/p6_t05_contract_freeze_20261105.json
# 打开测量依赖可选组 verification（Playwright + Chromium）；生产生成器禁止 import Playwright
```

依赖安装（精确版本，供复跑）：`python -m pip install "jinja2==3.1.6" "playwright==1.64.1"` +
`python -m playwright install chromium`；浏览器缓存默认位于 `%LOCALAPPDATA%\ms-playwright`
（Linux 为 `~/.cache/ms-playwright`），离线复跑复用同一缓存或本地 wheel 源，不把 Playwright
装进生产路径。

### P6 报告系统 T1 验收入口（只读投影层，非生产工具）

T1 交付只读投影层（稳定快照 / 权威内容核对 / 驾驶舱 projection / 自封存 manifest / 精确写
审计器），**仍未开始 HTML/CSV 渲染**。真实生产根核对归 T5，本入口只用合成夹具：

```bash
# T1 七组测试（P6-T1-01~17 全场景 + 整改负例）+ 治理面六组定向集
#  + 复杂度双口径（含 mccabe 正控制）+ 报告包 Playwright 零 import + 工作树指纹
python scripts/v2_p6/run_t1_acceptance.py --validation-id <new_id> --json
# 不传 --validation-id 时自动生成默认 id（p6_t1_acceptance_<UTC时间数字>）
# 产物：data/reports/p6/validations/research/<new_id>/t1_acceptance.json
# 已存在的 validation 目录一律拒绝（无覆盖）；失败即以新 id 重取证
# 证据 schema p6_t1_acceptance/v2：counts/expected_min_passed/mccabe_ran/worktree
```

实现位置：`src/lazybull/v2/reporting/{status,snapshot,authority,cockpit,instructions,
config_view,quality,sources,bundle,manifest,readonly_audit}.py`；契约见 `CLAUDE.md` §5
「P6 报告系统 T1 契约」。

直接调用写审计器时，前快照须由 `snapshot_tree` 获取并保留其 `FileTree.dirs` 目录证据；
转换成普通 `dict` 会丢失空目录身份，带新目录前缀的命令将拒绝判定。T6 合成兼容性用例使用
上游原生构造与封存函数生成 manifest、报告和判定表，不替代 T5 的真实 golden 验收。

### P6 报告系统 T2 生成入口（静态驾驶舱，只读离线）

T2 交付**自包含静态驾驶舱**：Jinja2 模板 + 内联静态 SVG，六区（账户 / 持仓 / 指令 /
pending / 调仓 / 模型·质量·新鲜度）+ 证据横幅 + 降级清单 + research 三端身份卡；
CSS/SVG/数据全内联，无运行时 JavaScript / CDN / 外部字体图片，CSP 固定，
companion 只链固定文件名（`projection.json` / `report_manifest.json`；全量 CSV 归 T3）。

```bash
# 生成并原子发布一份驾驶舱 run（默认写 data/reports/p6/<evidence>/dashboard/<report_id>/）
python scripts/p6_report.py generate --evidence-class production \
    --paper-root data/paper --quality-root data/reports/quality

# 带精确写审计（审计件落本次新 validation 目录；已存在即拒绝）
python scripts/p6_report.py generate --evidence-class research \
    --audit-out data/reports/p6/validations/research/<new_id> --json

# T2 验收驱动（六组 T2 + T1 七组 + T0.5 四组 + 棘轮；三份关键证据 + 双档性能 + 两次审计）
python scripts/v2_p6/run_t2_acceptance.py --validation-id <new_id>
```

口径要点：

- **证据域隔离 fail-closed**：`production` 只接受 production 输入，`calibration_only` 只接受
  calibration 输入，`research` 可含 production/research 但不得混入 calibration；混合或空声明即
  抛 `EvidenceIsolationError` 并非零退出（**不落任何产物**）；
- **原子新 run**：目标 `report_id` 目录已存在即拒绝（无 latest、不覆盖、不追加），失败清理本次
  staging，不留半成品；`attempt_id` 与 T1 快照同一字面量口径（禁点段/大写）；
- **安全**：来源文本一律 Unicode NFC + 转义 `< > & U+2028 U+2029` 后只写文本节点（scheme 字面量的
  冒号写成字符引用，可见文本不变）；**属性位只允许闭集常量**（`security.ALLOWED_ATTRIBUTES`，
  未登记属性与同元素重复属性即判定失败）；CSP
  `default-src 'none'; style-src 'unsafe-inline'; img-src data:; font-src data:; base-uri 'none'; form-action 'none'`
  且须恰一个；渲染结果在**发布前**强制自检（不可关闭）；除初始 file 导航外零 HTTP(S)/WebSocket 请求；
- **性能门（不放宽）**：标准+压力每页 ≤3MB、5 冷 + 5 热逐次 <2s、生成逐次 <10s；口径为
  “打开=页面自报 `loadEventEnd`、生成=in-process 生成调用”，验收载荷同时登记 `headroom`（实测÷阈值）；
- **确定性**：同输入且同 `generated_at` 的 HTML 逐字节一致；换墙钟即新 run；manifest 登记
  projection 链与渲染链的逐文件 SHA + 入口 SHA + `root_map`（含 `quality` 根）；
- **写域证据**：`readonly_audit.json` 内的 `scan_scope` 登记有界扫描域、排除清单、未登记盲区
  （须为空）与重解析点（须为空）；`logs/`、`temp/`、`__pycache__/` 任何写入即判定失败。

实现位置：`src/lazybull/v2/reporting/{render,dashboard_view,dashboard_sections,view_primitives,
svg,text_safety,html_check,security,evidence,publish,generate}.py` +
`templates/dashboard.html.j2`；契约见 `CLAUDE.md` §5「P6 报告系统 T2 契约」。

### P6 报告系统 T3 生成入口（实验报告 + 五份完整 CSV，只读离线）

T3 交付**实验报告**：显式基线/臂、链式 NAV、Δ 指标、分折榜、跨期稳定性、19 列执行归因、
成本/双边成交金额、旧 WF TopK 正日占比、制度摘要与四段证据卡；同时输出**五份完整 CSV**
（`arm_metrics.csv` / `fold_metrics.csv` / `chain_nav.csv` / `execution_attribution.csv` /
`stability.csv`），全值、非抽样、固定列序、UTF-8/LF、root-relative `source_ref`，
manifest 逐份登记 `columns,row_count,key_columns,sha256`。

```bash
# 显式基线与臂（runs=规范化 runs 批次；legacy_wf=旧 WF 运行目录）
python scripts/p6_report.py generate --kind experiment --evidence-class research \
    --arm base=runs:data/runs/<batch_base> \
    --arm arm=runs:data/runs/<batch_arm> \
    --arm legacy=legacy_wf:data/walk_forward/<run_id> \
    --baseline base --splits 0-13 \
    --regime-json <p5a_regime.json> \
    --audit-out data/reports/p6/validations/research/<new_id> --json

# 标准/压力合成语料（确定性；写 temp/，可双跑比对 content_sha256）
python scripts/v2_p6/build_t3_corpus.py --out temp/p6_t3_corpus --profile all

# T3 验收驱动（T3 十一组 + T1/T2/T0.5 + 治理面；双档生成/打开/审计/CSV 守恒/包内复算）
python scripts/v2_p6/run_t3_acceptance.py --validation-id <new_id>
```

口径要点：

- **不重算公式**：链式指标只调 `calculate_chain_metrics`，Δ/分折只调既有 `compare_runs`，
  旧 WF TopK 跨折均值只调 `aggregate_run`；生成链**禁止**调用 `bootstrap_delta/bootstrap_table`；
- **Δ 符号**：`Δ = 臂 − 基线`，`ΔMaxDD > 0` 表示回撤改善；MaxDD 为非正小数；只是符号约定，
  不构成采纳/闸门结论；
- **链式有效性**：逐行校验可解析、有限、NAV>0，任一坏行整臂 `invalid/nonfinite_chain_nav`；
  合法输入须 `rows_used = rows_total = N` 且 `trading_days = N − S`（折界重复点不计收益区间）；
- **可比性**：数据态、折集合、逐折窗口、host_mode/实现、基线确认任一不同即
  `not_comparable`（不产 Δ）；跨实现（runs vs 旧 WF）只做对账陈列；
- **TopK**：规范化 runs 固定 `not_available/no_unique_implementation`（禁止从 `topk_detail`
  私算）；旧 WF 仅把 `KEY_Top20/30_hit_rate` 改名，并在成品侧与 `aggregate_run` 对账；
- **稳定性**：`DYNAMIC_EXACT` 单一分类 + 外层 `stable_identity`（代码态/数据态/host/实现不并组）；
  缺 `综合得分`/指标列 ⇒ `missing_precondition` 且禁止打分；组内不足两条有限值 ⇒
  `insufficient_periods`（不产生数值 `stability_score`）；`batch_run_id` 等动态分组列统一占位，
  原始 run/折/窗口/模型版本仍在 CSV 与 `source_ref` 中保留；
- **制度摘要**：只读显式 P5a JSON 的既有分箱计数；无 bins 固定 `not_available/no_regime_bins`，
  不画伪分布、不重采样；
- **四段证据卡**（信号 / 执行 / β暴露 / 成本）是独立证据，**不是守恒瀑布**；成本金额差不等于
  NAV 贡献，β 暴露无来源即降级；
- **证据目录**：`data/reports/p6/<evidence>/experiment/<report_id>/`；审计落兄弟
  `validations/<evidence>/<id>_audit_<profile>/`；历史 run 不覆盖。

实现位置：`src/lazybull/v2/reporting/{experiment_sources,experiment_metrics,experiment_stability,
experiment_csv,experiment_projection,experiment_view}.py` + `templates/experiment.html.j2`；
契约见 `CLAUDE.md` §5「P6 报告系统 T3 契约」。

### P6 报告系统 T4 生成入口（趋势库 + 完整 run 索引，只读离线）

T4 交付**三类物理隔离的趋势库**：`production`（P4 纸面 run）/ `research`（规范化 runs 与旧 WF）/
`calibration_only`（P5b sealed bundle）。每个 run 一行身份（窗口/数据态/代码态/来源/哈希），
五类指标按证据类分列（年化 / 回撤 / 双边成交金额 / 成本 / TopK 正日占比），
并输出两份**完整 CSV**：`run_index.csv`（`index_mode=full`，永不抽样）与
`trend.csv`（行键 `run_key+metric_name`，缺失留状态行，不插值）。

```bash
# research：规范化 runs + 旧 WF（一次全量重建；批任务）
python scripts/p6_report.py generate --kind trend --evidence-class research \
    --source runs:data/runs --source legacy_wf:data/walk_forward \
    --audit-out data/reports/p6/validations/research/<new_id> --json

# research：上一不可变快照 + 新 run（秒级增量；只把新 run 作为来源）
python scripts/p6_report.py generate --kind trend --evidence-class research \
    --source runs:data/runs/<new_batch> \
    --previous data/reports/p6/research/trend/<previous_report_id> --json

# production：只收 P4 纸面根（收 runs/legacy/bundle 即失败）
python scripts/p6_report.py generate --kind trend --evidence-class production \
    --paper-root data/paper --json

# calibration_only：只收经封存校验的 P5b 回撤族 bundle
python scripts/p6_report.py generate --kind trend --evidence-class calibration_only \
    --bundle data/paper_shadow/p5b_testbed/<campaign>/reports --json

# 合成语料与验收（标准 500 / 压力 2,000 run；含 Linux canonical 包）
python scripts/v2_p6/build_t4_corpus.py --root temp/p6_t4_corpus --profile standard
python scripts/v2_p6/run_t4_acceptance.py --validation-id <new_id>
```

口径要点：

- **三类隔离**：根（`<evidence>/trend/<report_id>`）、索引、横幅与输入白名单四层隔离；
  不提供跨类总览或混合趋势；跨类 `--previous` 直接失败；
- **身份**：run 目录有 `manifest.json` 时 `source_identity_sha256` = manifest 字节 SHA；
  旧 WF 无 manifest 时 = 实际消费白名单文件（chain/summary）的 `path+sha256` 排序后
  canonical 哈希，`source_manifest_sha256` 为空并标 `manifest_absent`；不使用 mtime。
  `root_id` 由来源根的**仓库相对路径 SHA-256 前 8 位**派生（`src_<8>`/`legacy_<8>`/
  `bundle_<8>`，锚点为仓库根而非当前工作目录，仓库外根退化为绝对路径哈希属非可移植边界），
  `--source` 顺序不参与身份；同一物理 run 出现在多个来源根、或同一身份换 root_id 再现，
  一律拒绝（不静默双份）；**升级边界**：`0.209.39` 旧快照的 `src0/legacy1` 身份不能作为
  `--previous`（会身份冲突 fail-closed），升级后需一次全量重建（单独授权批任务）；
- **五类指标不同轴**：production 年化为自然日百分数、research 为 252 日小数 CAGR、
  calibration 保留 log-MDD/pp；不互转、不跨域复用；规范化 runs 的 TopK 固定
  `not_available/no_unique_implementation`，production 回撤固定
  `not_available/no_unique_drawdown_series`；calibration 的回撤显示名为
  “B0 冻结源窗口基准 MDD（500 样本，pp）”，**不是策略回撤**；
- **production 权威门**：年化必须先过 T1 权威链路（`paper_generation` + NAV 三原值核对）；
  缺权威即 `degraded/authority_missing`，记录形态冲突即 `invalid`，NAV 未核对不计算年化；
  `runs/` 下 `readiness_*`/`gap_*` 属宿主诊断旁路，不计入趋势 run；
- **金额来源与窗口**：金额行绑定逐折实际消费文件（run 目录 ref + 文件集合 canonical SHA），
  先按行窗口闭区间过滤；缺窗口/缺日期即 `not_available`（不贴窄窗口、不部分求和）；
- **可见性与抽样**：页面 `index_mode=visible` 只显示最近 500 行；图每序列最多
  `min(1,200, 可见行)` 点（首尾与极值必留），抽样只影响图，不改 CSV 行键/值；
- **增量与历史**：上一快照行原样继承、不重算不删除；`root_map` 由“本次来源 ∪ 上一快照 ∪ prev”
  合并，`inherited_roots`/`inherited_source` 记录继承来源的包哈希，输出行引用的 root_id
  未登记即生成期失败；同 `report_id` 再发布拒绝；
  全历史重建是单独授权批任务，不冒充秒级交互；
- **性能与门范围**：10s 门只覆盖 `generate`（projection + HTML + 两份 CSV + manifest）；
  真实发布含两次整树审计（约 59s/档）单独登记，不参与该门；
- **跨平台**：路径一律 `root_id:relative_posix`（NFC），拒绝盘符/URI/`..`；
  `run_t4_acceptance.py` 产出 canonical Linux 验收包（固定四文件 + 期望哈希 + **可执行**
  复跑命令：`--upto` 分步建语料、上一快照相对位置与其 SHA、显式 `--reports-root`），
  真实 Linux 执行门归 T5。

实现位置：`src/lazybull/v2/reporting/{csv_core,trend_contract,trend_csv,trend_metrics,
trend_observe,trend_sources,trend_projection,trend_svg,trend_view,trend_portability}.py` +
`templates/trend.html.j2`；契约见 `CLAUDE.md` §5「P6 报告系统 T4 契约」。

### P6 报告系统 T5 验收入口（联验 / 真实 golden / 真实 smoke / 双平台）

T5 只新增**验收侧**工具，生产生成入口不变。

```bash
# 1) 真实 P5b golden 独立核验（非 pytest；真实根缺失/不符即失败，禁止 skipif）
python scripts/v2_p6/check_real_golden.py \
    --root data/paper_shadow/p5b_testbed \
    --out data/reports/p6/validations/research/<new_id> \
    --freeze docs/contracts/p6_t5_real_golden_freeze_20261107.json

# 2) T5 联验全链（CI/本地一次跑完；产出 t5_acceptance.json 与兄弟证据目录）
python scripts/v2_p6/run_t5_acceptance.py --validation-id <new_id>
```

覆盖内容（逐项写入 `t5_acceptance.json`）：

- `real_golden`：三族真实 campaign（`t4_seq10`/`t5_seq7`/`t6_seq21`）的精确路径与文件集合、
  逐文件 SHA、原生封存、`hash_chain`/`identity` 链、冻结关键值；
- `real_smoke`：三条**只读**真实 smoke，每条都带产品精确写审计 + 驱动侧独立整树判定 +
  输入根 before/after 全树 path+SHA 比对：
  - P4 生产驾驶舱（`data/paper`）：账户 `degraded/run_record_pre_switch_shape`、指令
    `not_available/run_record_pre_switch_shape`、实跑模型 `not_available/no_persisted_actual_model`、
    NAV 六业务日三原值与权威 blob 逐值相等、末日总值 `619016.1802662`、可年化；
  - 真实 `data/runs` 规范化批次：整根当批次必须 fail-closed（异构根含 paper replay 目录），
    批次生成的 `run_index.csv`/`trend.csv` 行数与身份自洽；
  - 真实 P5b campaign：`calibration_only` 趋势面，`source_identity_sha256` = 回撤族 manifest
    字节 SHA，log-MDD 与上游原生报告值逐值相等；
- `three_faces`：dashboard/experiment/trend × standard/stress 的 5 次完整进程生成（<10s/次）、
  5 冷 5 热打开（<2s/次）、体积门（≤3,000,000B），原始 JSON 逐档落盘；
- `reproduce`：三面同输入同 `generated_at` 的**全链重建**逐字节一致（复算判据取层 3；
  层 2「包内 projection→HTML」**仅 dashboard 面实测**（experiment/trend 未做，probe 为 null）存在 footer 生成器序归一问题，只作诊断登记）；
- `dual_platform`：Windows 侧 canonical 哈希 + Linux 侧探测结论；**无可用 Linux 环境时显式
  `not_achieved`** 并给出**单入口**准备包（不以“跳过”代替，一条命令跑完）：
  `python scripts/v2_p6/run_t5_linux_replay.py --windows <windows t5_acceptance.json> --json`
  （按载荷记录的相对路径重建语料 → 三面复算 → 与 Windows 逐文件 SHA/canonical 比对；
  同一条闭环可先用 `python scripts/v2_p6/selftest_t5_linux_replay.py --json` 在本机自检）；
  另可用驱动内联分支 `run_t5_acceptance.py --linux-python <py312> --linux-cwd <repo> --windows-evidence <json>`；
- `zero_write` / `historical` / `global_scope`：白名单外零差异、既有文件零改写零删除、
  新增仅落在本次报告 run 与 validation 产物内（扫描域与排除树边界随载荷登记）。

P6 全部里程碑（M0.5~M5）的总收口报告见 `docs/reports/p6_closure_20261108.md`（证据索引、评审轨迹、
残余项移交）；T5 细节见下方。

契约见 `CLAUDE.md` §5「P6 报告系统 T5 契约」；证据与限制见
`docs/reports/p6_t5_closure_20261107.md`。未达成/未测项按 L1~L10 编号登记（载荷
`known_limitations`、计划 §24.4、闭合报告 §5 同编号）：L1 Linux 端到端（**2026-11-08 用户裁决豁免**）、
L2 层 2 包内重建（仅 dashboard 面实测；**同日豁免为诊断项，复算判据取层 3**）、
L3 audit 域量化、L4 UNC/file URI、L5 多源 manifest、L6 1,200 点抽样、L7 NFC/NFD、
L8 `prev/` 增强、L9 P4 钉住态、L10 trend 路径敏感（`by_design`）。
豁免后 `unmet_items` 为空、T5 验收为 `pass`（`p6_t5_acceptance_run11`）；**豁免不等于已达成**。

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

### 测试数据隔离

纯冷区测试使用 `bootstrap_manifest` 与 `append_archive_features` 在临时数据根构造归档分区，
并确认 `features/panel` 热区目录不存在。此类测试应使用独立夹具，避免混入共用夹具创建的热区数据。

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
3. **简化假设**: 当前版本存在多项简化（详见 [回测假设](docs/guide/backtest_assumptions.md)）
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
- 临时文件统一放 `temp/`、临时日志统一放 `logs/`（均已 gitignore，可随时清理）

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
