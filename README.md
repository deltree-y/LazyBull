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

#### 期末异常亏损风控模型（terminal_loss）

预测持仓在剩余持有期期末发生波动标准化异常亏损的概率，用于到期前风险退出
（方案见 `docs/plans/terminal_loss_risk_model_plan.md`）：

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

# 滚动 Walk-forward（8 折研究型 WF：排序信息量的时间稳定性验证）
powershell -ExecutionPolicy Bypass -File .\scripts\batch\batch_terminal_risk_wf.ps1
# 完成后自动汇总：data/walk_forward/terminal_risk_wf/summary.csv
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

把 E2 的逐日暴露系数表（`--export-table` 产物，两列 CSV：`日期, 暴露系数`）接入回测引擎，
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
归一化到子集起点，再复用 `ml/walk_forward/chain_metrics.py` 口径），输出 `折子集对比.csv`
（含 ΔCAGR / Δ最大回撤 / Δ夏普 与逐折同向数）与 `逐折对比.csv`（中文表头，默认落
`data/reports/wf_fold_subset/<时间戳>/`）。折集合、逐折窗口或数据态 ID 不一致会直接报错终止。

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
│   ├── compare/               # 实验对比分析子包（constants/loading/aggregate/scoring/fold_subset/...）
│   ├── compare_wf_fold_subset.py # 折子集链式对比（薄入口；省时消融实验的可比对照）
│   ├── analyze_factor_health.py # 因子体检（薄入口）
│   ├── factor_health/         # 因子体检子包（constants/scan/analysis/report）
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
