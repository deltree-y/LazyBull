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

### 当前版本 (v0.15.1)

**公共模块重构：消除三套脚本间的重复代码** (v0.15.1):
- ✅ **`TradingConfig` 统一策略参数**（新增 `src/lazybull/common/trading_config.py`）：
  - 将 `paper_trade.py` / `run_ml_backtest.py` / `bot_service.py` 中重复定义的 ~170 行 argparse 参数抽取为公共 dataclass + `add_trading_args()` 注册函数
  - 支持 `from_args()` / `from_dict()` / `to_dict()` 互转，`create_stop_loss_config()` / `create_equity_curve_config()` 统一构建
- ✅ **`create_signal()` 信号工厂**（新增 `src/lazybull/common/signal_factory.py`）：
  - 单模型 / 双模型 Ensemble 判断逻辑统一为一个入口
- ✅ **`check_positions_stop_loss()` 止损检查**（新增 `src/lazybull/risk/stop_loss_checker.py`）：
  - 将 `paper_trade.py` 和 `BacktestEngine` 中重复的止损检查逻辑提取为纯函数
- ✅ **`DataLoader.build_stock_names_dict()`**：股票名称构建逻辑下沉到 DataLoader 公共方法
- ✅ **`paper_trade.py model-info` 子命令** / **`bot_service.py model` 命令**：查看当前模型版本、训练参数、性能指标
- ✅ **大幅精简**：`paper_trade.py` 删除 ~200 行废弃代码，`run_ml_backtest.py` argparse 从 ~200 行缩减为 ~40 行
- ✅ **`runner.py` 行业映射修复**：`load_industry_mapping()` 改为从 `shenwan_industry` 数据加载

### v0.15.0

**基本面因子系统 + 行业/组合约束 + 钉钉机器人交易** (v0.15.0 新增):
- ✅ **基本面因子系统（全新模块）**：
  - 新增 `src/lazybull/factors/fundamental.py`：季度财务指标（fina_indicator）按 ann_date 前向填充到日频，防止前视偏差
  - 5个因子：`roe_waa`（加权ROE）、`or_yoy`（营收增速）、`netprofit_yoy`（净利增速）、`debt_to_assets`（资产负债率）、`q_gr_yoy`（单季营收增速）
  - 新增 `scripts/download_fina_indicator.py`：全市场财务指标下载（支持断点续传 `--resume`）
  - `TushareClient.get_fina_indicator()` / `DataLoader.load_fina_indicator()` 新增
  - `FeatureBuilder._add_fundamental_features()` 自动合并到截面特征
  - 训练脚本 `--enable-fundamental-features` 开关：`train_ml_model.py` / `walk_forward.py` / `build_features.py` / `build_clean_features.py` 均支持
- ✅ **行业约束 + 组合约束接入纸面交易**：
  - `paper_trade.py config` 新增 `--max-per-industry`、`--max-weight-per-stock`、`--exclude-st` / `--no-exclude-st`、`--min-list-days` 参数
  - `PaperTradingRunner` 信号生成 / T0 / 补位均传递行业约束和单股权重上限
  - `industry_constraint.py` 优先使用 `sw_l2`（申万二级）行业分类
- ✅ **钉钉机器人增强（`bot_service.py`）**：
  - 新增 `positions [日期]` 命令：手机友好的 Markdown 持仓展示（按收益率排序）
  - 新增 `trade [日期]` 命令：异步执行交易流程（止损→延迟卖出→T1→T0），完成后推送结果
  - 新增 `help` 命令：显示所有可用命令
  - `execute_trade()` 复用 paper_trade.py 核心逻辑，无需登录服务器即可远程交易
- ✅ **其他改进**：
  - `paper_trade.py`: horizon 默认值 5→20
  - `train_core.py`: 特征重要度打印全部特征（不再限制 Top20）
  - `compare_walk_forward.py`: 新增 `algorithm`（算法）列

### v0.14.0

**实时行情查看 + 树莓派 mini-LED 持仓显示** (v0.14.0 新增):
- ✅ **`paper_trade.py real` 子命令**：一键查看持仓实时行情，支持 `--ret-profit-only` 精简单行输出
- ✅ **`TushareClient.get_realtime_quote()`**：封装 Tushare `realtime_quote` 接口，按需获取实时价格
- ✅ **`get_realtime_portfolio_summary()`**：计算6项实时汇总指标（持仓数/市值/总资产/浮盈率/总盈亏/年化收益），供外部调用
- ✅ **`scripts/respi_disp_real.py`**：树莓派 mini-LED 持仓看板，每10分钟刷新一次，23:00-6:00 自动息屏
- ✅ **`src/lazybull/drv/mini_led/`**：128×64 OLED 硬件驱动（SSD1306，GPIO/SPI 方式），含 8×16 / 6×8 字库
- ✅ **`paper_trade.py run` 默认日期**：`--trade-date` 由必填改为默认当日日期（`pd.Timestamp.today()`）

### v0.13.9

**选股过滤重构、compare_walk_forward 增强与新股过滤优化**:
- ✅ **选股过滤重构（`ml_signal.py`）**：将百分位成交额过滤重构为 `_apply_selection_filters`，引入三个绝对阈值：成交额 `amount_ma20 ≥ 5000万`、市值 `total_mv ∈ [50亿, 1500亿]`、申万一级银行/非银金融剔除
- ✅ **新股过滤统一**：全局 `min_list_days` 默认值由60天提升至365天（约12个月），涉及 `cleaner.py`、`builder.py`、`paper/runner.py` 等多处
- ✅ **`compare_walk_forward.py` 大幅增强**：输出从 CSV 改为 Excel（两个 sheet：实验对比 + 指标说明）；新增综合评分体系（8指标加权百分位排名，0~100分）；标题行超链接跳转指标说明；参与评分列着浅→深绿底色；新增 `model_version_range` 列；按运行时间戳降序排列；所有列名改为中文

### v0.13.6

**修复市场状态与技术指标特征对 `--start-date` 敏感的问题** (v0.13.6 修复):
- ✅ **新增 `_slice_by_trading_days()` 通用切片方法**：
  以目标 `trade_date` 在全量 `trade_cal` 中的位置为锚点，向前回溯固定 **120 个交易日**
  作为 warmup 窗口，返回该起点之后的所有交易日数据子集，确保两次构建的输入起点相同。
- ✅ **`_add_market_state_features()` 切片修复**：
  首次建立 `_market_state_cache` 时，先对 `daily_adj` 与 `daily_basic_data` 应用
  `_slice_by_trading_days`，保证 rolling 计算（`mkt_adv_dec_ratio`、`mkt_vol_20`、
  `mkt_ret_avg_20` 等）的输入起点固定，不受 `--start-date` 影响。
- ✅ **`_get_tech_factor_today()` 切片修复**：
  首次建立 `_tech_factor_cache` 时，使用 `_slice_by_trading_days` 替换原有全量交易日
  过滤逻辑，确保 KDJ/MACD/RSI/布林带/波动率等 EWM/rolling 指标历史起点一致。
- ✅ **`build_clean_features.py` 扩展 warmup 加载范围**：
  数据加载起点从 `start_date - 1个月` 扩展为 `start_date - 7个月`（约覆盖 150 个
  交易日），为 warmup=120 提供充足的历史支撑。
- ✅ **历史不足时行为不变**：历史窗口 < 120 个交易日时，按 `min_periods=1` 行为
  计算，不抛异常。

### v0.13.5

**修复 `--start-date` 变化导致同一 `trade_date` 特征不稳定** (v0.13.5 修复):
- ✅ **新增 `_get_lookback_dates()` 私有方法**：
  以目标 `trade_date` 在**全量** `trade_cal` 中的位置为锚点，向前回溯恰好 N 个交易日，
  确保窗口日期只由全量交易日历决定，与构建脚本的 `start_date` 范围无关。
- ✅ **`_calculate_features()` 使用新方法**：
  替换旧的 `current_idx - window` 切片 + 区间筛选逻辑，改用 `_get_lookback_dates`，
  消除因 `trading_dates` 被截断时窗口错位导致的 `ret_N`、`vol_ratio_N`、`ma_deviation_N` 差异。
- ✅ **`_add_moneyflow_features()` 同步修复**：资金流 rolling 窗口也改用 `_get_lookback_dates`。
- ✅ **`_get_tech_factor_today()` 过滤 `daily_adj` 到全量交易日历**：
  预计算技术指标（RSI/KDJ/MACD/布林带/波动率）前，先将 `daily_adj` 过滤到 `trading_dates`
  集合，消除因 `daily_adj` 起始日期不同导致的滚动/EWM 指标差异。
- ✅ **历史不足时返回 NaN**：历史窗口回溯不足 N 个交易日时，相关特征置 NaN，不抛异常。

### v0.13.4

**修复 `label_transform=cs_zscore` 场景下的数据泄露/评估口径问题** (v0.13.4 修复):
- ✅ **新增 `split_train_val_by_date()` 共用切分函数**：
  `src/lazybull/ml/train_core.py` 新增按 `trade_date` 粒度切分训练集/验证集的函数，
  确保同一交易日的所有样本不会被拆分到不同集合，彻底避免截面统计量跨集合污染。
- ✅ **`prepare_training_data()` 改用日期切分**：
  替换旧的按行数比例（`iloc`）切分为按交易日列表切分；
  新增可选参数 `label_transform_fn`，支持切分后各自独立变换。
- ✅ **`train_ml_model.py` 与 `walk_forward.py` 修复**：
  `label_transform=cs_zscore` 时不再对全量数据预先变换，
  改为先按日期切分，再对训练集和验证集各自独立应用 `transform_labels_cs_zscore`，
  确保截面统计量（均值/标准差）互相独立，不跨集合混入。

**修复 volatility_20 / zscore_volatility_20 / spec_score 数值不一致** (v0.13.3 修复):
- ✅ **新增共用函数 `compute_ret_1()`**：
  新增 `src/lazybull/factors/returns.py`，统一 `ret_1` 构造口径：
  优先使用已有 `ret_1` → 若无则用 `close_adj` 按组 `pct_change()`（复权口径）→ fallback 到 `pct_chg/100`。
- ✅ **修复预计算口径**：`precompute_technical_factors` 波动率分支改为调用 `compute_ret_1`，
  消除旧版本在缺少 `ret_1` 时直接使用 `pct_chg/100` 导致的口径偏差。
- ✅ **衍生指标自动一致**：`zscore_volatility_20` 与 `spec_score` 均由 `volatility_20` 派生，无需额外处理。
- ✅ **性能优化不回退**：缓存机制、批量预计算均保持不变。

**技术指标与波动率批量预计算 + 实例级缓存** (v0.13.2 新增):
- ✅ **批量预计算函数 `precompute_technical_factors()`**：
  新增 `src/lazybull/factors/precompute_technical_factors.py`，
  对全量 `daily_adj` 一次性计算 RSI(14)、KDJ(9,3,3)、MACD(12,26,9)、布林带(20,2)
  及多窗口滚动波动率，输出宽表供按日查表。
- ✅ **`FeatureBuilder` 实例级缓存**：新增 `_tech_factor_cache` 字段，
  首次构建时触发批量预计算并缓存；后续每日仅做 `O(1)` 查表，
  彻底消除批量构建时逐日切片 + 重复计算的瓶颈。
- ✅ **输出口径不变**：复用现有 `calculate_rsi/kdj/macd/bollinger_bands/volatility`
  函数，数值精度与旧逻辑完全一致（< 1e-6），无需重新训练模型。
- ✅ **`build_features.py` 和 `build_clean_features.py` 自动受益**：
  两条链路均通过 `FeatureBuilder` 构建，无需修改脚本。

**市场状态特征性能优化** (v0.13.1 新增):
- ✅ **批量预计算 + 实例级缓存**：新增 `precompute_market_state_features()` 函数，
  对全量 `daily_data` / `daily_basic_data` 一次性 groupby + pandas rolling，
  消除批量构建时每日重复的 60 次滚动循环（每日约 10 秒 → 近似 O(1) 取值）。
- ✅ **`FeatureBuilder` 零侵入缓存**：首次调用 `_add_market_state_features` 时自动触发批量预计算并缓存；
  后续逐日仅按索引取一行，兼容 `build_features.py` 和 `build_clean_features.py` 两条链路。
- ✅ **输出口径不变**：6 个市场状态特征数值与旧实现完全一致（精度 < 1e-9），无需重新回测。

**申万行业升级三级（L3）+ 分层回退中性化** (v0.13.0 新增):
- ✅ **申万行业升级为三级（L3）**：下载口径从 L2 升级为 L3（约 200+ 子行业），精度更高
  - 行业映射表统一为单张表，包含 L1/L2/L3 三层字段
  - FeatureBuilder 主字段（`sw_industry*`）映射到 L3；同时输出 `sw_l2*` / `sw_l1*` 辅助字段
  - `update_basic_data.py --only-shenwan` 升级到 L3 下载
- ✅ **L3→L2→L1→全市场分层回退中性化**：样本不足时按层级回退，保证稳健性
  - L3 行业内 `tradable==1` 样本数 < `min_group_size(=5)` → 回退到 L2 统计
  - L2 仍不足 → 回退到 L1；L1 仍不足 → 回退到全市场统计
  - 新增可复用模块 `src/lazybull/factors/hierarchical_industry_neutralization.py`

**新增个股特征与市场状态特征** (v0.12.1 新增):
- ✅ **新增个股特征（4个）**：
  - `is_new_stock`：新股标记（上市<365天=1）
  - `size`：流通市值（`circ_mv`）
  - `zscore_size`：行业内流通市值 Z-Score（`log1p(size)` 按 `sw_industry`）
  - `spec_score`：个股特质得分（`zscore_volatility_20 × (−zscore_size)`）
- ✅ **新增市场状态特征（6个）**：每日一个标量值，广播到所有股票
  - `mkt_vol_cnt`：截面收益率标准差；`mkt_vol_20`：20日滚动均值
  - `mkt_turnover_ratio`：市场拥挤度；`mkt_ret_avg_20`：20日平均收益率之和
  - `mkt_turnover_std`：换手率截面标准差；`mkt_adv_dec_ratio`：60日涨跌比均值

**申万二级行业切换 + rank-weight 训练增强** (v0.12.0 新增):
- ✅ **申万行业切换为二级**：行业分类从一级（~30个）切换为**二级（~100个子行业）**，中性化精度更高
  - FeatureBuilder 输出字段统一为 `sw_industry` / `sw_industry_code` / `sw_industry_id`
  - 中性化分组基于 `sw_industry`（二级行业名称）
- ✅ **rank-weight 训练增强**：对每日截面 Top30 和 Bottom30 样本赋予更高训练权重（默认5倍）
  - 强化模型对头部/尾部样本的预测精度，提升 Top30 选股准确率
  - CLI 参数：`--rank-weight-topk`（默认30）、`--rank-weight-weight`（默认5.0）、`--no-rank-weight`
  - 默认**开启**；所有参数记录到 `ml_train_runs.csv` 便于回溯


### 核心功能

- ✅ **完整的项目骨架**: 模块化设计，易于扩展
- ✅ **TuShare数据接入**: 自动拉取交易日历、股票列表、日线行情、财务指标
- ✅ **Parquet存储**: 高效的列式存储，加速数据读取
- ✅ **回测引擎**: 支持日/周/月频调仓，**支持自定义天数调仓**（如每5天、10天）
- ✅ **T+1 交易规则**: T 日生成信号，T+1 日收盘价买入，T+n 日卖出（收盘价或开盘价可配置）
- ✅ **可配置卖出时机**: **支持 T+n 日开盘价卖出或收盘价卖出**，默认收盘价卖出（新增）
- ✅ **涨跌停与停牌处理**: **信号生成时基于T+1数据过滤并回填，确保top N可交易**（优化）
- ✅ **实时进度显示**: 回测时使用 tqdm 进度条实时显示当前日期、净值、耗时，**支持详细日志开关**
- ✅ **仓位补齐机制**: **调仓后未满仓时自动在补齐窗口期内尝试补齐，确保回测实盘一致**（新增）
- ✅ **止损机制**: 回撤止损、移动止损、连续跌停止损（可选）
- ✅ **权益曲线交易（ECT）**: **基于账户盈亏曲线的仓位/风险管理**（新增 v0.3.5）
  - 回撤分档控制仓位
  - 净值均线趋势过滤
  - 风险解除后逐步恢复仓位
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

### v0.3.5 更新内容（2026-02-05）

**权益曲线交易（Equity Curve Trading, ECT）** - 基于账户盈亏曲线的仓位/风险管理：

#### 核心功能
- **回撤分档控制**: 根据净值回撤程度分档降低仓位
  - 可配置多档回撤阈值（默认：5%、10%、15%、20%）
  - 对应不同仓位系数（默认：0.8、0.6、0.4、0.2）
- **均线趋势过滤**: 基于净值短期/长期均线判断趋势
  - 短期均线高于长期均线：允许持仓（系数1.0）
  - 短期均线低于长期均线：降低仓位（系数0.5）
- **逐步恢复机制**: 风险解除后仓位阶梯式回升
  - gradual 模式：每个调仓周期按步长增加仓位（默认0.1）
  - immediate 模式：立即恢复满仓
  - 可配置恢复前等待周期数
- **组合决策**: 同时考虑回撤和均线，取较小值（更保守）
- **适用范围**: 同时支持纸面交易和回测

#### 使用示例

**纸面交易**:
```bash
# 配置 ECT 参数
python scripts/paper_trade.py config \
  --buy-price close --sell-price close \
  --top-n 5 --initial-capital 500000 --rebalance-freq 5 \
  --equity-curve-enabled \
  --equity-curve-drawdown-thresholds 5.0 10.0 15.0 \
  --equity-curve-exposure-levels 0.8 0.6 0.4 \
  --equity-curve-ma-short 5 \
  --equity-curve-ma-long 20 \
  --equity-curve-recovery-mode gradual \
  --equity-curve-recovery-step 0.1

# 运行（ECT 会自动生效）
python scripts/paper_trade.py run --trade-date 20260205
```

**回测**:
```bash
python scripts/run_ml_backtest.py \
  --start-date 20230101 --end-date 20231231 \
  --top-n 5 --rebalance-freq 5 \
  --equity-curve-enabled \
  --equity-curve-drawdown-thresholds 5.0 10.0 15.0 \
  --equity-curve-exposure-levels 0.8 0.6 0.4 \
  --equity-curve-ma-short 5 \
  --equity-curve-ma-long 20
```

**Python API**:
```python
from src.lazybull.risk.equity_curve import EquityCurveConfig, EquityCurveMonitor
from src.lazybull.backtest import BacktestEngine

# 创建 ECT 配置
ect_config = EquityCurveConfig(
    enabled=True,
    drawdown_thresholds=[5.0, 10.0, 15.0, 20.0],
    exposure_levels=[0.8, 0.6, 0.4, 0.2],
    ma_short_window=5,
    ma_long_window=20,
    recovery_mode='gradual',
    recovery_step=0.1
)

# 传入回测引擎
engine = BacktestEngine(
    universe=universe,
    signal=signal,
    equity_curve_config=ect_config,
    # ... 其他参数
)
```

#### 工作流程

1. **计算当前状态**:
   - 从历史 NAV 序列计算最大回撤
   - 计算短期/长期均线
   
2. **确定仓位系数**:
   - 回撤系数: 根据回撤档位确定
   - 均线系数: 根据均线趋势确定
   - 最终系数: 取两者较小值
   
3. **应用恢复逻辑**:
   - 降仓: 立即执行
   - 增仓: 按恢复策略逐步执行
   
4. **调整目标权重**:
   - 所有目标权重乘以仓位系数
   - 系数为0时清仓

#### 输出日志示例

```
ECT 计算结果: [20260205] ECT: 回撤 12.50% (触发), 均线趋势向下, 系数=0.40
ECT 仓位系数: 0.40
应用 ECT 系数 0.40 到目标权重
已将 ECT 系数应用到 5 个目标权重
```

### v0.3.2 更新内容（2026-01-23）

**仓位补齐机制** - 解决回测实盘买入涨停股票后处理不一致问题：

#### 核心功能
- **自动补齐未满仓位**: 调仓日后如果组合未满（目标TopN股票未全部买入），自动进入补齐流程
- **配置补齐窗口**: 默认3天补齐窗口期（可配置），在窗口期内持续尝试补齐
- **智能候选选择**: 
  - 优先选择原未成交股票（如果仍在候选中）
  - 如不可买入（继续涨停/停牌），则尝试其他候选股票
  - 基于上一交易日数据重新生成信号，避免未来函数
- **权重动态调整**: 
  - 支持等权和非等权目标权重策略
  - 原未成交股票使用原目标权重
  - 新候选股票平均分配剩余权重
- **放弃机制**: 超过补齐窗口仍未完成，放弃补齐，对应权重持币，等待下次调仓
- **详细日志**: 明确打印每个补齐操作（买入成功、失败原因、放弃补齐等）
- **统计信息**: 输出补齐统计（累计未满仓次数、补齐成功/放弃次数、补齐尝试次数）

#### 使用方法

```python
from src.lazybull.backtest.engine import BacktestEngine

engine = BacktestEngine(
    universe=universe,
    signal=signal,
    enable_position_completion=True,  # 启用仓位补齐（默认True）
    completion_window_days=3,          # 补齐窗口期3天（默认3）
    verbose=True                       # 输出详细补齐日志
)
```

#### 补齐流程示例

```
T日: 生成信号选出10只股票
T+1日: 尝试买入，实际成功8只（2只涨停）
  -> 记录未成交: 股票A、股票B，启动补齐流程
  
T+2日: 补齐尝试1
  -> 股票A可买入，补齐成功
  -> 股票B继续涨停，本次跳过
  
T+3日: 补齐尝试2
  -> 股票B仍涨停，本次跳过
  
T+4日: 补齐尝试3（最后一次）
  -> 股票B开板，补齐成功
  -> 补齐完成，仓位已满
  
如果T+4日股票B仍涨停:
  -> 超过3天补齐窗口，放弃补齐
  -> 对应权重持币，等待下次调仓（如T+5日）
```

#### 设计特点
- **回测实盘一致**: 补齐逻辑基于真实交易状态，不使用回测特有的"填充"逻辑
- **复用现有模块**: 利用 `trade_status.py` 检查交易状态，利用信号生成逻辑
- **向后兼容**: 补齐功能可配置开关，默认启用但不影响现有功能
- **最小化改动**: 在现有架构上扩展，不破坏原有逻辑

#### 技术实现
- 修改 `BacktestEngine` 添加补齐配置和状态跟踪
- 修改 `_generate_signal` 保存完整候选列表
- 修改 `_execute_pending_buys` 跟踪未成交槽位
- 新增 `_process_position_completion` 实现补齐逻辑
- 输出补齐统计信息


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
- **Breaking Change**: 不再支持字母频率（D/W/M），仅支持正整数
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

- ✅ **纸面交易（Paper Trading）**: 日频工作流，T0 生成信号，T1 执行打印，支持状态持久化 ⭐ 新增
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

# 步骤2: 使用 ML 模型运行回测（使用新的默认值）
# 默认：Top N=5, 初始资金=50万, 周频调仓, 排除ST
python scripts/run_ml_backtest.py --start-date 20230101 --end-date 20231231

# 自定义参数示例
python scripts/run_ml_backtest.py --start-date 20230101 --end-date 20231231 \
    --model-version 1 --top-n 10 --initial-capital 1000000

# 指定调仓频率（每20个交易日调仓一次，约1个月）
python scripts/run_ml_backtest.py --start-date 20230101 --end-date 20231231 \
    --rebalance-freq 20 --top-n 5

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
│   ├── factors/               # 因子库模块 ✅ v0.9.0
│   │   ├── technical_indicators.py  # 技术指标（RSI/KDJ/MACD/布林带）
│   │   ├── candlestick.py          # K线形态（振幅/上下影线）
│   │   ├── volatility.py           # 波动率
│   │   ├── industry.py             # 行业相关（alpha/偏离）
│   │   ├── momentum.py             # 动量加速度
│   │   └── volume.py               # 量能突变
│   ├── features/              # 特征构建模块
│   │   └── builder.py         # 特征构建器（调用 factors 模块）
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
  rebalance_frequency: 5  # 每5个交易日调仓一次

costs:
  commission_rate: 0.0003    # 万3佣金
  min_commission: 5          # 最低5元
  stamp_tax: 0.001           # 千1印花税
  slippage: 0.001            # 0.1%滑点
```

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
