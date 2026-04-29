# LazyBull - A股量化研究与回测框架

<div align="center">

**专注价值红利策略的量化投资框架**

[![Python](https://img.shields.io/badge/Python-3.9.13-blue.svg)](https://www.python.org/)
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

### 当前版本 (v0.67.3)

**树莓派 3.5 寸 LCD 的中证800数据源已按实测可用链路调整** (v0.67.3):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 盘中中证800实时数据改为使用 AKShare `stock_zh_index_spot` 接口
- 盘外中证800日线改为使用 TuShare `index_daily(ts_code=000906.SH)`，降低 AKShare 东财日线接口不可达导致的数据缺失
- [tests/test_respi_35lcd_disp.py](tests/test_respi_35lcd_disp.py) 已同步更新对应回归测试

**树莓派 3.5 寸 LCD 的中证800 AKShare 取数诊断日志已增强** (v0.67.2):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 为中证800实时与日线取数补充了接口级失败日志（接口缺失、调用异常、代码未命中、字段不匹配、空结果）
- 运行日志落盘到 `data/paper/state/respi_35lcd_runtime.log`（失败时自动回退到系统临时目录），便于在树莓派上直接定位为何“拿不到中证800”

**树莓派 3.5 寸 LCD 中证800折线显示与盘中刷新稳定性修复** (v0.67.1):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 修复了旧缓存缺少 `csi800_pct` 时白线覆盖黄线的问题：现在不会再把中证800线回退成上证线来画
- 同时修复了 AKShare 临时缺少中证800实时值导致盘中整次跳过追加的问题：会沿用上一采样点的中证800值，保证 13:00 后盘中曲线持续刷新
- [tests/test_respi_35lcd_disp.py](tests/test_respi_35lcd_disp.py) 已新增对应回归测试

**树莓派 3.5 寸 LCD 下方图表新增中证800折线（AKShare）** (v0.67.0):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 现在在下方图表区域显示第 4 条中证800折线，颜色为亮白色；并与原有 3 条线保持相同刷新频率
- 中证800显示策略与其他折线一致：盘内展示当日日内线，盘外展示周期日线
- 图例布局已压缩，图例线段缩短，文字统一为 `上 / 深 / 持 / 中`
- 中证800数据通过 AKShare 获取：日线用于周期图，实时涨跌幅用于盘中图

**树莓派 3.5 寸 LCD 的年化收益会按宽度自动缩小字号** (v0.66.13):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 现在只对左侧总览面板的“年化收益”单元格启用宽度自适应字号；当数值像 `+123.4%` 这样超过当前列宽时，会自动下调字号，避免文字伸出格子
- [tests/test_respi_35lcd_disp.py](tests/test_respi_35lcd_disp.py) 新增回归测试，约束超宽收益率文本会触发缩字，而普通宽度场景仍保留默认字号

**paper 空账户会自动对齐当前初始资金，避免 45w/50w 双基准** (v0.66.12):
- [src/lazybull/paper/account.py](src/lazybull/paper/account.py) 在加载“无持仓且 last_update 为空”的空账户状态时，会检查现金与当前配置 `initial_capital` 是否一致；不一致会自动同步并保存
- 这能覆盖 `reset-t0` 前后配置变更或旧状态残留导致的基准漂移，避免日志中“本轮按 450000，总盈亏按 500000”这种并存现象
- [tests/test_paper_trading.py](tests/test_paper_trading.py) 新增回归测试，约束该行为不被后续改动回退

**paper_trade 的“本轮盈亏”现已纳入本轮内已实现盈亏** (v0.66.11):
- [src/lazybull/paper/broker.py](src/lazybull/paper/broker.py) 新增统一口径 `calculate_round_pnl_metrics()`：优先使用 `last_rebalance_nav -> 当前总资产` 计算本轮收益率，因此本轮中因止损、提前换出或止盈而卖出的已实现盈亏会被正确计入
- [src/lazybull/paper/reporting.py](src/lazybull/paper/reporting.py) 复用同一口径，CLI 持仓打印与共享展示（如钉钉摘要）不再出现本轮收益率定义不一致
- 当历史状态里没有 `last_rebalance_nav` 时会自动回退旧公式（当前持仓浮盈 / 当前持仓成本），保证旧账户数据可直接兼容

**paper_trade 的基金持仓季度补齐现在会走轻量读取和缓存** (v0.66.10):
- [src/lazybull/data/storage.py](src/lazybull/data/storage.py) 现在支持按列读取分区 parquet，[src/lazybull/features/ensure.py](src/lazybull/features/ensure.py) 在补齐 `fund_portfolio` 时只会读取聚合真正需要的 5 列，不再整表进内存
- [src/lazybull/factors/fund_portfolio.py](src/lazybull/factors/fund_portfolio.py) 会先把季度原始明细瘦身到最小列集再聚合，同时把个股级季度结果缓存到 `data/raw/fund_portfolio_agg/`
- 这一步主要是压低树莓派在 T0 最后阶段处理 `fund_portfolio` 季度大分区时的内存峰值，避免在“基金持仓历史补齐”之后直接被撑爆

**paper_trade 的单日 point-in-time 因子现在会直接取快照** (v0.66.9):
- [src/lazybull/factors/fundamental.py](src/lazybull/factors/fundamental.py)、[src/lazybull/factors/holder.py](src/lazybull/factors/holder.py)、[src/lazybull/factors/earnings.py](src/lazybull/factors/earnings.py)、[src/lazybull/factors/express.py](src/lazybull/factors/express.py)、[src/lazybull/factors/fund_portfolio.py](src/lazybull/factors/fund_portfolio.py) 在 `trading_dates` 只有目标交易日时，都会直接基于可见公告取每只股票的最新记录，不再走逐股票 Python 列表回放
- 这比上一版只把输出日期缩到 1 天更进一步，连单日场景内部的 Python 双层循环也一并避开，尤其针对树莓派上基本面、股东人数、业绩预告这几段的长时间卡顿
- [tests/test_single_day_factor_snapshots.py](tests/test_single_day_factor_snapshots.py) 新增了单日快照与多日查询一致性回归测试，保证优化不改 point-in-time 语义

**paper_trade 在树莓派上的 T0 因子补齐更轻量了** (v0.66.8):
- [src/lazybull/features/ensure.py](src/lazybull/features/ensure.py) 现在仍会加载计算所需的历史原始数据，但不会再为整段历史交易日批量物化完整的因子查询表，而是只为目标 `trade_date` 生成当天截面
- 这次优化重点覆盖基本面、股东人数、业绩预告、业绩快报、基金持仓、北向资金、龙虎榜、一致预期等因子加载链路，能明显降低树莓派在 T0 流程“步骤2: 生成信号”阶段的 CPU 与内存峰值
- [tests/test_ensure_and_t0_printing.py](tests/test_ensure_and_t0_printing.py) 新增了回归测试，约束 ensure 层保持“只输出目标交易日”的调用方式

**paper_trade 的 early_exit_mode=disabled 现已恢复为“原硬卖”语义** (v0.66.7):
- [src/lazybull/paper/runtime.py](src/lazybull/paper/runtime.py) 现在只受 `enable_profit_based_holding` 总开关控制；当 `early_exit_mode=disabled` 时，仍会执行 `early_exit_loss_threshold + early_exit_holding_ratio` 的基础亏损提前换出检查，不再把整个 early_exit 分支直接跳过
- [scripts/paper_trade.py](scripts/paper_trade.py) 的启动摘要里，`early_exit_mode=disabled` 现在会显示为“亏损换出=原硬卖”，不再把 `disabled` 误读成“关闭提前换出”
- [src/lazybull/paper/storage.py](src/lazybull/paper/storage.py)、[data/paper/config.yaml](data/paper/config.yaml) 与 [scripts/batch_walk_forward.ps1](scripts/batch_walk_forward.ps1) 也已经同步改成“基础阈值在前、二次确认子开关在后”的说明结构，明确 `early_exit_loss_threshold / early_exit_holding_ratio` 受 `enable_profit_based_holding` 控制，而 `early_exit_mode` 只控制 `strength_veto` 保护分支

**walk_forward 与 paper_trade 的亏损提前换出默认值现已对齐** (v0.66.6):
- [src/lazybull/common/trading_config.py](src/lazybull/common/trading_config.py) 里公共 `TradingConfig` 和 CLI 的 `early_exit_mode` 默认值已经统一改为 `disabled`，不再与回测引擎默认值和 paper 配置模板打架
- [scripts/batch_walk_forward.ps1](scripts/batch_walk_forward.ps1) 现在会在启用盈亏动态持仓时始终显式传递 `--early-exit-mode`，即使当前实验配置选的是 `disabled`，也不会再因为参数省略而回退到公共默认值
- 这次修复对应的现象是：batch 配置区里明明写着 `early_exit_mode_list = @('disabled')`，walk_forward 却仍然按照 `early_exit_loss_threshold` 在持有 12 天时触发提前换出

**纸面交易门控现在既会生效，也更容易观察** (v0.66.5):
- [src/lazybull/paper/runner.py](src/lazybull/paper/runner.py) 现已把单股限权放到门控之前，与 backtest 链路保持一致；当设置 max_weight_per_stock 时，composite/legacy 门控与滚动质量降仓不再被后续归一化重新抬回满仓
- [src/lazybull/signals/ml_signal.py](src/lazybull/signals/ml_signal.py) 在 paper 路径下遇到“满仓通过”时也会打印门控摘要，不再只有持币或半仓才有门控日志
- [src/lazybull/paper/runtime.py](src/lazybull/paper/runtime.py) 现在会在盈利延续模式为 disabled 时显式打印跳过原因，避免把“总开关打开但子模式关闭”误读为功能没接上

**纸面交易补位买入不再错误放大持仓数** (v0.66.4):
- `src/lazybull/paper/runner.py` 的 `generate_replacement_targets()` 现在会在补位场景显式按失败缺口数覆盖 `top_n`，不再因为传入整套 `TradingConfig` 而继续沿用主仓配置里的 `top_n=20`
- 同时新增补位目标数量保护；即使下游候选筛选或门控意外返回超过缺口数的结果，保存到 `pending_buys` 前也会截断到缺口数
- 这次修复对应的现象是：T1 里只有 2 个或 5 个买入失败时，下一交易日不应再生成整整 20 个补位计划，更不应把总持仓从 18 只一路补到 33 只

**reset-t0 现在会清掉树莓派大屏缓存的日内图历史** (v0.66.3):
- `src/lazybull/paper/storage.py` 的 `reset_t0()` 已改为递归清理运行目录下的嵌套子目录，不再只删除 `state/` 根目录的直接文件
- 因此 `data/paper/state/respi_35lcd_intraday/` 里的树莓派日内图 JSON 也会被一并删除，断电重启后不会再把 reset 前的旧折线重新加载出来
- 这次修复对应的现象是：`paper_trade adjust reset-t0` 后，即使重启大屏脚本或整机断电重启，也不应再看到上一轮缓存下来的旧画线

**paper_trade 现已支持 next 交易日解析并持久化最近执行日** (v0.66.2):
- `src/lazybull/paper/runner.py` 现在会在共享运行时内部直接解析 `trade_date=next`，优先使用最近执行日，其次回退到账户最后更新日，再回退到从今天起的最近交易日
- `src/lazybull/paper/runtime.py` 执行完成后会保存本次实际交易日，因此后续继续执行 `paper_trade run --trade-date next` 时不会再把字面量 `next` 传入数据补齐和价格加载链路
- `src/lazybull/paper/reporting.py` 与 `scripts/paper_trade.py` 的持仓打印也会显示解析后的真实日期，不再出现 `[next]持仓情况` 或 `不支持的日期格式: next`

**paper_trade 成本配置缺键兜底，reset-t0 后 next 不再在 T0 阶段报错** (v0.66.1):
- `src/lazybull/paper/runner.py` 不再直接手读 `configs/base.yaml` 并硬索引 `capital_retention_ratio` / `pendding_capital_retention_ratio`，统一改走公共成本配置读取
- 当 `configs/base.yaml` 缺少这些键时，现在会自动回退到安全默认值：T0 `capital_retention_ratio=0.0`，补位 `pendding_capital_retention_ratio=0.3`
- `configs/base.yaml` 已补齐这两个默认项，因此 `paper_trade adjust reset-t0` 后重新执行 `paper_trade next` 不会再在“步骤3: 生成交易指令”阶段因缺键中断

**树莓派 3.5 寸 LCD 顶栏升级为 CPU/内存双槽血条** (v0.66.0):
- `scripts/respi/3.5LCD_disp.py` 的顶部状态栏文字继续上移 2-3 像素，顶栏底部整整一行现在是一条约 5px 高的手机电量样式双槽血条，左侧是 CPU，占右侧是内存，依然不显示额外字符说明
- CPU 与内存都由显示线程按 2 秒节流采样：CPU 读取 `/proc/stat`，内存读取 `/proc/meminfo`，不会改变现有摘要、排行、日内图和周期图的刷新节奏
- 两个分槽都采用统一的绿黄红分段阈值；自动息屏恢复时会重置采样基准，避免首帧直接显示到旧采样值

**模型注册表按需加载，避免纸面交易首次推理卡在大 JSON** (v0.64.2):
- `src/lazybull/ml/model_registry.py` 现在会优先读取 `v*_metadata.json` 与 `latest_model_version.txt` 旁路文件，指定版本加载不再默认整包解析 `model_registry.json`
- 对旧模型也增加了按 version 流式提取 metadata 的兼容路径，树莓派这类慢存储环境下，`paper_trade` 首次懒加载模型时不会再先卡在 90MB 级注册表读取上

**跨时间段稳定性汇总补齐参数分组键** (v0.64.1):
- `scripts/compare_walk_forward.py` 现在会把 `bt_top_n`、signal gate v2、市场择时、因子开关、分批调仓等 summary 已写出的参数一并带入 `实验对比` 和 `跨时间段稳定性` 汇总
- `wf_comparison_batches.xlsx` 的 `跨时间段稳定性` sheet 不会再把不同 `bt_top_n` 或其他遗漏的扫描参数误合并为同一条记录

**paper_trade 与 bot_service 共用纸面交易运行时** (v0.64.0):
- `src/lazybull/paper/runtime.py` 现在统一承载纸面交易日执行编排，完整覆盖止损、亏损提前换出、整体止盈、延迟卖出、T1、T0 与明日指令整理
- `src/lazybull/paper/reporting.py` 现在统一承载模型信息、持仓快照、手机 Markdown 持仓展示与交易结果展示，`paper_trade.py` 与 `bot_service.py` 直接复用
- `bot_service.py trade / positions / model` 不再维护独立逻辑；`paper <paper_trade 子命令...>` 还可直接透传低频 CLI 子命令，减少后续同步维护成本

**batch_walk_forward 参数区按开关关系重排** (v0.63.3):
- `scripts/batch_walk_forward.ps1` 现在会把某个总开关和它真正控制的参数放在一起展示，例如盈亏动态持仓、ATR 动态阈值、strength_veto、止损、ECT、动态 Top-N、行业过滤、市场择时等
- 每组参数旁都补了“仅在何种开关 / 模式下生效”的中文注释，便于直接判断哪些参数当前有效、哪些只是备用扫描项

**纸面交易配置按开关控制关系分组展示** (v0.63.2):
- `data/paper/config.yaml` 现在会把某个开关和它真正控制的参数放在同一小段里，例如止损、ECT、市场择时、行业过滤、盈利延续、动态 Top-N 等
- `paper_trade.py config` 刷新 YAML 模板时也会保持这种“开关在前、受控参数紧跟”的排布，便于直接判断哪些参数当前有效

**纸面交易仅保留 YAML 主配置** (v0.63.1):
- `data/paper/config.yaml` 现在是唯一需要维护的纸面交易配置文件
- `PaperStorage.load_config()` 不再回退读取 `config.json`，`paper_trade.py config` 也不再生成 JSON 快照
- `reset_t0` 后保留的也是 `config.yaml`，配置目录更干净

**纸面交易主配置切换为带中文注释的 YAML 模板** (v0.63.0):
- `data/paper/config.yaml` 现在是纸面交易的主配置入口，按模型、门控、组合、止损、ECT、择时、行业、仓位管理等模块分段展示，并带中文注释
- `PaperStorage.load_config()` 会优先读取 `config.yaml`
- `paper_trade.py config` 保存时也会同步刷新这份 YAML 模板，便于你按 `batch_walk_forward.ps1` 风格长期维护

**纸面交易支持直接编辑全量配置文件** (v0.62.0):
- 核心配置已迁移到更适合手工编辑的 `data/paper/config.yaml`
- `PaperStorage.load_config()` 会自动兼容旧版 `weight_method` 字段并补齐缺失的新参数，避免因为 CLI 漏参导致配置不完整
- `paper_trade.py run` 在找不到配置时也会明确提示直接编辑 `data/paper/config.yaml`

**单次回测 / walk_forward / 纸面交易策略参数与运行时进一步对齐** (v0.61.0):
- `scripts/run_ml_backtest.py` 与 `scripts/walk_forward.py` 现在共用统一的回测运行时工厂，单次回测不再维护一套独立的 BacktestEngineML 长参数透传
- 纸面交易接入 `TradingConfig` 统一透传、滚动质量监控、动态 Top-N、行业轮动加权、Kelly/半Kelly、整体止盈、空仓提前调仓，并恢复延迟卖出的 T+1 语义
- 公共 signal 工厂现已支持 `model_version_b + ensemble_weight_a` 双模型加权集成，单次回测和纸面交易可直接复用

**跨时间段稳定性汇总新增运行ID列表** (v0.60.5):
- `wf_comparison_batches.xlsx` 的 `跨时间段稳定性` sheet 现在直接输出 `运行ID列表`
- 格式为 `批次时间段:运行ID`，例如 `0101:wf_xxx | 0209:wf_yyy`，可直接从稳定性汇总定位到底层 run

**raw 空目录也会生成占位报表** (v0.60.4):
- 无参 `compare_walk_forward.py` 在 `data/walk_forward/raw` 没有 summary 文件时，仍会生成 `data/walk_forward/wf_comparison_raw.xlsx`
- 占位文件会标明“无可用数据”，从而保证 batch 结束后 `raw / batches` 两份报表路径都存在

**跨时间段稳定性汇总按批次隔离 + batch 自动刷新两份总表** (v0.60.3):
- `wf_comparison_batches.xlsx` 的 `跨时间段稳定性` sheet 现在会按 `batch_run_id` 分开汇总；同参数重复跑两次 batch 不会再只剩一个批次ID
- `scripts/batch_walk_forward.ps1` 在生成当前批次的 `wf_comparison.xlsx` 后，会继续自动刷新 `data/walk_forward/wf_comparison_raw.xlsx` 与 `data/walk_forward/wf_comparison_batches.xlsx`

**compare_walk_forward 无参自动扫描 raw 与 batches** (v0.60.2):
- 直接执行 `py .\scripts\compare_walk_forward.py` 时，脚本会自动扫描 `data/walk_forward/raw` 与 `data/walk_forward/batches/*/raw`
- 默认分别生成 `data/walk_forward/wf_comparison_raw.xlsx` 与 `data/walk_forward/wf_comparison_batches.xlsx`，不再要求手工传 `--raw-dir`

**wf_comparison.xlsx 恢复输出到固定目录** (v0.60.1):
- `scripts/batch_walk_forward.ps1` 仍将本批次原始结果写入 `data/walk_forward/batches/<batch_id>/raw/`，保证 compare 只汇总本批次
- 最终对比 Excel 改回输出到 `data/walk_forward/wf_comparison.xlsx`，不再跟随批次目录变化

**walk_forward batch 支持多时间段与批次隔离汇总** (v0.60.0):
- `scripts/batch_walk_forward.ps1` 新增 `wf_period_configs`，可在同一批次内配置多组时间段；`skip_training` 下每个时间段独立配置 `StartModelVersion`
- 批量运行产物改为写入 `data/walk_forward/batches/<batch_id>/raw/`，随后仅对当前批次运行 `compare_walk_forward.py`，不再混入历史 raw 实验
- `compare_walk_forward.py` 新增 `跨时间段稳定性` sheet，按相同参数组合聚合不同时间段表现，输出时间段列表、综合得分均值/标准差、跨时间段 CAGR/回撤/跨切分 IR 与稳定性分

**精简每日特征构建日志** (v0.59.3):
- 删除 `开始构建 {trade_date} 的特征` 冗余行（上游已有更醒目的分隔条 ETA 日志）
- 3 个 horizon 的 `y_ret_N` 缺失 warning 合并为一行汇总，日志行数由 4 行降至 0~1 行

**特征构建进度日志增加 ETA 与分隔** (v0.59.2):
- `scripts/build_clean_features.py` 每日特征构建起始日志前增加换行与 `=====` 分隔条，作为生成一个 feature 文件的明显起点标志
- 基于已处理日期平均耗时线性外推，附带打印"预计完成"绝对时间，便于长任务安排下游工作

**抑制 storage.py 中 pandas concat FutureWarning** (v0.59.1):
- `load_raw_by_date_range` / `load_clean_by_date_range` 在合并多日分区时，pandas 1.5+ 对含 all-NA 列的 concat 会输出 FutureWarning（典型场景：龙虎榜 `reason` 字段在某些日子整列为 NaN）
- 用 `warnings.catch_warnings()` 局部抑制该特定 message，对其他 warning 无影响；对结果数据正确性无影响

**y_ret_N 标签语义对齐回测节奏** (v0.59.0):
- 旧公式 `close_adj(T+N) / close_adj(T) - 1` 假设 T 收盘买/T+N 收盘卖, 与回测引擎的 "T+1 收盘买入 → T+1+holding_period 开盘卖出" 不一致
- 新公式 `y_ret_N = open_adj(T+1+N) / close_adj(T+1) - 1`, 严格对齐回测引擎的实际成交价口径, 消除训练-回测之间的隐性偏差
- 配套调整: `train_core.py` 验证集切分 delta 由 `max(horizon, 5)` 改为 `max(horizon + 1, 5)`, 杜绝隔夜跳空潜在泄露
- 旧的 `data/features/cs_train/` 标签语义已变, 需删除并重跑 `build_clean_features.py` 重建

**下载并发 + 限流感知重试** (v0.58.0):
- **线程安全令牌桶**: `TushareClient._rate_limit_wait` 加锁, 多线程共享同一限频队列, 全局 QPS 受 `rate_limit` 严格约束, 不会超过 TuShare 官方配额
- **限流感知重试**: 识别"每分钟/访问/频次/rate/limit/429"等关键字 → 长等 `retry_rate_limit_sleep` (默认 15s); 其他错误 → 短等固定 `retry_delay`。消除原先 1+2+3=6 秒指数退避的雪球效应
- **按日并发**: `daily` / `margin_detail` / `cyq_perf` / `top_list` 并发下载, 预计 2012-2026 全量下载从 24h+ 压缩到 4-6h
- **两级降级**: `base.yaml → tushare.download_concurrency: 4` 全局; `--concurrency 1` 临时覆盖; 触发限流时改 1 即退化回串行

**download_raw.py 全量重写 — 错误汇总 + ETA 进度 + 多项隐患修复** (v0.57.0):
- **错误汇总**: 新增全局 `ErrorCollector`, 单条失败不中断, 脚本结束时在总结页统一列出所有错误, 无人值守场景可离线查日志; `finally` 保证异常/Ctrl+C 也会打印
- **ETA 进度**: 新增 `ProgressTracker`, 基于已完成项平均耗时估算剩余时间, 每 N 项打印 `elapsed / rate / ETA / 预计完成时刻`; 日线日志量下降约 80%
- **13 项隐患修复**: 默认 end-date 硬编码未来日期、trade_cal 短窗口截断历史、moneyflow_hsgt 断点续传失效、日线 6 接口非原子性(半日缺失永久化)、moneyflow 强制依赖静默降级、字符串日期字典序比较、分页多余空请求、report_rc force 语义不一致、dedup 顺序不明、stock_basic 生存者偏差(仅拉 L)、KeyboardInterrupt 未单独处理 等
- **退出码规范化**: 0/1/3/130 分别对应成功/初始化异常/有错误项/用户中断

**修复调仓决策摘要"最终"行显示与计算不一致** (v0.56.3):
- 原先"最终"行仅展示 `信号门控 x ECT x 市场层`，漏掉质量系数，导致显示的乘积与实际 `final_target_exposure` 对不上（例如 `45.8%[50.0% x 100.0% x 100.0%]`）
- 修复后完整展开所有参与相乘的分项为 `信号门控=xx% x 质量=xx% x ECT=xx% x 市场层=xx%`，确保计算链条透明且与最终数值自洽

**调仓买入 warning 三行汇总日志** (v0.56.2):
- walk_forward / run_ml_backtest 共用的回测引擎会在每个 T+1 调仓买入日输出 3 行 warning 汇总日志：首行展示计划买入数、计划资金占比、继承上轮持仓数量与资金占比，以及成功/失败数量
- 第二、三行分别展示成功仓位和失败仓位，格式统一为“数量 + 股票号列表 + 总资金占比”；失败项会保留涨停、停牌等原因，便于扫日志时快速定位问题

**另类因子扩展 — 北向资金 / 龙虎榜 / 一致预期** (v0.56.0):
- 新增 3 大另类因子模块，三者各自配备独立开关（默认关闭，保持基线行为）
  - **北向资金（north_flow）**：`moneyflow_hsgt` 市场级日度净流入，广播后做 5/20 日均、20 日 z-score、连续同方向天数等截面变换
  - **龙虎榜（lhb）**：`top_list` 个股日频上榜数据，同日多次上榜聚合，辅以 5/20 日滚动净额与 20 日上榜次数
  - **一致预期（consensus）**：`report_rc` 研报滚动聚合，90 日分析师覆盖 / FY1 EPS 均值 / 30 日 EPS 修订比例 / 目标价 / 评级五档量化
- 开关贯通：`batch_walk_forward.ps1` 新增 `$enable_north / $enable_lhb / $enable_consensus`；`walk_forward.py / train_ml_model.py / build_clean_features.py` 新增 `--enable-{north,lhb,consensus}-features`；`build_clean_features.py --build-all` 自动覆盖新 3 个因子
- 自动补齐：`features/ensure.py` 按日分区（hsgt / top_list）与按年分页（report_rc）增量下载

**行业轮动加权 + Kelly/半Kelly 仓位管理** (v0.55.0):
- 行业轮动加权：按行业动量排名对候选股票分数做乘性调整，强势行业股票获得分数加成，弱势行业分数下调但超强个股仍可入选（独立于硬过滤开关）
- Kelly / 半 Kelly 仓位管理：基于 `f* = μ/σ²` 的非线性仓位分配，低波动高分数的股票自动获得更高仓位；半 Kelly 更保守（仓位减半）；缺失数据优雅回退
- 两项优化均支持 `batch_walk_forward.ps1` 独立开关扫参，可通过 A/B 实验验证效果

**树莓派 3.5 寸 LCD 日内图边界点显示对齐** (v0.54.1):
- `scripts/respi/3.5LCD_disp.py` 的日内图显示层新增了边界吸附：`11:30` 与 `13:00` 的点会贴住午休虚线，`15:00` 的点会贴住最右侧边界，不再出现“差一点点才碰到边界”的视觉缝隙
- 这次修正只作用于绘图时的显示坐标，不改变实时采样、持久化历史和午休折叠逻辑

**树莓派 3.5 寸 LCD 日内图轻度平滑 + 抗锯齿渲染** (v0.54.0):
- `scripts/respi/3.5LCD_disp.py` 在日内图显示层新增了很轻的三点平滑，只作用于画线外观，不改变每次刷新记点、持久化历史和顶部展示数值
- 折线会先在更高分辨率的小画布上绘制，再缩回图表区域，因此高频刷新后原先那种“坑坑洼洼的小锯齿”会明显减轻
- 这次调整只作用于日内图显示层；实时采样频率、点位持久化和午休折叠逻辑都保持不变

**树莓派 3.5 寸 LCD 日内图改为每次刷新都记点** (v0.53.0):
- `scripts/respi/3.5LCD_disp.py` 的日内图现在不再按 10 分钟槽位覆盖同槽位内的新值，而是为每一轮实时刷新都保留一个采样点；同一个 10 分钟槽位里的多次刷新会完整保留下来
- 图表横坐标新增按真实盘中时间折叠后的 `x_positions`，因此这些新增点会沿 x 轴展开，午休仍然折叠，但盘中折线会更细、更接近真实刷新节奏
- 当日日内图持久化读回时也不再按槽位去重，脚本重启后仍会保留上午或下午同槽位内累积下来的多次刷新历史

**纸面交易功能对齐回测引擎** (v0.52.0):
- 将回测中已验证的高级功能移植到纸面交易：市场择时仓位管理（4 种模式 + MA250 + 回撤保护）、行业动量过滤、持仓保留奖励、盈利延续持有（pnl/strength）、亏损提前换出（支持 strength_veto 缓刑 + ATR 动态阈值）
- 所有新参数支持通过 `paper_trade.py config` 持久化，默认值与 `batch_walk_forward.ps1` 保持一致
- Position 新增 `buy_atr_pct` 字段，T1 买入自动记录 ATR 用于动态止损
- 统一使用 `create_signal(TradingConfig)` 创建信号，确保门控参数完整穿透

**亏损提前换仓二次确认门控 strength_veto** (v0.51.0):
- 亏损提前换出（early_exit）新增 `early_exit_mode` 参数，支持 `disabled`（原硬卖，默认兼容）和 `strength_veto`（二次确认门控）两种模式
- strength_veto 模式下，触发亏损阈值后用 `HoldingStrengthScorer` 评分，评分高于保护阈值时否决卖出（"缓刑"），防止把"暂时回调但趋势仍在"的股票过早换出
- 通过 `early_exit_max_reprieves`（默认 2）限制最大缓刑次数，防止无限拖延
- walk_forward / batch_walk_forward / compare_walk_forward 全链路透传，支持扫参对比

**盈利延续持有判据升级为多维度强势度评分** (v0.50.0):
- 原 `profit_extension_threshold` 单一浮盈率判据升级为可配置的 `profit_extension_mode`，支持 pnl（默认兼容）/ strength（5 维度强势度评分）/ disabled 三种模式
- strength 模式综合 5 个维度：ML 分数 30% + 动量加速 25% + 技术强度 15% + 资金筹码 15% + 回撤距离 15%，评分 ≥ 阈值（默认 0.6）才延续持有
- 完全向后兼容：默认 `mode="pnl"` 保持原有行为，显式启用 `--profit-extension-mode strength` 才激活新机制
- walk_forward / batch_walk_forward 全链路透传，支持扫参对比 pnl/strength 两种模式

**树莓派背光调节的硬件前提提示** (v0.49.1):
- 针对微雪 3.5inch RPi LCD (C)，脚本现在会明确提示一个关键硬件前提：官方要求先用 0R 电阻或焊锡接通背光控制焊盘，GPIO18 的 PWM 调光才会真正生效
- 如果这一步硬件改动没有做，即使 `scripts/respi/set_backlight.py` 输出“已通过 PWM 设置背光为 5%/10%/100%”，屏幕亮度也可能完全不变
- `scripts/respi/3.5LCD_disp.py` 现在也会输出同样的说明，因此主脚本与独立脚本在这个问题上的结论是一致的

**树莓派背光调节脚本默认显示亮度测试图** (v0.49.0):
- `scripts/respi/set_backlight.py` 在成功设置亮度后，默认会向 framebuffer 写入一张高对比度测试图，包含彩条、灰度条和棋盘块，方便你直接盯着屏幕看亮度变化
- 如果你只想调亮度、不想覆盖当前屏幕内容，可以显式加 `--no-preview`
- 预览目标也可通过 `--fb-path`、`--fb-width`、`--fb-height` 调整；默认仍按树莓派 3.5 寸 LCD 的 `480x320` 和 `/dev/fb1`

**树莓派 3.5 寸 LCD 主脚本背光路径与独立脚本对齐** (v0.48.2):
- `scripts/respi/3.5LCD_disp.py` 的背光控制现在直接复用 `scripts/respi/set_backlight.py` 同一套 helper，因此主显示脚本和你单独调亮度时走的是完全一致的后端选择逻辑
- 如果树莓派没有 `/sys/class/backlight` 节点，主 LCD 进程也会和独立脚本一样优先走 `lgpio` PWM，而不再卡在旧的 `RPi.GPIO` 路径上
- 这意味着只要你单独执行 `python scripts/respi/set_backlight.py 10` 能成功，主脚本理论上也会用同样的方式设置到相同亮度

**树莓派 LCD 背光脚本节点自动发现 + lgpio 回退** (v0.48.1):
- `scripts/respi/set_backlight.py` 不再只认固定的 `soc:backlight` 路径，而是会自动扫描 `/sys/class/backlight` 下所有可用背光设备；如果你的屏幕驱动导出的节点名不同，`--read` 和默认 `auto` 模式也能直接命中
- 当 sysfs 背光节点不存在时，脚本现在会优先尝试 `lgpio` 直接发 PWM，再回退到 `RPi.GPIO`，更适合 Bookworm / `rpi-lgpio` 环境
- 新增 `--list`、`--backlight-name`、`--gpiochip` 参数，便于你在树莓派现场先枚举设备，再逐个测试背光控制方式

**树莓派 LCD 独立背光调节脚本** (v0.48.0):
- 新增 `scripts/respi/set_backlight.py`，可以单独调树莓派 LCD 背光，不需要先启动 `3.5LCD_disp.py`
- 默认优先走 sysfs 背光节点；如果你的屏幕驱动没有挂出 sysfs 节点，也可以用 `--method pwm` 切到 GPIO PWM 方式测试亮度
- 常用示例：`python scripts/respi/set_backlight.py 20`、`python scripts/respi/set_backlight.py --read`、`python scripts/respi/set_backlight.py 15 --method pwm`

**树莓派 3.5 寸 LCD 午休前 11:30 尾点补齐修复** (v0.47.10):
- 日内图现在会在跨过 11:30 边界后保留一个很短的补尾窗口；如果上午最后一次常规轮询停在 11:29 左右，数据线程仍会立即再尝试一次，把上午收盘前的最后一个 10 分钟槽位补齐
- 午休刚开始时，数据线程不会因为进入休市就立刻睡到 13:00；只要 11:30 那一格还没补齐，就会继续按短间隔保留补抓机会
- 顶部 `更新:HH:MM` 现在优先显示实时行情自带的 `quote_time`，因此午休前最后一笔显示会更接近真实行情时间，而不是单纯的本地抓取时刻

**树莓派 3.5 寸 LCD 顶栏状态文案与零线细节优化** (v0.47.9):
- 顶栏调仓提示改为 `下次调仓:MM/DD/剩n天`，直接展示下一次调仓日期和剩余交易日，信息量比原来的“待调仓:n天”更完整
- 数据线程正在下载或处理时，顶部中间会短暂显示 `更新中...`；处理完成后恢复为 `更新:HH:MM`，能区分“上次更新时间”和“当前正在刷新”
- 日内图和周期图的 0% 参考线统一改为白色，并轻微下移 1 像素；周期图右上角角标也缩短为 `数据日:MM/DD`，整体更清爽

**树莓派 3.5 寸 LCD 收盘后延迟切回周期图** (v0.47.8):
- 收盘后不再在 `15:00` 一过就立刻从日内图切回周期图；如果周期图还没拿到当日数据，会继续显示当日日内折线，避免看到仍停留在前一日的周期图
- 日内图会在收盘后保留一个短的补齐窗口，继续尝试用 snapshot 里的 `quote_time` 补上 `15:00` 最后一格，不再轻易停在 `14:58` 或 `14:59`
- 只有当周期图已经拿到当日数据后，显示才会从日内图切回周期图；如果日内尾点已经补齐但周期图还未更新，则会继续显示静态的当日日内图等待切换

**树莓派 3.5 寸 LCD 午休分隔标记 + 0% 零线强化** (v0.47.7):
- 日内图在 11:30 与 13:00 的折叠边界上新增一条很淡的午休分隔虚线，并在 x 轴中部补一个“午休”标记，让折叠后的上午和下午衔接更直观
- 0% 参考线改成独立颜色，并额外加了一个小的“0%”标签，不再和普通网格线混在一起，日内走势的正负基准会更容易辨认
- 这些增强仅作用于图表显示层，不改变现有日内收益口径、持久化结构和刷新策略

**树莓派 3.5 寸 LCD 午休折叠显示 + 0% 参考线常显** (v0.47.6):
- 日内折线图的 x 轴改为只覆盖上午和下午两个实际交易时段，11:30~13:00 午休区间不再占用中间一大段水平空间，全天折线会更紧凑地拼接成连续走势
- 图表 y 轴范围现在会始终包含 0%，因此 0% 参考线会固定显示，即使三条线全天都在正区间或负区间运行也一样可见
- 旧版日内持久化数据仍兼容；带时间标签的历史点会在加载时按新槽位重新映射，午休不会再被画成一条很长的直线

**树莓派 3.5 寸 LCD 指数实时行情并入 snapshot + 周期图日线缓存** (v0.47.5):
- 上证和深证实时行情现在直接并入同一轮持仓 `realtime_quote` 结果，日内图优先复用 snapshot 里的指数涨跌幅，不再额外发起一次指数实时接口请求
- 周期图新增“同日且目标交易日已齐”缓存：同一天内、账户状态和调仓状态未变化时，重复刷新会直接复用已构建好的周期图 payload，减少重复 `daily` 和 `index_daily` 调用
- 收盘后如果当天周期数据尚未落库，缓存不会提前固化缺失结果；数据线程仍会按原有 10 分钟节奏继续重试，直到拿到最近应有交易日的数据

**树莓派 3.5 寸 LCD 持仓实时行情复用 + 分频刷新** (v0.47.4):
- `summary`、个股排行、日内折线现在复用同一份持仓实时行情快照，不再各自重复请求一次持仓 `realtime_quote`
- 盘中有效交易时段内，摘要/排行/日内图改为每 2 分钟刷新一次；开盘和午后开盘切换到新会话时会立即补刷
- 非交易时段只在周期图尚未拿到“最近一个应有交易日”数据时，每 10 分钟尝试补抓一次；拿到后停止请求，直到下一个待补交易日出现

**树莓派 3.5 寸 LCD 日内图恢复为前收基准** (v0.47.3):
- 修复 0.47.1 把日内显示值改成“相对当日首个有效点”的问题，这会让曲线看起来像按开盘价计算
- 现在日内图重新按前一交易日收盘价为基准显示，三条线的数值与原始当日涨跌幅保持一致
- 旧版日内持久化 JSON 仍兼容，加载后会保留 `raw_*` 字段，但显示序列不再做首点归零重写

**树莓派 3.5 寸 LCD 盘前不再误用无效实时价** (v0.47.2):
- 修复盘前或午休时段误把无效 `PRICE` 当成实时价，导致总览区出现 `-100%`、排行全绿 `-100%` 的异常显示
- 日内图显示窗口改为实际开盘后的 9:30-15:00，盘前不会再切到日内图；实时刷新只在有效交易时段进行
- 持仓总览与个股排行在遇到 `PRICE<=0` 时会回退到 `PRE_CLOSE`，旧版盘前日内脏点也会在加载持久化文件时自动清掉

**树莓派 3.5 寸 LCD 日内图改为当日起点归零** (v0.47.1):
- 修复日内图使用“相对昨收的当日涨跌幅”直接绘制，导致首个点不是 0、视觉上像把前一日收盘口径带进盘中的问题
- 现在日内图内部保留原始当日涨跌幅，但显示时按当日首个有效点归零，三条线都会从 0 开始，仅比较盘中变化
- 兼容已有的日内持久化 JSON：旧文件会在加载时自动重建为“原始值 + 归零显示值”两套数据

**P1优化: 因子增强 + 多子集集成 + 模型质量监控** (v0.47.0):
- 因子增强（`--enable-enhanced-features`）：新增开盘强度、日内波动结构、委托不平衡3个短线因子，经行业z-score标准化
- 多特征子集集成（`--subset-ensemble`）：动量/基本面/资金流3个子模型独立训练后加权融合，提升模型多样性
- 模型质量监控（`--model-quality-enabled`）：val_rankic_ir低于阈值时自动回退上一合格模型，避免质量退化传导到回测
- 清理已废弃的双模型信号代码（EnsembleMLSignal），SubsetEnsembleModel对MLSignal透明
- 三项功能均有独立开关，`batch_walk_forward.ps1` 中提供配置变量

**树莓派 3.5 寸 LCD 收盘切图后立即补刷周期图** (v0.46.2):
- 修复 15:30 从日内图切到周期图时，周期图仍短暂停留在 t-1 数据的问题
- 根因是数据线程按脚本启动时刻每 10 分钟轮询，昨天的修复保证了收盘后会继续检查，但没有在切图边界立刻补一次周期图刷新
- 现在数据线程在接近 15:30 时会缩短本次等待，并在离开日内窗口后立即再拉一次周期图数据，减少切图瞬间看到旧缓存的时间窗

**树莓派 3.5 寸 LCD 日内图脏点清洗与图例精简** (v0.46.1):
- 日内持仓收益在遇到 `PRICE<=0` 或明显异常实时价时会回退到昨收，避免单个坏点把整张日内图压扁
- 加载当天日内历史时会过滤异常指数/持仓点，已经写入 JSON 的脏点不会继续参与缩放
- 日内图例缩短为 `上证 / 深证 / 持仓`，且日内模式不再显示 `周期图最后数据日` 角标，顶部重叠明显缓解

**持仓保留奖励 & 市场自适应 Top-N** (v0.46.0):
- 新增持仓保留奖励（Holding Bonus）：调仓时对已持仓股票在截面分数上加分，降低不必要的换手；保留的持仓自动延续持有期，不产生交易成本
- 新增市场自适应 Top-N：根据近 20 日市场平均收益判断趋势，牛市集中持股、熊市分散持股
- 两个功能均可通过 CLI 参数独立开关，`batch_walk_forward.ps1` 中提供配置变量

**树莓派 3.5 寸 LCD 顶栏与左右面板布局微调** (v0.45.7):
- 顶部时间单独提高字号，并把顶栏高度略微上调；新增的高度从底部图表区扣减，不压缩中间数据面板
- 左侧总览面板宽度从 60% 调整到 55%，给右侧盈亏排行两个区域更多横向空间，长股票名和代码更不容易显得拥挤

**树莓派 3.5 寸 LCD 首帧渲染不卡交易日历加载** (v0.45.6):
- 修复显示线程首帧会在交易日判断里懒加载交易日历的问题，避免与数据线程启动期的大导入互相阻塞，导致屏幕停留在 `LCD启动中`
- 首帧渲染新增 `开始首帧渲染` / `已写出首帧` 一次性日志，便于在 SSH 上直接确认是否已经越过正式渲染
- 交易日历若尚未加载，LCD 显示逻辑先按工作日快速判断，不再为了精确节假日口径阻塞首屏显示

**树莓派 3.5 寸 LCD 启动测试页 + framebuffer 可切换** (v0.45.5):
- 脚本启动后会先尝试写入一张 `LCD启动中` 测试页，并在 SSH 中打印当前目标 framebuffer 和系统可见的 `/dev/fb*` 列表，便于快速判断脚本是否已经进入显示链路
- 新增环境变量 `LAZYBULL_LCD_FB_PATH`，可在树莓派上直接切换写入设备，例如 `LAZYBULL_LCD_FB_PATH=/dev/fb0 python ./scripts/respi/3.5LCD_disp.py`
- 主程序、配置加载、背光初始化、线程启动等关键阶段都改为同步输出到 SSH，避免只看到一行启动消息

**树莓派 3.5 寸 LCD 启动与 framebuffer 诊断增强** (v0.45.4):
- 修复 `/dev/fb1` 写入失败被静默吞掉的问题；现在如果 framebuffer 设备不存在、权限不对或设备号变化，会把失败原因写到 stderr 和运行诊断日志
- 新增启动阶段诊断日志，覆盖主程序启动、线程启动、自动息屏命中、背光初始化和 framebuffer 可用性，便于在“屏幕全黑且 SSH 没输出”时继续定位
- 诊断日志优先写入 `data/paper/state/respi_35lcd_runtime.log`，若项目目录不可写则兜底写到系统临时目录

**回测信号生成时排除已持仓股票** (v0.45.3):
- 修复"空仓/持有期拖尾提前调仓"场景下，残留持仓被信号重复选中导致 T+1 买入被跳过、槽位浪费的问题
- 信号生成阶段无条件从候选中排除当前 `self.positions`，让后续候选顶上空出的槽位
- 原仅在分批调仓（`stagger_tranches>1`）时过滤；现在单批模式同样过滤，常规场景等效于"先卖后买"

**树莓派 3.5 寸 LCD 黑屏诊断增强** (v0.45.3):
- 若显示线程出现运行时异常，脚本不再静默黑屏，而会直接在屏幕上显示 `LCD显示异常` 与异常类型/摘要，便于现场定位问题
- 同时会向终端输出简短错误信息，方便通过 SSH 或前台运行时查看

**树莓派 3.5 寸 LCD 周期图最后数据日显示 + 盘外刷新收口** (v0.45.2):
- 图表角落新增 `周期图最后数据日:MM/DD` 提示，便于直接确认周期图最新点已经更新到哪一天
- 盘外不再一直轮询刷新周期图：仅在交易日收盘后、且周期图还没拿到当天数据时继续按 10 分钟检查；一旦拿到当天数据，就停止盘外刷新
- 盘中 `8:30-15:30` 仍保持周期图和实时数据同步刷新

**树莓派 3.5 寸 LCD 盘外周期图晚间刷新修复** (v0.45.1):
- 修复盘外持仓周期折线图只在脚本启动时和盘中 8:30-15:30 刷新的问题，导致收盘后仍停留在昨天曲线
- 现在周期图会无论盘中盘外都按 10 分钟刷新一次；盘中实时数据仍只在交易日 `8:30-15:30` 刷新
- 因此收盘后当日 `index_daily` / 持仓日线数据可用时，无需重启脚本也会自动看到今天的新点

**树莓派 3.5 寸 LCD 三线图扩展（上证/深证/持仓）** (v0.45.0):
- 盘中图和盘外持仓周期图都新增深证指数曲线，图表统一显示“上证 + 深证 + 持仓”三条线
- 持仓曲线固定使用橘黄色；上证使用亮黄色，深证使用青蓝色，三条线和图例可明显区分
- 当日日内历史持久化文件同步升级为三线结构，脚本重启后会继续恢复当天的上证/深证/持仓三条历史曲线

**树莓派 3.5 寸 LCD 盘中图口径修正 + 当天历史持久化** (v0.44.0):
- 交易日 `8:30-15:30` 的盘中图改为显示“上证指数实时涨跌 vs 持仓股当日实时涨跌”，不再使用持仓周期累计涨跌作为盘中对比线
- 持仓股当日实时涨跌按持仓股票昨收市值加权计算，不含现金仓位，更符合盘中对比语义
- 盘中 10 分钟槽位历史会持久化到 `data/paper/state/respi_35lcd_intraday/YYYYMMDD.json`，脚本重启后会自动恢复当天已记录的历史点

**树莓派 3.5 寸 LCD 盘中/非盘中双图显示** (v0.43.0):
- 非交易日，以及交易日 `8:30-15:30` 之外继续显示持仓周期图，但 x 轴槽位固定为 `max(调仓周期, 当前持仓交易日数)`，持仓未满一轮时不再被压缩拉伸
- 交易日 `8:30-15:30` 之间切换为盘中图，按固定 10 分钟槽位显示上证指数当日涨跌幅和当前持仓组合相对本轮起点的累计涨跌幅
- 顶部时间显示改为 `4月7日(周二) 14:40:32` 格式，右上角文案由 `调仓:N天` 改为 `待调仓:N天`

**空仓/持有期拖尾提前调仓** (v0.42.0):
- 扩展 `enable_early_rebalance_on_empty` 语义：除空仓场景外，新增"持有期拖尾"触发路径
- 当 `cycle_day >= holding_period` 且仍有残留盈利延续持仓时，尝试生成新一轮信号
- 决策规则：`残留仓位占比 + 新信号权重合计 <= 100%` 方可入待买队列，否则继续等待
- 无论通过或拒绝均打印清晰评估日志
- `walk_forward.py` 新增 `--no-early-rebalance-on-empty` 开关；`batch_walk_forward.ps1` 新增 `$enable_early_rebalance_on_empty` 参数区

**空仓提前调仓** (v0.41.0):
- 新增：`enable_early_rebalance_on_empty` 参数，默认启用
- 当持仓为空且无待执行信号/补位时，立即触发新一轮信号生成（T+1 买入），无需等待预定调仓日
- 解决门控阻断、整体止盈清仓、止损清完持仓等场景下资金空转问题
- 与 `take_profit_refill` 补位机制协调，补位优先

**信号入口门控 v2 + 滚动模型质量监控** (v0.40.0):
- 新增：`signal_gate_mode` 参数支持 `legacy`/`composite`/`disabled` 三种门控模式
- 新增：composite 模式实现成本门控（预测收益<N×交易成本时持币）、绝对收益质量分、百分位归一化和自校准阈值
- 新增：滚动模型质量监控——追踪选股实际表现，模型"失灵"时自动降仓，恢复时自动放开
- 新增：`batch_walk_forward.ps1` 支持新门控参数扫描

**batch_walk_forward 显式支持 val_ratio 扫描与透传** (v0.39.0):
- 新增：`batch_walk_forward.ps1` 现在提供独立的 `$val_ratio_list`，可直接固定 `0.1` 或扫描 `0.1/0.15/0.2`
- 新增：批量任务总数统计、参数组合循环和 `walk_forward.py` 命令拼接均已接入 `--val-ratio`
- 优化：批量实验不再隐式依赖 `walk_forward.py` 默认验证集比例，调参时更容易验证 `val_ratio` 对早停轮次分布的影响

**walk-forward 默认矩阵收口到更均衡的 chain 指标区域** (v0.38.6):
- 修复：`batch_walk_forward.ps1` 在 `skip-training` 默认场景下不再重复扫描 `neu_y_ret_20`，避免对同一组旧模型做无效标签对照
- 优化：默认门控矩阵收口到 `TopK=10`、`thresholds=[0.04,0.12,0.25]`、两组仓位系数，以及 `vol_target=0.20/0.22` 的更窄区间
- 新增：`compare_walk_forward.py` 现在会输出基于 `chain_nav` 的全周期 `CAGR`、总收益、链式最大回撤和链式夏普
- 优化：综合评分改为优先参考全周期 chain 指标，避免把 split 均值误读成全周期收益

**batch_walk_forward 默认切换为更聚焦的防守型扫描矩阵** (v0.38.5):
- 优化：`batch_walk_forward.ps1` 现在默认以 `y_ret_20` 为主、`neu_y_ret_20` 为对照，减少在 2023-2024 防守阶段已证伪方向上的重复扫描
- 优化：信号门控参数由单组固定值改为可批量扫描的 gate 配置集合，默认围绕 `TopK=8/10/12`、三组阈值和两组仓位系数展开
- 优化：市场层默认只扫描 `vol_target=0.18/0.20/0.22`，同时将盈利延续天数 baseline 对齐到当前最佳防守型 run
- 优化：关闭门控时，脚本会自动退化为单一占位配置，避免无意义地展开门控相关笛卡尔积

**盈利延续持有日志增加预计卖出日期** (v0.38.4):
- 修复：`engine.py` 的“盈利延续持有”日志现在会在“延续至最多 N 天”后继续打印延期后的预计卖出交易日
- 修复：预计卖出日期按实际买入日和交易日序列计算，和当前持有期卖出规则保持一致
- 新增：单元测试覆盖盈利延续持有日志里的预计卖出日期输出

**回测轮次分隔线前移** (v0.38.3):
- 修复：`engine.py` 现在会在“新轮次第一天”的所有业务日志之前打印“新一轮回测”分隔线
- 修复：同一轮下的选股过滤、模型预测、卖出执行、买入执行与调仓摘要，不会再跑到分隔线前面造成错位
- 新增：单元测试覆盖新轮首日“先分隔线、后信号日志”的输出顺序

**统一调仓决策摘要改为单行表格式** (v0.38.2):
- 优化：每个调仓日的摘要日志改为单行 `|` 分隔格式，适合在 batch/walk-forward 长日志里快速横向比对
- 优化：门控阻断时，未进入后续评估的层明确显示为 `N/A`，不再把未评估误看成 `100%`
- 保持：信号门控、ECT、MA250/ATR、市场择时与最终目标仓位仍然固定输出，不再受条件分支影响

**统一调仓决策摘要日志** (v0.38.1):
- 优化：OOS 回测现在会在每个调仓日固定输出一段“调仓决策摘要”，集中展示信号门控、ECT、MA250/ATR、市场择时和最终目标仓位
- 优化：即使最终满仓通过，也会明确打印各层仓位结果，不再因为条件分支导致某些调仓日完全看不出是哪一层在控仓
- 优化：若信号门控直接把目标仓位压到 0，日志会明确标记“本次不进入待买队列”，不再只剩零散提示

**信号置信度门控 + 持币机制** (v0.38.0):
- 新增：`MLSignal` / `EnsembleMLSignal` 支持基于 Top-K 预测强度计算置信度分数，并按阈值映射不同仓位系数
- 新增：当置信度不足时可自动降仓或直接持币，不再在低边际优势环境里被动满仓
- 新增：`BacktestEngine`、`walk_forward.py`、`compare_walk_forward.py`、`batch_walk_forward.ps1` 与共享 `TradingConfig` 已全部接入该能力
- 新增：walk-forward 汇总与 compare 对比表会输出门控持币率、平均仓位、平均置信度，便于批量实验归因

**Walk-forward 主链接入更多 OOS 回测参数，并支持批量实验** (v0.37.0):
- 新增：`walk_forward.py` 现已支持回测卖出时机、ST/上市天数过滤、单股最大权重、单行业最大持仓数
- 新增：`walk_forward.py` 现已支持 OOS 止损参数与 ECT 权益曲线交易参数，并写入汇总 CSV
- 新增：`compare_walk_forward.py` 会保留以上参数列，`batch_walk_forward.ps1` 可直接逐组扫描这些回测风控组合

**每日回测日志新增当前持仓 ATR 统计** (v0.36.0):
- 新增：`engine.py` 的每日回测进度日志追加 `ATR(min/avg/max)` 字段，便于快速观察当前组合波动水平
- 新增：`engine_ml.py` 使用当日持仓股票的 `atr_pct_14` 计算最小值、均值、最大值；缺失时显示 `N/A`

**回测轮次分隔线只在首日输出** (v0.35.2):
- 修复：`engine.py` 的“新一轮回测”分隔线改为仅在 `本轮第[1/N]天` 前输出，不再每天重复打印
- 新增：单元测试覆盖轮次首日分隔线输出条件，避免后续回归

**MA250 日志改为简洁公式展示** (v0.35.1):
- 优化：`engine_ml.py` 的 MA250 日志前缀精简为 `MA250`，`ratio` 结果改为 `触发控仓/未触发控仓` 的直接表述
- 优化：ATR 缩放日志直接展开为 `atr_ma250/atr_now=...=...%` 计算式，便于快速核对仓位缩放来源

**每日回测日志增加持仓目标与当前仓位** (v0.35.0):
- 新增：每日回测进度日志显示 当前持仓数/目标持仓数 与 当前股票仓位比例，便于直接判断组合是否满仓
- 兼容：分批调仓时目标持仓数会按批次数自动放大，更贴近组合整体目标

**特征构建一键全因子开关** (v0.34.0):
- 新增：`build_clean_features.py` 支持 `--build-all`，一次性启用基本面、另类数据、融资融券、筹码胜率、基金持仓、业绩快报六类可选因子
- 保持：行业中性化仍由 `--enable-industry-neutralization` 单独控制，避免隐式改变截面处理逻辑

**MA250 ATR 特征缓存自动失效修复** (v0.33.3):
- 修复：将 `mkt_atr_pct`、`mkt_atr_pct_ma250` 纳入 features 缓存完整性校验，旧缓存会被识别为缺列
- 修复：`build_clean_features.py` 不再仅凭文件存在就跳过，发现旧 schema 会自动重建

**MA250 日志精细化** (v0.33.2):
- 优化：`engine_ml.py` 的 MA250 模块日志现在会明确打印 threshold 比较结果、是否触发硬条件、硬条件目标仓位、第一步基准仓位和 ATR 缩放后的最终仓位
- 优化：ATR 缩放开启时额外打印 scale、`atr_ma250`、`atr_now`，缺失 ATR 数据时明确标注

**MA250 硬条件可观测性修复** (v0.33.1):
- 优化：OOS 回测新增 MA250 交易日命中数与调仓信号日命中数统计日志，便于判断开关是否真正影响回测
- 优化：若当前 OOS 窗口没有任何调仓信号日命中，会明确提示“结果可能与关闭时接近”
- 优化：`compare_walk_forward.py` 汇总表新增 MA250 硬条件开关、阈值和仓位参数列

**MA250 ATR 动态仓位缩放** (v0.33.0):
- 新增：MA250 模块支持 ATR 动态仓位缩放（`--ma250-atr-scaling`），
  仓位 = 基准A × MA(ATR,250)/CurrentATR，高波动降仓、低波动恢复满仓
- 新增：市场状态特征 `mkt_atr_pct`（每日全市场 ATR% 中位数）、`mkt_atr_pct_ma250`（250日均值）
- 移除：个股层面 `atr_position_sizing` 功能，由整体仓位缩放替代

**条件卖出 T+1** (v0.31.0):
- 亏损提前换出、整体止盈改为"Tn 检查 → Tn+1 开盘价执行"，符合 A 股 T+1 规则
- `sell_timing` 默认值改为 `"open"`

**整体持仓止盈** (v0.30.0):
- 新增：`BacktestEngine` 支持整体持仓止盈（`take_profit_threshold`），
  当所有持仓的整体浮盈率（市值加权，后复权口径）≥ 阈值时，立即清空全部仓位并在 T+1 日自动补位买入
- `walk_forward.py`、`batch_walk_forward.ps1` 同步支持 `--take-profit-threshold` / `--no-take-profit-refill`

**回测风控优化** (v0.29.0):
- 新增：`mkt_ma250_ratio` 市场特征（大盘收益曲线 / MA250），用于识别系统性熊市
- 新增：`BacktestEngineML` 支持 MA250 长周期硬条件（`market_regime_ma250_hard_stop`），
  大盘跌破 250 日均线时强制降至指定仓位（默认完全空仓），优先级高于其他择时模式
- 新增：`BacktestEngine` 支持盈亏动态持仓时长（`enable_profit_based_holding`），
  亏损股提前换出 + 盈利股延续持有，提高换仓效率
- `walk_forward.py`、`batch_walk_forward.ps1` 同步支持以上所有新参数

**分批调仓** (v0.28.0):
- 新增 `--stagger-tranches` 参数，将资金分为 K 份错开调仓，降低单次调仓时点风险
- `walk_forward.py`、`run_ml_backtest.py`、`batch_walk_forward.ps1` 同步支持
- 修复：信号生成时排除已持仓股票并顺延补位，避免不同 tranche 选到重复股票浪费预算 (v0.28.1)
- 修复：仓位补齐相关日志（买入失败/仓位未满）始终输出，不再受 verbose 控制 (v0.28.2)
- 修复：分批调仓下仓位补齐预算未按 tranche 比例分配，导致补齐买入金额偏大 (v0.28.3)
- 优化：分批调仓日志增加 `[批次 N/K]` 前缀，方便区分各批次的信号/买入/补齐操作 (v0.28.4)
- 修复：仓位补齐不再对同一不可交易股票重复尝试，避免无效操作 (v0.28.5)
- 优化：回测进度日志增加持仓数、本轮收益率、年化收益率 (v0.28.6)
- 优化：卖出执行和止损卖出日志始终输出，不再受 verbose 控制 (v0.28.7)
- 新增：`walk_forward.py` 支持 `--bt-initial-capital` 配置回测初始资金，`batch_walk_forward.ps1` 同步支持 (v0.28.8)

**回撤归因分析** (v0.27.0):
- 新增 `scripts/ana/analyze_drawdown.py`，基于 walk-forward 数据做多维回撤归因（信号质量/市场环境/转化效率/回撤段详情）
- 输出 CSV + matplotlib 图表 + 文字报告

**修复特征构建数据加载范围不足** (v0.26.1):
- 修复 `fund_portfolio` 和 `cyq_perf` 在 `build_clean_features.py` 中使用精确日期范围加载导致起始段特征缺失的问题

**多偏移集成训练** (v0.26.0):
- 新增 `--ensemble-offsets` 参数，每个 split 训练3个偏移模型取平均，消除训练边界敏感性

**特征稳定性筛选** (v0.25.0):
- 新增 `--feature-stability-filter` 参数，训练前自动移除跨时期IC方向不一致的特征，提升模型泛化性

**钉钉机器人长时间命令进度报告** (v0.24.0):
- `trade` 命令执行期间每60秒自动推送当前步骤和已用时间，避免长时间无响应

**融资融券因子独立开关** (v0.23.0):
- 将融资融券4个因子从另类数据因子中剥离为独立 `--enable-margin-features` 开关
- `batch_walk_forward.ps1` 新增 `$enable_margin` 变量，可独立控制是否使用融资融券因子

**修复补位信号失败后无法重试** (v0.22.5):
- 补位失败时保存原始目标为 PendingBuy，T1 幂等检查后仍处理待补位计划

**修复融资融券因子缺失** (v0.22.4):
- 当日 margin_detail 未发布时额外重试下载，仍失败则阻止生成不完整 features 并输出明确错误

**基金持仓因子内存优化** (v0.22.3):
- 逐分区加载+聚合替代全量加载，fund_portfolio 峰值从 ~280MB 降至 ~35MB

**特征构建全链路内存优化** (v0.22.2):
- 基金持仓回溯缩短、技术指标缓存按需构建、消除冗余 `.copy()`

**因子加载内存优化** (v0.22.1):
- `_load_factor_data()` 每组因子处理完后即时释放中间数据，峰值内存降低约 40%

**Walk-forward 部署模型自动训练** (v0.22.0):
- Walk-forward 完成后默认自动追加部署训练，使用最后 split 的 test_end 作为 train_end，消除模型时间滞后
- 支持 `--no-deploy-train` 禁用，`batch_walk_forward.ps1` 同步支持

**修复纸面交易下载和使用未来数据的前视偏差** (v0.20.8):
- 修复 `ensure_features_for_date` 日期范围计算包含未来1个月交易日的问题
- cyq_perf/fund_portfolio/clean数据加载均严格截止到 `trade_date`，杜绝未来数据泄露

**全局代码审查修复14个隐藏bug** (v0.20.7):
- 修复 score 权重负分占位、上市天数默认值误判、训练集 delta 泄漏、补齐候选不足等 P0 问题
- 修复 ST 正则、涨跌停容差、日期转换、默认值不一致等 P1 问题
- 增强 Parquet 原子写入、NaN 防护、日期校验等 P2 鲁棒性

**修复 express/fund_portfolio 因子全 NULL** (v0.20.6):
- ✅ 修正 express 因子列名映射（TuShare 实际列名 `yoy_net_profit`/`diluted_roe`），新增营收同比自动计算
- ✅ 修复 fund_portfolio symbol 已含交易所后缀导致 `_symbol_to_ts_code` 重复拼接的问题

**修复41个失败测试，同步测试代码与源代码接口** (v0.20.5):
- 修复 `split_train_val_by_date` IndexError 源码bug
- 更新13个测试文件以适配最新接口变更

**修复特征构建时筹码胜率/基金持仓数据加载失败** (v0.20.4):
- ✅ `build_clean_features.py` 中 `load_cyq_perf`/`load_fund_portfolio` 传入日期范围，正确使用分区加载

**修复钉钉机器人消息丢失** (v0.20.3):
- ✅ `SimpleHandler` 改用 `AsyncChatbotHandler`，`process()` 在线程池中执行，不再阻塞 event loop 导致断线

**fund_portfolio 按季度分区存储** (v0.20.2):
- ✅ fund_portfolio 从单文件改为按季度末日期分区Parquet存储，支持增量加载

**批量下载优化** (v0.20.1):
- ✅ 6个因子数据从逐股下载改为按日期/季度/月份批量下载，API调用次数降低1~2个数量级

**新增高积分因子** (v0.20.0):
- ✅ **筹码胜率因子** (cyq_perf): winner_rate、成本偏离度、筹码集中度、胜率变化率
- ✅ **业绩快报因子** (express_vip): 营收/净利润增速、ROE、业绩惊喜（vs预告偏差）
- ✅ **基金持仓因子** (fund_portfolio): 持股比例、基金数量及其季度环比变化
- ✅ 3组因子独立开关: `--enable-cyq-features`, `--enable-express-features`, `--enable-fund-features`
- ✅ 全链路支持: 下载、加载、特征构建、训练、回测、批量脚本

### v0.18.2

**修复因子数据自动下载不完整** (v0.18.2):
- ✅ **修复**: fina_indicator/stk_holdernumber/forecast 首次运行时自动全量逐股下载（支持断点续传），而非仅下载单日公告
- ✅ **修复**: 申万行业分类数据自动下载，无需手动运行 `update_basic_data.py`
- ✅ **修复**: 增加最低记录数阈值，残留的不完整数据也会触发全量重下

### v0.18.1

**修复 moneyflow 自动下载被跳过** (v0.18.1):
- ✅ **修复**: `ensure_raw_data_for_date()` 中 moneyflow 下载被 `if not daily_exists:` 分支跳过，导致纸面交易缺失资金流向数据

### v0.18.0

**纸面交易 `adjust reset-t0` 子命令** (v0.18.0):
- ✅ **`adjust reset-t0` 子命令**：重置指定日期的T0运行记录并清空所有延迟交易订单，允许重新执行T0工作流

### v0.17.1

**AKShare 依赖补齐 + 纸面交易缺失数据自动下载** (v0.17.1):
- ✅ **补齐 `akshare` 项目依赖**：解决 `features/ensure.py` 中 `import akshare` 的 Pylance 告警与环境不一致问题
- ✅ **ensure 链路全覆盖**：`ensure_raw` 新增 `daily_basic`/`margin_detail`，`ensure_clean` 新增 `daily_basic`/`moneyflow` 自动构建
- ✅ **因子数据按日自动下载**：`fina_indicator`/`forecast`/`stk_holdernumber`/`hot_rank` 缺失时自动增量下载并追加保存
- ✅ **删除业绩快报（express）支持**：简化 `earnings.py` 逻辑，清理全部 express 引用

### v0.16.2

**数据下载去重 + 命名规范化** (v0.16.2):
- ✅ **删除 `_download_data()` 冗余方法**：T0 数据下载统一由 ensure 链完成，减少约 100 行重复代码
- ✅ **`load_industry_mapping()` 参数重命名**：`stock_basic` → `shenwan_industry`，消除命名歧义

### v0.16.1

**修复纸面交易因子数据缺失** (v0.16.1):
- ✅ **纸面交易 T0 自动加载因子数据**：`ensure_features_for_date()` 补全基本面和另类数据因子的加载逻辑
- ✅ **因子缺失可见性**：缺失因子数据时输出 WARNING 汇总日志（覆盖率 + 缺失列表 + 下载命令提示）

### v0.16.0

**市场择时多模式扩展 + 依赖精简** (v0.16.0):
- ✅ **市场择时扩展为 4 种模式**（`engine_ml.py` + `walk_forward.py`）：
  - `binary`（原有）：mkt_ret_avg_20 < threshold → bear_exposure，否则满仓
  - `vol_target`（新增）：exposure = target_vol / realized_vol，波动越大仓位越低
  - `trend`（新增）：基于 mkt_ma_trend（MA20/MA60）线性降仓，下行趋势自动减仓
  - `combined`（新增）：vol_target + trend 取最小值或相乘，双重保护
  - 回撤保护：已大幅下跌时停止降仓，避免底部踏空反弹（`--market-regime-drawdown-guard`）
  - 趋势保护：上行趋势时跳过 vol_target 强制满仓（`--market-regime-trend-guard`）
- ✅ **新增 `mkt_ret_vol_20` 市场特征**：20日全市场收益波动率，用于 vol_target 模式
- ✅ **移除业绩快报因子**：删除 `express_profit_yoy`、`express_revenue_yoy`（数据质量不稳定）
- ✅ **移除 tensorflow 依赖**：精简项目依赖
- ✅ **删除 `scripts/build_features.py`**：已由 `build_clean_features.py` 完全替代

### v0.15.1

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
  - `industry_constraint.py` 优先使用统一主字段 `sw_industry`（申万二级）行业分类
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

**申万行业多层字段 + 项目级主口径配置**:
- ✅ **申万行业数据保留 L1/L2/L3 三层字段**：下载与清洗结果仍保留三级明细，便于解释和扩展
  - 行业映射表统一为单张表，包含 L1/L2/L3 三层字段
  - FeatureBuilder 主字段（`sw_industry*`）统一绑定到 `configs/base.yaml` 的 `industry.shenwan_level`；默认 `l2`，同时保留 `sw_l3*` / `sw_l1*` 辅助字段
  - `update_basic_data.py --only-shenwan` 继续按 L3 数据源下载，再在特征阶段映射为系统统一主口径
- ✅ **行业中性化按主口径自适应回退**：样本不足时按层级回退，保证稳健性
  - `l3`：三级行业内 `tradable==1` 样本不足 → 回退到二级，再回退到一级，最后回退到全市场
  - `l2`：二级行业内 `tradable==1` 样本不足 → 回退到一级，再回退到全市场
  - `l1`：一级行业内 `tradable==1` 样本不足 → 直接回退到全市场
  - 训练、回测、纸面交易统一读取同一个项目配置，避免行业口径再次混用

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

### 核心功能模块

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

#### 权益曲线交易核心功能
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
# 方式1：直接编辑 data/paper/config.yaml（推荐，分段+中文注释）

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

# 步骤2: 构建clean和features（假设raw已存在）
# --horizon / --horizons 二选一必填：
#   --horizon 20         : 单值模式，仅按主 horizon 对应的 y_ret_20 非空过滤（推荐，保留停牌导致的辅助标签缺失样本）
#   --horizons 5 10 20   : 多值模式，AND 过滤，要求所有 horizons 对应 y_ret_N 同时非空
# 两种模式下生成的特征文件都包含 y_ret_5/10/20 三列，schema 一致
python scripts/build_clean_features.py --start-date 20230101 --end-date 20231231 --horizon 20

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
│   └── analyze_factor_importance.py # 因子重要性分析
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
