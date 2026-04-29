# Changelog

All notable changes to this project will be documented in this file.

## [0.67.0] - 2026-04-29

### 新增

- **树莓派 3.5 寸 LCD 下方图表新增中证800折线（亮白色）并复用盘内/盘外切换策略** ([scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py)):
  - 图表从 3 条线扩展为 4 条线：上证、深证、持仓、中证800，其中中证800折线颜色为亮白色
  - 刷新频率与既有三条线完全一致，继续复用现有数据线程节奏（盘中实时刷新，盘外按周期补齐）
  - 显示策略保持一致：盘内显示当日日内线，盘外显示周期日线
  - 图例横向空间压缩：图例线段缩短，标签统一展示为 `上 / 深 / 持 / 中`
  - 中证800数据接入 AKShare：
    - 日线通过 AKShare 拉取并参与周期图构建
    - 盘中涨跌幅通过 AKShare 现货接口拉取并参与日内图构建

### 测试

- **补充 3.5 寸 LCD 中证800折线回归测试** ([tests/test_respi_35lcd_disp.py](tests/test_respi_35lcd_disp.py)):
  - 新增周期图 payload 包含 `csi800_pct` 的断言
  - 新增/更新日内图写入 `raw_csi800_pct` 与 `csi800_pct` 的断言
  - 更新周期图缓存相关测试，补齐中证800 AKShare 数据 mock

## [0.66.13] - 2026-04-29

### 修复

- **树莓派 3.5 寸 LCD 的年化收益超长时会自动缩小字号** ([scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py)):
  - 左侧总览面板里的“年化收益”不再固定使用 24 号字体，而是按当前列宽自适应挑选可容纳的字号
  - 当收益率达到 `+123.4%` 这类 3 位数以上场景时，会自动降到更小字号，避免数值越过单元格右边界

### 测试

- **补充 3.5 寸 LCD 年化收益缩字回归测试** ([tests/test_respi_35lcd_disp.py](tests/test_respi_35lcd_disp.py)):
  - 校验 `_pick_fitting_font()` 在 `+123.4%` 这类超宽文本下会返回更小字号，而普通宽度文本仍保持默认字号

## [0.66.12] - 2026-04-26

### 修复

- **paper 空账户初始化会自动对齐当前配置初始资金，避免 45w/50w 基准并存** ([src/lazybull/paper/account.py](src/lazybull/paper/account.py)):
  - 当加载到的账户状态满足“无持仓 + last_update 为空”的初始空状态时，若 `account.cash` 与当前配置 `initial_capital` 不一致，会自动同步为配置值并落盘
  - 这样可覆盖 `reset-t0` 前后配置切换或历史状态残留导致的空账户现金基准漂移，避免运行日志中同时出现 450,000 与 500,000 两套初始基准

### 测试

- **补充空账户初始资金自动同步回归测试** ([tests/test_paper_trading.py](tests/test_paper_trading.py)):
  - 校验空账户旧状态现金为 450,000、当前配置初始资金为 500,000 时，`PaperAccount` 初始化后会自动同步到 500,000 并持久化

## [0.66.11] - 2026-04-26

### 修复

- **paper_trade 的本轮盈亏改为优先使用“上次调仓总资产 -> 当前总资产”口径** ([src/lazybull/paper/broker.py](src/lazybull/paper/broker.py), [src/lazybull/paper/reporting.py](src/lazybull/paper/reporting.py)):
  - `PaperBroker` 新增统一口径方法 `calculate_round_pnl_metrics()`：若策略状态里存在 `last_rebalance_nav`，本轮收益率按总资产变化计算，自动覆盖“本轮中已卖出（已实现）盈亏”
  - 缺少 `last_rebalance_nav` 时会自动回退到旧口径（当前持仓浮盈 / 当前持仓成本），保持历史数据兼容性
  - `print_positions_summary()` 与共享展示层 `load_position_snapshot()` 现在复用同一计算逻辑，避免 CLI 与钉钉展示口径不一致

### 测试

- **补充本轮盈亏口径回归测试** ([tests/test_paper_trading.py](tests/test_paper_trading.py)):
  - 校验存在 `last_rebalance_nav` 时，本轮收益率按总资产口径计算（可纳入已实现盈亏）
  - 校验缺少策略状态时，仍按旧的持仓浮盈口径回退

## [0.66.10] - 2026-04-26

### 修复

- **paper_trade 的基金持仓历史补齐改为最小列读取并带季度聚合缓存** ([src/lazybull/data/storage.py](src/lazybull/data/storage.py), [src/lazybull/features/ensure.py](src/lazybull/features/ensure.py), [src/lazybull/factors/fund_portfolio.py](src/lazybull/factors/fund_portfolio.py)):
  - `Storage.load_raw_by_date()` 现在支持可选列裁剪，`fund_portfolio` 历史补齐在读取季度原始分区时只会加载聚合真正需要的 `ts_code/symbol/ann_date/end_date/stk_float_ratio` 五列，不再把整张季度明细表全部读入内存
  - `_aggregate_fund_portfolio()` 现在会在聚合前先瘦身到最小必需列，避免 `mkv/amount` 等无关宽列在树莓派内存里继续占位
  - `_try_ensure_historical_fund_portfolio()` 新增 `raw/fund_portfolio_agg/` 季度聚合缓存；首次运行会把季度原始明细聚合成个股级缓存，后续 `paper_trade` 重跑时将直接复用缓存，不再反复把大季度明细重新拉进内存
  - 这主要针对 `fund_portfolio 20260331.parquet` 这类 10 万级季度分区在树莓派上把 T0 最后一步内存顶爆的问题

### 测试

- **补充基金持仓最小列聚合与缓存复用回归测试** ([tests/test_fund_portfolio_factor.py](tests/test_fund_portfolio_factor.py), [tests/test_ensure_and_t0_printing.py](tests/test_ensure_and_t0_printing.py)):
  - 校验 `_aggregate_fund_portfolio()` 在只提供最小必需列时仍能正确聚合
  - 校验 `_try_ensure_historical_fund_portfolio()` 会生成并复用季度聚合缓存，避免缓存存在时再次回读原始季度明细

## [0.66.9] - 2026-04-26

### 修复

- **paper_trade 的单日 point-in-time 因子改为走快照快速路径** ([src/lazybull/factors/fundamental.py](src/lazybull/factors/fundamental.py), [src/lazybull/factors/holder.py](src/lazybull/factors/holder.py), [src/lazybull/factors/earnings.py](src/lazybull/factors/earnings.py), [src/lazybull/factors/express.py](src/lazybull/factors/express.py), [src/lazybull/factors/fund_portfolio.py](src/lazybull/factors/fund_portfolio.py)):
  - 当 `trading_dates` 只有 `trade_date` 当天时，基本面、股东人数、业绩预告、业绩快报、基金持仓不再走“逐股票构造 Python 列表 + bisect 回放”的通用路径，而是直接基于 `ann_date <= trade_date` 的可见记录向量化提取每只股票的最新快照
  - 多日路径也去掉了内层重复构造 `ann_dates` 列表的开销，减少树莓派在 point-in-time 因子阶段的纯 Python 循环负担
  - 这进一步降低了 `paper_trade` 在 T0 工作流“步骤2: 生成信号”阶段卡在基本面 / 股东人数 / 业绩预告 / 业绩快报 / 基金持仓上的耗时风险

### 测试

- **补充单日快照与多日查询一致性回归测试** ([tests/test_single_day_factor_snapshots.py](tests/test_single_day_factor_snapshots.py)):
  - 校验 fundamental / holder / earnings / express / fund_portfolio 在单日快照快速路径下，与原多日 point-in-time 查询在同一目标交易日上的结果保持一致

## [0.66.8] - 2026-04-26

### 修复

- **paper_trade 的 T0 因子补齐改为仅物化目标交易日输出** ([src/lazybull/features/ensure.py](src/lazybull/features/ensure.py)):
  - `_load_factor_data()` 现在仍会加载构建因子所需的历史原始数据，但不再为整段历史交易日批量构建 `date -> DataFrame` 查询表，只会为 `trade_date` 当天生成截面结果
  - 这显著降低了树莓派运行 `paper_trade` 时在基本面、股东人数、业绩预告、业绩快报、基金持仓、一致预期等 point-in-time 因子上的 CPU 与内存峰值，减少在“股东人数因子”附近长时间卡住甚至死机的概率

### 测试

- **补充 ensure 单日因子输出回归测试** ([tests/test_ensure_and_t0_printing.py](tests/test_ensure_and_t0_printing.py)):
  - 校验 `_load_factor_data()` 调用各因子构建器时只传入 `[trade_date]` 作为输出日期列表，避免后续重构把整段历史输出重新带回来

## [0.66.7] - 2026-04-26

### 修复

- **paper_trade 的 early_exit_mode=disabled 语义现已与 backtest 对齐** ([src/lazybull/paper/runtime.py](src/lazybull/paper/runtime.py), [scripts/paper_trade.py](scripts/paper_trade.py)):
  - 纸面交易共享运行时现在只受 `enable_profit_based_holding` 总开关控制；当 `early_exit_mode=disabled` 时，仍会执行 `early_exit_loss_threshold + early_exit_holding_ratio` 的基础亏损提前换出检查，语义恢复为“原硬卖”而不是“整段跳过”
  - 纸面交易启动日志中的 `亏损换出=disabled` 现已改写为 `亏损换出=原硬卖`，避免把 `disabled` 误读成“关闭亏损提前换出”

- **paper 配置模板与 batch 参数区改为按真实控制关系展示** ([src/lazybull/paper/storage.py](src/lazybull/paper/storage.py), [data/paper/config.yaml](data/paper/config.yaml), [scripts/batch_walk_forward.ps1](scripts/batch_walk_forward.ps1)):
  - `early_exit_loss_threshold` 与 `early_exit_holding_ratio` 现在被明确标记为“启用 `enable_profit_based_holding` 后始终生效的基础阈值”
  - `early_exit_mode` 被明确标记为“二次确认子开关”，仅控制 `strength_veto` 保护分支，不再和基础阈值混排，减少“参数开着却看起来像没生效”的配置歧义

### 测试

- **补充 paper runtime disabled 模式与 YAML 注释回归测试** ([tests/test_paper_trade_runtime.py](tests/test_paper_trade_runtime.py), [tests/test_paper_trading_cli.py](tests/test_paper_trading_cli.py)):
  - 校验 `early_exit_mode=disabled` 时，paper 仍会执行基础亏损提前换出检查
  - 校验 `PaperStorage.save_config()` 生成的 YAML 模板包含新的基础阈值 / 二次确认分组注释

## [0.66.6] - 2026-04-26

### 修复

- **walk_forward / batch_walk_forward 的亏损提前换出默认值与 paper_trade 对齐** ([src/lazybull/common/trading_config.py](src/lazybull/common/trading_config.py), [scripts/batch_walk_forward.ps1](scripts/batch_walk_forward.ps1)):
  - `TradingConfig` 与公共 CLI 的 `early_exit_mode` 默认值改为 `disabled`，与回测引擎默认值和 paper 配置模板一致，不再在未显式传参时悄悄回退到 `strength_veto`
  - `batch_walk_forward.ps1` 现在会在启用盈亏动态持仓时始终显式透传 `--early-exit-mode`，即使脚本里选择的是 `disabled`，也不会再因为省略参数而落回公共 CLI 默认值
  - 这修复了“batch 脚本里写着 `early_exit_mode_list=@('disabled')`，walk_forward 实际却按 `early_exit_loss_threshold` 触发提前换出”的接线不一致问题

### 测试

- **补充 TradingConfig 默认 early_exit_mode 回归断言** ([tests/test_holding_strength.py](tests/test_holding_strength.py)):
  - 校验默认 `TradingConfig()` 现在为 `early_exit_mode='disabled'`，避免公共默认值再次偏离回测引擎与 paper 配置

## [0.66.5] - 2026-04-26

### 修复

- **纸面交易信号门控不再被单股限权归一化吞掉** ([src/lazybull/paper/runner.py](src/lazybull/paper/runner.py), [src/lazybull/signals/ml_signal.py](src/lazybull/signals/ml_signal.py), [src/lazybull/paper/runtime.py](src/lazybull/paper/runtime.py)):
  - paper 链路现在与 backtest 保持一致，先做单股限权，再应用 composite/legacy 门控与滚动质量降仓，避免 max_weight_per_stock 把降仓后的总权重重新归一化回满仓
  - paper 路径在门控“满仓通过”时也会显式打印门控结果，不再只在持币或半仓时有门控日志
  - 盈利延续总开关打开但 profit_extension_mode=disabled 时，现在会直接打印“盈利延续模式未启用，跳过”，避免误解为功能已运行但没有命中

### 测试

- **补充纸面交易门控保留仓位与日志回归测试** ([tests/test_buy_replacement.py](tests/test_buy_replacement.py)):
  - 校验在存在 max_weight_per_stock 时，半仓门控后的目标权重总和仍保留 0.5，不会被后续限权归一化抹掉
  - 校验门控满仓通过时，在 emit_log=True 下也会输出门控日志

## [0.66.4] - 2026-04-26

### 修复

- **纸面交易补位买入不再沿用主仓 top_n 扩大持仓数** ([src/lazybull/paper/runner.py](src/lazybull/paper/runner.py)):
  - `generate_replacement_targets()` 在传入完整 `TradingConfig` 时，会显式把补位选股的 `top_n` 覆盖为失败缺口数，不再错误沿用主配置里的 `top_n=20`
  - 增加补位目标数量保护，若下游门控或候选筛选返回数量超过缺口数，会在保存 `pending_buys` 前截断到缺口数，避免 2 个失败目标被扩成 20 个补位计划，随后把持仓数推高到 33 只之类的异常状态

### 测试

- **补充补位数量受失败数限制的回归测试** ([tests/test_buy_replacement.py](tests/test_buy_replacement.py)):
  - 校验传入完整 `TradingConfig(top_n=20)` 生成 2 个补位时，内部选股仍按 `failed_count=2` 执行，并且最终只返回 2 个补位目标

## [0.66.3] - 2026-04-26

### 修复

- **reset-t0 会递归清理树莓派大屏缓存子目录** ([src/lazybull/paper/storage.py](src/lazybull/paper/storage.py)):
  - `PaperStorage.reset_t0()` 不再只删除 `state/` 根目录下的直接文件，而是会递归清理 `state/`、`trades/`、`nav/`、`runs/`、`pending_*`、`instructions/` 下的嵌套子目录
  - 这会一并删除 `data/paper/state/respi_35lcd_intraday/` 里的日内图历史 JSON，避免 `paper_trade adjust reset-t0` 后树莓派大屏在重启甚至断电重启后仍加载旧折线

### 测试

- **补充 reset-t0 清理树莓派缓存回归测试** ([tests/test_adjust_truncate.py](tests/test_adjust_truncate.py)):
  - 校验 `reset_t0()` 会删除 `state/respi_35lcd_intraday/` 这类嵌套缓存目录

## [0.66.2] - 2026-04-26

### 修复

- **paper_trade 共享运行时支持解析 next 交易日** ([src/lazybull/paper/runner.py](src/lazybull/paper/runner.py), [src/lazybull/paper/runtime.py](src/lazybull/paper/runtime.py), [src/lazybull/paper/reporting.py](src/lazybull/paper/reporting.py), [scripts/paper_trade.py](scripts/paper_trade.py)):
  - `PaperTradingRunner._correct_trade_date()` 现在会先解析 `next`，优先基于 `last_trade_date`，缺失时回退到账户 `last_update`，再回退到“从今天起的最近交易日”
  - 共享 `execute_trade_workflow()` 现在会持久化本次实际执行日，后续 `paper_trade run --trade-date next` 与持仓查看都能复用同一条 next 推算链路
  - `load_position_snapshot()` 和 CLI 持仓打印也改为展示解析后的真实交易日，不再把字面量 `next` 传到底层数据加载

### 测试

- **补充 next 交易日解析回归测试** ([tests/test_ensure_and_t0_printing.py](tests/test_ensure_and_t0_printing.py), [tests/test_paper_trade_runtime.py](tests/test_paper_trade_runtime.py)):
  - 校验 `next` 会解析为上次执行日后的下一个交易日
  - 校验共享运行时执行完成后会写回最近执行日，供后续 `next` 继续推算

## [0.66.1] - 2026-04-26

### 修复

- **paper_trade 成本配置缺键不再导致 T0/补位流程崩溃** ([src/lazybull/paper/runner.py](src/lazybull/paper/runner.py), [src/lazybull/common/config.py](src/lazybull/common/config.py), [configs/base.yaml](configs/base.yaml)):
  - `PaperTradingRunner` 不再直接读取 `configs/base.yaml` 并硬索引 `costs.capital_retention_ratio` / `costs.pendding_capital_retention_ratio`，统一改走公共成本配置读取并对缺失键回退默认值
  - 补齐 `configs/base.yaml` 中两个资金保留比例默认项，其中 T0 保留比例默认 `0.0`，补位保留比例默认 `0.3`，避免 `adjust reset-t0` 后重新执行 `paper_trade next` 时在步骤3生成交易指令阶段因缺键报错

### 测试

- **补充纸面交易成本配置缺键回归测试** ([tests/test_ensure_and_t0_printing.py](tests/test_ensure_and_t0_printing.py)):
  - 校验缺少 `capital_retention_ratio` 时，T0 指令生成仍会使用默认值继续执行，不再抛出 KeyError

## [0.66.0] - 2026-04-26

### 新增

- **树莓派 3.5 寸 LCD 顶栏新增 CPU/内存双槽血条** ([scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py)):
  - 顶部状态栏文字整体上移，顶栏底部预留出一整条约 5px 的手机电量样式双槽血条，左侧显示 CPU 占用率，右侧显示内存占用率
  - CPU 与内存占用率共用显示线程内的 2 秒节流采样；CPU 从 `/proc/stat` 取样，内存从 `/proc/meminfo` 读取，不影响原有摘要、排行和图表的数据刷新频率
  - 两个分槽都采用统一的绿黄红分段阈值显示；自动息屏恢复后会重置采样基准，避免首帧误用跨长时间窗的旧值

### 测试

- **补充 CPU/内存双槽血条定向回归测试** ([tests/test_respi_35lcd_disp.py](tests/test_respi_35lcd_disp.py)):
  - 校验顶栏布局常量为双槽血条预留固定高度与分槽间距
  - 校验 CPU/内存共用的 2 秒采样节流，以及左右分槽各自的颜色填充逻辑

## [0.64.2] - 2026-04-26

### 修复

- **模型注册表改为按需加载并增加单模型 metadata 旁路** ([model_registry.py](src/lazybull/ml/model_registry.py)):
  - `ModelRegistry` 不再在初始化时无条件整包读取 `model_registry.json`，指定版本推理会优先读取 `v*_metadata.json`
  - 对只有旧版 `model_registry.json` 的模型目录，新增按版本流式提取单条 metadata 的兼容路径，避免树莓派等慢存储环境在 `paper_trade` 首次懒加载时卡在 90MB 级注册表解析

### 测试

- **补充模型注册表快速加载回归测试** ([test_ml.py](tests/test_ml.py)):
  - 校验无注册表文件时仍可通过 metadata 旁路加载指定版本
  - 校验指定版本和最新版本加载不会先回退到完整注册表解析

## [0.64.1] - 2026-04-25

### 修复

- **跨时间段稳定性汇总补齐回测参数分组键** ([compare_walk_forward.py](scripts/compare_walk_forward.py)):
  - `compare_walk_forward.py` 现在会把 `bt_top_n`、signal gate v2、市场择时、因子开关、分批调仓等 summary 已写出的训练/回测参数一并纳入对比表与跨时间段稳定性分组键
  - 修复 `wf_comparison_batches.xlsx` 的 `跨时间段稳定性` sheet 在 `skip_training` 扫描 `bt_top_n` 等参数时，误把不同参数组合合并成同一行的问题

### 测试

- **补充跨时间段稳定性 TopN 分组回归测试** ([test_ma250_observability.py](tests/test_ma250_observability.py)):
  - 校验同批次下不同 `bt_top_n` 会生成独立稳定性记录，不再错误折叠

## [0.64.0] - 2026-04-25

### 新增

- **paper_trade 与 bot_service 共用纸面交易运行时与展示层** ([runtime.py](src/lazybull/paper/runtime.py), [reporting.py](src/lazybull/paper/reporting.py), [paper_trade.py](scripts/paper_trade.py), [bot_service.py](scripts/bot_service.py)):
  - 新增 `src/lazybull/paper/runtime.py`，统一承载纸面交易日执行编排，完整覆盖止损、亏损提前换出、整体止盈、延迟卖出、T1、T0 与明日指令整理
  - 新增 `src/lazybull/paper/reporting.py`，统一承载模型信息、持仓快照、手机 Markdown 持仓展示与交易结果展示
  - `scripts/paper_trade.py run` 改为调用共享运行时，`scripts/bot_service.py trade / positions / model` 也改为复用同一层，不再各自维护一套交易步骤和格式化逻辑
  - `scripts/bot_service.py` 新增 `paper <paper_trade 子命令...>` 透传入口，低频 CLI 子命令可直接远程复用，无需再额外补机器人专用接线

### 测试

- **完成共享运行时与 CLI 相关验证**:
  - `d:/my_pro/LazyBull/.venv/Scripts/python.exe -m pytest tests/test_paper_trade_runtime.py tests/test_suspended_stock_handling.py`
  - `d:/my_pro/LazyBull/.venv/Scripts/python.exe -m pytest tests/test_paper_trading_cli.py`
  - `d:/my_pro/LazyBull/.venv/Scripts/python.exe -m py_compile scripts/paper_trade.py scripts/bot_service.py src/lazybull/paper/runtime.py src/lazybull/paper/reporting.py`

## [0.63.3] - 2026-04-25

### 优化

- **batch_walk_forward 参数区按开关控制关系重排并补充中文说明** ([scripts/batch_walk_forward.ps1](scripts/batch_walk_forward.ps1)):
  - 将 OOS 回测、信号门控、盈亏动态持仓、ATR 子开关、strength_veto、整体止盈、MA250、止损、ECT、行业过滤、行业轮动、市场择时等参数，按“总开关在前、受控参数紧跟”的方式重新排布
  - 为每组参数补充“仅在什么开关/模式下生效”的中文注释，减少无效扫描和参数误读

### 验证

- **完成脚本语法校验** ([scripts/batch_walk_forward.ps1](scripts/batch_walk_forward.ps1)):
  - 使用 PowerShell 语法解析器确认重排后脚本可正常解析

## [0.63.2] - 2026-04-25

### 优化

- **纸面交易 YAML 配置按开关控制关系分组展示** ([storage.py](src/lazybull/paper/storage.py), [config.yaml](data/paper/config.yaml)):
  - `data/paper/config.yaml` 现在会把“开关本身”和“仅在该开关开启时生效的参数”放在同一小段里，便于直接判断哪些参数当前有效
  - `PaperStorage.save_config()` 生成的 YAML 模板也同步采用相同分组方式，避免后续刷新模板后又回到旧的平铺布局

### 测试

- **补充配置模板分组文案断言** ([test_paper_trading_cli.py](tests/test_paper_trading_cli.py)):
  - 校验保存配置后生成的 YAML 模板包含“模型集成子开关”和“止损总开关”分组提示

## [0.63.1] - 2026-04-25

### 优化

- **纸面交易仅保留 YAML 配置文件** ([storage.py](src/lazybull/paper/storage.py), [paper_trade.py](scripts/paper_trade.py), [config.yaml](data/paper/config.yaml)):
  - `PaperStorage.load_config()` 现在只读取 `data/paper/config.yaml`
  - `PaperStorage.save_config()` 不再生成 `config.json` 快照
  - `reset_t0`、脚本提示和模板注释统一改为仅保留 `config.yaml`

### 测试

- **同步移除 JSON 兼容断言** ([test_paper_trading_cli.py](tests/test_paper_trading_cli.py), [test_adjust_truncate.py](tests/test_adjust_truncate.py)):
  - 校验保存配置后不再生成 `config.json`
  - 校验 reset 后仍保留的是 YAML 主配置

## [0.63.0] - 2026-04-25

### 新增

- **纸面交易主配置切换为带中文注释的 YAML 模板** ([config.yaml](data/paper/config.yaml), [storage.py](src/lazybull/paper/storage.py), [paper_trade.py](scripts/paper_trade.py)):
  - 新增 `data/paper/config.yaml` 作为纸面交易主配置入口，按模型、门控、组合、择时、止损、ECT、行业、仓位管理等模块分段展示，并带中文注释
  - `PaperStorage.load_config()` 现优先读取 `config.yaml`，仍兼容旧版 `config.json`
  - `PaperStorage.save_config()` 现在会同步写出带注释的 YAML 模板与 `config.json` 兼容快照，便于手工维护和兼容旧流程

### 测试

- **补充纸面交易 YAML 配置加载与模板写出测试** ([test_paper_trading_cli.py](tests/test_paper_trading_cli.py)):
  - 校验保存配置后会生成分段注释的 `config.yaml`
  - 校验分段 YAML 可以直接加载为扁平配置视图

## [0.62.0] - 2026-04-25

### 新增

- **纸面交易支持直接手工维护全量配置文件** ([storage.py](src/lazybull/paper/storage.py), [trading_config.py](src/lazybull/common/trading_config.py), [paper_trade.py](scripts/paper_trade.py), [config.json](data/paper/config.json)):
  - `PaperStorage.load_config()` 现在会自动补齐 `TradingConfig` 默认参数，并兼容旧版 `weight_method -> position_sizing` 字段映射
  - `PaperStorage.save_config()` 会写出完整配置面，后续可直接编辑 `data/paper/config.json`，不必再依赖长串 `paper_trade.py config --xxx` 命令
  - 仓库内现有纸面交易配置已升级为全量模板，便于按 `batch_walk_forward.ps1` 风格逐项填写和调整

### 测试

- **补充纸面交易旧配置自动升级测试** ([test_paper_trading_cli.py](tests/test_paper_trading_cli.py)):
  - 校验旧版 `config.json` 缺少新参数时，读取后会自动补齐完整默认值并迁移旧字段名

## [0.61.0] - 2026-04-25

### 新增

- **单次回测 / walk_forward 共享统一回测运行时** ([run_ml_backtest.py](scripts/run_ml_backtest.py), [walk_forward.py](scripts/walk_forward.py), [backtest_runtime.py](src/lazybull/common/backtest_runtime.py)):
  - 抽出统一的 `TradingConfig -> BacktestEngineML` 构造工厂，单次回测与 walk_forward OOS 回测改为共用同一套策略参数映射、signal 创建与滚动质量状态恢复逻辑
  - `run_ml_backtest.py` 现在与 walk_forward 一样支持并透传整体止盈、滚动质量监控、动态 Top-N、行业轮动加权、Kelly/半Kelly、空仓提前调仓等参数

- **纸面交易策略链补齐到 walk_forward 主策略面** ([paper_trade.py](scripts/paper_trade.py), [runner.py](src/lazybull/paper/runner.py), [broker.py](src/lazybull/paper/broker.py), [storage.py](src/lazybull/paper/storage.py)):
  - `PaperTradingRunner` 新增统一策略状态持久化，接入滚动质量监控、动态 Top-N、行业轮动加权、Kelly/半Kelly 仓位管理、整体止盈、空仓提前调仓
  - 纸面交易延迟卖出恢复为真正的 T+1 语义，整体止盈支持 `take_profit_refill` 控制是否允许次日自动回补
  - `paper_trade.py` 与补位链路改为透传 `TradingConfig`，减少脚本层与 runner 层手工拆参

- **公共 signal 工厂支持双模型加权集成** ([signal_factory.py](src/lazybull/common/signal_factory.py), [ensemble_signal.py](src/lazybull/signals/ensemble_signal.py)):
  - `model_version_b` + `ensemble_weight_a` 现在会创建统一的双模型集成 signal，单次回测和纸面交易可直接消费该参数组合

### 测试

- **补充并通过本次改造相关测试切片**:
  - `tests/test_equal_weight_lot_constraint.py`
  - `tests/test_pending_buy_estimation.py`
  - `tests/test_replenishment_no_sell.py`
  - `tests/test_holding_bonus_and_adaptive_topn.py`
  - `tests/test_industry_rotation_and_kelly.py`
  - `tests/test_paper_trading_cli.py`
  - `tests/test_paper_trade_realtime_summary.py`
  - `tests/test_suspended_stock_handling.py`
  - `tests/test_walk_forward.py`
  - `tests/test_market_and_new_features.py`

## [0.60.5] - 2026-04-25

### 优化

- **跨时间段稳定性汇总新增运行ID列表列** ([compare_walk_forward.py](scripts/compare_walk_forward.py)):
  - `wf_comparison_batches.xlsx` 的 `跨时间段稳定性` sheet 新增 `运行ID列表`
  - 每个稳定性分组会按 `批次时间段:运行ID` 的格式输出映射，例如 `0101:wf_xxx | 0209:wf_yyy`
  - 这样可以直接从稳定性汇总回溯到底层 run，无需再去 `实验对比` sheet 手工反查

### 测试

- **补充运行ID列表列断言** ([test_ma250_observability.py](tests/test_ma250_observability.py)):
  - 校验稳定性汇总在单批次和多批次场景下都会输出正确的 `运行ID列表`

## [0.60.4] - 2026-04-24

### 优化

- **compare_walk_forward 无数据来源也生成占位报表** ([compare_walk_forward.py](scripts/compare_walk_forward.py)):
  - 无参自动扫描模式下，即使 `data/walk_forward/raw` 当前没有 summary 文件，也会生成 `wf_comparison_raw.xlsx`
  - 占位报表会明确写出“无可用数据”，保证 batch 运行后 `raw / batches` 两份报表路径始终存在

### 测试

- **补充空 raw 目录占位报表测试** ([test_ma250_observability.py](tests/test_ma250_observability.py)):
  - 校验 raw 为空、batches 有数据时，无参 compare 仍会生成 `wf_comparison_raw.xlsx`

## [0.60.3] - 2026-04-24

### 修复

- **compare_walk_forward 跨时间段稳定性汇总按批次隔离** ([compare_walk_forward.py](scripts/compare_walk_forward.py)):
  - 同参数但不同 `batch_run_id` 的结果不再被错误合并到同一行稳定性汇总里
  - `wf_comparison_batches.xlsx` 的 `跨时间段稳定性` sheet 现在会为每次 batch 运行分别保留独立批次ID与时间段列表

### 优化

- **batch_walk_forward 运行后自动刷新 raw / batches 两份总表** ([batch_walk_forward.ps1](scripts/batch_walk_forward.ps1)):
  - 保留当前批次 `wf_comparison.xlsx` 输出不变
  - 额外自动执行一次无参 compare，刷新 `data/walk_forward/wf_comparison_raw.xlsx` 与 `data/walk_forward/wf_comparison_batches.xlsx`

### 测试

- **补充批次隔离回归测试** ([test_ma250_observability.py](tests/test_ma250_observability.py)):
  - 校验相同参数跨两个不同 batch 时，稳定性汇总会生成两行而不是被折叠为一行

## [0.60.2] - 2026-04-24

### 优化

- **compare_walk_forward 无参自动扫描 raw 与 batches** ([compare_walk_forward.py](scripts/compare_walk_forward.py)):
  - 直接执行 `py .\scripts\compare_walk_forward.py` 时，不再只读取 `data/walk_forward/raw`
  - 脚本现在会自动扫描 `data/walk_forward/raw` 与 `data/walk_forward/batches/*/raw`
  - 默认分别输出 `data/walk_forward/wf_comparison_raw.xlsx` 与 `data/walk_forward/wf_comparison_batches.xlsx`，避免要求手工指定 `--raw-dir`

### 测试

- **补充 compare 无参自动发现测试** ([test_ma250_observability.py](tests/test_ma250_observability.py)):
  - 校验临时 `data_root` 下同时存在 raw 与 batches 时，会自动生成两份对比 Excel

## [0.60.1] - 2026-04-24

### 优化

- **batch_walk_forward 恢复固定对比 Excel 输出路径** ([batch_walk_forward.ps1](scripts/batch_walk_forward.ps1)):
  - 保持本批次 `summary / chain_nav` 仍写入 `data/walk_forward/batches/<batch_id>/raw/`，继续隔离历史批次原始结果
  - 最终 `wf_comparison.xlsx` 改回输出到 `data/walk_forward/wf_comparison.xlsx`，便于沿用原有查找路径和人工查看习惯

## [0.60.0] - 2026-04-24

### 新增

- **walk_forward 批量实验支持多时间段批次** ([batch_walk_forward.ps1](scripts/batch_walk_forward.ps1), [walk_forward.py](scripts/walk_forward.py)):
  - `scripts/batch_walk_forward.ps1` 新增 `wf_period_configs` 时间段对象列表，可在同一批次内同时扫描多组 `WfStartDate/WfEndDate`
  - `skip_training` 模式下不再共用一个全局 `start_model_version`，而是改为每个时间段单独配置 `StartModelVersion`，避免跨时间段复用错误模型序列
  - 批量运行会把本批次 summary / chain_nav 输出到 `data/walk_forward/batches/<batch_id>/raw/`，与历史批次隔离，避免 compare 混入旧实验

### 优化

- **compare_walk_forward 增加本批次跨时间段稳定性汇总** ([compare_walk_forward.py](scripts/compare_walk_forward.py)):
  - 批量脚本调用 compare 时只读取当前批次 raw 目录，`wf_comparison.xlsx` 不再混入历史 raw 下的旧实验
  - Excel 新增 `跨时间段稳定性` sheet，按“相同参数组合、不同时间段”聚合，输出时间段数、时间段列表、综合得分均值/标准差、跨时间段 CAGR/回撤/跨切分 IR 统计与稳定性分
  - `compare_walk_forward.py` 读取 summary 时显式保留 `batch_run_id / batch_period_label / wf_start_date / wf_end_date` 字符串列，并增强综合评分对 `None`/字符串数值列的容错

### 测试

- **补充批次元数据与跨时间段稳定性测试** ([test_walk_forward.py](tests/test_walk_forward.py), [test_ma250_observability.py](tests/test_ma250_observability.py)):
  - 校验 walk_forward summary 会写出 `batch_run_id / batch_period_label`
  - 校验 compare 汇总表保留批次字段，并能对同参数的多时间段结果生成稳定性汇总

## [0.59.3] - 2026-04-23

### 优化

- **精简每日特征构建日志** ([builder.py](src/lazybull/features/builder.py)):
  - 删除 `开始构建 {trade_date} 的特征` 冗余日志（上游 `build_clean_features.py` 已有更醒目的分隔条 ETA 日志）
  - 将 3 个 horizon 的 `y_ret_N` 缺失 warning 合并为一行汇总：`{date} 标签缺失统计（...）: y_ret_5=52, y_ret_10=72, y_ret_20=106`
  - 日志行数由 4 行缩减为 0~1 行（无缺失时不打印）

## [0.59.2] - 2026-04-23

### 优化

- **特征构建进度日志增加 ETA 与分隔** ([build_clean_features.py](scripts/build_clean_features.py)):
  - 每日特征构建开始的日志前增加换行与 `=====` 分隔条，作为生成一个 feature 文件的明显起始标志，便于在长日志中快速定位
  - 基于已处理日期的平均耗时线性外推，附带打印"预计完成"绝对时间（`YYYY-MM-DD HH:MM:SS`），首轮显示"计算中"

## [0.59.1] - 2026-04-23

### 修复

- **抑制 storage.py 中 pandas concat FutureWarning** ([storage.py](src/lazybull/data/storage.py)):
  - `load_raw_by_date_range` / `load_clean_by_date_range` 在合并多日分区数据时，pandas 1.5+ 对含 all-NA 列的 concat 会输出 `FutureWarning`（典型场景：龙虎榜 `reason` 字段在某些日子整列为 NaN，北向资金在非交易日的占位记录等）
  - 在两处 `pd.concat` 调用外用 `warnings.catch_warnings()` 局部抑制该特定 message 的 FutureWarning，对其他 warning 无影响
  - 该警告对结果数据正确性无影响，待 pandas 2.x 默认行为变更后再统一适配

## [0.59.0] - 2026-04-23

### 变更

- **y_ret_N 标签语义对齐回测节奏** ([builder.py](src/lazybull/features/builder.py)):
  - 旧公式: `y_ret_N = close_adj(T+N) / close_adj(T) - 1` (T 收盘买 / T+N 收盘卖, 与回测引擎不一致)
  - 新公式: `y_ret_N = open_adj(T+1+N) / close_adj(T+1) - 1` (T+1 收盘买 / T+1+N 开盘卖)
  - 严格对齐回测引擎实际成交节奏 (`engine.py:600` "T 日生成信号 → T+1 日收盘价买入 → T+holding_period 日 sell_timing 卖出", 默认 `sell_timing="open"`), 消除训练-回测之间的成交价口径偏差
  - **影响**: 旧的 `data/features/cs_train/` 特征文件标签语义已变, 需要删除并重新跑 `python scripts/build_clean_features.py` 重建; 基于旧标签训练的模型不再可比
- **train_core.py 防泄露 delta 调整** ([train_core.py:660](src/lazybull/ml/train_core.py#L660)):
  - 新标签实际跨越 N+1 个交易日 (T+1 收盘 → T+1+N 开盘), 验证集切分间隔从 `max(horizon, 5)` 改为 `max(horizon + 1, 5)`, 杜绝隔夜跳空对截面排序的潜在泄露
- **测试更新** ([tests/test_features.py](tests/test_features.py)): mock_daily_data 补充 `open` 列, `test_forward_returns_calculation_correctness` 断言公式同步为新标签语义

## [0.58.0] - 2026-04-23

### 新增

- **下载并发 + 限流感知重试**: raw 数据下载从串行升级为可配置并发, 预计 2012-2026 全量下载从 24h+ 压缩到 4-6h
  - **线程安全令牌桶** ([tushare_client.py](src/lazybull/data/tushare_client.py)): `_rate_limit_wait` 用 `threading.Lock` 保护 `_last_request_time`, 多线程共享同一限频队列; 全局 QPS 始终受 `rate_limit` 严格约束, 不会超过 TuShare 配额
  - **限流感知重试**: 新增 `_is_rate_limit_error` 识别错误消息中的"每分钟/访问/频次/rate/limit/429"等关键字。命中限流 → 长等 `retry_rate_limit_sleep`(默认 15s, 让服务端限流窗口过去); 其他错误 → 固定 `retry_delay` 短等。彻底消除原先"token 错误也走 1+2+3=6 秒指数退避"的雪球效应
  - **按日并发** ([download_raw.py](scripts/download_raw.py)): 新增 `_run_concurrent` 执行器, 把"按交易日拉取"的 `daily` / `margin_detail` / `cyq_perf` / `top_list` 分发到线程池。`ErrorCollector` / `ProgressTracker` / 计数器全部加锁, 线程安全
  - **两级降级开关**:
    - `configs/base.yaml` → `tushare.download_concurrency: 4` 全局默认, 改 `1` 即退化回串行
    - 命令行 `python scripts/download_raw.py --concurrency 1` 临时覆盖, 排障时无需改配置
- **base.yaml 新增 tushare 配置项** ([base.yaml](configs/base.yaml)):
  - `rate_limit: 700` (从 500 调高, 8000 积分官方上限实测可承受; 若观察到限流频繁触发可回调至 500)
  - `download_concurrency: 4` (并发线程数, 主限瓶颈仍是 rate_limit 而非线程数)
  - `rate_limit_error_keywords: [每分钟, 访问, 频次, rate, limit, 频率, 429, 超过]` (限流识别词表)
  - `retry_rate_limit_sleep: 15` (命中限流关键字的重试等待秒数)

### 优化

- **download_raw.py 启动日志**: 新增"并发线程数 / 限频 / 限流重试等待"三行运行参数展示, 方便事后排障时从日志直接看出跑的是哪套参数

## [0.57.0] - 2026-04-23

### 新增

- **download_raw.py 全量重写**: 下载流程加入统一错误汇总、ETA 进度估算与多项隐患修复
  - **错误汇总**: 新增 `ErrorCollector`, 所有下载函数的单条失败都记录到全局收集器, 脚本结束时在总结页统一打印"按数据集分组的全部错误清单", 无人值守场景下运行完查日志即可定位; `finally` 保证即使中途异常/Ctrl+C 也会打印
  - **ETA 进度**: 新增 `ProgressTracker`, 基于已完成项的平均耗时估算剩余时间, 每 N 项打印一次 `elapsed / rate / ETA / 预计完成时刻`, 取代原"逐日 7 行日志"的噪声输出
  - **退出码**: 0=全部成功, 1=初始化/未预期异常, 3=存在错误项但脚本跑完, 130=用户中断, 便于批量脚本判断

### 修复

- **#1 默认 `--end-date` 硬编码未来日期**: 改为 `datetime.now().strftime("%Y%m%d")`, 避免日历过期/截断
- **#2 trade_cal 短窗口调用截断历史**: `download_basic_data` 改为"拉取并集窗口 + 合并旧数据去重排序"后保存, 小窗口调用不会抹掉历史日历
- **#3 moneyflow_hsgt 断点续传失效**: 先筛出真正待下载的交易日, 再基于缺失日期首末计算半年分段, 不再"全段批量重拉 + 逐日 skip"浪费 API
- **#4 moneyflow 强制依赖静默降级**: 返回空或异常时视为当日失败, 通过 `ErrorCollector` 记录并在总结页明确报错
- **#5 日线单日 6 接口非原子性**: 改为"先全部拉到内存, 任一失败整日不落盘"。避免"`daily` 成功但 `daily_basic` 失败 → 只有半个日子 → `is_data_exists` 跳过 → 永久缺失"
- **#6 字符串日期字典序比较**: `_generate_quarter_periods` / 日期参数校验改用 `_to_int_date`, 强制 8 位 YYYYMMDD 数值比较
- **#7 `_query_with_pagination` 整页多余空请求**: 恰好整页时用 `limit=1` 探测下页, 不再浪费一次 `page_limit` 调用
- **#8 `report_rc` --force 语义不一致**: force 模式下丢弃 `existing_df`, 与其他函数行为统一
- **#9 `_save_merged` dedup 顺序不明**: 新增 `sort_cols` 参数, dedup 前先排序, `keep="last"` 语义明确(保留最新 `ann_date`)
- **#10 `--resume` 死参数**: help 文案明确标注"等价于默认断点续传, 无需单独指定"
- **#11 `KeyboardInterrupt` 被当成未知异常**: 单独捕获, 退出码 130, 并在 `finally` 里打印已累积的错误汇总
- **#12 `stock_basic` 仅拉 L 导致生存者偏差**: 改为同时拉 `L/D/P` 三种状态并合并去重, 回测历史区间可覆盖已退市股票
- **#13 空交易日列表可能除零**: `download_daily_data` 等函数在 `trading_dates` 为空时提前返回

### 性能

- **下载慢根因分析 (2012-2026 全量耗时 24h+)**:
  - `rate_limit=500/分钟` (120ms/次) × 6 接口 × 3350 日 ≈ 40 分钟理论下限, 实测 24h+ 主因是
    TuShare 服务端限流触发客户端 `retry_delay * attempt` 指数退避 (1+2=3 秒/次)
  - 冷门接口共享同一 `_request_interval`, 无法按接口细分
  - 每日 7 行 logger.info 在 PowerShell stdout 下本身累计耗时
- **本次优化**:
  - 日线改为原子批处理, 每日最多 1 次失败 = 1 次进度 tick, 日志量下降约 80%
  - ProgressTracker 聚合打印, 日线/日 100 天/底层接口 1 行日志
  - 后续可进一步提速的建议(未在本版本实施, 需评估限流): rate_limit 提升到 700/分钟、识别限流错误后再长等避免对 404/token 错误做指数退避、按接口并发

### 优化

- **特征构建支持单值/多值两种标签过滤模式**：`scripts/build_clean_features.py` 将 `--horizon` 改为 `--horizon`（single）与 `--horizons`（all，AND 语义）互斥参数组，`required=True` 强制二选一
  - `--horizon 20`：新增 single 模式，仅按主 horizon 对应的 `y_ret_20` 非空过滤样本，仍生成 `y_ret_5/10/20` 三列标签，schema 不变
  - `--horizons 5 10 20`：保持 all 模式（AND 过滤，现有行为）
  - **修复停牌样本被 AND 过滤误丢的问题**：例如 T 日样本 T+10 停牌导致 `y_ret_10=NaN`，但 `y_ret_5/20` 均有值，single 模式下可保留该样本，恢复因停牌被无谓丢弃的训练数据
- [src/lazybull/features/builder.py](src/lazybull/features/builder.py)：`FeatureBuilder.__init__` 新增 `label_filter_mode` 参数（默认 `"all"`，向后兼容），`_apply_filters` 按模式分支执行
- [src/lazybull/features/builder.py](src/lazybull/features/builder.py)：`__init__` 新增防御性告警——当传入 `horizon` 不在 `horizons` 列表中时，日志警告并自动追加，避免静默回退到 `horizons[0]` 的坑
- [src/lazybull/features/ensure.py](src/lazybull/features/ensure.py)：更新缺数提示的推荐命令字符串，补充 `--horizon 20` 参数以匹配新的必填要求

### 测试

- 新增 [tests/test_builder_label_filter_mode.py](tests/test_builder_label_filter_mode.py)：覆盖 single/all 双模式在停牌场景下的过滤差异、默认模式兼容性、参数校验与 horizon 自动追加逻辑

## [0.56.3] - 2026-04-20

### 修复

- **调仓决策摘要"最终"行显示与计算结果不一致**：原先 `final_detail` 仅展示 `信号门控 x ECT x 市场层`，却未包含质量系数，导致 `final_target_exposure = 信号门控 x 质量` 与分项乘积对不上（例如 `45.8%[50.0% x 100.0% x 100.0%]`）
- 修改 [engine.py:_format_rebalance_decision_summary](src/lazybull/backtest/engine.py)，完整展开所有参与相乘的分项为 `信号门控=xx% x 质量=xx% x ECT=xx% x 市场层=xx%`，确保显示的计算链条与最终数值一致

### 测试

- 更新 `test_format_rebalance_decision_summary_is_explicit` 与 `test_ml_backtest_logs_unified_rebalance_summary_when_verbose_false` 断言以匹配新格式

## [0.56.2] - 2026-04-19

### 优化

- **walk_forward 调仓买入 warning 汇总日志改为三行格式**：首行展示计划买入数、计划资金占比、继承上轮持仓数量与资金占比，以及成功/失败数量；后两行分别展示成功仓位与失败仓位
- **成功/失败仓位按股票列表展示**：成功与失败行改为“数量 + 股票号列表 + 总资金占比”，失败项继续保留涨停、停牌等原因标记，减少每只股票逐个仓位打印的噪音

### 测试

- 更新调仓买入 warning 汇总日志测试，覆盖三行格式、继承上轮统计与成功/失败资金占比输出

## [0.56.1] - 2026-04-19

### 优化

- **walk_forward 调仓买入新增 warning 汇总日志**：`BacktestEngine._execute_pending_buys()` 现在会在每个 T+1 调仓买入日统一输出一条更醒目的 warning 日志，展示计划买入数、成功数、失败数，以及每只股票对应的目标仓位
- **失败明细直接带原因**：启用仓位补齐时，失败仓位明细会直接标注涨停、停牌等不可买入原因，便于快速定位当轮未满仓来源

### 测试

- 新增调仓买入 warning 汇总日志测试，覆盖计划/成功/失败数量与仓位明细输出

## [0.56.0] - 2026-04-18

### 新增

- **接入 C1/C2/C3 三大另类因子（北向资金 / 龙虎榜 / 一致预期）**，用于扩充 alpha 来源多样性
  - **北向资金（north_flow）**：TuShare `moneyflow_hsgt` 接口，市场级日度净流入（港股通/陆股通），广播到全样本后做 5/20 日滚动均值、20 日 z-score、连续同方向天数（sign_streak）等变换。6 个特征：`north_flow / _ma5 / _ma20 / _z20 / _sum5 / _sign_streak`
  - **龙虎榜（lhb）**：TuShare `top_list` 接口，个股日频上榜记录，按 `(trade_date, ts_code)` 聚合净买入、上榜理由数，再做 5/20 日滚动净额、20 日上榜次数。8 个特征：`lhb_on_list / _net_amount / _net_rate / _amount_rate / _up_days_20 / _net_sum_5 / _net_sum_20 / _reason_count`
  - **一致预期（consensus）**：TuShare `report_rc` 接口，研报滚动聚合（90 日窗分析师覆盖、FY1 EPS 均值、30 日 EPS 修订比例、目标价中位、评级五档量化）。5 个特征：`cons_analyst_count_30d / cons_eps_mean_fy1 / cons_eps_revision_30d / cons_target_price_mid / cons_rating_score`
  - **独立开关**：三类因子各自配备 `$enable_north / $enable_lhb / $enable_consensus` 开关（默认 `$false`，保持基线），贯通 PowerShell 批处理 → Python CLI (`--enable-{north,lhb,consensus}-features`) → `FeatureBuilder.build_features_for_day()` 新增可选参数 `north_flow_data / lhb_data / consensus_data`
  - **自动补齐链路**：`features/ensure.py` `_load_factor_data` 新增 3 段增量下载逻辑（moneyflow_hsgt 按日分区 / top_list 按日分区 / report_rc 按年分页），返回元组从 8 元素扩展到 11 元素
  - **train_core 因子前缀登记**：`NORTH_FEATURE_COLUMNS / LHB_FEATURE_COLUMNS / CONSENSUS_FEATURE_COLUMNS` 常量与 `prepare_training_data` 对应布尔开关
  - **`build_clean_features.py --build-all`** 已同步覆盖新 3 个因子（`OPTIONAL_FEATURE_FLAG_ATTRS` 扩展）
  - **`download_raw.py`** 新增 3 个数据集下载入口（`moneyflow_hsgt` 按区间批量 + `top_list` 按日分区空占位 + `report_rc` 按年分页增量），`ALT_DATASETS` / `--download` / `--all` 均已覆盖
  - **单元测试**：`tests/test_factor_north_flow.py / test_factor_lhb.py / test_factor_consensus.py` 共 13 条用例

## [0.55.5] - 2026-04-18

### 修复

- **Kelly/半Kelly 仓位权重重新设计**：彻底修复两个问题
  - **问题1（kelly = half_kelly）**：原实现 `f* *= 0.5` 对所有股票乘以相同常数，归一化后比例不变，两种模式结果完全相同。修复：`half_kelly` 改为 `50% kelly权重 + 50% 等权` 的混合模式，使其真正比 kelly 更保守
  - **问题2（量级不匹配导致全部 clip 等权）**：原始 ML 分数（0.5~0.9）直接除以方差（0.0001~0.001），`f*` ≈ 500~9000 全被 `kelly_max_leverage` clip 到相同值。修复：改用**分数百分位排名**（0~1）作为 μ，`f* = score_rank / σ²`，量级匹配且保留截面排序信息

## [0.55.4] - 2026-04-17

### 优化

- **统一仓位管理参数**：删除与 `position_sizing` 功能重叠的 `weight_method` / `bt_weight_method` 参数
  - `MLSignal.generate()` 不再做权重归一化，固定输出原始 ml_score（正分数），权重分配统一交由引擎层 `_normalize_signals` 处理
  - `BacktestEngine._normalize_signals` 补充 `position_sizing="score"` 分支实现（原为 fallback 等权）
  - `PaperTradingRunner` 新增 `_normalize_signals` 方法，支持 `equal` / `score` 两种模式（kelly 回退到 score）
  - 删除 `TradingConfig.weight_method` 字段和 `--weight-method` CLI 参数，`run_ml_backtest.py` / `walk_forward.py` / `paper_trade.py` / `batch_backtest.ps1` / `batch_walk_forward.ps1` 全部统一使用 `position_sizing`

## [0.55.3] - 2026-04-17

### 修复

- **`configs/base.yaml` 重新收口为真实默认值入口**：仅保留当前已接线的数据路径、TuShare、`industry.shenwan_level` 和交易成本配置，移除未实际驱动主流程的日志、回测、止损等历史段落，避免配置歧义
- **共享层默认值统一读取项目配置**：`Storage`、`PaperStorage`、`TushareClient`、`CostModel`、`ModelRegistry`、`Reporter`、`MLSignal` 及 `create_signal()` 的默认路径/参数现统一从 `configs/base.yaml` 派生；显式传参时仍优先于项目配置
- **训练/回测/分析脚本的 `--data-root` 改为“未指定时走 base.yaml”**：`train_ml_model.py`、`run_ml_backtest.py`、`walk_forward.py`、`compare_walk_forward.py`、`analyze_factor_importance.py` 与纸面交易相关脚本不再把 `./data` 写死为脚本级默认值
- **树莓派显示脚本改为跟随项目配置目录**：`respi_disp_real.py` 和 `3.5LCD_disp.py` 不再硬编码 `project_root/data` 与 `project_root/data/paper`
- **runtime 覆盖示例去除死配置**：`runtime_local.yaml`、`runtime_cloud.yaml` 仅保留当前仍有意义的路径/TuShare 覆盖示例，避免把未自动接线的旧段落误认为运行时默认值

### 测试

- 新增项目级数据路径、模型/报告目录、交易成本与 TuShare 默认值测试；配置/成本/存储/行业相关聚焦测试共 80 个用例通过

## [0.55.2] - 2026-04-17

### 修复

- **行业主口径升级为项目统一配置**：新增 `configs/base.yaml` 中的 `industry.shenwan_level` 项目级配置，支持 `l1` / `l2` / `l3`，默认 `l2`；训练、回测、纸面交易不再各自写死行业层级
- **`FeatureBuilder` 默认读取项目级行业口径**：`sw_industry*` 主字段不再固定绑定二级行业，而是按 `industry.shenwan_level` 映射到统一主口径；显式传入 `shenwan_level` 时优先于项目配置
- **行业中性化按主口径自适应回退**：`FeatureBuilder._apply_industry_neutralization()` 现按主口径自动选择 `L3→L2→L1→全市场`、`L2→L1→全市场` 或 `L1→全市场` 路径
- **行业约束与项目主口径对齐**：`load_industry_mapping()` 默认读取同一项目配置，回测和纸面交易的 `max_per_industry` 与特征构建、行业动量使用统一行业层级

### 测试

- 新增项目配置默认值、显式层级覆盖、行业约束层级切换相关测试；行业配置与行业约束聚焦测试共 51 个用例通过

## [0.55.1] - 2026-04-16

### 修复

- **行业主口径统一为申万二级**：`FeatureBuilder._merge_shenwan_industry()` 现将 `sw_industry*` 主字段统一绑定到申万二级行业；L3 明细继续保留在 `sw_l3*` 字段中，避免训练、回测、纸面交易在主行业口径上继续混用二级/三级
- **`ind_momentum_rank` 改为基于申万二级生成**：行业 alpha、行业动量绝对值和百分位排名统一按 `sw_industry` 计算；重建 features 后，训练、回测和纸面交易读取到的 `ind_momentum_rank` 将全部对应申万二级
- **行业约束映射与主字段对齐**：`industry_constraint.py` 优先使用统一主字段 `sw_industry`，使回测和纸面交易的单行业持仓约束与训练特征使用同一套申万二级口径
- **训练侧行业中性化统一到二级主口径**：`FeatureBuilder._apply_industry_neutralization()` 现使用 `L2→L1→全市场` 的回退路径，不再把三级行业作为训练主分组口径

### 测试

- 更新行业字段映射与分层中性化相关测试，覆盖申万二级主口径下的特征构建与行业约束行为

## [0.55.0] - 2026-04-16

### 新增

- **行业轮动加权**：`engine_ml.py` 的 `_post_filter_candidates` 新增步骤2——按行业动量排名对候选分数做乘性调整（`adjusted_score = score × (1 + alpha × (rank - 0.5))`），独立于已有的硬过滤开关，弱势行业中的超强个股仍有机会入选
- **Kelly / 半 Kelly 仓位管理**：`engine.py` 的 `_normalize_signals` 支持 4 种模式（`equal`/`score`/`kelly`/`half_kelly`），Kelly 公式 `f* = μ/σ²` 利用 ML 分数作为超额收益代理、近期价格方差估计波动率，并 clip 到 `kelly_max_leverage` 上限；缺失波动率或负分数的股票优雅回退到中位 Kelly 值
- **TradingConfig 新增 5 个参数**：`industry_rotation_enhanced`、`industry_rotation_alpha`、`position_sizing`、`kelly_vol_window`、`kelly_max_leverage`，支持 CLI 传参
- **batch_walk_forward.ps1 新增独立扫参开关**：行业轮动加权（`$industry_rotation_enhanced` + `$industry_rotation_alpha_list`）和仓位管理模式（`$position_sizing_list` + `$kelly_max_leverage_list`）可独立组合扫描
- **compare_walk_forward.py 新增展示列**：对比 Excel 中展示行业轮动加权和仓位管理相关参数

### 测试

- 新增 `tests/test_industry_rotation_and_kelly.py`（21 个测试用例），覆盖行业轮动加权、Kelly 仓位计算、参数校验、TradingConfig 字段

## [0.54.1] - 2026-04-15

### 修复

- **树莓派 3.5 寸 LCD 日内图边界点显示对齐**：在 `scripts/respi/3.5LCD_disp.py` 的显示层新增边界吸附，`11:30` 与 `13:00` 会贴住午休虚线，`15:00` 会贴住最右侧边界，避免日内折线在关键时点看起来“还差一点点”

### 测试

- 新增 `tests/test_respi_35lcd_disp.py` 中的边界吸附测试，覆盖午休分隔线与收盘边界的显示坐标修正

## [0.54.0] - 2026-04-15

### 新增

- **树莓派 3.5 寸 LCD 日内图轻度平滑显示**：`scripts/respi/3.5LCD_disp.py` 在日内图渲染前会对显示值做很轻的三点平滑，仅影响画线外观，不改实时采样、持久化历史和顶部数值
- **树莓派 3.5 寸 LCD 折线抗锯齿渲染**：图表折线现在会先在更高分辨率的小画布上绘制，再缩回图表区域，从而减轻高频刷新后明显的小锯齿感

### 测试

- 新增 `tests/test_respi_35lcd_disp.py` 中的平滑 helper 测试和真实 `ImageDraw` 抗锯齿渲染测试

## [0.53.0] - 2026-04-14

### 新增

- **树莓派 3.5 寸 LCD 日内图改为每次刷新都记点**：`scripts/respi/3.5LCD_disp.py` 的日内图不再按 10 分钟槽位覆盖同槽位内的新值，而是保留每一轮实时刷新采样点；同一 10 分钟内的多次刷新也会完整保存在当日历史中
- **日内图横坐标改为按真实盘中时间展开**：在继续折叠午休区间的前提下，新增基于真实盘中时间的 `x_positions`，因此同槽位内的多个点会沿 x 轴展开，折线会比原来更细、更接近实际刷新节奏
- **日内图持久化兼容升级**：重启脚本后加载当日历史时不再按槽位去重，旧版仅有 `slot_indices` 的持久化文件仍兼容；如果标签里带秒级时间，也会一并保留下来

### 测试

- 调整 `tests/test_respi_35lcd_disp.py` 的日内图测试，覆盖“同槽位多次刷新保留多点”“持久化读回不再去重”“旧版 payload 兼容归一化”

## [0.52.0] - 2026-04-13

### 修复

- **特征存储路径隔离**：纸面交易（推理）生成的特征文件存到独立目录 `cs_infer/`，与训练用 `cs_train/` 物理隔离，杜绝纸面交易先生成的无标签文件被训练路径误认为有效缓存的问题
- **build_clean_features.py require_label 恢复为 True**：批量构建特征时恢复标签非空过滤，尾部无标签样本不再混入训练数据
- **标签过滤逻辑从 OR 改为 AND**：`require_label=True` 时，原逻辑为"任一标签非空即保留"（OR），现改为"全部标签均非空才保留"（AND），确保 `y_ret_5/10/20` 均有值，避免部分 horizon 的训练集混入 NaN 标签
- **moneyflow 加载清理**：删除 `build_clean_features.py` 中已过时的 `AttributeError` fallback 分支及裸 `except:`，直接调用 `loader.load_clean_moneyflow`
- **特征缓存 schema 检查增强**：`_REQUIRED_FACTOR_COLS` 新增基本面、股东人数、业绩预告、筹码胜率、基金持仓、业绩快报的代表性列，确保旧缓存缺少高积分因子时能自动触发重建

### 新增

- **纸面交易功能对齐回测引擎**：将回测中已验证的多项高级功能移植到纸面交易，所有新参数均支持通过 `paper_trade.py config` 持久化，默认值与 `batch_walk_forward.ps1` 保持一致：
  - **市场择时仓位管理**：支持 binary/vol_target/trend/combined 4 种模式 + MA250 硬条件 + 回撤保护 + ATR 缩放，与 ECT 系数相乘得到最终仓位系数
  - **行业动量过滤**：在信号排序后、选取 Top-N 前，剔除行业动量排名后 `industry_momentum_bottom_pct` 的弱势行业股票
  - **持仓保留奖励**：对已持仓股票的 ML 分数加上 `holding_bonus_sigma × 截面标准差`，降低不必要换手
  - **盈利延续持有**：T0 生成卖出指令前，检查即将被卖出的持仓是否满足延续条件（pnl/strength 两种模式），满足则跳过卖出指令
  - **亏损提前换出 + strength_veto**：每日运行时评估持仓亏损情况，支持 ATR 动态阈值；strength_veto 模式下评分高于保护阈值时给予缓刑
  - **ATR 动态止损**：Position 新增 `buy_atr_pct` 字段，T1 买入时自动记录 ATR；亏损换出可使用 ATR 动态阈值替代固定阈值
  - **动态 Top-N 参数穿透**：统一使用 `create_signal(TradingConfig)` 创建信号，确保门控参数完整传递
- **TradingConfig 新增 ~27 个字段**：市场择时（15）、行业动量（2）、盈亏动态持仓（5）、亏损换出（3）、ATR 止损（2），均注册到 argparse
- **PaperStorage 新增缓刑状态持久化**：`save_early_exit_state()` / `load_early_exit_state()` 方法，支持 strength_veto 缓刑次数跨日持久化

### 测试

- 全部 702 个测试通过，无回归

## [0.51.2] - 2026-04-12

### 修复

- **特征存储路径隔离**：纸面交易（推理）生成的特征文件存到独立目录 `cs_infer/`，与训练用 `cs_train/` 物理隔离，杜绝纸面交易先生成的无标签文件被训练路径误认为有效缓存的问题
- **build_clean_features.py require_label 恢复为 True**：批量构建特征时恢复标签非空过滤，尾部无标签样本不再混入训练数据
- **moneyflow 加载清理**：删除 `build_clean_features.py` 中已过时的 `AttributeError` fallback 分支及裸 `except:`，直接调用 `loader.load_clean_moneyflow`
- **特征缓存 schema 检查增强**：`_REQUIRED_FACTOR_COLS` 新增基本面、股东人数、业绩预告、筹码胜率、基金持仓、业绩快报的代表性列，确保旧缓存缺少高积分因子时能自动触发重建

## [0.51.1] - 2026-04-12

### 修复

- **止损除零保护**：`StopLossMonitor.check_stop_loss` 新增 `buy_price <= 0` 前置检查；`check_stop_loss_for_positions` 中 `buy_price` 默认值为 0 时提前跳过，避免除零崩溃
- **ECT 回撤除零保护**：`EquityCurveTrader.calculate_exposure` 新增 `current_max <= 0` 检查，防止 NAV 异常时除零
- **信号权重除零保护**：`MLSignal.generate_ranked` 中 score 加权时新增 `scores.sum() < 1e-12` 回退等权，防止极小正分数导致 inf 权重
- **权重归一化空字典保护**：`BacktestEngine._normalize_signals` 新增空字典提前返回，防止 `1.0 / len(signals)` 除零
- **训练集标签泄漏防护加强**：`split_train_val_by_date` 在数据不足以保留 delta 间隔时，改为放弃验证集（返回空 DataFrame）而非使用有泄漏风险的数据
- **置信度门控空列表校验**：legacy 模式下新增 thresholds/exposure_levels 非空校验，避免配置为空列表时信号被永久阻断
- **因子除零阈值加强**：`FeatureBuilder` 中 `weight_avg_bias`、`opening_strength`、`vol_ratio` 的分母检查从 `> 0` 改为 `> 1e-6`，防止极小正数导致极值
- **盈亏率 NaN 显式检查**：亏损提前换出的盈亏率计算新增 `pd.isna(buy_pnl_price)` 检查，避免依赖 NaN 的隐式布尔行为

### 测试

- 新增 `test_delta_too_large_drops_val` 测试：验证 delta 大于数据量时验证集正确为空
- 调整 4 个 date_split 测试用例适配新的标签泄漏防护逻辑
- 全部 702 个测试通过，无回归

## [0.51.0] - 2026-04-12

### 新增

- **亏损提前换仓二次确认门控（strength_veto）**：原 `early_exit_loss_threshold` 触发后直接硬卖，现新增 `early_exit_mode` 参数支持二次确认：
  - `disabled`（默认，向后兼容）：保留原有"跌破阈值即卖"的逻辑
  - `strength_veto`：触发亏损阈值后调用 `HoldingStrengthScorer` 评分，评分 ≥ `early_exit_strength_protect_threshold`（默认 0.55）时否决卖出（"缓刑"），给暂时回调但趋势仍在的股票一次观察机会
- **缓刑次数兜底**：通过 `early_exit_max_reprieves`（默认 2）限制单只股票最大缓刑次数，防止无限拖延
- **early_exit 专用权重**：复用 `HoldingStrengthScorer` 但采用独立权重（ml_score=0.35, momentum=0.30, technical=0.20, fund_flow=0.15, drawdown=0.00），drawdown 归零因触发时已知亏损，侧重"模型是否看好"和"趋势是否恢复"
- **全链路透传**：walk_forward.py 新增 3 个 CLI 参数（`--early-exit-mode`、`--early-exit-strength-protect-threshold`、`--early-exit-max-reprieves`），batch_walk_forward.ps1 和 compare_walk_forward.py 同步支持参数扫描和对比报表

### 测试

- 新增 4 个 `TestEarlyExitModeValidation` 测试：校验非法 mode、disabled 无 scorer、strength_veto 创建独立 scorer、与 profit_extension scorer 互不干扰
- 全部 78 个回测引擎相关测试通过，无回归

## [0.50.0] - 2026-04-12

### 新增

- **盈利延续持有判据升级为多维度强势度评分**：原 `profit_extension_threshold` 单一浮盈率判据升级为可配置的 `profit_extension_mode`，支持三种模式：
  - `pnl`（默认，向后兼容）：保留原浮盈率 ≥ 阈值即延续的逻辑
  - `strength`：通过新建的 `HoldingStrengthScorer` 综合 5 个维度评分（ML 分数 30% + 动量加速 25% + 技术强度 15% + 资金筹码 15% + 回撤距离 15%），评分 ≥ `profit_extension_strength_threshold`（默认 0.6）才延续持有
  - `disabled`：持有期满直接卖出，不做延续
- **新文件 [src/lazybull/backtest/holding_strength.py](src/lazybull/backtest/holding_strength.py)**：包含 `HoldingStrengthWeights`（权重 dataclass，自动归一化）、`HoldingStrengthBreakdown`（评分分解，含日志格式化）、`HoldingStrengthScorer`（评分器，复用 features_by_date 中已有因子，不重复计算；缺失维度优雅降级到中位 0.5）。
- **TradingConfig 新增 3 个字段** + 对应 CLI 参数：`profit_extension_mode`、`profit_extension_strength_threshold`、`profit_extension_strength_weights`（5 个权重独立 CLI: `--profit-extension-strength-w-ml/momentum/technical/fund/drawdown`）。
- **engine_ml.py 覆写 `_get_holding_features_row` hook**：从 `features_by_date` 读取持仓股票当日截面特征行供 scorer 使用；基类 `BacktestEngine` 默认返回 None。
- **walk_forward.py / run_ml_backtest.py / batch_walk_forward.ps1 / compare_walk_forward.py 全链路透传**：支持 walk-forward 扫参对比 pnl/strength 两种模式，对比报表新增「盈利延持模式」「强势度阈值」两列。

### 测试

- 新建 [tests/test_holding_strength.py](tests/test_holding_strength.py)：覆盖权重归一化、评分分解日志、5 维度子评分（强势/弱势股票对比、ML 百分位排序、ATR 放大惩罚）、engine `__init__` 三种 mode 的校验与 scorer 注入，共 17 个测试全部通过。
- 更新 [tests/test_backtest_t1.py](tests/test_backtest_t1.py)：日志格式因新增 `[mode]` 标签变更，同步断言文案。
- 完整回归测试 690 通过（不含本变更前已存在的 3 个 `test_ml_backtest_engine.py` 失败用例）。

### 设计要点

- **完全向后兼容**：默认 `mode="pnl"` 保持原有行为，已有 walk-forward 实验基线不变；启用 `--profit-extension-mode strength` 才激活新机制
- **保留原浮盈维度**：浮盈率从"单一唯一判据"降格为"5 维度之一，权重 15%"，配合 ATR 波动率调整
- **不引入非调仓日卖出**：与 `stop_loss` / `take_profit_threshold` 职责边界清晰，仅改造持有期满的延续判据
- **轻量评分**：每只股票独立评分，使用经验阈值 + sigmoid 映射，不做截面 z-score（避免对全市场的批量依赖）

## [0.49.1] - 2026-04-11

### 修复

- **背光脚本补充官方硬件前提提示**：`scripts/respi/set_backlight.py` 在通过 PWM 路径成功设置亮度后，现在会明确提示微雪 3.5inch RPi LCD (C) 需要先用 0R 电阻或焊锡接通背光控制焊盘，否则 GPIO18 的 PWM 命令可能显示成功，但亮度不会变化。
- **主 LCD 脚本同步输出相同提示**：`scripts/respi/3.5LCD_disp.py` 初始化到 PWM 背光路径时，也会记录同样的硬件说明，避免把“软件调用成功但亮度不变”误判成主脚本逻辑问题。

### 测试

- 更新 `tests/test_respi_set_backlight.py`，新增 PWM 硬件说明文案与 `main` 输出断言。
- 更新 `tests/test_respi_35lcd_disp.py`，同步覆盖主 LCD 脚本初始化 PWM 路径时的硬件提示日志。

## [0.49.0] - 2026-04-11

### 新增

- **背光调节脚本默认显示亮度测试画面**：`scripts/respi/set_backlight.py` 在成功设置亮度后，现在会默认往 framebuffer 写入一张高对比度测试图，包含彩条、灰度条和棋盘块，便于在树莓派现场直接肉眼判断亮度是否合适。
- **预览参数可控**：新增 `--no-preview`、`--fb-path`、`--fb-width`、`--fb-height` 参数。默认会写预览图；如果只想静默调亮度，可以显式加 `--no-preview`。

### 测试

- 更新 `tests/test_respi_set_backlight.py`，新增测试图 framebuffer 数据生成、`main` 默认写入 preview、`--no-preview` 关闭 preview 等回归断言。

## [0.48.2] - 2026-04-11

### 修复

- **主 LCD 脚本的背光控制改为复用已验证的 helper**：`scripts/respi/3.5LCD_disp.py` 不再继续保留“固定 sysfs 节点 + RPi.GPIO.PWM”的旧实现，而是改为直接复用 `scripts/respi/set_backlight.py` 中已经验证过的背光 helper。
- **3.5LCD_disp.py 现已支持 sysfs 自动发现与 lgpio PWM**：在没有 `/sys/class/backlight` 节点时，主 LCD 进程会和独立调光脚本一样优先尝试 `lgpio`，并在亮屏/息屏切换时更新同一份 PWM 状态，而不是继续依赖只在部分环境可用的旧 GPIO 路径。
- **背光资源清理统一**：主 LCD 脚本退出时现在会通过统一 cleanup 入口释放 PWM/backlight 状态，避免主脚本与独立调光脚本走两套不同的资源释放逻辑。

### 测试

- 更新 `tests/test_respi_35lcd_disp.py`，新增主 LCD 脚本“初始化背光走 helper”“更新已有 PWM 状态”“退出时清理 helper 状态”等回归断言。
- 更新 `tests/test_respi_set_backlight.py`，新增 PWM 状态更新与 cleanup 的 helper 测试。

## [0.48.1] - 2026-04-11

### 修复

- **背光脚本不再只认固定的 `soc:backlight` 节点**：`scripts/respi/set_backlight.py` 现在会自动扫描 `/sys/class/backlight` 下所有可用设备，并优先选中发现到的真实背光节点；如果你的树莓派驱动导出的不是 `soc:backlight`，`--read` 和默认 `auto` 模式也能正常命中。
- **PWM 后端新增 direct `lgpio` 支持**：当没有 sysfs 背光节点时，脚本现在会先尝试 `lgpio` 直接 claim GPIO 并发送 PWM，再回退到 `RPi.GPIO`；比之前只走 `RPi.GPIO` 更适合 Bookworm / `rpi-lgpio` 环境。
- **增加现场排障能力**：新增 `--list`、`--backlight-root`、`--backlight-name`、`--gpiochip` 参数，便于在树莓派现场先枚举节点、再指定具体设备或 gpiochip 测试。

### 测试

- 更新 `tests/test_respi_set_backlight.py`，新增“自动发现非默认背光节点”“PWM 优先使用 lgpio 并在失败时回退 RPi.GPIO”“`main --read` 在默认路径缺失时仍能命中扫描到的节点”等回归断言。

## [0.48.0] - 2026-04-11

### 新增

- **树莓派 LCD 独立背光调节脚本**：新增 `scripts/respi/set_backlight.py`，支持通过命令行传入 `0~100` 亮度百分比，方便在树莓派上单独试背光，不必启动完整 LCD 显示程序。
- **自动选择背光控制方式**：脚本默认优先使用 sysfs 背光节点 `/sys/class/backlight/soc:backlight/brightness`；若节点不可用，可切换为 GPIO PWM 模式（默认 GPIO 18 / 1000Hz）。
- **支持读取当前亮度**：可通过 `--read` 直接查看当前 sysfs 背光原始值和百分比，便于现场摸清屏幕亮度到底如何调。

### 测试

- 新增 `tests/test_respi_set_backlight.py`，覆盖 sysfs 亮度换算、当前亮度读取、auto 模式优先走 sysfs、sysfs 不可用时回退 PWM 等行为。

## [0.47.12] - 2026-04-10

### 移除

- **多特征子集集成（SubsetEnsembleModel）**：完整移除 `SubsetEnsembleModel` 类、`_SHARED_MARKET_FEATURES`/`SUBSET_MOMENTUM_FEATURES`/`SUBSET_FUNDAMENTAL_FEATURES`/`SUBSET_CAPITAL_FLOW_FEATURES` 常量、`get_subset_ensemble_configs()` 函数、`_train_subset_ensemble_on_window()` 训练流程及所有 `--subset-ensemble` 参数。XGBoost 单模型通过跨特征交互天然优于分拆子集后融合，实测验证效果负向。
- **模型质量监控与降级**：移除 `--model-quality-enabled` / `--model-quality-ir-threshold` 参数及 walk-forward 主循环中的质量检查与降级逻辑。阈值概念在 walk-forward 场景中价值有限（每个模型仅用于一个测试期）。
- **市场自适应 Top-N**：移除 `TradingConfig` 中 `market_adaptive_topn_*` 三个字段、`BacktestEngine._compute_market_adaptive_topn_factor()` 基类方法与 `BacktestEngineML` 重写、信号生成中的调整逻辑、`--market-adaptive-topn-*` 参数及 `batch_walk_forward.ps1` 中对应开关。

### 测试

- `test_p1_optimizations.py` 仅保留 `TestEnhancedFeatures`（因子增强测试），移除 `TestSubsetEnsembleModel`/`TestSubsetConfigs`/`TestModelQualityDegradation`。
- `test_holding_bonus_and_adaptive_topn.py` 仅保留持仓保留奖励相关测试，移除 `TestMarketAdaptiveTopN`/`TestMarketAdaptiveTopNIntegration`/`TestCombinedFeatures` 及 TradingConfig 中自适应 Top-N 字段断言。

## [0.47.11] - 2026-04-10

### 优化

- **SubsetEnsembleModel 基本面子集增强**：`SUBSET_FUNDAMENTAL_FEATURES` 从 6 个扬展到 16 个，将 `FUNDAMENTAL_FEATURE_COLUMNS` 的 5 个财务质量因子（ROE/营收增速/净利润增速/负债率/单季度增速）直接内嵌为核心，不再与 `enable_fundamental_features` 开关耦合；并背诅加入行业动量锚点（ind_ret_avg/alpha_industry_20/ind_momentum_rank）、短期动量锚点（neu_ret_5）和综合评分（spec_score）。解决了基本面子集特征过少导致等权融合时被低 IC 子模型拉低整体排序质量的问题。
- **SubsetEnsembleModel 动态权重**：`_train_subset_ensemble_on_window` 改为收集每个子模型验证集 RankIC，按正 IC 値动态权重传入 `SubsetEnsembleModel`（替代原等权方案）；验证 IC 全部为负或无效时自动退化为等权。

### 修复

- **树莓派 3.5 寸 LCD 午休前最后一格可能缺失**：`scripts/respi/3.5LCD_disp.py` 之前在 11:30 一过就把午休视为非实时窗口，如果 11:29 是最后一次轮询，则不会再补抓上午收盘前的最后一笔，日内折线会停在离午休虚线前一格的位置。
- 本次为午休开始后增加了一个短暂的 11:30 补尾窗口：数据线程跨过 11:30 边界时会立即允许一次实时刷新，且在有限宽限时段内不会直接睡到 13:00，确保上午最后一个 10 分钟槽位有机会补齐。
- **顶部更新时间改为优先使用行情时间**：实时摘要成功后，顶部 `更新:HH:MM` 现在优先显示 snapshot/summary 中的 `quote_time`，不再简单显示本地抓取墙钟时间，因此午休前最后一次更新会更准确地反映为 11:30 而不是 11:29 之类的抓取时刻。

### 测试

- 更新 `tests/test_respi_35lcd_disp.py`，新增午休后仍允许补齐上午最后一格、11:30 边界强制刷新、午休补尾期间保持短间隔唤醒、顶部更新时间优先取 `quote_time` 等回归断言。
- 保留并通过 `tests/test_paper_trade_realtime_summary.py`，确认摘要逻辑未被本轮修复破坏。

## [0.47.9] - 2026-04-09

### 优化

- **树莓派 3.5 寸 LCD 顶栏状态文案更清晰**：`scripts/respi/3.5LCD_disp.py` 左上角调仓提示由“待调仓:n天”改为“下次调仓:MM/DD/剩n天”，同时数据线程刷新期间顶部中间会临时显示“更新中...”，处理结束后再恢复为“更新:HH:MM”。
- **0% 参考线改为白色并轻微下移**：日内图和周期图共用的 0% 基准线颜色统一改为白色，并整体下移 1 个像素，避免和中间网格线、标签视觉重叠过紧。
- **周期图右上角角标缩短**：周期图右上角文案从较长的“周期图最后数据日:MM/DD”缩短为“数据日:MM/DD”，减少标题栏拥挤感。

### 测试

- 更新 `tests/test_respi_35lcd_disp.py`，新增调仓文案格式化、0% 参考线偏移、渲染“更新中...”状态等回归断言，并同步更新周期图角标断言。
- 保留并通过 `tests/test_paper_trade_realtime_summary.py`，确认本轮显示层改动未影响实时摘要回退逻辑。

## [0.47.8] - 2026-04-09

### 修复

- **树莓派 3.5 寸 LCD 收盘切图过早**：`scripts/respi/3.5LCD_disp.py` 之前在 `15:00` 一过就按墙上时间切回周期图，即使周期图仍停留在前一交易日，也会立刻把当日日内图切走。
- 本次改为按数据状态切图：如果周期图尚未拿到当日数据，就继续显示当日日内图；只有周期图覆盖到当日目标交易日后，才切回周期图。
- **收盘后继续补齐日内尾点**：日内图构建现在优先使用 snapshot 的 `quote_time` 作为落点时间，收盘后保留一个有限的补齐窗口，继续尝试写入 `15:00` 最后一格，避免图线停在 `14:58` 或 `14:59`。
- **晚间恢复 10 分钟节奏**：收盘后的实时补齐只保留有限宽限时段，不会整晚维持 2 分钟高频轮询；超过宽限窗口后仍回到周期图的 10 分钟补数节奏。

### 测试

- 更新 `tests/test_respi_35lcd_disp.py`，新增“收盘后仍显示日内图直到周期更新”“收盘后按 quote_time 补齐 15:00 最后一格”“收尾期短周期唤醒与晚间恢复 10 分钟”等回归断言。

## [0.47.7] - 2026-04-09

### 优化

- **树莓派 3.5 寸 LCD 日内图新增午休分隔标记**：`scripts/respi/3.5LCD_disp.py` 现在会在 11:30 与 13:00 的折叠边界处绘制一条淡色竖向虚线，并在 x 轴中部补一个“午休”标记，帮助区分上午与下午数据的拼接位置。
- **0% 参考线强化显示**：0% 基准线改为单独颜色，并在图内追加 `0%` 小标签，不再与普通网格线共用同一种颜色，走势相对零轴的位置更容易判断。
- **不影响现有数据口径**：上述改动仅作用于显示层，不改变日内收益计算、持久化结构与数据刷新策略。

### 测试

- 更新 `tests/test_respi_35lcd_disp.py`，新增“日内图实际绘制出午休标记与 0% 标签”的渲染层回归测试。

## [0.47.6] - 2026-04-09

### 优化

- **树莓派 3.5 寸 LCD 日内图折叠午休时段**：`scripts/respi/3.5LCD_disp.py` 的日内 10 分钟槽位改为只覆盖 9:30-11:30 与 13:00-15:00 两段实际交易时段，11:30~13:00 午休区间不再占用中间 x 轴空间；日内折线会把上午收盘点与下午开盘点直接相邻显示，不再出现一长段难看的水平直线。
- **0% 参考线始终显示**：图表 y 轴范围改为统一通过 helper 计算，并强制包含 0%，因此无论三条线当天全部在正区间还是全部在负区间，0% 基准线都会持续可见。
- **兼容旧版日内持久化数据**：已有持久化 JSON 在加载时仍按时间标签重算槽位，迁移到新午休折叠坐标后不需要手动清理文件。

### 测试

- 更新 `tests/test_respi_35lcd_disp.py`，新增“午休折叠后 11:30 与 13:00 槽位连续”“图表 y 轴范围始终包含 0%”两组回归断言。

## [0.47.5] - 2026-04-09

### 优化

- **树莓派 3.5 寸 LCD 指数实时行情并入同一轮 snapshot**：`scripts/respi/3.5LCD_disp.py` 的 `_fetch_realtime_holdings_snapshot()` 现在会把上证和深证指数代码一起并入同一轮 `realtime_quote` 请求，日内图优先直接使用 snapshot 中解析出的指数涨跌幅，避免每次实时刷新再额外请求一次指数实时接口。
- **周期图新增同日缓存**：`scripts/respi/3.5LCD_disp.py` 的 `_fetch_cycle_chart_data()` 现在会按“自然日 + 目标交易日 + 调仓起点 + 账户持仓状态”缓存已经构建好的周期图 payload；同一天内重复刷新会直接复用缓存，减少重复 `daily` 和 `index_daily` 调用。
- **收盘补抓保持可重试**：只有当缓存结果已经覆盖“当前应有的最近交易日”时才会落缓存；如果收盘后当天日线数据尚未可用，周期图不会缓存缺失结果，10 分钟补抓仍会继续查询直到数据出现。

### 测试

- 更新 `tests/test_respi_35lcd_disp.py`，新增“同一轮 snapshot 合并指数行情”“优先复用 snapshot 指数涨跌幅”“周期图同日缓存命中”“收盘后目标交易日未齐时继续重试”等回归断言。

## [0.47.4] - 2026-04-09

### 优化

- **树莓派 3.5 寸 LCD 持仓实时行情去重复用**：`scripts/respi/3.5LCD_disp.py` 之前会分别为摘要、个股排行和日内图各取一轮持仓实时行情；本次改为先获取一份共享持仓快照，再复用于三块面板，避免同一次刷新里重复拉取持仓 `realtime_quote`。
- **盘中与非交易时段分频刷新**：盘中有效交易时段内，摘要/排行/日内图改为每 2 分钟刷新一次；周期图及非交易时段补数仍按 10 分钟节奏执行，但仅在尚未拿到“最近一个应有交易日”数据时才会继续尝试。
- **提取可复用的摘要计算 helper**：`scripts/paper_trade.py` 新增基于已获取行情 DataFrame 计算实时摘要的 helper，LCD 脚本可直接复用，保持市值、浮盈率、总盈亏率和年化收益口径一致。

### 测试

- 更新 `tests/test_respi_35lcd_disp.py`，新增共享快照复用、盘中 2 分钟刷新、目标交易日切换立即补抓、周末缺数重试等断言。
- 保留并通过 `tests/test_paper_trade_realtime_summary.py`，确认摘要在 `PRICE<=0` 时仍会回退 `PRE_CLOSE`。

## [0.47.3] - 2026-04-09

### 修复

- **树莓派 3.5 寸 LCD 日内图收益基准错误**：0.47.1 为了让首点归零，把日内显示序列改成了“相对当日首个有效点”的变化量，视觉效果接近按开盘价计算，不符合以昨日收盘价为基准的日收益口径。
- 本次恢复前收基准显示：`scripts/respi/3.5LCD_disp.py` 中的 `index_pct/shenzhen_pct/portfolio_pct` 重新直接使用原始当日涨跌幅，和 `raw_*` 字段保持一致，不再按首点重写。
- 旧版日内持久化 JSON 继续兼容：加载时仍会补齐 `raw_*` 字段，但显示值不再被重映射到 0 起点。

### 测试

- 更新 `tests/test_respi_35lcd_disp.py`，把日内图追加、持久化恢复和旧版载荷兼容断言切回前收基准口径。

## [0.47.2] - 2026-04-09

### 修复

- **树莓派 3.5 寸 LCD 盘前误显示实时异常值**：`scripts/respi/3.5LCD_disp.py` 之前把 8:30-15:30 统统当成盘中窗口，且数据线程启动时会立刻拉一次实时 summary/排行；盘前 `PRICE=0` 或无效报价会把总览和个股排行打成 `-100%`。
- 本次把日内图显示窗口调整为 9:30-15:00，并新增有效实时行情窗口判断；盘前不再生成日内点，午休时段也不会继续刷实时数据。
- `scripts/paper_trade.py` 的 `get_realtime_portfolio_summary()` 与 `scripts/respi/3.5LCD_disp.py` 的个股排行构建，现在在 `PRICE<=0` 时会回退到 `PRE_CLOSE`，避免盘前把持仓市值算成 0。
- 兼容旧版盘前日内 JSON：`_normalize_intraday_chart()` 现在会按时间标签重算槽位，并丢弃 9:30 之前的历史点。

### 测试

- 更新 `tests/test_respi_35lcd_disp.py`，新增盘前旧点清理、盘前不生成日内图、排行昨收回退、午休暂停实时刷新测试，并同步 9:30-15:00 的新窗口断言。
- 新增 `tests/test_paper_trade_realtime_summary.py`，覆盖实时 summary 在 `PRICE<=0` 时回退 `PRE_CLOSE` 的场景。

## [0.47.1] - 2026-04-09

### 修复

- **树莓派 3.5 寸 LCD 日内图首点不是零**：`scripts/respi/3.5LCD_disp.py` 原先直接使用“相对昨收的当日涨跌幅”绘制日内图，因此首个点会带着隔夜跳空，不符合“仅比较盘中变化”的口径。
- 本次新增日内图首点归零逻辑：持久化与内存中同时保留 `raw_*` 原始当日涨跌幅，以及 `index_pct/shenzhen_pct/portfolio_pct` 三条按首个有效点归零后的显示序列，三条线都会从 0 开始。
- 兼容旧版日内 JSON：加载旧文件时会把历史 `index_pct/shenzhen_pct/portfolio_pct` 当作原始值重建，不需要手动清文件。

### 测试

- 更新 `tests/test_respi_35lcd_disp.py`，新增旧版持久化载荷归零重建测试，并同步覆盖日内图追加、清洗和恢复后的 `raw_*` / 显示值双序列断言。

## [0.47.0] - 2026-04-08

### 新增

- **2.2 因子增强**（`--enable-enhanced-features`）：新增3个短线alpha因子
  - `opening_strength`（开盘强度）= open / pre_close - 1
  - `intraday_vol_structure`（日内波动结构）= (high - open) / (open - low)
  - `order_imbalance`（委托不平衡）= (buy_elg - sell_elg) / (buy_elg + sell_elg)，含5/20日滚动均值
  - 新因子经行业z-score标准化后加入训练特征
- **2.1 多特征子集集成**（`--subset-ensemble`）：训练3个独立模型（动量/基本面/资金流），加权融合预测
  - `SubsetEnsembleModel` 包装器：每个子模型使用各自特征子集预测，加权平均
  - 3类特征子集定义：`SUBSET_MOMENTUM_FEATURES`（技术动量19+）、`SUBSET_FUNDAMENTAL_FEATURES`（基本面价值6+）、`SUBSET_CAPITAL_FLOW_FEATURES`（资金博弈8+）
  - `get_subset_ensemble_configs()` 根据可用数据和启用的因子类别自动生成子集配置
  - `_train_subset_ensemble_on_window()` 一次加载数据、训练3个子模型并包装
- **3.3 模型质量监控与降级**（`--model-quality-enabled`）：训练质量不达标时自动回退上一合格模型
  - 当 `val_rankic_ir` < 阈值（默认 0.03）时，跳过当前模型，使用上一期合格模型做OOS回测
  - 支持连续降级（多个split质量不佳时持续使用最后一个好模型）
  - walk-forward完成时汇报降级次数

### 变更

- **删除 EnsembleMLSignal**：清理已废弃的双模型信号类（`ml_signal.py`、`signals/__init__.py`、`signal_factory.py`、`run_ml_backtest.py`），统一使用 `MLSignal`。`SubsetEnsembleModel` 对 `MLSignal` 透明，通过标准 `predict(X)` 接口即可。
- **batch_walk_forward.ps1**：新增 `$enable_enhanced`、`$subset_ensemble`、`$model_quality_enabled`、`$model_quality_ir_threshold` 开关
- **walk_forward_summary**：输出中新增 `enable_enhanced_features`、`subset_ensemble`、`model_quality_enabled`、`model_quality_ir_threshold` 字段

### 测试

- 新增 `tests/test_p1_optimizations.py`：16个测试覆盖增强因子常量、SubsetEnsembleModel预测/权重/特征并集、子集配置生成、模型降级逻辑（5种场景）

## [0.46.2] - 2026-04-08

### 修复

- **树莓派 3.5 寸 LCD 收盘切图后的周期图延迟刷新**：`scripts/respi/3.5LCD_disp.py` 的数据线程原先只按脚本启动时刻每 600 秒轮询。虽然 0.46.1 已修复“收盘后继续检查直到拿到当天周期图数据”，但在 15:30 切图边界并不会立刻再拉一次周期图，因此切换瞬间仍可能显示 t-1 缓存。
- 本次新增 `_get_data_worker_wait_seconds()`，在接近 15:30 时缩短等待时间，并在离开日内窗口后立即补一次周期图刷新，尽快把周期图推进到当天数据。

### 测试

- 更新 `tests/test_respi_35lcd_disp.py`，新增数据线程等待时序测试，覆盖常规轮询、15:30 前缩短等待和 15:30 后立即唤醒三个场景。

## [0.46.1] - 2026-04-08

### 修复

- **树莓派 3.5 寸 LCD 日内图异常点过滤**：`scripts/respi/3.5LCD_disp.py` 在计算日内持仓收益时，遇到 `PRICE<=0` 或明显异常实时价会回退到昨收；指数与持仓的异常涨跌幅不会再写入日内图，避免单个坏点把整张图压扁。
- **树莓派 3.5 寸 LCD 日内历史脏点清洗**：加载当天 `data/paper/state/respi_35lcd_intraday/YYYYMMDD.json` 时会过滤异常百分比点，已经持久化的坏数据不会继续影响当天图表缩放。
- **树莓派 3.5 寸 LCD 顶部图例拥挤**：日内图例改为短标签 `上证 / 深证 / 持仓`，并且日内模式不再显示 `周期图最后数据日` 角标，避免顶部文字重叠。

### 测试

- 更新 `tests/test_respi_35lcd_disp.py`，新增 0 价回退、历史异常点清洗、日内隐藏周期角标测试，并同步当前 LCD 布局常量断言。

## [0.46.0] - 2026-04-07

### 新增

- **持仓保留奖励（Holding Bonus）**：调仓时对已持仓股票在截面分数上加分（bonus = sigma × 截面std），降低不必要的换手。保留的持仓自动延续持有期，不产生交易成本。通过 `--holding-bonus-enabled` / `--holding-bonus-sigma` 控制。
- **市场自适应 Top-N（Market Adaptive Top-N）**：根据 `mkt_ret_avg_20` 判断市场趋势，牛市缩减选股数量（集中持股），熊市扩大选股数量（分散持股）。通过 `--market-adaptive-topn-enabled` / `--market-adaptive-topn-bull-factor` / `--market-adaptive-topn-bear-factor` 控制。
- **TradingConfig**：新增 5 个参数字段 `holding_bonus_enabled/sigma`、`market_adaptive_topn_enabled/bull_factor/bear_factor`，并注册对应 CLI 参数。
- **batch_walk_forward.ps1**：新增两组开关变量，可独立控制持仓奖励和市场自适应 Top-N。
- **run_ml_backtest.py**：支持传递持仓奖励和市场自适应 Top-N 参数到 BacktestEngineML。

### 测试

- 新增 `tests/test_holding_bonus_and_adaptive_topn.py`（21 个测试用例），覆盖持仓奖励加分/延续、市场自适应因子计算（牛市/熊市/缺失/NaN）、组合功能集成、TradingConfig 默认值。

## [0.45.7] - 2026-04-07

### 优化

- **树莓派 3.5 寸 LCD 顶栏时间和左右面板比例微调**：
  - `scripts/respi/3.5LCD_disp.py` 的顶部时间改为使用更大的独立字号，并小幅抬高顶栏高度；新增高度从底部图表区让出，保持中间数据面板高度不变。
  - 左侧总览面板宽度由 60% 收窄到 55%，给右侧两个盈亏排行面板更多横向空间，减少长文本挤压。

### 测试

- 更新 `tests/test_respi_35lcd_disp.py`，新增布局常量测试，覆盖顶栏高度、时间字号和左右面板比例。

## [0.45.6] - 2026-04-07

### 修复

- **树莓派 3.5 寸 LCD 首帧渲染可能卡在交易日历懒加载**：
  - `scripts/respi/3.5LCD_disp.py` 的显示线程首帧会先调用 `_select_chart_data()`，进而触发交易日判断；原实现可能在这里同步加载交易日历，并与数据线程启动期的导入竞争，现场表现为屏幕停留在 `LCD启动中`。
  - 修复后 LCD 显示线程在交易日历尚未加载时先按工作日快速判断，不再阻塞首帧显示；同时增加 `显示线程开始首帧渲染` 和 `显示线程已写出首帧` 的一次性日志，便于通过 SSH 判断是否已经越过正式渲染。

### 测试

- 更新 `tests/test_respi_35lcd_disp.py`，新增“未加载交易日历时走工作日兜底判断”的测试。

## [0.45.5] - 2026-04-07

### 修复

- **树莓派 3.5 寸 LCD 启动阶段信息不足，仍难判断是否写到了错误 framebuffer**：
  - `scripts/respi/3.5LCD_disp.py` 启动时新增 `LCD启动中` 测试页，先于正式渲染链路尝试写入 framebuffer，便于快速判断脚本是否已经进入显示阶段。
  - 新增环境变量 `LAZYBULL_LCD_FB_PATH`，可直接覆盖默认的 `/dev/fb1`，用于驱动把 LCD 挂到 `/dev/fb0` 或其他 framebuffer 设备的场景。
  - 主程序、配置加载、背光初始化、目标 framebuffer、线程启动等关键阶段现在都会同步输出到 stderr，便于通过 SSH 观察启动停在了哪一步。

### 测试

- 更新 `tests/test_respi_35lcd_disp.py`，新增 framebuffer 环境变量覆盖测试。

## [0.45.4] - 2026-04-07

### 修复

- **树莓派 3.5 寸 LCD framebuffer 打不开时静默黑屏且无输出**：
  - `scripts/respi/3.5LCD_disp.py` 中 `_write_fb()` 原先会直接吞掉 `/dev/fb1` 写入异常，导致 framebuffer 不存在、权限不对或设备号变化时，现场表现为“屏幕始终黑屏，SSH 也看不到任何错误”。
  - 修复后会把 framebuffer 写入失败原因输出到 stderr，并写入运行诊断日志，同时附带当前可见的 `/dev/fb*` 设备列表，便于确认驱动是否把 LCD 挂到了别的 framebuffer 号。
- **树莓派 3.5 寸 LCD 启动早期缺少可追踪日志**：
  - 新增 `data/paper/state/respi_35lcd_runtime.log` 运行诊断日志，覆盖主程序启动、配置加载、背光初始化、线程启动、自动息屏命中和线程异常退出等关键节点。
  - 若项目状态目录不可写，则自动兜底到系统临时目录中的同名日志文件。

### 测试

- 更新 `tests/test_respi_35lcd_disp.py`，新增 framebuffer 写入失败诊断测试。

## [0.45.3] - 2026-04-07

### 修复

- **树莓派 3.5 寸 LCD 显示线程异常时静默黑屏**：
  - `scripts/respi/3.5LCD_disp.py` 的显示线程原先若在 `_render()` 内抛异常，会直接停止刷新，现场表现为“执行后没反应、屏幕没有任何内容”。
  - 修复后显示线程会捕获异常，并把 `LCD显示异常` 和异常摘要直接画到屏幕上，同时向终端输出简短错误信息，避免无提示黑屏。
- **回测信号生成时排除已持仓股票，避免槽位被"重复买入"浪费**：
  - `src/lazybull/backtest/engine.py` 中原本仅在 `stagger_tranches > 1`（分批调仓）时排除已持仓股票。
  - 在"空仓/持有期拖尾提前调仓"场景下，残留持仓未到期，信号仍会选中它们，T+1 买入时被 `_buy_stock_direct` 的"已在持仓中"分支跳过，导致槽位浪费、实际持仓数低于目标。
  - 修复后无条件从候选中排除 `self.positions` 里的股票，让后续候选顶上空出的槽位；常规 `rebalance_freq == holding_period` 场景等效于"先卖后买"的结果，不影响正常换手逻辑。

### 测试

- 更新 `tests/test_respi_35lcd_disp.py`，新增错误消息裁剪格式测试。

## [0.45.2] - 2026-04-07

### 修复

- **树莓派 3.5 寸 LCD 周期图新增最后数据日角标，且盘外拿到当日数据后停止继续刷新**：
  - `scripts/respi/3.5LCD_disp.py` 的图表区右上角新增 `周期图最后数据日:MM/DD` 提示，直接显示当前周期图最后一个数据点的交易日。
  - 盘外刷新策略进一步收口：非交易时段不再持续刷新；仅在交易日收盘后且周期图最后数据日仍不是今天时，每 10 分钟继续检查一次。
  - 一旦周期图已经拿到今天的数据，当晚便停止继续轮询，避免不必要的外部请求。

### 测试

- 更新 `tests/test_respi_35lcd_disp.py`，新增周期图最后数据日格式测试，以及“盘外待当日数据/盘外已拿到当日数据/盘中”三种刷新策略测试。

## [0.45.1] - 2026-04-07

### 修复

- **树莓派 3.5 寸 LCD 盘外持仓周期图收盘后不刷新**：
  - 原实现中，`scripts/respi/3.5LCD_disp.py` 的数据线程虽然定义了 10 分钟刷新间隔，但循环里只有交易日 `8:30-15:30` 命中时才真正执行 `_fetch_data()`。
  - 结果是：如果脚本在盘中启动，盘外周期图会一直停留在最后一次盘中抓取结果；即使收盘后 `index_daily` 和个股日线已经包含今天，也不会自动刷新到今天。
  - 修复后将“周期图刷新”和“实时行情刷新”解耦：周期图和待调仓天数现在始终按 10 分钟刷新，实时行情/排名/盘中图仍只在盘中刷新。

### 测试

- 更新 `tests/test_respi_35lcd_disp.py`，新增刷新策略测试，覆盖“盘外仍刷新周期图、盘中同时刷新周期图和实时数据”的行为。

## [0.45.0] - 2026-04-07

### 新增

- **树莓派 3.5 寸 LCD 折线图扩展为上证/深证/持仓三线**：
  - `scripts/respi/3.5LCD_disp.py` 的盘中图和盘外持仓周期图都新增深证指数曲线，统一展示上证、深证和持仓三条线。
  - 持仓曲线颜色固定为橘黄色；上证使用亮黄色，深证使用青蓝色，三条线及图例可明显区分。
  - 周期图新增深证成指累计涨跌幅；盘中图新增深证成指当日实时涨跌幅；当天持久化文件同步升级为三线结构。

### 测试

- 更新 `tests/test_respi_35lcd_disp.py`，覆盖深证曲线在周期图、盘中图和当日历史恢复中的保留逻辑。

## [0.44.0] - 2026-04-07

### 新增

- **树莓派 3.5 寸 LCD 盘中图改为“上证实时涨跌 vs 持仓股当日实时涨跌”，并支持重启恢复当天历史**：
  - `scripts/respi/3.5LCD_disp.py` 的盘中第二条线由“持仓周期累计涨跌”改为“当前持仓股票相对昨收的实时涨跌汇总”，与上证指数当日实时涨跌直接对比。
  - 持仓股当日涨跌按持仓股昨收市值加权计算，不含现金仓位，更贴近“当前持仓股整体今日表现”的语义。
  - 新增 `data/paper/state/respi_35lcd_intraday/YYYYMMDD.json` 持久化文件，盘中每次刷新后都会落盘；脚本重启时会自动恢复当天已采集的 10 分钟槽位历史点。

### 测试

- 更新 `tests/test_respi_35lcd_disp.py`，新增覆盖持仓股当日涨跌口径和当天历史持久化恢复逻辑。

## [0.43.0] - 2026-04-07

### 新增

- **树莓派 3.5 寸 LCD 盘中/非盘中双图显示**：
  - `scripts/respi/3.5LCD_disp.py` 的底部图区改为双模式：非交易日及交易日 `8:30-15:30` 之外继续显示持仓周期图；交易日 `8:30-15:30` 之间切换为盘中图。
  - 持仓周期图的 x 轴改为固定槽位，规则为 `max(调仓周期, 当前持仓交易日数)`；例如调仓周期为 20 天时，持仓未满 20 天仍保持 20 个单元，超过后随持仓天数继续扩展。
  - 盘中图使用固定 10 分钟槽位，显示上证指数当日实时涨跌幅，以及当前持仓组合相对本轮持仓起点的累计涨跌幅，并随实时数据刷新同步更新。
  - 顶部时间格式改为 `4月7日(周二) 14:40:32`，顶部右侧文案由 `调仓:N天` 调整为 `待调仓:N天`。

### 测试

- 新增 `tests/test_respi_35lcd_disp.py`，覆盖顶部时间格式、持仓周期图固定 x 轴槽位、盘中图固定槽位更新与图表模式切换逻辑。

## [0.42.2] - 2026-04-07

### 修复

- **提前调仓污染门控/质量计算基准（补全 v0.42.1 遗漏字段）**：v0.42.1 仅快照了 `_separation_history`、`_composite_score_history`、`confidence_gate_history`、`_last_ranked_candidates`、`_signal_tracking` 是否包含当日 key，但 `_generate_signal` 还会修改以下状态，导致开/关 `enable_early_rebalance_on_empty` 时门控指标仍有漂移（实测 2026-02-06 `composite` 1.308 vs 1.292、`sep_pct` 0.62 vs 0.58、`hit_rate` 0.50 vs 0.61）：
  - `self._last_rebalance_nav`：每次调用都被重置为当前 NAV
  - `self._signal_tracking`：`_evaluate_expired_signal_quality` 会**删除**过期 key（仅判断"是否包含当日 key"无法还原删除）
  - `self._prediction_quality_history`：`_update_prediction_quality` 会追加质量记录
  - `self._rolling_quality_score` / `self._quality_warmup_remaining`：滚动质量评分与暖机倒计时
  
  修复：将 `_snapshot_early_rebalance_state` / `_restore_early_rebalance_state` 升级为对上述字段做完整深拷贝/还原（`_signal_tracking` 用 `dict` 全量拷贝而非长度标记），确保提前调仓拒绝路径完全无副作用。

## [0.42.1] - 2026-04-07

### 修复

- **提前调仓污染门控/质量计算基准**：`enable_early_rebalance_on_empty` 开启时，提前调仓尝试即使被门控阻断或拖尾拒绝，`_separation_history`、`_composite_score_history`、`confidence_gate_history`、`_signal_tracking` 等历史缓冲仍会追加条目，导致百分位归一化和滚动 hit_rate 的基准被"投机性评估"污染。表现为开/关同一开关时，正常调仓日的 `composite`、`sep_pct`、`quality hit_rate` 出现漂移。
  修复：新增 `_snapshot_early_rebalance_state` / `_restore_early_rebalance_state` 辅助方法；提前调仓 `_generate_signal` 调用前快照相关状态，信号若未真正入 `pending_signals` 则回滚所有追加条目。这样开/关该开关对正常调仓日的门控/质量计算完全一致。

## [0.42.0] - 2026-04-07

### 新增

- **持有期拖尾提前调仓（Early Rebalance on Holding Period Exceeded）**
  - 扩展 `enable_early_rebalance_on_empty` 开关的语义：除空仓场景外，新增"持有期拖尾"触发路径
  - 触发条件：`cycle_day >= holding_period` 且仍有残留持仓（通常为盈利延续持有的股票）
  - 决策规则：调用 `_generate_signal` 生成新一轮目标仓位，校验 `残留仓位占比 + 新信号权重合计 <= 100%`
    - 满足 → 信号入 `pending_signals`，T+1 买入，新旧持仓并存，旧盈利延续股票继续按原到期日卖出
    - 超过 → 撤回信号，继续等待残留持仓到期，次日再评估
  - 无论通过或拒绝均打印清晰日志：`持有期拖尾提前调仓评估/通过/拒绝`，含残留占比、新信号权重、合计占比
  - `walk_forward.py` 新增 `--no-early-rebalance-on-empty` CLI 开关
  - `batch_walk_forward.ps1` 新增 `$enable_early_rebalance_on_empty` 参数区

## [0.41.2] - 2026-04-07

### 修复

- **cycle_day 日志在门控阻断期间错误推进**：原 `cycle_day = idx % rebalance_freq + 1` 基于固定节奏，门控连续阻断导致持仓为空时 `cycle_day` 仍在每天推进（如显示 `本轮第[04/20]天`）。
  修复：引入 `_cycle_anchor_idx` 跟踪当前调仓周期起点，仅在"信号成功进入待买队列"（正常调仓日或空仓提前调仓）时才更新 anchor 并输出新一轮分隔线。语义变更：分隔线从"每 `rebalance_freq` 天固定输出"改为"每次信号成功入队列时输出"，更贴近"新一轮真正开始"的含义。

## [0.41.1] - 2026-04-07

### 修复

- **空仓提前调仓节奏冲突**：提前调仓成功后，原预定的调仓日仍会触发重复信号生成（导致"刚买完又调仓"）。
  修复：提前调仓成功后自动清理 `signal_dates` 中未来一个 `holding_period` 周期内的原预定调仓日，确保调仓节奏从提前触发日重新计算。

## [0.41.0] - 2026-04-07

### 新增

- **空仓提前调仓（Early Rebalance on Empty Position）——空仓时立即触发下一轮T0**
  - 新增 `enable_early_rebalance_on_empty` 参数（`BacktestEngine` 和 `TradingConfig`），默认启用
  - 主循环在每日卖出/买入执行完毕后检测：若持仓为空、无待执行信号、无活跃补齐槽位且当日非正常调仓日，立即调用 `_generate_signal()` 生成新一轮信号，T+1 执行买入
  - 解决场景：门控阻断导致持仓到期后空仓、整体止盈清仓、止损清完持仓等情况下，原先需傻等到下一个预定调仓日才重新入场，现在次日即可重新尝试建仓
  - 与现有 `take_profit_refill` 补位机制协调：补位槽有活跃条目时不会触发提前调仓，补位优先
  - 门控阻断时次日会自动再次尝试，直到门控放行或到达正常调仓日，行为自洽
  - 分批调仓场景下使用 `tranche_idx=0` 触发主批次

### 修复

- 修复提前调仓触发后，原预定调仓日仍会再次触发信号生成导致"刚买完又调仓"的问题：
  提前调仓成功后自动清理未来一个 `holding_period` 周期内的原预定调仓日，确保调仓节奏从提前触发日重新计算

### 测试

- `test_cycle_separator_is_logged_before_new_cycle_signal` 显式禁用该功能以保持原测试语义
- 全部 595 个测试通过（3个已有失败与本次无关）

## [0.40.0] - 2026-04-06

### 新增

- **信号入口门控 v2（composite 模式）——重新设计买入置信度门控公式**
  - 新增 `signal_gate_mode` 参数，支持三种模式：`legacy`（旧公式）、`composite`（新公式）、`disabled`（关闭）
  - **成本门控**（方案1）：预测收益低于交易成本 N 倍时直接持币，直接回答"这笔交易是否有正期望"
  - **绝对收益质量分**（方案3A）：`abs_quality_score = clip(top_mean / cost, 0, 2)`，衡量绝对预测收益而非仅看相对分离度
  - **百分位归一化**（方案3B）：用分离度在历史中的百分位替代不稳定的 `score_std` 归一化
  - **自校准阈值**（方案3C）：基于 `composite_score` 历史分位自动决定仓位，消除手动调参问题
  - 新增 `SignalConfidenceGateState` 字段：`abs_quality_score`、`separation_percentile`、`composite_score`、`cost_gate_passed`、`rolling_quality`

- **滚动模型质量监控（方案2）——根据模型近期实际表现动态调仓**
  - 新增 `signal_gate_quality_enabled` 参数，启用后追踪最近 N 个调仓周期的选股实际表现
  - 计算滚动 hit rate（选股跑赢全市场中位数的比例），低于阈值时线性降仓
  - 支持 EWM 半衰期平滑，模型"失灵"时自动收缩、恢复时自动放开
  - Walk-forward 模型换代时自动重置预热期，给新模型信任期

- **完整参数传递链路**
  - `TradingConfig`、`signal_factory`、`walk_forward.py`、`run_ml_backtest.py`、`batch_walk_forward.ps1` 全部支持新参数
  - `batch_walk_forward.ps1` 新增 `signal_gate_mode`、`signal_gate_cost_multiplier_list`、`signal_gate_quality_*` 参数区

### 测试

- 新增4个 composite 门控测试：成本门控阻断、高分数通过、历史缓冲区累积、disabled 模式
- 全部 591 个测试通过（1个已有失败与本次无关）

## [0.39.0] - 2026-04-03

### 新增

- **batch_walk_forward 显式支持 val_ratio 扫描与透传**
  - `batch_walk_forward.ps1` 新增 `$val_ratio_list` 参数，可直接固定单值或批量扫描多个训练集内部验证集比例
  - `batch_walk_forward.ps1` 的总任务数统计、参数笛卡尔积循环与 `walk_forward.py` 命令拼接现已同步接入 `--val-ratio`
  - 批量实验不再隐式依赖 `walk_forward.py` 的默认 `val_ratio=0.1`，便于直接对比 `0.1/0.15/0.2` 对早停轮次与样本外表现的影响

## [0.38.6] - 2026-04-03

### 修复

- **walk-forward 批量扫描默认去掉 skip-training 下的无效标签对比，并收口到更均衡的扫描区间**
  - `batch_walk_forward.ps1` 默认标签收口为单一 `y_ret_20`，避免在 `skip-training` 复用旧模型时重复跑一组无效的 `neu_y_ret_20` 对照
  - 默认门控矩阵收口到 `TopK=10`、`thresholds=[0.04,0.12,0.25]`、两组仓位系数，以及 `vol_target=0.20/0.22` 的均衡区间
  - 脚本执行阶段新增 `effective_label_list`，即使用户手工填入多个标签，`skip-training` 模式下也会自动只保留首个标签

- **compare_walk_forward 新增全周期 chain 指标，避免把 split 均值误读成全周期收益**
  - `compare_walk_forward.py` 现在会读取 `chain_nav_*.csv`，输出全周期 `CAGR`、总收益、链式最大回撤、链式夏普和链式交易日数
  - 综合评分从“回测年化收益均值 / 最差单 split 回撤”调整为优先参考“全周期 CAGR / 全周期链式最大回撤”
  - 指标说明同步明确区分“全周期 chain 指标”和“跨 split 均值指标”，减少实验复盘时的口径误用

### 测试

- 新增 `compare_walk_forward` 的全周期 chain 指标回归测试，覆盖 `chain_nav` 读取和 `CAGR / 最大回撤` 计算

## [0.38.5] - 2026-04-03

### 优化

- **batch_walk_forward 默认切换为更聚焦的防守型扫描矩阵**
  - `batch_walk_forward.ps1` 的默认标签由 `neu_y_ret_20` 调整为以 `y_ret_20` 为主、`neu_y_ret_20` 仅作对照，更贴近当前已有 OOS 最优防守型结果
  - 信号门控从单组固定参数升级为可批量扫描的配置集合，默认围绕 `TopK=8/10/12`、三组阈值和两组仓位系数做小范围搜索
  - 市场层默认只围绕 `vol_target` 的 `0.18/0.20/0.22` 做窄扫描，并把盈利延续天数 baseline 对齐到当前最佳防守型 run
  - 脚本在关闭门控时会自动退化为单配置占位，避免无效的门控笛卡尔积膨胀任务数

## [0.38.4] - 2026-04-02

### 修复

- **盈利延续持有日志增加延期后的预计卖出日期**
  - `engine.py` 在“盈利延续持有”日志末尾新增延期后的预计卖出交易日，便于直接核对延持后的退出时点
  - 日期按实际买入日和交易日序列推算，保持与当前持有期卖出规则一致；超出回测区间时会明确标注
  - 新增回归测试覆盖日志里同时出现“最多持有天数”和“预计卖出日期”

## [0.38.3] - 2026-04-02

### 修复

- **回测轮次分隔线前移到新轮首日业务日志之前**
  - `engine.py` 将“新一轮回测”分隔线从每日进度日志前，前移到新轮首日进入处理流程之前
  - 修复后同一轮的选股过滤、模型预测、卖买执行与调仓摘要都会落在正确的轮次标题下，不再出现“先选股再进入新一轮”的错位观感
  - 新增测试覆盖分隔线与新轮首日信号日志的先后顺序，防止后续回归

## [0.38.2] - 2026-04-02

### 修复

- **统一调仓决策摘要压缩为单行表格式**
  - `BacktestEngine` 的调仓决策摘要改为单行 `|` 分隔格式，批量回测时更适合横向扫读和 grep
  - 门控阻断时，ECT、MA250/ATR、市场择时等未评估层显示为 `N/A`，避免误读成 `100%`
  - 新增测试覆盖单行摘要格式、`verbose=False` 下的输出，以及门控阻断场景的单行展示

## [0.38.1] - 2026-04-02

### 修复

- **OOS 调仓决策日志统一为单一摘要输出**
  - `BacktestEngine` 新增统一的“调仓决策摘要”日志，固定展示信号门控、ECT、MA250/ATR、市场择时与最终目标仓位，不再因是否满仓而忽隐忽现
  - 信号门控在回测主链中改为由引擎统一汇总输出；即使门控直接阻断到 0 仓位，也会明确打印“不进入待买队列”
  - `BacktestEngineML` 将 MA250、ATR 缩放、市场择时与回撤保护结果写入同一摘要，避免多条分散日志难以判断最终生效层级
  - 新增测试覆盖摘要文案、`verbose=False` 时仍输出摘要、以及门控阻断场景

## [0.38.0] - 2026-04-01

### 新增

- **信号置信度门控 + 持币机制**
  - `MLSignal` / `EnsembleMLSignal` 新增 `signal_confidence_gate_*` 参数，根据 Top-K 候选的预测强度计算置信度分数，并映射为不同仓位系数
  - 置信度极低时支持直接返回空信号，实现低边际优势环境下的主动持币，而不是买入后再被动止损
  - `BacktestEngine` 在 `generate_ranked()` 主链上二次评估并应用门控，确保回测、walk-forward 与纸面交易行为一致
  - `walk_forward.py`、`compare_walk_forward.py`、`batch_walk_forward.ps1` 同步支持该能力，并新增门控持币率、平均仓位、平均置信度等实验观测列
  - 新增测试覆盖信号层、回测主链、walk-forward 汇总与 compare 输出

## [0.37.0] - 2026-04-01

### 新增

- **Walk-forward 主链接入更多 OOS 回测参数，并同步支持批量实验**
  - `walk_forward.py` 新增并透传 OOS 回测参数：`bt_sell_timing`、`bt_exclude_st`、`bt_min_list_days`、`bt_max_weight_per_stock`、`bt_max_per_industry`
  - `walk_forward.py` 新增 OOS 止损与 ECT 参数透传：`bt_stop_loss_*`、`bt_equity_curve_*`，直接复用回测引擎已有能力
  - `write_walk_forward_summary()` 新增写出上述参数，便于后续聚合分析
  - `compare_walk_forward.py` 新增对应参数列，避免实验对比时丢失关键风控配置
  - `batch_walk_forward.ps1` 新增对应批量扫描入口，可直接逐组实验卖出时机、单股/行业约束、止损与 ECT 组合
  - 新增测试覆盖 walk-forward 汇总与 compare 参数列保留行为

## [0.36.0] - 2026-04-01

### 新增

- **每日回测日志新增当前持仓 ATR 统计**
  - `engine.py` 的每日进度日志新增 `ATR(min/avg/max)` 段，统一展示当前持仓股票的 ATR 百分比区间
  - `engine_ml.py` 按当日持仓和当日 `atr_pct_14` 特征计算最小值、均值、最大值；缺失时回退为 `N/A`
  - 新增单元测试覆盖基础引擎的 `N/A` 占位输出，以及 ML 回测引擎的 ATR 统计输出

## [0.35.2] - 2026-04-01

### 修复

- **回测轮次分隔线只在本轮首日输出**
  - `engine.py` 将“新一轮回测”分隔线从每日输出修正为仅在 `本轮第[1/N]天` 前输出
  - 新增单元测试覆盖 `rebalance_freq=2` 下只在第 1、3 天打印分隔线的行为

## [0.35.1] - 2026-04-01

### 优化

- **MA250 日志改为简洁公式展示**
  - `engine_ml.py` 的 MA250 日志前缀从 `MA250模块` 精简为 `MA250`
  - `ratio` 比较结果改为 `ratio=数值:触发控仓/未触发控仓`，去掉 `threshold` 与 `hard_stop_exposure` 的冗余展示
  - ATR 缩放改为直接显示计算式 `atr_ma250/atr_now=...=...%`，更便于快速核对缩放结果

## [0.35.0] - 2026-03-31

### 新增

- **每日回测进度日志新增持仓目标与当前仓位显示**
  - `engine.py` 的每日进度日志从“持仓[当前持仓数]只”调整为“持仓/仓位[当前持仓数/目标持仓数]/[当前仓位%]”
  - 当前仓位按当日股票市值占组合总资产比例计算，便于快速识别未满仓、风控降仓或现金沉淀
  - 分批调仓时目标持仓数会按批次数自动放大，日志更符合组合整体视角
  - 新增单元测试覆盖日志格式和分批调仓目标持仓数计算

## [0.34.0] - 2026-03-31

### 新增

- **`build_clean_features.py` 新增 `--build-all` 参数**
  - 一次性启用基本面、另类数据、融资融券、筹码胜率、基金持仓、业绩快报六类可选因子，避免手工逐个输入开关
  - 保持行业中性化开关独立，不因 `--build-all` 隐式改变特征后处理逻辑
  - 新增单元测试覆盖 `--build-all` 打开全部可选因子、以及不启用时保持原值不变的行为

## [0.33.3] - 2026-03-31

### 修复

- **市场级 ATR 特征列未进入缓存失效判定，导致旧 features 持续缺列**
  - `features/ensure.py` 将 `mkt_atr_pct`、`mkt_atr_pct_ma250` 加入 `_REQUIRED_FACTOR_COLS`
  - `build_clean_features.py` 遇到旧 schema 时不再直接跳过，而是明确记录警告并自动重建
  - 修复后旧版 `cs_train` 缓存不会再长期缺少市场级 ATR 列，避免 `engine_ml.py` 持续打印“缺少有效ATR数据”

## [0.33.2] - 2026-03-31

### 优化

- **MA250 模块日志改为展示完整决策链**
  - `engine_ml.py` 新增 MA250 日志格式化函数，明确输出 `ratio` 与 `threshold` 的比较结果、是否触发硬条件、`hard_stop_exposure`、`base_after_ma250`、`final_after_atr`
  - ATR 缩放开启时追加输出 `scale`、`atr_ma250`、`atr_now`，避免 `base=20%, final=20%` 这类日志难以判断具体原因
  - ATR 数据缺失时明确标注 `final_after_atr=base_after_ma250`，减少误读

## [0.33.1] - 2026-03-31

### 修复

- **MA250 硬条件可观测性不足，导致开关看起来“无效”**
  - `walk_forward.py` 在 OOS 回测前新增 MA250 阈值命中统计，分别输出交易日命中数与调仓信号日命中数
  - 当 MA250 硬条件只命中过普通交易日、或完全没有命中调仓信号日时，明确输出提示日志，说明结果可能与关闭时接近
  - `compare_walk_forward.py` 补充 `market_regime_ma250_hard_stop`、`market_regime_ma250_threshold`、`market_regime_ma250_exposure` 参数列，避免汇总表中丢失关键对比维度

## [0.33.0] - 2026-03-31

### 新增

- **MA250 ATR 动态仓位缩放**（`market_regime_ma250_atr_scaling`）
  - 在 MA250 模块中新增第二步 ATR 缩放：`仓位B = 基准A × MA(ATR,250) / CurrentATR`
  - 市场级 ATR 使用每日全市场 atr_pct_14 截面中位数，抗异常值
  - 高波动自动降仓、低波动允许恢复到满仓（上限 1.0，下限 min_exposure）
  - 新增市场状态特征 `mkt_atr_pct`、`mkt_atr_pct_ma250`
  - 参数传递：`--ma250-atr-scaling` argparse 参数，batch_walk_forward.ps1 / compare_walk_forward.py 同步更新

### 移除

- **ATR 仓位缩放**（`atr_position_sizing`）
  - 删除个股层面 1/ATR 反比权重分配功能，由 MA250 ATR 整体仓位缩放替代
  - 清理 engine.py、engine_ml.py、walk_forward.py、batch_walk_forward.ps1、compare_walk_forward.py 中的相关代码

## [0.32.0] - 2026-03-30

### 新增

- **ATR 动态止损阈值**（`use_atr_for_early_exit` / `atr_multiplier`）
  - 用个股 ATR%（ATR÷收盘价）替代固定 `early_exit_loss_threshold`，亏损超过 N×ATR% 时提前换出
  - 高波动股阈值更宽、低波动股更严，比固定 -5% 更有个性
  - 需同时开启 `enable_profit_based_holding`
- **ATR 仓位缩放**（`atr_position_sizing`）
  - Top-N 入选后按 1/ATR% 反比分配个股权重，低波动股获得更高权重（类 risk parity）
  - 独立开关，与盈亏动态持仓无依赖关系
- **ATR 因子**（`atr_14`）
  - `factors/volatility.py` 新增 `calculate_atr()` 函数（True Range 的 14 日滚动均值）
  - `factors/precompute_technical_factors.py` 加入 ATR 预计算，输出 `atr_14` 列
  - `features/builder.py` 将 `atr_14` 纳入特征，可供 ML 模型训练使用
- **参数传递链**：`batch_walk_forward.ps1` → `walk_forward.py` → `BacktestEngineML` 完整打通
  - `batch_walk_forward.ps1`：新增 `$use_atr_for_early_exit`、`$atr_multiplier_list`、`$atr_position_sizing`
  - `walk_forward.py`：新增 argparse 参数 `--use-atr-for-early-exit`、`--atr-multiplier`、`--atr-position-sizing`
  - `compare_walk_forward.py`：新增 ATR 参数的中文表头映射

## [0.31.4] - 2026-03-29

### 修复

- **MA250 硬条件单独开启时意外触发 binary 择时逻辑**
  - `engine_ml.py` 在 `_get_market_regime_exposure` 中，MA250 检查未触发后、进入普通择时逻辑前，
    增加 `if not self.market_regime_enabled: return 1.0` 守卫
  - 根本原因：上一版修复将入口条件改为 `market_regime_enabled or market_regime_ma250_hard_stop`，
    导致 MA250 未触发时会继续执行 binary/vol_target 等常规择时，`market_regime_bear_exposure=0.3`
    被意外应用，表现为仓位始终被压到 30% 而与 `ma250_exposure` 设置无关

## [0.31.3] - 2026-03-29

### 修复

- **MA250 硬条件在 `market_regime` 关闭时完全失效**
  - `engine_ml.py` 修复 `_execute_pending_buys` 的入口条件：将 `if self.market_regime_enabled` 改为
    `if self.market_regime_enabled or self.market_regime_ma250_hard_stop`
  - 根本原因：MA250 硬条件的执行路径被包裹在 `market_regime_enabled` 判断内，单独开启
    `--market-regime-ma250-hard-stop` 而不开启 `--market-regime` 时，代码完全不进入该分支，
    导致无任何效果也无任何日志输出
  - 修复后 MA250 硬条件可独立于其他择时模式单独使用，符合其"系统级否决性保护"的设计意图

## [0.31.2] - 2026-03-29

### 修复

- **亏损提前换出误判停牌股票**
  - `engine.py` 修复 `_check_and_sell` 中停牌股票盈亏率计算错误的 bug
  - 根本原因：停牌日 `_get_pnl_price` 返回 None 时，错误地用不复权的 `buy_trade_price` 与后复权的 `buy_pnl_price` 比较，导致盈亏率严重失真（如算出 -66% 而实际未亏损）
  - 修复方案：价格不可用（停牌）时直接 `continue` 跳过该股票的亏损检查，等复牌后再评估

## [0.31.1] - 2026-03-28

### 优化

- **walk_forward.py 新增跳过训练模式**
  - 新增 `--skip-training` 参数：跳过模型训练，直接使用已有模型做 OOS 回测
  - 新增 `--start-model-version` 参数：指定第一个 split 对应的模型版本号，后续 split 依次 +1
  - skip-training 模式下自动跳过部署模型训练
  - `batch_walk_forward.ps1` 新增 `$skip_training` / `$start_model_version` 配置项，适用于只调回测参数而无需重新训练的场景

## [0.31.0] - 2026-03-28

### 新增

- **条件卖出 T+1 机制**
  - 亏损提前换出、整体止盈从"Tn 检查 + Tn 执行"改为"Tn 检查 + Tn+1 执行"，符合 A 股 T+1 规则和纸面交易实际操作
  - 新增 `pending_condition_sells` 待卖队列和 `_execute_pending_condition_sells()` 执行方法，参照止损 T+1 模式实现
  - 持有期到期、盈利延续到期为预定事件，保持 Tn 直接执行不变
  - `sell_timing` 参数默认值从 `"close"` 改为 `"open"`（Tn+1 开盘价卖出）

## [0.30.0] - 2026-03-27

### 新增

- **整体持仓止盈 + 自动补位**
  - `engine.py` 新增 `take_profit_threshold` 参数：当整体持仓浮盈率（市值加权，后复权口径）≥ 阈值时，
    立即清空全部仓位并通过 `unfilled_slots` 机制在 T+1 日自动补位买入（重新进入调仓流程）
  - `engine.py` 新增 `take_profit_refill` 参数：控制止盈后是否触发自动补位（默认开启）
  - `engine.py` 新增内部状态 `_last_ranked_candidates` / `_last_signal_date`，在每次 `_generate_signal()` 时更新，供止盈补位选股使用
  - 整体止盈优先级高于逐只盈亏判断（触发后直接清仓 return，跳过盈亏动态持仓逻辑）
  - `walk_forward.py` 新增 CLI 参数 `--take-profit-threshold` / `--no-take-profit-refill`
  - `batch_walk_forward.ps1` 新增 `$take_profit_threshold` / `$take_profit_refill` 配置项

## [0.29.0] - 2026-03-27

### 新增

- **方案一：Market Regime MA250 长周期硬条件（系统性熊市保护）**
  - `market_state.py` 新增 `mkt_ma250_ratio` 特征：全市场累积收益曲线 / MA250，< 1.0 表示大盘处于长期下行趋势
  - `engine_ml.py` 新增参数 `market_regime_ma250_hard_stop / threshold / exposure`：
    大盘跌破 MA250 时强制降至指定仓位（默认 0.0 = 完全空仓），优先级高于其他择时模式
  - `walk_forward.py` 新增对应 CLI 参数 `--market-regime-ma250-hard-stop / threshold / exposure`
  - `batch_walk_forward.ps1` 新增 `$market_regime_ma250_*` 配置项

- **方案二：盈亏动态持仓时长**
  - `engine.py` 新增 `enable_profit_based_holding` 参数，支持：
    1. **亏损提前换出**：持仓达到持有期 `early_exit_holding_ratio` 比例且亏损超过阈值时提前换出
    2. **盈利延续持有**：持有期满仍盈利超过阈值时，允许延续持有 `profit_extension_days` 天（趋势跟踪）
  - `walk_forward.py` 新增对应 CLI 参数（`--enable-profit-based-holding` 等5个参数）
  - `batch_walk_forward.ps1` 新增 `$enable_profit_based_holding` 等配置项

## [0.28.8] - 2026-03-27

### 新增

- **Walk-forward OOS 回测支持配置初始资金**
  - `walk_forward.py` 新增 `--bt-initial-capital` 参数（默认 100万），替换原有硬编码
  - `batch_walk_forward.ps1` 新增 `$bt_initial_capital` 配置项

## [0.28.7] - 2026-03-27

### 优化

- **卖出执行日志始终输出**
  - "卖出执行"（持有期到期）和"止损卖出执行"日志不再受 `verbose` 控制，始终打印
  - 与"买入执行"日志保持一致

## [0.28.6] - 2026-03-27

### 优化

- **回测进度日志增加持仓和收益信息**
  - 格式: `回测进度: 38/120 天, 日期: 2019-03-04(持仓20, 本轮:+1.23%, 年化+5.67%)`
  - 包含当前持仓数、本轮累计收益率、年化收益率（按 252 交易日折算）

## [0.28.5] - 2026-03-27

### 修复

- **仓位补齐不再对同一不可交易股票重复尝试**
  - 新增 `untradeable_stocks` 集合缓存当天已确认不可交易的股票（涨停/停牌等）
  - 后续槽位自动跳过，避免 5 个空槽位对同一涨停股尝试 5 次的无效操作
  - 不可交易的提示日志改为仅在 `verbose` 下输出一次
  - 补齐成功日志中 `候选池大小 9/5` 改为更直观的 `已补齐 1/5`

## [0.28.4] - 2026-03-27

### 优化

- **分批调仓日志增加批次标识**
  - 信号生成、买入失败、仓位未满、买入执行、补齐成功/失败/延迟/完成/放弃/跳过等日志均添加 `[批次 N/K]` 前缀
  - 分批调仓时，"信号生成"和"买入执行"汇总日志**始终输出**（不受 `verbose` 控制），便于确认各批次正常调度
  - 仅在 `stagger_tranches > 1` 时显示，不影响单批调仓的日志

## [0.28.3] - 2026-03-27

### 修复

- **分批调仓下仓位补齐预算未按 tranche 比例分配**
  - `_process_position_completion()` 补齐路径漏掉了 `current_value / stagger_tranches`
  - 导致补齐时每只股票的目标市值是正常买入的 K 倍（如 2 批时补齐按 5万买入而非 2.5万）
  - 现在补齐路径与正常买入路径预算一致

## [0.28.2] - 2026-03-27

### 修复

- **仓位补齐相关日志始终输出**
  - "买入失败"和"仓位未满"日志不再受 `verbose` 控制，始终打印
  - 与"补齐成功"日志保持一致，避免看到补齐成功却不知道为何触发补齐

## [0.28.1] - 2026-03-27

### 修复

- **分批调仓信号生成排除已持仓股票**
  - `_generate_signal()` 在 `stagger_tranches > 1` 时从候选列表排除 `self.positions` 中已持有的股票，顺延选择下一只补位
  - 避免不同 tranche 选到重复股票导致预算浪费（之前被 `_buy_stock_direct` 安全网跳过但资金闲置）
  - `stagger_tranches=1` 时行为完全不变

## [0.28.0] - 2026-03-27

### 新增

- **分批调仓 (`--stagger-tranches`)**
  - 将资金分为 K 份，各 tranche 错开 `rebalance_freq/K` 天依次调仓，降低单次调仓时点风险
  - `BacktestEngine` 新增 `stagger_tranches` 参数，`_get_rebalance_dates()` 返回 `Dict[date, tranche_idx]` 支持多 tranche 调度
  - `_execute_pending_buys()` 按 tranche 数量分配单次买入预算
  - `TradingConfig` 新增 `stagger_tranches` 字段及 argparse 参数
  - `walk_forward.py`、`run_ml_backtest.py`、`batch_walk_forward.ps1` 同步支持该参数
  - `stagger_tranches=1` 时行为与原有逻辑完全一致（默认值）

## [0.27.0] - 2026-03-26

### 新增

- **回撤归因分析脚本 (`scripts/ana/analyze_drawdown.py`)**
  - 基于 walk-forward 的 summary + chain_nav 数据，对指定时段回撤进行多维归因
  - 5个分析模块：Split级别对比、信号质量诊断（含信号→回测转化效率）、市场环境对比、净值回撤详情、综合归因报告
  - 3张 matplotlib 图表：全周期净值曲线+回撤、Split指标对比柱状图、Alpha vs 市场散点图
  - 自动检测最差splits或手动指定聚焦区间
  - 通过交易日历将 chain_nav 行号映射为真实日期
  - 新建 `scripts/ana/` 目录，后续分析脚本统一存放

## [0.26.1] - 2026-03-26

### 修复

- **特征构建时 fund_portfolio/cyq_perf 数据加载范围不足**
  - `build_clean_features.py` 中 `fund_portfolio` 使用精确日期范围加载，导致前序季度分区未被加载，point-in-time 查询在日期范围起始段无匹配 → fund 相关特征列缺失
  - `cyq_perf` 同理，精确日期范围导致 `diff(5)`/`diff(20)` 衍生特征在起始段全为 NaN
  - 修复：两者均改为使用7个月回溯的 `start_dt` 加载，与 `margin_detail` 的加载方式一致

## [0.26.0] - 2026-03-26

### 新增

- **多偏移集成训练 (`--ensemble-offsets`)**
  - 每个 walk-forward split 训练3个偏移模型（原始窗口 ± N个月），预测分数取平均
  - 新增 `EnsembleModel` 包装器（`src/lazybull/ml/ensemble.py`），对外提供与单模型相同的 `predict()` 接口
  - `MLSignal`、`BacktestEngine` 等下游代码无需修改，自动兼容
  - 通过 `--ensemble-offsets N` CLI参数控制偏移月数（0=禁用，1=±1个月→3模型）
  - `batch_walk_forward.ps1` 新增 `$ensemble_offsets` 变量，默认值 1
  - walk-forward summary 中记录 `ensemble_offsets` 参数
  - 部署模型训练同样支持多偏移集成
  - 重构 `execute_split_training`/`execute_deploy_training`，提取 `_train_model_on_window()` 公共函数消除代码重复

### 修改

- `batch_walk_forward.ps1` 中 `$feature_stability_filter` 默认值改为 `$false`（实验验证效果不佳）

## [0.25.0] - 2026-03-26

### 新增

- **特征稳定性筛选 (`--feature-stability-filter`)**
  - 新增 `filter_stable_features()` 函数（`train_core.py`），在模型训练前自动筛选跨时期IC方向一致的特征
  - 将训练集按时间等分成3段，逐段计算各特征的截面Spearman IC均值
  - 仅保留所有段IC方向一致且平均|IC|≥0.02的特征，移除不稳定特征
  - 通过 `--feature-stability-filter` CLI参数启用（`walk_forward.py` / `train_ml_model.py`）
  - `batch_walk_forward.ps1` 新增 `$feature_stability_filter` 开关
  - walk-forward summary 中记录筛选统计（feature_total/feature_stable/feature_removed）

## [0.24.2] - 2026-03-25

### 优化

- **修复树莓派 T0 信号生成阶段挂死问题**
  - 向量化 `_filter_untradeable_stocks`：将逐股 O(n×m) 循环替换为向量化 pandas 操作，Universe 停牌过滤从 ~60 秒降至 <1 秒
  - 新增 `FeatureBuilder.clear_caches()`：特征构建完成后释放内部缓存（市场状态、技术指标等），回收 ~20-50 MB 内存
  - 移除信号生成热路径上不必要的 `.copy()` 调用（runner/ml_signal），减少 ~10-15 MB 峰值内存
  - 信号生成前执行 `gc.collect()`，确保孤儿对象回收后再进入模型加载/预测
  - 模型加载和预测前增加诊断日志，便于定位未来可能的挂起问题

## [0.24.1] - 2026-03-25

### 优化

- **钉钉机器人交易结果增加收益信息**
  - `format_trade_result()` 末尾新增总资产、本轮收益率和总收益率展示
  - 与 `positions` 命令的收益计算逻辑保持一致
  - 价格数据不可用时静默跳过，不影响原有输出

## [0.24.0] - 2026-03-25

### 新增

- **钉钉机器人长时间命令进度报告**
  - 新增 `ProgressReporter` 类，在耗时命令执行期间每60秒自动向钉钉推送当前步骤和已用时间
  - `execute_trade` 新增 `progress_callback` 参数，在各关键步骤（止损检查、T1执行、T0数据下载/特征构建/模型推理等）报告进度
  - `handle_trade` 自动启动进度报告器，异常或完成时自动停止

### 修复

- **钉钉机器人命令无响应问题**
  - `process` 方法增加顶层 try/except，消息解析阶段的异常不再导致静默失败
  - 新增 `_safe_reply` / `_safe_reply_markdown` 方法：带重试（指数退避）+ Webhook 降级
  - 新增 `_ReplyFailureDetector`：拦截 dingtalk_stream 库内部吞掉的回复失败日志，触发重试
  - **所有** reply 调用（不仅是异常回复）统一使用安全方法，确保用户始终能收到反馈

## [0.23.0] - 2026-03-25

### 新增

- **融资融券因子独立开关**
  - 将融资融券因子（`rzye_chg_5`、`rzye_chg_20`、`rqye_rzye_ratio`、`margin_net_buy_ratio`）从另类数据因子（`enable_alt`）中剥离为独立开关
  - `train_core.py` 新增 `MARGIN_FEATURE_COLUMNS` 常量和 `enable_margin_features` 参数
  - `walk_forward.py`、`train_ml_model.py`、`build_clean_features.py` 新增 `--enable-margin-features` 命令行参数
  - `batch_walk_forward.ps1` 新增 `$enable_margin` 开关变量
  - `enable_alt` 仍控制股东人数和业绩预告因子，与融资融券因子互不影响

## [0.22.5] - 2026-03-24

### 修复

- **补位信号生成失败后无法重试的问题**
  - 补位信号生成失败时（如 margin 数据不可用），failed_buy_targets 被 `clear_failed_buy_targets()` 清空导致信息丢失，且 T1 run_record 已保存，重新运行时幂等检查直接跳过
  - 补位失败时将原始 failed_buy_targets 保存为 PendingBuy，确保失败目标不丢失
  - T1 幂等检查后增加 pending_buys 检查：指令已执行但有待处理的补位计划时，仍执行补位买入

## [0.22.4] - 2026-03-24

### 修复

- **纸面交易融资融券因子缺失导致推理失败**
  - 当日 margin_detail 数据尚未发布时，`margin_lookup.get(trade_date)` 返回 None，builder 跳过 merge，features 完全缺失 `rzye_chg_5`/`rzye_chg_20`/`rqye_rzye_ratio` 列
  - 新增当日数据额外重试下载逻辑：若查询表中无当日数据，单独调用 TuShare API 重试
  - 重试仍失败时抛出 `RuntimeError`，阻止生成不完整的 features 文件，并输出明确错误信息提示用户稍后重试

## [0.22.3] - 2026-03-24

### 优化

- **基金持仓因子内存优化，消除175万行全量加载瓶颈**
  - `_try_ensure_historical_fund_portfolio()` 改为逐分区加载+聚合，每次仅加载单季度数据（~44万行），聚合后立即释放
  - `_aggregate_fund_portfolio()` 去掉冗余 `.copy()`，避免百万行级 DataFrame 复制
  - `build_fund_portfolio_lookup_by_date()` 新增 `pre_aggregated` 参数，跳过重复聚合
  - fund_portfolio 峰值内存从 ~280MB 降至 ~35MB

## [0.22.2] - 2026-03-24

### 优化

- **特征构建全链路内存优化（第二轮），进一步降低树莓派内存占用**
  - 基金持仓回溯从2年缩短为1年，加载量减半
  - 纸面交易（单日模式）跳过技术指标 dict 缓存构建，节省 ~15MB
  - `build_features_for_day()` 返回后立即释放日线/因子数据
  - 消除 `hist_data` 和 `_slice_by_trading_days` 中的冗余 `.copy()`
  - 峰值内存从 ~250MB 降至 ~130MB

## [0.22.1] - 2026-03-24

### 优化

- **因子加载内存优化，修复树莓派 OOM 死机问题**
  - `_load_factor_data()` 每组因子处理完后立即释放原始 DataFrame 和 lookup 字典，配合 `gc.collect()` 强制回收
  - 复用 `forecast_df` 避免业绩快报段重复从磁盘加载（节省 ~20MB）
  - 峰值内存从 ~300MB 降至 ~170MB，树莓派 3B+（1GB RAM）可正常运行

## [0.22.0] - 2026-03-24

### 新增

- **Walk-forward 部署模型自动训练**
  - Walk-forward 所有 split 评估完成后，默认自动追加一次部署训练
  - 部署模型的 train_end 取最后一个 split 的 test_end，消除 test_window_months 导致的时间滞后
  - 模型注册到同一版本序列（`_wf`），metadata 中 `is_deploy=True` 标记区分
  - 支持 `--no-deploy-train` 参数禁用部署训练
  - `batch_walk_forward.ps1` 同步支持 `$deploy_train` 配置项

## [0.21.1] - 2026-03-24

### 移除

- **删除人气排名因子（hot_rank, hot_rank_chg_5）**
  - 因子重要性分析显示：100% 模型中 importance=0，对预测完全无贡献
  - 移除涉及文件：`factors/hot_rank.py`（删除）、`builder.py`、`ensure.py`、`train_core.py`、`loader.py`、`build_clean_features.py`、`download_raw.py`
  - ALT_FEATURE_COLUMNS 从 10 列缩减为 8 列
  - `_REQUIRED_FACTOR_COLS` 缓存校验列表同步移除，旧缓存将自动重建

## [0.21.0] - 2026-03-24

### 新增

- **因子重要性分析脚本 `scripts/analyze_factor_importance.py`**
  - 从已训练的 XGBoost 模型中提取 `feature_importances_`，跨模型聚合分析
  - 计算每个因子的平均/中位数重要性、排名、贡献占比、零值比例
  - 按因子类别（动量/技术指标/流动性/估值等19类）分组统计
  - 自动识别低价值因子（满足≥2项：低贡献 / 高零值率 / 排名靠后）
  - 支持 `--last-n` 只分析最近 N 个模型，`--output` 指定输出路径
  - 输出终端报告 + CSV 文件（`data/reports/factor_importance.csv`）

## [0.20.8] - 2026-03-23

### 修复

- **features/ensure.py — 禁止下载和使用未来数据（前视偏差）**
  - `end_dt` 从 `trade_date + 1个月` 改为 `trade_date`，`trading_dates_str` 不再包含未来交易日
  - `_try_ensure_historical_cyq_perf` 调用前过滤日期 `<= trade_date`，防止下载未来筹码胜率数据
  - `_try_ensure_historical_fund_portfolio` 调用前过滤日期 `<= trade_date`，防止下载未来季度基金持仓
  - clean 日线数据（daily/daily_basic/moneyflow）加载范围截止到 `trade_date`
  - `FEATURE_DATA_FUTURE_MONTHS` 常量从 1 改为 0

## [0.20.7] - 2026-03-16

### 修复

- **全局代码审查修复14个隐藏bug**（P0/P1/P2三级）
  - **ml_signal.py**: score权重模式下负分股票占top_n槽位但权重为0，现在预先过滤负分股票
  - **cleaner.py**: 上市天数默认值从999改为-1，防止未知上市日的股票误通过min_list_days过滤
  - **train_core.py**: 训练/验证集分割delta间隔不足时添加warning日志，提示标签泄漏风险
  - **engine.py**: 回测补齐候选数量从unfilled_count扩大为2倍buffer，解决不可交易候选消耗名额导致补齐不足
  - **cleaner.py**: ST正则表达式"退"字匹配改为"退市"精确匹配，使用非捕获分组避免pandas warning
  - **trading_config.py**: 统一dataclass和argparse默认值（horizon: 5, rebalance_freq: 20）
  - **express.py/fund_portfolio.py**: 日期转换增加datetime类型检测，避免`.str[:8]`截断丢失天数
  - **cleaner.py**: 涨跌停判断从相对容差(0.1%)改为绝对误差(0.01元)，避免高价股误判
  - **eval_utils.py**: 类型注解`any`修正为`Any`
  - **ml_signal.py**: ensemble预测移除fillna(0)，与单模型保持一致让XGBoost原生处理NaN
  - **storage.py**: Parquet/CSV写入改为先写.tmp再rename的原子操作，防止中断导致文件损坏
  - **fund_portfolio.py**: `_symbol_to_ts_code()`增加NaN/None防护
  - **storage.py**: 日期格式校验改用datetime.strptime验证，覆盖2月30日等无效日期
  - **cleaner.py**: 去重前按主键列排序，确保结果不依赖输入顺序

## [0.20.6] - 2026-03-15

### 修复

- **express 因子全 NULL** — TuShare express_vip 实际返回的列名与代码期望的不一致：`revenue_yoy`/`n_income_yoy`/`roe` 在 API 中不存在，实际为 `revenue`/`yoy_net_profit`/`diluted_roe`，导致 `row.get()` 全部回退为 NaN
  - 修正列名映射：`n_income_yoy` → `yoy_net_profit`，`roe` → `diluted_roe`
  - 新增 `_compute_revenue_yoy()` 函数：利用同公司去年同期 `revenue` 自行计算营收同比增速
- **fund_portfolio 因子全 NULL** — TuShare fund_portfolio API 返回的 `symbol` 已含交易所后缀（如 `600820.SH`），但 `_symbol_to_ts_code()` 假设输入是纯 6 位数字，导致重复拼接为 `600820.SH.SH`，merge 时完全无法匹配
  - 在 `_symbol_to_ts_code()` 开头检测：若 symbol 已含 `.SH`/`.SZ`/`.BJ` 后缀则直接返回

## [0.20.5] - 2026-03-15

### 修复

- **修复41个失败测试** — 源代码经过多次功能迭代后，部分测试未同步更新导致接口不匹配
  - 修复 `split_train_val_by_date` IndexError：当训练日期数 < delta 时索引越界，改用排序后的实际日期集合
  - 更新 `test_walk_forward.py`：适配 `write_walk_forward_summary` 新增的 `args` 和 `wf_run_id` 参数
  - 更新 `test_buy_replacement.py`：修正 DataLoader/Storage 的 mock 路径
  - 更新 `test_pending_order.py`：适配 `mark_success` 新增 `success_date` 参数和 `get_orders_to_retry` 元组返回值
  - 更新 `test_profit_tracking.py`：字段名 `profit_amount/profit_pct` → `pnl_profit_amount/pnl_profit_pct`
  - 更新 `test_suspended_stock_handling.py`：适配 `AccountState`、`TradeInstruction`、`PendingSell` dataclass 等接口变更，补充 `_get_suspend_calendar` mock
  - 更新 `test_trade_status.py`：适配缺失数据假定停牌的设计
  - 更新 `test_cost.py`：适配默认佣金率和印花税率变更
  - 更新 `test_market_and_new_features.py`：适配新增4个市场状态特征，跳过仅批量模式返回的特征
  - 更新 `test_ml.py`/`test_multi_horizon_labels.py`：适配模型注册严格版本检查
  - 更新 `test_new_features.py`：适配行业特征函数已注释的现状
  - 更新 `test_rebalance_freq.py`：修正测试逻辑（传入非法类型而非合法整数）
  - 更新 `test_position_completion.py`：适配有限候选池补齐策略

## [0.20.4] - 2026-03-15

### 修复

- **build_clean_features 筹码胜率/基金持仓数据加载失败** — `load_cyq_perf()` 和 `load_fund_portfolio()` 未传日期范围参数，导致走单文件加载路径而非按日期分区加载，数据返回 None，构建出的特征文件缺失对应列
  - 修改 `build_clean_features.py`: 传入 `start_date`/`end_date` 参数以正确使用 `load_raw_by_date_range`

## [0.20.3] - 2026-03-15

### 修复

- **钉钉机器人消息丢失** — `SimpleHandler` 从 `ChatbotHandler` 改为 `AsyncChatbotHandler`，`process()` 在线程池中执行，不再阻塞 asyncio event loop 导致 WebSocket 心跳超时断线
  - 去掉 `process()` 的 `async` 关键字（`AsyncChatbotHandler` 要求同步方法）
  - `handle_trade` 去掉手动 `threading.Thread`（线程池已自动处理）
  - 增加顶层异常捕获和消息日志，防止静默失败

## [0.20.2] - 2026-03-15

### 优化

- **fund_portfolio 改为按季度分区存储** — 数据量巨大（近年单季度100万+条），从单文件改为按季度末日期分区Parquet存储
  - `download_raw.py`: `download_by_period` 新增 `partition_by_period` 参数，fund_portfolio 每季度独立保存
  - `loader.py`: `load_fund_portfolio` 支持 `start_date`/`end_date` 范围加载（`load_raw_by_date_range`）
  - `ensure.py`: 新增 `_try_ensure_historical_fund_portfolio`，按季度分区补齐缺失数据（回溯2年），替代旧的单文件 `_try_download_fund_portfolio`

## [0.20.1] - 2026-03-15

### 优化

- **6个因子数据批量下载优化** — 从逐股下载改为按日期/季度/月份批量下载，API调用次数降低1~2个数量级
  - **cyq_perf**: 按 trade_date 逐日下载全市场（替代逐股），按日分区存储
  - **fina_indicator**: 按 period（季度）批量下载全市场（替代逐股~5000次调用）
  - **forecast**: 按 period（季度）批量下载全市场（替代逐股~5000次调用）
  - **express**: 按 period（季度）批量下载全市场（替代逐股~5000次调用）
  - **fund_portfolio**: 按 period（季度）批量下载全市场（替代逐基金~8000次调用）
  - **stk_holdernumber**: 按月份批量下载全市场（单次限3000条，替代逐股~5000次调用）
- **TuShare API 方法增强**
  - `get_fina_indicator_by_date` / `get_forecast_by_date` 新增 period 参数
  - `get_stk_holdernumber_by_date` 重构为 `get_stk_holdernumber`，支持 ts_code/ann_date/start_date/end_date
- **ensure 全量下载优化**: 数据不足时的全量回退从逐股下载改为按季度/月份批量
- **fund_portfolio ensure 自动下载**: 数据不足时自动按季度下载，不再仅提示离线下载

## [0.20.0] - 2026-03-15

### 新增

- **筹码胜率因子 (cyq_perf)** — 5000积分API
  - 新增因子模块 `src/lazybull/factors/cyq_perf.py`
  - 特征: winner_rate, weight_avg_bias, cost_concentration, winner_rate_chg_5, winner_rate_chg_20
  - CLI开关: `--enable-cyq-features`
- **业绩快报因子 (express_vip)** — 5000积分API
  - 新增因子模块 `src/lazybull/factors/express.py`
  - 特征: express_revenue_yoy, express_profit_yoy, express_roe, express_surprise
  - express_surprise 通过交叉引用 forecast 预告数据计算业绩惊喜
  - CLI开关: `--enable-express-features`
- **基金持仓因子 (fund_portfolio)** — 5000积分API
  - 新增因子模块 `src/lazybull/factors/fund_portfolio.py`
  - 特征: fund_hold_ratio, fund_hold_ratio_chg, fund_count, fund_count_chg
  - 基金级数据聚合到个股级，按 ann_date point-in-time 对齐
  - CLI开关: `--enable-fund-features`
- **TuShare客户端新增3个API方法**: get_cyq_perf, get_express_vip, get_fund_portfolio
- **DataLoader新增3个加载方法**: load_cyq_perf, load_express, load_fund_portfolio
- **下载脚本支持**: download_raw.py 新增 cyq_perf/express/fund_portfolio 数据类型
- **ensure自动补齐**: features/ensure.py 支持3组新因子的增量下载
- **批量脚本开关**: batch_walk_forward.ps1 新增 $enable_cyq, $enable_fund, $enable_express
- **全链路支持**: walk_forward.py, train_ml_model.py, build_clean_features.py 均支持新因子开关

## [0.19.3] - 2026-03-15

### 修复

- **trade next 连续执行卡在同一天** (bot_service.py, storage.py)
  - 原因: 无成交的交易日 `last_update` 不更新，`next` 始终解析到同一天
  - 修复: PaperStorage 新增独立的 `last_trade_date` 字段（state/last_trade_date.json），
    与账户 `last_update` 完全解耦，仅供 `trade next` 推算下一交易日
  - 每次 trade 执行后均记录 `last_trade_date`，reset-t0 时随 state 目录自动清理

## [0.19.1] - 2026-03-15

### 优化

- **trade 命令支持 next 参数** (bot_service.py)
  - 输入 `trade next` 自动获取下一个交易日并执行交易

## [0.19.0] - 2026-03-15

### 新增

- **钉钉机器人新增 reboot 指令** (bot_service.py)
  - 支持通过钉钉发送 `reboot` 远程重启树莓派系统
  - 使用 `sudo reboot` 执行，help 命令同步更新
- **钉钉机器人新增 reset-t0 指令** (bot_service.py)
  - 支持通过钉钉发送 `reset-t0` 远程重置纸面交易数据，恢复为新账户状态
  - 等同于 `paper_trade.py adjust reset-t0` 操作

## [0.18.6] - 2026-03-14

### 优化

- **钉钉机器人消息格式优化** (bot_service.py)
  - trade/positions 命令返回信息中增加调仓状态: "已持 Xd 剩 Yd"
  - 股票编号保留完整后缀(.SH/.SZ), 不再截断
  - T0 明日交易指令买卖分组显示(买入清单和卖出清单分开), 避免操作时混淆
  - T0 格式精简: 买入显示"量/权", 卖出显示"量/因", 移除无意义的"价格:open/close"
  - positions 输出精简: 股票名前置、代码带后缀放括号内, 持仓信息压缩为两行
  - T1 操作明细同样按买卖分组显示

## [0.18.5] - 2026-03-14

### 修复

- **纸面交易 reset-t0 后持有天数为负值**
  - `reset_t0` 只回滚了 `last_update`，未清理持仓和现金，导致 positions 中的
    `buy_date` 晚于重新运行时的当前交易日，`get_holding_days()` 计算出负值
  - 修复: `reset_t0` 改为全量重置——清空所有子目录（state/trades/nav/runs/instructions/
    pending_buys/pending_sells），仅保留 `config.json`，账户恢复为初始资金的新账户状态

## [0.18.4] - 2026-03-14

### 修复

- **纸面交易 hot_rank 特征缺失导致推理失败**
  - `_try_download_hot_rank` 将快照标记为当天日期，目标交易日非当天时 lookup 返回 None，
    特征 DataFrame 完全缺失 `hot_rank`/`hot_rank_chg_5` 列，导致模型推理报"特征列一致性检查失败"
  - 修复: FeatureBuilder 在 hot_rank 数据不可用时仍创建这两列（填充 NaN），
    保证特征 schema 与模型一致（XGBoost 原生支持 NaN）
  - 同时将 `hot_rank`/`hot_rank_chg_5` 加入 `_REQUIRED_FACTOR_COLS`，旧缓存自动淘汰重建

## [0.18.3] - 2026-03-14

### 修复

- **纸面交易推理特征缺失导致信号生成失败**
  - features 缓存校验 (`_REQUIRED_FACTOR_COLS`) 未包含行业中性化特征（`zscore_*`/`neu_*`/`alpha_industry_*`/`ind_*`），
    导致缺失中性化特征的旧缓存通过校验，推理时触发"特征列一致性检查失败"
  - 修复: 将中性化代表性特征加入校验列表，旧缓存自动淘汰并触发重建
  - 同时将申万行业数据缺失从 warning 降级为 error 并中断构建，避免静默生成不完整特征

## [0.18.2] - 2026-03-14

### 修复

- **纸面交易因子数据自动下载不完整**
  - **fina_indicator/stk_holdernumber/forecast**: 原增量下载仅获取单日公告（1~59条），
    point-in-time 查询需要全量历史数据（10万+条）。首次运行时改为逐股全量下载，
    支持断点续传；后续运行按公告日增量追加。同时增加最低记录数阈值，
    低于阈值的残留数据也会自动触发全量重下
  - **申万行业分类**: 原本需要手动运行 `update_basic_data.py --only-shenwan`，
    现在 ensure 链路自动检测并下载三级行业分类数据
  - **hot_rank（人气榜）**: 保持当日快照增量模式（历史回填需数小时，不适合自动触发）

## [0.18.1] - 2026-03-14

### 修复

- **moneyflow 自动下载在 daily 已存在时被跳过**
  - `ensure_raw_data_for_date()` 中 moneyflow 下载逻辑原本位于 `if not daily_exists:` 分支内
  - 当 daily 数据已缓存时，moneyflow 下载被完全跳过，导致纸面交易报"缺少 clean moneyflow 数据"
  - 修复: 将 moneyflow 下载移至与 `daily_basic`/`margin_detail` 同级的独立检查块，带 `is_data_exists` 判断

## [0.18.0] - 2026-03-14

### 新增功能

- **纸面交易 `adjust reset-t0` 子命令**
  - 自动查找最新T0运行记录并重置，允许重新执行T0工作流
  - 自动删除该T0生成的T1交易指令文件
  - 清空所有延迟买入/卖出队列
  - `PaperStorage` 新增 `find_latest_t0()` 和 `reset_t0()` 方法
  - `PendingBuy` 新增到 `paper` 模块的公开导出
  - 用法: `python scripts/paper_trade.py adjust reset-t0`

## [0.17.1] - 2026-03-14

### 修复

- **补齐 AKShare 依赖声明**
  - 在 `pyproject.toml` 和 `requirements.txt` 中新增 `akshare` 依赖
  - 修复 `src/lazybull/features/ensure.py` 中按需导入 `akshare` 时，Pylance 在工作区 `.venv` 下报 `无法解析导入“akshare”` 的问题

## [0.17.0] - 2026-03-13

### 新增功能

- **纸面交易缺失数据自动下载**（`ensure` 链路全覆盖）
  - `ensure_raw_data_for_date()` 新增 `daily_basic`、`margin_detail` 自动下载
  - `ensure_clean_data_for_date()` 新增 `daily_basic`、`moneyflow` 的 clean 层自动构建
  - `_load_factor_data()` 缺失因子数据时自动按日增量下载并追加保存：
    - `fina_indicator`：通过 `fina_indicator_vip` API 按 `ann_date` 获取当日全市场公告
    - `stk_holdernumber`：按 `ann_date` 获取当日全市场数据
    - `forecast`：通过 `forecast_vip` API 按 `ann_date` 获取当日全市场预告
    - `hot_rank`：通过 AKShare `stock_hot_rank_em()` 获取当日快照
  - `TushareClient` 新增 3 个按日查询方法：`get_fina_indicator_by_date()`、`get_forecast_by_date()`、`get_stk_holdernumber_by_date()`
  - 实现完整的 ensure 链路：`raw → clean → features` + 因子自动增量，纸面交易零手动干预

### 修复

- **features 缓存缺失融资融券列导致推理失败**
  - `ensure_raw_data_for_date()` 取消 `raw/daily` 存在即跳过的早期返回：`daily_basic` 和 `margin_detail` 现在独立检查，即使日线数据已存在也会补齐
  - `_load_factor_data()` 融资融券段改为总是先补齐历史分区再加载，确保 20+ 天历史数据供滚动变化率计算
  - `ensure_features_for_date()` 新增 Parquet schema 校验：通过 `pyarrow.read_schema()` 检查缓存文件是否包含必要因子列（`rzye_chg_5`、`rzye_chg_20`、`rqye_rzye_ratio`），缺失则自动触发重建

### 删除

- **移除业绩快报（express）支持**
  - 删除 `DataLoader.load_express()` 方法
  - 删除 `earnings.py` 中 express 相关逻辑（`express_profit_yoy`、`express_revenue_yoy` 列）
  - `build_earnings_lookup_by_date()` 签名简化：移除 `express_df` 参数
  - 清理 `download_raw.py`、`build_clean_features.py`、`features/ensure.py` 中的 express 引用

## [0.16.2] - 2026-03-13

### 重构

- **删除 `_download_data()` 冗余方法**（`src/lazybull/paper/runner.py`）
  - T0 工作流中数据下载已由 `ensure_features_for_date()` 内部的 ensure 链自动完成（raw → clean → features）
  - 移除约 100 行重复代码，ensure 链还额外下载 moneyflow 数据，覆盖更完整
- **`load_industry_mapping()` 参数重命名**（`src/lazybull/portfolio/industry_constraint.py`）
  - 参数 `stock_basic` → `shenwan_industry`，文档描述同步更新，消除命名歧义
- **`engine.py` 变量重命名**：`shenwan_stock_basic` → `shenwan_industry`

## [0.16.1] - 2026-03-13

### 修复

- **纸面交易 T0 特征构建缺少因子数据**（`src/lazybull/features/ensure.py`）
  - `ensure_features_for_date()` 此前未加载基本面和另类数据因子（fina_indicator、margin_detail、stk_holdernumber、forecast/express、hot_rank），导致纸面交易生成的 features 缺少这些列，模型预测质量下降
  - 新增 `_load_factor_data()` 函数：自动加载已下载的因子 raw 数据，构建 lookup 表并传入 builder
  - 因子缺失时输出 WARNING 级别汇总日志（覆盖率 N/5 + 缺失列表 + 下载命令提示），避免静默跳过
  - `ensure_features_for_date` 返回值扩展为 `Tuple[bool, List[str]]`，携带缺失因子列表
  - `PaperTradingRunner` 新增 `missing_factors` 属性，供上层读取
  - `bot_service.py` 钉钉交易结果消息中展示因子缺失警告（覆盖率 + 缺失列表）

## [0.16.0] - 2026-03-13

### 新增功能

- **市场择时扩展为 4 种模式**（`src/lazybull/backtest/engine_ml.py`、`scripts/walk_forward.py`）
  - `binary`：原有二值模式（mkt_ret_avg_20 < threshold → 降仓）
  - `vol_target`：波动率目标模式，exposure = target_vol / realized_vol，波动越大仓位越低
  - `trend`：趋势叠加模式，基于 mkt_ma_trend（MA20/MA60）线性降仓
  - `combined`：vol_target + trend 组合模式，双重保护
  - 新增回撤保护（`--market-regime-drawdown-guard`）：已大幅下跌时停止降仓，避免底部踏空反弹
  - 新增趋势保护（`--market-regime-trend-guard`）：上行趋势时跳过 vol_target 强制满仓
  - 新增仓位变动日志：每次择时变动输出 WARNING 级别日志
  - CLI 新增参数：`--market-regime-mode`、`--market-regime-vol-target`、`--market-regime-trend-threshold`、`--market-regime-min-exposure`、`--market-regime-combine-method`、`--market-regime-drawdown-threshold`

- **新增 `mkt_ret_vol_20` 市场特征**（`src/lazybull/factors/market_state.py`）
  - 近 20 日全市场日均收益的时间序列波动率（rolling std）
  - 用于 vol_target 模式的年化波动率计算

### 变更

- **移除业绩快报因子**（`src/lazybull/ml/train_core.py`）
  - 删除 `express_profit_yoy`（业绩快报净利润同比）、`express_revenue_yoy`（业绩快报营收同比）
  - 另类数据因子从 8 个缩减为 6 个
- **移除 tensorflow 依赖**（`pyproject.toml`）
- **删除 `scripts/build_features.py`**：已由 `build_clean_features.py` 完全替代
- **OOS 回测初始资金 50万→100万**（`scripts/walk_forward.py`）
- **`batch_walk_forward.ps1` 参数精简**：默认超参从网格搜索精简为单值，新增择时模式参数支持

### 版本变更

- 版本号：`0.15.1` → `0.16.0`

## [0.13.5] - 2026-02-23

### Bug 修复

- **修复 `--start-date` 变化导致同一 `trade_date` 特征值不稳定**
  - 新增 `_get_lookback_dates(trade_date, n, trading_dates)` 私有方法
    （`src/lazybull/features/builder.py`）：
    以 `trade_date` 在全量 `trading_dates` 中的位置为锚点，向前回溯恰好 `n` 个交易日，
    确保窗口日期集合只由全量 `trade_cal` 决定。
  - 修改 `_calculate_features()`：替换旧的 `current_idx - window` 切片 + 区间筛选
    为调用 `_get_lookback_dates`，消除 `trading_dates` 被截断时的窗口错位。
  - 修改 `_add_moneyflow_features()`：资金流 rolling 窗口同步使用 `_get_lookback_dates`。
  - 修改 `_get_tech_factor_today()`：新增 `trading_dates` 参数；预计算时先将 `daily_adj`
    过滤到全量交易日历日期集合，消除 `daily_adj` 起始日期不同导致的 EWM/滚动指标差异。
  - 历史不足（回溯不够 N 天）时返回空列表，对应特征置 NaN，不报错。

### 测试

- 新增 `test_get_lookback_dates_basic`：验证 `_get_lookback_dates` 基础行为与边界情况。
- 新增 `test_window_features_stable_across_start_dates`：
  相同全量 `trade_cal`、不同 `daily_data` 起始截断，同一 `trade_date` 的
  `ret_N`、`vol_ratio_N`、`ma_deviation_N` 应完全一致（精度 < 1e-9）。
- 新增 `test_window_features_nan_when_insufficient_history`：
  历史不足时，窗口特征应全部为 NaN。

## [0.13.4] - 2026-02-22

### Bug 修复

- **修复 `label_transform=cs_zscore` 场景下的数据泄露/评估口径问题**
  - 新增共用切分函数 `split_train_val_by_date(df, val_ratio, date_col)`（`src/lazybull/ml/train_core.py`）：
    以唯一交易日列表为单位切分，最后 `ceil(n_dates * val_ratio)` 个日期作为验证集，
    确保同一交易日的所有样本不会被拆分到不同集合，彻底避免截面统计量跨集合污染。
  - 修改 `prepare_training_data()` 使用 `split_train_val_by_date` 替换旧的按行数 `iloc` 切分；
    新增可选参数 `label_transform_fn`，若提供则在切分后对 train/val 各自独立调用，不共享统计量。
  - 修改 `scripts/train_ml_model.py`：`label_transform=cs_zscore` + `task=regression` 时，
    不再对全量 df 预先变换，改为通过 `label_transform_fn` 在切分后各自独立变换。
  - 修改 `scripts/walk_forward.py`：每个 split 内部同理，切分后分别对内部 train/val 独立变换。
  - `src/lazybull/ml/__init__.py` 新增 `split_train_val_by_date` 导出。

### 测试

- 新增 `tests/test_train_core_date_split.py`（12 个用例）：
  - `TestSplitTrainValByDate`（8 个）：
    - `test_no_same_day_split`：同日样本不被拆分
    - `test_all_samples_preserved`：切分后样本总数不变
    - `test_val_ratio_approximately_correct`：验证集日期数量正确
    - `test_val_dates_are_later_than_train_dates`：验证集日期晚于训练集
    - `test_stats_keys_present`：返回统计字段完整
    - `test_empty_dataframe`：空 DataFrame 不抛异常
    - `test_single_date`：单日情况处理正确
    - `test_custom_date_col`：支持自定义日期列名
  - `TestCsZscoreNoLeakage`（4 个）：
    - `test_old_logic_causes_leakage_on_boundary_date`：旧逻辑确实存在泄露（对照）
    - `test_new_logic_no_cross_set_statistics`：新逻辑各自独立统计量
    - `test_transform_labels_cs_zscore_per_date`：每日截面独立标准化
    - `test_split_train_val_date_sets_disjoint_with_transform`：切分后日期不交叉

### 文档

- 新增 `docs/PR/fix_label_cs_zscore_leakage.md`：详述问题根因、修复方案与影响
- 新增 `docs/guide/label_transform_cs_zscore.md`：cs_zscore 训练/验证口径指南

## [0.13.3] - 2026-02-21

### Bug 修复

- **修复 `volatility_20` / `zscore_volatility_20` / `spec_score` 数值不一致**
  - 新增共用函数 `compute_ret_1(daily_adj)`（`src/lazybull/factors/returns.py`）：
    统一 `ret_1` 构造优先级：
    1. 已有 `ret_1` 列 → 直接返回；
    2. 有 `close_adj` → 按 `ts_code` 分组 `pct_change()`（复权口径，无前瞻）；
    3. 有 `pct_chg` → `pct_chg / 100`（fallback，记录 WARNING）；
    4. 均无 → 全 NaN 并记录 WARNING。
  - 修改 `precompute_technical_factors.py` 步骤 6：调用 `compute_ret_1` 替换
    旧的直接使用 `pct_chg/100` 的逻辑，消除与复权收益率的口径偏差。
  - `zscore_volatility_20` 和 `spec_score` 由 `volatility_20` 派生，随之自动修复。
  - 性能优化（批量预计算 + 实例级缓存）**不回退**，`calculate_volatility` 公式不变。
  - `factors/__init__.py` 新增 `compute_ret_1` 导出。

### 测试

- 新增测试类 `TestComputeRet1`（6 个用例）和 `TestVolatilityRet1Consistency`（3 个用例）：
  - `test_priority1_uses_existing_ret_1`：已有 `ret_1` 时直接返回
  - `test_priority2_uses_close_adj_pct_change`：`close_adj` 路径结果正确
  - `test_priority2_no_cross_stock_leakage`：无跨股票边界差分
  - `test_priority3_fallback_pct_chg_with_warning`：fallback 行为可控且有 warning
  - `test_priority4_all_nan_with_warning`：全缺失返回 NaN 并有 warning
  - `test_result_aligned_to_original_index`：结果索引对齐
  - `test_volatility_20_consistent_with_close_adj_pct_change`：端到端一致性
  - `test_volatility_differs_from_pct_chg_path`：确认修复改变旧口径
  - `test_zscore_volatility_20_stable_with_close_adj`：zscore 一致性

## [0.13.2] - 2026-02-21

### 性能优化

- **技术指标与波动率批量预计算 + 实例级缓存（方案 A + A）**
  - 新增 `precompute_technical_factors(daily_adj, vol_windows)` 函数
    （`src/lazybull/factors/precompute_technical_factors.py`）：
    对全量 `daily_adj` 一次性调用现有
    `calculate_rsi / calculate_kdj / calculate_macd / calculate_bollinger_bands / calculate_volatility`，
    输出宽表（`ts_code, trade_date` + 15 个因子列），仅做一次排序与必要列裁剪。
  - `FeatureBuilder` 新增 `_tech_factor_cache` 实例级字段（内存缓存，不落盘）：
    首次调用 `_get_tech_factor_today()` 时触发批量预计算并缓存；后续每日
    仅按 `trade_date` 过滤（`O(1)` 查表），彻底消除批量构建时逐日切片
    50 天历史、重复计算 RSI/KDJ/MACD/BB/波动率 的瓶颈。
  - `FeatureBuilder._add_advanced_factors()` 技术指标与波动率分支改为调用
    `_get_tech_factor_today()`，不再按日 `hist_dates` 切片。
  - `factors/__init__.py` 新增 `precompute_technical_factors` 导出。
  - 复用现有各指标计算函数，**不改公式与实现细节**，输出口径完全不变。
  - 缓存生命周期为 `FeatureBuilder` 实例级，兼容
    `build_features.py` 和 `build_clean_features.py` 两条构建链路。

### 测试

- 新增 `tests/test_technical_indicators_precompute.py`（11 个测试用例）：
  - `TestPrecomputeTechnicalFactors`
    - `test_rsi_parity`：RSI(14) 与旧逻辑完全一致（< 1e-6）
    - `test_kdj_parity`：KDJ K/D/J 与旧逻辑完全一致
    - `test_macd_parity`：MACD DIF/DEA/HIST 与旧逻辑完全一致
    - `test_bollinger_bands_parity`：布林带中/上/下轨与旧逻辑完全一致
    - `test_volatility_parity`：volatility_5/10/20 与旧逻辑完全一致
    - `test_output_contains_required_columns`：输出包含全部预期列
    - `test_empty_input_returns_empty_df`：空输入不抛异常
  - `TestTechFactorCache`
    - `test_precompute_called_only_once`：多日构建时预计算只触发 1 次（monkeypatch）
    - `test_cache_returns_correct_date`：缓存查表返回正确日期的数据
    - `test_cache_is_instance_scoped`：不同实例缓存相互独立
    - `test_new_builder_instance_cache_is_none`：新建实例缓存初始为 None

### 文档

- 新增 `docs/PR/optimize_technical_indicators_performance.md`：
  性能瓶颈说明、优化方案 A+A、口径不变证明、对两条构建链路的影响

### 版本变更

- 版本号：`0.13.1` → `0.13.2`（patch bump，纯性能优化，不改口径）

## [0.13.1] - 2026-02-21

### 性能优化

- **市场状态特征批量预计算 + 实例级缓存**
  - 新增 `precompute_market_state_features(daily_data, trading_dates, daily_basic_data)` 函数
    （`src/lazybull/factors/market_state.py`）：对全量数据一次性 groupby + pandas rolling，
    将批量构建时逐日重复的 O(N×60) 计算降低为 O(N) 预计算 + O(1) 取值。
  - `FeatureBuilder._add_market_state_features()` 首次调用时自动触发批量预计算并存入
    `self._market_state_cache`，后续每日直接按索引取一行，兼容
    `build_features.py` 和 `build_clean_features.py` 两条构建链路。
  - 保留原 `compute_market_state_features()` 作为单日/回退入口，输出口径 **完全不变**
    （6 个字段数值精度 < 1e-9）。
  - `factors/__init__.py` 新增 `precompute_market_state_features` 导出。

### 测试

- 更新 `tests/test_market_and_new_features.py`（新增 6 个测试用例）：
  - `TestPrecomputeMarketStateFeatures`
    - `test_output_shape`：验证输出行数与 trading_dates 一致
    - `test_parity_with_single_day_no_basic`：无 daily_basic 时与逐日结果精确一致
    - `test_parity_with_single_day_with_basic`：含 daily_basic 时与逐日结果精确一致
    - `test_rolling_min_periods_1`：数据不足窗口时 min_periods=1 行为验证
    - `test_empty_data_returns_nan`：空数据不抛异常，返回全 NaN
    - `test_no_duplicate_compute_with_cache`：多次调用只触发一次批量预计算

### 文档

- 新增 `docs/PR/optimize_market_state_features.md`：性能问题说明、优化方案、影响范围
- 新增 `docs/guide/market_state_features.md`：如何验证输出一致性

### 版本变更

- 版本号：`0.13.0` → `0.13.1`（patch bump，纯性能优化，不改口径）

## [0.13.0] - 2026-02-21

### 新增功能

- **申万行业升级为三级（L3）**
  - `scripts/update_basic_data.py --only-shenwan` 下载口径升级到 `index_classify(level='L3', src='SW2021')`
    并通过 `index_member_all(l3_code=...)` 获取含 L1/L2/L3 完整层级信息的成分股
  - 行业映射表保存为单张表（`shenwan_industry.parquet`），字段包含：
    `ts_code`、`sw_l1_code`、`sw_l1`、`sw_l2_code`、`sw_l2`、`sw_l3_code`、`sw_l3`、`in_date`
  - `DataCleaner.clean_shenwan_industry()` 默认 `level_str='l3'`，新增 `_clean_shenwan_industry_l3()` 实现
    （旧式 `level_str='l2'/'l1'` 向后兼容，通过 `_clean_shenwan_industry_legacy()` 处理）
  - `FeatureBuilder._merge_shenwan_industry()` 自动检测 L3 格式（`sw_l3_code` 字段），产出：
    - `sw_industry` / `sw_industry_code` / `sw_industry_id`（映射到 L3）
    - `sw_l2` / `sw_l2_code` / `sw_l2_id`（L2 辅助字段）
    - `sw_l1` / `sw_l1_code` / `sw_l1_id`（L1 辅助字段）

- **L3→L2→L1→全市场 分层回退中性化**
  - 新增可复用模块 `src/lazybull/factors/hierarchical_industry_neutralization.py`，导出：
    - `hierarchical_zscore()`：指标的行业内 Z-Score（`zscore_` 前缀），支持 L3→L2→L1→全市场回退
    - `hierarchical_demean()`：收益率/标签的行业内去均值（`neu_` 前缀），支持同样的回退链路
  - 回退触发条件：L3 行业内 `tradable==1` 样本数 < `min_group_size(=5)` → 回退到 L2；
    L2 不足 → L1；L1 不足 → 全市场
  - `FeatureBuilder._apply_industry_neutralization()` 自动检测 L3 层级信息（`sw_industry_code`、
    `sw_l2_code`、`sw_l1_code`），存在时启用分层回退路径，否则退化为单层中性化

### 测试

- 更新 `tests/test_sw_industry_l2.py`（11 个测试用例）：
  - `TestMergeShenwanIndustry` 使用新式 L3 格式 fixture，验证 L2/L1 辅助字段输出
  - `TestCleanShenwanIndustryL3` 替换原 `TestCleanShenwanIndustryL2`，验证 L3 默认行为
- 新增 `tests/test_sw_industry_l3.py`（13 个测试用例）：
  - `TestHierarchicalZscore`：L3/L2/L1/全市场各层回退路径验证（数值精确断言）
  - `TestHierarchicalDemean`：分层去均值精确断言
  - `TestFeatureBuilderHierarchicalNeutralization`：集成测试（L3 路径/单层回退路径）

### 文档

- 更新 `docs/features_schema.md`：
  - 新增第 14 节「行业分层信息字段」（L1/L2/L3 字段命名说明）
  - 新增第 15 节「分层回退中性化规则」
- 新增 `docs/PR/shenwan_l3_upgrade.md`：申万三级行业升级 PR 说明文档
- 新增 `docs/guide/hierarchical_neutralization_guide.md`：分层回退验证指南

### 版本变更

- 版本号：`0.12.1` → `0.13.0`（minor bump）
- 重建 features 及重训模型命令：
  ```bash
  # 重新下载申万三级行业数据
  python scripts/update_basic_data.py --only-shenwan --force
  # 重建特征（以单日为例）
  python scripts/build_features.py --start-date 20240101 --end-date 20241231
  # 重训模型
  python scripts/train_ml_model.py
  ```

### Breaking Changes

- `DataCleaner.clean_shenwan_industry()` 默认 `level_str` 从 `'l2'` 改为 `'l3'`，
  产出字段从 `{ts_code, sw_code, sw_name, in_date}` 变更为
  `{ts_code, sw_l1_code, sw_l1, sw_l2_code, sw_l2, sw_l3_code, sw_l3, in_date}`
- 历史 features 文件中行业字段无变化（`sw_industry` 仍为主字段），但新增 `sw_l2*` / `sw_l1*` 列
- 若已有 `shenwan_industry.parquet` 且字段为旧式 L2 格式，需执行
  `python scripts/update_basic_data.py --only-shenwan --force` 重新下载



### 新增功能

- **新增个股特征**（在 `FeatureBuilder` 中生成）：
  - `is_new_stock`：上市不足 365 自然日则为 1，否则为 0（依赖 `list_days`）
  - `size`：流通市值（= `circ_mv`，来自 daily_basic）
  - `zscore_size`：对 `log1p(size)` 进行行业内 Z-Score（`sw_industry` 分组，tradable==1，min_group_size=5 回退）
  - `spec_score`：`zscore_volatility_20 × (−zscore_size)`（需 `apply_industry_neutralization=True`）

- **新增市场状态特征**（每日一个标量，广播至当日所有股票）：
  - `mkt_vol_cnt`：全市场收益率截面标准差（tradable==1）
  - `mkt_vol_20`：`mkt_vol_cnt` 过去 20 日滚动均值
  - `mkt_turnover_ratio`：市场拥挤度 `sum(amount)/sum(circ_mv)`（tradable==1）
  - `mkt_ret_avg_20`：过去 20 日全市场平均收益率之和
  - `mkt_turnover_std`：全市场换手率截面标准差（优先 `turnover_rate_f`）
  - `mkt_adv_dec_ratio`：过去 60 日涨跌家数比滚动均值

- **新增模块** `src/lazybull/factors/market_state.py`：可复用的市场状态特征计算模块

### 测试

- 新增 `tests/test_market_and_new_features.py`（17 个测试用例）：
  - `is_new_stock` 边界验证（刚好 365 天）
  - `zscore_size` 大样本行业 vs 小样本回退全市场
  - `spec_score` 公式验证与缺失依赖时 NaN 传播
  - 市场状态单日截面特征正确性
  - 市场状态滚动特征（窗口不足时 min_periods=1 行为）

### 文档

- 更新 `docs/features_schema.md`：
  - 新增第 12 节（新增个股特征）和第 13 节（市场状态特征）
  - 统一 zscore 列命名为 `zscore_` 前缀
- 新增 `docs/PR/market_and_stock_features.md`：PR 说明（新特征列表、依赖补齐方式、重建命令）
- 新增 `docs/guide/market_state_verification.md`：市场状态特征验证示例代码

## [0.12.0] - 2026-02-20

### 新增功能

- **申万行业从一级切换到二级**
  - `DataCleaner.clean_shenwan_industry()` 默认 `level_str='l2'`（原 `'l1'`）
  - `FeatureBuilder._merge_shenwan_industry()` 输出字段重命名：
    - `sw_name` -> `sw_industry`（申万二级行业名称）
    - `sw_code` -> `sw_industry_code`（申万二级行业指数代码）
    - `industry_id` -> `sw_industry_id`（稳定整数编码）
  - 中性化分组字段由 `sw_name` 更新为 `sw_industry`
  - 行业 alpha 计算使用 `sw_industry` 列（若存在）

- **训练 rank-weight：Top/Bottom K 样本权重增强**
  - 新增 `ml/train_core.py::build_rank_sample_weights()` 函数
    - 按 `trade_date` 分组，每日 Top K / Bottom K 样本权重 = `top_weight`
    - 退化处理：样本数 <= 2*topk 时全组赋予 top_weight
  - `train_xgboost_model()` 新增 `sample_weight` 参数，传给 XGBoost fit
  - `scripts/train_ml_model.py` 新增 CLI 参数：
    - `--rank-weight-enabled`（默认开启）/ `--no-rank-weight`
    - `--rank-weight-topk`（默认 30）
    - `--rank-weight-weight`（默认 5.0）
  - rank-weight 配置记录到 `ml_train_runs.csv`（`rank_weight_enabled` / `rank_weight_topk` / `rank_weight_weight` 列）

### 测试

- 新增 `tests/test_sw_industry_l2.py`（11 个测试用例）
  - 验证申万二级行业字段切换正确性
  - 验证 sw_industry_id 编码稳定性
- 新增 `tests/test_rank_sample_weight.py`（13 个测试用例）
  - 验证单日/多日 Top/Bottom K 权重逻辑
  - 验证多日分组独立不串
  - 验证 K 大于样本数时的退化处理

### 文档

- 新增 `docs/PR/sw_industry_l2_and_rank_weight.md`：PR 说明文档
- 新增 `docs/guide/rank_weight_guide.md`：rank-weight 使用与验证指南
- 更新 `docs/features_schema.md`：行业字段说明（sw_industry* 统一命名）

### Breaking Changes

- 特征文件中行业字段重命名：`sw_name` -> `sw_industry`，`sw_code` -> `sw_industry_code`，`industry_id` -> `sw_industry_id`
- 若已有历史特征文件，需重新运行 `build_features.py` 以生效
- `DataCleaner.clean_shenwan_industry()` 默认级别从一级改为二级，若需一级行业需显式传 `level_str='l1'`



### 重大变更 (Breaking Changes)

本版本完善行业中性化特征工程，**训练默认标签变更**。需要重新构建 features 并重训模型。

### Added

- **完整行业中性化实现**
  - 新增 `industry_demean()` 函数 - 行业去均值中性化（`src/lazybull/factors/normalization.py`）
    - 适用于收益率/标签列：`y_ret_5/10/20`, `ret_5/10/20`
    - 命名规则：`neu_` 前缀（如 `neu_y_ret_20`）
    - 公式：`neu_x = x - mean(x within industry)`
    - 统计范围：仅 `tradable==1`，小样本（<5）回退全市场均值
  - 完善 `_apply_industry_neutralization()` 方法 - 整合两类中性化
    - 去均值：收益率/标签列 → `neu_` 前缀
    - Z-Score：指标/特征列 → `_zscore` 后缀
    - 从 Z-Score 白名单移除 `ret_20`（用户明确只要去均值版本）
  
- **申万行业分类字段**（已在 v0.10.0 实现，本版本完善集成）
  - `sw_code`: 申万一级行业代码
  - `sw_name`: 申万一级行业名称（用于中性化分组）
  - `sw_l1_id` / `industry_id`: 整数编码（稳定映射）

- **新增中性化特征列**
  - 去均值列：`neu_y_ret_5/10/20`, `neu_ret_5/10/20`
  - Z-Score列：`pe_ttm_zscore`, `pb_zscore`, `bp_zscore`, `dv_ttm_zscore`, 
    `log_total_mv_zscore`, `amount_ma20_zscore`, `turnover_rate_zscore`,
    `volatility_5/10/20_zscore`, `net_mf_amount_zscore`, `ma_deviation_20_zscore`

### Changed

- **训练默认标签变更**
  - 旧默认：`y_ret_5`（未中性化的5日收益）
  - 新默认：`neu_y_ret_20`（行业中性化后的20日收益）
  - 更新 `scripts/train_ml_model.py` 默认参数和帮助信息
  - 支持的标签选项扩展：`y_ret_5/10/20`, `neu_y_ret_5/10/20`

- **行业中性化白名单优化**
  - 从 Z-Score 白名单移除 `ret_20`（只保留 `neu_ret_20` 去均值版本）
  - 保持其他指标列的 Z-Score 中性化

### Testing

- 新增 `tests/test_industry_demean.py` - 包含8个测试用例
  - 验证行业去均值基本功能
  - 验证 tradable==1 过滤
  - 验证小样本回退全市场
  - 验证缺失列和行业列错误处理
  - 验证多列同时去均值
  - 验证命名约定（neu_ vs _zscore）

### Documentation

- 新增 `docs/PR/industry_neutralization_v0.11.0.md` - 本版本完整说明
  - 申万行业分类接入方法
  - 两类中性化对比（去均值 vs Z-Score）
  - 训练默认标签变更说明
  - 重建 features 与重训模型命令
  - 扩展与验证指南
  - 常见问题排查
  
- 新增 `docs/guide/industry_neutralization_extension_guide.md` - 扩展指南
  - 如何扩展中性化白名单
  - 如何验证中性化效果
  - IC 分析方法
  
- 更新 `docs/features_schema.md`
  - 新增申万行业分类字段说明
  - 新增行业中性化字段说明（去均值 + Z-Score）
  - 详细说明两类中性化的区别和用途

### Migration Guide

**从 v0.10.0 升级到 v0.11.0**：

1. 更新代码：`git pull`
2. 确保申万行业数据已下载（v0.10.0已支持）：
   ```bash
   python scripts/update_basic_data.py --only-shenwan --force
   ```
3. 重新构建特征（启用行业中性化）：
   ```bash
   python scripts/build_features.py \
       --start-date 20230101 --end-date 20231231 \
       --apply-industry-neutralization
   ```
4. 重新训练模型（自动使用新默认标签 `neu_y_ret_20`）：
   ```bash
   python scripts/train_ml_model.py \
       --start-date 20230101 --end-date 20231130
   ```

**不兼容说明**：
- 旧版本构建的特征文件不包含去均值列（`neu_*`），需重新构建
- 训练脚本默认标签已改变，旧脚本使用新代码会自动使用新默认标签

---

## [0.10.0] - 2026-02-18

### Added

- **申万行业分类数据接入**
  - TuShare `index_classify` + `index_member` 接口
  - 申万一级行业分类（SW2021版本，约30个行业）
  - 数据存储在 `data/raw/shenwan_industry.parquet`
  - 更新脚本：`scripts/update_basic_data.py --only-shenwan`

- **行业内 Z-Score 中性化**（初版实现）
  - `src/lazybull/factors/normalization.py` - 中性化模块
  - `industry_neutralization()` - 行业内 Z-Score 标准化
  - `cross_sectional_zscore()` - 截面 Z-Score 标准化
  - FeatureBuilder 集成：`apply_industry_neutralization=True`

- **数据清洗与加载**
  - `DataCleaner.clean_shenwan_industry()` - 清洗申万行业数据
  - `DataLoader.load_shenwan_industry()` - 加载申万行业数据
  - `TushareClient.get_index_classify()` - 获取指数分类
  - `TushareClient.get_index_member()` - 获取指数成分股

### Testing

- 新增 `tests/test_industry_neutralization.py`
  - 验证截面 Z-Score 基本功能
  - 验证行业内 Z-Score 中性化
  - 验证小样本回退全市场统计

### Documentation

- 更新 `docs/features_schema.md` - 说明行业特征字段
- 新增行业中性化相关文档

---

## [0.9.0] - 2026-02-18

### 重大变更 (Breaking Changes)

本版本引入特征工程重构，**不兼容旧版本特征数据**。需要重新生成 features 并重训模型。

### Added

- **因子库模块 (src/lazybull/factors/)**
  - 新增 `technical_indicators.py` - 技术指标因子
    - RSI(14): 相对强弱指标
    - KDJ(9,3,3): 随机指标
    - MACD(12,26,9): 指数平滑移动平均线
    - 布林带(20,2): 输出带宽 (bb_width) 和 %B (bb_pct)
  - 新增 `candlestick.py` - K线形态因子
    - 振幅 (amplitude): (high_adj - low_adj) / pre_close_adj
    - 上影线 (upper_shadow): (high_adj - body_high) / close_adj
    - 下影线 (lower_shadow): (body_low - low_adj) / close_adj
    - 实体长度 (body_length): |close_adj - open_adj| / close_adj
  - 新增 `volatility.py` - 波动率因子
    - volatility_5/10/20: 基于 ret_1 的滚动标准差
  - 新增 `industry.py` - 行业相关因子
    - industry_id: 行业整数编码（稳定映射）
    - alpha_industry: 个股收益 - 行业平均收益
    - alpha_industry_5/10/20: 多窗口行业 alpha
  - 新增 `momentum.py` - 动量加速度
    - acceleration: ret_5 - ret_10 (短期动量 - 中期动量)
  - 新增 `volume.py` - 量能突变
    - vol_burst_5/10/20: vol_ratio 的截面 zscore

- **FeatureBuilder 增强**
  - 新增 `_add_advanced_factors()` 方法，统一整合因子库
  - 扩展 `_calculate_adj_close()` 自动计算 open_adj/high_adj/low_adj
  - 行业数据从 stock_basic 的 industry 字段获取
  - 缺失 industry 字段时抛出清晰错误提示

### Changed

- **特征删除**
  - 删除 `amount_ratio_5/10/20` 特征（不再生成）
  - 删除 `vol_ma5/10/20` 特征（不再生成）
  - 保留 `amount_ma5/10/20` 特征

- **FeatureBuilder 重构**
  - 特征计算逻辑拆分到 factors 模块，提升可维护性
  - features/builder.py 主要负责数据流编排，因子计算委托给 factors 模块

### Testing

- 新增 `tests/test_new_features.py` 包含12个测试用例
  - 验证已删除特征不再出现
  - 验证新增特征存在且可计算
  - 验证行业字段缺失报错
  - 验证 industry_id 编码稳定性

### Documentation

- 新增 `docs/PR/feature_refactoring.md` - 本次PR详细说明
- 新增 `docs/guide/factor_extension.md` - 因子扩展开发指南
- 更新 `docs/features_schema.md` - 反映新增/删除字段

### Migration Guide

旧模型和旧特征数据**不兼容**本版本，需要：
1. 重新生成 features: `python scripts/ensure_features.py --start-date 20200101 --end-date 20231231`
2. 重新训练模型: `python scripts/train_ml_model.py ...`

---

## [0.8.4] - 2026-02-15

### Added

- **Walk-forward 滚动训练能力**
  - 新增 `src/lazybull/ml/train_core.py` 模块，抽取训练核心逻辑供复用
    - `load_features_data()` - 加载特征数据
    - `prepare_training_data()` - 准备训练数据
    - `transform_labels_cs_zscore()` - 标签变换
    - `generate_classification_labels()` - 分类标签生成
    - `train_xgboost_model()` - 训练模型
    - `evaluate_validation_daily()` - 逐日评估
  - 新增 `src/lazybull/ml/walk_forward_utils.py` 模块，提供 walk-forward 切分工具
    - `generate_walk_forward_splits()` - 生成训练/测试区间切分
    - `WalkForwardSplit` - 切分数据结构
    - `print_splits_summary()` - 打印切分汇总
    - 支持按季度/月度/半年度滚动（monthly/quarterly/semiannual）
    - 支持可配置的训练窗口（默认5年）和测试窗口（默认6个月）
    - 所有日期自动对齐到交易日
  - 新增 `scripts/walk_forward.py` 脚本，实现完整的 walk-forward 流程
    - 生成多个训练/测试切分
    - 对每个切分执行完整训练（复用 train_core 逻辑）
    - 为每个切分注册模型版本并记录到 ml_train_runs.csv
    - 生成 walk_forward_summary.csv 汇总文件
    - 支持所有训练参数透传（task、label、pos_topk/pos_quantile、XGBoost 超参数等）
  - 扩展 `TrainingRunRecord` 支持 walk-forward 字段
    - `wf_run_id` - walk-forward 运行 ID
    - `split_index` - 切分索引
    - `step_frequency` - 滚动频率
    - `test_start_date` / `test_end_date` - 测试区间日期
  - 完整复用现有能力：训练、评估、逐日评估诊断、模型注册、CSV 训练运行日志

### Changed

- 重构 `scripts/train_ml_model.py`，使用 `train_core` 模块的函数
- 移除 `train_ml_model.py` 中的重复代码，改为导入 `train_core` 模块

### Documentation

- 新增 `docs/PR/walk_forward_implementation.md` - Walk-forward 实现说明
  - 功能特性：核心功能、切分口径、实现细节
  - 使用方法：基础用法、自定义参数、透传训练参数
  - 输出文件说明：ml_train_runs.csv 新增字段、walk_forward_summary.csv 字段
  - 与 train_ml_model.py 的关系：复用能力、区别
- 新增 `docs/guide/walk_forward_guide.md` - Walk-forward 使用指南
  - Walk-forward 原理介绍
  - 配置参数详解
  - 使用示例（按月度/季度/半年度滚动、分类/回归任务、超参数调优）
  - 输出文件说明
  - 结果分析方法（可视化、统计分析）
  - 常见问题与最佳实践

### Tests

- 新增 `tests/test_walk_forward.py` - Walk-forward 完整测试套件（11个测试用例）
  - 测试 split 生成逻辑（季度/月度/半年度）
  - 测试边界条件（窗口过大、日期范围不足、无效频率）
  - 测试切分验证（日期推进、无重叠、索引连续）
  - 测试汇总 CSV 生成
  - 测试与 run_logger 集成（wf 字段写入、动态列扩展）

## [0.8.3] - 2026-02-14

### Added

- **训练运行日志CSV追加记录功能**
  - 新增 `src/lazybull/ml/run_logger.py` 模块，提供训练运行记录的结构化存储与CSV追加功能
  - `TrainingRunRecord` 数据类：记录每次训练的完整信息
    - 基本信息：时间戳、版本号、训练日期区间、标签、任务类型
    - 训练配置：label_transform、winsorize_p、分类任务参数（pos_quantile/pos_topk/scale_pos_weight及模式）
    - XGBoost超参数：n_estimators、max_depth、learning_rate、subsample、colsample_bytree、gamma、reg_alpha、reg_lambda、early_stopping_rounds、tree_method、random_state、n_jobs
    - 数据统计：交易日数、总样本数、过滤后样本数、训练集/验证集样本数、验证集日期范围
    - 训练结果：best_iteration
    - 评估指标：训练集/验证集的MSE、RMSE、R2、IC、RankIC、ACC、AUC、Precision、Recall
    - 逐日评估：RankIC均值/标准差/IR、TopK收益统计
    - 诊断统计：全市场收益、样本数分布、TopK提升和分位数
  - `write_training_run_to_csv()` 函数：支持追加模式写入CSV，自动创建文件和表头
  - `create_training_run_record_from_training_session()` 函数：从训练会话信息创建记录对象
  - **动态列扩展**：新增字段时自动扩展表头，旧行缺失字段留空（向前兼容）

- **训练脚本集成日志记录**
  - `scripts/train_ml_model.py` 新增 `--run-log-csv` 参数，支持自定义日志文件路径（默认 `data/ml_train_runs.csv`）
  - 修改 `load_features_data()` 返回交易日数量
  - 修改 `prepare_training_data()` 返回数据统计（samples_after_filter、val_start_date、val_end_date）
  - 修改 `train_xgboost_model()` 在 train_params 中记录 best_iteration
  - 训练完成后自动记录运行日志到CSV（失败不影响模型保存）

### Documentation

- 新增 `docs/PR/training_run_logging.md` - 本 PR 详细说明
  - 功能介绍：CSV日志结构、字段说明、使用方法
  - 示例命令：如何使用 --run-log-csv 参数
  - 分析建议：如何利用CSV进行模型对比与超参数调优

### Tests

- 新增 `tests/test_ml_run_logger.py` - 训练运行日志模块完整测试套件（9个测试用例）
  - 测试CSV创建和首次写入
  - 测试追加记录功能
  - 测试自定义路径
  - 测试列扩展兼容性
  - 测试回归和分类任务记录
  - 测试完整工作流

## [0.8.2] - 2026-02-13

### Added

- **纸面交易与回测适配新模型/新特征（moneyflow + daily_basic）**
  - `DataLoader.load_clean_moneyflow()` - 新增资金流向数据加载方法，支持日期范围分区加载
  - `FeatureBuilder` 现在强制依赖 moneyflow 数据，缺失时会明确报错并提示补齐步骤
  - `MLSignal` 适配 classification 模型：自动使用 `predict_proba` 获取正类概率作为分数
  - `ensure_features_for_date()` 增强错误提示：moneyflow 缺失时提供详细的补数据命令

- **逐日评估诊断增强（排查 TopK/RankIC 不一致风险）**
  - `eval_utils.py` 新增 `compute_diagnostic_statistics()` 函数：
    - 全市场收益逐日均值/标准差统计
    - TopK 相对全市场提升计算（TopK - UniverseMean）
    - 每日样本数分布（min/median/max）
    - TopK 收益分位数（25%/50%/75%）
  - `eval_utils.py` 新增 `print_diagnostic_report()` 函数：格式化输出诊断报告
  - `train_ml_model.py` 集成诊断输出到验证集逐日评估流程

### Fixed

- **消除 Pandas FutureWarning（groupby.apply）**
  - `feature_utils.py` - `cross_sectional_zscore()` 改用矢量化 `transform` 方法，避免 `groupby.apply` 触发 FutureWarning
  - `train_ml_model.py` - `generate_classification_labels()` 改用矢量化方式计算百分比阈值，避免 `groupby.apply`

- **特征列一致性检查增强**
  - `ensure_features_for_date()` 增加 moneyflow 数据日志输出，记录加载的条数
  - moneyflow 缺失时的报错信息更友好，包含推荐的补数据命令

### Documentation

- 新增 `docs/PR/fix_paper_trade_moneyflow_v0.8.2.md` - 本 PR 详细说明
  - 说明纸面交易/回测为何缺特征列、如何修复、如何补齐数据
  - 说明 moneyflow 强制依赖的行为与补数据命令
  - 说明逐日评估新增诊断项的意义
  - 说明 FutureWarning 修复点

## [0.8.1] - 2026-02-13

### Fixed

- **修复 cs_zscore 的"重复 winsorize"问题**
  - `feature_utils.py` - 修复 `cross_sectional_zscore` 函数在按组标准化时重复 winsorize 的 bug
  - `train_ml_model.py` - 当 `label_transform=cs_zscore` 时，训练阶段不再对标签进行 winsorize（避免重复处理）
  - 新增 `skip_label_winsorize` 参数控制训练阶段是否跳过标签 winsorize
  - 目标行为：`label_transform=cs_zscore` 时仅在 cs_zscore 步骤做 winsorize；`label_transform=raw` 时仍保留训练阶段 winsorize 并在日志中说明

- **修复 Pandas FutureWarning（groupby.apply）**
  - `feature_utils.py` - 优化 `cross_sectional_zscore` 的 groupby 逻辑，避免 FutureWarning
  - `train_ml_model.py` - 改用 `rank(method='first')` 矢量化方式生成 pos_topk 标签，替代 groupby.apply
  - pos_topk 标签生成规则明确：正类数量严格等于 topk（使用 rank(method='first') 打散并列）

### Added

- **Classification 训练增强**
  - 新增 `--scale-pos-weight` CLI 参数，支持用户指定或自动计算（neg/pos）正类权重
  - 自动计算时在日志打印详细信息（负类数、正类数、计算值）
  - 新增 `src/lazybull/ml/eval_utils.py` 模块：提供可复用的逐日评估函数
    - `compute_daily_rankic()` - 计算单日 RankIC（Spearman）
    - `compute_daily_topk_returns()` - 计算单日 TopK 平均收益
    - `evaluate_predictions_by_date()` - 对多日预测进行逐日评估
    - `summarize_daily_metrics()` - 汇总逐日指标（均值、标准差、IR）
  - 分类任务训练后增加**验证集逐日评估**（贴近交易场景）
    - 逐日 RankIC（Spearman）：按每个 `trade_date` 计算预测概率与真实收益的秩相关，输出均值/标准差/IR
    - 逐日 TopK 收益评估：按每个 `trade_date` 以预测概率排序，计算 TopK（K=30/100/300）对应原始真实收益的均值，输出跨日均值/标准差
  - 统一 RankIC 计算口径：训练脚本中的 RankIC 改为"逐日计算后取均值"（与回测 eval panel 一致）

- **回测与纸面交易脚本适配新模型**
  - `model_registry.py` - 新增 `strict_version_check` 参数（默认 True），严格检查模型元数据
    - 检查必需字段：`feature_columns`、`train_params`、`model_type`
    - 缺少字段时明确报错并提示重新训练
  - `model_registry.py` - 新增 `check_feature_consistency()` 方法，检查推理数据特征列一致性
    - 验证推理数据是否包含模型训练时使用的所有特征列
    - 缺失特征时抛出详细错误（列出前 20 个缺失列）
  - `ml_signal.py` - 集成旧模型拒绝和特征列一致性检查
    - `_load_model` 方法调用 `strict_version_check=True` 拒绝旧模型
    - `generate` 和 `generate_ranked` 方法在预测前调用特征列一致性检查
  - **不兼容声明**：本版本明确不兼容旧模型（v1~v5），需重新训练

### Documentation

- 新增 `docs/PR/fix_cs_zscore_classification_enhancements_v0.8.1.md` - 本 PR 详细说明
  - 修复点说明
  - Classification 增强功能说明
  - 不兼容旧模型的决定与迁移方式
- 新增 `docs/guide/classification_evaluation_guide.md` - 分类模型评估指标指南
  - 说明应重点关注的指标（逐日 RankIC、TopK 收益）
  - 不要过度解读 Accuracy/Recall
  - 与回测结果对比的最佳实践

### Version

- 版本号从 0.8.0 升级到 0.8.1

---

## [0.8.0] - 2026-02-12

### Added

- **新增资金流数据源（moneyflow）**：提升模型在"价值红利"方向的选股能力
  - Raw/Ensure 层：新增 `TushareClient.get_moneyflow()` 方法，支持从 TuShare 获取个股资金流向数据
  - 在 `ensure_raw_data_for_date()` 中新增 moneyflow 下载逻辑，设为强制依赖（缺失时报错提示）
  - 在 `download_raw.py` 脚本中集成 moneyflow 下载
  - Clean 层：新增 `DataCleaner.clean_moneyflow()` 清洗方法
  - 在 `build_clean_features.py` 脚本中集成 moneyflow 清洗流程
  - 更新 `docs/data_contract.md` 补充 moneyflow 数据契约（主键、字段说明）
  
- **新增价值红利和资金流特征**：丰富因子库，支持价值投资和资金流分析
  - 新增 `feature_utils.py` 工具模块：提供 winsorize、log1p、zscore、cross_sectional_zscore 等通用特征处理函数
  - FeatureBuilder 新增 `_add_value_dividend_features()` 方法：
    - 基础因子：pb, pe_ttm, ps_ttm, dv_ttm, total_mv, circ_mv, turnover_rate, volume_ratio
    - 派生因子：ep_ttm (1/pe_ttm)、bp (1/pb)、log_total_mv、log_circ_mv
    - 亏损标记：is_loss（pe_ttm 为负或 NaN）
    - 处理 pe_ttm/pb 缺失和为0的情况
  - FeatureBuilder 新增 `_add_moneyflow_features()` 方法：
    - 当日净流入：net_mf_amount
    - 大单/特大单净流入：lg_net_amount、elg_net_amount
    - Rolling 特征（窗口 5/20）：net_mf_amount_sum/mean、lg_net_amount_sum、elg_net_amount_sum
    - 对重尾列自动应用 winsorize 处理
  - 更新 `build_features.py` 和 `build_clean_features.py` 加载并传递 daily_basic 和 moneyflow 数据

- **训练标签变换：cs_zscore（截面标准化）**：更稳定的回归标签，减少极端值影响
  - 新增 `transform_labels_cs_zscore()` 函数：对每个 trade_date 的标签进行截面 winsorize + zscore 变换
  - 变换后每个交易日标签均值≈0，标准差≈1
  - 新增 CLI 参数：`--label-transform {raw,cs_zscore}`（默认 raw）
  - 新增 CLI 参数：`--winsorize-p FLOAT`（默认 0.01，截断上下1%极端值）
  - 在模型元数据（model_registry.json）中记录 label_transform 和 winsorize_p

- **新增训练任务：classification（Top 分位分类）**：更贴近 TopN 选股的实际交易场景
  - 新增 `generate_classification_labels()` 函数：按每个交易日截面将标签转为 0/1 二分类标签
  - 支持两种模式（二选一，pos_topk 优先级更高）：
    - 百分比模式：`--pos-quantile FLOAT`（例如 0.2 表示 Top20% 为正类）
    - 数量模式：`--pos-topk INT`（例如 300 表示每日收益最高的 300 只为正类）
  - 新增 CLI 参数：`--task {regression,classification}`（默认 regression）
  - 新增 CLI 参数：`--pos-quantile FLOAT` 和 `--pos-topk INT`
  - 支持 XGBoost 分类器训练，目标函数自动切换为 `binary:logistic`
  - 分类任务评估指标：Accuracy、AUC、Precision、Recall
  - 在模型元数据中记录 task、pos_quantile、pos_topk
  - 模型类型标记为 `xgboost_classification` 以区分回归模型

### Changed

- **train_xgboost_model 函数增强**：统一支持回归和分类任务
  - 新增 `task` 参数，根据任务类型选择 XGBRegressor 或 XGBClassifier
  - 回归任务：保留 winsorize 处理和 IC/RankIC 评估
  - 分类任务：跳过 winsorize，使用 AUC/Precision/Recall 评估
  - 早停机制对两种任务均生效

### Documentation

- 更新 `docs/data_contract.md`：补充 moneyflow（资金流向）数据源的字段说明
- 新增 feature_utils.py 模块文档字符串：详细说明各工具函数的用法和示例

### Version

- 版本号从 0.7.0 升级到 0.8.0

---

## [0.7.0] - 2026-02-12

### Fixed
- **修复 `weight_method=score` 未生效问题**：修复了在回测引擎中 `score` 权重方法被等权覆盖的bug
  - 问题原因：`BacktestEngine._generate_signal()` 在信号生成阶段强制重新归一化权重，导致 MLSignal 已计算的按分数加权结果被覆盖
  - 修复方案：正确处理 `weight_method` 属性，当使用 `score` 时按预测分数归一化权重，而不是强制等权
  - 新增日志：明确显示当前使用的权重方法和前几只股票的权重示例，便于验证权重方法是否生效

### Added
- **权重后处理功能（限权/归一化）**：新增可复用的权重约束管理模块
  - 新增 `src/lazybull/portfolio/weight_processor.py` 模块
  - 实现 `cap_and_normalize_weights()` 函数：对权重进行限制并重新归一化
    - 支持设置单个股票最大权重 `max_weight_per_stock`（0-1之间）
    - 迭代式限权确保最终所有权重都不超过上限
    - 自动处理边界情况：空权重、全0、NaN、负数（过滤 <= 0 的权重）
  - BacktestEngine 新增 `max_weight_per_stock` 参数
  - CLI 新增 `--max-weight-per-stock` 参数（示例：`0.2` 表示单票最大 20%）
  - 单元测试：26 个测试全部通过，覆盖各种边界情况

- **行业持仓数量约束**：新增基于行业的持仓数量约束功能
  - 新增 `src/lazybull/portfolio/industry_constraint.py` 模块
  - 实现 `load_industry_mapping()` 函数：从 `stock_basic` 数据加载行业映射
    - 自动将行业缺失的股票归为"未知行业"
  - 实现 `apply_industry_constraint()` 函数：应用行业数量约束
    - 按分数排序选股，跳过已达到行业上限的股票并顺延
    - "未知行业"同样受约束限制
  - BacktestEngine 新增 `max_per_industry` 和 `stock_basic` 参数
  - CLI 新增 `--max-per-industry` 参数（示例：`3` 表示每个行业最多 3 只）
  - 单元测试：14 个测试全部通过，覆盖各种场景

### Documentation
- 新增 `docs/PR/portfolio_construction_enhancements.md`：详细说明三项改进的背景、实现和使用方法
- 新增测试文件：
  - `tests/test_weight_processor.py`：权重后处理模块测试（12个测试）
  - `tests/test_industry_constraint.py`：行业约束模块测试（14个测试）
- 扩展 `tests/test_ml_signal.py`：验证 score 权重方法产生非等权结果

### Technical Details
- 权重限权采用迭代算法：限权 → 归一化 → 检查收敛 → 重复（最多100次）
  - 确保最终所有权重都不超过设定上限
  - 处理多只股票同时被限权的情况
- 行业约束在信号生成阶段应用，在选择候选股票之前进行过滤
- 权重限权在权重归一化之后应用，确保最终权重满足约束

## [0.6.0] - 2026-02-12

### Added
- **统一评估面板（CSV输出）**：在回测运行时按日评估 MLSignal 的截面排序质量
  - 新增 `--export-eval` 参数：是否导出评估面板 CSV（默认开启）
  - 新增 `--eval-groups` 参数：分组数量（默认 10）
  - 新增 `--eval-topk` 参数：TopK 指标的 K（默认使用 --top-n）
  - 输出三个 CSV 文件：
    - `{output_name}_eval_daily.csv`：日度评估指标（RankIC、TopK收益、多空收益等）
    - `{output_name}_eval_groups.csv`：分组收益明细（每日每组的平均真实收益）
    - `{output_name}_eval_summary.csv`：汇总指标（参数配置和聚合统计）
  - 评估口径：
    - 真实收益标签直接使用 features 文件中的 label 列（y_ret_5/y_ret_10/y_ret_20）
    - 分组方式：按预测分数排序后等数量分组（默认 10 组）
    - RankIC 使用 Spearman 相关系数
  - CSV 统一使用 utf-8-sig 编码（Excel 兼容）

### Documentation
- 新增 `docs/PR/unified_eval_panel_csv_output.md`：详细说明评估面板功能的背景、实现和使用方法
- 新增 `docs/guide/ml_eval_panel_guide.md`：评估面板使用指南

## [0.5.0] - 2026-02-11

### Added
- **多 horizon 标签支持**：特征构建同时生成 `y_ret_5`, `y_ret_10`, `y_ret_20` 三个标签
  - 标签定义：未来 N 个交易日的后复权收益率，公式：`(close_adj(t+N) / close_adj(t)) - 1`
  - `FeatureBuilder` 新增 `horizons` 参数（默认 `[5, 10, 20]`），同时生成多个预测窗口的标签
  - `scripts/build_features.py` 新增 `--horizons` CLI 参数，支持自定义预测窗口列表

- **训练脚本支持选择标签**：`scripts/train_ml_model.py` 支持选择不同 horizon 的标签进行训练
  - 新增 `--label` CLI 参数（可选 `y_ret_5|y_ret_10|y_ret_20`，默认 `y_ret_5`）
  - 训练元数据自动记录所用标签到 `model_registry.json` 的 `label_column` 字段

- **回测脚本支持标签选择与自动调仓频率**：`scripts/run_ml_backtest.py` 增强标签和调仓频率管理
  - 新增 `--label` CLI 参数（可选 `y_ret_5|y_ret_10|y_ret_20`）
  - 当未显式指定 `--rebalance-freq` 时，根据标签自动设置默认值：
    - `y_ret_5` → 调仓频率 5 个交易日
    - `y_ret_10` → 调仓频率 10 个交易日
    - `y_ret_20` → 调仓频率 20 个交易日
  - 若同时指定 `--model-version` 和 `--label`，自动校验模型元数据中的标签一致性，不一致时给出清晰的中文报错提示

### Changed
- **`FeatureBuilder` 向后兼容**：保留 `horizon` 参数（已废弃），新参数 `horizons` 优先级更高
- **过滤逻辑优化**：`_apply_filters` 方法改为要求至少一个标签非空（而非所有标签都非空），更加灵活

### Documentation
- 新增 `docs/PR/multi_horizon_labels_and_selectable_training.md`：详细说明本次功能的背景、实现方案和使用方法
- 新增 `docs/guide/ml_label_horizon_guide.md`：完整的使用指南，包含标签定义、特征构建、训练和回测的详细说明

### Version
- 版本号从 0.4.2 升级到 0.5.0

## [0.4.2] - 2026-02-10

### Added
- **补位买入股数估算口径统一**：提示信息与实际执行逻辑完全一致
  - **问题背景**：生成 pending_buys（补位计划）时，提示的"预计购买数量"与实际执行时计算逻辑不一致，导致用户困惑
    - 提示逻辑：简单使用 `available_cash / len(targets)` 平均分配
    - 执行逻辑：考虑现金保留比例、成本预估、可用现金上限约束
  - **新增方法**：`PaperTradingRunner._estimate_pending_buy_shares()`
    - 封装统一的补位买入股数估算逻辑
    - 参数：`ts_code`, `price`, `target_weight`, `total_pending_count`, `pendding_capital_retention_ratio`
    - 计算逻辑与 `_execute_pending_buys()` 完全一致：
      1. `total_cash = account.cash * (1 - retention_ratio)` - 扣除保留比例
      2. `available_cash = total_cash / pending_count` - 平均分配
      3. `target_value = total_cash * target_weight` - 按权重计算
      4. 预估成本并检查是否超出可用现金上限
      5. `buy_shares = floor(target_value / price / 100) * 100` - 100股取整
  - **测试覆盖**：新增 `tests/test_pending_buy_estimation.py`，8个测试用例全部通过
    - 正常情况、现金受限、不足一手、异常价格、多目标分配、高保留比例、取整验证

### Changed
- **重构 `_execute_pending_buys()` 方法**：使用统一的 `_estimate_pending_buy_shares()` 计算股数
  - 简化原有逻辑约40行代码
  - 消除了重复的计算代码
  - 确保执行逻辑与估算逻辑完全一致

- **重构 `_print_replacement_targets()` 方法**：使用统一的估算逻辑并增加说明
  - 表头改为"估算股数"而非"建议股数"
  - 新增提示信息：
    - "注意：以下股数为估算值，基于当前价格与现金（保留比例 X%）"
    - "实际执行时会受到执行日价格变化、补位队列长度变化等因素影响，但计算规则一致"
  - 不足一手时显示为 "0 (不足一手)" 而非简单的 "0"

### Documentation
- 新增 `docs/PR/pending_buy_estimation_alignment.md` 详细说明本次改进的背景、方案和影响
- 新增 `docs/guide/pending_buy_estimation_guide.md` 说明补位买入股数的估算逻辑和影响因素

### Version
- 版本号从 0.4.1 升级到 0.4.2

## [0.3.15] - 2026-02-09

### Added
- **新增停牌判断统一工具类 SuspendCalendar**：基于 raw/suspend 数据提供统一的停牌判断接口
  - **问题背景**：停牌信息不在 daily/clean daily 中，停牌股票在 daily 中可能缺行或价格缺失/为0，导致：
    - 纸面交易止损检查可能误触发（0价造成大回撤）
    - 调仓/执行卖出可能因 sell_prices 缺失而直接跳过，既不卖出也不进入延迟卖出队列
    - 回测同样可能出现止损误触发或卖出静默失败
  - **新增模块**：`src/lazybull/common/suspend_calendar.py`
    - 实现 `SuspendCalendar` 工具类，提供 `is_suspended()`, `get_status_reason()`, `batch_is_suspended()` 方法
    - 判定规则基于 raw/suspend 数据：suspend_type='S' => 停牌，suspend_type='R' => 复牌，无记录 => 非停牌
    - 严格模式：suspend 数据文件缺失时抛出 FileNotFoundError 异常
    - 按 trade_date 缓存机制，提高查询效率
  - **测试覆盖**：新增 `tests/test_suspend_calendar.py`，8个测试用例全部通过

### Changed
- **纸面交易集成 SuspendCalendar**：
  - 修改 `scripts/paper_trade.py` 的 `_check_stop_loss()` 方法：
    - 使用 SuspendCalendar 判断停牌（而非依赖 daily 中的 is_suspended 列）
    - 停牌股票跳过止损检查，输出中文日志"停牌，跳过止损检查"
    - 无行情数据股票跳过止损检查，输出中文日志"无行情数据，跳过止损检查"
  - 修改 `src/lazybull/paper/broker.py` 的卖出流程：
    - `generate_orders()` 和 `execute_instructions()` 方法：停牌/无价格时创建 PendingSell 并持久化
    - reason 文案按优先级：停牌优先，否则无价格数据
    - 更新 `_check_can_sell()` 方法支持通过 trade_date 参数使用 SuspendCalendar
  - 修改 `src/lazybull/paper/runner.py`：传递 data_storage 给 broker，确保使用相同的数据根路径

- **回测引擎集成 SuspendCalendar**：
  - 修改 `src/lazybull/backtest/engine.py` 的止损检查：
    - `_check_stop_loss()` 方法使用 SuspendCalendar 判断停牌（而非依赖 price_data 中的 is_suspended 列）
    - 停牌时跳过止损检查，输出中文日志"股票 {stock} 停牌，跳过止损检查"
  - 修改回测引擎的卖出流程：
    - `_sell_stock_with_status_check()` 方法：停牌/无价格时进入延迟卖出队列
    - reason 文案按优先级：停牌优先，否则无价格数据或跌停
  - 新增 `data_storage` 参数支持传入 Storage 实例

### Documentation
- 新增 `docs/PR/suspend_detection_unified.md` 详细说明本次功能的问题、方案、影响范围和验证步骤

## [0.3.11] - 2026-02-09

### Fixed
- **修复 paper_trade positions 命令股票名称显示问题**：持仓明细现在能正确显示股票名称
  - **问题描述**：运行 `python scripts/paper_trade.py positions --trade-date YYYYMMDD` 时，所有股票名称都显示为 `(na)` 而非实际名称
  - **问题根因**：`print_positions()` 函数试图从 `daily_data` 的 `name` 列构建 `stock_names` 字典，但 clean daily 数据不包含 `name` 列
  - **解决方案**：从 `stock_basic` 表读取股票名称（包含 `ts_code` 和 `name` 列）
    - 新增 `build_stock_names_dict()` 辅助函数
    - 优先使用 `DataLoader.load_clean_stock_basic()`
    - 回退使用 `DataLoader.load_stock_basic()`
    - 若无法加载 stock_basic，输出清晰的中文提示日志，建议运行 `python scripts/update_basic_data.py`
  - **核心修改**：
    - `scripts/paper_trade.py`：新增 `build_stock_names_dict()` 函数，修改 `print_positions()` 函数
  - **验收测试**：新增 `tests/test_stock_names_display.py`，5个测试用例全部通过
    - 测试当提供股票名称字典时，持仓明细能正确显示股票名称
    - 测试当不提供股票名称字典时，持仓明细回退显示 `(na)`
    - 测试从 clean/raw stock_basic 加载

### Documentation
- 新增 `docs/PR/fix_stock_names_display.md` 详细说明本次修复的问题、方案和验证方法

## [0.3.9] - 2026-02-09

### Fixed
- **修复纸面交易日志/原因文案不清晰的问题**：卖出订单的 reason 文案现在能准确反映实际交易行为
  - **问题描述**：当目标权重为0时，所有卖出订单统一使用"退出持仓"，但实际可能只是减仓（部分卖出），容易误导用户
  - **解决方案**：根据实际卖出股数和持仓股数判断 reason 文案
    - 完全清仓（`sell_shares == pos.shares` 且 `target_weight == 0`）→ "退出持仓"
    - 部分清仓（`sell_shares < pos.shares` 且 `target_weight == 0`）→ "减仓(退出持仓未完全清仓)"
    - 普通减仓（`target_weight > 0`）→ "减仓"
  - **核心修改**：
    - `src/lazybull/paper/broker.py` 中的 `generate_orders()` 方法：重新组织卖出订单生成逻辑，在计算 sell_shares 后根据实际情况确定 reason
    - 同步更新 PendingSell 延迟卖出订单的 reason 逻辑
  - **验收测试**：新增 `tests/test_sell_order_reason.py`，7个测试用例全部通过

### Improved
- **增强执行日志统计**：在订单执行完成后增加详细的交易类型统计
  - 新增 `_calculate_execution_stats()` 方法，统计以下信息：
    - 买入：新建持仓笔数、加仓笔数
    - 卖出：清仓笔数、减仓笔数
  - 统计基于执行前的持仓快照，避免卖出/买入顺序影响判断
  - 日志格式示例：
    ```
    执行完成: 27 买，26 卖
      - 买入: 新建持仓 15 笔，加仓 12 笔
      - 卖出: 清仓 10 笔，减仓 16 笔
    ```
  - 这些统计以"成交 fill"为准，帮助用户更直观地了解交易结果

### Documentation
- 新增 `docs/PR/fix_sell_order_reason_clarity.md` 详细说明本次修复的动机与变更点
- 更新 CHANGELOG.md 记录版本变更

## [0.3.8] - 2026-02-09

### Fixed
- **修复补位机制导致的清仓问题**：补位目标不再覆盖全量组合目标，避免触发"退出持仓"卖出订单
  - **问题根因**：补位目标直接保存到 `pending/{next_date}.parquet`，T1执行时当作全量目标，导致现有持仓被清仓
  - **解决方案**：引入独立的 `pending_buys` 队列存储补位计划（增量买入），与 `pending_weights`（全量调仓）分离
  - **核心修改**：
    - `scripts/paper_trade.py`：将补位目标保存到 `pending_buys` 队列
    - `src/lazybull/paper/runner.py`：新增 `_execute_pending_buys()` 方法专门处理补位买入（仅买入，不触发卖出）
    - `run_t1()` 方法分别处理 pending_weights 和 pending_buys
    - 重构 `_execute_t1_if_pending()` 和新增 `_handle_failed_buys()` 辅助函数
  - **验收测试**：新增 `tests/test_replenishment_no_sell.py` 验证修复
    - 场景1：持有27只股票 + 3只补位计划 → 不生成卖出订单 ✓
    - 场景2：错误使用（3只补位作为全量目标） → 生成27个卖出订单（清仓）✓
    - 场景3：正确的补位流程 → 仅买入，不影响持仓 ✓

### Improved
- 补位机制更加健壮，与现有持仓管理解耦
- 补位执行不再影响全量调仓逻辑
- 更贴近真实交易场景：补位仅用于增量买入，不触发减仓/清仓

### Documentation
- 新增修复说明文档（详见本次提交的PR描述）
- 补位机制的生命周期和数据格式说明

## [0.3.6] - 2026-02-08

### Added
- **买入失败补位机制**：当 T1 买入因涨停/停牌/不可交易失败时，系统自动生成补位计划，在下一交易日继续买入
  - 新增 `PendingBuy` 数据模型，对称于现有的 `PendingSell`
  - 新增 `PaperStorage.save_pending_buys()` 和 `load_pending_buys()` 持久化方法
  - 新增 `PaperBroker.retry_pending_buys()` 重试补位订单
  - 新增 `PaperTradingRunner.generate_replacement_targets()` 生成补位目标
  - 在 `paper_trade.py run` 中新增步骤 3：处理延迟买入队列（补位计划）
- **补位重试机制**：最多重试 5 次，同日不重复推进 attempts 计数
- **一手可买约束**：补位目标必须满足至少能买入 100 股（1 手）的约束
- **补位输出格式化**：补位目标输出表格与 T0 输出格式保持一致
- **测试覆盖**：新增 `tests/test_buy_replacement.py` 测试文件，覆盖核心功能

### Changed
- **PaperBroker.generate_orders()**：增强买入失败检测，记录失败原因（涨停、停牌、无价格、现金不足、不足一手等）
- **T1 执行流程**：自动检测买入失败并生成补位计划，无需手工干预
- **手工操作指令汇总**：新增"延迟买入清单（补位计划）"部分

### Improved
- 提升资金使用效率，避免买入失败导致的资金长期闲置
- 增强纸面交易与回测的一致性，引入"候选顺延"机制
- 更贴近真实交易场景，自动处理买入受限情况

### Documentation
- 新增 `docs/PR/buy_replacement.md`：详细说明补位机制的设计与实现
- 说明纸面交易与回测在补位处理上的一致性与合理差异

## [0.3.5] - 2026-02-08

### Added
- 纸面交易T0等权策略的"一手可买约束 + 顺延补足"功能
  - 在等权模式下（`weight_method=="equal"`），对每只候选股票检查按资金分配是否能买入至少1手（100股）
  - 不足1手的股票将被跳过，并从排序候选中顺延选择下一只
  - 确保最终保存到pending的目标都是可有效购买的股票
  - 添加详细日志：原始候选数、跳过数、最终目标数、跳过示例

### Changed
- `PaperTradingRunner._generate_signals` 方法新增 `buy_price_type` 参数，用于确定一手判断价格
- 复用 `MLSignal.generate_ranked()` 方法获取完整排序候选列表，支持顺延补足

### Technical Details
- 新增 `PaperTradingRunner._generate_equal_weight_with_lot_constraint` 方法实现核心逻辑
- 等权策略下使用T1的买入价格类型（open/close）进行一手可买性判断
- Score加权策略暂不启用此约束，保持原有行为

## [0.3.4] - Previous Version
- 之前版本的功能
