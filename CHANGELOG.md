# Changelog

All notable changes to this project will be documented in this file.

## [0.90.21] - 2026-08-03

### Fixed

- **修复 `top_list` 下载限频笔误导致极慢**：`get_top_list`
  （`src/lazybull/data/tushare_client/alt.py`）的 `rate_limit_override` 误写为
  60（60 次/分钟 = 1 秒 1 请求），与注释意图"1000 次/分钟"不符，导致龙虎榜
  5240 个交易日全量下载耗时约 88 分钟（日志 rate=0.99/s）。
  - 实测 TuShare `top_list` 60ms 间隔连续请求 30 次无限流，实际瓶颈是服务端
    响应（~3.8s/请求），官方未对该接口设低频限制；
  - 改为 `rate_limit_override=1000` 后配合按日并发（`download_top_list` 已用
    `_run_concurrent`），吞吐从 60 次/分钟提升到数百次/分钟（受服务端响应与
    令牌桶约束），全量预计从 ~88 分钟压缩到 ~10 分钟量级；
  - 若触发限流，`client.query` 会自动解析"频率超限(X次/分钟)"并自适应降频兜底。
- **测试**：`test_tushare_client_rate_limit.py` 新增 `get_top_list` 传
  `rate_limit_override=1000` 的断言。

## [0.90.20] - 2026-08-03

### Changed

- **report_rc 下载并发化**：`download_report_rc`（`scripts/raw_download/alt.py`）
  原为按年串行下载（每年内部逐页翻页, 每请求响应约 5s, 22 年全量串行耗时很长）。
  现改为按年并发（复用 `_run_concurrent` + `collect=True`）：
  - 各年份 worker 独立下载（含超限年份自动二分分片），网络等待并行化，总 QPS
    仍受 TushareClient 令牌桶限频约束；
  - 全部完成后统一 `_save_merged` 合并去重落盘，避免并发写同一文件；
  - `success`/`empty` 计数加锁保护，`tracker.tick` 线程安全。
- **测试**：`test_download_report_rc_adaptive.py` 新增多年份并发合并、串行降级
  一致性用例。

## [0.90.19] - 2026-08-03

### Fixed

- **修复 `report_rc` 单次查询超限导致整年下载失败**：
  TuShare `report_rc` 接口对"一次查询 (start_date/end_date + offset 翻页)"的总
  行数上限为 100000 条 (offset 上限 100000)。`download_report_rc` 按年整段查询,
  数据量超限的年份 (2009 年约 10.2 万条, 2020/2023/2025 等约 20~30 万条) 翻页到
  offset > 100000 时服务端返回"查询数据失败, 请确认参数！", 整年数据全部丢失并
  记为失败 (实测 2009 年在 offset=102000 失败)。
  - `scripts/raw_download/alt.py` 新增 `_query_report_rc_adaptive`: 整段查询失败
    时自动把日期范围二分递归重试 (最多 2^6=64 段), 任意规模数据都能取全; 未超限
    年份仍按整年单次查询, 零额外开销。
  - 新增日期辅助 `_mid_date_str` / `_next_date_str`。
  - `download_report_rc` 改走 `_query_report_rc_adaptive`。
- **测试**: 新增 `tests/test_download_report_rc_adaptive.py` (7 用例: 整段成功/
  二分合并/深度耗尽抛错/空区间/超限年份下载/日期辅助)。既有
  `TestReportRcPagination` 保持通过。

## [0.90.18] - 2026-08-03

### Changed

- **按季度下载并发化（fund_portfolio 等大幅提速）**：
  `scripts/raw_download/periodic.py` 的 `download_by_period` 原为纯串行逐季度
  下载，每个季度内部还要按 `page_limit`（如 fund_portfolio=8000）逐页翻页，
  每请求服务端响应约 5s，导致大季度（上百万条）单季度耗时十几分钟、全量
  86 个季度约 2 小时。现改为复用 `_run_concurrent`（受
  `tushare.download_concurrency` 并发数与 TushareClient 令牌桶限频约束）并行下载：
  - 分区模式（`fund_portfolio`/`fina_indicator`）：各季度独立分区文件，worker
    内直接落盘，天然线程安全；
  - 非分区模式（`forecast`/`express`）：worker 返回各季度 df，全部完成后统一
    `_save_merged` 合并去重落盘；
  - `success`/`empty` 计数加锁保护，`ProgressTracker.tick` 保持线程安全。
- **`_run_concurrent` 支持收集返回值**：`scripts/raw_download/core.py` 的
  `_run_concurrent` 新增 `collect` 参数（默认 False，既有调用方行为不变），
  按 `work_items` 顺序返回 worker 返回值列表。
- **测试**：新增 `tests/test_download_periodic_concurrency.py`（`_run_concurrent`
  collect 串行/并发路径、download_by_period 分区/非分区模式行为、合并去重）。

## [0.90.17] - 2026-08-03

### Fixed

- **修复 `_save_merged` 的 concat FutureWarning**：`pd.concat` 遇到全 NA 片段（有行但所有
  值均为 NaN）时触发 pandas "empty/all-NA entries" 行为变更告警；现于 concat 前剔除全
  NA 片段并保留全部列集合，消除告警且 schema 不缺失。修复位于
  `scripts/raw_download/periodic.py`，影响 `express`/`cyq_perf`/`stk_holdernumber`/
  `report_rc` 等按季度批量下载路径。
- **测试**：`test_download_raw_fixes.py` 新增 `TestSaveMergedAllNA`（全 NA 片段剔除 +
  列集合保留 + 全 NA 仅存时结果为空）。

### Changed

- **cyq_perf 限频统一为 100 次/分钟**：`src/lazybull/data/tushare_client/core.py`
  工作区默认值已从 200 调整为 100（并新增 `margin_detail=200`），同步更新
  `test_tushare_client_rate_limit.py` 断言（interval 0.6s）与 `configs/base.yaml`、
  `core.py` 注释口径。

## [0.90.16] - 2026-08-03

### Removed

- **删除 `scripts/run_ml_backtest.py`（1059 行）**：用户已不再独立运行该脚本。
  - 评估面板能力（`evaluate_daily` / `export_evaluation_panel` / `equal_count_grouping` /
    `_append_dict_to_csv` / `_append_trades_to_cumulative_file`）下沉到
    `src/lazybull/backtest/eval_panel.py`。
  - `load_backtest_data` / `prepare_price_data` / `run_ml_backtest` / `_generate_run_id`
    无外部引用，随脚本删除。
  - 引用方同步切换：`tests/test_eval_panel.py`、`tests/test_ml_backtest_trades_runs.py`、
    `examples/demo_eval_panel.py` 改为直接 import `src.lazybull.backtest.eval_panel`。
  - 删除已弃用的 `scripts/batch/batch_backtest.ps1`。

### Changed

- **`scripts/build_clean_features.py`（1068 行）核心逻辑下沉**：
  - `build_clean_data` 下沉到 `src/lazybull/data/build_clean.py`。
  - `build_features_data` 与 `_build_features_parallel` 下沉到
    `src/lazybull/features/pipeline.py`。
  - 脚本保留 `main` + `apply_build_all_feature_flags`（CLI 参数处理）作为薄入口。
  - 下沉函数与原脚本逐字等价（AST 对比验证），全量回归通过。

## [0.90.15] - 2026-08-03

### Changed

- **清理 `scripts/paper_trade.py` 遗留死薄壳**：删除 6 个仅转发到
  `src.lazybull.paper.runtime` 且无人引用的本地薄壳
  （`_check_stop_loss` / `_process_pending_sells` / `_process_pending_buys` /
  `_execute_t1_if_pending` / `_handle_failed_buys` / `_execute_t0_if_rebalance_day`）
  及其对应 `shared_*` import，消除"搬一半"的双胞胎结构；顺带清理 unused 的 `List` import。
- **测试适配**：`test_suspended_stock_handling.py` 改直接引用
  `src.lazybull.paper.runtime._check_stop_loss`（不再经 paper_trade 薄壳）。

## [0.90.14] - 2026-08-03

### Fixed

- **`raw_download/cli.py` 运行时报错修复**：
  - 补 `import time`（`main` 中 `time.time()` 使用，原报 `NameError: name 'time' is not defined`）。
  - 补 `_fmt_duration` import（`finally` 中总耗时打印使用）。
  - **并发数写入 core 模块**：拆分后 `cli.main` 原用 `global _DOWNLOAD_CONCURRENCY` 只改 `cli`
    模块变量，不影响 `core._run_concurrent` 读取（始终为初始值 1=串行）；改为直接写
    `raw_core._DOWNLOAD_CONCURRENCY`，使 `--concurrency`/`download_concurrency` 配置真正生效。
- **测试**：`test_download_raw_fixes.py` 新增 `TestCliMainRuntime`（验证并发写入 core 模块）。

## [0.90.13] - 2026-08-03

### Changed

- **trade_cal 改为每次全量下载**：`download_basic_data` 不再按 `start/end` 裁剪区间，
  每次涉及 trade_cal 的下载（默认日线 / `--all` / `--only-basic`）都以
  `get_trade_cal(exchange="SSE")` 全量拉取（不传日期参数），合并旧数据去重排序后保存，
  保证交易日历始终完整最新；全量拉取失败时保留已有数据。仅针对 trade_cal 如此操作，
  其余数据仍按日期区间下载。
- **测试**：`test_download_raw_fixes.py` 新增 2 用例（全量拉取不传日期、失败时保留已有数据）。

## [0.90.12] - 2026-08-03

### Changed

- **`data/tushare_client.py`（约 940 行）→ `data/tushare_client/` 子包**：
  `__init__.py` 门面（`TushareClient` 由 `ClientCoreMixin/ClientBasicMixin/`
  `ClientDailyMixin/ClientFundamentalMixin/ClientAltMixin` 组合）+ `core/basic/`
  `daily/fundamental/alt` 5 个模块。旧路径 `from src.lazybull.data.tushare_client import X`
  经门面 re-export 保持兼容（含 `ts`、`FINA_INDICATOR_DEFAULT_FIELDS`）。
- **`scripts/download_raw.py` → `scripts/raw_download/` 子包 + 薄入口**：
  `raw_download/`（`core/basic/daily/periodic/daily_partition/alt/cli`）承载全部逻辑；
  `scripts/download_raw.py` 保留为薄入口（`python scripts/download_raw.py` 命令不变），
  `from scripts import download_raw` 经薄入口 re-export 兼容。
- **测试适配**：`test_download_raw_fixes.py` 的 monkeypatch 目标更新到
  `scripts.raw_download.alt`（门面 setattr 不影响实际模块绑定）。

## [0.90.11] - 2026-08-03

### Changed

- **TushareClient 接口级限频自适应**：
  - 限频改为**按接口分桶令牌桶**（每个接口独立 `interval`），不同限频的接口互不拖累
    （如 `cyq_perf=200次/分钟` 不影响全局 `rate_limit=500`）。
  - 新增接口限频表 `_API_RATE_LIMITS_DEFAULT`（默认收录 `cyq_perf=200`）；未知接口回退全局。
  - 收到限流错误时**自动解析**"频率超限(X次/分钟)"并动态更新该接口限频（自适应学习）。
  - 解决并发 36 下 `cyq_perf` 等低限频接口被限流 → 15s 长等 → 重试又限流的恶性循环。
- **测试**：新增 `tests/test_tushare_client_rate_limit.py`（4 用例）覆盖接口 interval、
  分桶独立性、限流错误自适应更新。

## [0.90.10] - 2026-08-02

### Fixed

- **download_raw 下载性能回归（20 小时 → 恢复）**：
  - `_query_with_pagination` 改用 `client.query`（走令牌桶限频 + 限流重试），
    原先直接 `client.pro.query` 绕过限频，在并发下触发 TuShare 限流后整段
    period 失败且不落盘，重跑反复重下导致耗时爆炸。
  - 移除 probe 探测逻辑（每整页多一次额外请求，且对不支持 offset 的接口存在
    死循环风险），恢复 `len(df) < page_limit` 终止条件，并新增 `max_pages` 兜底。
  - `download_stk_holdernumber` 增加断点续传（读已有最大 `ann_date`，仅下载其后
    月份段），修复每次全量重下 180 个月的问题；单月改为分页拉取，规避单次
    3000 条上限截断。
  - `download_report_rc` 按年改为分页拉取（`page_limit=2000`），规避单次 2000 条
    上限导致研报数据截断。
- **测试**：新增 `tests/test_download_raw_fixes.py`（6 用例）覆盖分页累积、
  max_pages 兜底、stk_holdernumber 断点续传、report_rc 分页。

## [0.90.9] - 2026-08-02

### Changed

- **features 层大文件规整为子包**：
  - `features/builder.py`（约 1440 行）→ `features/builder/` 子包：
    `__init__.py` 门面（`FeatureBuilder` 由 `cache/orchestration/helpers/factors`
    四个 mixin 组合）+ `static_core.py` / `static_extra.py`（12 个静态函数）。
  - `features/ensure.py`（约 1780 行）→ `features/ensure/` 子包：
    `__init__.py` 门面 + `entry/historical/factor_load/downloads/bulk/incremental/`
    `historical_assets/industry/schema/constants` 10 个模块。
  旧路径 `from src.lazybull.features.builder/ensure import X` 经门面 re-export 保持兼容。
- **测试适配**：更新 `test_ensure_and_t0_printing.py` 与
  `test_factor_wiring_cashflow_consensus_revision.py` 的 monkeypatch/patch 目标到
  实际子模块（门面 setattr 不影响实际模块绑定），
  `test_technical_indicators_precompute.py` 的 patch 目标到 `builder.helpers`。
- **性能验证**：拆分前后性能基线对比（200 股 × 250 交易日）：
  `precompute_daily_adj` 51.94→53.91ms、单日构建 686.40→655.26ms（均 ±5% 噪声范围），
  构建输出逐值一致（`assert_frame_equal` 通过），内存释放节奏（gc.collect 时序）与
  网络增量短路逻辑保持不变。

## [0.90.8] - 2026-08-02

### Changed

- **walk_forward 模块规整为子包**：将 `src/lazybull/ml/` 顶层 11 个散落的
  `walk_forward_*.py` 收敛为 `src/lazybull/ml/walk_forward/` 子包
  （`__init__.py` 门面 + `backtest/cli/deploy_training/reporting/runner/split_training/`
  `summary/training/training_core/training_reporting/utils`）。
  包内交叉引用统一改相对导入；对 `common`/`data`/`universe` 的相对导入层级 +1。
- **外部引用一次性迁移**：`scripts/walk_forward.py`、`scripts/ana/diagnose_training_stability.py`
  及 3 个测试文件（`test_walk_forward.py`、`test_walk_forward_training_modules.py`、
  `test_training_feature_flag_forwarding.py`）改用新路径；旧 `ml.walk_forward_*`
  模块路径不再保留（一次性破坏性路径变更）。

## [0.90.7] - 2026-08-02

### Changed

- **纸面交易目录规整为子包**：将散落的 `runner_*.py` / `broker_*.py` / `storage_*.py` mixin 文件
  收敛为三个子包：
  - `paper/runner/`（`__init__.py` 门面 + `calendar/rebalance/instructions/execution/pricing/signals/replacement`）
  - `paper/broker/`（`__init__.py` 门面 + `tradability/execution/retry/positions`）
  - `paper/storage/`（`__init__.py` 门面 + `state/config/records/queue/maintenance`）
  `from src.lazybull.paper.runner import PaperTradingRunner` 等既有导入路径保持不变。
- **测试适配**：更新 4 处 `monkeypatch`/`patch` 目标为子包路径
  （`paper.runner.signals` / `paper.runner.replacement` / `paper.runner.execution` / `paper.runner.calendar`）。

## [0.90.6] - 2026-08-02

### Changed

- **拆分训练核心为 `ml/train_core/` 子包**：2512 行的 `src/lazybull/ml/train_core.py` 拆分为
  `constants.py`（特征列清单/freshness 常量）、`labels.py`（标签变换）、`split.py`（数据切分）、
  `features.py`（特征清洗/因子排除，含 `_factor_exclude_cache` 缓存）、`prepare.py`
  （`prepare_training_data`）、`weights.py`（时间衰减/rank 权重）、`eval.py`（逐日评估/IC 指标）、
  `xgb.py` 与 `lgb.py`（按模型分拆，便于后续新增模型）。
  `train_core/__init__.py` 为门面，re-export 全部公共符号、常量与 `_factor_exclude_cache`，
  `from src.lazybull.ml.train_core import <X>` 与 `import ...train_core as tc` 既有用法保持不变。
- **测试适配**：将因子包化而失效的 `monkeypatch` 目标更新到实际模块
  （`xgboost.XGBRanker`、`train_core.prepare._load_factor_exclude_list`）。

## [0.90.5] - 2026-08-02

### Changed

- **拆分纸面交易大文件为 mixin**（沿用回测 `BacktestXxxMixin` 先例，扁平文件 + 门面组合）：
  - `paper/runner.py`：2390 行 → 150 行门面（`PaperTradingRunner` 组合 7 个 mixin）；
    拆分出 `runner_calendar.py`（日历/日期）、`runner_rebalance.py`（调仓日判断）、
    `runner_instructions.py`（指令生成）、`runner_execution.py`（T0/补位执行）、
    `runner_pricing.py`（价格/Kelly/净值）、`runner_signals.py`（信号）、
    `runner_replacement.py`（补位目标）。
  - `paper/broker.py`：1295 行 → 门面（`PaperBroker` 组合 4 个 mixin）；
    拆分出 `broker_tradability.py`、`broker_execution.py`、`broker_retry.py`、`broker_positions.py`。
  - `paper/storage.py`：1121 行 → 门面（`PaperStorage` 组合 5 个 mixin）；
    拆分出 `storage_state.py`、`storage_config.py`、`storage_records.py`、`storage_queue.py`、
    `storage_maintenance.py`。
  - 方法体原样搬运、行为不变，`from src.lazybull.paper.runner import PaperTradingRunner` 等
    既有导入路径保持不变。
- **新增 `common/constants.py`**：`SHARE_LOT_SIZE`/`SEPARATOR_LENGTH` 由 `runner.py` 迁移至此，
  供门面与 mixin 共用，避免循环依赖。
- **测试适配**：将因方法迁移而失效的 `monkeypatch` 目标更新到实际 mixin 模块
  （`runner_calendar`/`runner_signals`/`runner_replacement`/`runner_execution`），
  保证 mock 真正生效。

## [0.90.4] - 2026-08-02

### Removed

- **删除纸面交易历史遗留死代码**：
  - `paper/runner.py`：删除 `_regime_combined`（零调用且引用不存在属性）、
    `_estimate_pending_buy_shares_backtest_style`（零调用薄包装）、`run_t1`/`run_retry`
    （生产已被 `runtime.py` 内联等价逻辑替代）、`_build_pnl_price_map_for_date`/
    `_resolve_buy_pnl_price_for_position`（零调用，与 broker 版重复）、
    `_generate_ranked_with_lot_constraint` 内 `break` 后不可达孤儿块。
  - `paper/broker.py`：删除权重驱动旧路径 `generate_orders`/`execute_orders` 及仅被其
    调用的 `_print_order_detail`，生产执行统一走指令驱动 `execute_instructions`。
  - 修复删除 `@staticmethod` 方法后残留孤立装饰器误装饰 `_record_nav` 的问题。
- **清理测试**：删除 13 个仅覆盖已删除旧路径的测试，改写 16 个测试为指令驱动
  `execute_instructions` 与组合价值口径（`test_sell_order_reason` / `test_buy_replacement` /
  `test_suspended_stock_handling` / `test_pending_buy_estimation`）。

### Changed

- **停牌日历构建共享**：`_get_suspend_calendar` 抽为 `common/suspend_calendar.py` 的
  `get_suspend_calendar()`，回测引擎与纸面 broker 复用（保留各自延迟缓存与默认 Storage 赋值）。
- **持有天数计算共享**：交易日口径持有天数收敛为 `common/date_utils.py` 的
  `calc_holding_trade_days()`，`runner._calc_holding_days` 与 `broker._calc_holding_trade_days`
  改为薄包装复用。
- **统一补位股数估算口径**：`_print_replacement_targets` 展示由“现金均分”口径改为
  与实际执行一致的“组合价值×槽位权重”口径（对齐回测），删除旧现金均分实现
  `_estimate_pending_buy_shares`，展示建议股数与真实成交一致。
- **清理**：移除 `_generate_instructions` 中无意义的 `del sell_price_type, protected_stocks`
  消警残留（参数按兼容保留）。

## [0.90.3] - 2026-08-02

### Changed

- **拆分实验对比脚本为子包**：将 3467 行的 `scripts/compare_walk_forward.py` 按职责拆分为
  `scripts/compare/` 子包（`constants.py` / `loading.py` / `aggregate.py` / `scoring.py` /
  `metrics_desc.py` / `detail_display.py` / `excel.py` / `report.py`），
  原脚本保留为薄入口（CLI 参数解析 + 从子包 re-export 公共 API），
  列名、评分权重、聚合/评分/展示逻辑与 Excel 输出行为完全不变，
  `tests/test_compare_selection_score.py` 的既有导入路径保持不变。

## [0.90.2] - 2026-08-02

### Removed

- **移除无效信号门控历史接口**：删除统一交易参数和 walk-forward 中的信号置信度门控、
  composite 门控、滚动质量门控、动态 Top-N 与持仓奖励参数。这些参数未进入
  `TradingConfig`、信号或回测引擎，历史上接受参数但不会改变运行结果。
- **清理无效汇总字段**：walk-forward summary 不再写入上述参数，实验对比不再聚合
  从未由当前回测链路生成的门控持币率、平均仓位和平均置信度指标。
- **移除未接通的 ECT 入口**：删除 walk-forward 的权益曲线交易参数、日志和汇总列；
  风险模块中的独立实现保留，但不再暴露不会传入当前回测引擎的命令行接口。
- **移除无效滚动步长接口**：按 `split_count + final_date` 反推的切分从未使用 `--step`；
  现删除该 CLI、训练日志与汇总字段，并移除 batch 中无效的扫描维度和重复任务组合。

### Changed

- **统一可交易性状态判断**：新增单条状态记录的共享纯函数，回测/选股的 DataFrame 路径与
  纸面交易 broker 复用同一停牌、涨停和跌停判断；纸面交易特有的 `tradable` 买入过滤和
  `SuspendCalendar` 优先级保持不变。
- **统一整手买入股数计算**：新增金额、价格到整手股数的共享纯函数，回测引擎、纸面
  broker 与 runner 统一复用；各路径原有预算、手续费和现金缩量规则保持不变。
- **拆分 walk-forward 汇总模块**：将 split 指标整理、条件参数清洗和 summary CSV 写入
  从 `scripts/walk_forward.py` 迁移至 `src/lazybull/ml/walk_forward_summary.py`。
- **拆分 walk-forward OOS 回测模块**：将单 split 数据准备、引擎执行与绩效提取迁移至
  `src/lazybull/ml/walk_forward_backtest.py`，主脚本仅保留调用和结果编排。
- **合并训练运行记录构造器**：普通训练与 walk-forward 共用 `ml/run_logger.py` 的记录构造
  逻辑，统一验证隔离、TopK 与测试集指标落盘规则，并正式记录 `num_leaves`。
- **拆分 walk-forward 报告模块**：TopK 明细、成交归因和全周期串联净值统一迁移至
  `src/lazybull/ml/walk_forward_reporting.py`。
- **拆分 walk-forward 训练域模块**：将训练窗口构建、多偏移/多种子集成、split/deploy
  训练执行及训练评估辅助函数从 `scripts/walk_forward.py` 迁移至
  `src/lazybull/ml/walk_forward_training.py`；主脚本保留 CLI 编排并通过导入重导出兼容旧引用。
- **细分 walk-forward 训练子模块**：`src/lazybull/ml/walk_forward_training.py` 调整为兼容门面，
  训练核心函数与常量迁移至 `walk_forward_training_core.py`，日志/指标打印迁移至
  `walk_forward_training_reporting.py`，split 与 deploy 执行入口分别迁移至
  `walk_forward_split_training.py` 与 `walk_forward_deploy_training.py`；
  算法与训练行为保持不变。
- **拆分 walk-forward CLI 与 runner**：参数构建、解析、规范化与校验迁移至
  `src/lazybull/ml/walk_forward_cli.py`，运行编排与 split 过滤迁移至
  `src/lazybull/ml/walk_forward_runner.py`；`scripts/walk_forward.py` 调整为薄入口并保持历史导出。
- **拆分回测主循环状态机边界**：`BacktestEngine.run` 原样迁移至
  `src/lazybull/backtest/run_loop.py` 的 `BacktestRunLoopMixin.run`，
  每日 T0/T1 状态推进、早调仓回滚与统计输出顺序保持不变，`engine.py` 保留状态与执行组件实现。
- **拆分回测信号执行边界**：将 `_build_signal_data`、`_post_filter_candidates`、
  `_get_position_weight_for_planning`、`_queue_condition_sell_refill_signal`、
  `_get_holding_features_row`、`_generate_signal` 原样迁移至
  `src/lazybull/backtest/signal_execution.py` 的 `BacktestSignalExecutionMixin`，
  保持行业约束延迟导入与 `BacktestEngineML` 三个 hook 覆写行为不变。
- **拆分回测买入执行边界**：将 `_execute_pending_buys`、`_process_position_completion`、
  `_buy_stock_with_status_check`、`_build_position_extra_info`、`_buy_stock_direct`、
  `_buy_stock`、`_update_completion_attribution` 原样迁移至
  `src/lazybull/backtest/buy_execution.py` 的 `BacktestBuyExecutionMixin`，
  保持 T1 候选顺位、未成交槽位、补齐窗口、旁路归因、整手股数、手续费、
  最小买入阈值与 pending order 行为不变。
- **拆分回测卖出执行边界**：将 `_queue_rebalance_sells`、`_check_and_sell`、
  `_execute_pending_condition_sells`、`_check_stop_loss`、`_execute_pending_stop_loss_sells`、
  `_sell_stock`、`_sell_stock_with_status_check`、`_sell_stock_direct` 原样迁移至
  `src/lazybull/backtest/sell_execution.py` 的 `BacktestSellExecutionMixin`，
  保持调仓卖出候选、持有期/盈利延续、T0 触发 T1 执行、止损去重、
  停牌/跌停延迟、开盘/收盘口径及 PnL 与交易记录字段不变。
- **拆分回测延迟订单执行边界**：将 `_record_pending_order_event` 与
  `_process_pending_orders` 原样迁移至
  `src/lazybull/backtest/pending_execution.py` 的 `BacktestPendingExecutionMixin`，
  保持 `PendingOrderManager` 在 `__init__` 的 `event_sink=self._record_pending_order_event`
  绑定、每日重试流程、可交易性检查、买卖分发与成功/过期/继续延迟状态更新不变。
- **拆分回测报告与日志边界**：将调仓摘要 formatter 与日级日志/告警/信号汇总、
  决策 trace、进度日志等方法原样迁移至
  `src/lazybull/backtest/reporting.py` 的 `BacktestReportingMixin`，
  `engine.py` 保留 `_get_min_buy_value_threshold` 并通过导入重导出
  `_format_rebalance_decision_summary` 兼容既有引用。
- **清理回测引擎死代码**：删除 `engine.py` 顶层未调用的
  `_format_buy_execution_stock_list`、`_sum_buy_execution_weights`、
  `_format_buy_execution_summary`，不再保留重复实现。

## [0.90.1] - 2026-08-02

### Removed

- **移除 best_iteration 自适应候选重训能力**：`scripts/walk_forward.py` 删除
  `--adaptive-best-iter-retrain` 与 `--adaptive-low-iter-max-retries` 参数，以及对应候选重训、
  替换判定与元数据写入逻辑。
- **移除批量脚本透传开关**：`scripts/batch/batch_walk_forward.ps1` 删除
  `$adaptive_best_iter_retrain`、`$adaptive_low_iter_max_retries` 配置和命令行透传。
- **移除对比汇总中的相关参数列**：`scripts/compare_walk_forward.py` 删除
  `adaptive_best_iter_retrain` 与 `adaptive_low_iter_max_retries` 参数映射和候选列定义。
- **移除对应测试覆盖**：`tests/test_training_feature_flag_forwarding.py` 删除
  自适应重训相关单元测试，并同步更新多种子集成测试调用签名。

## [0.90.0] - 2026-08-01

### Added

- **行业中性与绝对收益混合标签**：`train_ml_model.py` 与 `walk_forward.py` 新增
  `--neutral-label-blend-weight`，训练期动态生成混合目标，不修改特征分区 schema；权重为 0
  时完全沿用原行业中性标签，权重为 1 时等价于原始收益标签。
- **混合标签批量实验**：`batch_walk_forward.ps1` 支持扫描混合权重，汇总与对比表同步记录
  `neutral_label_blend_weight`，便于独立比较行业内 Alpha 与绝对收益目标的平衡。

## [0.89.2] - 2026-08-01

### Fixed

- **纸面交易分批选股数量对齐回测**：`_generate_signals()` 在传入 `trading_config`
  时不再固定使用配置中的总 `top_n`，改为优先使用调用参数 `top_n`（即本批槽位数）。
  修复 `stagger_tranches > 1` 时首批仍按总 `top_n` 选股、导致首批即买满的问题。

## [0.89.1] - 2026-08-01

### Fixed

- **纸面交易分批槽位上限对齐回测**：`run_t0` 生成买入指令时，`desired_position_count`
  改为传递总 `top_n`（而非本批槽位数）。修复 `stagger_tranches > 1` 时首批建仓后，
  后续批次被误判为“无可用空槽”而无法继续买入的问题。

## [0.89.0] - 2026-08-01

### Added

- **Walk-forward 交易归因明细**：OOS 回测按 split 自动导出成交记录与买入执行归因，
  包含信号日、计划股、实际买入股、候选排名、目标权重、未成交/替换原因及 T0 到 T1
  价格变化；数据仅旁路记录，不参与交易决策。
- **信号到持仓收益分析**：新增 `scripts/ana/analyze_signal_execution_gap.py`，严格区分
  信号日 Top30 标签收益、实际买入股票标签收益与已平仓真实持仓收益，并汇总成交率、
  替换率、Top30 覆盖率、实际排名和失败原因。

## [0.88.0] - 2026-08-01

### Added

- **显式因子排除清单**：`train_ml_model.py` 与 `walk_forward.py` 新增
  `--factor-exclude-file`，启用 `--factor-prune` 时可为实验指定独立 JSON；未指定仍读取
  `data/models/factor_exclude_list.json`，保持生产默认行为。
- **稀疏因子首轮候选**：新增 `configs/factor_exclude_candidate_sparse_v1.json`，严格合并生产
  53 项与 `order_imbalance_mean_5`、`zscore_intraday_vol_structure`、`zscore_inv_turn`
  三个低使用、低覆盖根因子；batch walk-forward 默认指向该实验清单。

### Fixed

- **排除清单缓存隔离**：因子排除缓存改为按清单绝对路径保存，避免同一进程运行多个候选
  时错误复用首个清单。

## [0.87.0] - 2026-08-01

### Added

- **因子使用稳定性分析**：新增 `scripts/ana/analyze_factor_stability.py`，支持精确指定
  模型版本或版本区间，并递归展开 `EnsembleModel` 子模型，按归一化 importance、模型内
  排名、零值率和 Top50% 出现率聚合因子跨模型稳定性。
- **分层候选输出**：报告区分严格低使用候选和待 IC 复核观察名单；分析过程只读，不会
  修改模型注册表或 `factor_exclude_list.json`，避免候选未经消融验证直接进入生产裁剪。

## [0.86.7] - 2026-08-01

### Fixed

- **非整除分批调度均匀化**：分批信号日期不再使用 `floor(rebalance_freq / K)`
  固定偏移，改为按完整调仓周期比例均匀取整。例如 20 日分 3 批时，偏移由
  `0/6/12` 修正为 `0/7/13`，循环间隔由 `6/6/8` 改为 `7/6/7`；K1、K2、K4
  等可整除配置保持不变。

## [0.86.6] - 2026-08-01

### Fixed

- **分批调仓仓位修复**：`stagger_tranches > 1` 时不再由首批一次占满全部槽位、导致
  后续批次无法买入且组合长期只有约 `1/K` 仓位。现在将总 `TopN` 按批次拆分，并按
  各批槽位占比分配资金；`Top20/K4` 对应每批 5 只，最终仍为 20 只，而非 80 只。
- **非整除批次修复**：总持仓数不能被批次数整除时，按前批优先分配余数，并使用实际
  槽位比例分配预算。例如 `Top30/K4` 为 `8/8/7/7`，预算合计保持 100%。

## [0.86.5] - 2026-08-01

### Fixed

- **多偏移集成验证口径修复**：集成模型不再使用验证评分最高子模型的 calibration
  面板；改为选择起始日晚于所有保留子模型训练及早停截止日的共同未见面板，消除跨偏移
  窗口的验证泄漏与赢家偏差。无法证明面板独立，或 calibration 已参与子模型筛选时，
  禁用该次集成验证指标。
- **训练日期元数据补齐**：训练统计新增实际训练子集起止日期，供集成验证执行严格的
  时间边界检查；OOS 预测、模型集成和回测路径保持不变。

## [0.86.4] - 2026-07-29

### Added

- **freshness 归因实验策略**：新增 `state_keep_event_no_decay`，保留状态型 freshness、
  删除事件型 freshness，但不衰减事件因子原始值；默认策略仍为 `state_keep_event_decay`。
- **walk-forward 实验可追溯性**：汇总与对比结果新增 freshness 策略和事件衰减半衰期；
  非衰减策略的半衰期统一记为空，避免无效参数参与实验签名。

## [0.86.3] - 2026-07-28

### Changed

- **P2-C freshness 策略正式落地（状态型保持、事件型衰减）**：
  - `prepare_training_data()` 新增 `freshness_strategy`（默认 `state_keep_event_decay`）与 `event_freshness_half_life_days` 参数；
  - `state_keep_event_decay` 策略下：
    - 状态型 freshness（如 `fundamental_freshness_days`、`holder_freshness_days`）保留；
    - 事件型 freshness（如 `forecast_freshness_days`、`express_freshness_days`、`consensus_freshness_days`、`cons_revision_freshness_days`）不直接入模，改为用于对应事件因子的指数衰减；
  - 保留 `drop_all` 兼容策略用于纯硬删除模式。

### Fixed

- **训练入口 freshness 处理去噪优化**：事件型 freshness 不再作为独立特征直接输入模型，避免模型过拟合披露节律；同时仍保留其时效信息并注入到事件值本身。

### CLI

- `scripts/train_ml_model.py` 与 `scripts/walk_forward.py` 新增参数：
  - `--freshness-strategy`（`state_keep_event_decay|drop_all`）
  - `--event-freshness-half-life-days`
- `scripts/batch/batch_walk_forward.ps1` 新增批量配置透传：
  - `$freshness_strategy`
  - `$event_freshness_half_life_days`

## [0.86.2] - 2026-07-28

### Fixed

- **训练入口特征质量门禁**：`prepare_training_data()` 新增硬过滤：
  - 统一删除全部 `*freshness*` 特征，避免模型学习披露节律噪声；
  - 删除高缺失特征（默认缺失率阈值 `0.4`）；
  - 删除全空/常数特征；
  - 对 `zscore_*` 与 `zscore_*_sz` 增加联动剔除，避免派生列绕过过滤名单。
- **公告类多版本 PIT 对齐修复**：`fundamental`/`cashflow_quality`/`earnings`/`holder`/`express` 不再按 `ts_code+end_date` 仅保留最终版本，改为保留同报告期多公告版本并由交易日 PIT 查询选择当日可见版本。
- **快报惊喜值前视修复**：`express_surprise` 改为仅使用 `forecast_ann_date <= express_ann_date` 的历史预告版本计算，避免引用未来修订值。

### Changed

- **全历史截尾停用**：`fundamental` 与 `cashflow_quality` 中基于全样本分位数的 winsorize 截尾逻辑已移除，降低未来信息泄露风险。

## [0.86.1] - 2026-07-28

### Fixed

- **并行/串行特征构建对齐**：`features/parallel.py` 补齐价值红利、资金流、基本面代理回填步骤，修复并行路径列缺失导致的 schema 漂移问题。
- **复权因子缺失污染修复**：移除 `adj_factor=1.0` 伪默认值回退，改为按股票前后向填充；仍缺失时保留 NaN，避免伪造复权价污染标签与收益类因子。
- **涨跌停判定修复**：`cleaner` 层按主板/创业板/科创板/北交所及 ST 规则统一计算，并在有 `stk_limit` 时用涨跌停价覆盖阈值判定。
- **因子处理器安全性增强**：新增 `ts_code` 去重与 merge 行数校验，修复重复键静默错配；处理器异常时改为记录错误并填充 NaN 占位，保证 schema 稳定。
- **Storage 读取失败不再静默**：文件损坏等读取异常改为抛出错误，不再与“文件不存在”同构返回 None。

### Changed

- **日期契约统一**：`DataLoader` 与多个因子模块统一输出 YYYYMMDD 字符串日期；新增公共日期规范化函数，避免 `astype(str)` 产生字符串 `nan`。
- **load_clean_daily_by_date 去隐式副作用**：默认不再在“读取”方法里自动触发下载/清洗；如需自动补齐需显式传入 `auto_ensure=True`。

### Docs

- `docs/data_contract.md` 新增设计约束：
  - 涨跌停标记仅在 cleaner 层处理，features 层只复用。
  - 各层日期字段统一为 YYYYMMDD 字符串。

## [0.86.0] - 2026-07-29

### Added

- **回测/纸面交易共享决策核心**：新增 `src/lazybull/trading/` 包（`buy_plan.py`、`sell_rules.py`、`sizing.py`），将买入计划生成、卖出规则与仓位计算抽取为单一实现，回测 `engine.py` 与纸面 `runner.py`/`broker.py` 统一接入，消除两侧逻辑漂移；新增 30+ 共享核心单元测试

### Fixed

- **修复此前功能删除提交遗留的多处“截肢”损伤**：
  - `paper/storage.py`：从完整版本恢复并裁剪，修复配置模板/分段加载等能力缺失
  - `paper/runner.py`：恢复 `evaluate_holding_period_actions`/`_calc_holding_days`；“排除已持仓”逻辑修复为无条件生效
  - `paper/broker.py`：修复 `extension_mode` 引用残留
  - `ml/model_registry.py`：恢复 `get_latest_version()` 的 registry 尾部回退路径（避免全量加载）
  - `paper/reporting.py`：移除已删除字段 `ect_exposure`/`ect_reason` 的生产残留引用
- **树莓派 LCD35 无效价回退昨收逻辑恢复**：`_normalize_cycle_price` 实时价无效时回退昨收（昨收缺失时仍允许现价）；`_compute_holdings_intraday_pct` 移除无效价提前跳过，统一由 `_normalize_intraday_price` 回退处理

### Removed

- 删除重构后判明无用的废弃代码：runner 持仓奖励/置信度门控残留、`_reset_holding_anchor_for_kept_positions`、engine `_extend_holding_period`、runtime 止损状态中的死字段 `position_high_prices`、`train_ml_model.py` 的 `if False` 死块及未用 import
- 删除废弃测试项（约 20 个，涉及 signal gate、holding_bonus、ECT、holding_tail、已删除的 `3.5LCD_disp.py` 兼容入口等），重写多个过期断言测试（刷新文案、刷新间隔、配置模板）

### Tests

- 全量测试套件 949 个用例全部通过；新增 trading 共享核心、runtime 工作流、持有期对齐等回归测试；测试 stub 补齐 `PaperStorage` 的 `smb_reader` 参数

## [0.85.24] - 2026-07-28

### Fixed

- **风控因子预计算告警抑制**：新股上市前的日期在宽矩阵中全为 NaN，rolling 运算会触发 numpy 的 `All-NaN slice encountered` RuntimeWarning（结果本身是正确的 NaN，属预期行为），现已在 `precompute_risk_factors` 内定向抑制；新增后期上市股票场景的无告警回归测试

## [0.85.23] - 2026-07-28

### Performance

- **风控因子批量预计算**：新增 `src/lazybull/risk/precompute.py`，将 22 个基于 daily_adj 历史窗口的风控因子（A 类下行风险 8 个、B 类波动结构 6 个、D 类流动性 8 个）改为全周期一次性宽矩阵 rolling 向量化计算，替代原先每交易日的「全量切片 + groupby.tail + pivot + 逐股 Python 循环」模式。实测 2012-2026 全量数据预计算仅需约 80 秒（一次性），`build_clean_features` 整体耗时从 6+ 小时回落至约 2 小时。
  - `FeatureBuilder` 新增 `_risk_factor_cache_dict` 缓存槽位（与技术因子缓存模式一致），首次构建时预计算，之后每日 O(1) 查表合并；预计算失败时自动回退旧的逐日滑窗路径
  - 9 个公告类截面因子（pledge/unlock/block/short）不依赖历史窗口，仍逐日计算
  - `compute_all_risk_factors` 新增 `exclude` 参数，跳过已由预计算提供的因子
  - 语义说明：预计算窗口按「最近 N 个交易日」对齐（停牌日按 min_periods 跳过），而非原逐日路径的「该股最近 N 条观测」；对无停牌股票两者完全一致（17 个因子数值精确对齐，已有测试覆盖）

### Fixed

- **并行构建路径补齐风控因子**：`build_features_for_day_static`（`--parallel` 路径）此前完全缺失风控因子步骤，导致串行/并行产出的 cs_train schema 不一致；现已接入预计算缓存查表 + 公告类因子逐日计算，与串行路径保持一致

### Tests

- 新增 `tests/test_risk_precompute.py`（12 个用例）：预计算与逐日路径数值一致性、缺列降级、缓存复用、分位因子取值范围、exclude 参数、builder 集成等

## [0.85.22] - 2026-07-27

### Removed

- **移除盈亏动态持仓功能**：删除 `src/lazybull/backtest/holding_strength.py` 及所有相关代码、配置开关、测试。包括：
  - `enable_profit_based_holding` 总开关及所有子功能（亏损提前换出、盈利延续持有、ATR 动态止损、时间止损、strength_veto 二次确认）
  - 所有相关 dataclass 字段、CLI 参数、backtest engine 初始化参数
  - 纸面交易中的 `evaluate_profit_extension`、`evaluate_early_exit`、`_check_early_exit` 等方法
  - `batch_walk_forward.ps1` 中的全部分段配置
  - `compare_walk_forward.py` 中的中文标签映射和参数列表
  - `test_holding_strength.py` 及所有测试文件中的相关引用

- **移除整体持仓止盈功能**：删除 `take_profit_threshold`、`take_profit_refill` 及相关逻辑（整体止盈检查、止盈补位、元数据处理）

## [0.85.21] - 2026-07-27

### Removed

- **移除风险惩罚(Bad-Pick)功能**：删除 `src/lazybull/risk/bad_pick.py` 及所有相关代码、配置开关、测试。该功能在实际使用中效果不佳，简化核心架构。移除内容包括：
  - `BadPickConfig`、`RegimeBadPickConfig`、`apply_conditional_penalty`、`detect_market_regime` 等核心类/函数
  - `learn_risk_penalty_config` 训练函数及 `_apply_risk_penalty` / `_apply_risk_penalty_scores` 推理函数
  - `BAD_PICK_CLASSIFIER_FEATURES`、`MARKET_STATE_FEATURES`、`RISK_PENALTY_DEFAULT_LAMBDA_GRID` 等常量
  - `model_registry.py` 中的分类器内嵌逻辑
  - `walk_forward.py` 和 `train_ml_model.py` 中的风险惩罚学习/评估/参数定义
  - `batch_walk_forward.ps1` 中的风险惩罚配置/参数拼接/扫描循环
  - `compare_walk_forward.py` 中的风险惩罚指标列
  - `test_bad_pick_conditional.py` 及 `test_ml_signal.py`/`test_train_core_val_embargo.py`/`test_walk_forward.py` 中的相关测试

## [0.85.20] - 2026-07-26

### Changed

- **Bad-Pick 分类器特征恢复**：将 `BAD_PICK_CLASSIFIER_FEATURES` 恢复为此前 21 因子版本（波动/量价、成交额/振幅/布林、技术形态、动量/反转、开盘/资金、估值/行为），替换 v0.85.18–v0.85.19 引入的 20 因子重构版本。同步更新测试中 `kdj_d` → `kdj_j`。

## [0.85.19] - 2026-07-25

### Changed

- **Bad-Pick 去变体优化**：移除 3 个与主模型 zscore 版信息重复的原始值变体（`pb`/`zscore_bp`、`dv_ttm`/`zscore_dv_ttm`、`turnover_rate`/`zscore_turnover_rate`），新增 3 个真正独立的维度：`macd_dea`（补全 MACD 金叉/死叉判断）、`amount_ma5`（5日流动性枯竭预警，主模型只用20日）、`vol_ratio_5`（5日原始量比，区别于主模型 vol_ratio_20）。因子总数保持 20，信号维度从 6 类扩展为 7 类（新增流动性维度）。

## [0.85.18] - 2026-07-25

### Changed

- **Bad-Pick 因子重构**：从 18 个调整为 20 个，按 6 个独立信号维度重新组织。移除 5 个同质化/稀疏因子（`alpha_industry_10`、`ind_momentum_rank`、`margin_net_buy_ratio`、`lg_net_amount`、`vol_burst_10`），新增 7 个覆盖新维度的因子：`ret_5`（超短期反转）、`ma_deviation_5`/`ma_deviation_10`（均值回归）、`vol_burst_5`（5日量能异动）、`pb`/`dv_ttm`/`ep_ttm`（绝对估值锚定，区别于主模型的行业中性版本）。

## [0.85.17] - 2026-07-25

### Added

- **Bad-Pick AUC 阈值可配置**：新增 `--risk-penalty-clf-auc-threshold` 参数（walk_forward.py）和 `$risk_penalty_clf_auc_threshold_list` 批量变量（batch_walk_forward.ps1），默认 0.55。`learn_risk_penalty_config()` 新增 `clf_auc_threshold` 参数替代硬编码。

## [0.85.16] - 2026-07-25

### Fixed

- **Bad-Pick regime 样本门槛过高导致搜索被跳过**：`min_regime_samples` 从 `max(200, min_total_samples // 4)` 降为 `max(50, min_total_samples // 8)`，并新增全局兜底——当所有 regime 都不满足门槛时，回退到全量校准集做单网格搜索。修复小校准集（如 260 样本）下 threshold/lambda 恒为 (1.0, 0.0) 的问题。

## [0.85.15] - 2026-07-25

### Changed

- **Bad-Pick 分类器因子调优**：移除 `fund_hold_ratio`、`fund_hold_ratio_chg`（在 factor_exclude_list 中，ICIR 或覆盖率不达标）；新增 `margin_net_buy_ratio`（融资行为）、`weight_avg_bias`（筹码成本）、`turnover_rate`（原始换手率）、`vol_burst_10`（10日爆量），均不在主模型且不在排除列表。因子总数 16→18，非主模型因子 10→12。

## [0.85.14] - 2026-07-25

### Changed

- **Bad-Pick 因子按候选池实证重新筛选**：复现最新 split 的 Top150 候选池，并按覆盖率、单因子坏票分离度和三个时间段方向稳定性筛选。分类器特征由 27 个调整为 16 个，其中 10 个不在最新主模型中；移除在近期模型中始终未被使用的 3 个一致预期因子和 `zscore_fcf_yield`，以及方向翻转、低覆盖或与主模型重复度较高的因子。
- **Bad-Pick 改为严格样本外校准**：候选日期按 70%/10%/20% 拆分为训练、early-stop、calibration；AUC、regime 和惩罚参数只在最后 20% 日期上计算，随后按选定树数使用全部候选样本重训部署分类器。样本外 AUC 启用门槛调整为 0.55，并继续要求 TopK 中位数或 RankIC IR 改善。

### Fixed

- 修复旧版在分类器训练全量样本上计算 AUC、并在同批样本搜索惩罚参数导致的校准过拟合。近期记录中训练内 AUC 可达 0.80，但测试集 `swap_alpha` 仍为负，现改为严格时间留出评估。

## [0.85.13] - 2026-07-25

### Changed

- **Bad-Pick 特征回退至 32 因子版**：从 v0.85.12 的 20 特征精简版回退到 v0.85.10 的 27+5=32 特征版，保留全部候选因子供重新评估。

## [0.85.12] - 2026-07-25

### Changed

- **Bad-Pick 特征精简回退**：v0.85.8 过度追求占比目标，引入了多个有问题的因子导致分类器效果退化。
  - 移除 7 个：`mkt_ma250_ratio`/`mkt_turnover_ratio`（同日全市场相同值，截面零区分力）、`kdj_d`（与 kdj_j 相关系数>0.95）、`ps_ttm`/`net_mf_amount_mean_5`（原始量纲与 zscore 混合）、`zscore_cons_eps_dispersion_chg`/`zscore_cons_analyst_count_chg`（短窗口内稀疏）。
  - 保留 14 个核心 + 6 个新增 = 20 特征（+5 MARKET_STATE = 25）。
  - 设计原则改为"每个因子必须有明确的截面区分力"。

## [0.85.11] - 2026-07-25

### Fixed

- **Bad-Pick 空切片 NaN 填充警告消除**：新增因子在部分截面可能全列为 NaN，`Series.median()` 在空切片上触发 `RuntimeWarning`。修复：填充前先 `dropna()` 判断有效值数量，全 NaN 列直接填 0.0。覆盖 `prepare_classifier_features` 和 `learn_conditional_bad_pick_config` 两处。

## [0.85.10] - 2026-07-25

### Fixed

- **Bad-Pick 特征列表与 MARKET_STATE_FEATURES 去重**：`mkt_drawdown_20` 同时出现在 `BAD_PICK_CLASSIFIER_FEATURES`（v0.85.8 新增）和 `MARKET_STATE_FEATURES`（原有）中，导致训练时两列表拼接后列名重复，`X_clf[col]` 返回 DataFrame 而非 Series，触发 `ValueError: The truth value of a Series is ambiguous`。修复：从 `BAD_PICK_CLASSIFIER_FEATURES` 移除 `mkt_drawdown_20`（仍通过 MARKET_STATE 传入），并在 `learn_conditional_bad_pick_config` 加入 `dict.fromkeys` 去重防护。

## [0.85.9] - 2026-07-25

### Fixed

- **Bad-Pick 分类器实际可用特征数修复**：`prepare_training_data` 的内存优化裁剪掉了不在主模型 `feature_columns` 中的列，导致坏票分类器的 14 个新因子（`atr_pct_14`、`body_length`、`ps_ttm` 等）在训练时被丢弃，实际仅剩 15 个。修复后在 `needed_cols` 中显式保留 `BAD_PICK_CLASSIFIER_FEATURES` + `MARKET_STATE_FEATURES`。

## [0.85.8] - 2026-07-24

### Changed

- **Bad-Pick 分类器特征 v2 优化**：实现 50% 因子不在主模型中，确保惩罚信号与排序信号正交。
  - 移除 8 个冗余/低增量因子：`vol_ratio_20`（与 vol_burst_20 冗余）、`spec_score`（vol×size 衍生）、`rsi_14`（标准指标）、`zscore_acceleration`（动量饱和）、`zscore_opening_strength`（正交性弱）、`winner_rate`（主模型权重 0.007）、`zscore_or_yoy`（netprofit_yoy 更直接）、`zscore_quick_ratio`（debt_to_assets 已覆盖）。
  - 新增 13 个主模型未使用的因子：
    - 日内风险：`atr_pct_14`（ATR日内波幅）、`body_length`（K线实体，主模型明确排除）
    - 技术确认：`kdj_d`（KDJ慢线，主模型只用J值）、`net_mf_amount_mean_5`（5日平均资金流，主模型排除）
    - 市场环境：`mkt_ma250_ratio`（长期牛熊）、`mkt_drawdown_20`（回撤深度）、`mkt_turnover_ratio`（市场拥挤度）
    - 机构博弈：`fund_count_chg`（基金撤离）、`fund_hold_ratio_chg`（机构减持）
    - 价值陷阱：`ps_ttm`（市销率，主模型排除。高P/S+低P/E=暂时性盈利膨胀）
    - 一致预期修正：`zscore_cons_eps_dispersion_chg`（分歧度恶化）、`zscore_cons_analyst_count_chg`（分析师撤退）
    - 现金质量：`zscore_fcf_yield`（自由现金流，价值陷阱识别）
  - 注：`weight_avg_bias` 因数据中缺少 `close_adj` 列而不可用，替换为 `ps_ttm`。
  - 净变化：23 → 28 个特征，主模型内/外比例 14:14 = 50%:50%。

## [0.85.7] - 2026-07-24

### Changed

- **Bad-Pick 分类器特征优化**：减少与主模型高权重因子的重叠，增强风险维度的正交性。
  - 移除 7 个主模型高权重因子（`zscore_ma_deviation_20` 0.031、`lg_net_amount_sum_5` 0.022、`zscore_turnover_rate` 0.018、`zscore_amount_ma20` 0.015、`zscore_bb_width` 0.012、`amplitude` 0.011、`zscore_volatility_5` 0.010），这些因子与主模型排序信号高度相关，导致惩罚冗余。
  - 新增 8 个主模型低权重/未覆盖的风险维度因子：基本面恶化（`zscore_roe_dt`、`zscore_or_yoy`、`zscore_netprofit_yoy`）、杠杆与流动性（`zscore_debt_to_assets`、`zscore_quick_ratio`）、现金质量（`zscore_cf_nm`）、资金博弈（`zscore_order_imbalance`、`holder_num_chg`）。
  - 净变化：22 → 23 个特征，覆盖更全面的"质量陷阱"与"尾部风险"信号。

## [0.85.6] - 2026-07-24

### Fixed

- **批量汇总年化收益率与 broker 日志完全对齐**：`scripts/batch/batch_paper_trade.ps1` 的 `Get-NavSummary` 不再从 `nav.parquet` 重新计算（其价格来源与 broker 不同），改为直接读取 `config.yaml`（`initial_capital`/`account_start_date`）+ `account.json`（`cash`/`positions`）+ clean daily 收盘价，与 broker 的 `print_positions_summary` → `_calculate_annualized_return` 使用完全相同的数据源和 CAGR 公式。

## [0.85.5] - 2026-07-23

### Added

- **风险惩罚训练参数接入 batch_walk_forward.ps1 并打印**：
  - `scripts/walk_forward.py` 新增 10 个 CLI 参数：
    - 校准行为：`--risk-penalty-candidate-topk`、`--risk-penalty-bad-bottom-pct`、`--risk-penalty-min-bad-samples`、`--risk-penalty-min-total-samples`
    - 分类器超参：`--risk-penalty-clf-max-depth`、`--risk-penalty-clf-n-estimators`、`--risk-penalty-clf-learning-rate`、`--risk-penalty-clf-subsample`、`--risk-penalty-clf-colsample-bytree`、`--risk-penalty-clf-early-stopping-rounds`
  - 在 split 训练与部署训练两处 `learn_risk_penalty_config()` 调用中透传上述 10 个参数。
  - `learn_risk_penalty_config()` 签名新增 6 个分类器超参（`clf_max_depth`/`clf_n_estimators`/`clf_learning_rate`/`clf_subsample`/`clf_colsample_bytree`/`clf_early_stopping_rounds`），替代原硬编码常量。
  - `_log_risk_penalty_params()` 扩展打印，含分类器超参。
  - walk-forward 汇总 CSV 新增对应 10 列。
- **batch_walk_forward.ps1 接入**：
  - 配置区新增 6 个分类器超参列表变量。
  - 自动加入笛卡尔积遍历（+6 层 foreach）与命令行构建，参与总任务数统计。

## [0.85.4] - 2026-07-23

### Changed

- **walk-forward 风险惩罚强度支持可配调节（lambda scale/grid）**：
  - `scripts/walk_forward.py` 新增参数 `--risk-penalty-lambda-scale` 与 `--risk-penalty-lambda-grid`。
  - 在 split 训练与部署训练路径中，`learn_risk_penalty_config()` 现透传 `lambda_grid`，支持更温和/更激进的惩罚强度探索。
  - 当未显式提供 grid 且 `scale != 1.0` 时，自动按默认网格缩放生成候选，保持原默认行为可兼容。

### Added

- **walk-forward 汇总新增风险惩罚效果诊断列**：
  - `risk_penalty_penalized_ratio`（惩罚覆盖率）
  - `risk_penalty_penalty_mean`（惩罚均值）
  - `risk_penalty_topk_changed_days_ratio`（TopN变更日占比）
  - `risk_penalty_swap_alpha`（替换收益贡献）
- **批量对比报表接入风险惩罚效果指标**：
  - `scripts/compare_walk_forward.py` 已支持上述诊断列的聚合、中文映射与主表展示。
- **单元测试补充**：
  - `tests/test_walk_forward.py` 新增 lambda 网格解析与惩罚效果指标计算测试。

## [0.85.3] - 2026-07-23

### Fixed

- **修复 conditional bad-pick 在 walk-forward 评估侧的特征名不一致导致惩罚静默失效**：
  - `scripts/walk_forward.py` 的 `_apply_risk_penalty_scores()` 在 v2 路径中，现会按分类器 `feature_names_in_` 对 `X_clf` 做列对齐（补缺失列、剔除多余列、重排列顺序）。
  - 修复了分类器训练列不含 `mkt_drawdown_20` 但评估输入包含该列时 `predict_proba` 抛错并被静默回退到 `pred_score` 的问题。
  - v2 预测异常日志补充了 `bad_pick_model_version` 与具体异常信息，便于定位线上回退原因。
  - 新增 `tests/test_walk_forward.py` 回归测试，覆盖“输入含额外市场特征列仍能成功应用惩罚”的场景。

## [0.85.2] - 2026-07-23

### Fixed

- **修复 conditional bad-pick 在 walk-forward 评估侧的分类器兜底加载不可达分支**：
  - `scripts/walk_forward.py` 的 `_apply_risk_penalty_scores()` 原先在 `_clf_model` 缺失时会提前返回，导致后续按 `bad_pick_model_version` 兜底加载分类器的逻辑永远无法执行。
  - 现已调整为：先尝试按模型版本恢复分类器，兜底失败后再跳过惩罚，并输出明确失败原因日志。
  - 新增 `tests/test_walk_forward.py` 回归测试，覆盖“_clf_model 缺失 + bad_pick_model_version 兜底”路径。

## [0.85.1] - 2026-07-23

### Fixed

- **年化收益率统一为 CAGR 公式**：`src/lazybull/paper/broker.py` 的 `_calculate_annualized_return()` 从简单线性年化改为复合年化（CAGR），与批量汇总脚本一致。
- **批量纸面交易汇总结束日期修复**：`scripts/batch/batch_paper_trade.ps1` 的 `Get-NavSummary` 现在使用实际最终交易日（`$finalTradeDate`）而非 `nav.parquet` 末行日期，确保年化收益率计算覆盖完整的实际交易区间。

## [0.85.0] - 2026-07-23

### Changed

- **条件式 Bad-Pick 模型（完全替换线性惩罚）**：
  - src/lazybull/risk/bad_pick.py（新建）：BadPickConfig/RegimeBadPickConfig, detect_market_regime() 三层OR判断, apply_conditional_penalty() 阈值门控扣分。
  - src/lazybull/ml/train_core.py：learn_risk_penalty_config() 重写为训练XGB二分类器+分位数网格校准regime阈值+per-regime二维网格搜索。
  - src/lazybull/signals/ml_signal.py：_apply_risk_penalty() 支持v1/v2双模式，_load_model() 自动加载坏票分类器。
  - src/lazybull/ml/model_registry.py：register_model() 自动注册坏票分类器并回填版本号。
  - scripts/walk_forward.py 和 scripts/train_ml_model.py：日志与评估逻辑同步适配v2。
- **分类器特征从15个扩展到22个**：新增主力资金、动量衰竭、日内形态、极端估值、分析师分歧等维度。
- **所有阈值用分位数定义**：每个walk-forward split独立校准，自动适应市场数据分布进化。

### Added

- 	ests/test_bad_pick_conditional.py：21个测试覆盖配置序列化、regime检测、特征提取、门控逻辑、边界条件。

## [0.84.0] - 2026-07-22
