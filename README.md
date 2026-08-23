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

### 当前版本 (v0.95.9)

- **高缺失因子审计修复**：一致预期目标价在上下界仅一边有效时保留可用值，基础一致预期与修正因子两条链路口径一致；现金流状态改为报告期优先，晚发旧期更正不再覆盖最新财报；`fcf_yield` 将 `total_mv` 从万元换算为元后再计算；基本面与现金流同时启用时，训练入口自动移除 `zscore_cf_nm` 及其 `_sz` 别名，避免与 `zscore_ocf_to_profit` 确定性重复入模。
- **稀疏性结论**：`express_revenue_yoy` 依赖去年同期快报，`express_surprise` 依赖同报告期且先于快报披露的业绩预告，`cons_eps_revision_30d` 依赖相邻两个 30 日窗口，均保留真实 NaN，不做填零或跨期伪造。目标价/现金流修复需重建特征，训练去重需重新训练后生效。

### v0.95.8

- **现金流质量四列全空修复**：资本开支字段改用 TuShare 官方 `c_pay_acq_const_fiolta`，恢复 `zscore_capex_to_ocf` / `zscore_fcf_yield` 及对应市值中性化 `_sz` 列；客户端默认下载字段同步修正。现有 raw/cashflow 可直接复用，但需重建特征并重新训练。

### v0.95.7

- **亏损因子语义修复**：`is_loss` 对齐 TuShare“亏损公司的 `pe_ttm` 为空”契约，仅将当天存在 `daily_basic` 记录的空 PE 标记为亏损，避免把左连接未命中的数据缺失误判为亏损；兼容其他数据源返回的非正 PE。存量特征需重建并重新训练后生效。

### v0.95.6

- **业绩快报因子全链路修复**：同日多公告稳定排序（mergesort，稳定选报告期最新快报）；PIT 查询启用报告期优先（`end_col="end_date"`），晚发旧期更正公告不再回退因子值；接口缺列时输出全 NaN 列不崩溃；`express_surprise` 缺 forecast 时离线构建显式告警；express 由单文件迁移为按季度分区存储（loader/ensure 首次访问自动迁移旧单文件，增量路由写分区，`raw_download` 同步分区化）；纸面链路统一检查 1000 条最低记录门槛与同步水位，强制重建出现季度失败或重建后仍不足时明确报错；并补检 surprise/roe 缓存列。生效说明：升级后首次跑纸面/离线特征构建时旧单文件自动迁移，无需手工处理。

### v0.95.5

- **基金持仓因子全链路修复**：离线 `download_by_period` 分区模式落盘前去重生效（同报告期两批公告不再双重计数 `fund_hold_ratio`）；paper 端落盘去重与离线口径一致；paper 端披露季刷新门控（距报告期末 < 4 个月按最新公告日覆盖水位强制重下并覆盖 `fund_portfolio_agg` 缓存，消除部分快照永久冻结）；离线 fund 回溯窗口独立扩至 18 个月（chg 不再因 7 个月 warmup 整批 NaN）；下载失败日志升级 `warning`。生效说明：需重新下载 `fund_portfolio` 并重建特征分区后生效。

### v0.95.4

- **筹码胜率因子链路四项修复**：`weight_avg_bias` 偏离度改用未复权收盘价与未复权成本价同口径计算（此前因特征截面无 `close_adj` 从未产出，修复后正式入模，需重建特征+重训练）；`winner_rate_chg_5/20` 改为交易日历对齐（缺数据日不再静默跨期）；lookup 构建移除逐行 `iterrows` 全面向量化；纸面链路 `cyq_perf` 下载失败日志升级为 `warning` 不再静默。

### v0.95.3

- **训练入口特征清洗日志输出移除列明细**：训练启动时按类逐行打印被移除特征列（高缺失/全空/常数/zscore 联动）的详细名称，便于定位数据链路问题；仅日志增强，不改变特征移除逻辑与训练结果。

### v0.95.2

- **融资融券幽灵因子清理**：从未实现的 `margin_net_buy_ratio` 从主模型列清单移除，正式入模融资融券因子为 `rzye_chg_5` / `rzye_chg_20` / `rqye_rzye_ratio`；风控专用列（`margin_net_buy`、`short_balance_change_5`、`short_sell_vol_change_5`）独立为 `MARGIN_RISK_COLS` 继续供 PositionRiskModel 使用；
- **cs_infer 缓存补检**：缓存完整性校验新增 `short_balance_change_5`，旧推理缓存缺列自动重建，消除风控模型 train/infer 列差异。

### v0.95.1

- **事件衰减推理侧补齐**：`MLSignal` 与 OOS 评估按模型训练参数复现事件型 freshness 指数衰减，消除 train/serve skew（旧公告不再以原值全额入模）；
- **股东人数环比精确对齐**：`holder_num_chg` 基准改为公告日不晚于本版本、报告期早于本版本的最新已公告值，同报告期修正版本不再稀释跨期信号；
- **业绩预告类型语义修正**：未知/缺失预告类型保留 NaN（与"不确定"评分 0 区分）；公告 PIT 查询支持报告期优先，晚发旧期修正不覆盖新期预告。

### v0.95.0

- **龙虎榜连续异动信号**：新增 `lhb_cont_on_list` / `lhb_cont_up_days_5` / `lhb_cont_up_days_20` 事件级信号，区分连续异动与单日榜。

### v0.94.27

- **adj_factor 强制依赖**：批量下载和在线 ensure 遇到空响应时明确失败，clean 构建不再缓存全 NaN 复权价分区；
- **坏缓存自动修复**：已有 clean/daily 的 close_adj 全空时自动失效并重建；局部因子缺失保持 NaN，不再跨日沿用昨日因子。

### v0.94.26

- **daily 唯一代码完整性校验**：落盘前严格校验 `(ts_code, trade_date)` 主键并按唯一代码计数，重复、空代码和非目标日期数据直接失败；
- **历史停牌日交叉确认**：stock_basic 覆盖率粗筛偏低时，以 daily_basic 独立代码域二次确认并容忍经全量历史数据校准的 2% 接口差异，既拦截部分返回，也不误伤 2006 年和 2015 年大面积停牌日。

### v0.94.25

- **daily 完整性严格闭环**：新下载 daily 先验证覆盖度再落盘，覆盖率低于 85% 不落盘并返回 False；历史低覆盖重下后仍异常（重下空/仍低）也返回 False，不再告警放行；
- **覆盖度基准按上市日期过滤**：分母取 list_date <= 当日 的股票数，不再用当前全集误伤历史日期。

### v0.94.24

- **daily 覆盖度闭环**：历史 daily 覆盖度低于 stock_basic 全集的 85% 时强制重下一次，重下后仍低则 error 告警待人工核实（不再仅告警），历史截断/服务端部分返回可被检测并修复；
- **daily_basic 覆盖度门控恢复**：实测 daily_basic 与 daily 代码集合完全一致，恢复 `min_rows` 行数门控（仅 moneyflow 因缺北交所不设此门控）。

### v0.94.23

- **moneyflow 覆盖度门控修正**：moneyflow 天然不覆盖全部 daily 股票（如不含北交所 920xxx.BJ），不再以 daily 行数为下限（避免每次 ensure 重复下载）；`adj_factor`/`stk_limit` 保留行数门控；
- **KDJ 窗口不足掩码 NaN**：RSV 窗口有效观测不足时 kdj 输出为 NaN（内部状态可重置，但不给伪信号），与 volatility 等因子缺失语义一致；
- **daily 自身覆盖度告警**：以 stock_basic 全集为参照，daily 行数低于 70% 时 error 告警（仅告警不自动重下；0.94.24 已升级为强制重下）。

### v0.94.22

- **停牌窗口修复补全**：`ret_N`/`vol_ratio_N`/`ma_deviation_N`/`amount_maN` 在窗口内观测不足（停牌）时置 NaN；KDJ RSV 前向填充限长（`ffill(limit=3)`），长期停牌后重置而非僵化跳变；
- **daily 自身分页**：`get_daily` 全市场查询自动分页，堵住 daily 单次 6000 上限截断源头；
- **ensure 区分非交易日与接口故障**：依据缓存 trade_cal，非交易日正常跳过（True），交易日接口空或无法确认时返回 False，不再误报成功；
- **suspend 空值占位**：当日无停牌时写占位空文件，避免重复请求。

### v0.94.21

- **raw ensure 覆盖度门控**：`is_data_exists` 新增 `min_rows` 参数（pyarrow 快速读行数），以当日 daily 行数为参照对 `adj_factor`/`stk_limit`/`moneyflow`/`daily_basic` 做覆盖度检查，文件存在但行数不足视为未补齐并重新下载；
- **技术指标/波动率滚动窗口按交易日对齐**：`precompute_technical_factors` 支持 `trading_dates` 参数，停牌日补 NaN 行占位，长期停牌股复牌后 RSI/KDJ/MACD/BOLL/波动率/ATR 窗口不再按行数凑满；`compute_ret_1` 停牌缺口收益保持 NaN。

### v0.94.20

- **raw ensure 补齐解耦**：`adj_factor`/`suspend`/`stk_limit` 改为独立存在性检查，不再绑定 daily 缺失分支，防止单类数据缺失永久无法补齐；日线为空（非交易日）提前返回并告警，`daily_basic` 为空新增告警；
- **全市场查询自动分页**：`daily_basic`/`moneyflow`/`stk_limit` 未指定 `ts_code` 时自动分页取全，避免单次 6000 上限静默截断；
- **复权因子回填语义修正**：去掉 `bfill` 仅保留 `ffill`（累积因子前向填充），避免跨除权除息日回填污染复权价与标签；
- **daily_basic 单日整体缺失升级为 error 硬告警**：价值红利核心信号（bp/ep_ttm/dv_ttm/市值/换手）全空时明确可见。

### v0.94.19

- **门控 docstring 语义与实现对齐**：覆盖水位描述更新为"有水位以水位为准，无水位用数据最新日期初始化"；
- **同步水位写入使用唯一临时文件**：`tempfile.mkstemp` + `os.replace`，避免多进程并发同步同一数据集时互相覆盖。

### v0.94.18

- **门控覆盖判断与增量函数语义对齐**：有同步水位（连续成功前缀）时，门控覆盖判断只认水位，数据最新公告日不得越过水位后的未知区间（避免失败日被后续成功数据掩盖而永久漏掉）；
- **同步水位写入改为原子替换**：临时文件 + `os.replace`，崩溃时水位文件保持不变，只会触发安全重查。

### v0.94.17

- **同步水位语义修正（水位=连续成功前缀）**：起点改为"有水位则从水位之后开始"，区间内首个失败立即停止，水位只推进到最后一个成功日，失败日不被后续成功数据跨过；先落盘新数据、成功后才原子推进水位（落盘失败水位不提交）；数据无有效日期列时即使水位高也判定缺口（防止 parquet 被删/损坏后静默跳过重建）。

### v0.94.16

- **增量补齐引入持久化同步水位**：独立记录"已成功查询至日期"（无公告日也算已同步），仅在区间无失败时推进；门控用"有效水位 = max(数据最新公告日, 同步水位)"判断覆盖，空白日期不再被反复下载；
- **窗口外补行契约补全**：目标窗口内无分区时也从窗口前分区补充最新公告；移除"连续 4 空分区截断"启发式，完整遍历确保找到更早的有效股票；
- **update_flag 闭环补全**：`_FINA_REQUIRED_RAW_COLS` 补 `update_flag` 触发旧分区回补；批量分区保存接入修正版优先去重；仅显式识别 `update_flag == "1"` 为修正记录；
- **cf_nm 依赖告警去重**：缺 `ocf_to_profit` 时按构建会话只提示一次；
- **清理未使用导入**（`factor_load.py` 的 `_MIN_*` 等，F401）。

### v0.94.15

- **修复公告型基本面因子增量补齐"死路"**：门控从"记录数量充足"改为"最新公告日是否覆盖目标交易日"判断缺口，fina_indicator/cashflow/stk_holdernumber/forecast/express/report_rc 六类自动补齐真正生效，不再依赖人工全量下载；
- **fina_indicator/cashflow 增量写入季度分区**：分区读取 + 增量按 `end_date` 路由写对应季度分区（对齐 forecast/report_rc），消除"写单文件被分区遮蔽"；schema 回补结果同样写回分区；
- **季度窗口外股票保留"旧值 + 大 freshness"**：窗口截断外的最新公告从窗口前分区补充（`_load_pre_window_latest_rows`），不再硬缺失 NaN，消除 paper 与离线窗口起点不一致导致的 cs_train/cs_infer 漂移；
- **同日多公告稳定排序 + update_flag 修正版优先**：公告排序改 `mergesort` 稳定排序，同日披露"更正+新季报"时 PIT 确定选中最新报告期；`FINA_INDICATOR_DEFAULT_FIELDS` 补充 `update_flag`，多次修订去重可复现；
- **cf_nm 回填依赖显式提示 / bot 盘中触发当日交易显式提示**：cashflow 因子未启用导致 `cf_nm` 全 NaN 时显式 warning；bot 盘中触发当日 `/trade` 提示"当日公告可能尚未全部发布"。

### v0.94.14

- **统一串行/并行特征构建的基本面代理回填顺序**：并行路径回填从因子处理器之前移到之后，与串行一致（`ocf_to_revenue→cf_sales`、`ocf_to_profit→cf_nm` 两条链不再失效，消除训练/推理口径漂移）；
- **统一离线/在线缺失复权因子处理**：`build_clean.py` 不再伪造默认值 `1.0`，与在线 `ensure.py` 一致保留 `NaN` 交由清洗层处理；
- **缺失 dv_ttm/pe_ttm 不再编码成真实经济含义**：保留 `NaN`，新增 `dv_ttm_missing`/`pe_ttm_missing` 显式标记，`is_loss` 仅对已知亏损为 1；标记以可选方式加入训练候选特征（旧 schema 分区自动跳过，重训后模型可显式利用缺失状态）；
- **模型推理新增数值质量门禁**：`MLSignal` 预测前仅拒绝"全空"列（数据完全缺失）；全零/高缺失率（>50%）聚合为 WARNING 级警告不阻断，`mkt_*`/`north_*` 单日常量广播列为设计状态直接忽略，避免以截面分布一票否决合法状态。

---

### 历史版本 (v0.92.3)

- **修复纸面交易非调仓日缺数据不自动下载**：
  - 根因：`execute_trade_workflow` 仅在 T0 调仓日生成信号（`ensure_features_for_date`
    自动下载）或 T1 有指令时才触发数据下载；非调仓日/无指令时只做只读操作（止损
    检查、持仓打印），直接加载 clean 数据，缺失时止损仅跳过、持仓打印直接报错退出；
  - `src/lazybull/paper/runtime.py`：`execute_trade_workflow` 校正日期后主动调用
    `ensure_clean_data_for_date` 补齐当日 clean 数据（内部自动下载 raw），失败仅
    warning 不阻断主流程（保持降级语义）；
  - `src/lazybull/paper/reporting.py`：`load_position_snapshot` 加载前同样补齐当日
    clean 数据，避免 `positions` 查看/打印因缺数据崩溃；
  - 修复同时惠及钉钉机器人链路（`bot_service.py` 复用同一 runtime）；
  - **测试**：新增 `test_execute_trade_workflow_ensures_trade_date_clean_data`、
    `test_execute_trade_workflow_continues_when_clean_data_ensure_fails`、
    `test_load_position_snapshot_ensures_trade_date_clean_data`。

- **修复 walk-forward 多窗口集成子模型特征列不一致导致的训练失败**：
  - 多窗口（基础/前移/后移）集成时，以首个（基础窗口）子模型特征列为准，
    强制后续子模型通过 `feature_columns_override` 对齐，避免不同窗口高缺失
    门禁产生不一致的特征 schema（如 `express_revenue_yoy`）导致集成预测
    XGBoost `feature_names mismatch`；

- **一致预期（report_rc）因子恢复进入训练链路**：
  - `factor_exclude_list.json` 移除 20 个 `cons_*` / `zscore_cons_*` 排除项
    （53 → 33），因子精简不再拦截一致预期因子；
  - `max_feature_missing_ratio` 默认 0.4 → 0.6，`cons_analyst_count_30d` /
    `cons_eps_mean_fy0` / `cons_eps_mean_fy1` / `cons_eps_mean_fy2` /
    `cons_rating_score` 共 5 个因子实际进入训练；
  - `cons_eps_revision_30d` / `cons_target_price_mid` / `zscore_cons_*`
    因数据源缺失过高（82%+）仍被门禁剔除。

- **一致预期因子新增按 quarter 预测财年分组的 EPS 均值**：
  - `report_rc` 同一 `report_date` 含多个预测季度（如 2024Q4/2025Q4/2026Q4）；
  - 新增 `cons_eps_mean_fy0`（当前财年）/ `cons_eps_mean_fy2`（未来第二财年），
    `cons_eps_mean_fy1` 作为 FY1（未来第一财年）分组因子；
  - `cons_eps_revision_30d` 等其余因子保持原语义不变；
  - `quarter` 缺失时 EPS 财年列优雅降级为 NaN。

- **`download_raw` 启动时自动绕过终端/系统代理**：
  - PowerShell 等终端常通过环境变量注入 HTTP(S) 代理，导致 TuShare 请求走内网代理
    出现 `Read timed out`；
  - 脚本启动时在进程内清除 `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY`（含小写），
    只影响当前进程、不改终端设置；可用 `LAZYBULL_DOWNLOAD_BYPASS_PROXY=0` 关闭。

- **`forecast` / `report_rc` 改为按时间分区存储（告别超大独立文件）**：
  - `forecast` 按季度 `end_date` 分区，与 `fina_indicator`/`cashflow` 对齐；
    `report_rc` 按年 `report_date` 分区，与"按年下载/断点续传"节奏一致；
  - 离线下载：`forecast` 复用 `download_by_period(partition_by_period=True)`；
    `download_report_rc` 按年独立落盘（保留自适应二分 + 保守并发）；
  - 增量补齐：新增 `_append_and_save_partitioned` 按分区键路由写入，分区内去重，
    不再整文件读-合并-重写；
  - 加载：`load_forecast` / `load_report_rc` 纯分区加载（无旧单文件兜底）。

- **`report_rc` 下载告警清零 + 日志去噪**：
  - 二分合并的 concat 也改用 `_concat_no_warning`，所有 pandas empty/all-NA
    告警全部屏蔽（数据原样）；
  - 预期流程（确定性错误不重试、自动二分重试）日志降为 debug，只有真错误
    （如 `Read timed out`）保留 warning。

- **修复 `report_rc` 长期高并发被 TuShare 拒绝**：
  - 实测确认并发数量本身没问题（16 并发 320 请求全成功），根因是长期高请求
    频率 + 超限错误重试 3 次放大请求量；
  - 修复：确定性错误（"查询数据失败"）不重试、`report_rc` 接口级限频 200、
    并发降到 8。

- **下载 concat 只屏蔽告警、不再改动数据**：
  - `_query_with_pagination` 与 `_save_merged` 恢复数据**原样 concat**，
    不再剔除全 NA 行/全 NaN 列、不做 reindex 补列（避免破坏 raw 层 schema，
    导致训练用到列、预测时列被删）；
  - 仅通过 `_concat_no_warning` 按消息精确屏蔽 pandas 的 empty/all-NA
    entries FutureWarning。

- **彻底修复 `_query_with_pagination` concat FutureWarning（真正根源）**：
  - `report_rc` 改用保守并发（`_REPORT_RC_CONCURRENCY=8`），避免 48 并发
    打爆本地 HTTP 代理（`192.168.1.21:18081`）导致 `Read timed out`；
  - `_query_report_rc_adaptive` 只对"查询数据失败"超限错误二分，网络超时等
    其它错误直接抛出（不无意义递归），重跑断点续传；
  - `_run_concurrent` 新增 `max_workers` 参数，可按接口单独设置并发。

- **修复 `_query_with_pagination` concat FutureWarning**：
  - 翻页中"有行但全 NaN"的片段在 concat 前剔除，并保留全部列集合，
    消除 pandas FutureWarning 且 schema 不缺失（与 `_save_merged` 修复一致）。

- **修复 `top_list` 限频绕过配置导致限流**：
  - `get_top_list` 不再硬编码 `rate_limit_override`（它绕过接口级/全局令牌桶），
    改走 `_API_RATE_LIMITS_DEFAULT["top_list"]` 配置（默认 400，低于官方
    500 次/分钟留余量）；
  - 之前 v0.90.21 设 1000 在并发下超官方 500 限频触发大量限流，现已修复。

- **修复 `top_list` 下载限频笔误导致极慢**：
  - `get_top_list` 的 `rate_limit_override` 误写为 60（1 秒 1 请求），与注释
    意图 1000 次/分钟不符，5240 个交易日全量下载耗时约 88 分钟；
  - 实测 TuShare 该接口 60ms 间隔连续请求无限流，改为 1000 后配合按日并发，
    吞吐从 60 次/分钟提升到数百次/分钟，全量预计压缩到 ~10 分钟量级。

- **`report_rc` 下载并发化**：
  - `download_report_rc` 从按年串行改为按年并发下载，网络等待并行化；
  - 各年份独立下载（超限年份自动二分分片），全部完成后统一合并去重落盘；
  - 配合 `--concurrency` / `tushare.download_concurrency` 生效，22 年全量
    下载耗时显著下降。

- **修复 `report_rc` 单次查询超限导致整年下载失败**：
  - TuShare `report_rc` 接口单次查询（`start_date`/`end_date` + offset 翻页）
    总行数上限 100000 条，超限年份（2009 起多数年份 10~30 万条）翻页到
    offset > 100000 会返回"查询数据失败，请确认参数！"并整年失败；
  - `download_report_rc` 现通过 `_query_report_rc_adaptive` 自动二分日期范围
    重试，任意规模数据都能取全；未超限年份仍单次查询，零额外开销。

- **按季度下载并发化（fund_portfolio 等大幅提速）**：
  - `download_by_period`（`scripts/raw_download/periodic.py`）从纯串行逐季度
    下载改为复用 `_run_concurrent` 并行下载，受 `tushare.download_concurrency`
    并发数与 TushareClient 令牌桶限频约束；
  - 分区模式（`fund_portfolio`/`fina_indicator`）各季度独立落盘，天然线程安全；
    非分区模式（`forecast`/`express`）各季度下载完成后统一合并去重落盘；
  - 实测 `fund_portfolio` 全量 86 个季度从约 2 小时压缩到数分钟量级。

- **移除无效信号门控历史接口**：
  - 删除信号置信度门控、composite 门控、动态 Top-N 与持仓奖励 CLI 参数；
  - 这些参数未进入 `TradingConfig`、信号或回测引擎，历史上接受参数但不会改变运行结果；
  - 同步移除未接入回测引擎的 ECT 参数，风险模块中的独立实现不受影响；
  - walk-forward 汇总与实验对比不再输出对应空字段和幽灵指标。
  - 按切分数量反推的主链不再暴露无效 `--step`，batch 也不再生成仅步长不同的重复任务。

- **统一交易状态判断**：
  - 回测、选股与纸面交易 broker 共用停牌、涨停和跌停状态决策；
  - 纸面交易仍保留 `tradable` 买入过滤和停牌日历优先级。
  - 回测与纸面交易统一复用金额、价格到 A 股整手股数的计算函数。

- **Walk-forward 汇总模块拆分**：
  - split 指标整理、条件参数清洗与 summary CSV 写入已迁移到独立 ML 模块；
  - 单 split 的 OOS 数据准备、回测执行和绩效提取已迁移到独立回测模块；
  - 训练窗口构建、多偏移/多种子集成与种子筛选评分迁移到
    `src/lazybull/ml/walk_forward_training_core.py`；
  - OOS 重点面板与回测前模型摘要迁移到
    `src/lazybull/ml/walk_forward_training_reporting.py`；
  - split 与 deploy 训练执行入口分别迁移到
    `src/lazybull/ml/walk_forward_split_training.py` 与
    `src/lazybull/ml/walk_forward_deploy_training.py`；
  - `src/lazybull/ml/walk_forward_training.py` 保留为兼容门面重导出；
  - 普通训练和 walk-forward 共用训练运行记录构造器；
  - TopK 明细、成交归因和串联净值由独立报告模块负责；
  - 主脚本继续保留训练和部署编排职责。

- **Walk-forward CLI/Runner 拆分**：
  - 参数解析与校验迁移至 `src/lazybull/ml/walk_forward_cli.py`，提供 `build_walk_forward_parser()` 与 `parse_walk_forward_args(argv=None)`；
  - split 过滤与运行编排迁移至 `src/lazybull/ml/walk_forward_runner.py`，提供 `_filter_splits_by_selected_indices()` 与 `run_walk_forward(args)`；
  - `scripts/walk_forward.py` 调整为薄入口，保留历史导出符号兼容既有测试与外部调用。

- **回测主循环状态机边界拆分**：
  - `BacktestEngine.run` 原样迁移到 `src/lazybull/backtest/run_loop.py` 的 `BacktestRunLoopMixin.run`；
  - 每日 T0/T1 状态推进、早调仓回滚、signal_dates 修剪、finally 清理和统计输出顺序保持一致；
  - `src/lazybull/backtest/engine.py` 聚焦状态字段与执行组件，不改动既有交易算法。

- **回测信号执行边界拆分**：
  - `_build_signal_data`、`_post_filter_candidates`、`_get_position_weight_for_planning`、`_queue_condition_sell_refill_signal`、`_get_holding_features_row`、`_generate_signal`
    已原样迁移到 `src/lazybull/backtest/signal_execution.py` 的 `BacktestSignalExecutionMixin`；
  - `BacktestEngine` 与 `BacktestEngineML` 的既有覆写关系保持不变，行业约束继续采用延迟导入。

- **回测买入执行边界拆分**：
  - `_execute_pending_buys`、`_process_position_completion`、`_buy_stock_with_status_check`、`_build_position_extra_info`、`_buy_stock_direct`、`_buy_stock`、`_update_completion_attribution`
    已原样迁移到 `src/lazybull/backtest/buy_execution.py` 的 `BacktestBuyExecutionMixin`；
  - T1 候选顺位、未成交槽位、补齐窗口、旁路归因、整手股数、手续费、最小买入阈值与 pending order 行为保持不变。

- **回测卖出执行边界拆分**：
  - `_queue_rebalance_sells`、`_check_and_sell`、`_execute_pending_condition_sells`、`_check_stop_loss`、`_execute_pending_stop_loss_sells`、`_sell_stock`、`_sell_stock_with_status_check`、`_sell_stock_direct`
    已原样迁移到 `src/lazybull/backtest/sell_execution.py` 的 `BacktestSellExecutionMixin`；
  - 调仓卖出候选、持有期/盈利延续、T0 触发 T1 执行、止损去重、停牌/跌停延迟、开盘/收盘口径以及 PnL 与交易记录字段保持不变。

- **回测延迟订单执行边界拆分**：
  - `_record_pending_order_event`、`_process_pending_orders`
    已原样迁移到 `src/lazybull/backtest/pending_execution.py` 的 `BacktestPendingExecutionMixin`；
  - `PendingOrderManager` 在 `BacktestEngine.__init__` 中继续通过
    `event_sink=self._record_pending_order_event` 绑定；
  - 每日重试、可交易性检查、买卖分发与成功/过期/继续延迟状态保持不变。

- **回测报告与日志边界拆分**：
  - 调仓摘要 formatter 与日级日志/告警/信号汇总、决策 trace、进度日志等方法
    已原样迁移到 `src/lazybull/backtest/reporting.py` 的 `BacktestReportingMixin`；
  - `src/lazybull/backtest/engine.py` 通过导入重导出 `_format_rebalance_decision_summary`，
    兼容现有测试与外部引用。
  - 删除 `engine.py` 顶层未调用的 `_format_buy_execution_stock_list`、
    `_sum_buy_execution_weights`、`_format_buy_execution_summary` 死代码。

- **移除 best_iteration 自适应候选重训**：
  - `walk_forward.py` 不再支持 `--adaptive-best-iter-retrain` 与 `--adaptive-low-iter-max-retries`；
  - `batch_walk_forward.ps1` 已删除对应开关与参数透传；
  - 对比汇总不再输出相关实验字段，训练与汇总口径更简洁稳定。

- **行业中性与绝对收益混合标签**：
  - 通过 `--neutral-label-blend-weight` 在训练期动态混合 `neu_y_ret_N` 与 `y_ret_N`，无需重建特征；
  - `0` 保持纯行业中性标签，`0.25` 表示引入 25% 原始收益目标，`1` 等价于纯原始收益；
  - 混合发生在 `cs_zscore` 之前，walk-forward 的早停、rank-weight、验证和 OOS 标签指标统一使用混合目标；
  - `batch_walk_forward.ps1` 可用 `@(0.0, 0.25, 0.5)` 做真实重训快筛，不能在 skip-training 模式复用旧模型。

- **Walk-forward 信号到成交收益归因**：
  - OOS 回测按 split 自动导出 `walk_forward_trades_*` 与
    `walk_forward_execution_attribution_*` 两类明细；
  - 记录计划股、实际买入股、候选排名、替换/未成交原因及 T0 到 T1 价格变化；
  - 使用 `python scripts/ana/analyze_signal_execution_gap.py --raw-dir <raw目录> --wf-run-id <运行ID> --focus-splits 6,9`
    分析信号日 Top30、实际买入标签收益和真实持仓收益的转化差异。

- **独立因子裁剪实验清单**：
  - 单次训练和 walk-forward 支持 `--factor-exclude-file <JSON路径>`；
  - 未指定路径时仍使用生产 `data/models/factor_exclude_list.json`；
  - 候选实验可显式使用 `configs/factor_exclude_candidate_sparse_v1.json`，不会覆盖生产清单。

- **因子使用稳定性分析**：
  - 支持按模型版本区间展开 `EnsembleModel` 的全部子模型并聚合归一化 importance；
  - 输出严格低使用候选与待 IC 复核观察名单，不会自动修改生产因子排除清单；
  - 示例：`python scripts/ana/analyze_factor_stability.py --versions 22626-22639`。

- **分批调仓均匀排期**：
  - 批次数不能整除调仓周期时，按周期比例均匀分布各 tranche；
  - 20 日分 3 批使用 `0/7/13` 日偏移，避免旧排期 `0/6/12` 留下 8 日尾部空档。

- **分批调仓仓位修复**：
  - `top_n` 始终表示组合总持仓数，分批模式不再将目标持仓数放大 K 倍；
  - 总 TopN 按批次拆分槽位与资金，修复首批后其余批次因无可用槽位而无法买入的问题；
  - 纸面交易 T0 下发的 `desired_position_count` 统一使用总 `top_n`，避免第二批起被误判为“无可用空槽”；
  - 纸面交易 `_generate_signals` 在分批模式下改为优先使用调用参数 `top_n`（本批槽位数），不再被 `trading_config.top_n` 覆盖，避免首批按总 `top_n` 选股。
  - 支持 TopN 不能整除批次数的场景，各批预算按实际槽位占比分配。

- **多偏移集成验证口径修复**：
  - 集成验证统一使用所有保留子模型共同未见的 calibration 面板；
  - 面板起始日必须晚于全部子模型的训练及早停截止日，避免后移窗口污染验证集；
  - calibration 已参与子模型筛选时不再复用为独立验证集；
  - 无法证明验证面板无泄漏时不输出集成验证指标，OOS 与回测路径不受影响。

- **freshness 归因实验支持**：
  - 新增 `state_keep_event_no_decay` 可选策略，仅保留状态型 freshness，事件型 freshness 不入模且不衰减事件值；
  - 默认生产策略仍为 `state_keep_event_decay`，实验结束后无需删除临时代码；
  - walk-forward 汇总与对比表记录 freshness 策略及有效半衰期，避免依赖运行顺序人工追溯。

- **P2-C freshness 完整方案上线**：
  - 训练入口支持 `state_keep_event_decay`（默认）、`state_keep_event_no_decay` 与 `drop_all` 三种策略；
  - 状态型 freshness 保留，事件型 freshness 改为驱动对应事件因子做指数衰减（半衰期可配）；
  - 事件型 freshness 不再直接作为独立特征入模，降低“披露节律噪声”学习风险。

- **训练/滚动训练参数可控化**：
  - `train_ml_model.py` 与 `walk_forward.py` 新增 `--freshness-strategy`、`--event-freshness-half-life-days`；
  - `batch_walk_forward.ps1` 同步新增批量参数透传。

- **因子训练防污染第一阶段落地**：
  - 训练入口统一移除 `*freshness*` 特征；
  - 新增高缺失（默认 >40%）、全空、常数特征硬门禁；
  - `factor_prune` 增加 `zscore_*` 与 `zscore_*_sz` 联动剔除，防止派生列绕过过滤。

- **公告因子 PIT 可见性修复**：
  - `fundamental` / `cashflow_quality` / `earnings` / `holder` / `express` 改为保留同报告期多公告版本，由交易日 PIT 回放选择当日可见值；
  - `express_surprise` 仅使用 `forecast_ann_date <= express_ann_date` 的历史预告，修复前视污染。

- **未来统计泄露风险收敛**：
  - 停用 `fundamental` / `cashflow_quality` 中基于全历史样本分位数的 winsorize 截尾。

- **因子链路稳定性修复与约束落地**：
  - 并行/串行特征构建步骤完全对齐，修复并行路径缺失价值红利、资金流与基本面代理回填导致的 schema 漂移。
  - 涨跌停标记收敛为 cleaner 层唯一入口，features 层不再重算。
  - 日期字段契约统一为 YYYYMMDD 字符串，清理多处 `astype(str)` 产生字符串 `nan` 的隐患。
  - 因子处理器新增 ts_code 安全合并与失败 NaN 占位，避免重复键错配与静默丢列。

- **回测/纸面交易共享决策核心**：新增 `src/lazybull/trading/` 包（买入计划、卖出规则、仓位计算），回测与纸面交易统一接入单一实现；同步修复此前功能删除遗留的多处代码损伤（storage/runner/broker/model_registry/reporting），清理废弃代码与废弃测试，全量 949 个测试通过。

- **风控因子预计算告警抑制**：定向抑制新股上市前全 NaN 窗口触发的 `All-NaN slice encountered` 告警（结果正确，属预期行为）。(v0.85.24)

- **风控因子批量预计算**：22 个历史窗口风控因子改为全周期一次性向量化预计算（约 80 秒）+ 每日 O(1) 查表，`build_clean_features` 整体耗时从 6+ 小时回落至约 2 小时；并行路径同步补齐风控因子，修复串行/并行 schema 不一致。(v0.85.23)

- 完全移除风险惩罚功能及其所有相关代码、配置、测试。该功能在实际使用中效果不佳，移除后简化核心架构。(v0.85.21)

**批量汇总年化收益率与 broker 日志完全对齐** (v0.85.6):
- `scripts/batch/batch_paper_trade.ps1` 汇总不再从 `nav.parquet` 重新计算，改为读取 `config.yaml` + `account.json` + clean daily 收盘价，与 broker 使用完全一致的数据源和 CAGR 公式。

- `scripts/compare_walk_forward.py` 已接入这些列的聚合与展示，便于在 `wf_comparison_batches.xlsx` 直接观察“覆盖率/替换收益贡献”。

- 修复了评估输入多出 `mkt_drawdown_20` 时 `predict_proba` 失败并静默回退 `pred_score` 的问题，避免“惩罚已启用但实际未生效”。
- 新增回归测试覆盖该场景，并在失败回退日志中输出模型版本与异常上下文。

- 修复了此前“兜底加载逻辑写在提前返回之后，永远不会执行”的控制流问题。
- 新增回归测试覆盖该路径，确保 OOS 评估能正确生成 `final_score`。

**年化收益率统一为 CAGR 公式 + 批量汇总结束日期修复** (v0.85.1):
- `src/lazybull/paper/broker.py` 年化收益率从简单线性改为复合年化（CAGR）。
- `scripts/batch/batch_paper_trade.ps1` 汇总使用实际最终交易日，确保区间计算准确。

**风险惩罚学习口径修正并对齐真实 TopN 执行** (v0.85.0):
- `src/lazybull/ml/train_core.py` 现在把 `lambda=0` 作为正式候选解参与 calibration 比较；如果 calibration 段没有证明正惩罚更优，就会保留零惩罚，而不是因为“学出了风险画像”就默认启用惩罚。
- `scripts/walk_forward.py` 的风险惩罚学习目标不再固定为 `Top30`，而是跟随真实回测使用的 `bt_top_n`；当前批量脚本默认 `bt_top_n=20`，因此学习与执行口径保持一致。
- `scripts/train_ml_model.py` 的单次训练路径也同步对齐默认 `Top20`，减少训练期诊断和实际持仓目标不一致的问题。

- 风险惩罚不再依赖手工拍脑袋扣分，而是从现有风险类特征中学习出单调分位权重，并在同一 calibration 段上用小网格自动选择 `penalty_lambda`。

**wf_comparison_batches 忽略 seed 作为参数分组维度** (v0.83.1):
- `wf_comparison_batches.xlsx` 在 `跨时间段稳定性`、`模型Alpha评分`、`交易参数收益评分`、`实盘候选评分` 中，不再把仅 seed 不同的运行拆成不同参数组。
- 多种子相关列也不会再出现在 `实验对比` 页的“有区分度参数”里，避免主表把重复试验误读成新参数扫描。
- seed 差异仍保留在 `模型Seed稳定性` 工作表里，继续用于观察同一套非 seed 超参的波动。

**模型 Seed 稳定性改为突出中位数口径** (v0.83.0):
- `模型Seed稳定性` 工作表新增 `模型Alpha分中位数`，用于表示同一套非 seed 超参在多 seed 下的典型表现。
- `Seed稳健分` 改为优先使用中位数，而不是均值，避免单个 seed 异常好时抬高整套参数的代表值。

**wf_comparison_batches 主表默认精简** (v0.83.0):
- `wf_comparison_batches.xlsx` 的 `实验对比` 页现在只保留核心指标和真正有区分度的关键参数。
- 训练/交易常量列与低价值尾部列不再全部平铺，批量扫参时更容易直接看结论。
- 完整指标仍保留在 `逐Split明细`、`模型Alpha评分`、`模型Seed稳定性`、`交易参数收益评分` 等工作表中。

**walk-forward 新增每个 split 的逐日 Top20/Top30 明细导出** (v0.82.0):
- OOS 评估阶段会默认额外导出 `walk_forward_topk_details_{wf_run_id}_splitXX.csv`。
- 每条记录包含 `trade_date`、`topk`、`rank`、`ts_code`、`pred_score` 与 `true_return`，便于直接比较不同 seed 在同一天的前排名单与分数分叉。
- 可用 `--no-export-topk-details` 关闭，适合只做大批量扫参、不保留逐日明细的场景。

**训练验证协议拆分为 `ES / Calibration / Embargo` 三段** (v0.81.1):
- `val_es` 只参与 early stopping 与 `best_iteration` 选择。
- `val_calib` 只参与候选比较、验证评估与稳定性诊断，避免同一块验证集被反复用于“训练内选优”。
- `val_embargo` 继续隔离测试期前沿标签窗口，并新增 `val_calib_*` 统计字段落盘，方便回看每次训练真实使用的验证区间。

**TopK 逐日评估新增样本覆盖保护** (v0.81.1):
- 当日有效样本数小于 K 时，不再把 `TopK` 退化成更小样本的伪 `TopK`。
- 诊断输出新增 `Top{k}_有效交易日数` 与 `Top{k}_样本覆盖率`，帮助排查“Top30 看起来很好，但其实只有少数日期样本数够用”的假稳定性。

**FeatureBuilder 复用时新增跨窗口缓存失效保护** (v0.81.1):
- `precompute_daily_adj()` 在切换历史窗口前会先失效旧的技术因子、市场状态和交易日索引缓存。
- 用于保证 `ensure_features_for_date()` 与纸面交易复用同一个 `FeatureBuilder` 时，不会把上一轮窗口的派生缓存错误复用到下一轮。

**walk-forward 对比报表新增模型 Seed 稳定性视角** (v0.81.0):
- `wf_comparison*.xlsx` 新增 `模型Seed稳定性` 工作表。
- 按“排除多种子字段后的同一套模型超参”聚合不同 seed 的模型 Alpha 表现。
- 新增 `模型Alpha分均值`、`模型Alpha分标准差`、`模型Alpha分最差`、`Seed稳健分` 等列，帮助识别超参是否跨 seed 稳定。

**walk-forward 对比报表最新记录置顶** (v0.80.1):
- `实验对比` 与三张评分表默认按最新 `wf_run_id` 时间倒序输出。
- 报表新增 `最新运行时间` 列，刚跑完的训练结果会直接显示在前面，不再需要从 ID 里手工查找。

**walk-forward 对比报表新增三视角评分体系** (v0.80.0):
- `wf_comparison*.xlsx` 新增 `模型Alpha评分`、`交易参数收益评分`、`实盘候选评分` 三张工作表。
- `模型Alpha分` 聚焦选股统计，帮助判断哪个模型/超参更优秀。
- `交易收益分` 在相同模型参数 + 相同时间段下比较不同交易参数，再跨模型、跨时间段平均，避免收益评分被强模型污染。
- `实盘候选分` 加入硬门槛，不满足模型 Alpha、有效配对环境、最大回撤和最差 CAGR 要求的组合直接置 0，并在 Excel 中用浅红标出失败原因。

**批量纸面交易汇总新增 `append-summary-only` 开关** (v0.77.43):
- `scripts/batch/batch_paper_trade.ps1` 支持 `-AppendSummaryOnly`，开启后仅对已有汇总 CSV 重新整合排序，不会重跑交易。
- 适合在批跑结束后，只想把 `data/reports/paper_trade_batch_summary.csv` 重新按最新记录置顶时使用。

**批量纸面交易汇总 CSV 改为单文件增量更新** (v0.77.42):
- `scripts/batch/batch_paper_trade.ps1` 现在固定使用 `data/reports/paper_trade_batch_summary.csv`。
- 每次运行会先读取旧记录，再把本次结果按最新置顶后写回同一个 CSV，不再生成新的时间戳文件。
- 文件内按 `模型编号 + 计划开始 + 计划结束` 去重，确保同一批次重复执行时会更新旧记录而不是无限累积。

**纸面交易停牌持仓估值修复** (v0.77.41):
- 持仓估值优先使用 `close`，无效时自动回退 `pre_close`，避免停牌日现价为 0 导致市值错误归零。
- 账户估值与持仓明细新增无效价格兜底（回退买入价），避免出现 `-100%` 假亏损与总资产异常下跳。

**walk_forward 支持 yearly 步进窗口** (v0.77.40):
- `scripts/walk_forward.py` 的 `--step` 选项新增 `yearly`。
- `src/lazybull/ml/walk_forward_utils.py` 同步支持 `yearly=12` 月步长，避免 `invalid choice: 'yearly'` 报错。

**batch_walk_forward 的 `step_list` 增加 yearly 选项** (v0.77.39):
- `scripts/batch/batch_walk_forward.ps1` 的 `step_list` 配置注释已补充 `yearly`，可直接在批量实验中使用年度步进窗口。

**批量纸面交易脚本修复 `param` 位置错误导致无法启动** (v0.77.38):
- `scripts/batch/batch_paper_trade.ps1` 现将默认值直接写在 `param` 参数默认值中，确保脚本可被 PowerShell 正常解析。
- 保留“在脚本内改默认参数”与“命令行覆盖参数”两种使用方式。

**批量纸面交易脚本支持脚本内默认日期与多模型汇总** (v0.77.37):
- `scripts/batch/batch_paper_trade.ps1` 可直接在脚本顶部填写默认 `开始日期/结束日期/模型编号列表`，同时支持命令行覆盖。
- 支持一次填写多个模型编号并依次执行，避免手工重复启动。
- 每个模型跑完后会自动统计并输出汇总表（总资产、总收益率、年化收益率、状态），并导出 CSV 到 `data/reports/`。

**批量纸面交易脚本修复 Windows 下 `py -c` 引号丢失问题** (v0.77.36):
- `scripts/batch/batch_paper_trade.ps1` 的 next 交易日解析改为临时 Python 文件执行，不再依赖 `py -c` 参数拼接。
- 修复部分 PowerShell/路径环境下 `read_parquet("./data/clean/trade_cal.parquet")` 引号丢失导致的 `SyntaxError: invalid syntax`。

**批量纸面交易 next 日期解析失败告警修复** (v0.77.35):
- `scripts/batch/batch_paper_trade.ps1` 在调用内嵌 Python 解析下一交易日时，现会严格检查子进程退出码。
- 若 Python 语法或执行异常，不再误判为“后续无可用交易日”，而是直接抛出错误，便于定位根因。

**批量纸面交易脚本支持日期范围+模型编号三参数执行** (v0.77.34):
- `scripts/batch/batch_paper_trade.ps1` 改为必填参数：`-StartDate`、`-EndDate`、`-ModelVersion`。
- 执行前自动写入模型配置：`paper_trade.py config --model-version <ModelVersion>`，并默认执行 `adjust reset-t0` 清空旧状态。
- 执行顺序为：首日 `run --trade-date <StartDate>`，后续每天 `run --trade-date next`，直到达到结束日期停止。

**修复 compare 列重排 duplicate labels 报错** (v0.77.33):
- 修复 `scripts/compare_walk_forward.py` 在“实验对比”列重排阶段可能出现的 `ValueError: cannot reindex on an axis with duplicate labels`。
- 通过“最终导出列按首次出现去重”保证前置参数列与参数区同名列不再冲突。
- `py .\scripts\compare_walk_forward.py` 已可正常完成 raw/batches 两份对比表导出。

**wf_comparison_batches 列顺序与参数可读性优化** (v0.77.32):
- `scripts/compare_walk_forward.py` 将 `最大深度`、`学习率`、`rank权重TopK`、`rank权重值` 前置到实验对比表前部，便于扫参时快速横向比较。
- `选股综合得分` 后紧跟其 3 个构成指标：`RankIC均值`、`ICIR`、`Top30超额均值`。

**模型持久化改用 XGBoost 原生格式，消除跨版本 pickle 告警** (v0.77.28):
- `register_model()` 优先使用 `model.save_model()` 保存为 `.json` 原生格式，避免 XGBoost 版本升级后 pickle 反序列化产生 `UserWarning`。
- `load_model()` 优先从 `.json` 加载，旧 `.joblib` 文件自动静默回退。

**WF 选股综合得分改为三指标口径** (v0.77.27):
- `scripts/compare_walk_forward.py` 的 `选股综合得分` 改为 `RankIC均值 30% + ICIR 30% + Top30超额均值 40%`。
- batch 跑完后优先看这 3 个选股指标：排序方向是否有效、排序稳定性是否够强、Top30 是否真正跑赢全市场。
- `分层单调性(近似)` 保留为辅助观察列，但不再参与 `选股综合得分`。

**实验对比工作表移除 D/E/F 三列冗余信息** (v0.77.26):
- `scripts/compare_walk_forward.py` 的“实验对比”导出列删除 `KEY_说明`、`KEY_Top20_list`、`KEY_Top30_list`，`wf_comparison_batches.xlsx` 更简洁。

**rank-weight 双侧 linear_decay（Top/Bottom 末位权重=2）** (v0.77.25):
- `src/lazybull/ml/train_core.py` 中 `build_rank_sample_weights()` 的 `linear_decay` 模式调整为：TopK 末位权重从 1 提升到 2。
- BottomK 同步采用与 TopK 一致的线性衰减权重（最差样本= `rank_weight`，BottomK 末位=2）。
- `flat` 模式保持 Top/Bottom K 同权，便于与线性模式做 A/B 对照。

**移除验证集逐日评估明细日志** (v0.77.23):
- `src/lazybull/ml/train_core.py` 的 `evaluate_validation_daily()` 不再打印“验证集逐日评估”标题、评估天数、逐日 RankIC 明细及 TopK 跨日均值/标准差逐行日志。
- 保留逐日诊断摘要输出，日志更短更聚焦。

**逐日评估诊断报告瘦身** (v0.77.22):
- `src/lazybull/ml/eval_utils.py` 的逐日评估诊断输出改为 2-3 行摘要，保留全市场基线、样本数区间、TopK 概览和最强 TopK 结论，避免长篇分段日志。

**修复 walk-forward OOS 布尔掩码重建索引告警** (v0.77.21):
- `scripts/walk_forward.py` 测试集过滤掩码改为使用 `df_test_eval.index` 构建，避免布尔索引与 DataFrame 索引不一致。
- 消除 `UserWarning: Boolean Series key will be reindexed to match DataFrame index`，OOS 评估日志更干净。

**rank-weight TopK 线性衰减 + BottomK 降权到1** (v0.77.20):
- `src/lazybull/ml/train_core.py` 的 rank-weight 默认从 TopK 同权改为 `linear_decay`：Top1 权重=`rank_weight`，线性递减到 TopK 末位=1。
- BottomK 不再加权，统一为 1（与 TopK 之外样本一致），减少尾部样本过度放大。
- `scripts/walk_forward.py` 与 `scripts/train_ml_model.py` 新增 `--rank-weight-topk-weight-mode`（`linear_decay|flat`）开关。
- `scripts/batch/batch_walk_forward.ps1` 新增 `$rank_weight_topk_weight_mode` 并自动透传，便于批量实验切换。

**bot_service 调仓诊断 + 次日计划打印** (v0.77.19):
- `ensure_features_for_date()` 返回详细失败原因，bot 调仓日不再仅提示"数据可能不足"，而是输出具体缺失原因和缺失因子列表。
- `format_trade_result()` 末尾追加下一交易日买卖计划（`format_next_day_instructions`），每日 trade 后即可在结果中查看次日买卖指令。

**训练/评估统一主板过滤（与交易口径一致）** (v0.77.18):
- `scripts/walk_forward.py` 在训练窗口、验证评估、测试评估与部署训练链路统一加入主板过滤（`market=主板`），训练数据不再混入 `.BJ` 样本。
- OOS 重点面板中的 Top20/Top30 最新名单与 hit rate 口径与交易链路保持一致，不再出现“评估含.BJ、交易不含.BJ”的偏差。

**OOS重点指标面板 + 对比表重点前置** (v0.77.17):
- `scripts/walk_forward.py` 新增 OOS 重点面板（Top20/Top30 最新名单、命中率、收益中位数、超额）并默认启用简洁输出；新增 `--oos-detail-metrics` 可按需切回每个 split 的详细对比日志。
- `scripts/compare_walk_forward.py` 将 KEY 重点字段前置到“实验对比/逐Split明细”，并在 Excel 中用浅黄色高亮重点列，便于快速定位 Top20/Top30 核心信号。

**训练特征重要性三列紧凑打印** (v0.77.14):
- 特征重要性输出从单列长列表改为 3 列对齐格式，行数缩减约 2/3，内容完整保留。

**多种子筛选参数可配置** (v0.77.12):
- `scripts/walk_forward.py` 新增 `--ensemble-seed-keep-top-ratio` 与 `--ensemble-seed-keep-min-models`
- `scripts/batch/batch_walk_forward.ps1` 新增 `$ensemble_seed_keep_top_ratio` 和 `$ensemble_seed_keep_min_models`，可在批处理配置区直接调整
- 默认仍保持“前30%且至少3个”，仅从固定规则改为可调规则

**多种子 ensemble 子模型筛选规则升级** (v0.77.11):
- 在多种子构建模型时，最终仅保留排序指标最好的前 `30%` 子模型参与集成
- 当 `30%` 对应数量小于 3 时，自动保底保留 3 个子模型
- 排序依据为逐日 `RankIC IR` 优先、逐日 `RankIC` 均值次之，部署训练与 walk-forward 训练保持一致

**LambdaRank + rank_ic 并发参数兼容修复** (v0.77.6):
- 修复 `objective=lambdarank` 且 `early_stopping_metric=rank_ic` 时，XGBoost sklearn 包装器可调用评估指标路径触发 `max_workers must be greater than 0` 的问题
- `src/lazybull/ml/train_core.py` 新增 `n_jobs` 规范化，确保传入线程池参数始终为正整数
- 保持你要求的“早停指标继续 rank_ic，目标函数继续 rank:pairwise”不变

**LambdaRank 支持 rank_ic 早停** (v0.77.5):
- `src/lazybull/ml/train_core.py` 中 `train_xgboost_model()` 调整为统一 eval_metric 选择逻辑
- 当 `objective=lambdarank` 且 `early_stopping_metric=rank_ic` 时，训练保持 `rank:pairwise` 目标函数，同时早停监控使用 `neg_rank_ic`
- 满足“目标函数切换为排序目标、但不改 rank_ic 早停指标”的实盘调参诉求

**TreeLimitedModel 序列化兼容一次性加固** (v0.77.4):
- 通过模型文件探针确认并修复旧版扁平序列化状态：缺失 `base_model` 时反序列化自动重建基础模型
- 新增稳定 `__getstate__` 最小状态输出，避免再次出现 wrapper 状态被底层模型字段覆盖
- 预测路径改为显式安全读取基础模型，修复 `AttributeError: base_model` 导致的 OOS 中断

**TreeLimitedModel 历史模型兼容性一次收口** (v0.77.3):
- 修复旧模型缺失 `tree_limit` 导致 OOS 回测阶段 `AttributeError: tree_limit` 中断
- 对 `tree_limit/max_trees` 缺失或非法值做统一恢复：优先回退到模型实际可用树数
- 若历史状态无法解析有效树数上限，预测自动回退基础模型默认路径（不带限树参数），确保批任务不断裂

**TreeLimitedModel 旧模型 max_trees 兼容修复** (v0.77.2):
- 修复旧版本模型文件加载时缺少 `max_trees` 导致的 `AttributeError: max_trees`，避免 walk-forward OOS 阶段中断
- `src/lazybull/ml/ensemble.py` 新增 `__setstate__`：历史模型反序列化后自动补齐 `max_trees`
- 兼容逻辑保持与前一版防递归修复一致，确保反序列化早期和预测阶段都稳定

**TreeLimitedModel 反序列化递归修复** (v0.77.1):
- 修复 `src/lazybull/ml/ensemble.py` 中 `TreeLimitedModel.__getattr__` 在 joblib 反序列化阶段的递归访问问题，避免 `maximum recursion depth exceeded`
- 当 `base_model` 尚未恢复时，属性代理会安全返回 `AttributeError`，不再进入无限递归
- 该修复直接覆盖 walk-forward OOS 回测加载已注册模型路径，避免 split 训练后在回测加载阶段中断

**特征稳定性筛选低自由度告警修复** (v0.76.4):
- `src/lazybull/ml/train_core.py` 的 `filter_stable_features()` 移除无效的 `np.errstate` 告警抑制，改为在相关系数计算前做有效配对样本数和零方差校验
- 仅当有效样本不少于 2 且截面 rank 方差大于 0 时才计算相关系数，避免 `Degrees of freedom <= 0 for slice` 告警刷屏
- 新增 `tests/test_feature_stability_filter_warning.py` 回归测试，强制将 RuntimeWarning 升级为异常，确认该路径已被彻底规避

**walk-forward 按 split 下标定向训练** (v0.76.3):
- `scripts/walk_forward.py` 新增 `--selected-split-indices`（例如 `--selected-split-indices 0 4 5 7 9`），可只训练指定 split；不传该参数时默认训练全部 split
- `scripts/batch/batch_walk_forward.ps1` 的 `wf_period_configs` 新增 `SelectedSplits`，可直接在每个 `PSCustomObject` 内配置（如 `SelectedSplits = @(0,4,5,7,9)`）；`SelectedSplits = @()` 或不填表示全部 split
- 对非法下标会在启动阶段直接报错，避免批量任务长时间运行后才发现配置问题

**多种子 bagging（walk-forward 训练）** (v0.76.0):
- `scripts/walk_forward.py` 新增 `--ensemble-seeds`（逗号分隔随机种子，如 `42,1,2,3,4`）：启用后每个 split 在多个种子上各训一个子模型，预测取平均，降低单一 holdout 早停带来的训练随机方差
- 复用现有 `EnsembleModel`，推理端 / 模型注册 / 信号生成零改动；与 `--ensemble-offsets`（多偏移窗口）正交可叠加
- 默认为空，严格向后兼容（单种子 = 原行为）；`batch_walk_forward.ps1` 新增 `$ensemble_seeds` 透传

**训练稳定性诊断脚本** (v0.75.1):
- 新增 `scripts/ana/diagnose_training_stability.py`，在与 batch 训练完全一致的配置下，对指定 walk-forward split（默认最近期 `splits[-1]`）用多种子 × 多早停指标（auto / rank_ic）重复训练同一份数据，量化训练随机不稳定性
- 量化三项：`best_iteration`（树数量）种子间方差与变异系数；全池 RankIC 与按日截面 RankIC 的口径差距；多种子预测平均（bagging）对按日截面 RankIC 种子间方差的收敛效果
- 评估按日截面 RankIC 时直接用「保留 NaN」的验证集自行 predict，绕开 `fillna(0)` 口径污染；纯只读分析工具，不改训练主链路，结果输出到 `data/reports/`

**LCD35 SMB 远端数据架构** (v0.75.0):
- 树莓派 LCD35 显示现在支持从远端群晖 NAS 通过 SMB 协议读取纸面交易数据（持仓、调仓状态、净值等）
- 配置文件 `configs/runtime_respi.yaml` 中设置 `data.paper_remote` 启用远端模式
- 日志和缓存文件迁移到本地 `/tmp/lazybull_lcd35/`，不再写入远端 paper 目录

**调仓日卖出与买入同步到同一交易日** (v0.73.1):
- 调仓日生成买入信号时，同步生成非保护持仓的卖出指令，使卖出与买入在同一 T+1 日执行，消除此前卖出滞后一天导致新旧仓位重叠的问题。
- 回测与纸面交易均已对齐：回测通过 `_queue_rebalance_sells()` 排队到 `pending_condition_sells`；纸面交易通过 `_build_rebalance_sell_instructions()` 合并到次日指令文件。
- 调仓卖出自动排除盈利延续保护、新信号目标保留、已在其他卖出队列中的持仓，不与现有止损/提前换出/整体止盈逻辑冲突。

**WF 对比报表新增选股指标组合与选股综合得分** (v0.73.0):
- [scripts/compare_walk_forward.py](scripts/compare_walk_forward.py) 的 `实验对比` sheet 现新增 `RankIC均值`、`ICIR`、`分层单调性(近似)` 三列，并新增独立的 `选股综合得分` 列（与原 `综合得分` 并存）。
- `选股综合得分` 当前采用三指标权重：`RankIC均值 30% + ICIR 30% + Top30超额均值 40%`，并对旧批次缺失字段按有效项自动重归一，避免历史数据报表失真或生成失败。
- `指标说明` sheet 与控制台对比输出同步补充上述选股指标口径，便于仅从选股维度做参数筛选。

**纸面交易支持拖尾残留持仓提前调仓** (v0.72.14):
- [src/lazybull/paper/runtime.py](src/lazybull/paper/runtime.py) 新增 `_resolve_early_rebalance_context()`，非调仓日除“完全空仓”外，只要当前仍有满足盈利延续保护的拖尾持仓，且不存在 `pending_sells`、`pending_buys`、待执行 instruction，也允许提前执行 T0。
- 同一批拖尾保护仓位会直接复用到后续 T0 规划，不再在提前调仓阶段和正式 T0 阶段各自重复判定一次，纸面行为与回测侧“空仓/持有期拖尾提前调仓”入口保持一致。

**纸面交易补位重试槽位上限修复** (v0.72.13):
- [src/lazybull/paper/runtime.py](src/lazybull/paper/runtime.py) 的 `_plan_pending_buy_retry_instructions()` 生成次日补位买单时，不再把“本轮补位候选数”误写成 `desired_position_count`，而是改为沿用组合目标 `top_n`。
- [src/lazybull/paper/runner.py](src/lazybull/paper/runner.py) 的 `_generate_instructions()` 现支持显式传入目标持仓数，因此像“目标仓位 20、当前持仓 16、补位候选 5 只”这样的场景，会允许开出 4 个新槽位，只拦下超出的第 5 只，而不会错误提示全部“无可用空槽”。

**纸面交易补位原因跨日重复叠加修复** (v0.72.12):
- [src/lazybull/paper/models.py](src/lazybull/paper/models.py) 新增 `normalize_trade_reason()`，统一折叠重复的 `补位槽位-`、`（无可用空槽）`、`（无价格数据）` 等补位标签。
- [src/lazybull/paper/broker.py](src/lazybull/paper/broker.py)、[src/lazybull/paper/runner.py](src/lazybull/paper/runner.py) 和 [src/lazybull/paper/runtime.py](src/lazybull/paper/runtime.py) 在失败买单转补位、跨日重试再规划次日 instruction 时都改为复用该归一化逻辑，不再出现 reason 每过一天就再包一层的问题。
- [src/lazybull/paper/reporting.py](src/lazybull/paper/reporting.py) 的次日指令预览展示也会对历史脏 reason 做兜底清洗，因此旧数据展示会立即变干净。

**纸面交易仓位展示前追加下一交易日买卖明细** (v0.72.11):
- [scripts/paper_trade.py](scripts/paper_trade.py) 的 `positions`/`run` 持仓展示现在会先输出“下一交易日指令”区块，再打印当前持仓表。
- [src/lazybull/paper/reporting.py](src/lazybull/paper/reporting.py) 新增 `format_next_day_instructions()`，统一展示下一交易日待执行的卖出、买入明细，包含股数、目标权重和触发原因。
- [src/lazybull/paper/reporting.py](src/lazybull/paper/reporting.py) 的 `format_positions_mobile()` 也同步加入该区块，因此钉钉 `positions` 命令会先展示次日计划、再展示持仓明细。

**一致预期修正有效样本不足告警修复** (v0.72.10):
- [src/lazybull/factors/consensus_revision.py](src/lazybull/factors/consensus_revision.py) 的 EPS 分歧度计算现在会先统计非 NaN 有效样本数，再决定是否计算标准差。
- 当窗口里虽然有多条研报、但真正有值的 EPS 预测不足 2 条时，直接输出 `NaN`，不再触发 numpy 的 `Degrees of freedom <= 0 for slice` RuntimeWarning。
- 更新 [tests/test_factor_consensus_revision.py](tests/test_factor_consensus_revision.py) 回归测试，用 `warnings.simplefilter("error", RuntimeWarning)` 确认不会再冒出这类告警。

**纸面交易下一个交易日缺失告警降噪** (v0.72.9):
- [src/lazybull/paper/runner.py](src/lazybull/paper/runner.py) 的 `_get_next_trade_date()` 在确实没有下一交易日可返回时，内部日志已从 warning 降为 debug。
- 真正依赖下一交易日的业务分支仍会在调用方按原逻辑报错；仅用于摘要展示、尾部状态或可选流程的查询不再反复打印无害告警。
- [scripts/paper_trade.py](scripts/paper_trade.py) 的运行完成摘要也不再显示 `[None]`，改为 `无`。

**纸面交易非交易日输入先刷新交易日历再自动补数** (v0.72.8):
- [src/lazybull/paper/runner.py](src/lazybull/paper/runner.py) 的交易日校正与前后交易日解析现在会在本地日历覆盖不足时先刷新基础交易日历，再从 clean/raw 两层挑选覆盖更完整的开市日列表。
- 纸面交易输入非交易日时，会先顺延到下一交易日再触发 features/raw/clean 自动补齐，不再出现“旧交易日历下先报当日无数据、随后才下载下一交易日数据”的错序日志。
- 更新 [tests/test_ensure_and_t0_printing.py](tests/test_ensure_and_t0_printing.py) 回归测试，覆盖 stale trade_cal 刷新后的顺延行为。

**纸面交易 T1 全面改为只执行前一日 T0 指令** (v0.72.7):
- [src/lazybull/paper/runtime.py](src/lazybull/paper/runtime.py) 的共享执行顺序已调整为“先执行当日 T1 既有 instruction，再在当日 T0 统一规划下一交易日 instruction”。
- 止损、亏损提前换出、整体止盈、持有期到期卖出，以及卖出失败/买入失败后的下一日重试，现全部在 T0 写入次日 instruction 文件；T1 不再临时计算新卖单，也不再在同一交易日立即顺延补位。
- [src/lazybull/paper/models.py](src/lazybull/paper/models.py) 的 `TradeInstruction` 新增 `retry_attempt` 元数据，补位失败重试可在 instruction 文件里保留轮次。
- 更新 [tests/test_paper_trade_runtime.py](tests/test_paper_trade_runtime.py) 与 [tests/test_paper_holding_period_alignment.py](tests/test_paper_holding_period_alignment.py) 回归测试，覆盖“T1 只执行预生成指令”的纸面交易新口径。

**回测非动态持仓分支补齐 T0->T1 卖后买链路** (v0.72.6):
- [src/lazybull/backtest/engine.py](src/lazybull/backtest/engine.py) 在 `enable_profit_based_holding=False` 的 `holding_period` 卖出路径中，现会同步写入次日补位买入计划。
- 执行语义恢复为 T+1 当日“先卖后买”，减少了“T+1 只卖不买”导致的空仓提前调仓与短轮次日志。
- 更新 [tests/test_backtest_t1.py](tests/test_backtest_t1.py) 回归测试，新增未启用盈亏动态持仓场景下的 T1 卖后买断言。

**调仓决策摘要“最终”段精简为短标签** (v0.72.5):
- [src/lazybull/backtest/engine.py](src/lazybull/backtest/engine.py) 的 `调仓决策摘要` 里，`最终` 字段不再重复展开 `信号门控 x 质量 x ECT x 市场层` 的完整乘法细节。
- 正常路径改为 `最终=xx.x%[入队/不入队]`；门控阻断路径改为 `最终=0.0%[门控阻断, 不入队]`，在保留关键结论的同时显著缩短单行日志长度。
- 更新 [tests/test_ml_backtest_engine.py](tests/test_ml_backtest_engine.py) 回归测试，覆盖新的 `最终` 段文案。

**回测信号与交易摘要统一改为卖在前、买在后** (v0.72.4):
- [src/lazybull/backtest/engine.py](src/lazybull/backtest/engine.py) 的日终 `交易:` 摘要现在会先输出卖出、再输出买入；`信号:` 单行也同步调整为先展示卖出信号，再展示买入信号。
- 这样日志顺序会更贴合当前回测语义：先看释放了哪些仓位，再看补入了哪些仓位或计划补哪些仓位。
- 更新 [tests/test_backtest_daily_progress_log.py](tests/test_backtest_daily_progress_log.py) 回归测试，覆盖 `交易:` 与 `信号:` 的新顺序。

**回测信号摘要新增延续数量并调整输出顺序** (v0.72.3):
- [src/lazybull/backtest/engine.py](src/lazybull/backtest/engine.py) 的 `信号:` 单行现在会额外展示当日 `盈利延续` 数量；如果当天没有新的买/卖信号，但存在延续持有，也会打印类似 `信号: 延续[20]` 的摘要。
- 同一行的输出位置已从交易摘要之前调整到 `交易: 买... / 卖...` 之后，便于先看 T+1 实际成交，再看当日新生成的 T 日信号。
- 更新 [tests/test_backtest_daily_progress_log.py](tests/test_backtest_daily_progress_log.py) 回归测试，覆盖“延续数量统计”和“信号摘要顺序后移”的行为。

**回测新增 T 日信号数量摘要日志** (v0.72.2):
- [src/lazybull/backtest/engine.py](src/lazybull/backtest/engine.py) 现在会在每个产生新买入/卖出信号的 T 日，额外输出一条简洁的 `信号:` 统计，按类别汇总当日新生成的买卖信号数量。
- 统计口径覆盖调仓买入、补槽买入，以及持有期卖出、亏损换出、时间止损、回撤止损、移动止损、连续跌停等卖出信号，且输出位置固定在调仓决策摘要之后、实际成交摘要之前，便于区分“T 日决策”和“T+1 成交”。
- 更新 [tests/test_backtest_daily_progress_log.py](tests/test_backtest_daily_progress_log.py) 回归测试，覆盖文本格式与日志顺序。

**回测盈利延续/持有期卖出后次日补买修复** (v0.72.1):
- [src/lazybull/backtest/engine.py](src/lazybull/backtest/engine.py) 现在会在 `holding_period` 卖出信号生成日同步写入对应的待买计划；到下一交易日先按实际卖出结果释放空槽，再按 T0 的 ML 候选顺位补回，不再出现盈利延续失败后仓位一路下滑但没有对应补买的问题。
- 同文件的待买计划新增显式 `desired_position_count`，部分补买场景会按实际卖出成功数量收缩开仓槽位，而不是把“待补槽位数”误当成组合总目标持仓数。
- 更新 [tests/test_backtest_t1.py](tests/test_backtest_t1.py) 回归测试，覆盖“盈利延续未通过后，下一交易日先卖再买”的行为。

**纸面交易买入改为 T0 计划、T+1 执行** (v0.72.0):
- [src/lazybull/paper/runner.py](src/lazybull/paper/runner.py) 与 [src/lazybull/paper/runtime.py](src/lazybull/paper/runtime.py) 现在会在 T0 保存 ranked_candidates，并把计划买单写入下一交易日 instruction。
- [src/lazybull/paper/broker.py](src/lazybull/paper/broker.py) 现按 T0 目标持仓数限制新开仓；如果当日卖出失败导致空槽未真正释放，当日不会继续超配买入。
- 当前版本的进一步对齐见 v0.72.7：T1 已不再同日顺延补位，所有 T1 动作均来自前一日 T0 instruction。

**回测买入改为 T0 计划、T+1 按优先级顺位执行** (v0.71.79):
- [src/lazybull/backtest/engine.py](src/lazybull/backtest/engine.py) 现在会在 T0 保存 ML 候选优先级和原始槽位权重，T+1 执行买入时若原计划股票涨停、停牌或未成交，会在同日按候选顺序继续顺延，而不是提前在 T0 用 T+1 状态做前视过滤。
- 同文件中，若 T+1 卖出失败并进入延迟卖出队列，当日买入会按实际空槽数收缩，不再因为账户里仍有现金而继续超配加仓。
- 更新 [tests/test_backtest_t1.py](tests/test_backtest_t1.py) 与 [tests/test_position_completion.py](tests/test_position_completion.py) 回归测试，覆盖“同日顺延优先，跨日补齐兜底”的新口径。

**回测持有期/盈利延续卖出改为 T0 信号、T+1 执行** (v0.71.78):
- [src/lazybull/backtest/engine.py](src/lazybull/backtest/engine.py) 将“持有期到期、盈利延续未通过、盈利延续到期”三条预定卖出路径从当日直卖改为先写入 `pending_condition_sells`，由下一交易日统一执行，和现有条件卖出时序对齐。
- 同文件初始化日志中的交易规则说明已同步更新，明确卖出变为“满持有期后 T0 生成卖出信号，下一交易日成交”。
- 更新 [tests/test_backtest_t1.py](tests/test_backtest_t1.py) 回归测试，覆盖持有期到期、盈利延续未通过、盈利延续到期三条 T0->T1 卖出路径。

**download_raw 支持仅下载 ST 状态数据** (v0.71.77):
- [scripts/download_raw.py](scripts/download_raw.py) 新增 `--only-is-st` 参数，可只下载 `stock_st`（is_st 来源数据），不触发其它日线/另类数据下载。
- 同文件新增参数冲突校验：`--only-is-st` 不能与 `--only-basic`、`--all`、`--download` 同时使用，避免执行语义不明确。
- `--all` 仍会覆盖 `stock_st`：其走日线组下载路径，而日线集合 `DAILY_SUBSETS` 已包含 `stock_st`。

**ST 判定改为 stock_st 按日口径写入** (v0.71.76):
- [src/lazybull/data/cleaner.py](src/lazybull/data/cleaner.py) 新增 `clean_stock_st()`，并将 `add_tradable_universe_flag()` 的 `is_st` 判定改为“`stock_st` 优先、名称规则兜底”，修复历史回测中因名称快照引起的 ST 漏判。
- [scripts/download_raw.py](scripts/download_raw.py) 的默认日线下载已纳入 `stock_st`，并在 [src/lazybull/data/tushare_client.py](src/lazybull/data/tushare_client.py) 新增 `get_stock_st()` 封装，保证 raw 层可按交易日落盘 ST 状态。
- [scripts/build_clean_features.py](scripts/build_clean_features.py) 与 [src/lazybull/data/ensure.py](src/lazybull/data/ensure.py) 均已接入 `stock_st`，离线构建与自动补齐链路现在使用同一 ST 数据口径。
- 更新 [tests/test_cleaner.py](tests/test_cleaner.py) 与 [tests/test_ensure_and_t0_printing.py](tests/test_ensure_and_t0_printing.py) 回归测试。

**回测最小买入后市值阈值对齐纸面交易** (v0.71.75):
- [src/lazybull/backtest/engine.py](src/lazybull/backtest/engine.py) 新增 `min_buy_value_ratio`，并在回测买入执行路径中按与纸面交易一致的口径拦截过小买单：`阈值=总资产/目标持仓数*ratio`。
- 该拦截逻辑对正常买入、补齐买入、延迟买入重试统一生效，避免出现 `0.1w` 这类高成本低效率小单。
- [src/lazybull/common/backtest_runtime.py](src/lazybull/common/backtest_runtime.py) 已透传 `trading_config.min_buy_value_ratio` 到回测引擎，保证两端配置一致。
- 新增 [tests/test_backtest_min_buy_value_threshold.py](tests/test_backtest_min_buy_value_threshold.py) 回归测试，覆盖阈值开启/关闭两种行为。

**交易买卖明细不再省略** (v0.71.74):
- [src/lazybull/backtest/engine.py](src/lazybull/backtest/engine.py) 的 `交易: 买...` 与 `交易: 卖...` 日志由压缩展示改为完整展示，当日有多少成交就打印多少条股票明细，不再出现 `...+N` 折叠。
- 更新 [tests/test_backtest_daily_progress_log.py](tests/test_backtest_daily_progress_log.py) 回归测试，覆盖“买卖明细不折叠”的口径。

**回测 T 标签按调仓周期重置** (v0.71.73):
- [src/lazybull/backtest/engine.py](src/lazybull/backtest/engine.py) 的每日进度摘要 `T` 序号由全局累计改为按调仓周期内计数：每个调仓日为 `T0`，随后 `T1/T2/...`，到下一调仓周期再次从 `T0` 开始。
- 更新 [tests/test_backtest_daily_progress_log.py](tests/test_backtest_daily_progress_log.py) 回归测试，覆盖 `T` 标签按周期重置的口径。

**回测每日日志标签与买入摘要可读性优化** (v0.71.72):
- [src/lazybull/backtest/engine.py](src/lazybull/backtest/engine.py) 的每日进度摘要前缀由 `回测[日期]` 改为 `T序号[日期]`（见 v0.71.73：已进一步修正为按调仓周期重置）。
- 同文件中，当日交易摘要里的买入项不再展示 `0d,+0.0%`，改为展示单票买入现金支出（万元），格式如 `买3[002383.SZ(10.3w), ...]`。
- 更新 [tests/test_backtest_daily_progress_log.py](tests/test_backtest_daily_progress_log.py) 回归测试，覆盖新的日汇总前缀与买入摘要格式。

**walk-forward 验证集自动尾部隔离** (v0.71.71):
- [src/lazybull/ml/train_core.py](src/lazybull/ml/train_core.py) 在 `prepare_training_data()` 中新增验证集尾部自动隔离：按标签自动推导 `label_delta` 后，仅让隔离后的 `val_es` 参与 `eval_set`、early stopping 与 `best_iteration` 选择，避免测试期价格窗口通过 val 影响模型选择。
- 同文件新增 `split_val_for_early_stopping_by_date()`，输出 `val_raw / val_es / val_embargo` 三段统计，并在样本过短时允许 `val_es` 为空，训练流程自动退化为“无验证集早停”。
- [scripts/walk_forward.py](scripts/walk_forward.py)、[scripts/train_ml_model.py](scripts/train_ml_model.py) 与 [src/lazybull/ml/run_logger.py](src/lazybull/ml/run_logger.py) 已透传并落盘 `val_raw_* / val_es_* / val_embargo_*` 字段，便于回看每次训练的隔离规模与生效区间。

**补齐跳过与延迟订单放弃继续压缩** (v0.71.70):
- [src/lazybull/backtest/engine.py](src/lazybull/backtest/engine.py) 现在会把 `补齐跳过` 的当日无行情、前日无行情、无数据、无候选、候选已持仓、候选不可交易等情况统一压成每日白字单行摘要，不再在补齐流程中散落多条 warning
- [src/lazybull/execution/pending_order.py](src/lazybull/execution/pending_order.py) 的买入延迟订单超次/超期放弃也改为交给 [src/lazybull/backtest/engine.py](src/lazybull/backtest/engine.py) 汇总，日终以 `延迟订单放弃: 超次买... | 超期买...` 的形式输出

**延迟订单日志继续压缩** (v0.71.69):
- [src/lazybull/execution/pending_order.py](src/lazybull/execution/pending_order.py) 不再逐条输出 `添加延迟订单` 和 `延迟订单执行成功`，而是将事件交给 [src/lazybull/backtest/engine.py](src/lazybull/backtest/engine.py) 按交易日统一汇总
- 每日日终现在会以白字单行展示 `延迟订单: 新增买/卖...` 与 `延迟订单成交: 成功买/卖...`，避免跌停批量延迟卖出时刷出多行重复日志

**回测交易与补齐日志进一步压缩** (v0.71.68):
- [src/lazybull/backtest/engine.py](src/lazybull/backtest/engine.py) 现在会把当日 `交易` 摘要拆成 `买`、`卖` 两行，便于快速扫读；`调仓决策摘要` 也提前到了交易结果之前展示
- 同文件中，`重复买入跳过`、`亏损提前换出`、`仓位未满`、`补齐放弃` 改为按交易日统一汇总成白色单行，不再逐条刷屏
- `盈利延续[strength]` 单行摘要会显示更多股票后再折叠，减少过早出现 `...+N`

**调仓摘要与 warning 日志继续压缩** (v0.71.67):
- [src/lazybull/backtest/engine.py](src/lazybull/backtest/engine.py) 现在会把 `调仓决策摘要` 压成单行白色日志，并提前到信号日输出，不再等到 T+1 执行日；在 ML 回测中，[src/lazybull/backtest/engine_ml.py](src/lazybull/backtest/engine_ml.py) 会同步把市场层与最终仓位提前写入该摘要
- [src/lazybull/signals/ml_signal.py](src/lazybull/signals/ml_signal.py) 不再输出 `选股/预测(ranked)` 入口日志，减少与当日汇总重复的信息
- [src/lazybull/backtest/engine.py](src/lazybull/backtest/engine.py) 将 `空仓提前调仓`、`时间止损`、`整体止盈` 三类事件改为按交易日统一汇总，以白色单行显示在每日总结之下

**回测日志按交易日重排并压缩** (v0.71.66):
- [src/lazybull/backtest/engine.py](src/lazybull/backtest/engine.py) 现在会把单个交易日的明细日志先缓冲，待日终按“彩色每日总结 -> 当日买卖摘要 -> 两空格缩进细节”顺序统一输出，避免同一天的杂项日志出现在总结之前
- 每日总结新增买/卖股票数量，移除 ATR 展示；若当日有成交，则下一行紧凑显示买入/卖出股票代码、持有天数与收益率
- `盈利延续持有[strength]` 与 `补齐成功/补齐延迟` 改为按日单行汇总，原有买入/卖出执行日志不再单独输出
- [src/lazybull/signals/ml_signal.py](src/lazybull/signals/ml_signal.py) 将 `选股过滤合计` 与 `开始模型预测(ranked)` 合并为 `选股/预测` 单行日志，减少重复噪音

**回测补齐成功日志补充实际成交口径** (v0.71.65):
- [src/lazybull/backtest/engine.py](src/lazybull/backtest/engine.py) 的“补齐成功”日志现在会同时展示目标市值、实际成交市值、成交股数与交易成本，现金受限缩量时不会再被误读为按目标金额满额成交
- 更新 [tests/test_position_completion.py](tests/test_position_completion.py) 回归测试，覆盖“目标市值 5 万但实际只够买一手低价股”的场景

**树莓派 3.5LCD 盘内指数刷新时机修复** (v0.71.64):
- [scripts/respi/lcd35/core.py](scripts/respi/lcd35/core.py) 与 [scripts/respi/lcd35/data_pipeline.py](scripts/respi/lcd35/data_pipeline.py) 现在会把“指数缓存是否过期”作为独立条件判断，完整缓存也会按盘中刷新节奏续刷
- 持仓快照开始前会先并行预热后台指数刷新，因此本轮更有机会直接拿到新指数值，减少盘内指数线“慢一拍”的现象
- 更新 [tests/test_respi_35lcd_disp.py](tests/test_respi_35lcd_disp.py) 回归测试，覆盖完整缓存续刷和快照前预热两种场景

**树莓派 3.5LCD 盘内行业统计口径修复** (v0.71.63):
- [scripts/respi/lcd35/industry.py](scripts/respi/lcd35/industry.py) 在盘内行业面板计算时，如果实时快照缺少 `PRE_CLOSE`，会改用 `PCT_CHG` 反推昨收，避免把大量个股直接排除在盘内统计之外
- [scripts/respi/lcd35/rendering.py](scripts/respi/lcd35/rendering.py) 在盘中模式下不再回退展示周期行业面板，避免当日盘面走弱时仍显示一片红的持仓周期收益
- 更新 [tests/test_respi_35lcd_disp.py](tests/test_respi_35lcd_disp.py) 回归测试，覆盖缺昨收和盘中渲染回退两类场景

**树莓派 3.5LCD 盘前/盘后启动快照短路修复** (v0.71.62):
- [scripts/respi/lcd35/data_pipeline.py](scripts/respi/lcd35/data_pipeline.py) 在盘前、盘后和非交易日会直接优先使用收盘日线快照，不再先卡在 efinance 或 AKShare 的慢路径上
- 同文件里的日线 fallback 已收缩为 1 次全市场 `daily` 加 3 次 `index_daily`，启动恢复速度明显快于逐股查询
- 更新 [tests/test_respi_35lcd_disp.py](tests/test_respi_35lcd_disp.py) 回归测试，覆盖盘前短路和单次 `daily` 查询行为

**树莓派 3.5LCD 盘后重启面板恢复修复** (v0.71.61):
- [scripts/respi/lcd35/data_pipeline.py](scripts/respi/lcd35/data_pipeline.py) 在盘后或非实时窗口内，如果 efinance 与 AKShare 都拿不到快照，会自动改用目标交易日的个股日线和指数日线合成持仓快照
- 这样即使软件在盘后重启，左侧摘要、右侧个股排行和行业统计也能恢复，不会只剩下周期图
- 顶栏更新来源新增 `D` 标识，表示当前展示基于收盘日线数据

**树莓派 3.5LCD 盘后停止重复抓取修复** (v0.71.60):
- [scripts/respi/lcd35/charting.py](scripts/respi/lcd35/charting.py) 的 `_get_refresh_policy()` 现在直接复用盘后实时补齐判定 helper，盘后只会在“周期图尚未追平且仍处于补尾宽限期”时继续抓实时数据
- 一旦周期图已经覆盖到当日目标交易日，或者已经超过收盘后的补尾宽限窗口，实时抓取就会停掉，不再每 10 分钟重复拉取快照
- 更新 [tests/test_respi_35lcd_disp.py](tests/test_respi_35lcd_disp.py) 回归测试，覆盖盘后追平和宽限期结束两种停止条件

**树莓派 3.5LCD 周期图旧状态兼容修复** (v0.71.59):
- [scripts/respi/lcd35/data_pipeline.py](scripts/respi/lcd35/data_pipeline.py) 在周期图计算前会统一归一化 `cash`、`shares`、`buy_price` 与日线 `close/trade_date`，避免旧状态文件中的字符串数值或脏值导致收盘后周期图构建异常
- 同文件中的“抓周期阶段异常”日志现在会带上异常类型和异常消息，树莓派现场日志可以直接看出具体失败点，而不是只看到笼统的阶段失败
- 新增 [tests/test_respi_35lcd_disp.py](tests/test_respi_35lcd_disp.py) 回归测试，覆盖旧状态数值字段为字符串时周期图仍能正常生成

**一致预期 freshness 特征补齐** (v0.71.58):
- [src/lazybull/factors/consensus.py](src/lazybull/factors/consensus.py) 为 report_rc 一致预期聚合新增 `consensus_freshness_days`，表示最近一次可见研报距当日的天数
- [src/lazybull/ml/train_core.py](src/lazybull/ml/train_core.py) 已将该列接入一致预期训练特征清单
- [src/lazybull/features/builder.py](src/lazybull/features/builder.py) 在一致预期启用但当日无研报数据时补 NaN 占位，避免 freshness 列时有时无
- [src/lazybull/features/ensure.py](src/lazybull/features/ensure.py) 将其纳入特征 schema 校验，旧缓存缺列时会触发重建

**fina_indicator / cashflow 季度分区与窗口读取** (v0.71.57):
- [scripts/download_raw.py](scripts/download_raw.py) 将 `fina_indicator` 与 `cashflow` 的全量 raw 下载切到按季度分区落盘，`fina_indicator` 继续强制显式请求字段
- [src/lazybull/data/loader.py](src/lazybull/data/loader.py) 为这两类公告型季度数据增加“按窗口读取、旧单文件回退”的加载逻辑，避免 paper_trade 与构建特征时反复整表读取
- [src/lazybull/features/ensure.py](src/lazybull/features/ensure.py) 与 [scripts/build_clean_features.py](scripts/build_clean_features.py) 已接到新窗口读取路径，覆盖 paper_trade 缺数自动补齐和离线特征构建两条主链路
- 新增 [tests/test_announcement_partition_window_loading.py](tests/test_announcement_partition_window_loading.py) 并扩展相关 ensure 回归测试，验证季度分区读写闭环

**fina_indicator 显式字段与基本面代理列修复** (v0.71.56):
- [src/lazybull/data/tushare_client.py](src/lazybull/data/tushare_client.py) 抽取 `FINA_INDICATOR_DEFAULT_FIELDS`，并由 [src/lazybull/features/ensure.py](src/lazybull/features/ensure.py) 在全量下载和旧 schema 回补时强制显式请求，修复 `q_gr_yoy` / `inv_turn` 被默认 schema 漏拉的问题
- [src/lazybull/features/ensure.py](src/lazybull/features/ensure.py) 在检测到旧版 `fina_indicator` raw 缺关键列时，会先按已有报告期回补后再继续增量补齐；paper_trade 走同一条 ensure 链路，因此自动下载也同步修复
- [src/lazybull/factors/fundamental.py](src/lazybull/factors/fundamental.py)、[src/lazybull/features/builder.py](src/lazybull/features/builder.py)、[src/lazybull/ml/train_core.py](src/lazybull/ml/train_core.py) 将无法稳定获取的 `cf_sales` / `cf_nm` / `goodwill` 替换为可稳定落地的代理实现，并补齐训练侧对应 zscore 列
- 新增/扩展回归测试，覆盖训练入口透传、fina_indicator 全量下载显式字段、旧 schema 回补和基本面代理列回填

**训练入口新增因子开关透传修复** (v0.71.55):
- [scripts/walk_forward.py](scripts/walk_forward.py) 补齐 `cashflow_quality` / `consensus_revision` 到训练选列层的透传，修复 batch 命令已带开关但特征数不变的问题
- [scripts/train_ml_model.py](scripts/train_ml_model.py) 新增 `--enable-cashflow-quality-features` 与 `--enable-consensus-revision-features` 参数，并同步透传到训练选列层
- [src/lazybull/features/builder.py](src/lazybull/features/builder.py) 在行业中性化白名单中补回 `fcf_yield`，修复 `zscore_fcf_yield` 长期未写入特征文件的问题
- 新增 [tests/test_training_feature_flag_forwarding.py](tests/test_training_feature_flag_forwarding.py) 回归测试，覆盖 walk-forward 与单次训练两条入口链路

**3.5LCD 渲染线程屏幕状态字段修复** (v0.71.54):
- [scripts/respi/lcd35/state.py](scripts/respi/lcd35/state.py) 为 `DisplayState` 补回 `is_screen_on` 默认值
- 修复树莓派启动后显示线程持续报错 `AttributeError: 'DisplayState' object has no attribute 'is_screen_on'` 并无法正常渲染的问题
- 新增 [tests/test_respi_35lcd_disp.py](tests/test_respi_35lcd_disp.py) 回归用例，覆盖默认亮屏状态字段

**3.5LCD 主入口 src 导入修复** (v0.71.53):
- [scripts/respi/lcd35_display.py](scripts/respi/lcd35_display.py) 在加载 `_context` 之前先注入项目根目录与 `scripts` 目录到 `sys.path`
- 修复树莓派上使用 `nohup python ./scripts/respi/lcd35_display.py` 时，启动阶段报 `ModuleNotFoundError: No module named 'src'`
- 新增 [tests/test_respi_35lcd_disp.py](tests/test_respi_35lcd_disp.py) 回归用例，覆盖缺少项目根路径时的入口自举导入

**3.5LCD 图表中文乱码恢复** (v0.71.52):
- [scripts/respi/lcd35/charting.py](scripts/respi/lcd35/charting.py) 恢复被编码破坏的中文注释、docstring 与屏幕文案
- 保留拆分后的现有逻辑与函数结构，只修正文本文字层

**3.5LCD 显示入口与模块命名整理** (v0.71.51):
- [scripts/respi/lcd35_display.py](scripts/respi/lcd35_display.py) 成为新的 3.5LCD 主入口，命名更正常，也更适合后续直接引用
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 退化为历史兼容壳，继续兼容旧命令和旧测试
- [scripts/respi/lcd35](scripts/respi/lcd35) 目录去掉数字前缀，并把运行时进一步拆成状态、渲染、应用三层职责

**3.5LCD 共享加载与告警收敛** (v0.71.51):
- 主入口保持共享命名空间装配，兼顾旧测试 monkeypatch 与环境变量重读
- 行业、图表、系统 IO 和渲染页补齐显式依赖后，编辑器告警和运行时 NameError 一并收敛
- 行业面板计数改为只统计已知行业，避免未知行业污染展示口径

**3.5LCD 行业贡献比例口径优化** (v0.71.48):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 行业页贡献比例改为正负方向分别归一化
- 正收益行业贡献比例合计 `+100%`，负收益行业贡献比例合计 `-100%`
- 更新 [tests/test_respi_35lcd_disp.py](tests/test_respi_35lcd_disp.py) 对应回归用例

**3.5LCD 诊断日志时间戳化** (v0.71.47):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 的 stderr 诊断输出前缀改为 `[%H:%M:%S]`，更便于现场按时间追日志
- 新增 [tests/test_respi_35lcd_disp.py](tests/test_respi_35lcd_disp.py) 回归用例，覆盖 stderr 前缀格式

**一致预期修正告警修复** (v0.71.46):
- [src/lazybull/factors/consensus_revision.py](src/lazybull/factors/consensus_revision.py) 新增 `_safe_nanmean`，修复窗口内全 NaN 时的 `RuntimeWarning: Mean of empty slice`
- 新增 [tests/test_factor_consensus_revision.py](tests/test_factor_consensus_revision.py) 回归用例，覆盖“目标价窗口全 NaN”场景

**一致预期修正二次提速** (v0.71.45):
- [src/lazybull/factors/consensus_revision.py](src/lazybull/factors/consensus_revision.py) 改为按股票遍历活跃交易日窗口，并使用 `searchsorted` 做窗口定位
- 预构建 `close_adj` 哈希索引，避免循环内逐日逐股筛选 DataFrame
- 进度日志改为按股票批次输出，长耗时阶段可持续看到推进

**一致预期修正构建性能优化** (v0.71.44):
- [src/lazybull/factors/consensus_revision.py](src/lazybull/factors/consensus_revision.py) 预先按股票分组复用，避免按交易日重复 `groupby` 带来的高耗时
- 新增有效交易日范围裁剪（基于 report_date 覆盖区间），减少无效日期遍历
- 新增阶段进度日志（每 50 个交易日打印），避免长时间无输出

**一致预期修正因子双口径兼容修复** (v0.71.43):
- [src/lazybull/factors/consensus_revision.py](src/lazybull/factors/consensus_revision.py) 同时支持 `rec_fore_Netprofit/rec_target` 与 `np/tp` 字段口径
- 当 `rec_target` 缺失时，自动回退使用 `max_price/min_price` 构建目标价相关因子
- 新增 [tests/test_factor_consensus_revision.py](tests/test_factor_consensus_revision.py) 回归测试，覆盖 `rec_*` 与 `np/tp + max/min` 两套口径

**新增因子接线补齐与口径对齐** (v0.71.42):
- [scripts/build_clean_features.py](scripts/build_clean_features.py) 补齐 `cashflow_quality` 与 `consensus_revision` 的实际加载、lookup 构建与传参
- [src/lazybull/features/ensure.py](src/lazybull/features/ensure.py) 补齐 `cs_infer` 自动补齐链路中的 cashflow / consensus_revision
- [src/lazybull/features/builder.py](src/lazybull/features/builder.py) 当当日无数据时补齐 freshness 占位列，确保 schema 稳定
- [src/lazybull/features/ensure.py](src/lazybull/features/ensure.py) `fina_indicator_vip` 下载改为全字段，覆盖新增基本面因子

### 当前版本历史 (v0.71.38)

**3.5LCD 盘后重启空面板快速重试修复** (v0.71.38):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 当实时抓取失败且无可用快照时，改为 15 秒快速重试，而非等待常规 180/600 秒
- 盘后重启场景下，摘要/排行/行业会尽快恢复，不再长时间空白
- 新增快速重试诊断日志，便于现场确认重试链路生效

### 当前版本历史 (v0.71.37)

**3.5LCD 指数抓取单次调用优化** (v0.71.37):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 单次调用 `stock_zh_index_spot_sina` 直接提取上证/深证/中证800
- 正常路径下指数抓取调用次数从 2 次降到 1 次
- 仅在单次返回未命中中证800时触发兜底二次请求，兼顾性能与稳定性

### 当前版本历史 (v0.71.36)

**3.5LCD 盘中折线图不刷新修复** (v0.71.36):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 在实时快照中新增 `PCT_CHG` 字段，兼容 `PRE_CLOSE` 缺失场景
- 盘中收益计算支持“现价+涨跌幅反推昨收”，避免 `holdings_pct` 长期为 `None` 导致日内图不落点
- 新增“盘中图跳过”与“盘中收益计算”诊断日志，现场可直接看到未刷新的具体原因

### 当前版本历史 (v0.71.35)

**3.5LCD 指数抓取可观测性增强与超时定位修复** (v0.71.35):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 新增指数抓取阶段日志（触发、开始、主接口返回、结束耗时）
- 后台指数抓取新增独立超时配置 `LAZYBULL_REALTIME_INDEX_ASYNC_TIMEOUT_SECONDS`（默认60秒）
- 超时时明确打印“后台指数抓取超时”，不再出现“10秒后重来但无日志”
- 补充“后台指数刷新跳过（上一轮仍在执行）”日志，排查并发刷新更直观

### 当前版本历史 (v0.71.34)

**3.5LCD 超时配置快速修复** (v0.71.34):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 将实时快照总超时增加最小值保护（>=30秒），防止被10秒误截断
- efinance 连接/读取超时改为环境变量可配置：`LAZYBULL_EFINANCE_CONNECT_TIMEOUT_SECONDS`、`LAZYBULL_EFINANCE_READ_TIMEOUT_SECONDS`
- efinance 读取超时默认值提升到30秒，减少慢网络下的过早中断
- 刷新日志新增 `ef_timeout`，直接显示当前生效超时参数

### 当前版本历史 (v0.71.33)

**3.5LCD 排行、行业与指数抓取链路修复** (v0.71.33):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 将排行与行业的持仓周期口径价格从“依赖昨收”改为“昨收缺失时仍可直接使用现价”，修复持仓快照里已有现价但排行为空的问题
- 行业面板不再在缺少实时价时静默回退买入价，避免所有行业被算成 0%
- 新增排行样本日志，直接打印 ts_code、price、pre_close、current_price、buy_price，便于现场核对输入
- 指数涨跌幅改为缓存优先、后台刷新，避免每轮实时快照都被 AKShare 指数抓取拖到超时后重头再来

### 当前版本历史 (v0.71.32)

**3.5LCD 下午数据卡顿根本修复** (v0.71.32):
- 问题根因：实时快照主路径会被慢数据源（efinance/akshare 需 3-5 分钟）阻塞
- 解决方案：缩短快照等待窗口并优先回退缓存数据渲染，避免 UI 长时间卡屏
- 后台线程继续异步获取新数据，缓存回退逻辑已就位，确保不会展示过时数据超过缓存期限
- 结果：即使数据源需要 3-5 分钟，用户也不会看到长时间卡屏，显示更流畅

### 当前版本历史 (v0.71.31)

**3.5LCD 排行无结果诊断与下午数据滞后修复** (v0.71.31):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 排行构建新增详细诊断，区分"匹配成功但计算失败"的原因
- 新增 `pnl_calc_failed` 与 `buy_price_invalid` 统计，并打印失败样本，快速定位排行为空的根本原因
- 将盘中刷新间隔从 300 秒缩短到 120 秒，改善代理环境下的下午数据更新延迟
- 新增盘中刷新触发诊断日志，便于观察是否正在定时更新或触发开盘补齐

### 当前版本历史 (v0.71.30)

**修复树莓派大屏盘中不显示分钟折线图问题** (v0.71.28):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 在快照抓取阶段主动预取上证/深证/中证800实时涨跌幅并写入 `snapshot['index_pct_map']`
- 避免 `_build_intraday_chart` 在 12 秒超时窗口内重复触发慢速指数接口请求，导致日内图长期构建失败

### 当前版本历史 (v0.71.27)

**ContinueDays 改为向后推进，并修复跨时间段稳定性缺失** (v0.71.27):
- [scripts/batch/batch_walk_forward.ps1](scripts/batch/batch_walk_forward.ps1) 的 ContinueDays 展开改为从 FinalDate 起逐日向后推进
- 非交易日继续顺延到最近后一交易日
- 多天展开时自动使用带 final_date 的 batch_period_label，确保 `wf_comparison_batches.xlsx` 的跨时间段稳定性能产生记录

### 当前版本历史 (v0.71.26)

**batch walk-forward 修复多行 Python 内嵌调用的引号解析问题** (v0.71.26):
- [scripts/batch/batch_walk_forward.ps1](scripts/batch/batch_walk_forward.ps1) 不再用 `py -c` 传递多行 Python，而是改为通过标准输入传给 `py -`
- 修复 Windows PowerShell 下中文字符串与多行脚本触发的 `SyntaxError`

### 当前版本历史 (v0.71.25)

**batch walk-forward 改为走项目统一交易日历读取链路** (v0.71.25):
- [scripts/batch/batch_walk_forward.ps1](scripts/batch/batch_walk_forward.ps1) 不再硬编码读取 `data/clean/trade_cal.csv`
- 改为通过项目 `Storage` / `DataLoader` 读取交易日历，兼容当前 `trade_cal.parquet` 存储格式
- 避免因为底层文件格式不是 csv 而导致脚本启动即失败

### 当前版本历史 (v0.71.24)

**batch walk-forward 连续展开结果按最终交易日去重** (v0.71.24):
- [scripts/batch/batch_walk_forward.ps1](scripts/batch/batch_walk_forward.ps1) 对 `ContinueDays` 展开后顺延得到的最终 `FinalDate` 做去重
- 多个候选自然日若映射到同一个交易日，只保留一条任务，避免重复执行

### 当前版本历史 (v0.71.23)

**batch walk-forward 非交易日改为顺延到后一交易日** (v0.71.23):
- [scripts/batch/batch_walk_forward.ps1](scripts/batch/batch_walk_forward.ps1) 对 `ContinueDays` 展开后的候选 `FinalDate`，若命中非交易日，现改为顺延到最近后一交易日
- 避免把训练终点错误回退到更早历史交易日

### 当前版本历史 (v0.71.22)

**batch walk-forward 支持 FinalDate 连续多日展开** (v0.71.22):
- [scripts/batch/batch_walk_forward.ps1](scripts/batch/batch_walk_forward.ps1) 的 `wf_period_configs` 新增 `ContinueDays`
- 同一组时间段配置可从 `FinalDate` 开始按自然日回退展开多个任务，例如 `ContinueDays=2` 会额外包含前一天的 `FinalDate`
- 若展开后的日期命中非交易日，脚本会自动对齐到交易日
- `skip-training` 模式下不同 `FinalDate` 共用同一组 `StartModelVersion` 起点，不再因多日展开而改动模型编号

### 当前版本历史 (v0.71.21)

**3.5LCD efinance 重试间隔下限修复** (v0.71.21):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 对 efinance 快照增加有限重试（总尝试最多2次）
- 重试间隔强制不小于 2 秒，避免短时间连续请求触发上游风控
- 重试日志新增尝试序号与等待时长，便于线上定位

### 当前版本历史 (v0.71.20)

**3.5LCD AKShare 兜底代码口径兼容修复** (v0.71.20):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 新增 `_extract_stock_code6()`，兼容 `sh600000`、`1.600000`、`600000.SH` 等代码口径
- AKShare 兜底过滤改为先提取 6 位代码再匹配持仓，修复“全市场返回有数据但命中0条”的问题
- 当 AK 命中 0 条时，新增 `code_col`、原始样本与未命中样本日志，便于现场快速定位

### 当前版本历史 (v0.71.19)

**3.5LCD 排行匹配增强与诊断日志补强** (v0.71.19):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 新增代码归一化匹配（支持持仓与快照代码在大小写、带不带后缀时自动对齐）
- 排行构建新增匹配统计日志与未命中样本日志，便于快速确认是否为 `SH/SZ` 后缀导致 miss
- 当持仓或快照为空时，输出明确的分支日志，避免仅看到“排行为空”但无法定位

### 当前版本历史 (v0.71.18)

**3.5LCD 盘内个股实时行情改为 efinance 主源 + AKShare 兜底** (v0.71.18):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 新增 `efinance` 个股快照主链路，优先调用 `ef.stock.get_latest_quote(...)`
- 实时持仓快照链路调整为 `E(efinance) -> A(akshare)`，不再走 TuShare 实时兜底
- 上证/深证/中证800 的实时获取链路保持现状不变

### 当前版本历史 (v0.71.17)

**3.5LCD 快照超时配置简化为单变量直读** (v0.71.17):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 仅支持 `LAZYBULL_REALTIME_SNAPSHOT_TIMEOUT_SECONDS`
- 移除 `REALTIME_SNAPSHOT_TIMEOUT_SECONDS` 旧变量兼容读取
- 移除最小 5 秒下限保护，按配置值直接生效
- 启动日志固定输出单变量与默认值说明（`default=60.0s`）

### 当前版本历史 (v0.71.16)

**修复快照超时导致上半区空白，并兼容120秒旧配置名** (v0.71.16):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 兼容读取 `REALTIME_SNAPSHOT_TIMEOUT_SECONDS`（旧）与 `LAZYBULL_REALTIME_SNAPSHOT_TIMEOUT_SECONDS`（新）
- 启动日志会明确打印快照超时配置来源，避免“明明配了120秒却仍按60秒执行”的误判
- 快照超时/异常时自动回退最近有效快照缓存，避免摘要/排行/行业持续为空
- AKShare 快照优先尝试 `stock_zh_a_spot`，并对 `stock_zh_a_spot_em` 的分页进度含义给出日志提示

### 当前版本历史 (v0.71.15)

**3.5LCD 新增分阶段诊断日志（定位上半区空白）** (v0.71.15):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 在刷新主流程记录开始/结束日志，明确 `summary/rank/industry/cycle_chart` 是否更新成功
- 新增 AKShare/TuShare 快照链路日志（开始、失败原因、兜底切换、耗时）
- 新增数据线程调度决策日志，便于判断是“策略未触发刷新”还是“抓数失败”

### 当前版本历史 (v0.71.14)

**3.5LCD 实时快照超时默认提升到 60 秒** (v0.71.14):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 将 `REALTIME_SNAPSHOT_TIMEOUT_SECONDS` 默认值从 18 秒提高到 60 秒
- 新增环境变量 `LAZYBULL_REALTIME_SNAPSHOT_TIMEOUT_SECONDS` 可按现场网络质量调整超时阈值
- 增加最小 5 秒下限保护，防止误配置导致超时过短

### 当前版本历史 (v0.71.13)

**3.5LCD 抓数阶段临时绕开代理（抓完即恢复）** (v0.71.13):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 新增 `_fetch_network_context()`，仅在数据抓取调用期间禁用代理
- 覆盖 AKShare 实时快照/指数与 TuShare 实时/日线抓数，降低代理超时对屏幕刷新的影响
- 支持 `LAZYBULL_FETCH_BYPASS_PROXY` 开关（默认开启绕过）

### 当前版本历史 (v0.71.12)

**树莓派实时快照改为 AKShare 主源（TuShare 兜底）** (v0.71.12):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 的 `_fetch_realtime_holdings_snapshot` 调整为先拉 AKShare 实时行情
- AKShare 不可用或空数据时才回退 TuShare，避免 TuShare 实时接口能力变化导致上半区长期空白
- [tests/test_respi_35lcd_disp.py](tests/test_respi_35lcd_disp.py) 已更新为 `A` 主 `T` 备链路测试

### 当前版本历史 (v0.71.11)

**修复树莓派启动后数据全空（持仓摘要不显示）** (v0.71.11):
- `_build_realtime_portfolio_summary` 在函数内部用 `__file__` 推导 `scripts/` 绝对路径并插入 `sys.path`，修复非项目根目录运行时 `from paper_trade import ...` 静默失败、summary 始终为 None 的问题

### 当前版本历史 (v0.71.10)

**修复“总盈亏有值但年化收益为0.0%”与“更新:--:--”问题** (v0.71.10):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 在轻量快照路径下恢复年化收益函数构建（起始日优先 `account_start_date`，其次 NAV 最早日）
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 实时快照成功后即刷新顶部更新时间，避免摘要失败时长期显示 `--:--`
- [tests/test_respi_35lcd_disp.py](tests/test_respi_35lcd_disp.py) 新增年化计算与更新时间兜底回归测试

**顶部刷新状态支持显示实时数据源标记 `[T]/[A]`** (v0.71.9):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 更新中显示 `更:<步骤>[T/A]`，更新完成显示 `更新:HH:MM[T/A]`
- `T` 表示 TuShare 主源，`A` 表示 AKShare 回退源
- 快照结构新增 `quote_source` 字段并透传到渲染状态
- [tests/test_respi_35lcd_disp.py](tests/test_respi_35lcd_disp.py) 新增来源标记显示与来源断言测试

**修复“抓快照可恢复但数据不更新”问题（新增数据源回退）** (v0.71.8):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 新增 `_fetch_realtime_quotes_akshare(...)`
- 当 TuShare 实时快照异常或空数据时，自动回退 AKShare 持仓行情，继续驱动摘要/排行更新
- 回退数据统一映射到 `TS_CODE/NAME/PRICE/PRE_CLOSE/TIME`，不改变后续显示与计算逻辑
- [tests/test_respi_35lcd_disp.py](tests/test_respi_35lcd_disp.py) 新增快照回退回归测试

**修复树莓派 3.5LCD“更:抓快照”长时间卡住问题** (v0.71.7):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 将盘中刷新间隔恢复为 `90` 秒（此前被改成 `120` 秒）
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 的 `_fetch_realtime_holdings_snapshot()` 改为直接读取账户状态文件，移除每轮初始化 `PaperTradingRunner` 的重链路
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 新增更新状态看门狗，超过阈值自动复位“更新中”状态
- [tests/test_respi_35lcd_disp.py](tests/test_respi_35lcd_disp.py) 新增看门狗复位测试并更新快照路径测试

**优化树莓派 3.5LCD 顶部刷新状态可观测性（步骤名 + 小字体）** (v0.71.6):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 顶部中间状态改为显示实时步骤名（最大 5 个汉字）
- 刷新链路分阶段显示 `抓快照/算摘要/盘中图/算排行/算行业/算调仓/抓周期`，可直接观察卡在哪一步
- 顶部元信息字体从 `15` 下调到 `13`，提升小屏状态栏可读性与容纳能力
- [tests/test_respi_35lcd_disp.py](tests/test_respi_35lcd_disp.py) 新增步骤名显示与结束清空的回归测试

**修复树莓派 3.5LCD 盘中刷新卡住与更新时间滞后问题** (v0.71.5):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 新增 `_call_with_timeout(...)`，为实时快照抓取与盘中图构建增加超时保护
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 的 `_refresh_display_state(...)` 增加 `finally` 兜底，保证刷新结束后 `is_updating` 必然复位
- [scripts/paper_trade.py](scripts/paper_trade.py) 新增 `_extract_latest_quote_time(...)`，顶部更新时间改为使用整批行情里的最新 `TIME`
- 修复效果：外部实时接口偶发阻塞时不再长期显示“更新中”，且“更新:HH:MM”不再因首行旧时间而明显滞后

**公告型 PIT 因子新增 freshness 特征，不再把“过期”硬编码进数据层** (v0.71.4):
- [src/lazybull/factors/announcement_utils.py](src/lazybull/factors/announcement_utils.py) 新增公告型公共 helper，统一处理 `ann_date` point-in-time 查询并输出 freshness
- [src/lazybull/factors/express.py](src/lazybull/factors/express.py), [src/lazybull/factors/fundamental.py](src/lazybull/factors/fundamental.py), [src/lazybull/factors/holder.py](src/lazybull/factors/holder.py), [src/lazybull/factors/fund_portfolio.py](src/lazybull/factors/fund_portfolio.py), [src/lazybull/factors/earnings.py](src/lazybull/factors/earnings.py) 五类公告因子新增 `freshness_days` 列
- [src/lazybull/ml/train_core.py](src/lazybull/ml/train_core.py) 将上述 freshness 列接入显式特征清单，让模型自行学习陈旧公告的折价
- [src/lazybull/features/ensure.py](src/lazybull/features/ensure.py) 将 freshness 纳入 features 缓存 schema 校验，旧的 `cs_train/cs_infer` 文件缺这些列时会自动重建
- 当前语义：旧公告值仍可见，但模型能区分“刚公告”和“很久没更新”的样本

**新增公告因子 freshness 回归测试** (v0.71.4):
- [tests/test_announcement_factor_freshness.py](tests/test_announcement_factor_freshness.py) 覆盖 `express/fundamental/holder/fund_portfolio/earnings` 五类公告因子的 freshness 行为
- [tests/test_ma250_observability.py](tests/test_ma250_observability.py) 新增缓存 schema 测试，验证缺 freshness 的旧 features 文件会自动触发重建

**公告/研报类因子增量补齐升级为“日期区间补齐”** (v0.71.3):
- [src/lazybull/features/ensure.py](src/lazybull/features/ensure.py) 新增 `_incremental_catchup_by_calendar_date(...)`，根据本地最大 `ann_date/report_date` 自动补齐到目标交易日
- `fina_indicator/stk_holdernumber/forecast/express/report_rc` 五类增量路径统一改为区间补齐，不再仅查询 `ann_date=trade_date` 或 `report_date=trade_date` 单点
- 修复效果：周末/节假日公告不会因“仅在调仓日触发单点增量”而长期停留在旧日期

**新增回归测试覆盖公告类区间补齐** (v0.71.3):
- [tests/test_ensure_and_t0_printing.py](tests/test_ensure_and_t0_printing.py) 新增自然日补齐测试，验证会补齐非交易日公告
- 同文件新增参数化测试，覆盖 `fina_indicator/stk_holdernumber/forecast/express/report_rc` 五类增量函数都走区间补齐入口

**修复实时摘要/树莓派显示路径错误回写空账户初始资金** (v0.71.2):
- [scripts/paper_trade.py](scripts/paper_trade.py) 新增 `_create_realtime_runner()`，`run_real` 与 `get_realtime_portfolio_summary()` 统一先读取纸面配置，再按配置初始化 `PaperTradingRunner`
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 的 `_fetch_realtime_holdings_snapshot()` 同步按配置 `initial_capital/position_sizing/horizon` 初始化 runner
- 修复效果：当 `data/paper/config.yaml` 中 `initial_capital` 非 500000 时，实时摘要与树莓派总资产显示不再把空账户现金回写为默认值

**新增回归测试覆盖实时初始资金口径** (v0.71.2):
- [tests/test_paper_trade_realtime_summary.py](tests/test_paper_trade_realtime_summary.py) 新增 `test_get_realtime_portfolio_summary_uses_config_initial_capital`
- [tests/test_respi_35lcd_disp.py](tests/test_respi_35lcd_disp.py) 更新 DummyRunner 初始化签名，覆盖 LCD 快照路径参数透传

**walk-forward 反推切分消除相邻测试区间重叠** (v0.71.1):
- [src/lazybull/ml/walk_forward_utils.py](src/lazybull/ml/walk_forward_utils.py) 的 `generate_walk_forward_splits_by_count(...)` 改为“逐段带上界搜索”：每段 `test_end` 受下一段 `test_start` 前一交易日约束
- 生成结果保证相邻 split 满足 `prev.test_end < next.test_start`，避免 OOS 测试窗口重叠
- [tests/test_walk_forward.py](tests/test_walk_forward.py) 新增相邻测试区间不重叠断言

**walk-forward 输入改为“split数量 + 最终日期”自动反推切分** (v0.71.0):
- [scripts/walk_forward.py](scripts/walk_forward.py) CLI 改为 `--split-count` + `--final-date`，按配置自动反推每个 split 的训练/测试区间
- [src/lazybull/ml/walk_forward_utils.py](src/lazybull/ml/walk_forward_utils.py) 新增 `generate_walk_forward_splits_by_count(...)`，避免末尾无效 split（`test_start > test_end`）
- [scripts/batch/batch_walk_forward.ps1](scripts/batch/batch_walk_forward.ps1) 同步改为 `SplitCount/FinalDate` 配置与新参数透传
- [scripts/compare_walk_forward.py](scripts/compare_walk_forward.py) 增加 `split_count/final_date` 列展示，保持对比输出可读

**walk-forward 切分汇总增加部署训练日期展示** (v0.70.12):
- [src/lazybull/ml/walk_forward_utils.py](src/lazybull/ml/walk_forward_utils.py) 新增 `resolve_deploy_train_window(...)`，统一按交易日对齐部署训练区间
- [src/lazybull/ml/walk_forward_utils.py](src/lazybull/ml/walk_forward_utils.py) 的 `print_splits_summary(...)` 支持追加“部署训练”日期行
- [scripts/walk_forward.py](scripts/walk_forward.py) 在一开始打印切分时同步展示最终部署训练的开始/结束日期

**修复批量纸面交易脚本 PowerShell 未认可动词告警** (v0.70.11):
- [scripts/batch_paper_trade.ps1](scripts/batch_paper_trade.ps1) 将 `Parse-DateText` 重命名为 `Get-DateFromText`
- 同步替换调用点，消除 PSScriptAnalyzer `UseApprovedVerbs` 告警

**修复补位真实执行路径漏掉最小买入后市值阈值** (v0.70.10):
- [src/lazybull/paper/runner.py](src/lazybull/paper/runner.py) 在 `PaperTradingRunner._execute_pending_buys()` 增加最小买入后市值校验
- 阈值口径与其他路径统一：`(总资产 / top_n) * min_buy_value_ratio`，低于阈值则跳过并保留补位计划
- [tests/test_buy_replacement.py](tests/test_buy_replacement.py) 新增回归测试 `test_execute_pending_buys_skip_tiny_buy_value_by_ratio`

**纸面交易新增最小买入后市值阈值，避免碎仓买入** (v0.70.9):
- [src/lazybull/paper/broker.py](src/lazybull/paper/broker.py) 在 `T0 订单买入`、`T1 指令买入`、`补位重试买入` 三条路径统一增加最小买入后市值校验
- [src/lazybull/common/trading_config.py](src/lazybull/common/trading_config.py) 新增 `min_buy_value_ratio` 配置（默认 `0.2`，可通过 `paper_trade.py config --min-buy-value-ratio` 调整）
- 阈值口径：`(总资产 / top_n) * min_buy_value_ratio`，低于阈值则跳过并保留现金

**补位延迟日志简化为短句摘要** (v0.70.8):
- [src/lazybull/paper/runner.py](src/lazybull/paper/runner.py) 移除“示例股票”明细输出，仅保留“候选数 + 原因汇总 + 下次重试”
- 结果：在不丢失关键原因信息的前提下，日志更短、更易读

**纸面持仓表“剩余D”纳入延期持有天数** (v0.70.5):
- [src/lazybull/paper/broker.py](src/lazybull/paper/broker.py) 在 `enable_profit_based_holding=true` 且 `profit_extension_mode!=disabled` 时，将 `profit_extension_days` 计入“剩余D”
- 计算口径从 `rebalance_freq - 持有交易日` 升级为 `rebalance_freq + profit_extension_days - 持有交易日`
- 对应你反馈的场景，超过基础持有期但仍处于延续窗口的持仓，不会再显示为 `剩余D=0`

**补位延迟日志支持分项原因统计（可直接判断是否现金不足）** (v0.70.7):
- [src/lazybull/paper/runner.py](src/lazybull/paper/runner.py) 在补位失败日志中新增原因统计与示例股票，区分“不可交易/无价格/已持仓/当日已买入”等场景
- [src/lazybull/paper/runner.py](src/lazybull/paper/runner.py) 新增 `_analyze_pending_buy_shares_backtest_style()`，对“目标金额不足一手、现金不足、手续费不足、缩量后不足一手”给出明确判定
- 结果：出现“未找到可买入股票”时，可直接从日志确认是否主要由现金约束导致

**纸面补位策略进一步对齐回测（保留原槽位权重 + 回测口径买入金额）** (v0.70.6):
- [src/lazybull/paper/runtime.py](src/lazybull/paper/runtime.py) `_handle_failed_buys()` 不再重算等权补位目标，改为保留失败槽位原始 `target_weight`
- [src/lazybull/paper/runner.py](src/lazybull/paper/runner.py) `_execute_pending_buys()` 在 D-1 重算候选时使用 `universe/exclude_st/min_list_days` 股票池约束，并统一走 `is_tradeable()` 判定
- [src/lazybull/paper/runner.py](src/lazybull/paper/runner.py) 新增 `_estimate_pending_buy_shares_backtest_style()`，按“组合价值×槽位权重”估算买入股数，现金不足时缩量买入

**纸面补位执行改为回测同款"有限候选池逐槽位匹配"** (v0.70.5):
- [src/lazybull/paper/runner.py](src/lazybull/paper/runner.py) `_execute_pending_buys()` 基于上一交易日重算候选池（槽位数×2），逐槽匹配可交易且未持仓的股票
- 排除已持仓、已买入候选，与回测 `_process_position_completion` 逻辑对齐
- 修复"纸面一次性补满所有槽位 vs 回测逐日有限补齐"导致的持仓数量大幅分叉

**纸面 T0 在 `holding_bonus=false` 时完全排除已持仓股票** (v0.70.5):
- [src/lazybull/paper/runner.py](src/lazybull/paper/runner.py) `_generate_ranked_with_lot_constraint()` `holding_bonus=False` 时先剔除已持仓，不再生成补差买单

**纸面持仓表支持交易日口径持有天数与持有剩余** (v0.70.4):
- [src/lazybull/paper/broker.py](src/lazybull/paper/broker.py) 将持仓表“持有天数”改为按交易日计算（不含买入当日）
- 新增“持有剩余”列，按 `rebalance_freq` 实时展示距到期卖出的剩余交易日
- 当交易日历缺失时自动回退自然日口径，保证兼容性

**批量纸面交易脚本输出文案恢复中文** (v0.70.3):
- [scripts/batch_paper_trade.ps1](scripts/batch_paper_trade.ps1) 的注释、日志与错误提示已统一为中文
- 保留已验证稳定的执行逻辑：首日指定日期，后续使用 `trade-date=next` 自动推进

**批量纸面交易改为 next 交易日推进** (v0.70.2):
- [scripts/batch_paper_trade.ps1](scripts/batch_paper_trade.ps1) 采用“首日指定日期，后续 `trade-date=next`”执行模式
- 不再依赖“仅排除周六周日”的近似逻辑，可覆盖节假日/临时休市等非交易日
- 通过 `trade_cal.parquet` 校验下一开市日边界，确保不会超过 `end_date`

**纸面 T0 在 `holding_bonus=false` 时完全排除已持仓股票** (v0.69.18):
- [src/lazybull/paper/runner.py](src/lazybull/paper/runner.py) 在 `_generate_ranked_with_lot_constraint()` 中，`holding_bonus_enabled=False` 时先从候选池剔除 `existing_positions`
- 同方法的一手约束阶段仅在 `holding_bonus_enabled=True` 时允许已持仓直通入选
- 结果：不再给已持仓股票生成“补差买单”，减少你反馈的“9/10 纸面买入数高于回测”现象
- [tests/test_equal_weight_lot_constraint.py](tests/test_equal_weight_lot_constraint.py) 新增回归测试覆盖该行为

**修复 ranked_candidates 漏保存导致的 `ml_raw=0` 持仓评分分叉** (v0.69.17):
- [src/lazybull/paper/runtime.py](src/lazybull/paper/runtime.py) 将 ranked_candidates 持久化条件从“严格信号日期匹配”放宽为“`t0_status=success` 且候选池非空”
- 避免 `_last_signal_date` 类型/格式差异引发 `data/paper/state/ranked_candidates.json` 未落盘，导致次日 `strength` 评分大面积 `raw=0.000`
- 该修复与 v0.69.16 的“strength 按日评估前 ensure 当日特征”配合使用，可同时修复 `raw=0` 与分项退化问题

**修复 9/11 场景下纸面 strength 评分退化导致的回测分叉** (v0.69.16):
- [src/lazybull/paper/runtime.py](src/lazybull/paper/runtime.py) 仅在真实 T0 成功且信号日期匹配时保存 `ranked_candidates`，避免非调仓日被补位流程候选池覆盖
- [src/lazybull/paper/runner.py](src/lazybull/paper/runner.py) 在 `profit_extension_mode=strength` 的按日评估前先 ensure 当日 `cs_infer`，避免 `mom/tech/fund` 因缺特征退化为 `0.50`
- [tests/test_paper_trade_runtime.py](tests/test_paper_trade_runtime.py)、[tests/test_paper_holding_period_alignment.py](tests/test_paper_holding_period_alignment.py) 新增回归测试覆盖上述边界

**修复纸面 T0 买单顺序无序导致的现金受限漂移** (v0.69.15):
- [src/lazybull/paper/runner.py](src/lazybull/paper/runner.py) 中 `_generate_instructions()` 改为按 `targets` 原始顺序生成买入/加仓指令，不再使用无序 `set` 遍历
- 在现金不足时，买单执行先后将稳定可复现，减少纸面在同调仓日出现“持仓数量/成分与回测偏移”的随机性
- [tests/test_ensure_and_t0_printing.py](tests/test_ensure_and_t0_printing.py) 新增回归测试，约束买单顺序与目标顺序一致

**修复纸面交易到期日 early_exit 抢跑导致的回测分叉** (v0.69.14):
- [src/lazybull/paper/runner.py](src/lazybull/paper/runner.py) 中 `evaluate_early_exit()` 对齐回测语义：仅在 `holding_days < rebalance_freq` 的正常持有期内评估亏损提前换出
- 到达/超过持有期后，不再由 early_exit 生成延迟卖出，改由“持有期到期 + 盈利延续”路径统一决策，避免同日双路径冲突
- [tests/test_paper_holding_period_alignment.py](tests/test_paper_holding_period_alignment.py) 新增回归测试，覆盖“到期后即使亏损也不触发 early_exit”边界
- [src/lazybull/paper/runtime.py](src/lazybull/paper/runtime.py) 为 `ranked_candidates` 恢复增加返回值格式校验，异常结构仅告警跳过，避免工作流因解包失败中断

**ensure 构建 cs_infer 与离线 build_clean_features 口径对齐** (v0.69.13):
- [src/lazybull/features/ensure.py](src/lazybull/features/ensure.py) 将 `cs_infer` 构建窗口对齐为 `7个月 warmup + 1个月扩展`，不再使用 1 个月短窗口
- 在 ensure 链路新增 `precompute_daily_adj(...)` 预计算步骤，保持与离线 `cs_train` 构建的特征预热流程一致
- 历史 clean 数据补齐窗口扩展到 warmup 口径（最多 180 个交易日），降低历史缺口导致的技术因子偏移
- 保持唯一差异：纸面交易依然使用 `require_label=False`（不要求 y 标签），其余特征构建流程与离线构建尽量一致

**持仓保留奖励命中后重置持有期起点到 T+1（对齐回测）** (v0.69.10):
- [src/lazybull/paper/runner.py](src/lazybull/paper/runner.py) 在 `holding_bonus_enabled` 且命中留仓时，新增锚点重置：将持仓 `buy_date` 推进到 T+1
- 新增 `_reset_holding_anchor_for_kept_positions()`，并持久化账户状态，避免“延续但持有期未后移”导致的提前到期卖出
- [tests/test_paper_holding_period_alignment.py](tests/test_paper_holding_period_alignment.py) 新增回归测试，约束留仓后锚点重置行为

**纸面交易卖出主路径与回测实现对齐，盈亏统一后复权绩效价口径** (v0.69.9):
- [src/lazybull/paper/runner.py](src/lazybull/paper/runner.py) 将 T0 目标指令生成改为仅买入/加仓，目标减仓/清仓不再直接下卖单
- 卖出统一走持有期/条件触发（到期、盈利延续、提前换出）路径，减少与回测实现偏差
- [src/lazybull/paper/broker.py](src/lazybull/paper/broker.py)、[src/lazybull/paper/models.py](src/lazybull/paper/models.py)、[src/lazybull/paper/storage.py](src/lazybull/paper/storage.py) 新增并持久化 `buy_pnl_price`，保障跨日绩效口径稳定
- [tests/test_paper_holding_period_alignment.py](tests/test_paper_holding_period_alignment.py) 新增同窗口回测-纸面决策对齐与后复权口径回归测试

**纸面交易按日执行持有期到期/盈利延续评估，减少与回测口径偏差** (v0.69.8):
- [src/lazybull/paper/runner.py](src/lazybull/paper/runner.py) 新增 `evaluate_holding_period_actions()`，按交易日评估到期卖出与盈利延续（`pnl/strength/disabled`）
- [src/lazybull/paper/runtime.py](src/lazybull/paper/runtime.py) 将该评估接入 T1 执行链，即使无调仓指令也会执行持有期到期卖出
- [tests/test_paper_holding_period_alignment.py](tests/test_paper_holding_period_alignment.py) 新增回归测试覆盖“非调仓日延续触发”和“无调仓指令仍可到期卖出”

**修复纸面交易 T0 路径 protected_stocks 未定义回归** (v0.69.7):
- [src/lazybull/paper/runner.py](src/lazybull/paper/runner.py) 为 `_generate_signals()` 增加 `protected_stocks` 参数并在 `run_t0()` 中透传
- 修复 `T0 执行失败: name 'protected_stocks' is not defined`，恢复 T0 正常执行

**钉钉机器人交易结果新增盈利延续保护名单展示** (v0.69.6):
- [src/lazybull/paper/runtime.py](src/lazybull/paper/runtime.py) 将 T0 阶段命中的 `protected_stocks` 纳入共享执行结果
- [src/lazybull/paper/reporting.py](src/lazybull/paper/reporting.py) 在钉钉 Markdown 中新增紧凑的 `盈利延续保护` 区块，明确列出被保护保留的股票
- 为适配钉钉消息长度限制，最多展示前 8 只，超出时只追加余量提示

**T0 目标详情现与盈利延续保护后的最终指令对齐** (v0.69.5):
- [src/lazybull/paper/runner.py](src/lazybull/paper/runner.py) 在打印 `T0 建仓目标详情` 时，会识别 `protected_stocks`，对已触发盈利延续保护的持仓显示为 `保留`
- 不再把这类股票误显示为 `清仓/减仓`，减少“日志显示要卖、实际指令不卖”的误导
- [tests/test_paper_trading.py](tests/test_paper_trading.py) 新增回归测试，验证受保护股票展示为 `保留`

**纸面交易 strength 路径特征读取接口修复（避免 AttributeError）** (v0.69.4):
- [src/lazybull/paper/runner.py](src/lazybull/paper/runner.py) 将两处错误的 `load_features_by_date` 调用改为现有 `load_cs_train_day(trade_date, subdir="cs_infer")`
- 修复纸面交易执行到“T0 -> 计算盈利延续保护”时，可能抛出的 `AttributeError: 'Storage' object has no attribute 'load_features_by_date'`
- [tests/test_paper_trading.py](tests/test_paper_trading.py) 新增回归测试，确保 Storage 仅提供 `load_cs_train_day` 时强势度评分仍可正常执行

**纸面交易盈利延续 strength 命中时改为 warning 日志** (v0.69.3):
- [src/lazybull/paper/runner.py](src/lazybull/paper/runner.py) 在 `profit_extension_mode=strength` 且强势度达到阈值时，改为输出 `warning` 级别日志
- 日志内容会明确显示 `强势度`、`阈值`、`pnl` 和强势度分解，便于快速识别“本该到期卖出但因盈利延续被保护”的持仓
- [tests/test_paper_trading.py](tests/test_paper_trading.py) 新增回归测试，验证 warning 日志输出

**batch walk-forward 新增 MA250 阈值/仓位参数扫描并写入对比 xlsx** (v0.69.2):
- [scripts/batch_walk_forward.ps1](scripts/batch_walk_forward.ps1) 将 `market_regime_ma250_threshold`、`market_regime_ma250_exposure` 升级为 `_list` 扫描参数，并纳入总任务数与批量组合循环
- 运行批量实验后，输出的对比表会保留 `MA250阈值` 与 `MA250仓位` 两列，便于直接在 [data/walk_forward/wf_comparison.xlsx](data/walk_forward/wf_comparison.xlsx) 与批次总表中筛选

**compare 跨时间段稳定性汇总去重修复** (v0.69.1):
- [scripts/compare_walk_forward.py](scripts/compare_walk_forward.py) 在“跨时间段稳定性”聚合前，先按 `参数组 + 时间段标签` 去重，并保留每个时间段最新 `wf_run_id`
- 修复 [data/walk_forward/wf_comparison_batches.xlsx](data/walk_forward/wf_comparison_batches.xlsx) 中同一时间段重复堆叠导致 `时间段数` 异常放大的问题
- [tests/test_ma250_observability.py](tests/test_ma250_observability.py) 新增回归测试，约束同时间段重复 run 只保留最新一条

**树莓派 3.5 寸 LCD 行业页计数口径调整（平盘计入正收益）** (v0.68.8):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 行业页顶部 `正/负收益股票数量` 中，`0` 收益股票改为计入正收益（`+`）
- 行业内每行 `+x/-y` 的计数规则同步调整为平盘计入 `+`
- [tests/test_respi_35lcd_disp.py](tests/test_respi_35lcd_disp.py) 新增平盘计入正收益回归测试

**compare 汇总时避免全 NA 条件列触发 pandas FutureWarning** (v0.68.9):
- [scripts/compare_walk_forward.py](scripts/compare_walk_forward.py) 在合并多个 walk-forward summary CSV 前，会先按单个 frame 排除全 NA 列，再在拼接后按原列顺序补回
- 这样保留了当前 compare 输出语义，同时修复批量扫描历史 batch 时 `pd.concat(frames)` 重复输出 FutureWarning 的问题
- [tests/test_ma250_observability.py](tests/test_ma250_observability.py) 新增回归测试，约束局部全 NA 条件列参与汇总时不再告警

**walk-forward / compare 参数展示按实际生效状态清洗，避免默认值误导** (v0.68.8):
- [scripts/walk_forward.py](scripts/walk_forward.py) 在写 summary CSV 时，会根据 OOS 总开关、父开关和模式开关清空未生效子参数
- [scripts/compare_walk_forward.py](scripts/compare_walk_forward.py) 在读取历史 raw summary 时也会执行同一套清洗，重新生成 compare 表即可纠正旧 batch 的误导性默认值
- 修复 `enable_profit_based_holding=false` 却显示 `profit_extension_mode=pnl` 一类记录
- 同时修复 `signal_gate_mode=composite` 下 `signal_confidence_gate_top_k` 被误清空，以及信号门控失活时 `dynamic Top-N`、回撤保护关闭时 `drawdown_threshold` 仍显示的问题
- 修复 `market_regime_mode=binary` 下 `drawdown_guard` 被误清空的问题
- [tests/test_walk_forward.py](tests/test_walk_forward.py) 新增/更新回归测试覆盖上述场景

**树莓派 3.5 寸 LCD 行业页贡献口径与零值配色优化** (v0.68.7):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 行业贡献比例新增口径标记：盘外 `cycle_total_pnl`、盘内 `intraday_total_pnl`
- 行业页在盘内模式下明确按当日总盈亏计算贡献比例，盘外模式继续按持仓周期总盈亏计算
- 行业页 `0` 值（行业名/明细数值）由白色调整为浅灰，降低“0”视觉突兀感
- [tests/test_respi_35lcd_disp.py](tests/test_respi_35lcd_disp.py) 新增对应口径与配色回归测试

**树莓派 3.5 寸 LCD 行业页补齐盘内/盘外双口径统计** (v0.68.6):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 行业统计新增双口径：盘外继续按持仓周期（买入成本）计算，盘内改为按当日口径（昨收）计算
- 刷新阶段同时缓存两套行业面板，渲染阶段按图表当前模式自动选择，确保行业页与折线图口径一致
- [tests/test_respi_35lcd_disp.py](tests/test_respi_35lcd_disp.py) 新增盘内/盘外基准价差异回归断言，并更新刷新状态双口径断言

**树莓派 3.5 寸 LCD 行业页表格行高与分页时长修复** (v0.68.5):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 修复行业表格最后一行高度偏小问题，统一为等高行
- 多页行业展示时长改为按每页行业数量占比自动分配（例如 `8:4 -> 20秒:10秒`）

**树莓派 3.5 寸 LCD 行业页布局与统计口径再优化** (v0.68.4):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 页码移到首行最右并缩小字号
- 行业表格改为 2 列 × 4 行，并放大表格内字体
- 行业统计改为按申万 L1 聚合
- 首行左右统计区间距压缩，提升同屏信息密度

**树莓派 3.5 寸 LCD 行业页与顶部资源条显示样式进一步优化** (v0.68.3):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 图表页/行业页轮播时长调整为 `30秒/30秒`
- 行业列表按收益从高到低排序（收益低的排在后）
- 行业首行统计增加底色区分；行业区新增轻量表格边框并加粗中间分栏线
- 行业名可显示长度从 10 字提升到 12 字
- 顶部 CPU/内存进度条移除边框

**树莓派 3.5 寸 LCD 行业页改为 2 列 × 5 行紧凑彩色展示** (v0.68.2):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 行业页每页固定显示 10 个行业（2 列 × 5 行），并适度增大字号
- 每个行业仅显示紧凑数值：`+正收益数量/-负收益数量/+贡献比例`，减少中文占位
- 数值按正负实时着色：正红、负绿、零白
- 首行显示 `行业1/2/3:<L1>/<L2>/<L3>` 与 `正/负收益股票数量:+x/-y`，并对正负数量着色

**树莓派 3.5 寸 LCD 行业页改为全行业汇总分页显示** (v0.68.1):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 行业页不再展示行业内 TOP1/BOTTOM1 个股，改为仅展示每个行业的 `正收益数量 / 负收益数量 / 贡献比例`
- 当持仓行业较多时，行业页会在 20 秒窗口内自动分页轮播并显示页码，确保可覆盖全部持仓行业
- 行业统计构建同步移除 TOP 个股计算，减少不必要开销

**树莓派 3.5 寸 LCD 图表区新增“图表页/行业统计页”轮播与进度条** (v0.68.0):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 图表区按 `40秒(折线图) : 20秒(行业统计)` 自动轮播，仅切换图表区域内容
- 图表区顶部新增 3 像素无边框进度条，实时展示当前页时长进度，走满自动切页
- 行业统计页展示持仓行业清单、各行业正负收益数量、行业内正负收益 TOP1、行业对总收益贡献比例，以及全持仓正负收益股票数量
- 行业分类严格使用申万行业映射并按 `industry.shenwan_level` 动态选择层级；行业名颜色按贡献收益实时变化（正红/负绿/零白）

**树莓派 3.5 寸 LCD 图例顺序与中证800日线残留逻辑已清理** (v0.67.6):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 折线图图例顺序调整为 `上 / 深 / 中 / 持`
- 删除未被调用的中证800 AKShare 日线函数与解析函数，避免误导为仍存在日线兜底链路
- 脚本已通过语法编译检查，3.5LCD 回归测试通过

**树莓派 3.5 寸 LCD 实时指数链路已改为仅使用新浪接口** (v0.67.5):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 上证/深证实时获取仅调用 AKShare `stock_zh_index_spot_sina`
- 中证800实时获取同样仅调用 `stock_zh_index_spot_sina` 并匹配 `sh000906`（兼容 `000906` / `000906.SH`）
- 已移除 `stock_zh_index_spot_em` 与 `stock_zh_index_spot` 的实时兜底链路

**树莓派 3.5 寸 LCD 的中证800实时接口已切换为新浪链路优先** (v0.67.4):
- [scripts/respi/3.5LCD_disp.py](scripts/respi/3.5LCD_disp.py) 中证800盘中实时涨跌幅现在优先使用 AKShare `stock_zh_index_spot_sina()`，并按 `sh000906`（兼容 `000906` / `000906.SH`）匹配
- 若新浪实时接口在当前环境不可用，会自动回退到 `stock_zh_index_spot`，保持兼容
- [tests/test_respi_35lcd_disp.py](tests/test_respi_35lcd_disp.py) 已同步修正实时快照断言，允许中证800在快照缺失时由补抓路径返回

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
  - **龙虎榜（lhb）**：`top_list` 个股日频上榜数据，同日多次上榜优先取单日榜（净买入绝对值最大的一条，仅连续类理由时保留），按交易日历重采样计算近 5/20 日滚动净额与 20 日上榜次数（未上榜日保留历史累计衰减）；新增 `lhb_cont_on_list`（当日是否因"连续异动"类理由上榜，reason 含"连续"，事件级信号）及 `lhb_cont_up_days_5/20`（近 5/20 交易日连续异动上榜次数累计）；近期空响应不落盘延迟重试、近期空占位自动重新查询防假空
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

**回测**（`scripts/run_ml_backtest.py` 已删除；回测已并入 `walk_forward` 滚动回测，或经 `src.lazybull.common.backtest_runtime` 工厂驱动 `BacktestEngineML`）:

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

# 风控公告类数据（质押/解禁/大宗，供风控模型专用因子使用）
python scripts/download_raw.py --start-date 20230101 --end-date 20231231 --download pledge_stat share_float block_trade

# 步骤2: 构建clean和features（假设raw已存在）
# --horizon / --horizons 二选一必填：
#   --horizon 20         : 单值模式，仅按主 horizon 对应的 y_ret_20 非空过滤（推荐，保留停牌导致的辅助标签缺失样本）
#   --horizons 5 10 20   : 多值模式，AND 过滤，要求所有 horizons 对应 y_ret_N 同时非空
# 两种模式下生成的特征文件都包含 y_ret_5/10/20 三列，schema 一致
python scripts/build_clean_features.py --start-date 20230101 --end-date 20231231 --horizon 20

# 启用风控公告类因子（质押/解禁/大宗，需先下载 pledge_stat/share_float/block_trade）
python scripts/build_clean_features.py --start-date 20230101 --end-date 20231231 --horizon 20 --enable-announcement-risk-features

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
# 注意：scripts/run_ml_backtest.py 已删除，回测已并入 walk_forward 滚动回测，
# 或经 src.lazybull.common.backtest_runtime 工厂驱动 BacktestEngineML
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
