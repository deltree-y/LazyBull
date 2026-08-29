# Changelog

All notable changes to this project will be documented in this file.

## [0.98.3] - 2026-08-29

### Fixed

- **分红原始数据脏日期容错**：合并后在去重与分年落盘前校验 `ann_date`，缺失或非法记录按数量及股票样例告警后剔除，避免生成 `None-12-31` 非法分区并中断全市场下载。
- **分红合并告警降噪**：raw 历史分区与新下载结果合并时，仅精确屏蔽 pandas empty/all-NA dtype 推断 `FutureWarning`，不删除全空列或改变原始 schema。

## [0.98.2] - 2026-08-29

### Added

- **利润表归母净利润链路**：新增 TuShare `income_vip` 合并报表下载、季度分区加载与纸面交易自动补齐；版本键使用 `(ts_code, end_date, f_ann_date)`（缺失回退 `ann_date`），支付率只读取 `n_income_attr_p`，不再混用现金流量表 `net_profit`。

### Fixed

- **分红事件与缺失语义（schema v3）**：首次实施公告至除息前不再改变 continuity/yield/history 状态；`dividend_days_to_ex_date` 改为自然日 `0~30`，成熟股票 30 日内无已公告除息事件显式编码为 `31`，避免被 60% 缺失门禁静默剔除。
- **训练事件列有效性**：两个稀有事件因子改用原始稠密值入模，避免全市场同值时 zscore 因零方差整列 NaN；稳定哨兵 `dividend_schema_v1` 升为 `3`。
- **可选因子缓存契约**：现金流质量、一致预期修正、分红政策三组 schema 按实际构建开关校验；关闭的组不再误伤基础缓存，分红组开启时完整校验原始输出、训练列、freshness、缺失标记与当前哨兵。
- **分红日增量门控**：逐股历史覆盖完成后不再以全市场总行数决定是否执行日增量；低于 3 万行的完整库也会按 `ann_date` 与 `imp_ann_date` 推进自然日水位，避免新实施公告长期停留在旧快照。

### Changed

- **分红构建性能**：缓存日期预筛前移到数据加载前，离线 lookup 只物化待构建日期；股票日期计算按 64 日分块向量化，并消除逐股 pandas 排序、日期转换和 DataFrame 构造。同机 `1000×250` 基准由 3.254 秒降至 0.549 秒，5000 股单日由 3.793 秒降至 1.090 秒。纸面 dividend 数据复用预加载 DataFrame，消除重复全分区扫描。
- **income 预热单一职责**：批量与纸面调用方仅传目标日期区间，统一由 `load_income` 默认回看六年，避免调用方先减六年后 loader 再减六年而读取约十二年分区。

### Migration

- 启用分红政策因子的训练区间需先执行 `python scripts/download_raw.py ... --download income --force`，再强制重建 `cs_train` / `cs_infer` 并重训；schema v2 模型不可与 v3 特征混用。

## [0.98.1] - 2026-08-29

### Fixed

- **分红年度状态即时生效（schema v2）**：已实施正分红在 `ex_date` 当日进入连续性、稳定性与增长率窗口；尚未成熟财年不提前记零，财年在次年 9 月 1 日成熟后才将无实施记录解释为停发。稳定哨兵列 `dividend_schema_v1` 的当前值升为 2，旧特征缓存必须重建。
- **首次分红缺失语义**：仅已发生 `ex_date` 的事件构成分红历史；首次实施公告至除息前继续保持 `dividend_hist_missing=1`（上市满一年），不会被事件行提前翻转。
- **逐股下载覆盖状态**：新增 `raw/dividend/_stock_coverage.json`，原子持久化 `data/empty/pending/failed`；成功空结果不再重复请求，失败或进程中断保持可重试。
- **`--force` 权威替换**：按股全历史查询成功后整体替换该股票旧行并清理空年度分区；失败股票保留旧行且标为 `failed`，后续非 force 自动重试。ensure 使用补齐后的数据继续判断，不再在同次运行重复全量请求。
- **回归保护**：补充 force 部分失败、失败重试、成功空结果、旧分区清理、年度分区范围加载、除息即时生效与首次分红状态测试。

## [0.98.0] - 2026-08-29

### Added

- **分红政策质量因子组（Roadmap Phase 3 红利因子）**：新增 `dividend` 数据接入与 8 个因子：
  - 状态因子（可用日统一 `ex_date`，仅 `div_proc=实施` 行入因子）：`dividend_continuity_5y`（近 5 个已实施财年正分红年占比，窗口以最新可见年度为锚、上市前年份不计分母）、`dividend_stability_5y`（年度 DPS 的 1-MAD/中位数，样本≥3 且中位数>0）、`dividend_growth_3y/5y`（每股调整 DPS 有界对称增长率）、`dividend_payout_ratio`（现金分红总额/Q4 归母净利润，PIT 双可见）、`dividend_yield_hist_12m`（近 12 个月每股现金分红/未复权收盘价）；
  - 事件因子（双日期市场效应）：`dividend_days_to_ex_date`（距最近**已公告**未除息事件天数，`imp_ann_date ≤ T` 可见、缺失回退 ann_date、clip 30 日）、`dividend_recent_imp_ann_10d`（近 10 交易日实施公告计数，纯回看窗口 [T-9, T]，公告只影响发布日及之后）；
  - 附列：`dividend_freshness_days`（注册状态型 freshness）、`dividend_hist_missing`（从未分红=1/有历史=0/上市不足一年=NaN，**随开关显式入模**不进全局缺失标记列表）。
- **每股调整口径（PIT 截断）**：事件调整基数 `base_i = cash_div_tax_i × Π_{k<i}(1+stk_div_k)`，T 日口径除以 `G(T) = Π_{k: ex_k<=T}(1+stk_div_k)`（仅含 T 前已发生送转，未来送转不影响历史截面）；比率类因子与共同因子无关直接用 base。
- **数据链路**：TuShare `dividend` 接口接入（按股全历史查询 + 断点续传 + 失败股票告警清单）；raw 按 `ann_date` **年分区**存储（分区模式按枚举加载，与 report_rc 契约一致）；`download_raw.py --download dividend`；纸面 ensure 自动补齐（水位单日增量**同时查询 ann_date 与 imp_ann_date**，避免漏掉实施状态更新）。
- **单开关与哨兵**：`--enable-dividend-policy-features`（默认关，接入 `batch_walk_forward.ps1`）；哨兵列 `dividend_schema_v1` 由 handler 对当日全截面恒写，训练入口 fail-fast 校验缺失/NaN/版本不符；模型 metadata 记录开关与 schema 版本；skip-training 校验与旧模型加载告警四层兜底。
- **训练特征列**：行业中性 `zscore_dividend_*` 8 列 + `dividend_freshness_days` + `dividend_hist_missing`（`DIVIDEND_POLICY_FEATURE_COLUMNS`）。
- **单日/批量一致性**：`build_dividend_lookup_by_date` 支持 `calendar_dates`（完整预热日历），单日推理的近 10 交易日回看窗口与批量构建口径一致。

### Changed

- `features/ensure/factor_load.py` 因子组计数 15 → 16；`ensure/schema.py` 必检列新增分红哨兵与 freshness（旧特征缓存自动重建）。

### Removed

- 首轮移除 `dividend_cash_ratio`（现金总额"元"与送转股数"股"量纲不可比，审查意见）。

### Fixed

- **送转调整方向与 PIT 截断**：改为前复权式当前股本口径 `D_i × G_{i-1} / G_T`（送转后历史每股分红缩小；修复前方向相反）；G(T) 仅含 T 前已发生送转，未来送转不影响历史截面。
- **同财年多次分红事件级可见**：年度聚合不再取最大 ex 作为可见边界，改为事件级前缀分段，未来第二次分红不污染历史截面。
- **成熟财年窗口**：财年 y 成熟于 (y+1)-09-01；窗口右端按 T 推算，停发年份作为 0 进入窗口（连续性随停发下降），未实施上年分红成熟前不入窗口。
- **下载失败与 force 语义**：`--force` 部分失败时旧分区数据保留不删；ensure 每日先补未覆盖股票（失败股票未写分区自动重试）再日期增量。
- **年分区范围加载**：`load_dividend(start,end)` 改为按年枚举分区后按 `ann_date` 过滤（按日分区接口会漏读 `YYYY-12-31` 年度分区）。

## [0.97.1] - 2026-08-29

### Removed

- **清理 ECT 遗留接口与文档**：`_execute_t0_if_rebalance_day` 返回签名由 5 元组收窄为 3 元组，移除 ECT 下线后遗留的 `_dummy_exposure`/`_dummy_reason` 兼容占位；删除 `tests/test_paper_trading.py` 中"盈亏动态持仓已移除"的残留桩测试残躯；README 移除已下线的"权益曲线交易（ECT）"功能特性说明。

## [0.97.0] - 2026-08-29

### Added

- **纸面分批排期追赶与状态迁移**：`rebalance_state` 分离记录实际执行日和已履行计划日；漏过计划日时逐日补执行最早未完成批次，`rebalance_freq` 或 `stagger_tranches` 变化时从当前 T0 重建 anchor。T0 运行记录持久化完整排期快照，手工 truncate 可恢复 anchor、批次和计划进度。
- **分批参数强校验**：统一拒绝 `stagger_tranches > rebalance_freq`、`stagger_tranches > top_n` 及无法形成满仓组合的 `max_weight_per_stock * top_n < 1` 配置，避免排期覆盖、空批次和隐式超限。

### Fixed

- **纸面/OOS 分批买入全链路对齐**：一手预筛按总 `top_n` 的单槽预算并计入现金保留比例；批内权重保存为全组合权重，失败补位不再被批次数放大；新批次排除已预留补位标的，补位队列只清理已成功转写的槽位。
- **T1 买入失败同日顺延**：涨停、停牌、价格或资金约束导致首选买入失败时，纸面交易与 OOS 均按 T0 原始排名在同一 T1 继续尝试下一候选；仅同日仍未成交的槽位进入后续补位队列，卖出失败继续延期而不释放虚假槽位。
- **组合约束跨批一致**：行业上限同时计入已有持仓和预留买单；单股上限按本批资金占比换算后在纸面/OOS 两侧统一应用，严格限权且保持候选顺序。
- **重试状态持久化**：交易指令新增补位槽位身份，延迟卖出队列补存 `last_attempt_date`，重启后仍能避免同日重复推进尝试次数。

## [0.96.4] - 2026-08-28

### Fixed

- **ensure 因子补齐链路 FutureWarning 刷屏**：纸面交易 ensure 自动补齐路径的 5 个模块（bulk/downloads/incremental/historical/historical_assets）共 11 处裸 `pd.concat` 未沿用项目既定屏蔽模式，TuShare 稀疏财务列（部分季度整列为空）在分页合并与新旧合并时触发 pandas empty/all-NA FutureWarning 大量刷屏；新增共享辅助 `_concat_no_warning`（ensure/concat_utils.py）统一替换，与 scripts/raw_download/periodic.py、data/loader.py 既有口径一致——只屏蔽告警，不对数据做任何剔除/补列，raw 层 schema 保持不变。

### Changed

- **现金流修订刷新并发化**：`_refresh_cashflow_revisions_if_due` 的逐季度串行循环改为并发执行，worker 数直读 `tushare.download_concurrency`（为 1 或仅单季度时保持串行路径，与 scripts/raw_download 的 `_run_concurrent` 降级口径一致）；各季度读写各自独立分区文件（客户端令牌桶限频线程安全、存储按目标文件派生临时名原子替换），逐季结果统一汇总后仍按原逻辑门控两个水位（部分失败不推进）。每 90 天一次的全历史复查（86 个季度）由约 14 分钟降至约 1 分钟级，并新增每 20 季度的进度日志与总耗时输出，消除长时间静默。

## [0.96.3] - 2026-08-28

### Fixed

- **现金流质量因子全链路审计修复（schema v3）**：
  - **版本化 PIT**：数据可用时间改取 `f_ann_date`（实际公告日，缺失回退 `ann_date`），去重键改为 `(ts_code, end_date, f_ann_date)`，同报告期修订按版本保留、按交易日选择当日可见最新版本。修复前修订值被回填到原始公告日（实测 002111.SZ 2020 年报修订值 784M 在 2021-04-20 即被特征使用且 freshness=0；全量 6,105 行 `f_ann_date > ann_date`，中位延迟 232 天）；
  - **事件驱动 TTM**：全部因子改为滚动四季度口径；当前期、去年同季度或去年 Q4 任一版本可用时都会重算当前期快照，修复依赖期晚到修订后 TTM 仍冻结旧值及 freshness 不重置；
  - **供应商口径自由现金流**：`fcf`/`fcf_yield` 改用 TuShare `free_cashflow` 字段（TTM），不再使用本地 OCF−|capex| 代理混同口径；
  - **数值稳定性**：分母经济尺度下限（1000 万元）+ 比值有界裁剪（capex_to_ocf ±50、ocf_to_revenue ±10、ocf_to_profit ±50、fcf_yield ±1 且总市值分母下限 1 亿元），替代仅 1e-6 元的弱过滤（修复前 raw 层 capex_to_ocf 最大 197,303）；
  - **确定性版本去重**：同一 `(ts_code, end_date, f_ann_date)` 内冲突行优先按 TuShare 官方 `update_flag=1` 最新语义选择；标志相同或缺失时以全行内容哈希稳定决胜。离线下载、ensure 合并与因子构建复用同一工具，消除供应商 FCF 随输入行序变化；
  - **加载与下载完整性**：现金流默认预热由一年改为两年，真实 `20260105` 单日 OCF 非空覆盖由 26 只恢复到 5,689 只；季度和公告日查询均按接口上限 6400 分页；近 8 季度每日刷新，首次升级及之后每 90 天全历史复查，空响应或部分失败不推进水位；
  - **fail-fast 与哨兵**：`enable_cashflow_quality_features=True` 时训练入口缺列直接失败；稳定列名 `cashflow_quality_schema_v2` 的当前值升为 3，ensure 与训练入口均校验版本，强制旧语义分区重建；
  - **训练/推理追溯**：WF summary 的 `cashflow_quality_cols_live` 记录门禁后实际入模列并包含 freshness；split/deploy/单次训练记录 schema 与实际列；普通模型加载和 skip-training 均告警旧现金流模型的 train/serve 语义偏差。

### Migration

- 离线启用现金流质量因子前，对所用历史范围执行一次 `python scripts/download_raw.py --start-date <训练起点前至少两年> --end-date <目标日> --download cashflow --force`，随后重建 `cs_train`/`cs_infer`。
- 纸面 ensure 首次运行会自动重查全部现有 cashflow 分区，之后每 90 天深度复查；近期 8 个季度仍每日刷新。
- schema v2 及更早缓存和模型与 v3 同名异义，必须重建特征并重训；哨兵值或模型 metadata 版本不符会失败或告警。

## [0.96.2] - 2026-08-27

### Docs

- **README 与 CHANGELOG 分工**：移除 README.md 中积累的全部版本 changelog 内容，README 仅保留项目介绍、功能特性、快速开始、使用指南等文档内容；所有版本变更统一记录在 CHANGELOG.md。项目共识（.github/copilot-instructions.md、CLAUDE.md）已同步约定：README 不包含 changelog，禁止再向 README 追加版本变更记录。

## [0.96.1] - 2026-08-27

### Changed

- **一致预期修正因子 v2 重做（经济语义修复）**：`consensus_revision.py` 全面重写，修复全链路审计发现的语义缺陷：
  - EPS 源列改回 `eps`（v1 误用 `np` 净利润），并按**绝对预测财年**分组（锚定报告年的 FY1 与 FY0 按覆盖报告日数多者优先、持平取 FY1），不再混合多预测期，也杜绝跨年报告的不同 FY1（如 2025 与 2026 财年）混入同一序列；
  - `cons_eps_dispersion` 改为同日同财年研报级分歧度的时间平均（v1 先按报告日聚合，衡量的是预测随时间波动而非分析师分歧）；
  - `cons_eps_revision_accel` 改为按报告日真实日历天数的一阶斜率（v1 按研报行序号，初版 v2 误用 yyyymmdd 整数回归）；
  - `cons_rating_upgrade_ratio` 真实读取 `rating` 列（v1 是目标价变化 2% 的 0/1 别名）；
  - 删除与基础因子重复的 `cons_revision_target_upside` / `zscore_cons_revision_target_upside`；
  - 基线指标（目标价/覆盖/评级/分歧度变化）统一为近 30 日 vs 此前 90 日，且两侧窗口在第 -30 日不重叠；
  - 研报级与身份去重数组显式按报告日升序，杜绝乱序分区导致未来研报进入历史窗口（PIT 前视）；
  - 输出按当日股票截面 1%/99% winsorize，降低下游 Z-Score 被极值牵引的风险。
- **哨兵列强制旧缓存重建**：新增 `cons_revision_schema_v2` 列，由 handler 对当日全截面恒写当前版本号（含无修正数据的股票）；ensure schema 必检清单与训练入口均严格校验“哨兵列存在且全量等于当前版本”，旧语义分区或哨兵缺失/NaN 直接失败。
- **存活列可追溯**：WF summary 新增 `consensus_revision_cols_live` 记录每个 split 门禁后实际存活列（含 `zscore_cons_*_sz` 市值中性化派生列）；正常训练透传 `feature_columns`，skip-training 从独立 `v*_features.json` 补齐；skip-training 复用旧模型时核验开关与 schema 版本一致性并告警（异常版本值安全处理，不中断 split 循环）。
- **schema 版本全链路记录**：split/deploy/单次训练三个注册入口统一记录 `cons_revision_schema_version`；`ModelRegistry.load_model` 严格加载时对含修正列但未记录 v2 版本（v1 旧语义模型）告警 train/serve 语义偏差风险。

### Fixed

- 目标价与评级聚合改为研报级（同研报多预测期行不重复加权），覆盖计数与数值窗口口径统一；split 结果透传 `feature_columns`，修复存活列汇总始终为空。

### Migration

- v0.96.1 升级后，启用 `--enable-consensus-revision-features` 的区间必须重建 `cs_train` / `cs_infer`（旧缓存缺哨兵列会自动触发 ensure 重建）；无需重下 raw 数据。
- **旧模型风险提示**：v1 时期的旧模型（如 v22715）注册的 `zscore_cons_*` 列与新构建分区同名，继续推理时会静默读到 v2 语义数值，存在 train/serve 语义偏差；建议停用旧模型或以同一重建后的特征重新训练。

## [0.96.0] - 2026-08-26

### Added

- **一致预期经济归一化特征**：新增 FY-1 EPS 信息及 `cons_eps_yield_fym1/fy0/fy1/fy2`（预测 EPS / 当日未复权收盘价）、`cons_target_upside`（目标价中值 / 收盘价 - 1）；训练候选切换到可跨股票比较的归一化列，旧 EPS/目标价绝对值列继续产出，仅供存量模型兼容推理。
- **report_rc 统一身份契约**：以 `ts_code + report_date + org_name + author_name + report_title` 标识唯一研报，追加 `quarter` 标识唯一预测行；离线批量、在线 ensure 与因子覆盖计数共用同一契约。

### Fixed

- **report_rc 下载完整性闭环**：ensure 单日查询与历史年度回补均按接口上限 2000 行分页，历史年度累计查询触发 10 万行上限时自动按日期二分；回补窗口以目标 `trade_date` 为锚取近六年，避免历史回放误拉机器当前年份。批量下载对当前年度已有分区从最大 `report_date` 次日续传并合并写回，不再因年分区存在而全年断更。
- **研报覆盖数与全聚合去重**：`cons_analyst_count_30d` 及 `cons_analyst_count_chg` 改为唯一研报数，同一研报 FY0/FY1/FY2 多条预测行不再放大覆盖数；基础与修正聚合入口均按统一预测行键去重，标题或作者不同的真实研报不会被旧四键误删。
- **旧 report_rc schema 明确失败**：DataLoader、公共去重、因子入口与离线/ensure 写入边界均要求完整研报身份列，缺少机构、作者、标题或预测期时直接提示 `--download report_rc --force`，不再静默退化为弱去重键。
- **评级缺失语义**：补齐 `跑赢行业`、`优于大市`、`Overweight`、`买进`、`BUY` 等映射并统一英文大小写；`无`、空值和未知文本保留 NaN，不再伪造中性 3 分。
- **一致预期状态与修正稳健性**：基础与修正状态均以最新研报为锚聚合 90 日，最多保留 365 日并交由 freshness 指数衰减，消除第 90 天硬断崖；`cons_eps_revision_30d` 保持全预测期口径，改为近 30 日至少 2 个报告日对比此前 90 日至少 3 个报告日的日度中值，并使用有界对称变化率消除小分母爆炸。
- **基础/修正目标价口径隔离**：修正因子改用独立的 `cons_revision_target_upside` / `zscore_cons_revision_target_upside`，分母统一为当日未复权收盘价，不再覆盖基础 `cons_target_upside` 或在训练/推理间切换为复权价。
- **训练开关强契约与元数据**：`enable_consensus_features=True` 与 `enable_consensus_revision_features=True` 均要求对应构建 schema 完整，缺列立即提示重建特征，不再静默少训；split/deploy 模型 `train_params` 均记录全部可选特征开关，并由注册入口测试覆盖。

### Migration

- v0.96.0 首次升级必须对目标训练区间以 `--download report_rc --force` 重下 `raw/report_rc`：旧四键曾误删的同日同机构不同作者/标题研报无法靠增量续传恢复。完成后需以 `--enable-consensus`（按需追加 `--enable-consensus-revision-features`）重建 `cs_train` / `cs_infer`，并重新训练启用一致预期的模型；此后可恢复正常增量续传。

## [0.95.14] - 2026-08-24

### Tests

- **回归测试与既有行为对齐**：`test_tushare_client_rate_limit.py` 中 `daily` 已配置接口级限频 480（官方 500 次/分钟留 10% 余量），改用真正未配置的接口验证“未知接口回退全局 500/分钟”与“非限流错误不更新限频”；`test_respi_35lcd_disp.py` 中 efinance→AKShare 实时快照链路测试显式 mock `_should_prefer_daily_holdings_snapshot`，避免盘后运行时段被“优先日线快照”分支短路。全量 1265 个测试全部通过。

## [0.95.13] - 2026-08-24

### Fixed

- **季度公告窗口加载告警降噪**：`DataLoader` 统一通过私有封装执行 DataFrame 合并，精确屏蔽窗口前补充记录及其与窗口内数据合并时 pandas empty/all-NA concat 的 `FutureWarning`；现金流等季度公告数据的行、全 NA 列、dtype 推断输入和加载顺序均保持原样。

### Tests

- `test_announcement_partition_window_loading.py` 新增现金流窗口内外多分区回归测试，验证两层 concat 告警不再泄露，且股票记录与全 NA 可选列完整保留；既有 `report_rc` 告警测试继续通过。

## [0.95.12] - 2026-08-24

### Fixed

- **业绩快报旧单文件迁移告警降噪**：`Storage.migrate_raw_single_file_to_partitions()` 在旧单文件与已有季度分区合并时，精确屏蔽 pandas 对 empty/all-NA concat 的 `FutureWarning`；合并数据、全 NA 列、去重顺序与 raw schema 均保持不变。

### Tests

- `test_express_partition_migration.py` 新增混合态迁移回归测试，验证该告警不再泄露且全 NA 可选列原样保留。

## [0.95.11] - 2026-08-24

### Fixed

- **北向资金跨制度 OOS 因果隔离**：删除同一 `north_flow*` 列跨越净买入/成交额两种语义的设计，拆为 6 个 `north_net_buy*` 与 6 个 `north_turnover*` 因子，并保留 `north_turnover_flag`。两组因子仅在所属口径激活，另一口径统一为 0；因此训练截止 2024-08-19 前的模型不会在首个跨制度 OOS 中把成交额误读为净买入，有新口径训练样本后则可独立学习成交额因子。生产、稀疏候选、复现实验及 north-on 排除清单同步为新 13 列，保持各清单原有策略。
- **风控百分位完整截面闭环**：`PositionRiskMonitor` 改为接收完整当日特征截面，先按与训练一致的全市场日截面生成 `pct_*`，再筛选单股或持仓，修复持仓子集内排名导致的数值漂移；单股 Monitor 不再因原始 cs 行缺少 `pct_*` 而提前返回 HOLD。`PositionRiskModel.predict_single()` / `predict_batch()` 缺少模型特征时明确报错，不再用 NaN 冒充训练期百分位或在未知范围的批次内自行排名。
- **风控测试穿透真实入口**：新增 Monitor 单股与批量路径测试，验证三股票截面中的最低值在单股和持仓子集预测中均保持 `1/3` 分位；原单股 NaN 占位测试改为拒绝无截面预测。

### Migration

- 需重新构建 `cs_train` / `cs_infer` 以生成 `north_net_buy*` / `north_turnover*` 新列，并重新训练主模型与风控模型；不兼容旧 north 特征 schema。

## [0.95.10] - 2026-08-24

### Fixed

- **北向资金因子口径切换适配**：2024-08-19 起交易所调整北向披露口径，`moneyflow_hsgt` 的 `hgt/sgt/north_money` 由"当日净买入（可为负）"变为"当日成交额（恒正）"；同时证实该接口**全程单位为百万元**（切换前后一致，沪港通首日 2014-11-17 `hgt=13000` = 130 亿元额度）。`north_flow.py` 现按口径分段处理：全期统一 ÷100 换算亿元；`north_flow_ma5/ma20/z20/sum5` 按口径段独立滚动，窗口不跨切换日；新增 `north_turnover_flag` 口径指示列（0=净流入, 1=成交额）供模型显式区分跨口径样本；`north_flow_sign_streak` 窗口化为近 20 日并按口径段计算方向（净流入符号 / 成交额环比方向），全期有值且不受加载范围裁剪影响；`north_flow_z20` 段内预热不足时置 0 中性，避免全 NaN 列触发推理侧特征质量门禁拒绝整日预测。
- **北向 ensure 重复查询治理**：北向市场级数据每日必存在，空响应不落空占位（防永久丢数）；单日推理侧将补齐范围裁剪为近 40 个交易日（2 倍滚动窗口预热），接口临时故障/停更时不再对全部缺失历史重复分段查询；远历史缺口由 `download_raw.py` 批量脚本补齐。
- **风控特征与推理闭环**：`train_position_risk_model.py` 保留 `north_flow_sum5` 候选并新增 `north_turnover_flag`；广播列不做截面百分位（无信息）。`position_risk.py` 推理侧补齐 `pct_*` 截面百分位列（按当日截面 rank 与训练同口径），单股推理缺失 pct 列以 NaN 占位并告警，消除训练含 pct 特征而推理缺列的 KeyError 隐患。
- **批量脚本配置澄清**：`batch_walk_forward.ps1` 修正 `$enable_north` 注释与值矛盾，`$factor_exclude_file` 改为留空使用生产默认清单（原 `#configs/...` 伪路径不存在时会导致因子精简整体跳过、全部因子保留）；注释修正 `factor_exclude_list_north_on.json` 的真实含义（从排除清单放回 6 个 north 因子）。
- **生效说明**：存量 `cs_train` / `cs_infer` 分区中的北向因子列仍为旧口径（前段未换算亿元），需重建特征；使用含 north 列模型的实验需重训后生效（当前生产主模型 v22318 不含 north 列，不受影响）；风控模型需重训后吸收新口径与 flag 列。

### Tests

- `test_factor_north_flow.py` 更新：全期单位换算（百万元→亿元）、口径指示列、切换前后 streak 方向定义、streak 20 日窗口封顶、z20 预热置 0、滚动窗口不跨口径。
- `test_ensure_and_t0_printing.py` 新增：北向空响应不落占位分区。
- 新增 `test_position_risk_pct_inference.py`：风控推理侧 pct 截面重建、存量 pct 列不覆盖、单股 NaN 占位。

## [0.95.9] - 2026-08-23

### Fixed

- **一致预期单边目标价恢复**：`consensus.py` 与 `consensus_revision.py` 改为逐行计算目标价代理；上下界同时存在时取均值，仅一边存在时保留该值，避免 `report_rc` 同时含两列但 `max_price` 行级稀疏时将有效 `min_price` 一并丢弃。
- **现金流报告期选择修复**：现金流公告 PIT 查询启用报告期优先，同报告期取最新公告，晚发的旧报告期更正不再覆盖最新报告期状态。
- **自由现金流收益率单位修复**：计算 `fcf_yield` 前将 TuShare `total_mv` 从万元换算为元，与现金流量表金额单位对齐。
- **现金流利润比重复入模修复**：基本面与现金流质量因子同时启用时，训练入口保留 `zscore_ocf_to_profit`，移除确定性别名 `zscore_cf_nm` 及其 `_sz` 派生列；特征分区仍保留原列，兼容既有模型推理与只启用基本面的实验。

### Audit

- `express_revenue_yoy` 的缺失来自去年同期快报不可用，`express_surprise` 的缺失来自同报告期先验业绩预告不可用，`cons_eps_revision_30d` 的缺失来自相邻两个 30 日窗口不同时满足；三者保留真实 NaN，不做填零或跨期回填。
- 存量目标价与现金流特征需重建后采用新口径；训练候选去重需重新训练后生效。

### Tests

- 新增一致预期单边目标价、现金流报告期优先、`fcf_yield` 单位换算及训练候选别名去重测试。

## [0.95.8] - 2026-08-23

### Fixed

- **现金流质量四列全空修复**：`cashflow_quality.py` 原先读取不存在的 `c_pay_for_assets`，现统一改用 TuShare 现金流量表官方字段 `c_pay_acq_const_fiolta`（购建固定资产、无形资产和其他长期资产支付的现金），恢复 `fcf` 与 `capex_to_ocf` 计算，进而恢复训练列 `zscore_capex_to_ocf`、`zscore_capex_to_ocf_sz`、`zscore_fcf_yield`、`zscore_fcf_yield_sz`；`TushareClient` 的默认 cashflow/cashflow_vip 请求字段同步修正。
- **生效说明**：存量 `cs_train` / `cs_infer` 分区中的上述四列仍为全空，需重建特征并重新训练模型后生效；现有 raw/cashflow 分区已包含正确字段，无需重新下载。

### Tests

- `test_factor_wiring_cashflow_consensus_revision.py` 新增 TuShare 官方资本开支字段公式测试、四个训练列端到端非空测试及客户端默认字段契约测试。

## [0.95.7] - 2026-08-23

### Fixed

- **亏损因子恢复真实区分度**：`is_loss` 对齐 TuShare `daily_basic` 契约，将“已匹配股票的 `pe_ttm` 为空”识别为亏损，同时保留 `pe_ttm <= 0` 以兼容其他数据源；通过当日 `daily_basic` 股票集合区分源记录存在性，左连接未命中的数据缺失不再误判为亏损。此前仅以 `pe_ttm <= 0` 判断，但 TuShare 对亏损公司返回空 PE，导致历史 `is_loss` 全为 0 并在每次训练时作为常数列删除。
- **生效说明**：存量 `cs_train` / `cs_infer` 分区中的 `is_loss` 不会自动改写，需重建特征并重新训练模型后生效。

### Tests

- 更新 `test_value_dividend_missing_semantics_preserved`，覆盖正 PE、兼容性负 PE、TuShare 匹配空 PE及 `daily_basic` 未匹配四种场景。

## [0.95.6] - 2026-08-23

### Fixed

- **业绩快报因子全链路修复（express 审计）**：
  - 同日多公告稳定排序：`express.py` 最终排序改用 `mergesort`，保持“同日多公告按 end_date 升序”相对顺序，PIT 查询同日取最后一条时稳定选中报告期最新的快报（与 fundamental 同构修复对齐）；
  - 报告期优先 PIT：`build_express_lookup_by_date` 启用 `end_col="end_date"`，晚发的旧报告期更正公告不再把因子值整体回退到旧报告期（此前 express 是公告型因子中少数未启用该模式的之一）；
  - 接口缺列优雅降级：`yoy_net_profit` / `diluted_roe` 缺失时输出全 NaN 列而非 KeyError，保持 cs_train/cs_infer schema 稳定；
  - 离线构建显式告警：`enable_express` 且 forecast 缺失时输出 warning（提示 `express_surprise` 将全 NaN），与纸面 ensure 强制补齐口径差异可见化；
  - 迁移季度分区存储：express 由单文件改为按季度 `end_date` 分区存储（与 forecast/fina_indicator 对齐）；loader 与 ensure 首次访问自动迁移旧单文件（分组写分区 + 删除旧文件），增量补齐路由写入季度分区（消除整文件读-合并-重写）；`raw_download` CLI 的 express 下载同步分区化，关闭单文件写入入口；
  - 门控强化：`_MIN_EXPRESS_RECORDS` 500→1000（防止损坏残留误判为充足）；cs_infer 缓存补检新增 `express_profit_yoy` / `express_roe` / `express_surprise`，旧缓存缺列自动重建。

- **express 分区迁移三项数据完整性加固（审查反馈）**：
  - 混合态不漏读：迁移条件从"无分区"改为"旧单文件仍存在"，迁移时与同季度已有分区合并去重（此前"部分分区 + 旧单文件"混合态会完全忽略旧数据）；
  - 数据不足真正全量重建：`_bulk_download_by_period` 新增 `force` 参数（忽略断点续传全量重下），express 数据不足分支 `force=True` 补齐残缺季度（此前已迁移的残缺季度会被跳过）；
  - 无效分区键不静默丢数：迁移按 `dropna=False` 统计无效分区键记录，存在跳过记录时不删除旧单文件（保留待人工处理），仅在全部记录成功迁移后删除旧文件。

- **express 分区迁移三项一致性加固（复审反馈）**：
  - 同键冲突新分区优先：迁移合并顺序调整为已有分区在后 + `keep="last"`，旧单文件不再覆盖新分区数据（此前 `concat([existing, part])` 导致旧值胜出）；
  - 异常旧文件不遮蔽有效分区：空旧文件视为垃圾清理、缺分区列时返回 None，loader/ensure 仅在迁移结果非 None 时才覆盖已加载分区（此前空/破损旧文件会把有效分区覆盖为 None/破损数据）；
  - 非法日历日期计入无效：`_partition_key_to_date` 新增真实日历校验（如 20230230），非法日期计入 skipped 而非在保存分区时抛 ValueError 中断迁移。

- **express 完整性门控与失败契约加固**：
  - 纸面因子链路统一通过 `_try_download_express` 检查记录数量与同步水位，水位已覆盖时也不会绕过 `_MIN_EXPRESS_RECORDS` 损坏门槛；
  - `force=True` 强制全量下载出现季度异常时明确抛错，重建完成后再次校验最低记录数，禁止残缺分区被当作重建成功继续生成特征。

### Tests

- 新增 `test_express_partition_migration.py`：单文件迁移（写分区/删旧文件/缺分区列保持原状）、混合态合并（部分分区 + 旧单文件不漏读）、无效分区键保留旧文件、同键冲突已有分区优先、非法日历日期计入无效、空/破损旧文件不遮蔽有效分区（loader 与 ensure 两侧）、`load_express` 分区优先与迁移回退、`_try_download_express` 迁移后分区增量与数据不足 force 全量下载、强制下载异常及重建后仍不足的失败契约、`_bulk_download_by_period` force 断点续传跳过语义；
- `test_ensure_and_t0_printing.py` 补充纸面 express 即使同步水位已覆盖也必须进入完整性检查入口的回归断言；
- `test_express_factor.py` 新增同日多公告选最新报告期、晚发旧期更正不回退、缺列输出全 NaN 列；
- 既有单文件示例用例（`test_forecast_report_rc_partition.py` / `test_ensure_and_t0_printing.py`）改挂 `stk_holdernumber`，express 纳入分区数据集用例。

## [0.95.5] - 2026-08-23

### Fixed

- **基金持仓因子全链路修复（fund_portfolio 审计）**：
  - 离线分区模式去重生效：`download_by_period` 分区落盘前按 `dedup_cols` 去重（此前去重仅作用于非分区合并路径，分区数据集原样落盘），同一报告期"季报前十大 + 半年报/年报全量"两批公告不再让聚合 `sum(stk_float_ratio)` 双重计数（`fund_count` 用 `nunique` 原本幸免）；`cli.py` 的 `fund_portfolio` 下载补充 `sort_cols`，保证 keep="last" 保留 ann_date 最晚记录；
  - paper 端落盘去重：`_try_ensure_historical_fund_portfolio` 下载后按 `(ts_code, symbol, end_date)` 去重，与离线口径一致；
  - paper 端披露季刷新门控：距报告期末不足 4 个月的分区，若分区内最新公告日未覆盖到当前日则强制重下并覆盖重写，同时强制重算 `fund_portfolio_agg` 缓存，消除披露季中期首次下载的部分快照永久冻结（train/serve 口径分裂）；
  - 离线 fund 回溯窗口独立扩至 18 个月（此前统一 7 个月 warmup 无法加载 chg 所需同口径上期分区，短窗口增量构建 chg 整批 NaN，与 paper 端回溯口径分裂）；
  - paper 端 `fund_portfolio` 下载失败日志从 `debug` 升级为 `warning`（与 cyq_perf 同类口径风险），docstring 回溯年限描述修正。

### Tests

- 新增 `test_fund_portfolio_ensure_refresh.py`：披露窗口边界、落盘去重（保留最晚公告日）、窗口内水位刷新 + agg 缓存重算、下载失败警告与降级；
- `test_download_periodic_concurrency.py` 新增分区模式落盘前去重用例；
- `test_ensure_and_t0_printing.py` 聚合缓存复用用例改用披露窗口外日期（新语义下窗口内会回读明细检查覆盖水位）。

## [0.95.4] - 2026-08-23

### Fixed

- **筹码胜率因子链路四项修复**：
  - `weight_avg_bias` 复权口径修复：偏离度改用未复权收盘价（`current_data.close`）与未复权成本价 `weight_avg` 同口径计算；此前因特征截面从不含 `close_adj` 列，该因子从未被产出（幽灵因子），修复后 5 列齐备正式入模（需重建特征分区 + 重训练后生效，建议按因子裁剪实验契约先 A/B 对比）；
  - `winner_rate_chg_5` / `winner_rate_chg_20` 改为按交易日历对齐：缺失数据日（含个股缺席与全市场缺某日）保留空位，严格取 5/20 个交易日前的值，对齐位置缺数据为 NaN，不再因 `dropna` 先行删除缺数据日而静默跨越更长日历区间；日历 = 数据内日期 ∪ `trading_dates`（裁剪到数据范围），并新增 `calendar_dates` 参数供 ensure 链路传入完整历史交易日（ensure 仅输出单日截面，不物化完整 lookup）；
  - `build_cyq_perf_lookup_by_date` 向量化重写：筹码集中度列运算、变化率日历对齐、按日切分全部向量化，移除逐行 `iterrows`（6 年约 800 万次迭代）；
  - 纸面链路下载失败静默降级修复：`cyq_perf` 当日分区下载失败日志从 `debug` 升级为 `warning`，避免当日因子静默缺列造成 train/serve 口径不一致。

- **`_REQUIRED_FACTOR_COLS` 同步补检 `weight_avg_bias`**：随口径修复直接加入缓存校验清单（用户决策：不等切换），旧 4 列缓存分区（缺该列）会被 ensure 自动判定为不完整并重建为 5 列新口径；注意这意味着重建后的推理特征与现役 4 列旧口径模型不匹配，需同步推进重训练，勿在重建后用旧模型跑纸面。

### Tests

- 新增 `test_cyq_factor_handler.py`：未复权口径偏离度计算、后复权价不混入口径、缺 `close` 兑底不产出偏离度；
- `test_cyq_perf_factor.py` 新增缺失数据日日历对齐测试（对齐日缺失 → NaN，后续日正确跨 5 个交易日）。

## [0.95.3] - 2026-08-22

### Changed

- **训练入口特征清洗日志输出移除列明细**：`ml/train_core/prepare.py` 在原有统计日志（高缺失/全空/常数计数）之后，按类逐行打印被移除特征列的详细名称（高缺失、全空、常数、zscore 联动移除），便于快速定位数据链路问题（如某类因子整体缺失）；仅日志变更，不影响特征移除逻辑与训练结果。

## [0.95.2] - 2026-08-22

### Fixed

- **融资融券幽灵因子清理（margin_net_buy_ratio）**：
  - `margin_net_buy_ratio` 自引入以来从未被计算，仅在 `MARGIN_COLS` 与 `MARGIN_FEATURE_COLUMNS` 中声明，被 lookup 存在性过滤与训练列过滤双重掩盖；现已从两份清单移除，主模型入模融资融券因子正式确定为 `rzye_chg_5` / `rzye_chg_20` / `rqye_rzye_ratio` 三个；
  - `factors/margin.py` 将列清单拆分为 `MARGIN_COLS`（主模型）与 `MARGIN_RISK_COLS`（风控专用：`margin_net_buy` / `short_balance_change_5` / `short_sell_vol_change_5`），后者继续由 lookup 输出供 PositionRiskModel 使用，不改变既有特征分区 schema；
  - `features/ensure/schema.py` 缓存补检新增 `short_balance_change_5`，2026-08-06 之前构建的旧 `cs_infer` 缓存（缺该列）会被判定不完整并自动重建，消除风控模型 train/infer 列差异；
  - 顺带修复：`rqye_rzye_ratio` 补上源列存在性保护（同函数其他因子均具备），`rqye` 缺失时优雅降级为 NaN 而非 KeyError。

### Tests

- 新增 `test_margin_factor_cleanup.py`：lookup 输出列集合、主模型/风控列清单划分、变化率/多空比/净买入数值、融券源列缺失降级、缓存补检列断言。

## [0.95.1] - 2026-08-20

### Fixed

- **推理侧补齐事件型 freshness 衰减（消除 train/serve skew）**：
  - 训练侧 `state_keep_event_decay` 策略对事件型因子按 freshness 指数衰减，但 OOS 评估与 `MLSignal`（回测/纸面）一直用未衰减原值，导致模型在压缩分布上学习、推理时旧公告以原值全额进入模型；
  - 衰减逻辑抽取为公共模块 `ml/train_core/freshness.py`（`apply_event_freshness_decay` / `apply_serving_event_decay`），训练侧与推理侧共用同一实现；
  - `MLSignal.generate` / `generate_ranked` 按模型 `train_params` 中记录的 `freshness_strategy` / `event_freshness_half_life_days` 复现衰减；旧模型缺参数时按默认 `state_keep_event_decay` + 45 天处理；
  - walk-forward OOS 评估（`split_training.py`）同样按运行参数复现衰减；新注册模型（split、deploy 与 `train_ml_model.py` 单次训练）在 `train_params` 中记录 freshness 参数，供推理侧精确复现；
  - 注意：历史 `decay` 组实验结论基于未衰减推理，修复后需重跑对比才有可比性。
- **股东人数环比改为跨报告期精确对齐（holder_num_chg / holder_num_chg_2q）**：
  - 原实现按 `ann_date` 朴素 shift，同一报告期的修正公告会稀释跨期信号；现改为每个公告版本的环比基准 = 公告日不晚于本版本、报告期早于本版本的最新已公告值（同报告期多版本取最新修正版）；
  - PIT 查询启用报告期优先（`end_col`），晚发的旧报告期修正公告不再覆盖已公告的新报告期。
- **业绩预告因子两处修正**：
  - `forecast_type_score` 未知/缺失类型不再 `fillna(0.0)`（与"不确定"评分混淆），保留 NaN 交由模型原生处理；
  - PIT 查询启用报告期优先（`end_col`），同报告期修正公告发布后自动替代首发版，晚发旧期修正不覆盖新期预告。
- **公告 PIT 通用函数新增 `end_col` 参数**：提供时在当日可见公告中优先选择报告期最新的记录（同报告期取最新公告），未提供时保持原行为，现有调用不受影响。

### Tests

- 新增 `test_serving_freshness_decay.py`：衰减权重/半衰期、缺失与负 freshness、策略门控、`MLSignal` 推理侧复现（decay/no_decay/旧模型默认）等 8 个用例；
- 新增 `test_holder_end_date_alignment.py`：同报告期修正版本跨期对齐、晚发旧期修正不覆盖新期、首期 NaN、跨股票隔离；
- 新增 `test_earnings_pit_and_type_nan.py`：未知类型 NaN、修正版本切换、报告期优先查询、`end_col` 兼容行为等 7 个用例。

## [0.95.0] - 2026-08-09

### Added

- **龙虎榜新增连续异动信号 `lhb_cont_on_list`**：
  - 语义：当日是否存在"连续异动"类上榜理由（`top_list.reason` 含"连续"，如连续三个交易日涨幅偏离累计达 20%），0/1 事件级信号；
  - 与选中主记录无关：当日单日榜与连续类理由并存时，主记录仍取单日榜（净买入绝对值最大），但连续异动信号记为 1；
  - 相比 `lhb_reason_count`（非零比例约 1.6%、值域几乎二元）区分度更高：真实数据上榜当日连续异动占比约 39%，覆盖度约 4.8%；
  - 新增 `lhb_cont_up_days_5` / `lhb_cont_up_days_20`：按 `lhb_cont_on_list` 做近 5/20 交易日滚动求和（未上榜日补 0，含时序衰减），覆盖度分别约 2.9% / 7.9%，与 `lhb_up_days_20`（普通上榜 20 日累计）互补；
  - 已登记入 `LHB_COLS`、`LHB_FEATURE_COLUMNS` 与 ensure `_REQUIRED_FACTOR_COLS`，批量 build 特征后自动进入训练（`build_clean_features.py --build-all` 或增量区间重建）。
- **`lhb_reason_count` 纳入生产因子排除清单**：该列仍保留在特征 schema（不破坏历史一致性），但已加入 `data/models/factor_exclude_list.json`，训练时由因子精简逻辑剔除（重要性恒为 0 且与 `lhb_on_list` 冗余）；如需复启用，从清单移除即可。
- **ensure 缓存 schema 补检 `lhb_cont_on_list`**：`features/ensure/schema.py` 的 `_REQUIRED_FACTOR_COLS` 新增 `lhb_cont_on_list`。旧 `cs_infer`/`cs_train` 缓存（v0.95.0 之前构建、缺该列）会被 `_check_features_schema` 判定不完整并自动触发重建，避免推理时该列缺失被 `MLSignal` 补 NaN 后触发"全空拒绝预测"。

### Tests

- 新增 `lhb_cont_on_list` 事件级信号、非上榜日置 0、reason 列缺失降级测试；既有单日/连续类并存与仅连续类用例补充断言。

## [0.94.27] - 2026-08-09

### Fixed

- **adj_factor 强制依赖闭环**：
  - raw 批量下载不再允许 adj_factor 空响应静默通过，空响应按整日失败处理且不部分落盘；moneyflow 仍允许因发布时间较晚而暂时为空，并在后续运行重试；
  - 在线 ensure 获取不到 adj_factor 时返回 False，离线 clean 批量构建则跳过该日期，不再构造全 NaN 占位并缓存无效 clean/daily。
- **无效 clean/daily 缓存自愈**：
  - 在线和离线构建在跳过已有分区前检查 close_adj 是否至少包含一个有效值；旧版本生成的全空复权价分区会自动失效并覆盖重建；
  - DataCleaner 拒绝空、缺必要列或全无有效数值的 adj_factor，防止其他调用路径绕过入口保护。
- **复权因子缺失不再跨日填充**：
  - cleaner 与 FeatureBuilder helper 均取消 adj_factor 前向填充；局部缺失保留 NaN 并告警，避免除权除息事件日错误沿用昨日因子；
  - 真实历史数据允许少量股票缺少因子，不要求 daily 与 adj_factor 代码集合完全一致。

### Tests

- 新增 adj_factor 空响应整日不落盘、在线补齐失败、离线跳过、全无效值拒绝和旧坏缓存自动重建测试；更新中间日期缺失因子测试为保持 NaN。

## [0.94.26] - 2026-08-09

### Fixed

- **daily 唯一代码完整性校验**：
  - 新下载和历史重下数据在落盘前校验 `(ts_code, trade_date)` 主键，重复记录、空代码或混入非目标日期时直接失败，避免分页重叠用重复行虚增覆盖率；
  - 覆盖度统一使用唯一 `ts_code` 数量，不再使用 Parquet 物理行数判断。
- **历史停牌日覆盖度交叉确认**：
  - stock_basic 粗筛低于 85% 时，改由独立 daily_basic 代码域二次确认，避免 2006 年和 2015 年大面积停牌等合法历史日期被硬阈值误杀；
  - 基于全部 241 个低覆盖历史分区校准 2% 跨源容差，实际最大差异为 1.96%，明显部分返回仍会失败；
  - daily_basic 自身改用代码集合校验，替代严格行数相等门控；正常发布时间延迟仍保持告警降级。
- **缺失基准安全降级**：stock_basic 缺少 list_date 时跳过历史覆盖率粗筛并告警，不再错误回退当前股票全集。

### Tests

- 新增重复 daily 不落盘、历史低覆盖由 daily_basic 确认、2% 历史容差与明显部分返回拒绝用例。

## [0.94.25] - 2026-08-09

### Fixed

- **daily 完整性严格闭环（评审高危）**：
  - 新下载 daily **先验证覆盖度再落盘**：覆盖率低于当日已上市股票域的 85% 时不落盘并返回 False，
    避免缺陷分区（截断/部分返回）被保存；
  - 历史低覆盖强制重下后，重下返回空（接口故障）或重下后仍低（数据不可靠）均返回 False，
    不再"告警后继续放行"；旧分区在重下异常时保留但明确失败；
  - 测试覆盖全部失败路径（首次低覆盖不落盘、重下空、重下仍低）。
- **覆盖度基准按 list_date 过滤（评审中高）**：
  - `_is_daily_coverage_low` 分母改为"截至 trade_date 已上市（list_date <= trade_date）"股票数，
    不再用当前 stock_basic 全集（5878 只）做所有日期的分母，避免 2005 年等历史日期
    被必然误判低覆盖而每次重复下载；缺 list_date 列时退化为全集并保留告警。

### Tests

- `test_ensure_and_t0_printing.py`：新增首次下载低覆盖不落盘、重下返回空、重下仍低三条失败路径用例；
  新增 list_date 过滤（历史日期不误伤）断言。

## [0.94.24] - 2026-08-09

### Fixed

- **daily 覆盖度闭环（评审高危）**：
  - `_check_daily_coverage` 重构为 `_is_daily_coverage_low`，阈值 0.7 → 0.85
    （实测丢失 1000/5500、覆盖率 81.8% 的场景可被检测）；
  - 历史已有 daily 覆盖度偏低时触发一次**强制重下**（不再仅告警），重下后覆盖度恢复即成功；
  - 重下后仍低（可能为真实停牌潮或服务端部分返回）时 error 告警待人工核实
    （不返回失败，避免停牌潮日误报阻断流程）；
  - 集合级错配（缺一只混入另一只且行数相近）仍属已知局限。
- **daily_basic 覆盖度门控恢复（评审中危）**：
  - 实测 daily_basic 与 daily 代码集合 30/30 天完全一致（不同于 moneyflow 缺北交所），
    恢复 `min_rows=daily_rows` 门控；上一版（0.94.23）将其与 moneyflow 一并移除属误删，已纠正。

### Tests

- `test_ensure_and_t0_printing.py`：coverage_gate 恢复为 daily_basic；新增历史 daily 低覆盖
  触发强制重下、`_is_daily_coverage_low` 判定用例。

## [0.94.23] - 2026-08-09

### Fixed

- **moneyflow 覆盖度门控修正（评审高危）**：
  - `moneyflow` 天然不覆盖全部 daily 股票（实测每日稳定少 319~331 只，主要为北交所 920xxx.BJ），
    从 `ensure_raw_data_for_date` 的 `min_rows=daily_rows` 门控中移除，恢复存在性检查，
    避免每次 ensure 重复下载；`adj_factor`/`stk_limit` 保留行数门控。
    （注：`daily_basic` 实测与 daily 代码集合一致，其门控保留——见 0.94.24 修正说明。）
- **KDJ 窗口不足掩码（评审中危）**：
  - `technical_indicators.py::calculate_kdj` 在 RSV 窗口有效观测不足（low_n/high_n 为 NaN）时，
    最终 kdj_k/kdj_d/kdj_j 输出掩码为 NaN：内部状态可重置（ffill/fillna 仅用于 EWM 演化），
    但不再给"看似有效"的伪信号，与 volatility 等窗口因子保持一致的缺失语义。
- **daily 自身覆盖度告警（评审中危）**：
  - `ensure.py` 新增 `_check_daily_coverage`：以 stock_basic 全集为参照，daily 行数低于 70%
    时记录 error 告警（覆盖历史截断分区/服务端部分返回），仅告警不自动重下；
    集合级错配（缺一只混入另一只且行数相近）仍属已知局限。
    （注：0.94.24 已升级为低覆盖度触发强制重下，见下。）

### Tests

- `test_ensure_and_t0_printing.py`：coverage_gate 改用 adj_factor；新增 moneyflow 不受 daily
  行数门控、daily 覆盖度告警用例；
- `test_technical_indicators_precompute.py`：补充停牌股复牌首日 KDJ 掩码为 NaN 断言。

## [0.94.22] - 2026-08-09

### Fixed

- **停牌窗口修复补全（评审问题1）**：
  - `static_core.py::_calculate_window_features_static` 统计窗口内观测数，`ret_N`/`vol_ratio_N`/
    `ma_deviation_N`/`amount_maN` 在观测不足（停牌/上市不足导致行数 < window）时置 NaN，
    不再用不足观测的 first/last 聚合出跨停牌失真值；
  - `technical_indicators.py::calculate_kdj` 的 RSV 前向填充改为 `ffill(limit=3)`：短期停牌保持
    停牌前 RSV，长期停牌回退 `fillna(50)` 重置，避免指标长期僵化后在复牌日跳变。
- **daily 自身截断源头修复（评审问题2）**：
  - `get_daily` 全市场查询（`ts_code is None`）改走 `_query_with_pagination` 分页，
    消除 daily 单次 6000 上限静默截断；覆盖度门控的 daily 行数基准因此更可靠。
- **ensure 区分非交易日与接口故障（评审问题3）**：
  - `ensure_raw_data_for_date` 日线为空时依据缓存 trade_cal 判断：非交易日返回 True（正常跳过）；
    交易日接口返回空或无法确认交易日历时返回 False（不再把接口故障误报为成功）。
- **suspend 空值占位（评审问题4）**：
  - 停复牌接口返回空（当日无停牌）时写占位空文件，避免下次 ensure 重复请求。

### Tests

- `test_ensure_and_t0_printing.py`：非交易日用例预置 trade_cal；新增交易日接口故障返回 False、
  suspend 空占位不再重复请求用例；
- `test_tushare_client_pagination.py`：新增 daily 全市场分页用例；
- `test_technical_indicators_precompute.py`：新增 static_core 窗口观测不足置 NaN 用例；
- `test_audit_fixes_quality_consistency.py`：新增 daily_basic 空值 `logger.error` 断言（评审问题5）。

## [0.94.21] - 2026-08-09

### Fixed

- **raw ensure 覆盖度门控（审计高危）**：
  - `storage.py::is_data_exists` 新增可选 `min_rows` 参数（配合 `count_rows` 用 pyarrow 元数据快速读行数），
    文件存在但行数不足即视为未补齐，防止历史截断/中断落盘后缺口永久驻留；
  - `ensure.py` 以当日 daily 行数为覆盖度参照，对 `adj_factor`/`stk_limit`/`moneyflow`/`daily_basic`
    独立检查时启用 `min_rows` 门控（按日返回的应为同一批交易股票，行数不足触发重新补齐）。
- **技术指标/波动率滚动窗口按交易日对齐（审计中危）**：
  - `precompute_technical_factors` 新增可选 `trading_dates` 参数，经 `_align_to_trading_dates`
    将日线序列与交易日对齐（停牌日补 NaN 行占位），滚动窗口按交易日跨度而非实际行数，
    避免长期停牌股复牌后 RSI/KDJ/MACD/BOLL/波动率/ATR 窗口语义错位；输出仍以原始行为键；
  - `compute_ret_1` 的 `pct_change` 改 `fill_method=None`，停牌缺口收益保持 NaN，
    不再跨停牌期计算（对无缺口的正常股票行为不变）；
  - `helpers.py` 技术因子缓存构建时传入窗口交易日列表（生产路径默认生效）。

### Tests

- `test_storage.py` 新增 `count_rows` 与 `is_data_exists(min_rows)` 覆盖度门控用例；
- `test_ensure_and_t0_printing.py` 新增 daily_basic 行数不足触发重新补齐用例；
- `test_technical_indicators_precompute.py` 新增 `_align_to_trading_dates` 补行与
  停牌股复牌初期滚动窗口按交易日对齐（NaN）用例。

## [0.94.20] - 2026-08-09

### Fixed

- **raw ensure 补齐解耦（审计高危）**：
  - `adj_factor` / `suspend` / `stk_limit` 从"daily 缺失分支"中移出，改为与 moneyflow/daily_basic 一致的独立存在性检查，
    防止 daily 已落盘但某类数据因接口抖动缺失时永久无法补齐；
  - 日线为空（非交易日）时提前返回并告警，不再无意义调用其余接口；`daily_basic` 为空时新增告警。
- **daily_basic / moneyflow / stk_limit 全市场查询自动分页（审计中危）**：
  - `TushareClient` 新增 `_query_with_pagination`（走 query 限频与限流重试）；三个 getter 在未指定 `ts_code` 的
    全市场查询下自动分页取全，避免单次 6000 上限静默截断。
- **复权因子回填语义修正（审计中偏高）**：
  - `cleaner.py` 与 `helpers.py` 的 `adj_factor` 处理去掉 `bfill` 仅保留 `ffill`（累积因子前向填充语义正确），
    避免将除权除息事件后的因子回填到事件前造成复权价虚假跳变、污染动量/波动率因子与标签。
- **daily_basic 单日整体缺失硬告警（审计中危）**：
  - `_add_value_dividend_features_static` 由 warning 升级为 error 级告警，价值红利核心信号
    （bp/ep_ttm/dv_ttm/市值/换手）全空时明确可见。

### Tests

- 新增 `tests/test_tushare_client_pagination.py`（分页拼接/单股不翻页/空降级）；
- `test_ensure_and_t0_printing.py` 新增 adj_factor 独立补齐与非交易日短路用例；
- `test_features.py` 新增复权因子头部缺失不回填、中间缺失前向填充用例；
- `test_audit_fixes_quality_consistency.py` 新增 daily_basic 空值不熔断用例。

## [0.94.19] - 2026-08-09

### Fixed

- **门控 docstring 语义与实现对齐（审计低危）**：
  - `factor_load.py::_has_announcement_gap` docstring 原写"有效水位 = max(...)"，
    与"有水位以水位为准、无水位用数据最新日期初始化"的实现不符，已更新描述。
- **同步水位写入使用唯一临时文件（审计低危）**：
  - `storage.py::save_sync_watermark` 改用同目录唯一临时文件
    （`tempfile.mkstemp` + `os.replace`），避免多进程并发同步同一数据集时
    固定临时文件名互相覆盖。

## [0.94.18] - 2026-08-09

### Fixed

- **门控覆盖判断与增量函数语义对齐（审计高危）**：
  - `factor_load.py::_has_announcement_gap` 原先在有水位时仍取
    `max(watermark, latest_data)`，数据最新公告日可能跨过失败日（后续成功落盘），
    导致门控误判已覆盖、阻止增量运行，失败日永久漏掉；
  - 现改为"有水位（连续成功前缀）时覆盖判断只认水位"（`covered_to = watermark`），
    数据不得越过水位后的未知区间，与 `_incremental_catchup_by_calendar_date` 起点语义一致；
  - **测试**：新增反例用例（watermark=11、数据已有 13 公告、target=13 仍为缺口）。

- **同步水位写入改为原子替换（审计低危）**：
  - `storage.py::save_sync_watermark` 改为临时文件 + `os.replace` 原子替换，
    崩溃时正式水位文件保持不变，只会触发安全重查。

## [0.94.17] - 2026-08-09

### Fixed

- **同步水位语义修正：水位只代表"连续成功前缀"（审计高危×2 + 中危）**：
  - 失败日不再被后续成功数据跨过：`_incremental_catchup_by_calendar_date` 起点
    改为"有水位则从水位之后开始，不可用数据最新公告日越过水位"；区间内遇到首个
    失败立即停止，水位只推进到最后一个成功日（含空日），下次从水位之后重试；
  - 原子推进：先落盘新数据，成功后才推进水位；落盘失败则水位保持原值，
    避免"水位已提交但数据缺失"导致永久缺数据；
  - 数据缺失不能仅凭水位跳过：`_has_announcement_gap` 在本地数据无有效日期列
    （parquet 被删/损坏）时即使水位高也判定缺口，触发初始化/恢复；
  - **测试**：新增失败日不被后续成功跨过、落盘失败水位不推进用例，
    更新首个失败即停/水位只推进到前缀用例。

## [0.94.16] - 2026-08-09

### Fixed

- **增量补齐引入持久化同步水位，避免反复下载无公告日期（审计高危）**：
  - `src/lazybull/data/storage.py` 新增 `load_sync_watermark`/`save_sync_watermark`
    （存 `{raw}/{dataset}/_sync_watermark.json`），记录该数据集已成功查询至的日期；
  - `_incremental_catchup_by_calendar_date` 仅在区间内无失败时把水位推进到目标日
    （无公告空日也算已同步）；有失败则不推进，下次从原水位之后重试，不跳过失败日；
  - 门控 `_has_announcement_gap` 用"有效水位 = max(数据最新公告日, 同步水位)"
    判断覆盖，本地数据无公告日不再被反复下载；
  - **测试**：新增同步水位覆盖/推进/失败不推进用例。

- **窗口外补行契约补全（审计中危）**：
  - `_load_quarter_partitioned_raw` 目标窗口内无任何分区时，同样从窗口前分区补充
    股票最新公告（原实现直接返回 None，窗口外股票整体变 NaN）；
  - `_load_pre_window_latest_rows` 移除"连续 4 分区无新股票提前终止"的启发式截断，
    完整遍历窗口前分区，确保能找到更早的有效股票（实测补全 545 只窗口外股票，0.6s）；
  - **测试**：新增窗口内无数据仍回填用例。

- **update_flag 闭环补全（审计中危）**：
  - `_FINA_REQUIRED_RAW_COLS` 补充 `update_flag`，现有旧季度分区缺该列时触发
    schema 回补（`fields` 已含 update_flag）；
  - `_bulk_download_by_period` 分区保存前调用 `_drop_duplicates_keep_updated` 去重；
  - `_drop_duplicates_keep_updated`/`fundamental.py` 仅显式识别 `update_flag == "1"`
    为修正版（TuShare 文档：1 表示更新记录），不臆测任意非空值；
  - **测试**：新增非 "1" 非空值不视为修正的用例。

- **cf_nm 依赖告警按构建会话只提示一次（审计低危）**：
  - `static_core.py` 缺 `ocf_to_profit` 导致 `cf_nm` 全 NaN 时，仅首次提示并说明
    可能是 cashflow 未启用或数据缺失，不再按交易日刷屏。

- **清理未使用导入（代码质量）**：
  - `factor_load.py` 移除门控改版后不再使用的 `_MIN_*` 常量导入（F401）；
    顺带清理 `fundamental.py` 的 `numpy` 与 `bulk.py` 的 `Dict` 未使用导入。

## [0.94.15] - 2026-08-09

### Fixed

- **修复公告型基本面因子增量补齐"死路"（审计高危1）**：
  - `src/lazybull/features/ensure/factor_load.py` 门控原先基于"记录数量是否充足"
    （`len < _MIN_XXX_RECORDS` 才触发下载），分区/单文件数据一旦齐全（无论新旧）
    数量永远充足，导致 `_incremental_catchup_by_calendar_date` 增量补齐永不触发，
    纸面交易基本面数据新鲜度完全依赖人工全量下载；
  - 现改为基于"本地最新公告日是否覆盖目标交易日"判断缺口（新增
    `_has_announcement_gap`），fina_indicator/cashflow/stk_holdernumber/forecast/
    express/report_rc 六类统一生效；
  - **测试**：新增 `tests/test_announcement_incremental_gate_fix.py`。

- **修复 fina_indicator/cashflow 增量写入单文件被分区遮蔽（审计高危2）**：
  - `src/lazybull/features/ensure/downloads.py` 的 `_try_download_fina_indicator` /
    `_try_download_cashflow` 原先用 `storage.load_raw` 读旧单文件判断存量（实盘纯
    季度分区时读不到 → 误判为全量重下），增量分支未传分区参数 → 追加写单文件被
    loader 分区优先遮蔽；
  - 现改为 `_load_all_partitions` 分区读取 + 增量按 `end_date` 季度分区路由写入
    （对齐 forecast/report_rc 已有正确写法）；schema 回补结果同样写回分区
    （`historical.py::_refresh_existing_period_rows` 支持分区保存）；
  - **测试**：更新 `test_factor_wiring_cashflow_consensus_revision.py` 与
    `test_ensure_and_t0_printing.py`。

- **季度窗口外股票保留"旧值 + 大 freshness"而非硬缺失（审计中危）**：
  - `src/lazybull/data/loader.py::_load_quarter_partitioned_raw` 原先按窗口
    `[起点年-1, 目标日]` 截断加载，窗口外最新报告的股票（长期停牌/年报延迟）
    直接 NaN，且 paper 单日窗口起点与离线构建不一致产生 cs_train/cs_infer 漂移；
  - 现从窗口起点之前最近的分区补充窗口外股票的最新一条公告
    （新增 `_load_pre_window_latest_rows`），保持 freshness 契约；
  - **测试**：新增 `tests/test_announcement_incremental_gate_fix.py`。

- **同日多公告稳定排序（审计低危1）**：
  - `src/lazybull/factors/announcement_utils.py` 排序改为 `kind="mergesort"` 稳定
    排序，配合 `fundamental.py` 上游按 `[ts_code, end_date, ann_date]` 排序，同日
    披露"更正公告 + 新季报"时 PIT 确定选中最新报告期；
  - **测试**：新增稳定排序用例。

- **update_flag 修正版优先去重（审计低危2）**：
  - `FINA_INDICATOR_DEFAULT_FIELDS` 补充 `update_flag` 字段；
  - 新增 `_drop_duplicates_keep_updated`：同 `(ts_code, end_date, ann_date)` 多次
    修订时优先保留 `update_flag` 非空的修正记录，保证跨次下载可复现；接入分区/单
    文件/批量/回补各保存路径（incremental/bulk/historical）；
  - **测试**：新增 `tests/test_announcement_incremental_gate_fix.py`。

- **cf_nm 回填依赖显式提示（审计低危3）**：
  - `src/lazybull/features/builder/static_core.py` 在未启用 cashflow 因子导致
    `cf_nm` 全 NaN 时显式 warning，避免与启用 cashflow 的实验特征集悄然不一致。

- **bot 盘中触发当日交易显式提示（审计低危4）**：
  - `scripts/bot_service.py` 盘中（A股连续竞价时段）触发当日 `/trade` 时显式提示
    "当日公告可能尚未全部发布"，降低盘后才披露公告被误用的风险。

## [0.94.14] - 2026-08-08

### Fixed

- **统一串行/并行特征构建的基本面代理回填顺序（问题1）**：
  - `src/lazybull/features/parallel.py` 原先在因子处理器 `apply_all` 之前执行
    `_backfill_fundamental_proxy_features_static`，与串行路径（`orchestration.py`
    先 `apply_all` 再回填）相反，导致 `ocf_to_revenue → cf_sales`、
    `ocf_to_profit → cf_nm` 两条回填链在并行路径失效，串行/并行产出不同特征；
  - 现统一为"先因子处理器、后回填"，与串行路径保持一致，消除训练/推理口径漂移；
  - **测试**：新增 `tests/test_audit_fixes_quality_consistency.py` 端到端用例
    （mock 因子处理器生成 `q_ocf_to_sales/ocf_to_revenue`，断言 `cf_sales` 被回填）。

- **统一离线/在线缺失复权因子处理（问题2）**：
  - `src/lazybull/data/build_clean.py` 原先在复权因子缺失时伪造默认值 `1.0`，
    与在线 `ensure.py` 保留 `NaN` 相反，导致同一原始缺失状态产生不同复权价格
    与收益；现统一保留 `NaN` 交由清洗层处理（cleaner 已有 ffill/bfill 兜底）；
  - **测试**：新增 `test_build_clean_missing_adj_factor_keeps_nan`。

- **缺失 dv_ttm/pe_ttm 不再编码成真实经济含义（问题3）**：
  - `src/lazybull/features/builder/static_extra.py` 原先将缺失 `dv_ttm` 填 `0`
    （与"不分红"混淆）、缺失 `pe_ttm` 标为 `is_loss=1`（与"亏损"混淆），且会
    掩盖数据链路失败；现保留 `NaN` 并新增 `dv_ttm_missing`/`pe_ttm_missing`
    显式缺失标记，`is_loss` 仅对已知亏损（非缺失且 <=0）为 1；
  - 影响：新增 `dv_ttm_missing`/`pe_ttm_missing` 列，不影响既有 schema 列；
    `is_loss` 语义修正为仅真实亏损，模型推理输入口径更准确；
  - `dv_ttm_missing`/`pe_ttm_missing` 以可选方式加入训练候选特征
    （`ml/train_core/constants.py` 定义 + `prepare.py` 按存在性加入），
    新构建的特征分区自动启用、旧 schema 分区（无此列）自动跳过并 warning，
    不破坏存量特征数据直接训练；重新训练后模型可显式利用缺失状态；
  - **测试**：新增 `test_value_dividend_missing_semantics_preserved`；
    训练侧新增 `test_prepare_training_data_skips_missing_markers_when_absent`
    （旧 schema 无标记列不报错）与
    `test_prepare_training_data_includes_missing_markers_when_present`。

- **模型推理新增数值质量门禁（问题5）**：
  - `src/lazybull/signals/ml_signal.py` 原先仅做列结构一致性检查，数据失效列仍会
    静默预测（实际案例：`dv_ttm` 全零）；现新增 `_check_feature_quality`，在
    `generate`/`generate_ranked` 预测前仅对"全空"列（缺失率 100%）硬拒绝本次预测
    （返回空信号）；
  - 全零/截面常量/高缺失（>50%）仅记录 WARNING 级聚合警告（不逐列刷 ERROR
    日志）且不阻断——市场环境特征（`mkt_*`/`north_*`）本就是单日常量广播到
    全部股票，常量/全零为设计状态直接忽略不警告；全零也可能是合法状态（如
    全部不分红/未上榜/未亏损），不能以截面分布一票否决整日预测；训练侧已在
    `prepare.py` 按整个训练期判定高缺失/全空/常数并移除；
  - **测试**：新增 `test_check_feature_quality_rejects_only_all_nan`、
    `test_check_feature_quality_accepts_market_state_broadcast`、
    `test_generate_all_zero_column_warns_but_predicts`、
    `test_generate_ranked_rejects_all_nan_column`、
    `test_generate_passes_quality_gate_with_good_data`。

## [0.94.13] - 2026-08-08

### Fixed

- **修复龙虎榜（lhb）因子三审问题（0.94.12 的补充）**：
  - **groupby.first() 拼接不同记录字段**：`src/lazybull/factors/lhb.py` 原先排序后
    用 `groupby.first()` 选择同日代表记录，它会逐列取首个非空值，当净买入最大那条
    的 `net_rate`/`amount_rate` 缺失时会把另一条记录的字段拼接进来，生成不存在的
    行（真实历史影响 1 个股票日）。改为按 `(trade_date, ts_code, is_cont, abs)`
    排序后 `drop_duplicates(subset=["trade_date", "ts_code"])` 保留第一整行；
  - **补充关键接线测试**：
    - `tests/test_pipeline_lhb_wiring.py`（新增）：断言 `features/pipeline.py`
      批量构建龙虎榜时确实把含预热期的完整日历作为 `calendar_dates` 传入（防止
      以后重构时预热接线复发）；并覆盖 `scripts/raw_download/alt.py download_top_list`
      对近期空占位分区重新查询覆盖、非近期空占位跳过两个分支；
    - `tests/test_factor_lhb.py`：新增"同日组内不同列 NaN 分布时保留第一整行
      不拼接字段"回归用例；
  - 影响：`lhb_*` 列名与 schema 不变；修复极小概率的字段错配。

## [0.94.12] - 2026-08-08

### Fixed

- **修复龙虎榜（lhb）因子二轮审计问题（0.94.11 的补充）**：
  - **批量构建缺少预热日历**：`features/pipeline.py` 批量构建时输出交易日从
    `start_date` 开始，未传预热期交易日历，导致加载的前 7 个月 top_list 数据在
    `reindex` 时被丢弃，区间前 19 个交易日的 `lhb_up_days_20` / `lhb_net_sum_*`
    历史累计为空，造成批量构建与单日推理不一致。现通过
    `loader.get_trading_dates(start_dt, end_date)` 获取含预热期的完整日历并作为
    `calendar_dates` 传入；
  - **仅连续类理由被误标为未上榜**：`lhb.py` 原先无条件删除所有含"连续"的
    reason 记录，若股票当天只有连续异动理由（2024-2026 共 11816 个股票日）会被
    误标为未上榜。现改为：同日同时存在单日榜与连续类时优先单日榜；仅连续类时
    保留（`lhb_on_list=1`）；`lhb_reason_count` 与选中类别一致；
  - **净额全 NaN 组 idxmax 崩溃**：`lhb.py` 原先对净买入全 NaN 的分组调用
    `idxmax()` 返回 NaN 索引导致 `df.loc[idx]` 崩溃。现改为
    `sort_values(na_position="last") + groupby.first()`，全 NaN 组取第一条，不崩溃；
  - **已有假空分区不会被修复**：新逻辑只能阻止新的近期空响应落盘，已存在的
    0 行空占位（如 2026-08-03）因 `is_data_exists` 检查通过仍被永久跳过。现在下载
    与推理补齐对"10 个自然日内"已存在的空占位分区重新查询一次，查询到真实数据则
    覆盖旧占位；
  - 测试：`tests/test_factor_lhb.py` 新增"仅连续类理由保留""净额全 NaN 不崩溃"
    "日历预热一致性"用例；`tests/test_lhb_false_empty_guard.py` 新增"近期空占位
    重新下载覆盖""非近期空占位跳过"用例；
  - 影响：`lhb_*` 列名与 schema 不变；仅连续类理由的股票日恢复真实上榜标记。

## [0.94.11] - 2026-08-08

### Fixed

- **修复龙虎榜（lhb）因子三处实质性问题 + 空响应防假空**：
  - **滚动窗口语义错误**：`lhb_up_days_20` / `lhb_net_sum_5` / `lhb_net_sum_20`
    原先按"上榜事件条数"滚动（top_list 稀疏，20 条记录可能跨越数年），实际含义是
    "历史上榜次数封顶 20 / 最近 N 次上榜的净买入累计"，而非"近 5/20 个交易日"。
    `src/lazybull/factors/lhb.py` 改为按完整交易日历重采样（未上榜日补 0）后再滚动，
    并新增 `calendar_dates` 参数供单日推断传入历史交易日历；
  - **时序连续性缺失**：lookup 原先只输出当日上榜股票，一笔上榜事件次日即归零，
    历史累计只在再次上榜当天出现。现每个交易日输出所有"近 20 个交易日内上过榜"
    的股票（含当日未上榜者），`lhb_up_days_20` 等历史累计在非上榜日持续衰减保留；
  - **同日多理由重复放大**：同一股票同日可能同时给出单日榜与"连续 N 个交易日"榜
    （统计周期重叠，直接求和会翻倍甚至符号反转）。现排除"连续"类 reason，并对
    同周期重复行取净买入绝对值最大的一条（禁止 sum）；`lhb_reason_count` 改为
    当日去重后的理由数；
  - **空响应防假空**：下载与推理补齐对"近期日期（3 个自然日内）"的空响应不再落盘
    占位，延迟重试，避免接口数据未发布时被过早下载成"假空"永久缓存、真实数据丢失；
    历史日期的空占位统一为 0 行 6 列（与下载脚本 schema 一致，加载时自动过滤）；
    ensure 原先的 1 行 1 列空占位会触发 `build_lhb_lookup_by_date` 异常，已由
    ts_code 缺失防御 + 占位 schema 统一双重修复；
  - 单日推断（纸面交易）的滚动日历裁剪到近 40 个交易日，避免每次对全历史重算；
  - 测试：`tests/test_factor_lhb.py` 重写为按交易日窗口 / 时序连续性 / 排除连续类 /
    同日取绝对值最大语义；新增 `tests/test_lhb_false_empty_guard.py` 覆盖近期空响应
    不落盘与历史空占位 schema；
  - 影响：`lhb_*` 列名与 schema 不变，但取值语义变化较大（窗口口径、聚合口径、
    时序连续性），建议重建特征后重新训练并对比 walk-forward 稳定性。

## [0.94.10] - 2026-08-07

### Fixed

- **修复 fund_portfolio 变化量因子的披露口径断层**：
  - 背景：公募基金一季报/三季报**仅披露前十大重仓股**，半年报/年报**披露全量持仓**。
    `fund_hold_ratio` / `fund_count` 按季度相邻作差时，Q1↔H1、H1↔Q3、Q3↔年报
    四对里有三对跨口径，`fund_hold_ratio_chg` / `fund_count_chg` 实际上大部分时候
    在度量"披露口径切换"而非"机构增减持"，噪声量级大于 v0.94.9 修掉的前视；
  - `src/lazybull/factors/fund_portfolio.py`：
    - `_prev_quarter_end()` 改为 `_prev_same_scope_end()`，返回**同口径上一报告期**
      （相隔两个季度：0331←上年 0930、0630←上年 1231、0930←0331、1231←0630）；
    - 变化量不再用 `shift(1)` 取排序后的紧邻记录，改为按 `(symbol, 同口径上期 end_date)`
      精确 `reindex` 查找；该报告期缺失（如某季度无基金持有）则为 NaN；
    - 保留 v0.94.9 的延迟披露校验：同口径上期公告日晚于本期时置 NaN；
  - `src/lazybull/ml/train_core/constants.py`：同步更新 `FUND_FEATURE_COLUMNS` 注释；
  - 测试：`tests/test_fund_portfolio_factor.py` 改造环比用例为同口径语义，新增
    "相邻季度跨口径不作差"与"年报对半年报"两个用例；
  - 影响：列名与 schema 不变；跨口径的相邻季度作差由错误取值变为 NaN，同口径期间
    的变化量口径统一为"半年度变化"。因子取值分布变化较大，建议重建特征后重新训练。

## [0.94.9] - 2026-08-07

### Fixed

- **修复公告型因子跨期计算的前视泄漏（express 同比 / fund_portfolio 环比）**：
  - 背景：`build_latest_announcement_lookup_by_date` 的 PIT 门控只作用于
    **当前记录的 ann_date**，但因子值内部引用了其它报告期的数据，这部分跨期引用
    此前完全没有公告日约束；
  - `src/lazybull/factors/express.py`：`_compute_revenue_yoy` 原先用全量数据构建
    `(ts_code, end_date) -> revenue` 字典，因入参已按 ann_date 排序，同一报告期
    保留的是公告日最晚的版本。若去年同期存在晚于本期快报的重述/更正公告，
    `express_revenue_yoy` 会在本期公告日就用上未来信息。改为按
    `(ts_code, end_date) -> (ann_date 升序列表, revenue 列表)` 存储，用
    `bisect_right(本期 ann_date) - 1` 取本期公告日当天及之前已披露的最新版本
    （与同文件 `express_surprise` 的 `fc_lookup` 口径统一）；取不到则为 NaN；
  - `src/lazybull/factors/fund_portfolio.py`：季度环比原先仅按 `end_date` 排序后
    `shift(1)`，未校验上期公告日。新增 `_prev_quarter_end()` 与两项校验，满足任一
    条件即把 `fund_hold_ratio_chg` / `fund_count_chg` 置 NaN：
    - 上期聚合公告日晚于本期（基金延迟披露/补充更正，构成前视）；
    - 上期不是紧邻上一季度（该股某季度无基金持仓导致记录缺失，环比口径不可比）；
  - 顺带把 `_compute_revenue_yoy` 的双重 `iterrows()` 改为列级 `zip` 遍历，并补充
    `end_date` 数字校验，避免异常报告期触发 `int()` 抛错；
  - 测试：`tests/test_express_factor.py` 新增未来重述不采用 / 已公开重述采用两个用例；
    `tests/test_fund_portfolio_factor.py` 新增上期延迟披露、上期非紧邻季度两个用例；
  - 影响：仅收紧上述两个因子的可用信息边界，正常披露路径下取值不变；异常路径下
    由错误取值变为 NaN，`freshness_days` 与 schema 均不受影响。

## [0.94.8] - 2026-08-07

### Fixed

- **修复纸面交易链路 report_rc 加载触发 pandas concat FutureWarning**：
  - 背景：纸面交易 run 加载"一致预期研报（report_rc）"时，`load_report_rc` 合并按年
    分区的 DataFrame，部分分区存在整列全 NA，触发 pandas FutureWarning
    （DataFrame concatenation with empty or all-NA entries is deprecated），
    日志被黄色告警刷屏（每次 run 出现两次）；
  - 根因：`src/lazybull/data/loader.py` 的 `load_report_rc` 使用裸
    `pd.concat(dfs, ignore_index=True)`，未像 `storage.py` 那样屏蔽该告警；
  - `src/lazybull/data/loader.py`：新增 `import warnings`，`load_report_rc` 的
    concat 使用 `warnings.catch_warnings()` + `filterwarnings` 精确屏蔽该
    FutureWarning（与 storage.py 统一模式一致，不改动任何数据）；
  - 影响：仅消除告警刷屏，concat 结果与 schema 完全不变。

## [0.94.7] - 2026-08-07

### Fixed

- **修复部署训练大窗口 OOM（MemoryError）**：
  - 背景：部署训练使用长训练窗口（如 20200206 ~ 20260205，约 610 万样本 × 293 列）
    时，`_filter_to_main_board` 中 `df[布尔掩码].copy()` 触发 `Unable to allocate
    8.76 GiB` OOM；
  - 根因：布尔掩码索引 `df[mask]` 本身已按行 take 复制出独立数据（约 8.76 GiB），
    紧随的 `.copy()`（deep=True）再次触发 pandas `BlockManager._consolidate_inplace`
    把 293 个列块合并为一块 293 × 4014232 的连续 float64 数组（又 8.76 GiB 连续内存），
    峰值内存翻倍导致 OOM；
  - `src/lazybull/ml/walk_forward/training_core.py`：`_filter_to_main_board` 去除
    冗余 `.copy()`（布尔索引已生成独立副本），消除 consolidate 峰值分配；
  - `src/lazybull/ml/train_core/prepare.py`：训练样本过滤 `df[mask].copy()` 去除
    冗余 `.copy()`（后续 `df_train[col] = np.nan` 等写操作作用于独立副本，不写穿
    原 df），消除同链路下一个同类 OOM 风险点；
  - 影响：仅降低峰值内存，过滤语义与训练结果不变。

## [0.94.6] - 2026-08-06

### Fixed

- **全数据集下载完整性审计：修复 4 个数据集的单次查询截断**：
  - 背景：对全部 19 个 raw 数据集做整体筛查（pyarrow 元数据扫描 + TuShare 单次
    limit 上限实测），发现多个数据集存在与 share_float 同类的"单次查询被服务端
    limit 截断、分页未生效"问题；
  - 实测各接口单次 limit 上限：cashflow_vip=6400、fina_indicator_vip=12000、
    pledge_stat=3000、stk_limit=6000、express_vip=5000、forecast_vip=6500、
    fund_portfolio=8000；daily/daily_basic/moneyflow/margin_detail/cyq_perf/
    adj_factor/suspend/stock_st/top_list 单日数据量均低于上限（无截断）；
  - **cashflow**（23 个季度恰好 6400 行被截断）：`download_cashflow` 设
    `page_limit=6400`（旧默认 50000 导致首屏 6400 被误判取完）；实测重下后
    恰好 6400 季度 23→0、总行数 +18%；
  - **fina_indicator**（2 个季度恰好 12000）：cli 调用设 `page_limit=12000`；
  - **pledge_stat**（3 期恰好 3000）：`download_pledge_stat` 改用
    `_query_with_pagination(page_limit=3000)` 分页；实测 2017Q1/Q2、2019Q3
    从 3000 → 3123/3252/3171；
  - **stk_limit**（单日约 7411 条 > 6000 上限被截断）：新增
    `_fetch_stk_limit_paginated` 按日分页 fetcher，`download_daily_data` 支持
    `subsets` 参数，CLI 支持 `--download stk_limit`（及 daily/daily_basic/
    adj_factor/suspend/moneyflow/stock_st 日线子集单独下载）；实测单日
    7397-7411 条取全；
  - 确认无截断：forecast（未触及 6500）、express（每期 <5000）、stk_holdernumber
    （page_limit=3000 已正确）、fund_portfolio（page_limit=8000 已正确）；
    moneyflow/stk_limit 的分区空洞为数据源起始时间（2005-2006 无数据），非下载不全；
  - **多记录处理审计**（问题 B）：逐一检查全部 lookup builder——top_list 已按
    (trade_date,ts_code) 聚合（sum/mean/count）、fund_portfolio 已预聚合、
    block_trade 已按日聚合、PIT 多版本（fina/cashflow/earnings/express/holder）
    取最新版本、consensus 滚动聚合，均无"同事实多行只取一条"偏差
    （share_float 多持有人问题已在 v0.94.4 修复）；
  - **待重下**：`python scripts/download_raw.py --download fina_indicator --force`
    （补齐 2 个截断季度）；cashflow/pledge_stat/stk_limit 已在本轮实测重下。

## [0.94.5] - 2026-08-06

### Fixed

- **修复 `share_float` 单日公告超 6000 条被截断（下载缺约 45%）**：
  - 现象：检查全量重下后的各年分区，发现 2017/2019/2021/2022/2023/2024 年大量
    交易日"满额 6000 条"（2023 年 132/242 个交易日满额、单日中位数=6000），
    疑似单日查询被 TuShare 单次 limit 上限 6000 截断；2020 年 0 天满额
    （max 3604），确认 2020 年下载完整、数据源本就偏少（非下载失败）；
  - 根因：`_query_share_float_by_ann_date` 用 `client.query(ann_date=date)`
    直接单次查询、无 offset 翻页，解禁明细高峰日单日公告超 6000 条被截断；
  - 修复（`scripts/raw_download/announcement_risk.py`）：改为 `_query_with_pagination`
    按 6000 粒度 offset 翻页取全单日全部公告（复用既有翻页模板）；
  - 效果（实测 2023 年重下）：93.6 万条 → **171.6 万条**（截断漏 45%），
    242 个交易日全部成功、64 秒、零错误；
  - **注意**：其他年份（2017/2018/2019/2021/2022/2024 等含满额天）分区仍为
    截断数据，需重跑全量：`python scripts/download_raw.py --download share_float
    --start-date 20120101 --end-date 20261231 --force`；
  - **测试**：`tests/test_announcement_lookup.py` 新增
    `test_query_share_float_by_ann_date_paginates_full_day`（mock 单日 6000+2000
    条分页取全，断言 8000 条全部取回）。

## [0.94.4] - 2026-08-06

### Fixed

- **修复 `unlock_ratio` 因子对同批多持有人解禁只取单条记录导致严重低估**：
  - 现象：`share_float` 原始数据中同一 (ts_code, float_date)（同一批解禁）有大量
    持有人记录（80.6% 的批次多条，最多 6000 条），`float_ratio` 是**单持有人**
    占总股本比例（百分比 0-100，多数 0.0002%）；
  - 旧实现 `build_share_float_lookup_by_date` 取"已公告未解禁"中最近解禁日
    **一条**记录，`unlock_ratio` 变成单持有人比例（0.0002% 级），而非该批
    解禁总比例（实测聚合后中位数 10%），低估约百倍；
  - 修复（`src/lazybull/factors/risk/announcement_lookup.py`）：先按
    `(ts_code, float_date)` 分组，`float_ratio` 求和（同批解禁总比例，实测
    中位数 10.1%、最大 99.5% <100% 合理）、`ann_date` 取该批最早公告日，
    再走原 PIT 逻辑（取最近解禁日一条）；`days_to_unlock` 不受影响；
  - 效果（真实数据）：20180115 截面 `unlock_ratio` median 从 0.0002% 级 →
    4.76%（24 只有解禁，max 77.33%）；
  - **测试**：`tests/test_announcement_lookup.py` 新增
    `test_share_float_lookup_aggregates_multi_holder_same_float_date`
    （603080.SH 两条 33.09+5.26 → 38.35）；
  - **注意**：`unlock_ratio` 因子列需重新 build features 后进入 cs_train。

## [0.94.3] - 2026-08-06

### Fixed

- **修复 `share_float` 下载器查询语义错误导致解禁数据严重缺失**：
  - 现象：2018/2019/2021/2023 年分区只剩 12 月公告的残缺数据（2019 年仅 35 条），
    解禁因子 `days_to_unlock`/`unlock_ratio` 在 features 里几乎全空（非空率 0%）；
  - 根因 1（语义）：TuShare `share_float` 的 `start_date/end_date` 语义是
    **float_date（解禁日）** 而非 ann_date（公告日），旧代码按 float_date 查询后
    再按 `ann_date[:4]==year` 过滤，只保留"当年公告且当年解禁"的记录，而大部分
    "前几年公告、当年解禁"的记录被误删；
  - 根因 2（翻页）：按 float_date 查询返回解禁明细（每年 30-55 万条），且单次
    `limit` 上限 6000、offset 深翻页有上限（约 100000，超限报"查询数据失败"），
    旧实现 `page_limit=20000` 导致首屏 6000 条被误判"已取完"，每年只取到
    float_date 最晚（12 月末）的一批；
  - 修复（`scripts/raw_download/announcement_risk.py`）：改为按 **ann_date 单值
    逐交易日查询**（`ann_date=YYYYMMDD`，每天公告约 10-2000 条，单次查询即取全、
    无分页），用交易日历生成目标公告年内交易日，并发 8 逐日拉取，按公告年分组落盘；
  - 效果（实测 2018-2019）：2018 分区从 5713 条（仅 12 月）→ 340049 条（1-12 月
    全年覆盖），2019 从 35 条 → 551711 条；487 个交易日 40 秒完成、零错误；
  - **注意**：其他年份（2012-2017/2020-2026）分区仍为旧逻辑残缺数据，需重跑
    `python scripts/download_raw.py --download share_float --start-date 20120101
    --end-date 20261231 --force` 全量覆盖；
  - **测试**：更新 `tests/test_announcement_lookup.py` 下载器回归测试为
    ann_date 逐日查询语义 + 公告年分区 + `get_trade_cal` 调用断言。

## [0.94.2] - 2026-08-06

### Fixed

- **修复质押因子阈值单位 bug（`pledge_ratio` 为百分比口径 0-100）**：
  - `pledge_high_flag` 分档阈值原按小数写（`> 0.50` / `< 0.30`），实际数据为百分比
    （median≈3.6、max≈81），导致 0.5% 以上的质押率全部误判高危——真实构建的
    `cs_train` 中 56.7% 股票被标记高危（median=1.0），而真实 >50% 的高危仅 0.44%；
    修复为 `> 50` / `< 30`（百分比口径），实测高危从 2656 只降到 12 只；
  - `pledge_delta` 的实质变化阈值原为 `abs() < 0.005`（0.005 个百分点，几乎不清零），
    修复为 `< 0.5`（0.5 个百分点才算实质变化）；
  - 删除 `announcement_factors.py` 中未被使用的 `_align_to_df` 死代码及未使用 import；
  - **测试**：`tests/test_announcement_lookup.py` 新增
    `test_pledge_high_flag_percentage_threshold`（含 0.6%/29/30/50/51/75.09 边界）与
    `test_pledge_delta_threshold_percentage_points`（0.4 清零 / 1.0 保留）；
    端到端测试数据统一为百分比口径；相关测试全部通过。
  - **注意**：修复后需重新 build features（`pledge_high_flag`/`pledge_delta` 列才会
    以正确口径进入 cs_train）。

## [0.94.1] - 2026-08-06

### Fixed

- **修复风控公告类因子与 handler 原始列重名导致 build_clean_features 失败**：
  - 现象：启用 `--enable-announcement-risk-features` 真实构建时，首个交易日即报
    `Duplicate column names found`（重复列 `unlock_ratio` / `block_discount_avg_10d` /
    `block_discount_days_10d` / `short_balance_change_5`），build 中断；
  - 根因：`factor_handlers` 已把原始列（如 `unlock_ratio`）合并进 features，而
    `announcement_factors.py` 中 4 个透传型因子（`compute_unlock_ratio` /
    `compute_block_discount_avg_10d` / `compute_block_discount_days_10d` /
    `compute_short_balance_change_5`）读取同名列并原样返回，`_attach_risk_factors_static`
    再次 `pd.concat` 时列名重复。此前测试未暴露是因为无数据时 handler 走 NaN 占位、
    features 中不存在这些原始列，透传型因子返回 NaN 占位不冲突；
  - 修复：`factors/risk/factor_registry.py` 的 `compute_all_risk_factors` 统一跳过
    `name in df.columns` 的因子（透传型因子不再重复输出同名列），覆盖串行
    `_add_risk_factors` 回退路径与 `_attach_risk_factors_static` 主路径；
    加工型因子（`pledge_high_flag` / `unlock_risk_flag` 等输出名 ≠ 输入列名）不受影响；
  - **测试**：`tests/test_announcement_lookup.py` 新增
    `test_attach_risk_factors_no_duplicate_columns`（handler 合并原始列后
    `_attach_risk_factors_static` 无重复列、原始列保留、加工因子仍生成），
    更新端到端断言（透传列不在因子层重复输出）；相关测试全部通过。

## [0.94.0] - 2026-08-06

### Added

- **风控公告类因子侧接线（质押/解禁/大宗原始列接入 features）**：完成 0.93.0
  下载链 → 因子生效的最后一环，使 `announcement_factors.py` 的 7 个公告因子
  （质押 3 + 解禁 2 + 大宗 2）从「全 NaN 被有效性过滤剔除」变为真实计算：
  - `src/lazybull/factors/risk/announcement_lookup.py`（新建）：3 个 PIT 日频
    查询表 builder（与 fundamental/consensus_revision 同模式），把低频公告数据
    转换为 `{trade_date: DataFrame}` 当日截面：
    - `build_pledge_lookup_by_date`：按公告日 `ann_date`（缺失回退 `end_date`）
      前向填充，输出 `pledge_ratio` / `pledge_freshness_days` / `pledge_ratio_prev`；
    - `build_share_float_lookup_by_date`：仅 `ann_date <= T` 且 `float_date > T`
      （未解禁）的公告可见，取最近解禁日一条，输出 `days_to_unlock` / `unlock_ratio`；
    - `build_block_trade_lookup_by_date`：折价率 =（成交价 - 未复权收盘价）/ 收盘价，
      按近 10 个交易日聚合，输出 `block_discount_avg_10d` / `block_discount_days_10d`；
    - **修复**：YYYYMMDD 不能直接整数相减算自然日差（20240301-20240210=91 但
      实际 20 天），统一经 datetime64 转自然日；大宗聚合窗口延伸到最后一笔交易后
      9 个交易日（折价影响持续存在）；
  - `src/lazybull/features/handlers_announcement.py`（新建）：3 个 FactorHandler
    （`PledgeFactorHandler` / `ShareFloatFactorHandler` / `BlockTradeFactorHandler`），
    将当日截面原始列合并进 features；空数据输出 NaN/0 占位（schema 稳定）；
  - `FeatureContext` + `builder/orchestration.py` 新增 `pledge_data` /
    `share_float_data` / `block_trade_data` 三个字段（向后兼容透传）；
  - `features/factor_handlers.py`：注册 3 个新 handler 并补充失败占位默认列；
  - `features/pipeline.py`：新增 `enable_announcement_risk` 开关，加载三类原始
    数据并构建日频查询表（并行 + 串行双路径）；
  - `features/ensure/factor_load.py` + `entry.py`：纸面交易链路自动加载三类数据
    （有则加载、缺失记 missing；自动下载补齐为下一步工作）；
  - `scripts/build_clean_features.py`：新增 `--enable-announcement-risk-features`
    开关（纳入 `--build-all`）；
  - **测试**：`tests/test_announcement_lookup.py`（新建，10 个测试）覆盖 PIT
    前向填充、自然日差、解禁清零、大宗窗口聚合、handler 占位/合并、端到端因子
    计算；`tests/test_ensure_and_t0_printing.py` 更新适配 16 元组返回；
    既有 features/ensure/下载测试全部通过。

## [0.93.0] - 2026-08-06

### Added

- **风控公告类数据下载链（质押/解禁/大宗）**：新增 3 个数据集的完整下载链路，
  全部复用现有下载模板（`_download_by_trade_date` / `_query_with_pagination` /
  `_generate_quarter_periods` / `_run_concurrent`），零新并发/限频/断点续传架构：
  - `scripts/raw_download/announcement_risk.py`（新建）：
    - `block_trade`（大宗交易）：按 `trade_date` 逐日查询 → 日分区
      `raw/block_trade/{YYYY-MM-DD}.parquet`（对齐 margin_detail/top_list），
      全量约 4400 交易日，预计 10-20 分钟；
    - `pledge_stat`（股权质押统计）：按 `end_date`(季末) 查询 → 季分区
      `raw/pledge_stat/{YYYY-MM-DD}.parquet`（对齐 fina_indicator），
      全量约 52 期，预计 1-3 分钟；
    - `share_float`（限售解禁）：按 `ann_date` 年区间查询 → 年分区
      `raw/share_float/{YYYY}-12-31.parquet`（对齐 report_rc），
      **PIT 契约**：按公告日分区/过滤（`ann_date <= T` 才可见，`float_date` 为未来解禁日），
      全量约 22 年，预计 5-15 分钟；
  - `scripts/raw_download/cli.py`：`--download pledge_stat/share_float/block_trade`
    接入分发，`ALT_DATASETS`（`scripts/raw_download/core.py`）纳入 3 数据集
    （`--all` 一并下载）；
  - `src/lazybull/data/loader_announcement.py`（新建 Mixin）：`DataLoader` 组合
    该 Mixin，新增 `load_pledge_stat` / `load_share_float` / `load_block_trade`
    三个加载方法（日/季/年分区路由，日期统一 YYYYMMDD）；
  - 接口级限频写入 `_API_RATE_LIMITS_DEFAULT`（block_trade=200 / pledge_stat=120 /
    share_float=120，仅当无配置或更宽松时写入，不覆盖用户配置）；
  - 因子侧接线（FeatureContext + factor_handlers 合并 pledge/unlock/block 原始列）
    为下一步工作，本次仅落地下载与加载；
  - **测试**：mock 验证三个下载器的查询参数与分区路由（含 share_float 跨年
    `ann_date` 归位）；既有下载测试 42 个全部通过。

## [0.92.4] - 2026-08-06

### Changed

- **重构：风控因子统一归入 `src/lazybull/factors/risk` 子包**：
  - 将 `src/lazybull/risk/` 下的 7 个风控因子模块迁移到 `src/lazybull/factors/risk/`：
    `factor_registry`、`downside_factors`、`volatility_factors`、`liquidity_factors`、
    `announcement_factors`、`derived_factors`、`position_features`；
  - 依据：因子构建统一归入 `factors/`（风险因子放 `factors/risk` 子目录），
    `risk/` 仅保留风控逻辑（止损/止盈/PositionRiskModel/label_builder/precompute）；
  - 更新引用：`features/builder/factors.py`、`features/builder/static_extra.py`、
    `tests/test_risk_precompute.py` 的导入路径改为 `...factors.risk.factor_registry`；
  - 新增 `src/lazybull/factors/risk/__init__.py`（导出 registry 三接口）；
  - **测试**：`tests/test_risk_precompute.py`（13 passed）验证迁移后无回归。

- **风控模型特征增强（融券因子 + 截面百分位归一化）**：
  - `src/lazybull/factors/margin.py` 新增 2 个融券因子（零新下载，复用已有
    margin_detail）：`short_balance_change_5`（融券余额 5 日变化率，rqye）、
    `short_sell_vol_change_5`（融券卖出量 5 日变化率，rqmcl）；经
    `build_margin_lookup_by_date` → FeatureContext → MarginFactorHandler
    全链路自动并入 features（含 ensure 增量路径）；
  - `scripts/train_position_risk_model.py` 新增截面百分位归一化特征
    （`--add-pct-features`，默认开启）：对 24 个绝对值跨市场环境漂移严重的
    因子（动量/波动率/流动性类，诊断证实 `ret_20` 月均值漂移 8.6 倍）按
    交易日截面 `rank(pct=True)` 转 0~1 相对分位，消除牛熊市阈值漂移，
    输出 `pct_<原名>` 列并自动加入候选列表；
  - **注意**：融券列需重新 build features（build_clean_features）后进入
    cs_train/cs_infer；截面百分位特征为训练期现算，无需新列。

## [0.92.3] - 2026-08-05

### Fixed

- **修复纸面交易非调仓日缺数据不自动下载导致止损跳过与持仓打印崩溃**：
  - 根因：`execute_trade_workflow` 仅在 T0 调仓日生成信号（`ensure_features_for_date`
    自动下载）或 T1 有指令时才触发数据下载；非调仓日/无指令时只做只读操作
    （止损检查、持仓打印），直接加载 clean 数据——缺失时止损检查仅 warning 跳过、
    `load_position_snapshot` 直接抛 `ValueError` 退出（复现：`trade_date=next` 解析
    为 20260723，距上次调仓 17 个交易日 < 20 非调仓日且无 T1 指令，当日 clean 数据
    从未下载）；
  - `src/lazybull/paper/runtime.py`：`execute_trade_workflow` 校正日期后、T1 前主动
    调用 `ensure_clean_data_for_date` 补齐当日 clean 数据（缺失时内部触发
    `ensure_raw_data_for_date` 自动下载），补齐失败仅 warning 不阻断主流程（保持
    原降级语义）；修复同时惠及钉钉机器人链路（`bot_service.py` 复用同一 runtime）；
  - `src/lazybull/paper/reporting.py`：`load_position_snapshot` 加载前同样补齐当日
    clean 数据，避免 `positions` 命令/run 后打印因缺数据直接崩溃；
  - **测试**：`tests/test_paper_trade_runtime.py` 新增
    `test_execute_trade_workflow_ensures_trade_date_clean_data` 与
    `test_execute_trade_workflow_continues_when_clean_data_ensure_fails`；
    `tests/test_ensure_and_t0_printing.py` 新增
    `test_load_position_snapshot_ensures_trade_date_clean_data`。

## [0.92.2] - 2026-08-05

### Fixed

- **修复 walk-forward 多窗口集成子模型特征列不一致导致训练失败**：
  - 根因：多窗口（基础/前移/后移）集成时，每个子模型独立在各自训练窗口上执行
    “训练入口特征质量门禁”（缺失率 > 0.6 移除特征），不同窗口稀疏因子缺失率
    不同，导致子模型间特征 schema 不一致（如 `express_revenue_yoy` 在基础/前移
    窗口缺失率 > 0.6 被移除、在后移窗口保留），集成预测时按基础窗口特征列调用
    全部子模型，XGBoost 报 `feature_names mismatch`；
  - `src/lazybull/ml/train_core/prepare.py`：`prepare_training_data` 新增
    `feature_columns_override` 可选参数，训练入口特征质量门禁之后强制将特征列
    对齐到指定列表，数据中缺失的列补 NaN（默认 None，行为不变）；
  - `src/lazybull/ml/walk_forward/training_core.py`：`_train_model_on_window`
    透传 `feature_columns_override`；`_build_ensemble_sub_models` 以首个（基础
    窗口）子模型特征列为准，强制后续子模型对齐，保证集成内子模型特征 schema
    完全一致（部署链路 `deploy_training.py` 复用同一入口，同步受益）；
  - **测试**：`tests/test_train_core_val_embargo.py` 新增
    `test_prepare_training_data_feature_columns_override` 与
    `test_prepare_training_data_without_override_removes_high_missing`；
    `tests/test_training_feature_flag_forwarding.py` 新增
    `test_ensemble_sub_models_align_feature_columns_to_base_window`，并同步
    更新两个 fake `_train_model_on_window` 的签名。

## [0.92.1] - 2026-08-04

### Changed

- **clean 层日志精简：build_clean_features 逐日仅保留一行进度**：
  - `src/lazybull/data/cleaner.py` 所有步骤级 INFO 日志（开始清洗/清洗完成/
    复权价格计算/可交易标记统计等）统一收敛到 `verbose` 门控下，仅在
    `DataCleaner(verbose=True)` 时输出；
  - `scripts/build_clean_features.py` 使用默认 `verbose=False` 的清洗器，逐日
    仅保留 `[i/N] (x%) 处理 YYYYMMDD...` 一行进度，异常类 warning 保持不变；
  - `src/lazybull/data/build_clean.py` 逐日的“已保存 clean 记录”与“clean daily
    已存在，跳过”日志降级为 debug；
  - 纸面交易链路（`PaperTradingRunner` 默认 `verbose=True`）行为不受影响；
  - **测试**：`tests/test_cleaner.py` 新增
    `test_step_logs_gated_by_verbose`（默认安静、verbose=True 输出步骤日志）。

- **一致预期（report_rc）因子恢复进入训练链路**：
  - `data/models/factor_exclude_list.json` 移除 20 个 report_rc 相关排除项
    （`cons_*` / `zscore_cons_*`，53 → 33），因子精简不再拦截一致预期因子；
  - `src/lazybull/ml/train_core/prepare.py` 的 `max_feature_missing_ratio`
    默认值 0.4 → 0.6，使缺失率约 50~60% 的一致预期基础因子通过训练入口
    缺失率门禁；
  - 效果（模拟验证）：打开 `--enable-consensus-features` 后，
    `cons_analyst_count_30d` / `cons_eps_mean_fy0` / `cons_eps_mean_fy1` /
    `cons_eps_mean_fy2` / `cons_rating_score` 共 5 个因子实际进入训练；
  - 仍被门禁剔除（数据源本身缺失过高，82%+）：`cons_eps_revision_30d` /
    `cons_target_price_mid` / `zscore_cons_*` 一致预期修正因子；
  - **注意**：`factor_exclude_list.json` 由 `generate_factor_exclude_list.py`
    重新生成时会覆盖手工调整；`max_feature_missing_ratio=0.6` 为全局阈值，
    会同时放宽其他因子组的缺失率门槛。

## [0.92.0] - 2026-08-03

### Added

- **一致预期基础因子新增按 quarter 预测财年分组的 EPS 均值因子**：
  - `report_rc` 同一 `report_date` 会同时含多个预测季度（如 2024Q4/2025Q4/2026Q4），
    研报内各季度预测相互独立；
  - `src/lazybull/factors/consensus.py` 新增 `_parse_quarter_year` 解析研报
    `quarter` 预测年份，相对 `report_date` 发布年份定位财年位置
    （FY0=当年, FY1=次年, FY2=后年）；
  - 新增 `cons_eps_mean_fy0`（当前财年）与 `cons_eps_mean_fy2`（未来第二财年），
    `cons_eps_mean_fy1` 作为 FY1（未来第一财年）分组因子；
  - `cons_eps_revision_30d` / `cons_target_price_mid` / `cons_rating_score` /
    `cons_analyst_count_30d` 均保持原语义不变；`quarter` 缺失时 EPS 财年分组列
    优雅降级为 NaN；
  - `src/lazybull/ml/train_core/constants.py` 同步更新
    `CONSENSUS_FEATURE_COLUMNS` 与
    `EVENT_FRESHNESS_TO_VALUE_COLUMNS["consensus_freshness_days"]`，纳入
    `cons_eps_mean_fy0` / `cons_eps_mean_fy2`；
  - **测试**：`tests/test_factor_consensus.py` 数据补齐 `quarter`，新增
    `test_consensus_eps_mean_by_fy`（FY0/FY1/FY2 分组断言）、
    `test_consensus_revision_all_periods_mix`（revision 全预测期口径回归）与
    `test_consensus_without_quarter_column_degrades`（缺失 quarter 优雅降级）；
  - **注意**：新增/变更列需重新构建 features（build_clean_features）并重训模型
    后方可用于新模型。

## [0.91.0] - 2026-08-03

### Added

- **`download_raw` 启动时自动绕过终端/系统注入的代理**：
  - PowerShell 等终端常通过环境变量注入 HTTP(S) 代理（如 `http://192.168.1.21:18081`），
    导致 TuShare 请求走内网代理并出现 `Read timed out`；
  - `scripts/raw_download/cli.py` 在 `main()` 启动时于进程内清除
    `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY`（含小写），使 TuShare/requests 直连，
    仅影响当前进程、不修改终端/系统设置；
  - 开关：`LAZYBULL_DOWNLOAD_BYPASS_PROXY=0` 可关闭（默认启用，单变量直读、单默认值）；
  - 新增 `tests/test_download_raw_proxy_bypass.py` 覆盖默认清除、开关关闭、空操作
    三种场景。

## [0.90.29] - 2026-08-03

### Changed

- **`forecast` / `report_rc` 由超大独立单文件改为按时间分区存储**，与
  `fina_indicator`/`cashflow`/`fund_portfolio` 对齐：
  - `forecast` 按季度 `end_date` 分区（`data/raw/forecast/{YYYY-MM-DD}.parquet`），
    离线下载复用 `download_by_period(partition_by_period=True)`，增量补齐按
    `end_date` 路由写对应季度分区；
  - `report_rc` 按年 `report_date` 分区（`data/raw/report_rc/{YYYY}-12-31.parquet`），
    `download_report_rc` 按年独立落盘（保留自适应二分与保守并发），增量补齐按
    年份路由写分区；
  - 新增 `_append_and_save_partitioned` / `_partition_date_str` /
    `_load_all_partitions`（`features/ensure/incremental.py`），分区内去重、
    跨分区天然隔离，增量补齐从"整文件读-合并-重写 (O(全量))"降为"只写增量分区"；
  - `DataLoader.load_forecast` / `load_report_rc` 改为纯分区加载（复用
    `_load_quarter_partitioned_raw` / `list_partitions + concat`），不再保留
    旧单文件路径；
  - 修复潜在 bug：`ensure/downloads.py` 的 `_try_download_report_rc` 全量回溯
    路径使用了未导入的 `_query_with_pagination`（NameError 被 except 吞掉，
    全量回溯从未成功），现正确导入。
- **测试**：新增 `test_forecast_report_rc_partition.py`（分区键映射、分区路由
  去重、增量补齐写分区、loader 纯分区加载）；同步适配
  `test_download_report_rc_adaptive.py` / `test_download_raw_fixes.py` /
  `test_download_periodic_concurrency.py` / `test_ensure_and_t0_printing.py`。

## [0.90.28] - 2026-08-03

### Changed

- **屏蔽 `_query_report_rc_adaptive` 二分合并的 concat FutureWarning**：
  `report_rc` 大年份自动二分后，`pd.concat(parts)` 合并各子段仍会因页内全 NaN
  列（如 `max_price`）触发 pandas 告警；现统一改用 `_concat_no_warning`
  （`scripts/raw_download/alt.py`），数据仍原样保留。
- **预期流程日志降级为 debug（减少黄色噪音）**：仅真正需要关注的错误（如
  `Read timed out`）保留 warning 级别：
  - `client.query` 的"确定性错误，不重试"日志 warning → debug（大年份超限走
    二分属预期分流）；
  - `_query_report_rc_adaptive` 的"整段查询失败，自动二分重试"日志
    warning → debug（二分是预期正常流程）。
- **测试**：`test_download_report_rc_adaptive.py` 新增二分合并屏蔽告警用例。

## [0.90.27] - 2026-08-03

### Fixed

- **修复 `report_rc` 长期高并发被 TuShare 拒绝（"查询数据失败"）**：
  - 现象：无代理时，16 并发长期下载大年份（需二分），TuShare 对请求返回
    "查询数据失败，请确认参数！" 拒绝，二分后的子段也被拒。
  - 实测定位（用户实际环境）：并发 8×80 请求、16×320 请求均全部成功——
    并发数量本身不是问题；根因是**长期高请求频率** + **超限错误重试 3 次
    放大请求量**（"查询数据失败"是确定性错误，重试必再失败，白白放大 3 倍）。
  - 修复（`src/lazybull/data/tushare_client/core.py`、
    `scripts/raw_download/alt.py`）：
    1. `client.query` 对"查询数据失败，请确认参数"类**确定性错误不重试**
       （直接抛，节省 2/3 请求量）；
    2. `_API_RATE_LIMITS_DEFAULT["report_rc"] = 200`（接口级限频，长期运行
       不再超出 TuShare 承受频率）；
    3. `_REPORT_RC_CONCURRENCY` 16 → 8（保守并发）。
- **测试**：`test_tushare_client_rate_limit.py` 新增确定性错误不重试、
  `report_rc` 接口限频生效两个用例。

## [0.90.26] - 2026-08-03

### Changed

- **不再为消除 concat FutureWarning 而处理数据（撤回 0.90.23/0.90.25 的剔除/reindex
  逻辑）**：下载层的 `_query_with_pagination` 与 `_save_merged`
  （`scripts/raw_download/periodic.py`）恢复**数据原样 concat**——不剔除任何
  全 NA 行/全 NaN 列、不做 reindex 补列。此前为消除 pandas 的 empty/all-NA
  entries FutureWarning 而剔除全 NaN 列（如 `report_rc` 的 `max_price`）会破坏
  raw 层 schema，导致训练用到某列、预测时该列被删而失败。
  - 新增 `_concat_no_warning` 辅助：`concat` 时仅按消息精确屏蔽该 FutureWarning
    （`filterwarnings` 的 `message` 用 `re.match` 从头匹配，需含
    "The behavior of " 前缀），**数据完全原样**。
- **测试**：`TestQueryWithPaginationAllNA` / `TestSaveMergedAllNA` 更新为
  "数据原样保留（含全 NA 行/全 NaN 列）+ 不泄露 empty/all-NA FutureWarning" 语义。

## [0.90.25] - 2026-08-03

### Fixed

- **彻底修复 `_query_with_pagination` 的 concat FutureWarning（真正根源为页内
  "全 NaN 列"）**：
  - 在用户实际环境（pandas 2.3.0）真实复现确认：触发警告的不是"全 NA 行/页"，
    而是**某些页存在整列全 NaN 的列**——如 `report_rc` 的 `max_price`（预测
    最高价）字段在部分页完全缺失（2026 年 42 页中有 7 页 `max_price` 全 NaN）；
  - 0.90.23 的修复（剔除全 NA 行）方向错误、未生效，本次改为逐页
    `dropna(axis=1, how="all")` 剔除全 NaN 列后再 concat，最后 `reindex` 补回
    全部列集合（缺失列填 NaN），schema 不缺失且告警消除；
  - 全 NA 页（剔除后 0 列）一并过滤。
  - 修复位于 `scripts/raw_download/periodic.py`。
- **测试**：`test_download_raw_fixes.py` 新增 `test_all_na_column_skipped_and_reindexed`
  （全 NaN 列剔除 + reindex 补回）。

## [0.90.24] - 2026-08-03

### Fixed

- **修复 `report_rc` 并发过高导致代理超时/全局失败**：`report_rc` 单请求服务端
  响应约 5s，按年并发使用全局限频 48 会打爆本地 HTTP 代理（如
  `192.168.1.21:18081`），大量 `Read timed out (read timeout=30)`，进而连锁
  触发 TuShare "查询数据失败，请确认参数！" 全局拒绝；此时自适应二分的子段
  （如 5 万条/半年）也失败，说明是全局性问题而非超限，原二分逻辑仍会无谓递归。
  - `scripts/raw_download/core.py`：`_run_concurrent` 新增 `max_workers` 参数，
    可按接口覆盖全局 `_DOWNLOAD_CONCURRENCY`（默认不变，向后兼容）；
  - `scripts/raw_download/alt.py`：`download_report_rc` 改用保守并发
    `_REPORT_RC_CONCURRENCY = 8`，避免打爆代理；
  - `_query_report_rc_adaptive` 只对"查询数据失败，请确认参数！"类**超限**错误
    二分；网络超时/其它错误直接上抛（由 `download_report_rc` 记录该年份失败，
    重跑断点续传），不再对全局性问题做无意义递归。
- **测试**：新增 `_run_concurrent` 的 `max_workers` 覆盖/串行用例、
  `_query_report_rc_adaptive` 非超限错误不二分用例、`download_report_rc`
  保守并发断言。

## [0.90.23] - 2026-08-03

### Fixed

- **修复 `_query_with_pagination` 的 concat FutureWarning**：翻页过程中某些页
  "有行但所有值均为 NaN"（全 NA 片段）时，`pd.concat` 触发 pandas
  "empty/all-NA entries" 行为变更告警（与 0.90.17 修复的 `_save_merged` 同类）。
  现于 concat 前剔除全 NA 片段并保留全部列集合，消除告警且 schema 不缺失。
  修复位于 `scripts/raw_download/periodic.py`，影响所有走分页下载的数据集
  （如 `report_rc`/`forecast`/`express` 等）。
- **测试**：`test_download_raw_fixes.py` 新增 `TestQueryWithPaginationAllNA`
  （全 NA 页剔除 + 列集合保留 + 全 NA 页仅存时结果为空）。
- **测试适配**：`test_tushare_client_rate_limit.py` 中硬编码 `cyq_perf=100`
  的断言改为动态读取 `_API_RATE_LIMITS_DEFAULT`，与用户对接口级限频的调整解耦。

## [0.90.22] - 2026-08-03

### Fixed

- **修复 `top_list` 限频 override 绕过配置导致大量限流**（v0.90.21 引入的回归）：
  - v0.90.21 将 `get_top_list` 的 `rate_limit_override` 从 60 改为 1000，但 TuShare
    `top_list` 官方限频实测为 **500 次/分钟**；`rate_limit_override` 会绕过
    `_API_RATE_LIMITS_DEFAULT` 与全局令牌桶，1000 在 48 并发下远超官方限频，
    触发海量"频率超限(500次/分钟)"限流错误。
  - 同时导致用户对 `_API_RATE_LIMITS_DEFAULT["top_list"]` 的配置（如 400）不生效。
  - 修复：`get_top_list` **移除 `rate_limit_override` 硬编码**，改走接口级限频
    `_API_RATE_LIMITS_DEFAULT["top_list"]`（默认 400，低于官方 500 留余量），
    用户可直接在该表配置；仍触发限流时 `client.query` 会自动解析并自适应降频。
- **测试**：`test_tushare_client_rate_limit.py` 更新为断言 `get_top_list` 不传
  override、接口级间隔取配置限频且不高于官方 500。

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
