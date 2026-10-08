# v2 P1 数据底座·构建冻结物（开工前冻结）

> 版本：v3（2026-10-06）　状态：**已生效**（P1 开工闸门，随计划批准生效）
> **版本行**：v1（2026-10-02，commit `3d2bd78`）首次生效；v2（2026-10-03）**重冻结修订**——
> ① §1 `label_filter_mode` 勘误 all→**single**（单元 2 开关标定实证：cs_train 全历史分区含
> y_ret_5/y_ret_10 NaN 行而 y_ret_20 恒非空 ⇒ 生产构建走 `--horizon 20` 单值分支，仅按主
> horizon 过滤；若按 all 回填每日多滤 2~9 行、占比 4e-4~1.8e-3 > 验收门 1e-4。基线产物不变，
> 属冻结文字勘误，差异台账回溯标注见 §8）；② §3 labels 列名适配登记（y_ret_N/neu_y_ret_N →
> store 口径 label_value/neu_label_value）；③ §1 补 build_daily 捕获区间口径（[D−7月, D]
> 捕获、只落 D，对齐 cs_infer 通路 `FEATURE_DATA_HISTORY_MONTHS=7`；[D,D] 单日捕获会把
> 北向 rolling 列在长度 1 交易日序列上算错，标定实证 4684/4684 行超容差）。
> v3（2026-10-06）**R3-03 裁决登记**（§9.2 待裁决→已裁决 + §1 三处补注 + §8 D-15）——
> 行域由构建通路（写侧）决定：backfill 冻结域 / build_daily 结构域，消费侧统一读时过滤。
> **登记时点构建行为不变**（build_daily 仍 fail-fast；require_label=False 日更通路随
> P2a/M1.5 前置工作项实施落地）。
> 依据：`docs/contracts/v2_architecture_plan.md` §8-P1 行「开工前冻结构建窗口口径」+ §4.1 数据存储契约 + P1 实施计划（2026-10-02 批准）。
> 冻结纪律：本文 commit（git `d729f82` 之后首个 P1 提交）后口径即冻结；此后任何改变构建行为的代码改动触发**重冻结程序**（对应段基线重冻结 + 差异台账回溯标注，沿方案 §9.1 基线稳定条款）。

## 1. 构建窗口口径（回填构建器必须逐位遵守）

| 项 | 冻结值 | 依据 |
|---|---|---|
| 回填区间 | **20120104 ~ 20260702**（= cs_train 现产物覆盖区间，3,518 交易日） | 现产物实测 |
| 预热窗口 | 加载窗口 = 起点**前 7 个自然月** + 终点**后 1 个自然月**；fund_portfolio 单独回溯 18 个月 | `features/pipeline.py:383-385,490-495` |
| 标签口径 | T 日信号，**T+1 收盘（close_adj）买入，T+1+N 开盘（open_adj）卖出**；`y_ret_N = open_adj(T+1+N)/close_adj(T+1) − 1` | `features/labels.py:6-8,98-101` |
| horizons | **{5, 10, 20} 全量生成**（builder `horizons=[5,10,20]`），主 horizon=20，**`label_filter_mode="single"`**（仅按 y_ret_20 非空过滤；生产构建走 `--horizon 20` 单值分支——v2 重冻结勘误：v1 误记 "all"，实证 cs_train 全历史分区含 y_ret_5/10 NaN 行而 y_ret_20 恒非空） | `scripts/build_clean_features.py:254-262`；单元 2 标定 `temp/p1_flag_calib_20261003.json` |
| require_label | True（cs_train 语义；无标签拒绝落盘沿 `storage.py:592-597`）——**限回填/冻结段**；build_daily 日更通路裁决为 False（R3-03，v3 补注，见 §9.2） | 现产物契约 |
| 样本过滤 | `is_st==0 & list_days>=365 & is_suspended==0`（min_list_days=365） | `builder/static_extra.py:450` |
| 中性化 | **行业（申万）+ 规模双开**（现产物同时含 `zscore_*` 57 列与 `zscore_*_sz` 56 列） | 现产物实测 |
| 可选因子组 | 以现产物实际列集为准（哨兵 `cashflow_quality_schema_v2` / `cons_revision_schema_v2` / `dividend_schema_v1` 在列；holdertrade/repurchase/top10fh/top_inst 四族**不**在 cs_train 列集，见 §6 物化拍板） | 现产物实测 |

**截面口径（核心，0.201.1 探测教训登记）**：构建编排顺序冻结为——合并标签 → 过滤标记 → **行过滤（步骤 11）→ 中性化（步骤 12）→ 个股特征 → 市场状态**（`features/builder/orchestration.py:204-229`）。即 **`zscore_*` / `neu_*` / `mkt_*` 全部在「过滤后截面」上计算**（过滤后截面 = 上述样本过滤 + y_ret_20 非空后的同日行集）。回填构建器必须保持同一顺序与同一截面；单日增量构建（影子通路）必须使用与批量回填**完全一致**的过滤口径（全市场统一截面），禁止以当日局部截面重排。
**（v3 补注·R3-03 裁决）**上句「完全一致的过滤口径」语义修订为：**样本过滤三条件与全市场统一截面口径不变**（禁止当日局部截面重排的约束继续有效），但日更通路（build_daily，价格水位 = 当日）**不叠加标签过滤**（require_label=False，当日标签物理不可算）⇒ 行域 = 结构域；历史影子通路（对账复刻，目标日为历史日、端点数据已在）仍走与回填完全一致口径。详见 §9.2。

**单日增量构建捕获区间（v2 补登）**：`build_daily(D)` 的构建驱动区间为 **[D−7 个自然月, D]**（捕获后 sink 只落 D 日），对齐纸面 cs_infer 通路 `FEATURE_DATA_HISTORY_MONTHS=7`（`features/ensure/constants.py:7`）——[D, D] 单日捕获会把北向 rolling 列（`north_*_ma5/ma20/z20/sum5/sign_streak` 等）在长度 1 的交易日序列上算错（单元 2 标定实证 4684/4684 行超容差）。

**panel 行口径拍板**：v2 panel 与 cs_train **逐行一致**（同一过滤后截面）——等价复刻优先；全截面分母需求（pct_* 母截面、M1.5 覆盖率剖面）由 normalized + 母截面构建器承担，不从 panel 取（沿 `risk/terminal_loss/mother_section.py` 契约不变）。
**（v3 补注·R3-03）**「逐行一致」限**冻结段**（回填区间 20120104~20260702）；水位线后日更段行域 = 结构域（§9.2 裁决），单一 store 在水位线两侧行集口径不同（分段内容属性）；训练消费行集 = 全段统一读时按标签非空过滤（`ml/train_core/prepare.py` 既有 dropna 升格为正式口径），冻结段写时已过滤 ⇒ no-op，读时过滤 ≡ v1 写时过滤行集。

## 2. 列族分组表（383 列 → panel 8 族 + labels 3 表；逐列映射冻结）

校验：383 列全覆盖、无重复、无遗漏、无越界（产物归档 `tests/fixtures/p1_column_groups_20261002.json`，生成脚本为一次性 temp 物已按 §7.7 清理；单元 2 起代码化为 `v2/store/column_groups.py` 单一来源）。

| 族 | 列数 | 内容 |
|---|---|---|
| `core` | 52 | 键（trade_date/ts_code）+ clean 复用列（vol/amount/is_st/is_suspended/is_limit_*/list_days/tradable）+ 基础量价（ret_*/vol_ratio_*/ma_deviation_*/amount_ma*/opening_strength/intraday_vol_structure）+ daily_basic 直通与价值红利派生（pb/pe_ttm/ps_ttm/dv_ttm/total_mv/circ_mv/turnover_rate/volume_ratio/ep_ttm/bp/log_*/is_loss/*_missing）+ 申万行业参考（sw_*/in_date） |
| `fundamental` | 69 | fina_indicator 19 + forecast 3 + express 5 + 一致预期/修订 23 + 现金流 9 + 分红 11 |
| `moneyflow` | 44 | 资金流向 14 + 融资融券 6 + 北向 13 + 龙虎榜 11 |
| `technical` | 43 | 筹码 5 + 基金持仓 5 + 高级量价 30 + 个股特征 3 |
| `announcement` | 10（+物化） | 股东户数 3 + 质押 3 + 解禁 2 + 大宗 2；**运行时派生四族 ht_*/rp_*/tfh_*/ti_* 与 has_* 标记物化后入此族（§6）** |
| `risk` | 29 | 步骤 9.5 `_add_risk_factors` 产出块（@register_risk_factor 注册族的 panel 侧部分；注册族另 4 列 short_balance_change_5/unlock_ratio/block_discount_*/short_sell_vol_change_5 物理上随 moneyflow/announcement 族，注册表为逻辑视图不作存储分组依据） |
| `market_state` | 13 | mkt_* 市场状态广播列 |
| `neutralized` | 117 | zscore_* 57 + zscore_*_sz 56 + zscore_size + neu_ret_1/5/10/20 |
| **labels**（独立成表） | 6 | y_ret_5/10/20 + neu_y_ret_5/10/20（训练默认标签 = `neu_y_ret_20`，`ml/train_core/prepare.py:63`） |

逐列清单见附录 A（与 `tests/fixtures/p1_column_groups_20261002.json` 逐项一致）。

## 3. 冷热分层参数（P1 一次性定型）

- **热区**：最近 **12 个自然月**按日分区 `data/features/panel/YYYYMMDD/<group>.parquet`（N=12 依据：覆盖 7 个月预热窗口 + 影子对账窗口 + 单日增量构建期）。
- **冷区**：封存月压实 `data/features/panel_archive/YYYY-MM/<group>.parquet`，范围 2012-01 ~ 2025-06（162 月 × 8 族 ≈ 1,300 文件，贴合方案 ~1,000 量级估计）。
- **单文件规则**：软上限 256MB（超限按列子族拆分）、软下限 32MB（欠限触发压实评审）；回填时逐文件校验并登记。
- **labels**：`data/labels/y_ret_{5,10,20}/YYYYMMDD.parquet`（日分区，不分冷热；单表列 = `ts_code, trade_date, label_value, neu_label_value, maturity_status`——**v2 适配登记**：v1 文字 `y_ret_N, neu_y_ret_N` 按 store 统一口径命名为 `label_value`（= y_ret_N 值）/ `neu_label_value`（= neu_y_ret_N 值），`v2/store/labels_builder.py` docstring 同源登记）。
- **labels 成熟度生命周期**：封存时点 = T + max(h)=T+20 端点可算日（T+1 起算第 20 个交易日存在）；封存前幂等可重写（`maturity_status=forming`），封存后拒绝改写（`maturity_status=sealed`）；回填区间内全部标签已成熟（区间末端 20260702 的 T+20 ≈ 20260730 ≤ raw 末端 2026-07-31）。
  **措辞补注（2026-10-06 评审 P2）**：「T+20 端点可算日」的代码语义 = 交易日历中 `idx(T)+21` 存在且端点 ≤ 数据水位（T+1 起第 20 个交易日），与 §1 标签公式端点 T+1+N 严格一致——**代码正确，勿按字面「修」成 off-by-one**；本轮整改已把数据水位纳入判定（评审 R3-06）。

## 4. manifest 规格要点（单元 1 实现的输入）

- 列级：`列名 → {group, source, definition_version, available_from, backfilled_at, status}`；`available_from` 初值 = 列族实际可用起点（单元 5 剖面报告回填精化；回填的新因子只允许在其可用起点后参与训练）。
- per-分区**内容指纹**（sha256-16，封存/压实/备份/恢复复用同一指纹）；**依赖声明**（raw 清单 + factors 派生函数标识）；**两阶段提交**（临时名写入 → 原子纳入 manifest → 孤儿 GC + 单机写锁）。
- 拼接键固定 `(trade_date, ts_code)`；core 族为左表 left join；族缺日整族 NaN + 计数登记（§4.1 补丁 1）。

## 5. raw 快照（水位 + 指纹 = 快照；raw 只追加不可变，不物理复制）

- **数据态锚点**：与 B0/B1 基线冻结同数据态 **`9d0408ee`**——8 个数据态数据集（daily/adj_factor/daily_basic/moneyflow/stk_limit/suspend/stock_st/margin_detail）水位全部 = 2026-07-31 ✓（2026-10-02 复核一致），cs_train 最新 = 20260702 ✓。
- **全量水位**（34 数据集 + 4 单文件，2026-10-02 采集）：日频类 2026-07-31（top_list 2026-08-07 / block_trade 2026-08-05）；季频类 2026-06-30；年分区类 2026-12-31（dividend/share_float/report_rc/repurchase/stk_holdertrade/top10_floatholders/top_inst）；pledge_stat 2025-12-31；cyq_perf/moneyflow_hsgt 2026-07-31；单文件 trade_cal/stock_basic/shenwan_industry/stk_holdernumber/hot_rank。
- **抽验指纹**（sha256-16，附录 B 全表）：daily `2783617a92ab2ce9`、adj_factor `43c16bd328e4e29f`、cs_train(20260702) `54678be3decbd5d5`、clean/daily(2026-07-31) `d6a6f7857393896f` 等 10 项。
- **代码态**：git `d729f82`（v0.203.5，F9），工作树干净。
- 回填期间纸面软生产继续运行（raw 只追加，水位 ≤2026-07-31 的冻结段不受影响）。

## 6. 运行时派生四族处置（拍板）

按方案 §4.3「运行时派生退役：历史区间一律物化到列族」——holdertrade / repurchase / top10_floatholders / top_inst 四族及 availability 标记（has_*）在 v2 panel 中**物化进 `announcement` 族**，经 `build_*_feature_frame` 单一实现接线（与现四侧运行时派生同一数值路径）。manifest 登记 `status=deprecated`（已终结家族，默认链路不消费；物化是数据资产合规，不代表重开实验）。对账口径 = **列交集**（现 cs_train 无此四族列，panel 多出部分为预期差异，先登记后比对）。

## 7. 验收闸门参数（单元 4 执行）

- 对账门：母截面三段式（块级逐值 atol=1e-6 → 合并 → 判定）；实现漂移判据 = 超容差行占比 > 1e-4 或任一列 max|Δ| > 0.05 ⇒ raise；极稀疏差异 ⇒ 数据态漂移告警 + 离群登记（沿 `MOTHER_OUTLIER_SHARE_LIMIT` / `MOTHER_HARD_ATOL`）。
- **注（2026-10-06 评审整改，R2-4②）**：上句「超门 ⇒ raise」的字面口径已被 D-13 机械化出口取代——终态出口为「**超门 + 落入登记归因类 ⇒ 放行**」（登记类 = L1 / D-04 / D-11 / D-12 豁免 / 纯数据态孤立单点，类外 ⇒ FAIL）。此处补注防后人照字面回改实现。
- 出口 = 残差 100% 归因（差异逐列归因：截面口径 / 派生物化 / 数据态）。
- 性能闸门：全历史跨列族加载 ≤ 现状 1.5 倍（双口径：抽 5 列 + 全列）。
- 新增列只写新文件（历史分区指纹零改动校验）。

## 8. 差异台账（重冻结/口径修订回溯标注，append-only）

| 日期 | 条目 | 内容 | 证据 |
|---|---|---|---|
| 2026-10-03 | D-01 | **§1 `label_filter_mode` 勘误 all→single**：v1 依「三列标签都在」推断多值模式；单元 2 标定实证 cs_train 全历史分区含 y_ret_5/10 NaN 行而 y_ret_20 恒非空 ⇒ 生产走 `--horizon 20` 单值分支。影响：若按 all 回填每日多滤 2~9 行（4e-4~1.8e-3 > 门限 1e-4）。处置：冻结文字修订，基线产物（cs_train）不变、无需重跑；`V2PanelBuilder` 按 single 实现。 | `temp/p1_flag_calib_20261003.json` |
| 2026-10-03 | D-02 | **build_daily 捕获区间补登 [D−7月, D]**：[D,D] 单日捕获使北向 rolling 列在长度 1 交易日序列上算错（4684/4684 行超容差）；对齐 cs_infer 通路 7 个月历史窗口，sink 只落 D。 | 同上 |
| 2026-10-03 | D-03 | **§3 labels 列名适配**：y_ret_N/neu_y_ret_N → label_value/neu_label_value（store 统一口径）。 | `v2/store/labels_builder.py` |
| 2026-10-03 | D-04 | **已知数据态漂移预告（单元 4 对账将现身，先登记）**：① share_float/block_trade/top_list raw 水位（2026-08-05/07）晚于基线数据态 8 数据集（2026-07-31）⇒ days_to_unlock/unlock_ratio/unlock_risk_flag 等列对账差异属预期数据态漂移（§5 水位已如实登记，不重冻结）；② raw daily 2024-01-16/17 分区丢 2 只停牌股行 ⇒ downside_corr_20 等权市场序列微移（当日 max 0.011）；③ 季频财务/基金持仓 raw 修订 ⇒ dividend_payout_ratio/cf_nm/fund_* 等 NaN 型差异。均走 §7 数据态漂移告警 + 离群登记通道，残差 100% 归因。 | 单元 2 标定报告 |
| 2026-10-03 | D-05 | **pandas 3.0 rolling.skew/kurt 回归与修复（用户裁决：修复后回填）**：pandas 3.0.x 的 `rolling.skew`（及 `rolling.kurt` 部分模式）在序列曾出现 ≥ 窗口长度的全 NaN 段后，后续所有窗口卡死为 NaN（pandas 2.3.3 无此问题，venv 双版本实证 3.0.0/3.0.4/3.0.5 均中招）；本机 pandas 于 2026-09-01 21:55 升级 3.0.5 ⇒ cs_train（2026-08-31 构建）值正确、此后构建（cs_infer 0923 起/重放）中毒（长停牌股 skewness_20/kurtosis_20 系统性 NaN，cs_infer 20260702 实测 5 股 vs cs_train 1 股）。**现役模型 154 列不含 22 个批量风控列 ⇒ 生产信号无直接影响**（已核实 v24288_features.json）。处置：① `risk/precompute.py` 的 A7/A8 改为中心矩公式 NaN 安全实现（与 pandas 2.x 逐位一致 max|Δ|≈1e-15，验证 `temp/p1_verify_skew_formula.py`；真实数据全截面验证 20260401 全部 5,069 股 skewness_20 ≡ cs_train max|Δ|=0、kurtosis_20 max|Δ|=6e-8——中毒纯属环境漂移，无数据态残留）；② 修复改变旧链 cs_infer 日构建行为（恢复与冻结基线一致，仅限非模型消费列）；③ 冻结参照全量 3,518 分区两列补丁重建（`temp/p1_patch_reference_skew_kurt.py`；首轮 ≤20211229，后发现 chunk 3 经模块破损窗口期构建遂扩至全量）；④ 回归测试 `tests/test_risk_precompute.py::TestRollingSkewKurtNanSafety`。 | 单元 3 探测证据链；venv 双版本对照；全截面验证 |
| 2026-10-03 | D-06 | **cs_train 2012 构建血缘（生产首建冷启动零预热）**：cs_train 2012 年代分区为 330~336 列 schema 且无 7 个月预热（首 ~20 交易日 ret_*/资金流 sum/mkt_* 等 1,600+ 股 NaN；至 2012-02 为 19-valid 口径；macd_* EMA 长尾延续 2012 年内）。冻结参照/v2 panel 按 §1 冻结预热口径构建，与该年代分区差异属预期（归因门登记，报告 §4a-L1；如需严格逐行一致，区间起点 ~20 天须专项口径评审）。 | 单元 3 报告 §4/§4a |
| 2026-10-03 | D-07 | ~~cs_train 2013~2015 公式版本血缘~~ **（2026-10-05 更正，见 D-09）**：单元 3 曾把 2013~2015 风控族差异（downside_vol 比值 2.87~2.96 等）归因为 cs_train 公式版本血缘；单元 4 严格门复核证实**真正的异常方是冻结参照的分块 1 逐日回退污染**（D-09），cs_train 与 panel 在该年代逐值一致（20130104/20150105 全截面抽样 diff=0）。moneyflow 2011-12 值域修订与 north 2015-01 覆盖/修订两子项维持有效（单元 3 §4a 实证）。**2018 起参照/panel/cs_train 三方逐值一致**。 | 单元 4 复核（D-09 证据链） |
| 2026-10-05 | D-09 | **冻结参照分块 1 逐日回退污染事件（已修复）**：首轮全量重放的分块 1（2012~2016）分区风控 22 列为 float64 逐日回退路径产物（与批量路径值系统性不同，downside_vol 比值 ~0.33~0.38）——疑因原始重放期间并发文件编辑致批量缓存 import 破损窗口（回填期冻结依赖链纪律的执行细则来源）；经 dtype 探出（批量路径=float32）→ 删除分块 1 分区重筑（当前修复后代码，批量路径复归，20120104 全截面与 panel max\|Δ\|=0）。**附查发现重叠区覆盖纪律**：`_check_features_schema` 对 <383 列分区一律判 False ⇒ 后续分块把 2015~2017 重叠区以自身（较短）预热链重建覆盖，污染 warmup 敏感列（mkt_ma250_ratio/macd 等）——回填侧 keep_dates 纪律（已写日不重建不覆盖）天然免疫，参照侧以「--end-date 20161231 单分块重筑」修复。**教训：v2 侧一律以 manifest 存在性（而非 schema 校验）做已写判定。** | 单元 4 闸门证据链 |
| 2026-10-05 | D-10 | **单元 4 分块边界 5 日缺口与修复性合并**：分块末标签不成熟（T+21 端点超出分块加载窗）致 20161228/29/30、20211230/31 未落盘，且已写判定月粒度盲区掩盖（月文件已登记 ≠ 日内覆盖完整）。处置：① 新增 `PanelDataStore.repair_archive_partition`（行超集追加 + 键不重校验 + 理由强制登记进 manifest.repairs；区别于「口径修正=新列名/版本升级」，本通道服务构建缺陷修复）；② 按参照同预热窗重建 5 日并合并入月归档（labels 直写日分区）；③ `_written_days` 改冷区日级精度（核 core 族分区实际日期集）防复发。 | 单元 4；测试 `tests/test_v2_store_repair_boundary.py` |
| 2026-10-05 | D-11 | **市场状态/技术缓存锚定机制（构建器口径探明）**：`precompute_market_state_cache` 等批量缓存以**首个 pending 日为锚点**、向**前**取 120 交易日切片计算（`_slice_by_trading_days` 的 `trading_dates[anchor_idx-120:]`）——分块/局部重建时锚点随 pending 集漂移 ⇒ 同一日值随构建上下文变化（mkt_ma250_ratio/macd 等长记忆列）。**结论：须与 cs_train 逐值一致的参照/回填，锚定必须与 cs_train 生产构建相同——cs_train 为单调用、管线起点 = 20120104（预热由管线内部 -7 个月完成），故参照终态 = 单调用 `--chunk-years 0`（不带 --warmup-start，锚定 20120104）**；v2 panel 回填经「force=True 全量构建使锚点恒落分块起点」+ keep_dates sink 纪律，锚定效应与 cs_train 逐日兼容（panel ≡ cs_train 已实证）。注：首版曾误判 warmup-start 20100101 为正确锚定，实证其致 2012 段与 panel 口径不一，以本行为准。 | 单元 4 严格门复核证据链（20211230 锚定差异实证 + 单调用双口径对照） |
| 2026-10-05 | D-12 | **公告前瞻查询的区间终点截断（PIT 完整性缺陷，口径裁决待办）**：`load_share_float`/`load_block_trade` 按构建区间 [start, end+1mo] 加载（share_float 按 float_date 年分区）——分块回填/单日 cs_infer 的加载窗截断 ⇒ 解禁/大宗的「已公告未解禁」（ann_date≤T<float_date）远年记录在窗尾丢失 ⇒ unlock_ratio/days_to_unlock 等前瞻列在分块早期年代与 cs_train（全区间覆盖）值不同（实证：panel 分块 1 的 2014-01 段 vs 全区间参照 max\|Δ\|=89.8）。**质押族已按「全量历史起点加载」处理（pipeline.py:736 先例），share_float/block_trade 未跟进**。影响面：现役模型 154 列不含该族 ⇒ 无生产信号影响；但袖子 B/D（事件/公司行为）将消费公告数据。**处置：差异台账登记 + 口径裁决待办**（候选：公告前瞻查询改为全量历史起点+全量前向覆盖加载，与 pledge 先例对齐；该改动改变旧链行为，按行为冻结纪律须随对应段切换一并评审，或作 P1 后独立修复项），建议随袖子 B 立项的数据工作一并裁决。 | 单元 4 严格门/归因门复核证据链 |
| 2026-10-06 | D-13 | **D-12 口径裁决落地（用户裁决：panel 保留全量 PIT 修复 + 严格门豁免登记）**：① 单元 4 已对 panel 执行 D-12 修复（全量 share_float 2005~2026 分区 + 全日历重建解禁查询表，`rewrite_partition` 重写 174 分区 / 233,194 单元格；首轮在 manifest 落盘阶段撞 Windows 瞬时锁中断，第二轮完成——manifest.save 已加瞬时锁重试，事后只读扫描 3,248 登记分区指纹零失同步）。② 修复后严格门该族差异不降反升（unlock_ratio 超门行 138,717→428,106 等）——实证根因：**panel 语义 = PIT 完整，参照/cs_train = 旧链加载窗截断**（[起点−7月, 终点+1月] ⇒ 丢 ann_date ≤2010 分区与末年分区）；探针 000011.SZ@20120104 panel=201 天/0.1667 与全量手工核算一致（2009-10-28 公告批），参照 NaN。③ 裁决：**panel 保留修复值**（新底座直接消费正确语义，袖子 B 数据工作输入）；参照与 cs_train 生产链保持旧行为（行为冻结不动）；旧链 pipeline 加载口径修复转为 P1 后独立项随袖子 B 评审；现役模型 154 列不含该族 ⇒ 零生产影响。④ 豁免登记：严格门/归因门对 days_to_unlock/unlock_ratio/unlock_risk_flag 从「逐值一致」改「豁免 + 差异全归因登记」（`panel_reconcile.py::D12_EXEMPT_COLUMNS`）；**严格门判定同步机械化为 §7 出口「残差 100% 归因」**（登记类 = L1 / D-04 / D-11 / D-12 豁免 / 纯数据态孤立单点，类外 ⇒ FAIL）——同步细化：D-11 锚定影响窗定界 20170103~20170831 / 20220104~20220930（120 交易日切片 + 复牌股 EMA 收敛尾，实证 002260.SZ 2022-05 复牌簇几何收敛 ~0.8/日；窗内非长记忆列稀疏伴随单点同归 D-11），纯孤立单点阈 = 日数 ≤10 且总量 ≤20 行且单日 ≤1% 且 max\|Δ\|≤0.05（百分位秩单步翻转类，超幅/密集 ⇒ 调查）。⑤ 量化证据：`data/reports/v2_p1_d12_exemption_evidence_20261006.json`（panel ≡ 全量查询表 且 参照 ≡ 窗口查询表 双恒等式全量核对 ⇒ 残差 = 加载窗内容差，无第四来源）。⑥ **勘误 D-12 原文**：「share_float 按 float_date 年分区」有误——实际按 **ann_date** 年分区（`loader_announcement.py:8,60-67` 模块契约与 raw 分区内容实证），本条目以此为准。 | 严格门终版 `v2_p1_panel_reconcile_final_20261006.json`；豁免证据 JSON；探针 000011.SZ/000156.SZ |
| 2026-10-06 | D-14 | **归因门 5 列越界的复核定界（单元 3 遗留「待调查」闭环；panel ≡ 参照 ⇒ 差异源在 cs_train 侧）**：归因门（panel vs cs_train）复核中 downside_corr_20 / kurtosis_20 / skewness_20 / zscore_macd_hist / zscore_macd_hist_sz 五列越出原五类清单，经全量差异日诊断（`data/reports/v2_p1_attrib_violators_diag_20261006.json`，11,032,382 股票-日）+ 严格门交叉验证（panel 与冻结参照在该五列零差异或仅 D-11 簇）+ git/pandas 环境比对，定界为三个已登记机理的组合，均非实现漂移：① **skewness_20/kurtosis_20 = L1 主体 + D-05 实现对偶数值尾**——cs_train 由 pandas 2.x `rolling.skew/kurt` 计算（2026-08-31 构建），panel/参照为中心矩 NaN 安全实现（D-05 修复），数值不稳定股（近恒定收益/停牌簇）翻转：晚日逐日 ≤25 股且 ≤1.5%（实证 skewness ≤3 股/0.18%；kurtosis ≤23 股/1.32% 聚于 2015-04~08 极端行情簇；max\|Δ\|=35.1 @20150326 300288.SZ 同簇）；② **zscore_macd_hist(_sz) = L1 macd-EMA 冷启动长尾 + D-11 + 孤立单点**——cs_train 2012 冷启动 EMA 链收敛实测拖尾至 2013-09-04（1 行拖尾至 2013-09-18；单元 3 报告将该列_sz 列「待调查」，本条目闭环）+ 2022-05 复牌簇（002260.SZ，同严格门 D-11）+ 稀疏孤立单点（≤5 行且 ≤1%）；③ **downside_corr_20 = L1 + 2013-09/10 cs_train 构建环境血缘块 + D-04② + 孤立单点**——20130917~20131023 连续 17 日近全截面差异（~98% 行）：cs_train 由 pandas 2.x 构建（2026-08-31）、panel/参照由 pandas 3.0.5 构建，输入数据与代码逐字节相同（clean/daily 2013-09 分区 08-31 后未变、`risk/precompute.py` A2 实现 08-31 至今未变，git 实证），推断为 pandas 版本滚动聚合行为差被该年代数据触发的局部板块（20 日窗扫过触发日 ⇒ 连续 17 日）；另含 D-04②（2024-01 窗）与 2021-12 两点稀疏单点（1 行/日）。**共同前提：风控 22 列与 zscore 派生均不在现役模型 154 列 ⇒ 零生产影响（D-05 已登记）**。处置：`panel_reconcile.py::_classify_attrib_special` 按上述定界归因（越界仍 FAIL）。 | 诊断 JSON；严格门终版；`git show 77adea3^:src/lazybull/risk/precompute.py` 对照 |
| 2026-10-06 | D-13 增补①（三评审：第二份 §3.1 / R2-5②） | **D-12 修复计数终态口径**：共 **651 个分区有变化** = 首轮 476 条 `[rewrite_partition]` + 中断分区 1 个（`panel_archive/2016-10/announcement`——rewrite 登记随 manifest 落盘失败（WinError 5）丢失，由 02:46 resync 记录承接）+ 第二轮 174 条 `[rewrite_partition]`；manifest.repairs 终态 **707 条** = 650 `[rewrite_partition]` + 40 边界回补 + 16 `[replace_days]` + 1 `[resync]`；两轮修正单元格合计 758,672 + 233,194 = **991,866**（中断分区未计入）；守恒 812 = 651 + 161（无变化分区 161）。D-13 ①原文「重写 174 分区 / 233,194 单元格」为第二轮续跑段局部计数，以本增补为准。 | 第二份评审 §3.1（两轮修复运行日志 + manifest + resync 三方闭合）；总验收报告 §7.1 |
| 2026-10-06 | D-13 增补②（三评审：R2-5①） | **repairs reason 编号映射说明**：40 条边界回补记录的 reason 文本写「单元 4 分块边界缺口回补（**D-09**）」——该事件在差异台账中的编号为 **D-10**（D-09 是参照污染事件）；repairs 记录 append-only 不回改，特此注明编号映射。 | 第二份评审 R2-5 |
| 2026-10-06 | D-15 | **R3-03 日更样本域契约裁决（用户拍板，构建行为变更待实施）**：行域由构建通路（写侧）决定——backfill 维持 require_label=True 冻结域（3,518 分区不动）、build_daily 裁决为 require_label=False 结构域（`is_st==0 & list_days>=365 & is_suspended==0`，对齐旧链 cs_infer 先例 `paper/runner/__init__.py:117-118`）；消费侧统一读取语义 = 训练全段读时按标签非空过滤（`prepare.py:787-789` 既有 dropna 升格为正式口径，冻结段 no-op）。因果性：标签过滤依赖 T+21 未来价格、日更时点物理不可构造；重建冻结分区 = 放弃等价复刻验收基准。差异拆两笔登记：旧链既有（训练 vs 服务边缘行集差异）+ v2 新引入（训练集内部跨缝并存，新事实）。标签刷新义务点名：T+21 端点到达后 forming→sealed 重算归属随 P2a/M1.5 前置工作项立项（标签专用重算路径 + 端点≤实际加载水位触发）。**登记时点行为不变**（build_daily 仍 fail-fast）。完整裁决内容与验收清单见 §9.2。 | §9.2；评审第三份 R3-03；2026-10-06 裁决对话（用户 + 外部意见核验：prepare.py/neutralization.py/labels_builder.py 逐处实证） |

## 9. 评审整改登记（2026-10-06，三评审）

> 依据：`docs/review/v2_p1/v2_p1_impl_review{,2,3}_20261006.md`。本节登记三评审驱动的整改落点；
> 实现常量名以代码为准（本节只写机制与阈值数值，不引用常量名）。

### 9.1 对账门收紧（P1-1/2/3 + R2-1，四件套）

1. **verdict 结构项判定**：`cols_only_*` / `missing_ref_days` 非空不再静默 PASS——加合法性
   判定（日期上界 + 白名单 + 键列噪声剔除），越界即 FAIL；结构项白名单：CYQ 5 列仅允许
   `last ≤ 20171229` 的日；结构项 2012 冷启动段上界 `20121231`。
2. **D-04 归因收紧**：① 子项（share_float/block_trade/top_list 水位族）日期下界
   `20260101`；③/④ 子项（季频财务/基金持仓 raw 修订族）只允许**纯 NaN 形态 + 日期窗
   `[20161201, 20260702]`**——值错位型漂移不再可经 D-04 放行。
3. **D-14 专项三上限**（天数 / 累计行数 / max|Δ|，任一超限即 FAIL）：skewness ≤250 天 /
   ≤40000 行 / max|Δ|≤5；kurtosis ≤750 天 / ≤40000 行 / max|Δ|≤40；zscore_macd_hist
   ≤500 天 / ≤600000 行 / max|Δ|≤20；downside_corr ≤200 天 / ≤300000 行 / max|Δ|≤1.5。
   **D-11 锚定敏感列补窗内超幅上限 5.0**（关闭「窗内任意量级放行」形态，R2-1）。
4. **比较分母修正（R3-04）**：三段式比较的超门占比分母 = **全窗口可比行数**（零差异日
   不再从分母剔除；两套比较器共用统计逻辑）。

### 9.2 已知限制登记

- **R3-03（契约级裁决项，2026-10-06 用户拍板：已裁决；构建行为变更待日更通路实施）**
  **裁决内容：行域由构建通路（写侧）决定，消费侧统一读取语义。**
  ① **写侧双通路**：`backfill`（历史回填）维持 `require_label=True` + single 过滤 = 冻结段
  行集（label_filtered 域），已回填 3,518 分区不动；`build_daily`（当日日更）走
  `require_label=False` = 结构域（`is_st==0 & list_days>=365 & is_suspended==0`，对齐旧链
  cs_infer 先例 `paper/runner/__init__.py:117-118`——v1 双通路先例：cs_train 写时过滤 /
  cs_infer 结构域，生产长期运行）。因果性：标签过滤条件（y_ret_20 非空）依赖 T+21 未来
  价格，日更时点物理不可构造；重建冻结分区 = 放弃「等价复刻 cs_train」验收基准 ⇒ 无第二条
  路。**禁止将行域实现为消费方可配置开关**——单表资产下「哪些行存在」是资产的分段内容属性，
  消费方能定的只是读取过滤。
  ② **消费侧统一读取语义（硬条款）**：训练消费行集 = **全段统一按标签非空读时过滤**（行先
  落盘、消费时剔除空标签行）。落点 = 训练入口既有行为 `ml/train_core/prepare.py:787-789`
  `dropna(subset=[label_column])`，本条为既有机制**升格为正式口径**（非新工程）；冻结段写时
  已过滤 ⇒ no-op。经 `features/neutralization.py:228-229`（neu 标签 = 截面去均值，NaN 同步
  传播）读时过滤 ≡ v1 写时过滤行集。稳态下日更段日期必然进入滚动训练窗口（水位线后无回填），
  无此条款则空标签行进训练矩阵与否取决于实现默认——现 dropna 在 v2 语境下从防御性清洗变为
  行集语义唯一关卡，必须契约保护（防"优化掉"或换读取路径后静默漂移）。
  ③ **labels 表同带分段属性**：冻结段只含非空行、日更段含 forming 行（label_value=NaN）；
  全历史类扫描工具须知晓该分段。
  **差异登记（拆两笔）**：a) **旧链既有**（不阻断）——同日 cs_train(过滤) ↔ cs_infer(结构)
  行集差异，位置在「训练 vs 当日服务」边缘；量化证据 = 口径 A 同日比对（20260702，5,158 行
  交集，非截面列逐值一致率 100%，截面口径差异 104 列全归因，CHANGELOG 0.203.x）。
  〔占位：2026-09-23 纸面 vs OOS 核查（实测同日 cs_infer=5166 / cs_train=5158、+8 行全为
  T+21 无 open_adj 股、Top20 对齐 19~20/20、Spearman 0.9987~0.9995）原件仓库内不可追溯，
  若补落正式报告后升级引用〕。b) **v2 新引入**（新事实，单独登记）——同一族差异挪到训练集
  **内部跨缝并存**（冻结段旧口径 + 日更段新口径同入训练窗口）与跨缝回放路径，幅度预期同级
  （~0.1~0.2%），不得归入「沿旧链既有」。**注**：不得引用母截面契约「分母窄约 10%」
  （那是 cs_train vs 全 clean 截面含 ST/新股/停牌，另一比较，混用误导量级）。
  **运营义务点名（标签刷新归属）**：`maturity_status_for(data_end=...)` 水位感知已落地
  （§3 措辞补注 / 0.204.1 R3-06 整改），剩余义务 = **谁在 T+21 端点数据到达后把 forming
  标签重算并升格 sealed**——现链路无归属（日更只写自己那天、回填不覆盖已写日），不落实则
  日更段标签永久 forming + NaN、训练侧读时过滤丢弃整段（每一天都是空日），日更段永久不可
  训练。实现约束：㆒ 做成**标签专用重算路径**（复用标签计算实现，禁止整日重建——特征指纹
  因数据修订漂移时追加直接报错，会把刷新卡死）；ㆁ 刷新触发按「**端点 ≤ 实际加载水位**」
  判定，不得按「+1 自然月」加载窗（T+21 个交易日跨长假可超窗，D-10 边界缺口为同类先例）。
  归属落点：随日更通路实施立项（P2a/M1.5 前置工作项）。
  **登记机制**：段边界（冻结段右界日）+ 分区行域（`label_filtered` / `structural`）入
  manifest 机器可读字段，读取端可断言/标注；前置依赖 R3-02（core 左表锚定）已于 0.204.1
  （commit 83df5d9）整改落地（已满足）。
  **验收清单（6 条，日更通路实施时执行）**：① 完整未来日历 + 价格水位仅到 T + 足够历史
  预热 + 无任何未来价格 ⇒ 当日特征可产出、行集 = 结构域；② 历史回填行集与数值保持原冻结
  口径逐位不变；③ 负向断言——日更运行后冻结段抽样分区指纹不变（不触碰历史）；④ 行集
  锚——日更产出与同日旧链 cs_infer 产物行集对齐（断言而非声明）；⑤ 生命周期回归——完整
  日历 × 两档水位：forming→sealed 合法改写、未来日历不提前封存、封存后拒改；⑥ 跨缝影响
  声明——当前基线 14 折不跨缝（≤20260702），OOS 窗口推进过水位线后新实验跨缝口径须预登记
  处理，禁止「悄悄跨缝」。
  **登记时点状态**：构建行为不变——`build_daily` 仍 fail-fast（R3-03 防御保留至日更通路
  实施）；`panel_builder.py` 报错文案与 docstring 已同步「已裁决待实施」指向。
- **available_from 全 null 非缺陷**：M1.5 立项前已批准延期（单元 5 §5 语义冲突登记；
  第三份评审 R3-07 复核确认不作为新缺陷）。

### 9.3 冻结参照与 fixture 迁移（P1-8）

- 冻结参照迁至 **`data/frozen_reference/v2_p1/`**（原 `temp/p1_frozen_reference/` 作废；
  迁移后复核 features/cs_train 分区数 = 3,518 一致；24GB 已加入 .gitignore）。
  注：本文档正文不含该路径引用（评审所述「冻结文档 6 处引用」实分布在总验收报告 /
  单元 3 报告 / 术语库 / scripts 默认值——文档侧随各处勘误与术语库修订同步，scripts 侧
  默认路径由代码整改组处置）。
- 列族分组表 JSON 副本落 **`tests/fixtures/p1_column_groups_20261002.json`**（
  `tests/test_v2_store_column_groups.py` 改读该路径，防漂移断言不再依赖 temp/）；
  本文 §2/附录 A 引用的 temp 原物仍在 temp/ 留档，既有引用不受影响。

## 附录 A：列族分组逐列清单

（与 `tests/fixtures/p1_column_groups_20261002.json` 一致；单元 2 起以 `v2/store/column_groups.py` 为代码单一来源，本附录留档备查。）

- **core（52）**：amount, amount_ma10, amount_ma20, amount_ma5, bp, circ_mv, dv_ttm, dv_ttm_missing, ep_ttm, in_date, intraday_vol_structure, is_limit_down, is_limit_up, is_loss, is_st, is_suspended, list_days, log_circ_mv, log_total_mv, ma_deviation_10, ma_deviation_20, ma_deviation_5, opening_strength, pb, pe_ttm, pe_ttm_missing, ps_ttm, ret_1, ret_10, ret_20, ret_5, sw_industry, sw_industry_code, sw_industry_id, sw_l1, sw_l1_code, sw_l1_id, sw_l2, sw_l2_code, sw_l2_id, sw_l3, sw_l3_code, total_mv, tradable, trade_date, ts_code, turnover_rate, vol, vol_ratio_10, vol_ratio_20, vol_ratio_5, volume_ratio
- **fundamental（69）**：assets_turn, capex_to_ocf, cashflow_freshness_days, cashflow_quality_schema_v2, cf_nm, cf_sales, cons_analyst_count_30d, cons_analyst_count_chg, cons_eps_dispersion, cons_eps_dispersion_chg, cons_eps_mean_fy0, cons_eps_mean_fy1, cons_eps_mean_fy2, cons_eps_mean_fym1, cons_eps_revision_30d, cons_eps_revision_accel, cons_eps_yield_fy0, cons_eps_yield_fy1, cons_eps_yield_fy2, cons_eps_yield_fym1, cons_rating_score, cons_rating_upgrade_ratio, cons_revision_freshness_days, cons_revision_schema_v2, cons_target_price_mid, cons_target_upside, cons_target_upside_chg, consensus_freshness_days, current_ratio, debt_to_assets, dividend_continuity_5y, dividend_days_to_ex_date, dividend_freshness_days, dividend_growth_3y, dividend_growth_5y, dividend_hist_missing, dividend_payout_ratio, dividend_recent_imp_ann_10d, dividend_schema_v1, dividend_stability_5y, dividend_yield_hist_12m, equity_yoy, express_freshness_days, express_profit_yoy, express_revenue_yoy, express_roe, express_surprise, fcf, fcf_yield, forecast_chg_mid, forecast_freshness_days, forecast_type_score, fundamental_freshness_days, grossprofit_margin, int_to_talcap, inv_turn, netprofit_margin, netprofit_yoy, ocf, ocf_to_profit, ocf_to_revenue, or_yoy, profit_dedt, q_gr_yoy, q_ocf_to_sales, quick_ratio, roa, roe_dt, roe_waa
- **moneyflow（44）**：elg_net_amount, elg_net_amount_sum_20, elg_net_amount_sum_5, lg_net_amount, lg_net_amount_sum_20, lg_net_amount_sum_5, lhb_amount_rate, lhb_cont_on_list, lhb_cont_up_days_20, lhb_cont_up_days_5, lhb_net_amount, lhb_net_rate, lhb_net_sum_20, lhb_net_sum_5, lhb_on_list, lhb_reason_count, lhb_up_days_20, margin_net_buy, net_mf_amount, net_mf_amount_mean_20, net_mf_amount_mean_5, net_mf_amount_sum_20, net_mf_amount_sum_5, north_net_buy, north_net_buy_ma20, north_net_buy_ma5, north_net_buy_sign_streak, north_net_buy_sum5, north_net_buy_z20, north_turnover, north_turnover_change_streak, north_turnover_flag, north_turnover_ma20, north_turnover_ma5, north_turnover_sum5, north_turnover_z20, order_imbalance, order_imbalance_mean_20, order_imbalance_mean_5, rqye_rzye_ratio, rzye_chg_20, rzye_chg_5, short_balance_change_5, short_sell_vol_change_5
- **technical（43）**：acceleration, alpha_industry_10, alpha_industry_20, alpha_industry_5, amplitude, atr_14, atr_pct_14, bb_lower, bb_middle, bb_pct, bb_upper, bb_width, body_length, cost_concentration, fund_count, fund_count_chg, fund_hold_ratio, fund_hold_ratio_chg, fund_portfolio_freshness_days, ind_momentum_rank, ind_ret_avg, is_new_stock, kdj_d, kdj_j, kdj_k, lower_shadow, macd_dea, macd_dif, macd_hist, rsi_14, size, spec_score, upper_shadow, vol_burst_10, vol_burst_20, vol_burst_5, volatility_10, volatility_20, volatility_5, weight_avg_bias, winner_rate, winner_rate_chg_20, winner_rate_chg_5
- **announcement（10 + 物化四族）**：block_discount_avg_10d, block_discount_days_10d, days_to_unlock, holder_freshness_days, holder_num_chg, holder_num_chg_2q, pledge_freshness_days, pledge_ratio, pledge_ratio_prev, unlock_ratio（物化追加：ht_*/rp_*/tfh_*/ti_*/has_* 全列，以 manifest 登记为准）
- **risk（29）**：amihud_illiq_20, amount_cv_20, cvar_95_20, downside_corr_20, downside_vol_20, drawdown_duration, earnings_yield, gap_risk, garch_persistence, high_low_range_ratio, kurtosis_20, max_drawdown_20, momentum_decay, parkinson_vol_20, pledge_delta, pledge_high_flag, pledge_ratio_decayed, ret_volatility_ratio, skewness_20, turnover_cv_20, turnover_percentile, unlock_risk_flag, up_down_vol_ratio, var_95_20, vol_of_vol_20, vol_ratio_5_20, vol_regime_percentile, volume_climax_days, volume_price_divergence
- **market_state（13）**：mkt_adv_dec_ratio, mkt_atr_pct, mkt_atr_pct_ma250, mkt_drawdown_20, mkt_ma250_ratio, mkt_ma_trend, mkt_ret_avg_20, mkt_ret_avg_60, mkt_ret_vol_20, mkt_turnover_ratio, mkt_turnover_std, mkt_vol_20, mkt_vol_cnt
- **neutralized（117）**：neu_ret_1, neu_ret_10, neu_ret_20, neu_ret_5, zscore_acceleration, zscore_acceleration_sz, zscore_amount_ma20, zscore_amount_ma20_sz, zscore_assets_turn, zscore_assets_turn_sz, zscore_bb_width, zscore_bb_width_sz, zscore_bp, zscore_bp_sz, zscore_capex_to_ocf, zscore_capex_to_ocf_sz, zscore_cf_nm, zscore_cf_nm_sz, zscore_cf_sales, zscore_cf_sales_sz, zscore_cons_analyst_count_chg, zscore_cons_analyst_count_chg_sz, zscore_cons_eps_dispersion, zscore_cons_eps_dispersion_chg, zscore_cons_eps_dispersion_chg_sz, zscore_cons_eps_dispersion_sz, zscore_cons_eps_revision_accel, zscore_cons_eps_revision_accel_sz, zscore_cons_rating_upgrade_ratio, zscore_cons_rating_upgrade_ratio_sz, zscore_cons_target_upside, zscore_cons_target_upside_chg, zscore_cons_target_upside_chg_sz, zscore_cons_target_upside_sz, zscore_current_ratio, zscore_current_ratio_sz, zscore_debt_to_assets, zscore_debt_to_assets_sz, zscore_dividend_continuity_5y, zscore_dividend_continuity_5y_sz, zscore_dividend_days_to_ex_date, zscore_dividend_days_to_ex_date_sz, zscore_dividend_growth_3y, zscore_dividend_growth_3y_sz, zscore_dividend_growth_5y, zscore_dividend_growth_5y_sz, zscore_dividend_payout_ratio, zscore_dividend_payout_ratio_sz, zscore_dividend_recent_imp_ann_10d, zscore_dividend_recent_imp_ann_10d_sz, zscore_dividend_stability_5y, zscore_dividend_stability_5y_sz, zscore_dividend_yield_hist_12m, zscore_dividend_yield_hist_12m_sz, zscore_dv_ttm, zscore_dv_ttm_sz, zscore_elg_net_amount_sum_20, zscore_elg_net_amount_sum_20_sz, zscore_equity_yoy, zscore_equity_yoy_sz, zscore_fcf_yield, zscore_fcf_yield_sz, zscore_grossprofit_margin, zscore_grossprofit_margin_sz, zscore_int_to_talcap, zscore_int_to_talcap_sz, zscore_intraday_vol_structure, zscore_intraday_vol_structure_sz, zscore_inv_turn, zscore_inv_turn_sz, zscore_log_total_mv, zscore_log_total_mv_sz, zscore_ma_deviation_20, zscore_ma_deviation_20_sz, zscore_macd_hist, zscore_macd_hist_sz, zscore_net_mf_amount, zscore_net_mf_amount_sz, zscore_netprofit_margin, zscore_netprofit_margin_sz, zscore_netprofit_yoy, zscore_netprofit_yoy_sz, zscore_ocf_to_profit, zscore_ocf_to_profit_sz, zscore_ocf_to_revenue, zscore_ocf_to_revenue_sz, zscore_opening_strength, zscore_opening_strength_sz, zscore_or_yoy, zscore_or_yoy_sz, zscore_order_imbalance, zscore_order_imbalance_sz, zscore_pb, zscore_pb_sz, zscore_pe_ttm, zscore_pe_ttm_sz, zscore_profit_dedt, zscore_profit_dedt_sz, zscore_q_gr_yoy, zscore_q_gr_yoy_sz, zscore_quick_ratio, zscore_quick_ratio_sz, zscore_roa, zscore_roa_sz, zscore_roe_dt, zscore_roe_dt_sz, zscore_roe_waa, zscore_roe_waa_sz, zscore_size, zscore_turnover_rate, zscore_turnover_rate_sz, zscore_volatility_10, zscore_volatility_10_sz, zscore_volatility_20, zscore_volatility_20_sz, zscore_volatility_5, zscore_volatility_5_sz
- **labels（6，独立成表）**：y_ret_5, y_ret_10, y_ret_20, neu_y_ret_5, neu_y_ret_10, neu_y_ret_20

## 附录 B：raw 快照抽验指纹（sha256 前 16 位，2026-10-02）

| 数据集（分区） | 指纹 |
|---|---|
| daily（2026-07-31） | 2783617a92ab2ce9 |
| adj_factor（2026-07-31） | 43c16bd328e4e29f |
| daily_basic（2026-07-31） | 48ad81b4621f63ba |
| moneyflow（2026-07-31） | 9071cc45a5946ebf |
| stk_limit（2026-07-31） | f79334470d421dc8 |
| suspend（2026-07-31） | a4db2d44e1e1e9f2 |
| stock_st（2026-07-31） | 31124dcea2facd25 |
| margin_detail（2026-07-31） | 2bfdbeb9ee41de34 |
| cs_train（20260702） | 54678be3decbd5d5 |
| clean/daily（2026-07-31） | d6a6f7857393896f |
