# v2 P1 单元 3：口径 B 探测 + 冻结参照全量重放对账报告（2026-10-03）

> 依据：`docs/data/v2_p1_build_freeze.md`（v2，§8 差异台账 D-01~D-05）+ 契约 R1-4。
> 机制：旧代码（`features/pipeline.py::build_features_data`，**串行生产路径**）+ 数据态 B
> 全量重放 cs_train 等价物 → `temp/p1_frozen_reference/`；只读生产数据，写入仅 scratch/报告。
> 对账器：`src/lazybull/v2/store/replay_compare.py`（atol=1e-6，NaN==NaN 一致；
> 门：超容差行占比 >1e-4 或 max|Δ| > 0.05；joblib loky 逐日并行）。

## 1. 执行摘要

- **口径 B（20260401~20260630，60 交易日）**：行集 60/60 零差异；23 个差异列全部归因——
  17 列命中 D-04 登记项，6 个「待调查」列调查结案（2 列 = pandas 3.0 rolling.skew/kurt
  回归 [D-05]，4 列 = margin_detail 分区修订 [D-04④]）。
- **全量重放**：3,518/3,518 分区全数落盘（串行，5 年 ×3 分块，167.3 min），0 失败日；
  分块间 schema 演进（336/378/383 列）与 cs_train **逐日精确一致**（抽验 4 日）。
- **全量对账（终版 00:4x，全量 skew/kurt 补丁后）**：**126/381 列有差异**（23:07 快照为
  130 列，全量补丁后收敛 4 列），行集 3,518/3,518 零差异；**126 列 100% 完成归因**
  （五类：L1 2012 构建血缘 / L2 2013~15 公式版本血缘 / D-04 数据态漂移 / D-05 环境漂移
  [已修复] / 42 列 2012 schema 缺失），**无实现漂移证据**——2018 起（当前公式版本构建区）
  参照与 cs_train 全列逐值一致。终版复核见 §4a。
- **基线判定（R1-4）**：**重建基线**——以主会话今日修复后代码（D-05 skew/kurt NaN 安全实现）
  重放的冻结参照为基线（参照补丁重建已由主会话并发执行，见 §6）；本重放提供了
  修复前代码态下的完整差异解剖与归因台账。

## 2. 重放配置

| 项 | 值 |
|---|---|
| 构建驱动 | `features/pipeline.py::build_features_data`（串行；生产默认路径，见 §5-③并行留白） |
| builder | `FeatureBuilder(min_list_days=365, horizon=20, horizons=[5,10,20], require_label=True, label_filter_mode="single")` |
| flags | 383 生产全集（中性化双开 + 15 因子族开；ht/rp/tfh 关） |
| 写侧 | `Storage(root_path=temp/p1_frozen_reference*)`（分区存在即跳过 ⇒ 断点续跑） |
| 读侧 | 生产 root `DataLoader(Storage())`（只读） |
| 分块 | 5 年 ×3（[20120104,20161231] / [20150101,20211231] / [20200101,20260702]，起点钳制 Y−2 年 ⇒ 新建日有效历史 ≥19 个月） |
| 口径 B 预热 | `--warmup-start 20250601`（比对窗口 13 个月历史，消除单元 2 已登记的单日/短批次预热伪差异） |

## 3. 口径 B 结论（20260401~20260630）

构建 262 分区（含预热段），串行 **12.6 min**；对账 `data/reports/v2_p1_probe_b_compare_20261003.json`：

- 日级：60/60 比对，0 缺分区，**行集差日 0**；
- 列级：差异列 23（价格-量价/短窗特征**逐值一致**；长记忆特征经预热段后零差异）；
- 归因：D-04③ 8 列、D-04① 3 列、D-04③ 派生连锁 6 列、D-05（skewness_20/kurtosis_20）2 列、
  D-04④（rzye_chg_5/20、short_sell_vol_change_5、short_balance_change_5）4 列。

**可复现性（口径 B 层面）**：成立（差异全部归属已登记数据态/环境条目）。

## 4. 全量对账差异解剖（23:07 快照，130/381 列）

| 簇 | 代表列 | 规模 | 归因 |
|---|---|---|---|
| 区间起点零预热 artifact（生产首建批次） | ret_5/10/20、mkt_*、moneyflow sums、ind_*、kdj_*（2012-01 NaN 型：cs=None/replay=值）；macd_*=0 种子及 EMA/分位衰减（~2012 全年数值差，2013 起收敛） | ~60 列；2012-01 ~20 天 NaN 型 + 衰减期 | **生产首建零预热**（cs_train 2012-01 分区自身即零预热 artifact； replay 按冻结 7 个月预热获全窗口值）。例：cs macd_dif@20120104 ≡ 0.0 |
| 列集演进 | — | 0 差异 | replay 与 cs_train **逐日一致**（2012~2017=378 列、2018+=383 列；cyq_perf raw 起点 2018-01-02 ⇒ 筹码 5 列 2018 起现身；cs_train 20120104 独为 336 列） |
| moneyflow 2011-12 修订 | net_mf_amount_sum/mean_5/20、lg/elg_net_amount_sum_*、order_imbalance_mean_* + zscore 派生（~30 列） | 区间前 ~15 个交易日 | 今日 clean moneyflow 任意窗口**拼不出** cs 值 ⇒ 2011-12 资金流值域构建后修订（覆盖/数值演进；replay 计算已按当前代码验证正确） |
| north 2015-01 覆盖/修订 | north_net_buy_ma20/z20（+派生） | ~12 天（2015-01） | 北向（2014-11 启动）早期数据覆盖/修订 |
| D-04① | days_to_unlock、unlock_ratio、unlock_risk_flag、block_discount_*（5 列） | 全窗口 NaN 型，占比 1~9% | share_float/block_trade/top_list 水位晚于基线数据态 |
| D-04③ | dividend_payout_ratio、cf_nm、ocf_to_profit、fund_*（8 列）+ zscore/neu 派生（6 列） | NaN 型；zscore 派生占比 82~100% | 季频财务/基金持仓 raw 修订 |
| D-04④ | rzye_chg_5/20、short_sell_vol_change_5、short_balance_change_5（4 列） | NaN 型，占比 ≤4e-4 | margin_detail 分区修订丢行（2026-04-15~17 实证） |
| D-05 | skewness_20、kurtosis_20（2 列） | 长停牌股系统性 NaN | **pandas 3.0 rolling.skew/kurt 回归**（主会话 D-05 全文：本机 2026-09-01 升 3.0.5 ⇒ cs_train（08-31 构建）干净、此后构建中毒；已修复为中心矩 NaN 安全实现，全截面验证 skew max\|Δ\|=0 / kurt 6e-8） |
| D-04② | downside_corr_20（1 列） | 当日微移 max 0.011 | raw daily 2024-01-16/17 丢停牌股行 |

注：23:25~23:31 间我曾就 2013~2014 风险族数值差（amihud/downside_vol 等）抽样 scratch，
后证实当时主会话正在并发补丁重建参照分区（23:26 起 mtime），该抽样为补丁中间态、
不作归因依据；独立复算显示当前修复后代码 + 今日数据可精确复现 cs_train 对应值
（000002.SZ@20140102 双列命中）。

## 4a. 终版复核（主会话，00:4x，全量 skew/kurt 补丁后；126/381 列 100% 归因）

全量补丁（3,518 分区，9.3 min）后复跑全量对账：差异列收敛 130→126。按**日期边界分解法**
（逐列定位差异日分布）+ 全截面抽样复核，126 列归入五类、无第六类残留：

| 类 | 列数 | 差异时空边界（复核证据） |
|---|---|---|
| **L1：cs_train 2012 构建血缘（冷启动零预热）** | ~75（ret_*/vol_ratio_*/amount_ma_*/资金流 sum/alpha_industry/kdj/mkt_* 等，median total_rows 3.1 万） | NaN 类集中 2012-01（首日 1,600+ 股 NaN）；19-valid 值类至 2012-02（20120131 抽样 1,669 行，20120229 收敛至 5 行）；macd_* EMA 长尾延续 2012 年内；**2013 起逐值一致**（20130104 全截面 diff=0）。参照语义正确性实证：20120131 000001.SZ 日历窗口×obs-ret 手工复算 = 参照值 1.309730 逐位一致 |
| **L2：cs_train 2013~2015 公式版本血缘** | ~27（风控 22 族多数 + macd_* + mkt_atr_pct_ma250 等，total_rows ~2.4M） | 差异集中 2013~2015 分区：downside_vol_20 比值中位 **2.87~2.96**（旧年化口径）、macd_* max\|Δ\| ~3e3（未复权价口径）；**2018 起逐值一致**（downside_vol ratio 恒 1.00000、diff=0；amihud/garch/vol_regime/turnover_pct/up_down/cvar/macd/kdj/mkt_* 2018/2026 全部 diff=0） |
| **D-04 数据态漂移（raw 水位/修订）** | 21 + 6（zscore/neu 连锁） | 命中登记事件日：2024-01-16/17 丢行（downside_corr_20）、share_float/block_trade/top_list 水位、季频财务/基金持仓修订、margin 2026-04-17 丢行 |
| **D-05 环境漂移（pandas rolling.skew/kurt 回归）** | 2（skewness_20、kurtosis_20） | 已修复消除；修复后残留 = 上述 L1/L2/D-04 类 + float32 ulp 噪声（大值 kurt，归属 L2） |
| **cols_only_replay（2012 schema 缺失）** | 42 | 仅 2012 年代分区（330 列 schema 缺 ret_*/vol_ratio_*/amount_ma*/资金流 sum/alpha_industry/vol_burst/neu_ret_5~20 及对应 zscore）——逐日列交集口径不作差异计，登记为 cs_train 构建血缘事实 |

**归因完备性**：126 列按日期边界全部收敛于 2018 前构建血缘区或登记漂移事件日；
**2018 起（= 当前公式版本构建区）参照与 cs_train 全列逐值一致**——实现路径可复现性确立，
单元 4 归因门的预期差异清单 = 本表五类（禁止新增第六类）。

## 5. 调查与机制发现（证据链）

### ① skewness_20/kurtosis_20 ⇒ pandas 3.0 rolling.skew/kurt 回归（D-05 已由主会话定案修复）

本单元独立证据链：000525.SZ clean 窗口完整、手工重算命中 cs 值；全量帧复现 NaN、
小帧正常；截断实验定位毒害段（该股 20241111~20241212 停牌 25 个交易日）；合成实验
（pandas 3.0.5）证实 **≥ 窗口长度全 NaN 段 ⇒ rolling.skew/kurt 在线状态永久 NaN**。
主会话 D-05 补全定案：pandas 2.3.3 无此问题（3.0.0/3.0.4/3.0.5 均中招）、本机 09-01
升级分层、修复实现与 2.x 逐位一致（max|Δ|≈1e-15）、现役模型 154 列不含 22 个批量
风控列 ⇒ 生产信号无直接影响；修复 + 参照两列补丁重建 + 回归测试已落地。

### ② margin 4 列 ⇒ margin_detail 分区修订（D-04④ 已由主会话吸收入归因表）

300300.SZ：今日 raw margin_detail 2026-04-15/16/17 无行（04-20/21 有）⇒ chg 端点
缺失出 NaN；cs_train 有值 ⇒ 构建后分区被修订。与 D-04② 同机制（两融数据集）。

### ③ 首轮并行路径零落盘 ⇒ 旧代码留白（`_trading_date_index` 缓存）

`_build_features_parallel` 读 `builder._trading_date_index or {}`——该缓存仅由串行
`_get_trading_dates` 建立；standalone 并行 262 天"0 成功 0 失败"零分区（生产默认串行
从未踩到；现有并行单测手工传索引）。本任务全量改走串行（与 cs_train 机制一致；实测
串行 ~1.6s/天优于 loky 开销）；runner `--parallel` 已在薄入口侧补预热（未改旧文件）。
**建议主会话**：pipeline 内自预热一行修复，或文档标注并行前置条件。

### ④ schema 演进实录（available_from 机制的直接输入）

cs_train 列集：20120104 = 336 列、20120105~2017 = 378 列、2018-01-02 起 = 383 列
（cyq_perf raw 起点即 2018-01-02；winner_rate/weight_avg_bias/cost_concentration 等
5 列 2018 起）。重放逐日精确复现该演进。旁证：`_check_features_schema` 对 <383 列
历史分区一律判 False ⇒ 分块重跑时会无效重建 2015~2017（本重放中 ~730 天空转，
值不变、仅耗时）；**建议单元 4 统一分块约定避免重复**。

## 6. 全量重放执行与基线判定（R1-4）

- 重放：3 分块 31.0 + 64.2 + 72.1 = **167.3 min**，3,518/3,518 分区，0 失败；
  对账（3,518 天 joblib）~1.2 min；日志 `logs/v2_p1_full_replay_20261003.log`。
- **基线判定：重建基线**。理由：① 修复前代码态下存在 pandas 环境回归（D-05）与
  生产首建零预热 artifact，现产物 cs_train 与「正确语义」之间存在环境分层；
  ② 主会话今日已修复 `risk/precompute.py`（D-05 NaN 安全实现，全截面验证与
  cs_train max|Δ|=0/6e-8）并并发补丁重建参照分区——**修复后代码 + 数据态 B 的
  重放才是干净基线**；③ 本报告 §4 给出修复前差异的 100% 归因（无实现漂移证据）。
- **单元 4 对账门建议**：
  - **严格门** = v2 panel vs 修复后冻结参照（同代码态 + 数据态 B ⇒ 任何差异 = 实现
    漂移 ⇒ raise；门限沿冻结 §7：占比 1e-4 / max|Δ| 0.05）；
  - **归因门** = v2 panel vs cs_train（差异 = 实现漂移 + 数据态漂移并集；按冻结 §8
    D-01~D-05 + 本报告 §4 新增簇做 100% 归因：区间起点零预热簇、moneyflow 2011-12
    修订、north 2015-01 覆盖、schema 演进 [非差异，按列交集]）；
  - 回填构建与冻结参照**必须统一分块约定**（5 年 ×3 + Y−2 钳制），规避窗口类伪差异
    与 schema 校验无效重建。

## 7. 冻结文档 §8 差异台账追加条目建议（主会话统一登记）

- **D-06（生产首建零预热 artifact）**：cs_train 2012-01 分区（~20 交易日）为零/短预热
  构建产物（macd ≡0 种子、ret_5/10/20 与 moneyflow 和值 NaN、mkt_* 部分窗口），
  EMA/分位族差异衰减至 ~2012 年底；replay 无法也不应复刻（冻结 §1 预热窗口 = 7 个月）。
  处置建议：归因门登记；如需严格逐行一致，区间起点 ~20 天走专项口径评审。
- **D-07（moneyflow 2011-12 值域修订）**：今日 clean moneyflow 任意窗口拼不出
  2012-01 前 ~15 个交易日的 cs 值 ⇒ net_mf/lg/elg/order_imbalance 滚动族 + zscore 派生
  区间起点差异（~30 列）；归因门登记为数据态（覆盖/数值演进）。
- **D-08（north 2015-01 覆盖/修订）**：北向启动初期（2015-01）数据覆盖/修订 ⇒
  north_net_buy_ma20/z20 等 ~12 天差异。
- （D-04④ margin_detail 扩展、D-05 skew/kurt、并行缓存留白三项主会话已登记/处置，
  见 §5-②③。）

## 8. 机器耗时台账（vs 授权 ~2.5h；实际 ~6.2h，超时如实报告）

| 环节 | 实际 | 授权估计 | 说明 |
|---|---|---|---|
| 并行首轮探测（失败转串行） | 24.3 min | — | 换来 §5-③ 留白定位（非计划内但必要） |
| 口径 B 构建（串行 262 分区） | 12.6 min | ≤15 min ✓ | 含 13 个月预热段与查找表固定成本 |
| 口径 B 对账（60 天） | ~1 min | — | |
| 全量重放（3,518 天 ×3 分块） | 167.3 min | ≈120 min | 超时主因：分块 2/3 固定成本（7.7 年日历的共识/现金流查找表各 ~10 min）+ schema 演进致 2015~2017 无效重建 ~730 天 |
| 首轮重放中止后续跑（主会话，20211230 起 1,089 天） | ~57 min | — | 首段子代理生命周期终止于分块 3 加载期（2,429/3,518）；修复后代码续跑（schema 校验重建 2012~2017 + 新建 2022+） |
| skew/kurt 补丁重建（主会话 ×2） | 5.2 + 9.3 min | — | 首轮 ≤20211229；次轮全量（发现 chunk 3 经模块破损窗口期构建后扩展） |
| 全量对账（3,518 天 ×2） | ~1.2 ×2 min | ~20 min ✓ | joblib 20 workers；第二轮 = 全量补丁后终版 |
| 合计（机器） | **~372 min** | ~155 min | 超时 ~217 min：调查驱动（pandas 回归定位/修复/验证、chunk3 破损窗口重建、首轮并行留白）+ 分块固定成本与无效重建——均属契约「口径漂移处置另计」范畴 |

**单元 4 防复发清单**（主会话采纳）：① 回填与冻结参照**统一分块约定**（5 年 ×3 + Y−2 钳制，串行——并行 `_trading_date_index` 缓存留白见 §5-③，生产默认路径一致）；② `_check_features_schema` 对 <383 列历史分区一律判 False 致无效重建 ~730 天——回填跳过逻辑须按「同日同族指纹一致」判定而非 schema 列数（DataStore 两阶段提交的防冲突语义天然覆盖）；③ 并发期修改被依赖模块会导致在跑进程 import 破损窗口——回填期间冻结构建依赖链代码（行为冻结的执行细则）。

## 9. 产物清单

- 脚本：`scripts/v2_p1/run_frozen_reference.py`、`scripts/v2_p1/compare_replay.py`（薄入口）
- 对账器：`src/lazybull/v2/store/replay_compare.py` + `tests/test_v2_store_replay_compare.py`（13 例全绿）
- 重放分区：`temp/p1_frozen_reference/`（3,518 个；skew/kurt 全量补丁重建已完成）
- 对账 JSON：`data/reports/v2_p1_probe_b_compare_20261003.json`（口径 B）、
  `data/reports/v2_p1_frozen_ref_compare_20261003.json`（全量**终版** = 全量补丁后口径，
  126/381 列五类归因，见 §4a）
- 日志：`logs/v2_p1_probe_b_run*_20261003.log`、`logs/v2_p1_full_replay_20261003.log`、
  `logs/v2_p1_full_compare_20261003.log`

---

## 增补（2026-10-05，单元 4 严格门复核驱动）——参照修复与 §4a 归因更正

单元 4 严格门（v2 panel vs 本参照）首轮复核暴露**本参照自身的两处构建缺陷**，修复后
§4a 的 L2 归因须更正：

1. **分块 1 逐日回退污染（冻结文档 §8 D-09）**：首轮重放的分块 1（2012~2016）分区
   风控 22 列为 float64 逐日回退路径产物（批量路径应输出 float32）——逐日回退与批量
   语义系统性不同（downside_vol 比值 ~0.33~0.38）。dtype 探出后删除重筑（当前修复后
   代码，批量路径复归，20120104 全截面 vs panel max|Δ|=0）。
2. **重叠区 schema 校验驱动的重建覆盖（同 D-09）**：`_check_features_schema` 对
   <383 列分区一律判 False ⇒ 后续分块把 2015~2017 重叠区以自身较短预热链重建覆盖，
   污染 warmup 敏感列（mkt_ma250_ratio/macd_* 等）。回填侧 keep_dates（已写日不重建）
   免疫；参照侧以「单分块重筑」修复。**结论更正：§4a 的 L2 类（"cs_train 2013~15 公式
   版本血缘"）误归因——真实异常方是本参照；cs_train 与 panel 在该年代逐值一致。**
   D-07 已在冻结文档同步更正。v2 侧纪律：**已写判定一律走 manifest 存在性，不走
   schema 校验**。
3. **驱动等价性实证（A/B 对照实验）**：同区间 [20141201, 20150131] 同预热下，
   run_frozen_reference（旧存储直写）与 V2PanelBuilder.backfill（捕获拆分）逐位一致
   （mkt_ma250_ratio/downside_vol_20/ret_20/macd_dif/skewness_20 五列 1,908 行 max|Δ|=0）
   ⇒ 两个驱动的数值路径等价。
4. **分块 1 末 3 日（20161228/29/30）回补**：分块 1 末标签不成熟（T+21 超分块加载窗）致
   缺；以分块 2 预热口径构建回补（2014-06 起，2.5 年深度处与分块 1 链逐位一致——panel 侧
   实证 macd/mkt max|Δ|≤1e-14）。

修复后参照 = 3,518 分区、全量批量路径（float32）、与 panel 同分块约定同预热链。
单元 4 严格门终版结果见单元 4 报告（`docs/reports/v2_p1_backfill_reconcile_20261005.md`）。
