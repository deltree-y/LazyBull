# LazyBull v2 runs 产物契约（正式契约）

> **P0 落档标记**：本文档冻结回测 / 实验产物的字段级 schema（账本 / 逐日明细 / 报告分片）。
> **证据机器只认本 schema**；旧 walk-forward 产物由一次性转换器对齐（P5a-1 的"3 个历史实验重算一致"验收同时验收转换器）。
> 目的（方案 v1.4）：防止 P5a 建在旧产物格式上、P2 切换后被"格式适配税"反噬。
> 生效状态：**待 P0 确认**。

| 契约版本 | 落档日期 | 来源 | 变更摘要 |
|---|---|---|---|
| F1 | 2026-10-01 | 方案 v1.8 §4.1 / §8 + 基线批次（wf_batch_20260930_171221 / 172037）产物实测表头 | 首次落档 |
| F2 | 2026-10-01 | P0 评审第二轮（5-A~5-E） | 补 topk_detail.csv schema（信号层尺子输入，卡 P5a-1）；trades 补 4 列映射 + policy_lambda 条件条款 + attribution 头部 3 列；附录 A 改 fail-safe 指纹口径 + baseline_freeze 补快照；重建标注位；文件缺失三态通则 |

---

## 1. 目录结构

```
data/runs/<batch_id>/                 # batch_id = ASCII：<类型>_<YYYYMMDD>_<HHMMSS>_<短指纹>
├── batch_meta.json                   # 批次元数据（配置指纹 + 数据态 + 代码态）
├── summary.csv                       # 逐折汇总（指标 + 超参签名，自描述）
├── chain_nav.csv                     # 链式净值
└── folds/<split_id>/                 # split_id = split00, split01, ...
    ├── trades.csv                    # 成交账本
    ├── daily.csv                     # 逐日组合明细（持仓市值 / 现金 / NAV / 当日指令计数）
    ├── topk_detail.csv               # 逐日 Top-K 明细（信号层尺子输入；可缺，见 §9 通则）
    ├── attribution.csv               # 执行归因（旁路语义，不参与成交判断）
    ├── holdings_snapshot.csv         # 持仓快照（可选旁路；引擎默认关闭、OOS 显式开启）
    └── policy_lambda.csv/.json       # 政策层 λ 台账（条件条款：仅政策层保留时存在，见 §8.1）
```

**命名规则**：目录 / 文件名 / 字段名一律 **ASCII**（禁中文、禁空格——PowerShell 转码教训；旧产物"持仓快照"等中文名属转换器映射范围，中文展示由报告层渲染负责）。

## 2. batch_meta.json（字段级）

| 字段 | 类型 | 说明 |
|---|---|---|
| `schema_version` | int | 当前 = 1 |
| `batch_id` | str | 同目录名 |
| `created_at` | str | ISO 时间戳 |
| `host_mode` | str | `backtest` / `paper_replay` / `paper_shadow` |
| `code_state` | object | `{git_commit, git_dirty}`（复用 `ml/walk_forward/data_state.py` 口径） |
| `data_state` | object | `{data_state_id, sources: {数据集: 最新分区}, cs_train_latest}` |
| `config` | object | 配置指纹全量键值（影响持仓的参数字典，键清单见附录 A） |
| `config_fingerprint` | str | `config` 规范化序列化的 sha 短指纹 |
| `baseline_ref` | str \| null | 对照批次 id（A/B 实验必填，基线批次为 null） |
| `arms` | list[str] | 本批次包含的臂标签 |

**指纹语义**：改任一附录 A 键 ⇒ 指纹变化 ⇒ 对比工具必须显式告警（跨配置结论需登记）；配置指纹一致 + 数据态一致 = 可比性的必要前提（沿折子集对比契约）。

## 3. summary.csv（字段级）

每行 = 一折。字段分四组（现 summary 的 ~120 列自描述原则保留，命名 ASCII 化）：

- **标识**：`batch_id, split_index, model_version, wf_run_id`
- **窗口**：`train_start, train_end, test_start, test_end`（+ 可选 `val_start, val_end`）
- **指标**：`bt_total_return, bt_annual_return, bt_max_drawdown, bt_volatility, bt_sharpe, bt_calmar, bt_trading_days` +
  信号层 `key_top20_hit_rate, key_top20_avg_return_median, key_top20_lift_mean, key_top30_*`（旧 `KEY_*` 列名映射为小写蛇形）
- **配置回显与血缘**：附录 A 全量键 + `data_state_id, git_commit, git_dirty`

## 4. chain_nav.csv（字段级）

| 字段 | 类型 | 说明 |
|---|---|---|
| `date` | str/int | 交易日（YYYYMMDD 或 ISO，批次内统一） |
| `nav` | float | 链式净值（折边界延续，起点 1.0） |
| `split_index` | int | 所属折 |

不变量：折内 `date` 严格递增；各折重复的起始净值点**不计入**交易日数（链式绩效契约）；CAGR / Sharpe 必须经 `chain_metrics.py` 口径重算可复现。

## 5. folds/<split_id>/trades.csv（账本，字段级）

自旧 `walk_forward_trades_*` 映射（数值逐位不动，仅命名规范化）：

| 新字段 | 旧字段 | 说明 |
|---|---|---|
| `wf_run_id, split_index, model_version` | 同左 | 血缘标识 |
| `trade_date` | `date` | 成交日（T+1） |
| `signal_date` | `signal_date` | 信号日（T0） |
| `ts_code` | `stock` | 股票代码 |
| `action` | `action` | `buy` / `sell`（闭集校验） |
| `price, shares, amount, cost` | 同左 | 成交价 / 股数 / 金额 / 费用 |
| `buy_date, buy_price` | 同左 | 卖出行的对应买入信息（卖出行 `buy_date` 必填） |
| `buy_pnl_price, sell_pnl_price, pnl_profit_amount, pnl_profit_pct` | 同左 | 盈亏口径（现语义不变） |
| `sell_type, sell_timing` | 同左 | 卖出类型 / 时点 |
| `sell_reason, trigger_type, buy_type, buy_reason` | 同左（F2 补，P0 评审 5-B 实测偏差） | 卖出/买入原因与触发类型明细（语义沿用现引擎） |
| `lot_id` | —（v2 新增） | 归属 lot（净额化冻结文档 §5；旧产物转换时按 FIFO 重建并标注，见 §5.1） |
| `tranche_idx` | —（v2 新增，可空） | 分批调仓批次号（沿分批调仓契约） |

### 5.1 重建标注字段位（F2，P0 评审 5-D）

- `lot_id` 重建标注 = **文件级 meta**：`folds/<split_id>/_meta.json` 的 `trades.lot_reconstructed=true`；
- `daily.csv` 重建标注 = 同文件 `daily.daily_reconstructed=true`（§9 通则 6）；
- 不加布尔列（行级标注会混入数值列，违背「字段级 schema 全 ASCII 且语义单一」）。

## 6. folds/<split_id>/attribution.csv（执行归因，字段级）

自旧 `walk_forward_execution_attribution_*` 映射：`wf_run_id, split_index, model_version`（F2 补头部三列，P0 评审 5-B）、`signal_date, ranking_date, execution_date, execution_stage, tranche_idx,`
`planned_ts_code(←planned_stock), actual_ts_code(←actual_stock), planned_rank, actual_rank, pred_score, target_weight,`
`status, reason, buy_price, signal_price, signal_to_buy_return`。
语义不变：**旁路记录，不参与候选选择、资金计算或成交判断**（交易归因契约）。

## 7.1 folds/<split_id>/topk_detail.csv（F2 新增，P0 评审 5-A——信号层尺子输入）

逐日 Top-K 明细是信号层尺子（`src/lazybull/compare/signal_metrics.py`）的唯一输入，必须在 schema 内：

| 字段 | 说明 |
|---|---|
| `wf_run_id, split_index, test_start, test_end, model_version` | 头部血缘（现产物同名字段） |
| `trade_date, topk, rank, ts_code` | 交易日 / Top-K 口径（20/30）/ 名次 / 股票 |
| `pred_score, true_return, score_column, ml_score, risk_score, final_score` | 分数与真实收益（现产物同名字段） |

**缺失语义（P0 评审 7-C 实测）**：基线批次 B0/B1 **均无** topk 明细，四个历史实验批次全有 ⇒ 同一 schema 下文件时有时无；
属「可缺 + 可重建」类（见 §9 通则）——缺失时转换器从 trades + summary 无法完整重建（pred_score 不可复原），
必须标注 `_meta.json: topk_detail.missing=true` 并在证据机器侧降级（该折不参与信号层尺子，只参与净值层）。

## 7. folds/<split_id>/holdings_snapshot.csv（持仓快照，字段级）

旧中文表头的 ASCII 映射（存储层 ASCII，报告层负责中文渲染）：

| 新字段 | 旧字段 |
|---|---|
| `run_id, split_index, model_version` | 运行标识, 折序号, 模型版本 |
| `trade_date, ts_code, shares, market_value, weight, total_value` | 日期, 股票代码, 持仓股数, 持仓市值, 持仓权重, 组合总值 |
| `buy_date, signal_date, held_days, due_date, remaining_days` | 买入日, 信号日, 持有交易日数, 到期执行日, 剩余持有交易日 |

旁路纪律沿用：只读记录、不回写持仓状态、不参与买卖判断（开关对成交与净值逐位一致）。

## 8. folds/<split_id>/daily.csv（逐日组合明细，字段级）

| 字段 | 说明 |
|---|---|
| `trade_date` | 交易日 |
| `total_value, cash, market_value` | 组合总值 / 现金 / 持仓市值 |
| `nav` | 折内净值（起点 1.0） |
| `daily_return` | 日收益（引擎口径） |
| `n_positions, n_buys, n_sells, turnover_amount` | 持仓数 / 当日买卖计数 / 换手金额 |
| `exposure_lambda` | 暴露系数（政策层启用时记录；未启用恒 1.0）——λ 序列逐日一致验收的载体 |

### 8.1 folds/<split_id>/policy_lambda.csv/.json（F2 新增，P0 评审 5-B 条件条款）

政策层 λ 台账是 P1.5 裁决的验收依赖物（政策层保留时）：
- **条件条款**：仅当政策层（e2online_r 等）启用时存在；退役时不收（明确登记「退役批次无此文件」）；
- csv 字段（实测）：`date, multiplier, mkt_vol_20, p_loss_mean, holdings, weight_sum, day_mean_return, event_rate, day_weighted_return, loss_day, fold`；
- json = 配套 meta（策略口径 / 折源 / 指纹）；
- 注意口径教训（R-007 §6 登记）：`holdings / weight_sum` 是 terminal_loss 政策面板口径**非真实持仓**，消费方不得当真实持仓用。

## 9. 读取不变量（证据机器读入时的硬校验）

**文件缺失语义通则（F2，P0 评审 5-E）**：每类文件必须归属三态之一——
**必须报错**（trades / summary / chain_nav / batch_meta，缺失即批次非法）/
**可缺 + 可重建**（daily 从账本+快照重建、holdings_snapshot 旁路）/
**可缺 + 不可重建**（topk_detail 缺失即该折降级，见 §7.1）。禁止默认「缺 = 跳过」。

1. `trades.action ∈ {buy, sell}`；卖出行 `buy_date` 非空且不晚于 `trade_date`。
2. `chain_nav` 折内日期严格递增；`summary.bt_total_return` 与 chain_nav 折内起止重算值容差 **1e-6**。
3. `batch_meta.config_fingerprint` 重算一致；`data_state.data_state_id` 与逐折 summary 列一致。
4. 同日同 `(action, ts_code)` 指令已在引擎内合并（账本内不得出现同日同股同向两行）；出现 ⇒ 报错（属引擎 bug，不是数据问题）。
5. 文件名 / 字段名全 ASCII；发现中文字段 ⇒ 提示先跑转换器。
6. 缺失 `daily.csv`（旧产物无此文件）⇒ 转换器从账本 + 快照重建，重建产物记 `_meta.json: daily.daily_reconstructed=true`。
7. **列级校验（F2，P0 评审 5-B 通则）**：目标列集合 = 契约集合——转换器必须产出「源→目标映射 + **丢弃列清单**」，
   行数校验（源 vs 目标）之外，列集合不一致必须报错（静默丢列发现不了）。
8. **跨折衔接（F2 补）**：chain_nav 后折首行日期 = 前折末行日期（折界重复点语义，不计入交易日数）；
   daily 与 trades 交叉对账（`n_buys / n_sells` 与 trades 当日计数一致、末行 `total_value` 与 chain_nav 衔接）。

## 10. 转换器契约（旧 WF 产物 → 本 schema）

- **数值逐位不动**：只做目录搬迁 / 重命名 / 字段名映射 / 编码规范化；禁止任何数值重算（lot_id 与 daily.csv 为标注性重建，例外已注明）。
- 转换报告：行数校验（源 vs 目标）、字段映射表、抽样 md5（每文件 ≥3 行）一并产出。
- 验收：P5a-1 用 3 个已登记历史实验（holdertrade A2 / repurchase / top10fh）重算与既有报表逐项一致——同时验收转换器与证据机器读入链路。

## 附录 A：配置指纹键清单（F2 修订，P0 评审 5-C——fail-safe 方向）

- **口径写死**：**全键入指纹 − 显式排除清单**（fail-safe；新增键默认进指纹，防「漏加进清单就逃逸校验」）。
- **排除清单**（唯一例外，列全）：运行标识类（`wf_run_id, batch_run_id, batch_period_label, registered_at`）+
  统计输出类（`bt_*, key_*, *_samples, best_iteration*`）。**任何其他键一律入指纹**。
- **127 键快照**：以 `docs/contracts/baseline_freeze.md` 附录 A 为唯一参照（F2 补全——原悬空引用已修复）；
  代码承载（独立模块 + 单测）随 P5a-1 转换器一并落地，实现必须从该快照导入排除清单，禁止重写。
