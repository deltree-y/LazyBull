# 数据层新候选盘点（2026-09-18）

> 方法：**实测盘点**（对每个候选接口发最小查询，看可用性/批量能力/字段/PIT 锚点），不是读文档推断。
> 触发背景：路线图 3 个候选（stk_holdertrade / repurchase / disclosure_date）已全部有结论，
> 需要新的候选清单才能继续推进数据层。

## 0. 已消费数据集（26 个，盘点基线）

`adj_factor / block_trade / cashflow / cyq_perf / daily / daily_basic / dividend / express /
fina_indicator / forecast / fund_portfolio(+agg) / income / margin_detail / moneyflow /
moneyflow_hsgt / pledge_stat / report_rc / repurchase / share_float / stk_holdertrade /
stk_limit / stock_st / suspend / top_list`

⇒ 新候选必须在这些之外**提供正交信息**，否则只是加列稀释（列集扰动实测：加 4 列信号 −9.5%）。

## 1. 候选实测结果

| 接口 | 可用性 | **批量能力**（决定成本） | 关键字段 / **PIT 锚点** | 体量（估） |
|---|---|---|---|---|
| `top10_floatholders` | ✓ | **按 `period` 批量**（单页上限 6,000 行，需 offset 翻页） | `ts_code, ann_date, end_date, holder_name, hold_amount, hold_ratio, hold_float_ratio, hold_change, holder_type`；**锚点 `ann_date`** ✓ | ~5 万行/期 × 80 期 ≈ 400 万行 |
| `fina_audit` | ✓ | ✗ **必须传 `ts_code`**（按日期/报告期均报「必填参数, ts_code」）⇒ 全市场逐股循环 ≈ 5,000 次调用 | `audit_result / audit_fees / audit_agency / audit_sign`；**锚点 `ann_date`**（缺失 0.37%）✓ | ~5,000 行/年 ≈ 13 万行全历史 |
| `stk_surv` | ✓ | **按日期批量**（单页上限 400 行） | `surv_date, rece_org, org_type, fund_visitors, rece_mode`；**无 `ann_date`** ⚠️ | 低（每月数百行） |
| `index_weight` | ✓ | 按 `trade_date`（单月 7,000 行） | `index_code, con_code, trade_date, weight`；`trade_date`（月频快照） | 低 |
| `fina_mainbz` | ✓ | ✗ **必须传 `ts_code`** ⇒ 逐股 × 逐期 ≈ **40 万次调用，不可行** | 无 `ann_date`（只有 `end_date`） | — |
| `stk_factor_pro` | ✓ | 按 ts_code/日期 | **261 列**行情+技术因子 | 与自算技术因子高度冗余 |
| `cyq_chips` | ✓ | 按 ts_code + trade_date（单股单日 104 个价位行） | `price, percent` 逐价位筹码 | **全市场约 20 亿行，不可行** |
| `anns_d`（公告） | ✗ | — | 返回「您没有接口权限」 | — |

## 2. `fina_audit` 抽样实测（148 只等距抽样，2,416 行）

| 项 | 实测 |
|---|---|
| `audit_result` 分布 | 标准无保留 **94.87%**；带强调事项段 **2.98%**；**无法表示意见 1.32%**；**保留意见 0.83%** ⇒ **非标合计 5.13%** |
| `audit_fees` | **缺失 57.4%**（只能做辅助项），单位=元，中位 90 万元 |
| 逐年行数 | 报告期维度 ~130~145 行/年/148 只 ⇒ 全市场约 5,000 行/年（**体量极小**） |
| `ann_date − end_date` | 中位 **110 天**（年报 4 月中下旬），5% 分位 51 天 |
| 修订行 | `(ts_code, end_date)` 重复 6 行（重述/修订）⇒ 需版本口径（同 key 保留最新 `ann_date`） |
| 失败率 | 147/148（1 只无记录），无权限/限流失败 |

## 3. 排序建议

### 首选：`top10_floatholders`（十大流通股东）
- **成本最低**：按报告期批量可取，全历史 ≈ 几十~几百次调用（分页 6,000/页）。
- **PIT 干净**：`ann_date` 直接可用（季报公告日）。
- **正交轴**：机构/大户持仓结构（保险、社保、QFII、私募、个人大户）+ 集中度变化；
  与现有 `holder_num`（**散户户数**）、`fund_portfolio`（**公募**）、`margin` 互补——社保/保险持仓是价值域的经典信号。
- 预判风险：`holder_name` 归一化（同名机构识别）、`holder_type` 编码语义需实测、季频滞后 1~4 个月、
  且**只披露前 10 名** ⇒ 结构类因子分辨率有限。

### 次选：`fina_audit`（审计意见/审计费）
- **信号正交度高**：审计师视角的"财报可信度"轴，与 `forecast`/`express`/`fina_indicator` 不重叠；
  非标意见（5.13%）+ 审计机构变更 + 审计费异常（缺失 57%，弱）⇒ 天然是**风控**型状态因子。
- 体量极小（13 万行全历史）。
- **代价**：接口必须逐股查询 ⇒ 全量回补 ≈ 5,000 次调用（按 500 次/分 ≈ 10~25 分钟）；增量维护需周期性
  全市场重扫（无法按日期增量）。这是"低体量、高调用数"的组合，需在 Phase 1 方案里写定刷新策略与水位口径
  （**注意**：本接口无"按日期查询"，水位只能记"上次全扫完成时间"）。

### 备选：`stk_surv`（机构调研）
- 成本低、按日期批量；信号（机构关注度）在 A 股有先例。
- **PIT 风险必须登记**：无 `ann_date`，`surv_date` 是**调研发生日**，信息实际公开（投资者关系活动记录表）
  在其后 1~3 个交易日 ⇒ 直接按 `surv_date` 生效等于**提前知道了尚未公开的信息**。
  可接受做法：可用日 = `surv_date + 3 交易日`（保守），并在方案里登记为"近似锚"。

### 备选：`index_weight`（指数成分与权重）
- 成本低、PIT 干净（月频快照），但主要承载"调入调出事件 / 被动资金"，
  与价值红利策略匹配度一般，优先级低。

## 4. 淘汰清单

| 接口 | 淘汰理由 |
|---|---|
| `fina_mainbz` | 必须逐股 × 逐期 ⇒ 40 万次调用；且无 `ann_date`（可用日需 join `income`，口径成本高） |
| `stk_factor_pro` | 261 列几乎全是我方自算技术因子的重复（现有 cs_train 已含 ~150 技术/量价列） |
| `cyq_chips` | 逐价位筹码 ⇒ 全市场约 20 亿行；且已有 `cyq_perf`（胜率/成本/集中度） |
| `anns_d` | 无接口权限（积分不足） |
| 北向个股持仓类 | 2024-08-19 口径切换后个股持仓不再披露（既有契约已登记） |

## 5. 建议下一步

**Phase 0 审计（首选 `top10_floatholders`）**，按既有协议一次性写定并登记：
① 单页上限与分页读满（实测周期 6,000 行触顶）；② 跨页重复与去重口径；③ PIT 严格性
（`ann_date` 缺失率、窄窗/宽窗切片一致、历史回溯起点）；④ 唯一键（`ts_code+end_date+holder_name` 语义，
含 `hold_ratio` 是否为"占流通股%"）；⑤ 稀疏度与覆盖（每期股票数、每股票行数、`holder_type` 取值域）；
⑥ **raw 存储布局**（按年分区 / 季度分区，沿新增数据集契约在方案阶段写定）。

**并行备选**：若更看重"正交风控轴"，可把 Phase 0 换成 `fina_audit`（本文件 §2 已完成大半抽样实测，
剩余只需补：全市场口径的非标率、`ann_date` 与季报公告日的一致性、逐股循环的限流实测）。

## 6. 复现

```powershell
python temp/api_survey_20260918.py        # → logs/api_survey_20260918.log
python temp/api_bulk_probe_20260918.py    # → logs/api_bulk_probe_20260918.log
python temp/fina_audit_sample_20260918.py # → logs/fina_audit_sample_20260918.log
```

（三个脚本均为一次性盘点脚本，跑完已删除；调用方式与关键结论已固化在本文件。）
