# v2 P1 单元 5：起点剖面 + 新增列演练 + events 映射 + 影子通路验收

> 日期：2026-10-06　状态：**终版（四项完成；M1.5 立项与否待用户裁决）**
> 依据：方案 §8-P1 行「raw 单一写入 + 只读转换器 + 样本起点诊断 + 新增列演练」+
> §8.1-M1.5 + events 契约 F2 §2 欠账。

## 1. 列级可用起点剖面（M1.5 前置诊断）

产物：`scripts/v2_p1/profile_column_availability.py`（只读，0.4 min）+
`data/reports/v2_p1_column_availability_20261006.json` +
**专题报告 `docs/reports/v2_p1_column_availability_20261006.md`**（全量口径与数据表）。

结论摘要：2007~2011 段现役模型 154 列门禁幸存 **85.7~87.0%**（被淘汰 20 列全部源头硬缺席
/内禀稀疏：north 13 + cyq 4 + margin 3），瓶颈非「门禁拖死」；可挽回样本 ≈210 万股票-日
（≈+19%，>15% 停止线），regime 多样性 4 类。**建议：立项「分时段变列集训练」，样本起点
2007（不取 2005——moneyflow 源 2007 起 + forecast/express 覆盖爬坡）**。**裁决点：立项
与否待用户拍板**（M1.5 成本登记：编码 2~3 单元 + 重训 1 批 1.5~2h 需授权）。

附带登记（专题报告 §5）：manifest `available_from` 剖面值与 store 层硬查询屏障语义冲突
——剖面值已落 JSON 作治理输入，**manifest 写入推迟到 M1.5 立项实施**（届时先裁决
available_from 语义 = 训练列集元数据 vs 加载屏障）。

## 2. 新增列演练（可执行操作剧本）

产物：`tests/test_v2_new_column_drill.py`（2 例全绿）。演练「新增因子列」全生命周期语义：
① manifest 登记（含 available_from）是写入前提；② 追溯改写已写分区拒绝（热区日指纹冲突 /
封存月原地修改拒绝）；③ 新增列只写登记后新分区，旧分区指纹零变化；④ 登记前分区读取
该列整列 NaN（reindex 补齐）；⑤ available_from 越界查询拒绝；⑥ 封存月热区写入拒
（已封存）+ 新月份冷区写入成功（新增列只写新文件）。

## 3. events 四族映射补全（F2 §2 欠账 → F3 落地）

产物：`docs/contracts/events_state_schema.md` **F3 修订**——§2 映射表补 forecast /
express / dividend / share_float_unlock 四行（七元组规格），键与重复率经全量实证扫描：
- forecast：键 `(ts_code, ann_date, end_date, type)` 唯一（重复 0）；同 `(ts_code,end_date,type)`
  多版 6.42%（首次+修正链，`first_ann_date`/`change_reason` 标记）⇒ `keep_all`。
- express：键 `(ts_code, ann_date, end_date)` 唯一（重复 0；同 `(ts_code,end_date)` 多版 0.05%）
  ⇒ `keep_all`。
- dividend：`event_date = ex_date`（除息日=经济时点；仅实施阶段 22.3% 行可得，缺失回退
  `end_date`）；knowledge_date 实施行取 `imp_ann_date`（清洗规则挂 `_meta.json:
  cleaning_pipeline_ref`）⇒ `keep_all`。
- share_float_unlock：`event_date = float_date`；键 `(ts_code, ann_date, float_date,
  holder_name)` 唯一（重复 0）；同 `(ts_code,float_date,holder_name)` 多公告 20.3%（再公告/
  更新）⇒ `keep_all`（单持有人明细全留，同批聚合下游做）。

封闭注册表语义不变：其余未映射存量数据集仍不得被事件构建器消费。

## 4. 影子通路验收（build_daily(D) 三口径对账）

产物：`scripts/v2_p1/shadow_path_acceptance.py` +
`data/reports/v2_p1_shadow_path_acceptance_20260702.json`。**判定：PASS（问题项 0）**。

- **判据 B（主，同窗等价）**：shadow ≡ 同窗批量参照（`build_features_data` 同捕获窗
  [20251202, 20260702] 直写）——特征 375 列 + 标签 6 列 **0 差异**、结构差异 0
  （34 物化列按冻结 §6 登记为 panel 独有）。**v2 单日通路与批量通路同窗逐位一致。**
- **参照判据（vs panel 当日）**：22 列差异**全部落入登记类**——D-13 豁免族（unlock ×3）/
  公告季频加载窗截断族（cf_nm/ocf_to_profit/dividend_payout_ratio/fund_* 5 列/
  has_fund_holding：shadow [D−7月,D] 窗 vs 回填分块窗；cs_infer 与 shadow 同窗同口径）/
  macd EMA 重排噪声（×3，相对 ~3e-9）/ mkt_atr_pct_ma250 warmup 深度差 / zscore 连锁 ×6。
- **cs_infer 对照（信息登记项）**：124 列差异 + 行集 8 股 = cs_infer 冻结于 2026-07-02
  数据态（raw 2026-08-30/31 全量刷新前）与当前数据态的边界（D-04 类），非路径问题。

**调查教训登记**：判据 B 首轮出现 4 列疑似差异（macd ×3 + mkt_atr_pct_ma250），定位链 =
三方对数（panel/shadow/ref）→ A/B flags 对照（ht/rp/tfh 开关零影响实证）→ 发现
**参照窗口起点取错一天**（D−7 个自然月 = 日历月减法：20260702→**20251202**，而非
20251130；一天之差经 120 日切片锚点传导放大为 mkt/macd 全行小差）。教训：锚定敏感列的
对账窗口必须精确到交易日，「自然月」一律日历月减法。

## 5. 机器耗时台账

| 环节 | 耗时 |
|---|---|
| 起点剖面（含两轮口径修正） | ~1 min |
| 新增列演练（测试编写 + 跑通） | —（秒级） |
| events 四族实证扫描 + F3 修订 | ~3 min |
| 影子通路验收（首轮 20260630 + 20260702 + 同窗参照 ×2 + A/B 对照） | ~40 min |
| 合计 | ~45 min（计划口径「起点剖面分钟级」内，验收对账为重复构建耗时段） |
