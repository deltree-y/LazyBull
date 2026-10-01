# LazyBull v2 事件库 / 状态库统一 schema（正式契约）

> **P0 落档标记**：本文档冻结 `events/` 与 `state/` 两类资产的统一 schema，并落档**公告源试数据协议**
> （公告源是袖子 B 的"第一优先"数据，协议优先级不低于分钟线试数据协议）。
> 生效状态：**待 P0 确认**。
> 上游依据：方案 §4.1 补丁 2（events/state schema 提前到 P0）、§7（试数据协议）、R2.5-7（披露时点 PIT 审计）。

| 契约版本 | 落档日期 | 来源 | 变更摘要 |
|---|---|---|---|
| F1 | 2026-10-01 | 方案 v1.8 §4.1 / §7 + 四族数据集契约 | 首次落档 |
| F2 | 2026-10-01 | P0 评审第二轮（6-A~6-C） | 扩 revision_policy/dedup_rule 封闭词表（修 §2 违反 §1 的自相矛盾）；存量公告型数据集映射补全计划；公告源审计样本量升级（≥60）+ 前视/延迟分离 + 分层抽样 |

---

## 1. events 统一 schema（七元组）

所有事件表（`data/events/<event_type>/YYYY.parquet`，按 `knowledge_date` 年分区）逐行携带：

| 字段 | 类型 | 语义 |
|---|---|---|
| `event_type` | str | 事件类型（封闭注册表，未知类型 fail-fast） |
| `ts_code` | str | 标的 |
| `event_date` | str | **事件发生的经济时点**（YYYYMMDD；如报告期 / 除息日 / 交易日；无独立经济时点时 = `knowledge_date`） |
| `knowledge_date` | str | **系统可得时点**（YYYYMMDD；PIT 锚点，一律取公告日 / 披露日口径） |
| `revision_key` | str | 版本键（同事件多版本的区分键，序列化后的元组） |
| `revision_policy` | str | 版本策略：`keep_all`（明细全留，聚合下游做）/ `keep_latest`（仅最新版本生效）/ `keep_max:<field>`（同键取某字段最大行） |
| `dedup_rule` | str | 去重口径声明（如 `full_row` = 全字段整行去重） |

**规则**：

- 双时间轴是硬约束：`event_date` 与 `knowledge_date` 缺一不入库；PIT 消费一律以 `knowledge_date ≤ T` 过滤。
- `revision_policy` / `dedup_rule` 是**每张事件表的注册属性**（写在表头 `_meta.json`），逐行重复存储只为自描述；
  两字段取值必须在封闭词表内（F2 扩，P0 评审 6-A）：
  - `revision_policy` 词表：`keep_all` / `keep_latest` / `keep_max:<field>` / `priority:<rule_name>`（按命名规则优先级选留，如 `priority:single_day_first`）；
  - `dedup_rule` 词表：`full_row` / `key_tuple` / `correct_swap_then_key_tuple`（先列修正再按键三元组去重，top_inst 用）；
  - 长描述 / 多步管线挪到 `_meta.json` 自由字段 `cleaning_pipeline_ref`（指向清洗过程文档 / 代码版本）。
- 事件表 = 去重对齐后的一等公民资产（不再是"窗口聚合因子列"的附属）；窗口聚合留在 `factors/` 消费事件表产出。

## 2. 四族既有契约 → 实例规格映射

| 族 | `event_type` | `event_date` | `knowledge_date` | `revision_key` | `revision_policy` | `dedup_rule` |
|---|---|---|---|---|---|---|
| 股东增减持 | `holder_trade` | = `ann_date`（无变动期间字段，契约定死只认公告日） | `ann_date` | `(ts_code, ann_date, holder_name, in_de)`（F2：标注源字段名 `in_de`） | `keep_all`（无自然唯一键，多行明细保留） | `full_row`（源内整行重复 24.7%） |
| 股票回购 | `repurchase` | `end_date`（进度报告期，可空 ⇒ 空时 = `ann_date`） | `ann_date` | `(ts_code, ann_date, proc)` | `keep_all`（阶段机多行明细：预案/通过/实施/完成） | `full_row`（跨页重复 0.31%） |
| 十大流通股东 | `top10_float_holders` | `end_date`（报告期） | `ann_date` | `(ts_code, end_date, ann_date, holder_name)` | `keep_max:hold_amount`（同键多行 0.06%，取金额最大行，禁止求和） | `full_row`（跨页重复 0） |
| 龙虎榜机构席位 | `top_inst_seat` | `trade_date` | `trade_date`（当日收盘后披露） | `(ts_code, trade_date, exalter, side, reason)` | `priority:single_day_first`（F2：单日榜优先，剔除 reason 含「连续/累计」行，无单日榜才保留连续榜） | `correct_swap_then_key_tuple`（F2：先列修正 buy/sell 互换，再按键 + 金额三元组去重；多步管线见 `_meta.json: cleaning_pipeline_ref`） |

- 四族的抓取协议（分页上限 / 水位 / 回拉窗口）仍归各自数据集契约；本表只冻结**事件抽象层的映射口径**。
- 新事件族入库 = 在本表追加一行并经评审（closed registry 语义）。
- **存量公告型数据集映射补全计划（F2，P0 评审 6-B）**：方案 §3.1 规划的公告流含预告 / 快报 / 分红 / 解禁等，
  §2 当前只映射四族；**迁移路径**——P1 数据底座阶段按「袖子 B 立项所需优先」顺序补映射（forecast / express / dividend / share_float），
  未映射的存量数据集**不进入**封闭注册表、不得被事件构建器消费（防「目录在但语义未冻结」的灰区）。

## 3. state 统一 schema

- 物理形态：`data/state/<state_family>/YYYYMMDD.parquet`（日分区）。
- 字段：`trade_date` + 状态列（波动 / 流动性 / 宽度 / 风格动量……按族登记）+ **`definition_version`**（定义版本，整数）。
- **版本冻结**：历史区间按当时 `definition_version` 冻结封存；定义变更 = 版本递增 + 从变更点起算，**禁止原地改写历史**（重算历史须另存新族或显式全量版本迁移并登记）。
- **写入约束（F2 补）**：同一日同族**仅一个版本**（重复写入同日同族同版本 ⇒ 报错；同日新版本 ⇒ 显式覆盖并登记审计行）。
- 血缘：state 由**状态构建器**从 `normalized/` 派生（写入者唯一）；策略层与证据层共用同一状态库，禁止各自重算。

## 4. 公告源试数据协议（袖子 B 第一优先数据）

公告全文 / 结构化事件流是事件漂移袖子的弹药。采购 / 接入前必须走完三步（与分钟线试数据协议对齐）：

1. **质量对账**：
   - 结构化字段交叉核对（公告标题 / 类型 / 标的 vs 已落库 express / forecast / fina 记录）；
   - **披露时点 PIT 抽样审计（本协议核心）**：公告时间戳 1 日误差可吞掉整个 PEAD——每源每季**分层抽样 ≥60 条**
     （F2，P0 评审 6-C：原 ≥30 条统计功效不足——真实错误率 6% 的源漏检概率 ≈45%；按公告类型分层，防随机抽不到关键类型），
     以交叉源（交易所披露页 / 巨潮发布时间）人工核对 `knowledge_date` 与「真实可获取时点」；
     **误差方向分离（F2）**：前视（时间戳早于真实可得，虚增回测收益）与延迟（保守偏差）性质不同，分列两档阈值——
     **前视 >2% 即停采**（或逐条复核后降级），延迟 >5% 降级（消费侧 `knowledge_date` 保守 +1 自然日，注明「自然日 or 下一交易日」须在审计报告写明）；
     审计记录进台账。
2. **价值预筛**：2~3 个事件候选（如业绩预告超预期漂移）过信号层尺子粗测——**只作"值不值得买 / 接"的预筛，不算正式实验结论**。
3. **成本评估**：覆盖速率（只/日）、更新延迟（公告 → 可抓取的时滞分布）、全历史回填体量与增量拉取耗时、采购价格。

**决策门**：三项过关 → 立项接入（事件构建器 + 本契约 §1 schema）；任一不过 ⇒ 登记台账，不采购。

**反爬降级方案**（抓取工程纪律）：

- 降级阶梯：降频 → 分片错峰 → 暂停该源 + 告警 → 人工介入；每级触发条件与恢复条件写入采集器配置。
- 缺口管理：抓取缺口**逐日登记**（数据集 `_meta.json` 水位 + 缺口清单），禁止静默跳过；
  连续失败超阈值升级告警（沿每日运行契约的失败处理分级）。
- 已落库数据不受降级影响（raw 不可变只追加）。
