# LazyBull v2 正式契约区（P0 起）

> 本目录承载 v2 重构的**正式契约文档**（与 `docs/plans/` 工作稿区分——plans 是本地计划区，contracts 是落档生效区）。
> 版本治理：契约正文 append-only，修订保留历史；每次修订在文档头部登记版本行。

## 目录

| 文件 | 内容 | 状态 |
|---|---|---|
| [v2_architecture_plan_frozen.md](v2_architecture_plan_frozen.md) | v2 架构方案冻结版（F5 = 工作稿 v1.11，P0 评审第二轮） | P0 待确认 |
| [protocols.md](protocols.md) | 模块协议层（F5 = 草案 v0.7；§11 对照表已打钩 22/22 + 止盈行 + 双层归属） | P0 待确认 |
| [baseline_freeze.md](baseline_freeze.md) | 基线双臂冻结（B0/B1 配置指纹 + 逐折收益路径 + 附录 A 127 键快照） | 已冻结（2026-09-30；附录 A F2 补） |
| [runs_artifact_contract.md](runs_artifact_contract.md) | runs 产物契约（账本 / 逐日明细 / 报告字段级 schema + 转换器契约 + topk_detail + policy_lambda 条件条款） | P0 待确认（F2） |
| [events_state_schema.md](events_state_schema.md) | 事件库 / 状态库统一 schema + 公告源试数据协议（词表已修 + 审计样本量升级） | P0 待确认（F2） |
| [hypothesis_ledger_schema.md](hypothesis_ledger_schema.md) | 假设台账 schema（含候选池分母 / α 预算 / 顺序检验规格——块方差已拍板） | P0 待确认（F2） |
| [netting_freeze.md](netting_freeze.md) | 净额化冻结文档（分层语义 / 状态机表 / 三恒等式 / lot 模型 + 同向分摊 / 双口径 / 兜底链） | P0 待确认（F3） |

配套（非契约但属 P0 交付）：术语库 `docs/glossary/terms.yaml`（首版 71 条）；
复杂度 hook `scripts/check/check_complexity.py`（存量棘轮基线同目录 `complexity_baseline.json`）；
术语引用统计 `scripts/check/glossary_refs.py`。

## 落档规则

1. 任何契约修改必须经评审并登记版本行；
2. 契约条文分级（裁决级机器校验 / 卫生级自动化）经评审**登记为暂不采纳**（方案 §0.12：当前 hook + 评审节奏已够，待首份契约失效实例出现再立）；本索引层保留可选的状态标注，不产生条文级分级义务；
3. 与代码不一致时，以契约为准修改代码，或先改契约再改代码（禁止静默漂移）。
