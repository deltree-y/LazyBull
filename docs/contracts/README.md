# LazyBull v2 正式契约区（P0 起）

> 本目录承载 v2 重构的**正式契约文档**（与 `docs/plans/` 工作稿区分——plans 是本地计划区，contracts 是落档生效区）。
> 版本治理：契约正文 append-only，修订保留历史；每次修订在文档头部登记版本行。

## 目录

| 文件 | 内容 | 状态 |
|---|---|---|
| [v2_architecture_plan_frozen.md](v2_architecture_plan_frozen.md) | v2 架构方案冻结版（自 plans/v2_architecture_plan.md v1.7b 落档） | P0 待确认 |
| [protocols.md](protocols.md) | 模块协议层（自 plans/v2_protocols.md v0.3 转正） | P0 待确认 |
| [runs_artifact_contract.md](runs_artifact_contract.md) | runs 产物契约（账本 / 逐日明细 / 报告字段级 schema） | P0 待补全 |
| [events_state_schema.md](events_state_schema.md) | 事件库 / 状态库统一 schema | P0 待补全 |
| [hypothesis_ledger_schema.md](hypothesis_ledger_schema.md) | 假设台账 schema（含候选池分母 / α 预算 / 顺序检验规格） | P0 待补全 |
| [netting_freeze.md](netting_freeze.md) | 净额化冻结文档（分层语义 / 状态机表 / 三恒等式 / lot 模型） | P0 待补全 |

## 落档规则

1. 任何契约修改必须经评审并登记版本行；
2. 契约条文分为**裁决级**（机器校验进 hook）与**卫生级**（AI 自动化 + 季度审计）——分级标注随各文档落档时补齐；
3. 与代码不一致时，以契约为准修改代码，或先改契约再改代码（禁止静默漂移）。
