# LazyBull v2 正式契约区（P0 起）

> 本目录承载 v2 重构的**正式契约文档**。自 2026-10-01（F6）起方案与协议实行**单文件制**：
> 契约文件本身即唯一载体（原 `docs/plans/` 双轨工作稿已退役，历史版本备份于 `docs/plans/v2/backup/`）。
> 版本治理：契约正文 append-only，修订保留历史；每次修订在文档头部登记版本行（状态=草案 → 评审通过改生效）；
> **生效才 commit**——git 历史只保留生效版本快照，裁决引用「F 版本号 + commit」。

## 目录

| 文件 | 内容 | 状态 |
|---|---|---|
| [v2_architecture_plan.md](v2_architecture_plan.md) | v2 架构方案（**F11 = v1.17**：P2a 开工前——§4.10 迁移期过渡例外逐项到期表 / M1.5 方向立项 + 标签刷新立项 / P2a 预算 ≈5 / **M1 判据 2→3 倍（停止线 57 单元 / 48h）**；**F10 = v1.16**：P2a 前置在用全域对账清零、PositionRiskModel 不迁移；**F9 = v1.15**：P5a-2 执行登记；**F8 = v1.14**：P1.5 裁决——e2online_r 退役、迁移目标 B0；**F7 = v1.13**：§5 一次性修订窗口；原名 `v2_architecture_plan_frozen.md`，2026-10-01 更名去 frozen 后缀） | **已生效（2026-10-07，F11）** |
| [protocols.md](protocols.md) | 模块协议层（**F8 = v0.8**：P2a T0 勘误包——止盈行无在用对照物改退役机制描述 / 估值回退 4→5 处 / 15 钩子括注按代码实测 / Kelly 包装合并兑现节点 / §11 快速复核已执行；**F7**：P1 R3-10 勘误——manifest_version 为 schema 版本、内容身份由新增 manifest_fingerprint 承载；F6 = v0.7 内容零变更，载体合并） | **已生效（2026-10-07，F8）** |
| [baseline_freeze.md](baseline_freeze.md) | 基线双臂冻结（B0/B1 配置指纹 + 逐折收益路径 + 附录 A 127 键快照） | **已生效（2026-09-30 冻结，2026-10-01 随 P0 确认）** |
| [runs_artifact_contract.md](runs_artifact_contract.md) | runs 产物契约（账本 / 逐日明细 / 报告字段级 schema + 转换器契约 + topk_detail + policy_lambda 条件条款） | **已生效（2026-10-01 P0 确认）** |
| [events_state_schema.md](events_state_schema.md) | 事件库 / 状态库统一 schema + 公告源试数据协议（词表已修 + 审计样本量升级） | **已生效（2026-10-01 P0 确认）** |
| [hypothesis_ledger_schema.md](hypothesis_ledger_schema.md) | 假设台账 schema（含候选池分母 / α 预算 / 顺序检验规格——块方差已拍板；**F4**：`supersede_reasons` 字段登记，2026-10-07） | **已生效（2026-10-01 P0 确认；F4 = 2026-10-07）** |
| [netting_freeze.md](netting_freeze.md) | 净额化冻结文档（分层语义 / 状态机表 / 三恒等式 / lot 模型 + 同向分摊 / 双口径 / 兜底链） | **已生效（2026-10-01 P0 确认）** |

配套（非契约但属 P0 交付）：术语库 `docs/glossary/terms.yaml`（首版 71 条）；
复杂度 hook `scripts/check/check_complexity.py`（存量棘轮基线同目录 `complexity_baseline.json`）；
术语引用统计 `scripts/check/glossary_refs.py`。

## 落档规则

1. 任何契约修改必须经评审并登记版本行；
2. 契约条文分级（裁决级机器校验 / 卫生级自动化）经评审**登记为暂不采纳**（方案 §0.12：当前 hook + 评审节奏已够，待首份契约失效实例出现再立）；本索引层保留可选的状态标注，不产生条文级分级义务；
3. 与代码不一致时，以契约为准修改代码，或先改契约再改代码（禁止静默漂移）。
