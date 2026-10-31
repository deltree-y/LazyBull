# -*- coding: utf-8 -*-
"""P4 阶段验收工具包（阶段专用，按临时开发文件契约豁免复杂度门）。

- ``t05b_step_driver``：T0.5b 步进等价性 spike——连续 run vs 逐日步进 + 每日落盘恢复原型；
- ``t05b_live_anchor``：live 持久化锚定四场景 + 隔离根（R14）实证；
- ``t05b_gate2b_binding``：门 2b 数据态绑定留证；
- ``t05b_idempotency_probe``：旧臂 ensure_* 禁下载幂等性实证探针（D4/R14）；
- ``t05b_spike_runner``：T0.5b 编排器（证据落盘 data/frozen_reference/v2_p4/）；
- ``t1_ast_enum``：T1 登记枚举 AST 核对（EngineState 字段归类注册表一致性）；
- ``t1_worktree_digest``：T1 工作树摘要留证；
- ``t2_cs_infer_panel_smoke``：T2 cs_infer↔panel 只读冒烟（列集/行数对照，非判据）；
- ``t2_daily_dryrun``：T2 单日 dry-run（隔离根驱动 + 生产根 containment 双快照）。

全部工具不改 src/ 生产代码、不写生产数据根；仅驱动现有实现并落证据。
"""
