# -*- coding: utf-8 -*-
"""v2 训练通用引擎层（P3：数据只读适配 + 训练内核 + 滚动编排）。

依赖方向契约（§4.3 / §13.3.1）：``train → common.types, common.protocols, store``；
禁止 import sleeves / universe / factors / 业务包。cs_train 验证源模式对旧
``data.loader/storage`` 的依赖为验收专用过渡例外（§3.3，去除节点 = P4 评审）。
编排器对 v2 内核回测（hosts）与信号（core）的依赖一律经宿主注入（依赖倒置，
§17.6）；协议实现：``trainer.V2Trainer``（Trainer）/
``walk_forward.orchestrator.V2WalkForwardOrchestrator``（WalkForwardOrchestrator）。
"""
