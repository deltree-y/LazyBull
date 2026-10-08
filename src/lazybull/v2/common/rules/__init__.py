"""v2 跨端共享纯函数（common/rules/，契约 §4.3）。

「禁止任何一侧自行重算」契约的代码形态：止损/到期判定/分批排期等
纯判定函数的唯一归属地；调用编排在 core/decision（双层归属，v1.11）。

当前成员（P2a-T1 自旧链路复制，行为冻结、双源并存至切换）：

- ``sell_rules``：持有期到期判定与调仓卖出候选筛选（旧 ``trading/sell_rules.py``）
- ``stagger``：分批调仓排期、槽位拆分、预算比例（旧 ``trading/stagger.py``）
- ``stop_loss``：StopLossConfig / StopLossMonitor（旧 ``risk/stop_loss.py``）
- ``stop_loss_checker``：统一止损检查纯函数（旧 ``risk/stop_loss_checker.py``）
"""
