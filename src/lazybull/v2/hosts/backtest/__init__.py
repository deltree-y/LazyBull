"""v2 回测宿主（P2a-T7）：OOS 装配 / 重放驱动 / 独立报告 / runs 出口。

- ``runtime``：TradingConfig 映射、信号创建、引擎构造（复制自旧
  ``common/backtest_runtime.py``）；
- ``replay``：walk-forward 单折 OOS 重放驱动（复制改指自旧
  ``ml/walk_forward/backtest.py``，B2 蓝本；exposure 政策族按 D6 摘除并
  非默认值 fail-fast）；
- ``reporter``：独立回测报告生成器（复制自旧 ``backtest/reporter.py``）；
- ``runs_writer``：runs 产物契约（F2 schema）的正式出口（全新代码，
  非临时适配器，见 ``docs/contracts/runs_artifact_contract.md`` §5-8）。
"""
