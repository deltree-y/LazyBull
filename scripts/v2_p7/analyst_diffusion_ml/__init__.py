"""AD-ML-20-v1 分析师预期扩散 ML 袖研究包（经济预筛，不进生产链路）。

协议：`docs/experiments/p7_analyst_diffusion_ml_protocol_20261118.md`
"""

from __future__ import annotations

from scripts.v2_p7.analyst_diffusion_ml import (  # noqa: F401
    artifacts,
    diffusion,
    labels,
    panel,
    report_events,
    spec,
    universe,
)

__all__ = [
    "artifacts",
    "diffusion",
    "labels",
    "panel",
    "report_events",
    "spec",
    "universe",
]
