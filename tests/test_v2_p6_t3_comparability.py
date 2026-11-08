# -*- coding: utf-8 -*-
"""P6 T3 可比性（计划 §3-T3：不同数据态/折集合/窗口/host/实现/基线未确认 ⇒ not_comparable）。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.reporting import experiment_metrics, status  # noqa: E402
from tests.p6_t3_env import arm_from_nav  # noqa: E402


def _splits(arm):
    return sorted(int(item) for item in arm.windows["split_index"].tolist())


def test_p6_t3_06_data_state_mismatch_not_comparable(tmp_path):
    base = arm_from_nav(tmp_path, arm_id="base", navs=[1.0, 0.9, 1.0])
    arm = arm_from_nav(tmp_path, arm_id="arm", navs=[1.0, 0.95, 1.0], data_state_id="ds_other")
    result = experiment_metrics.compare_pair(base, arm, _splits(base))
    assert result["status"]["report_status"] == "not_available"
    assert result["status"]["degradation_reason_code"] == "not_comparable"
    assert "数据态" in result["status"]["detail"]
    assert result["summary"] == {}


def test_p6_t3_07_split_or_window_mismatch_not_comparable(tmp_path):
    base = arm_from_nav(tmp_path, arm_id="base", navs=[1.0, 0.9, 1.0])
    shorter = arm_from_nav(tmp_path, arm_id="short", navs=[1.0, 0.95])
    result = experiment_metrics.compare_pair(base, shorter, _splits(base))
    assert result["status"]["degradation_reason_code"] == "not_comparable"
    assert result["summary"] == {}
    # 折数相同但窗口终点不同：仍不可比
    shifted = arm_from_nav(tmp_path, arm_id="shift", navs=[1.0, 0.9, 1.0])
    shifted.windows = shifted.windows.assign(test_end="20991231")
    result_window = experiment_metrics.compare_pair(base, shifted, _splits(base))
    assert result_window["status"]["degradation_reason_code"] == "not_comparable"
    assert result_window["summary"] == {}


def test_p6_t3_08_host_implementation_baseline_not_comparable(tmp_path):
    base = arm_from_nav(tmp_path, arm_id="base", navs=[1.0, 0.9, 1.0])
    other_host = arm_from_nav(
        tmp_path, arm_id="host", navs=[1.0, 0.9, 1.0], host_mode="paper_replay"
    )
    result = experiment_metrics.compare_pair(base, other_host, _splits(base))
    assert result["status"]["degradation_reason_code"] == "not_comparable"
    assert "host_mode" in result["status"]["detail"]
    # 基线不可用 ⇒ 任何臂都不产差值（不拿坏基线做分母）
    broken = arm_from_nav(tmp_path, arm_id="broken", navs=[1.0, 0.9, 1.0])
    broken.status = status.make_status(
        "invalid", reason_code="chain_source_missing", detail="合成负例"
    )
    arm = arm_from_nav(tmp_path, arm_id="arm", navs=[1.0, 0.95, 1.0])
    blocked = experiment_metrics.compare_pair(broken, arm, _splits(arm))
    assert blocked["status"]["degradation_reason_code"] == "not_comparable"
    assert "基线" in blocked["status"]["detail"]
