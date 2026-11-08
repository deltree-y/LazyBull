# -*- coding: utf-8 -*-
"""P6 T3 生成性能门与包内复算（计划 §3-T3：标准+压力性能与包内复算通过）。"""

from __future__ import annotations

import json
import sys
import time
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.reporting import experiment_csv, generate, manifest, perf_gate  # noqa: E402
from tests.p6_t3_env import load_arms, write_runs_batch  # noqa: E402


def _request(tmp_path: Path, *arms: tuple[str, Path]) -> generate.ExperimentRequest:
    return generate.ExperimentRequest(
        reports_root=tmp_path / "reports",
        evidence_class="research",
        generated_at="20261106T000000Z",
        attempt_id="gen_t3_perf",
        input_evidence_classes=("research",),
        arms=tuple(f"{arm_id}=runs:{path}" for arm_id, path in arms),
        baseline=arms[0][0],
        git_record={"git_commit": "abc1234", "git_dirty": True},
        publish_run=False,
    )


def test_p6_t3_22_generation_gate_and_csv_schemas(tmp_path):
    """完整进程生成 < 10s、页面 < 3MB，manifest 登记五份 CSV 的 schema。"""
    base = write_runs_batch(tmp_path / "runs_base", batch_id="batch_base")
    arm = write_runs_batch(tmp_path / "runs_arm", batch_id="batch_arm")
    started = time.perf_counter()
    result = generate.generate_experiment(_request(tmp_path, ("base", base), ("arm", arm)))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    assert elapsed_ms < perf_gate.MAX_GENERATION_MS
    html_bytes = result["files"]["report.html"]
    assert perf_gate.judge_size(len(html_bytes))["passed"]
    # 阈值负例：注入合成度量记录只验判定逻辑（不制造真实慢页面）
    assert perf_gate.evaluate_metric({"kind": "generation", "value": 9_999.0})["passed"]
    assert not perf_gate.evaluate_metric({"kind": "generation", "value": 10_000.0})["passed"]
    assert not perf_gate.evaluate_metric({"kind": "size", "value": 3_000_001})["passed"]
    series = perf_gate.judge_generation_series([elapsed_ms] * 5)
    assert series["passed"] and series["repeat"] == 5
    record = json.loads(result["files"]["report_manifest.json"].decode("utf-8"))
    assert set(record["csv_schemas"]) == set(experiment_csv.CSV_FILENAMES)
    assert set(record["outputs"]) == set(experiment_csv.CSV_FILENAMES) | {
        "projection.json",
        "report.html",
    }
    assert record["report_kind"] == "experiment"


def test_p6_t3_23_in_package_recompute_and_tamper_detection(tmp_path):
    """包内复算通过；篡改产出或**源输入**字节都能定位失败（输入环不得自证）。"""
    base = write_runs_batch(tmp_path / "runs_base", batch_id="batch_base")
    arm = write_runs_batch(tmp_path / "runs_arm", batch_id="batch_arm")
    request = _request(tmp_path, ("base", base), ("arm", arm))
    request = generate.ExperimentRequest(
        **{**request.__dict__, "publish_run": True}
    )
    result = generate.generate_experiment(request)
    record = json.loads(result["files"]["report_manifest.json"].decode("utf-8"))
    outputs = {
        name: sha256(payload).hexdigest()
        for name, payload in result["files"].items()
        if name != "report_manifest.json"
    }
    chain = manifest.recompute_chain(
        manifest=record,
        projection=result["projection"],
        outputs=outputs,
        inputs=record["inputs"],
    )
    assert chain["report_status"] == "available", chain
    tampered = dict(outputs)
    tampered["execution_attribution.csv"] = "0" * 64
    broken = manifest.recompute_chain(
        manifest=record,
        projection=result["projection"],
        outputs=tampered,
        inputs=record["inputs"],
    )
    assert broken["report_status"] == "invalid"
    assert broken["degradation_reason_code"] == "output_sha_mismatch"
    assert "outputs.execution_attribution.csv" in broken["detail"]
    # 输入环：按源文件字节重算，篡改源文件后必须 input_drift（B-01）
    from scripts.v2_p6 import run_t3_acceptance as driver

    report_path = Path(result["report_path"])
    roots = {"arm_0": base, "arm_1": arm}
    verified = driver.recompute_published(report_path, roots)
    assert verified["passed"] is True
    assert verified["inputs_verified_from_source"] == len(record["inputs"])
    tampered_source = base / "batch_meta.json"
    tampered_source.write_bytes(tampered_source.read_bytes() + b"\n")
    drift = driver.recompute_published(report_path, roots)
    assert drift["passed"] is False
    assert drift["degradation_reason_code"] == "input_drift"
    assert drift["drift"]
    # 未知输入根：不得静默复用 manifest 自述行（B-R2-02）
    partial_roots = {"arm_0": base}
    uncovered = driver.recompute_published(report_path, partial_roots)
    assert uncovered["passed"] is False
    assert "arm_1" in uncovered["unverified_roots"]
