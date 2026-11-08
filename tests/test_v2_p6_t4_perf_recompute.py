# -*- coding: utf-8 -*-
"""P6 T4 复算与性能门（设计 §7 / §8.2）。

同输入且同 ``generated_at`` 的 HTML/JSON/CSV 逐字节一致；manifest 自封存与包内复算通过；
体积门 ≤3,000,000 字节、生成门 <10,000ms（阈值负例走注入合成记录，不制造真实慢页面）。
"""

from __future__ import annotations

from hashlib import sha256

import pytest

from src.lazybull.v2.reporting import generate, manifest, perf_gate, publish
from tests import p6_t4_env


def test_p6_t4_payload_paths_detects_kind_absolute_and_drive_letter() -> None:
    """载荷巡检须抓住 ``kind:D:\\…`` 与字符串中任意位置的盘符（T4-R3-R3B-01）。"""
    from scripts.v2_p6 import run_t4_acceptance as acc

    assert acc._payload_paths({"x": "runs:temp/a/b"})["passed"] is True
    assert acc._payload_paths({"x": "runs:D:\\abs\\x"})["count"] == 1
    assert acc._payload_paths({"x": "见 C:\\Users\\BBMM\\python.exe"})["count"] == 1
    assert acc._payload_paths({"x": "file:///D:/x"})["count"] == 1
    assert acc._payload_paths({"x": "12:30 说明"})["passed"] is True
    assert acc._payload_paths({"environment": {"p": "C:\\Users"}})["passed"] is True


def test_p6_t4_perf_and_recompute(tmp_path_factory: pytest.TempPathFactory) -> None:
    """字节确定性 + 链复算 + 体积/生成门（阈值负例只注入记录）。"""
    root = tmp_path_factory.mktemp("p6_t4_recompute")
    corpus = root / "corpus"
    p6_t4_env.build_corpus(corpus)
    reports = root / "reports"
    first = generate.generate_trend(
        p6_t4_env.research_request(reports, corpus, attempt="recompute_a")
    )
    second = generate.generate_trend(
        p6_t4_env.research_request(reports, corpus, attempt="recompute_b")
    )
    assert first["files"] == second["files"]
    assert first["report_id"] == second["report_id"]
    assert set(first["files"]) == set(publish.TREND_FILENAMES)
    assert len(first["files"]) == 5
    record = first["manifest"]
    assert manifest.verify_manifest(record)["report_status"] == "available"
    chain = manifest.recompute_chain(
        manifest=record,
        projection=first["projection"],
        outputs={
            name: sha256(payload).hexdigest()
            for name, payload in first["files"].items()
            if name != "report_manifest.json"
        },
        inputs=record["inputs"],
    )
    assert chain["report_status"] == "available"
    # 体积门：整页字节 ≤3,000,000（真实）；3,000,001 注入记录必须失败
    page = first["files"]["report.html"]
    assert perf_gate.judge_size(len(page))["passed"]
    assert not perf_gate.evaluate_metric({"kind": "size", "value": 3_000_001})["passed"]
    # 生成门：本进程实测耗时 <10,000ms；注入 10,001ms 记录必须失败
    assert perf_gate.judge_generation(float(first["elapsed_ms"]))["passed"]
    assert not perf_gate.evaluate_metric({"kind": "generation", "value": 10_001})["passed"]
