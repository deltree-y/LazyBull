# -*- coding: utf-8 -*-
"""P6 T0.5 外部请求、历史不覆盖与 validation 隔离（P6-T05-05/07/08）。

覆盖：参考 HTML 零外部引用与 CSP、请求清单门；参考生成双跑逐字节一致与已有目录拒绝；
validation 写盘不回写源语料/原报告、只落 validation 根、JSON 不覆盖且拒绝 NaN；
测量入口拒绝写入 validation 根之外。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p6 import (  # noqa: E402
    build_corpus,
    measure_generation,
    measure_report,
    reference_generator,
    run_t05_acceptance,
)
from src.lazybull.v2.reporting import perf_gate, validation  # noqa: E402

GENERATED_AT = "20260101T000000Z"


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): validation.sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    build_corpus.build(corpus, profile="standard")
    return corpus


def test_p6_t05_05_reference_html_is_offline_with_csp(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    out = tmp_path / "reference"
    reference_generator.generate(corpus, out, generated_at=GENERATED_AT)

    html = (out / "report.html").read_text(encoding="utf-8")
    assert perf_gate.scan_external_references(html) == []
    assert "default-src 'none'" in html
    assert "http://" not in html.replace("http://www.w3.org/2000/svg", "")

    assert perf_gate.scan_request_urls(["file:///tmp/report.html", "data:text/html,<p>x</p>"]) == []
    offenders = perf_gate.scan_request_urls(
        ["file:///tmp/report.html", "https://cdn.example/x.js", "ws://example/socket"]
    )
    assert offenders == ["https://cdn.example/x.js", "ws://example/socket"]


def test_p6_t05_07_reference_deterministic_and_no_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _corpus(tmp_path)
    first = reference_generator.generate(corpus, tmp_path / "first", generated_at=GENERATED_AT)
    second = reference_generator.generate(corpus, tmp_path / "second", generated_at=GENERATED_AT)

    assert first["record_sha256"] == second["record_sha256"]
    assert (tmp_path / "first" / "report.html").read_bytes() == (
        tmp_path / "second" / "report.html"
    ).read_bytes()

    with pytest.raises(validation.ValidationError):
        reference_generator.generate(corpus, tmp_path / "first", generated_at=GENERATED_AT)

    with pytest.raises(measure_report.MeasureSetupError):
        measure_report._assert_new_validation_out(tmp_path / "outside-open")
    with pytest.raises(measure_generation.GenerationSetupError):
        measure_generation._assert_new_validation_out(tmp_path / "outside-generation")

    # R1-3：validation 根形态守卫——evidence_class 白名单（validations 根自身亦拒绝）
    reports_root = tmp_path / "reports"
    monkeypatch.setattr(measure_report, "_REPORTS_ROOT", reports_root)
    monkeypatch.setattr(measure_generation, "_REPORTS_ROOT", reports_root)
    with pytest.raises(measure_report.MeasureSetupError):
        measure_report._assert_new_validation_out(reports_root / "validations")
    with pytest.raises(measure_report.MeasureSetupError):
        measure_report._assert_new_validation_out(
            reports_root / "validations" / "bogus_class" / "probe_x"
        )
    with pytest.raises(measure_generation.GenerationSetupError):
        measure_generation._assert_new_validation_out(
            reports_root / "validations" / "bogus_class" / "probe_x"
        )
    positive = measure_report._assert_new_validation_out(
        reports_root / "validations" / "research" / "p6_t05_probe_nonexistent"
    )
    assert positive.name == "p6_t05_probe_nonexistent"


def test_p6_t05_08_validation_is_isolated_from_source(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    before = _tree_hashes(corpus)
    out = tmp_path / "reference"
    reference_generator.generate(corpus, out, generated_at=GENERATED_AT)

    html_path = out / "report.html"
    html_sha = validation.sha256_file(html_path)
    html = html_path.read_text(encoding="utf-8")
    assert perf_gate.judge_size(html_path.stat().st_size)["passed"] is True  # 评审 R1-7
    assert perf_gate.scan_external_references(html) == []
    assert validation.sha256_file(html_path) == html_sha
    assert _tree_hashes(corpus) == before
    assert all(path == out or out in path.parents for path in out.rglob("*"))

    reports_root = tmp_path / "reports"
    existing = reports_root / "validations" / "research" / "p6_t05_existing"
    existing.mkdir(parents=True)
    with pytest.raises(validation.ValidationError):
        validation.resolve_validation_dir(reports_root, "research", "p6_t05_existing")

    fresh = validation.resolve_validation_dir(reports_root, "research", "p6_t05_fresh")
    validation.ensure_new_dir(fresh)
    validation.write_json_new(fresh / "result.json", {"ok": True})
    with pytest.raises(validation.ValidationError):
        validation.write_json_new(fresh / "result.json", {"ok": False})
    with pytest.raises(ValueError):
        validation.write_json_new(fresh / "nan.json", {"value": float("nan")})


def test_p6_t05_extra_default_validation_id_valid() -> None:
    """R1-1/R2-T05-05 回归守卫：默认复跑命令的 id 生成即合法（无需显式传参）。"""
    identifier = run_t05_acceptance._default_validation_id()
    assert validation.validate_validation_id(identifier) == identifier
    assert identifier.startswith("p6_t05_acceptance_")
    assert identifier[len("p6_t05_acceptance_") :].isdigit()


def _write_probe_html(tmp_path: Path, body: str) -> Path:
    html_path = tmp_path / "probe.html"
    html_path.write_text(
        "<!DOCTYPE html><html><head><meta charset='utf-8'></head>"
        f"<body>{body}</body></html>",
        encoding="utf-8",
    )
    return html_path


def test_p6_t05_extra_local_subresource_fails_open_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2-T05-01 负例：本地 file 子资源（CSS）必须被资源门判违规。"""
    pytest.importorskip("playwright")
    reports_root = tmp_path / "reports"
    monkeypatch.setattr(measure_report, "_REPORTS_ROOT", reports_root)
    html_path = _write_probe_html(tmp_path, "<link rel='stylesheet' href='probe.css'>")
    (tmp_path / "probe.css").write_text("body { color: #111; }\n", encoding="utf-8")
    payload = measure_report.measure(
        html_path, reports_root / "validations" / "research" / "p6_t05_extra_css_probe"
    )
    assert payload["verdict"] == "fail"
    request_gate = payload["gates"]["requests"]
    assert request_gate["passed"] is False
    assert request_gate["websocket_count"] == 0
    assert any(url.endswith("probe.css") for url in request_gate["offenders"])


def test_p6_t05_extra_websocket_fails_open_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2-T05-01 负例：运行期 WebSocket 必须被捕获并使资源门失败（静态扫描盲区）。"""
    pytest.importorskip("playwright")
    reports_root = tmp_path / "reports"
    monkeypatch.setattr(measure_report, "_REPORTS_ROOT", reports_root)
    html_path = _write_probe_html(
        tmp_path, "<script>new WebSocket('ws://127.0.0.1:9/probe');</script>"
    )
    payload = measure_report.measure(
        html_path, reports_root / "validations" / "research" / "p6_t05_extra_ws_probe"
    )
    assert payload["verdict"] == "fail"
    request_gate = payload["gates"]["requests"]
    assert request_gate["websocket_count"] >= 1
    assert request_gate["passed"] is False
    assert any("ws://127.0.0.1:9/probe" in url for url in request_gate["offenders"])
    # R2-T05-02 续：写域三方证据——user-data-dir 位于 validation、profile 有落盘、
    # Playwright 可控临时目录重定向后 OS TEMP 零新增
    domain = payload["gates"]["browser_write_domain"]
    assert domain["passed"] is True
    assert domain["profile_files_total"] > 0
    assert domain["os_temp_new_entries"] == []


def test_p6_t05_extra_popup_request_fails_open_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2-T05-01 续 负例：popup 发起的动态 HTTP 请求必须在 context 级被捕获并使门失败。"""
    pytest.importorskip("playwright")
    reports_root = tmp_path / "reports"
    monkeypatch.setattr(measure_report, "_REPORTS_ROOT", reports_root)
    html_path = _write_probe_html(
        tmp_path, "<script>window.open('http://127.0.0.1:8765/probe');</script>"
    )
    payload = measure_report.measure(
        html_path, reports_root / "validations" / "research" / "p6_t05_extra_popup_probe"
    )
    assert payload["verdict"] == "fail"
    request_gate = payload["gates"]["requests"]
    assert request_gate["passed"] is False
    assert any("http://127.0.0.1:8765/probe" in url for url in request_gate["offenders"])
    # 原始记录可复核：context 级观测必须在请求清单中看到 popup 请求（reviewer2 复现场景）
    observations = payload["observations"]["cold"] + payload["observations"]["hot"]
    assert any(
        any("127.0.0.1:8765/probe" in entry["url"] for entry in record["requests"])
        for record in observations
    )
