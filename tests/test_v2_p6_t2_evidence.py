# -*- coding: utf-8 -*-
"""P6 T2 证据横幅与 research 三端身份卡（证据与三端 2 项，设计 §2.2 / §4.1）。

- 三类证据横幅为**固定文案**（production 蓝 / research 灰 / calibration 琥珀整页边框），
  切换前记录形态必须在横幅区可见；
- 三端卡**只在 research 面**出现，且实盘端固定 `not_available`（不伪造实盘读数）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import p6_report  # noqa: E402
from src.lazybull.v2.reporting import evidence, publish  # noqa: E402
from src.lazybull.v2.reporting.generate import (  # noqa: E402
    IMPLEMENTED_KINDS,
    DashboardRequest,
    generate_dashboard,
)
from tests.p6_t1_env import build_paper_root, build_projection  # noqa: E402
from tests.p6_t2_env import (  # noqa: E402
    build_view,
    case_paper_root,
    fact_value,
    render_projection,
    section_slice,
)


def _projection(tmp_path: Path, name: str, **kwargs: Any) -> dict[str, Any]:
    """合成纸面根 → 投影（``kwargs`` 同时喂给根构造与投影装配）。"""
    root_kwargs = {
        key: kwargs.pop(key)
        for key in ("drop_record_keys", "record_extra")
        if key in kwargs
    }
    built = build_paper_root(tmp_path / name, **root_kwargs)
    projection, _snapshot, _audit, _info = build_projection(
        built["root"], tmp_path / f"{name}_staging", attempt_id=f"attempt_{name}", **kwargs
    )
    return projection


def test_p6_t2_03_evidence_banner_fixed_and_pre_switch_visible(tmp_path: Path) -> None:
    """三类横幅固定文案 + 整页边框类；切换前形态说明必须可见（含原因码）。"""
    for evidence_class in ("production", "research", "calibration_only"):
        projection = _projection(
            tmp_path,
            f"banner_{evidence_class}",
            evidence_class=evidence_class,
        )
        html = render_projection(projection)
        banner = evidence.EVIDENCE_BANNERS[evidence_class]
        assert banner["text"] in html
        assert banner["class"] in html
        assert 'id="evidence-banner"' in html
        if evidence_class == "calibration_only":
            assert "body.banner--calibration { border: 6px solid var(--calibration); }" in html
        # 正常记录形态不得出现切换前说明
        assert 'id="generation-shape-note"' not in html
        assert evidence.PRE_SWITCH_NOTE not in html

    pre_switch = _projection(
        tmp_path,
        "banner_pre_switch",
        evidence_class="production",
        drop_record_keys=("run_kind", "engine"),
    )
    assert pre_switch["record"]["degradation_reason_code"] == evidence.PRE_SWITCH_REASON
    html = render_projection(pre_switch)
    assert 'id="generation-shape-note"' in html
    assert evidence.PRE_SWITCH_NOTE in html
    assert "切换前记录形态" in html
    # 账户退化为 degraded，指令区不得伪造未来
    assert pre_switch["account"]["report_status"] == "degraded"
    assert pre_switch["instructions"]["report_status"] == "not_available"
    assert pre_switch["instructions"]["degradation_reason_code"] == evidence.PRE_SWITCH_REASON

    with pytest.raises(evidence.EvidenceIsolationError):
        evidence.assert_input_evidence("production", ["production", "calibration_only"])


def test_p6_t2_04_three_ends_only_research_without_live_fabrication(tmp_path: Path) -> None:
    """三端卡只在 research；实盘端固定不可用；非 research 面完全不渲染该卡。"""
    for evidence_class in ("production", "calibration_only"):
        projection = _projection(tmp_path, f"ends_{evidence_class}", evidence_class=evidence_class)
        html = render_projection(projection)
        assert 'id="sec-three-ends"' not in html
        assert "三端身份卡" not in html
        view = build_view(projection)
        assert view["three_ends"]["visible"] is False
        assert view["three_ends"]["rows"] == []

    research = _projection(tmp_path, "ends_research", evidence_class="research")
    html = render_projection(research)
    section = section_slice(html, "sec-three-ends")
    for end in ("回测端", "纸面回放端", "纸面端", "实盘端"):
        assert end in section
    assert "P8 实盘端未建" in section
    view = build_view(research)
    rows = {row["end"]: row for row in view["three_ends"]["rows"]}
    assert set(rows) == {"回测端", "纸面回放端", "纸面端", "实盘端"}
    assert rows["实盘端"]["status"]["word"] == "not_available"
    assert rows["回测端"]["status"]["word"] == "not_available"
    assert rows["纸面端"]["status"]["word"] == "available"
    assert "run_kind=t1" in rows["纸面端"]["identity"]

    # 额外三端输入一律拒绝：它不进入封存包，包内无法复算（T2-R3-03）
    paper, quality = case_paper_root(tmp_path, "standard")
    injected = DashboardRequest(
        paper_root=paper,
        reports_root=tmp_path / "ends_injected",
        evidence_class="research",
        generated_at="2026-06-04T00:00:00Z",
        attempt_id="ends_inject",
        input_evidence_classes=("research",),
        quality_root=quality,
        three_ends={"backtest": {"report_status": "available"}},
    )
    with pytest.raises(publish.PublishError):
        generate_dashboard(injected)
    assert not (tmp_path / "ends_injected").exists()


@pytest.mark.parametrize(
    ("evidence_class", "inputs"),
    [
        ("production", ["production", "calibration_only"]),
        ("research", ["production", "calibration_only"]),
        ("calibration_only", ["calibration_only", "production"]),
        ("calibration_only", ["production"]),
    ],
)
def test_p6_t2_14_evidence_isolation_matrix(
    evidence_class: str, inputs: list[str]
) -> None:
    """证据域隔离矩阵：四类跨类组合全部 fail-closed（函数级）。"""
    with pytest.raises(evidence.EvidenceIsolationError):
        evidence.assert_input_evidence(evidence_class, inputs)
    # 本类内组合合法
    allowed = {
        "production": ["production"],
        "research": ["production", "research"],
        "calibration_only": ["calibration_only"],
    }[evidence_class]
    evidence.assert_input_evidence(evidence_class, allowed)


def test_p6_t2_15_cli_isolation_nonzero_and_no_artifact(tmp_path: Path) -> None:
    """CLI 级：混合输入非零退出，且**不落任何产物**（不只看退出码）。"""
    paper, quality = case_paper_root(tmp_path, "standard")
    for index, inputs in enumerate(
        ("production,calibration_only", "research,calibration_only", "")
    ):
        reports_root = tmp_path / f"reports_{index}"
        code = p6_report.main(
            [
                "generate",
                "--paper-root",
                str(paper),
                "--quality-root",
                str(quality),
                "--reports-root",
                str(reports_root),
                "--evidence-class",
                "production" if index == 0 else "research",
                "--input-evidence-classes",
                inputs,
                "--attempt-id",
                f"iso_probe_{index}",
            ]
        )
        assert code == p6_report.EXIT_CODES["policy_fail"]
        assert not reports_root.exists(), f"混合输入不得落产物：{reports_root}"

    # 合法组合可发布（对照组，证明上面的“零产物”不是路径写错）
    ok_root = tmp_path / "reports_ok"
    code = p6_report.main(
        [
            "generate",
            "--paper-root",
            str(paper),
            "--quality-root",
            str(quality),
            "--reports-root",
            str(ok_root),
            "--evidence-class",
            "research",
            "--input-evidence-classes",
            "production",
            "--attempt-id",
            "iso_probe_ok",
        ]
    )
    assert code == p6_report.EXIT_CODES["pass"]
    assert ok_root.is_dir()


def test_p6_t2_16_generation_identity_and_degradation_parity(tmp_path: Path) -> None:
    """权威代必须显示 ``run_kind@trade_date``；页面降级清单与 manifest 同源同序。"""
    from src.lazybull.v2.reporting.generate import DashboardRequest, generate_dashboard

    paper, quality = case_paper_root(tmp_path, "standard")
    result = generate_dashboard(
        DashboardRequest(
            paper_root=paper,
            quality_root=quality,
            reports_root=tmp_path / "reports",
            evidence_class="production",
            generated_at="20261106T080000Z",
            attempt_id="identity_probe",
            input_evidence_classes=("production",),
            git_record={"git_commit": "<synthetic>", "git_dirty": False},
        )
    )
    html = result["files"]["report.html"].decode("utf-8")
    account = section_slice(html, "sec-account")
    assert fact_value(account, "权威代") == "t1@20260603"
    assert fact_value(account, "记录形态").startswith("available")

    projection = result["projection"]
    degradations = projection["degradations"]
    assert result["manifest"]["degradations"] == degradations
    section = section_slice(html, "sec-degradations")
    for entry in degradations:
        assert entry["field_path"] in section
        assert entry["reason_code"] in section

    # production 面：全文不得出现实盘端读数
    assert "P8" not in html and "实盘端" not in html
    assert "sec-three-ends" not in html


def test_p6_t2_22_cli_snapshot_error_maps_to_registered_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI 级：快照协议失败 → 退出码 3（``stage=stable_snapshot``），不落产物。"""
    from src.lazybull.v2.reporting import generate as generate_module
    from src.lazybull.v2.reporting.snapshot import UnstableSnapshotError

    paper, quality = case_paper_root(tmp_path, "standard")
    reports_root = tmp_path / "reports_snapshot"

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise UnstableSnapshotError("合成：读取期持续漂移")

    monkeypatch.setattr(generate_module, "capture_stable_snapshot", _boom)
    code = p6_report.main(
        [
            "generate",
            "--paper-root",
            str(paper),
            "--quality-root",
            str(quality),
            "--reports-root",
            str(reports_root),
            "--evidence-class",
            "research",
            "--attempt-id",
            "snapshot_probe",
        ]
    )
    assert code == p6_report.EXIT_CODES["policy_fail"]
    assert not reports_root.exists()

    # 参数形态非法归 2（SetupError），不是 3；设备名负例用**真设备名**（`LPT1` 会被
    # 正则的大写分支先拦，无区分力，仅保留作正则分支参考）
    for bad_attempt in ("..", "", "con", "aux", "prn", "nul", "com1", "lpt9", "LPT1"):
        code = p6_report.main(
            [
                "generate",
                "--paper-root",
                str(paper),
                "--reports-root",
                str(reports_root),
                "--evidence-class",
                "research",
                "--attempt-id",
                bad_attempt,
            ]
        )
        assert code == p6_report.EXIT_CODES["setup_error"], bad_attempt
        assert not reports_root.exists()

    # 载荷级：快照协议失败必须带 stage=stable_snapshot（不能只在 docstring 里声称）
    args = p6_report.build_parser().parse_args(
        [
            "generate",
            "--paper-root",
            str(paper),
            "--quality-root",
            str(quality),
            "--reports-root",
            str(tmp_path / "reports_stage"),
            "--evidence-class",
            "research",
            "--attempt-id",
            "stage_probe",
        ]
    )
    code, payload = p6_report.run_generate(args)
    assert code == p6_report.EXIT_CODES["policy_fail"]
    assert payload["stage"] == "stable_snapshot"
    assert payload["attempt_id"] == "stage_probe"
    assert payload["detail"].startswith("UnstableSnapshotError")
    assert not (tmp_path / "reports_stage").exists()

    # 文件系统失败与运行期守卫拦截同样映射到登记退出码（不被裸 traceback 逃逸）
    from src.lazybull.v2.reporting import readonly_audit

    for exc_type, expected_stage in (
        (OSError, "file_system"),
        (readonly_audit.ReadonlyViolationError, "readonly_guard"),
    ):
        def _raise(*args: Any, _exc: type = exc_type, **kwargs: Any) -> Any:
            raise _exc("合成：结构性兜底探针")

        monkeypatch.setattr(generate_module, "generate_dashboard", _raise)
        probe_args = p6_report.build_parser().parse_args(
            [
                "generate",
                "--paper-root",
                str(paper),
                "--reports-root",
                str(tmp_path / f"reports_{expected_stage}"),
                "--evidence-class",
                "research",
                "--attempt-id",
                f"probe_{expected_stage}",
            ]
        )
        code, payload = p6_report.run_generate(probe_args)
        assert code == p6_report.EXIT_CODES["policy_fail"]
        assert payload["stage"] == expected_stage
        assert not (tmp_path / f"reports_{expected_stage}").exists()


def _request(
    paper: Path, quality: Path, reports_root: Path, *, attempt: str, **kwargs: Any
) -> Any:
    """构造一次 research 请求（默认测试口径，可覆盖任意字段）。"""
    payload = {
        "paper_root": paper,
        "reports_root": reports_root,
        "evidence_class": "research",
        "generated_at": "20260604T000000Z",
        "attempt_id": attempt,
        "input_evidence_classes": ("research",),
        "quality_root": quality,
    }
    payload.update(kwargs)
    return DashboardRequest(**payload)


def test_p6_t2_31_existing_staging_refused_and_preserved(tmp_path: Path) -> None:
    """同 ID 的既有 staging 必须**拒绝且原样保留**，不得被异常清理删除（T2-R3-01）。"""
    from hashlib import sha256

    paper, quality = case_paper_root(tmp_path, "standard")
    reports = tmp_path / "reports_reuse"
    crash_site = reports / ".staging" / "reuse_probe" / "r1"
    crash_site.mkdir(parents=True)
    evidence_file = crash_site / "evidence.txt"
    evidence_file.write_text("崩溃现场", encoding="utf-8")
    before = sha256(evidence_file.read_bytes()).hexdigest()

    for attempt in ("reuse_probe", "reuse_probe"):
        with pytest.raises(publish.PublishError):
            generate_dashboard(_request(paper, quality, reports, attempt=attempt))
        assert evidence_file.is_file(), "既有崩溃现场被删除"
        assert sha256(evidence_file.read_bytes()).hexdigest() == before

    # 正常路径不受影响：新建 attempt 仍能与既有现场并存并在成功后清理
    ok = generate_dashboard(_request(paper, quality, reports, attempt="fresh_probe"))
    assert ok["cleanup"]["staging_discard_ok"] is True
    assert not (reports / ".staging" / "fresh_probe").exists()
    assert evidence_file.is_file()


def test_p6_t2_32_unimplemented_kind_rejected_before_capture(tmp_path: Path) -> None:
    """未接入/错入口的类型必须拒绝且零产物（T2-R3-04；T3/T4 已分别接入实验/趋势）。

    三条独立判据：
    1. ``experiment``/``trend`` 已接入，但**不得**经驾驶舱入口生成（反向冒充 fail-closed）；
    2. 未登记类型（如 ``unknown_kind``）在 CLI 参数层即被拒绝（argparse 闭集，退出码 2）；
    3. ``trend`` 已进入 ``IMPLEMENTED_KINDS``（T4 接入事实锁，防止回退成“未实现”）。
    """
    paper, quality = case_paper_root(tmp_path, "standard")

    for kind in ("experiment", "trend"):
        reports = tmp_path / f"reports_kind_{kind}"
        with pytest.raises(publish.PublishError):
            generate_dashboard(
                _request(paper, quality, reports, attempt=f"kind_{kind}", report_kind=kind)
            )
        assert not reports.exists(), kind

    cli_reports = tmp_path / "reports_cli_unknown"
    with pytest.raises(SystemExit) as excinfo:
        p6_report.main(
            [
                "generate",
                "--paper-root",
                str(paper),
                "--quality-root",
                str(quality),
                "--reports-root",
                str(cli_reports),
                "--evidence-class",
                "research",
                "--kind",
                "unknown_kind",
            ]
        )
    assert excinfo.value.code == 2
    assert not cli_reports.exists()
    assert "trend" in IMPLEMENTED_KINDS


def test_p6_t2_33_same_attempt_concurrency_never_deletes_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同 ID 并发：持有方成功、冲突方在进入采集前被拒，互不删除、不产生伪缺失（T2-R3-01-R1）。"""
    import threading

    from src.lazybull.v2.reporting import cockpit

    paper, quality = case_paper_root(tmp_path, "standard")
    reports = tmp_path / "reports_concurrent"
    inside = threading.Event()
    release = threading.Event()
    results: dict[str, Any] = {}
    original = cockpit.capture_paper_inputs

    def slow_capture(snapshot: Any) -> dict[str, Any]:
        inside.set()
        assert release.wait(20), "持有方未被放行"
        return original(snapshot)

    monkeypatch.setattr(cockpit, "capture_paper_inputs", slow_capture)

    def run_owner() -> None:
        results["a"] = generate_dashboard(
            _request(paper, quality, reports, attempt="shared_probe", publish_run=False)
        )

    owner = threading.Thread(target=run_owner, daemon=True)
    owner.start()
    assert inside.wait(20), "持有方未进入采集"

    # 冲突方：同 attempt，必须在采集前被拒（不得共用/删除持有方的快照）
    with pytest.raises(publish.PublishError):
        generate_dashboard(_request(paper, quality, reports, attempt="shared_probe"))

    release.set()
    owner.join(30)
    assert not owner.is_alive()
    assert results["a"]["report_id"].startswith("p6_dashboard_")
    # 持有方拿到的是**真实权威**（不是被对手删掉快照后的 authority_missing）
    assert results["a"]["projection"]["authority"]["report_status"] == "available"
    assert not (reports / ".staging" / "shared_probe").exists()


def test_p6_t2_35_release_once_does_not_affect_new_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """成功路径只释放一次：旧调用释放后新调用取得目录，旧调用收尾不得再动它（T2-R3-01-R2）。"""
    import threading

    from src.lazybull.v2.reporting import cockpit
    from src.lazybull.v2.reporting import generate as generate_module
    from src.lazybull.v2.reporting import publish as publish_module

    paper, quality = case_paper_root(tmp_path, "standard")
    reports = tmp_path / "reports_release"
    old_released = threading.Event()
    new_captured = threading.Event()
    old_done = threading.Event()
    results: dict[str, Any] = {}
    counts = {"discard": 0}

    original_capture = cockpit.capture_paper_inputs
    original_discard = publish_module.discard_attempt
    original_compose = generate_module._compose

    state = {"new_owner_active": False, "owner_thread": None}

    def capture(snapshot: Any) -> dict[str, Any]:
        payload = original_capture(snapshot)
        if state["new_owner_active"]:
            new_captured.set()
        return payload

    def discard(reports_root: Any, attempt_id: str) -> None:
        original_discard(reports_root, attempt_id)
        if threading.current_thread() is state["owner_thread"] and attempt_id == "shared_release":
            counts["discard"] += 1
            old_released.set()
            assert new_captured.wait(20), "新所有者未完成采集"

    def compose(snapshot: Any, request: Any, info: Any) -> dict[str, Any]:
        if state["new_owner_active"]:
            assert old_done.wait(20), "旧调用未结束"
        return original_compose(snapshot, request, info)

    monkeypatch.setattr(cockpit, "capture_paper_inputs", capture)
    monkeypatch.setattr(publish_module, "discard_attempt", discard)
    monkeypatch.setattr(generate_module, "_compose", compose)

    def run_old() -> None:
        state["owner_thread"] = threading.current_thread()
        try:
            results["a"] = generate_dashboard(
                _request(paper, quality, reports, attempt="shared_release", publish_run=False)
            )
        finally:
            old_done.set()

    old = threading.Thread(target=run_old, name="old-owner", daemon=True)
    old.start()
    assert old_released.wait(20), "旧调用未释放 attempt"

    state["new_owner_active"] = True
    results["b"] = generate_dashboard(
        _request(paper, quality, reports, attempt="shared_release", publish_run=True)
    )
    old.join(30)
    assert not old.is_alive()
    assert counts["discard"] == 1, "成功路径只能释放一次"
    assert results["b"]["projection"]["authority"]["report_status"] == "available"
    assert results["b"]["report_path"] is not None
    assert Path(str(results["b"]["report_path"])).is_dir()


def test_p6_t2_36_preexisting_empty_reports_root_is_kept(tmp_path: Path) -> None:
    """调用前已存在的空 reports_root 不得被失败清理删除（T2-R3-N1）。"""
    _paper, quality = case_paper_root(tmp_path, "standard")
    reports = tmp_path / "reports_preexisting"
    reports.mkdir()
    missing_paper = tmp_path / "missing_paper"

    with pytest.raises(Exception):
        generate_dashboard(_request(missing_paper, quality, reports, attempt="preexisting_probe"))

    assert reports.is_dir(), "调用前已存在的空报告根被删除"
