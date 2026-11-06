# -*- coding: utf-8 -*-
"""P6 T1 整改补测（评审 R1-01/02/04/08/09/14/16、R2-11）：真实形态与畸形输入负例。

覆盖：跳空 NAV 合法、nav 独有日 invalid、权威成交投影与宿主**逐字段 parity**、持仓缺字段
不抛异常、权威不可读时区键恒定、坏 JSON 逐区降级、空 portfolio_values、空未来指令帧、
报告包重型依赖 AST 守卫、staging 受信与副本复核。
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.common.cost import CostModel  # noqa: E402
from src.lazybull.v2.hosts.paper.host import PaperHost  # noqa: E402
from src.lazybull.v2.reporting import authority  # noqa: E402
from src.lazybull.v2.reporting.cockpit import SECTION_KEYS  # noqa: E402
from src.lazybull.v2.reporting.snapshot import (  # noqa: E402
    SnapshotError,
    UntrustedRootError,
    capture_stable_snapshot,
)
from tests.p6_t1_env import (  # noqa: E402
    TOTAL_VALUE,
    TRADE_DATE,
    build_engine_blob,
    build_paper_root,
    build_projection,
    instruction_rows,
    nav_rows,
    portfolio_values,
)


def _project(tmp_path: Path, **kwargs: Any) -> dict:
    built = build_paper_root(tmp_path / "paper", **kwargs)
    projection, _snapshot, _audit, _info = build_projection(
        built["root"], tmp_path / "staging"
    )
    return projection


def test_p6_t1_sparse_nav_is_legal(tmp_path: Path) -> None:
    """R1-01：真实跳空语义（仅成交日有 NAV 行）⇒ available 且 broken_links 为空。"""
    projection = _project(tmp_path)
    nav = projection["nav"]
    assert nav["report_status"] == "available"
    assert nav["broken_links"] == []
    assert nav["verified_rows"] == [TRADE_DATE]
    assert nav["nav_date"] == TRADE_DATE
    assert nav["nav_only_days"] == []
    assert projection["degradations"] == []
    assert projection["config"]["annualization"]["report_status"] == "available"


def test_p6_t1_nav_extra_day_is_invalid(tmp_path: Path) -> None:
    """R1-04：nav 含权威不存在的业务日 ⇒ invalid（两载体业务日集合冲突）。"""
    extra = nav_rows() + [
        {
            "trade_date": "20260610",
            "cash": 1.0,
            "position_value": 1.0,
            "total_value": 2.0,
            "nav": 1.0,
        }
    ]
    projection = _project(tmp_path, nav=extra)
    nav = projection["nav"]
    assert nav["report_status"] == "invalid"
    assert nav["degradation_reason_code"] == "authority_mismatch"
    assert nav["nav_only_days"] == ["20260610"]
    assert "20260610" in nav["detail"]


def test_p6_t1_authority_trade_projection_parity() -> None:
    """R1-02：报告包内的成交投影与宿主 ``_project_trade_row`` **逐字段**一致。"""
    cost_model = CostModel(
        commission_rate=0.00025, min_commission=5.0, stamp_tax=0.001, slippage=0.0005
    )
    stub = SimpleNamespace(
        cost_model=cost_model,
        _normalize_date_text=lambda value: authority.normalize_date(value),
        _trade_reason=PaperHost._trade_reason,  # noqa: SLF001
    )
    for record in (
        {
            "date": "20260603",
            "stock": "600000.SH",
            "action": "buy",
            "shares": 100,
            "price": 10.0,
            "amount": 1000.0,
            "cost": 5.5,
        },
        {
            "date": "20260603",
            "stock": "000001.SZ",
            "action": "sell",
            "shares": 200,
            "price": 12.5,
            "amount": 2500.0,
            "cost": 8.0,
            "sell_type": "holding_period",
        },
        {
            "date": "20260603",
            "stock": "000001.SZ",
            "action": "sell",
            "shares": 100,
            "price": 12.0,
            "amount": 1200.0,
            "cost": 6.0,
            "sell_reason": "自定义原因",
        },
    ):
        mirrored = authority.project_authority_trade_record(record)
        for name in ("trade_date", "ts_code", "action", "shares", "price", "amount", "reason"):
            expected = PaperHost._project_trade_row(  # noqa: SLF001
                stub, record, validate_cost=False
            )[name]
            assert mirrored[name] == expected, name
        assert mirrored["trade_date"] == "20260603"
        assert isinstance(mirrored["shares"], int)


def test_p6_t1_position_missing_field_is_invalid_not_raise(tmp_path: Path) -> None:
    """R1-08：权威持仓缺 ``buy_trade_price`` ⇒ 账户区 invalid（不抛宿主异常）。"""
    positions = {
        "600000.SH": {
            "shares": 10000,
            "buy_date": "20260506",
            "signal_date": "20260505",
            "buy_trade_price": 10.0,
            "buy_pnl_price": 10.0,
            "buy_cost_cash": 100050.0,
        },
        "000001.SZ": {"shares": 20000},
    }
    blob = build_engine_blob(positions=positions)
    projection = _project(tmp_path, blob=blob)
    account = projection["account"]
    assert account["report_status"] == "invalid"
    assert account["degradation_reason_code"] == "missing_precondition"
    assert "000001.SZ" in account["detail"] or "持仓" in account["detail"]


def test_p6_t1_authority_unreadable_keeps_all_sections(tmp_path: Path) -> None:
    """R1-09：权威不可读时 projection 键集恒定，各区独立 ``not_available``。"""
    blob = build_engine_blob(portfolio=[])
    projection = _project(tmp_path, blob=blob)
    assert projection["authority"]["report_status"] == "invalid"
    for key in SECTION_KEYS:
        assert key in projection, key
    assert projection["nav"]["report_status"] == "not_available"
    assert projection["pending"]["buys"]["report_status"] == "not_available"
    assert projection["pending"]["sells"]["report_status"] == "not_available"
    paths = {row["field_path"] for row in projection["degradations"]}
    assert {"nav", "account", "pending.buys", "pending.sells"} <= paths


@pytest.mark.parametrize(
    ("relative", "target_path"),
    [
        ("state/account.json", "account"),
        ("runs/rebalance_state.json", "rebalance"),
    ],
)
def test_p6_t1_bad_json_degrades_one_section(
    tmp_path: Path, relative: str, target_path: str
) -> None:
    """R1-16：单文件坏 JSON 只影响该区（其余区仍可用）。"""
    built = build_paper_root(tmp_path / "paper")
    (built["root"] / relative).write_text("{bad json", encoding="utf-8")
    projection, _snapshot, _audit, _info = build_projection(
        built["root"], tmp_path / "staging"
    )
    section = projection[target_path]
    assert section["report_status"] == "invalid"
    assert section["degradation_reason_code"] == "invalid_json"
    assert projection["nav"]["report_status"] == "available"


def test_p6_t1_bad_run_record_degrades_record_only(tmp_path: Path) -> None:
    """R1-16：run record 坏 JSON ⇒ 记录区 invalid，其余区不受影响。"""
    built = build_paper_root(tmp_path / "paper")
    (built["root"] / "runs" / f"t1_{TRADE_DATE}.json").write_text("{bad", encoding="utf-8")
    projection, _snapshot, _audit, _info = build_projection(
        built["root"], tmp_path / "staging"
    )
    assert projection["record"]["report_status"] == "invalid"
    assert projection["record"]["degradation_reason_code"] == "invalid_json"
    assert projection["nav"]["report_status"] == "available"


def test_p6_t1_empty_portfolio_values_invalid(tmp_path: Path) -> None:
    """R1-16：权威 portfolio_values 为空 ⇒ authority invalid/authority_mismatch。"""
    blob = build_engine_blob(portfolio=[])
    projection = _project(tmp_path, blob=blob)
    assert projection["authority"]["report_status"] == "invalid"
    assert projection["authority"]["degradation_reason_code"] == "authority_mismatch"


def test_p6_t1_recovery_with_empty_future_frame(tmp_path: Path) -> None:
    """R1-16：恢复代缺计数 + 未来指令文件为**空帧** ⇒ 仍按零指令语义处置。"""
    projection = _project(
        tmp_path,
        drop_record_keys=("instructions_count", "fills_count"),
        instructions={"20260604": []},
    )
    row = projection["instructions"]
    assert row["report_status"] in ("empty_valid", "degraded")
    assert row["instruction_count"] == 0


def test_p6_t1_instructions_declared_zero_with_history_only(tmp_path: Path) -> None:
    """历史文件存在但声明 0 指令且无未来文件 ⇒ empty_valid（历史不参与未来推断）。"""
    projection = _project(
        tmp_path,
        instructions={"20260601": instruction_rows(source_date="20260530", count=2)},
        record_extra={"instructions_count": 0},
    )
    assert projection["instructions"]["report_status"] == "empty_valid"
    assert projection["instructions"]["history_files"] == 1


def test_p6_t1_reporting_package_no_heavy_dependencies() -> None:
    """R1-14：报告包全部模块禁止 import Registry/Storage/交易日历/Playwright。"""
    forbidden = {
        "ModelRegistry",
        "PanelDataStore",
        "Storage",
        "load_trade_cal",
        "playwright",
        "subprocess",
    }
    offenders: dict[str, set[str]] = {}
    package = Path("src/lazybull/v2/reporting")
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
                names.update(alias.name for alias in node.names)
        hit = names & forbidden
        if hit:
            offenders[path.name] = hit
    assert offenders == {}


def test_p6_t1_snapshot_rejects_untrusted_staging_and_detects_copy_rewrite(
    tmp_path: Path,
) -> None:
    """R2-11：staging 根受信校验 + 副本被改写即抛（副本字节不得静默漂移）。"""
    built = build_paper_root(tmp_path / "paper")
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "staging_link"
    try:
        import os

        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        link = None
    if link is not None:
        with pytest.raises(UntrustedRootError):
            capture_stable_snapshot(
                roots={"paper": built["root"]},
                staging_root=link,
                attempt_ids=["p6_t1_attempt_a"],
                reader=lambda snapshot: None,
                max_retries=0,
            )

    staging = tmp_path / "staging"
    snapshot, _audit = capture_stable_snapshot(
        roots={"paper": built["root"]},
        staging_root=staging,
        attempt_ids=["p6_t1_attempt_b"],
        reader=lambda snap: snap.capture("paper", "state/account.json"),
        max_retries=0,
    )
    assert snapshot.verify_copy() == []
    target = snapshot.copy_path("paper", "state/account.json")
    target.write_text('{"tampered": true}', encoding="utf-8")
    assert snapshot.verify_copy() and "副本字节变化" in snapshot.verify_copy()[0]
    with pytest.raises(SnapshotError):
        snapshot.read_bytes("paper", "state/account.json")


def test_p6_t1_nav_sparse_matches_authority_total(tmp_path: Path) -> None:
    """稀疏 NAV 的单行三原值仍必须等于权威（跳空不等于免检）。"""
    projection = _project(tmp_path)
    nav = projection["nav"]
    assert nav["nav_values"][TRADE_DATE]["total_value"] == pytest.approx(TOTAL_VALUE)
    assert nav["rows"] and all(row["present"] is True for row in nav["rows"] if
                               row["trade_date"] == TRADE_DATE)
    dense = _project(tmp_path / "dense", nav=nav_rows())
    assert dense["nav"]["report_status"] == "available"
    assert len(dense["nav"]["verified_rows"]) == len(portfolio_values())
    assert dense["nav"]["broken_links"] == []


@pytest.mark.parametrize("case", ["nav_only", "nav_missing_file", "authority_empty"])
def test_p6_t1_invalid_nav_forbids_annualization(tmp_path: Path, case: str) -> None:
    """R2-01（复核轮）：invalid/not_available 的 NAV 结果不得标 ``annualization_allowed=True``。"""
    if case == "nav_only":
        extra = nav_rows() + [
            {
                "trade_date": "20260610",
                "cash": 1.0,
                "position_value": 1.0,
                "total_value": 2.0,
                "nav": 1.0,
            }
        ]
        projection = _project(tmp_path, nav=extra)
    elif case == "nav_missing_file":
        projection = _project(tmp_path, nav_file=False)
    else:
        projection = _project(tmp_path, blob=build_engine_blob(portfolio=[]))
    nav = projection["nav"]
    assert nav["report_status"] in ("invalid", "not_available")
    assert nav["annualization_allowed"] is False
    assert "rows" in nav  # 明细不得因早退而消失
    assert "verified_rows" in nav
    # R3-01：config 子键集同样恒定（T2 取 config.annualization.report_status 不得 KeyError）
    assert projection["config"]["annualization"]["report_status"] in ("not_available", "invalid")
    assert "model" in projection["config"]


def test_p6_t1_guard_rejects_reparse_worktree_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2-02（复核轮）：执法根本身经重解析点/别名 ⇒ fail-fast（不允许静默不执法）。"""
    from src.lazybull.v2.reporting.readonly_audit import ReadonlyGuard, ReadonlyViolationError

    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    created = False
    try:
        os.symlink(real, alias, target_is_directory=True)
        created = True
    except (OSError, NotImplementedError):
        created = _make_junction(alias, real)
    if not created:
        # 无建链接权限：用 resolve 别名模拟“执法根经重解析点”的等价条件
        original = Path.resolve

        def fake_resolve(self: Path, *args: Any, **kwargs: Any) -> Path:
            if Path(os.path.abspath(self)) == Path(os.path.abspath(alias)):
                return original(real)
            return original(self, *args, **kwargs)

        alias.mkdir()
        monkeypatch.setattr(Path, "resolve", fake_resolve)
    with pytest.raises(ReadonlyViolationError):
        ReadonlyGuard(allowed_patterns=(), worktree_root=alias)


def _make_junction(link: Path, target: Path) -> bool:
    """尝试用 ``mklink /J`` 建目录联接（Windows 无需管理员权限）。"""
    try:
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0 and link.exists()


def test_p6_t1_nav_flag_cannot_contradict_status() -> None:
    """R2-N8：显式 ``annualization_allowed=True`` 与 invalid/not_available 矛盾即报错。"""
    from src.lazybull.v2.reporting import status as status_module

    row = status_module.make_status("invalid", reason_code="authority_mismatch", detail="x")
    with pytest.raises(ValueError):
        authority._nav_result(row, annualization_allowed=True)  # noqa: SLF001
    ok = authority._nav_result(row)  # noqa: SLF001
    assert ok["annualization_allowed"] is False


def test_p6_t1_nested_guards_both_enforced(tmp_path: Path) -> None:
    """R2-N3：嵌套 guard 必须**全部**执法（外层更严时内层活动期不得放宽）。"""
    from src.lazybull.v2.reporting.readonly_audit import ReadonlyGuard, ReadonlyViolationError

    worktree = tmp_path / "worktree"
    (worktree / "scripts").mkdir(parents=True)
    strict = ReadonlyGuard(allowed_patterns=(), worktree_root=worktree)
    loose = ReadonlyGuard(allowed_patterns=["scripts/**"], worktree_root=worktree)
    with strict, loose:
        with pytest.raises(ReadonlyViolationError):
            (worktree / "scripts" / "x.py").write_text("x", encoding="utf-8")


def test_p6_t1_json_roundtrip_of_projection(tmp_path: Path) -> None:
    """projection 必须 JSON 可序列化（NaN/Inf/时间戳不得泄漏）。"""
    projection = _project(tmp_path)
    payload = json.dumps(projection, ensure_ascii=False, allow_nan=False, default=str)
    assert '"report_status"' in payload
    assert inspect.isclass(type(projection))


# ---------------------------------------------------------------------------
# 2026-11-06 复审（reviewer2）新增负例
# ---------------------------------------------------------------------------


def test_p6_t1_generation_gate_blocks_dependents(tmp_path: Path) -> None:
    """P1-02：record 未通过完整代门禁 ⇒ 账户/成交/配置不得认证为可用。"""
    pre_switch = _project(tmp_path / "pre", drop_record_keys=("run_kind", "engine"))
    assert pre_switch["record"]["degradation_reason_code"] == "run_record_pre_switch_shape"
    assert pre_switch["account"]["report_status"] == "degraded"
    assert pre_switch["account"]["gate_reason_code"] == "run_record_pre_switch_shape"
    assert pre_switch["trades"]["report_status"] in ("degraded", "invalid", "empty_valid")
    assert pre_switch["account"]["account"]  # 值仍展示

    missing_marker = _project(
        tmp_path / "missing",
        run_kind="t1",
        record_extra={},
        drop_record_keys=("run_kind", "engine"),
    )
    assert missing_marker["record"]["report_status"] == "degraded"

    partial = _project(tmp_path / "partial", engine=None)
    assert partial["record"]["report_status"] == "invalid"
    assert partial["account"]["report_status"] == "invalid"
    assert partial["account"]["gate_reason_code"] == "authority_mismatch"


def test_p6_t1_broken_authority_trade_is_invalid(tmp_path: Path) -> None:
    """P1-04：权威成交缺字段不得被静默丢弃成“零成交”。"""
    broken = {"date": TRADE_DATE, "stock": "600000.SH", "action": "buy", "shares": 100}
    projection = _project(
        tmp_path,
        blob=build_engine_blob(authority_trades=[broken]),
        trades_rows=[],
        record_extra={"fills_count": 0},
    )
    row = projection["trades"]
    assert row["report_status"] == "invalid"
    assert row["degradation_reason_code"] == "authority_mismatch"
    assert row["broken_rows"]
    assert "不可判定" in row["detail"]
    # 有成交证据（即使损坏）⇒ NAV 区不得按“无成交”放行
    assert projection["nav"]["report_status"] in ("invalid", "stale", "available")


def test_p6_t1_nonfinite_amounts_rejected(tmp_path: Path) -> None:
    """P2-06：账户/NAV 的双侧 NaN 不得通过内容门。"""
    account_nan = _project(tmp_path / "acc", account_override={"cash": float("nan")})
    account_row = account_nan["account"]
    assert account_row["report_status"] == "invalid"
    assert any(item["path"] == "cash" for item in account_row["mismatches"])

    from tests.p6_t1_env import nav_rows

    rows = nav_rows()
    rows[-1]["total_value"] = float("nan")
    nav_nan = _project(tmp_path / "nav", nav=rows)
    assert nav_nan["nav"]["report_status"] == "invalid"
    assert nav_nan["nav"]["annualization_allowed"] is False


def test_p6_t1_late_file_after_missing_capture_is_drift(tmp_path: Path) -> None:
    """P2-05：请求时缺失、读取期间出现的文件也必须触发漂移（不得判稳定）。"""
    from src.lazybull.v2.reporting.snapshot import UnstableSnapshotError

    built = build_paper_root(tmp_path / "paper")
    paper = built["root"]
    late = paper / "state" / "late.json"

    def reader(snapshot: Any) -> None:
        assert snapshot.capture("paper", "state/late.json") is False
        late.write_text("{}", encoding="utf-8")

    with pytest.raises(UnstableSnapshotError):
        capture_stable_snapshot(
            roots={"paper": paper},
            staging_root=tmp_path / "staging",
            attempt_ids=["p6_t1_attempt_a"],
            reader=reader,
            max_retries=0,
        )


def test_p6_t1_instructions_gated_by_record(tmp_path: Path) -> None:
    """R2-20261106-02（复审）：record 不合格时指令区不得给出 authority_verified。"""
    from tests.p6_t1_env import FUTURE_DATE, instruction_rows

    partial = _project(
        tmp_path,
        engine=None,
        instructions={FUTURE_DATE: instruction_rows()},
        record_extra={"instructions_count": 3},
    )
    assert partial["record"]["report_status"] == "invalid"
    row = partial["instructions"]
    assert row["report_status"] == "invalid"
    assert row["gate_reason_code"] == "authority_mismatch"
    assert row["verification_level"] == "projection_only"
    assert "instructions" not in row

    missing_marker = _project(
        tmp_path / "marker",
        drop_record_keys=("run_kind", "engine"),
        instructions={FUTURE_DATE: instruction_rows()},
        record_extra={"instructions_count": 3},
    )
    assert missing_marker["instructions"]["degradation_reason_code"] == (
        "run_record_pre_switch_shape"
    )


def test_p6_t1_unparsable_trade_date_is_invalid(tmp_path: Path) -> None:
    """R2-20261106-04（复审）：字段齐全但日期不可解析的权威成交不得被当历史行。"""
    broken = {
        "date": "not-a-date",
        "stock": "600000.SH",
        "action": "buy",
        "price": 10.0,
        "shares": 100,
        "amount": 1000.0,
        "cost": 5.0,
    }
    projection = _project(
        tmp_path,
        blob=build_engine_blob(authority_trades=[broken]),
        trades_rows=[],
        record_extra={"fills_count": 0},
    )
    row = projection["trades"]
    assert row["report_status"] == "invalid"
    assert any("日期不可解析" in item for item in row["broken_rows"])


def test_p6_t1_quality_duplicate_rows_are_invalid(tmp_path: Path) -> None:
    """P2-09：同值重复行同样是“重复”，必须 invalid 而不是 available。"""
    import pandas as pd

    from src.lazybull.v2.reporting.quality import project_quality

    frame = pd.DataFrame(
        [
            {"layer": "raw", "dataset": "daily", "partition": None, "metric": "partition_count",
             "value": "2", "status": "ok"},
            {"layer": "raw", "dataset": "daily", "partition": None, "metric": "partition_count",
             "value": "2", "status": "ok"},
        ]
    )
    row = project_quality(rows=frame.to_dict("records"), present=True)
    summary = row["datasets"][0]
    assert summary["partition_count_status"]["report_status"] == "invalid"
    assert summary["partition_count_status"]["degradation_reason_code"] == "multiple_quality_value"
    assert summary["partition_count"] is None
