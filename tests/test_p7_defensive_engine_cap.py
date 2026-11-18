"""引擎侧行业容量硬复核（用户裁决：改引擎）的反例测试。"""

from __future__ import annotations

from types import SimpleNamespace

from src.lazybull.v2.core.execution.buy_execution import _industry_capacity_ok


def _engine(positions: dict[str, dict], mapping: dict[str, str], cap: int | None) -> object:
    return SimpleNamespace(
        positions=positions, industry_mapping=mapping, max_per_industry=cap
    )


def test_allows_when_cap_disabled() -> None:
    engine = _engine({}, {"A": "银行"}, None)
    assert _industry_capacity_ok(engine, "A") is True


def test_rejects_when_bucket_full() -> None:
    positions = {code: {} for code in ("A", "B", "C", "D", "E")}
    mapping = {code: "银行" for code in ("A", "B", "C", "D", "E")} | {"F": "银行"}
    engine = _engine(positions, mapping, 5)
    assert _industry_capacity_ok(engine, "F") is False


def test_allows_when_bucket_has_room() -> None:
    positions = {code: {} for code in ("A", "B", "C", "D")}
    mapping = {code: "银行" for code in ("A", "B", "C", "D")} | {"F": "银行"}
    engine = _engine(positions, mapping, 5)
    assert _industry_capacity_ok(engine, "F") is True


def test_other_bucket_not_blocked_by_full_one() -> None:
    positions = {code: {} for code in ("A", "B", "C", "D", "E")}
    mapping = {code: "银行" for code in ("A", "B", "C", "D", "E")}
    mapping["G"] = "医药生物"
    engine = _engine(positions, mapping, 5)
    assert _industry_capacity_ok(engine, "G") is True


def test_unknown_bucket_passes_through() -> None:
    """行业不可判定时保持原行为（是否可买由选择期规则决定）。"""
    engine = _engine({}, {}, 5)
    assert _industry_capacity_ok(engine, "Z") is True
