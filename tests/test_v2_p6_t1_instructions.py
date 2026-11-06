# -*- coding: utf-8 -*-
"""P6 T1 指令、pending 与调仓投影（P6-T1-01/04/12/14）。

覆盖：合成 v2 T1 有未来执行日指令（显示未来且不出现 legacy ``no_new_t0``）、恢复代缺计数
两分支、多历史+一个未来合法与双未来歧义、pending 五态（非 list/坏 JSON/合法空/缺文件/远端
不支持）与 rebalance 非 dict。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.reporting import instructions  # noqa: E402
from tests.p6_t1_env import (  # noqa: E402
    EARLIER_DATE,
    FUTURE_DATE,
    PREV_DATE,
    TRADE_DATE,
    build_paper_root,
    build_projection,
    instruction_rows,
)

_ALL_COUNT_KEYS = ("instructions_count", "fills_count", "pending_buys_count", "pending_sells_count")


def _project(tmp_path: Path, **kwargs: object) -> dict:
    built = build_paper_root(tmp_path / "paper", **kwargs)  # type: ignore[arg-type]
    projection, _snapshot, _audit, _info = build_projection(
        built["root"], tmp_path / "staging"
    )
    return projection


def test_p6_t1_01_future_instruction_is_shown(tmp_path: Path) -> None:
    """合成 v2 T1 有未来 E 指令 ⇒ 展示 E 与行明细，且不出现 legacy ``no_new_t0``。"""
    projection = _project(
        tmp_path,
        run_kind="t1",
        instructions={FUTURE_DATE: instruction_rows()},
        record_extra={"instructions_count": 3},
    )
    row = projection["instructions"]
    assert row["report_status"] == "available"
    assert row["execution_date"] == FUTURE_DATE
    assert row["instruction_count"] == 3
    assert len(row["instructions"]) == 3
    assert row["history_files"] == 0
    assert "no_new_t0" not in json.dumps(projection, ensure_ascii=False, default=str)


def test_p6_t1_14a_history_plus_single_future_is_legal(tmp_path: Path) -> None:
    """多历史指令 + 一个当前未来文件 ⇒ 合法；历史文件不参与未来推断。"""
    projection = _project(
        tmp_path,
        instructions={
            EARLIER_DATE: instruction_rows(source_date=PREV_DATE, count=2),
            PREV_DATE: instruction_rows(source_date=EARLIER_DATE, count=2),
            FUTURE_DATE: instruction_rows(),
        },
        record_extra={"instructions_count": 3},
    )
    row = projection["instructions"]
    assert row["report_status"] == "available"
    assert row["execution_date"] == FUTURE_DATE
    assert row["history_files"] == 2


def test_p6_t1_14b_two_future_files_is_invalid(tmp_path: Path) -> None:
    """两个未来执行日文件 ⇒ invalid/instruction_future_ambiguous（不择一）。"""
    projection = _project(
        tmp_path,
        instructions={
            FUTURE_DATE: instruction_rows(),
            "20260605": instruction_rows(),
        },
        record_extra={"instructions_count": 3},
    )
    row = projection["instructions"]
    assert row["report_status"] == "invalid"
    assert row["degradation_reason_code"] == "instruction_future_ambiguous"
    assert row["future_files"] == [FUTURE_DATE, "20260605"]


def test_p6_t1_14c_future_source_date_mismatch_is_invalid(tmp_path: Path) -> None:
    """未来指令行 ``source_date`` 不等于当前权威日 ⇒ 跨代 invalid。"""
    projection = _project(
        tmp_path,
        instructions={FUTURE_DATE: instruction_rows(source_date="20260530")},
        record_extra={"instructions_count": 3},
    )
    row = projection["instructions"]
    assert row["report_status"] == "invalid"
    assert row["degradation_reason_code"] == "instruction_source_date_mismatch"


def test_p6_t1_12a_recovery_without_counts_degrades_but_usable(tmp_path: Path) -> None:
    """恢复代缺计数 + 有未来指令 ⇒ degraded/recovered_generation_count_missing（可用）。"""
    projection = _project(
        tmp_path,
        drop_record_keys=_ALL_COUNT_KEYS,
        instructions={FUTURE_DATE: instruction_rows()},
    )
    row = projection["instructions"]
    assert row["report_status"] == "degraded"
    assert row["degradation_reason_code"] == "recovered_generation_count_missing"
    assert row["execution_date"] == FUTURE_DATE
    assert len(row["instructions"]) == 3
    assert row["instruction_count"] == 3


def test_p6_t1_12b_recovery_without_counts_and_no_trades(tmp_path: Path) -> None:
    """恢复代缺计数 + 无成交/无未来指令 ⇒ 零成交 empty_valid、指令区专用 not_available。"""
    projection = _project(
        tmp_path,
        drop_record_keys=_ALL_COUNT_KEYS,
        authority_trades=[],
        trades_rows=[],
    )
    trades = projection["trades"]
    assert trades["report_status"] == "empty_valid"
    assert "支持零成交语义" in trades["detail"]
    row = projection["instructions"]
    assert row["report_status"] == "not_available"
    assert row["degradation_reason_code"] == "recovery_counts_missing"
    assert "instructions" not in row


def test_p6_t1_04_pending_five_states(tmp_path: Path) -> None:
    """pending：{} invalid、坏 JSON invalid、[] empty_valid、缺文件 not_available、远端 unsupported。"""
    bad_type = _project(tmp_path / "a", pending_buys={"not": "a list"})
    row = bad_type["pending"]["buys"]
    assert row["report_status"] == "invalid"
    assert row["degradation_reason_code"] == "pending_not_list"

    broken = tmp_path / "b"
    built = build_paper_root(broken / "paper")
    (built["root"] / "pending_buys" / "pending_buys.json").write_text("{bad json", encoding="utf-8")
    projection, _snapshot, _audit, _info = build_projection(built["root"], broken / "staging")
    row = projection["pending"]["buys"]
    assert row["report_status"] == "invalid"
    assert row["degradation_reason_code"] == "invalid_json"

    empty = _project(tmp_path / "c", pending_buys=[], pending_sells=[])
    assert empty["pending"]["sells"]["report_status"] == "empty_valid"

    missing = _project(tmp_path / "d", pending_buys_present=False)
    row = missing["pending"]["buys"]
    assert row["report_status"] == "not_available"
    assert row["degradation_reason_code"] == "no_source"

    unsupported = instructions.project_pending(
        present=False,
        payload=None,
        parse_error=None,
        declared_count=None,
        label="远端队列",
        supported=False,
    )
    assert unsupported["report_status"] == "not_available"
    assert unsupported["degradation_reason_code"] == "unsupported"


def test_p6_t1_04b_pending_count_conflict_is_invalid(tmp_path: Path) -> None:
    """pending 计数与队列长度矛盾 ⇒ invalid/count_mismatch。"""
    projection = _project(
        tmp_path, pending_buys=[{"ts_code": "600000.SH"}], record_extra={"pending_buys_count": 5}
    )
    row = projection["pending"]["buys"]
    assert row["report_status"] == "invalid"
    assert row["degradation_reason_code"] == "count_mismatch"


def test_p6_t1_04c_rebalance_non_dict_is_invalid(tmp_path: Path) -> None:
    """rebalance 非 dict ⇒ invalid/rebalance_not_dict。"""
    projection = _project(tmp_path, rebalance=["not", "a", "dict"])
    row = projection["rebalance"]
    assert row["report_status"] == "invalid"
    assert row["degradation_reason_code"] == "rebalance_not_dict"


def test_p6_t1_04d_rebalance_original_values(tmp_path: Path) -> None:
    """rebalance 五字段原值展示（不重算）。"""
    payload = {
        "last_rebalance_date": "20260601",
        "last_scheduled_rebalance_date": "20260601",
        "tranche_anchor_date": "20260601",
        "rebalance_freq": 20,
        "stagger_tranches": 2,
    }
    projection = _project(tmp_path, rebalance=payload)
    row = projection["rebalance"]
    assert row["report_status"] == "available"
    assert row["rebalance"] == payload
    assert projection["authority"]["trade_date"] == TRADE_DATE


def test_p6_t1_instructions_empty_valid_when_declared_zero(tmp_path: Path) -> None:
    """record 声明 0 指令且无未来文件 ⇒ empty_valid；声明 >0 但缺文件 ⇒ invalid。"""
    zero = _project(tmp_path / "zero", record_extra={"instructions_count": 0})
    assert zero["instructions"]["report_status"] == "empty_valid"

    missing = _project(tmp_path / "missing", record_extra={"instructions_count": 3})
    row = missing["instructions"]
    assert row["report_status"] == "invalid"
    assert row["degradation_reason_code"] == "instruction_missing_future"


@pytest.mark.parametrize("run_kind", ["t0", "t1"])
def test_p6_t1_instruction_record_kind_matches_authority(tmp_path: Path, run_kind: str) -> None:
    """T0/T1 两相位都能装配未来指令区（记录相位必须与权威一致）。"""
    projection = _project(
        tmp_path / run_kind,
        run_kind=run_kind,
        instructions={FUTURE_DATE: instruction_rows()},
        record_extra={"instructions_count": 3},
    )
    assert projection["authority"]["run_kind"] == run_kind
    assert projection["record"]["report_status"] == "available"
    assert projection["instructions"]["report_status"] == "available"
