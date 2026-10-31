# -*- coding: utf-8 -*-
"""门 5 指纹与因果归因工具的正反例；不读取生产数据。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p4 import t5_shadow_compare as compare  # noqa: E402


def _input(*, shares: int = 100, pending: int = 0) -> dict:
    return {
        "date_phase": {"trade_date": "20261030", "phase": "T0"},
        "merged_target": [{"ts_code": "A", "weight": 1.0}],
        "account": {"cash": 1000.0, "positions": {"A": {"shares": shares}}},
        "pending": {"orders": pending},
        "schedule_anchor": {"anchor": "20261001"},
        "ranked_candidates": [{"ts_code": "A", "score": 0.8}],
        "stop_loss_state": {},
        "market_data": {"features": [{"ts_code": "A", "x": 1.0}], "close": 10.0},
        "effective_identity": {"model": "v24022", "buy_price": "close", "sell_price": "open"},
    }


def _order(ts_code: str = "A", shares: int = 100) -> dict:
    return {
        "date": "20261030",
        "phase": "T0",
        "action": "buy",
        "ts_code": ts_code,
        "shares": shares,
        "price_basis": "close",
        "reason": "rebalance",
        "execution_date": "20261102",
    }


def _panel_input() -> dict:
    record = _input()
    record["market_data"]["features"][0]["x"] = 2.0
    return record


def _evidence(category: str = "data_source") -> dict:
    payload = {
        "category": category,
        "input_difference": "panel.x 与 cs_infer.x 不同",
        "affected_instruction": "20261030|buy|A",
        "validation": {"passed": True, "method": "同状态换输入重算"},
    }
    if category == "data_source":
        payload.update({"feature_difference": "x: 1→2", "score_difference": "A: 0.8→0.7"})
    return payload


def test_same_full_input_and_output_passes() -> None:
    result = compare.compare_shadow_day(
        left_input=_input(), right_input=_input(), left_orders=[_order()], right_orders=[_order()]
    )
    assert result["verdict"] == "PASS"
    assert result["causal_evidence_rate"] == 1.0


def test_same_full_input_with_output_difference_is_unexplained() -> None:
    result = compare.compare_shadow_day(
        left_input=_input(),
        right_input=_input(),
        left_orders=[_order()],
        right_orders=[_order(shares=200)],
        evidence={"20261030|buy|A": _evidence("holding_path")},
    )
    assert result["verdict"] == "FAIL"
    assert result["hard_gates"]["same_full_input_same_output"] is False
    assert result["unexplained"] == ["20261030|buy|A"]


def test_model_input_same_but_holding_differs_does_not_trigger_hard3() -> None:
    result = compare.compare_shadow_day(
        left_input=_input(shares=100),
        right_input=_input(shares=200),
        left_orders=[_order()],
        right_orders=[_order(shares=200)],
        evidence={"20261030|buy|A": _evidence("holding_path")},
    )
    assert result["model_input_equal"] is True
    assert result["full_input_equal"] is False
    assert result["hard_gates"]["same_full_input_same_output"] is True
    assert result["verdict"] == "PASS"


def test_pending_difference_requires_causal_evidence() -> None:
    result = compare.compare_shadow_day(
        left_input=_input(pending=0),
        right_input=_input(pending=1),
        left_orders=[_order()],
        right_orders=[_order(shares=200)],
    )
    assert result["verdict"] == "FAIL"
    assert result["unexplained"] == ["20261030|buy|A"]


def test_legal_panel_difference_with_complete_chain_passes() -> None:
    result = compare.compare_shadow_day(
        left_input=_input(),
        right_input=_panel_input(),
        left_orders=[_order()],
        right_orders=[_order(shares=200)],
        evidence={"20261030|buy|A": _evidence()},
    )
    assert result["verdict"] == "PASS"
    assert result["bucket_counts"]["data_source"] == 1


def test_legal_panel_difference_cannot_cover_extra_wrong_order() -> None:
    result = compare.compare_shadow_day(
        left_input=_input(),
        right_input=_panel_input(),
        left_orders=[_order()],
        right_orders=[_order(shares=200), _order(ts_code="B")],
        evidence={"20261030|buy|A": _evidence()},
    )
    assert result["verdict"] == "FAIL"
    assert result["unexplained"] == ["20261030|buy|B"]
    assert result["causal_evidence_rate"] == 0.5


def test_data_source_evidence_requires_feature_and_score_links() -> None:
    evidence = _evidence()
    del evidence["score_difference"]
    result = compare.compare_shadow_day(
        left_input=_input(),
        right_input=_input(),
        left_orders=[_order()],
        right_orders=[_order(shares=200)],
        evidence={"20261030|buy|A": evidence},
    )
    assert result["verdict"] == "FAIL"
    assert result["hard_gates"]["causal_evidence_complete"] is False


def test_irrelevant_evidence_key_is_rejected() -> None:
    result = compare.compare_shadow_day(
        left_input=_input(),
        right_input=_input(),
        left_orders=[_order()],
        right_orders=[_order(shares=200)],
        evidence={"20261030|buy|B": _evidence("holding_path")},
    )
    assert result["verdict"] == "FAIL"
    assert any("不存在" in error for error in result["errors"])


def test_policy_append_cannot_pass_acceptance_edge() -> None:
    result = compare.compare_shadow_day(
        left_input=_input(),
        right_input=_input(),
        left_orders=[_order()],
        right_orders=[_order(shares=200)],
        evidence={"20261030|buy|A": _evidence("policy_append")},
    )
    assert result["verdict"] == "FAIL"
    assert any("不得用于验收边" in error for error in result["errors"])


def test_policy_append_is_allowed_for_informational_edge() -> None:
    result = compare.compare_shadow_day(
        left_input=_input(),
        right_input=_panel_input(),
        left_orders=[_order()],
        right_orders=[_order(shares=200)],
        evidence={"20261030|buy|A": _evidence("policy_append")},
        acceptance_edge=False,
    )
    assert result["verdict"] == "PASS"


def test_decision_fingerprint_requires_all_nine_groups() -> None:
    record = _input()
    del record["pending"]
    with pytest.raises(compare.ShadowCompareError, match="pending"):
        compare.decision_fingerprints(record)


def test_duplicate_order_key_is_rejected() -> None:
    with pytest.raises(compare.ShadowCompareError, match="未唯一合并"):
        compare.diff_orders([_order(), _order()], [])


def test_robust_z_requires_five_samples() -> None:
    result = compare.robust_z_anomalies([1.0, 1.0, 1.0, 1.0])
    assert result["status"] == "insufficient_samples"


def test_robust_z_flags_frozen_threshold() -> None:
    result = compare.robust_z_anomalies([1.0, 1.0, 1.0, 1.0, 10.0])
    assert result["special_attribution_required"] is True
    assert result["anomalies"][0]["index"] == 4


def test_canonical_orders_ignores_display_weight_suffix_only() -> None:
    base = {
        "date": "20260717", "phase": "T0", "action": "buy", "ts_code": "600496.SH",
        "shares": 11500, "price_basis": None, "execution_date": "20260720",
    }
    left = compare.canonical_orders([{**base, "reason": "信号生成"}])
    right = compare.canonical_orders([{**base, "reason": "信号生成 (权重=0.1301)"}])
    assert left == right
    sell = compare.canonical_orders(
        [{**base, "action": "sell", "reason": "持有期到期: 持有19天"}]
    )
    assert sell[0]["reason"] == "持有期到期: 持有19天"
    repeated = compare.canonical_orders(
        [{**base, "reason": "补位槽位-信号生成（买入后市值过小）（买入后市值过小）"}]
    )
    assert repeated[0]["reason"] == "补位槽位-信号生成"
