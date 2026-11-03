# -*- coding: utf-8 -*-
"""P5b T4 配对账本样本（设计 v4 §5.1：``d_t`` 只从配对账本取）。

覆盖：``query_shards`` 只读门 + 校验分片读 NAV 的路径、``d_t`` 手工逐值核对、
INVALIDATED 区间 fail-closed（含审计旁路）、缺 NAV/非正 NAV/单点样本拒绝、
某臂缺日形成的缺口拒绝、读侧零写入。

注：两臂「日期不对齐」在合法账本里不可达（同一 index 共享日集合）；缺日必然先被
``query_shards`` 报成缺口，因此这里以缺口分支覆盖该失败面（见
``test_missing_arm_day_is_rejected_as_gap``）。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import p5b_t4_env as env  # noqa: E402

from src.lazybull.v2.hosts.testbed.ledger_common import LedgerInvalidatedError  # noqa: E402
from src.lazybull.v2.hosts.testbed.referee import (  # noqa: E402
    LedgerSampleError,
    ledger_sample_identity,
    nav_series_from_ledger,
    paired_returns_from_ledger,
)
from src.lazybull.v2.hosts.testbed.layout import layout_for  # noqa: E402

CAMPAIGN = "t4_ledger_campaign"


def _campaign(tmp_path: Path, *, days: int = 4, omit_valuation=None, nav_a=None, nav_b=None):
    dates = env.ledger_dates(days)
    nav = {
        "aa_1": nav_a or env.linear_nav(1.0, 0.001, days),
        "aa_2": nav_b or env.linear_nav(1.0, 0.0004, days),
    }
    root = env.synthetic_campaign(
        tmp_path,
        nav=nav,
        dates=dates,
        campaign_id=CAMPAIGN,
        omit_valuation=omit_valuation,
    )
    return root, dates, nav


def test_paired_returns_golden(tmp_path):
    root, dates, nav = _campaign(tmp_path)
    sample = paired_returns_from_ledger(layout_for(root), "aa_1", "aa_2")
    assert sample["sample_days"] == len(dates) - 1
    assert sample["d0_date"] == dates[0]
    assert sample["dates"] == dates[1:]
    assert sample["arm_order"] == ["aa_1", "aa_2"]
    for index, value in enumerate(sample["returns"]):
        expected = math.log(nav["aa_1"][index + 1] / nav["aa_1"][index]) - math.log(
            nav["aa_2"][index + 1] / nav["aa_2"][index]
        )
        assert value == pytest.approx(expected, rel=0, abs=1e-15)
    assert almost_constant(sample["returns"], target=0.0006)


def almost_constant(values, *, target: float) -> bool:
    return bool(np.allclose(np.asarray(values), target, atol=1e-12))


def test_paired_returns_read_path_and_generations(tmp_path):
    root, _, _ = _campaign(tmp_path)
    sample = paired_returns_from_ledger(layout_for(root), "aa_1", "aa_2")
    assert "query_shards" in sample["read_path"]
    assert sample["published_generation"] == {"aa_1": 4, "aa_2": 4}
    assert sample["generation_span"]["aa_1"] == [1, 4]
    assert len(sample["chain_digests"]["aa_1"]) == 64


def test_invalidated_window_fails_closed(tmp_path):
    root, dates, _ = _campaign(tmp_path)
    env.register_invalidated(root, arm_id="aa_1", date=dates[1], campaign_id=CAMPAIGN)
    with pytest.raises(LedgerInvalidatedError):
        paired_returns_from_ledger(layout_for(root), "aa_1", "aa_2")


def test_invalidated_window_audit_bypass(tmp_path):
    root, dates, nav = _campaign(tmp_path)
    env.register_invalidated(root, arm_id="aa_1", date=dates[1], campaign_id=CAMPAIGN)
    sample = paired_returns_from_ledger(layout_for(root), "aa_1", "aa_2", allow_invalidated=True)
    assert sample["sample_days"] == len(dates) - 1


def test_missing_valuation_is_rejected(tmp_path):
    root, _, _ = _campaign(tmp_path, omit_valuation="aa_2")
    with pytest.raises(LedgerSampleError):
        nav_series_from_ledger(layout_for(root), "aa_2")


def test_non_positive_nav_is_rejected(tmp_path):
    root, _, _ = _campaign(tmp_path, nav_b=[1.0, 0.0, 1.0, 1.0])
    with pytest.raises(LedgerSampleError):
        paired_returns_from_ledger(layout_for(root), "aa_1", "aa_2")


def test_single_valuation_point_is_rejected(tmp_path):
    root, _, _ = _campaign(tmp_path, days=1)
    with pytest.raises(LedgerSampleError):
        paired_returns_from_ledger(layout_for(root), "aa_1", "aa_2")


def test_missing_arm_day_is_rejected_as_gap(tmp_path):
    root, dates, _ = _campaign(tmp_path)
    env.publish_extra_day(root, campaign_id=CAMPAIGN, date="20260713", nav={"aa_1": 1.01})
    with pytest.raises(LedgerSampleError) as exc:
        paired_returns_from_ledger(layout_for(root), "aa_1", "aa_2")
    assert "缺口" in str(exc.value)


def test_ledger_read_is_zero_write(tmp_path):
    root, _, _ = _campaign(tmp_path)
    before = env.tree_state(root)
    paired_returns_from_ledger(layout_for(root), "aa_1", "aa_2")
    nav_series_from_ledger(layout_for(root), "aa_1")
    assert env.tree_state(root) == before


def test_ledger_sample_identity_summary(tmp_path):
    root, dates, _ = _campaign(tmp_path)
    identity = ledger_sample_identity(paired_returns_from_ledger(layout_for(root), "aa_1", "aa_2"))
    assert identity["d0_date"] == dates[0]
    assert identity["first_return_date"] == dates[1]
    assert identity["last_return_date"] == dates[-1]
    assert identity["sample_days"] == len(dates) - 1
    assert "returns" not in identity
