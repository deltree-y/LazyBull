# -*- coding: utf-8 -*-
"""P5b T6 机械 A/A 测试（tests/ 豁免复杂度）。

覆盖：逐日逐字段相等、非真空（比对行数/字段数非零）、单日单字段改动即失败、
日期序列不一致即失败、空输入即失败、真实账本复算（合成 3 日）。
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import p5b_t2_env as t2env  # noqa: E402

from src.lazybull.v2.hosts.testbed.rehearsal.aa import (  # noqa: E402
    AA_FIELDS,
    ArmDaySnapshot,
    compare_snapshots,
    mechanical_aa,
)


def _snapshot(day: str, nav: float = 1.0, generation: int = 1) -> ArmDaySnapshot:
    return ArmDaySnapshot(
        virtual_date=day,
        generation=generation,
        cash=1000.0,
        positions={"000001.SZ": 100},
        fills=[{"date": day, "side": "buy"}],
        nav=nav,
    )


def test_identical_snapshots_pass_with_nonzero_counts():
    base = [_snapshot("20260706"), _snapshot("20260707", generation=2)]
    clone = [_snapshot("20260706"), _snapshot("20260707", generation=2)]
    result = compare_snapshots(base, clone)
    assert result["identical"] is True
    assert result["compared_days"] == 2
    assert result["compared_fields"] == 2 * len(AA_FIELDS)
    assert set(result["field_counts"]) == set(AA_FIELDS)
    assert all(count == 2 for count in result["field_counts"].values())


def test_single_nav_change_fails():
    base = [_snapshot("20260706"), _snapshot("20260707", generation=2)]
    clone = [_snapshot("20260706"), _snapshot("20260707", nav=1.0000001, generation=2)]
    result = compare_snapshots(base, clone)
    assert result["identical"] is False
    mismatch = [item for item in result["mismatches"] if item["kind"] == "field_mismatch"]
    assert len(mismatch) == 1
    assert mismatch[0]["field"] == "nav" and mismatch[0]["virtual_date"] == "20260707"


def test_cash_and_positions_changes_fail():
    base = [_snapshot("20260706")]
    cash_changed = [dataclasses.replace(base[0], cash=999.0)]
    positions_changed = [dataclasses.replace(base[0], positions={"000001.SZ": 99})]
    assert compare_snapshots(base, cash_changed)["identical"] is False
    assert compare_snapshots(base, positions_changed)["identical"] is False


def test_empty_and_misaligned_inputs_fail():
    assert compare_snapshots([], [])["identical"] is False
    assert compare_snapshots([_snapshot("20260706")], [])["identical"] is False
    misaligned = compare_snapshots(
        [_snapshot("20260706"), _snapshot("20260707", generation=2)],
        [_snapshot("20260706", generation=1)],
    )
    assert misaligned["identical"] is False
    assert misaligned["mismatches"][0]["kind"] == "date_sequence_mismatch"


def test_mechanical_aa_from_real_ledger(tmp_path):
    env = t2env.build_env(tmp_path)
    prepared = t2env.prepare_campaign(
        env,
        campaign_id="t6_aa",
        arms=("base", "mech_aa"),
        roles={"base": "aa_clone", "mech_aa": "aa_clone"},
    )
    t2env.run_campaign(env, prepared, days=env.days[:3])
    from src.lazybull.v2.hosts.testbed.layout import layout_for

    layout = layout_for(prepared.root)
    result = mechanical_aa(layout, arm_base="base", arm_clone="mech_aa")
    assert result["identical"] is True
    assert result["compared_days"] == 3
    assert result["compared_fields"] == 3 * len(AA_FIELDS)
    assert result["source"].startswith("shard_recompute")
    # 负例：把真实快照的一日 NAV 改动 1 ulp ⇒ 必须失败（断言非真空）
    from src.lazybull.v2.hosts.testbed.rehearsal.aa import load_snapshots

    base = load_snapshots(layout, "base")
    clone = load_snapshots(layout, "mech_aa")
    doctored = list(clone)
    doctored[1] = dataclasses.replace(doctored[1], nav=doctored[1].nav + 1e-12)
    tampered = compare_snapshots(base, doctored)
    assert tampered["identical"] is False
    assert any(item["field"] == "nav" for item in tampered["mismatches"])


def test_mechanical_aa_requires_ledger(tmp_path):
    from src.lazybull.v2.hosts.testbed.layout import CampaignLayout

    with pytest.raises(Exception):
        mechanical_aa(
            CampaignLayout(root=tmp_path / "missing"), arm_base="base", arm_clone="mech_aa"
        )
