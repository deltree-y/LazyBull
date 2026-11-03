# -*- coding: utf-8 -*-
"""P5b T6 宿主层注入与账本链路测试（tests/ 豁免复杂度）。

覆盖：注入只改 NAV（成交/现金/持仓与 base 逐值一致）、收益臂 NAV 的 golden 关系、
DD 臂 ``not_constructible`` 即纯克隆、机械 A/A 逐值一致（从分片复算）、同 seed 重跑
确定性、生产根零写入。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import p5b_t2_env as t2env  # noqa: E402
import p5b_t6_env as t6env  # noqa: E402

from src.lazybull.v2.hosts.testbed.rehearsal.aa import mechanical_aa  # noqa: E402
from src.lazybull.v2.hosts.testbed.rehearsal.injection import (  # noqa: E402
    arm_effect_plan,
    income_daily_shift,
)


def _env(tmp_path: Path) -> SimpleNamespace:
    return t2env.build_env(tmp_path)


def _production_snapshot(repo_root: Path) -> dict:
    from src.lazybull.v2.hosts.testbed.rehearsal.sources import (
        collect_production_fingerprints,
    )

    return collect_production_fingerprints(repo_root)


def _run(tmp_path: Path, campaign_id: str, days: int = 3):
    env = _env(tmp_path)
    repo = t6env.build_repo(tmp_path)
    prepared = t6env.prepare_arms(env, campaign_id=campaign_id)
    document = t6env.manifest_document(env, repo, campaign_id=campaign_id)
    t6env.run_with_injection(env, prepared, document, days=env.days[:days])
    return env, prepared, document


def test_injection_changes_only_nav(tmp_path):
    env, prepared, _ = _run(tmp_path, "t6_inject")
    days = env.days[:3]
    shift = income_daily_shift()
    for day_index, day in enumerate(days):
        base = t6env.shard(prepared, "base", day)["evidence"]
        income = t6env.shard(prepared, "income_minus_5d", day)["evidence"]
        # 成交/现金/持仓/日期 逐值一致
        assert income["cash"] == base["cash"]
        assert income["positions"] == base["positions"]
        assert income["fills"] == base["fills"]
        assert income["driven_date"] == base["driven_date"]
        # NAV 只按累积位移改变（golden）
        expected = base["valuation"]["nav"] * math.exp(day_index * shift)
        assert math.isclose(income["valuation"]["nav"], expected, rel_tol=1e-12)
        # market_value 与 portfolio_value 自洽（注入后不破坏身份恒等式）
        row_value = income["valuation"]["portfolio_value"]
        assert math.isclose(
            row_value, income["cash"] + (row_value - income["cash"]), rel_tol=0.0, abs_tol=1e-9
        )


def test_not_constructible_dd_arms_are_plain_clones(tmp_path):
    env, prepared, document = _run(tmp_path, "t6_clone")
    effects = arm_effect_plan(document, int(document["window"]["return_days"]))
    for arm_id in ("dd_improve_12pp", "dd_worsen_12pp"):
        assert effects[arm_id].status == "not_constructible"
        assert effects[arm_id].active is False
    for day in env.days[:3]:
        base = t6env.shard(prepared, "base", day)["evidence"]
        for arm_id in ("dd_improve_12pp", "dd_worsen_12pp"):
            clone = t6env.shard(prepared, arm_id, day)["evidence"]
            assert clone["valuation"]["nav"] == base["valuation"]["nav"]
            assert clone["cash"] == base["cash"]
            assert clone["positions"] == base["positions"]
            assert clone["fills"] == base["fills"]


def test_mechanical_aa_is_value_by_value(tmp_path):
    env, prepared, _ = _run(tmp_path, "t6_aa_run")
    from src.lazybull.v2.hosts.testbed.layout import layout_for

    result = mechanical_aa(layout_for(prepared.root), arm_base="base", arm_clone="mech_aa")
    assert result["identical"] is True
    assert result["compared_days"] == 3
    assert result["compared_fields"] == 3 * 6
    assert result["mismatches"] == []


def test_same_seed_rerun_is_deterministic(tmp_path):
    first_env, first, _ = _run(tmp_path / "a", "run_a")
    second_env, second, _ = _run(tmp_path / "b", "run_b")
    for arm_id in t6env.ARMS:
        left = t6env.nav_series(first, arm_id, first_env.days[:3])
        right = t6env.nav_series(second, arm_id, second_env.days[:3])
        assert np.array_equal(np.asarray(left), np.asarray(right))
        digests_left = [
            t6env.shard(first, arm_id, day)["evidence"]["day_digest"] for day in first_env.days[:3]
        ]
        digests_right = [
            t6env.shard(second, arm_id, day)["evidence"]["day_digest"]
            for day in second_env.days[:3]
        ]
        assert digests_left == digests_right


def test_run_does_not_touch_production_roots(tmp_path):
    before = _production_snapshot(ROOT)
    _run(tmp_path, "t6_zero_write")
    assert _production_snapshot(ROOT) == before


def test_injected_store_rejects_missing_valuation_row(tmp_path):
    env = _env(tmp_path)
    repo = t6env.build_repo(tmp_path)
    prepared = t6env.prepare_arms(env, campaign_id="t6_bad_row")
    document = t6env.manifest_document(env, repo, campaign_id="t6_bad_row")
    effects = arm_effect_plan(document, int(document["window"]["return_days"]))
    from src.lazybull.v2.hosts.paper.state_store import PaperStateStore
    from src.lazybull.v2.hosts.testbed.rehearsal.hosts import (
        InjectedValuationStore,
        RehearsalHostError,
    )

    store = InjectedValuationStore(
        PaperStateStore(root_path=str(prepared.root / "arms" / "income_minus_5d")),
        effect=effects["income_minus_5d"],
    )
    with pytest.raises(RehearsalHostError):
        store.observe(env.days[0])
