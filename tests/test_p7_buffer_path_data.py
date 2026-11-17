"""共享排名缓存的合成验证，不读取真实配置、模型或生产数据。"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.v2_p7 import buffer_path_data as module

DAYS = ("20260102", "20260105", "20260106", "20260107")
SPEC = "a" * 64


def topk(rows: list[tuple]) -> pd.DataFrame:
    """构造与既有辅证 dtype 一致的合成表。"""
    return pd.DataFrame(rows, columns=module.TOPK_COLUMNS).astype(
        {"trade_date": str, "rank": np.int64, "ts_code": str, "score": np.float64}
    )


def identity() -> dict:
    """合成 r6 身份；固定 SHA 仅代表允许的漂移口径。"""
    group = {
        "digest": "test",
        "partitions": 4,
        "frozen_manifest_checked": 4,
        "frozen_manifest_mismatch": [],
        "accepted_input_drift": [],
    }
    return {
        "cs_train": copy.deepcopy(group),
        "clean_daily": copy.deepcopy(group),
        "clean_meta": {
            "data/clean/stock_basic.parquet": {
                "sha256": module.STOCK_BASIC_SHA,
                "frozen_sha256": "old",
                "status": "accepted_drift",
            }
        },
        "clean_meta_mismatch": [],
        "accepted_input_drift": [
            {
                "path": "data/clean/stock_basic.parquet",
                "current": module.STOCK_BASIC_SHA,
                "frozen": "old",
            }
        ],
    }


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    """只创建四天、单折的隔离证据，所有生产入口用合成对象替换。"""
    source = tmp_path / module.SOURCE
    folder = source / "folds/split00"
    folder.mkdir(parents=True)
    reference = topk(
        [
            (DAYS[0], 1, "A", 0.3),
            (DAYS[0], 2, "B", 0.3),
            (DAYS[0], 3, "C", 0.1),
            (DAYS[3], 1, "B", 0.4),
        ]
    )
    reference.to_parquet(folder / "signal_topk.parquet", index=False)
    digest = module._sha(folder / "signal_topk.parquet")
    data = identity()
    module._write_json(source / "manifest.json", {"input_identity": data})
    module._write_json(
        source / "scores.json",
        {"stock_connect_closure_days": [DAYS[1]], "empty_ranking_days": [DAYS[1]]},
    )
    module._write_json(
        source / "fold_registry.json",
        {
            "folds": [
                {"split_index": 0, "model_version": 7, "scores": {"signal_topk_sha256": digest}}
            ]
        },
    )
    module._write_json(
        folder / "fold.json",
        {
            "model_version": 7,
            "signal_topk_sha256": digest,
            "signal_topk_rows": len(reference),
            "signal_topk_empty_days": 1,
            "closure_days": [DAYS[1]],
        },
    )
    params = SimpleNamespace(
        signal_days=DAYS,
        extended_days=DAYS + ("20260108",),
        v20_schedule={DAYS[0]: 0},
        data_root=tmp_path / "data",
        precheck_dir=tmp_path / "precheck",
        model_identity={7: "model"},
        folds=[SimpleNamespace(split_index=0, model_version=7, signal_days=DAYS)],
    )
    monkeypatch.setattr(module, "build_params", lambda root: params)
    monkeypatch.setattr(module, "_load_model_identity", lambda *args: params.model_identity)
    accepted_calls = []

    def capture(_params, accepted_drift):
        accepted_calls.append(accepted_drift)
        return copy.deepcopy(data)

    monkeypatch.setattr(module, "compute_input_identity", capture)
    code = {"digest": "code"}
    monkeypatch.setattr(module, "_code_identity", lambda root: dict(code))
    rankings = {
        DAYS[0]: [("B", 0.3), ("A", 0.3), ("C", 0.1)],
        DAYS[1]: [],
        DAYS[2]: [],
        DAYS[3]: [("B", 0.4)],
    }
    calls = []

    class Predictor:
        def __init__(self, _params):
            pass

        def __call__(self, day):
            calls.append(day)
            return list(rankings[day])

    monkeypatch.setattr(module, "RankedProvider", Predictor)
    monkeypatch.setattr(
        module,
        "_runtime_inputs",
        lambda p, r, i, v: module.ResearchInputs(
            p,
            tuple(p.extended_days),
            tuple(p.signal_days),
            dict(p.v20_schedule),
            r,
            "synthetic_variance",
            "synthetic_loader",
            i,
            v,
        ),
    )
    return SimpleNamespace(
        root=tmp_path,
        cache=tmp_path / "cache",
        rankings=rankings,
        reference=reference,
        calls=calls,
        data=data,
        code=code,
        accepted_calls=accepted_calls,
        params=params,
    )


def test_prepare_and_load_preserve_kernel_ties_and_do_not_predict_twice(synthetic):
    value = synthetic
    result = module.prepare_inputs(value.root, value.cache, spec_sha256=SPEC)
    assert value.calls == list(DAYS)
    assert result.ranked[DAYS[0]][0][0] == "B"
    assert result.validation["top20_auxiliary_bitwise_exact"]
    assert not result.validation["kernel_top20_order_exact"]
    assert result.validation["ordinary_empty_ranking_days"] == [DAYS[2]]
    assert result.calendar[-1] == "20260108" and result.schedule == {DAYS[0]: 0}
    loaded = module.load_inputs(value.root, value.cache, spec_sha256=SPEC)
    assert loaded.ranked == result.ranked and value.calls == list(DAYS)
    assert loaded.validation["cache_reused"]
    assert all(
        call == (f"data/clean/stock_basic.parquet:{module.STOCK_BASIC_SHA}",)
        for call in value.accepted_calls
    )


@pytest.mark.parametrize("change", ["data", "code", "spec", "artifact"])
def test_cache_identity_changes_fail_before_any_new_inference(synthetic, change):
    value = synthetic
    module.prepare_inputs(value.root, value.cache, spec_sha256=SPEC)
    spec = SPEC
    if change == "data":
        value.data["cs_train"]["digest"] = "changed"
    elif change == "code":
        value.code["digest"] = "changed"
    elif change == "spec":
        spec = "b" * 64
    else:
        path = value.cache / "ranked.parquet"
        path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        module.load_inputs(value.root, value.cache, spec_sha256=spec)
    assert value.calls == list(DAYS)


def test_existing_cache_and_missing_spec_are_rejected(synthetic):
    value = synthetic
    value.cache.mkdir()
    with pytest.raises(FileExistsError):
        module.prepare_inputs(value.root, value.cache, spec_sha256=SPEC)
    with pytest.raises(ValueError, match="spec_sha256"):
        module.prepare_inputs(value.root, value.root / "unused", spec_sha256="")
    assert value.calls == []


def test_unapproved_metadata_drift_is_not_inherited():
    original = identity()
    changed = copy.deepcopy(original)
    changed["accepted_input_drift"].append(
        {"path": "data/clean/trade_cal.parquet", "current": "changed"}
    )
    with pytest.raises(ValueError, match="裁决"):
        module.validate_input_identity(changed, original)
    changed = copy.deepcopy(original)
    changed["clean_meta"]["data/clean/stock_basic.parquet"]["sha256"] = "other"
    with pytest.raises(ValueError, match="stock_basic"):
        module.validate_input_identity(changed, original)


@pytest.mark.parametrize("change", ["score_bit", "missing_day", "duplicate", "nan", "closure"])
def test_invalid_or_changed_rankings_fail(synthetic, change):
    value = synthetic
    ranked = copy.deepcopy(value.rankings)
    if change == "score_bit":
        ranked[DAYS[3]][0] = ("B", np.nextafter(0.4, 1.0))
    elif change == "missing_day":
        del ranked[DAYS[3]]
    elif change == "duplicate":
        ranked[DAYS[0]].append(("A", 0.05))
    elif change == "nan":
        ranked[DAYS[3]][0] = ("B", np.nan)
    else:
        ranked[DAYS[1]] = [("A", 0.3)]
    with pytest.raises(ValueError):
        module.validate_rankings(ranked, value.reference, DAYS, [DAYS[1]], [DAYS[2]])


def test_daily_loader_never_ensures_and_rejects_outside_calendar():
    calls = []

    def load(day, auto_ensure):
        calls.append((day, auto_ensure))
        return pd.DataFrame({"trade_date": [day], "ts_code": ["A"], "close_adj": [10.0]})

    loader = module.ReadOnlyDailyLoader(SimpleNamespace(load_clean_daily_by_date=load), DAYS)
    assert len(loader.load_clean_daily_by_date(DAYS[0])) == 1
    with pytest.raises(ValueError, match="auto_ensure"):
        loader.load_clean_daily_by_date(DAYS[0], auto_ensure=True)
    with pytest.raises(ValueError, match="日历"):
        loader.load_clean_daily_by_date("20300101")
    assert calls == [(DAYS[0], False)]


def test_variance_uses_only_rows_at_or_before_signal_day():
    calendar = pd.bdate_range("20260101", periods=65).strftime("%Y%m%d").tolist()
    cutoff = calendar[59]

    def make_book(future_price):
        def load(day):
            index = calendar.index(day)
            price = 10 + index * 0.1 + (index % 3) * 0.03 if day <= cutoff else future_price
            return pd.DataFrame({"ts_code": ["A"], "close_adj": [price]})

        return module.VarianceBook(calendar, load)

    before, altered = make_book(20), make_book(1000000)
    assert before.variance("A", calendar[18]) is None
    assert before.variance("A", cutoff) == altered.variance("A", cutoff)
    assert before.variance("A", calendar[-1]) != altered.variance("A", calendar[-1])
