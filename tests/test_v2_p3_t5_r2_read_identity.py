# -*- coding: utf-8 -*-
"""R2-T5A-R2-02：归因日历、末次读取与报告结束身份的合成回归。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts.v2_p3 import run_data_impact_attribution as attribution
from scripts.v2_p3 import t5_attribution_io as io
from scripts.v2_p3 import t5_attribution_sections as sections
from scripts.v2_p3.t05b_isolated_root import _ensure_link
from tests.test_v2_p3_t5_data_impact_attribution import (
    _DAYS,
    _OBSERVER,
    _build_arm,
    _build_b0p,
    _build_data_root,
)


def _environment(base: Path, data_root: Path | None = None) -> dict[str, Path]:
    """所有臂证据真实派生，快照含实际交易日历及数据分区。"""
    data_root = data_root if data_root is not None else _build_data_root(base)
    return {
        "data_root": data_root,
        "arm2": _build_arm(base, "arm2", data_root=data_root),
        "arm3": _build_arm(base, "arm3", data_root=data_root),
        "b0p": _build_b0p(base),
    }


def _report(env: dict[str, Path], **kwargs: Any) -> dict[str, Any]:
    return attribution.build_report(
        env["arm2"],
        env["arm3"],
        env["b0p"],
        env["data_root"],
        observer_identity=_OBSERVER,
        **kwargs,
    )


def _errors(report: dict[str, Any]) -> str:
    return "\n".join(report["verdict"]["errors"])


def _change_cs(path: Path, values: list[float]) -> None:
    frame = pd.read_parquet(path)
    frame["ret_1"] = values
    frame.to_parquet(path, index=False)


def test_other_calendar_cannot_shrink_domain_with_bound_features(tmp_path: Path) -> None:
    """原评审反例：features 指向原文件、另一份日历缩域，必须因日历实读越界拒绝。"""
    data_root = _build_data_root(tmp_path)
    _change_cs(io.cs_train_day_path(data_root, _DAYS[-1]), [0.8, 0.9])
    env = _environment(tmp_path, data_root)
    assert _report(env)["verdict"]["verdict"] == "FAIL"
    other = tmp_path / "other_data_root"
    (other / "clean").mkdir(parents=True)
    pd.DataFrame(
        [{"cal_date": _DAYS[0], "is_open": 1}, {"cal_date": _DAYS[1], "is_open": 0}]
    ).to_parquet(other / "clean" / "trade_cal.parquet", index=False)
    link = other / "features"
    _ensure_link(link, data_root / "features")
    try:
        env["data_root"] = other
        report = _report(env)
        assert report["verdict"]["verdict"] == "FAIL"
        assert "实读越界" in _errors(report) and "trade_cal.parquet" in _errors(report)
    finally:
        # 先移除联接自身，保证 pytest 清理不会递归进入目标目录。
        if os.name == "nt":
            os.rmdir(link)
        else:
            link.unlink()


def test_empty_clean_fallback_records_both_calendar_files(tmp_path: Path) -> None:
    data_root = _build_data_root(tmp_path)
    clean = data_root / "clean" / "trade_cal.parquet"
    raw = data_root / "raw" / "trade_cal.parquet"
    raw.parent.mkdir(exist_ok=True)
    pd.read_parquet(clean).to_parquet(raw, index=False)
    pd.DataFrame(columns=["cal_date", "is_open"]).to_parquet(clean, index=False)
    ledger = io.ReadLedger()
    calendar = io.load_trade_calendar(data_root, ledger=ledger)
    assert set(ledger.records) == {ledger.normalize(clean), ledger.normalize(raw)}
    assert calendar.attrs["source_path"] == str(raw.resolve())
    assert calendar.attrs["source_sha256"] == io.file_sha256(raw)
    report = _report(_environment(tmp_path, data_root))
    assert report["verdict"]["verdict"] == "PASS", _errors(report)
    assert report["sections"]["a_column_input_diffs"]["folds"][0]["calendar_source"] == str(
        raw.resolve()
    )


def test_calendar_change_between_pre_hash_and_read_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _environment(tmp_path)
    from src.lazybull.data.loader import DataLoader

    original = DataLoader.load_clean_trade_cal

    def update_then_read(loader: DataLoader) -> pd.DataFrame | None:
        path = env["data_root"] / "clean" / "trade_cal.parquet"
        frame = pd.read_parquet(path)
        frame.loc[frame["cal_date"] == _DAYS[-1], "is_open"] = 0
        frame.to_parquet(path, index=False)
        return original(loader)

    monkeypatch.setattr(DataLoader, "load_clean_trade_cal", update_then_read)
    report = _report(env)
    assert report["verdict"]["verdict"] == "FAIL"
    assert "读后核验失败" in _errors(report) and "trade_cal.parquet" in _errors(report)


@pytest.mark.parametrize("changed_day", _DAYS)
def test_last_fold_cs_change_between_pre_hash_and_parquet_read_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_day: str
) -> None:
    """实际改写 parquet 且不恢复；最后一折/最后一天也必须即时拒绝。"""
    data_root = _build_data_root(tmp_path)
    _change_cs(io.cs_train_day_path(data_root, changed_day), [0.8, 0.9])
    env = _environment(tmp_path, data_root)
    for key in ("arm2", "arm3"):
        models = env[key] / "root" / "models" / "stock_selection"
        (models / "v1_features.json").write_text(
            json.dumps(["days_to_unlock", "dividend_payout_ratio"]), encoding="utf-8"
        )
        (models / "v2_features.json").write_text(json.dumps(["ret_1"]), encoding="utf-8")
    original = io.load_cs_train_day
    calls: dict[str, int] = {}

    def update_on_last_fold(storage: Any, day: str) -> pd.DataFrame | None:
        calls[day] = calls.get(day, 0) + 1
        if day == changed_day and calls[day] == 2:
            _change_cs(io.cs_train_day_path(data_root, day), [0.01, 0.02])
        return original(storage, day)

    monkeypatch.setattr(io, "load_cs_train_day", update_on_last_fold)
    report = _report(env)
    assert report["verdict"]["verdict"] == "FAIL"
    assert "读后核验失败" in _errors(report)
    assert "运行后实读内容漂移" in _errors(report)
    assert calls[changed_day] == 2


def test_disk_cache_hit_must_recheck_cs_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _environment(tmp_path)
    cache_dir = tmp_path / "cache"
    assert _report(env, cache_dir=cache_dir)["verdict"]["verdict"] == "PASS"
    original = sections.DayCompareCache.get
    changed = False

    def drift_after_cache_lookup(
        cache: sections.DayCompareCache, day: str, columns_fp: str, identity: dict[str, Any]
    ) -> dict[str, Any] | None:
        nonlocal changed
        result = original(cache, day, columns_fp, identity)
        if result is not None and not changed:
            _change_cs(io.cs_train_day_path(env["data_root"], day), [0.8, 0.9])
            changed = True
        return result

    monkeypatch.setattr(sections.DayCompareCache, "get", drift_after_cache_lookup)
    report = _report(env, cache_dir=cache_dir)
    assert changed and report["verdict"]["verdict"] == "FAIL"
    assert "读后核验失败" in _errors(report)


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("source", ["cs_train", "panel", "calendar"])
def test_report_end_rechecks_actual_inputs_after_other_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, warm: bool, source: str
) -> None:
    """a 节结束后的真实文件变更也拒绝；暖缓存及月帧复用不能洗白结束状态。"""
    env = _environment(tmp_path)
    cache_dir = tmp_path / "cache" if warm else None
    if warm:
        assert _report(env, cache_dir=cache_dir)["verdict"]["verdict"] == "PASS"
    original = attribution._collect_runs_sections

    def update_after_runs(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        if source == "cs_train":
            _change_cs(io.cs_train_day_path(env["data_root"], _DAYS[-1]), [0.8, 0.9])
        elif source == "calendar":
            path = env["data_root"] / "clean" / "trade_cal.parquet"
            frame = pd.read_parquet(path)
            frame.loc[frame["cal_date"] == _DAYS[-1], "is_open"] = 0
            frame.to_parquet(path, index=False)
        else:
            path = next((env["data_root"] / "features" / "panel" / _DAYS[-1]).glob("*.parquet"))
            frame = pd.read_parquet(path)
            frame["reviewer2_drift"] = 1
            frame.to_parquet(path, index=False)
        return result

    monkeypatch.setattr(attribution, "_collect_runs_sections", update_after_runs)
    report = _report(env, cache_dir=cache_dir)
    assert report["verdict"]["verdict"] == "FAIL"
    assert "运行后实读内容漂移" in _errors(report)
    identity = report["sections"]["g_identity_appendix"]["arm2"]
    assert identity["consumption_snapshot_verified"] is True
    assert identity["consumption_post_verified"] is False
    assert identity["consumption_post_drift_count"] >= 1


def test_correct_cold_warm_reports_include_successful_end_verification(tmp_path: Path) -> None:
    env = _environment(tmp_path)
    cache_dir = tmp_path / "cache"
    cold = _report(env, cache_dir=cache_dir)
    warm = _report(env, cache_dir=cache_dir)
    for report in (cold, warm):
        assert report["verdict"]["verdict"] == "PASS", _errors(report)
        for key in ("arm2", "arm3"):
            identity = report["sections"]["g_identity_appendix"][key]
            assert identity["consumption_post_verified"] is True
            assert identity["consumption_post_drift_count"] == 0
    left = dict(cold["sections"]["a_column_input_diffs"])
    right = dict(warm["sections"]["a_column_input_diffs"])
    left.pop("cache_stats")
    right.pop("cache_stats")
    assert json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def test_end_verification_runs_after_identity_appendix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """结束复核须覆盖 a~g 全部分节，身份附录组装期间的更新也不得漏检。"""
    env = _environment(tmp_path)
    original = attribution._build_identity_appendix

    def update_after_appendix(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        _change_cs(io.cs_train_day_path(env["data_root"], _DAYS[-1]), [0.8, 0.9])
        return result

    monkeypatch.setattr(attribution, "_build_identity_appendix", update_after_appendix)
    report = _report(env)
    assert report["verdict"]["verdict"] == "FAIL"
    assert "运行后实读内容漂移" in _errors(report)
    assert report["sections"]["g_identity_appendix"]["arm2"]["consumption_post_verified"] is False
