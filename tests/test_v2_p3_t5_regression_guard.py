# -*- coding: utf-8 -*-
"""回归边受保护复跑入口测试（R2-T5AE-03 / R1-P2-2，全合成数据）。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v2_p3.regression_edge_runner import (  # noqa: E402
    BASELINE_REF,
    SOURCE,
    collect_guarded_inputs,
    run_guarded_regression_edge,
)
from src.lazybull.v2.hosts.backtest.runtime import LEGACY_BYPASS  # noqa: E402


def _make_data_root(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    for sub in ("features/cs_train", "clean/daily", "raw/suspend"):
        (data / sub).mkdir(parents=True, exist_ok=True)
    (data / "features" / "cs_train" / "20240102.parquet").write_bytes(b"cs")
    (data / "clean" / "daily" / "2024-01-02.parquet").write_bytes(b"daily")
    (data / "raw" / "suspend" / "2024-01-02.parquet").write_bytes(b"suspend")
    (data / "clean" / "stock_basic.parquet").write_bytes(b"basic")
    (data / "raw" / "trade_cal.parquet").write_bytes(b"cal")
    models = data / "models" / "stock_selection"
    models.mkdir(parents=True)
    for version in range(24008, 24022):
        for suffix in ("features.json", "metadata.json", "model.joblib"):
            (models / f"v{version}_{suffix}").write_bytes(f"{version}-{suffix}".encode())
    return data


def _fake_compare(new_dir: Path, frozen_dir: Path) -> Dict[str, object]:
    return {"verdict": "PASS", "errors": [], "new": str(new_dir), "frozen": str(frozen_dir)}


class TestGuardedRegressionEdge:
    def test_inputs_include_models_and_merger_tuple_is_fixed(self, tmp_path):
        data = _make_data_root(tmp_path)
        inputs = collect_guarded_inputs(data)
        assert len(inputs) == 5 + 14 * 3
        assert any(path.name == "v24021_model.joblib" for path in inputs)

        calls: List[Dict[str, object]] = []

        def replay_fn(**kwargs: object) -> Path:
            calls.append(dict(kwargs))
            return tmp_path / "new_batch"

        verdict = run_guarded_regression_edge(
            data_root=data,
            out_dir=tmp_path / "evidence",
            batch_id="new_batch",
            replay_fn=replay_fn,
            compare_fn=_fake_compare,
        )
        assert verdict["verdict"] == "PASS"
        assert calls[0]["merger_mode"] == LEGACY_BYPASS
        assert calls[0]["arms"] == (LEGACY_BYPASS,)
        assert calls[0]["source"] == SOURCE
        assert calls[0]["baseline_ref"] == BASELINE_REF
        assert (tmp_path / "evidence" / "consumption_before.json").is_file()
        assert (tmp_path / "evidence" / "regression_edge_verdict.json").is_file()

    def test_input_drift_fails_even_when_comparison_passes(self, tmp_path):
        data = _make_data_root(tmp_path)
        target = data / "features" / "cs_train" / "20240102.parquet"

        def replay_fn(**kwargs: object) -> Path:
            target.write_bytes(b"changed-during-replay")
            return tmp_path / "new_batch"

        verdict = run_guarded_regression_edge(
            data_root=data,
            out_dir=tmp_path / "evidence",
            batch_id="new_batch",
            replay_fn=replay_fn,
            compare_fn=_fake_compare,
        )
        assert verdict["verdict"] == "FAIL"
        assert verdict["guard"]["after_drift"]
        assert any("消费输入漂移" in message for message in verdict["errors"])
