"""B0 发布隔离与折条件门测试（F05/C3）。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.v2_p7.defensive_ml import artifacts, training


def _make_stage(camp: Path, run_dir: Path, payload: dict) -> None:
    artifacts.write_json_checked(run_dir / "dataset.json", {"stage": "dataset"}, camp)
    artifacts.seal_stage(run_dir, "dataset", payload, artifact_base=camp.parent.parent)


def test_publish_then_require_roundtrip(tmp_path: Path) -> None:
    camp = tmp_path / "camp"
    work = camp / "work"
    (work / "dataset").mkdir(parents=True)
    (work / "dataset" / "panel_2013.parquet").write_bytes(b"panel")
    run_dir = camp / "runs" / "dataset_r1"
    run_dir.mkdir(parents=True)
    base = tmp_path
    pub = artifacts.publish_artifacts(
        work, run_dir, "dataset", ("dataset",), {"x": 1}, artifact_base=base
    )
    assert "artifacts/dataset/panel_2013.parquet" in pub["files"]
    artifacts.seal_stage(run_dir, "dataset", {"artifacts": pub}, artifact_base=base)
    source = artifacts.require_published(camp, "dataset_r1", "dataset")
    assert (source / "panel_2013.parquet").read_bytes() == b"panel"


def test_publish_refuses_second_time(tmp_path: Path) -> None:
    camp = tmp_path / "camp"
    work = camp / "work"
    (work / "dataset").mkdir(parents=True)
    (work / "dataset" / "a.parquet").write_bytes(b"x")
    run_dir = camp / "runs" / "r1"
    run_dir.mkdir(parents=True)
    artifacts.publish_artifacts(
        work, run_dir, "dataset", ("dataset",), {}, artifact_base=tmp_path
    )
    with pytest.raises(artifacts.ArtifactError):
        artifacts.publish_artifacts(
            work, run_dir, "dataset", ("dataset",), {}, artifact_base=tmp_path
        )


def test_require_detects_content_drift(tmp_path: Path) -> None:
    camp = tmp_path / "camp"
    work = camp / "work"
    (work / "dataset").mkdir(parents=True)
    (work / "dataset" / "a.parquet").write_bytes(b"x")
    run_dir = camp / "runs" / "r1"
    run_dir.mkdir(parents=True)
    pub = artifacts.publish_artifacts(
        work, run_dir, "dataset", ("dataset",), {}, artifact_base=tmp_path
    )
    artifacts.seal_stage(run_dir, "dataset", {"artifacts": pub}, artifact_base=tmp_path)
    (run_dir / "artifacts" / "dataset" / "a.parquet").write_bytes(b"changed")
    with pytest.raises(artifacts.ArtifactError):
        artifacts.require_published(camp, "r1", "dataset")


def test_require_rejects_unpublished_name(tmp_path: Path) -> None:
    camp = tmp_path / "camp"
    run_dir = camp / "runs" / "r1"
    run_dir.mkdir(parents=True)
    artifacts.write_json_checked(run_dir / "dataset.json", {"stage": "dataset"}, tmp_path)
    artifacts.seal_stage(run_dir, "dataset", {"artifacts": {"files": {}}}, artifact_base=tmp_path)
    with pytest.raises(artifacts.ArtifactError):
        artifacts.require_published(camp, "r1", "accounts")


def _panel(train_days: list[str], val_days: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    def rows(days: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "trade_date": days,
                "ts_code": ["A"] * len(days),
                "label_status": ["valid"] * len(days),
                "label_end_date": days,
            }
        )

    return rows(train_days), rows(val_days)


def test_fold_condition_gate_rejects_short_train() -> None:
    days = [f"20{20 + i // 12:02d}{i % 12 + 1:02d}01" for i in range(40)]
    val_days = [
        "20250102", "20250103", "20250106", "20250107", "20250108", "20250109",
        "20250110", "20250113", "20250114", "20250115", "20250116",
    ]
    train, val = _panel(days, val_days)
    with pytest.raises(training.FoldConditionError):
        training.assert_fold_conditions({"fold_id": 0}, train, val)


def test_fold_condition_gate_accepts_valid_fold() -> None:
    days = [f"20{13 + i // 12:02d}{i % 12 + 1:02d}01" for i in range(120)]
    val_days = [f"20{23 + i // 12:02d}{i % 12 + 1:02d}01" for i in range(12)]
    train, val = _panel(days, val_days)
    training.assert_fold_conditions({"fold_id": 0}, train, val)
