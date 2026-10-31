# -*- coding: utf-8 -*-
"""M-P4-1 数据层单一协调入口（scripts/v2_p4/daily_data_coordinator.py）单测。

只覆盖纯函数与命令装配（不触网、不读写生产 data 根）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p4 import daily_data_coordinator as coord  # noqa: E402


def test_gap_days_inclusive_bounds() -> None:
    calendar = ["20260731", "20260803", "20260804", "20261026", "20261027"]
    assert coord.gap_days(calendar, "20260803", "20261026") == [
        "20260803",
        "20260804",
        "20261026",
    ]
    assert coord.gap_days(calendar, "20260731", "20260731") == ["20260731"]


def test_gap_days_rejects_unknown_target() -> None:
    with pytest.raises(ValueError, match="不在交易日历"):
        coord.gap_days(["20260803"], "20260803", "20260804")


def test_partition_dates_normalizes_and_filters(tmp_path: Path) -> None:
    root = tmp_path / "raw" / "daily"
    root.mkdir(parents=True)
    names = ("2026-08-03.parquet", "20260804.parquet", "_stock_coverage.json", "readme.parquet")
    for name in names:
        (root / name).write_bytes(b"")
    assert coord.partition_dates(root) == ["20260803", "20260804"]


def test_raw_commands_three_non_overlapping_calls() -> None:
    commands = coord._raw_commands(["20260803", "20260804"], "20261026")
    assert len(commands) == 3
    daily, top10fh, periodic = commands
    assert "--force" not in daily
    for dataset in coord.DAILY_PARTITION_DATASETS:
        assert dataset in daily
    assert "stk_holdertrade" in daily and "report_rc" in daily and "stk_holdernumber" in daily
    # top10fh 起点必须早于上一报告期（否则无法回拉该期尾部公告）
    assert top10fh[top10fh.index("--start-date") + 1] == "20260401"
    assert top10fh[top10fh.index("--download") + 1] == "top10_floatholders"
    # 季分区族必须 --force 才重写上一期
    assert "--force" in periodic
    assert periodic[periodic.index("--start-date") + 1] == "20260401"
    assert set(coord.PERIODIC_DATASETS).issubset(set(periodic))


def test_missing_daily_partitions_reports_per_dataset(tmp_path: Path) -> None:
    ok_dir = tmp_path / "raw" / "daily"
    ok_dir.mkdir(parents=True)
    (ok_dir / "20260803.parquet").write_bytes(b"")
    (ok_dir / "20260804.parquet").write_bytes(b"")
    missing = coord._missing_daily_partitions(["20260803", "20260804"], root=tmp_path)
    assert "daily" not in missing
    assert missing["adj_factor"] == ["20260803", "20260804"]
    assert set(missing) == set(coord.DAILY_PARTITION_DATASETS) - {"daily"}


def test_split_source_gaps_separates_known_and_unexpected() -> None:
    missing = {"moneyflow_hsgt": ["20261019", "20261020"], "daily": ["20260803"]}
    known = {"moneyflow_hsgt": ("20261019",)}
    explained, unexpected = coord._split_source_gaps(missing, known)
    assert explained == {"moneyflow_hsgt": ["20261019"]}
    assert unexpected == {"moneyflow_hsgt": ["20261020"], "daily": ["20260803"]}


def test_split_source_gaps_all_known() -> None:
    explained, unexpected = coord._split_source_gaps(
        {"moneyflow_hsgt": ["20261019"]}, {"moneyflow_hsgt": ("20261019",)}
    )
    assert explained == {"moneyflow_hsgt": ["20261019"]}
    assert unexpected == {}


def test_watermark_of_supports_partition_and_single(tmp_path: Path) -> None:
    part = tmp_path / "raw" / "daily"
    part.mkdir(parents=True)
    (part / "20260803.parquet").write_bytes(b"")
    (part / "20260804.parquet").write_bytes(b"")
    single = tmp_path / "raw" / "stk_holdernumber.parquet"
    single.write_bytes(b"payload")
    assert coord._watermark_of("daily", root=tmp_path) == {
        "kind": "partition",
        "n": 2,
        "max": "20260804",
    }
    info = coord._watermark_of("stk_holdernumber", root=tmp_path)
    assert info["kind"] == "single" and info["bytes"] == len(b"payload")
    assert coord._watermark_of("absent_ds", root=tmp_path) == {"kind": "missing"}


def test_cs_infer_builder_flags_match_legacy_runner(tmp_path: Path) -> None:
    """cs_infer 必须复刻 paper/runner 的构成口径：require_label=False。"""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("paper_trade:\n  horizon: 20\n", encoding="utf-8")
    assert coord._cs_infer_builder_flags(cfg) == (20, False)
    assert coord._cs_infer_builder_flags(tmp_path / "absent.yaml") == (20, False)

    from src.lazybull.features import FeatureBuilder

    horizon, require_label = coord._cs_infer_builder_flags(cfg)
    builder = FeatureBuilder(horizon=horizon, require_label=require_label)
    assert builder.require_label is False
    assert builder.horizon == 20


def test_bypass_proxy_clears_process_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Windows 环境变量名大小写不敏感：统一用大写键断言
    monkeypatch.setenv("HTTP_PROXY", "http://192.168.1.21:18081")
    monkeypatch.setenv("HTTPS_PROXY", "http://192.168.1.21:18081")
    removed = coord._bypass_proxy()
    assert {"HTTP_PROXY", "HTTPS_PROXY"}.issubset(set(removed))
    import os

    assert "HTTP_PROXY" not in os.environ
    assert "HTTPS_PROXY" not in os.environ


def test_main_overwrites_stale_dry_run_marker(monkeypatch, tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "checkpoint.json").write_text(
        '{"run_id":"r1","stages":{"raw":{"dry_run":true,"status":"done"}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(coord, "_evidence_dir", lambda run_id: evidence)
    monkeypatch.setattr(coord, "_setup_logger", lambda run_id, stage: tmp_path / "run.log")
    monkeypatch.setitem(coord.STAGE_FUNCS, "raw", lambda run_id, args, log: {"gap_days": 1})
    coord.main(["--stages", "raw", "--run-id", "r1", "--execute"])
    checkpoint = coord._checkpoint("r1")
    assert checkpoint["stages"]["raw"]["dry_run"] is False
    assert checkpoint["stages"]["raw"]["status"] == "done"
    coord.main(["--stages", "raw", "--run-id", "r1"])
    assert coord._checkpoint("r1")["stages"]["raw"]["dry_run"] is True


def test_dash_formatting() -> None:
    assert coord._dash("20260803") == "2026-08-03"


def test_parse_args_all_expands_and_rejects_unknown() -> None:
    args = coord._parse_args(["--stages", "all", "--run-id", "r1"])
    assert args.stages == ",".join(coord.STAGE_ORDER)
    assert args.target_date == "20261026"
    with pytest.raises(SystemExit):
        coord._parse_args(["--stages", "nope", "--run-id", "r1"])


def test_stage_order_covers_registered_stages() -> None:
    assert set(coord.STAGE_ORDER) == set(coord.STAGE_FUNCS)
    assert coord.FROZEN_CS_TRAIN_THROUGH < coord.FROZEN_PANEL_THROUGH


def test_registered_gaps_are_not_in_pull_lists() -> None:
    """裁决 2026-10-29：登记不补的家族不得出现在任何拉取命令里。"""
    commands = coord._raw_commands(["20260803"], "20261026")
    flat = [token for cmd in commands for token in cmd]
    for dataset in coord.REGISTERED_GAPS:
        assert dataset not in flat
