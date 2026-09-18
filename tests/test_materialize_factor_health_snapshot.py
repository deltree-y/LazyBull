#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""体检快照物化脚本（materialize_factor_health_snapshot）专项测试。

只使用合成数据：临时数据根当作"生产根"，验证物化产物（列集、运行时派生列、
特征清单、非法 out-root 拦截）。
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import materialize_factor_health_snapshot as snapshot
from src.lazybull.common.config import get_config


@pytest.fixture()
def fake_production(tmp_path, monkeypatch):
    """构造合成生产根：features/cs_train 分区 + raw/stk_holdertrade 年分区。"""
    root = tmp_path / "prod"
    cs_dir = root / "features" / "cs_train"
    cs_dir.mkdir(parents=True)
    for day in ("20200102", "20200103", "20200106", "20200107", "20200108", "20200109"):
        pd.DataFrame(
            {
                "trade_date": [day, day],
                "ts_code": ["000001.SZ", "000002.SZ"],
                "neu_y_ret_20": [0.01, -0.02],
                "mkt_vol_20": [0.2, 0.2],
                "zscore_size": [0.1, -0.1],
                "some_feature": [1.0, 2.0],
            }
        ).to_parquet(cs_dir / f"{day}.parquet", index=False)

    raw_dir = root / "raw" / "stk_holdertrade"
    raw_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            ("000001.SZ", "20200102", "G", "IN", 0.5),
            ("000001.SZ", "20200102", "G", "IN", 0.3),
            ("000002.SZ", "20200103", "C", "DE", 1.0),
        ],
        columns=["ts_code", "ann_date", "holder_type", "in_de", "change_ratio"],
    ).to_parquet(raw_dir / "2020-12-31.parquet", index=False)

    monkeypatch.setattr(snapshot, "get_data_root", lambda: str(root))
    get_config().set("data.root", str(root))
    yield root
    get_config().set("data.root", str(Path.cwd() / "data"))


def test_materialize_writes_runtime_columns_and_feature_file(fake_production, tmp_path):
    out_root = tmp_path / "snapshot"
    feature_file = tmp_path / "features.json"
    feature_file.write_text(json.dumps(["some_feature", "missing_feature"]), encoding="utf-8")

    args = snapshot.build_parser().parse_args(
        [
            "--out-root",
            str(out_root),
            "--start",
            "20200101",
            "--end",
            "20200131",
            "--every",
            "1",
            "--with-holdertrade",
            "--feature-file",
            str(feature_file),
        ]
    )
    meta = snapshot.materialize(args)

    assert meta["files"] == 6
    written = sorted((out_root / "features" / "cs_train").glob("*.parquet"))
    assert [p.stem for p in written] == [
        "20200102",
        "20200103",
        "20200106",
        "20200107",
        "20200108",
        "20200109",
    ]
    day = pd.read_parquet(out_root / "features" / "cs_train" / "20200102.parquet")
    # 运行时列已派生且非空（窗口外显式 0）
    assert day.loc[day["ts_code"] == "000001.SZ", "ht_net_ratio_30d"].iloc[0] == pytest.approx(0.8)
    assert day.loc[day["ts_code"] == "000002.SZ", "ht_net_ratio_30d"].iloc[0] == pytest.approx(0.0)
    assert day["holdertrade_schema_v1"].eq(1).all()
    # 不存在的列不进快照列集；支撑列保留
    assert "missing_feature" not in day.columns
    for col in ("trade_date", "ts_code", "neu_y_ret_20", "mkt_vol_20"):
        assert col in day.columns

    features = json.loads((out_root / "feature_file.json").read_text(encoding="utf-8"))
    assert "ht_net_ratio_30d" in features
    assert "holdertrade_schema_v1" in features
    assert "missing_feature" not in features


def test_materialize_requires_raw_holdertrade_when_enabled(fake_production, tmp_path):
    for path in (fake_production / "raw" / "stk_holdertrade").glob("*.parquet"):
        path.unlink()
    out_root = tmp_path / "snapshot2"
    feature_file = tmp_path / "features2.json"
    feature_file.write_text(json.dumps(["some_feature"]), encoding="utf-8")
    args = snapshot.build_parser().parse_args(
        [
            "--out-root",
            str(out_root),
            "--start",
            "20200101",
            "--end",
            "20200131",
            "--every",
            "1",
            "--with-holdertrade",
            "--feature-file",
            str(feature_file),
        ]
    )
    with pytest.raises(ValueError, match="stk_holdertrade"):
        snapshot.materialize(args)


def test_materialize_derives_repurchase_columns(fake_production, tmp_path):
    """repurchase 家族：派生需要 circ_mv/amount/vol，但派生后不得写进快照。"""
    cs_dir = fake_production / "features" / "cs_train"
    for path in cs_dir.glob("*.parquet"):
        frame = pd.read_parquet(path)
        frame["circ_mv"] = [100000.0, 200000.0]
        frame["amount"] = [10000.0, 20000.0]
        frame["vol"] = [10000.0, 20000.0]
        frame.to_parquet(path, index=False)
    raw_dir = fake_production / "raw" / "repurchase"
    raw_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            ("000001.SZ", "20200102", "实施", 1.0e8, 1.0e6, 12.0),
            ("000002.SZ", "20200103", "预案", 5.0e7, None, 8.0),
        ],
        columns=["ts_code", "ann_date", "proc", "amount", "vol", "high_limit"],
    ).to_parquet(raw_dir / "2020-12-31.parquet", index=False)

    out_root = tmp_path / "snapshot_rp"
    feature_file = tmp_path / "features_rp.json"
    feature_file.write_text(json.dumps(["some_feature"]), encoding="utf-8")
    args = snapshot.build_parser().parse_args(
        [
            "--out-root",
            str(out_root),
            "--start",
            "20200101",
            "--end",
            "20200131",
            "--every",
            "1",
            "--with-repurchase",
            "--feature-file",
            str(feature_file),
        ]
    )
    meta = snapshot.materialize(args)
    assert meta["files"] == 6

    day = pd.read_parquet(out_root / "features" / "cs_train" / "20200102.parquet")
    # 000001.SZ：增量 1.0e8 元 ÷ 流通市值 100000 万元(=1e9 元) = 0.1
    assert day.loc[day["ts_code"] == "000001.SZ", "rp_amount_to_mv_90d"].iloc[0] == pytest.approx(
        0.1
    )
    # 000002.SZ 窗口内无公告 ⇒ 显式 0
    assert day.loc[day["ts_code"] == "000002.SZ", "rp_amount_to_mv_90d"].iloc[0] == 0.0
    assert day["repurchase_schema_v1"].eq(1).all()
    # 派生支撑列不落盘
    for col in ("circ_mv", "amount", "vol"):
        assert col not in day.columns

    features = json.loads((out_root / "feature_file.json").read_text(encoding="utf-8"))
    assert "rp_amount_to_mv_90d" in features
    assert "repurchase_schema_v1" in features


def test_materialize_requires_raw_repurchase_when_enabled(fake_production, tmp_path):
    out_root = tmp_path / "snapshot_rp2"
    feature_file = tmp_path / "features_rp2.json"
    feature_file.write_text(json.dumps(["some_feature"]), encoding="utf-8")
    args = snapshot.build_parser().parse_args(
        [
            "--out-root",
            str(out_root),
            "--start",
            "20200101",
            "--end",
            "20200131",
            "--every",
            "1",
            "--with-repurchase",
            "--feature-file",
            str(feature_file),
        ]
    )
    with pytest.raises(ValueError, match="repurchase"):
        snapshot.materialize(args)


def test_main_rejects_out_root_inside_production(fake_production):
    args = [
        "--out-root",
        str(fake_production / "features"),
        "--start",
        "20200101",
        "--end",
        "20200131",
        "--skip-health",
        "--skip-diagnosis",
    ]
    import sys

    old_argv = sys.argv
    sys.argv = ["materialize_factor_health_snapshot.py"] + args
    try:
        with pytest.raises(ValueError, match="out-root"):
            snapshot.main()
    finally:
        sys.argv = old_argv
