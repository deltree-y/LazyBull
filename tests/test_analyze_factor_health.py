# -*- coding: utf-8 -*-
"""因子体检脚本（scripts/analyze_factor_health.py）测试。"""

import json

import numpy as np
import pandas as pd

from scripts.factor_health.analysis import (
    add_year_profiles,
    assemble_register,
    attach_clusters,
    attach_families,
    attach_twin_info,
    candidate_tables,
    cluster_features,
    compute_model_usage,
    family_of,
    flag_candidates,
    map_booster_scores,
    parse_version_spec,
    prefer_plain_representatives,
)
from scripts.factor_health.constants import HealthThresholds
from scripts.factor_health.report import (
    build_exclude_lists,
    build_report_markdown,
    summarize,
    write_health_outputs,
)
from scripts.factor_health.scan import (
    average_correlation,
    compute_daily_rank_ic,
    pick_corr_dates,
    pick_partition_files,
    scan_features,
)

# ---------- 基础工具 ----------


def test_map_booster_scores_maps_generic_feature_names():
    scores = {"f0": 2.0, "f1": 1.0, "other": 0.5}
    mapped = map_booster_scores(scores, ["alpha", "beta"])
    assert mapped == {"alpha": 2.0, "beta": 1.0, "other": 0.5}


def test_family_of_inherits_twin_family_and_freshness():
    assert family_of("zscore_roe_waa_sz") == "fundamental"
    assert family_of("zscore_ma_deviation_20_sz").startswith("base-")
    assert family_of("unknown_factor_freshness_days") == "meta-freshness"
    assert family_of("some_unknown_factor") == "base-其他"


def test_parse_version_spec_supports_ranges():
    assert parse_version_spec("v24008-24010, 24010, 24052") == [24008, 24009, 24010, 24052]


# ---------- 扫描 ----------


def _synthetic_day(date: str, n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    good = rng.normal(size=n)
    base = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "trade_date": date,
            "ts_code": [f"{600000 + i}.SH" for i in range(n)],
            "neu_y_ret_20": good * 0.05 + rng.normal(scale=0.001, size=n),
            "mkt_vol_20": np.full(n, 0.2 + seed * 0.01),
            "good_feature": good,
            "noise_feature": rng.normal(size=n),
            "twin_feature": base,
            "twin_feature_sz": base + rng.normal(scale=0.01, size=n),
        }
    )
    return frame


def test_compute_daily_rank_ic_ignores_constant_columns():
    frame = _synthetic_day("20240102")
    frame["const_feature"] = 1.0
    ic = compute_daily_rank_ic(
        frame, ["good_feature", "noise_feature", "const_feature"], "neu_y_ret_20", min_pairs=30
    )
    assert ic["good_feature"] > 0.7
    assert abs(ic["noise_feature"]) < 0.3
    assert "const_feature" not in ic


def test_pick_partition_files_and_corr_dates(tmp_path):
    stems = ["20200102", "20200103", "20200106", "20200604", "20201020", "20210104"]
    for stem in stems:
        (tmp_path / f"{stem}.parquet").write_bytes(b"")
    files_2020 = pick_partition_files(tmp_path, "20200101", "20201231", every=1)
    assert [f.stem for f in files_2020] == stems[:5]
    assert [f.stem for f in pick_partition_files(tmp_path, "20200102", "20201231", 2)] == [
        "20200102",
        "20200106",
        "20201020",
    ]
    files_all = pick_partition_files(tmp_path, "20200101", "20211231", every=1)
    dates = pick_corr_dates(files_all, [2020, 2021])
    # 2021 年最早的采样月日（0415）之后无分区 -> 该年不产出采样日
    assert dates == ["20200604", "20201020"]
    assert pick_corr_dates(files_all, [2021], month_days=("0101",)) == ["20210104"]


# ---------- 汇编与候选 ----------


def test_add_year_profiles_counts_flips():
    register = pd.DataFrame(
        {
            "ic_mean": [0.02],
            "ic_2020": [0.05],
            "ndays_2020": [80],
            "ic_2021": [-0.04],
            "ndays_2021": [80],
            "ic_2022": [0.03],
            "ndays_2022": [80],
            "ic_2023": [-0.02],
            "ndays_2023": [80],
        },
        index=["f1"],
    )
    result = add_year_profiles(register, 0.01)
    assert result.loc["f1", "years_opposite"] == 2
    assert result.loc["f1", "years_flips"] == 3


def test_cluster_features_groups_twins():
    corr = pd.DataFrame(
        [[1.0, 0.99, 0.1], [0.99, 1.0, 0.1], [0.1, 0.1, 1.0]],
        index=["twin", "twin_sz", "solo"],
        columns=["twin", "twin_sz", "solo"],
    )
    scores = pd.Series({"twin": 3.0, "twin_sz": 2.0, "solo": 1.0})
    clusters = cluster_features(corr, scores, abs_threshold=0.85)
    multi = clusters[clusters["size"] == 2]
    assert len(multi) == 1
    assert multi.iloc[0]["representative"] == "twin"
    assert multi.iloc[0]["members"] == "twin|twin_sz"


def test_flag_candidates_rules_and_tables():
    register = pd.DataFrame(
        {
            "coverage": [0.2, 0.99, 0.99, 0.99],
            "coverage_min_year": [0.1, 0.99, 0.99, 0.99],
            "ic_mean": [0.01, np.nan, 0.0, 0.0],
            "ic_t": [2.0, np.nan, 0.5, 3.0],
            "gain_share_mean": [0.01, 0.01, 0.001, 0.001],
            "split_use_frac": [1.0, 1.0, 0.2, 1.0],
            "years_opposite": [0.0, np.nan, 3.0, 0.0],
            "years_flips": [0.0, np.nan, 2.0, 0.0],
            "cluster_rep": ["f1", "f2", "f3", "f4"],
        },
        index=["f1", "f2", "f3", "f4"],
    )
    flagged = flag_candidates(register, HealthThresholds())
    assert flagged.loc["f1", "flag_low_coverage"]
    assert flagged.loc["f2", "market_level"]
    assert flagged.loc["f3", "flag_weak"] and flagged.loc["f3", "flag_flip"]
    assert flagged.loc["f3", "flag_unused"]
    tables = candidate_tables(flagged)
    assert set(tables) == {"low_coverage", "sign_flip", "weak", "unused", "dedup"}
    assert len(tables["low_coverage"]) == 1  # 市场级常数不参与低覆盖清单


# ---------- 模型使用度 ----------


class _FakeBooster:
    def __init__(self, scores):
        self._scores = scores

    def get_score(self, importance_type):
        return dict(self._scores)


class _FakeModel:
    feature_names_in_ = ["alpha", "beta", "gamma"]

    def get_booster(self):
        return _FakeBooster({"f0": 3.0, "f1": 1.0})


class _FakeEnsemble:
    def __init__(self, models):
        self.models = models


def test_prefer_plain_representatives_swaps_sz_side():
    clusters = pd.DataFrame(
        [
            {
                "cluster_id": 1,
                "size": 2,
                "members": "zscore_bp|zscore_bp_sz",
                "representative": "zscore_bp_sz",
            },
            {
                "cluster_id": 2,
                "size": 2,
                "members": "zscore_fcf_yield_sz|zscore_ocf_to_profit_sz",
                "representative": "zscore_fcf_yield_sz",
            },
        ]
    )
    register = pd.DataFrame(
        {"ic_ir": [0.9, 0.5, 0.3, 0.2]},
        index=["zscore_bp_sz", "zscore_bp", "zscore_fcf_yield_sz", "zscore_ocf_to_profit_sz"],
    )
    swapped = prefer_plain_representatives(clusters, register).set_index("cluster_id")
    # 有 plain 成员的簇：代表换成 plain（即便原本 _sz 的 |ic_ir| 更高）
    assert swapped.loc[1, "representative"] == "zscore_bp"
    assert bool(swapped.loc[1, "swapped"])
    # 全为 _sz 的簇：保持不变
    assert swapped.loc[2, "representative"] == "zscore_fcf_yield_sz"
    assert not bool(swapped.loc[2, "swapped"])


def test_build_exclude_lists_includes_plain_swap():
    register = pd.DataFrame(
        {
            "flag_weak": [False, False, False, False],
            "flag_unused": [False, False, False, False],
            # 默认去重清单剔除非代表成员：此处模拟真实情形（孪生对里代表为 _sz、被删的是 plain）
            "flag_dup": [True, False, False, True],
        },
        index=["zscore_bp", "zscore_bp_sz", "lhb_on_list", "lhb_reason_count"],
    )
    clusters = pd.DataFrame(
        [
            {
                "cluster_id": 1,
                "size": 2,
                "members": "zscore_bp|zscore_bp_sz",
                "representative": "zscore_bp_sz",
            },
            {
                "cluster_id": 2,
                "size": 2,
                "members": "lhb_on_list|lhb_reason_count",
                "representative": "lhb_on_list",
            },
        ]
    )
    register["ic_ir"] = [0.6, 0.7, 0.5, 0.4]
    lists = build_exclude_lists(register, clusters)
    assert set(lists) == {
        "exclude_weak_v1",
        "exclude_dedup_v1",
        "exclude_weak_dedup_v1",
        "exclude_dedup_plain_v1",
    }
    assert lists["exclude_dedup_v1"]["factors"] == ["lhb_reason_count", "zscore_bp"]
    # 口径交换：孪生对改为剔除 _sz 一侧（lhb 簇保持默认代表）
    assert lists["exclude_dedup_plain_v1"]["factors"] == ["lhb_reason_count", "zscore_bp_sz"]


def test_compute_model_usage_with_fake_loader(tmp_path):
    (tmp_path / "v2_model.joblib").write_bytes(b"")
    usage = compute_model_usage(
        tmp_path,
        [1, 2],
        ["alpha", "beta", "gamma"],
        loader=lambda path: _FakeEnsemble([_FakeModel()]),
    )
    assert usage.attrs["loaded_versions"] == [2]
    assert usage.attrs["sub_model_count"] == 1
    assert usage.loc["alpha", "gain_share_mean"] == 0.75
    assert usage.loc["beta", "gain_share_mean"] == 0.25
    assert usage.loc["gamma", "split_use_frac"] == 0.0


# ---------- 端到端 ----------


def _write_synthetic_partitions(root, dates):
    for seed, date in enumerate(dates):
        _synthetic_day(date, seed=seed).to_parquet(root / f"{date}.parquet", index=False)


def test_end_to_end_scan_register_and_outputs(tmp_path):
    cs_train = tmp_path / "cs_train"
    cs_train.mkdir()
    dates = ["20200102", "20200103", "20200106", "20210104", "20210105", "20210106"]
    _write_synthetic_partitions(cs_train, dates)

    features = ["good_feature", "noise_feature", "twin_feature", "twin_feature_sz", "mkt_vol_20"]
    files = pick_partition_files(cs_train, "20200101", "20211231", every=1)
    corr_dates = pick_corr_dates(files, [2020, 2021])

    scan_result = scan_features(
        files=files,
        feature_names=features,
        label_column="neu_y_ret_20",
        market_vol_column="mkt_vol_20",
        corr_dates=corr_dates,
        min_pairs=30,
        corr_min_periods=20,
        progress_every=0,
    )
    assert scan_result.corr_avg is not None
    assert len(scan_result.daily_ic) > 0

    register = assemble_register(
        scan_result.coverage_by_date, scan_result.daily_ic, scan_result.market_vol_by_date
    )
    register = add_year_profiles(register, 0.01)
    register = attach_twin_info(register)
    register = attach_families(register)
    clusters = cluster_features(scan_result.corr_avg, register["ic_ir"], 0.85)
    register = attach_clusters(register, clusters)
    register["gain_share_mean"] = 0.01
    register["split_use_frac"] = 1.0
    register = flag_candidates(register, HealthThresholds())

    assert register.loc["mkt_vol_20", "market_level"]
    assert not register.loc["good_feature", "market_level"]
    dup_cluster = register.loc[["twin_feature", "twin_feature_sz"], "cluster_id"]
    assert dup_cluster.nunique() == 1
    assert int(register["flag_dup"].sum()) == 1

    summary = summarize(register, clusters)
    assert summary["twin_pairs"] == 1

    candidates = candidate_tables(register)
    report = build_report_markdown(
        register,
        clusters,
        HealthThresholds(),
        {
            "feature_file": "synthetic",
            "start": dates[0],
            "end": dates[-1],
            "every": 1,
            "file_count": len(files),
            "label": "neu_y_ret_20",
            "code_count": 80,
            "mainboard_filter": False,
            "model_versions": [],
            "sub_model_count": 0,
            "out_dir": str(tmp_path / "out"),
        },
        candidates,
    )
    assert "# 选股因子体检报告" in report

    out_dir = tmp_path / "out"
    write_health_outputs(
        out_dir,
        register,
        scan_result.daily_ic,
        scan_result.corr_avg,
        clusters,
        candidates,
        report,
        build_exclude_lists(register, clusters),
    )
    for name in (
        "factor_register.csv",
        "daily_ic.csv.gz",
        "corr_matrix.csv",
        "clusters.csv",
        "factor_health_report.md",
        "exclude_weak_v1.json",
        "exclude_dedup_v1.json",
        "exclude_weak_dedup_v1.json",
        "exclude_dedup_plain_v1.json",
    ):
        assert (out_dir / name).exists(), name
    payload = json.loads((out_dir / "exclude_dedup_v1.json").read_text(encoding="utf-8"))
    assert payload["exclude_count"] == len(payload["exclude_factors"]) == 1
    plain_payload = json.loads(
        (out_dir / "exclude_dedup_plain_v1.json").read_text(encoding="utf-8")
    )
    # 孪生对（twin_feature / twin_feature_sz）在口径交换清单中应剔除 _sz 一侧
    assert plain_payload["exclude_factors"] == ["twin_feature_sz"]


def test_average_correlation_handles_empty_and_nan():
    assert average_correlation([]) is None
    first = pd.DataFrame([[1.0, np.nan], [np.nan, 1.0]], index=["a", "b"], columns=["a", "b"])
    averaged = average_correlation([first, first])
    assert averaged.loc["a", "b"] == 0.0
