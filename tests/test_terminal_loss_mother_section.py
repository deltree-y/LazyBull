"""pct_* 完整母截面重建测试（合成数据，不依赖真实数据）"""

import numpy as np
import pandas as pd
import pytest

from src.lazybull.risk.terminal_loss import (
    MOTHER_HARD_ATOL,
    MOTHER_OUTLIER_SHARE_LIMIT,
    MOTHER_SECTION_FACTORS,
    assert_mother_section_validation_ok,
    build_mother_section,
    load_clean_daily_long,
    merge_mother_section_validation,
    validate_mother_section_against_cs_train,
)

N_DAYS = 45
CODES = ["000001.SZ", "000002.SZ", "000003.SZ"]


def _write_env(tmp_path, drop_columns=()):
    """构造 clean/trade_cal + clean/daily 合成目录。"""
    dates = pd.bdate_range("2024-01-02", periods=N_DAYS)
    cal = [d.strftime("%Y%m%d") for d in dates]
    clean_dir = tmp_path / "clean"
    daily_dir = clean_dir / "daily"
    daily_dir.mkdir(parents=True)
    pd.DataFrame({"exchange": "SSE", "cal_date": cal, "is_open": 1}).to_parquet(
        clean_dir / "trade_cal.parquet"
    )

    rng = np.random.default_rng(7)
    prices = {c: 100.0 for c in CODES}
    for d in cal:
        rows = []
        for c in CODES:
            ret = float(rng.normal(0, 0.02))
            prices[c] = prices[c] * (1 + ret)
            close = prices[c]
            open_ = close / (1 + ret)
            row = {
                "ts_code": c,
                "trade_date": d,
                "close_adj": close,
                "open_adj": open_,
                "high_adj": max(open_, close) * 1.01,
                "low_adj": min(open_, close) * 0.99,
                "vol": 1_000_000.0 + rng.random() * 1000,
                "amount": 100_000.0 + rng.random() * 1000,
            }
            for col in drop_columns:
                row.pop(col, None)
            rows.append(row)
        pd.DataFrame(rows).to_parquet(daily_dir / f"{d[:4]}-{d[4:6]}-{d[6:8]}.parquet")
    return cal


def _write_suspension_env(tmp_path):
    """构造"停牌复牌"合成环境：单只股票停牌 16 个交易日后复牌。

    价格路径：day 9 相对 day 8 跌 30%；day 10~25 无行（停牌）；day 26 复牌
    相对 day 9（停牌前最后一行）跌 35%；其后为小幅波动。目标日 = day 44。

    Returns:
        (cal, resume_ret)：交易日列表与复牌日相对停牌前一行的收益
    """
    dates = pd.bdate_range("2024-01-02", periods=N_DAYS)
    cal = [d.strftime("%Y%m%d") for d in dates]
    clean_dir = tmp_path / "clean"
    daily_dir = clean_dir / "daily"
    daily_dir.mkdir(parents=True)
    pd.DataFrame({"exchange": "SSE", "cal_date": cal, "is_open": 1}).to_parquet(
        clean_dir / "trade_cal.parquet"
    )

    code = CODES[0]
    other = CODES[1]
    gap = range(10, 26)
    close = 100.0
    rows_by_date = {}
    for i, d in enumerate(cal):
        if i in gap:
            # 停牌期间该股无行，但其它股票仍有行（真实链路：分区不会整天为空）
            rows_by_date[d] = pd.DataFrame(
                [
                    {
                        "ts_code": other,
                        "trade_date": d,
                        "close_adj": 50.0,
                        "open_adj": 50.0,
                        "high_adj": 50.5,
                        "low_adj": 49.5,
                        "vol": 1_000_000.0,
                        "amount": 100_000.0,
                    }
                ]
            )
            continue
        if i == 9:
            close = close * 0.70
        elif i == 26:
            close = close * 0.65
        elif i > 26:
            close = close * 1.01
        open_ = close * 0.995
        rows_by_date[d] = pd.DataFrame(
            [
                {
                    "ts_code": code,
                    "trade_date": d,
                    "close_adj": close,
                    "open_adj": open_,
                    "high_adj": max(open_, close) * 1.005,
                    "low_adj": min(open_, close) * 0.995,
                    "vol": 1_000_000.0,
                    "amount": 100_000.0,
                }
            ]
        )
    for d, frame in rows_by_date.items():
        frame.to_parquet(daily_dir / f"{d[:4]}-{d[4:6]}-{d[6:8]}.parquet")
    return cal, -0.35


def _write_cs_train(tmp_path, dates, mother, offset=0.0):
    """按母截面写 cs_train 分区（offset 用于制造不一致）。"""
    cs_dir = tmp_path / "features" / "cs_train"
    cs_dir.mkdir(parents=True, exist_ok=True)
    for d in dates:
        frame = mother[d].copy()
        for col in MOTHER_SECTION_FACTORS:
            frame[col] = frame[col] + offset
        frame.to_parquet(cs_dir / f"{d}.parquet")


class TestBuildMotherSection:
    def test_returns_all_targets_with_four_factors(self, tmp_path):
        cal = _write_env(tmp_path)
        targets = cal[30:33]
        mother = build_mother_section(str(tmp_path), targets)
        assert sorted(mother) == sorted(targets)
        for frame in mother.values():
            assert list(frame.columns) == ["ts_code"] + MOTHER_SECTION_FACTORS
            assert len(frame) == len(CODES)

    def test_factor_values_reproduce_feature_pipeline(self, tmp_path):
        """母截面必须复现 cs_train 中同名因子的取值（同一实现）。"""
        cal = _write_env(tmp_path)
        targets = cal[30:33]
        mother = build_mother_section(str(tmp_path), targets)
        _write_cs_train(tmp_path, targets, mother)
        stats = validate_mother_section_against_cs_train(str(tmp_path), mother, targets)
        assert stats["checked_dates"] == 3
        assert all(v == 0.0 for v in stats["max_abs_diff"].values())

    def test_universe_is_quoted_stocks_of_day(self, tmp_path):
        """证券域 = 当日 clean/daily 有行的股票（PIT 口径，不含退市/停牌股）。"""
        cal = _write_env(tmp_path)
        path = tmp_path / "clean" / "daily" / f"{cal[31][:4]}-{cal[31][4:6]}-{cal[31][6:8]}.parquet"
        day = pd.read_parquet(path)
        day = day[day["ts_code"] != CODES[0]]
        day.to_parquet(path)
        mother = build_mother_section(str(tmp_path), [cal[31]])
        assert list(mother[cal[31]]["ts_code"]) == CODES[1:]

    def test_ret20_available_after_warmup(self, tmp_path):
        cal = _write_env(tmp_path)
        mother = build_mother_section(str(tmp_path), [cal[-1]])
        assert mother[cal[-1]]["ret_20"].notna().all()

    def test_missing_daily_partition_raises(self, tmp_path):
        cal = _write_env(tmp_path)
        path = tmp_path / "clean" / "daily" / f"{cal[31][:4]}-{cal[31][4:6]}-{cal[31][6:8]}.parquet"
        path.unlink()
        with pytest.raises(FileNotFoundError, match="缺少 clean/daily 分区"):
            build_mother_section(str(tmp_path), [cal[31]])

    def test_missing_required_column_raises(self, tmp_path):
        cal = _write_env(tmp_path, drop_columns=("close_adj",))
        with pytest.raises(ValueError, match="缺少母截面重建必需列"):
            load_clean_daily_long(str(tmp_path), [cal[31]])

    def test_vol_amount_are_required_for_ret20(self, tmp_path):
        """vol/amount 是 ret_20 共享实现的必需输入：缺列必须报错不静默降级。"""
        cal = _write_env(tmp_path, drop_columns=("vol",))
        with pytest.raises(ValueError, match="缺少母截面重建必需列"):
            load_clean_daily_long(str(tmp_path), [cal[31]])

    def test_optional_price_columns_absent_is_tolerated(self, tmp_path):
        """开高低价为可选列：缺失不影响四个基列的计算与证券域。"""
        cal = _write_env(tmp_path, drop_columns=("high_adj", "low_adj", "open_adj"))
        mother = build_mother_section(str(tmp_path), [cal[31]])
        frame = mother[cal[31]]
        assert len(frame) == len(CODES)
        assert frame["ret_20"].notna().all()

    def test_resume_day_return_uses_pre_gap_row(self, tmp_path, monkeypatch):
        """停牌复牌日的收益必须取停牌前最后一行（历史窗口与特征流水线一致）。

        收益按股票自身可用行计算：停牌期间无行，复牌日的收益要拿停牌前那一行。
        旧 30 交易日窗口会把该行切出窗外 → 复牌日收益在母截面为 NaN，而 cs_train
        中有效（流水线窗口为 7 个月）→ 被误判成数据态漂移。实测触发：300630.SZ
        停牌 20240430→20240708，母截面 cvar_95_20=-0.196356 vs cs_train=-0.2。
        """
        import src.lazybull.risk.terminal_loss.mother_section as mother_mod

        cal, resume_ret = _write_suspension_env(tmp_path)
        target = cal[-1]

        mother = build_mother_section(str(tmp_path), [target])
        frame = mother[target]
        value = frame.loc[frame["ts_code"] == CODES[0], "cvar_95_20"].iloc[0]
        assert value == pytest.approx(resume_ret, abs=1e-9)

        # 反证：把历史窗口缩到 1 个月（旧口径的等效效果）→ 停牌前行在窗外，
        # 复牌日收益消失，最差收益退化为窗口内其它交易日。
        monkeypatch.setattr(mother_mod, "MOTHER_HISTORY_MONTHS", 1)
        weak = build_mother_section(str(tmp_path), [target])
        weak_frame = weak[target]
        weak_value = weak_frame.loc[weak_frame["ts_code"] == CODES[0], "cvar_95_20"].iloc[0]
        assert not np.isclose(weak_value, resume_ret, atol=1e-3)


def _validation_stub(
    checked_rows, checked_dates, max_diff, outlier_total, outlier_dates, outlier_rows=None
):
    """构造单块校验统计（用于窗口级聚合的单元测试）。"""
    return {
        "checked_dates": checked_dates,
        "checked_rows": checked_rows,
        "max_abs_diff": max_diff,
        "atol": 1e-6,
        "outliers": {
            "total": outlier_total,
            "share": outlier_total / checked_rows if checked_rows else 0.0,
            "rows": outlier_rows or {"cvar_95_20": outlier_total},
            "dates": outlier_dates,
            "examples": [],
            "limits": {
                "hard_atol": MOTHER_HARD_ATOL,
                "share_limit": MOTHER_OUTLIER_SHARE_LIMIT,
            },
        },
    }


def _validated(tmp_path, mother, targets, **kwargs):
    """逐块统计 → 窗口级合并（训练脚本的实际调用方式）。"""
    return merge_mother_section_validation(
        [validate_mother_section_against_cs_train(str(tmp_path), mother, targets, **kwargs)]
    )


class TestValidateAgainstCsTrain:
    def test_exact_match_registers_zero_outliers(self, tmp_path):
        cal = _write_env(tmp_path)
        targets = cal[30:33]
        mother = build_mother_section(str(tmp_path), targets)
        _write_cs_train(tmp_path, targets, mother)
        merged = _validated(tmp_path, mother, targets)
        assert merged["checked_dates"] == 3
        assert all(v == 0.0 for v in merged["max_abs_diff"].values())
        assert merged["outliers"]["total"] == 0
        assert assert_mother_section_validation_ok(merged)["checked_rows"] == 9

    def test_systematic_drift_raises(self, tmp_path):
        """全部行偏移（如换了一套实现）→ 判为系统性差异，必须报错。"""
        cal = _write_env(tmp_path)
        targets = cal[30:31]
        mother = build_mother_section(str(tmp_path), targets)
        _write_cs_train(tmp_path, targets, mother, offset=1e-3)
        merged = _validated(tmp_path, mother, targets)
        with pytest.raises(ValueError, match="差异呈系统性"):
            assert_mother_section_validation_ok(merged)

    def test_large_magnitude_raises_even_when_sparse(self, tmp_path):
        """幅度硬上限：稀疏但幅度巨大（超过 hard_atol）仍须报错。"""
        cal = _write_env(tmp_path)
        targets = cal[30:33]
        mother = build_mother_section(str(tmp_path), targets)
        cs_dir = tmp_path / "features" / "cs_train"
        cs_dir.mkdir(parents=True, exist_ok=True)
        for d in targets:
            frame = mother[d].copy()
            frame.loc[0, "cvar_95_20"] += 0.5
            frame.to_parquet(cs_dir / f"{d}.parquet")
        merged = _validated(tmp_path, mother, targets)
        with pytest.raises(ValueError, match="差异呈系统性"):
            assert_mother_section_validation_ok(merged)

    def test_single_stock_window_drift_tolerated_and_registered(self, tmp_path):
        """数据态漂移（同一只股票连续多个交易日）：不阻断训练，但必须登记。

        生产阈值（行占比 1e-4）按真实规模（十万行级交集）设定，小样本 fixture
        无法自然落入区间，故显式放宽阈值以覆盖该分支；生产阈值的严格性由
        test_production_limits_reject_small_fixture_drift 锁定。
        """
        cal = _write_env(tmp_path)
        targets = cal[30:33]
        mother = build_mother_section(str(tmp_path), targets)
        cs_dir = tmp_path / "features" / "cs_train"
        cs_dir.mkdir(parents=True, exist_ok=True)
        for d in targets:
            frame = mother[d].copy()
            frame.loc[0, "cvar_95_20"] += 1e-3
            frame.to_parquet(cs_dir / f"{d}.parquet")
        merged = _validated(tmp_path, mother, targets, outlier_share_limit=0.5)
        outliers = merged["outliers"]
        assert outliers["total"] == 3  # 每日 1 行
        assert outliers["dates"] == 3
        assert outliers["rows"]["cvar_95_20"] == 3
        assert len(outliers["examples"]) == 3
        assert outliers["examples"][0]["date"] in targets
        assert outliers["examples"][0]["column"] == "cvar_95_20"
        # cs_train 落盘为 float32，偏移量会有浮点误差
        assert outliers["examples"][0]["abs_diff"] == pytest.approx(1e-3, rel=1e-5)
        assert assert_mother_section_validation_ok(merged) is merged

    def test_production_limits_reject_small_fixture_drift(self, tmp_path):
        """同一单点漂移在小样本 fixture 下会被生产阈值拦下（阈值未被偷偷放宽）。"""
        cal = _write_env(tmp_path)
        targets = cal[30:33]
        mother = build_mother_section(str(tmp_path), targets)
        cs_dir = tmp_path / "features" / "cs_train"
        cs_dir.mkdir(parents=True, exist_ok=True)
        for d in targets:
            frame = mother[d].copy()
            frame.loc[0, "cvar_95_20"] += 1e-3
            frame.to_parquet(cs_dir / f"{d}.parquet")
        merged = _validated(tmp_path, mother, targets)
        with pytest.raises(ValueError, match="差异呈系统性"):
            assert_mother_section_validation_ok(merged)

    def test_missing_cs_train_partition_raises(self, tmp_path):
        cal = _write_env(tmp_path)
        targets = cal[30:31]
        mother = build_mother_section(str(tmp_path), targets)
        with pytest.raises(FileNotFoundError, match="缺少 cs_train 分区"):
            validate_mother_section_against_cs_train(str(tmp_path), mother, targets)

    def test_no_intersection_raises(self, tmp_path):
        """没有任何交集行时不得静默通过（空校验等于没校验）。"""
        cal = _write_env(tmp_path)
        with pytest.raises(ValueError, match="没有任何交集行"):
            validate_mother_section_against_cs_train(str(tmp_path), {}, cal[30:31])

    def test_empty_mother_frame_is_skipped(self, tmp_path):
        """空母截面的日期跳过统计（其他日期仍正常校验，不触发无交集报错）。"""
        cal = _write_env(tmp_path)
        targets = [cal[30]]
        mother = build_mother_section(str(tmp_path), targets)
        _write_cs_train(tmp_path, targets, mother)
        mother["20990101"] = mother[cal[30]].iloc[0:0]
        stats = validate_mother_section_against_cs_train(
            str(tmp_path), mother, targets + ["20990101"]
        )
        assert stats["checked_dates"] == 1


class TestWindowLevelValidation:
    """窗口级聚合判定：块内占比不得直接当判定口径（块粒度伪影）。"""

    def test_merge_recomputes_share_over_window(self):
        """单块 1% 的离群占比，在全窗口（总行数大得多）下应被稀释。"""
        chunk_a = _validation_stub(100, 1, {"cvar_95_20": 1e-3}, 1, 1)
        chunk_b = _validation_stub(100_000, 1, {"cvar_95_20": 0.0}, 0, 0)
        merged = merge_mother_section_validation([chunk_a, chunk_b])
        assert merged["checked_rows"] == 100_100
        assert merged["checked_dates"] == 2
        assert merged["outliers"]["dates"] == 1
        assert merged["outliers"]["share"] == pytest.approx(1 / 100_100)
        # 块 A 单独看行占比 1e-2 已超生产阈值，聚合后被稀释到阈值内
        assert chunk_a["outliers"]["share"] > MOTHER_OUTLIER_SHARE_LIMIT
        assert merged["outliers"]["share"] < MOTHER_OUTLIER_SHARE_LIMIT

    def test_merge_takes_column_max(self):
        a = _validation_stub(10, 1, {"cvar_95_20": 1e-3, "ret_20": 0.0}, 0, 0)
        b = _validation_stub(10, 1, {"cvar_95_20": 2e-4, "ret_20": 5e-4}, 0, 0)
        merged = merge_mother_section_validation([a, b])
        assert merged["max_abs_diff"]["cvar_95_20"] == pytest.approx(1e-3)
        assert merged["max_abs_diff"]["ret_20"] == pytest.approx(5e-4)

    def test_merge_empty_parts_raises(self):
        with pytest.raises(ValueError, match="校验结果为空"):
            merge_mother_section_validation([])

    def test_assert_raises_when_row_share_exceeds_limit(self):
        """行占比超限（大面积不同源，实现漂移的特征）→ 报错。"""
        merged = merge_mother_section_validation(
            [_validation_stub(1000, 50, {"cvar_95_20": 1e-3}, 5, 5)]
        )
        assert merged["outliers"]["share"] > MOTHER_OUTLIER_SHARE_LIMIT
        with pytest.raises(ValueError, match="差异呈系统性"):
            assert_mother_section_validation_ok(merged)

    def test_assert_tolerates_sparse_drift_even_across_many_dates(self):
        """稀疏数据态漂移不因"涉及日期多"而报错（日占比不是判据）。

        实测形态：同一只股票连续 10 个交易日各 1 行；若按日占比判定，
        10/123 日 = 8% 会误报，而行占比 1.4e-5 才是真实稀疏度。
        """
        parts = [_validation_stub(700_000, 50, {"cvar_95_20": 1e-3}, 10, 10)]
        merged = merge_mother_section_validation(parts)
        assert merged["outliers"]["date_share"] > 0.05  # 日占比确实超 5%
        assert merged["outliers"]["share"] < MOTHER_OUTLIER_SHARE_LIMIT
        assert assert_mother_section_validation_ok(merged)["outliers"]["total"] == 10
