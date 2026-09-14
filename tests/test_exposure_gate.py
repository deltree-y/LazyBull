"""E2 暴露门控（policy exposure gate）单元测试：规则、校准、契约与中文表头。

测试全部基于合成台账，不依赖真实数据与真实配置。
"""

import numpy as np
import pandas as pd
import pytest

from src.lazybull.common.sidecar_schema import (
    GATE_CALIBRATION_COLUMNS_ZH,
    GATE_DAILY_COLUMNS_ZH,
    GATE_EVAL_COLUMNS_ZH,
    select_chinese,
)
from src.lazybull.risk.terminal_loss.exposure_gate import (
    ARMS,
    ExposureGateConfig,
    RollingGateConfig,
    _CALIBRATION_KEYS,
    _DAILY_KEYS,
    _EVAL_KEYS,
    apply_gate,
    apply_rolling_gate,
    assert_disjoint,
    build_daily_frame,
    calibrate_gate,
    evaluate_gate,
    evaluate_gate_by_fold,
    read_ledger_frames,
    run_exposure_gate,
    run_rolling_gate,
)


def make_ledger(days: int = 40, per_day: int = 3) -> pd.DataFrame:
    """构造合成台账（按月推进的合法日期）。

    按 4 日循环铺出 2×2 结构（与 2026-09-14 实测的坏格模式一致）::

        i%4 == 0  低波动 + 低分  → 事后收益 +0.010
        i%4 == 1  低波动 + 高分  → +0.005
        i%4 == 2  高波动 + 低分  → +0.020（高波动里的好格）
        i%4 == 3  高波动 + 高分  → -0.030（唯一坏格，异常亏损）

    这样三臂的触发集合互不相同，可直接验证"只有 E2 臂命中坏格"。
    """
    profile = {
        0: (0.02, 0.10, 0.010, 0.0),
        1: (0.02, 0.20, 0.005, 0.0),
        2: (0.05, 0.10, 0.020, 0.0),
        3: (0.05, 0.20, -0.030, 1.0),
    }
    rows = []
    year, month = 2023, 1
    for i in range(days):
        date = f"{year}{month:02d}01"
        vol, score, ret, label = profile[i % 4]
        for j in range(per_day):
            rows.append(
                {
                    "日期": date,
                    "风险模型折": "折A" if i < days // 2 else "折B",
                    "股票代码": f"60000{j}.SH",
                    "持仓权重": 0.1,
                    "风险概率": score,
                    "市场波动状态": vol,
                    "事后实际收益": ret,
                    "是否异常亏损": label,
                    "标签状态": "valid",
                }
            )
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return pd.DataFrame(rows)


def write_ledger(tmp_path, frame: pd.DataFrame):
    path = tmp_path / "风险台账.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def test_schema_keys_match_chinese_headers():
    """三张表的中文表头与内部键数量一致，且可无损映射。"""
    assert len(GATE_CALIBRATION_COLUMNS_ZH) == len(_CALIBRATION_KEYS)
    assert len(GATE_DAILY_COLUMNS_ZH) == len(_DAILY_KEYS)
    assert len(GATE_EVAL_COLUMNS_ZH) == len(_EVAL_KEYS)
    for keys, chinese in (
        (_CALIBRATION_KEYS, GATE_CALIBRATION_COLUMNS_ZH),
        (_DAILY_KEYS, GATE_DAILY_COLUMNS_ZH),
        (_EVAL_KEYS, GATE_EVAL_COLUMNS_ZH),
    ):
        frame = pd.DataFrame([{k: 1 for k in keys}])
        out = select_chinese(frame, dict(zip(keys, chinese)), keys)
        assert list(out.columns) == chinese


def test_schema_missing_column_raises():
    """中文表头映射缺列必须报错（禁止静默丢列）。"""
    frame = pd.DataFrame([{k: 1 for k in _EVAL_KEYS[:-1]}])
    with pytest.raises(ValueError, match="缺少内部列"):
        select_chinese(frame, dict(zip(_EVAL_KEYS, GATE_EVAL_COLUMNS_ZH)), _EVAL_KEYS)


def test_config_rejects_invalid_values():
    """参数校验：臂、分位、暴露系数、成本、层下限。"""
    with pytest.raises(ValueError, match="未知臂"):
        ExposureGateConfig(arm="nope")
    with pytest.raises(ValueError, match="regime_quantile"):
        ExposureGateConfig(regime_quantile=1.5)
    with pytest.raises(ValueError, match="score_quantile"):
        ExposureGateConfig(arm="combined", score_quantile=0.0)
    with pytest.raises(ValueError, match="de_exposure_multiplier"):
        ExposureGateConfig(de_exposure_multiplier=0.0)
    with pytest.raises(ValueError, match="de_exposure_multiplier"):
        ExposureGateConfig(de_exposure_multiplier=1.2)
    with pytest.raises(ValueError, match="cost_bps"):
        ExposureGateConfig(cost_bps=-1.0)
    with pytest.raises(ValueError, match="min_layer_days"):
        ExposureGateConfig(min_layer_days=0)


def test_read_ledger_filters_invalid_labels_and_normalizes_dates(tmp_path):
    """非 valid 标签行必须剔除；日期规范为 YYYYMMDD。"""
    frame = make_ledger(days=3)
    frame.loc[frame.index[:2], "标签状态"] = "sigma_unavailable"
    frame.loc[frame.index[0], "日期"] = "2023-01-01"
    ledger = read_ledger_frames([write_ledger(tmp_path, frame)])
    assert len(ledger) == len(frame) - 2
    assert ledger["date"].str.len().eq(8).all()
    assert set(ledger["label_status"]) == {"valid"}


def test_read_ledger_missing_column_raises(tmp_path):
    """台账缺列必须明确失败。"""
    frame = make_ledger(days=2).drop(columns=["市场波动状态"])
    with pytest.raises(ValueError, match="缺少台账列"):
        read_ledger_frames([write_ledger(tmp_path, frame)])


def test_read_ledger_all_invalid_raises(tmp_path):
    """全部行标签非 valid 时必须报错，不得产出空评估。"""
    frame = make_ledger(days=2)
    frame["标签状态"] = "immature"
    with pytest.raises(ValueError, match="全部行标签非 valid"):
        read_ledger_frames([write_ledger(tmp_path, frame)])


def test_build_daily_frame_weighted_return_and_regime_uniqueness(tmp_path):
    """日级加权收益 = Σw·r / Σw；同日 regime 不唯一必须报错。"""
    frame = make_ledger(days=2, per_day=2)
    frame.loc[frame.index[0], "事后实际收益"] = 0.05
    frame.loc[frame.index[1], "事后实际收益"] = -0.01
    daily = build_daily_frame(read_ledger_frames([write_ledger(tmp_path, frame)]))
    assert len(daily) == 2
    first = daily.iloc[0]
    assert first["day_weighted_return"] == pytest.approx((0.05 - 0.01) / 2)
    assert first["holdings"] == 2
    assert bool(first["loss_day"]) is False

    frame.loc[frame.index[1], "市场波动状态"] = 999.0
    with pytest.raises(ValueError, match="市场波动状态不唯一"):
        build_daily_frame(read_ledger_frames([write_ledger(tmp_path, frame)]))


def test_segments_must_be_disjoint():
    """校准段与评估段重叠必须报错（禁止用评估段信息拟合阈值）。"""
    with pytest.raises(ValueError, match="重叠"):
        assert_disjoint("20220701", "20231231", "20231231", "20251204")
    assert_disjoint("20220701", "20231231", "20240101", "20251204")


def test_calibrate_thresholds_are_quantiles_of_calibration_segment(tmp_path):
    """阈值必须取自校准段分位：regime 全段（所有臂）、combined 层内、score 全段。"""
    ledger = read_ledger_frames([write_ledger(tmp_path, make_ledger(days=120))])
    daily = build_daily_frame(ledger)
    calib_end = daily["date"].iloc[79]
    segment = daily[daily["date"] <= calib_end]
    layer = segment[segment["mkt_vol_20"] >= segment["mkt_vol_20"].quantile(2.0 / 3.0)]
    assert len(layer) >= 20  # 合成数据规模足以覆盖层内下限

    for arm in ARMS:
        calib = calibrate_gate(daily, ExposureGateConfig(arm=arm), daily["date"].iloc[0], calib_end)
        assert calib.calib_days == len(segment)
        assert calib.arm == arm
        assert calib.layer_days == len(layer)
        # regime 阈值对所有臂都拟合（仅用于分层报告）
        assert calib.regime_quantile == pytest.approx(2.0 / 3.0)
        assert calib.regime_threshold == pytest.approx(
            segment["mkt_vol_20"].quantile(2.0 / 3.0)
        )
        if arm == "score":
            assert calib.score_quantile == pytest.approx(0.5)
            assert calib.score_threshold == pytest.approx(segment["p_loss_mean"].quantile(0.5))
        elif arm == "combined":
            assert calib.score_threshold == pytest.approx(layer["p_loss_mean"].quantile(0.5))
        else:
            assert calib.score_threshold is None
            assert calib.score_quantile is None
        assert 0.0 <= calib.calib_trigger_share <= 1.0


def test_calibrate_rejects_small_layer(tmp_path):
    """层内校准日数不足必须报错（阈值不可靠）。"""
    ledger = read_ledger_frames([write_ledger(tmp_path, make_ledger(days=10))])
    daily = build_daily_frame(ledger)
    with pytest.raises(ValueError, match="层内日数"):
        calibrate_gate(
            daily,
            ExposureGateConfig(arm="combined", min_layer_days=50),
            daily["date"].iloc[0],
            daily["date"].iloc[-1],
        )


def test_calibrate_empty_segment_raises(tmp_path):
    """校准段无交易日必须报错。"""
    ledger = read_ledger_frames([write_ledger(tmp_path, make_ledger(days=5))])
    daily = build_daily_frame(ledger)
    with pytest.raises(ValueError, match="无交易日"):
        calibrate_gate(daily, ExposureGateConfig(), "20100101", "20110101")


def test_arm_trigger_truth_table(tmp_path):
    """三臂触发逻辑真值表：regime 只看波动、score 只看分数、combined 取交。"""
    frame = make_ledger(days=6, per_day=1)
    frame["市场波动状态"] = [0.01, 0.01, 0.05, 0.05, 0.01, 0.05]
    frame["风险概率"] = [0.10, 0.20, 0.10, 0.20, 0.15, 0.15]
    ledger = read_ledger_frames([write_ledger(tmp_path, frame)])
    daily = build_daily_frame(ledger)
    start, end = daily["date"].iloc[0], daily["date"].iloc[-1]

    expect = {
        "regime": [False, False, True, True, False, True],
        "score": [False, True, False, True, True, True],
        # 层内得分中位 = 0.15（idx2=0.10 落选，idx5=0.15 命中）
        "combined": [False, False, False, True, False, True],
    }
    for arm, flags in expect.items():
        calib = calibrate_gate(
            daily,
            ExposureGateConfig(arm=arm, regime_quantile=0.5, score_quantile=0.5, min_layer_days=1),
            start,
            end,
        )
        judged = apply_gate(daily, calib, start, end)
        assert judged["triggered"].tolist() == flags, arm


def test_first_trigger_marks_run_start_only(tmp_path):
    """首触只标记触发区间第一天（折内），跨折边界按折内首日重计。"""
    frame = make_ledger(days=6, per_day=1)
    frame["市场波动状态"] = [0.01, 0.05, 0.05, 0.05, 0.01, 0.05]
    frame["风险概率"] = 0.2
    ledger = read_ledger_frames([write_ledger(tmp_path, frame)])
    daily = build_daily_frame(ledger)
    start, end = daily["date"].iloc[0], daily["date"].iloc[-1]
    # 合成数据的 风险模型折 在第 4 天切换（折A → 折B），触发区间不跨折延续
    calib = calibrate_gate(daily, ExposureGateConfig(arm="regime", regime_quantile=0.4), start, end)
    judged = apply_gate(daily, calib, start, end)
    assert judged["triggered"].tolist() == [False, True, True, True, False, True]
    assert judged["first_trigger"].tolist() == [False, True, False, True, False, True]
    assert judged["exposure_multiplier"].tolist() == [1.0, 0.5, 0.5, 0.5, 1.0, 0.5]


def test_gate_evaluation_direction_and_cost(tmp_path):
    """E2 臂在合成坏格上应给出正收益项；成本项按首触计一次往返。"""
    ledger = read_ledger_frames([write_ledger(tmp_path, make_ledger(days=120))])
    daily = build_daily_frame(ledger)
    calib_end = daily["date"].iloc[79]
    eval_start, eval_end = daily["date"].iloc[80], daily["date"].iloc[-1]

    calib = calibrate_gate(
        daily, ExposureGateConfig(arm="combined"), daily["date"].iloc[0], calib_end
    )
    judged = apply_gate(daily, calib, eval_start, eval_end)
    evaluated = evaluate_gate(judged, calib.de_exposure_multiplier, cost_bps=15.0)
    overall = evaluated[(evaluated["scope"] == "全部触发日") & (evaluated["group"] == "全部")]
    assert len(overall) == 1
    assert set(evaluated["fold"]) == {"全部"}
    by_fold = evaluate_gate_by_fold(judged, calib.de_exposure_multiplier, cost_bps=15.0)
    assert set(by_fold["fold"]) != {"全部"}
    row = overall.iloc[0]
    # 合成数据里"波动高 × 得分高"日收益为负 → 触发日平均收益更低、收益项为正
    assert row["trigger_mean_return"] < row["nontrigger_mean_return"]
    assert row["gain_term"] > 0
    assert row["net_daily_delta"] == pytest.approx(row["gain_term"] - row["cost_term"])
    assert row["cost_term"] == pytest.approx(
        0.5 * 2.0 * 15.0 / 1e4 * row["first_trigger_share"]
    )
    assert set(evaluated["group"]) == {"全部", "高波动层", "低波动层"}
    assert set(evaluated["scope"]) == {"全部触发日", "首触日"}


def test_regime_arm_has_no_within_layer_contrast(tmp_path):
    """纯 regime 臂在高波动层内触发占比必为 1（层内没有对照）。"""
    ledger = read_ledger_frames([write_ledger(tmp_path, make_ledger(days=120))])
    daily = build_daily_frame(ledger)
    calib_end, eval_start = daily["date"].iloc[79], daily["date"].iloc[80]
    calib = calibrate_gate(
        daily, ExposureGateConfig(arm="regime"), daily["date"].iloc[0], calib_end
    )
    judged = apply_gate(daily, calib, eval_start, daily["date"].iloc[-1])
    evaluated = evaluate_gate(judged, calib.de_exposure_multiplier, cost_bps=15.0)
    row = evaluated[
        (evaluated["scope"] == "全部触发日") & (evaluated["group"] == "高波动层")
    ].iloc[0]
    assert row["trigger_share"] == pytest.approx(1.0)
    assert np.isnan(row["return_gap"])


def test_run_exposure_gate_returns_three_tables(tmp_path):
    """端到端：校准表三行、逐日表含三臂中文标签、评估表含按折行。"""
    ledger = read_ledger_frames([write_ledger(tmp_path, make_ledger(days=120))])
    probe = build_daily_frame(ledger)
    calib_seg = (probe["date"].iloc[0], probe["date"].iloc[79])
    eval_seg = (probe["date"].iloc[80], probe["date"].iloc[-1])
    configs = [ExposureGateConfig(arm=arm) for arm in ARMS]
    calibration, daily, (overall, by_fold) = run_exposure_gate(ledger, configs, calib_seg, eval_seg)
    assert len(calibration) == len(ARMS)
    assert set(daily["arm"]) == {"对照A纯regime", "对照B纯模型分数", "E2(regime×分数)"}
    assert set(overall["fold"]) == {"全部"}
    assert set(by_fold["fold"]) != {"全部"}
    eval_days = int(probe["date"].iloc[80:].nunique())
    assert len(overall) == len(ARMS) * 2 * 3  # 臂 × 口径 × 分组
    # "全部" 行 = 评估段全量；两个层行之和必须回到全量（分层不重不漏）
    assert (overall.loc[overall["group"] == "全部", "days"] == eval_days).all()
    layer_sum = overall[overall["group"] != "全部"].groupby(["arm", "scope"])["days"].sum()
    assert (layer_sum == eval_days).all()


def test_run_exposure_gate_rejects_overlap(tmp_path):
    """端到端也必须拦截时段重叠。"""
    ledger = read_ledger_frames([write_ledger(tmp_path, make_ledger(days=20))])
    with pytest.raises(ValueError, match="重叠"):
        run_exposure_gate(
            ledger,
            [ExposureGateConfig()],
            ("20230101", "20231231"),
            ("20231201", "20991231"),
        )


def test_defaults_are_single_valued():
    """默认值单变量直读（无多层回退）。"""
    assert ARMS == ("regime", "score", "combined")
    config = ExposureGateConfig()
    assert config.arm == "combined"
    assert config.regime_quantile == pytest.approx(2.0 / 3.0)
    assert config.score_quantile == pytest.approx(0.5)
    assert config.de_exposure_multiplier == pytest.approx(0.5)
    assert config.cost_bps == pytest.approx(15.0)
    assert config.min_layer_days == 20


# ── 滚动分位口径（R-004 缓解路径 ①）────────────────────────────────────


def test_rolling_config_rejects_invalid_values():
    """滚动配置参数校验：窗口、层下限、分位、系数。"""
    with pytest.raises(ValueError, match="未知臂"):
        RollingGateConfig(arm="nope")
    with pytest.raises(ValueError, match="window_days"):
        RollingGateConfig(window_days=1)
    with pytest.raises(ValueError, match="min_window_days"):
        RollingGateConfig(window_days=100, min_window_days=200)
    with pytest.raises(ValueError, match="regime_quantile"):
        RollingGateConfig(regime_quantile=0.0)
    with pytest.raises(ValueError, match="score_quantile"):
        RollingGateConfig(arm="combined", score_quantile=1.0)
    with pytest.raises(ValueError, match="de_exposure_multiplier"):
        RollingGateConfig(de_exposure_multiplier=1.5)


def test_rolling_thresholds_use_only_past_data(tmp_path):
    """阈值只能来自 [t-W, t-1]：改动**当日**数据不得影响当日阈值；改动未来日不得影响过去。"""
    ledger = read_ledger_frames([write_ledger(tmp_path, make_ledger(days=80))])
    daily = build_daily_frame(ledger)
    start, end = daily["date"].iloc[40], daily["date"].iloc[-1]

    base, skipped = apply_rolling_gate(
        daily, RollingGateConfig(window_days=30, min_window_days=10), start, end
    )
    assert skipped == 0
    target = base["date"].iloc[5]

    bumped = daily.copy()
    row = bumped.index[bumped["date"] == target][0]
    bumped.loc[row, "p_loss_mean"] = 1.0
    bumped.loc[row, "mkt_vol_20"] = 9.9
    changed, _ = apply_rolling_gate(
        bumped, RollingGateConfig(window_days=30, min_window_days=10), start, end
    )
    before = base.loc[base["date"] == target].iloc[0]
    after = changed.loc[changed["date"] == target].iloc[0]
    assert after["regime_threshold"] == pytest.approx(before["regime_threshold"])
    assert after["score_threshold"] == pytest.approx(before["score_threshold"])
    # 当日得分变了 → 触发判定可以变，但阈值不能变
    assert after["p_loss_mean"] == pytest.approx(1.0)


def test_rolling_threshold_values_match_manual_quantiles(tmp_path):
    """阈值必须等于窗口内分位（regime 全窗口、combined 取窗口内高波动层）。"""
    ledger = read_ledger_frames([write_ledger(tmp_path, make_ledger(days=80))])
    daily = build_daily_frame(ledger)
    start, end = daily["date"].iloc[40], daily["date"].iloc[-1]
    config = RollingGateConfig(window_days=30, min_window_days=10)
    judged, _ = apply_rolling_gate(daily, config, start, end)

    ordered = daily.sort_values("date").reset_index(drop=True)
    target = judged["date"].iloc[7]
    pos = ordered.index[ordered["date"] == target][0]
    window_start = max(0, pos - 30)
    window = ordered.iloc[window_start:pos]
    expected_vol_thr = window["mkt_vol_20"].quantile(config.regime_quantile)
    layer = window[window["mkt_vol_20"] >= expected_vol_thr]
    expected_score_thr = layer["p_loss_mean"].quantile(config.score_quantile)
    row = judged.iloc[7]
    assert row["regime_threshold"] == pytest.approx(expected_vol_thr)
    assert row["score_threshold"] == pytest.approx(expected_score_thr)
    assert row["window_days"] == 30
    assert row["threshold_mode"] == "滚动分位"


def test_rolling_skips_short_window(tmp_path):
    """窗口不足 min_window_days 的交易日不判定（记空阈值、系数 1.0）并计数。"""
    ledger = read_ledger_frames([write_ledger(tmp_path, make_ledger(days=30))])
    daily = build_daily_frame(ledger)
    judged, skipped = apply_rolling_gate(
        daily,
        RollingGateConfig(window_days=250, min_window_days=60),
        daily["date"].iloc[0],
        daily["date"].iloc[-1],
    )
    assert skipped == len(judged)
    assert (judged["exposure_multiplier"] == 1.0).all()
    assert judged["regime_threshold"].isna().all()


def test_rolling_regime_arm_has_nan_score_threshold(tmp_path):
    """纯 regime 臂不设得分阈值（记 NaN），并用逐日阈值切层。"""
    ledger = read_ledger_frames([write_ledger(tmp_path, make_ledger(days=80))])
    daily = build_daily_frame(ledger)
    start, end = daily["date"].iloc[40], daily["date"].iloc[-1]
    judged, _ = apply_rolling_gate(
        daily, RollingGateConfig(arm="regime", window_days=30, min_window_days=10), start, end
    )
    assert judged["score_threshold"].isna().all()
    evaluated = evaluate_gate(judged, 0.5, cost_bps=15.0)
    # 逐日阈值下"高波动层"恒等于触发集合，层内触发占比必为 1
    row = evaluated[(evaluated["scope"] == "全部触发日") & (evaluated["group"] == "高波动层")].iloc[0]
    assert row["trigger_share"] == pytest.approx(1.0)


def test_run_rolling_gate_end_to_end(tmp_path):
    """端到端：无校准段输入，产物结构与固定口径一致（可直接对比）。"""
    ledger = read_ledger_frames([write_ledger(tmp_path, make_ledger(days=120))])
    probe = build_daily_frame(ledger)
    eval_seg = (probe["date"].iloc[60], probe["date"].iloc[-1])
    configs = [
        RollingGateConfig(arm="combined", window_days=40, min_window_days=20, label="E2-滚动分位40日"),
        RollingGateConfig(arm="score", window_days=40, min_window_days=20, label="对照B-滚动分位40日"),
    ]
    calibration, daily, (overall, by_fold) = run_rolling_gate(ledger, configs, eval_seg)
    assert len(calibration) == len(configs)
    assert set(daily["arm"]) == {"E2-滚动分位40日", "对照B-滚动分位40日"}
    assert set(daily["threshold_mode"]) == {"滚动分位"}
    assert set(overall["fold"]) == {"全部"}
    assert set(by_fold["fold"]) != {"全部"}
    # 分层不重不漏
    eval_days = int(probe["date"].iloc[60:].nunique())
    layer_sum = overall[overall["group"] != "全部"].groupby(["arm", "scope"])["days"].sum()
    assert (layer_sum == eval_days).all()
