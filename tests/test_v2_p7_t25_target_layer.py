# -*- coding: utf-8 -*-
"""P7 T2.5 目标层与原始意图预检的纯函数回归（不碰真实数据、不建证据根）。

覆盖点
======
- 相位平移：只移日历、不环绕、超尾部丢弃；
- 目标选择：持仓排除后按内核顺序截断、档位数 = 总目标 / 分批数；
- 目标级账户：T+1 买入、20 交易日到期卖出、跨折模型版本承接；
- 状态机归类：S1/S2/S3/S4/S5/S6 六种情形；
- 折界/右删失/重叠/相关性的登记口径。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import pandas as pd
import pytest

from scripts.v2_p7.t25.intent_precheck import (
    _classify,
    _recall_ledger,
    classify_intents,
    sleeve_correlation,
    summarize_scenarios,
)
from scripts.v2_p7.t25.params import _sleeve_specs, days_digest, shift_schedule
from scripts.v2_p7.t25.signal_layer import compare_anchor_scores
from scripts.v2_p7.t25.target_layer import (
    PlannerKernel,
    holdings_overlap,
    runoff_obligations,
    select_targets,
    simulate_arm,
    target_overlap,
)


class _StubParams:
    """``simulate_arm`` 只读取日期→折/模型映射，故用最小桩件。"""

    def __init__(self, splits: Sequence[Mapping[str, object]], runoff: Sequence[str] = ()):
        self.folds = [type("Fold", (), dict(item)) for item in splits]
        self.runoff_days = tuple(runoff)
        self.signal_days = tuple(
            day for item in splits for day in list(item["signal_days"])  # type: ignore[index]
        )

    def current_model_version(self, day: str) -> int:
        for item in self.folds:
            days = list(item.signal_days)
            if days[0] <= day <= days[-1]:
                return int(item.model_version)
        return int(self.folds[-1].model_version)

    def current_split_index(self, day: str) -> int:
        for item in self.folds:
            days = list(item.signal_days)
            if days[0] <= day <= days[-1]:
                return int(item.split_index)
        return int(self.folds[-1].split_index)


def _ranked(codes: Sequence[str]) -> List[Tuple[str, float]]:
    return [(code, 1.0 - 0.01 * idx) for idx, code in enumerate(codes)]


def _calendar(count: int) -> List[str]:
    base = pd.Timestamp("20200101")
    return [(base + pd.Timedelta(days=idx)).strftime("%Y%m%d") for idx in range(count)]


# ---------------------------------------------------------------- 相位平移
def test_shift_schedule_moves_only_calendar_and_drops_tail() -> None:
    calendar = [f"202001{idx:02d}" for idx in range(1, 22)]
    schedule = {"20200101": 0, "20200111": 1, "20200121": 0}

    shifted = shift_schedule(schedule, calendar, 3)

    assert shifted == {"20200104": 0, "20200114": 1}
    assert set(shifted) - set(schedule)  # 日期变了
    assert set(shifted.values()) == {0, 1}  # 档位标签不变


def test_shift_schedule_rejects_negative_and_unknown_days() -> None:
    calendar = ["20200101", "20200102"]
    with pytest.raises(Exception):
        shift_schedule({"20200101": 0}, calendar, -1)
    with pytest.raises(Exception):
        shift_schedule({"20200103": 0}, calendar, 0)


def test_days_digest_is_order_sensitive_and_stable() -> None:
    left = days_digest(["20200101", "20200102"])
    assert left == days_digest(["20200101", "20200102"])
    assert left != days_digest(["20200102", "20200101"])


# ---------------------------------------------------------------- 袖口径
def test_sleeve_specs_default_matches_frozen_and_records_deviation() -> None:
    top_n, capital, deviation = _sleeve_specs(None, None)
    assert top_n == {"v20": 20, "v10": 20}
    assert capital["v10"] == 300_000.0
    assert deviation == ()

    top_n, capital, deviation = _sleeve_specs(10, 350_000.0)
    assert top_n["v20"] == 20 and top_n["v10"] == 10
    assert capital["v20"] == 700_000.0 and capital["v10"] == 350_000.0
    assert len(deviation) == 2  # 深度 + 初资各一条


def test_sleeve_specs_rejects_illegal_values() -> None:
    with pytest.raises(Exception):
        _sleeve_specs(1, None)
    with pytest.raises(Exception):
        _sleeve_specs(None, 0.0)


# ---------------------------------------------------------------- 目标选择
def test_select_targets_excludes_held_and_caps_by_tranche() -> None:
    kernel = PlannerKernel(top_n=20, stagger_tranches=2)
    ranked = _ranked([f"S{idx:02d}" for idx in range(1, 11)])

    picks, info = select_targets(kernel, ranked, "20200102", 0, ["S01", "S02"])

    assert [code for code, _ in picks] == ["S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10"]
    assert info["target_n"] == 10
    assert info["overall_target_n"] == 20
    assert info["excluded_held"] == 2


def test_select_targets_respects_counterpart_depth() -> None:
    kernel = PlannerKernel(top_n=10, stagger_tranches=2)
    picks, info = select_targets(
        kernel, _ranked([f"S{idx:02d}" for idx in range(1, 21)]), "20200102", 1, []
    )
    assert len(picks) == 5
    assert info["target_n"] == 5


def test_sleeve_weights_follow_frozen_half_kelly() -> None:
    from scripts.v2_p7.t25.target_layer import sleeve_weights

    signals = {f"S{idx:02d}": 1.0 - 0.01 * idx for idx in range(10)}
    weights = sleeve_weights(signals, "20200102", None)
    assert set(weights) == set(signals)
    # 无方差输入时内核回退中位数 ⇒ 按分数百分位加权，总和归一到 1
    assert sum(weights.values()) == pytest.approx(1.0, rel=1e-9)
    assert weights["S00"] > weights["S09"]
    # 0.15 单股上限换算到批内 = 0.15 / 0.5 = 0.3
    assert max(weights.values()) <= 0.3 + 1e-12

    # 两只股票时单股 0.5 被压到 0.3，总和 < 1（剩余留现金），与内核一致
    capped = sleeve_weights({"A": 1.0, "B": 1.0}, "20200102", None)
    assert sum(capped.values()) == pytest.approx(0.6, rel=1e-9)


# ---------------------------------------------------------------- 账户模拟
def _prices(day: str, codes: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    return {code: (10.0, 10.0) for code in codes}


def _run_two_fold_account() -> object:
    calendar = _calendar(60)
    splits = [
        {"split_index": 0, "model_version": 101, "signal_days": calendar[:30]},
        {"split_index": 1, "model_version": 102, "signal_days": calendar[30:]},
    ]
    params = _StubParams(splits, runoff=calendar[-10:])
    codes = [f"S{idx:02d}" for idx in range(1, 41)]
    lookup = lambda day: _ranked(codes)  # noqa: E731
    price = lambda day: _prices(day, codes)  # noqa: E731
    schedule = {calendar[0]: 0, calendar[20]: 1, calendar[40]: 0}
    return simulate_arm(
        arm="v20",
        calendar=calendar,
        schedule=schedule,
        ranked_lookup=lookup,
        price_lookup=price,
        params=params,  # type: ignore[arg-type]
        capital=700_000.0,
        top_n=20,
        stagger_tranches=2,
    )


def test_simulate_arm_buys_t_plus_1_and_sells_after_holding() -> None:
    result = _run_two_fold_account()
    calendar = _calendar(60)

    first = result.plans.sort_values("planned_rank").iloc[0]
    assert str(first["signal_date"]) == calendar[0]
    assert str(first["execution_date"]) == calendar[1]
    assert str(first["due_date"]) == calendar[21]

    # 到期日在当日开盘卖出：到期前一日仍持有，到期日收盘后不再持有
    held_before_due = result.holdings[result.holdings["trade_date"] == calendar[20]]
    held_on_due = result.holdings[result.holdings["trade_date"] == calendar[21]]
    assert first["ts_code"] in set(held_before_due["ts_code"])
    assert first["ts_code"] not in set(held_on_due["ts_code"])


def test_simulate_arm_never_repeats_a_held_stock() -> None:
    result = _run_two_fold_account()
    plans = result.plans
    grouped = plans.groupby("signal_date")["ts_code"].apply(list)
    for date, codes in grouped.items():
        assert len(codes) == len(set(codes)), f"{date} 同批重复选股"


def test_simulate_arm_carries_positions_across_fold_boundary() -> None:
    result = _run_two_fold_account()
    calendar = _calendar(60)
    boundary = calendar[30]
    snapshot = result.holdings[result.holdings["trade_date"] == boundary]
    assert not snapshot.empty
    assert (snapshot["origin_model_version"] != snapshot["current_model_version"]).any()


def test_plan_integrity_detects_wrong_model_version() -> None:
    result = _run_two_fold_account()
    assert result.integrity["plans"] == len(result.plans)
    assert result.integrity["empty_ranking_days"] == []


# ---------------------------------------------------------------- 锚分数比对
def _attribution_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # 正常行
            {
                "ranking_date": "20200102",
                "signal_date": "20200102",
                "planned_ts_code": "X",
                "actual_ts_code": "X",
                "pred_score": 1.0,
            },
            # T+1 停牌顺延：actual != planned，pred_score 属 actual
            {
                "ranking_date": "20200102",
                "signal_date": "20200102",
                "planned_ts_code": "Y",
                "actual_ts_code": "Z",
                "pred_score": 2.0,
            },
            # 补位阶段：ranking_date 晚于 signal_date
            {
                "ranking_date": "20200105",
                "signal_date": "20200102",
                "planned_ts_code": "W",
                "actual_ts_code": "V",
                "pred_score": 3.0,
            },
        ]
    )


def _predictions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"trade_date": "20200102", "ts_code": "X", "score": 1.0},
            {"trade_date": "20200102", "ts_code": "Z", "score": 2.0},
            {"trade_date": "20200105", "ts_code": "V", "score": 3.0},
            # 计划股票自身分数与锚 pred_score 无关（顺延后未成交）
            {"trade_date": "20200102", "ts_code": "Y", "score": 9.9},
        ]
    )


def test_compare_anchor_scores_uses_ranking_date_and_actual_code() -> None:
    result = compare_anchor_scores(_attribution_frame(), _predictions_frame())
    assert result["anchor_rows"] == 3
    assert result["substituted_rows"] == 2
    assert result["exact_rows"] == 3
    assert result["all_exact"] is True
    assert result["compare_key"] == "ranking_date+actual_ts_code"


def test_compare_anchor_scores_flags_real_mismatch() -> None:
    attribution = _attribution_frame()
    attribution.loc[0, "pred_score"] = 1.5
    result = compare_anchor_scores(attribution, _predictions_frame())
    assert result["exact_rows"] == 2
    assert result["all_exact"] is False
    assert result["max_abs_diff"] == pytest.approx(0.5)


# ---------------------------------------------------------------- 目标重放与续跑
def test_compare_frames_detects_share_and_due_tampering() -> None:
    from scripts.v2_p7.t25.signal_layer import compare_frames

    base = pd.DataFrame(
        [
            {"ts_code": "X", "shares": 100, "due_date": "20200120", "model_version": 1},
            {"ts_code": "Y", "shares": 200, "due_date": "20200121", "model_version": 1},
        ]
    )
    assert compare_frames(base, base.copy())["identical"] is True

    tampered = base.copy()
    tampered.loc[0, "shares"] = -1
    result = compare_frames(base, tampered)
    assert result["identical"] is False
    assert "shares" in result["column_diffs"]

    tampered = base.copy()
    tampered.loc[1, "due_date"] = "19000101"
    result = compare_frames(base, tampered)
    assert result["identical"] is False
    assert "due_date" in result["column_diffs"]

    shorter = base.iloc[:1].reset_index(drop=True)
    assert compare_frames(base, shorter)["rows_match"] is False


def test_open_size_ignores_execution_day_close() -> None:
    """因果性：只改 T+1 收盘价，开盘计划股数与预算必须逐位不变。"""
    from scripts.v2_p7.t25.signal_layer import compare_frames

    def run_with_close(close: float) -> object:
        calendar = _calendar(60)
        splits = [{"split_index": 0, "model_version": 101, "signal_days": calendar}]
        params = _StubParams(splits, runoff=calendar[-5:])
        codes = [f"S{idx:02d}" for idx in range(1, 41)]
        return simulate_arm(
            arm="v20",
            calendar=calendar,
            schedule={calendar[0]: 0, calendar[20]: 1, calendar[40]: 0},
            ranked_lookup=lambda day: _ranked(codes),
            price_lookup=lambda day: {code: (10.0, close) for code in codes},
            params=params,  # type: ignore[arg-type]
            capital=700_000.0,
            top_n=20,
            stagger_tranches=2,
        )

    base = run_with_close(10.0)
    shifted = run_with_close(37.5)
    result = compare_frames(base.plans, shifted.plans)
    assert result["identical"] is True, result["column_diffs"]
    assert compare_frames(base.holdings, shifted.holdings)["identical"] is True


def test_invalid_open_never_falls_back_to_execution_close() -> None:
    """D1：open 无效时不得用当日 close 顶替——改 close 仍不能影响股数。"""
    from scripts.v2_p7.t25.signal_layer import compare_frames

    def run_with_close(close: float) -> object:
        calendar = _calendar(40)
        splits = [{"split_index": 0, "model_version": 101, "signal_days": calendar}]
        params = _StubParams(splits, runoff=calendar[-3:])
        codes = [f"S{idx:02d}" for idx in range(1, 21)]

        def prices(day: str) -> Dict[str, Tuple[float, float]]:
            table = {code: (10.0, close) for code in codes}
            # 第二批执行日：S00 的 open 置 0（无效），close 随参数变化
            if day == calendar[11]:
                table["S00"] = (0.0, close)
            return table

        return simulate_arm(
            arm="v20",
            calendar=calendar,
            schedule={calendar[0]: 0, calendar[10]: 1},
            ranked_lookup=lambda day: _ranked(codes),
            price_lookup=prices,
            params=params,  # type: ignore[arg-type]
            capital=700_000.0,
            top_n=20,
            stagger_tranches=2,
        )

    base = run_with_close(10.0)
    shifted = run_with_close(20.0)
    assert compare_frames(base.plans, shifted.plans)["identical"] is True
    assert compare_frames(base.daily, shifted.daily)["identical"] is True


def test_variance_book_counts_rows_not_values() -> None:
    """D2：有行情行但 close_adj 为 NaN，仍按**行数**过门（不得提前回退）。"""
    import numpy as np

    from scripts.v2_p7.t25.target_layer import VarianceBook

    days = [f"202001{idx:02d}" for idx in range(1, 21)]
    values = {day: 10.0 + 0.02 * pos + np.sin(pos) * 0.1 for pos, day in enumerate(days)}
    values[days[5]] = float("nan")  # 有行但价格无效

    def loader(day: str) -> pd.DataFrame:
        return pd.DataFrame({"ts_code": ["X"], "close_adj": [values[day]]})

    book = VarianceBook(days, loader)
    assert book.variance("X", days[-1]) is not None

    # 真正缺行（该日无该股行）时，只有 19 行 ⇒ 与内核一致地回退
    missing_day = days[3]

    def loader_missing(day: str) -> pd.DataFrame:
        if day == missing_day:
            return pd.DataFrame({"ts_code": ["Y"], "close_adj": [10.0]})
        return loader(day)

    assert VarianceBook(days, loader_missing).variance("X", days[-1]) is None


def test_variance_cutoff_is_signal_day() -> None:
    """因果性：half_kelly 方差只用信号日及以前的行（截止日 = 信号日，不是执行日）。"""
    seen: List[str] = []

    class _RecordingBook:
        def variance(self, code: str, day: str) -> float:
            seen.append(day)
            return 1e-4

    calendar = _calendar(30)
    splits = [{"split_index": 0, "model_version": 101, "signal_days": calendar}]
    params = _StubParams(splits, runoff=calendar[-3:])
    codes = [f"S{idx:02d}" for idx in range(1, 21)]
    result = simulate_arm(
        arm="v20",
        calendar=calendar,
        schedule={calendar[0]: 0, calendar[10]: 1},
        ranked_lookup=lambda day: _ranked(codes),
        price_lookup=lambda day: {code: (10.0, 12.0) for code in codes},
        params=params,  # type: ignore[arg-type]
        capital=700_000.0,
        top_n=20,
        stagger_tranches=2,
        variance_book=_RecordingBook(),  # type: ignore[arg-type]
    )
    assert result.plans.shape[0] > 0
    assert set(seen) == {calendar[0], calendar[10]}
    assert all(day in {calendar[0], calendar[10]} for day in seen)


def test_compare_frames_is_bitwise_and_dtype_strict() -> None:
    """复审 R4 三条反例：+0/-0、float32 vs float64、int64 减法溢出。"""
    from scripts.v2_p7.t25.signal_layer import compare_frames

    pos = pd.DataFrame({"x": [0.0]})
    neg = pd.DataFrame({"x": [-0.0]})
    assert compare_frames(pos, neg)["identical"] is False

    narrow = pd.DataFrame({"x": pd.Series([1.0], dtype="float32")})
    wide = pd.DataFrame({"x": pd.Series([1.0], dtype="float64")})
    assert compare_frames(narrow, wide)["identical"] is False

    left = pd.DataFrame({"shares": pd.Series([100], dtype="int64")})
    right = pd.DataFrame({"shares": pd.Series([-9223372036854775708], dtype="int64")})
    assert compare_frames(left, right)["identical"] is False


def test_resume_identity_rejects_wrong_split_and_sha(tmp_path: Path) -> None:
    from scripts.v2_p7.common import sha256_file
    from scripts.v2_p7.t25.runner import _assert_resume_fold_identity

    fold_dir = tmp_path / "split00"
    fold_dir.mkdir()
    scores = fold_dir / "scores.parquet"
    scores.write_bytes(b"score-bytes")
    topk = fold_dir / "signal_topk.parquet"
    topk.write_bytes(b"topk-bytes")
    fold = type(
        "Fold",
        (),
        {"split_index": 0, "model_version": 101, "signal_days": ("20200102",)},
    )()
    payload = {
        "split_index": 0,
        "model_version": 101,
        "params_digest": "p",
        "source_digest": "s",
        "predictions_sha256": sha256_file(scores),
        "signal_topk_sha256": sha256_file(topk),
    }
    _assert_resume_fold_identity(
        payload=payload,
        fold=fold,
        fold_dir=fold_dir,
        params_digest="p",
        source_digest="s",
    )

    for field, value in (("split_index", 999), ("model_version", 999), ("params_digest", "x")):
        broken = dict(payload)
        broken[field] = value
        with pytest.raises(RuntimeError):
            _assert_resume_fold_identity(
                payload=broken,
                fold=fold,
                fold_dir=fold_dir,
                params_digest="p",
                source_digest="s",
            )

    (fold_dir / "signal_topk.parquet").unlink()
    with pytest.raises(RuntimeError):
        _assert_resume_fold_identity(
            payload=payload,
            fold=fold,
            fold_dir=fold_dir,
            params_digest="p",
            source_digest="s",
        )


def test_mechanical_line_treats_s1_as_registered_gap_only() -> None:
    """用户 2026-11-12 裁决：S1 缺位不阻断机械线；S2/S3/S4/S5 缺位仍阻断。"""
    from scripts.v2_p7.t25.runner import _mechanical_line_status

    s1_only = _mechanical_line_status({"missing_scenarios": ["S1"], "counts": {"S1": 0}})
    assert s1_only["natural_intent_status"] == "observed_with_registered_gap"
    assert s1_only["non_blocking_gap"] == ["S1"]
    assert s1_only["missing_scenarios"] == ["S1"]  # 原始计数不修饰
    assert "T4~T6 可继续" in s1_only["downstream"]

    for blocking in ("S2", "S3", "S4", "S5"):
        result = _mechanical_line_status({"missing_scenarios": [blocking], "counts": {}})
        assert result["natural_intent_status"] == "incomplete/not_observed"
        assert result["blocking_missing"] == [blocking]

    full = _mechanical_line_status({"missing_scenarios": [], "counts": {}})
    assert full["natural_intent_status"] == "observed"


def test_stop_on_hard_failures_blocks_publication() -> None:
    from scripts.v2_p7.t25.runner import _stop_on_hard_failures

    ok = {
        "score_anchor_exact": {"ok": True},
        "targets_replay": {"all_identical": True},
        "replay": {"checked": True, "all_bitwise_identical": True},
        "no_bt_overlap": {"ok": True},
        "anchor_plan_same_source": {"unexplained_days": 0},
        "input_identity": {"frozen_manifest_mismatch": []},
    }
    _stop_on_hard_failures(ok)

    for key, broken in (
        ("score_anchor_exact", {"ok": False}),
        ("targets_replay", {"all_identical": False}),
        ("replay", {"checked": True, "all_bitwise_identical": False}),
        ("input_identity", {"frozen_manifest_mismatch": ["data/features/cs_train/x.parquet"]}),
    ):
        case = dict(ok)
        case[key] = broken
        with pytest.raises(RuntimeError):
            _stop_on_hard_failures(case)


def test_daily_row_uses_one_timepoint_for_target_and_market_value() -> None:
    """换仓日目标不得算成 30 槽；普通持仓日市值不得置零。"""
    result = _run_two_fold_account()
    daily = result.daily
    rebalance_days = daily[daily["new_count"] > 0]
    assert not rebalance_days.empty
    # 目标名义 = 成交后持仓预算合计，非负且不超过当日 NAV 上限（含浮盈，故留余量）
    assert (rebalance_days["target_notional"] <= 1_200_000.0).all()
    assert (rebalance_days["target_notional"] > 0).all()
    # 非再平衡、无新买但有持仓的日子，市值必须 > 0
    plain = daily[(daily["new_count"] == 0) & (daily["held_count"] > 0)]
    assert not plain.empty
    assert (plain["mv_hold"] > 0).all()
    # 同一时点：raw_demand = max(0, G − MV)
    gap = (daily["target_notional"] - daily["mv_hold"]).clip(lower=0.0)
    assert (daily["raw_demand"] - gap).abs().max() == pytest.approx(0.0)


# ---------------------------------------------------------------- 状态机
def test_classify_covers_all_six_scenarios() -> None:
    assert _classify(1000, 0, 0, 1000) == "S1"
    assert _classify(0, 1000, 500, 0) == "S2"
    assert _classify(1000, 0, 0, 400) == "S3"
    assert _classify(1000, 0, 500, 0) == "S4"
    assert _classify(0, 1000, 0, 400) == "S5"
    assert _classify(1000, 0, 0, 0) == "S6"
    assert _classify(0, 0, 0, 0) == "S6"


def _intent_frames() -> Tuple[pd.DataFrame, pd.DataFrame]:
    buys = pd.DataFrame(
        [
            {"arm": "a", "trade_date": "20200102", "ts_code": "X", "shares": 1000, "notional": 1e4},
            {"arm": "b", "trade_date": "20200102", "ts_code": "Y", "shares": 500, "notional": 5e3},
        ]
    )
    sells = pd.DataFrame(
        [
            {"arm": "b", "trade_date": "20200102", "ts_code": "X", "shares": 1000, "notional": 1e4},
            {"arm": "a", "trade_date": "20200102", "ts_code": "Y", "shares": 500, "notional": 5e3},
        ]
    )
    return buys, sells


def test_classify_intents_pairs_by_arm_name() -> None:
    buys, sells = _intent_frames()
    records = classify_intents(buys, sells, arms=("a", "b"))
    by_code = {item["ts_code"]: item for item in records}
    assert by_code["X"]["scenario"] == "S1"
    assert by_code["X"]["offset_shares"] == 1000
    assert by_code["X"]["net_shares"] == 0


def test_summarize_scenarios_keeps_reverse_and_merge_separate() -> None:
    buys, sells = _intent_frames()
    summary = summarize_scenarios(classify_intents(buys, sells, arms=("a", "b")))
    assert summary["reverse_events"] == 2
    assert summary["merge_events"] == 0
    assert summary["scenarios"]["S1"]["events"] == 2
    assert summary["scenarios"]["S4"]["events"] == 0


# ---------------------------------------------------------------- 额度/召回
def _daily_frame(rows: Sequence[Mapping[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_recall_ledger_flags_recall_when_lender_needs_cash_back() -> None:
    calendar = ["20200102", "20200103"]
    frames = {
        # 第 1 日：b 缺钱、a 有闲置 ⇒ a 借出；第 2 日：a 自己缺钱 ⇒ 召回需求
        "v20": _daily_frame(
            [
                {
                    "trade_date": "20200102",
                    "buy_need": 100.0,
                    "sell_proceeds": 0.0,
                    "cash_open": 500.0,
                    "funding_gap": 0.0,
                },
                {
                    "trade_date": "20200103",
                    "buy_need": 600.0,
                    "sell_proceeds": 0.0,
                    "cash_open": 100.0,
                    "funding_gap": 500.0,
                },
            ]
        ),
        "v10": _daily_frame(
            [
                {
                    "trade_date": "20200102",
                    "buy_need": 800.0,
                    "sell_proceeds": 0.0,
                    "cash_open": 200.0,
                    "funding_gap": 600.0,
                },
                {
                    "trade_date": "20200103",
                    "buy_need": 300.0,
                    "sell_proceeds": 0.0,
                    "cash_open": 500.0,
                    "funding_gap": 0.0,
                },
            ]
        ),
    }
    ledger = _recall_ledger(frames, calendar)
    assert ledger["borrow_events"] == 2
    assert ledger["borrow_amount"] == pytest.approx(600.0)
    assert ledger["recall_events"] == 1
    assert ledger["recall_amount"] == pytest.approx(400.0)


def test_sleeve_correlation_is_one_for_identical_series() -> None:
    series = {"v20": [100.0, 101.0, 103.0, 106.0], "v10": [100.0, 101.0, 103.0, 106.0]}
    result = sleeve_correlation(series)
    assert result["pearson"] == pytest.approx(1.0)


# ---------------------------------------------------------------- 重叠/runoff
def test_holdings_and_target_overlap() -> None:
    holdings = pd.DataFrame(
        [
            {"arm": "v20", "trade_date": "20200102", "ts_code": "X"},
            {"arm": "v10", "trade_date": "20200102", "ts_code": "X"},
            {"arm": "v20", "trade_date": "20200102", "ts_code": "Y"},
        ]
    )
    assert holdings_overlap(holdings)["mean_jaccard"] == pytest.approx(0.5)

    plans = pd.DataFrame(
        [
            {"arm": "v20", "signal_date": "20200102", "ts_code": "X"},
            {"arm": "v10", "signal_date": "20200102", "ts_code": "X"},
        ]
    )
    assert target_overlap(plans)["mean_jaccard"] == pytest.approx(1.0)


def test_runoff_obligations_marks_right_censoring() -> None:
    calendar = _calendar(30)
    splits = [{"split_index": 0, "model_version": 101, "signal_days": calendar}]
    params = _StubParams(splits, runoff=calendar[-5:])
    codes = [f"S{idx:02d}" for idx in range(1, 21)]
    result = simulate_arm(
        arm="v20",
        calendar=calendar,
        # 第二档在靠尾部建仓，到期跨越日历末端 ⇒ 末信号日仍有未决持仓
        schedule={calendar[0]: 0, calendar[20]: 1},
        ranked_lookup=lambda day: _ranked(codes),
        price_lookup=lambda day: _prices(day, codes),
        params=params,  # type: ignore[arg-type]
        capital=700_000.0,
        top_n=20,
        stagger_tranches=2,
    )
    report = runoff_obligations(result, params)  # type: ignore[arg-type]
    assert report["runoff_window"] == [calendar[-5], calendar[-1]]
    assert report["obligations"]
    assert all("right_censored" in item for item in report["obligations"])
