"""walk-forward 对比表选股评分测试"""

import pandas as pd

from scripts.compare_walk_forward import (
    COL_NAMES,
    build_live_candidate_score_table,
    build_model_alpha_score_table,
    build_model_seed_stability_table,
    build_trade_param_score_table,
    compute_selection_score,
    sort_by_run_time,
)


def test_selection_score_uses_three_stock_selection_metrics() -> None:
    """选股综合得分应只由 RankIC、ICIR、Top30 超额三项驱动。"""
    df = pd.DataFrame(
        {
            COL_NAMES["daily_rankic_mean"]: [0.10, 0.20, 0.30],
            COL_NAMES["icir"]: [1.0, 2.0, 3.0],
            COL_NAMES["oos_top30_lift_mean"]: [0.01, 0.02, 0.03],
            COL_NAMES["selection_monotonicity"]: [1.0, 0.0, 0.0],
        }
    )

    scores = compute_selection_score(df)

    assert scores.tolist() == [33.3, 66.7, 100.0]


def test_selection_score_single_row_is_neutral() -> None:
    """单个实验没有横向排序对象时，选股综合得分固定为中性 50。"""
    df = pd.DataFrame(
        {
            COL_NAMES["daily_rankic_mean"]: [0.30],
            COL_NAMES["icir"]: [3.0],
            COL_NAMES["oos_top30_lift_mean"]: [0.03],
        }
    )

    scores = compute_selection_score(df)

    assert scores.tolist() == [50.0]


def _base_comp_df() -> pd.DataFrame:
    rows = []
    for period, final_date in [("早期", "20231231"), ("近期", "20241231")]:
        for model_name, rankic, icir, lift, worst, monotonicity, gap in [
            ("model_good", 0.08, 2.0, 0.04, 0.01, 1.0, 0.05),
            ("model_bad", 0.01, 0.4, 0.00, -0.05, 0.2, 0.80),
        ]:
            for top_n, cagr, total_return, sharpe, calmar, win_rate, drawdown in [
                (20, 0.30, 0.60, 1.8, 2.5, 0.80, -0.12),
                (40, 0.05, 0.10, 0.6, 0.7, 0.45, -0.28),
            ]:
                rows.append(
                    {
                        "运行ID": (f"wf_{final_date}_090000_{model_name}_{top_n}"),
                        "批次时间段": period,
                        "最终日期": final_date,
                        "切分数量": 3,
                        "标签列": model_name,
                        "最大深度": 6 if model_name == "model_good" else 3,
                        "学习率": 0.03 if model_name == "model_good" else 0.10,
                        "多种子bagging种子": "42",
                        "回测TopN": top_n,
                        "回测调仓频率": 5,
                        "选股综合得分": 90.0 if model_name == "model_good" else 20.0,
                        COL_NAMES["daily_rankic_mean"]: rankic,
                        COL_NAMES["icir"]: icir,
                        COL_NAMES["oos_top30_lift_mean"]: lift,
                        COL_NAMES["oos_top30_win_rate"]: 0.8 if model_name == "model_good" else 0.3,
                        COL_NAMES["oos_top30_worst_median"]: worst,
                        COL_NAMES["selection_monotonicity"]: monotonicity,
                        COL_NAMES["train_val_ir_gap"]: gap,
                        COL_NAMES["chain_cagr"]: cagr,
                        COL_NAMES["chain_total_return"]: total_return,
                        COL_NAMES["chain_sharpe"]: sharpe,
                        COL_NAMES["bt_calmar_mean"]: calmar,
                        COL_NAMES["bt_win_rate"]: win_rate,
                        COL_NAMES["chain_max_drawdown"]: drawdown,
                    }
                )
    return pd.DataFrame(rows)


def test_model_alpha_score_ignores_trade_return_columns() -> None:
    """模型Alpha评分应只受选股统计驱动，不被交易收益变化污染。"""
    df = _base_comp_df()
    before = build_model_alpha_score_table(df).set_index("标签列")["模型Alpha分"].to_dict()

    changed = df.copy()
    changed.loc[changed["标签列"] == "model_bad", COL_NAMES["chain_cagr"]] = 9.99
    changed.loc[changed["标签列"] == "model_bad", COL_NAMES["chain_total_return"]] = 9.99
    after = build_model_alpha_score_table(changed).set_index("标签列")["模型Alpha分"].to_dict()

    assert before == after
    assert before["model_good"] > before["model_bad"]


def test_trade_param_score_uses_paired_context_and_skips_single_candidate() -> None:
    """交易参数评分应在同模型同时间段内配对，且跳过只有一个交易候选的环境。"""
    df = _base_comp_df()
    single_candidate = df.iloc[[0]].copy()
    single_candidate["批次时间段"] = "单候选"
    single_candidate["最终日期"] = "20251231"
    df = pd.concat([df, single_candidate], ignore_index=True)

    trade_df = build_trade_param_score_table(df).set_index("回测TopN")

    assert trade_df.loc[20, "交易收益分"] > trade_df.loc[40, "交易收益分"]
    assert trade_df.loc[20, "有效配对环境数"] == 4
    assert trade_df.loc[40, "有效配对环境数"] == 4


def test_live_candidate_score_zero_when_hard_gate_fails() -> None:
    """硬门槛未通过时，实盘候选分应置0并写明失败原因。"""
    df = _base_comp_df()
    model_df = build_model_alpha_score_table(df)
    trade_df = build_trade_param_score_table(df)
    candidate_df = build_live_candidate_score_table(df, model_df, trade_df)

    failed = candidate_df[candidate_df["模型Alpha分"] < 60].iloc[0]

    assert failed["实盘候选分"] == 0
    assert "模型Alpha分<60" in failed["候选门槛失败原因"]


def test_score_tables_put_latest_run_first() -> None:
    """评分表应把最新生成记录排在最前，并展示可读运行时间。"""
    df = _base_comp_df()
    model_df = build_model_alpha_score_table(df)

    assert model_df.iloc[0]["最新运行时间"] == "2024-12-31 09:00:00"


def test_comparison_table_sort_adds_latest_run_time() -> None:
    """实验对比表也应按运行ID时间倒序，并补充最新运行时间列。"""
    df = pd.DataFrame(
        {
            "运行ID": ["wf_20240101_090000_old", "wf_20250101_090000_new"],
            "综合得分": [90.0, 10.0],
        }
    )

    result = sort_by_run_time(df)

    assert result.iloc[0]["运行ID"] == "wf_20250101_090000_new"
    assert result.iloc[0]["最新运行时间"] == "2025-01-01 09:00:00"


def test_model_seed_stability_groups_same_params_except_seed() -> None:
    """模型Seed稳定性应忽略 seed 字段，聚合同一套超参的多 seed 结果。"""
    df = pd.DataFrame(
        {
            "运行ID": ["wf_20250101_090000_seed42", "wf_20250101_100000_seed99"],
            "批次时间段": ["seed_test", "seed_test"],
            "最终日期": ["20250101", "20250101"],
            "切分数量": [3, 3],
            "标签列": ["model_same", "model_same"],
            "最大深度": [6, 6],
            "学习率": [0.03, 0.03],
            "多种子bagging种子": ["42", "99"],
            "选股综合得分": [90.0, 70.0],
            COL_NAMES["daily_rankic_mean"]: [0.08, 0.06],
            COL_NAMES["icir"]: [2.0, 1.4],
            COL_NAMES["oos_top30_lift_mean"]: [0.04, 0.02],
            COL_NAMES["oos_top30_win_rate"]: [0.8, 0.7],
            COL_NAMES["oos_top30_worst_median"]: [0.01, 0.00],
            COL_NAMES["selection_monotonicity"]: [1.0, 0.8],
            COL_NAMES["train_val_ir_gap"]: [0.05, 0.10],
            COL_NAMES["chain_cagr"]: [0.3, 0.2],
            COL_NAMES["chain_max_drawdown"]: [-0.12, -0.15],
        }
    )

    model_df = build_model_alpha_score_table(df)
    seed_df = build_model_seed_stability_table(df, model_df)

    assert len(seed_df) == 1
    assert seed_df.iloc[0]["Seed稳定性样本数"] == 2
    assert "42" in seed_df.iloc[0]["Seed列表"]
    assert "99" in seed_df.iloc[0]["Seed列表"]
    assert "模型Alpha分均值" in seed_df.columns
    assert "模型Alpha分标准差" in seed_df.columns
    assert "模型Alpha分最差" in seed_df.columns
