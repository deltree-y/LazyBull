"""walk-forward 对比表选股评分测试"""

import pandas as pd

from scripts.compare_walk_forward import COL_NAMES, compute_selection_score


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