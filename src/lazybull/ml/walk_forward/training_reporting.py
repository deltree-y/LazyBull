#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Walk-forward 训练报告与格式化辅助函数。"""

from typing import Dict, Optional

import numpy as np
from loguru import logger


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(result) or np.isinf(result):
        return None
    return result


def _fmt_metric(value: Optional[float], fmt: str = ".4f") -> str:
    return "nan" if value is None else f"{value:{fmt}}"


def _fmt_pct(value: Optional[float]) -> str:
    return "nan" if value is None else f"{value * 100:.1f}%"


def _metric_value(metrics: Dict, key: str) -> Optional[float]:
    return _safe_float(metrics.get(key))


def _topk_key_metrics(metrics: Dict, topk: int) -> Dict[str, Optional[float]]:
    return {
        "median": _metric_value(metrics, f"diagnostic_Top{topk}_逐日均值_50分位"),
        "mean": _metric_value(metrics, f"top{topk}_return_mean"),
        "lift": _metric_value(metrics, f"diagnostic_Top{topk}_相对全市场提升_均值"),
        "hit_rate": _metric_value(metrics, f"diagnostic_Top{topk}_命中率_日均收益为正"),
        "excess_hit_rate": _metric_value(metrics, f"diagnostic_Top{topk}_超额命中率_跑赢全市场"),
    }


def _print_oos_focus_panel(split_index: int, test_daily_metrics: Dict) -> None:
    """打印 OOS 重点指标面板，避免关键 TopK 信息被普通日志淹没。"""
    top20 = _topk_key_metrics(test_daily_metrics, 20)
    top30 = _topk_key_metrics(test_daily_metrics, 30)
    top20_list = str(test_daily_metrics.get("diagnostic_Top20_最新股票列表") or "")
    top30_list = str(test_daily_metrics.get("diagnostic_Top30_最新股票列表") or "")
    latest_date = str(
        test_daily_metrics.get("diagnostic_Top20_最新日期")
        or test_daily_metrics.get("diagnostic_Top30_最新日期")
        or ""
    )

    logger.opt(colors=True).warning("<cyan><bold>" + "=" * 92 + "</bold></cyan>")
    logger.opt(colors=True).warning(
        f"<cyan><bold>Split {split_index} OOS 重点 TopK 指标</bold></cyan> "
        f"<cyan>(hit rate=TopK逐日平均收益>0占比, list=最新OOS日期 {latest_date})</cyan>"
    )
    logger.opt(colors=True).warning(
        "<yellow><bold>Top20</bold></yellow> | "
        f"hit={_fmt_pct(top20['hit_rate'])} | "
        f"均值中位数={_fmt_metric(top20['median'], '.6f')} | "
        f"超额均值={_fmt_metric(top20['lift'], '.6f')} | "
        f"跑赢全市场={_fmt_pct(top20['excess_hit_rate'])}"
    )
    logger.opt(colors=True).warning(
        "<yellow><bold>Top30</bold></yellow> | "
        f"hit={_fmt_pct(top30['hit_rate'])} | "
        f"均值中位数={_fmt_metric(top30['median'], '.6f')} | "
        f"超额均值={_fmt_metric(top30['lift'], '.6f')} | "
        f"跑赢全市场={_fmt_pct(top30['excess_hit_rate'])}"
    )
    logger.opt(colors=True).warning(f"<yellow>Top20 list:</yellow> {top20_list}")
    logger.opt(colors=True).warning(f"<yellow>Top30 list:</yellow> {top30_list}")
    logger.opt(colors=True).warning("<cyan><bold>" + "=" * 92 + "</bold></cyan>")


def _print_pre_backtest_model_summary(
    split_index: int,
    ensemble_meta: Dict,
    val_daily_metrics: Dict,
    test_daily_metrics: Dict,
) -> None:
    """在回测前打印模型摘要：各子模型迭代轮数 + 验证集/测试集关键指标。"""
    sub_iters = ensemble_meta.get("sub_model_best_iterations", [])
    if sub_iters:
        seed_parts = []
        for seed_val, best_iter_val in sub_iters:
            seed_str = str(seed_val) if seed_val is not None else "?"
            iter_str = str(best_iter_val) if best_iter_val is not None else "?"
            seed_parts.append(f"seed={seed_str}:best_iter={iter_str}")
        logger.opt(colors=True).warning(
            f"<yellow><bold>Split {split_index} 子模型迭代轮数:</bold></yellow> "
            f"{', '.join(seed_parts)}"
        )

    def _extract(m: Dict) -> Dict[str, str]:
        def _f(key: str, fmt: str = ".6f") -> str:
            v = _safe_float(m.get(key))
            return "nan" if v is None else f"{v:{fmt}}"

        top20_med = _safe_float(m.get("diagnostic_Top20_逐日均值_50分位"))
        top30_med = _safe_float(m.get("diagnostic_Top30_逐日均值_50分位"))
        univ_mean = _safe_float(m.get("diagnostic_全市场收益_逐日均值的均值"))
        lift20_med = (
            top20_med - univ_mean if top20_med is not None and univ_mean is not None else None
        )
        lift30_med = (
            top30_med - univ_mean if top30_med is not None and univ_mean is not None else None
        )
        return {
            "RankIC均值": _f("daily_rankic_mean", ".4f"),
            "Top20中位数": _f("diagnostic_Top20_逐日均值_50分位"),
            "Top20命中率": _fmt_pct(_safe_float(m.get("diagnostic_Top20_命中率_日均收益为正"))),
            "Top20提升中位数": "nan" if lift20_med is None else f"{lift20_med:.6f}",
            "Top30中位数": _f("diagnostic_Top30_逐日均值_50分位"),
            "Top30命中率": _fmt_pct(_safe_float(m.get("diagnostic_Top30_命中率_日均收益为正"))),
            "Top30提升中位数": "nan" if lift30_med is None else f"{lift30_med:.6f}",
            "Top30提升均值": _f("diagnostic_Top30_相对全市场提升_均值"),
        }

    val_info = _extract(val_daily_metrics) if val_daily_metrics else {}
    test_info = _extract(test_daily_metrics) if test_daily_metrics else {}

    metric_labels = [
        "RankIC均值",
        "Top20中位数",
        "Top20命中率",
        "Top20提升中位数",
        "Top30中位数",
        "Top30命中率",
        "Top30提升中位数",
        "Top30提升均值",
    ]
    logger.opt(colors=True).warning(
        f"<yellow><bold>Split {split_index} 模型指标对比（验证集 vs 测试集）:</bold></yellow>"
    )
    for label in metric_labels:
        v_val = val_info.get(label, "-")
        v_test = test_info.get(label, "-")
        logger.opt(colors=True).warning(
            f"  <yellow>{label:16s}</yellow>  " f"验证={v_val:>12s}  |  测试={v_test:>12s}"
        )
