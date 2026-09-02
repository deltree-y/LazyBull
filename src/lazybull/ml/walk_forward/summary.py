"""Walk-forward 分段结果汇总。"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from src.lazybull.common.config import get_data_root
from src.lazybull.ml.train_core.constants import (
    CONSENSUS_REVISION_FEATURE_COLUMNS,
)


def _live_consensus_revision_cols(result: Dict) -> str:
    """提取单个 split 实际存活的一致预期修正列（以门禁后 feature_columns 为准）。

    同时识别入模的市值中性化派生列（zscore_cons_*_sz）。
    """
    feature_cols = result.get("feature_columns") or []
    base_cols = {
        col for col in CONSENSUS_REVISION_FEATURE_COLUMNS if col != "cons_revision_freshness_days"
    }
    live = sorted(
        col
        for col in feature_cols
        if col in base_cols or (col.startswith("zscore_cons_") and col.endswith("_sz"))
    )
    return ",".join(live)


def _live_cashflow_quality_cols(result: Dict) -> str:
    """提取单个 split 实际入模的现金流质量列（以门禁后 feature_columns 为准）。

    与名义开关不同，此处记录的是缺失率/常数门禁之后真正进入模型的列，
    避免"开关开启"被误读为"实验使用了同一组因子"。
    """
    from src.lazybull.factors.cashflow_quality import cashflow_quality_live_columns

    return ",".join(cashflow_quality_live_columns(result.get("feature_columns") or []))


def _topk_key_metrics(metrics: Dict, topk: int) -> Dict[str, Optional[float]]:
    """提取 summary 首列展示的 TopK 重点指标。"""
    return {
        "hit_rate": metrics.get(f"top{topk}_positive_day_ratio"),
        "median": metrics.get(f"top{topk}_avg_return_median"),
        "lift": metrics.get(f"top{topk}_lift_mean"),
    }


def _build_summary_key_fields(test_daily_metrics: Dict) -> Dict[str, object]:
    top20 = _topk_key_metrics(test_daily_metrics, 20)
    top30 = _topk_key_metrics(test_daily_metrics, 30)
    return {
        "KEY_说明": "重点: hit rate=TopK逐日平均收益>0占比; list=最新OOS日期预测名单",
        "KEY_Top20_list": test_daily_metrics.get("diagnostic_Top20_最新股票列表"),
        "KEY_Top30_list": test_daily_metrics.get("diagnostic_Top30_最新股票列表"),
        "KEY_Top20_hit_rate": top20["hit_rate"],
        "KEY_Top20_avg_return_median": top20["median"],
        "KEY_Top20_lift_mean": top20["lift"],
        "KEY_Top30_hit_rate": top30["hit_rate"],
        "KEY_Top30_avg_return_median": top30["median"],
        "KEY_Top30_lift_mean": top30["lift"],
    }


def _sanitize_train_params(raw_params: Dict[str, Any]) -> Dict[str, Any]:
    """清空未启用功能的子参数，避免对比表出现误导性默认值。"""
    params = dict(raw_params)

    def clear(*keys: str) -> None:
        for key in keys:
            if key in params:
                params[key] = None

    if params.get("freshness_strategy") != "state_keep_event_decay":
        clear("event_freshness_half_life_days")

    if not params.get("oos_backtest"):
        clear(
            "oos_backtest_months",
            "bt_top_n",
            "bt_sell_timing",
            "bt_exclude_st",
            "bt_min_list_days",
            "bt_max_weight_per_stock",
            "bt_max_per_industry",
            "bt_stop_loss_enabled",
            "bt_stop_loss_drawdown_pct",
            "bt_stop_loss_consecutive_limit_down",
            "position_sizing",
            "kelly_vol_window",
            "kelly_max_leverage",
            "stagger_tranches",
            "enable_early_rebalance_on_empty",
        )
        return params

    if not params.get("bt_stop_loss_enabled"):
        clear("bt_stop_loss_drawdown_pct", "bt_stop_loss_consecutive_limit_down")

    if params.get("position_sizing") not in ("kelly", "half_kelly"):
        clear("kelly_vol_window", "kelly_max_leverage")

    return params


def _collect_data_state_fields(args, wf_run_id: str, output_path: str) -> Dict[str, Any]:
    """采集数据态血缘：摘要列并入 summary，完整快照落盘为 data_state JSON。

    数据态采集失败仅告警并返回空列，不影响训练结果输出。
    """
    from .data_state import collect_data_state, data_state_summary_columns, write_data_state_file

    try:
        state = collect_data_state(
            data_root=getattr(args, "data_root", None) or get_data_root(),
            wf_run_id=wf_run_id,
            batch_run_id=getattr(args, "batch_run_id", None),
        )
    except Exception as exc:
        logger.warning(f"采集数据态血缘失败，本次汇总不含数据态列: {exc}")
        return {}

    write_data_state_file(Path(output_path).parent, state)
    summary_cols = data_state_summary_columns(state)
    logger.info(
        f"数据态血缘: id={summary_cols['data_state_id']}"
        f" git={summary_cols['git_commit']}"
        f" daily={summary_cols['data_daily_latest']}"
    )
    return summary_cols


def write_walk_forward_summary(results: List[Dict], output_path: str, args, wf_run_id: str) -> None:
    """将所有 walk-forward split 的指标和公共参数写入 CSV。"""
    if not results:
        logger.warning("没有结果可以写入汇总文件")
        return

    logger.info(f"生成 walk-forward 汇总文件: {output_path}")
    derived_wf_start_date = getattr(args, "wf_start_date", results[0]["train_start"])
    derived_wf_end_date = getattr(args, "wf_end_date", results[-1]["test_end"])

    data_state_cols = _collect_data_state_fields(args, wf_run_id, output_path)
    train_params_cols = {
        "wf_run_id": wf_run_id,
        "batch_run_id": getattr(args, "batch_run_id", None),
        "batch_period_label": getattr(args, "batch_period_label", None),
        "split_count": getattr(args, "split_count", len(results)),
        "final_date": getattr(args, "final_date", derived_wf_end_date),
        "wf_start_date": derived_wf_start_date,
        "wf_end_date": derived_wf_end_date,
        "algorithm": args.algorithm,
        "train_window_years": args.train_window_years,
        "test_window_months": args.test_window_months,
        "val_ratio": args.val_ratio,
        "label_column": args.label_column,
        "neutral_label_blend_weight": getattr(args, "neutral_label_blend_weight", 0.0),
        "task": args.task,
        "label_transform": args.label_transform if args.task == "regression" else None,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "num_leaves": getattr(args, "num_leaves", None),
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "gamma": args.gamma,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "early_stopping_rounds": args.early_stopping_rounds,
        "early_stopping_metric": args.early_stopping_metric,
        "rank_weight_enabled": args.rank_weight_enabled,
        "rank_weight_topk": args.rank_weight_topk,
        "rank_weight": args.rank_weight,
        "rank_weight_topk_weight_mode": getattr(
            args, "rank_weight_topk_weight_mode", "linear_decay"
        ),
        "time_decay_half_life": args.time_decay_half_life,
        "freshness_strategy": getattr(args, "freshness_strategy", "state_keep_event_decay"),
        "event_freshness_half_life_days": getattr(args, "event_freshness_half_life_days", 45.0),
        "objective": getattr(args, "objective", "mse"),
        "enable_fundamental": args.enable_fundamental_features,
        "enable_alt": args.enable_alt_features,
        "enable_margin": args.enable_margin_features,
        "enable_cyq": args.enable_cyq_features,
        "enable_fund": args.enable_fund_features,
        "enable_express": args.enable_express_features,
        "feature_stability_filter": args.feature_stability_filter,
        "factor_prune": getattr(args, "factor_prune", False),
        "factor_exclude_file": getattr(args, "factor_exclude_file", None),
        "ensemble_offsets": getattr(args, "ensemble_offsets", 0),
        "ensemble_seeds": getattr(args, "ensemble_seeds", None),
        "ensemble_seed_keep_top_ratio": getattr(args, "ensemble_seed_keep_top_ratio", None),
        "ensemble_seed_keep_min_models": getattr(args, "ensemble_seed_keep_min_models", None),
        "enable_enhanced_features": getattr(args, "enable_enhanced_features", False),
        "enable_north_features": getattr(args, "enable_north_features", False),
        "enable_lhb_features": getattr(args, "enable_lhb_features", False),
        "enable_consensus_features": getattr(args, "enable_consensus_features", False),
        "enable_cashflow_quality_features": getattr(
            args, "enable_cashflow_quality_features", False
        ),
        "enable_consensus_revision_features": getattr(
            args, "enable_consensus_revision_features", False
        ),
        "enable_dividend_policy_features": getattr(args, "enable_dividend_policy_features", False),
        "oos_backtest": getattr(args, "oos_backtest", False),
        "oos_backtest_months": getattr(args, "oos_backtest_months", None),
        "bt_top_n": getattr(args, "bt_top_n", None),
        "bt_rebalance_freq": getattr(args, "bt_rebalance_freq", None),
        "bt_initial_capital": getattr(args, "bt_initial_capital", None),
        "bt_sell_timing": getattr(args, "bt_sell_timing", "open"),
        "bt_exclude_st": getattr(args, "bt_exclude_st", True),
        "bt_min_list_days": getattr(args, "bt_min_list_days", 365),
        "bt_max_weight_per_stock": getattr(args, "bt_max_weight_per_stock", None),
        "bt_max_per_industry": getattr(args, "bt_max_per_industry", None),
        "bt_stop_loss_enabled": getattr(args, "bt_stop_loss_enabled", False),
        "bt_stop_loss_drawdown_pct": getattr(args, "bt_stop_loss_drawdown_pct", 30.0),
        "bt_stop_loss_consecutive_limit_down": getattr(
            args, "bt_stop_loss_consecutive_limit_down", 2
        ),
        "position_sizing": getattr(args, "position_sizing", "equal"),
        "kelly_vol_window": getattr(args, "kelly_vol_window", 60),
        "kelly_max_leverage": getattr(args, "kelly_max_leverage", 0.25),
        "stagger_tranches": getattr(args, "stagger_tranches", 1),
        "enable_early_rebalance_on_empty": getattr(args, "enable_early_rebalance_on_empty", True),
        "no_deploy_train": getattr(args, "no_deploy_train", False),
        "skip_training": getattr(args, "skip_training", False),
        "start_model_version": getattr(args, "start_model_version", None),
        "selected_split_indices": getattr(args, "selected_split_indices", None),
    }
    train_params_cols = _sanitize_train_params(train_params_cols)
    # 数据态血缘摘要列（采集失败时为空 dict，历史对比按缺失列处理）
    train_params_cols.update(data_state_cols)

    summary_rows = []
    for result in results:
        test_daily = result.get("test_daily_metrics", {})
        row = {
            **_build_summary_key_fields(test_daily),
            "split_index": result["split_index"],
            "train_start": result["train_start"],
            "train_end": result["train_end"],
            "test_start": result["test_start"],
            "test_end": result["test_end"],
            "model_version": result["model_version"],
            "consensus_revision_cols_live": _live_consensus_revision_cols(result),
            "cashflow_quality_cols_live": _live_cashflow_quality_cols(result),
            "train_samples": result.get("train_samples"),
            "val_samples": result.get("val_samples"),
            "test_samples": result.get("test_samples"),
            "best_iteration": result.get("best_iteration"),
            "val_rankic_ir": result.get("val_rankic_ir"),
        }
        row.update(test_daily)
        row.update(result.get("bt_metrics", {}))
        row.update(train_params_cols)
        summary_rows.append(row)

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"汇总文件已保存: {output_path}")
    logger.info(f"  共 {len(summary_rows)} 个切分")
