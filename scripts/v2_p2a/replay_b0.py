# -*- coding: utf-8 -*-
"""B0 基线重放驱动（P2a-T7）：v2 内核 native replay → runs 契约批次。

B0 配置按 ``docs/contracts/baseline_freeze.md`` §2 写死为常量（其来源 =
``scripts/batch/batch_walk_forward.ps1`` 配置区启动日 2026-09-30 当前值）；
127 指纹键逐键从**实际运行输入**组装（窗口/模型版本实取自切分与折模型
元数据，禁止从冻结 CSV 抄值）。

只读过渡依赖登记（P2a 规划 §3.5 新增行，T7）：

| 依赖 | 形态 | 去除节点 |
|---|---|---|
| ``common.config``（get_data_root / get_stock_selection_models_root） | 只读（全局单例） | P4 展平/切换评审 |
| ``data``（Storage / DataLoader） | 只读过渡 | P2b/P3 数据面切换 |
| ``ml.walk_forward.utils``（切分生成） | 只读过渡 | P3 WF 编排迁移段 |
| ``ml.walk_forward.data_state``（数据态采集） | 只读过渡 | P3 WF 编排迁移段 |
| ``ml.train_core.constants``（CONSENSUS_REVISION_FEATURE_COLUMNS） | 只读 | P3 train/* 归位 |
| ``factors.cashflow_quality``（cashflow_quality_live_columns） | 只读 | 永久（factors 沿用） |
| ``universe.domains``（resolve_stock_domain） | 只读 | 永久（universe 沿用） |

**机器时间纪律**：本脚本读取真实生产数据与折模型，属规划 §5 门 1-5 的机器
时间任务（预检 3 折 ≈30min / 全量 14 折 1.5~2h，分次申请），不进 pytest。

用法：
    python scripts/v2_p2a/replay_b0.py                      # 全量 14 折
    python scripts/v2_p2a/replay_b0.py --splits 0,1,2       # 预检子集
    python scripts/v2_p2a/replay_b0.py --data-root ./data --runs-root ./data/runs
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from dateutil.relativedelta import relativedelta
from loguru import logger

# 脚本直接运行时把仓库根插入 sys.path（v2 包用 src.lazybull 绝对导入，同
# scripts/v2_p1、v2_p5a1 既有先例）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.lazybull.common.config import (
    get_cost_settings,
    get_data_root,
    get_stock_selection_models_root,
)
from src.lazybull.data import DataLoader, Storage
from src.lazybull.factors.cashflow_quality import cashflow_quality_live_columns
from src.lazybull.ml.train_core.constants import CONSENSUS_REVISION_FEATURE_COLUMNS
from src.lazybull.ml.walk_forward.data_state import collect_data_state, compute_data_state_id
from src.lazybull.ml.walk_forward.utils import (
    WalkForwardSplit,
    generate_walk_forward_splits_by_count,
)
from src.lazybull.universe.domains import resolve_stock_domain
from src.lazybull.v2.core.signal.ml_signal import MLSignal
from src.lazybull.v2.evidence.fingerprint_keys import fingerprint
from src.lazybull.v2.hosts.backtest.replay import run_oos_backtest
from src.lazybull.v2.hosts.backtest.runs_writer import write_runs_batch

# ── B0 配置常量（baseline_freeze.md §2 写死；指纹键值与冻结批次的核对 = 验收门 3） ──
B0_SPLIT_COUNT = 14
B0_FINAL_DATE = "20260105"
B0_START_MODEL_VERSION = 24008
B0_LABEL_COLUMN = "neu_y_ret_20"
B0_TRAIN_WINDOW_YEARS = 6
B0_TEST_WINDOW_MONTHS = 6
# oos_backtest_months：批脚本传 0，runner 原地解析为 test_window_months（=6）后入 summary
B0_OOS_BACKTEST_MONTHS = 6
B0_BT_TOP_N = 20
B0_BT_REBALANCE_FREQ = 20
B0_BT_INITIAL_CAPITAL = 1000000
B0_BT_SELL_TIMING = "open"
B0_BT_MIN_LIST_DAYS = 365
B0_BT_MAX_WEIGHT_PER_STOCK = 0.15
B0_STAGGER_TRANCHES = 2
B0_POSITION_SIZING = "half_kelly"
B0_KELLY_VOL_WINDOW = 60
B0_KELLY_MAX_LEVERAGE = 0.2
B0_STOCK_DOMAIN = "main"

#: 静态配置段（batch 级公共键；逐折差异键 cols_live 经 config_overrides 注入）
B0_STATIC_CONFIG: Dict[str, object] = {
    # 训练窗口与切分
    "split_count": B0_SPLIT_COUNT,
    "final_date": B0_FINAL_DATE,
    "train_window_years": B0_TRAIN_WINDOW_YEARS,
    "test_window_months": B0_TEST_WINDOW_MONTHS,
    "val_ratio": 0.2,
    # 标签与任务
    "algorithm": "xgboost",
    "label_column": B0_LABEL_COLUMN,
    "neutral_label_blend_weight": 0,
    "task": "regression",
    "label_transform": "cs_zscore",
    "objective": "mse",
    # 模型超参
    "n_estimators": 500,
    "max_depth": 5,
    "num_leaves": 63,  # 仅 LightGBM 有效，XGBoost 忽略（批脚本配置区原值）
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.3,
    "min_child_weight": 200,
    "gamma": 0.5,
    "reg_alpha": 0.03,
    "reg_lambda": 7,
    "early_stopping_rounds": 50,
    "early_stopping_metric": "rank_ic_daily",
    # rank-weight / 时间衰减 / freshness
    "rank_weight_enabled": True,
    "rank_weight_topk": 120,
    "rank_weight": 100,
    "rank_weight_topk_weight_mode": "linear_decay",
    "time_decay_half_life": 0,
    "freshness_strategy": "state_keep_event_decay",
    "event_freshness_half_life_days": 120,
    # 因子开关（baseline_freeze §2「因子开关」行）
    "enable_fundamental": True,
    "enable_alt": True,
    "enable_margin": True,
    "enable_cyq": True,
    "enable_fund": True,
    "enable_express": True,
    "enable_north_features": True,
    "enable_lhb_features": True,
    "enable_consensus_features": True,
    "enable_enhanced_features": True,
    "enable_cashflow_quality_features": True,
    "enable_consensus_revision_features": False,
    "enable_dividend_policy_features": False,
    "enable_availability_markers": False,
    "enable_holdertrade_features": False,
    "holdertrade_feature_set": "full",
    "enable_repurchase_features": False,
    "repurchase_feature_set": "full",
    "enable_top10fh_features": False,
    "top10fh_feature_set": "full",
    "enable_top_inst_features": False,
    "feature_stability_filter": False,
    "factor_prune": False,
    "factor_exclude_file": None,
    # 集成
    "ensemble_offsets": 0,
    "ensemble_seeds": "42,61,82",
    "ensemble_seed_keep_top_ratio": 1,
    "ensemble_seed_keep_min_models": 3,
    # 回测配置（B0 = 无暴露政策基线；exposure 五参全默认，由 replay fail-fast 兜底）
    "stock_domain": B0_STOCK_DOMAIN,
    "oos_backtest": True,
    "oos_backtest_months": B0_OOS_BACKTEST_MONTHS,
    "bt_top_n": B0_BT_TOP_N,
    "bt_rebalance_freq": B0_BT_REBALANCE_FREQ,
    "bt_initial_capital": B0_BT_INITIAL_CAPITAL,
    "bt_sell_timing": B0_BT_SELL_TIMING,
    "bt_exclude_st": True,
    "bt_min_list_days": B0_BT_MIN_LIST_DAYS,
    "bt_max_weight_per_stock": B0_BT_MAX_WEIGHT_PER_STOCK,
    "bt_max_per_industry": None,
    "bt_stop_loss_enabled": False,
    "bt_stop_loss_drawdown_pct": None,  # sanitize 口径：未启用止损时子参数清空
    "bt_stop_loss_consecutive_limit_down": None,
    "position_sizing": B0_POSITION_SIZING,
    "kelly_vol_window": B0_KELLY_VOL_WINDOW,
    "kelly_max_leverage": B0_KELLY_MAX_LEVERAGE,
    "stagger_tranches": B0_STAGGER_TRANCHES,
    "enable_early_rebalance_on_empty": True,
    "no_deploy_train": False,
    "skip_training": True,
    "skip_training_eval": False,
    "start_model_version": B0_START_MODEL_VERSION,
    "selected_split_indices": "[]",
    "downside_penalty": 0.0,
    "downside_penalty_column": None,  # sanitize 口径：λ=0 时清空
    # 训练产出统计（skip-training 复用折模型，无训练输出；排除键，不影响指纹）
    "train_samples": None,
    "val_samples": None,
    "test_samples": None,
    "best_iteration": None,
    "best_iteration_floor_triggered": None,
    "val_rankic_ir": None,
    # 运行标识类（排除键；batch_period_label 沿用批脚本时间段标签）
    "batch_period_label": "0101",
}


def _live_feature_columns(models_root: str, model_version: int) -> List[str]:
    """读取折模型 features.json 的特征列清单（skip-training 复用模型的实际入模列）。"""
    features_file = Path(models_root) / f"v{model_version}_features.json"
    features = json.loads(features_file.read_text(encoding="utf-8"))
    if not isinstance(features, list):
        raise ValueError(f"{features_file} 不是特征列清单（list）")
    return [str(c) for c in features]


def _live_config_overrides(feature_columns: List[str]) -> Dict[str, object]:
    """逐折 cols_live 配置键（口径同旧 summary.py 的 _live_*_cols 谓词）。

    依赖 ``ml.train_core.constants`` / ``factors.cashflow_quality`` 的原始谓词
    集合（只读过渡/永久沿用，见模块头登记表），禁止重写谓词逻辑。
    """
    base_cols = {
        col for col in CONSENSUS_REVISION_FEATURE_COLUMNS if col != "cons_revision_freshness_days"
    }
    live_cons = sorted(
        col
        for col in feature_columns
        if col in base_cols or (col.startswith("zscore_cons_") and col.endswith("_sz"))
    )
    return {
        "consensus_revision_cols_live": ",".join(live_cons) or None,
        "cashflow_quality_cols_live": ",".join(cashflow_quality_live_columns(feature_columns))
        or None,
    }


def build_batch_config(splits: Sequence[WalkForwardSplit]) -> Dict[str, object]:
    """组装批次 config 字典（127 指纹键；窗口键实取自切分，禁止抄冻结 CSV）。"""
    config = dict(B0_STATIC_CONFIG)
    config["wf_start_date"] = splits[0].train_start
    config["wf_end_date"] = splits[-1].test_end
    return config


def build_manifest_config_snapshot(
    splits: Sequence[WalkForwardSplit],
) -> Dict[str, object]:
    """门 1 有效配置快照（R2-T7-02）：与宿主实际消费同源的非敏感配置全集。

    = 驱动 config（``build_batch_config``，127 指纹键）+ **成本设置**
    （``get_cost_settings()``——引擎 ``CostModel`` 构造的同一读取源，
    v2/旧两侧共享同一单例；驱动常量不覆盖成本参数，缺它则成本配置漂移
    无法被门 1 捕获）。
    """
    return {
        "driver_config": build_batch_config(splits),
        "cost_settings": {k: float(v) for k, v in get_cost_settings().items()},
    }


def _generate_b0_splits(trade_cal) -> List[WalkForwardSplit]:
    """B0 全量 14 折切分（单一来源：主流程与门 1 快照构建共用）。

    与旧 runner 调用逐字一致（rebalance_freq=20 将 test_end 对齐调仓边界；
    省略则折窗口整体错位——实测 None 与 20 的 split0 窗口不同，B0 冻结窗口仅后者复现）。
    """
    return generate_walk_forward_splits_by_count(
        trade_cal=trade_cal,
        split_count=B0_SPLIT_COUNT,
        final_date=B0_FINAL_DATE,
        train_window_years=B0_TRAIN_WINDOW_YEARS,
        test_window_months=B0_TEST_WINDOW_MONTHS,
        rebalance_freq=B0_BT_REBALANCE_FREQ,
    )


def _load_splits_for_snapshot(data_root: str) -> List[WalkForwardSplit]:
    """为快照构建加载交易日历并生成全量 14 折切分（与驱动主流程同参数）。"""
    storage = Storage(root_path=data_root)
    loader = DataLoader(storage)
    trade_cal = loader.load_clean_trade_cal()
    if trade_cal is None:
        trade_cal = loader.load_trade_cal()
    if trade_cal is None or len(trade_cal) == 0:
        raise RuntimeError("无法加载交易日历（clean/raw 均缺失）")
    return _generate_b0_splits(trade_cal)


def make_batch_id(config: Dict[str, object], now: Optional[datetime] = None) -> str:
    """批次 ID：``p2a_b0replay_<YYYYMMDD>_<HHMMSS>_<fp6>``（fp6 = 静态配置指纹前 6 位）。"""
    now = now or datetime.now()
    fp6 = fingerprint(config)[:6]
    return f"p2a_b0replay_{now:%Y%m%d_%H%M%S}_{fp6}"


def replay_b0(
    *,
    splits_subset: Optional[Sequence[int]] = None,
    data_root: Optional[str] = None,
    runs_root: Optional[str] = None,
    batch_id: Optional[str] = None,
) -> Path:
    """B0 重放主流程：切分 → 逐折 v2 OOS 回测 → runs 契约批次写出。

    Args:
        splits_subset: 折子集（预检用，如 [0, 1, 2]）；None = 全量 14 折
        data_root: 数据根目录（None = 配置默认）
        runs_root: runs 根目录（None = <data_root>/runs）
        batch_id: 批次 ID（None = 自动生成 p2a_b0replay_<时间戳>_<fp6>）
    """
    data_root = data_root or get_data_root()
    runs_root = runs_root or str(Path(data_root) / "runs")
    storage = Storage(root_path=data_root)
    loader = DataLoader(storage)
    stock_basic = loader.load_clean_stock_basic()
    if stock_basic is None:
        stock_basic = loader.load_stock_basic()
    if stock_basic is None:
        raise RuntimeError("无法加载 stock_basic（clean/raw 均缺失）")
    trade_cal = loader.load_clean_trade_cal()
    if trade_cal is None:
        trade_cal = loader.load_trade_cal()
    if trade_cal is None or len(trade_cal) == 0:
        raise RuntimeError("无法加载交易日历（clean/raw 均缺失）")

    splits = _generate_b0_splits(trade_cal)
    if splits_subset is not None:
        selected = set(int(i) for i in splits_subset)
        splits = [s for s in splits if s.split_index in selected]
        if not splits:
            raise ValueError(f"--splits 子集 {sorted(selected)} 与生成的切分无交集")

    config = build_batch_config(splits)
    batch_id = batch_id or make_batch_id(config)
    config["batch_run_id"] = batch_id
    logger.info(f"B0 重放批次: {batch_id}（{len(splits)} 折）")

    # 数据态血缘（只读过渡依赖，见模块头登记表）
    data_state = collect_data_state(data_root, wf_run_id=batch_id, batch_run_id=batch_id)
    data_state["data_state_id"] = compute_data_state_id(data_state)
    code_state = {
        "git_commit": data_state.get("git_commit"),
        "git_dirty": data_state.get("git_dirty"),
    }

    # 跨折持久化 MLSignal（v2 信号；构造参数同旧 runner.py:508 装配段）
    models_root = get_stock_selection_models_root(data_root)
    bt_domain = resolve_stock_domain(B0_STOCK_DOMAIN)
    persistent_signal = MLSignal(
        top_n=B0_BT_TOP_N,
        model_version=None,  # 首折经 update_model_version 设置
        models_dir=models_root,
        min_total_mv=float(bt_domain["min_total_mv_wan"]),  # type: ignore[arg-type]
        max_total_mv=float(bt_domain["max_total_mv_wan"]),  # type: ignore[arg-type]
        verbose=False,
        downside_penalty=0.0,
        downside_penalty_column="downside_vol_20",
    )

    results: List[Dict[str, object]] = []
    for split in splits:
        model_version = B0_START_MODEL_VERSION + split.split_index
        bt_start = split.test_start
        bt_end = (
            datetime.strptime(bt_start, "%Y%m%d") + relativedelta(months=B0_OOS_BACKTEST_MONTHS)
        ).strftime("%Y%m%d")
        logger.info(f"[Split {split.split_index}] v{model_version} 回测 {bt_start} ~ {bt_end}")
        metrics = run_oos_backtest(
            model_version=model_version,
            bt_start=bt_start,
            bt_end=bt_end,
            storage=storage,
            loader=loader,
            trade_cal=trade_cal,
            stock_basic=stock_basic,
            label_column=B0_LABEL_COLUMN,
            bt_top_n=B0_BT_TOP_N,
            bt_rebalance_freq=B0_BT_REBALANCE_FREQ,
            data_root=data_root,
            persistent_signal=persistent_signal,
            bt_exclude_st=True,
            bt_min_list_days=B0_BT_MIN_LIST_DAYS,
            bt_sell_timing=B0_BT_SELL_TIMING,
            bt_max_weight_per_stock=B0_BT_MAX_WEIGHT_PER_STOCK,
            bt_max_per_industry=None,
            bt_stop_loss_enabled=False,
            position_sizing=B0_POSITION_SIZING,
            kelly_vol_window=B0_KELLY_VOL_WINDOW,
            kelly_max_leverage=B0_KELLY_MAX_LEVERAGE,
            stagger_tranches=B0_STAGGER_TRANCHES,
            enable_early_rebalance_on_empty=True,
            initial_capital=float(B0_BT_INITIAL_CAPITAL),
            split_num=split.split_index,
            stock_domain=B0_STOCK_DOMAIN,
            # B0 四族运行时派生全关（baseline_freeze §2：enable_* = false）
            holdertrade_lookup=None,
            repurchase_lookup=None,
            top10fh_panel=None,
            top_inst_lookup=None,
        )
        if not metrics:
            raise RuntimeError(f"Split {split.split_index} OOS 回测返回空（数据缺失或日历异常）")
        feature_columns = _live_feature_columns(models_root, model_version)
        results.append(
            {
                "split_index": split.split_index,
                "model_version": model_version,
                "train_start": split.train_start,
                "train_end": split.train_end,
                "test_start": split.test_start,
                "test_end": split.test_end,
                "metrics": {k: v for k, v in metrics.items() if not k.startswith("_")},
                "nav_curve": metrics["_nav_curve"],
                "trades": metrics["_trades"],
                "attribution": metrics["_execution_attribution"],
                "holdings_snapshot": metrics["_holdings_snapshot"],
                "config_overrides": _live_config_overrides(feature_columns),
            }
        )

    out_dir = write_runs_batch(
        results,
        batch_id=batch_id,
        runs_root=Path(runs_root),
        config=config,
        code_state=code_state,
        data_state=data_state,
        baseline_ref=None,  # B0 基线批（契约 §2：基线批为 null）
        arms=(),
        source="v2_p2a_native_replay",
    )
    logger.info(f"B0 重放完成: {out_dir}")
    return out_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="B0 基线重放（P2a-T7，v2 内核 → runs 出口）")
    parser.add_argument("--splits", default=None, help="折子集（逗号分隔，如 0,1,2；预检用）")
    parser.add_argument("--data-root", default=None, help="数据根目录（默认取配置）")
    parser.add_argument("--runs-root", default=None, help="runs 根目录（默认 <data_root>/runs）")
    parser.add_argument("--batch-id", default=None, help="批次 ID（默认自动生成）")
    args = parser.parse_args(argv)

    splits_subset = (
        [int(token) for token in args.splits.split(",") if token.strip()] if args.splits else None
    )
    replay_b0(
        splits_subset=splits_subset,
        data_root=args.data_root,
        runs_root=args.runs_root,
        batch_id=args.batch_id,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
