"""AD-ML-20-v1 冻结规格（唯一来源）。

本模块是协议 `docs/experiments/p7_analyst_diffusion_ml_protocol_20261118.md` 的机器镜像。
任何会改变数值输出的改动都必须新 campaign；本模块只登记，不提供扫描入口。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

CANDIDATE_ID = "AD-ML-20-v1"
PROTOCOL_VERSION = "0.1"
PROTOCOL_RELATIVE = "docs/experiments/p7_analyst_diffusion_ml_protocol_20261118.md"
DEFAULT_CAMPAIGN = "adml20_v1_20261118"

#: 研究正式工件根（不得写到生产根）
ARTIFACT_ROOT_RELATIVE = "data/reports/p7_analyst_diffusion_ml_20261118"

#: 冻结 15 列特征（顺序即训练列序，禁止自动增删）
FEATURES: tuple[str, ...] = (
    "cons_eps_revision_accel",
    "cons_eps_dispersion",
    "cons_eps_dispersion_chg",
    "cons_target_upside_chg",
    "cons_analyst_count_chg",
    "cons_rating_upgrade_ratio",
    "cons_revision_freshness_days",
    "ad_eps_net_diffusion_90",
    "ad_eps_up_ratio_90",
    "ad_eps_revision_median_90",
    "ad_eps_revision_breadth_90",
    "ad_target_net_diffusion_90",
    "ad_rating_net_diffusion_90",
    "ad_active_analysts_90_log",
    "ad_latest_report_age_days",
)

#: 前 7 列来自 `factors/consensus_revision.py`（沿用其既有 winsorize 语义）
EXISTING_FEATURES: tuple[str, ...] = FEATURES[:7]
NEW_FEATURES: tuple[str, ...] = FEATURES[7:]

#: 只进台账、不进训练列的分母与计数
DENOMINATOR_COLUMNS: tuple[str, ...] = (
    "ad_n_active_90",
    "ad_n_eps_90",
    "ad_n_eps_up_90",
    "ad_n_eps_down_90",
    "ad_n_eps_flat_90",
    "ad_n_target_90",
    "ad_n_target_up_90",
    "ad_n_target_down_90",
    "ad_n_rating_90",
    "ad_n_rating_up_90",
    "ad_n_rating_down_90",
)

#: 研报事件口径（第 5 节）
REPORT: dict[str, Any] = {
    "revision_lookback_days": 180,
    "diffusion_window_days": 90,
    "fy_offsets": [0, 1],
    "eps_abs_tolerance_yuan": 0.01,
    "eps_relative_tolerance": 0.01,
    "eps_denominator_floor_yuan": 0.10,
    "eps_relative_clip": 2.0,
    "target_tolerance": 0.01,
    "rating_tolerance": 0.0,
    "max_eps_fy_match": True,
    "require_named_analyst": True,
}

#: 信号域与资格（第 3 / 4 节）
UNIVERSE: dict[str, Any] = {
    "markets": ["主板"],
    "exclude_financial": True,
    "exclude_st": True,
    "min_list_days": 252,
    "min_total_mv_yuan": 5.0e9,
    "max_total_mv_yuan": 1.5e11,
    "min_amount20_yuan": 5.0e7,
    "min_scored_per_day": 100,
    "min_active_analysts_90": 2,
    # 协议 §4.1 为「20 日平均成交额」，不得用更短观察期放宽域
    "amount_min_periods": 20,
}

#: 标签（第 7 节）
LABEL: dict[str, Any] = {
    "horizon": 20,
    "entry_offset": 1,
    "exit_offset": 21,
    "entry_price": "close_adj",
    "exit_price": "open_adj",
    "formula": "open_adj(T+21)/close_adj(T+1)-1",
    "neutralize": "industry_l1_median_on_base_eligible_domain",
    "min_industry_members": 5,
    "column": "y_ad_neu20",
    "raw_column": "y_ad_raw20",
}

#: 模型（第 8.1 节）
MODEL: dict[str, Any] = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "device": "cpu",
    "max_depth": 3,
    "n_estimators": 300,
    "learning_rate": 0.03,
    "min_child_weight": 200.0,
    "reg_lambda": 10.0,
    "reg_alpha": 0.0,
    "gamma": 0.0,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "max_bin": 128,
    "random_state": 42,
    "max_n_jobs": 8,
    "early_stopping": False,
    "winsorize": False,
    "standardize": False,
}

#: 时间切分（第 8.2 节）
SPLITS: dict[str, Any] = {
    "refresh_anchor": "20190401",
    "refresh_months": 6,
    "train_months": 72,
    "val_months": 6,
    "sample_every_trading_days": 5,
    "sample_anchor": "20100104",
    "expected_folds": 16,
    "models_per_fold": 1,
    "expected_models": 16,
    "min_train_span_months": 36,
    "min_sampled_train_days": 100,
    "min_sampled_val_days": 10,
}

#: 组合与生命周期（第 3 / 10 节）
PORTFOLIO: dict[str, Any] = {
    "top_n": 6,
    "holding_period": 20,
    "rebalance_freq": 20,
    "stagger_tranches": 2,
    "tranche_slots": [3, 3],
    "tranche_offsets": [0, 10],
    "position_sizing": "equal",
    "max_per_industry": 2,
    "max_single_stock_weight": 0.20,
    "buy_phase": "next_close",
    "sell_phase": "open",
    "buy_retry_days": 0,
    "buy_retries_per_day": 0,
    "fill_window_days": 0,
    "score_formula": "xgboost_prediction_industry_neutral_20d",
}

#: 成本（第 10.4 节）
COSTS: dict[str, Any] = {
    "commission_rate": 0.0001954,
    "min_commission": 5.0,
    "stamp_tax": 0.0005,
    "slippage": 0.0005,
}

#: 单笔买入下限（第 10.3 节）
MIN_TICKET_FLOOR_YUAN = 25000.0


def min_ticket_value(costs: dict[str, Any] | None = None) -> float:
    """`V = max(25000, min_commission / commission_rate)`。"""
    table = costs or COSTS
    return float(
        max(MIN_TICKET_FLOOR_YUAN, float(table["min_commission"]) / float(table["commission_rate"]))
    )


#: 经济目标（第 2.1 节；只决定资源建议，不是正式准入门）
TARGETS: dict[str, Any] = {
    "cagr_delta_min": 0.005,
    "mdd_loss_delta_max": 0.01,
    "must_beat_cash_arm": True,
    "capital_scenarios": [1_000_000, 2_000_000],
    "initial_b_fraction": 0.2,
    "arms": ["A100", "A80_CASH20", "A80_AD20", "A80_AD_SIMPLE20"],
}

#: 窗口（第 3 / 13.1 节）
WINDOW: dict[str, Any] = {
    "cash_anchor": "20190329",
    "start": "20190401",
    "end": "20261029",
    "segments": [
        ["20190401", "20211231"],
        ["20220104", "20241231"],
        ["20250102", "20261029"],
    ],
    "startup_exclusion_days": 20,
    #: 数据集装载起点 = 首折训练起点（L），由 run 阶段按 folds 计算后校验
    "dataset_load_warmup_trading_days": 60,
}

#: 描述性重抽样（第 13.2 节）
BOOTSTRAP: dict[str, Any] = {
    "method": "circular_block",
    "block": 20,
    "repetitions": 2000,
    "seed": 20261118,
}

#: 固定压力（第 13.3 节）
STRESS: dict[str, Any] = {
    "report_lag_trading_days": 1,
    "slippage_doubled": 0.0010,
    "cash_equity_discount": 0.20,
    "zero_recovery_when_unresolved": True,
}

#: A 基准（第 10.1 节）
A_BASELINE: dict[str, Any] = {
    "relative": "data/reports/p7_operating_monthly_economic_20261116/run_r3",
    "expected_manifest_sha256": (
        "541933bbbf3ecc2bc96a5bb572bdc8fb16cedde6b369c302b3501b4c19d5fd81"
    ),
    "required_arm_dirs": [
        "1000000/A100",
        "1000000/A80",
        "2000000/A100",
        "2000000/A80",
    ],
}

#: 证据输入（R0 核验对象）
INPUTS: tuple[dict[str, Any], ...] = (
    {"name": "raw_report_rc", "relative": "data/raw/report_rc", "kind": "partition"},
    {"name": "clean_daily", "relative": "data/clean/daily", "kind": "partition"},
    {"name": "clean_daily_basic", "relative": "data/clean/daily_basic", "kind": "partition"},
    {"name": "clean_stock_basic", "relative": "data/clean/stock_basic.parquet", "kind": "file"},
    {"name": "clean_trade_cal", "relative": "data/clean/trade_cal.parquet", "kind": "file"},
    {
        "name": "clean_industry_pit",
        "relative": "data/clean/sw_industry_pit.parquet",
        "kind": "file",
    },
    {
        "name": "clean_industry_table",
        "relative": "data/clean/sw_industry_code_table.parquet",
        "kind": "file",
    },
    {"name": "raw_dividend", "relative": "data/raw/dividend", "kind": "partition"},
    {"name": "raw_stock_st", "relative": "data/raw/stock_st", "kind": "partition"},
    {"name": "raw_suspend", "relative": "data/raw/suspend", "kind": "partition"},
    {"name": "raw_stk_limit", "relative": "data/raw/stk_limit", "kind": "partition"},
    {"name": "raw_stock_basic", "relative": "data/raw/stock_basic.parquet", "kind": "file"},
)

#: 阶段清单（第 12 节）
STAGES: tuple[str, ...] = (
    "readiness",
    "dataset",
    "model_probe",
    "train_predict",
    "replay",
    "diagnostics",
    "report",
)

#: 选择史（第 2.2 / 16 节；R0 必须原样登记）
SELECTION_HISTORY: tuple[dict[str, str], ...] = (
    {
        "direction": "A 中分析师水平值",
        "relation": "既有生产 A 使用评级/目标价/EPS 水平",
        "difference": "本袖只用同分析师前后修正与分析师间扩散，不读 A 分数或仓位",
    },
    {
        "direction": "B0 业绩预告/快报/正式财报事件",
        "relation": "既有事件袖消费公司三源公告",
        "difference": "本袖事件主体是卖方研报修正，不消费 forecast/express/fina",
    },
    {
        "direction": "旧 C5 分析师修正设想",
        "relation": "旧材料只有候选描述、无经济试验",
        "difference": "本文件冻结宽域、同分析师同 FY 配对与 ML 组合；保留其候选选择史",
    },
    {
        "direction": "固定续持 / 低波 / 赎回压力等近期路线",
        "relation": "同期并行提出的方向",
        "difference": "本袖用新输入、新标签、新模型与新资金路径，不读取其结果调参",
    },
)


def protocol_path(root: Path | None = None) -> Path:
    """协议原文路径。"""
    return (root or REPO_ROOT) / PROTOCOL_RELATIVE


def protocol_sha256(root: Path | None = None) -> str:
    """协议原文的字节 SHA256（R0 封存对象，不填占位）。"""
    return hashlib.sha256(protocol_path(root).read_bytes()).hexdigest()


def numeric_identity() -> dict[str, Any]:
    """会改变数值输出的全部冻结参数（不含墙钟、路径与运行元数据）。"""
    return {
        "candidate_id": CANDIDATE_ID,
        "protocol_version": PROTOCOL_VERSION,
        "features": list(FEATURES),
        "feature_count": len(FEATURES),
        "report": REPORT,
        "universe": UNIVERSE,
        "label": LABEL,
        "model": MODEL,
        "splits": SPLITS,
        "portfolio": PORTFOLIO,
        "costs": COSTS,
        "min_ticket_value_yuan": min_ticket_value(),
        "targets": TARGETS,
        "window": WINDOW,
        "bootstrap": BOOTSTRAP,
        "stress": STRESS,
    }


def spec_sha256() -> str:
    """冻结规格的规范化 JSON SHA256（数值身份）。"""
    payload = json.dumps(
        numeric_identity(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def machine_spec_document(protocol_sha: str | None = None) -> dict[str, Any]:
    """第 18.1 节机器规格文档（字段全部为真实值，无占位）。"""
    return {
        "candidate_id": CANDIDATE_ID,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_path": PROTOCOL_RELATIVE,
        "protocol_sha256": protocol_sha or protocol_sha256(),
        "spec_sha256": spec_sha256(),
        "purpose": "economic_prescreen",
        "intended_admission_type": "enhancement",
        "formal_admission": "not_evaluated",
        "capital_scenarios": TARGETS["capital_scenarios"],
        "initial_b_fraction": TARGETS["initial_b_fraction"],
        "horizon": LABEL["horizon"],
        "label": "industry_median_neutralized_open_T21_over_close_T1",
        "feature_count": len(FEATURES),
        "features": list(FEATURES),
        "model_count_expected": SPLITS["expected_models"],
        "top_n": PORTFOLIO["top_n"],
        "stagger_tranches": PORTFOLIO["stagger_tranches"],
        "min_scored_per_day": UNIVERSE["min_scored_per_day"],
        "min_active_analysts_90": UNIVERSE["min_active_analysts_90"],
        "revision_lookback_days": REPORT["revision_lookback_days"],
        "diffusion_window_days": REPORT["diffusion_window_days"],
        "random_seed": MODEL["random_state"],
        "not_virgin_market_holdout": True,
        "production_changed": False,
        "data_purchase_yuan": 0,
        "window": WINDOW,
        "targets": TARGETS,
        "costs": COSTS,
        "min_ticket_value_yuan": min_ticket_value(),
        "portfolio": PORTFOLIO,
        "model": MODEL,
        "splits": SPLITS,
        "bootstrap": BOOTSTRAP,
        "stress": STRESS,
        "a_baseline": A_BASELINE,
        "selection_history": [dict(item) for item in SELECTION_HISTORY],
        "inputs": [dict(item) for item in INPUTS],
    }


def n_jobs(logical_cpus: int) -> int:
    """线程数 = min(8, 可用逻辑 CPU)。"""
    return max(1, min(int(MODEL["max_n_jobs"]), int(logical_cpus)))
