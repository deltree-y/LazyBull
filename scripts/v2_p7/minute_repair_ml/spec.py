"""LR5M-ML-10-v1 冻结规格（唯一来源）。

本模块是协议 `docs/experiments/p7_minute_repair_ml_protocol_20261118.md` 的机器镜像。
任何会改变数值输出的改动都必须新 campaign；本模块只登记，不提供扫描入口。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

CANDIDATE_ID = "LR5M-ML-10-v1"
PROTOCOL_VERSION = "0.1"
PROTOCOL_RELATIVE = "docs/experiments/p7_minute_repair_ml_protocol_20261118.md"
DEFAULT_CAMPAIGN = "lr5m10_v1_20261118"

#: 研究正式工件根（不得写到生产根）
ARTIFACT_ROOT_RELATIVE = "data/reports/p7_minute_repair_ml_20261118"

#: 日线控制 6 列（顺序即训练列序）
DAILY_FEATURES: tuple[str, ...] = (
    "d_ret_1",
    "d_ret_5",
    "d_vol_20",
    "d_amount_rel_20",
    "d_turnover_20",
    "d_log_mv",
)

#: 分钟新增 18 列（顺序即训练列序）
MINUTE_FEATURES: tuple[str, ...] = (
    "m_am_ret",
    "m_pm_ret",
    "m_repair_ret",
    "m_path_dd",
    "m_repair_fraction",
    "m_low_time",
    "m_rv",
    "m_down_var_share",
    "m_neg_amount_share",
    "m_drop3_amount_share",
    "m_impact_am",
    "m_impact_pm",
    "m_impact_repair",
    "m_close_vwap_gap",
    "m_pm_amount_share",
    "m_late_amount_surprise",
    "m_core_amount_surprise",
    "m_amount_hhi",
)

#: 主模型 24 列（禁止自动增删）
FEATURES: tuple[str, ...] = DAILY_FEATURES + MINUTE_FEATURES

#: `minute_informed` 判据：18 列中至少 12 列有限
MIN_INFORMED_MINUTE_COLUMNS = 12

#: 分钟数据与时段（第 4/5 节）
MINUTE: dict[str, Any] = {
    "api": "stk_mins",
    "freq": "5min",
    "probe_freq": "1min",
    "fields": ["ts_code", "trade_time", "open", "close", "high", "low", "vol", "amount"],
    "page_limit": 8000,
    "chunk_trading_days": 120,
    "grid_slots": 48,
    "core_slots": [2, 47],
    "am_slots": [2, 24],
    "pm_slots": [26, 47],
    "late_slots": [42, 47],
    "auction_time": "09:30:00",
    "sse_closing_auction_from": "20180820",
    "rate_limit_start_per_minute": 100,
    #: 吞吐**实测**样本（预登记；不得按结果调整）：用接近真实块的 120 交易日窗口，
    #: 连续请求直到请求数或时长上限；样本期不足即判"未测得"，不做外推。
    "throughput_sample": {
        "max_requests": 300,
        "max_seconds": 120,
        "trading_days_per_request": 120,
        "min_requests_for_estimate": 30,
        "min_seconds_for_estimate": 20,
    },
    "rate_limit_ceiling_per_minute": 500,
    "max_recoverable_retries": 5,
    "r0_max_requests": 1000,
    "r0_max_seconds": 7200,
    "r0_sample_days": ["20130401", "20150803", "20190701", "20261012"],
    "r0_sample_window_days": 5,
    "research_start": "20160101",
    "max_request_trading_days": 30,
}

#: 单位与对账（第 5.5 节）
UNITS: dict[str, Any] = {
    # 分钟：vol=股、amount=元
    "minute_vol": "share",
    "minute_amount": "yuan",
    # 日线 clean/daily：vol=手、amount=千元
    "daily_vol": "hand",
    "daily_amount": "thousand_yuan",
    "daily_vol_to_share": 100.0,
    "daily_amount_to_yuan": 1000.0,
    # daily_basic.total_mv 单位为万元
    "total_mv": "ten_thousand_yuan",
    "total_mv_to_yuan": 10000.0,
    #: 协议 §5.5 的**调查线**（不是准入线）：越线须分类、披露、处置
    "minute_anomaly_volume_rel_error": 0.005,
    "minute_anomaly_ohlc_ticks": 1,
    "tick_size_yuan": 0.01,
    #: 机械可解释阈值：整手差、整元差
    "volume_rounding_shares": 100.0,
    "amount_rounding_yuan": 1.0,
}

#: 时段映射（R0 冻结；本值为本地 1 分钟实样证据推定，5 分钟须复核）
SESSION_MAPPING: dict[str, Any] = {
    "vendor_convention": "end",
    "auction_record": "separate_0930",
    "core_grid": "k=1..48 收盘时刻 09:35..11:30 / 13:05..15:00",
    "evidence": "本地 600000.SH 20261113 1 分钟 241 行：09:30 竞价 + 09:31..11:30 + 13:01..15:00",
    "verified_freqs": ["1min"],
    "unverified_freqs": ["5min"],
}

#: 资格（第 5.1 节）
UNIVERSE: dict[str, Any] = {
    "markets": ["主板"],
    "exclude_financial": True,
    "exclude_st": True,
    "min_list_days": 252,
    "min_total_mv_yuan": 5.0e9,
    "max_total_mv_yuan": 1.5e11,
    "min_amount20_yuan": 5.0e7,
    "amount_min_periods": 16,
    "amount_window": 20,
    "min_scored_per_day": 100,
    "below_min_share_max": 0.01,
    "below_min_max_streak": 5,
}

#: 标签（第 7 节）
LABEL: dict[str, Any] = {
    "horizon": 10,
    "entry_offset": 1,
    "exit_offset": 11,
    "entry_price": "close_adj",
    "exit_price": "open_adj",
    "formula": "open_adj(T+11)/close_adj(T+1)-1",
    "neutralize": "industry_l1_median_on_base_eligible_domain",
    "min_industry_members": 5,
    "column": "y_lr_neu10",
    "raw_column": "y_lr_raw10",
}

#: 模型（第 8.1 节）
MODEL: dict[str, Any] = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "device": "cpu",
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,
    "min_child_weight": 200.0,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 10.0,
    "reg_alpha": 0.0,
    "gamma": 0.0,
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
    "train_months": 36,
    "val_months": 6,
    "sample_every_trading_days": 5,
    "sample_anchor": "20100104",
    "expected_folds": 16,
    "models_per_fold": 2,
    "expected_deploy_models": 32,
    "expected_fits": 64,
    "min_train_span_months": 24,
    "min_sampled_train_days": 100,
    "min_sampled_val_days": 10,
}

#: 组合与生命周期（第 3/9 节）
PORTFOLIO: dict[str, Any] = {
    "top_n": 6,
    "holding_period": 10,
    "rebalance_freq": 10,
    "stagger_tranches": 2,
    "tranche_slots": [3, 3],
    "tranche_offsets": [0, 5],
    "position_sizing": "equal",
    "max_per_industry": 2,
    "max_single_stock_weight": 0.20,
    "intra_tranche_cap": 0.40,
    "buy_phase": "next_close",
    "sell_phase": "open",
    "buy_retry_days": 0,
    "buy_retries_per_day": 0,
    "fill_window_days": 0,
    "score_formula": "xgboost_prediction_industry_neutral_10d",
}

#: 成本（第 9.5 节）
COSTS: dict[str, Any] = {
    "commission_rate": 0.0001954,
    "min_commission": 5.0,
    "stamp_tax": 0.0005,
    "slippage": 0.0005,
}

#: 单笔买入下限（第 9.5 节）
MIN_TICKET_FLOOR_YUAN = 25000.0
LOT_SIZE = 100


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
    "minute_increment_min": 0.002,
    "capital_scenarios": [1_000_000, 2_000_000],
    "initial_b_fraction": 0.2,
    "arms": ["A100", "A80_CASH20", "A80_LR20", "A80_DAILY20"],
}

#: 窗口（第 3/11.2 节）
WINDOW: dict[str, Any] = {
    "cash_anchor": "20190329",
    "start": "20190401",
    "end": "20261029",
    "segments": [
        ["20190401", "20201231"],
        ["20210101", "20231231"],
        ["20240101", "20261029"],
    ],
    "dataset_load_warmup_trading_days": 60,
    "minute_preheat_trading_days": 20,
}

#: 描述性重抽样（第 11.3 节）
BOOTSTRAP: dict[str, Any] = {
    "method": "paired_circular_block",
    "block": 20,
    "repetitions": 2000,
    "seed": 42,
}

#: 固定压力（第 11.4 节）
STRESS: dict[str, Any] = {
    "b_slippage": 0.0010,
    "delay_trading_days": 1,
    "zero_recovery_when_unresolved": True,
}

#: 链式指标口径（第 11.2 节）
METRICS: dict[str, Any] = {
    "annual_trading_days": 252,
    "annual_risk_free_rate": 0.03,
}

#: A 基准（第 10 节）
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

#: 预算（第 14 节；机器小时）
BUDGET: dict[str, Any] = {
    "total_machine_hours": 64.0,
    "stage_upper_hours": {"r0": 8.0, "r1": 28.0, "r2": 14.0, "r3": 14.0},
    "r0_capture_max_hours": 2.0,
    "r1_download_max_hours": 22.0,
    "r2_compute_max_hours": 6.0,
    "r3_replay_max_hours": 6.0,
    "new_procurement_yuan": 0,
    "procurement_request_yuan": 2000,
}

#: 阶段与状态枚举（第 11.5 节）
STAGES: tuple[str, ...] = ("r0", "r1", "r2", "r3")
RESEARCH_STATUSES: tuple[str, ...] = (
    "access_not_ready",
    "data_not_ready",
    "engineering_invalid",
    "economic_target_not_met",
    "continue_limited",
    "economic_candidate",
)
FORMAL_ADMISSION = "not_evaluated"

#: 证据输入（R0 核验对象）
INPUTS: tuple[dict[str, Any], ...] = (
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
)

#: 选择史（第 2.3 / 16 节；R0 原样登记）
SELECTION_HISTORY: tuple[dict[str, str], ...] = (
    {
        "direction": "方案1 日线反转/超跌",
        "relation": "同期另一候选，已由用户另行处置",
        "difference": "本袖新增盘中路径与量额分布输入，不复用其结论、不重判",
    },
    {
        "direction": "AD-ML-20-v1 分析师预期扩散",
        "relation": "同期并行候选（分析师研报输入、20 日标签）",
        "difference": "本袖输入为分钟量价，标签 10 日，不读取其结果调参",
    },
    {
        "direction": "执行成本预筛（2026-11-15）",
        "relation": "只评估冻结订单与成本口径",
        "difference": "未验证分钟 ML 信号；本包不复用其因子结论",
    },
    {
        "direction": "日线反转/资金流/赎回压力等已终结家族",
        "relation": "已按契约终结，禁止新臂",
        "difference": "本袖是新输入新标签，不重开旧家族",
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
        "minute": MINUTE,
        "units": UNITS,
        "session_mapping": SESSION_MAPPING,
        "daily_features": list(DAILY_FEATURES),
        "minute_features": list(MINUTE_FEATURES),
        "feature_count": len(FEATURES),
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
        "metrics": METRICS,
    }


def spec_sha256() -> str:
    """冻结规格的规范化 JSON SHA256（数值身份）。"""
    payload = json.dumps(
        numeric_identity(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def machine_spec_document(protocol_sha: str | None = None) -> dict[str, Any]:
    """机器规格文档（字段全部为真实值，无占位）。"""
    return {
        "candidate_id": CANDIDATE_ID,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_path": PROTOCOL_RELATIVE,
        "protocol_sha256": protocol_sha or protocol_sha256(),
        "spec_sha256": spec_sha256(),
        "phase": "r0",
        "purpose": "economic_prescreen",
        "intended_admission_type": "enhancement",
        "formal_admission": FORMAL_ADMISSION,
        "capital_scenarios": TARGETS["capital_scenarios"],
        "initial_b_fraction": TARGETS["initial_b_fraction"],
        "minute_freq": MINUTE["freq"],
        "feature_count": len(FEATURES),
        "features": list(FEATURES),
        "daily_control_features": list(DAILY_FEATURES),
        "minute_new_features": list(MINUTE_FEATURES),
        "session_mapping": SESSION_MAPPING,
        "horizon": LABEL["horizon"],
        "label": LABEL["formula"],
        "portfolio": PORTFOLIO,
        "model": MODEL,
        "splits": SPLITS,
        "costs": COSTS,
        "min_ticket_value_yuan": min_ticket_value(),
        "targets": TARGETS,
        "window": WINDOW,
        "bootstrap": BOOTSTRAP,
        "stress": STRESS,
        "budget_limits": BUDGET,
        "authorization_scope": "existing_entitlements_only_no_new_procurement",
        "baseline_sha": A_BASELINE["expected_manifest_sha256"],
        "selection_history": [dict(item) for item in SELECTION_HISTORY],
        "inputs": [dict(item) for item in INPUTS],
        "not_virgin_market_holdout": True,
        "production_changed": False,
        "data_purchase_yuan": int(BUDGET["new_procurement_yuan"]),
    }


def n_jobs(logical_cpus: int) -> int:
    """线程数 = min(8, 可用逻辑 CPU)。"""
    return max(1, min(int(MODEL["max_n_jobs"]), int(logical_cpus)))
