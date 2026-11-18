"""DEF-ML-60-v1 冻结规格（唯一来源）。

本模块是协议 `docs/experiments/p7_defensive_ml_protocol_20261117.md` 的机器镜像。
改动任何会改变输出的数值都必须新 campaign；本模块只登记、不提供扫描入口。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

CANDIDATE_ID = "DEF-ML-60-v1"
PROTOCOL_VERSION = "0.1"
PROTOCOL_RELATIVE = "docs/experiments/p7_defensive_ml_protocol_20261117.md"
DEFAULT_CAMPAIGN = "defml60_v1_20261117"

#: 研究正式工件根（不得写到生产根）
ARTIFACT_ROOT_RELATIVE = "data/reports/p7_defensive_ml_20261117"

#: 冻结 19 列特征（顺序即训练列序，禁止自动增删）
FEATURES: tuple[str, ...] = (
    "def_ret20",
    "def_ret60",
    "def_ret120",
    "def_vol20",
    "def_vol60",
    "def_downvol60",
    "def_mdd120",
    "def_beta120",
    "def_downbeta120",
    "def_idiovol120",
    "def_amihud20",
    "def_amount_cv20",
    "def_log_amount20",
    "def_log_total_mv",
    "def_turnover20",
    "def_zero_return60",
    "def_ocf_to_sales_cash",
    "def_capex_to_ocf",
    "def_cashflow_age",
)

#: 单位换算（已核 clean 层 schema；转换只允许一次）
UNIT_RULES: dict[str, Any] = {
    "daily_amount": {
        "source_column": "amount",
        "source_unit": "千元",
        "to_unit": "元",
        "multiplier": 1000.0,
        "verified_on": "clean/daily",
    },
    "daily_basic_total_mv": {
        "source_column": "total_mv",
        "source_unit": "万元",
        "to_unit": "元",
        "multiplier": 10000.0,
        "verified_on": "clean/daily_basic",
    },
    "daily_basic_turnover_rate": {
        "source_column": "turnover_rate",
        "source_unit": "百分数",
        "to_unit": "比例",
        "multiplier": 0.01,
        "verified_on": "clean/daily_basic",
    },
}

#: 信号域与资格（第 4 节）
UNIVERSE: dict[str, Any] = {
    "markets": ["主板"],
    "exclude_financial": True,
    "exclude_st": True,
    "min_list_days": 252,
    "min_total_mv_yuan": 5.0e9,
    "max_total_mv_yuan": 1.5e11,
    "min_amount20_yuan": 5.0e7,
    "min_scored_per_day": 100,
    "price_window_days": 120,
    "min_valid_returns_in_window": 96,
    "min_window_valid_ratio": 0.8,
}

#: 标签（第 6 节）
LABEL: dict[str, Any] = {
    "horizon": 60,
    "entry_offset": 1,
    "exit_offset": 61,
    "entry_price": "close_adj",
    "exit_price": "open_adj",
    "formula": "open_adj(T+61)/close_adj(T+1)-1",
}

#: 模型（第 7 节）
MODEL: dict[str, Any] = {
    "objective": "reg:quantileerror",
    "quantiles": [0.10, 0.50, 0.90],
    "tree_method": "hist",
    "device": "cpu",
    "max_depth": 3,
    "n_estimators": 300,
    "learning_rate": 0.03,
    "min_child_weight": 200,
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

#: 时间切分（第 7.2 节）
SPLITS: dict[str, Any] = {
    "refresh_anchor": "20190401",
    "refresh_months": 6,
    "train_months": 72,
    "val_months": 6,
    "sample_every_trading_days": 5,
    "sample_anchor": "20100104",
    "expected_folds": 16,
    "models_per_fold": 3,
    "expected_models": 48,
    "min_train_span_months": 36,
    "min_sampled_train_days": 100,
    "min_sampled_val_days": 10,
}

#: 组合与生命周期（第 9.2 节）
PORTFOLIO: dict[str, Any] = {
    "top_n": 20,
    "holding_period": 60,
    "rebalance_freq": 60,
    "stagger_tranches": 3,
    "tranche_slots": [7, 7, 6],
    "tranche_offsets": [0, 20, 40],
    "position_sizing": "equal",
    "max_per_industry": 5,
    "max_single_stock_weight": 0.10,
    "buy_phase": "next_close",
    "sell_phase": "open",
    "buy_retry_days": 10,
    "buy_retries_per_day": 5,
    "fill_window_days": 5,
    "score_formula": "0.5*q10_fixed+0.5*q50_fixed",
}

#: 成本（第 9.3 节）
COSTS: dict[str, Any] = {
    "commission_rate": 0.0001954,
    "min_commission": 5.0,
    "stamp_tax": 0.0005,
    "slippage": 0.0005,
}

#: 经济目标（第 2 节；只决定资源建议，不是正式准入门）
TARGETS: dict[str, Any] = {
    "dd_improvement_min": 0.03,
    "cagr_delta_min": -0.01,
    "capital_scenarios": [1_000_000, 2_000_000],
    "initial_b_fraction": 0.2,
    "arms": ["A100", "A80_CASH20", "A80_DEF20", "A80_LOWVOL20"],
}

#: 窗口
WINDOW: dict[str, Any] = {
    "cash_anchor": "20190329",
    "start": "20190401",
    "end": "20261029",
    "segments": [
        ["20190401", "20211231"],
        ["20220104", "20241231"],
        ["20250102", "20261029"],
    ],
    "startup_exclusion_days": 60,
    "first_train_signal_start": "20130401",
    "price_warmup_trading_days": 252,
    "cashflow_warmup_years": 2,
}

#: 描述性重抽样（第 11.2 节）
BOOTSTRAP: dict[str, Any] = {
    "method": "circular_block",
    "block": 60,
    "repetitions": 2000,
    "seed": 20261117,
    "alternative_blocks": [20, 120],
}

#: 固定压力（第 11.3 节）
STRESS: dict[str, Any] = {
    "slippage_doubled": 0.0010,
    "rank_lag_trading_days": 1,
    "cash_equity_discount": 0.20,
    "zero_recovery_when_unresolved": True,
}

#: A 基准（第 9.1 节）
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
    {"name": "clean_daily", "relative": "data/clean/daily", "kind": "partition"},
    {"name": "clean_daily_basic", "relative": "data/clean/daily_basic", "kind": "partition"},
    {"name": "clean_stock_basic", "relative": "data/clean/stock_basic.parquet", "kind": "file"},
    {"name": "clean_trade_cal", "relative": "data/clean/trade_cal.parquet", "kind": "file"},
    {
        "name": "clean_industry_pit",
        "relative": "data/clean/sw_industry_pit.parquet",
        "kind": "file",
    },
    {"name": "raw_cashflow", "relative": "data/raw/cashflow", "kind": "partition"},
    {"name": "raw_dividend", "relative": "data/raw/dividend", "kind": "partition"},
    {"name": "raw_suspend", "relative": "data/raw/suspend", "kind": "partition"},
    {"name": "raw_stk_limit", "relative": "data/raw/stk_limit", "kind": "partition"},
    {"name": "raw_stock_st", "relative": "data/raw/stock_st", "kind": "partition"},
    {"name": "raw_stock_basic", "relative": "data/raw/stock_basic.parquet", "kind": "file"},
)

#: 阶段清单（第 12 节）
STAGES: tuple[str, ...] = ("readiness", "dataset", "model_probe", "train_predict", "replay",
                           "diagnostics", "report")


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
        "unit_rules": UNIT_RULES,
        "universe": UNIVERSE,
        "label": LABEL,
        "model": MODEL,
        "splits": SPLITS,
        "portfolio": PORTFOLIO,
        "costs": COSTS,
        "targets": {
            "dd_improvement_min": TARGETS["dd_improvement_min"],
            "cagr_delta_min": TARGETS["cagr_delta_min"],
            "capital_scenarios": TARGETS["capital_scenarios"],
            "initial_b_fraction": TARGETS["initial_b_fraction"],
            "arms": TARGETS["arms"],
        },
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
    """第 16 节机器规格文档（字段全部为真实值或显式 not_available）。"""
    return {
        "candidate_id": CANDIDATE_ID,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_path": PROTOCOL_RELATIVE,
        "protocol_sha256": protocol_sha or protocol_sha256(),
        "spec_sha256": spec_sha256(),
        "purpose": "economic_prescreen",
        "intended_admission_type": "risk_adjusted",
        "formal_admission": "not_evaluated",
        "capital_scenarios": TARGETS["capital_scenarios"],
        "initial_b_fraction": TARGETS["initial_b_fraction"],
        "horizon": LABEL["horizon"],
        "quantiles": MODEL["quantiles"],
        "score_formula": PORTFOLIO["score_formula"],
        "feature_count": len(FEATURES),
        "features": list(FEATURES),
        "candidate_pool_and_selection_history": (
            "protocol §2.2 与 §12 R0 选择史登记；正式台账未写"
        ),
        "not_virgin_market_holdout": True,
        "production_changed": False,
        "data_purchase_yuan": 0,
        "window": WINDOW,
        "targets": {
            "dd_improvement_min": TARGETS["dd_improvement_min"],
            "cagr_delta_min": TARGETS["cagr_delta_min"],
        },
        "model": MODEL,
        "splits": SPLITS,
        "portfolio": PORTFOLIO,
        "costs": COSTS,
        "bootstrap": BOOTSTRAP,
        "stress": STRESS,
        "a_baseline": A_BASELINE,
        "inputs": [dict(item) for item in INPUTS],
    }


def n_jobs(logical_cpus: int) -> int:
    """线程数 = min(8, 可用逻辑 CPU)。"""
    return max(1, min(int(MODEL["max_n_jobs"]), int(logical_cpus)))
