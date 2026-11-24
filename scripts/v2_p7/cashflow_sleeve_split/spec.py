# -*- coding: utf-8 -*-
"""CF-SPLIT-20-v1 冻结规格（协议 §3~§8 的机器化唯一口径）。

本模块只承载**规格常量与派生哈希**，不读数据、不做 IO 校验（校验归
:mod:`identity` / :mod:`feature_routes` / :mod:`fold_clock` / :mod:`readiness`）。

三条硬纪律
==========
1. **任何会改变数值结果的参数只能写在这里**；改变即新 campaign，不得就地覆盖。
2. ``numeric_identity()`` 只含数值口径（不含墙钟、路径、运行元数据），其 SHA256
   即机器预登记的 ``spec_sha256``。
3. ``formal_admission`` 恒为 ``not_evaluated``、``alpha_consumed`` 恒为 ``False``：
   本包是历史经济预筛（证据等级 H），不消费正式准入 α、不写生产假设台账。

协议未写明的取值一律标 ``source`` 与 ``status``（``documented`` / ``derived`` /
``reference_only``），禁止把推断值伪装成已冻结值。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from scripts.v2_p7.common import REPO_ROOT

# ── 身份 ───────────────────────────────────────────────────────────────
CANDIDATE_ID = "CF-SPLIT-20-v1"
PROTOCOL_VERSION = "CF-SPLIT-20-v1"
PROTOCOL_RELATIVE = "docs/experiments/p7_cashflow_sleeve_split_protocol_20261120.md"
DEFAULT_CAMPAIGN = "cf_split_v1"
#: 正式研究产物根（协议 §2.2；写域见 artifacts.write_roots）
ARTIFACT_ROOT_RELATIVE = "data/reports/p7_cashflow_split_20261120"

#: 结论等级与本轮授权（协议 §1 末段、§8）
EVIDENCE_GRADE = "H"
PURPOSE = "historical_economic_prescreen"
FORMAL_ADMISSION = "not_evaluated"
ALPHA_CONSUMED = False
DATA_PURCHASE_YUAN = 0
PRODUCTION_CHANGED = False

# ── 现金流组 G（协议 §3，顺序即冻结顺序）────────────────────────────────
CASHFLOW_GROUP: Tuple[str, ...] = (
    "zscore_ocf_to_revenue",
    "zscore_ocf_to_revenue_sz",
    "zscore_ocf_to_profit",
    "zscore_ocf_to_profit_sz",
    "zscore_fcf_yield",
    "zscore_fcf_yield_sz",
    "zscore_capex_to_ocf",
    "zscore_capex_to_ocf_sz",
    "cashflow_freshness_days",
    "zscore_cf_sales",
    "zscore_cf_sales_sz",
)

#: 组内「经济比率」子集（协议 §5.1：仅 freshness / 规模中性副本不能凭空新增股票）
#: 判定「B 该股有真实来源」时至少要有一列命中本集合。
CASHFLOW_ECONOMIC_COLUMNS: Tuple[str, ...] = (
    "zscore_ocf_to_revenue",
    "zscore_ocf_to_profit",
    "zscore_fcf_yield",
    "zscore_capex_to_ocf",
    "zscore_cf_sales",
)

#: 规模中性副本（``_sz``）与 freshness：不构成独立来源证明。
CASHFLOW_SIZE_NEUTRAL_COLUMNS: Tuple[str, ...] = tuple(
    f"{name}_sz" for name in CASHFLOW_ECONOMIC_COLUMNS
)
CASHFLOW_FRESHNESS_COLUMN = "cashflow_freshness_days"

#: schema 哨兵（值 = 3；来源 ``v2/common/model_schema.py``）
CASHFLOW_SENTINEL_COLUMN = "cashflow_quality_schema_v2"
CASHFLOW_SENTINEL_VALUE = 3

#: 禁止进入任何一臂矩阵的列（标签 / 目标 / 哨兵 / 已删除别名）
FORBIDDEN_MATRIX_COLUMNS: Tuple[str, ...] = (
    "y_ret_5",
    "y_ret_10",
    "y_ret_20",
    "y_ret_20_raw",
    "neu_y_ret_20",
    CASHFLOW_SENTINEL_COLUMN,
    "zscore_cf_nm",
    "zscore_cf_nm_sz",
)

#: 训练帧六标签列（data_adapter LABEL_VALUE_COLUMNS 同名单；PIT 叠加覆盖目标）
LABEL_VALUE_COLUMNS: Tuple[str, ...] = (
    "y_ret_5",
    "y_ret_10",
    "y_ret_20",
    "neu_y_ret_5",
    "neu_y_ret_10",
    "neu_y_ret_20",
)

# ── 折与模型（协议 §3 末段、§4）───────────────────────────────────────
FOLD_VERSIONS: Tuple[int, ...] = tuple(range(24008, 24022))  # 14 折
DEPLOY_VERSION = 24022

#: 协议 §3 冻结的逐折 F 列数（v24008..v24021）+ 部署 v24022，逐值断言
EXPECTED_FEATURE_COUNTS: Mapping[int, int] = {
    24008: 140,
    24009: 140,
    24010: 141,
    24011: 149,
    24012: 149,
    24013: 148,
    24014: 148,
    24015: 148,
    24016: 148,
    24017: 148,
    24018: 148,
    24019: 147,
    24020: 154,
    24021: 154,
    24022: 154,
}

MODEL_ROOT_RELATIVE = "data/models/stock_selection"

# ── 窗口与时钟（协议 §4）──────────────────────────────────────────────
WINDOW: Dict[str, Any] = {
    "cash_anchor": "20190329",
    "start": "20190401",
    "end": "20261029",
    #: 协议 §4 冻结值：1,838 个收益区间 / 1,839 个净值点（含锚）
    "return_intervals": 1838,
    "nav_points": 1839,
    "seam": ["20260106", "20260202"],
    "seam_days": 20,
    "deploy": ["20260203", "20261029"],
    "deploy_days": 176,
    "segments": {
        "full": ["20190329", "20261029"],
        "s2019_2021": ["20190401", "20211231"],
        "s2022_2024": ["20220104", "20241231"],
        "s2025_end": ["20250102", "20261029"],
        "post_seam": ["20260203", "20261029"],
    },
    #: run-a 结构重建的预热下限（价格 sizing 至少向前 60 条逐股记录）
    "price_warmup_records": 60,
    "feature_warmup_trading_days": 120,
}

#: 折内切分口径（协议 §5.2-2；``val_ratio`` 与生产 value 袖一致 = 0.2）
#:
#: 整改（复审第 1 项）：es/校准/末尾隔离段一律取**原训练日志实际记录值**
#: （``ml_train_runs.csv``），不再用日历比例近似；比例规则只留作参照核对。
#: 标签成熟日按 ``T+21`` 计算（T+1 收盘买入、T+1+20=T+21 开盘卖出）。
SPLIT_CLOCK: Dict[str, Any] = {
    "val_ratio": 0.2,
    "label_horizon": 20,
    #: 标签成熟偏移 = 1（T+1 买入）+ horizon（T+21 开盘卖出）
    "label_maturity_offset": 21,
    #: 冻结规则的训练窗尾扣参数（仅用于参照重算，不用于产出时钟）
    "label_isolation_days": 20,
    "label_name": "neu_y_ret_20",
    "raw_label_name": "y_ret_20",
    "label_definition": "T+1 收盘买入 / T+21 开盘卖出的复权价格收益（industry 中性后截面 zscore）",
    "label_implementation": "src/lazybull/features/labels.py::compute_forward_returns",
    "segment_source": "data/models/stock_selection/ml_train_runs.csv",
    "segment_date_status": "logged",
    "derived_reference_source": (
        "src/lazybull/v2/train/walk_forward/training_core.py::resolve_explicit_split_dates"
    ),
    "derived_reference_status": "reference_only",
    "derived_reference_note": (
        "比例规则重算的 val 起日在 v24008/v24010/v24016 比原日志早 1 个交易日；"
        "时钟以日志为准，比例结果只作参照并逐版登记差值"
    ),
    #: 部署模型 v24022 早停净化（r0_003 复核第 2 项）：部署早停段截至 2025-08-28、
    #: 校准始于 2025-08-29 ⇒ 早停最后可用信号日 = 2025-07-30（其 T+21 = 08-28）；
    #: 07-31 的标签 08-29 成熟，已进校准段，须剔除（实测 21 个早停信号日）。
    "deploy_es_last_usable_expected": "20250730",
    "deploy_es_purged_expected": 21,
}

#: 原训练日志（阶段边界的唯一权威来源）
TRAINING_LOG: Dict[str, Any] = {
    "relative": "data/models/stock_selection/ml_train_runs.csv",
    "run_hint": "wf_20260907_111212_fba16ce5",
    "version_column": "model_version",
    "timestamp_column": "timestamp",
    "columns": {
        "outer_train_start": "start_date",
        "outer_train_end": "end_date",
        "es_start": "val_start_date",
        "es_end": "val_end_date",
        "calib_start": "val_calib_start_date",
        "calib_end": "val_calib_end_date",
        "test_start": "test_start_date",
        "test_end": "test_end_date",
        "split_index": "split_index",
        "best_iteration": "best_iteration",
    },
    "note": "同一 model_version 在 CSV 中恰一行；物理行号 = CSV 行序（含表头后 1-based）",
}

#: 阶段命名（``fold_clock.csv`` 列名前缀）
SEGMENTS: Tuple[str, ...] = ("es", "calib", "tail_isolation", "test")

#: 折窗控制文件（逐折 test 域与阶段日历；与日志逐值对账）
ANCHOR_SUMMARY_RELATIVE = "data/runs/p2a_b0replay_20261008_144307_d79f3f/summary.parquet"
T25_PRECHECK_RELATIVE = "data/paper_shadow/p7_testbed/p7_t05_20261111_r6/precheck"

#: 行域事实（协议 §5.1）
ROW_DOMAIN: Dict[str, Any] = {
    "label_filtered_until": "20260702",
    "structural_from": "20260703",
    "cs_infer_partitions": 80,
    "cs_infer_insufficient_for_window": True,
    "historical_kernel_rejection_dates": ["20261019"],
}

#: 行业 PIT 注入（r0_003 复核第 1 项：筛股/中性化/标签必须用**当日**行业区间）
#:
#: 生产通路默认行业快照（``data/raw/shenwan_industry.parquet``）是**当前**申万
#: 分类（无日期维度），用在历史日会把未来行业改写写入特征与标签中性化分组
#: （实例：600783.SH 在 2019-04-01 属 SW2014 非银金融 490000/801790.SI，快照写
#: SW2021 机械设备 801890.SI）。研究构建入口必须从 ``sw_industry_pit`` 区间命中
#: 构造与生产快照同 schema 的当日帧并显式注入（``_run_capture`` 的
#: ``shenwan_industry`` 参数），资格过滤、特征行业中性化与 ``neu_y_ret_*`` 标签
#: 分组共用同一输入。
INDUSTRY_PIT_INJECTION: Dict[str, Any] = {
    "pit_relative": "data/clean/sw_industry_pit.parquet",
    "code_table_relative": "data/clean/sw_industry_code_table.parquet",
    "interval_rule": "in_date ≤ T < out_date（out_date 空 = 仍有效）",
    "snapshot_schema": [
        "ts_code",
        "sw_l1_code",
        "sw_l1",
        "sw_l2_code",
        "sw_l2",
        "sw_l3_code",
        "sw_l3",
        "in_date",
    ],
    "code_mapping": (
        "l1_code/l2_code/ind_code 旧 6 位码 → code_table(src=sw_version, level) 反查 index_code"
    ),
    "unmapped_handling": "反查失败的行保留 NaN 并逐只登记（资格过滤按 industry_known 剔除）",
    "identity_probe": {
        "day": "20190401",
        "ts_code": "600783.SH",
        "expected_l1_code": "490000",
        "expected_l1_name": "非银金融",
        "expected_l1_index": "801790.SI",
    },
}

#: 单日端到端验证日（r0_004；评审：「先只验证 2019-04-01，并让探针消费这次真实输出」）
SINGLE_DAY_VALIDATION = "20190401"

# ── 资金与账户（协议 §6）──────────────────────────────────────────────
ACCOUNTS: Dict[str, Any] = {
    "capitals": [1_000_000, 2_000_000],
    #: 腿 = 初始资本的固定比例，只分配一次，之后不再恢复固定权重
    "legs": {"A100": 1.00, "A80": 0.80, "B20": 0.20},
    "independent_cash": True,
    "no_borrow": True,
    "no_cross_sleeve_netting": True,
    "combination_rule": "各腿同一收盘日总权益金额相加后除以初始 K（禁止归一曲线后相加）",
}

#: 大腿规格（协议 §6「大腿规格」；``current_config=False``）
BIG_LEG: Dict[str, Any] = {
    "top_n": 20,
    "holding_days": 20,
    "rebalance_freq": 20,
    "stagger_tranches": 2,
    "position_sizing": "half_kelly",
    "kelly_vol_window": 60,
    "kelly_max_leverage": 0.2,
    "max_weight_per_stock": 0.15,
    "max_per_industry": None,
    "min_buy_value_ratio": 0.2,
    "signal_day": "T 收盘产生信号，T+1 收盘买入",
    "sell_path": "原生开盘卖出（无次日顺延重建）",
}

#: 小腿规格（协议 §6「小腿规格」；研究者事前设计，审点一确认后锁死）
SMALL_LEG: Dict[str, Any] = {
    "top_n": 6,
    "stagger_tranches": 2,
    "per_tranche": 3,
    "weighting": "批内等权",
    "max_weight_per_stock": "小腿当时权益 / 6",
    "max_per_industry": None,
    "holding_days": 20,
    "rebalance_freq": 20,
    "same_clock_as_big_leg": True,
    "cross_leg_overlap_allowed": True,
    "signal_object": "显式注入 Top6 信号对象（禁止给 Top20 传 6 行）",
}

#: 账户角色的执行配置（复审第 4 项：主腿 Top20/15% 上限，六槽只属小腿）
#: ``max_weight_per_stock`` 是**新买**单股上限（占该腿当时总权益），不是槽位金额。
ACCOUNT_ROLES: Dict[str, Dict[str, Any]] = {
    "F100": {"leg": "A100", "capital_share": 1.00, "signal": "big", "top_n": 20},
    "R100": {"leg": "A100", "capital_share": 1.00, "signal": "big", "top_n": 20},
    "F80": {"leg": "A80", "capital_share": 0.80, "signal": "big", "top_n": 20},
    "R80": {"leg": "A80", "capital_share": 0.80, "signal": "big", "top_n": 20},
    "B20": {"leg": "B20", "capital_share": 0.20, "signal": "small", "top_n": 6},
    "F20": {"leg": "B20", "capital_share": 0.20, "signal": "small", "top_n": 6},
}

#: 两只信号的执行参数（主腿 vs 小腿）
SIGNAL_EXECUTION: Dict[str, Dict[str, Any]] = {
    "big": {
        "top_n": 20,
        "position_sizing": "half_kelly",
        "kelly_vol_window": 60,
        "max_weight_per_stock": 0.15,
        "stagger_tranches": 2,
        "slots_per_tranche": None,
        "weighting": "half-Kelly（单股新买上限 = 15% × 该腿当时权益）",
    },
    "small": {
        "top_n": 6,
        "position_sizing": "equal_weight",
        "kelly_vol_window": None,
        "max_weight_per_stock": 1.0 / 6.0,
        "stagger_tranches": 2,
        "slots_per_tranche": 3,
        "weighting": "批内等权（单股上限 = 当时权益 / 6）",
    },
}

#: 成本（协议 §6「净成本与账务」；CostModel(0.0001954, 5.0, 0.0005, 0.0005)）
COSTS: Dict[str, Any] = {
    "commission_rate": 0.0001954,
    "min_commission": 5.0,
    "stamp_tax": 0.0005,
    "slippage_per_side": 0.0005,
}

#: 小额腿单笔买入下限 V = max(25000, min_commission / commission_rate)
MIN_TICKET_FLOOR_YUAN = 25000.0


def min_ticket_value(costs: Mapping[str, Any] | None = None) -> float:
    """小腿单笔新增买入的成交额下限（费用前，元）。"""
    table = costs or COSTS
    return float(
        max(
            MIN_TICKET_FLOOR_YUAN,
            float(table["min_commission"]) / float(table["commission_rate"]),
        )
    )


# ── 判据（协议 §1、§7）────────────────────────────────────────────────
TARGETS: Dict[str, Any] = {
    #: R 的单袖研究目标（≥26%，含等号）
    "r_cagr_min": 0.26,
    #: R 的回撤损失幅度必须**严格**小于同口径完整 A（F）
    "r_mdd_loss_strictly_below_f": True,
    #: 唯一候选组合：每档同时满足净 CAGR >25% 且回撤损失 <20%（两档合取）
    "portfolio_cagr_min": 0.25,
    "portfolio_cagr_strict": True,
    "portfolio_mdd_loss_max": 0.20,
    "portfolio_mdd_loss_strict": True,
    "portfolio_capital_scenarios": [1_000_000, 2_000_000],
    "portfolio_conjunction": True,
    "comparison_precision": "未四舍五入的原始浮点",
}

#: 链式指标口径（协议 §7.1；唯一实现 = v2/common/chain_metrics.py）
CHAIN_METRICS: Dict[str, Any] = {
    "trading_days_per_year": 252,
    "risk_free_rate": 0.03,
    "include_anchor": True,
    "split_index": None,
    "mdd_loss_convention": "mdd_loss = -max_drawdown（底层 max_drawdown 为负）",
    "implementation": "src/lazybull/v2/common/chain_metrics.py::calculate_chain_metrics",
}

#: 描述性重抽样（协议 §7.6）：主窗完整配对日收益、同一组索引
BOOTSTRAP: Dict[str, Any] = {
    "method": "circular_block",
    "block": 20,
    "repetitions": 2000,
    "seed": 42,
    "pairs": ["R-F", "portfolio-F", "portfolio-cash"],
    "usage": "只描述路径不确定性，不把重抽达标占比称为未来概率",
}

#: 固定压力（协议 §7.6）
STRESS: Dict[str, Any] = {
    "slippage_per_side_doubled": 0.0010,
    "replay_arms": ["F", "R", "cash", "portfolio"],
    "unresolved_corporate_default": "unbounded（给不出可解释范围时）",
}

# ── 旧 A 锚（协议 §4）─────────────────────────────────────────────────
A_BASELINE: Dict[str, Any] = {
    "relative": "data/reports/p7_operating_monthly_economic_20261116/run_r3",
    "expected_manifest_sha256": (
        "541933bbbf3ecc2bc96a5bb572bdc8fb16cedde6b369c302b3501b4c19d5fd81"
    ),
    #: 协议 §4 已发表参照读数（只作查错，验收用冻结底账）
    "published_reference": {
        "1000000": {"cagr": 0.163743, "mdd_loss": 0.317443},
        "2000000": {"cagr": 0.164341, "mdd_loss": 0.317787},
    },
    "harness": {
        "1000000": {
            "a100_daily": "1000000/A100_daily.parquet",
            "a80_daily": "1000000/A80/daily.json",
            "a80_cash_daily": "1000000/A80_B_cash_daily.parquet",
        },
        "2000000": {
            "a100_daily": "2000000/A100_daily.parquet",
            "a80_daily": "2000000/A80/daily.json",
            "a80_cash_daily": "2000000/A80_B_cash_daily.parquet",
        },
    },
    "a80_account_detail": [
        "A80/attribution.json",
        "A80/ending.json",
        "A80/validation.json",
        "A80/trades.json",
        "A80/holdings_snapshot.json",
        "A80/corporate.json",
    ],
}

# ── 探针日（协议 §8 审点一）───────────────────────────────────────────
PROBE_DATES: Tuple[Dict[str, str], ...] = (
    {"date": "20190401", "segment": "window_start", "note": "账户首日，属折 0 测试域"},
    {"date": "20220104", "segment": "s2022_2024", "note": "2022 首个交易日"},
    {"date": "20260106", "segment": "seam", "note": "接缝首日，沿用末折模型"},
    {"date": "20260203", "segment": "deploy", "note": "部署段首日，固定各臂部署模型"},
    {"date": "20260702", "segment": "tail_label_filtered", "note": "label_filtered 行域末日"},
    {"date": "20260703", "segment": "tail_structural", "note": "structural 行域首日"},
    {"date": "20261019", "segment": "tail_rejection", "note": "原内核拒绝日（协议 §5.1）"},
)

#: 资格探针的**唯一**键与分步排除（复审第 2 项）
#:
#: 母表 = 当日 ``clean/daily`` 全部行；键 = ``(trade_date, ts_code)``。每一步都记录
#: 排除数与「资料未知」数，缺列一律记 ``source_not_ready``（禁止静默跳过该项）。
PROBE_SELECTION: Dict[str, Any] = {
    "mother": "data/clean/daily/<date>.parquet（当日全部行，不做任何预筛）",
    "key": ["trade_date", "ts_code"],
    "market_whitelist": ["主板"],
    "steps": [
        {
            "id": "has_feature_row",
            "rule": "特征分区存在同键行（无行 = 当日无特征）",
            "needs": ["features partition"],
        },
        {
            "id": "main_board",
            "rule": "stock_basic.market ∈ domain_market_whitelist('main')",
            "needs": ["data/clean/stock_basic.parquet::market"],
        },
        {
            "id": "listed_days",
            "rule": "clean/daily.list_days ≥ 365（上市年限，信号日可知）",
            "needs": ["data/clean/daily::list_days"],
        },
        {
            "id": "not_st",
            "rule": "clean/daily.is_st = 0 且 name 不含 ST/退（历史 ST 两部分都查）",
            "needs": ["data/clean/daily::is_st", "data/clean/stock_basic.parquet::name"],
        },
        {
            "id": "not_suspended",
            "rule": "clean/daily.is_suspended = 0 且 tradable = 1",
            "needs": ["data/clean/daily::is_suspended", "data/clean/daily::tradable"],
        },
        {
            "id": "industry_known",
            "rule": "sw_industry_pit 区间命中（in_date ≤ T < out_date）且 l1_code 非空",
            "needs": ["data/clean/sw_industry_pit.parquet"],
        },
        {
            "id": "not_financial",
            "rule": "sw_l1_code ∉ {801780 银行, 801790 非银金融}",
            "needs": ["sw_industry_pit::l1_code"],
        },
        {
            "id": "amount_floor",
            "rule": "特征 amount_ma20 ≥ 50,000（千元，MLSignal 口径）",
            "needs": ["features partition::amount_ma20"],
        },
        {
            "id": "market_cap_band",
            "rule": "特征 total_mv ∈ [500,000, 15,000,000]（万元）",
            "needs": ["features partition::total_mv"],
        },
    ],
    "financial_l1_codes": ["801780", "801790"],
    "financial_pit_codes": ["480000", "490000"],
    "financial_code_note": (
        "金融剔除必须用 T 日行业区间（sw_industry_pit）的**旧 6 位码** 480000/490000，"
        "与中性化、相关标签共用同一映射；特征里的 sw_l1_code（801780.SI 等）是旧截面，"
        "只作冲突对照。MLSignal 的裸码常量在带后缀数据上命中 0（已登记发现）"
    ),
    "amount_floor_kilo_yuan": 50000.0,
    "market_cap_band_wan": [500000.0, 15000000.0],
    "min_list_days": 365,
    "unknown_status": "source_not_ready",
}

#: 三臂「至少一项当时可知的经济输入」取证白名单（复审第 3 项）
#:
#: 标签/目标/哨兵一律不在任何白名单内；F 与 R 各自独立取证，B 必须命中
#: **经济比率或其有原值血缘的销售现金代理**（标准化 zscore 单独出现不算来源证明）。
#:
#: r0_003 复核修正（Codex 2026-11-20）：经营现金流/自由现金流**绝对额**
#: （``ocf``/``fcf``）单独存在不构成 B 的来源——它们不是比率，无现金流「质量」
#: 语义，且 B 的模型列是比率族；只有绝对额、没有可用比率的行单独计数
#: （``B_absolute_only``）并在夹具核对，不计入 B 的来源宽度。
SOURCE_EVIDENCE: Dict[str, Any] = {
    "F": {"columns": "F_k（该折有序白名单）", "exclude": list(FORBIDDEN_MATRIX_COLUMNS)},
    "R": {"columns": "R_k = F_k − G（该折有序白名单）", "exclude": list(FORBIDDEN_MATRIX_COLUMNS)},
    "B": {
        "columns": "G（11 列）",
        #: B 来源白名单 = 经济比率 + cf_sales（协议 §3：cf_sales 缺失时由
        #: ocf_to_revenue 回填，属「有原值血缘的销售现金代理」）
        "raw_columns": [
            "ocf_to_revenue",
            "ocf_to_profit",
            "fcf_yield",
            "capex_to_ocf",
            "cf_sales",
        ],
        #: 绝对额参照列：单独存在不算来源，只作 `B_absolute_only` 计数
        "absolute_amount_columns": ["ocf", "fcf"],
        "announcement_proxy": CASHFLOW_FRESHNESS_COLUMN,
        "note": (
            "经济比率/销售现金代理非空（有限值）= 有真实财报来源；freshness 非空 = "
            "有实际公告日（或回退代理）；仅有 zscore（标准化结果）、_sz 副本或 "
            "ocf/fcf 绝对额不算来源证明"
        ),
    },
    "label_columns_excluded": list(FORBIDDEN_MATRIX_COLUMNS),
}

# ── 数据源清单（协议 §9「预登记与输入」）──────────────────────────────
#: ``kind``: file / partition_daily / partition_month / partition_quarter / file_tree
SOURCES: Tuple[Dict[str, Any], ...] = (
    {
        "name": "protocol",
        "relative": PROTOCOL_RELATIVE,
        "kind": "file",
        "role": "冻结协议原文（字节 SHA 入 spec）",
    },
    {
        "name": "model_features",
        "relative": MODEL_ROOT_RELATIVE,
        "kind": "model_artifacts",
        "role": "逐折/部署模型特征清单（F 列唯一来源）",
    },
    {
        "name": "a_baseline",
        "relative": A_BASELINE["relative"],
        "kind": "file_tree",
        "role": "冻结旧 A 账户与 manifest",
    },
    {"name": "clean_trade_cal", "relative": "data/clean/trade_cal.parquet", "kind": "file"},
    {"name": "clean_stock_basic", "relative": "data/clean/stock_basic.parquet", "kind": "file"},
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
    {"name": "clean_daily", "relative": "data/clean/daily", "kind": "partition_daily"},
    {"name": "clean_daily_basic", "relative": "data/clean/daily_basic", "kind": "partition_daily"},
    {"name": "features_cs_train", "relative": "data/features/cs_train", "kind": "partition_daily"},
    {"name": "features_cs_infer", "relative": "data/features/cs_infer", "kind": "partition_daily"},
    {"name": "panel_hot", "relative": "data/features/panel", "kind": "partition_daily"},
    {"name": "panel_archive", "relative": "data/features/panel_archive", "kind": "partition_month"},
    {"name": "labels_y_ret_20", "relative": "data/labels/y_ret_20", "kind": "partition_daily"},
    {"name": "raw_cashflow", "relative": "data/raw/cashflow", "kind": "partition_quarter"},
    {"name": "raw_dividend", "relative": "data/raw/dividend", "kind": "partition_year"},
    {"name": "raw_stock_st", "relative": "data/raw/stock_st", "kind": "partition_daily"},
    {"name": "raw_suspend", "relative": "data/raw/suspend", "kind": "partition_daily"},
    {"name": "raw_stk_limit", "relative": "data/raw/stk_limit", "kind": "partition_daily"},
    {"name": "raw_adj_factor", "relative": "data/raw/adj_factor", "kind": "partition_daily"},
    # ── 复审第 6 项：实际读取的折窗/日志/公司行为输入必须逐个入身份 ──
    {
        "name": "training_log",
        "relative": TRAINING_LOG["relative"],
        "kind": "file",
        "role": "折阶段边界唯一权威来源（es/校准/末尾隔离/测试）",
    },
    {
        "name": "anchor_summary",
        "relative": ANCHOR_SUMMARY_RELATIVE,
        "kind": "file",
        "role": "折表 test 域与模型版本（实际读取）",
    },
    {
        "name": "t25_fold_check",
        "relative": f"{T25_PRECHECK_RELATIVE}/fold_check.json",
        "kind": "file",
        "role": "T0.5 已核验折表（逐值对账）",
    },
    {
        "name": "t25_phase_calendar",
        "relative": f"{T25_PRECHECK_RELATIVE}/phase_calendar.json",
        "kind": "file",
        "role": "接缝/部署段日历与批次数（逐值对账）",
    },
    {
        "name": "corporate_manifest",
        "relative": f"{A_BASELINE['relative']}/corporate_input/manifest.json",
        "kind": "file",
        "role": "冻结公司行为清单（版本、日期覆盖、逐文件源/出 SHA）",
    },
    {
        "name": "corporate_dividend",
        "relative": f"{A_BASELINE['relative']}/corporate_input/dividend",
        "kind": "file_tree",
        "role": "冻结公司行为修订表（实际消费）",
    },
)

# ── 选择史与已看过的信息（协议 §1 末段）──────────────────────────────
SELECTION_HISTORY: Tuple[Dict[str, str], ...] = (
    {
        "direction": "A 袖（价值红利）全列模型",
        "relation": "冻结旧 A100/A80 账户即当前生产基准",
        "difference": "本包把 11 列现金流组整体移出 A，A 只保留 F−G",
    },
    {
        "direction": "旧现金流家族 A/B（列入口）",
        "relation": "现金流质量族已按列入口路径进入生产 A，未单独裁决过家族整体",
        "difference": "本包不改因子、不筛列、不消融，只做整体拆分与独立账户",
    },
    {
        "direction": "B0 / C1 / IDX-IN300 等既有袖子",
        "relation": "均为已登记结果或已终结家族",
        "difference": "本包不重开任何旧候选，也不以本读数改写其结论",
    },
    {
        "direction": "2026-11-16 经营月报 / 涤纶 / 生猪经济预筛",
        "relation": "同窗（2019-04-01~2026-10-29）与尾段的历史结果已被看过",
        "difference": "本包登记为同窗重复观察，不声称 virgin holdout",
    },
)

# ── 阶段与授权（协议 §8）──────────────────────────────────────────────
STAGES: Tuple[str, ...] = ("prepare", "run-a", "run-b", "verify")

#: 审点一交付只授权 ``prepare``；其余批次须用户按已通过审核后另行命名授权
AUTHORIZED_STAGES: Tuple[str, ...] = ("prepare",)

STAGE_ARTIFACTS: Mapping[str, Tuple[str, ...]] = {
    "prepare": (
        "spec.json",
        "input_manifest.json",
        "feature_routes.json",
        "fold_clock.csv",
        "readiness.json",
        "pilot_costs.json",
        "tests.txt",
    ),
}

#: 机器批次预算基线（协议 §8「预算」；单机墙钟，含上下限）
#:
#: 复审第 7 项：A 批**只**含 F/R 两臂；B 的训练与经济读取列在审点二后的独立预算，
#: 历史结构重建单列（须固定分块实测，见 ``budget.structural_rebuild``）。
BUDGET_REFERENCE: Dict[str, Any] = {
    "pilot": {"low_minutes": 10, "high_minutes": 20},
    "run_a_hours": {
        "low": 4.0,
        "high": 9.5,
        "scope": "仅 F 与 R：14 折 + 部署重训、全域逐日打分、两档 F100/F80/R100/R80",
        "basis": "历史 14 折单臂 5h09m；两臂并行口径按臂线性拆分，两档账户回放分钟级",
    },
    "run_b_hours": {
        "low": 2.0,
        "high": 8.0,
        "scope": "B 大腿与六槽小腿训练、主组合与四类对照、压力与对账",
        "gate": "审点二通过后单独立项，不并入 A 批",
    },
    "verify_hours": {"low": 0.5, "high": 3.0},
    "total_hours": {"low": 6.5, "high": 20.5},
    "structural_rebuild": {
        "status": "separate_line_item",
        "note": "全窗历史结构重建不属于任何一批的既定预算，须先做固定分块（含预热）实测",
    },
    "excludes": "编码、人工等待",
    "source": "协议 §8「预算」段（历史计时：14 折约 5h09m；两档 A 账户分钟级回放）",
    "approval_required_over_minutes": 20,
}


def protocol_path(root: Path | None = None) -> Path:
    """协议原文路径。"""
    return (Path(root) if root is not None else REPO_ROOT) / PROTOCOL_RELATIVE


def protocol_sha256(root: Path | None = None) -> str:
    """协议原文的字节 SHA256（封存对象，不允许占位）。"""
    return hashlib.sha256(protocol_path(root).read_bytes()).hexdigest()


def numeric_identity() -> Dict[str, Any]:
    """会改变数值输出的全部冻结参数（不含墙钟、路径与运行元数据）。"""
    return {
        "candidate_id": CANDIDATE_ID,
        "cashflow_group": list(CASHFLOW_GROUP),
        "cashflow_group_size": len(CASHFLOW_GROUP),
        "cashflow_economic_columns": list(CASHFLOW_ECONOMIC_COLUMNS),
        "sentinel": {
            "column": CASHFLOW_SENTINEL_COLUMN,
            "value": CASHFLOW_SENTINEL_VALUE,
        },
        "fold_versions": list(FOLD_VERSIONS),
        "deploy_version": DEPLOY_VERSION,
        "expected_feature_counts": {str(k): v for k, v in EXPECTED_FEATURE_COUNTS.items()},
        "window": WINDOW,
        "split_clock": SPLIT_CLOCK,
        "segments": list(SEGMENTS),
        "training_log": TRAINING_LOG,
        "row_domain": ROW_DOMAIN,
        "accounts": ACCOUNTS,
        "account_roles": ACCOUNT_ROLES,
        "signal_execution": SIGNAL_EXECUTION,
        "big_leg": BIG_LEG,
        "small_leg": SMALL_LEG,
        "probe_selection": PROBE_SELECTION,
        "source_evidence": SOURCE_EVIDENCE,
        "industry_pit_injection": INDUSTRY_PIT_INJECTION,
        "single_day_validation": SINGLE_DAY_VALIDATION,
        "min_ticket_value_yuan": min_ticket_value(),
        "costs": COSTS,
        "targets": TARGETS,
        "chain_metrics": CHAIN_METRICS,
        "bootstrap": BOOTSTRAP,
        "stress": STRESS,
        "probe_dates": [item["date"] for item in PROBE_DATES],
    }


def spec_sha256() -> str:
    """冻结规格规范化 JSON 的 SHA256（数值身份）。"""
    payload = json.dumps(
        numeric_identity(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def machine_spec_document(protocol_sha: str | None = None) -> Dict[str, Any]:
    """审点一机器预登记文档（提交物 ``spec.json`` 的正文）。

    字段全部为真实值/显式状态，不允许 ``TBD`` 之外的占位；未知项必须写
    ``status = not_estimable`` 与原因。
    """
    return {
        "candidate_id": CANDIDATE_ID,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_path": PROTOCOL_RELATIVE,
        "protocol_sha256": protocol_sha or protocol_sha256(),
        "spec_sha256": spec_sha256(),
        "package": "scripts/v2_p7/cashflow_sleeve_split",
        "purpose": PURPOSE,
        "evidence_grade": EVIDENCE_GRADE,
        "formal_admission": FORMAL_ADMISSION,
        "alpha_consumed": ALPHA_CONSUMED,
        "data_purchase_yuan": DATA_PURCHASE_YUAN,
        "production_changed": PRODUCTION_CHANGED,
        "not_virgin_holdout": True,
        "known_look_history": {
            "same_window_seen_until": "20261029",
            "tail_window_seen": True,
            "statement": "同窗与 2026 尾段均已被既有研究看过；本包不得称前向/日历验证通过",
        },
        "research_questions": [
            "完整 A 拿走现金流质量及其直接代理后，能否提高自身净收益并降低最大回撤",
            "被拿走的因子独立训练、交易并占用真实资金后，在 A 深跌时贡献了什么",
        ],
        "targets": TARGETS,
        "accounts": ACCOUNTS,
        "account_roles": ACCOUNT_ROLES,
        "signal_execution": SIGNAL_EXECUTION,
        "big_leg": BIG_LEG,
        "small_leg": SMALL_LEG,
        "min_ticket_value_yuan": min_ticket_value(),
        "lot_size": 100,
        "costs": COSTS,
        "window": WINDOW,
        "split_clock": SPLIT_CLOCK,
        "segments": list(SEGMENTS),
        "training_log": TRAINING_LOG,
        "probe_selection": PROBE_SELECTION,
        "source_evidence": SOURCE_EVIDENCE,
        "row_domain": ROW_DOMAIN,
        "chain_metrics": CHAIN_METRICS,
        "bootstrap": BOOTSTRAP,
        "stress": STRESS,
        "feature_group": {
            "name": "现金流质量及其直接代理",
            "columns": list(CASHFLOW_GROUP),
            "economic_columns": list(CASHFLOW_ECONOMIC_COLUMNS),
            "size_neutral_copies": list(CASHFLOW_SIZE_NEUTRAL_COLUMNS),
            "freshness": CASHFLOW_FRESHNESS_COLUMN,
            "cf_sales_fallback": (
                "cf_sales 缺失时由 ocf_to_revenue 回填；本规格将 cf_sales 与其 _sz "
                "整体划给 B，故结论指「现金流质量及其直接代理」组合"
            ),
            "cf_nm_note": (
                "cf_nm 可由 ocf_to_profit 回填；冻结列集中无 zscore_cf_nm/_sz，"
                "禁止因关闭家族开关使该别名回流"
            ),
            "routing_rule": "R = [c for c in F_k if c not in G]；B = [c for c in F_k if c in G]",
            "no_selection": "不做重要性筛选、逐列消融或事后翻符号",
        },
        "a_baseline": A_BASELINE,
        "inputs": [dict(item) for item in SOURCES],
        "probe_dates": [dict(item) for item in PROBE_DATES],
        "selection_history": [dict(item) for item in SELECTION_HISTORY],
        "stages": list(STAGES),
        "authorized_stages": list(AUTHORIZED_STAGES),
        "stage_artifacts": {k: list(v) for k, v in STAGE_ARTIFACTS.items()},
        "budget_reference": BUDGET_REFERENCE,
        "explicitly_not_run": [
            "真实收益回放（run-a / run-b / verify）",
            "整窗训练（14 折 + 部署的 F/R/B 三臂重训）",
            "数据采购（新增数据费 0 元）",
            "生产接入（模型、分区、配置、纸面链路零改动）",
        ],
    }


def feature_group_payload() -> Dict[str, Any]:
    """``feature_routes.json`` 的组定义段（列集合的唯一机器来源）。"""
    return {
        "group": list(CASHFLOW_GROUP),
        "size": len(CASHFLOW_GROUP),
        "economic": list(CASHFLOW_ECONOMIC_COLUMNS),
        "size_neutral": list(CASHFLOW_SIZE_NEUTRAL_COLUMNS),
        "freshness": CASHFLOW_FRESHNESS_COLUMN,
        "sentinel": {"column": CASHFLOW_SENTINEL_COLUMN, "value": CASHFLOW_SENTINEL_VALUE},
        "forbidden_in_matrix": list(FORBIDDEN_MATRIX_COLUMNS),
        "routing": {
            "F": "F_k（原折已注册有序列表，逐字继承）",
            "R": "[c for c in F_k if c not in G]（保序，不排序重排）",
            "B": "[c for c in F_k if c in G]（保序，不排序重排）",
        },
    }
