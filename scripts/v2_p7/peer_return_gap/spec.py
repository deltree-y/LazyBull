# -*- coding: utf-8 -*-
"""P7 候选一 ``PRG-M1-v1`` 冻结规格（协议 §1—§6 的唯一参数来源）。

任何数值改动都会改变研究身份：本模块的 SHA256 进入每份 stage manifest，
因此禁止在运行后按结果调整任何阈值、窗口或排序口径。
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

PROTOCOL_REL = "docs/experiments/p7_peer_return_gap_protocol_20261119.md"
CAMPAIGN = "prg_m1_v1"
STUDY = "p7_peer_return_gap"
REPORT_DATE = "20261119"
OUTPUT_REL = f"data/reports/{STUDY}_{REPORT_DATE}"
TEMP_REL = "temp"
LOG_REL = "logs"

# §1 研究窗与本金
MAIN_START = "20190401"
MAIN_END = "20261029"
ANCHOR_DATE = "20190329"
ANCHOR_PREV = "20190328"
CAPITALS = (1_000_000, 2_000_000)
B_SHARE = 0.20
TARGET_DCAGR = 0.005
TARGET_DMDD = 0.010
SEGMENTS = (
    ("full", "20190401", "20261029"),
    ("seg_2019_2021", "20190401", "20211231"),
    ("seg_2022_2024", "20220101", "20241231"),
    ("seg_2025_2026", "20250101", "20261029"),
)

# §2 基本域
MARKET_MAIN = "主板"
MIN_LISTED_TRADING_DAYS = 252
MV_MIN_YUAN = 5.0e9
MV_MAX_YUAN = 1.5e11
AMOUNT_WINDOW = 20
AMOUNT_MIN_YUAN = 5.0e7
AMOUNT_MIN_DAYS = 16
EXCLUDED_L1 = ("银行", "非银金融")
BREADTH_MIN = 100
BREADTH_STOP_MONTH_RATIO = 0.05

# §3 建图
CORR_WINDOW = 252
CORR_MIN_PAIRS = 180
TOP_PEERS = 10
PEER_MIN_VALID = 8
BETA_MIN_PAIRS = 180
VARIANCE_FLOOR = 1e-12
ROUND_DECIMALS = 12

# §4 月分数与匹配
MONTH_MIN_COVERAGE = 0.80
MATCH_RANK_TOL = 0.15

# §5 账户
SLOTS = 6
SECTOR_CAP = 2
SLOT_WEIGHT = 1.0 / SLOTS
COMMISSION_RATE = 0.0001954
MIN_COMMISSION = 5.0
STAMP_TAX = 0.0005
SLIPPAGE = 0.0005
MIN_NOTIONAL = max(25000.0, 5.0 / COMMISSION_RATE)
LOT_SIZE = 100
BUDGET_SLIPPAGE_DOUBLE = 0.0010

# §6 冻结基准
BASELINE_REL = "data/reports/p7_operating_monthly_economic_20261116/run_r3"
BASELINE_MANIFEST_SHA = (
    "541933bbbf3ecc2bc96a5bb572bdc8fb16cedde6b369c302b3501b4c19d5fd81"
)
BASELINE_DIR = REPO_ROOT / "data/reports/p7_operating_monthly_economic_20261116/run_r3"
DIVIDEND_DIR = BASELINE_DIR / "corporate_input/dividend"
BASELINE_CONSUMED = (
    "{cap}/A100_daily.parquet",
    "{cap}/A80/daily.json",
    "{cap}/A80_B_cash_daily.parquet",
)
BASELINE_DISPLAY = {
    1_000_000: {"a100_cagr": 0.1637, "a100_mdd": 0.3174, "cash_cagr": 0.1407, "cash_mdd": 0.2896},
    2_000_000: {"a100_cagr": 0.1643, "a100_mdd": 0.3178, "cash_cagr": 0.1414, "cash_mdd": 0.2898},
}

# §8 压力
STRESS_SLIPPAGE = 0.0010
STRESS_DIVIDEND_HAIRCUT = 0.20
CORR_BOOTSTRAP = {"resamples": 10000, "seed": 42, "upper": 0.975}
CRASH_YEARS = ("2015", "2018", "2022")

STAGES = ("R0", "R1", "R2", "R3", "R4", "R5", "R6")
REQUIRED_UPSTREAM = {
    "R0": (),
    "R1": ("R0",),
    "R2": ("R1",),
    "R3": ("R0", "R2"),
    "R4": ("R3", "R2"),
    "R5": ("R4", "R2"),
    "R6": ("R4", "R5"),
}
