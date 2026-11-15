# -*- coding: utf-8 -*-
"""P7 正式 T2/T3 程序（IDX-IN300）冻结常量与 schema。"""

from __future__ import annotations

#: 单候选总 α（P=1；P-1 不消费 α）
ALPHA_TOTAL = 0.05
#: 核心格数（IDX-IN300 只有一个核心格；B0 的三源口径不适用）
CORE_CELLS = 1
#: λ 冻结值（无随机选择流：单候选确定性门）
LAMBDA_FROZEN = 1
#: 组件筛查分位（随机型组件按 α/(核心格数·λ) 缩放）
COMPONENT_QUANTILE = 1.0 - ALPHA_TOTAL / (CORE_CELLS * LAMBDA_FROZEN)

#: 重采样冻结参数
RESAMPLES = 10000
SEED_MAIN = 42
NULL_SEED = 4242
#: Q1 独立夏普：10 交易日循环块
BLOCK_Q1 = 10
TRADING_DAYS = 252
RF_ANNUAL = 0.03
#: 零假设路径数（机器预算内冻结；CP 分位按该样本量报告）
NULL_PATHS = 200
#: MDE 注入网格（δ_S：夏普位移）
MDE_GRID = (0.25, 0.5, 1.0, 2.0)
MIN_POWER = 0.8

#: 既有门槛（沿用，不新增）
CORR_UPPER = 0.3
UTIL_MEDIAN_MIN = 0.5
ES1_MIN_PP = 1.0
ES2_MAX_PP = 2.0
ES3_TOL_PP = 1.0
ES3_HARD_PP = 3.0
#: 单笔最低容量 V = max(25000, 最低佣金/佣金率)
MIN_TICKET_CNY = 25000.0

#: A/A 校准容差（模拟器 vs 实际账户；预登记冻结）
AA_TOTAL_RETURN_TOL = 0.02
AA_SHARPE_TOL = 0.05

# ── 型 II（风险调整改善）判据（F29 草案 v5；用户 2026-11-15 裁决） ──
#: 袖型枚举（预登记锁定，不得事后更改）
SLEEVE_ENHANCEMENT = "enhancement"
SLEEVE_RISK_ADJUSTED = "risk_adjusted"
SLEEVE_TYPES = (SLEEVE_ENHANCEMENT, SLEEVE_RISK_ADJUSTED)
#: ΔSharpe 配对自举：10 日循环块（与 T2 同块长，独立种子域）
DELTA_SHARPE_BLOCK = 10
#: 型 II 主门与护栏的最小样本
MIN_SHARPE_OBSERVATIONS = 252
MIN_SHARPE_BLOCKS = 20
#: 仓位资格门：完整冻结窗平均 ≥10%；市场最差 20% 交易日平均 ≥20%
POSITION_MIN = 0.10
WORST_DAY_SHARE = 0.20
WORST_DAY_POSITION_MIN = 0.20
WORST_DAY_MIN_DAYS = 5
#: 单个冻结窗内市场日收益的最小有效日数（不足 ⇒ not_estimable）
MIN_MARKET_DAYS = 25
#: 冻结窗内市场日收益的最小覆盖率（不足 ⇒ not_estimable）
MARKET_COVERAGE_MIN = 0.95
#: 非危机期净收益的最小有效日数
MIN_NONCRISIS_DAYS = 60

SCHEMA = "p7_f2_formal_t2t3/v1"
PREREG_SCHEMA = "p7_f2_prereg/v1"

VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"
VERDICT_FROZEN = "inconclusive_frozen"
