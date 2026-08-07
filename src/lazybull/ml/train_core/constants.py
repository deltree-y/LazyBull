# -*- coding: utf-8 -*-
"""train_core 特征列清单与 freshness 策略常量。"""

FUNDAMENTAL_FEATURE_COLUMNS = [
    # 盈利能力（原始5个 + 新增5个）
    "zscore_roe_waa",              # 加权平均ROE
    "zscore_roe_dt",               # 扣非ROE（新版）
    "zscore_roa",                  # 总资产收益率（新版）
    "zscore_or_yoy",               # 营业收入同比增速
    "zscore_netprofit_yoy",        # 净利润同比增速
    "zscore_profit_dedt",          # 扣非净利润（新版，需zscore后从profit_dedt_yoy得到）
    "zscore_q_gr_yoy",             # 单季度营收同比增速
    "zscore_equity_yoy",           # 净资产同比增长率（新版）
    "zscore_grossprofit_margin",   # 毛利率（新版）
    "zscore_netprofit_margin",     # 净利率（新版）
    # 盈利质量（2个新版）
    "zscore_cf_sales",             # 经营现金流/营业收入（新版）
    "zscore_cf_nm",                # 经营现金流/净利润（新版）
    # 偿债/流动性（原有1个 + 新增2个）
    "zscore_debt_to_assets",       # 资产负债率
    "zscore_current_ratio",        # 流动比率（新版）
    "zscore_quick_ratio",          # 速动比率（新版）
    # 无形资产风险代理 + 运营效率（3个新版）
    "zscore_int_to_talcap",        # 无形资产/总资本比（替代 goodwill）
    "zscore_assets_turn",          # 总资产周转率（新版）
    "zscore_inv_turn",             # 存货周转率（新版）
    # 新鲜度
    "fundamental_freshness_days",  # 最近一次基本面公告距当日天数
]

MARGIN_FEATURE_COLUMNS = [
    "rzye_chg_5",  # 融资余额5日变动率
    "rzye_chg_20",  # 融资余额20日变动率
    "rqye_rzye_ratio",  # 融券/融资余额比
    "margin_net_buy_ratio",  # 融资净买入/成交额
]

ALT_FEATURE_COLUMNS = [
    # 股东人数 (2)
    "holder_num_chg",  # 股东人数环比变动率
    "holder_num_chg_2q",  # 股东人数两期变动率
    "holder_freshness_days",  # 最近一次股东人数公告距当日天数
    # 业绩预告 (2)
    "forecast_type_score",  # 业绩预告类型评分
    "forecast_chg_mid",  # 业绩预告变动幅度中值
    "forecast_freshness_days",  # 最近一次业绩预告公告距当日天数
]

CYQ_FEATURE_COLUMNS = [
    "winner_rate",  # 胜率
    "weight_avg_bias",  # 加权平均成本偏离度
    "cost_concentration",  # 筹码集中度
    "winner_rate_chg_5",  # 5日胜率变化
    "winner_rate_chg_20",  # 20日胜率变化
]

FUND_FEATURE_COLUMNS = [
    "fund_hold_ratio",  # 基金持股占流通股比例
    "fund_hold_ratio_chg",  # 基金持股比例较同口径上一报告期的变化
    "fund_count",  # 持仓基金数量
    "fund_count_chg",  # 持仓基金数量较同口径上一报告期的变化
    "fund_portfolio_freshness_days",  # 最近一次基金持仓公告距当日天数
]

EXPRESS_FEATURE_COLUMNS = [
    "express_revenue_yoy",  # 营业收入同比增速
    "express_profit_yoy",  # 净利润同比增速
    "express_roe",  # 快报ROE
    "express_surprise",  # 业绩惊喜
    "express_freshness_days",  # 最近一次业绩快报公告距当日天数
]

NORTH_FEATURE_COLUMNS = [
    "north_flow",  # 当日北向净流入（亿元）
    "north_flow_ma5",  # 5日移动均值
    "north_flow_ma20",  # 20日移动均值
    "north_flow_z20",  # 20日滚动 z-score
    "north_flow_sum5",  # 5日累计净流入
    "north_flow_sign_streak",  # 连续同方向天数
]

LHB_FEATURE_COLUMNS = [
    "lhb_on_list",  # 当日是否上榜
    "lhb_net_amount",  # 龙虎榜净买入额
    "lhb_net_rate",  # 净买入占流通市值比
    "lhb_amount_rate",  # 龙虎榜成交占比
    "lhb_up_days_20",  # 近 20 日累计上榜次数
    "lhb_net_sum_5",  # 近 5 日净买入累计
    "lhb_net_sum_20",  # 近 20 日净买入累计
    "lhb_reason_count",  # 当日上榜理由数
]

CONSENSUS_FEATURE_COLUMNS = [
    "cons_analyst_count_30d",  # 近 30 日覆盖的研报数
    "cons_eps_mean_fy0",  # 近 90 日当前财年 (FY0) EPS 预测均值
    "cons_eps_mean_fy1",  # 近 90 日未来第一财年 (FY1) EPS 预测均值
    "cons_eps_mean_fy2",  # 近 90 日未来第二财年 (FY2) EPS 预测均值
    "cons_eps_revision_30d",  # 近 30 日 EPS 预测修正率（全预测期）
    "cons_target_price_mid",  # 近 90 日目标价中值
    "cons_rating_score",  # 近 90 日平均评级得分
    "consensus_freshness_days",  # 最近一次研报距当日天数
]

ENHANCED_FEATURE_COLUMNS = [
    "zscore_opening_strength",       # 开盘强度（隔夜情绪代理）
    "zscore_intraday_vol_structure",  # 日内波动结构（多空力量对比）
    "zscore_order_imbalance",        # 特大单订单失衡
    "order_imbalance_mean_5",        # 5日订单失衡均值
    "order_imbalance_mean_20",       # 20日订单失衡均值
]

CASHFLOW_QUALITY_FEATURE_COLUMNS = [
    "zscore_ocf_to_revenue",         # OCF / 营业收入（现金含量）
    "zscore_ocf_to_profit",          # OCF / 净利润（利润质量）
    "zscore_fcf_yield",              # 自由现金流 / 总市值（现金回报率）
    "zscore_capex_to_ocf",           # 资本支出 / OCF
    "cashflow_freshness_days",       # 最近一次现金流公告距当日天数
]

CONSENSUS_REVISION_FEATURE_COLUMNS = [
    "zscore_cons_eps_revision_accel",     # EPS 修正加速度
    "zscore_cons_eps_dispersion",         # 分析师分歧度（负向预警）
    "zscore_cons_eps_dispersion_chg",     # 分歧度月度变化
    "zscore_cons_target_upside",          # 目标价上行空间
    "zscore_cons_target_upside_chg",      # 目标价上行空间月度变化
    "zscore_cons_analyst_count_chg",      # 覆盖分析师数变化
    "zscore_cons_rating_upgrade_ratio",   # 评级上调比例
    "cons_revision_freshness_days",       # 最近一次研报距当日天数
]

FRESHNESS_STRATEGY_DROP_ALL = "drop_all"

FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY = "state_keep_event_decay"

FRESHNESS_STRATEGY_STATE_KEEP_EVENT_NO_DECAY = "state_keep_event_no_decay"

STATE_FRESHNESS_COLUMNS = {
    "fundamental_freshness_days",
    "holder_freshness_days",
    "fund_portfolio_freshness_days",
    "cashflow_freshness_days",
}

EVENT_FRESHNESS_TO_VALUE_COLUMNS = {
    "forecast_freshness_days": [
        "forecast_type_score",
        "forecast_chg_mid",
    ],
    "express_freshness_days": [
        "express_revenue_yoy",
        "express_profit_yoy",
        "express_roe",
        "express_surprise",
    ],
    "consensus_freshness_days": [
        "cons_analyst_count_30d",
        "cons_eps_mean_fy0",
        "cons_eps_mean_fy1",
        "cons_eps_mean_fy2",
        "cons_eps_revision_30d",
        "cons_target_price_mid",
        "cons_rating_score",
    ],
    "cons_revision_freshness_days": [
        "zscore_cons_eps_revision_accel",
        "zscore_cons_eps_dispersion",
        "zscore_cons_eps_dispersion_chg",
        "zscore_cons_target_upside",
        "zscore_cons_target_upside_chg",
        "zscore_cons_analyst_count_chg",
        "zscore_cons_rating_upgrade_ratio",
    ],
}

FACTOR_EXCLUDE_LIST_FILE = "factor_exclude_list.json"
