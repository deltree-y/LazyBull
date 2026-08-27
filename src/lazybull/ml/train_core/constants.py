# -*- coding: utf-8 -*-
"""train_core 特征列清单与 freshness 策略常量。"""

FUNDAMENTAL_FEATURE_COLUMNS = [
    # 盈利能力（原始5个 + 新增5个）
    "zscore_roe_waa",  # 加权平均ROE
    "zscore_roe_dt",  # 扣非ROE（新版）
    "zscore_roa",  # 总资产收益率（新版）
    "zscore_or_yoy",  # 营业收入同比增速
    "zscore_netprofit_yoy",  # 净利润同比增速
    "zscore_profit_dedt",  # 扣非净利润（新版，需zscore后从profit_dedt_yoy得到）
    "zscore_q_gr_yoy",  # 单季度营收同比增速
    "zscore_equity_yoy",  # 净资产同比增长率（新版）
    "zscore_grossprofit_margin",  # 毛利率（新版）
    "zscore_netprofit_margin",  # 净利率（新版）
    # 盈利质量（2个新版）
    "zscore_cf_sales",  # 经营现金流/营业收入（新版）
    "zscore_cf_nm",  # 经营现金流/净利润（新版）
    # 偿债/流动性（原有1个 + 新增2个）
    "zscore_debt_to_assets",  # 资产负债率
    "zscore_current_ratio",  # 流动比率（新版）
    "zscore_quick_ratio",  # 速动比率（新版）
    # 无形资产风险代理 + 运营效率（3个新版）
    "zscore_int_to_talcap",  # 无形资产/总资本比（替代 goodwill）
    "zscore_assets_turn",  # 总资产周转率（新版）
    "zscore_inv_turn",  # 存货周转率（新版）
    # 新鲜度
    "fundamental_freshness_days",  # 最近一次基本面公告距当日天数
]

# 估值缺失标记列（可选：由 features/builder/static_extra.py 生成，
# 旧 schema 特征分区可能缺失，训练时按存在性自动加入/跳过）
MISSING_MARKER_FEATURE_COLUMNS = [
    "dv_ttm_missing",  # 股息率缺失标记
    "pe_ttm_missing",  # PE 缺失标记
]

MARGIN_FEATURE_COLUMNS = [
    "rzye_chg_5",  # 融资余额5日变动率
    "rzye_chg_20",  # 融资余额20日变动率
    "rqye_rzye_ratio",  # 融券/融资余额比
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
    "north_net_buy",  # 切换前当日净买入（亿元），切换后为 0
    "north_net_buy_ma5",  # 切换前净买入 5 日均值
    "north_net_buy_ma20",  # 切换前净买入 20 日均值
    "north_net_buy_z20",  # 切换前净买入 20 日 z-score
    "north_net_buy_sum5",  # 切换前净买入 5 日累计
    "north_net_buy_sign_streak",  # 切换前近 20 日连续净流入/流出方向
    "north_turnover",  # 切换后当日成交额（亿元），切换前为 0
    "north_turnover_ma5",  # 切换后成交额 5 日均值
    "north_turnover_ma20",  # 切换后成交额 20 日均值
    "north_turnover_z20",  # 切换后成交额 20 日 z-score
    "north_turnover_sum5",  # 切换后成交额 5 日累计
    "north_turnover_change_streak",  # 切换后近 20 日连续放量/缩量方向
    "north_turnover_flag",  # 口径指示（0=净买入, 1=成交额）
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
    "lhb_cont_on_list",  # 当日是否连续异动上榜（reason 含"连续"）
    "lhb_cont_up_days_5",  # 近 5 交易日连续异动上榜次数累计
    "lhb_cont_up_days_20",  # 近 20 交易日连续异动上榜次数累计
]

CONSENSUS_FEATURE_COLUMNS = [
    "cons_analyst_count_30d",  # 近 30 日覆盖的研报数
    "cons_eps_yield_fym1",  # 上一财年 EPS / 当日未复权收盘价
    "cons_eps_yield_fy0",  # 当前财年 EPS / 当日未复权收盘价
    "cons_eps_yield_fy1",  # 未来第一财年 EPS / 当日未复权收盘价
    "cons_eps_yield_fy2",  # 未来第二财年 EPS / 当日未复权收盘价
    "cons_eps_revision_30d",  # 近 30 日相对此前 90 日有界修正率（全预测期）
    "cons_target_upside",  # 目标价中值 / 当日未复权收盘价 - 1
    "cons_rating_score",  # 近 90 日平均评级得分
    "consensus_freshness_days",  # 最近一次研报距当日天数
]

ENHANCED_FEATURE_COLUMNS = [
    "zscore_opening_strength",  # 开盘强度（隔夜情绪代理）
    "zscore_intraday_vol_structure",  # 日内波动结构（多空力量对比）
    "zscore_order_imbalance",  # 特大单订单失衡
    "order_imbalance_mean_5",  # 5日订单失衡均值
    "order_imbalance_mean_20",  # 20日订单失衡均值
]

CASHFLOW_QUALITY_FEATURE_COLUMNS = [
    "zscore_ocf_to_revenue",  # OCF / 营业收入（现金含量）
    "zscore_ocf_to_profit",  # OCF / 净利润（利润质量）
    "zscore_fcf_yield",  # 自由现金流 / 总市值（现金回报率）
    "zscore_capex_to_ocf",  # 资本支出 / OCF
    "cashflow_freshness_days",  # 最近一次现金流公告距当日天数
]

CONSENSUS_REVISION_FEATURE_COLUMNS = [
    "zscore_cons_eps_revision_accel",  # EPS 修正速度（按日历时间斜率）
    "zscore_cons_eps_dispersion",  # 分析师分歧度（同日同 FY 研报级）
    "zscore_cons_eps_dispersion_chg",  # 分歧度变化
    "zscore_cons_target_upside_chg",  # 目标价均值变化
    "zscore_cons_analyst_count_chg",  # 研报覆盖数变化
    "zscore_cons_rating_upgrade_ratio",  # 评级上调比例
    "cons_revision_freshness_days",  # 最近一次研报距当日天数
]

FRESHNESS_STRATEGY_DROP_ALL = "drop_all"

FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY = "state_keep_event_decay"

FRESHNESS_STRATEGY_STATE_KEEP_EVENT_NO_DECAY = "state_keep_event_no_decay"

DEFAULT_EVENT_FRESHNESS_HALF_LIFE_DAYS = 45.0

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
        # 旧绝对值列仅供存量模型推理，新模型使用下方经济归一化列。
        "cons_eps_mean_fym1",
        "cons_eps_mean_fy0",
        "cons_eps_mean_fy1",
        "cons_eps_mean_fy2",
        "cons_eps_yield_fym1",
        "cons_eps_yield_fy0",
        "cons_eps_yield_fy1",
        "cons_eps_yield_fy2",
        "cons_eps_revision_30d",
        "cons_target_price_mid",
        "cons_target_upside",
        "cons_rating_score",
    ],
    "cons_revision_freshness_days": [
        "zscore_cons_eps_revision_accel",
        "zscore_cons_eps_dispersion",
        "zscore_cons_eps_dispersion_chg",
        "zscore_cons_target_upside_chg",
        "zscore_cons_analyst_count_chg",
        "zscore_cons_rating_upgrade_ratio",
    ],
}

FACTOR_EXCLUDE_LIST_FILE = "factor_exclude_list.json"


def attach_cons_revision_schema_version(train_params: dict, enable_flag: bool) -> dict:
    """当一致预期修正因子开关开启时，在训练元数据中记录修正 schema 版本。

    所有模型注册入口（split/deploy/单次训练）必须共用此函数，保证新旧模型
    可通过 `cons_revision_schema_version` 机器级区分，推理侧据此识别 v1 旧模型。
    """
    if enable_flag:
        from src.lazybull.factors.consensus_revision import CONSENSUS_REVISION_SCHEMA_VERSION

        train_params["cons_revision_schema_version"] = CONSENSUS_REVISION_SCHEMA_VERSION
    return train_params


def read_cons_revision_schema_version(train_params: dict) -> int:
    """安全读取修正 schema 版本；缺失或异常内容（如 "v2" 字符串）返回 -1。"""
    raw = (train_params or {}).get("cons_revision_schema_version")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return -1
