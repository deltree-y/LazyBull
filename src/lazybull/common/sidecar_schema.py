"""政策旁路（policy sidecar）产物表结构：中文表头与内部键的唯一来源。

设计约束（2026-09-14 共识）：

- **用户可见的政策旁路产物一律中文表头**（持仓快照 / 风险台账 / 触发清单 /
  阈值扫描），因此中文列名必须集中定义，禁止在生产者与消费者两侧各写一份。
- 本模块是**叶子模块**（不导入项目内任何其它模块），供两条链路共同引用：
  生产者 ``backtest.holdings_snapshot``（引擎只读快照）与消费者
  ``risk.terminal_loss.policy_sidecar``（打分与报告）。放在 ``common`` 下是为了
  避免 ``backtest → risk → ml`` 之类的前向依赖形成循环导入。
- 内部键保持英文（代码可读、便于 join）；落盘/展示统一经 :func:`to_chinese`
  改名，缺失列明确报错，不静默丢列。
"""

from typing import Dict, List, Sequence

# ── 表 1：持仓快照（回测侧逐日导出，一行 = 某日某持仓）────────────────

SNAPSHOT_KEYS: List[str] = [
    "date",
    "ts_code",
    "shares",
    "market_value",
    "weight",
    "portfolio_value",
    "buy_date",
    "signal_date",
    "holding_days",
    "planned_exit_date",
    "remaining_intervals",
]

SNAPSHOT_COLUMNS_ZH: Dict[str, str] = {
    "date": "日期",
    "ts_code": "股票代码",
    "shares": "持仓股数",
    "market_value": "持仓市值",
    "weight": "持仓权重",
    "portfolio_value": "组合总值",
    "buy_date": "买入日",
    "signal_date": "信号日",
    "holding_days": "持有交易日数",
    "planned_exit_date": "到期执行日",
    "remaining_intervals": "剩余持有交易日",
}

# ── 表 2：持仓风险台账（打分后，一行 = 某持仓的某个剩余持有期）────────

LEDGER_COLUMNS_ZH: List[str] = [
    "运行标识",
    "折序号",
    "风险模型折",
    "日期",
    "股票代码",
    "持仓权重",
    "持仓市值",
    "买入日",
    "持有交易日数",
    "到期执行日",
    "剩余持有交易日",
    "风险概率",
    "当日截面分位",
    "当日截面样本数",
    "日波动率",
    "持有期预期波动",
    "市场波动状态",
    "事后实际收益",
    "是否异常亏损",
    "标签状态",
    "事后可避免损失",
    "主策略最后持仓日",
]

# ── 表 3：触发清单（双重条件命中 / 漏报，一行 = 一个关注事件）─────────

TRIGGER_COLUMNS_ZH: List[str] = [
    "日期",
    "股票代码",
    "事件类型",
    "风险概率",
    "当日截面分位",
    "剩余持有交易日",
    "持仓权重",
    "日波动率",
    "持有期预期波动",
    "市场波动状态",
    "事后实际收益",
    "是否异常亏损",
    "事后可避免损失",
]

# ── 表 4：阈值扫描（一行 = 一组阈值参数）──────────────────────────────

SCAN_COLUMNS_ZH: List[str] = [
    "截面分位阈值",
    "绝对概率阈值",
    "触发笔数",
    "触发占比",
    "事件拦截率",
    "误杀率",
    "误杀且上涨占比",
    "漏报率",
    "触发样本平均事后收益",
    "未触发样本平均事后收益",
    "事后可避免损失合计",
]


# ── 表 5：暴露门控校准（E2，一行 = 一个臂）──────────────────────────

GATE_CALIBRATION_KEYS: List[str] = [
    "arm",
    "calib_start",
    "calib_end",
    "calib_days",
    "regime_quantile",
    "regime_threshold",
    "score_quantile",
    "score_threshold",
    "de_exposure_multiplier",
    "layer_days",
    "calib_trigger_days",
    "calib_trigger_share",
]

GATE_CALIBRATION_COLUMNS_ZH: List[str] = [
    "臂",
    "校准段起",
    "校准段止",
    "校准交易日数",
    "波动分位参数",
    "市场波动阈值",
    "得分分位参数",
    "得分阈值",
    "降暴露系数",
    "层内校准日数",
    "校准段触发日数",
    "校准段触发占比",
]

# ── 表 6：暴露门控逐日判定（一行 = 某臂某交易日）────────────────────

GATE_DAILY_KEYS: List[str] = [
    "arm",
    "date",
    "fold",
    "threshold_mode",
    "window_days",
    "regime_threshold",
    "mkt_vol_20",
    "vol_percentile",
    "p_loss_mean",
    "score_threshold",
    "layer_score_percentile",
    "holdings",
    "weight_sum",
    "day_weighted_return",
    "day_mean_return",
    "triggered",
    "first_trigger",
    "exposure_multiplier",
    "in_calibration",
]

GATE_DAILY_COLUMNS_ZH: List[str] = [
    "臂",
    "日期",
    "风险模型折",
    "阈值口径",
    "窗口日数",
    "市场波动阈值",
    "市场波动状态",
    "波动分位",
    "组合平均风险概率",
    "得分阈值",
    "层内得分分位",
    "持仓数",
    "组合权重和",
    "当日加权事后收益",
    "当日平均事后收益",
    "是否触发",
    "是否首触",
    "暴露系数",
    "是否校准段",
]

# ── 表 7：暴露门控评估（一行 = 某臂 × 口径 × 分组/折）────────────────

GATE_EVAL_KEYS: List[str] = [
    "arm",
    "scope",
    "group",
    "fold",
    "days",
    "trigger_days",
    "trigger_share",
    "first_trigger_days",
    "first_trigger_share",
    "trigger_mean_return",
    "nontrigger_mean_return",
    "return_gap",
    "trigger_loss_day_rate",
    "nontrigger_loss_day_rate",
    "trigger_event_rate",
    "nontrigger_event_rate",
    "gain_term",
    "cost_term",
    "net_daily_delta",
]

GATE_EVAL_COLUMNS_ZH: List[str] = [
    "臂",
    "口径",
    "分组",
    "风险模型折",
    "交易日数",
    "触发日数",
    "触发占比",
    "首触日数",
    "首触占比",
    "触发日平均加权收益",
    "未触发日平均加权收益",
    "收益差",
    "触发日亏损日频率",
    "未触发日亏损日频率",
    "触发日事件率",
    "未触发日事件率",
    "收益项",
    "成本项",
    "净增量（日均口径）",
]


def to_chinese(frame, mapping: Dict[str, str]):
    """按映射把内部英文列名改为中文表头（缺列报错，不静默丢列）。

    Args:
        frame: 待改名的 DataFrame（必须含映射中的全部键）
        mapping: {内部键: 中文列名}

    Returns:
        列名已中文化的副本（行序不变）

    Raises:
        ValueError: 缺少任一内部键
    """
    missing: List[str] = [key for key in mapping if key not in frame.columns]
    if missing:
        raise ValueError(
            f"旁路产物缺少内部列 {sorted(missing)}；中文表头映射要求列齐，"
            f"禁止静默丢列（实际列: {sorted(frame.columns)}）"
        )
    return frame.rename(columns=dict(mapping))


def select_chinese(frame, mapping: Dict[str, str], keys: Sequence[str]):
    """按给定键顺序取列并中文化（用于控制输出列顺序）。"""
    missing: List[str] = [key for key in keys if key not in frame.columns]
    if missing:
        raise ValueError(f"旁路产物缺少内部列 {sorted(missing)}；无法按约定列序输出")
    return to_chinese(frame[list(keys)], mapping)
