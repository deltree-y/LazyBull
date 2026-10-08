"""v2 公共表结构：快照列名与中文表头的唯一来源（P2a-T1，F10R-2 拆分复制）。

来源与边界（F10R-2 裁决）：旧模块 ``common/sidecar_schema.py`` 非政策专用——
持仓快照 / WF 报告 / 政策链三方共享。本模块迁移其中的**共享部分**：

- 持仓快照内部键（``SNAPSHOT_KEYS``）与中文表头（``SNAPSHOT_COLUMNS_ZH``）；
- 中文化工具函数（``to_chinese`` / ``select_chinese``）。

**政策专用键不迁移**（随 F8 政策层退役，仍留旧模块）：风险台账
``LEDGER_COLUMNS_ZH``、触发清单 ``TRIGGER_COLUMNS_ZH``、阈值扫描
``SCAN_COLUMNS_ZH``、暴露门控 ``GATE_*`` 系列。

迁移期与旧模块双源并存（B5）：``SNAPSHOT_*`` 与两个工具函数的逐值一致由
``tests/test_v2_p2a_t1_equivalence.py`` 锁定；旧模块在切换前保持生产可用，
本模块为 v2 侧唯一来源。
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
