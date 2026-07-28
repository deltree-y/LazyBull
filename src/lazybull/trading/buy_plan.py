"""买入计划共享逻辑：槽位-候选匹配骨架

回测（_execute_pending_buys / 仓位补齐）与纸面交易（补位买入）共用的
槽位匹配语义：

- 按槽位顺序遍历，每个槽位从排序候选池头部开始顺位匹配
- 同一交易日内已被其他槽位买入的候选自动去重
- 候选约束检查与实际下单通过回调注入（数据源/执行层两侧各异）
- 某候选成交即填充该槽位，槽位候选耗尽则记入未成交列表

历史上两侧此循环各自实现，多次出现顺位与去重语义漂移，
现统一收敛到 fill_slots_from_candidates。
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# 骨架内部产生的拒绝原因常量（调用侧可按需翻译为展示文案）
REASON_ALREADY_BOUGHT = "__already_bought__"  # 当日已被其他槽位买入
REASON_EXECUTION_FAILED = "__execution_failed__"  # 下单执行失败


@dataclass
class SlotMatchResult:
    """槽位匹配结果。"""

    filled: List[Dict] = field(default_factory=list)  # [{"slot": 槽位, "stock": 实际买入代码}]
    unfilled: List[Dict] = field(default_factory=list)  # 未成交槽位（原样返回）
    bought: List[str] = field(default_factory=list)  # 本次成交的股票代码（按成交顺序）


def fill_slots_from_candidates(
    slots: List[Any],
    candidates: List[str],
    evaluate_candidate: Callable[[str, Any], Tuple[bool, str]],
    execute_buy: Callable[[str, Any], bool],
    on_reject: Optional[Callable[[Any, str, str], None]] = None,
) -> SlotMatchResult:
    """按共享槽位匹配语义执行买入。

    Args:
        slots: 槽位列表（含目标权重等信息，骨架不解析内容，原样传给回调）
        candidates: 排序候选股票代码列表（优先级从高到低）
        evaluate_candidate: (候选代码, 槽位) -> (是否可买, 拒绝原因)；
            应包含已持仓、价格、可交易性、资金约束等全部检查
        execute_buy: (候选代码, 槽位) -> 是否成交；负责实际下单副作用
        on_reject: 可选回调 (槽位, 候选代码, 原因)，用于失败原因统计/日志；
            原因可能为 evaluate 返回值或本模块的 REASON_* 常量

    Returns:
        SlotMatchResult（成交映射、未成交槽位、成交代码列表）
    """
    result = SlotMatchResult()
    bought_set = set()

    for slot in slots:
        filled_for_slot = False

        for candidate in candidates:
            if candidate in bought_set:
                if on_reject is not None:
                    on_reject(slot, candidate, REASON_ALREADY_BOUGHT)
                continue

            ok, reason = evaluate_candidate(candidate, slot)
            if not ok:
                if on_reject is not None:
                    on_reject(slot, candidate, reason)
                continue

            if execute_buy(candidate, slot):
                bought_set.add(candidate)
                result.bought.append(candidate)
                result.filled.append({"slot": slot, "stock": candidate})
                filled_for_slot = True
                break

            if on_reject is not None:
                on_reject(slot, candidate, REASON_EXECUTION_FAILED)

        if not filled_for_slot:
            result.unfilled.append(slot)

    return result
