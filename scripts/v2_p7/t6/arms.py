# -*- coding: utf-8 -*-
"""P7 T6 两臂运行：VV-H（同配额、禁 credit）与 VV-S（允许 credit）全窗逐日推进。

复用面（**只读**，逐件 SHA 已进 ``identity.json``）
================================================
- ``scripts.v2_p7.t5.runner._run_arm``：单臂宿主构造与运行（含内部价源隔离）；
- ``scripts.v2_p7.t5.runner._cost_model``：Q3=A 费用模型的唯一装配；
- ``scripts.v2_p7.t4.host.build_intent_index``：买/卖意图索引（T2.5 冻结计划）。

两臂唯一差异 = ``T5Mode.credit``；初资、袖资本、目标、成本、日历、执行语义、
宿主类、价源构造方式、运行窗口、checkpoint **判定函数**全部相同（G2）。

注意：checkpoint 日集合可能因两臂 coverage 不同而不同（``wants_checkpoint`` 依赖各自状态），
这不是可比性缺陷——checkpoint 只影响续跑/重放抽样，不影响逐日推进语义。

确定性（G5）
===========
``rerun`` 用同一 ``T6Params`` 独立重跑两臂（新建价源与宿主），逐位比对
状态摘要 / NAV / 交易 / 守恒行 / 逐袖资产负债 / credit 台账；结果只进
``identity.json``，不覆盖首跑产物。

**两次运行各自携带的宿主自检（``replay`` / ``restart``）也计入 G5**（C6-01）：
``checked`` 为假、载荷缺失或 ``all_bitwise_identical`` 为假 ⇒ 两臂任一即拒绝发布。
宿主只把这些结果当返回值、不会自行抛错，故必须在 T6 层合取。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from loguru import logger

from scripts.v2_p7.t4.host import build_intent_index
from scripts.v2_p7.t5.host import MODE_VVH, MODE_VVS, T5HostResult
from scripts.v2_p7.t5.runner import _cost_model, _run_arm
from scripts.v2_p7.t6.params import T6Params

#: 臂名 -> T5 模式（任务书 §4.1：两臂共用 T5 宿主，唯一差异是 credit 开关）
ARM_MODES: Mapping[str, Any] = {"vvh": MODE_VVH, "vvs": MODE_VVS}

#: 臂名 -> 证据目录名（``arms/{vvh,vvs}/``）
ARM_DIRS: Mapping[str, str] = {"vvh": "vvh", "vvs": "vvs"}


class ArmError(RuntimeError):
    """两臂运行的机械前置条件不满足（参数不可比、缺失臂），一律 fail-fast。"""


def cost_model(params: T6Params) -> Any:
    """Q3=A 费用模型（与 T5 同一装配，不新增任何费率或口径）。"""
    return _cost_model(params.t5)


def intent_index(params: T6Params) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    """买/卖意图索引（按执行日 / 到期日；只读 T2.5 冻结计划）。"""
    return build_intent_index(params.t25.root, params.arms)


def run_arms(
    *,
    params: T6Params,
    days: Sequence[str],
    run_id: str,
    arms: Sequence[str] = ("vvh", "vvs"),
) -> Dict[str, T5HostResult]:
    """两臂全窗运行（每臂独立价源，防跨臂污染）。"""
    _assert_comparable(params, arms)
    model = cost_model(params)
    buy_index, sell_index = intent_index(params)
    results: Dict[str, T5HostResult] = {}
    for name in arms:
        if name not in ARM_MODES:
            raise ArmError(f"未登记的臂：{name!r}（登记={sorted(ARM_MODES)}）")
        logger.info(f"T6 运行臂 {name}（{ARM_MODES[name].label}）：{len(days)} 日")
        results[name] = _run_arm(
            params=params.t5,
            days=tuple(days),
            run_id=f"{run_id}:{name}",
            mode=ARM_MODES[name],
            cost_model=model,
            buy_index=buy_index,
            sell_index=sell_index,
        )
    return results


def _assert_comparable(params: T6Params, arms: Sequence[str]) -> None:
    """两臂可比性前置：臂集合冻结、参数快照唯一、初资/袖资本/配额结构自洽。

    冻结值只在这里**引用**（``calibration.PHYSICAL_CAPITAL`` 与 ``params``），
    不再复制字面量，避免同一冻结量出现两个来源。
    """
    from scripts.v2_p7.t6 import calibration

    missing = [name for name in ("vvh", "vvs") if name not in arms]
    if missing:
        raise ArmError(f"两臂必须同跑（缺失：{missing}）；分臂单跑不得出 verdict")
    if float(params.physical_capital) != float(calibration.PHYSICAL_CAPITAL):
        raise ArmError(
            "物理初资与预登记常量不符："
            f"{params.physical_capital!r} != {calibration.PHYSICAL_CAPITAL!r}"
        )
    if abs(sum(params.sleeve_capital.values()) - float(params.physical_capital)) > 1e-9:
        raise ArmError("袖资本之和与物理初资不等（两臂分母不可比）")
    for sleeve, capital in sorted(params.sleeve_capital.items()):
        expected = float(params.ratio_of(sleeve)) * float(params.physical_capital)
        if abs(float(capital) - expected) > 1e-9:
            raise ArmError(
                f"袖 {sleeve} 资本与冻结配额 ρ 不一致：{capital!r} != ρ·PC = {expected!r}"
            )


def mode_payload() -> Dict[str, Any]:
    """两臂模式载荷（G2：差异表只允许 ``credit`` 一项）。"""
    return {name: dict(mode.payload()) for name, mode in sorted(ARM_MODES.items())}


def determinism_check(
    *,
    params: T6Params,
    days: Sequence[str],
    first: Mapping[str, T5HostResult],
    run_id: str,
) -> Dict[str, Any]:
    """同参独立重跑两臂并逐位比对（G5）；差异或宿主自检失败即阻断发布。"""
    second = run_arms(params=params, days=days, run_id=f"{run_id}:rerun")
    arms: Dict[str, Any] = {}
    failures: List[str] = []
    for name in ("vvh", "vvs"):
        payload = arm_determinism_payload(first[name], second[name])
        if not payload["bitwise_ok"]:
            failures.append(f"{name} 同参重跑不一致")
        if not payload["self_checks_ok"]:
            failures.append(f"{name} 宿主重放/续跑自检未通过（checked/identical 不全为真）")
        arms[name] = payload
    return {
        "schema": "p7_t6_determinism/v2",
        "ok": not failures,
        "failures": failures,
        "arms": arms,
        "rule": (
            "同参独立重跑：状态摘要 / NAV / 交易 / 守恒行 / 逐袖资产负债 / credit 台账逐位一致，"
            "且两次运行各自的宿主 replay/restart 自检 checked 且 all_bitwise_identical 全为真"
        ),
    }


def arm_determinism_payload(left: Any, right: Any) -> Dict[str, Any]:
    """单臂的确定性载荷（纯函数：两次运行的一致项 + 各自宿主自检的合取，C6-01）。"""
    from scripts.v2_p7.t4.state import stable_digest

    payload: Dict[str, Any] = {
        "digest_series_equal": [a[1] for a in left.digest_series]
        == [b[1] for b in right.digest_series],
        "nav_series_equal": _series_equal(left.nav_series, right.nav_series),
        "trades_equal": _rows_equal(left.trades, right.trades),
        "conservation_equal": _rows_equal(left.conservation_rows, right.conservation_rows),
        "balance_equal": _rows_equal(left.balance_rows, right.balance_rows),
        "credit_payload_equal": _rows_equal([left.credit_payload], [right.credit_payload]),
        "state_digest": {
            "first": left.envelope.state_digest(),
            "second": right.envelope.state_digest(),
        },
        "final_digest": stable_digest(
            {"first": left.envelope.digest(), "second": right.envelope.digest()}
        ),
        "self_checks": {"first": self_check_rows(left), "second": self_check_rows(right)},
    }
    payload["bitwise_ok"] = bool(
        payload["digest_series_equal"]
        and payload["nav_series_equal"]
        and payload["trades_equal"]
        and payload["conservation_equal"]
        and payload["balance_equal"]
        and payload["credit_payload_equal"]
        and payload["state_digest"]["first"] == payload["state_digest"]["second"]
    )
    payload["self_checks_ok"] = bool(
        self_checks_ok(payload["self_checks"]["first"])
        and self_checks_ok(payload["self_checks"]["second"])
    )
    payload["ok"] = bool(payload["bitwise_ok"] and payload["self_checks_ok"])
    return payload


#: 宿主自检类别（任务书 §6.1 G5 后半句：续跑 / 重放抽样一致）
SELF_CHECK_KINDS: Tuple[str, ...] = ("replay", "restart")


def self_check_rows(result: Any) -> Dict[str, Dict[str, Any]]:
    """把宿主的 ``replay`` / ``restart`` 自检抽成可门控的行（缺失即视为未执行）。"""
    rows: Dict[str, Dict[str, Any]] = {}
    for kind in SELF_CHECK_KINDS:
        payload = getattr(result, kind, None)
        present = isinstance(payload, Mapping)
        rows[kind] = {
            "present": bool(present),
            "checked": bool((payload or {}).get("checked")) if present else False,
            "identical": bool((payload or {}).get("all_bitwise_identical")) if present else False,
            "note": str((payload or {}).get("note") or "") if present else "",
        }
    return rows


def self_checks_ok(rows: Mapping[str, Mapping[str, Any]]) -> bool:
    """三类缺一即失败：**未登记** / **未执行** / **不一致**（C6-01）。"""
    if not rows:
        return False
    for kind in SELF_CHECK_KINDS:
        row = rows.get(kind)
        if not row or not row.get("present") or not row.get("checked") or not row.get("identical"):
            return False
    return True


def _series_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if sorted(left) != sorted(right):
        return False
    return all(list(left[key]) == list(right[key]) for key in sorted(left))


def _rows_equal(left: Sequence[Any], right: Sequence[Any]) -> bool:
    return list(left) == list(right)
