# -*- coding: utf-8 -*-
"""P7 T4 重启/续跑审计（C4-04）。

职责
====
1. 从检查点「该日**开始**」的**版本化完整信封**（代际 + 幂等历史 + 双口径 + 续跑态）
   与费用台账累计快照还原（先做 JSON 往返，确保还原对象真正可持久化）；
2. 用**新宿主**连续推进若干交易日，逐日比对完整账本 / 费用累计 / 实际与独立 NAV /
   逐日账本投影行；续跑回到全窗末日时再比完整费用 payload（含逐键明细）；
3. 覆盖停牌中重启、顺延残差与禁买状态；禁买态用「同一信封 + 注入禁买集合」的可判据探针，
   不依赖 ``E_i <= 0`` 在真实数据上必然触发；
4. 在续跑信封上复核幂等语义（同键同代无操作、跨代拒绝）。

模块边界
========
只依赖 ``state`` 的类型与宿主实例的读取面；宿主在 ``_finalize`` 内**延迟导入**本模块，
避免 ``host <-> restart`` 的循环导入。``host.py`` 因此不必承载这段审计代码。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Sequence

from scripts.v2_p7.t4.state import StateEnvelope, StateError

#: 重启检查点续跑的交易日数（>=2 才是“多日推进”；末个检查点自动截断）
RESTART_CONTINUATION_DAYS = 3


def _market_buy_shares(result: Any, sleeve_id: str) -> int:
    """某袖在该结果里市场买入的成交股数（禁买探针与基线对照用）。"""
    total = 0
    for trade in result.trades:
        if trade.get("kind") != "market_buy":
            continue
        total += int((trade.get("sleeves") or {}).get(str(sleeve_id), 0))
    return total


def _first_mismatch(
    window: Sequence[str], flags: Mapping[str, Sequence[bool]]
) -> Dict[str, Any] | None:
    """第一个出现差异的交易日与差异维度（全部一致返回 None，便于证据定位）。"""
    for pos, day in enumerate(window):
        failed = [name for name in sorted(flags) if not flags[name][pos]]
        if failed:
            return {"trade_date": str(day), "failed": failed}
    return None


def _restart_child(
    host: Any,
    *,
    envelope: Mapping[str, Any],
    fee_state: Mapping[str, Any],
    window: Sequence[str],
) -> Any:
    """用**新宿主**从给定信封 + 费用快照续跑一个窗口（不递归自检）。"""
    from scripts.v2_p7.t4.host import HistoricalVirtualSleeveHost

    child = HistoricalVirtualSleeveHost(
        params=host.params,
        run_id=host.run_id,
        prices=host._fresh_prices(),  # noqa: SLF001 - 同一价源隔离策略
        cost_model=host.cost_model,
        buy_index=host.buy_index,
        sell_index=host.sell_index,
        days=window,
        price_factory=host._price_factory,  # noqa: SLF001 - 同上
        restore={"envelope": dict(envelope), "fee_book": dict(fee_state)},
        self_check=False,
    )
    return child.run()


def _no_new_buy_probe(
    host: Any,
    *,
    envelope: Mapping[str, Any],
    fee_state: Mapping[str, Any],
    baseline: Any,
    window: Sequence[str],
    already_blocked: Sequence[str],
) -> Dict[str, Any]:
    """禁买态探针：从**同一信封**注入 ``no_new_buy``，验证续跑日该袖新增买入为 0。

    C4-04 要求覆盖「禁买状态」；真实数据上 ``E_i <= 0`` 可能始终不触发，
    因此主路径用「同一信封 + 注入禁买集合」的可判据探针覆盖，不依赖真实触发。
    基线对照取同一窗口里该袖**确实有成交**的情形，避免空过。
    """
    blocked_existing = {str(item) for item in already_blocked or ()}
    candidates = sorted(
        {intent.sleeve_id for day in window for intent in host.buy_index.get(day, ())}
        - blocked_existing
    )
    tried: List[Dict[str, Any]] = []
    for sleeve_id in candidates:
        expected = _market_buy_shares(baseline, sleeve_id)
        if expected <= 0:
            tried.append({"sleeve": sleeve_id, "baseline_buy_shares": 0})
            continue
        probed = json.loads(json.dumps(envelope, ensure_ascii=False, sort_keys=True))
        resume = dict(probed.get("resume_state") or {})
        resume["no_new_buy"] = sorted(blocked_existing | {sleeve_id})
        probed["resume_state"] = resume
        probe = _restart_child(host, envelope=probed, fee_state=fee_state, window=window)
        observed = _market_buy_shares(probe, sleeve_id)
        return {
            "sleeve": sleeve_id,
            "baseline_buy_shares": expected,
            "blocked_buy_shares": observed,
            "blocked": bool(observed == 0),
            "probe_generation": int(probe.envelope.generation),
            "skipped": tried,
        }
    return {
        "sleeve": "",
        "baseline_buy_shares": 0,
        "blocked_buy_shares": 0,
        "blocked": None,
        "skipped": tried,
        "note": "窗口内无可成交买单，禁买探针跳过（不判定）",
    }


def _idempotency_probe(envelope: StateEnvelope) -> Dict[str, Any]:
    """同键同代重复提交 => 状态不变；跨代提交 => 拒绝。"""
    key = envelope.applied and next(iter(envelope.applied)) or ""
    before = envelope.digest()
    same_generation_rejected = False
    cross_generation_rejected = False
    if key:
        generation, fingerprint = envelope.applied[key]
        try:
            applied = envelope.commit(
                key=key,
                generation=generation,
                fingerprint=fingerprint,
                sleeves=envelope.sleeves,
                physical=envelope.physical,  # type: ignore[arg-type]
                as_of=envelope.as_of,
            )
            same_generation_rejected = applied is False
        except StateError:
            same_generation_rejected = False
        try:
            envelope.commit(
                key=f"{key}:cross",
                generation=generation,
                fingerprint="x",
                sleeves=envelope.sleeves,
                physical=envelope.physical,  # type: ignore[arg-type]
                as_of=envelope.as_of,
            )
        except StateError:
            cross_generation_rejected = True
    after = envelope.digest()
    return {
        "probe_key": key,
        "same_key_same_generation_is_noop": bool(same_generation_rejected),
        "state_unchanged_after_replay": bool(before == after),
        "cross_generation_rejected": bool(cross_generation_rejected),
    }


def audit_restarts(host: Any, state: Any) -> Dict[str, Any]:
    """从序列化信封重启：还原原代际/幂等历史/价源/费用累计，用新宿主续跑多日。

    每个检查点都从「该日**开始**」的完整版本化信封 + 费用台账快照还原，再由**新宿主**
    连续推进 ``RESTART_CONTINUATION_DAYS`` 个交易日，逐日比对：

    - 完整状态本体（lot 级袖账本 + 物理账本）摘要；
    - 费用台账累计快照（双通道 + 逐键 + 浮点闭合轨迹）；
    - 逐袖实际/独立 NAV（含 total_value 与独立权益）；
    - 逐日账本投影行（场景/抵消/成交/阻断/作废/顺延/现金/残差）。

    覆盖：停牌中重启、顺延残差、禁买状态（前两类真实触发，禁买另有注入探针）。
    """
    if not state.checkpoints:
        return {"checked": False, "note": "无检查点"}
    digest_by_day = dict(state.digest_series)
    fee_by_day = dict(state.fee_series)
    rows_by_day = {row["trade_date"]: row for row in state.day_rows}
    nav_by_day = {
        arm: {row["trade_date"]: row for row in series} for arm, series in state.nav_series.items()
    }
    rows: List[Dict[str, Any]] = []
    probes: List[Dict[str, Any]] = []
    for item in state.checkpoints:
        index = int(item["index"])
        span = min(RESTART_CONTINUATION_DAYS, len(host._days) - index)  # noqa: SLF001
        window = host._days[index : index + span]  # noqa: SLF001
        # 先做 JSON 往返：还原对象必须是**真正可持久化**的信封，而不是内存里的同一份 dict
        envelope_payload = json.loads(
            json.dumps(item.get("envelope") or {}, ensure_ascii=False, sort_keys=True)
        )
        fee_payload = json.loads(
            json.dumps(item.get("fee_state") or {}, ensure_ascii=False, sort_keys=True)
        )
        restored_resume = dict(envelope_payload.get("resume_state") or {})
        day_start_cache = dict(restored_resume.get("price_last_known") or {})
        # 续跑态只在信封里取（``_restore_state`` 以 resume_state 为唯一来源）
        replay = _restart_child(
            host, envelope=envelope_payload, fee_state=fee_payload, window=window
        )
        flags = {
            "ledger": [
                replay.digest_series[pos][1] == digest_by_day[day] for pos, day in enumerate(window)
            ],
            "fee": [replay.fee_series[pos][1] == fee_by_day[day] for pos, day in enumerate(window)],
            "nav": [
                all(
                    replay.nav_series[arm][pos] == nav_by_day[arm][day]
                    for arm in sorted(nav_by_day)
                )
                for pos, day in enumerate(window)
            ],
            "daily_row": [
                replay.daily_rows[pos] == rows_by_day[day] for pos, day in enumerate(window)
            ],
        }
        end_at_last = index + span == len(host._days)  # noqa: SLF001
        checks = {
            "generation_continued": int(replay.envelope.generation) == index + span,
            "ledger_identical": all(flags["ledger"]),
            "fee_identical": all(flags["fee"]),
            "nav_identical": all(flags["nav"]),
            "daily_row_identical": all(flags["daily_row"]),
            # 续跑回到全窗末日 => 完整费用 payload（含逐键明细）必须逐位相等
            "full_fee_payload_identical": (
                replay.fee_book.payload() == host.fee_book.payload() if end_at_last else None
            ),
        }
        # C4-04「禁买状态」探针：同一信封注入禁买集合，验证续跑日该袖新增买入为 0
        no_buy_probe = _no_new_buy_probe(
            host,
            envelope=envelope_payload,
            fee_state=fee_payload,
            baseline=replay,
            window=window,
            already_blocked=restored_resume.get("no_new_buy") or (),
        )
        checks["no_new_buy_probe_blocked"] = no_buy_probe.get("blocked")
        required = {key: value for key, value in checks.items() if value is not None}
        probes.append(_idempotency_probe(replay.envelope))
        rows.append(
            {
                "trade_date": str(item["trade_date"]),
                "index": index,
                "window": [window[0], window[-1]],
                "days_continued": span,
                "restored_generation": int(envelope_payload.get("generation") or 0),
                "restored_idempotency_keys": len(envelope_payload.get("idempotency") or {}),
                "price_cache_restored": bool(day_start_cache),
                "restored_no_new_buy": sorted(
                    {str(code) for code in (restored_resume.get("no_new_buy") or ())}
                ),
                "coverage": dict(item.get("coverage") or {}),
                "no_new_buy_probe": dict(no_buy_probe),
                **checks,
                "all_bitwise_identical": all(required.values()),
                "first_mismatch": _first_mismatch(window, flags),
            }
        )
    keys = ("carry_residual", "suspension", "no_new_buy")
    coverage = {key: any(bool(row["coverage"].get(key)) for row in rows) for key in keys}
    return {
        "checked": True,
        "policy": (
            "检查点 = 该日**开始**的完整版本化信封（代际 + 幂等历史 + 双口径 + 续跑态）"
            "+ 费用台账累计快照；新宿主连续推进 "
            f"{RESTART_CONTINUATION_DAYS} 日，逐日比对账本/费用/NAV/账本投影行。"
        ),
        "checkpoints": rows,
        "checkpoint_count": len(rows),
        "continuation_days": sorted({row["days_continued"] for row in rows}),
        "all_bitwise_identical": all(row["all_bitwise_identical"] for row in rows),
        "generation_all_continued": all(row["generation_continued"] for row in rows),
        "ledger_all_identical": all(row["ledger_identical"] for row in rows),
        "fee_all_identical": all(row["fee_identical"] for row in rows),
        "nav_all_identical": all(row["nav_identical"] for row in rows),
        "daily_row_all_identical": all(row["daily_row_identical"] for row in rows),
        "coverage": coverage,
        "coverage_note": (
            "停牌/顺延/禁买三类各自首次命中日必设检查点；``coverage`` 记**真实触发**，"
            "未真实触发的「禁买状态」由 ``no_new_buy_probe`` 从同一信封注入禁买集合同口径覆盖。"
        ),
        # C4-04「禁买状态」：不依赖 E_i<=0 真实触发的可判据探针
        "no_new_buy_probe": {
            "covered": any(row["no_new_buy_probe"].get("blocked") for row in rows),
            "policy": (
                "取同一检查点信封，向 resume_state.no_new_buy 注入候选袖，"
                "续跑同一窗口后该袖市场买入必须为 0，且基线同窗口该袖确有成交。"
            ),
            "rows": [
                {
                    "trade_date": row["trade_date"],
                    "window": row["window"],
                    **dict(row["no_new_buy_probe"]),
                }
                for row in rows
            ],
        },
        "price_book_isolated": host._price_factory is not None,  # noqa: SLF001
        "idempotency": probes[0] if probes else {},
    }
