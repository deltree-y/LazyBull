# -*- coding: utf-8 -*-
"""P7 T4 编排器：场景提取 -> 构造例 -> 全窗生命周期 -> 七件套证据落盘。

写域
====
只写 ``data/paper_shadow/p7_testbed/<t4_run_id>/``（``scripts.v2_p7.common`` 写域守卫，
``mkdir`` 前整体预检，拒绝路径零副作用）；不写 ``data/raw|clean|features|models|runs|paper``，
不覆盖任何完成根（``resume`` 遇 ``manifest.json`` 必须拒绝）。

硬门（不通过即停、且已通过项按 ``checks`` 登记）
==============================================
1. 自然实例逐笔复算与 T2.5 ``intent_precheck.json`` 及冻结计数一致；
2. 构造例全部通过（``synthetic`` / ``constructed_on_history``）；
3. I1/I2/I3 逐日闭合（``identity_failures`` 为空）；
4. 意图守恒 ``original = offset + filled + voided + carry`` 零违例；
5. 费用台账 ``Σ实际费用 = Σ物理费用`` 精确闭合；
6. 重放/重启逐位一致 + 同代幂等 + 跨代拒绝；
7. 受审旧面 SHA 与旧证据交叉核对一致。

``alpha_consumed=false``：本 task 不产生任何 pass/fail 经济判定。
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger

from scripts.v2_p7.common import (
    REPO_ROOT,
    assert_targets_within_write_roots,
    resolve_write_roots,
    sha256_file,
    write_json,
)
from scripts.v2_p7.t4 import scenarios as scenario_mod
from scripts.v2_p7.t4.fees import FEE_FLOAT_CLOSURE_TOLERANCE, FEE_LEDGER_TOLERANCE
from scripts.v2_p7.t4.host import (
    HistoricalVirtualSleeveHost,
    HostResult,
    build_intent_index,
    build_nav_frames,
)
from scripts.v2_p7.t4.params import (
    DEFAULT_RUN_ID,
    TESTBED_REL,
    T4Params,
    build_params,
    market_identity,
    params_digest,
)
from scripts.v2_p7.t4.price import PriceBook

SCHEMA = "p7_t4/v1"
TASK = "P7 T4 — P7 专用生命周期、价源、双口径与状态"

#: 浮点闭合容差（元）：精度内闭合的偏差超过此值即视为口径破缺（<= 1e-9 元）
FLOAT_CLOSURE_TOLERANCE = FEE_FLOAT_CLOSURE_TOLERANCE

#: 构造例取样日（真实交易日；窗口内且行情齐全）
CONSTRUCTED_SAMPLE_DAY = "20240102"


class RunnerError(RuntimeError):
    """硬门不通过或写域越界（停止并保留已登记证据）。"""


def run(
    repo_root: Path = REPO_ROOT,
    run_id: str = DEFAULT_RUN_ID,
    resume: bool = False,
    sample_day: str = CONSTRUCTED_SAMPLE_DAY,
    windows: Optional[Sequence[Sequence[str]]] = None,
) -> Dict[str, Any]:
    """执行 T4 全流程（默认全窗；``windows`` 可限定子窗口用于分段验证）。"""
    started = time.time()
    params = build_params(repo_root)
    run_root = Path(repo_root) / TESTBED_REL / run_id
    _prepare_run_root(run_root, resume=resume)
    source_identity = _source_identity(params)

    scenario_started = time.time()
    precheck = _read_json(params.t25.root / "intent_precheck.json")
    natural = scenario_mod.recompute_natural(
        root=params.t25.root,
        arms=params.arms,
        precheck=precheck,
        frozen_counts=dict(_frozen_counts()),
    )
    if not natural["matches_frozen_counts"] or not natural["matches_t25_precheck"]:
        raise RunnerError(
            "自然实例逐笔复算与 T2.5 不一致："
            f"matches_frozen={natural['matches_frozen_counts']} "
            f"matches_precheck={natural['matches_t25_precheck']}"
        )
    cost_model = _cost_model(params)
    cases = scenario_mod.run_constructed_cases(
        params=params, cost_model=cost_model, sample_day=sample_day
    )
    failed_cases = [case.name for case in cases if not case.passed]
    if failed_cases:
        raise RunnerError(f"构造例未通过：{failed_cases}")
    scenario_seconds = round(time.time() - scenario_started, 1)

    host_started = time.time()
    from src.lazybull.data.loader import DataLoader
    from src.lazybull.data.storage import Storage

    loader = DataLoader(Storage(root_path=str(params.data_root)))
    prices = PriceBook(loader)
    buy_index, sell_index = build_intent_index(params.t25.root, params.arms)
    host_result: HostResult

    def _fresh_prices() -> PriceBook:
        """独立价源工厂（重放/重启自验与分段运行各用独立实例，防跨段污染）。"""
        return PriceBook(DataLoader(Storage(root_path=str(params.data_root))))

    if windows:
        segments = []
        for window in windows:
            host = HistoricalVirtualSleeveHost(
                params=params,
                run_id=f"{run_id}:window",
                prices=_fresh_prices(),
                cost_model=cost_model,
                buy_index=buy_index,
                sell_index=sell_index,
                days=window,
            )
            segments.append(host.run())
        host_result = segments[-1]
        segment_payload = [
            {
                "window": [item.days[0], item.days[-1]],
                "days": len(item.days),
                "scenarios": item.scenario_counts,
                "offset_shares": item.offset_shares,
                "market_filled": item.market_filled,
                "identity_failures": len(item.identity_failures),
                "replay": item.replay,
            }
            for item in segments
        ]
    else:
        host = HistoricalVirtualSleeveHost(
            params=params,
            run_id=run_id,
            prices=prices,
            cost_model=cost_model,
            buy_index=buy_index,
            sell_index=sell_index,
            price_factory=_fresh_prices,
        )
        host_result = host.run()
        segment_payload = []
    host_seconds = round(time.time() - host_started, 1)

    host_result.fee_book.assert_closed()
    # C4-03：执行结束后重新采集同一全窗身份，确认**执行期间未换输入**后才允许发布
    input_identity_after = _input_identity_stability(params)
    checks = _acceptance_checks(
        params=params,
        natural=natural,
        cases=cases,
        result=host_result,
        input_identity_after=input_identity_after,
    )
    _stop_on_hard_failures(checks)

    artifacts_started = time.time()
    written = _write_artifacts(
        params=params,
        run_root=run_root,
        natural=natural,
        cases=cases,
        result=host_result,
        segment_payload=segment_payload,
    )
    artifacts_seconds = round(time.time() - artifacts_started, 1)

    manifest = _manifest(
        params=params,
        run_id=run_id,
        run_root=run_root,
        natural=natural,
        cases=cases,
        result=host_result,
        checks=checks,
        source_identity=source_identity,
        timings={
            "scenarios_seconds": scenario_seconds,
            "host_seconds": host_seconds,
            "artifacts_seconds": artifacts_seconds,
            "total_seconds": round(time.time() - started, 1),
        },
        artifacts=written,
    )
    write_json(run_root / "manifest.json", manifest)
    logger.info(f"T4 证据已落盘：{run_root}（{manifest['timing']['total_seconds']}s）")
    return manifest


# ────────────────────────── 前置 ──────────────────────────


def _frozen_counts() -> Mapping[str, int]:
    from scripts.v2_p7.t4.params import FROZEN_INTENT_COUNTS

    return FROZEN_INTENT_COUNTS


def _cost_model(params: T4Params) -> Any:
    from src.lazybull.v2.common.cost import CostModel

    return CostModel(
        commission_rate=params.cost_settings["commission_rate"],
        min_commission=params.cost_settings["min_commission"],
        stamp_tax=params.cost_settings["stamp_tax"],
        slippage=params.cost_settings["slippage"],
    )


def _source_identity(params: T4Params) -> Dict[str, Any]:
    from src.lazybull.v2.store.store_identity import workspace_identity

    return workspace_identity(params.repo_root)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RunnerError(f"冻结输入缺件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _prepare_run_root(run_root: Path, resume: bool) -> None:
    """写域预检 + 完成根只读。"""
    roots = resolve_write_roots()
    assert_targets_within_write_roots([run_root], roots)
    if run_root.exists() and not resume:
        raise FileExistsError(f"输出根已存在（禁止覆盖）：{run_root}")
    if resume and (run_root / "manifest.json").exists():
        raise FileExistsError(
            f"完成根只读（manifest 已存在，禁止续跑改写）：{run_root}；整改请用新 run_id 重取"
        )
    run_root.mkdir(parents=True, exist_ok=True)


def _stop_on_hard_failures(checks: Mapping[str, Any]) -> None:
    failures: List[str] = []
    if not checks["natural_instances"]["ok"]:
        failures.append("自然实例复算不一致")
    if not checks["constructed_cases"]["ok"]:
        failures.append(f"构造例未通过：{checks['constructed_cases']['failed']}")
    if not checks["identities"]["ok"]:
        failures.append(f"恒等式破缺日数={checks['identities']['failure_days']}")
    if not checks["intent_conservation"]["ok"]:
        failures.append(f"意图守恒违例={checks['intent_conservation']['violations']}")
    if not checks["fees"]["closed"]:
        failures.append(
            "费用台账未闭合或浮点闭合越界"
            f"（max_abs_ulp_gap={checks['fees']['float_closure']['max_abs_ulp_gap']!r}，"
            f"declared_minus_applied={checks['fees']['declared_minus_applied']!r}）"
        )
    if not checks["identity"]["ok"]:
        failures.append(f"幂等/重放未通过：{checks['identity']['failures']}")
    if not checks["physical_cash"]["ok"]:
        failures.append(
            "物理现金为负：非杠杆前提破缺"
            f"（{checks['physical_cash']['negative_days']} 日，"
            f"min={checks['physical_cash']['min_physical_cash']!r}）"
        )
    if not checks["price_source"]["ok"]:
        failures.append(
            "价源门未过："
            f"unavailable={checks['price_source']['distribution'].get('unavailable', 0)} "
            f"dates_missing={checks['price_source']['valuation_dates_missing']} "
            f"suspension_drift={checks['price_source']['suspension_valuation']['drift']} "
            f"resume_mismatch={len(checks['price_source']['resume_consistency']['mismatch'])}"
        )
    if not checks["legacy_sha"]["ok"]:
        failures.append(f"受审旧面不一致：{checks['legacy_sha']['modified_in_worktree']}")
    if not checks["identity"].get("input_identity_stable", False):
        failures.append(
            "执行中行情/日历输入发生变化（发布拒绝）："
            f"{checks['identity'].get('input_identity', {}).get('changed')}"
        )
    if failures:
        raise RunnerError(f"T4 硬失败，禁止发布：{'；'.join(failures)}")


# ────────────────────────── 验收门 ──────────────────────────


def _sleeve_metrics(frame: Any) -> Tuple[Any, str, List[str]]:
    """独立（袖自身初资）口径的链式指标；出现无效权益日即整袖标 invalid（C4-06）。

    无效日（非正权益）已被上游标 ``None``。**不得**用 ``dropna`` 把它删掉再算，
    否则无效日前后的 NAV 会被当成相邻有效区间，凭空造出跨越无效日的收益。
    """
    from src.lazybull.v2.common.chain_metrics import calculate_chain_metrics

    invalid_rows = frame[frame["independent_nav"].isna()]
    invalid_dates = [str(value) for value in invalid_rows["trade_date"].tolist()]
    if invalid_dates:
        return None, "invalid_nonpositive_equity_day", invalid_dates
    series = frame[["trade_date", "independent_nav", "split_index"]].rename(
        columns={"independent_nav": "nav"}
    )
    return calculate_chain_metrics(series.reset_index(drop=True)), "ok", []


def _input_identity_stability(params: T4Params) -> Dict[str, Any]:
    """执行前后同一全窗行情/日历身份的重采对比（C4-03 发布门）。

    只比较**内容摘要**：执行期间若任一 clean/daily 分区、日历或证券主表被替换，
    本 run 的账本就不再对应同一输入，必须拒绝发布。
    """
    before = params.market_identity
    after = market_identity(Path(params.repo_root), list(params.extended_days))
    pairs = (
        ("clean_daily", "digest"),
        ("clean_trade_cal", "sha256"),
        ("clean_stock_basic", "sha256"),
    )
    changed = [
        {
            "input": name,
            "before": before[name][field],
            "after": after[name][field],
        }
        for name, field in pairs
        if before[name][field] != after[name][field]
    ]
    return {
        "stable": not changed,
        "scope_days": int(after["scope"]["days"]),
        "before": {name: before[name][field] for name, field in pairs},
        "after": {name: after[name][field] for name, field in pairs},
        "changed": changed,
    }


def _acceptance_checks(
    *,
    params: T4Params,
    natural: Mapping[str, Any],
    cases: Sequence[Any],
    result: HostResult,
    input_identity_after: Mapping[str, Any],
) -> Dict[str, Any]:
    summary = scenario_mod.summarize_cases(cases)
    fee_payload = result.fee_book.payload()
    savings = fee_payload["savings_by_sleeve"]
    identity_probe = (result.restart or {}).get("idempotency") or {}
    legacy = params.legacy_sha
    return {
        "physical_cash": _physical_cash_check(result),
        "natural_instances": _natural_check(natural),
        "constructed_cases": _constructed_check(summary),
        "executed_scenarios": _executed_check(result),
        "identities": {
            "ok": bool(not result.identity_failures),
            "failure_days": len(result.identity_failures),
            "tolerance": {"I1": 1e-6, "I1b": 0.0, "I2": 0.0, "I3": 1e-6, "I3b": 1e-6},
            "invalid_equity_days": dict(result.invalid_days),
        },
        "intent_conservation": {
            "ok": bool(
                result.conservation["violations"] == 0
                and result.cross_day["violations"] == 0
                and _conservation_anchors(result)["offset_ok"]
                and _conservation_anchors(result)["filled_ok"]
            ),
            "rows": result.conservation["rows"],
            "violations": result.conservation["violations"],
            "cross_day": dict(result.cross_day),
            "anchors": _conservation_anchors(result),
            "rule": (
                "逐日逐意图 original = offset + filled + voided + carry；"
                "跨日 first_original = Σoffset + Σfilled + Σvoided + 末次 carry（容差 0 股）；"
                "另有两条独立锚：Σ逐键 offset = 2×抵消股数、Σ逐键 filled = 市场成交量"
            ),
        },
        "fees": _fees_check(fee_payload, savings, result),
        "identity": {
            "ok": bool(
                identity_probe.get("same_key_same_generation_is_noop")
                and identity_probe.get("state_unchanged_after_replay")
                and identity_probe.get("cross_generation_rejected")
                and (result.replay or {}).get("all_bitwise_identical")
                and (result.restart or {}).get("all_bitwise_identical")
                and bool(input_identity_after.get("stable"))
            ),
            "generations": int(result.envelope.generation),
            "idempotency_keys": len(result.envelope.applied),
            "replay": result.replay,
            "restart": result.restart,
            # C4-03：本 run 的身份 = 参数身份 + 全窗输入内容身份（执行前后必须一致）
            "input_identity": dict(input_identity_after),
            "input_identity_stable": bool(input_identity_after.get("stable")),
            "input_identity_digest": str(params.market_identity["clean_daily"]["digest"]),
            "params_digest": params_digest(params),
            "failures": [
                name
                for name, ok in (
                    (
                        "same_key_same_generation_is_noop",
                        identity_probe.get("same_key_same_generation_is_noop"),
                    ),
                    (
                        "state_unchanged_after_replay",
                        identity_probe.get("state_unchanged_after_replay"),
                    ),
                    ("cross_generation_rejected", identity_probe.get("cross_generation_rejected")),
                    ("replay_bitwise", (result.replay or {}).get("all_bitwise_identical")),
                    ("restart_bitwise", (result.restart or {}).get("all_bitwise_identical")),
                    ("input_identity_stable", input_identity_after.get("stable")),
                )
                if not ok
            ],
        },
        "price_source": {
            "ok": bool(
                result.price_payload["distribution"].get("unavailable", 0) == 0
                and result.price_payload["suspension_valuation"]["drift"] == 0
                and result.price_payload["valuation_dates_missing"] == 0
                and not result.price_payload["resume_consistency"]["mismatch"]
            ),
            "distribution": result.price_payload["distribution"],
            "execution_block_reasons": result.price_payload["execution_block_reasons"],
            "valuation_dates_recorded": result.price_payload["valuation_dates_recorded"],
            "valuation_dates_missing": result.price_payload["valuation_dates_missing"],
            "suspension_episodes": len(result.price_payload["suspension_episodes"]),
            "suspension_valuation": result.price_payload["suspension_valuation"],
            "resume_consistency": result.price_payload["resume_consistency"],
        },
        "legacy_sha": _legacy_check(legacy),
        "contract_deviations": {
            "gate": False,
            "requires_user_confirmation": False,
            "all_resolved": True,
            "items": [
                {
                    "id": "CD-1",
                    "status": "resolved_by_user",
                    "decided_on": "2026-11-12",
                    "title": "S4 费用终端闭合：以 100 ulp(F) 精度容差替代契约的「拒绝」语义",
                    "contract": "netting_freeze.md §5 `_close_terminal_fee`：一次残差修正后仍不等即拒绝",
                    "deviation": (
                        "P7 在「直接 / 一次修正 / 仅终端袖 ulp 微调」均不闭合成精确和时，"
                        "把**可用物理费用**取为可达精确和 `Σf_i`（`ulp_gap <= 100 ulp(F)`），"
                        "不改任何袖的费用取值；逐笔计数见 `fees.float_closure`。"
                    ),
                    "why": (
                        "受审件对真实数据直接拒绝整笔成交（`audited_waterfall_rejects_1ulp` "
                        "已落盘样例），T4 不能因此丢单；偏差量级 5.68e-14 元。"
                    ),
                    "decision": (
                        "用户裁决：改为设定精度——逐事件 100 ulp(F) 内视为闭合，全窗硬门 1e-9 元；"
                        "契约 P7 注记已同步。"
                    ),
                    "needs": "已裁决（2026-11-12），无需再确认。",
                },
                {
                    "id": "CD-2",
                    "status": "resolved_by_user",
                    "decided_on": "2026-11-12",
                    "title": "逐袖 `cash_i >= 0` 的归属：T4 登记不拒判，门归 T5",
                    "contract": "netting_freeze.md §4 I1 注记（P6.5 无借用子集要求逐袖现金非负）",
                    "deviation": (
                        "T4 `credit ≡ 0` 字面上属「无借用子集」；净额化内部转账是无 credit 背书的虚拟付款。"
                        "**C4-01 整改后内部转账已按买方自有现金限额**，故逐袖现金实测不再为负"
                        "（`negative_cash_days = 0`）。本条保留为**语义边界**：T5 引入 credit 后，"
                        "同一笔欠款将表现为 `cash_i < 0` 且 `credit_i > 0`，逐袖非负门归 T5。"
                    ),
                    "measured": {
                        "min_sleeve_cash": result.min_sleeve_cash,
                        "negative_cash_days": int(result.negative_cash_days),
                        "min_physical_cash": result.min_physical_cash,
                        "negative_physical_cash_days": int(result.negative_physical_cash_days),
                    },
                    "decision": (
                        "用户 2026-11-12 二次裁决（口径 a）：内部转账额 <= 买方袖自有现金，超出部分不打抵消、"
                        "残差按既有资金约束语义顺延——杜绝用负现金充值出可花费购买力；"
                        "另加 `物理现金 >= 0` 逐日硬门。逐袖非负门仍归 T5。"
                    ),
                    "needs": "已裁决（2026-11-12），无需再确认。",
                },
            ],
        },
        "alpha_consumed": False,
    }


# ────────────────────────── 落盘 ──────────────────────────


def _write_artifacts(
    *,
    params: T4Params,
    run_root: Path,
    natural: Mapping[str, Any],
    cases: Sequence[Any],
    result: HostResult,
    segment_payload: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """写七件套 + 场景子目录（返回逐件 SHA/行数）。"""
    written: Dict[str, Dict[str, Any]] = {}
    written["state_matrix.json"] = _write_json_counted(
        run_root / "state_matrix.json", _state_matrix(params, result)
    )
    written["price_source.json"] = _write_json_counted(
        run_root / "price_source.json", dict(result.price_payload)
    )
    written["dual_caliber.json"] = _write_json_counted(
        run_root / "dual_caliber.json", _dual_caliber(params, result)
    )
    written["fees.json"] = _write_json_counted(run_root / "fees.json", result.fee_book.payload())
    written["identity.json"] = _write_json_counted(run_root / "identity.json", _identity(result))
    written["legacy_sha.json"] = _write_json_counted(
        run_root / "legacy_sha.json", dict(params.legacy_sha)
    )
    written["scenarios/natural.json"] = _write_json_counted(
        run_root / "scenarios" / "natural.json", dict(natural)
    )
    written["scenarios/constructed.json"] = _write_json_counted(
        run_root / "scenarios" / "constructed.json",
        scenario_mod.summarize_cases(cases),
    )
    written["scenarios/executed.json"] = _write_json_counted(
        run_root / "scenarios" / "executed.json", _executed(result)
    )
    written["fees_by_key.parquet"] = _write_fee_frame(run_root / "fees_by_key.parquet", result)
    written["trades.parquet"] = _write_frames(run_root / "trades.parquet", result.trades)
    written["conservation.parquet"] = _write_frames(
        run_root / "conservation.parquet", result.conservation_rows
    )
    written["daily.parquet"] = _write_frames(run_root / "daily.parquet", result.daily_rows)
    if segment_payload:
        written["segments.json"] = _write_json_counted(
            run_root / "segments.json",
            {
                "schema": "p7_t4_segments/v1",
                "windows": [dict(item) for item in segment_payload],
                "note": "分段验证窗口（不替代全窗证据）。",
            },
        )
    return written


def _write_json_counted(path: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    digest = write_json(path, dict(payload))
    return {"sha256": digest, "bytes": int(path.stat().st_size), "file": path.name}


def _write_frames(path: Path, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """写帧（嵌套 dict/list 列规范化为 JSON 字符串，Parquet 无空 struct）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([_flatten_row(row) for row in rows])
    frame.to_parquet(path, index=False)
    return {"sha256": sha256_file(path), "rows": int(len(frame)), "file": path.name}


def _flatten_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """把嵌套容器转成规范 JSON 字符串（空 dict/list 亦安全）。"""
    out: Dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (dict, list, tuple)):
            out[str(key)] = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        else:
            out[str(key)] = value
    return out


def _write_fee_frame(path: Path, result: HostResult) -> Dict[str, Any]:
    rows = result.fee_book.payload()["by_key"]
    return _write_frames(path, rows)


def _state_matrix(params: T4Params, result: HostResult) -> Dict[str, Any]:
    envelope = result.envelope
    physical = envelope.physical
    return {
        "schema": "p7_t4_state_matrix/v1",
        "run_id": envelope.run_id,
        "generation": int(envelope.generation),
        "as_of": envelope.as_of,
        "state_digest": envelope.state_digest(),
        "envelope_digest": envelope.digest(),
        "sleeves": {
            sid: {
                "cash": float(state.cash),
                "credit": float(state.credit),
                "cash_minus_credit": float(state.equity),
                "lots": len(state.lots),
                "shares": int(sum(lot.shares for lot in state.lots)),
                "residuals": len(state.residuals),
                "borrow_lots": len(state.borrow_lots),
                "recalled_amount": float(state.recalled_amount),
                "deadlocked_amount": float(state.deadlocked_amount),
                "origin_model_versions": sorted(
                    {int(lot.origin_model_version) for lot in state.lots}
                ),
            }
            for sid, state in sorted(envelope.sleeves.items())
        },
        "physical": (
            None
            if physical is None
            else {
                "cash": float(physical.cash),
                "lots": len(physical.lots),
                "shares": int(sum(lot.shares for lot in physical.lots.values())),
                "total_fee": float(physical.total_fee),
                "total_buy_amount": float(physical.total_buy_amount),
                "total_sell_amount": float(physical.total_sell_amount),
            }
        ),
        "blocks": result.blocks,
        "voids": result.voids,
        "carries": result.carries,
        "fold_boundaries": result.fold_boundaries,
        "reserved_for_t5": list(envelope.reserved_for_t5),
        "final_envelope": envelope.to_payload(),
        "note": (
            "P7 专用状态信封（不改 P4 EngineState）；借款 lot / 召回 / 死局字段在本 task"
            " 恒为空，结构与证据位置已就位，T5 只做填充。"
        ),
    }


def _dual_caliber(params: T4Params, result: HostResult) -> Dict[str, Any]:
    from src.lazybull.v2.common.chain_metrics import calculate_chain_metrics

    per_identity: Dict[str, Dict[str, Any]] = {}
    daily: List[Dict[str, Any]] = []
    for row in result.daily_rows:
        day = str(row["trade_date"])
        per_sleeve = {sid: float(value) for sid, value in sorted(row["sleeves_cash"].items())}
        daily.append(
            {
                "trade_date": day,
                "physical_cash": float(row["physical_cash"]),
                "sleeves_cash": per_sleeve,
                "identity_ok": bool(row["identity_ok"]),
            }
        )
    nav_frames = build_nav_frames(result, params)
    for arm in params.arms:
        series = result.nav_series.get(arm, [])
        frame = pd.DataFrame(series)
        if frame.empty:
            continue
        chain = frame[["trade_date", "nav", "split_index"]].dropna(subset=["nav"])
        metrics = calculate_chain_metrics(chain.reset_index(drop=True))
        # C4-06：非正权益日被标 None 是正确的，但**不得**用 dropna 把它从指标输入里删掉——
        # 那会把无效日前后的 NAV 当成相邻有效区间，凭空算出跨越无效日的收益。
        # 口径：只要出现无效日，该袖的独立绩效整体标 invalid，保留日期与原因，不做删日修饰。
        ind_metrics, independent_status, invalid_dates = _sleeve_metrics(frame)
        per_identity[arm] = {
            "contribution_nav_metrics": metrics,
            "contribution_nav_denominator": float(params.physical_capital),
            "sleeve_nav_metrics": ind_metrics,
            "sleeve_nav_metrics_status": independent_status,
            "sleeve_nav_invalid_dates": invalid_dates,
            "sleeve_nav_policy": (
                "出现无效权益日即整袖标 invalid：**不跨该日拼接收益**、不删日修饰；"
                "展示序列仍保留无效日（C4-06）。"
            ),
            "sleeve_nav_denominator": float(params.sleeve_capital.get(arm, 0.0)),
            "denominator_note": (
                "contribution_* 的 NAV 分母 = 物理初资（I3 口径，跨袖可加）；"
                "sleeve_* 的 NAV 分母 = 袖自身初资（独立/反事实口径，**不可**与前者相加）。"
            ),
            "days": int(len(frame)),
            "invalid_equity_days": int(result.invalid_days.get(arm, 0)),
            "nav_first_last": [
                float(chain["nav"].iloc[0]) if len(chain) else None,
                float(chain["nav"].iloc[-1]) if len(chain) else None,
            ],
        }
    fee_payload = result.fee_book.payload()
    return {
        "schema": "p7_t4_dual_caliber/v1",
        "physical_initial_capital": float(params.physical_capital),
        "sleeve_capital": dict(params.sleeve_capital),
        "daily_closure": {
            "days": len(daily),
            "all_ok": bool(all(item["identity_ok"] for item in daily)),
            "failures": len(result.identity_failures),
            "failure_examples": list(result.identity_failures[:3]),
        },
        "daily": daily,
        "nav_series": {
            arm: [
                {
                    "trade_date": item["trade_date"],
                    "nav": item["nav"],
                    "independent_nav": item["independent_nav"],
                    "split_index": item["split_index"],
                }
                for item in result.nav_series.get(arm, [])
            ]
            for arm in params.arms
        },
        "chain_metrics": per_identity,
        "nav_frames_chain_metrics": nav_frames,
        "savings_sensitivity": {
            "ruling": "F26 §5.1 双口径：独立评审默认不含节省，含节省作敏感性并列",
            "savings_by_sleeve": fee_payload["savings_by_sleeve"],
            "independent_fee_total": fee_payload["independent_fee_total"],
            "actual_fee_total": fee_payload["actual_fee_running"],
            "counterfactual_basis": (
                "只读一阶反事实：E_nosave = total_value - 累计节省；"
                "不重跑成交路径（sizing 外生，但现金约束会改变成交），故只作节省敏感性上界。"
            ),
            "nonpositive_policy": "E ≤ 0 的袖标 invalid_nonpositive_equity，保留当日不删。",
        },
        "notes": {
            "I3": "I3 允许 E 为零/负的有符号贡献；只要求各分量与物理总值有限、物理初资 > 0。",
            "nav": "NAV 只经 calculate_nav；链指标只经 calculate_chain_metrics。",
        },
    }


def _identity(result: HostResult) -> Dict[str, Any]:
    envelope = result.envelope
    series = result.digest_series
    return {
        "schema": "p7_t4_identity/v1",
        "run_id": envelope.run_id,
        "generation": int(envelope.generation),
        "idempotency_keys": int(len(envelope.applied)),
        "first_key": next(iter(envelope.applied), ""),
        "last_key": next(reversed(envelope.applied), "") if envelope.applied else "",
        "replay": dict(result.replay),
        "restart": dict(result.restart),
        "digest_series": {
            "days": len(series),
            "first": series[0] if series else None,
            "middle": series[len(series) // 2] if series else None,
            "last": series[-1] if series else None,
        },
        "cross_generation_policy": "同键同代重复提交 => 无操作；跨 generation 或同键异内容 => 拒绝。",
        "restart_policy": (
            "重启检查点 = ~50% / ~90% / 末日**开始时**的状态 + 该日**结束后**的状态本体；"
            "逐个重放必须逐位一致。"
        ),
    }


def _executed(result: HostResult) -> Dict[str, Any]:
    return {
        "schema": "p7_t4_executed_scenarios/v1",
        "counts": result.scenario_counts,
        "offset_legs": int(result.offset_legs),
        "offset_shares": int(result.offset_shares),
        "market_filled": result.market_filled,
        "blocks": result.blocks,
        "voids": result.voids,
        "carries": result.carries,
        "note": (
            "执行层计数 = 实际落账的场景分布（含持仓截断、资金约束与顺延后重净额）；"
            "与 natural.json 的意图层计数不同源，禁止混报。"
        ),
    }


def _manifest(
    *,
    params: T4Params,
    run_id: str,
    run_root: Path,
    natural: Mapping[str, Any],
    cases: Sequence[Any],
    result: HostResult,
    checks: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    timings: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    outputs: Dict[str, str] = {}
    for path in sorted(run_root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            outputs[path.relative_to(run_root).as_posix()] = sha256_file(path)
    return {
        "schema": SCHEMA,
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": TASK,
        "alpha_consumed": False,
        "verdict": None,
        "code_state": {
            "git_commit": source_identity["git_commit"],
            "git_dirty": source_identity["git_dirty"],
            "source_digest": source_identity["source_digest"],
            "environment": source_identity["environment"],
        },
        "python": sys.version.split()[0],
        "params": params.params_payload(),
        "params_digest": params_digest(params),
        "inputs": {
            "t25_root": params.t25.payload(),
            "cost_settings": dict(params.cost_settings),
            # C4-03：上游登记 SHA 的**核验**结果（不是只采集）
            "input_verification": dict(params.t25.input_verification),
            # C4-03：本 run **实际全窗读入**的行情/日历内容身份（含 runoff，非只信号域），
            # 与上游冻结 digest 对账；执行前后一致性见 checks.identity.input_identity
            "market_identity": dict(params.market_identity),
        },
        "calendar": {
            "signal_domain": [params.calendar[0], params.calendar[-1]],
            "signal_days": len(params.calendar),
            "runoff_window": [params.runoff_days[0], params.runoff_days[-1]],
            "runoff_days": len(params.runoff_days),
            "extended_days": len(params.extended_days),
            "folds": [
                {
                    "split_index": split,
                    "signal_start": start,
                    "signal_end": end,
                    "model_version": model,
                }
                for split, start, end, model in params.t25.folds
            ],
        },
        "checks": dict(checks),
        "natural_observations": {
            "counts": natural["natural_counts"],
            "missing_scenarios": natural["missing_scenarios"],
            "s1_status": natural["s1_status"],
            "status": (
                "observed_with_registered_gap" if natural["missing_scenarios"] else "observed"
            ),
        },
        "constructed": scenario_mod.summarize_cases(cases)["cases"],
        "timing": dict(timings),
        "legacy_sha": {
            "file_count": params.legacy_sha["file_count"],
            "old_evidence_crosscheck_all_match": params.legacy_sha[
                "old_evidence_crosscheck_all_match"
            ],
        },
        "write_scope": {
            "root": run_root.relative_to(params.repo_root).as_posix(),
            "forbidden": [
                "data/raw",
                "data/clean",
                "data/features",
                "data/models",
                "data/runs",
                "data/paper",
                "src/lazybull",
            ],
        },
        "artifacts": {key: dict(value) for key, value in sorted(artifacts.items())},
        "outputs": outputs,
    }


def _natural_check(natural: Mapping[str, Any]) -> Dict[str, Any]:
    """自然实例门（与 T2.5 逐项一致；S1 缺口单列）。"""
    return {
        "ok": bool(natural["matches_frozen_counts"] and natural["matches_t25_precheck"]),
        "counts": natural["natural_counts"],
        "frozen_counts": natural["frozen_counts"],
        "matches_t25_precheck": natural["matches_t25_precheck"],
        "day_scenario_counts_match": natural["day_scenario_counts_match"],
        "s1_status": natural["s1_status"],
        "note": "S1 自然计数为 0 属已登记缺口，由构造黄金例覆盖，不合并计数。",
    }


def _constructed_check(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """构造例门（全过才放行；全部标 synthetic）。"""
    return {
        "ok": bool(summary["failed"] == []),
        "total": summary["total"],
        "passed": summary["passed"],
        "failed": summary["failed"],
        "synthetic_all": summary["synthetic_all"],
        "constructed_on_history": summary["constructed_on_history"],
    }


def _conservation_anchors(result: HostResult) -> Dict[str, Any]:
    """两条独立锚（复审 B-04 残余 N-6）：防「同组多意图重复归属」等记账内重分配。"""
    rows = result.conservation_rows
    offset_attributed = sum(int(row.get("offset", 0)) for row in rows)
    filled_attributed = sum(int(row.get("filled", 0)) for row in rows)
    offset_legs = int(result.offset_shares)
    filled_market = int(result.market_filled.get("buy", 0)) + int(
        result.market_filled.get("sell", 0)
    )
    return {
        "offset_attributed": offset_attributed,
        "offset_expected": 2 * offset_legs,
        "offset_ok": bool(offset_attributed == 2 * offset_legs),
        "filled_attributed": filled_attributed,
        "filled_expected": filled_market,
        "filled_ok": bool(filled_attributed == filled_market),
        "note": (
            "每次内部抵消在买卖**两袖**各记一次 offset，故逐键和应为抵消股数的 2 倍；"
            "逐键 filled 之和必须等于市场成交股数（买入+卖出）。"
        ),
    }


def _fees_check(
    fee_payload: Mapping[str, Any],
    savings: Mapping[str, float],
    result: HostResult,
) -> Dict[str, Any]:
    """费用门：逐事件 / 按袖 / 物理账本**三条独立累计通道** + 浮点闭合阈值。"""
    channels = dict(fee_payload["closure_channels"])
    channels["physical_book_vs_ledger"] = _physical_book_gap(result)
    channels["physical_book_tolerance"] = FEE_LEDGER_TOLERANCE
    return {
        "closed": bool(
            fee_payload["closed"]
            and _float_closure_ok(fee_payload["float_closure"])
            and _closure_channels_ok(channels)
        ),
        "events": fee_payload["events"],
        "physical_fee_total": fee_payload["physical_fee_total"],
        "declared_fee_total": fee_payload["declared_fee_total"],
        "declared_minus_applied": fee_payload["declared_minus_applied"],
        "actual_fee_running": fee_payload["actual_fee_running"],
        "independent_fee_total": fee_payload["independent_fee_total"],
        "savings_by_sleeve": dict(savings),
        "float_closure": fee_payload["float_closure"],
        "closure_channels": channels,
        "float_closure_tolerance": FLOAT_CLOSURE_TOLERANCE,
        "ruling": "Q3=A",
    }


def _physical_book_gap(result: HostResult) -> float:
    """第三通道：``PhysicalBook.total_fee``（按日小计累加）与费用台账（逐事件累加）之差。"""
    physical = result.envelope.physical
    if physical is None:  # pragma: no cover - 运行结束必有物理账本
        return 0.0
    return float(physical.total_fee) - float(result.fee_book.physical_fee_total)


def _closure_channels_ok(channels: Mapping[str, Any]) -> bool:
    """三条累计通道都不许破缺（逐事件 / 按袖 / 物理账本）。"""
    by_sleeve = abs(float(channels.get("by_sleeve_vs_physical", 0.0)))
    by_book = abs(float(channels.get("physical_book_vs_ledger", 0.0)))
    return (
        bool(channels.get("per_event_exact"))
        and by_sleeve <= float(channels.get("by_sleeve_tolerance", 1e-6))
        and by_book <= float(channels.get("physical_book_tolerance", 1e-6))
    )


def _float_closure_ok(closure: Mapping[str, Any]) -> bool:
    """浮点闭合阈值门（复审 N-1 / N-2）：吸收偏差必须可忽略。"""
    max_gap = float(closure.get("max_abs_ulp_gap", 0.0))
    total_gap = abs(float(closure.get("ulp_gap_total", 0.0)))
    return max_gap <= FLOAT_CLOSURE_TOLERANCE and total_gap <= FLOAT_CLOSURE_TOLERANCE


def _legacy_check(legacy: Mapping[str, Any]) -> Dict[str, Any]:
    """受审旧面判定：``src`` 面（可验证）与 ``t25`` 面（无基线）分列（复审 A-05）。"""
    files = legacy["files"]
    src_files = {key: item for key, item in files.items() if key.startswith("src/")}
    t25_files = {key: item for key, item in files.items() if key.startswith("scripts/")}
    modified = sorted(key for key, item in files.items() if item["modified_in_worktree"] is True)
    crosscheck = legacy["old_evidence_crosscheck"]
    crosscheck_ok = all(item["match"] for item in crosscheck.values()) if crosscheck else None
    return {
        "ok": bool(not modified and crosscheck_ok is not False),
        "src_ok": bool(not modified and crosscheck_ok is not False),
        "verified_files": len(src_files),
        "unverifiable_files": len(t25_files),
        "gate_scope": (
            "``ok`` 只覆盖**可验证面**（9 件 ``src/**``）；8 件 ``scripts/v2_p7/t25/**`` "
            "无逐件基线，登记为 ``not_verifiable`` 但**不冒充已核对**。"
        ),
        "file_count": legacy["file_count"],
        "verifiable": {
            "files": len(src_files),
            "crosschecked_against_old_evidence": sorted(crosscheck),
            "crosscheck_all_match": crosscheck_ok,
            "note": (
                "``src/**`` 受审面以「当前 SHA + 最近提交 + 工作区状态」登记；其中 P6.5 T5 "
                "证据登记过逐件源码 SHA 的两件（order_chain / virtual_native_engine）"
                "做了**逐字节**交叉核对。"
            ),
        },
        "not_verifiable": {
            "files": len(t25_files),
            "paths": sorted(t25_files),
            "reason": (
                "T2.5 证据根只登记工作区级 source_digest，未登记逐件源码 SHA，"
                "因此这几件**无基线可比**：本轮只登记当前 SHA 供下一代比对，"
                "不声称「未改动」。"
            ),
            "git_status": sorted({item["git_status"] or "" for item in t25_files.values()}),
        },
        "modified_in_worktree": modified,
        "untracked": sorted(key for key, item in files.items() if item["git_status"] == "??"),
        "note": "``modified_in_worktree=True`` 才算改动；``??``（未跟踪）属当前开发态。",
    }


def _s1_provenance(result: HostResult) -> Dict[str, Any]:
    """执行层 S1 逐条归属（复审 A-08 / B-11）：不得被读作自然 S1 覆盖。"""
    instances = [
        {
            "trade_date": trade["trade_date"],
            "ts_code": trade["ts_code"],
            "shares": int(trade["shares"]),
            "from_sleeve": trade.get("from_sleeve", ""),
            "to_sleeve": trade.get("to_sleeve", ""),
        }
        for trade in result.trades
        if trade.get("kind") == "internal_offset" and trade.get("scenario") == "S1"
    ]
    return {
        "count": len(instances),
        "instances": instances,
        "provenance": (
            "执行层 S1 = 顺延残差次日重净额后恰好等量的抵消；**不是**自然意图层 S1"
            "（自然计数 0），也**不是**构造例。"
        ),
    }


def _physical_cash_check(result: HostResult) -> Dict[str, Any]:
    """物理现金硬门（C4-01）：非杠杆前提的可执行性判据。

    I1 只给 ``Σ cash_i ≡ 物理现金``（**相等**），两侧一起变负仍通过；
    此处另验 ``物理现金 >= 0``（**非负**），才是「账户付得起」。
    """
    return {
        "ok": bool(result.negative_physical_cash_days == 0),
        "min_physical_cash": result.min_physical_cash,
        "negative_days": int(result.negative_physical_cash_days),
        "examples": list(result.negative_physical_cash_examples),
        "note": (
            "非杠杆前提的**可执行性**判据：I1 只给 `Σ cash_i ≡ 物理现金`（相等），"
            "两侧一起变负仍通过；此处另验 `物理现金 >= 0`（C4-01）。"
        ),
    }


def _executed_check(result: HostResult) -> Dict[str, Any]:
    """执行层场景计数与 T4 边界（informational，不设门）。"""
    return {
        "ok": "informational",
        "gate": False,
        "counts": result.scenario_counts,
        "offset_legs": result.offset_legs,
        "offset_shares": result.offset_shares,
        "market_filled": result.market_filled,
        "s1_provenance": _s1_provenance(result),
        "t4_boundaries": _t4_boundaries(result),
        "note": (
            "执行层场景计数（含持仓截断/资金约束/顺延后重净额）与自然意图层计数"
            "不同源：后者见 natural_instances，禁止混报；本块不设门（informational）。"
        ),
    }


def _t4_boundaries(result: HostResult) -> Dict[str, Any]:
    """T4 边界登记（复审 B-10）：资金约束残差按日终作废，归 T5 授权/借用。"""
    funding_shares = int(result.voids.get("insufficient_cash_voided_eod", 0))
    buy_shares = int(result.market_filled.get("buy", 0))
    return {
        "funding_no_borrow": {
            "voided_shares": funding_shares,
            "market_buy_shares": buy_shares,
            "voided_share_of_buy_intent": (
                float(funding_shares) / float(funding_shares + buy_shares)
                if (funding_shares + buy_shares)
                else None
            ),
            "policy": (
                "T4 不实现配额与借用：逐袖现金不足的部分按日终作废（不作跨日顺延），"
                "T5 将由授权 + 借用改为可成交。"
            ),
        },
        "s4_remainder_voided": {
            "voided_shares": int(result.voids.get("s4_remainder_voided_eod", 0)),
            "policy": "S4 剩余意图日终作废（Q22 保留于 F26）。",
        },
        "exit_schedule_from_plan": {
            "policy": (
                "退出由 T2.5 计划行的 due_date 驱动（冻结 20 日持期，本 task 不改）；"
                "延迟成交的买入不再重排到期（重排即改 20 日到期口径，属 §2 非目标）。"
            ),
            "note": "抵消承接的 lot 保留原 due_date（lot 谱系不重置）。",
        },
        "transfer_cash_cap": {
            "capped_shares": int(result.offset_capped_shares),
            "offset_shares": int(result.offset_shares),
            "policy": (
                "内部转账必须由**买方袖自有现金**支撑（C4-01 / 口径 a）：撑不住的部分不打内部抵消，"
                "残差留在意图上按既有资金约束语义顺延。"
            ),
        },
        "sleeve_cash_sign": {
            "min_sleeve_cash": result.min_sleeve_cash,
            "negative_cash_days": int(result.negative_cash_days),
            "min_physical_cash": result.min_physical_cash,
            "negative_physical_cash_days": int(result.negative_physical_cash_days),
            "examples": list(result.negative_cash_examples),
            "policy": (
                "I1 只约束 ``Σ_i cash_i ≡ 物理现金``（已逐日校验）；**逐袖** ``cash_i >= 0`` "
                "属 T5 的授权/借用门。T4 无配额/无 credit，净额化的内部转账是无 credit 背书的"
                "虚拟付款，故逐袖现金可能为负（实测为**连续挂账**而非日内瞬态）——"
                "此处登记不拒判，T5 由授权+借用消除。"
            ),
        },
        "equity_nonpositive_policy": {
            "events": len(result.equity_policy_events),
            "detail": list(result.equity_policy_events),
            "policy": "``E_i <= 0`` 的袖禁新增买入、不被动卖出（F26）；本 run 未触发。",
        },
        "no_credit_no_quota": "``credit`` 恒 0、``borrow_lots`` 为空；召回/死局字段仅占位。",
    }
