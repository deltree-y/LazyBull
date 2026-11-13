# -*- coding: utf-8 -*-
"""P7 T5 编排器：机械互证（T4 兼容门）→ VV-H → VV-S → 证据落盘。

写域
====
只写 ``data/paper_shadow/p7_testbed/<t5_run_id>/``（``scripts.v2_p7.common`` 写域守卫，
``mkdir`` 前整体预检，拒绝路径零副作用）；不写 ``data/raw|clean|features|models|runs|paper``，
不覆盖任何完成根（``resume`` 遇 ``manifest.json`` 必须拒绝）。

硬门（不通过即停；已通过项按 ``checks`` 登记）
============================================
1. **T4 兼容门**：``credit=off & quota=off`` 与 T4 完成根逐位一致（逐日账本投影行、
   费用台账、实际/独立 NAV、守恒锚、终态摘要）；
2. **配额门**：``R ∈ [0, D]``、整手、cap-binding 诊断齐全、``A ≤ min(D, R+L)``；
3. **授权门**：``A + Fee(A) ≤ cash_open + L``，``credit_outstanding`` 不参与授权；
4. **借用门**：``Σ credit = 0`` 逐日精确、本金逐笔对消、本金转移零 P&L；
5. **非负门**：``cash_i + credit_i ≥ 0`` 逐日 0 违例（VV-H 与 VV-S 都要过）；
6. **恒等式门**：I1/I2/I3 逐日闭合（含 credit），非正 E 不删日；
7. **构造例门**：黄金例/负例全过（``synthetic`` / ``constructed_on_history`` 分列）；
8. **身份门**：重放与 3 类 credit 检查点续跑逐位一致；旧受审面 SHA 未变。

``alpha_consumed=false``：本 task 不产生任何 pass/fail 经济判定。
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd
from loguru import logger

from scripts.v2_p7.common import (
    REPO_ROOT,
    assert_targets_within_write_roots,
    resolve_write_roots,
    sha256_file,
    write_json,
)
from scripts.v2_p7.t4.host import build_intent_index
from scripts.v2_p7.t4.price import PriceBook
from scripts.v2_p7.t4.runner import _flatten_row
from scripts.v2_p7.t5 import scenarios as scenario_mod
from scripts.v2_p7.t5.artifacts import write_artifacts
from scripts.v2_p7.t5.checks import _acceptance_checks, _stop_on_hard_failures
from scripts.v2_p7.t5.errors import RunnerError
from scripts.v2_p7.t5.host import (
    MODE_T4_PARITY,
    MODE_VVH,
    MODE_VVS,
    QuotaCreditHost,
    T5HostResult,
)
from scripts.v2_p7.t5.params import (
    DEFAULT_RUN_ID,
    TESTBED_REL,
    T5Params,
    build_params,
    params_digest,
)

SCHEMA = "p7_t5/v1"
TASK = "P7 T5 — 原始需求、配额、日内净额、借用会计与死局"


def run(
    repo_root: Path = REPO_ROOT,
    run_id: str = DEFAULT_RUN_ID,
    resume: bool = False,
    arms: Sequence[str] = ("parity", "vvh", "vvs"),
    windows: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """执行 T5 全流程（默认全窗三臂：T4 互证 / VV-H / VV-S）。"""
    started = time.time()
    params = build_params(repo_root)
    run_root = Path(repo_root) / TESTBED_REL / run_id
    _prepare_run_root(run_root, resume=resume)
    source_identity = _source_identity(params)
    cost_model = _cost_model(params)
    days = tuple(params.extended_days)
    if windows:
        days = tuple(day for day in days if str(windows[0]) <= day <= str(windows[1]))
        if not days:
            raise RunnerError(f"子窗口内无交易日：{windows}")

    scenario_started = time.time()
    cases = scenario_mod.run_constructed_cases(cost_model=cost_model)
    failed_cases = [case.name for case in cases if not case.passed]
    if failed_cases:
        raise RunnerError(f"构造例未通过：{failed_cases}")
    scenario_seconds = round(time.time() - scenario_started, 1)

    host_started = time.time()
    buy_index, sell_index = build_intent_index(params.t25.root, params.arms)
    results: Dict[str, T5HostResult] = {}
    for name, mode in (("parity", MODE_T4_PARITY), ("vvh", MODE_VVH), ("vvs", MODE_VVS)):
        if name not in arms:
            continue
        results[name] = _run_arm(
            params=params,
            days=days,
            run_id=f"{run_id}:{name}",
            mode=mode,
            cost_model=cost_model,
            buy_index=buy_index,
            sell_index=sell_index,
        )
    host_seconds = round(time.time() - host_started, 1)

    parity = (
        _t4_parity(params, results["parity"], segment=bool(windows))
        if "parity" in results
        else {"ok": False, "skipped": True}
    )
    if "vvh" in results and not windows:
        reference = _vvh_reference(params, days, results["vvh"], cost_model, buy_index, sell_index)
    else:
        reference = {
            "ok": True,
            "skipped": True,
            "reason": ("未运行 VV-H 臂或为分段验证子窗口：参照互证只在全窗三臂同跑时判定"),
        }
    checks = _acceptance_checks(
        params=params,
        results=results,
        cases=cases,
        parity=parity,
        reference=reference,
        segment=bool(windows),
    )
    # 运行后重采身份：代码/输入/行情未显式接受的漂移必须阻断发布（C-T5-08）
    drift = _identity_drift(params=params, before=source_identity)
    if drift:
        raise RunnerError(f"运行期身份漂移（未显式接受）→ 阻断发布：{drift}")
    _stop_on_hard_failures(checks)

    artifacts_started = time.time()
    written = write_artifacts(
        params=params,
        run_root=run_root,
        results=results,
        cases=cases,
        parity=parity,
        reference=reference,
    )
    artifacts_seconds = round(time.time() - artifacts_started, 1)

    manifest = _manifest(
        params=params,
        run_id=run_id,
        run_root=run_root,
        results=results,
        cases=cases,
        checks=checks,
        source_identity=source_identity,
        parity=parity,
        reference=reference,
        timings={
            "scenarios_seconds": scenario_seconds,
            "host_seconds": host_seconds,
            "artifacts_seconds": artifacts_seconds,
            "total_seconds": round(time.time() - started, 1),
        },
        artifacts=written,
        days=days,
    )
    write_json(run_root / "manifest.json", manifest)
    logger.info(f"T5 证据已落盘：{run_root}（{manifest['timing']['total_seconds']}s）")
    return manifest


def _run_arm(
    *,
    params: T5Params,
    days: Sequence[str],
    run_id: str,
    mode: Any,
    cost_model: Any,
    buy_index: Mapping[str, Any],
    sell_index: Mapping[str, Any],
) -> T5HostResult:
    """单臂全窗运行（每臂独立价源，防跨臂污染）。"""
    from src.lazybull.data.loader import DataLoader
    from src.lazybull.data.storage import Storage

    def _prices() -> PriceBook:
        return PriceBook(DataLoader(Storage(root_path=str(params.data_root))))

    host = QuotaCreditHost(
        params=params,
        run_id=run_id,
        prices=_prices(),
        cost_model=cost_model,
        buy_index=buy_index,
        sell_index=sell_index,
        days=tuple(days),
        price_factory=_prices,
        mode=mode,
    )
    return host.run()


def _cost_model(params: T5Params) -> Any:
    from src.lazybull.v2.common.cost import CostModel

    return CostModel(
        commission_rate=params.cost_settings["commission_rate"],
        min_commission=params.cost_settings["min_commission"],
        stamp_tax=params.cost_settings["stamp_tax"],
        slippage=params.cost_settings["slippage"],
    )


def _source_identity(params: T5Params) -> Dict[str, Any]:
    from src.lazybull.v2.store.store_identity import workspace_identity

    return workspace_identity(params.repo_root)


def _identity_drift(*, params: T5Params, before: Mapping[str, Any]) -> List[str]:
    """开工身份 vs 发布前重采身份（排除时间类易变字段），返回漂移明细。"""
    after = _source_identity(params)
    volatile = ("time", "mtime", "now", "as_of", "seconds", "elapsed")
    drift: List[str] = []
    for key in sorted(set(before) | set(after)):
        if any(token in str(key).lower() for token in volatile):
            continue
        if before.get(key) != after.get(key):
            drift.append(f"identity.{key}: {before.get(key)!r} -> {after.get(key)!r}")
    rebuilt = build_params(params.repo_root)
    if params_digest(params) != params_digest(rebuilt):
        drift.append("params_digest 在运行期变化")
    return drift


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


# ────────────────────────── T4 兼容门 ──────────────────────────


def _t4_parity(params: T5Params, result: T5HostResult, *, segment: bool = False) -> Dict[str, Any]:
    """``credit=off & quota=off`` 与 T4 完成根逐位互证（逐日 + 终态 + 双口径 + 费用）。

    ``segment=True``（分段验证子窗口）时互证**不成立亦不判失败**：子窗口与 r31 的
    逐日序列长度不同，只有全窗才能逐位比对；此时期望值登记为 ``skipped``。
    """
    if segment:
        return {
            "schema": "p7_t5_t4_parity/v1",
            "ok": True,
            "skipped": True,
            "reason": "分段验证子窗口：逐日比对只对全窗成立，本 run 不作互证结论",
            "reference_root": str(
                params.t4_reference.root.relative_to(params.repo_root).as_posix()
            ),
            "days": len(result.days),
        }
    root = params.t4_reference.root
    reference_daily = _frame_rows(root / "daily.parquet")
    reference_trades = _frame_rows(root / "trades.parquet")
    reference_conservation = _frame_rows(root / "conservation.parquet")
    reference_fees = _frame_rows(root / "fees_by_key.parquet")
    reference_dual = _read_json(root / "dual_caliber.json")
    reference_state = _read_json(root / "state_matrix.json")

    # 我方行必须按 T4 **同一式**拍平（嵌套容器 -> 规范 JSON 字符串），否则与落盘件不可比
    ours_daily = [_flatten_row(row) for row in result.daily_rows]
    ours_trades = [_flatten_row(row) for row in result.trades]
    ours_conservation = [_flatten_row(row) for row in result.conservation_rows]
    ours_fees = [_flatten_row(row) for row in result.fee_book.payload()["by_key"]]

    daily_mismatch = _diff_rows(reference_daily, ours_daily)
    trades_mismatch = _diff_rows(reference_trades, ours_trades)
    conservation_mismatch = _diff_rows(reference_conservation, ours_conservation)
    fee_mismatch = _diff_rows(reference_fees, ours_fees)
    nav_mismatch = _nav_mismatch(reference_dual, result)
    state_ok = reference_state.get("state_digest") == result.envelope.state_digest()
    reference_conservation_summary = reference_state.get("blocks") or {}
    del reference_conservation_summary
    return {
        "schema": "p7_t5_t4_parity/v1",
        "reference_root": str(root.relative_to(params.repo_root).as_posix()),
        "reference_artifacts": dict(sorted(params.t4_reference.artifacts.items())),
        "days": len(result.days),
        "daily": {
            "rows": len(ours_daily),
            "equal": int(len(ours_daily) - len(daily_mismatch)),
            "mismatch": daily_mismatch[:5],
            "mismatch_count": len(daily_mismatch),
        },
        "trades": {
            "rows": len(ours_trades),
            "equal": int(len(ours_trades) - len(trades_mismatch)),
            "mismatch": trades_mismatch[:5],
            "mismatch_count": len(trades_mismatch),
        },
        "conservation": {
            "rows": len(ours_conservation),
            "equal": int(len(ours_conservation) - len(conservation_mismatch)),
            "mismatch": conservation_mismatch[:5],
            "mismatch_count": len(conservation_mismatch),
        },
        "fees_by_key": {
            "rows": len(ours_fees),
            "equal": int(len(ours_fees) - len(fee_mismatch)),
            "mismatch": fee_mismatch[:5],
            "mismatch_count": len(fee_mismatch),
        },
        "nav": {"equal": not nav_mismatch, "mismatch": nav_mismatch[:5]},
        "final_state_digest": {
            "reference": reference_state.get("state_digest"),
            "ours": result.envelope.state_digest(),
            "equal": bool(state_ok),
        },
        "credit_inert": {
            "credit_all_zero": bool(
                all(abs(float(row["credit"])) == 0.0 for row in result.balance_rows)
            ),
            "borrow_lots": int(result.credit_payload["totals"]["lots"]),
            "quota_rows": len(result.quota_rows),
        },
        "ok": bool(
            not daily_mismatch
            and not trades_mismatch
            and not conservation_mismatch
            and not fee_mismatch
            and not nav_mismatch
            and state_ok
            and all(abs(float(row["credit"])) == 0.0 for row in result.balance_rows)
            and int(result.credit_payload["totals"]["lots"]) == 0
        ),
        "rule": (
            "逐日账本投影行 / 交易 / 守恒行 / 费用逐键 / 实际与独立 NAV / 终态摘要全部相等，"
            "且 credit 恒 0、无借款 lot、无配额决策 ⇒ 视为逐位一致"
        ),
    }


def _vvh_reference(
    params: T5Params,
    days: Sequence[str],
    result: T5HostResult,
    cost_model: Any,
    buy_index: Mapping[str, Any],
    sell_index: Mapping[str, Any],
) -> Dict[str, Any]:
    """VV-H 与「同配额、禁 credit」参照逐位一致（独立宿主 + 独立价源重跑）。"""
    reference = _run_arm(
        params=params,
        days=days,
        run_id="vvh-reference",
        mode=MODE_VVH,
        cost_model=cost_model,
        buy_index=buy_index,
        sell_index=sell_index,
    )
    return {
        "schema": "p7_t5_vvh_reference/v1",
        "days": len(result.days),
        "digest_equal": [a[1] for a in result.digest_series]
        == [b[1] for b in reference.digest_series],
        "nav_equal": result.nav_series == reference.nav_series,
        "trades_equal": result.trades == reference.trades,
        "conservation_equal": result.conservation_rows == reference.conservation_rows,
        "fee_equal": [a[1] for a in result.fee_series] == [b[1] for b in reference.fee_series],
        "quota_equal": result.quota_rows == reference.quota_rows,
        "credit_all_zero": all(abs(float(row["credit"])) == 0.0 for row in result.balance_rows),
        "borrow_lots": int(result.credit_payload["totals"]["lots"]),
        "requests_without_grant": bool(
            all(
                float(item.get("granted") or 0.0) == 0.0
                for item in result.credit_payload["requests"]
            )
        ),
        "ok": bool(
            [a[1] for a in result.digest_series] == [b[1] for b in reference.digest_series]
            and result.nav_series == reference.nav_series
            and result.trades == reference.trades
            and result.conservation_rows == reference.conservation_rows
            and [a[1] for a in result.fee_series] == [b[1] for b in reference.fee_series]
            and int(result.credit_payload["totals"]["lots"]) == 0
        ),
        "rule": "VV-H = 同配额 + 禁 credit；credit 必须完全惰性（无 lot、无授信）",
    }


def _frame_rows(path: Path) -> List[Dict[str, Any]]:
    frame = pd.read_parquet(path)
    return [dict(row) for row in frame.to_dict(orient="records")]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RunnerError(f"冻结输入缺件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _diff_rows(
    reference: Sequence[Mapping[str, Any]], ours: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    """**多重集**逐行比对（不按 key 折叠）：返回不一致项（最多 20 条）。

    账本投影行/交易/守恒行/费用逐键都可能有同键多行（例如同一日同一意图的
    「意图行 + 截断行」），按键折叠会漏判或误判，故直接用规范 JSON 多重集比较。
    """
    left = sorted(_canonical_row(row) for row in reference)
    right = sorted(_canonical_row(row) for row in ours)
    mismatched: List[Dict[str, Any]] = []
    for pos in range(max(len(left), len(right))):
        item_left = left[pos] if pos < len(left) else None
        item_right = right[pos] if pos < len(right) else None
        if item_left != item_right:
            mismatched.append({"pos": pos, "reference": item_left, "ours": item_right})
        if len(mismatched) >= 20:
            break
    return mismatched


def _canonical_row(row: Mapping[str, Any]) -> str:
    return json.dumps(_normalize_row(row), ensure_ascii=False, sort_keys=True, default=str)


def _normalize_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """归一化：numpy 标量转原生；**缺值与 NaN 等价**（Parquet 往返会补 NaN 键）。

    落盘件由 ``pd.DataFrame`` 写出，列并集会让「某些行没有的键」以 ``NaN`` 出现；
    因此比较前必须把这些键按**不存在**处理，否则会把「字段缺席」误判为差异。
    """
    out: Dict[str, Any] = {}
    for name, value in sorted(row.items()):
        if hasattr(value, "item") and not isinstance(value, (str, bytes)):
            try:
                value = value.item()
            except (TypeError, ValueError):  # pragma: no cover - 非标量容器
                pass
        if value is None:
            continue
        if isinstance(value, float):
            if math.isnan(value):
                continue
            # Parquet 混合列（多数行缺值）会整体升为 float（``3500`` -> ``3500.0``）；
            # 跨存储比较必须把**整值浮点**与整数视作相等，否则会把存储类型差异当口径差异。
            out[str(name)] = int(value) if value.is_integer() else float(value)
            continue
        out[str(name)] = value
    return out


def _nav_mismatch(reference: Mapping[str, Any], result: T5HostResult) -> List[Dict[str, Any]]:
    mismatched: List[Dict[str, Any]] = []
    reference_series = reference.get("nav_series") or {}
    for arm, series in sorted(reference_series.items()):
        ours = result.nav_series.get(arm) or []
        reference_rows = [
            {
                "trade_date": str(item["trade_date"]),
                "nav": item.get("nav"),
                "independent_nav": item.get("independent_nav"),
                "split_index": int(item.get("split_index") or 0),
            }
            for item in series
        ]
        ours_rows = [
            {
                "trade_date": str(item["trade_date"]),
                "nav": item.get("nav"),
                "independent_nav": item.get("independent_nav"),
                "split_index": int(item.get("split_index") or 0),
            }
            for item in ours
        ]
        if reference_rows != ours_rows:
            first = next(
                (
                    {"pos": pos, "reference": reference_rows[pos], "ours": ours_rows[pos]}
                    for pos in range(min(len(reference_rows), len(ours_rows)))
                    if reference_rows[pos] != ours_rows[pos]
                ),
                {"rows": [len(reference_rows), len(ours_rows)]},
            )
            mismatched.append({"arm": arm, **first})
    return mismatched


def _manifest(
    *,
    params: T5Params,
    run_id: str,
    run_root: Path,
    results: Mapping[str, T5HostResult],
    cases: Sequence[Any],
    checks: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    parity: Mapping[str, Any],
    reference: Mapping[str, Any],
    timings: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    days: Sequence[str],
) -> Dict[str, Any]:
    outputs: Dict[str, str] = {}
    for path in sorted(run_root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            outputs[path.relative_to(run_root).as_posix()] = sha256_file(path)
    summary = scenario_mod.summarize_cases(cases)
    arms: Dict[str, Any] = {}
    for name, result in sorted(results.items()):
        arms[name] = {
            "mode": dict(result.mode),
            "days": len(result.days),
            "window": [result.days[0], result.days[-1]],
            "generation": int(result.envelope.generation),
            "state_digest": result.envelope.state_digest(),
            "scenarios": result.scenario_counts,
            "market_filled": result.market_filled,
            "voids": result.voids,
            "blocks": result.blocks,
            "carries": result.carries,
            "credit_totals": result.credit_payload["totals"],
            "credit_by_sleeve": result.credit_balance["credit_by_sleeve"],
            "gate": {
                "ok": result.gate_payload["ok"],
                "violation_days": result.gate_payload["violation_days"],
                "min_cash_plus_credit": result.gate_payload["min_cash_plus_credit"],
                "per_sleeve_min": result.gate_payload["per_sleeve_min"],
            },
            "identity_failure_days": len(result.identity_failures),
            "invalid_equity_days": dict(result.invalid_days),
            "min_physical_cash": result.min_physical_cash,
            "negative_physical_cash_days": int(result.negative_physical_cash_days),
            "min_sleeve_cash": result.min_sleeve_cash,
            "negative_sleeve_cash_days": int(result.negative_cash_days),
            "equity_blocks": len(result.equity_blocks),
            "conservation_violations": int(result.conservation["violations"]),
            "cross_day_violations": int(result.cross_day["violations"]),
        }
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
            "t4_reference": params.t4_reference.payload(),
            "cost_settings": dict(params.cost_settings),
            "market_identity": dict(params.market_identity),
            "target_notional_days": len(params.account_target),
        },
        "calendar": {
            "window": [str(days[0]), str(days[-1])],
            "days": len(days),
            "signal_days": len(params.calendar),
            "runoff_days": len(params.runoff_days),
        },
        "arms": arms,
        "checks": dict(checks),
        "constructed": summary["cases"],
        "constructed_summary": {
            "total": summary["total"],
            "passed": summary["passed"],
            "failed": summary["failed"],
            "synthetic": len(summary["synthetic"]),
            "constructed_on_history": len(summary["constructed_on_history"]),
        },
        "t4_parity": {
            "ok": bool(parity.get("ok")),
            "skipped": bool(parity.get("skipped")),
            "reason": str(parity.get("reason") or ""),
            "days": parity.get("days"),
            "daily_mismatch_count": (parity.get("daily") or {}).get("mismatch_count"),
            "trade_mismatch_count": (parity.get("trades") or {}).get("mismatch_count"),
            "conservation_mismatch_count": (parity.get("conservation") or {}).get("mismatch_count"),
            "fee_mismatch_count": (parity.get("fees_by_key") or {}).get("mismatch_count"),
            "state_digest_equal": (parity.get("final_state_digest") or {}).get("equal"),
            "vvh_reference_ok": bool(reference.get("ok")),
            "vvh_reference_skipped": bool(reference.get("skipped")),
        },
        "timing": dict(timings),
        "write_scope": {
            "root": run_root.relative_to(params.repo_root).as_posix(),
            "forbidden": [
                "data/raw",
                "data/clean",
                "data/features",
                "data/models",
                "data/runs",
                "data/paper",
                "data/frozen_reference",
                "src/lazybull",
            ],
        },
        "artifacts": {key: dict(value) for key, value in sorted(artifacts.items())},
        "outputs": outputs,
    }
