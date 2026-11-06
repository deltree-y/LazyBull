# -*- coding: utf-8 -*-
"""驾驶舱 projection 装配（设计 §2.3 / §5.2；渲染归 T2）。

单向数据流：受信根 → 私有稳定快照 → 既有纯函数 → 白名单 projection。本模块**只读副本**，
不取锁、不修复、不 ensure、不构造 Registry/Storage/Calendar，也不产生任何指标新实现：
账户投影走 ``build_account_projection``，年化走 ``calculate_annualized_return``，
代形态/NAV/成交核对走 ``authority`` 模块的纯函数。

降级纪律：每区独立给状态行（``report_status/degradation_reason_code``），缺文件与合法空严格
区分；任何冲突一律 ``invalid`` 且零修复。
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

from src.lazybull.v2.core.accounting.engine_state import (
    EngineState,
    EngineStateError,
    loads_blob,
    state_from_blob,
)
from src.lazybull.v2.hosts.paper.state_store import (
    PaperStateError,
    PaperStateStore,
    build_account_projection,
)
from src.lazybull.v2.reporting import authority, config_view, instructions, quality, status
from src.lazybull.v2.reporting.snapshot import StableSnapshot

__all__ = [
    "PAPER_FIXED_FILES",
    "PAPER_ROOT_ID",
    "PROJECTION_SCHEMA",
    "QUALITY_ROOT_ID",
    "build_dashboard_projection",
    "capture_paper_inputs",
    "collect_degradations",
    "open_copy_store",
]

#: projection schema（进入 manifest ``projection_sha256`` 的载荷版本）
PROJECTION_SCHEMA = "p6_dashboard_projection/v1"
#: 纸面生产根标识（manifest ``root_map`` 只用标识与语义，不写绝对路径）
PAPER_ROOT_ID = "paper"
#: 质量来源根标识
QUALITY_ROOT_ID = "quality"
#: 每次投影必读的固定文件（其余按权威代动态补采）
PAPER_FIXED_FILES: tuple[str, ...] = (
    "state/engine_state.bin",
    "config.yaml",
    "state/account.json",
    "nav/nav.parquet",
    "trades/trades.parquet",
    "runs/rebalance_state.json",
    "pending_buys/pending_buys.json",
    "pending_sells/pending_sells.json",
)
_INSTRUCTIONS_DIR = "instructions"
#: 权威 blob 反序列化/形态校验的全部可预期异常（损坏字节一律降级，不抛裸异常）
_BLOB_ERRORS = (
    pickle.UnpicklingError,
    EOFError,
    ValueError,
    TypeError,
    AttributeError,
    EngineStateError,
)


@dataclass
class CockpitContext:
    """一次投影的只读上下文（全部来自同一次稳定副本）。"""

    snapshot: StableSnapshot
    store: PaperStateStore
    state: EngineState
    day: str
    run_kind: str
    record: Mapping[str, Any] | None
    record_row: dict[str, Any]
    instruction_names: list[str]
    degradations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int | None]:
        """run record 计数字段（**缺键与 0 严格区分**）。"""
        return instructions.record_counts(self.record)

    @property
    def pre_switch_shape(self) -> bool:
        """切换前记录形态（同时缺 run_kind 与 engine ⇒ 账户允许 degraded）。"""
        return self.record_row.get("degradation_reason_code") == "run_record_pre_switch_shape"


def open_copy_store(snapshot: StableSnapshot, *, root_id: str = PAPER_ROOT_ID) -> PaperStateStore:
    """在**副本**上构造只读 Store（``create_layout=False``，绝不建目录）。"""
    return PaperStateStore(root_path=str(snapshot.root_copy(root_id)), create_layout=False)


def capture_paper_inputs(
    snapshot: StableSnapshot, *, root_id: str = PAPER_ROOT_ID
) -> dict[str, Any]:
    """采集回调：固定文件 + 指令目录枚举 + 按权威代补采 run record 与未来指令文件。

    只对**未来**执行日指令文件读字节（历史 ``E ≤ D`` 文件仅参与文件名清单，不推断未来）。
    """
    for relative in PAPER_FIXED_FILES:
        snapshot.capture(root_id, relative)
    names = snapshot.list_dir(root_id, _INSTRUCTIONS_DIR, suffix=".parquet")
    day, run_kind = _peek_generation(snapshot, root_id)
    if day and run_kind:
        snapshot.capture(root_id, f"runs/{run_kind}_{day}.json")
        future, _ = instructions.select_future_instruction_files(names, current_date=day)
        for execution_day in future:
            snapshot.capture(root_id, f"{_INSTRUCTIONS_DIR}/{execution_day}.parquet")
    return {"instructions": names, "authority_date": day, "run_kind": run_kind}


def _peek_generation(snapshot: StableSnapshot, root_id: str) -> tuple[str | None, str | None]:
    """从副本字节窥探权威代（损坏/缺失返回 (None, None)，不抛异常）。"""
    raw = snapshot.read_bytes(root_id, "state/engine_state.bin")
    if raw is None:
        return None, None
    try:
        blob = loads_blob(raw)
    except _BLOB_ERRORS:
        return None, None
    generation = blob.get("paper_generation")
    if not isinstance(generation, Mapping):
        return None, None
    return authority.normalize_date(generation.get("trade_date")), _text(generation.get("run_kind"))


def _text(value: Any) -> str | None:
    """非空文本化（空白视为缺省）。"""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_authority(
    snapshot: StableSnapshot, *, root_id: str = PAPER_ROOT_ID
) -> tuple[dict[str, Any], EngineState | None, dict[str, Any] | None]:
    """读权威 blob 并核对代/相位/最后处理日一致；返回 ``(状态行, 状态对象, 代信息)``。"""
    raw = snapshot.read_bytes(root_id, "state/engine_state.bin")
    if raw is None:
        return (
            status.make_status(
                "not_available", reason_code="authority_missing", detail="缺 state/engine_state.bin"
            ),
            None,
            None,
        )
    try:
        blob = loads_blob(raw)
        state = state_from_blob(blob)
    except _BLOB_ERRORS as exc:
        return (
            status.make_status(
                "invalid",
                reason_code="authority_unreadable",
                detail=f"{type(exc).__name__}: {exc}",
            ),
            None,
            None,
        )
    generation = blob.get("paper_generation")
    if not isinstance(generation, Mapping):
        return (
            status.make_status(
                "invalid", reason_code="authority_unreadable", detail="缺 paper_generation 顶层键"
            ),
            None,
            None,
        )
    day = authority.normalize_date(generation.get("trade_date"))
    run_kind = _text(generation.get("run_kind"))
    if day is None or run_kind not in ("t0", "t1"):
        return (
            status.make_status(
                "invalid",
                reason_code="authority_unreadable",
                detail=f"paper_generation 非法：{dict(generation)}",
            ),
            None,
            None,
        )
    last = (
        authority.normalize_date(state.portfolio_values[-1].get("date"))
        if state.portfolio_values
        else None
    )
    if last != day:
        return (
            status.make_status(
                "invalid",
                reason_code="authority_mismatch",
                detail=f"paper_generation 日 {day} 与 portfolio_values 末日 {last} 不一致",
            ),
            None,
            None,
        )
    row = status.make_status(
        "available",
        detail=f"权威代 {run_kind}@{day}",
        verification_level="authority_verified",
        extra={"trade_date": day, "run_kind": run_kind},
    )
    return row, state, {"trade_date": day, "run_kind": run_kind}


def _instruction_names(
    snapshot: StableSnapshot, info: Mapping[str, Any] | None
) -> list[str]:
    """指令文件名清单（优先取采集结果，其次取已登记的目录枚举，绝不额外读原根）。"""
    if info and info.get("instructions") is not None:
        return [str(name) for name in info["instructions"]]
    entry = snapshot.listings.get(f"{PAPER_ROOT_ID}:{_INSTRUCTIONS_DIR}")
    return list(entry[1]) if entry else []


def _load_record(
    snapshot: StableSnapshot, *, run_kind: str, day: str
) -> tuple[Mapping[str, Any] | None, dict[str, Any]]:
    """读当前代 run record 并判定形态（坏 JSON/缺失都落到状态行）。"""
    relative = f"runs/{run_kind}_{day}.json"
    if not snapshot.exists(PAPER_ROOT_ID, relative):
        return None, authority.classify_run_record(None, run_kind=run_kind, trade_date=day)
    payload, parse_error = snapshot.read_json(PAPER_ROOT_ID, relative)
    if parse_error is not None:
        return None, status.make_status(
            "invalid", reason_code=parse_error, detail=f"{relative} JSON 损坏"
        )
    return payload, authority.classify_run_record(payload, run_kind=run_kind, trade_date=day)


def _build_context(
    snapshot: StableSnapshot,
    store: PaperStateStore,
    state: EngineState,
    generation: Mapping[str, Any],
    info: Mapping[str, Any] | None,
) -> CockpitContext:
    """装配只读上下文（record 形态 + 指令清单一次取定）。"""
    day = str(generation["trade_date"])
    run_kind = str(generation["run_kind"])
    record, record_row = _load_record(snapshot, run_kind=run_kind, day=day)
    return CockpitContext(
        snapshot=snapshot,
        store=store,
        state=state,
        day=day,
        run_kind=run_kind,
        record=record,
        record_row=record_row,
        instruction_names=_instruction_names(snapshot, info),
    )


def build_dashboard_projection(
    *,
    snapshot: StableSnapshot,
    generated_at: str,
    evidence_class: str = "production",
    capture_info: Mapping[str, Any] | None = None,
    quality_rows: Sequence[Mapping[str, Any]] | None = None,
    quality_present: bool = False,
) -> dict[str, Any]:
    """装配驾驶舱 projection（无 HTML/CSV；渲染归 T2）。"""
    if evidence_class not in ("production", "research", "calibration_only"):
        raise ValueError(f"未知 evidence_class：{evidence_class!r}")
    store = open_copy_store(snapshot)
    authority_row, state, generation = load_authority(snapshot)
    projection: dict[str, Any] = {
        "schema": PROJECTION_SCHEMA,
        "generated_at": generated_at,
        "evidence_class": evidence_class,
        "root_map": {PAPER_ROOT_ID: "v2 纸面生产根（只读副本）"},
        "authority": authority_row,
        "quality": quality.project_quality(rows=quality_rows, present=quality_present),
    }
    if state is None or generation is None:
        projection.update(_unavailable_sections(authority_row))
        projection["degradations"] = collect_degradations(
            [status.degradation_entry(authority_row, field_path="authority")]
            + [
                status.degradation_entry(row, field_path=path)
                for path, row in _unavailable_entries(authority_row)
            ]
        )
        return projection
    context = _build_context(snapshot, store, state, generation, capture_info)
    projection.update(project_sections(context))
    projection["degradations"] = collect_degradations(context.degradations)
    return projection


#: 权威不可读时仍需存在的区键（保证 projection 键集恒定，渲染侧不分支）
SECTION_KEYS: tuple[str, ...] = (
    "record",
    "account",
    "nav",
    "trades",
    "instructions",
    "pending",
    "rebalance",
    "config",
)


def _unavailable_row(authority_row: Mapping[str, Any]) -> dict[str, Any]:
    """权威不可用时各区的统一 ``not_available`` 状态行（沿用权威侧原因码）。"""
    return status.make_status(
        "not_available",
        reason_code=str(authority_row.get("degradation_reason_code") or "authority_missing"),
        detail="权威 engine_state.bin 不可用，本区无法投影",
    )


def _unavailable_entries(authority_row: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """逐区登记项（``pending`` 拆 buys/sells 两路）。"""
    entries: list[tuple[str, dict[str, Any]]] = []
    for key in SECTION_KEYS:
        if key == "pending":
            entries.append(("pending.buys", _unavailable_row(authority_row)))
            entries.append(("pending.sells", _unavailable_row(authority_row)))
            continue
        entries.append((key, _unavailable_row(authority_row)))
    return entries


def _unavailable_sections(authority_row: Mapping[str, Any]) -> dict[str, Any]:
    """权威不可用时的恒定区结构（各区独立状态行，禁止整段缺键）。"""
    sections: dict[str, Any] = {}
    for path, row in _unavailable_entries(authority_row):
        if "." in path:
            parent, child = path.split(".", 1)
            sections.setdefault(parent, {})[child] = dict(row)
        else:
            sections[path] = dict(row)
    sections["nav"].update(
        {
            "annualization_allowed": False,
            "nav_date": None,
            "nav_date_status": "missing",
            "rows": [],
            "verified_rows": [],
            "broken_links": [],
            "nav_only_days": [],
        }
    )
    # 子键集也须恒定：T2 按 ``config.annualization.report_status`` 取字段时不得 KeyError
    sections["config"].update(
        {
            "model": _unavailable_row(authority_row),
            "annualization": _unavailable_row(authority_row),
        }
    )
    return sections


def _generation_gate(context: CockpitContext) -> dict[str, Any] | None:
    """完整代资格：record 非可用时向各依赖区传递资格（R2-20261106-02）。

    只有两类需要传递：① 切换前形态（缺 marker）⇒ 依赖区 ``degraded``（值可展示但不认证）；
    ② record ``invalid``（缺 marker / 只缺一项 / 坏 JSON）⇒ 依赖区 ``invalid``。
    ``recovered_generation_count_missing`` 属“代身份已由权威证明、仅缺计数”，不拦。
    """
    row = context.record_row
    state = row.get("report_status")
    reason = str(row.get("degradation_reason_code") or "authority_mismatch")
    if state == "invalid":
        return status.make_status(
            "invalid",
            reason_code=reason,
            detail=f"run record 未通过完整代门禁（{reason}），依赖区不得认证为可用",
        )
    if state == "degraded" and reason == "run_record_pre_switch_shape":
        return status.make_status(
            "degraded",
            reason_code=reason,
            detail="切换前记录形态：值保留但不认证为正常 v2 代",
        )
    return None


def _apply_gate(row: dict[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    """把代门禁叠加到依赖区结果上（不覆盖更严重的既有状态，保留原明细）。"""
    order = {"available": 0, "empty_valid": 0, "stale": 1, "degraded": 2, "invalid": 3}
    current = str(row.get("report_status"))
    gated = str(gate.get("report_status"))
    if order.get(gated, 3) <= order.get(current, 3):
        return row
    merged = dict(row)
    merged["report_status"] = gated
    merged["degradation_reason_code"] = gate.get("degradation_reason_code")
    merged["verification_level"] = "projection_only"
    merged["detail"] = f"{row.get('detail') or ''} | {gate.get('detail')}".strip(" |")
    merged["gate_reason_code"] = gate.get("degradation_reason_code")
    return merged


def project_sections(context: CockpitContext) -> dict[str, Any]:
    """逐区装配；非可用状态一律登记到 ``context.degradations``。"""
    nav_row = _track(context, "nav", project_nav(context))
    gate = _generation_gate(context)
    account = project_account(context)
    trades = project_trades(context)
    config_row = project_config_section(context, nav_row)
    if gate is not None:
        account = _apply_gate(account, gate)
        trades = _apply_gate(trades, gate)
        config_row = _apply_gate(config_row, gate)
    sections: dict[str, Any] = {
        "record": _track(context, "record", dict(context.record_row)),
        "account": _track(context, "account", account),
        "nav": nav_row,
        "trades": _track(context, "trades", trades),
        "instructions": _track(
            context, "instructions", project_instructions_section(context, gate)
        ),
        "pending": {
            "buys": _track(context, "pending.buys", project_pending_section(context, "buys")),
            "sells": _track(context, "pending.sells", project_pending_section(context, "sells")),
        },
        "rebalance": _track(context, "rebalance", project_rebalance_section(context)),
    }
    sections["config"] = _track(context, "config", config_row)
    return sections


def _track(context: CockpitContext, field_path: str, row: dict[str, Any]) -> dict[str, Any]:
    """登记非可用状态并原样返回该区（``available/empty_valid`` 不登记）。"""
    if not status.is_usable(row):
        context.degradations.append(status.degradation_entry(row, field_path=field_path))
    return row


def project_account(context: CockpitContext) -> dict[str, Any]:
    """账户投影：权威重建 projection 与 ``account.json`` 逐字段全等核对。"""
    try:
        expected = build_account_projection(
            context.state.current_capital, context.state.positions, context.day
        )
    except PaperStateError as exc:
        return status.make_status(
            "invalid",
            reason_code="missing_precondition",
            detail=f"权威持仓形态非法，账户投影失败（不猜测、不补默认值）：{exc}",
        )
    except (KeyError, TypeError, ValueError) as exc:
        return status.make_status(
            "invalid", reason_code="missing_precondition", detail=f"权威账户投影失败：{exc}"
        )
    payload, parse_error = context.snapshot.read_json(PAPER_ROOT_ID, "state/account.json")
    if parse_error is not None:
        return status.make_status(
            "invalid", reason_code=parse_error, detail="state/account.json JSON 损坏"
        )
    row = authority.compare_account_projection(expected=expected, actual=payload)
    row["account"] = expected
    return row


def _parquet_rows(frame: Any) -> list[dict[str, Any]] | None:
    """DataFrame → 原生类型行列表（缺失返回 None，表示来源不存在）。"""
    if frame is None:
        return None
    if frame.empty:
        return []
    return [_plain(row) for row in frame.to_dict("records")]


def _plain_scalar(value: Any) -> Any:
    """单个标量降级：pandas/numpy → 原生类型，非有限浮点 → None（禁止 NaN/Inf 进 JSON）。"""
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (str, bytes)):
        value = item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _plain(row: Mapping[str, Any]) -> dict[str, Any]:
    """把一行记录降级为原生类型（保证 JSON 可序列化且判定口径一致）。"""
    return {str(key): _plain_scalar(value) for key, value in row.items()}


def project_nav(context: CockpitContext) -> dict[str, Any]:
    """NAV 区：权威 portfolio_values 三原值逐行核对 + 当前年化可用性。"""
    nav_rows = _parquet_rows(context.store.load_all_nav())
    trades, broken = _current_day_trades(context)
    row = authority.verify_nav_authority(
        portfolio_values=context.state.portfolio_values,
        nav_rows=nav_rows,
        authority_date=context.day,
        current_day_has_trades=bool(trades) or bool(broken),
    )
    return row


def _current_day_trades(
    context: CockpitContext,
) -> tuple[list[dict[str, Any]], list[str]]:
    """权威当前日成交 + **不可判定行的问题清单**（禁止静默丢弃证据）。

    R2-20261106-04：非 Mapping、日期不可判定、七元组投影失败的行不能当成“历史行”跳过，
    否则损坏的当前日成交会被转成“权威确认零成交”。
    """
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for index, entry in enumerate(context.state.trades):
        if not isinstance(entry, Mapping):
            problems.append(f"第 {index} 条权威成交不是对象（{type(entry).__name__}）")
            continue
        try:
            projected = authority.project_authority_trade_record(entry)
        except ValueError as exc:
            problems.append(f"第 {index} 条权威成交不可投影：{exc}")
            continue
        trade_date = projected.get("trade_date") or ""
        if not trade_date:
            # R2-20261106-04（复审）：日期不可判定 ≠ 历史行，必须进 broken_rows
            problems.append(
                f"第 {index} 条权威成交日期不可解析（date={entry.get('date')!r}），"
                "无法判定是否属于当前日"
            )
            continue
        if trade_date == context.day:
            rows.append(_plain(entry))
    return rows, problems


def project_trades(context: CockpitContext) -> dict[str, Any]:
    """当前日成交：record 计数 + trades 投影 + 权威七元组多重集核对。"""
    authority_rows, broken = _current_day_trades(context)
    if broken:
        return status.make_status(
            "invalid",
            reason_code="authority_mismatch",
            detail=(
                f"权威 {len(context.state.trades)} 条成交中 {len(broken)} 条不可判定"
                f"（{broken[:3]}），禁止当作零成交"
            ),
            extra={"broken_rows": broken, "trades": []},
        )
    projection_rows: list[dict[str, Any]] = []
    history_rows = 0
    all_rows = _parquet_rows(context.store.load_all_trades())
    for entry in all_rows or []:
        if authority.normalize_date(entry.get("trade_date")) == context.day:
            projection_rows.append(entry)
        else:
            history_rows += 1
    row = authority.verify_current_day_trades(
        authority_rows=authority_rows,
        projection_rows=projection_rows,
        record_count=context.counts.get("fills_count"),
        history_rows=history_rows,
    )
    row["trades"] = projection_rows
    return row


def project_instructions_section(
    context: CockpitContext, gate: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """未来指令区（选择规则见 ``instructions`` 模块；完整代门禁一并传递）。"""
    return instructions.project_instructions(
        names=context.instruction_names,
        current_date=context.day,
        record=context.record,
        pre_switch_shape=context.pre_switch_shape,
        loader=lambda name: context.store.load_instructions_frame(name[0:8]),
        generation_gate=gate,
    )


def project_pending_section(context: CockpitContext, kind: str) -> dict[str, Any]:
    """pending 买/卖队列（非 list 与计数冲突 invalid）。"""
    filename = "pending_buys.json" if kind == "buys" else "pending_sells.json"
    relative = f"pending_{kind}/{filename}"
    payload, parse_error = context.snapshot.read_json(PAPER_ROOT_ID, relative)
    declared = context.counts.get(f"pending_{kind}_count")
    return instructions.project_pending(
        present=context.snapshot.exists(PAPER_ROOT_ID, relative),
        payload=payload,
        parse_error=parse_error,
        declared_count=declared,
        label=f"{kind} 队列",
    )


def project_rebalance_section(context: CockpitContext) -> dict[str, Any]:
    """调仓状态五字段原值投影。"""
    relative = "runs/rebalance_state.json"
    payload, parse_error = context.snapshot.read_json(PAPER_ROOT_ID, relative)
    return instructions.project_rebalance(
        present=context.snapshot.exists(PAPER_ROOT_ID, relative),
        payload=payload,
        parse_error=parse_error,
    )


def project_config_section(
    context: CockpitContext, nav_row: Mapping[str, Any]
) -> dict[str, Any]:
    """配置/起始日/模型/年化（冲突检测在展平前完成；年化只读已装配的 NAV 区）。"""
    return config_view.project_config(
        store=context.store, raw_config=context.store.load_config(), nav_result=nav_row
    )


def collect_degradations(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """降级清单稳定排序去重（manifest ``degradations[]`` 唯一入口）。"""
    return status.collect_degradations(entries)


def load_quality_rows(frame: Any) -> list[dict[str, Any]] | None:
    """质量长表 → 行列表（供 T2 装配；缺失返回 None = 无来源）。"""
    if frame is None:
        return None
    return [_plain(row) for row in frame.to_dict("records")]
