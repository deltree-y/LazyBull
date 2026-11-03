# -*- coding: utf-8 -*-
"""P5b 试车场 T2 臂运行时：臂规格、宿主装配、虚拟日成功资格与权威估值读取。

契约来源：设计 v4 §4.1（逐日推进 / 固定 ``schedule_mode=replay`` / 不联网不补齐）、
§4.1.1（虚拟日成功资格 A/B 两条路径）、§7.2（每臂 driven/日期/完整代/权威估值条数）。

成功资格（§4.1.1，**不得**把 ``run_t1`` 的无异常/None 返回当成功）：

- 路径 A（首次执行）：必须从宿主同日结果读到 ``driven=True``、``run_kind="t1"``（小写），
  且 ``run_kind`` 与校正后日期与实际请求日期一致；
- 路径 B（幂等恢复）：``driven=False, t0_status="already_run"`` 只作为信号上抛，是否接纳由
  调用方（runner/族级事务）按已发布代核验决定——本模块**不**直接接纳；
- ``driven=False, t0_status="not_ready"`` ⇒ 上抛 :class:`VirtualDayNotReady`，整族停留同日。

权威估值（§4.1.1 A3/A4）：每臂当日必须**恰有一条** ``portfolio_values``，且从**完整
state 字段**（``state/engine_state.bin`` → ``fields.portfolio_values``）读取；禁止用稀疏
``nav.parquet`` 或展示 NAV 代替；无成交日同样必须有完整估值行。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from src.lazybull.v2.core.accounting.engine_state import loads_blob
from src.lazybull.v2.core.accounting.nav import calculate_nav
from src.lazybull.v2.hosts.paper.factory import create_paper_host
from src.lazybull.v2.hosts.paper.host import PaperHost
from src.lazybull.v2.hosts.paper.replay import assert_single_day_row_domain
from src.lazybull.v2.hosts.paper.state_store import PaperStateStore
from src.lazybull.v2.hosts.testbed.d0 import load_d0_record
from src.lazybull.v2.hosts.testbed.layout import FamilyTxnError
from src.lazybull.v2.hosts.testbed.registration import list_arms, load_arm, load_campaign
from src.lazybull.v2.hosts.testbed.storage import read_json
from src.lazybull.v2.store.row_domain import STRUCTURAL
from src.lazybull.v2.store.store_identity import canonical_json_sha256

__all__ = [
    "ArmSpec",
    "DrivenQualificationError",
    "HostFactory",
    "VirtualDayAlreadyRun",
    "VirtualDayNotReady",
    "drive_virtual_day",
    "load_arm_specs",
    "make_default_host_factory",
    "read_authoritative_valuation",
]

#: 宿主工厂：``(臂规格, 本次暂存根) -> PaperHost``（测试注入桩宿主的唯一接缝）
HostFactory = Callable[["ArmSpec", Path], PaperHost]
#: 允许的驱动入口（§4.1.1 A1：必须是小写 ``t1``）
REQUIRED_RUN_KIND = "t1"


class DrivenQualificationError(FamilyTxnError):
    """虚拟日成功资格不成立（§4.1.1 A1~A4）。"""

    __test__ = False


class VirtualDayNotReady(FamilyTxnError):
    """宿主当日未就绪（整族必须停留同一虚拟日，不写 index、不推进）。"""

    __test__ = False

    def __init__(self, arm_id: str, virtual_date: str, gaps: Sequence[Any]) -> None:
        self.arm_id = arm_id
        self.virtual_date = virtual_date
        self.gaps = list(gaps)
        super().__init__(
            f"臂 {arm_id!r} 在虚拟日 {virtual_date} 未就绪（driven=False/not_ready），"
            f"整族停留同日不推进；readiness gaps={self.gaps}"
        )


class VirtualDayAlreadyRun(FamilyTxnError):
    """宿主同日已驱动（``already_run``）：是否接纳由已发布代核验决定（§4.1.1 B2）。"""

    __test__ = False

    def __init__(self, arm_id: str, virtual_date: str, detail: Mapping[str, Any]) -> None:
        self.arm_id = arm_id
        self.virtual_date = virtual_date
        self.detail = dict(detail)
        super().__init__(
            f"臂 {arm_id!r} 在虚拟日 {virtual_date} 返回 already_run（driven=False）——"
            "未发布 attempt 不得直接接纳，须按 §4.3 终止并按已发布代核验"
        )


@dataclass(frozen=True)
class ArmSpec:
    """已注册模拟臂的运行时规格（身份全部来自 T1 注册台账，禁止另造）。"""

    arm_id: str
    arm_role: str
    root: Path
    config_flat: Dict[str, Any]
    config_fingerprint: str
    model_version: int
    models_dir: str
    model_content_fingerprint: str
    train_end_date: str
    data_root: str
    initial_capital: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "arm_role": self.arm_role,
            "root": self.root.as_posix(),
            "config_fingerprint": self.config_fingerprint,
            "model_version": self.model_version,
            "models_dir": self.models_dir,
            "model_content_fingerprint": self.model_content_fingerprint,
            "train_end_date": self.train_end_date,
            "initial_capital": self.initial_capital,
        }


def load_arm_specs(campaign_root: str | Path) -> Dict[str, ArmSpec]:
    """读取 campaign 全部已注册臂的运行时规格（未注册/未 D0 ⇒ fail-fast）。"""
    root = Path(campaign_root)
    campaign = load_campaign(root)
    d0_record = load_d0_record(root)
    campaign_identity = campaign.get("identity") or {}
    data_root = str((campaign_identity.get("input") or {})["data_root"])
    initial_capital = float(d0_record["initial_capital"])
    specs: Dict[str, ArmSpec] = {}
    for arm_id in sorted(list_arms(root)):
        record = load_arm(root, arm_id)
        identity = record.get("identity") or {}
        config = identity.get("config") or {}
        model = identity.get("model") or {}
        # 冻结配置唯一来源 = 注册台账的 config.json（与 D0 materialize 同源；不另解析）
        config_flat = read_json(
            root / "registered" / "arms" / arm_id / "config.json",
            what=f"臂 {arm_id!r} 冻结配置",
        )
        specs[arm_id] = ArmSpec(
            arm_id=arm_id,
            arm_role=str(record["arm_role"]),
            root=root / "arms" / arm_id,
            config_flat=dict(config_flat),
            config_fingerprint=str(config.get("fingerprint")),
            model_version=int(model.get("version")),
            models_dir=str(model.get("models_dir")),
            model_content_fingerprint=str(model.get("content_fingerprint")),
            train_end_date=str(model.get("train_end_date")),
            data_root=data_root,
            initial_capital=initial_capital,
        )
    if not specs:
        raise FamilyTxnError(f"campaign {root.name!r} 无已注册臂，禁止启动虚拟时钟运行")
    return specs


def make_default_host_factory(
    *,
    data_root: str,
    replay_calendar: Sequence[pd.Timestamp],
    host_overrides: Optional[Mapping[str, Any]] = None,
) -> HostFactory:
    """默认宿主工厂：``create_paper_host`` + 固定 ``schedule_mode="replay"``。

    交易日历交给 ``create_paper_host`` 自 ``data_root`` 的 trade_cal 装配（不另造日历
    口径）；``replay_calendar`` = 本 campaign 虚拟窗口。``host_overrides`` 只供测试/
    影子注入桩信号与数据面，**不接受** ``schedule_mode`` 覆盖——回放口径不得切到 live。
    """
    overrides = dict(host_overrides or {})
    if "schedule_mode" in overrides:
        raise FamilyTxnError("试车场宿主工厂禁止覆盖 schedule_mode（固定 replay，禁止切到 live）")

    def _factory(spec: ArmSpec, store_root: Path) -> PaperHost:
        store = PaperStateStore(root_path=str(store_root))
        return create_paper_host(
            store,
            config_flat=dict(spec.config_flat),
            data_root=data_root,
            schedule_mode="replay",
            replay_calendar=list(replay_calendar),
            **overrides,
        )

    return _factory


def _day_text(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y%m%d")
    return str(value).strip().replace("-", "")


def _normalise(value: Any) -> Any:
    """canonical JSON 前的递归归一（Timestamp → YYYYMMDD、numpy 标量 → Python 标量）。"""
    if isinstance(value, Mapping):
        return {str(key): _normalise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y%m%d")
    if value is None or isinstance(value, (bool, str)):
        return value
    if hasattr(value, "item"):  # numpy 标量
        return _normalise(value.item())
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise DrivenQualificationError(f"证据数值非有限（{type(value).__name__}）: {value!r}")
        return value
    return str(value)


def read_authoritative_valuation(
    store: PaperStateStore, virtual_date: str, initial_capital: float
) -> Dict[str, Any]:
    """读取当日唯一权威估值行（完整 state 字段；不接受 nav.parquet/展示 NAV）。"""
    raw = store.load_engine_state_blob()
    if raw is None:
        raise DrivenQualificationError(
            f"虚拟日 {virtual_date} 驱动后缺权威完整快照（state/engine_state.bin）"
        )
    blob = loads_blob(raw)
    fields = blob.get("fields") or {}
    rows = [
        row
        for row in (fields.get("portfolio_values") or [])
        if _day_text((row or {}).get("date")) == virtual_date
    ]
    if len(rows) != 1:
        raise DrivenQualificationError(
            f"虚拟日 {virtual_date} 的权威估值行必须恰 1 条（完整 state 字段读取），"
            f"实际 {len(rows)} 条；禁止用稀疏 nav.parquet 或缺行代替"
        )
    row = rows[0]
    total_value = float(row["portfolio_value"])
    if not math.isfinite(total_value):
        raise DrivenQualificationError(
            f"虚拟日 {virtual_date} 的 NAV 非有限: portfolio_value={total_value!r}"
        )
    nav = float(calculate_nav(total_value, float(initial_capital)))
    if not math.isfinite(nav):
        raise DrivenQualificationError(f"虚拟日 {virtual_date} 的 nav 非有限: {nav!r}")
    last_date = _day_text((fields.get("portfolio_values") or [{}])[-1].get("date"))
    if last_date != virtual_date:
        raise DrivenQualificationError(
            f"虚拟日 {virtual_date} 不是权威最后处理日（实际 {last_date}）——禁止跳日或回退"
        )
    return {
        "source": "engine_state.portfolio_values",
        "row_count": 1,
        "portfolio_value": total_value,
        "nav": nav,
        "cash": float(row.get("capital", 0.0)),
        "position_value": float(row.get("market_value", 0.0)),
        "initial_capital": float(initial_capital),
    }


def _day_trades(fields: Mapping[str, Any], virtual_date: str) -> List[Dict[str, Any]]:
    """当日成交（引擎记录，取自完整 state 字段；非 parquet 投影）。"""
    return [
        _normalise(row)
        for row in (fields.get("trades") or [])
        if _day_text((row or {}).get("date")) == virtual_date
    ]


def _positions(state: Any) -> Dict[str, Any]:
    return _normalise(dict(state.positions or {}))


def drive_virtual_day(
    host: PaperHost,
    spec: ArmSpec,
    virtual_date: str,
    *,
    expected_domain: str = STRUCTURAL,
) -> Dict[str, Any]:
    """驱动一个虚拟日并按 §4.1.1 路径 A 校验成功资格，返回可落盘证据。

    Returns:
        证据字典（确定性、无墙钟）：driven/run_kind/日期/权威估值/持仓/成交/行域回执/
        内容身份摘要，供 journal 与分片共用。

    Raises:
        VirtualDayNotReady: ``driven=False`` 且 ``not_ready``（整族停留同日）。
        VirtualDayAlreadyRun: ``driven=False`` 且 ``already_run``（接纳与否由已发布代决定）。
        DrivenQualificationError: 资格不成立（run_kind/日期/估值行数/非有限 NAV）。
        ReplayRowDomainError: 实际读取分区行域非 structural（paper/replay 回执）。
    """
    receipt = assert_single_day_row_domain(
        host.panel_store, virtual_date, expected_domain=expected_domain
    )
    host.run_t1(virtual_date)
    result = host.last_day_result
    if result is None:
        raise DrivenQualificationError(
            f"臂 {spec.arm_id!r} 在虚拟日 {virtual_date} 驱动后无宿主同日结果"
        )
    _assert_driven(result, spec, virtual_date)
    if not host.store.check_run_exists(REQUIRED_RUN_KIND, virtual_date):
        raise DrivenQualificationError(
            f"臂 {spec.arm_id!r} 在虚拟日 {virtual_date} 缺完整代发布标记 "
            f"runs/{REQUIRED_RUN_KIND}_{virtual_date}.json（半提交代不得计入成功日）"
        )
    valuation = read_authoritative_valuation(host.store, virtual_date, spec.initial_capital)
    state = host.store.load_engine_state()
    if state is None:
        raise DrivenQualificationError(f"臂 {spec.arm_id!r} 驱动后无法重建完整代状态")
    blob = loads_blob(host.store.load_engine_state_blob())
    fills = _day_trades(blob.get("fields") or {}, virtual_date)
    positions = _positions(state)
    evidence: Dict[str, Any] = {
        "arm_id": spec.arm_id,
        "arm_role": spec.arm_role,
        "virtual_date": virtual_date,
        "requested_date": str(result.requested_date),
        "driven_date": str(result.corrected_date),
        "driven": True,
        "run_kind": str(result.run_kind),
        "t0_status": str(result.t0_status),
        "complete_generation": True,
        "valuation": valuation,
        "cash": float(state.current_capital),
        "positions": positions,
        "fills": fills,
        "row_domain_receipt": _receipt_payload(receipt),
    }
    evidence["day_digest"] = canonical_json_sha256(
        {
            "virtual_date": virtual_date,
            "driven_date": evidence["driven_date"],
            "requested_date": evidence["requested_date"],
            "cash": evidence["cash"],
            "positions": positions,
            "fills": fills,
            "valuation": valuation,
        }
    )
    return evidence


def _assert_driven(result: Any, spec: ArmSpec, virtual_date: str) -> None:
    """§4.1.1 A1 资格：driven=True、``run_kind="t1"``（小写）、实际驱动日期 == 请求日期。"""
    if not result.driven:
        if result.t0_status == "not_ready":
            gaps = [] if result.readiness is None else result.readiness.gaps
            raise VirtualDayNotReady(spec.arm_id, virtual_date, gaps)
        if result.t0_status == "already_run":
            raise VirtualDayAlreadyRun(
                spec.arm_id,
                virtual_date,
                {"driven_date": result.corrected_date, "run_kind": result.run_kind},
            )
        raise DrivenQualificationError(
            f"臂 {spec.arm_id!r} 在虚拟日 {virtual_date} 未驱动（t0_status={result.t0_status!r}）"
        )
    if result.run_kind != REQUIRED_RUN_KIND:
        raise DrivenQualificationError(
            f"臂 {spec.arm_id!r} 的驱动入口必须是 {REQUIRED_RUN_KIND!r}（小写），"
            f"实际 {result.run_kind!r}"
        )
    if result.corrected_date != virtual_date:
        raise DrivenQualificationError(
            f"臂 {spec.arm_id!r} 的实际驱动日期 {result.corrected_date!r} ≠ 请求日期 "
            f"{virtual_date!r}（禁止跳日/校正绕过）"
        )


def _receipt_payload(receipt: Mapping[str, Any]) -> Dict[str, Any]:
    """行域回执落盘形态（只保留可复算的日期/文件身份/行域，不含读取时刻）。"""
    return {
        "declared_row_domain": receipt["declared_row_domain"],
        "actual_row_domains": receipt["actual_row_domains"],
        "files": receipt["files"],
        "query_start": receipt["query_start"],
        "query_end": receipt["query_end"],
        "passed": True,
    }
