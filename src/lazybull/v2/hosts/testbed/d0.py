# -*- coding: utf-8 -*-
"""P5b 试车场 D0 fresh 起点（空现金账户 + 共用起点核验）。

设计 v4 §4.1 第 3 条的落地口径：

- 所有臂从**同一历史 D0 的空现金账户**起步（``fresh_cash``）；
- **不复制**生产持仓、pending、instructions、nav/trades、coverage history；
- D0 只是估值日，记录里显式写明 D1 才产生第一个相邻收益；
- 每臂根由 ``PaperStateStore`` 建立，并核验一致性 = ``fresh``（无权威快照/无投影）。

评审处置（R1-T1-06/07/11、R2-T1-06/07/08/09/17/18/19/20）：空账户证据改为**从盘上
观测**（七子目录逐一空判 + 配置初值），不再由入参自证；D0 记录封存并在复核时复算；
复核路径只读（``create_layout=False``）、拒绝记录内的路径穿越、复检臂集合与漂移门；
``d0_fresh.json`` 写入纳入同一回滚事务。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml

from src.lazybull.v2.evidence.fingerprint_keys import fingerprint
from src.lazybull.v2.hosts.paper.config_adapter import load_paper_config, to_trading_config
from src.lazybull.v2.hosts.paper.state_store import CONFIG_FILENAME, NAV_PARQUET_FILENAME
from src.lazybull.v2.hosts.paper.state_store import PaperStateStore, TRADES_PARQUET_FILENAME
from src.lazybull.v2.hosts.testbed.clock import VirtualClock
from src.lazybull.v2.hosts.testbed.identity import (
    collect_code_identity,
    collect_config_identity,
    collect_data_state_identity,
)
from src.lazybull.v2.hosts.testbed.paths import (
    D0_SCHEMA,
    EVIDENCE_CLASS,
    SCOPE,
    TestbedError,
    assert_write_scope,
    project_root,
)
from src.lazybull.v2.hosts.testbed.registration import (
    DuplicateRegistrationError,
    list_arms,
    load_arm,
    load_campaign,
)
from src.lazybull.v2.hosts.testbed.storage import (
    raise_rollback_failure,
    read_json,
    seal_record,
    verify_seal,
    write_json_exclusive,
)
from src.lazybull.v2.store.store_identity import canonical_json_sha256, verify_file_identities

__all__ = [
    "D0FreshError",
    "D0_FRESH_FILENAME",
    "EmptyStateObservation",
    "empty_account_digest",
    "initialize_d0_fresh",
    "observe_empty_account",
    "verify_d0_fresh",
]

#: D0 fresh 记录文件名（相对 ``registered/``）
D0_FRESH_FILENAME = "d0_fresh.json"
#: PaperStateStore 七子目录：D0 fresh 下必须逐一为空
_EMPTY_DIRS = (
    "state",
    "trades",
    "nav",
    "runs",
    "pending_buys",
    "pending_sells",
    "instructions",
)


class D0FreshError(TestbedError):
    """D0 fresh 起点不合法（非空根 / 状态非 fresh / 各臂不一致）。"""


@dataclass(frozen=True)
class EmptyStateObservation:
    """从盘上观测到的臂起点状态（D0 fresh 的唯一证据形态）。"""

    arm_root: Path
    cash: float
    cash_basis: str
    positions: Dict[str, Any]
    pending_buys: list
    pending_sells: list
    instructions: list
    nav_rows: int
    trade_rows: int
    coverage_history: bool
    state_files: list
    store_consistency: str

    def payload(self) -> Dict[str, Any]:
        return {
            "cash": self.cash,
            "cash_basis": self.cash_basis,
            "positions": self.positions,
            "pending_buys": self.pending_buys,
            "pending_sells": self.pending_sells,
            "instructions": self.instructions,
            "nav_rows": self.nav_rows,
            "trade_rows": self.trade_rows,
            "coverage_history": self.coverage_history,
            "state_files": self.state_files,
            "store_consistency": self.store_consistency,
        }

    def digest(self) -> str:
        return canonical_json_sha256(self.payload())


@dataclass(frozen=True)
class ArmD0State:
    """单臂 D0 状态（写进 d0_fresh.json，供后续逐臂核验）。"""

    arm_id: str
    arm_role: str
    root: str
    config_fingerprint: str
    model_content_fingerprint: str
    initial_capital: float
    observation: EmptyStateObservation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "arm_role": self.arm_role,
            "root": self.root,
            "config_fingerprint": self.config_fingerprint,
            "model_content_fingerprint": self.model_content_fingerprint,
            "initial_capital": self.initial_capital,
            "empty_state": self.observation.payload(),
            "empty_state_sha256": self.observation.digest(),
        }


def empty_account_digest(initial_capital: float) -> str:
    """空现金账户的**期望**摘要（无盘上观测时的对比基线）。"""
    payload = {
        "cash": float(initial_capital),
        "cash_basis": "config_initial_capital",
        "positions": {},
        "pending_buys": [],
        "pending_sells": [],
        "instructions": [],
        "nav_rows": 0,
        "trade_rows": 0,
        "coverage_history": False,
        "state_files": [],
        "store_consistency": "fresh",
    }
    return canonical_json_sha256(payload)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _require(mapping: Mapping[str, Any], key: str, *, what: str) -> Any:
    """取必需字段；缺键抛带上下文的 ``D0FreshError``（禁止裸 KeyError）。"""
    if key not in mapping:
        raise D0FreshError(f"{what}缺必需字段 {key}")
    return mapping[key]


def _campaign_clock(campaign: Mapping[str, Any]) -> VirtualClock:
    payload = _require(campaign, "clock", what="campaign 记录")
    return VirtualClock(
        d0_date=str(payload["d0_date"]),
        days=tuple(str(day) for day in payload["days"]),
    )


def _campaign_data_root(campaign: Mapping[str, Any]) -> str:
    return str((campaign.get("identity") or {})["input"]["data_root"])


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _dir_files(path: Path) -> list[str]:
    if not path.is_dir():
        return []
    return sorted(entry.name for entry in path.iterdir())


def _parquet_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    import pandas as pd

    return int(len(pd.read_parquet(path)))


def observe_empty_account(
    arm_root: Path, initial_capital: float, *, create_layout: bool = False
) -> EmptyStateObservation:
    """观测臂根是否处于 fresh 空账户状态（只读；``create_layout=False``）。

    Raises:
        D0FreshError: 臂根缺失 / 七子目录非空 / 状态库一致性非 ``fresh``。
    """
    root = Path(arm_root)
    if not root.is_dir():
        raise D0FreshError(f"臂根不存在（禁止用不存在的根冒充 D0 fresh）: {root}")
    store = PaperStateStore(root_path=str(root), create_layout=create_layout)
    dirty = {name: files for name in _EMPTY_DIRS if (files := _dir_files(root / name))}
    if dirty:
        raise D0FreshError(f"臂根不是 fresh（存在既有产物 {dirty}）: {root}")
    status = store.check_consistency(auto_repair=False)
    if status != "fresh":
        raise D0FreshError(f"臂根状态库一致性必须为 fresh，实际 {status!r}: {root}")
    return EmptyStateObservation(
        arm_root=root,
        cash=float(initial_capital),
        cash_basis="config_initial_capital",
        positions={},
        pending_buys=[],
        pending_sells=[],
        instructions=[],
        nav_rows=_parquet_rows(root / "nav" / NAV_PARQUET_FILENAME),
        trade_rows=_parquet_rows(root / "trades" / TRADES_PARQUET_FILENAME),
        coverage_history=(root / "runs" / "coverage_history.json").exists(),
        state_files=_dir_files(root / "state"),
        store_consistency=status,
    )


def _assert_drift_free(campaign: Mapping[str, Any]) -> None:
    """D0 前置复检：窗口数据态与执行闭包必须与 campaign 登记一致。"""
    clock = _campaign_clock(campaign)
    data_root = _campaign_data_root(campaign)
    window_state = collect_data_state_identity(data_root=data_root, virtual_days=clock.days)
    campaign_identity = _require(campaign, "identity", what="campaign 记录")
    recorded_window = (campaign_identity.get("window_state") or {}).get("window_state_id")
    if window_state["window_state_id"] != recorded_window:
        raise D0FreshError(
            "窗口数据态漂移：campaign 登记后窗口内历史分区内容已变化，禁止初始化 D0: "
            f"{window_state['window_state_id']} vs {recorded_window}"
        )
    code_identity = collect_code_identity(str(campaign.get("repo_root") or project_root()))
    recorded_code = (campaign_identity.get("code") or {}).get("execution_digest")
    if code_identity["execution_digest"] != recorded_code:
        raise D0FreshError(
            "执行闭包代码态漂移：campaign 登记后试车场/复用件源码已变化，禁止初始化 D0: "
            f"{code_identity['execution_digest']} vs {recorded_code}"
        )


def _materialize_config(arm_root: Path, frozen_flat: Mapping[str, Any]) -> Dict[str, Any]:
    """把注册时冻结的扁平配置落到臂根 ``config.yaml`` 并核验闭环。"""
    text = yaml.safe_dump(dict(frozen_flat), allow_unicode=True, sort_keys=True)
    path = arm_root / CONFIG_FILENAME
    path.write_text(text, encoding="utf-8")
    reloaded = load_paper_config(arm_root)
    frozen_fp = collect_config_identity(dict(frozen_flat))
    reloaded_fp = collect_config_identity(reloaded)
    if reloaded_fp["fingerprint"] != frozen_fp["fingerprint"]:
        removed = sorted(set(frozen_fp["execution"]) - set(reloaded_fp["execution"]))
        changed = sorted(
            key
            for key, value in frozen_fp["execution"].items()
            if key in reloaded_fp["execution"] and reloaded_fp["execution"][key] != value
        )
        raise D0FreshError(
            "臂配置落盘后有效执行指纹漂移（禁止 D0 使用与注册不一致的配置）: "
            f"{arm_root}；缺失键={removed}；取值不同键={changed}；"
            f"full_config_changed={frozen_fp['config_sha256'] != reloaded_fp['config_sha256']}"
        )
    return reloaded


def initialize_d0_fresh(
    campaign_root: str | Path,
    *,
    arm_ids: Optional[Sequence[str]] = None,
    allow_external_root: bool = False,
) -> Dict[str, Any]:
    """为已注册臂建立 D0 fresh 空现金起点（一次成功，重复即拒绝）。

    Args:
        campaign_root: 已登记 campaign 根。
        arm_ids: 期望覆盖的臂；缺省 = 全部已注册臂。显式给出时必须**等于**
            全部已注册臂（不允许半初始化）。
        allow_external_root: 仅测试/临时脚本放开外部根（不解除生产根禁令）。

    Returns:
        落盘的 ``registered/d0_fresh.json`` 内容。

    Raises:
        UnregisteredArmError: campaign 未注册，或指定了未注册臂。
        DuplicateRegistrationError: D0 已初始化过。
        D0FreshError: 臂根非空 / 状态非 fresh / 各臂起点不一致 / 数据态漂移。
    """
    root = Path(campaign_root)
    campaign = load_campaign(root)
    assert_write_scope(
        root, data_root=_campaign_data_root(campaign), allow_external_root=allow_external_root
    )
    record_path = root / "registered" / D0_FRESH_FILENAME
    if record_path.is_file():
        raise DuplicateRegistrationError(
            f"campaign {campaign['campaign_id']!r} 的 D0 已初始化，禁止重复初始化: {record_path}"
        )
    arms = list_arms(root)
    if not arms:
        raise D0FreshError(
            f"campaign {campaign['campaign_id']!r} 无已注册臂，禁止初始化 D0（先注册臂）"
        )
    requested = [str(arm_id) for arm_id in (arm_ids if arm_ids is not None else arms)]
    for arm_id in requested:
        load_arm(root, arm_id)  # 未注册 ⇒ UnregisteredArmError
    if sorted(requested) != sorted(arms):
        raise D0FreshError(
            "D0 fresh 必须覆盖 campaign 全部已注册臂（禁止半初始化）: "
            f"requested={sorted(requested)} registered={sorted(arms)}"
        )
    _assert_drift_free(campaign)
    clock = _campaign_clock(campaign)
    # 先整体预检，避免部分臂已建根、部分臂失败形成半初始化现场
    for arm_id in requested:
        if (root / "arms" / arm_id).exists():
            raise D0FreshError(
                f"D0 fresh 要求全新臂根，实际已存在（禁止覆盖或续跑）: {root / 'arms' / arm_id}"
            )
    states: Dict[str, ArmD0State] = {}
    created: list[Path] = []
    arms_root = root / "arms"
    arms_created = False
    try:
        if not arms_root.exists():
            arms_root.mkdir(parents=True, exist_ok=False)
            arms_created = True
        for arm_id in requested:
            frozen = read_json(
                root / "registered" / "arms" / arm_id / "config.json",
                what=f"臂 {arm_id!r} 冻结配置",
            )
            arm_root = arms_root / arm_id
            arm_root.mkdir(parents=False, exist_ok=False)
            created.append(arm_root)
            flat = _materialize_config(arm_root, frozen)
            initial_capital = float(to_trading_config(flat).initial_capital)
            observation = observe_empty_account(arm_root, initial_capital, create_layout=True)
            arm_record = arms[arm_id]
            states[arm_id] = ArmD0State(
                arm_id=arm_id,
                arm_role=str(arm_record["arm_role"]),
                root=f"arms/{arm_id}",
                config_fingerprint=collect_config_identity(flat)["fingerprint"],
                model_content_fingerprint=str(
                    arm_record["identity"]["model"]["content_fingerprint"]
                ),
                initial_capital=initial_capital,
                observation=observation,
            )
        identical = _assert_arms_consistent(states)
        record = seal_record(
            {
                "schema": D0_SCHEMA,
                "campaign_id": campaign["campaign_id"],
                "scope": SCOPE,
                "evidence_class": EVIDENCE_CLASS,
                "not_forward_evidence": True,
                "not_sla_evidence": True,
                "not_real_candidate": True,
                "initialized_at": _now(),
                "source": "fresh_cash",
                "production_state_copied": False,
                "row_domain": _require(campaign, "row_domain", what="campaign 记录"),
                "row_domain_verified": False,
                "row_domain_receipt": None,
                "row_domain_note": "row_domain 由 campaign 声明继承；逐日行域回执属 mechanical replay",
                "evidence_notes": {
                    "empty_state": (
                        "由七子目录盘上观测生成；store_consistency 为前置空目录断言蕴含的"
                        "防御性字段，不得单独引用"
                    ),
                    "all_arms_identical": (
                        "由各臂盘上观测摘要比对得出；正式结论以 verify_d0_fresh 复算为准"
                    ),
                },
                "clock": dict(_require(campaign, "clock", what="campaign 记录")),
                "d0_date": clock.d0_date,
                "d0_role": "valuation_only",
                "first_return_date": clock.first_return_date,
                "initial_capital": next(iter(states.values())).initial_capital,
                "all_arms_identical": identical,
                "arms": {arm_id: state.to_dict() for arm_id, state in states.items()},
                "campaign_identity_summary": dict(
                    _require(campaign, "identity_summary", what="campaign 记录")
                ),
            }
        )
        write_json_exclusive(record_path, record)
    except FileExistsError as exc:
        raise_rollback_failure(
            created,
            DuplicateRegistrationError(f"D0 并发初始化被独占创建拒绝: {record_path}（{exc}）"),
            empty_dirs=(arms_root,) if arms_created else (),
        )
    except Exception as exc:  # noqa: BLE001 - 统一转回滚后重抛，不吞异常
        # 只回滚本次创建的叶子臂根；arms/ 仅在本次创建且删空后才回收
        raise_rollback_failure(created, exc, empty_dirs=(arms_root,) if arms_created else ())
    return record


def _assert_arms_consistent(states: Mapping[str, ArmD0State]) -> bool:
    """各臂必须同起点：初始资金与观测摘要逐位一致（不一致即 fail-closed）。"""
    digests = {arm_id: state.observation.digest() for arm_id, state in states.items()}
    capitals = {arm_id: state.initial_capital for arm_id, state in states.items()}
    if len(set(digests.values())) != 1:
        raise D0FreshError(f"各臂 D0 空账户不一致（禁止不同起点）: {digests}")
    if len(set(capitals.values())) != 1:
        raise D0FreshError(f"各臂 D0 初始资金不一致（禁止不同起点）: {capitals}")
    return True


def _load_d0_for_verify(root: Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """读取并校验 campaign + D0 记录（封存、schema、时钟一致）。"""
    campaign = load_campaign(root)
    record_path = root / "registered" / D0_FRESH_FILENAME
    record = read_json(record_path, what="D0 fresh 记录")
    verify_seal(record, what=f"D0 fresh 记录（{record_path}）")
    if record.get("schema") != D0_SCHEMA:
        raise D0FreshError(f"D0 记录 schema 非法: {record.get('schema')!r}")
    if record.get("clock") != dict(_require(campaign, "clock", what="campaign 记录")):
        raise D0FreshError("D0 记录时钟与 campaign 注册时钟不一致（时钟被改写）")
    if not (record.get("arms") or {}):
        raise D0FreshError(f"D0 记录缺臂清单: {record_path}")
    return campaign, record


def _verify_arm_registration(
    root: Path, arm_id: str, payload: Mapping[str, Any], campaign: Mapping[str, Any]
) -> tuple[str, str]:
    """复核臂注册身份与 campaign/D0 记录一致，并复算模型三件套物理内容。

    Returns:
        ``(config_fingerprint, model_content_fingerprint)``（注册记录登记值）。
    """
    what = f"臂 {arm_id!r} 的注册记录"
    d0_what = f"臂 {arm_id!r} 的 D0 记录"
    record = load_arm(root, arm_id)
    identity = _require(record, "identity", what=what)
    campaign_identity = _require(campaign, "identity", what="campaign 记录")
    window = _require(identity, "window_state", what=what)
    campaign_window = _require(campaign_identity, "window_state", what="campaign 记录")
    if _require(window, "window_state_id", what=what) != _require(
        campaign_window, "window_state_id", what="campaign 记录"
    ):
        raise D0FreshError(f"臂 {arm_id!r} 的窗口数据态与 campaign 不一致")
    source = _require(identity, "input", what=what)
    campaign_source = _require(campaign_identity, "input", what="campaign 记录")
    if _require(source, "input_sha256", what=what) != _require(
        campaign_source, "input_sha256", what="campaign 记录"
    ):
        raise D0FreshError(f"臂 {arm_id!r} 的输入身份与 campaign 不一致")
    config = _require(identity, "config", what=what)
    config_fingerprint = _require(config, "fingerprint", what=what)
    if config_fingerprint != _require(payload, "config_fingerprint", what=d0_what):
        raise D0FreshError(f"臂 {arm_id!r} 的注册配置身份与 D0 记录不一致（台账被改）")
    code = _require(identity, "code", what=what)
    campaign_code = _require(campaign_identity, "code", what="campaign 记录")
    if _require(code, "execution_digest", what=what) != _require(
        campaign_code, "execution_digest", what="campaign 记录"
    ):
        raise D0FreshError(f"臂 {arm_id!r} 的执行闭包代码态与 campaign 不一致")
    model = _require(identity, "model", what=what)
    files = _require(model, "files", what=what)
    artifacts = _require(model, "artifacts", what=what)
    if not files or not artifacts:
        raise D0FreshError(f"{what} 的模型身份为空清单（禁止空清单冒充已登记）")
    model_fingerprint = _require(model, "content_fingerprint", what=what)
    if fingerprint(artifacts) != model_fingerprint:
        raise D0FreshError(f"{what} 的模型内容指纹与 artifacts 不一致")
    if model_fingerprint != _require(payload, "model_content_fingerprint", what=d0_what):
        raise D0FreshError(f"臂 {arm_id!r} 的注册模型身份与 D0 记录不一致（台账被改）")
    mismatches = verify_file_identities(files, base_dir=_require(model, "models_dir", what=what))
    if mismatches:
        raise D0FreshError(f"臂 {arm_id!r} 的模型产物内容已漂移（注册后文件被改动）: {mismatches}")
    return config_fingerprint, model_fingerprint


def _verify_arm_d0(
    root: Path, arm_id: str, payload: Mapping[str, Any], campaign: Mapping[str, Any]
) -> tuple[str, float, str]:
    """单臂复核：根归属、配置指纹、空账户观测、初始资金，返回 (状态, 资金, 摘要)。"""
    what = f"臂 {arm_id!r} 的 D0 记录"
    expected_root = f"arms/{arm_id}"
    if str(payload.get("root")) != expected_root:
        raise D0FreshError(f"{what} root 非法（只允许 {expected_root}）: {payload.get('root')!r}")
    arm_root = (root / expected_root).resolve()
    if not _is_within(arm_root, root.resolve()):
        raise D0FreshError(f"{what} root 越出 campaign 根: {arm_root}")
    if not arm_root.is_dir():
        raise D0FreshError(f"臂根不存在（D0 现场已被删除）: {arm_root}")
    registered_config, registered_model = _verify_arm_registration(root, arm_id, payload, campaign)
    flat = load_paper_config(arm_root)
    capital = float(to_trading_config(flat).initial_capital)
    fingerprint = collect_config_identity(flat)["fingerprint"]
    if fingerprint != registered_config:
        raise D0FreshError(
            f"臂 {arm_id!r} 臂根配置指纹与注册记录不一致: {fingerprint} vs {registered_config}"
        )
    observation = observe_empty_account(arm_root, capital, create_layout=False)
    recorded_digest = _require(payload, "empty_state_sha256", what=what)
    if observation.digest() != recorded_digest:
        raise D0FreshError(
            f"臂 {arm_id!r} 空账户观测与 D0 记录不一致: "
            f"{observation.digest()} vs {recorded_digest}"
        )
    recorded_capital = float(_require(payload, "initial_capital", what=what))
    if capital != recorded_capital:
        raise D0FreshError(
            f"臂 {arm_id!r} 初始资金与 D0 记录不一致: {capital} vs {recorded_capital}"
        )
    return observation.store_consistency, capital, observation.digest()


def verify_d0_fresh(campaign_root: str | Path) -> Dict[str, Any]:
    """只读复核 D0 fresh 记录与各臂实际状态（不建目录、不写盘）。"""
    root = Path(campaign_root)
    campaign, record = _load_d0_for_verify(root)
    arms: Dict[str, Any] = record["arms"]
    registered = list_arms(root)
    if sorted(arms) != sorted(registered):
        raise D0FreshError(
            "D0 记录臂集合与已注册臂不一致: "
            f"record={sorted(arms)} registered={sorted(registered)}"
        )
    campaign_summary = _require(campaign, "identity_summary", what="campaign 记录")
    if record.get("campaign_identity_summary") != dict(campaign_summary):
        raise D0FreshError("D0 记录的 campaign 身份摘要与 campaign 注册记录不一致")
    _assert_drift_free(campaign)
    capitals: Dict[str, float] = {}
    digests: Dict[str, str] = {}
    statuses: Dict[str, str] = {}
    for arm_id, payload in arms.items():
        statuses[arm_id], capitals[arm_id], digests[arm_id] = _verify_arm_d0(
            root, arm_id, payload, campaign
        )
    if len(set(digests.values())) != 1 or len(set(capitals.values())) != 1:
        raise D0FreshError(f"复核发现各臂起点不一致: capitals={capitals} digests={digests}")
    top_capital = float(_require(record, "initial_capital", what="D0 记录"))
    if top_capital != next(iter(capitals.values())):
        raise D0FreshError(f"D0 记录顶层 initial_capital 与各臂不一致: {top_capital} vs {capitals}")
    if record.get("all_arms_identical") is not True:
        raise D0FreshError("D0 记录的 all_arms_identical 必须为 true")
    return {
        "campaign_id": campaign["campaign_id"],
        "d0_date": _require(record, "d0_date", what="D0 记录"),
        "first_return_date": record.get("first_return_date"),
        "arms": statuses,
        "initial_capital": next(iter(capitals.values())),
        "empty_state_sha256": next(iter(digests.values())),
        "all_arms_identical": True,
        "record_sha256": _require(record, "record_sha256", what="D0 记录"),
    }


def load_d0_record(campaign_root: str | Path) -> Dict[str, Any]:
    """读取 D0 fresh 记录并复算封存（供后续阶段复用，禁止另立读取口径）。"""
    record = read_json(Path(campaign_root) / "registered" / D0_FRESH_FILENAME, what="D0 fresh 记录")
    verify_seal(record, what="D0 fresh 记录")
    return record
