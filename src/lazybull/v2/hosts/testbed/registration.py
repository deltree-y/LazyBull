# -*- coding: utf-8 -*-
"""P5b 试车场模拟臂注册（campaign + arm 身份台账）。

设计 v4 §4.2/§7.1 的落地：

- campaign 记录 ``campaign_id`` / 行域声明 / 虚拟时钟（含日历身份）/ input+code+window-state 身份；
- arm 记录 ``arm_id`` / ``arm_role`` / config+model+code+input+window-state 身份；
- **重复注册一律拒绝**：注册根以 ``mkdir(exist_ok=False)`` 独占创建，记录以
  ``O_CREAT|O_EXCL`` 独占落盘，先查后写的竞态窗口被内核裁决；
- **未注册臂不得初始化**：``load_arm`` 缺失即抛 ``UnregisteredArmError``；
- **不注册真实候选**：角色白名单只含模拟臂四类，真实候选/前向影子角色显式拒绝。

评审处置：注册记录一律封存 ``record_sha256`` 并在读取时复算（R2-T1-05）；臂注册
补齐写域守卫（R1-T1-05/R2-T1-01）、半初始化现场显式拒绝（R2-T1-04）、代码态漂移
门改用执行闭包（R2-T1-13）、数据态漂移门改用窗口内容（R1-T1-08/R2-T1-12）。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from src.lazybull.v2.hosts.paper.config_adapter import load_paper_config, resolve_engine
from src.lazybull.v2.hosts.testbed.clock import (
    CONTIGUITY_VACUOUS,
    VirtualClock,
    build_virtual_clock,
    calendar_identity,
)
from src.lazybull.v2.hosts.testbed.identity import (
    collect_code_identity,
    collect_config_identity,
    collect_data_state_identity,
    collect_input_identity,
    collect_model_identity,
    identity_summary,
)
from src.lazybull.v2.hosts.testbed.paths import (
    ARM_ROLES,
    EVIDENCE_CLASS,
    SCOPE,
    TestbedError,
    assert_write_scope,
    resolve_campaign_root,
    validate_campaign_id,
)
from src.lazybull.v2.hosts.testbed.storage import (
    raise_rollback_failure,
    read_json,
    seal_record,
    verify_seal,
    write_json_atomic,
    write_json_exclusive,
)

__all__ = [
    "ARM_SCHEMA",
    "CAMPAIGN_SCHEMA",
    "DuplicateRegistrationError",
    "UnregisteredArmError",
    "list_arms",
    "load_arm",
    "load_campaign",
    "register_arm",
    "register_campaign",
]

#: campaign 注册记录 schema
CAMPAIGN_SCHEMA = "p5b_testbed_campaign/v1"
#: arm 注册记录 schema
ARM_SCHEMA = "p5b_testbed_arm/v1"
#: 历史行域：机械层只允许 structural（设计 v4 §4.1）
ROW_DOMAIN_STRUCTURAL = "structural"
#: 试车场只复用 v2 纸面宿主（P4 生产自 2026-10-31 起 engine=v2）
REQUIRED_ENGINE = "v2"
#: 显式拒绝的“真实候选/前向影子/生产”角色词（不属 P5b 试车场）
_FORBIDDEN_ARM_ROLES = frozenset(
    {
        "real_candidate",
        "candidate",
        "real",
        "production",
        "prod",
        "forward_shadow",
        "shadow",
        "live",
        "real_forward",
    }
)


class DuplicateRegistrationError(TestbedError):
    """重复注册（campaign 或 arm 已存在）。"""


class UnregisteredArmError(TestbedError):
    """未注册臂不得初始化/不得参与试车场。"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _campaign_record_path(campaign_root: Path) -> Path:
    return Path(campaign_root) / "registered" / "campaign.json"


def _arm_dir(campaign_root: Path, arm_id: str) -> Path:
    return Path(campaign_root) / "registered" / "arms" / arm_id


def _validate_arm_role(arm_role: str) -> str:
    role = str(arm_role).strip().lower()
    if role in _FORBIDDEN_ARM_ROLES:
        raise TestbedError(
            f"禁止注册真实候选/前向影子/生产角色: {arm_role!r}；"
            f"P5b 试车场只允许模拟臂 {list(ARM_ROLES)}"
        )
    if role not in ARM_ROLES:
        raise TestbedError(f"未知模拟臂角色: {arm_role!r}（允许 {list(ARM_ROLES)}）")
    return role


def _contiguity_payload(clock: VirtualClock, calendar: Optional[Sequence[str]]) -> Dict[str, Any]:
    """连续性证据：多日窗口必须来自交易日历，单日窗口显式标为真空连续。"""
    if calendar is None:
        return {
            "verified": True,
            "basis": CONTIGUITY_VACUOUS,
            "calendar": None,
            "note": "单日窗口无相邻关系，连续性真空成立",
        }
    return {
        "verified": True,
        "basis": "trading_calendar",
        "calendar": calendar_identity(calendar),
        "window_days": list(clock.days),
    }


def _declaration_block() -> Dict[str, Any]:
    """campaign 级声明字段：行域与标签截止（不得以"已核验"形态呈现）。"""
    return {
        "row_domain": ROW_DOMAIN_STRUCTURAL,
        "row_domain_verified": False,
        "row_domain_receipt": None,
        "row_domain_note": (
            "structural 为声明；逐日行域回执由 mechanical replay 产出并核验，"
            "T1 不冒充回执（设计 §4.1(1)）"
        ),
        "label_cutoff": {
            "value": None,
            "status": "not_applicable_t1",
            "reason": "T1 只建起点与身份，不消费标签；标签截止属统计层登记（设计 §7.2）",
        },
    }


def register_campaign(
    *,
    campaign_id: str,
    d0_date: str,
    virtual_days: Optional[Sequence[str]] = None,
    calendar: Optional[Sequence[str]] = None,
    data_root: Optional[str] = None,
    base_root: Optional[str] = None,
    repo_root: Optional[str] = None,
    allow_external_root: bool = False,
) -> Dict[str, Any]:
    """登记一个历史试车场 campaign（重复登记直接拒绝）。

    Args:
        campaign_id: campaign 标识（小写/数字/下划线/连字符）。
        d0_date: 历史 D0（估值日，不产生收益样本）。
        virtual_days: 已登记虚拟日（含 D0，缺省只有 D0）。**多日窗口必须连续**。
        calendar: 完整交易日历；多日窗口必填（否则"不跳日"无法校验）。
        data_root: 只读历史输入根（缺省 = 配置 data 根）。
        base_root: 试车场根（缺省 ``{data_root}/paper_shadow/p5b_testbed``）。
        repo_root: 代码态采集根（缺省仓库根）。
        allow_external_root: 仅测试/临时脚本放开外部根（不解除生产根禁令）。

    Returns:
        落盘的 campaign 记录（含 input/code/window-state 身份）。

    Raises:
        DuplicateRegistrationError: campaign 根已存在。
        TestbedError: 标识/虚拟时钟非法、多日窗口缺日历、数据分区缺失。
    """
    validate_campaign_id(campaign_id)
    campaign_root = resolve_campaign_root(
        campaign_id,
        base_root=base_root,
        data_root=data_root,
        allow_external_root=allow_external_root,
    )
    if campaign_root.exists():
        raise DuplicateRegistrationError(
            f"campaign 重复注册（根已存在，禁止覆盖或续用）: {campaign_root}"
        )
    resolved_data_root = _resolve_data_root(data_root)
    clock = build_virtual_clock(
        d0_date=d0_date,
        virtual_days=virtual_days,
        calendar=calendar,
        require_calendar=True,
    )
    resolved_repo_root = str(Path(repo_root).resolve()) if repo_root else _repo_root()
    window_state = collect_data_state_identity(
        data_root=resolved_data_root, virtual_days=clock.days
    )
    input_identity = collect_input_identity(
        data_root=resolved_data_root,
        row_domain=ROW_DOMAIN_STRUCTURAL,
        d0_date=clock.d0_date,
        virtual_days=clock.days,
        window_state_id=window_state["window_state_id"],
    )
    code_identity = collect_code_identity(resolved_repo_root)
    identity: Dict[str, Any] = {
        "input": input_identity,
        "config": None,
        "model": None,
        "code": code_identity,
        "window_state": window_state,
    }
    record = seal_record(
        {
            "schema": CAMPAIGN_SCHEMA,
            "campaign_id": campaign_id,
            "scope": SCOPE,
            "evidence_class": EVIDENCE_CLASS,
            "virtual_clock": True,
            "not_forward_evidence": True,
            "not_sla_evidence": True,
            "not_real_candidate": True,
            **_declaration_block(),
            "clock": {**clock.to_dict(), "contiguity": _contiguity_payload(clock, calendar)},
            "created_at": _now(),
            "campaign_root": str(campaign_root).replace("\\", "/"),
            "repo_root": _normalize_root(resolved_repo_root),
            "identity": identity,
            "identity_summary": identity_summary(identity),
        }
    )
    try:
        campaign_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise DuplicateRegistrationError(
            f"campaign 重复注册（根已存在，禁止覆盖或续用）: {campaign_root}"
        ) from exc
    try:
        record_dir = campaign_root / "registered"
        record_dir.mkdir(parents=True, exist_ok=False)
        write_json_exclusive(record_dir / "campaign.json", record)
    except FileExistsError as exc:
        raise_rollback_failure(
            [campaign_root],
            DuplicateRegistrationError(
                f"campaign 重复注册（并发写入被独占创建拒绝）: {campaign_root}（{exc}）"
            ),
        )
    except Exception as exc:  # noqa: BLE001 - 统一转回滚后重抛，不吞异常
        raise_rollback_failure([campaign_root], exc)
    return record


def load_campaign(campaign_root: str | Path) -> Dict[str, Any]:
    """读取并校验 campaign 注册记录（未注册 ⇒ fail-fast）。"""
    root = Path(campaign_root)
    path = _campaign_record_path(root)
    if not path.is_file():
        raise UnregisteredArmError(
            f"campaign 未注册（缺 {path}）；禁止在未登记 campaign 上初始化臂"
        )
    record = read_json(path, what="campaign 注册记录")
    verify_seal(record, what=f"campaign 注册记录（{path}）")
    if record.get("schema") != CAMPAIGN_SCHEMA:
        raise TestbedError(
            f"campaign 注册记录 schema 非法: {record.get('schema')!r}（期望 {CAMPAIGN_SCHEMA}）"
        )
    if record.get("campaign_id") != root.name:
        raise TestbedError(
            f"campaign 注册记录与目录不一致: {record.get('campaign_id')!r} vs {root.name!r}"
        )
    if record.get("scope") != SCOPE or record.get("evidence_class") != EVIDENCE_CLASS:
        raise TestbedError(
            "campaign 记录范围/证据类别被篡改（只允许 "
            f"scope={SCOPE} / evidence_class={EVIDENCE_CLASS}）: "
            f"{record.get('scope')!r} / {record.get('evidence_class')!r}"
        )
    recorded_root = str(record.get("campaign_root") or "").replace("\\", "/")
    actual_root = str(root.resolve()).replace("\\", "/")
    if recorded_root != actual_root:
        raise TestbedError(
            f"campaign 记录根与目录不一致（禁止搬移/复制记录续用）: "
            f"{recorded_root} vs {actual_root}"
        )
    return record


def load_arm(campaign_root: str | Path, arm_id: str) -> Dict[str, Any]:
    """读取单臂注册记录；未注册臂抛 ``UnregisteredArmError``。"""
    campaign = load_campaign(campaign_root)
    path = _arm_dir(Path(campaign_root), arm_id) / "registration.json"
    if not path.is_file():
        raise UnregisteredArmError(
            f"臂 {arm_id!r} 未在 campaign {campaign['campaign_id']!r} 注册；" "未注册臂不得初始化"
        )
    record = read_json(path, what=f"臂 {arm_id!r} 注册记录")
    verify_seal(record, what=f"臂 {arm_id!r} 注册记录（{path}）")
    if record.get("schema") != ARM_SCHEMA:
        raise TestbedError(f"臂注册记录 schema 非法: {record.get('schema')!r}（期望 {ARM_SCHEMA}）")
    if record.get("arm_id") != arm_id or record.get("campaign_id") != campaign["campaign_id"]:
        raise TestbedError(
            f"臂注册记录身份不一致: arm={record.get('arm_id')!r} "
            f"campaign={record.get('campaign_id')!r}（请求 {arm_id!r}）"
        )
    _validate_arm_role(record.get("arm_role"))
    return record


def list_arms(campaign_root: str | Path) -> Dict[str, Dict[str, Any]]:
    """列出已注册臂（按 arm_id 排序；无臂返回空 dict）。"""
    root = Path(campaign_root) / "registered" / "arms"
    if not root.is_dir():
        return {}
    return {
        path.name: load_arm(campaign_root, path.name)
        for path in sorted(root.iterdir())
        if path.is_dir()
    }


def register_arm(
    *,
    campaign_root: str | Path,
    arm_id: str,
    arm_role: str,
    config_flat: Optional[Mapping[str, Any]] = None,
    config_source: Optional[str | Path] = None,
    models_dir: Optional[str] = None,
    data_root: Optional[str] = None,
    repo_root: Optional[str] = None,
    note: Optional[str] = None,
    allow_external_root: bool = False,
) -> Dict[str, Any]:
    """登记一个模拟臂（重复登记直接拒绝，身份必须与 campaign 一致）。

    Args:
        campaign_root: 已登记 campaign 根。
        arm_id: 臂标识（同 campaign_id 形态）。
        arm_role: ``aa_clone`` / ``mechanical_perturbation`` /
            ``synthetic_effect`` / ``fault_injection``。
        config_flat: 已解析的扁平配置（与 ``config_source`` 二选一）。
        config_source: 含 ``config.yaml`` 的纸面根（只读，经 ``load_paper_config``
            按 P4 分组模板解析，禁止本模块另写解析规则）。
        models_dir: 选股模型家族目录（缺省 ``{data_root}/models/stock_selection``）。
        data_root: 与 campaign 一致的输入根（不一致即拒绝，防跨数据态拼臂）。
        repo_root: 代码态采集根（缺省仓库根）。
        note: 臂用途备注（写入注册台账）。
        allow_external_root: 仅测试/临时脚本放开外部根。

    Returns:
        落盘的臂注册记录。
    """
    validate_campaign_id(arm_id, name="arm_id")
    role = _validate_arm_role(arm_role)
    root = Path(campaign_root)
    campaign = load_campaign(root)
    campaign_data_root = str((campaign.get("identity") or {})["input"]["data_root"])
    assert_write_scope(root, data_root=campaign_data_root, allow_external_root=allow_external_root)
    if (root / "registered" / "d0_fresh.json").is_file():
        raise DuplicateRegistrationError(
            f"campaign {campaign['campaign_id']!r} 的 D0 已初始化，禁止事后新增臂: {arm_id}"
        )
    arm_dir = _arm_dir(root, arm_id)
    _assert_arm_dir_free(arm_dir, arm_id)
    flat = _resolve_flat_config(config_flat, config_source)
    engine = resolve_engine(flat)
    if engine != REQUIRED_ENGINE:
        raise TestbedError(
            f"臂 {arm_id!r} 的 engine={engine!r} 不受支持：P5b 试车场只复用 v2 纸面宿主"
            f"（{REQUIRED_ENGINE}）"
        )
    _assert_same_data_root(data_root, campaign_data_root, arm_id)
    clock = _campaign_clock(campaign)
    # 数据态/代码态漂移门在前置核验，避免同一 campaign 混入多实现/多数据态
    window_state = collect_data_state_identity(
        data_root=campaign_data_root, virtual_days=clock.days
    )
    recorded_window = (campaign["identity"].get("window_state") or {}).get("window_state_id")
    if window_state["window_state_id"] != recorded_window:
        raise TestbedError(
            "窗口数据态漂移：campaign 登记后窗口内历史分区内容已变化，禁止新增臂: "
            f"{window_state['window_state_id']} vs {recorded_window}"
        )
    campaign_repo_root = _normalize_root(campaign.get("repo_root") or _repo_root())
    if repo_root is not None:
        provided_repo = str(Path(repo_root).resolve()).replace("\\", "/")
        if provided_repo != campaign_repo_root:
            raise TestbedError(
                f"臂 {arm_id!r} 的 repo_root 与 campaign 不一致（禁止跨代码态拼臂）: "
                f"{provided_repo} vs {campaign_repo_root}"
            )
    code_identity = collect_code_identity(campaign_repo_root)
    recorded_code = (campaign["identity"].get("code") or {}).get("execution_digest")
    if code_identity["execution_digest"] != recorded_code:
        raise TestbedError(
            "执行闭包代码态漂移：campaign 登记后试车场/复用件源码已变化，禁止新增臂: "
            f"{code_identity['execution_digest']} vs {recorded_code}"
        )
    config_identity = collect_config_identity(flat)
    model_identity = collect_model_identity(
        flat, models_dir=models_dir or _models_root(campaign_data_root), d0_date=clock.d0_date
    )
    input_identity = collect_input_identity(
        data_root=campaign_data_root,
        row_domain=str(campaign["row_domain"]),
        d0_date=clock.d0_date,
        virtual_days=clock.days,
        window_state_id=window_state["window_state_id"],
    )
    identity = {
        "input": input_identity,
        "config": config_identity,
        "model": model_identity,
        "code": code_identity,
        "window_state": window_state,
    }
    record = seal_record(
        {
            "schema": ARM_SCHEMA,
            "campaign_id": campaign["campaign_id"],
            "arm_id": arm_id,
            "arm_role": role,
            "scope": SCOPE,
            "evidence_class": EVIDENCE_CLASS,
            "virtual_clock": True,
            "not_forward_evidence": True,
            "not_sla_evidence": True,
            "not_real_candidate": True,
            "root": f"arms/{arm_id}",
            "registered_at": _now(),
            "note": note,
            "identity": identity,
            "identity_summary": identity_summary(identity),
        }
    )
    try:
        arm_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise DuplicateRegistrationError(
            f"臂重复注册（并发写入被独占创建拒绝，{arm_id!r}）: {arm_dir}（{exc}）"
        ) from exc
    try:
        write_json_atomic(arm_dir / "config.json", dict(flat))
        # registration.json 为提交点：独占创建成功即视为注册完成
        write_json_exclusive(arm_dir / "registration.json", record)
    except Exception as exc:  # noqa: BLE001 - 统一转回滚后重抛，不吞异常
        raise_rollback_failure([arm_dir], exc)
    return record


def _assert_arm_dir_free(arm_dir: Path, arm_id: str) -> None:
    """臂目录必须不存在；残留无注册记录的目录属半初始化现场，显式拒绝。"""
    if not arm_dir.exists():
        return
    if (arm_dir / "registration.json").is_file():
        raise DuplicateRegistrationError(f"臂重复注册（{arm_id!r} 已有注册记录）: {arm_dir}")
    leftover = sorted(path.name for path in arm_dir.iterdir())
    raise TestbedError(
        f"臂目录已存在但无注册记录（半初始化现场，禁止覆盖或跳过）: {arm_dir}；"
        f"残留={leftover}；请人工核对后清理该目录再重试"
    )


def _assert_same_data_root(data_root: Optional[str], campaign_data_root: str, arm_id: str) -> None:
    if data_root is None:
        return
    provided = _normalize_root(data_root)
    if provided != campaign_data_root:
        raise TestbedError(
            f"臂 {arm_id!r} 的 data_root 与 campaign 不一致（禁止跨数据态拼臂）: "
            f"{provided} vs {campaign_data_root}"
        )


def _campaign_clock(campaign: Mapping[str, Any]) -> VirtualClock:
    payload = campaign["clock"]
    return VirtualClock(
        d0_date=str(payload["d0_date"]),
        days=tuple(str(day) for day in payload["days"]),
    )


def _resolve_flat_config(
    config_flat: Optional[Mapping[str, Any]], config_source: Optional[str | Path]
) -> Dict[str, Any]:
    if (config_flat is None) == (config_source is None):
        raise TestbedError("config_flat 与 config_source 必须二选一（两者同传或同缺都拒绝）")
    if config_flat is not None:
        if not isinstance(config_flat, Mapping):
            raise TestbedError(f"config_flat 必须是 Mapping，实际 {type(config_flat).__name__}")
        return dict(config_flat)
    source_root = Path(config_source)
    if not source_root.is_dir():
        raise TestbedError(
            f"config_source 必须是含 config.yaml 的纸面根（只读），实际不是目录: {source_root}"
        )
    return load_paper_config(source_root)


def _normalize_root(value: Any) -> str:
    """路径归一（绝对化 + 正斜杠）：落盘与比较必须同口径。"""
    return str(Path(str(value)).resolve()).replace("\\", "/")


def _resolve_data_root(data_root: Optional[str]) -> str:
    if data_root:
        return str(Path(data_root).resolve())
    from src.lazybull.common.config import get_data_root

    return str(Path(get_data_root()).resolve())


def _models_root(data_root: str) -> str:
    from src.lazybull.common.config import get_stock_selection_models_root

    return get_stock_selection_models_root(data_root)


def _repo_root() -> str:
    from src.lazybull.v2.hosts.testbed.paths import project_root

    return str(project_root())
