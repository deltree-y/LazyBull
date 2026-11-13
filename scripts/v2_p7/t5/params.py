# -*- coding: utf-8 -*-
"""P7 T5 冻结参数：配额比率、T4 完成根引用、原始目标来源与旧面身份。

原则
====
- T5 **不改** T4 语义：T4 参数（T2.5 输入身份、全窗行情身份、成本快照、受审旧面 SHA）
  一律经 ``scripts.v2_p7.t4.params.build_params`` 派生，本模块只**追加** T5 自己的
  冻结量（配额比率 ρ、T4 完成根引用、T4 模块逐件 SHA），不另立第二套口径；
- 原始目标 ``G`` 的唯一来源是 T2.5 已收口的 ``targets/daily_{arm}.parquet``
  的 ``target_notional`` 列（合成账户级原始目标 = 各相位之和）：不改写逐袖归一化
  权重对象，也不在 T5 侧重算权重的任何分量；
- 任一冻结量缺件、字节不符 => fail-fast（``ParamsError``），校验发生在任何写盘之前。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from scripts.v2_p7.common import REPO_ROOT, sha256_file
from scripts.v2_p7.t4.params import (
    T4Params,
)
from scripts.v2_p7.t4.params import build_params as build_t4_params
from scripts.v2_p7.t4.params import params_digest as t4_params_digest

#: T4 已收口完成根（只读引用；本 task 不覆盖、不重跑）
T4_ROOT_REL = "data/paper_shadow/p7_testbed/p7_t4_20261112_r31"

#: T5 测试床工作根（唯一写域）
TESTBED_REL = "data/paper_shadow/p7_testbed"

#: 默认 run 身份（整改一律新 run_id 重取，不覆盖任何完成根）
DEFAULT_RUN_ID = "p7_t5_20261113_r1"

#: 冻结配额比率（任务书 §3：``(ρ_V20, ρ_V10) = (0.7, 0.3)``，合计 1.0 = 物理账户资本份额）
QUOTA_RATIO: Mapping[str, float] = {"v20": 0.7, "v10": 0.3}

#: 配额诊断 schema
QUOTA_SCHEMA = "p7_t5_quota/v1"

#: 借用台账 schema
CREDIT_SCHEMA = "p7_t5_credit/v1"

#: 逐袖资产负债 schema
SLEEVE_BALANCE_SCHEMA = "p7_t5_sleeve_balance/v1"

#: T5 版本化状态信封（**不得**与 ``p7_t4_state/v2`` 混用）
T5_ENVELOPE_SCHEMA = "p7_t5_state/v1"

#: 臂：VV-H（同配额、禁 credit 参照）/ VV-S（允许 credit）
ARM_VVH = "VV-H"
ARM_VVS = "VV-S"
#: 机械互证臂：与 T4 完成根逐位一致的模式（quota=off & credit=off）
ARM_T4_PARITY = "T4-parity"

#: 模式标签（``quota`` / ``credit`` 两个独立开关）
MODE_OFF = "off"
MODE_ON = "on"

#: T4 模块（本 task 禁改语义；逐件 SHA 进入 ``legacy_sha.json``）
T4_MODULE_GLOB = "scripts/v2_p7/t4"

#: T4 完成根必须存在的登记产物（缺件即拒绝做互证）
T4_REQUIRED_ARTIFACTS: Tuple[str, ...] = (
    "manifest.json",
    "state_matrix.json",
    "dual_caliber.json",
    "identity.json",
    "daily.parquet",
    "trades.parquet",
    "conservation.parquet",
    "fees_by_key.parquet",
)


class ParamsError(RuntimeError):
    """T5 冻结输入缺件、字节不符或与冻结口径不一致（fail-fast，零副作用）。"""


@dataclass(frozen=True)
class T4Reference:
    """T4 完成根身份（只读引用；逐件 SHA 登记，写入前核验）。"""

    root: Path
    artifacts: Mapping[str, str]

    def payload(self) -> Dict[str, Any]:
        return {
            "root": T4_ROOT_REL,
            "artifacts": dict(sorted(self.artifacts.items())),
            "artifact_count": len(self.artifacts),
        }


@dataclass(frozen=True)
class T5Params:
    """T5 冻结参数 = T4 全部冻结输入（委托） + T5 追加的冻结量。"""

    t4: T4Params
    quota_ratio: Mapping[str, float]
    t4_reference: T4Reference
    t4_module_sha: Mapping[str, Any]
    #: 账户级原始目标 ``{交易日: G}``（= 各相位 ``target_notional`` 之和）
    account_target: Mapping[str, float]
    #: 逐相位原始目标 ``{交易日: {arm: target_notional}}``
    arm_target: Mapping[str, Mapping[str, float]]

    # ── T4 读取面委托（宿主/编排器只读这些属性） ──
    @property
    def repo_root(self) -> Path:
        return self.t4.repo_root

    @property
    def data_root(self) -> Path:
        return self.t4.data_root

    @property
    def t25(self) -> Any:
        return self.t4.t25

    @property
    def calendar(self) -> Tuple[str, ...]:
        return self.t4.calendar

    @property
    def runoff_days(self) -> Tuple[str, ...]:
        return self.t4.runoff_days

    @property
    def extended_days(self) -> Tuple[str, ...]:
        return self.t4.extended_days

    @property
    def arms(self) -> Tuple[str, ...]:
        return self.t4.arms

    @property
    def physical_capital(self) -> float:
        return float(self.t4.physical_capital)

    @property
    def sleeve_capital(self) -> Mapping[str, float]:
        return dict(self.t4.sleeve_capital)

    @property
    def lot_size(self) -> int:
        return int(self.t4.lot_size)

    @property
    def cost_settings(self) -> Mapping[str, float]:
        return dict(self.t4.cost_settings)

    @property
    def market_identity(self) -> Mapping[str, Any]:
        return dict(self.t4.market_identity)

    @property
    def legacy_sha(self) -> Mapping[str, Any]:
        return dict(self.t4.legacy_sha)

    def split_of(self, day: str) -> int:
        return int(self.t4.split_of(day))

    def model_of(self, day: str) -> int:
        return int(self.t4.model_of(day))

    # ── T5 追加读取面 ──
    def ratio_of(self, sleeve_id: str) -> float:
        """该袖的冻结配额比率 ρ_i（未登记即拒绝，不静默补 0/1）。"""
        key = str(sleeve_id)
        if key not in self.quota_ratio:
            raise ParamsError(f"袖 {key} 无冻结配额比率 ρ（登记={sorted(self.quota_ratio)}）")
        return float(self.quota_ratio[key])

    def target_of(self, day: str) -> float:
        """账户级原始目标 G（当日各相位 ``target_notional`` 之和；缺日即 0）。"""
        return float(self.account_target.get(str(day), 0.0))

    def arm_target_of(self, day: str, arm: str) -> float:
        return float((self.arm_target.get(str(day)) or {}).get(str(arm), 0.0))

    def params_payload(self) -> Dict[str, Any]:
        return {
            "t4_params": self.t4.params_payload(),
            "t4_params_digest": t4_params_digest(self.t4),
            "quota_ratio": {str(key): float(value) for key, value in self.quota_ratio.items()},
            "quota_ratio_sum": float(sum(self.quota_ratio.values())),
            "t4_reference": self.t4_reference.payload(),
            "t4_module_sha": {
                str(key): str((value or {}).get("sha256") or "")
                for key, value in sorted(self.t4_module_sha.items())
            },
            "account_target_days": len(self.account_target),
            "account_target_first_last": [
                float(self.account_target.get(self.extended_days[0], 0.0)),
                float(self.account_target.get(self.extended_days[-1], 0.0)),
            ],
            "envelope_schema": T5_ENVELOPE_SCHEMA,
            "quota_schema": QUOTA_SCHEMA,
            "credit_schema": CREDIT_SCHEMA,
            "arms": {"vvh": ARM_VVH, "vvs": ARM_VVS, "t4_parity": ARM_T4_PARITY},
            "q3_ruling": "Q3=A（复用 T4 实现，本 task 不改费用口径）",
            "authorization": (
                "A^H_i = max{a <= min(D_i, R_i): a + Fee(a) <= cash_open_i}；"
                "A^S_i(L) = max{a <= min(D_i, R_i + L): a + Fee(a) <= cash_open_i + L}"
            ),
            "credit_sign": "借入为正、出借为负；E_i = cash_i + MV_i - credit_i（本金转移零 P&L）",
        }


def quota_ratio() -> Dict[str, float]:
    """冻结配额比率 ρ（``(ρ_V20, ρ_V10) = (0.7, 0.3)``，合计 1.0 = 资本份额）。"""
    return {str(key): float(value) for key, value in QUOTA_RATIO.items()}


def params_digest(params: T5Params) -> str:
    """T5 参数快照稳定摘要（续跑/证据身份）。"""
    import hashlib

    text = json.dumps(params.params_payload(), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_t4_reference(repo_root: Path) -> T4Reference:
    """登记 T4 完成根逐件 SHA（缺件即拒绝，不伪造互证基准）。"""
    root = Path(repo_root) / T4_ROOT_REL
    if not root.exists():
        raise ParamsError(f"T4 完成根不存在：{root}")
    artifacts: Dict[str, str] = {}
    missing: List[str] = []
    for name in T4_REQUIRED_ARTIFACTS:
        path = root / name
        if not path.exists():
            missing.append(name)
            continue
        artifacts[name] = sha256_file(path)
    if missing:
        raise ParamsError(f"T4 完成根缺件（禁止以缺件做互证）：{missing}")
    return T4Reference(root=root, artifacts=artifacts)


def t4_module_identity(repo_root: Path) -> Dict[str, Any]:
    """``scripts/v2_p7/t4/**`` 逐件 SHA + 字节（证明 T5 未改 T4 语义）。"""
    base = Path(repo_root) / T4_MODULE_GLOB
    if not base.exists():
        raise ParamsError(f"T4 模块目录缺件：{base}")
    rows: Dict[str, Any] = {}
    for path in sorted(base.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(Path(repo_root)).as_posix()
        rows[rel] = {"sha256": sha256_file(path), "bytes": int(path.stat().st_size)}
    if not rows:
        raise ParamsError("T4 模块目录为空（不可能用于互证）")
    return rows


def load_target_notional(
    root: Path, arms: Sequence[str]
) -> Tuple[Dict[str, Mapping[str, float]], Dict[str, float]]:
    """读取 T2.5 ``targets/daily_{arm}.parquet`` 的 ``target_notional``（原始目标 G）。

    Returns:
        ``(逐相位 {day: {arm: value}}, 账户级 {day: Σ arm})``。

    Raises:
        ParamsError: 缺件、缺列或数值非法（不静默补 0）。
    """
    import pandas as pd

    per_arm: Dict[str, Dict[str, float]] = {str(arm): {} for arm in arms}
    for arm in arms:
        path = Path(root) / "targets" / f"daily_{arm}.parquet"
        if not path.exists():
            raise ParamsError(f"T2.5 逐日帧缺件：{path}")
        frame = pd.read_parquet(path)
        for column in ("trade_date", "target_notional"):
            if column not in frame.columns:
                raise ParamsError(f"{path} 缺列 {column}（禁止重算目标口径）")
        for row in frame.itertuples(index=False):
            value = float(getattr(row, "target_notional"))
            if value < 0.0:
                raise ParamsError(f"{path} 的 target_notional 为负：{value!r}")
            per_arm[str(arm)][str(row.trade_date)] = value
    days = sorted({day for values in per_arm.values() for day in values})
    if not days:
        raise ParamsError("T2.5 逐日帧为空（原始目标不可派生）")
    by_day: Dict[str, Mapping[str, float]] = {
        day: {arm: float(per_arm[arm].get(day, 0.0)) for arm in arms} for day in days
    }
    account: Dict[str, float] = {day: float(sum(by_day[day][arm] for arm in arms)) for day in days}
    return by_day, account


def build_params(
    repo_root: Path = REPO_ROOT,
    accept_cost_drift: bool = False,
    accept_input_drift: bool = False,
) -> T5Params:
    """组装并校验 T5 全部冻结输入（任何不符即 fail-fast，零副作用）。"""
    t4 = build_t4_params(
        repo_root=repo_root,
        accept_cost_drift=accept_cost_drift,
        accept_input_drift=accept_input_drift,
    )
    root = Path(t4.repo_root)
    reference = load_t4_reference(root)
    per_arm, account = load_target_notional(t4.t25.root, t4.arms)
    missing = [arm for arm in t4.arms if arm not in QUOTA_RATIO]
    if missing:
        raise ParamsError(f"配额比率未覆盖全部袖：{missing}（登记={sorted(QUOTA_RATIO)}）")
    if abs(sum(QUOTA_RATIO[arm] for arm in t4.arms) - 1.0) > 1e-12:
        raise ParamsError(
            "配额比率之和必须为 1.0（资本份额口径）："
            f"{sum(QUOTA_RATIO[arm] for arm in t4.arms)!r}"
        )
    return T5Params(
        t4=t4,
        quota_ratio={arm: float(QUOTA_RATIO[arm]) for arm in t4.arms},
        t4_reference=reference,
        t4_module_sha=t4_module_identity(root),
        account_target=account,
        arm_target={day: dict(values) for day, values in per_arm.items()},
    )
