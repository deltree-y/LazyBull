# -*- coding: utf-8 -*-
"""P7 T6 冻结参数：T5 冻结面委托 + T5 完成根身份 + T5 模块 SHA + 预登记摘要。

原则
====
- T6 **不新增任何口径**：初资、袖资本、目标、成本、日历、执行语义一律经
  ``scripts.v2_p7.t5.params.build_params``（再委托 T4/T2.5）派生；
- T6 只**追加**两件冻结量：① T5 完成根 ``p7_t5_20261113_r15`` 的逐件 SHA（证明上游未改）；
  ② ``scripts/v2_p7/t5/**`` 与 ``scripts/v2_p7/t4/**`` 的逐件 SHA（证明本 task 只读复用）；
- 任一缺件、字节不符、身份漂移 => ``ParamsError``（fail-fast，先于任何写盘）。

登记缺口（G1 相关，如实登记）
==========================
T5 完成根只登记了 ``scripts/v2_p7/t4/**`` 与受审旧面/T2.5 的逐件 SHA，**未登记自身**
``scripts/v2_p7/t5/**`` 的 SHA。T6 的对策：① 用 T5 根 ``manifest.outputs`` 逐件自洽核验
证明该根未被修改；② 由 T6 首采 T5 模块 SHA 并写入证据（作为后续 task 的基线），
不声称是「与 T5 登记值比对通过」。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from scripts.v2_p7.common import REPO_ROOT, sha256_file
from scripts.v2_p7.t5.params import T5Params
from scripts.v2_p7.t5.params import build_params as build_t5_params
from scripts.v2_p7.t5.params import params_digest as t5_params_digest
from scripts.v2_p7.t6 import calibration

#: T5 已收口完成根（只读引用；本 task 不覆盖、不重跑）
T5_ROOT_REL = "data/paper_shadow/p7_testbed/p7_t5_20261113_r15"

#: 测试床工作根（唯一写域，与 T4/T5 同）
TESTBED_REL = "data/paper_shadow/p7_testbed"

#: 默认 run 身份（整改一律新 run_id 重取，不覆盖任何完成根）
DEFAULT_RUN_ID = "p7_t6_20261113_r1"

#: T5 完成根必须登记的 schema / 状态
T5_REQUIRED_SCHEMA = "p7_t5/v1"

#: T4/T5 模块目录（只读复用；逐件 SHA 进入 identity.json）
T4_MODULE_GLOB = "scripts/v2_p7/t4"
T5_MODULE_GLOB = "scripts/v2_p7/t5"
T6_MODULE_GLOB = "scripts/v2_p7/t6"

#: T2.5 输入身份需逐项比对的键（与 ``t25.payload()`` 同名）
T25_SHA_KEYS: Tuple[str, ...] = (
    "manifest_sha256",
    "intent_precheck_sha256",
    "calendar_sha256",
    "plans_sha256",
    "holdings_sha256",
    "daily_sha256",
)

#: T6 追加的 schema
T6_IDENTITY_SCHEMA = "p7_t6_identity/v1"


class ParamsError(RuntimeError):
    """冻结输入缺件、字节不符或与冻结口径不一致（fail-fast，零副作用）。"""


@dataclass(frozen=True)
class T5Reference:
    """T5 完成根身份（只读引用；逐件 SHA 与自洽核验结果）。"""

    root: Path
    manifest_sha256: str
    artifacts: Mapping[str, str]
    self_check: Mapping[str, Any]

    def payload(self) -> Dict[str, Any]:
        return {
            "root": T5_ROOT_REL,
            "manifest_sha256": self.manifest_sha256,
            "artifacts": dict(sorted(self.artifacts.items())),
            "artifact_count": len(self.artifacts),
            "self_check": dict(self.self_check),
        }


@dataclass(frozen=True)
class T6Params:
    """T6 冻结参数 = T5 冻结参数（委托） + 完成根身份 + 模块 SHA + 预登记摘要。"""

    t5: T5Params
    t5_reference: T5Reference
    t4_module_sha: Mapping[str, Any]
    t5_module_sha: Mapping[str, Any]
    calibration_digest: str

    # ── T5 读取面委托 ──
    @property
    def repo_root(self) -> Path:
        return self.t5.repo_root

    @property
    def data_root(self) -> Path:
        return self.t5.data_root

    @property
    def t25(self) -> Any:
        return self.t5.t25

    @property
    def calendar(self) -> Tuple[str, ...]:
        return self.t5.calendar

    @property
    def runoff_days(self) -> Tuple[str, ...]:
        return self.t5.runoff_days

    @property
    def extended_days(self) -> Tuple[str, ...]:
        return self.t5.extended_days

    @property
    def arms(self) -> Tuple[str, ...]:
        return self.t5.arms

    @property
    def physical_capital(self) -> float:
        return float(self.t5.physical_capital)

    @property
    def sleeve_capital(self) -> Mapping[str, float]:
        return dict(self.t5.sleeve_capital)

    @property
    def lot_size(self) -> int:
        return int(self.t5.lot_size)

    @property
    def cost_settings(self) -> Mapping[str, float]:
        return dict(self.t5.cost_settings)

    @property
    def market_identity(self) -> Mapping[str, Any]:
        return dict(self.t5.market_identity)

    @property
    def legacy_sha(self) -> Mapping[str, Any]:
        return dict(self.t5.legacy_sha)

    def split_of(self, day: str) -> int:
        return int(self.t5.split_of(day))

    def model_of(self, day: str) -> int:
        return int(self.t5.model_of(day))

    def target_of(self, day: str) -> float:
        return float(self.t5.target_of(day))

    def arm_target_of(self, day: str, arm: str) -> float:
        return float(self.t5.arm_target_of(day, arm))

    def ratio_of(self, sleeve_id: str) -> float:
        return float(self.t5.ratio_of(sleeve_id))

    def params_payload(self) -> Dict[str, Any]:
        return {
            "t5_params": self.t5.params_payload(),
            "t5_params_digest": t5_params_digest(self.t5),
            "t5_reference": self.t5_reference.payload(),
            "t4_module_sha": {
                str(key): str((value or {}).get("sha256") or "")
                for key, value in sorted(self.t4_module_sha.items())
            },
            "t5_module_sha": {
                str(key): str((value or {}).get("sha256") or "")
                for key, value in sorted(self.t5_module_sha.items())
            },
            "calibration": {
                "schema": calibration.CALIBRATION_SCHEMA,
                "digest": str(self.calibration_digest),
                "registered_date": calibration.REGISTERED_DATE,
            },
            "arms": {"treatment": calibration.TREATMENT, "control": calibration.CONTROL},
            "alpha_consumed": True,
        }


def params_digest(params: T6Params) -> str:
    """T6 参数快照稳定摘要（两臂共用同一摘要：参数与臂无关）。"""
    import hashlib

    text = json.dumps(params.params_payload(), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def module_identity(repo_root: Path, glob: str) -> Dict[str, Any]:
    """模块目录逐件 SHA + 字节（``__pycache__`` 剔除；空目录即拒绝）。"""
    base = Path(repo_root) / glob
    if not base.exists():
        raise ParamsError(f"模块目录缺件：{base}")
    rows: Dict[str, Any] = {}
    for path in sorted(base.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(Path(repo_root)).as_posix()
        rows[rel] = {"sha256": sha256_file(path), "bytes": int(path.stat().st_size)}
    if not rows:
        raise ParamsError(f"模块目录为空（不可能用于身份核验）：{base}")
    return rows


def load_t5_reference(repo_root: Path) -> T5Reference:
    """登记 T5 完成根：manifest 状态 + ``outputs`` 逐件 SHA 自洽核验。"""
    root = Path(repo_root) / T5_ROOT_REL
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise ParamsError(f"T5 完成根缺 manifest：{manifest_path}")
    manifest = _read_json(manifest_path)
    if str(manifest.get("schema") or "") != T5_REQUIRED_SCHEMA:
        raise ParamsError(f"T5 完成根 schema 不符：{manifest.get('schema')!r}")
    if bool(manifest.get("alpha_consumed")):
        raise ParamsError("T5 完成根标记 alpha_consumed=true（与 T6 消费边界冲突）")
    if manifest.get("verdict") is not None:
        raise ParamsError(f"T5 完成根已有 verdict（不得作为 T6 基准）：{manifest.get('verdict')!r}")
    outputs = manifest.get("outputs") or {}
    if not isinstance(outputs, Mapping) or not outputs:
        raise ParamsError("T5 完成根未登记 outputs（无法逐件核验）")
    artifacts: Dict[str, str] = {"manifest.json": sha256_file(manifest_path)}
    mismatch = []
    for rel, expected in sorted(outputs.items()):
        path = root / str(rel)
        if not path.exists():
            mismatch.append({"file": str(rel), "status": "missing"})
            continue
        actual = sha256_file(path)
        artifacts[str(rel)] = actual
        if actual != str(expected):
            mismatch.append({"file": str(rel), "expected": str(expected), "actual": actual})
    extra = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and path.relative_to(root).as_posix() not in artifacts
    )
    if mismatch or extra:
        raise ParamsError(
            "T5 完成根已被改动（禁止以改动后的根做 T6 基准）："
            f"mismatch={mismatch[:3]} extra={extra[:5]}"
        )
    self_check = {
        "outputs_registered": len(outputs),
        "artifacts_verified": len(artifacts),
        "mismatch": [],
        "extra_files": [],
        "run_id": str(manifest.get("run_id") or ""),
        "created_at": str(manifest.get("created_at") or ""),
        "params_digest": str(manifest.get("params_digest") or ""),
        "arms": sorted((manifest.get("arms") or {}).keys()),
    }
    return T5Reference(
        root=root,
        manifest_sha256=sha256_file(manifest_path),
        artifacts=artifacts,
        self_check=self_check,
    )


def verify_t4_module(reference: T5Reference, current: Mapping[str, Any]) -> Dict[str, Any]:
    """T4 模块逐件 SHA 必须与 T5 完成根登记值一致（G1）。"""
    recorded = _read_json(reference.root / "legacy_sha.json").get("t4_modules") or {}
    rows: Dict[str, Any] = {}
    mismatch = []
    for rel, item in sorted(recorded.items()):
        expected = str((item or {}).get("sha256") or "")
        actual = str((current.get(rel) or {}).get("sha256") or "")
        rows[rel] = {"recorded": expected, "current": actual, "match": expected == actual}
        if expected != actual:
            mismatch.append(rel)
    missing = sorted(set(current) - set(recorded))
    if mismatch or missing:
        raise ParamsError(f"T4 模块与 T5 登记值不一致：{mismatch[:5]}；新增={missing[:5]}")
    return {
        "files": rows,
        "count": len(rows),
        "match": True,
        "source": f"{T5_ROOT_REL}/legacy_sha.json#t4_modules",
    }


def verify_t25_input(reference: T5Reference, current: Mapping[str, Any]) -> Dict[str, Any]:
    """T2.5 输入身份（manifest/precheck/calendar/plans/holdings/daily）逐项一致（G1）。"""
    manifest = _read_json(reference.root / "manifest.json")
    recorded = ((manifest.get("params") or {}).get("t4_params") or {}).get("t25_input") or {}
    if not recorded:
        raise ParamsError("T5 完成根未登记 t25_input（G1 无基准）")
    rows: Dict[str, Any] = {}
    mismatch = []
    for key in T25_SHA_KEYS:
        expected = recorded.get(key)
        actual = current.get(key)
        if isinstance(expected, Mapping) or isinstance(actual, Mapping):
            same = dict(expected or {}) == dict(actual or {})
        else:
            same = str(expected or "") == str(actual or "")
        rows[key] = {"match": bool(same)}
        if not same:
            mismatch.append(key)
    if mismatch:
        raise ParamsError(f"T2.5 输入身份与 T5 登记值不一致：{mismatch}")
    return {
        "keys": rows,
        "count": len(rows),
        "match": True,
        "root": str(recorded.get("root") or ""),
        "source": f"{T5_ROOT_REL}/manifest.json#params.t4_params.t25_input",
    }


def build_params(repo_root: Path = REPO_ROOT) -> T6Params:
    """组装并校验 T6 全部冻结输入（任何不符即 fail-fast，零副作用）。"""
    t5 = build_t5_params(repo_root=repo_root)
    root = Path(t5.repo_root)
    reference = load_t5_reference(root)
    t4_module = module_identity(root, T4_MODULE_GLOB)
    verify_t4_module(reference, t4_module)
    verify_t25_input(reference, t5.t4.t25.payload())
    return T6Params(
        t5=t5,
        t5_reference=reference,
        t4_module_sha=t4_module,
        t5_module_sha=module_identity(root, T5_MODULE_GLOB),
        calibration_digest=calibration.digest(),
    )


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ParamsError(f"冻结输入缺件：{path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:  # pragma: no cover - 防御分支
        raise ParamsError(f"冻结输入不是合法 JSON：{path}（{error}）") from error
