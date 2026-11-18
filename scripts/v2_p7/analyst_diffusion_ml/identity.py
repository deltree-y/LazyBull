"""运行身份闭环：就绪门、输入漂移复核、模型与账户证据链核验。

设计要点（对应 AD-ML-20-v1 复审第 7 条）：
- 下游阶段必须能指到一个**已发布且为 ready** 的 readiness run，不允许只在
  campaign 目录里"看起来有 spec"就继续；
- 输入 SHA 在每个数据/诊断阶段重算一次并与 R0 登记的 `source_manifest.json` 比对，
  任何漂移都阻断（不静默接受）；
- 模型与账户一律以 run 级发布件或 run 级摘要为准，不从可变的 campaign 目录取数。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from scripts.v2_p7.analyst_diffusion_ml import artifacts, sources, spec


class IdentityError(RuntimeError):
    """证据链/身份核验不通过。"""


READY_STATUSES: tuple[str, ...] = ("ready", "ready_with_limits")


def require_ready(camp: Path, run_id: str) -> dict[str, Any]:
    """读取指定 readiness run 的封存件，status 不在 ready 白名单即阻断。

    `readiness.json` 写在 run 根并被 `manifest.json` 的逐文件 SHA 覆盖，
    因此这里先核 SHA 再读状态，防止事后改写。
    """
    if not run_id:
        raise IdentityError("缺少 readiness run 身份：数据/训练/回放阶段必须显式指定")
    run_dir = Path(camp) / "runs" / run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise IdentityError(f"就绪 run 缺 manifest：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = (manifest.get("files") or {}).get("readiness.json")
    path = run_dir / "readiness.json"
    if not path.exists():
        raise IdentityError(f"就绪 run 缺 readiness.json：{path}")
    digest = artifacts.sha256_file(path)
    if str(recorded) != digest:
        raise IdentityError(f"readiness.json 与封存 SHA 不符：{recorded} != {digest}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    status = str(payload.get("status"))
    if status not in READY_STATUSES:
        raise IdentityError(f"readiness run {run_id} 状态为 {status}，不得进入下游")
    return {"run_id": run_id, "status": status, "path": str(path), "sha256": digest}


def verify_inputs(root: Path, camp: Path) -> dict[str, Any]:
    """重算输入清单并与 R0 登记的 SHA 逐项比对；任何差异即阻断。"""
    manifest_path = Path(camp) / "inputs" / "source_manifest.json"
    if not manifest_path.exists():
        raise IdentityError(f"缺 R0 输入身份：{manifest_path}")
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected: Mapping[str, Mapping[str, Any]] = {
        str(item["name"]): item for item in recorded.get("files") or []
    }
    current = {str(item["name"]): item for item in sources.inventory(Path(root))}
    drift = []
    for name, want in expected.items():
        got = current.get(name) or {}
        if str(want.get("content_sha256")) != str(got.get("content_sha256")):
            drift.append(
                {
                    "name": name,
                    "expected": want.get("content_sha256"),
                    "actual": got.get("content_sha256"),
                }
            )
    if drift:
        raise IdentityError(f"输入内容漂移 {len(drift)} 项：{drift[:3]}")
    return {
        "status": "pass",
        "files": len(expected),
        "source_manifest_sha256": artifacts.sha256_file(manifest_path),
    }


def models_digest(camp: Path) -> dict[str, str]:
    """campaign 内隔离模型注册表的逐文件 SHA。"""
    return artifacts.file_digests(Path(camp) / "models")


def require_models(camp: Path, expected: Mapping[str, Any] | None) -> dict[str, Any]:
    """当前模型注册表必须与训练阶段登记的摘要逐文件一致。"""
    if not expected:
        raise IdentityError("上游未登记模型摘要，无法证明模型与预测同源")
    current = models_digest(camp)
    want = {str(key): str(value) for key, value in dict(expected).items()}
    missing = sorted(set(want) - set(current))
    extra = sorted(set(current) - set(want))
    changed = sorted(key for key in want if key in current and want[key] != current[key])
    if missing or extra or changed:
        raise IdentityError(
            f"模型注册表与训练阶段不一致：missing={missing[:3]} extra={extra[:3]} "
            f"changed={changed[:3]}"
        )
    return {"status": "pass", "files": len(current)}


def spec_identity(camp: Path) -> dict[str, Any]:
    """当前代码规格与 campaign 冻结规格必须一致。"""
    path = Path(camp) / "specification" / "machine_spec.json"
    if not path.exists():
        raise IdentityError("缺 R0 机器规格；先跑 readiness")
    recorded = json.loads(path.read_text(encoding="utf-8"))
    current = spec.machine_spec_document(recorded.get("protocol_sha256"))
    if current["spec_sha256"] != recorded.get("spec_sha256"):
        raise IdentityError(
            "当前代码的规格摘要与 campaign 冻结值不同："
            f"recorded={recorded.get('spec_sha256')} current={current['spec_sha256']}"
        )
    return {
        "status": "pass",
        "spec_sha256": recorded["spec_sha256"],
        "protocol_sha256": recorded.get("protocol_sha256"),
    }
