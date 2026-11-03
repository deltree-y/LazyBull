# -*- coding: utf-8 -*-
"""P5b T6 测试环境（tests/ 全量豁免复杂度；不是测试用例模块）。

复用 T2 的合成 structural 段（真实 parquet + 确定性桩信号），并补：
- 五个演练臂的注册（角色名与 manifest 分节一致）；
- 假仓库根（上游五项文件 + runbook 骨架），用于 manifest 冻结/校验/报告一致性；
- 宿主层注入工厂（``make_injected_host_factory``）。

**不读生产 data/paper、不联网、不补齐、不写生产根。**
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import p5b_t2_env as envmod  # noqa: E402

#: 五臂（顺序与 manifest 分节一致）
ARMS = ("base", "mech_aa", "income_minus_5d", "dd_improve_12pp", "dd_worsen_12pp")
#: 臂角色
ROLES = {
    "base": "aa_clone",
    "mech_aa": "aa_clone",
    "income_minus_5d": "synthetic_effect",
    "dd_improve_12pp": "synthetic_effect",
    "dd_worsen_12pp": "synthetic_effect",
}
#: 假上游六项（相对假仓库根；含被引用的 T4 holdout）
UPSTREAM_FILES = {
    "t4_manifest": "docs/plans/v2/p5b_t4_calibration_manifest_20261102.json",
    "t4_thresholds": "t4/calibration_thresholds.json",
    "t4_holdout": "t4/income_holdout.json",
    "t5_manifest": "docs/plans/v2/p5b_t5_drawdown_manifest_20261102.json",
    "t5_thresholds": "t5/drawdown_thresholds.json",
    "t5_power": "t5/drawdown_power.json",
}
#: 合成冻结崩盘窗（严格落在 T2 的 9 个收益样本内）
FROZEN_WINDOWS = (
    {
        "peak": 1,
        "trough": 3,
        "window_start_ordinal": 1,
        "window_stop_ordinal": 3,
        "depth_pp": -3.0,
        "rank": 0,
        "window_samples": 2,
    },
    {
        "peak": 4,
        "trough": 6,
        "window_start_ordinal": 4,
        "window_stop_ordinal": 6,
        "depth_pp": -2.0,
        "rank": 1,
        "window_samples": 2,
    },
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_repo(tmp_path: Path) -> Path:
    """构造假仓库根：上游五项 + 运行手册骨架 + 版本/README。"""
    repo = tmp_path / "repo"
    for key, relative in UPSTREAM_FILES.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"key": key, "value": 1}, sort_keys=True), encoding="utf-8")
    guide = repo / "docs" / "guide" / "p5b_testbed_runbook.md"
    guide.parent.mkdir(parents=True, exist_ok=True)
    guide.write_text(runbook_text({}), encoding="utf-8")
    (repo / "pyproject.toml").write_text('version = "0.209.6"\n', encoding="utf-8")
    (repo / "CHANGELOG.md").write_text("## 0.209.6\n\n- test\n", encoding="utf-8")
    (repo / "README.md").write_text("见 docs/guide/p5b_testbed_runbook.md\n", encoding="utf-8")
    return repo


def upstream_payload(repo: Path) -> Dict[str, Dict[str, str]]:
    """按假仓库的实际字节生成 upstream 声明。"""
    return {
        key: {
            "path": relative,
            "sha256": hashlib.sha256((repo / relative).read_bytes()).hexdigest(),
        }
        for key, relative in UPSTREAM_FILES.items()
    }


def runbook_text(artifact_hashes: Mapping[str, str]) -> str:
    """运行手册骨架（含必需章节、产物名与哈希块）。"""
    lines = [
        "# P5b 试车场运行手册（测试骨架）",
        "",
        "> 正式演练根 `c1`（`lineage.sequence=1`）；更早的 run 标“被取代”，不参与读数。",
        "",
    ]
    for index in range(1, 9):
        lines.extend([f"## {index}. 章节 {index}", "", "示例命令与说明。", ""])
    from src.lazybull.v2.hosts.testbed.rehearsal import artifacts as art

    for name in art.ARTIFACT_NAMES:
        lines.append(f"- 产物：`{name}`")
    lines.extend(["", "<!-- rehearsal-artifacts -->", "| 产物 | sha256 |", "|---|---|"])
    for name, digest in sorted(artifact_hashes.items()):
        lines.append(f"| `{name}` | `{digest}` |")
    lines.extend(["<!-- /rehearsal-artifacts -->", ""])
    return "\n".join(lines)


def manifest_document(
    env: SimpleNamespace,
    repo: Path,
    *,
    days: Optional[Sequence[str]] = None,
    parameters: Optional[Mapping[str, Optional[float]]] = None,
    campaign_id: str = "t6_campaign",
    campaign_root: Optional[str] = None,
) -> Dict[str, Any]:
    """按假仓库与合成环境构造 T6 manifest 文档（未封存）。"""
    from src.lazybull.v2.hosts.testbed.rehearsal.manifest import build_manifest_document

    frozen = [dict(entry) for entry in FROZEN_WINDOWS]
    text = json.dumps(frozen, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return build_manifest_document(
        run_id="t6_run",
        created_at="2026-11-02",
        campaign_id=campaign_id,
        campaign_root=str(campaign_root or env.base_root / campaign_id),
        base_root=str(env.base_root),
        data_root=str(env.data_root),
        days=tuple(days or env.days),
        watermark=str((days or env.days)[-1]),
        selection_rule="测试窗口：全部合成 structural 交易日",
        frozen_windows=frozen,
        frozen_list_sha256=sha256_text(text),
        parameters=dict(
            parameters if parameters is not None else {"dd_improve": 0.3, "dd_worsen": 0.2}
        ),
        upstream=upstream_payload(repo),
    )


def prepare_arms(env: SimpleNamespace, *, campaign_id: str = "t6_campaign") -> SimpleNamespace:
    """登记 campaign + 五臂 + D0（合成环境；双臂 A/A 由 base/mech_aa 承担）。"""
    return envmod.prepare_campaign(
        env,
        campaign_id=campaign_id,
        arms=ARMS,
        roles=ROLES,
    )


def run_with_injection(
    env: SimpleNamespace,
    prepared: SimpleNamespace,
    document: Mapping[str, Any],
    *,
    days: Optional[Sequence[str]] = None,
    extra_factory=None,
) -> Dict[str, Any]:
    """以注入工厂运行（``extra_factory`` 用于非注入对照）。"""
    from src.lazybull.v2.hosts.testbed.rehearsal.hosts import make_injected_host_factory
    from src.lazybull.v2.hosts.testbed.rehearsal.injection import (
        arm_effect_plan,
        window_mask_from_frozen,
    )
    from src.lazybull.v2.hosts.testbed.runner import run_virtual_campaign

    return_days = int(document["window"]["return_days"])
    effects = arm_effect_plan(document, return_days)
    masks: Dict[str, Any] = {}
    try:
        mask, _ = window_mask_from_frozen(
            list(document["injection_window"]["frozen_list"]), return_days
        )
        masks = {
            arm: mask
            for arm, payload in dict(document["arms"]).items()
            if str((payload.get("injection") or {}).get("kind")) == "crash_window_transform"
        }
    except Exception:  # noqa: BLE001 - 窗越界时无掩码（各臂 not_constructible）
        masks = {}
    base = extra_factory or envmod.make_host_factory(env)
    factory = make_injected_host_factory(base, effects=effects, masks=masks)
    return run_virtual_campaign(
        prepared.root,
        days=list(days) if days is not None else None,
        host_factory=factory,
        write_report=False,
        allow_external_root=True,
    )


def shard(prepared: SimpleNamespace, arm_id: str, day: str) -> Dict[str, Any]:
    return envmod.shard(prepared, arm_id, day)


def nav_series(prepared: SimpleNamespace, arm_id: str, days: Sequence[str]) -> List[float]:
    return [float(shard(prepared, arm_id, day)["evidence"]["valuation"]["nav"]) for day in days]


def ledger_layout(prepared: SimpleNamespace):
    from src.lazybull.v2.hosts.testbed.layout import layout_for

    return layout_for(prepared.root)


def report_kwargs(document: Dict[str, Any]) -> Dict[str, Any]:
    """构造 build_report 的最小确定性入参（用于逐字节确定性用例）。"""
    from src.lazybull.v2.hosts.testbed.rehearsal.injection import arm_effect_plan

    thresholds = {
        "look_days": [120, 240, 360, 500],
        "look_blocks": [6, 12, 18, 25],
        "h": 20,
        "c_k": [4.7, 3.4, 3.0, 2.9],
        "c_w": [-17.8, -30.5, -31.0, -31.1],
        "c_i": [7.9, 6.5, 5.5, 5.1],
        "delta_log_daily": 0.0001936117626,
        "anchor": 500,
    }
    days = len(document["window"]["virtual_days"])
    arms = {arm: {"driven_days": days, "authoritative_values": 1} for arm in ARMS}
    return {
        "campaign_id": str(document["campaign"]["campaign_id"]),
        "run_id": str(document["run_id"]),
        "manifest": document,
        "manifest_sha256": "a" * 64,
        "calibrator": {"implementation_id": "i", "sha256": "b" * 64, "modules": {}},
        "upstream": dict(document["upstream"]),
        "run_report": {
            "row_domain": {"declared": "structural", "verified": True, "per_day": {}},
            "identity": {"manifest_sha256": "a" * 64},
            "model_availability": {},
            "arms": arms,
            "transactions": [],
            "published": {"published_generation": 1},
            "recovery": {},
            "fault_injections": [],
        },
        "aa": {
            "identical": True,
            "compared_days": 1,
            "compared_fields": 6,
            "mismatches": [],
            "source": "shard_recompute",
        },
        "h6_items": [
            {"name": name, "status": "not_executed", "judgment": "未执行"}
            for name in ("income_minus_5d", "dd_improve_12pp", "dd_worsen_12pp")
        ],
        "income_reading": {"sample_days": 0, "rows": []},
        "drawdown_readings": {},
        "thresholds": thresholds,
        "effects": {
            arm: effect.to_dict() for arm, effect in arm_effect_plan(document, days - 1).items()
        },
        "artifact_hashes": {"rehearsal_manifest.json": "a" * 64},
        "machine_time": {"virtual_days_total": 1, "batches": [{"days": 1}]},
        "authorization": None,
        "production": {"unchanged": True, "window": "test", "before": {}, "after": {}},
        "referenced": {"note": "引用 T4/T5"},
        "t4_reference": {"path": "x", "sha256": "c" * 64, "c_k": thresholds["c_k"]},
        "t5_reference": {
            "path": "y",
            "sha256": "d" * 64,
            "c_w": thresholds["c_w"],
            "c_i": thresholds["c_i"],
        },
    }
