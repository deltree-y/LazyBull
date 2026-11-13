# -*- coding: utf-8 -*-
"""P7 T1 冻结参数（逐项取自生效复合 prereg 与设计 §6）。

原则
====
- **参数只从已生效 prereg 派生**，不在此处另立阈值；读取时逐项断言一致性，
  任一缺件/字节不符即 fail-fast（不进入计算、不产生任何写出）；
- 未写入 prereg 但设计 §6 已冻结的口径（对照个数 5、协变量回看 20 日、裁剪 ±3、
  入场/离场最长顺延 10 日）在此以常量登记，并在 manifest 中逐项留档；
- 统计流的种子**互斥且显式登记**：校准流用 prereg 的 42，误放行评估流与功效复核流
  各用固定偏移种子，避免同一随机量重复使用。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from src.lazybull.common.config import get_cost_settings

#: T0.5 已生效证据根（只读引用）
T05_EVIDENCE_REL = "data/paper_shadow/p7_testbed/p7_t05_20261111_r6"
PREREG_REL = f"{T05_EVIDENCE_REL}/prereg/b0_composite_prereg_v1.json"
CRASH_SPEC_REL = f"{T05_EVIDENCE_REL}/specs/crash_windows_v6.json"

#: 生效身份前缀（防止误读历史版本）
PREREG_SHA_PREFIX = "e88efd87"
CRASH_SPEC_SHA_PREFIX = "61c6b7d1"

#: 设计 §6 冻结常量（prereg 未单列，按设计原文固定）
CONTROL_COUNT = 5
COVARIATE_LOOKBACK_DAYS = 20
MAD_CLIP = 3.0
MAD_SCALE = 1.0
ENTRY_MAX_DEFER_DAYS = 10
EXIT_MAX_DEFER_DAYS = 10
LOT_SIZE = 100

#: 统计流的互斥种子（校准流 = prereg 冻结的 42）
SEED_CALIBRATION = 42
SEED_EVALUATION = 1042
SEED_POWER = 2042

#: 分量门的名义尾概率分母（Q5：三源各一组件，λ=1 由 T3 选择，T1 只用 λ=1）
COMPONENT_COUNT = 3
LAMBDA_T1 = 1

DEFAULT_RUN_ID = "p7_t1_20261111_r1"
EVIDENCE_ROOT_REL = "data/frozen_reference/v2_p7"


class ParamsError(RuntimeError):
    """prereg/规格缺件、字节不符或字段缺失（fail-fast，零副作用）。"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


@dataclass(frozen=True)
class CostRates:
    """交易成本四项（显式传入 `CostModel`，不读运行期配置）。"""

    commission_rate: float
    min_commission: float
    stamp_tax: float
    slippage: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "commission_rate": self.commission_rate,
            "min_commission": self.min_commission,
            "stamp_tax": self.stamp_tax,
            "slippage": self.slippage,
        }

    def min_ticket(self) -> float:
        """B0 最低实际成交额 `V = max(25000, min_commission / commission_rate)`。"""
        return max(25000.0, self.min_commission / self.commission_rate)


@dataclass(frozen=True)
class P1Params:
    """T1 全部冻结参数（manifest 直接序列化本对象）。"""

    repo_root: Path
    prereg_path: Path
    prereg_sha256: str
    prereg_id: str
    crash_spec_path: Path
    crash_spec_sha256: str
    common_data_cutoff: str
    evaluation_start: str
    evaluation_end: str
    core_cells: Tuple[Tuple[str, int], ...]
    sensitivity_cells: Tuple[Tuple[str, int], ...]
    block_days: int
    guard_days: int
    min_effective_blocks: int
    implementations: int
    delta_r_grid: Tuple[float, ...]
    mde_min_power: float
    alpha_total: float
    component_tail_alpha: float
    component_quantile: float
    cost: CostRates
    min_ticket: float
    seeds: Mapping[str, int]
    precheck_attachments: Mapping[str, str] = field(default_factory=dict)

    # ---- 派生口径 ----
    def cell_keys(self) -> Tuple[Tuple[str, int], ...]:
        return self.core_cells + self.sensitivity_cells

    def params_payload(self) -> Dict[str, Any]:
        """manifest 用参数快照（不含绝对仓库前缀，保证跨机器可读）。"""
        return {
            "common_data_cutoff": self.common_data_cutoff,
            "evaluation_window": [self.evaluation_start, self.evaluation_end],
            "core_cells": [list(item) for item in self.core_cells],
            "sensitivity_cells": [list(item) for item in self.sensitivity_cells],
            "block_days": self.block_days,
            "guard_days": self.guard_days,
            "min_effective_blocks": self.min_effective_blocks,
            "implementations_per_cell": self.implementations,
            "delta_r_grid": list(self.delta_r_grid),
            "mde_min_power": self.mde_min_power,
            "alpha_total": self.alpha_total,
            "component_tail_alpha": self.component_tail_alpha,
            "component_quantile": self.component_quantile,
            "control_count": CONTROL_COUNT,
            "covariate_lookback_days": COVARIATE_LOOKBACK_DAYS,
            "mad_clip": MAD_CLIP,
            "mad_scale": MAD_SCALE,
            "entry_max_defer_days": ENTRY_MAX_DEFER_DAYS,
            "exit_max_defer_days": EXIT_MAX_DEFER_DAYS,
            "lot_size": LOT_SIZE,
            "cost": self.cost.as_dict(),
            "min_ticket": self.min_ticket,
            "seeds": dict(self.seeds),
            "prereg_id": self.prereg_id,
            "prereg_sha256": self.prereg_sha256,
            "crash_spec_sha256": self.crash_spec_sha256,
        }


def _require(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    node: Any = payload
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            raise ParamsError(f"prereg 缺少字段：{'.'.join(path)}")
        node = node[key]
    return node


def load_prereg(repo_root: Path, prereg_path: Path) -> Tuple[Dict[str, Any], str]:
    """读取 prereg 并校验生效身份前缀与 schema。"""
    path = Path(prereg_path)
    if not path.exists():
        raise ParamsError(f"prereg 缺件：{path}")
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if not digest.startswith(PREREG_SHA_PREFIX):
        raise ParamsError(f"prereg SHA 前缀不符：{digest[:16]} != {PREREG_SHA_PREFIX}")
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("schema") != "p7_b0_composite_prereg/v2":
        raise ParamsError(f"prereg schema 不符：{payload.get('schema')}")
    if payload.get("status") != "effective":
        raise ParamsError(f"prereg 未生效：{payload.get('status')}")
    return payload, digest


def verify_precheck_attachments(repo_root: Path, prereg: Mapping[str, Any]) -> Dict[str, str]:
    """逐字节校验 prereg 内嵌的 T0.5 附件 SHA（缺件/不符即 fail-fast）。"""
    attachments = dict(_require(prereg, ["data_gate", "attachments_sha256"]))
    evidence_id = str(_require(prereg, ["data_gate", "precheck_evidence_id"]))
    base = Path(repo_root) / "data" / "paper_shadow" / "p7_testbed" / evidence_id / "precheck"
    verified: Dict[str, str] = {}
    for name, expected in sorted(attachments.items()):
        path = base / name
        if not path.exists():
            raise ParamsError(f"预检附件缺件：{path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ParamsError(f"预检附件字节不符：{name} {actual[:16]} != {str(expected)[:16]}")
        verified[name] = actual
    return verified


def build_params(
    repo_root: Path,
    prereg_path: Path | None = None,
    crash_spec_path: Path | None = None,
) -> P1Params:
    """从生效 prereg 派生全部 T1 参数。"""
    root = Path(repo_root).resolve()
    prereg_file = Path(prereg_path) if prereg_path else root / PREREG_REL
    prespec_file = Path(crash_spec_path) if crash_spec_path else root / CRASH_SPEC_REL
    prereg, prereg_sha = load_prereg(root, prereg_file)
    attachments = verify_precheck_attachments(root, prereg)

    if not prespec_file.exists():
        raise ParamsError(f"崩盘规格缺件（仅登记引用）：{prespec_file}")
    spec_sha = sha256_file(prespec_file)
    if not spec_sha.startswith(CRASH_SPEC_SHA_PREFIX):
        raise ParamsError(f"崩盘规格 SHA 前缀不符：{spec_sha[:16]}")
    embedded_spec = str(_require(prereg, ["crash_gate", "spec_sha256"]))
    if embedded_spec != spec_sha:
        raise ParamsError("崩盘规格字节与 prereg 内嵌 SHA 不一致")

    core_cells = tuple(
        tuple(item) for item in _require(prereg, ["gates", "Q5_composite", "core_cells"])
    )
    if [list(item) for item in core_cells] != [
        ["forecast", 10],
        ["express", 10],
        ["fina_report", 10],
    ]:
        raise ParamsError(f"Q5 核心格与冻结口径不符：{core_cells}")
    sensitivity = tuple(
        tuple(item) for item in _require(prereg, ["gates", "Q5_composite", "sensitivity_cells"])
    )
    block_rule = _require(prereg, ["gates", "Q5_composite", "precheck_block_rule"])
    randomization = _require(prereg, ["gates", "Q5_composite", "randomization"])
    if int(block_rule["block_days"]) != 20 or int(block_rule["guard_days"]) != 10:
        raise ParamsError(f"评价块/guard 与冻结口径不符：{block_rule}")
    if block_rule.get("assignment_key") != "planned_entry_day=T+1":
        raise ParamsError(f"归块键与冻结口径不符：{block_rule.get('assignment_key')}")

    cutoff = str(_require(prereg, ["data_identity", "common_data_cutoff"]))
    evaluation_start = str(_require(prereg, ["data_identity", "evaluation_start"]))
    window = _require(prereg, ["data_identity", "evaluation_window"])
    if cutoff != "20261029" or evaluation_start != "20111107":
        raise ParamsError(f"数据窗口与冻结口径不符：cutoff={cutoff} start={evaluation_start}")

    implementations = int(randomization["implementations_per_cell"])
    if implementations != 10000:
        raise ParamsError(f"内层实现次数必须为 10000：{implementations}")
    if int(randomization["seed"]) != SEED_CALIBRATION:
        raise ParamsError(f"随机化种子必须为 42：{randomization['seed']}")

    precheck_blocks = _require(prereg, ["gates", "Q5_composite", "precheck_effective_blocks"])
    min_blocks = int(precheck_blocks["forecast"]["min_required"])
    if min_blocks != 60:
        raise ParamsError(f"有效块下限必须为 60：{min_blocks}")

    alpha_total = float(_require(prereg, ["alpha_accounting", "alpha_total"]))
    tail_alpha = alpha_total / COMPONENT_COUNT / LAMBDA_T1
    settings = get_cost_settings()
    cost = CostRates(
        commission_rate=float(settings["commission_rate"]),
        min_commission=float(settings["min_commission"]),
        stamp_tax=float(settings["stamp_tax"]),
        slippage=float(settings["slippage"]),
    )
    return P1Params(
        repo_root=root,
        prereg_path=prereg_file,
        prereg_sha256=prereg_sha,
        prereg_id=str(_require(prereg, ["prereg_id"])),
        crash_spec_path=prespec_file,
        crash_spec_sha256=spec_sha,
        common_data_cutoff=cutoff,
        evaluation_start=evaluation_start,
        evaluation_end=str(window[1]),
        core_cells=core_cells,
        sensitivity_cells=sensitivity,
        block_days=int(block_rule["block_days"]),
        guard_days=int(block_rule["guard_days"]),
        min_effective_blocks=min_blocks,
        implementations=implementations,
        delta_r_grid=tuple(float(item) for item in randomization["delta_R_grid"]),
        mde_min_power=float(
            _require(prereg, ["gates", "Q1_absolute_sharpe", "injection", "mde_min_power"])
        ),
        alpha_total=alpha_total,
        component_tail_alpha=tail_alpha,
        component_quantile=1.0 - tail_alpha,
        cost=cost,
        min_ticket=cost.min_ticket(),
        seeds={
            "calibration": SEED_CALIBRATION,
            "evaluation": SEED_EVALUATION,
            "power": SEED_POWER,
        },
        precheck_attachments=attachments,
    )


def prereg_verdict_rules(params: P1Params) -> Dict[str, Any]:
    """只读复用 prereg 的三向判定规则文本（报告需逐条引用）。"""
    payload, _ = load_prereg(params.repo_root, params.prereg_path)
    return dict(payload.get("verdict_rules") or {})


def cost_model_from_params(params: P1Params):
    """显式四参数构造 CostModel（不读配置，保证可复算）。"""
    from src.lazybull.common.cost import CostModel

    return CostModel(
        commission_rate=params.cost.commission_rate,
        min_commission=params.cost.min_commission,
        stamp_tax=params.cost.stamp_tax,
        slippage=params.cost.slippage,
    )


def list_cell_keys(params: P1Params) -> List[Tuple[str, int]]:
    return list(params.cell_keys())
