# -*- coding: utf-8 -*-
"""P7 P-1 重开（p1r）冻结参数：只从生效 prereg 派生，缺件/字节不符即 fail-fast。

设计约束
========
- prereg 的 ``status`` 必须为 ``effective``，文件 SHA 前缀必须匹配登记值；
- ``criteria_digest`` 每次读取都复算，任何判据段落改动都会 fail-fast（防止“跑完再改判据”）；
- 成本、最低成交额、块长、复制次数、种子全部来自 prereg，不在代码里另立阈值。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PREREG_REL = "data/frozen_reference/v2_p7/p7_p1r_20261113_r1/prereg/p7_p1_prereg_v1.json"
PREREG_SHA_PREFIX = "f3c4dcbc"
PREREG_ID = "p7-p1-redesign-20261113"
EVIDENCE_ROOT_REL = "data/frozen_reference/v2_p7/p7_p1r_20261113_r1"
DEFAULT_RUN_ID = "p7_p1r_20261113_r1"
SCHEMA = "p7_p1_redesign_prereg/v1"
CRITERIA_SCHEMA = "p7_p1_criteria_digest/v1"


class P1RParamsError(RuntimeError):
    """prereg 缺件、字节不符、判据摘要不一致（fail-fast，零副作用）。"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def canonical_json(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _require(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    node: Any = payload
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            raise P1RParamsError(f"prereg 缺少字段：{'.'.join(path)}")
        node = node[key]
    return node


@dataclass(frozen=True)
class CostRates:
    """交易成本四项（与冻结 prereg 一致）。"""

    commission_rate: float
    min_commission: float
    stamp_tax: float
    slippage: float

    def as_dict(self) -> dict[str, float]:
        return {
            "commission_rate": self.commission_rate,
            "min_commission": self.min_commission,
            "stamp_tax": self.stamp_tax,
            "slippage": self.slippage,
        }


@dataclass(frozen=True)
class P1RParams:
    """P-1 重开全部冻结参数（manifest 直接序列化）。"""

    repo_root: Path
    prereg_path: Path
    prereg_sha256: str
    prereg_id: str
    criteria_digest: str
    evidence_root: Path
    run_id: str
    common_data_cutoff: str
    evaluation_start: str
    evaluation_end: str
    evaluation_trading_days: int
    horizon_primary: int
    horizon_sensitivity: int
    entry_max_defer: int
    lot_size: int
    min_ticket: float
    cost: CostRates
    initial_capital: float
    max_concurrent_positions: int
    maturity_marker_days: int
    bootstrap_main_block: int
    bootstrap_aux_block: int
    bootstrap_replicates: int
    bootstrap_sensitivity: tuple[int, ...]
    calibration_samples: int
    calibration_replicates: int
    seeds: Mapping[str, int]

    def params_payload(self) -> dict[str, Any]:
        """manifest 用参数快照（不含绝对仓库前缀）。"""
        return {
            "prereg_id": self.prereg_id,
            "prereg_sha256": self.prereg_sha256,
            "criteria_digest": self.criteria_digest,
            "common_data_cutoff": self.common_data_cutoff,
            "evaluation_window": [self.evaluation_start, self.evaluation_end],
            "evaluation_trading_days": self.evaluation_trading_days,
            "horizon_primary": self.horizon_primary,
            "horizon_sensitivity": self.horizon_sensitivity,
            "entry_max_defer_days": self.entry_max_defer,
            "lot_size": self.lot_size,
            "min_ticket": self.min_ticket,
            "cost": self.cost.as_dict(),
            "initial_capital": self.initial_capital,
            "max_concurrent_positions": self.max_concurrent_positions,
            "maturity_marker_days": self.maturity_marker_days,
            "bootstrap": {
                "main_block": self.bootstrap_main_block,
                "aux_block": self.bootstrap_aux_block,
                "replicates": self.bootstrap_replicates,
                "sensitivity": list(self.bootstrap_sensitivity),
            },
            "calibration": {
                "samples_per_theta": self.calibration_samples,
                "replicates": self.calibration_replicates,
            },
            "seeds": dict(self.seeds),
        }


def load_prereg(repo_root: Path, prereg_path: Path | None = None) -> tuple[dict, str]:
    """读取并校验生效 prereg（status / SHA 前缀 / schema / 判据摘要复算）。"""
    root = Path(repo_root).resolve()
    path = Path(prereg_path) if prereg_path else root / PREREG_REL
    if not path.exists():
        raise P1RParamsError(f"prereg 缺件：{path}")
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if not digest.startswith(PREREG_SHA_PREFIX):
        raise P1RParamsError(f"prereg SHA 前缀不符：{digest[:16]} != {PREREG_SHA_PREFIX}")
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("schema") != SCHEMA:
        raise P1RParamsError(f"prereg schema 不符：{payload.get('schema')}")
    if payload.get("status") != "effective":
        raise P1RParamsError(f"prereg 未生效：{payload.get('status')}")

    block = payload.get("criteria_digest") or {}
    if block.get("schema") != CRITERIA_SCHEMA:
        raise P1RParamsError(f"criteria_digest schema 不符：{block.get('schema')}")
    criteria = {key: payload[key] for key in block.get("covers", [])}
    recomputed = sha256_bytes(canonical_json(criteria))
    if recomputed != block.get("sha256"):
        raise P1RParamsError(
            f"criteria_digest 复算不一致：{recomputed[:16]} != {str(block.get('sha256'))[:16]}"
        )
    return payload, digest


def build_params(
    repo_root: Path,
    prereg_path: Path | None = None,
    *,
    run_id: str = DEFAULT_RUN_ID,
) -> P1RParams:
    """从生效 prereg 派生全部 P-1 参数。"""
    root = Path(repo_root).resolve()
    payload, digest = load_prereg(root, prereg_path)
    prereg_file = Path(prereg_path) if prereg_path else root / PREREG_REL

    frozen = payload["frozen_params"]
    cost_payload = payload["execution"]["cost"]
    cost = CostRates(
        commission_rate=float(cost_payload["commission_rate"]),
        min_commission=float(cost_payload["min_commission"]),
        stamp_tax=float(cost_payload["stamp_tax"]),
        slippage=float(cost_payload["slippage"]),
    )
    min_ticket = max(25000.0, cost.min_commission / cost.commission_rate)
    if abs(min_ticket - float(frozen["min_ticket"])) > 1e-9:
        raise P1RParamsError(f"min_ticket 与冻结值不符：{min_ticket}")

    horizon_primary = int(frozen["horizon_primary"])
    if horizon_primary != int(payload["intent_domain"]["horizon_primary"]):
        raise P1RParamsError("horizon_primary 与 intent_domain 不一致")
    evaluation_days = int(payload["data_identity"]["evaluation_trading_days"])
    if evaluation_days != int(frozen["evaluation_trading_days"]):
        raise P1RParamsError("evaluation_trading_days 与 data_identity 不一致")

    sampling = payload["sampling"]
    calibration = payload["calibration"]["layer2_statistical"]
    seeds = {
        key: int(value) for key, value in payload["seeds"].items() if key != "mutually_exclusive"
    }

    return P1RParams(
        repo_root=root,
        prereg_path=prereg_file,
        prereg_sha256=digest,
        prereg_id=str(payload["prereg_id"]),
        criteria_digest=str(payload["criteria_digest"]["sha256"]),
        evidence_root=root / EVIDENCE_ROOT_REL,
        run_id=run_id,
        common_data_cutoff=str(payload["data_identity"]["common_data_cutoff"]),
        evaluation_start=str(payload["data_identity"]["evaluation_start"]),
        evaluation_end=str(payload["data_identity"]["evaluation_window"][1]),
        evaluation_trading_days=evaluation_days,
        horizon_primary=horizon_primary,
        horizon_sensitivity=int(frozen["horizon_sensitivity"]),
        entry_max_defer=int(frozen["entry_max_defer_days"]),
        lot_size=int(frozen["lot_size"]),
        min_ticket=min_ticket,
        cost=cost,
        initial_capital=float(frozen["initial_capital"]),
        max_concurrent_positions=int(frozen["max_concurrent_positions"]),
        maturity_marker_days=int(frozen["maturity_marker_days"]),
        bootstrap_main_block=int(sampling["main"]["mean_block"]),
        bootstrap_aux_block=int(sampling["aux"]["block"]),
        bootstrap_replicates=int(sampling["main"]["replicates"]),
        bootstrap_sensitivity=tuple(int(item) for item in sampling["sensitivity"]["blocks"]),
        calibration_samples=int(calibration["samples_per_theta"]),
        calibration_replicates=2000,
        seeds=seeds,
    )
