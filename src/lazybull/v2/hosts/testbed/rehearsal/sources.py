# -*- coding: utf-8 -*-
"""P5b T6 演练的输入源解析：structural 窗口、交易日历、上游链接、冻结崩盘窗、λ/ζ。

契约来源：设计 v4 §4.1（机械层只选 ``row_domain=structural`` 窗口）、§5.4.3（冻结窗与求解
语义）、§5.5「H6 数值」行、§6.2（五臂同一起点）。

全部函数**只读**：不写数据根、不写生产根、不改 T4/T5 冻结件。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from src.lazybull.v2.hosts.testbed.paths import TestbedError
from src.lazybull.v2.hosts.testbed.rehearsal.injection import solve_anchor_parameter
from src.lazybull.v2.hosts.testbed.rehearsal.sections import ANCHOR_HORIZON
from src.lazybull.v2.hosts.testbed.rehearsal.manifest import file_sha256
from src.lazybull.v2.store.row_domain import STRUCTURAL, STRUCTURAL_FROM, row_domain_for_date

__all__ = [
    "anchor_parameters",
    "collect_production_fingerprints",
    "frozen_windows_from_t5",
    "production_flat_config",
    "production_model_train_end",
    "require_repo_manifest",
    "resolve_structural_window",
    "resolve_trading_calendar",
    "resolve_upstream",
]

#: 生产零写入快照面（递归统计文件数与字节数）
PRODUCTION_DIRS: Sequence[str] = ("paper", "runs", "models")
#: 上游五项（T4/T5 冻结仪器与两侧临界值、T5 功效产物）
UPSTREAM_PATHS: Mapping[str, str] = {
    "t4_manifest": "docs/plans/v2/p5b_t4_calibration_manifest_20261104b.json",
    "t4_thresholds": (
        "data/paper_shadow/p5b_testbed/p5b_t4_referee_20261104b/reports/"
        "calibration_thresholds.json"
    ),
    "t4_holdout": (
        "data/paper_shadow/p5b_testbed/p5b_t4_referee_20261104b/reports/income_holdout.json"
    ),
    "t5_manifest": "docs/plans/v2/p5b_t5_drawdown_manifest_20261104b.json",
    "t5_thresholds": (
        "data/paper_shadow/p5b_testbed/p5b_t5_drawdown_20261104b/reports/drawdown_thresholds.json"
    ),
    "t5_power": (
        "data/paper_shadow/p5b_testbed/p5b_t5_drawdown_20261104b/reports/drawdown_power.json"
    ),
}


class SourcesError(TestbedError):
    """输入源不可用（缺分区/缺冻结件/窗口不足）。"""

    __test__ = False


def resolve_structural_window(
    data_root: str | Path, *, model_train_end: str, watermark_cap: Optional[str] = None
) -> Dict[str, Any]:
    """解析可用的 structural 虚拟窗口（= 全部可用日，不做挑选）。

    规则（写进 manifest ``selection_rule``）：取 ``clean/daily`` 全部分区中
    **行域为 structural** 且**晚于模型训练截止**的交易日，按升序整体纳入。

    Raises:
        SourcesError: 无可用日。
    """
    root = Path(data_root)
    daily = root / "clean" / "daily"
    if not daily.is_dir():
        raise SourcesError(f"数据根缺 clean/daily: {daily}")
    days = sorted(
        path.stem.replace("-", "")
        for path in daily.glob("*.parquet")
        if len(path.stem.replace("-", "")) == 8
    )
    available = [
        day
        for day in days
        if day > str(model_train_end)
        and row_domain_for_date(day) == STRUCTURAL
        and (watermark_cap is None or day <= str(watermark_cap))
    ]
    if not available:
        raise SourcesError(
            f"无可用 structural 虚拟日（structural_from={STRUCTURAL_FROM}，"
            f"模型截止={model_train_end}）: {root}"
        )
    return {
        "days": available,
        "day_count": len(available),
        "d0_date": available[0],
        "watermark": available[-1],
        "return_days": max(0, len(available) - 1),
        "selection_rule": (
            "全部可用 structural 交易日（clean/daily 分区、行域 structural、晚于模型训练截止），"
            "按升序整体纳入；不按读数删减或挑选"
        ),
        "structural_from": STRUCTURAL_FROM,
    }


def resolve_trading_calendar(data_root: str | Path) -> List[str]:
    """读取交易日历（``clean/trade_cal`` 的开市日；缺失即报错）。"""
    path = Path(data_root) / "clean" / "trade_cal.parquet"
    if not path.is_file():
        raise SourcesError(f"数据根缺交易日历: {path}")
    frame = pd.read_parquet(path)
    column = "cal_date" if "cal_date" in frame.columns else "trade_date"
    if "is_open" in frame.columns:
        frame = frame[frame["is_open"].astype(int) == 1]
    return sorted(str(value).replace("-", "") for value in frame[column].tolist())


def collect_production_fingerprints(repo_root: str | Path) -> Dict[str, Any]:
    """生产根零写入快照（文件数 + 字节数 + 路径/大小/mtime 摘要）。

    摘要口径：对 ``(相对路径, size, mtime_ns)`` 排序后取 sha256 ⇒ 同尺寸内容替换、同名
    覆盖都能被发现（只比 files+bytes 会漏）。
    """
    root = Path(repo_root) / "data"
    payload: Dict[str, Any] = {}
    for name in PRODUCTION_DIRS:
        base = root / name
        files = 0
        total = 0
        digest = hashlib.sha256()
        max_mtime = 0
        if base.is_dir():
            entries = []
            for path in base.rglob("*"):
                if path.is_file():
                    stat = path.stat()
                    files += 1
                    total += stat.st_size
                    max_mtime = max(max_mtime, stat.st_mtime_ns)
                    entries.append(
                        (path.relative_to(base).as_posix(), stat.st_size, stat.st_mtime_ns)
                    )
            for relative, size, mtime in sorted(entries):
                digest.update(f"{relative}|{size}|{mtime}\n".encode("utf-8"))
        payload[name] = {
            "files": files,
            "bytes": total,
            "path_digest": digest.hexdigest(),
            "max_mtime_ns": max_mtime,
        }
    return payload


def resolve_upstream(repo_root: str | Path) -> Dict[str, Any]:
    """复算上游五项哈希（哈希不符即由 manifest 校验阶段拒绝运行）。"""
    root = Path(repo_root)
    resolved: Dict[str, Dict[str, str]] = {}
    for key, relative in UPSTREAM_PATHS.items():
        path = root / relative
        if not path.is_file():
            raise SourcesError(f"上游冻结件缺失（禁止在仪器缺失时演练）: {path}")
        resolved[key] = {"path": relative, "sha256": file_sha256(path)}
    return resolved


def frozen_windows_from_t5(repo_root: str | Path) -> Dict[str, Any]:
    """读取 T5 冻结崩盘窗清单（原文引用 + 规范化哈希；不重选）。"""
    path = Path(repo_root) / str(UPSTREAM_PATHS["t5_manifest"])
    doc = json.loads(path.read_text(encoding="utf-8"))
    frozen = [dict(entry) for entry in doc["drawdown_referee"]["windows"]["frozen_list"]]
    text = json.dumps(frozen, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "frozen_list": frozen,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_manifest_sha256": file_sha256(path),
    }


def anchor_parameters(repo_root: str | Path, frozen: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """在冻结 B0 锚窗上求 λ/ζ（盲态；不读演练读数）。

    Returns:
        ``{dd_improve, dd_worsen, dd_improve_status, dd_worsen_status, *_pure_pp}``。
    """
    from src.lazybull.v2.hosts.testbed.referee.drawdown.calibration import (
        source_from_pair_document,
    )
    from src.lazybull.v2.hosts.testbed.referee.drawdown.mdd import (
        DIRECTION_IMPROVE,
        DIRECTION_WORSEN,
    )

    root = Path(repo_root)
    manifest = json.loads((root / str(UPSTREAM_PATHS["t5_manifest"])).read_text(encoding="utf-8"))
    source = source_from_pair_document(
        manifest["source_pair"],
        dgp_doc=manifest["calibration"],
        repo_root=root,
        h=int(manifest["drawdown_referee"]["reading"]["H"]),
        window_samples=int(manifest["drawdown_referee"]["windows"]["window_samples"]),
        expected_window_mdd=float(manifest["source_window"]["window"]["mdd_percent"]),
    )
    thresholds = json.loads(
        (root / str(UPSTREAM_PATHS["t5_thresholds"])).read_text(encoding="utf-8")
    )
    look_days = [int(value) for value in thresholds["plan"]["look_days"]]
    anchor_index = look_days.index(ANCHOR_HORIZON)
    c_i_anchor = float(thresholds["c_i_by_look"][anchor_index])
    c_w_anchor = float(thresholds["c_w_by_look"][anchor_index])
    # 预登记目标（盲态）：DD-I 取 max(12pp, c^I+2pp)；DD-W 取 min(-12pp, c^W-2pp)。
    # 依据：T5 冻结 c^W_anchor 远深于 -12pp（-31.1pp），12pp 的恶化结构上无法触发否决事件，
    # 故目标必须越过阈值本身；DD-I 的 12pp 已高出 c^I_anchor 约 6.8pp，无需上调。
    targets = {
        DIRECTION_IMPROVE: max(12.0, c_i_anchor + 2.0),
        DIRECTION_WORSEN: min(-12.0, c_w_anchor - 2.0),
    }
    result: Dict[str, Any] = {
        "targets_pp": {key: float(value) for key, value in targets.items()},
        "anchor_thresholds_pp": {"c_i": c_i_anchor, "c_w": c_w_anchor},
        "target_derivation": (
            "DD-I: max(12pp, c^I_anchor+2pp)；DD-W: min(-12pp, c^W_anchor-2pp)"
            "（c^W_anchor 深于 -12pp，故必须越过阈值才能触发恶化否决事件）"
        ),
    }
    for direction, key in ((DIRECTION_IMPROVE, "dd_improve"), (DIRECTION_WORSEN, "dd_worsen")):
        target = targets[direction]
        solution = solve_anchor_parameter(
            source.window.log_returns, frozen, direction=direction, target_pp=target
        )
        ok = bool(solution.constructible[0])
        result[key] = float(solution.parameter[0]) if ok else None
        result[f"{key}_status"] = str(solution.status[0])
        result[f"{key}_pure_pp"] = float(solution.achieved_pp[0])
    return result


def production_model_train_end(repo_root: str | Path) -> str:
    """生产纸面配置的冻结模型训练截止日（D0 必须严格晚于它）。"""
    from src.lazybull.v2.hosts.paper.config_adapter import (
        load_paper_config,
        resolve_model_version,
    )

    root = Path(repo_root)
    flat = load_paper_config(root / "data" / "paper")
    models_dir = str(root / "data" / "models" / "stock_selection")
    version = resolve_model_version(flat.get("model_version"), models_dir)
    payload = json.loads(
        (Path(models_dir) / f"v{version}_metadata.json").read_text(encoding="utf-8")
    )
    return str(payload["train_end_date"]).replace("-", "")


def production_flat_config(repo_root: str | Path) -> Dict[str, Any]:
    """生产纸面扁平配置（冻结模型/配置的 fresh_cash 克隆来源）。"""
    from src.lazybull.v2.hosts.paper.config_adapter import load_paper_config

    return dict(load_paper_config(Path(repo_root) / "data" / "paper"))


def require_repo_manifest(repo_root: str | Path) -> Path:
    """取最新的 T6 仓库冻结件（按修改时间，兼顾重冻后缀；缺省即报错）。"""
    candidates = list(
        (Path(repo_root) / "docs" / "plans" / "v2").glob("p5b_t6_rehearsal_manifest_*.json")
    )
    if not candidates:
        raise SourcesError("缺 T6 仓库冻结件（先执行 freeze --apply）")

    def _sequence(path: Path) -> tuple[int, str]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return int((payload.get("lineage") or {}).get("sequence") or 0), path.name
        except (OSError, ValueError):
            return 0, path.name

    return max(candidates, key=_sequence)
