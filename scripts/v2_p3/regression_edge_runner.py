# -*- coding: utf-8 -*-
"""回归边受保护复跑入口（验收专用；P2-2 / R2-T5AE-03）。

固定 P2b ``legacy_bypass`` 四元组，禁止 P2a 默认模式误用；运行前采集冻结模型与
实际消费输入的字节身份，运行后复核并落盘。比较仍复用
``t5_compare_gates.compare_regression_edge``（判据不变）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from loguru import logger

# 独立运行时仓库根不入 sys.path，按薄入口惯例显式引导
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.v2_p2a.replay_b0 import B0_START_MODEL_VERSION, replay_b0  # noqa: E402
from scripts.v2_p3.t5_compare_gates import (  # noqa: E402
    compare_regression_edge,
    write_verdict_json,
)
from scripts.v2_p3.t05b_baselines import enumerate_consumption_files  # noqa: E402
from scripts.v2_p3.t05b_isolated_root import (  # noqa: E402
    collect_consumption_snapshot,
    verify_consumption_snapshot_closed,
)
from src.lazybull.v2.hosts.backtest.runtime import LEGACY_BYPASS  # noqa: E402

#: 回归边比较冻结批（P2b 关闭臂；与 §19.22 登记一致）
FROZEN_BATCH = "p2b_t5_closed_20261017_203508"
#: 复跑批的 baseline_ref（沿用既有闭批的 P2a 基线身份）
BASELINE_REF = "p2a_b0replay_20261008_144307_d79f3f"
#: P2b legacy_bypass 复跑来源标签（与 §19.22 登记一致）
SOURCE = "v2_p2b_merger_replay"
#: B0 14 折模型版本数
_MODEL_COUNT = 14
_MODEL_SUFFIXES = ("features.json", "metadata.json", "model.joblib")


def _model_files(models_root: Path, start_version: int = B0_START_MODEL_VERSION) -> List[Path]:
    """回归边实际消费的 14 个折模型文件（缺任一 ⇒ 明确失败）。"""
    files: List[Path] = []
    for version in range(int(start_version), int(start_version) + _MODEL_COUNT):
        for suffix in _MODEL_SUFFIXES:
            path = Path(models_root) / f"v{version}_{suffix}"
            if not path.is_file():
                raise FileNotFoundError(f"回归边模型文件缺失: {path}")
            files.append(path)
    return files


def collect_guarded_inputs(
    data_root: Path, start_version: int = B0_START_MODEL_VERSION
) -> List[Path]:
    """回归边消费输入清单：B0 输入枚举 + 14 折模型字节。"""
    data_root = Path(data_root)
    by_dir = enumerate_consumption_files(data_root, include_panel_labels=False)
    files = [path for paths in by_dir.values() for path in paths]
    files.extend(_model_files(data_root / "models" / "stock_selection", start_version))
    return files


def run_guarded_regression_edge(
    *,
    data_root: Path,
    out_dir: Path,
    batch_id: Optional[str] = None,
    frozen_batch: str = FROZEN_BATCH,
    replay_fn: Callable[..., Path] = replay_b0,
    compare_fn: Callable[..., Dict[str, object]] = compare_regression_edge,
) -> Dict[str, object]:
    """执行「身份前置 + 冻结 legacy_bypass 复跑 + 身份前后核验 + 成对比较」。

    ``out_dir`` 只放守卫快照与最终 verdict；runs 产物仍由 ``replay_fn`` 写入
    ``<data_root>/runs/<batch_id>``。守卫漂移与比较失败均为 FAIL，二者独立登记。
    """
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    before_path = out_dir / "consumption_before.json"

    inputs = collect_guarded_inputs(data_root)
    collect_consumption_snapshot(inputs, base_dir=data_root, out_path=before_path)
    new_batch = replay_fn(
        data_root=str(data_root),
        runs_root=str(data_root / "runs"),
        batch_id=batch_id,
        merger_mode=LEGACY_BYPASS,
        extra_meta_notes=None,
        baseline_ref=BASELINE_REF,
        arms=(LEGACY_BYPASS,),
        source=SOURCE,
    )
    # 收尾闭合核验（R2-T5AE-R1-02）：重枚举消费候选 + 逐文件身份（新增/替换/变更均 FAIL）
    drift = verify_consumption_snapshot_closed(
        before_path, base_dir=data_root, rescan_paths=collect_guarded_inputs(data_root)
    )
    frozen_path = Path(frozen_batch)
    if not frozen_path.is_absolute():
        frozen_path = data_root / "runs" / frozen_batch
    comparison = compare_fn(Path(new_batch), frozen_path)
    guard = {
        "input_files": len(inputs),
        "model_versions": _MODEL_COUNT,
        "before_snapshot": str(before_path),
        "after_drift": drift,
    }
    passed = not drift and comparison.get("verdict") == "PASS"
    verdict: Dict[str, object] = {
        "gate": "regression_edge_guarded",
        "verdict": "PASS" if passed else "FAIL",
        "criterion": "冻结 legacy_bypass 四元组 + 模型/消费输入前后字节身份核验"
        " + 与冻结批成对精确比较（R2-T5AE-03 / R1-P2-2）",
        "new_batch": str(new_batch),
        "frozen_batch": frozen_batch,
        "merger_mode": LEGACY_BYPASS,
        "source": SOURCE,
        "guard": guard,
        "comparison": comparison,
        "errors": list(comparison.get("errors") or [])
        + ([f"消费输入漂移: {drift}"] if drift else []),
    }
    write_verdict_json(verdict, out_dir / "regression_edge_verdict.json")
    logger.info(f"回归边受保护复跑 verdict={verdict['verdict']}，证据: {out_dir}")
    return verdict


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="回归边受保护复跑（legacy_bypass 固定）")
    parser.add_argument("--data-root", required=True, help="生产数据根（含 models/ 与 runs/）")
    parser.add_argument("--out-dir", required=True, help="守卫快照与 verdict 输出目录")
    parser.add_argument("--batch-id", default=None, help="新 runs 批次 ID（默认自动生成）")
    parser.add_argument("--frozen-batch", default=FROZEN_BATCH, help="冻结批批次 ID")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    verdict = run_guarded_regression_edge(
        data_root=Path(args.data_root),
        out_dir=Path(args.out_dir),
        batch_id=args.batch_id,
        frozen_batch=args.frozen_batch,
    )
    json.dumps(verdict, ensure_ascii=False, default=str)
    return 0 if verdict["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
