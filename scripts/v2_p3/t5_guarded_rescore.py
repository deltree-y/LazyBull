# -*- coding: utf-8 -*-
"""单折受保护补评分（验收专用；R2-T5AE-02 的闭市输入取证）。

对指定臂/折重新打分，跑前采集实际消费输入（含 raw/moneyflow_hsgt 与模型字节），
跑后复核；指纹与既存折指纹、excluded_days 均须一致。不重训、不覆盖既有折产物。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.v2_p3.t1_reconcile_adapter import _panel_partition_files  # noqa: E402
from scripts.v2_p3.t05b_fingerprint import (  # noqa: E402
    SOURCE_MODE_CS_TRAIN_VERIFY,
    SOURCE_MODE_PANEL,
    _load_model_feature_columns,
    load_main_board_codes,
    resolve_panel_score_days,
    score_model_fingerprint,
)
from scripts.v2_p3.t05b_isolated_root import (  # noqa: E402
    collect_consumption_snapshot,
    verify_consumption_snapshot_closed,
)
from scripts.v2_p3.t05b_spike_runner import (  # noqa: E402
    list_expected_oos_days,
    resolve_score_days,
)
from scripts.v2_p3.t05b_stock_connect import (  # noqa: E402
    raw_read_guard,
    raw_read_ledger_snapshot,
)
from src.lazybull.v2.store.data_store import PanelDataStore  # noqa: E402


def _read_json(path: Path) -> Dict[str, Any]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"JSON 顶层须为 object: {path}")
    return doc


def _model_files(models_dir: Path, model_version: int) -> List[Path]:
    files = [
        models_dir / f"v{model_version}_{suffix}"
        for suffix in ("features.json", "metadata.json", "model.joblib")
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"模型文件缺失: {missing}")
    return files


def _fold_input_candidates(
    root: Path, fold_dir: Path, model_version: int, panel: bool
) -> tuple[List[Path], List[Path]]:
    """单折消费候选（现存/缺失分列；R2-T5AE-R1-02）。

    逐日候选（cs_train/clean/daily/raw/suspend/raw/moneyflow_hsgt）采集时**不存在**
    者不再跳过，而是作为缺失候选返回（快照登记 ``missing`` 态，评分期间出现即判
    「快照后新增实读候选」）；其余（单文件 glob / panel 分区 / 模型字节）缺失即
    显式失败或跳过语义不变。
    """
    boundary = _read_json(fold_dir / "boundary.json")
    start, end = str(boundary["test_start"]), str(boundary["test_end"])
    expected = list_expected_oos_days(root, start, end)
    existing: List[Path] = []
    missing: List[Path] = []
    for day in expected:
        hyphen = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        for path in (
            root / "features" / "cs_train" / f"{day}.parquet",
            root / "clean" / "daily" / f"{hyphen}.parquet",
            root / "raw" / "suspend" / f"{hyphen}.parquet",
            root / "raw" / "moneyflow_hsgt" / f"{hyphen}.parquet",
        ):
            if path.is_file():
                existing.append(path)
            else:
                missing.append(path)
    for pattern in ("clean/stock_basic*", "clean/trade_cal*", "raw/trade_cal*"):
        existing.extend(sorted(root.glob(pattern)))
    if panel:
        existing.extend(_panel_partition_files(root, start, end))
    models_dir = root / "models" / "stock_selection"
    existing.extend(_model_files(models_dir, model_version))
    if not existing:
        raise ValueError(f"折 {fold_dir.name} 未解析到任何消费文件: {root}")
    return existing, missing


def guarded_rescore_fold(
    arm_dir: Path, fold: str, out_dir: Path, *, score_fn=score_model_fingerprint
) -> Dict[str, object]:
    """单折补评分 + 前后身份核验 + 与既有折指纹/excluded_days 比对。"""
    arm_dir = Path(arm_dir)
    root = arm_dir / "root"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _read_json(arm_dir / "arm_meta.json")
    arm = str(meta["arm"])
    panel = arm == "v2_panel"
    fold_dir = arm_dir / "folds" / fold
    boundary = _read_json(fold_dir / "boundary.json")
    model_version = int((fold_dir / "model_version.txt").read_text(encoding="utf-8").strip())
    start, end = str(boundary["test_start"]), str(boundary["test_end"])
    expected = list_expected_oos_days(root, start, end)
    models_dir = root / "models" / "stock_selection"
    extra: Dict[str, object] = {}
    if panel:
        store = PanelDataStore(root)
        score_days = resolve_panel_score_days(expected, store, start, end)
        extra = {
            "source_mode": SOURCE_MODE_PANEL,
            "store": store,
            "feature_columns": _load_model_feature_columns(models_dir, model_version),
        }
    else:
        score_days = resolve_score_days(expected, root / "features")
        extra = {"source_mode": SOURCE_MODE_CS_TRAIN_VERIFY}

    inputs, missing_candidates = _fold_input_candidates(root, fold_dir, model_version, panel)
    before = out_dir / "consumption_before.json"
    collect_consumption_snapshot(
        inputs, base_dir=root, out_path=before, missing_files=missing_candidates
    )
    # 读取门（R2-T5AE-R2-01）：跑前缺失候选一旦被实际读取（含「创建→读取→移除」）
    # 即在读取动作处立即拒绝；已存在 raw 读取时点内容须与跑前登记身份相符。
    snapshot_records = json.loads(before.read_text(encoding="utf-8"))
    registered = {}
    for record in snapshot_records:
        if record.get("missing"):
            continue
        path = root / str(record["path"])
        if path.is_file():
            registered[path.resolve().as_posix()] = str(record["sha256"])
    with raw_read_guard(missing_paths=missing_candidates, registered_identities=registered):
        fingerprint = score_fn(
            models_dir=models_dir,
            model_version=model_version,
            data_root=root,
            score_days=score_days,
            main_board_codes=load_main_board_codes(root, stock_domain="main"),
            **extra,
        )
        raw_reads = [
            {"path": record.path, "sha256": record.sha256} for record in raw_read_ledger_snapshot()
        ]
    # 收尾闭合核验（R2-T5AE-R1-02）：重枚举现存候选 + 缺失候选态核验 + 逐文件身份
    rescanned, _ = _fold_input_candidates(root, fold_dir, model_version, panel)
    drift = verify_consumption_snapshot_closed(before, base_dir=root, rescan_paths=rescanned)
    existing = _read_json(fold_dir / "fingerprint.json")
    fp_match = fingerprint.get("fingerprint") == existing.get("fingerprint")
    days_match = sorted(str(day) for day in fingerprint.get("excluded_days") or []) == sorted(
        str(day) for day in existing.get("excluded_days") or []
    )
    verdict: Dict[str, object] = {
        "gate": "guarded_fold_rescore",
        "arm": arm,
        "fold": fold,
        "model_version": model_version,
        "fingerprint": fingerprint.get("fingerprint"),
        "existing_fingerprint": existing.get("fingerprint"),
        "fingerprint_match": fp_match,
        "excluded_days": fingerprint.get("excluded_days") or [],
        "existing_excluded_days": existing.get("excluded_days") or [],
        "excluded_days_match": days_match,
        "input_files": len(inputs),
        "missing_candidates": len(missing_candidates),
        "raw_reads": raw_reads,
        "after_drift": drift,
        "before_snapshot": str(before),
        "verdict": "PASS" if fp_match and days_match and not drift else "FAIL",
    }
    (out_dir / "rescore_verdict.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    logger.info(f"{arm}/{fold} 受保护补评分 verdict={verdict['verdict']}，证据: {out_dir}")
    return verdict


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="单折受保护补评分（闭市输入取证）")
    parser.add_argument("--arm-dir", required=True)
    parser.add_argument("--fold", default="split06")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    verdict = guarded_rescore_fold(Path(args.arm_dir), args.fold, Path(args.out_dir))
    return 0 if verdict["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
