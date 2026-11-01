# -*- coding: utf-8 -*-
"""结构性漂移单列豁免登记（B′ 决策 a，2026-11-01 用户批准）。

复杂度豁免：CLAUDE.md §7.5（阶段专用运维工具按临时开发文件处理）。

背景
====
76 日影子对照（决策 b 后）仍有 6 静态 / 8 滚动列 error 交易日 >20%，全部为
一致预期 `cons_*` 族与估值 `zscore_pe_ttm*` 族的结构性漂移（长期贴边/高于训练
p99）。用户批准按 B1② 原文**单列登记豁免**：豁免列 error 封顶为 warning，
不参与聚合升级计数；**全空（100%）不豁免**。

登记载体（可审计，禁止裸列名）
==============================
`train_params["feature_missing_rate_drift_exempt"] = {columns, registered_at,
reason, evidence, review_condition, review_by}`——`model_schema.read_drift_exempt`
强校验三要素（columns/registered_at/review_condition），缺失即不启用。

写入
====
sidecar `v{N}_metadata.json` + `model_registry.json` 条目**定向**更新（双份一致），
备份 + 原子写 + 回读校验；证据 JSON 落 `data/frozen_reference/v2_p4/`。
默认 dry-run。

用法
====
``python scripts/v2_p4/register_drift_exempt.py``（dry-run）
``python scripts/v2_p4/register_drift_exempt.py --write``
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from loguru import logger  # noqa: E402

from src.lazybull.common.config import get_stock_selection_models_root  # noqa: E402
from src.lazybull.v2.common.model_schema import (  # noqa: E402
    DRIFT_EXEMPT_KEY,
    read_drift_exempt,
    read_feature_missing_rate,
)

REVIEW_CONDITION = (
    "观察期 ≥10 交易日复核（切换后第 10 个交易日）：任一豁免列 error 率 >50%、"
    "出现未登记列 >20% 交易日、或豁免列全空 ⇒ 重新裁决；20 交易日内重估分位或重训收敛"
)


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def build_block(columns: Sequence[str], evidence_rel: str, reason: str | None = None) -> dict:
    return {
        "columns": list(dict.fromkeys(columns)),
        "registered_at": datetime.now().strftime("%Y-%m-%d"),
        "reason": reason
        or "76 日影子对照：结构性漂移列（一致预期 cons_* 族 + 估值 zscore_pe_ttm* 族），"
        "长期贴边/高于训练 p99；决策 a 单列登记（error 封顶 warning，不参与聚合计数）",
        "evidence": evidence_rel,
        "review_condition": REVIEW_CONDITION,
        "review_by": "切换后第 10 个交易日（预登记 20261113，实际以首个生产驱动日起算）",
    }


def _columns_from_shadow(shadow_json: Path) -> list[str]:
    """从影子对照证据自动推导「超 20% 交易日」列（新模型登记的唯一推荐入口）。"""
    payload = json.loads(shadow_json.read_text(encoding="utf-8"))
    freq = (payload.get("summary") or {}).get("freq") or []
    columns = [item["column"] for item in freq if item.get("over_20pct")]
    if not columns:
        raise ValueError(f"影子证据无超 20% 列（无需登记）: {shadow_json}")
    return columns


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="登记结构性漂移单列豁免（决策 a）")
    parser.add_argument("--version", type=int, default=24022)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--evidence-dir", default=None)
    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="登记列（缺省：v24022 用内置 9 列；其他模型必须显式给出或 --from-shadow）",
    )
    parser.add_argument(
        "--from-shadow",
        default=None,
        help="从 shadow_76d.json 自动推导超 20% 交易日列（新模型推荐入口）",
    )
    parser.add_argument("--reason", default=None)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    if args.from_shadow:
        columns = _columns_from_shadow(Path(args.from_shadow))
    elif args.columns:
        columns = list(args.columns)
    else:
        raise SystemExit(
            "登记列必须显式给出：--columns 或 --from-shadow（禁止内置清单/跨模型复制）"
        )

    models_dir = Path(args.models_dir or get_stock_selection_models_root())
    sidecar = models_dir / f"v{args.version}_metadata.json"
    registry_file = models_dir / "model_registry.json"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    evidence_dir = Path(
        args.evidence_dir
        or (_REPO_ROOT / "data" / "frozen_reference" / "v2_p4" / f"register_drift_exempt_{stamp}")
    )
    evidence_rel = str(
        (evidence_dir / "registration.json").relative_to(_REPO_ROOT)
        if str(evidence_dir).startswith(str(_REPO_ROOT))
        else evidence_dir / "registration.json"
    )

    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    params = dict(metadata.get("train_params") or {})
    distribution, error = read_feature_missing_rate(params)
    if error is not None:
        raise ValueError(f"v{args.version} 未通过分布校验（{error}）——先补录再登记豁免")
    unknown = [col for col in columns if col not in distribution]
    if unknown:
        raise ValueError(f"登记列不在分布键集内（拒绝登记幽灵列）: {unknown}")

    block = build_block(columns, evidence_rel, args.reason)
    print(f"目标模型: v{args.version}（{sidecar}）")
    print(f"登记列（{len(block['columns'])}）: {block['columns']}")
    print(f"复核条件: {block['review_condition']}")
    print(f"证据: {evidence_rel}")

    if not args.write:
        logger.warning("dry-run：未写入（加 --write 落盘）")
        return 0

    params[DRIFT_EXEMPT_KEY] = block
    metadata["train_params"] = params
    sidecar_backup = sidecar.with_name(f"{sidecar.name}.bak_{stamp}")
    shutil.copy2(sidecar, sidecar_backup)
    _atomic_write_json(sidecar, metadata)

    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    entry = next(
        (
            item
            for item in registry.get("models", [])
            if int(item.get("version", -1)) == args.version
        ),
        None,
    )
    if entry is None:
        raise ValueError(f"model_registry.json 未找到 v{args.version} 条目")
    entry_params = dict(entry.get("train_params") or {})
    entry_params[DRIFT_EXEMPT_KEY] = block
    entry["train_params"] = entry_params
    registry_backup = registry_file.with_name(f"{registry_file.name}.bak_{stamp}")
    shutil.copy2(registry_file, registry_backup)
    _atomic_write_json(registry_file, registry)

    sidecar_after = json.loads(sidecar.read_text(encoding="utf-8"))["train_params"]
    registry_after = next(
        item
        for item in json.loads(registry_file.read_text(encoding="utf-8"))["models"]
        if int(item.get("version", -1)) == args.version
    )["train_params"]
    if sidecar_after.get(DRIFT_EXEMPT_KEY) != registry_after.get(DRIFT_EXEMPT_KEY):
        raise RuntimeError("双份回读不一致：sidecar 与 registry 登记块不同值")
    readback = read_drift_exempt(sidecar_after)
    if readback != frozenset(columns):
        raise RuntimeError(f"回读校验失败：生效豁免集 {sorted(readback)} 与登记清单不一致")

    evidence_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        evidence_dir / "registration.json",
        {
            "model_version": args.version,
            "registered_at": block["registered_at"],
            "block": block,
            "readback_equal": True,
            "sidecar_backup": str(sidecar_backup),
            "registry_backup": str(registry_backup),
        },
    )
    print(f"登记完成：双份一致 + 读回 PASS；证据 {evidence_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
