# -*- coding: utf-8 -*-
"""coverage_history.json 初始种子（B′ 决策 B3 补充，2026-10-31 用户批准）。

复杂度豁免：CLAUDE.md §7.5（阶段专用运维工具按临时开发文件处理）。

背景
====
滚动分位基线（`coverage_gate._rolling_baseline`）要求近 60 日窗口内 ≥20 日样本，
否则回落训练期静态分布。上线首日起 `runs/coverage_history.json` 为空 ⇒ 前 20 个
交易日均走静态基线（零点/贴边列噪音）。本工具把 **76 日影子对照的域内逐日缺失率**
（`shadow_76d.json` 的 `rows[].rates_domain`）写入生产纸面根的滚动历史，使上线当日
即可用滚动基线。

约束
====
- 只写 `<paper_root>/runs/coverage_history.json` 一个文件；已存在 ⇒ 备份后覆盖；
- 默认 dry-run（打印摘要）；`--write` 才落盘；原子写 + 读回校验（`load_coverage_history`）；
- 仅接受 ≥20 日（滚动最小样本）的源数据；保留最近 90 日（模块口径）。

用法
====
``python scripts/v2_p4/seed_coverage_history.py``（dry-run）
``python scripts/v2_p4/seed_coverage_history.py --write``
``python scripts/v2_p4/seed_coverage_history.py --shadow-json <path> --paper-root <path>``
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

from src.lazybull.common.config import get_paper_root  # noqa: E402
from src.lazybull.v2.hosts.paper.coverage_gate import ROLLING_MIN_DAYS  # noqa: E402
from src.lazybull.v2.hosts.paper.coverage_history import (  # noqa: E402
    HISTORY_FILENAME,
    HISTORY_KEEP_DAYS,
    load_coverage_history,
)


def _latest_shadow_json() -> Path:
    candidates = sorted(
        (_REPO_ROOT / "data" / "frozen_reference" / "v2_p4").glob("coverage_shadow_76d_*")
    )
    if not candidates:
        raise FileNotFoundError("未找到 coverage_shadow_76d_* 证据目录（先跑 76 日对照）")
    return candidates[-1] / "shadow_76d.json"


def _extract_days(shadow_json: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(shadow_json.read_text(encoding="utf-8"))
    days: dict[str, dict[str, float]] = {}
    for row in payload.get("rows", []):
        if row.get("error") is not None:
            continue
        # 与生产一致：阻断日不回灌滚动历史（reviewer3 R2-P2-3）
        if row.get("blocked"):
            continue
        rates = row.get("rates_domain") or {}
        if rates:
            days[str(row["date"])] = {str(col): float(rate) for col, rate in rates.items()}
    return days


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="写入 coverage_history.json 初始种子")
    parser.add_argument("--shadow-json", default=None)
    parser.add_argument("--paper-root", default=None)
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="允许覆盖含种子窗外活数据的现有历史（默认拒绝）"
    )
    args = parser.parse_args(argv)

    shadow_json = Path(args.shadow_json) if args.shadow_json else _latest_shadow_json()
    paper_root = Path(args.paper_root or get_paper_root())
    target = paper_root / "runs" / HISTORY_FILENAME

    days = _extract_days(shadow_json)
    kept = sorted(days)[-HISTORY_KEEP_DAYS:]
    if len(kept) < ROLLING_MIN_DAYS:
        raise ValueError(f"源数据仅 {len(kept)} 日 < 滚动最小样本 {ROLLING_MIN_DAYS} 日，拒绝写入")
    columns = {col for day in kept for col in days[day]}
    print(f"源: {shadow_json}")
    print(f"目标: {target}")
    print(f"日数: {len(kept)}（{kept[0]} ~ {kept[-1]}）｜列数: {len(columns)}")

    if not args.write:
        logger.warning("dry-run：未写入（加 --write 落盘）")
        return 0

    # 源数据先全量校验（reviewer3 R2-P1-4）：非法率不得进入生产历史
    for day, rates in ((day, days[day]) for day in kept):
        for col, rate in rates.items():
            if not isinstance(rate, float) or not (0.0 <= rate <= 1.0) or rate != rate:
                raise SystemExit(f"源数据非法率（{day}/{col}={rate!r}）：拒绝写入")

    runs_path = target.parent
    runs_path.mkdir(parents=True, exist_ok=True)
    backup: Optional[Path] = None
    if target.exists():
        # 严格解析现有文件（reviewer3 R2-P1-4）：损坏/结构非法默认拒绝，不得静默覆盖
        try:
            existing_payload = json.loads(target.read_text(encoding="utf-8"))
            existing_days = existing_payload["days"]
            if not isinstance(existing_days, dict):
                raise ValueError("days 不是对象")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            if not args.force:
                raise SystemExit(
                    f"现有历史无法严格解析（{target}：{exc}）：默认拒绝覆盖"
                    "（reviewer3 R2-P1-4）；确需覆盖请显式 --force"
                ) from exc
            existing_days = {}
            logger.warning("现有历史损坏但 --force 指定覆盖，将继续（已备份原文件）")
        future_live = sorted(day for day in existing_days if day > kept[-1])
        if future_live and not args.force:
            raise SystemExit(
                f"现有历史含种子窗外活数据（{future_live[:5]}…共 {len(future_live)} 日）："
                "拒绝回退覆盖（reviewer1 P2-4）；确需覆盖请显式 --force"
            )
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup = target.with_name(f"{target.name}.bak_{stamp}")
        shutil.copy2(target, backup)
        logger.warning(f"已存在历史，先备份: {backup}")

    payload = {
        "days": {day: days[day] for day in kept},
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, target)

    # 全量回读（reviewer3 P2-3）：逐日逐列比较，不只比末日/单键
    readback = load_coverage_history(runs_path)
    expected = {day: days[day] for day in kept}
    if readback != expected:
        diff_days = sorted(set(readback) ^ set(expected))[:5]
        # 回读失败恢复备份（reviewer3 R2-P1-4）
        if backup is not None and backup.exists():
            shutil.copy2(backup, target)
        else:
            target.unlink(missing_ok=True)
        raise RuntimeError(f"读回校验失败：历史与写入内容不一致（差异日 {diff_days}）；已恢复备份")
    print(f"写入完成：{len(readback)} 日；读回校验 PASS；备份={backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
