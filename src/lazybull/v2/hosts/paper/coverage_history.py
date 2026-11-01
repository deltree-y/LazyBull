# -*- coding: utf-8 -*-
"""逐日缺失率滚动历史（B′ 决策 B3：近 N 日分位在线自校准）。

从 ``host.py`` 拆出（复杂度契约 §4.4：新增功能对应新增文件）。

载体 = 纸面状态根 ``runs/coverage_history.json``（**旁路诊断产物**，非账户状态
投影；不经 commit_day、不参与买卖判断；写盘失败告警不阻断主链路）：

```json
{"days": {"20261030": {"bp": 0.0, "cons_target_upside": 0.66}, ...},
 "updated_at": "2026-10-31 21:00:00"}
```

用途：就绪门门 4 的滚动分位基线（``coverage_gate._rolling_baseline``）——窗口
``ROLLING_WINDOW_DAYS``（60）、最少 ``ROLLING_MIN_DAYS``（20）日；样本不足时
门侧回落训练期静态分布（``baseline_source="static"``）。默认保留 90 日（窗口 +
余量），按日期升序裁剪。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Mapping

from loguru import logger

__all__ = [
    "HISTORY_FILENAME",
    "HISTORY_KEEP_DAYS",
    "append_coverage_rates",
    "load_coverage_history",
]

#: 历史文件名（runs 目录下）与保留天数（窗口 60 + 余量）
HISTORY_FILENAME = "coverage_history.json"
HISTORY_KEEP_DAYS = 90


def _history_path(runs_path: Path) -> Path:
    return Path(runs_path) / HISTORY_FILENAME


def load_coverage_history(runs_path: Path) -> dict[str, dict[str, float]]:
    """读取历史（缺失/损坏 ⇒ 空 dict + 告警；门侧自动回落静态基线）。"""
    path = _history_path(runs_path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"缺失率历史读取失败，本日回落训练期静态基线: {path}（{exc}）")
        return {}
    days = payload.get("days")
    if not isinstance(days, dict):
        logger.warning(f"缺失率历史结构非法（缺 days），回落静态基线: {path}")
        return {}
    history: dict[str, dict[str, float]] = {}
    rejected = 0
    for day, rates in days.items():
        if not isinstance(rates, dict):
            continue
        clean: dict[str, float] = {}
        for col, rate in rates.items():
            if isinstance(rate, bool) or not isinstance(rate, (int, float)):
                rejected += 1
                continue
            value = float(rate)
            # P1-3：历史只接受有限 [0,1] 率（NaN/Inf/越界拒绝，防污染后续基线）
            if (
                value != value
                or value in (float("inf"), float("-inf"))
                or not (0.0 <= value <= 1.0)
            ):
                rejected += 1
                continue
            clean[str(col)] = value
        if clean:
            history[str(day)] = clean
    if rejected:
        logger.warning(f"缺失率历史含 {rejected} 个非法值（NaN/Inf/越界/非数值），已剔除")
    return history


def append_coverage_rates(
    runs_path: Path,
    date: str,
    rates: Mapping[str, float],
    *,
    keep_days: int = HISTORY_KEEP_DAYS,
) -> None:
    """追加当日逐列缺失率（原子写；旁路失败只告警，不阻断主链路）。"""
    clean = {
        str(col): float(rate)
        for col, rate in rates.items()
        if isinstance(rate, (int, float))
        and not isinstance(rate, bool)
        and float(rate) == float(rate)
        and float(rate) not in (float("inf"), float("-inf"))
        and 0.0 <= float(rate) <= 1.0
    }
    if not clean:
        return
    path = _history_path(runs_path)
    history = load_coverage_history(runs_path)
    history[str(date)] = clean
    kept_days = sorted(history)[-int(keep_days) :]
    payload = {
        "days": {day: history[day] for day in kept_days},
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    tmp = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning(f"缺失率历史写盘失败（旁路，不阻断主链路）: {path}（{exc}）")
