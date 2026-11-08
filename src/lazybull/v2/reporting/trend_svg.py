# -*- coding: utf-8 -*-
"""趋势图元（静态内联 SVG；P6 T4）。

硬口径（设计 §4.3 / §7）：

- 每序列最多 ``trend_contract.MAX_CHART_POINTS`` 点：确定性抽样且**强制保留首尾与全局
  极值**（复用 T2 ``svg.downsample_indices`` 唯一实现）；
- 抽样只影响**图上的点**，不改变任何 CSV 行键/值（CSV 在 ``trend_csv`` 由完整行渲染）；
- 缺值点保留位置并**断开连线**（不插值、不补 0）；
- 全部数值固定小数位格式化，标签只经 ``text_safety.escape_text`` 进入文本节点；
- 纯函数、无 I/O、无随机数（同输入同输出，服务字节一致性门）。
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from src.lazybull.v2.reporting import svg, text_safety, trend_contract

__all__ = ["MAX_POINTS", "metric_chart"]

#: 图每序列点数上限（契约常量转导出）
MAX_POINTS = trend_contract.MAX_CHART_POINTS
#: 画布几何（固定，避免依赖渲染环境）
_W, _H, _PAD_L, _PAD_R, _PAD_T, _PAD_B = 900.0, 200.0, 56.0, 16.0, 14.0, 26.0
_LABEL = "trend-chart"


def _empty(message: str) -> str:
    """无点/单点时的固定尺寸说明（不绘制曲线）。"""
    return (
        f'<svg class="chart" viewBox="0 0 {_W:.0f} {_H:.0f}" role="img" '
        f'xmlns="http://www.w3.org/2000/svg" aria-label="chart-empty">'
        f'<text x="{_PAD_L:.0f}" y="{_H / 2:.0f}" class="chart-note">'
        f"{text_safety.escape_text(message)}</text></svg>"
    )


def _value_of(point: Mapping[str, Any]) -> float | None:
    """点值（缺失/非有限一律不可绘）。"""
    value = point.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _segments(kept: Sequence[int], values: Sequence[float | None]) -> list[list[int]]:
    """按缺值点切分连线（缺值两侧不连线；单点段不画线）。"""
    segments: list[list[int]] = []
    current: list[int] = []
    previous: int | None = None
    for index in kept:
        if values[index] is None:
            if len(current) > 1:
                segments.append(current)
            current = []
            previous = None
            continue
        if previous is not None and index != previous + 1:
            if len(current) > 1:
                segments.append(current)
            current = []
        current.append(index)
        previous = index
    if len(current) > 1:
        segments.append(current)
    return segments


def metric_chart(
    points: Sequence[Mapping[str, Any]],
    *,
    caption: str,
    unit_label: str,
    limit: int = MAX_POINTS,
) -> str:
    """单指标折线图（``points`` 形如 ``{run_id, window_end, value}``，按窗口升序）。"""
    rows = [dict(point) for point in points if isinstance(point, Mapping)]
    values = [_value_of(row) for row in rows]
    numeric = [value for value in values if value is not None]
    if len(numeric) < 2:
        return _empty(f"{caption}：可绘制点不足 2 个，不绘制曲线（不插值）")
    kept = svg.downsample_indices(
        [value if value is not None else float("nan") for value in values], limit
    )
    low, high = min(numeric), max(numeric)
    span = (high - low) or 1.0
    total = max(1, len(rows) - 1)
    width = _W - _PAD_L - _PAD_R
    height = _H - _PAD_T - _PAD_B
    xs = {index: _PAD_L + (index / total) * width for index in kept}
    ys = {
        index: _PAD_T + height - ((values[index] - low) / span) * height
        for index in kept
        if values[index] is not None
    }
    shapes: list[str] = []
    for segment in _segments(kept, values):
        shape = " ".join(f"{xs[index]:.2f},{ys[index]:.2f}" for index in segment if index in ys)
        shapes.append(f'<polyline class="trend-line" fill="none" points="{shape}"/>')
    endpoints = sorted({min(xs), max(xs)}) if xs else []
    for index in endpoints:
        if index in ys:
            shapes.append(
                f'<circle class="trend-point" cx="{xs[index]:.2f}" cy="{ys[index]:.2f}" r="2.5"/>'
            )
    shapes.extend(_labels(rows, xs))
    shapes.append(
        f'<text x="{_PAD_L:.0f}" y="{_PAD_T - 2:.0f}" class="chart-axis">'
        f"{text_safety.escape_text(unit_label)}｜min {low:.6f} / max {high:.6f}</text>"
    )
    return (
        f'<svg class="chart" viewBox="0 0 {_W:.0f} {_H:.0f}" role="img" '
        f'xmlns="http://www.w3.org/2000/svg" aria-label="{_LABEL}">'
        f"{''.join(shapes)}</svg>"
    )


def _labels(rows: Sequence[Mapping[str, Any]], xs: Mapping[int, float]) -> list[str]:
    """首尾窗口标签（固定锚点，来源文本只进文本节点）。"""
    if not xs:
        return []
    first, last = min(xs), max(xs)
    texts: list[str] = []
    for index, anchor in ((first, "start"), (last, "end")):
        label = str(rows[index].get("window_end") or "（无窗口）")
        texts.append(
            f'<text x="{xs[index]:.2f}" y="{_H - 8:.0f}" text-anchor="{anchor}" '
            f'class="chart-axis">{text_safety.escape_text(label)}</text>'
        )
    return texts
