# -*- coding: utf-8 -*-
"""静态内联 SVG 图元（设计 §6.1：无运行时 JavaScript、无外部图片）。

硬口径（P6 T2）：

- 只画**已有来源**的点：NAV 曲线只用经权威核对的行；点不足两个不画曲线（显式说明，
  不插值、不补 0）；
- 抽样只影响**图上的点**，不影响 projection/后续 CSV：``MAX_CHART_POINTS`` 内的确定性
  等距抽样并**强制保留首尾与全局极值**；
- 全部数值以固定小数位格式化（不受 locale 影响），标签文本走 ``text_safety.escape_text``；
- 本模块是纯函数、无 I/O、无随机数（同输入同输出，服务字节一致性门）。
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from src.lazybull.v2.reporting import text_safety

__all__ = [
    "MAX_CHART_POINTS",
    "downsample_indices",
    "nav_chart",
    "quality_bars",
]

#: 图上每序列最多点数（设计 §3“1,200 点抽样”在纸面 NAV 面的收紧实现）
MAX_CHART_POINTS = 600
#: 画布几何（固定，避免依赖渲染环境）
_W, _H, _PAD_L, _PAD_R, _PAD_T, _PAD_B = 900.0, 220.0, 56.0, 16.0, 16.0, 28.0
_MISSING_VALUE = "（无值）"


def downsample_indices(values: Sequence[float], limit: int = MAX_CHART_POINTS) -> list[int]:
    """确定性等距抽样：保留首尾与全局极值，其余等距取点（返回升序索引）。"""
    total = len(values)
    if limit < 4:
        raise ValueError(f"抽样上限至少 4 个点：{limit}")
    if total <= limit:
        return list(range(total))
    picks = {0, total - 1}
    numeric = [(index, float(value)) for index, value in enumerate(values)
               if isinstance(value, (int, float)) and math.isfinite(float(value))]
    if numeric:
        picks.add(min(numeric, key=lambda item: item[1])[0])
        picks.add(max(numeric, key=lambda item: item[1])[0])
    span = limit - len(picks)
    for step in range(span):
        picks.add(round(step * (total - 1) / max(1, span - 1)))
    return sorted(picks)


def _point(value: Any) -> Mapping[str, Any]:
    """图元输入点形态校验（缺值/非有限一律视为不可绘，不猜值）。"""
    if not isinstance(value, Mapping):
        raise ValueError(f"图元点必须是对象：{value!r}")
    return value


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


#: SVG 的 ``aria-label`` 一律用**固定常量**（来源文本不得进入属性位；设计 §6.4）
_LABEL_NAV = "nav-chart"
_LABEL_QUALITY = "quality-bars"
_LABEL_EMPTY = "chart-empty"


def _empty_chart(message: str, *, height: float = _H) -> str:
    """无点/单点时的不绘制说明（保持固定尺寸，避免布局跳动）。"""
    return (
        f'<svg class="chart" viewBox="0 0 {_W:.0f} {height:.0f}" role="img" '
        f'xmlns="http://www.w3.org/2000/svg" aria-label="{_LABEL_EMPTY}">'
        f'<text x="{_PAD_L:.0f}" y="{height / 2:.0f}" class="chart-note">'
        f"{text_safety.escape_text(message)}</text></svg>"
    )


def _segments(kept: Sequence[int], broken: frozenset[int]) -> list[list[int]]:
    """按“被核对失败点中断”切分连线（保留失败点两侧不连线的语义）。"""
    segments: list[list[int]] = []
    current: list[int] = []
    for index in kept:
        if index in broken:
            if len(current) > 1:
                segments.append(current)
            current = []
            continue
        if current and any(item in broken for item in range(current[-1] + 1, index)):
            if len(current) > 1:
                segments.append(current)
            current = []
        current.append(index)
    if len(current) > 1:
        segments.append(current)
    return segments


def _nav_geometry(
    series: Sequence[Mapping[str, Any]],
) -> tuple[list[int], dict[int, float], dict[int, float]]:
    """返回 ``(保留索引, 索引→x, 索引→y)``；值域取已核对点的最小/最大。"""
    values = [_finite(row.get("value")) for row in series]
    kept = downsample_indices([value if value is not None else float("nan") for value in values])
    numeric = [value for value in values if value is not None]
    low, high = min(numeric), max(numeric)
    span = (high - low) or 1.0
    total = max(1, len(series) - 1)
    width = _W - _PAD_L - _PAD_R
    height = _H - _PAD_T - _PAD_B
    xs = {index: _PAD_L + (index / total) * width for index in kept}
    ys = {
        index: _PAD_T + height - ((values[index] - low) / span) * height
        for index in kept
        if values[index] is not None
    }
    return kept, xs, ys


def _polyline(indexes: Sequence[int], xs: Mapping[int, float], ys: Mapping[int, float]) -> str:
    points = " ".join(f"{xs[index]:.2f},{ys[index]:.2f}" for index in indexes if index in ys)
    return f'<polyline class="nav-line" fill="none" points="{points}"/>'


def _markers(
    series: Sequence[Mapping[str, Any]],
    kept: Sequence[int],
    xs: Mapping[int, float],
    ys: Mapping[int, float],
) -> list[str]:
    shapes: list[str] = []
    for index in kept:
        if index not in ys:
            continue
        row = _point(series[index])
        css = "nav-point" if bool(row.get("matched")) else "nav-point nav-point--broken"
        shapes.append(
            f'<circle class="{css}" cx="{xs[index]:.2f}" cy="{ys[index]:.2f}" r="2.5"/>'
        )
    return shapes


def _labels(series: Sequence[Mapping[str, Any]], xs: Mapping[int, float]) -> list[str]:
    if not xs:
        return []
    first, last = min(xs), max(xs)
    rows = [
        (first, str(_point(series[first]).get("date") or _MISSING_VALUE), "start"),
        (last, str(_point(series[last]).get("date") or _MISSING_VALUE), "end"),
    ]
    texts: list[str] = []
    anchor = {"start": "start", "end": "end"}
    for index, label, side in rows:
        texts.append(
            f'<text x="{xs[index]:.2f}" y="{_H - 8:.0f}" text-anchor="{anchor[side]}" '
            f'class="chart-axis">{text_safety.escape_text(label)}</text>'
        )
    return texts


def nav_chart(
    series: Sequence[Mapping[str, Any]], *, caption: str = "已核对 NAV（总资产）"
) -> str:
    """NAV 折线图：``series`` 行形如 ``{date, value, matched}``（未核对点中断连线）。"""
    rows = [_point(row) for row in series]
    drawable = [row for row in rows if _finite(row.get("value")) is not None]
    if len(drawable) < 2:
        return _empty_chart(f"{caption}：已核对点不足 2 个，不绘制曲线（不插值）")
    kept, xs, ys = _nav_geometry(rows)
    # 断线集合必须来自**完整原序列**：被抽样跳过的失败点也要断开相邻连线（T2-R3-02）。
    # 只对 ``kept`` 求 ``broken`` 时，长历史中未被抽到的失败点会被直接跨越。
    broken = frozenset(
        index for index, row in enumerate(rows) if not bool(row.get("matched"))
    )
    shapes = [_polyline(segment, xs, ys) for segment in _segments(kept, broken)]
    shapes.extend(_markers(rows, kept, xs, ys))
    shapes.extend(_labels(rows, xs))
    return (
        f'<svg class="chart" viewBox="0 0 {_W:.0f} {_H:.0f}" role="img" '
        f'xmlns="http://www.w3.org/2000/svg" aria-label="{_LABEL_NAV}">'
        f"{''.join(shapes)}</svg>"
    )


def quality_bars(datasets: Sequence[Mapping[str, Any]], *, limit: int = 40) -> str:
    """质量数据集条形图（error 红 / warning 琥珀，按 error 降序、同值按名称稳定排序）。"""
    rows = [dict(_point(row)) for row in datasets[:limit]]
    if not rows:
        return _empty_chart("质量长表：无可绘制的数据集分组")
    rows.sort(key=lambda row: (-int(row.get("error_count") or 0), str(row.get("dataset") or "")))
    peak = max(
        [int(row.get("error_count") or 0) + int(row.get("warning_count") or 0) for row in rows]
        + [1]
    )
    width = 320.0
    height = 22.0 * len(rows) + 8.0
    shapes: list[str] = []
    for index, row in enumerate(rows):
        error = int(row.get("error_count") or 0)
        warning = int(row.get("warning_count") or 0)
        y = 6.0 + index * 22.0
        error_width = width * error / peak
        warning_width = width * warning / peak
        label = f"{row.get('layer') or '?'}/{row.get('dataset') or '?'}"
        shapes.append(
            f'<text x="0" y="{y + 12:.2f}" class="chart-axis">'
            f"{text_safety.escape_text(label)}</text>"
        )
        shapes.append(
            f'<rect class="bar bar--error" x="200" y="{y:.2f}" width="{error_width:.2f}" '
            'height="14"/>'
        )
        shapes.append(
            f'<rect class="bar bar--warning" x="{200 + error_width:.2f}" y="{y:.2f}" '
            f'width="{warning_width:.2f}" height="14"/>'
        )
    return (
        f'<svg class="chart chart--bars" viewBox="0 0 560 {height:.0f}" role="img" '
        f'xmlns="http://www.w3.org/2000/svg" aria-label="{_LABEL_QUALITY}">'
        f"{''.join(shapes)}</svg>"
    )
