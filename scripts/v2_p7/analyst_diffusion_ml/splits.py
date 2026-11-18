"""半年刷新时钟、5 日采样相位与 label_end purge（协议 §8.2）。

时间口径：
  L = C 向前 72 自然月后的首个交易日
  V = C 向前 6 自然月后的首个交易日
  训练：L <= T < V 且 label_end_date < V
  验证：V <= T < C 且 label_end_date < C
  OOS ：C <= T < 下一刷新日（末期截至主窗终点）
采样：以 2010-01-04 为公共锚，按完整日历的整数距离 mod 5 取截面。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from scripts.v2_p7.analyst_diffusion_ml import spec


def _shift_months(day: str, months: int) -> str:
    """YYYYMMDD 减若干自然月（月末按 pandas 规则回退）。"""
    stamp = pd.Timestamp(f"{day[:4]}-{day[4:6]}-{day[6:]}")
    shifted = stamp - pd.DateOffset(months=months)
    return shifted.strftime("%Y%m%d")


def _first_on_or_after(calendar: Sequence[str], target: str) -> str | None:
    """首个 >= target 的交易日。"""
    for day in calendar:
        if day >= target:
            return day
    return None


def refresh_dates(calendar: Sequence[str], end: str | None = None) -> list[str]:
    """刷新时钟：锚点起每 6 自然月取首个可用交易日。"""
    anchor = str(spec.SPLITS["refresh_anchor"])
    months = int(spec.SPLITS["refresh_months"])
    limit = end or calendar[-1]
    points: list[str] = []
    step = 0
    while True:
        target = _shift_months(anchor, -months * step)
        if target < str(spec.WINDOW["cash_anchor"]):
            break
        day = _first_on_or_after(calendar, target)
        if day is None or day > limit:
            break
        points.append(day)
        step += 1
        if step > 64:
            break
    return sorted(set(points))


def sampled_dates(calendar: Sequence[str]) -> set[str]:
    """公共采样相位：与锚点的整数日历距离 mod 5 == 0。"""
    anchor = str(spec.SPLITS["sample_anchor"])
    every = int(spec.SPLITS["sample_every_trading_days"])
    if anchor not in calendar:
        anchor = calendar[0]
    start = list(calendar).index(anchor)
    return {day for i, day in enumerate(calendar) if i >= start and (i - start) % every == 0}


def build_folds(calendar: Sequence[str], end: str | None = None) -> list[dict[str, object]]:
    """构造全部刷新折（含 L/V/C 与可达训练/验证日期）。"""
    limit = end or calendar[-1]
    points = refresh_dates(calendar, limit)
    sample = sampled_dates(calendar)
    folds: list[dict[str, object]] = []
    for index, refresh in enumerate(points):
        train_months = int(spec.SPLITS["train_months"])
        val_months = int(spec.SPLITS["val_months"])
        low = _first_on_or_after(calendar, _shift_months(refresh, train_months))
        val = _first_on_or_after(calendar, _shift_months(refresh, val_months))
        if low is None or val is None:
            continue
        train_days = [d for d in calendar if low <= d < val and d in sample]
        val_days = [d for d in calendar if val <= d < refresh and d in sample]
        next_refresh = points[index + 1] if index + 1 < len(points) else None
        oos_days = [
            d
            for d in calendar
            if d >= refresh and (next_refresh is None or d < next_refresh) and d <= limit
        ]
        folds.append(
            {
                "fold_id": f"f{index + 1:02d}",
                "C": refresh,
                "V": val,
                "L": low,
                "train_first": train_days[0] if train_days else None,
                "train_last": train_days[-1] if train_days else None,
                "val_first": val_days[0] if val_days else None,
                "val_last": val_days[-1] if val_days else None,
                "n_train_days": len(train_days),
                "n_val_days": len(val_days),
                "oos_first": oos_days[0] if oos_days else None,
                "oos_last": oos_days[-1] if oos_days else None,
                "n_oos_days": len(oos_days),
                "train_days": train_days,
                "val_days": val_days,
                "oos_days": oos_days,
            }
        )
    return folds


def fold_table(folds: Iterable[dict[str, object]]) -> pd.DataFrame:
    """折清单（去掉日期列表，便于落盘）。"""
    rows = []
    for fold in folds:
        rows.append({k: v for k, v in fold.items() if not isinstance(v, list)})
    return pd.DataFrame(rows)


def purge_mask(
    frame: pd.DataFrame, cutoff: str, columns: tuple[str, ...] = ("trade_date", "label_end_date")
) -> pd.Series:
    """标签必须在该段结束前完全成熟（严格小于 cutoff）。"""
    return frame[columns[1]].astype(str) < str(cutoff)


def split_masks(
    frame: pd.DataFrame, fold: dict[str, object], require_valid: bool = True
) -> tuple[pd.Series, pd.Series]:
    """训练/验证行掩码（日期窗 + purge + 仅 valid 标签）。"""
    dates = frame["trade_date"].astype(str)
    train = dates.isin(set(fold["train_days"])) & purge_mask(frame, str(fold["V"]))
    val = dates.isin(set(fold["val_days"])) & purge_mask(frame, str(fold["C"]))
    if require_valid:
        train &= frame["label_status"].astype(str) == "valid"
        val &= frame["label_status"].astype(str) == "valid"
    return train, val


def check_fold_requirements(fold: dict[str, object]) -> list[str]:
    """每折最低条件检查；返回未满足项（空表示通过）。"""
    problems: list[str] = []
    months = int(spec.SPLITS["min_train_span_months"])
    low = pd.Timestamp(f"{str(fold['L'])[:4]}-{str(fold['L'])[4:6]}-{str(fold['L'])[6:]}")
    high = pd.Timestamp(f"{str(fold['V'])[:4]}-{str(fold['V'])[4:6]}-{str(fold['V'])[6:]}")
    span_months = (high - low).days / 30.44
    if span_months < months:
        problems.append(f"train_span_{span_months:.1f}_months_lt_{months}")
    if int(fold["n_train_days"]) < int(spec.SPLITS["min_sampled_train_days"]):
        need_train = spec.SPLITS["min_sampled_train_days"]
        problems.append(f"train_days_{fold['n_train_days']}_lt_{need_train}")
    if int(fold["n_val_days"]) < int(spec.SPLITS["min_sampled_val_days"]):
        problems.append(f"val_days_{fold['n_val_days']}_lt_{spec.SPLITS['min_sampled_val_days']}")
    return problems
