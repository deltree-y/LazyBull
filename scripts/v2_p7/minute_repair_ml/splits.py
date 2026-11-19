"""半年刷新时钟、5 日采样相位与 label_end purge（协议 §8.2）。

时间口径：
  L = C 向前 36 自然月后的首个交易日
  V = C 向前 6 自然月后的首个交易日
  训练：L <= T < V 且 label_end_date < V
  验证：V <= T < C 且 label_end_date < C
  部署：L <= T < C 且 label_end_date < C（重训后用本折 OOS）
  OOS ：C <= T < 下一刷新日（末折截至主窗终点）
采样：以 2010-01-04 为公共锚，按完整日历整数距离 mod 5 取截面。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from scripts.v2_p7.minute_repair_ml import spec


class FoldError(RuntimeError):
    """折清单与协议要求不符（数量、跨度、采样日不足）。"""


def _shift_months(day: str, months: int) -> str:
    """YYYYMMDD 减若干自然月（月末按 pandas 规则回退）。"""
    stamp = pd.Timestamp(f"{day[:4]}-{day[4:6]}-{day[6:]}")
    return (stamp - pd.DateOffset(months=months)).strftime("%Y%m%d")


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
    while step <= 64:
        target = _shift_months(anchor, -months * step)
        if target < str(spec.WINDOW["cash_anchor"]):
            break
        day = _first_on_or_after(calendar, target)
        if day is None or day > limit:
            break
        points.append(day)
        step += 1
    return sorted(set(points))


def sampled_dates(calendar: Sequence[str]) -> set[str]:
    """公共采样相位：与锚点的整数日历距离 mod 5 == 0（全局锚点，不按折/股重置）。"""
    anchor = str(spec.SPLITS["sample_anchor"])
    every = int(spec.SPLITS["sample_every_trading_days"])
    if anchor not in calendar:
        anchor = calendar[0]
    start = list(calendar).index(anchor)
    return {day for i, day in enumerate(calendar) if i >= start and (i - start) % every == 0}


def build_folds(calendar: Sequence[str], end: str | None = None) -> list[dict[str, object]]:
    """构造全部刷新折（含 L/V/C 与可达训练/验证/OOS 日期）。"""
    limit = end or calendar[-1]
    points = refresh_dates(calendar, limit)
    sample = sampled_dates(calendar)
    folds: list[dict[str, object]] = []
    for index, refresh in enumerate(points):
        low = _first_on_or_after(calendar, _shift_months(refresh, int(spec.SPLITS["train_months"])))
        val = _first_on_or_after(calendar, _shift_months(refresh, int(spec.SPLITS["val_months"])))
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


def assert_fold_count(folds: Sequence[dict[str, object]]) -> None:
    """16 折清单必须完整，不得按有数据的折数静默缩减。"""
    expected = int(spec.SPLITS["expected_folds"])
    if len(folds) != expected:
        identifiers = [f["fold_id"] for f in folds]
        raise FoldError(f"折数 {len(folds)} != 冻结值 {expected}：{identifiers}")


def check_fold_requirements(fold: dict[str, object]) -> list[str]:
    """每折最低条件（跨度/抽样日）；返回未满足项（空表示通过）。"""
    problems: list[str] = []
    low = pd.Timestamp(f"{str(fold['L'])[:4]}-{str(fold['L'])[4:6]}-{str(fold['L'])[6:]}")
    high = pd.Timestamp(f"{str(fold['V'])[:4]}-{str(fold['V'])[4:6]}-{str(fold['V'])[6:]}")
    span_months = (high - low).days / 30.44
    need_span = float(spec.SPLITS["min_train_span_months"])
    if span_months < need_span:
        problems.append(f"train_span_{span_months:.1f}_months_lt_{need_span:g}")
    if int(fold["n_train_days"]) < int(spec.SPLITS["min_sampled_train_days"]):
        problems.append(
            f"train_days_{fold['n_train_days']}_lt_{spec.SPLITS['min_sampled_train_days']}"
        )
    if int(fold["n_val_days"]) < int(spec.SPLITS["min_sampled_val_days"]):
        problems.append(f"val_days_{fold['n_val_days']}_lt_{spec.SPLITS['min_sampled_val_days']}")
    return problems


def fold_table(folds: Iterable[dict[str, object]]) -> pd.DataFrame:
    """折清单（去掉日期列表，便于落盘）。"""
    return pd.DataFrame(
        [{k: v for k, v in fold.items() if not isinstance(v, list)} for fold in folds]
    )


def purge_mask(frame: pd.DataFrame, cutoff: str) -> pd.Series:
    """标签必须在该段结束前完全成熟（严格小于 cutoff）。"""
    return frame["label_end_date"].astype(str) < str(cutoff)


def split_masks(
    frame: pd.DataFrame, fold: dict[str, object], require_valid: bool = True
) -> tuple[pd.Series, pd.Series]:
    """诊断模型的训练/验证行掩码（日期窗 + purge + 仅 valid）。"""
    dates = frame["trade_date"].astype(str)
    train = dates.isin(set(fold["train_days"])) & purge_mask(frame, str(fold["V"]))
    val = dates.isin(set(fold["val_days"])) & purge_mask(frame, str(fold["C"]))
    if require_valid:
        train &= frame["label_status"].astype(str) == "valid"
        val &= frame["label_status"].astype(str) == "valid"
    return train, val


def deploy_mask(
    frame: pd.DataFrame, fold: dict[str, object], require_valid: bool = True
) -> pd.Series:
    """部署模型训练行：L <= T < C 且标签在 C 前成熟。"""
    dates = frame["trade_date"].astype(str)
    mask = dates.isin(set(fold["train_days"]) | set(fold["val_days"])) & purge_mask(
        frame, str(fold["C"])
    )
    if require_valid:
        mask &= frame["label_status"].astype(str) == "valid"
    return mask


def oos_mask(frame: pd.DataFrame, fold: dict[str, object]) -> pd.Series:
    """OOS 行：C <= T < 下一刷新日（评分不需要未来标签）。"""
    return frame["trade_date"].astype(str).isin(set(fold["oos_days"]))
