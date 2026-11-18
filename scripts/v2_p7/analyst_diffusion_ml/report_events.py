"""研报事件层：身份去重、冲突隔离、同分析师同 FY 配对与 revision 事件。

口径（协议 §4.3~§5.3）：
- 时点只认 `report_date <= T`（T 日收盘后进入信息集，T+1 执行）；
- 行级键含 quarter，研报级键不含 quarter；经济字段冲突整组隔离、不按行序取末条；
- 分析师团队 = (org_name, author_name) 均非空的组合；缺任一者不投票，只计年龄；
- `absolute_fy = parse_quarter_year(quarter)`，只允许 FY0/FY1 进入 EPS 主特征；
- 每篇研报 EPS 主预测期唯一：FY1 有效则 FY1，否则 FY0；
- 配对窗口 180 自然日，首次覆盖不是上调/下调。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from scripts.v2_p7.analyst_diffusion_ml import spec
from scripts.v2_p7.common import sha256_bytes
from src.lazybull.factors.consensus import parse_quarter_year, rating_to_score

FIELDS: tuple[str, ...] = ("eps", "target", "rating")
_SEP = "\u0001"


def _emit(log: Callable[[str], None] | None) -> Callable[[str], None]:
    return log or (lambda message: None)


def _join_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """列拼接（向量化，避免逐行 apply）。

    必须先 `fillna("")`：pandas 3.0 的 `astype(str)` 会把 NaN 保留为缺失值，
    拼接后整行变 NaN，会让"同身份不同经济值"的冲突检测失效。
    """
    def _text(name: str) -> pd.Series:
        return frame[name].fillna("").astype(str)

    text = _text(columns[0])
    for name in columns[1:]:
        text = text + _SEP + _text(name)
    return text


def load_report_rc(root: Path, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """按年分区读取 report_rc（只读）。"""
    directory = Path(root) / "data/raw/report_rc"
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"report_rc 分区为空：{directory}")
    raw = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    raw["report_date"] = raw["report_date"].astype(str)
    if start is not None:
        raw = raw[raw["report_date"] >= str(start)]
    if end is not None:
        raw = raw[raw["report_date"] <= str(end)]
    return raw.reset_index(drop=True)


def normalize(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """规范化 + 完整重复去重 + 经济字段冲突隔离（冲突行留原行 SHA）。"""
    work = frame.copy()
    work["report_date"] = work["report_date"].astype(str).str.replace("-", "", regex=False)
    valid_date = work["report_date"].str.fullmatch(r"\d{8}").fillna(False)
    dropped_date = int((~valid_date).sum())
    work = work[valid_date].copy()

    for name in ("org_name", "author_name"):
        work[name] = work[name].fillna("").astype(str).str.strip()
    work["report_title"] = work["report_title"].fillna("").astype(str).str.strip()
    work["quarter"] = work["quarter"].fillna("").astype(str).str.strip()
    work["rating"] = work["rating"].fillna("").astype(str)

    before = len(work)
    work = work.drop_duplicates(keep="first")
    exact_duplicates = before - len(work)

    identity = ["ts_code", "report_date", "org_name", "author_name", "report_title", "quarter"]
    work["_identity"] = _join_columns(work, identity)
    econ = ["eps", "max_price", "min_price", "rating"]
    work["_econ"] = _join_columns(work, econ)
    spread = work.groupby("_identity", sort=False)["_econ"].transform("nunique")
    conflict_mask = spread > 1
    conflict_ids = sorted(set(work.loc[conflict_mask, "_identity"].tolist()))
    conflict_sha: list[str] = []
    if conflict_ids:
        keys = [
            "ts_code",
            "report_date",
            "report_title",
            "org_name",
            "author_name",
            "quarter",
            "eps",
            "max_price",
            "min_price",
            "rating",
        ]
        conflicted = work[conflict_mask]
        conflict_sha = sorted(
            set(_join_columns(conflicted, keys).str.encode("utf-8").map(sha256_bytes).tolist())
        )[:20]
        work = work[~work["_identity"].isin(conflict_ids)].copy()

    audit = {
        "rows_before": int(before),
        "dropped_invalid_date": dropped_date,
        "dropped_exact_duplicate": int(exact_duplicates),
        "conflict_identity_groups": len(conflict_ids),
        "conflict_rows": int(conflict_mask.sum()),
        "conflict_row_sha": conflict_sha,
        "rows_after": int(len(work)),
    }
    return work, audit


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def visibility_map(
    report_dates: pd.Series, trading_days: Sequence[str], lag_trading_days: int = 0
) -> np.ndarray:
    """`report_date` -> 首个可见交易日（主口径 0 延迟时与 `report_date <= T` 等价）。

    研报在非交易日发布时，主口径下个交易日起可见；延迟压力把可见日再推
    `lag_trading_days` 个交易日（协议 §13.3 第 1 条）。超出日历返回空串。
    """
    days = np.array(sorted({str(day) for day in trading_days}))
    if len(days) == 0:
        raise ValueError("交易日历为空")
    values = report_dates.astype(str).to_numpy()
    index = np.searchsorted(days, values, side="left") + int(lag_trading_days)
    out = np.where(index < len(days), days[np.clip(index, 0, len(days) - 1)], "")
    out = np.where(index >= len(days), "", out)
    return out


def prepare_fields(
    frame: pd.DataFrame,
    trading_days: Sequence[str] | None = None,
    lag_trading_days: int = 0,
) -> pd.DataFrame:
    """解析财年、数值字段与评级分；不在此处做聚合。

    传入交易日历时，`report_date` 被替换为**可见交易日**（财年锚点仍用原始报告年）。
    """
    work = frame.copy()
    work["eps_value"] = _numeric(work["eps"])
    max_price = _numeric(work["max_price"])
    min_price = _numeric(work["min_price"])
    positive_max = max_price.where(max_price > 0)
    positive_min = min_price.where(min_price > 0)
    work["target_value"] = pd.concat([positive_max, positive_min], axis=1).mean(
        axis=1, skipna=True
    )
    work["rating_value"] = work["rating"].map(rating_to_score).astype(float)
    work["fy_year"] = work["quarter"].map(parse_quarter_year)
    work["report_year"] = work["report_date"].str[:4].astype(int)
    if trading_days is not None:
        work["report_date_original"] = work["report_date"]
        work["report_date"] = visibility_map(
            work["report_date"], trading_days, lag_trading_days
        )
        work = work[work["report_date"] != ""].copy()
    work["named_analyst"] = (work["org_name"] != "") & (work["author_name"] != "")
    work["analyst_team"] = np.where(
        work["named_analyst"], work["org_name"] + "|" + work["author_name"], ""
    )
    return work


def eps_table(frame: pd.DataFrame) -> pd.DataFrame:
    """EPS 表：`(ts_code, analyst, report_date, absolute_fy)` 取中位数。"""
    work = frame[
        frame["named_analyst"] & frame["eps_value"].notna() & frame["fy_year"].notna()
    ].copy()
    work["absolute_fy"] = work["fy_year"].astype(int)
    keep = (work["absolute_fy"] == work["report_year"]) | (
        work["absolute_fy"] == work["report_year"] + 1
    )
    work = work[keep]
    if work.empty:
        return pd.DataFrame(
            columns=["ts_code", "analyst_team", "report_date", "absolute_fy", "eps"]
        )
    return (
        work.groupby(["ts_code", "analyst_team", "report_date", "absolute_fy"], sort=False)[
            "eps_value"
        ]
        .median()
        .reset_index()
        .rename(columns={"eps_value": "eps"})
    )


def report_level_table(frame: pd.DataFrame) -> pd.DataFrame:
    """研报级表：先按研报身份取中位数，再按 (股票, 分析师, 日期) 取中位数。

    同日同一分析师的多篇研报无法区分先后，协议 §5.4 要求"同一分析师只投最新一票"，
    因此在日期层做一次中位数合并（登记于 manifest 的 `same_day_aggregation`）。
    """
    identity = frame[
        frame["named_analyst"] & (frame["target_value"].notna() | frame["rating_value"].notna())
    ].copy()
    if identity.empty:
        return pd.DataFrame(columns=["ts_code", "analyst_team", "report_date", "target", "rating"])
    by_report = (
        identity.groupby(["ts_code", "analyst_team", "report_date", "report_title"], sort=False)[
            ["target_value", "rating_value"]
        ]
        .median()
        .reset_index()
    )
    return (
        by_report.groupby(["ts_code", "analyst_team", "report_date"], sort=False)[
            ["target_value", "rating_value"]
        ]
        .median()
        .reset_index()
        .rename(columns={"target_value": "target", "rating_value": "rating"})
    )


def activity_table(frame: pd.DataFrame) -> pd.DataFrame:
    """报告活动表：每篇研报的日期与分析团队（含无作者者，供年龄与覆盖分母）。"""
    work = frame[["ts_code", "report_date", "analyst_team", "named_analyst"]].copy()
    return work.drop_duplicates()


def _pair_series(
    frame: pd.DataFrame, value_column: str, *, lookback_days: int
) -> pd.DataFrame:
    """同一 (股票, 分析师) 序列内配对前一份有效值（≤180 自然日）。"""
    columns = [
        "ts_code",
        "analyst_team",
        "report_date",
        value_column + "_new",
        value_column + "_old",
        "gap_days",
    ]
    work = frame[frame[value_column].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=columns)
    work = work.sort_values(["ts_code", "analyst_team", "report_date"], kind="mergesort")
    work["_ord"] = pd.to_datetime(work["report_date"], format="%Y%m%d").map(
        pd.Timestamp.toordinal
    )
    grouped = work.groupby(["ts_code", "analyst_team"], sort=False)
    work["_ord_prev"] = grouped["_ord"].shift(1)
    work[value_column + "_old"] = grouped[value_column].shift(1)
    work["gap_days"] = work["_ord"] - work["_ord_prev"]
    work = work[work["gap_days"].notna() & (work["gap_days"] <= lookback_days)].copy()
    work = work.rename(columns={value_column: value_column + "_new"})
    work["gap_days"] = work["gap_days"].astype(int)
    return work[columns]


def eps_revision_events(primary: pd.DataFrame) -> pd.DataFrame:
    """同分析师同绝对 FY 的 EPS 配对与方向判定（§5.2）。

    输入必须是**主预测期**表（每篇研报一行）：协议 §5.1 规定同一篇研报不得同时用
    FY0 与 FY1 投两票，因此非主预测期的 FY 不形成 revision 事件。
    """
    columns = [
        "ts_code",
        "analyst_team",
        "report_date",
        "absolute_fy",
        "eps_new",
        "eps_old",
        "eps_delta",
        "eps_rel",
        "direction",
        "gap_days",
    ]
    if primary.empty:
        return pd.DataFrame(columns=columns)
    work = primary.rename(columns={"eps": "eps"}).sort_values(
        ["ts_code", "analyst_team", "absolute_fy", "report_date"], kind="mergesort"
    ).copy()
    work["_ord"] = pd.to_datetime(work["report_date"], format="%Y%m%d").map(
        pd.Timestamp.toordinal
    )
    grouped = work.groupby(["ts_code", "analyst_team", "absolute_fy"], sort=False)
    work["_ord_prev"] = grouped["_ord"].shift(1)
    work["eps_old"] = grouped["eps"].shift(1)
    work["gap_days"] = work["_ord"] - work["_ord_prev"]
    lookback = int(spec.REPORT["revision_lookback_days"])
    work = work[work["gap_days"].notna() & (work["gap_days"] <= lookback)].copy()
    work = work.rename(columns={"eps": "eps_new"})
    work["eps_delta"] = work["eps_new"] - work["eps_old"]

    abs_tol = float(spec.REPORT["eps_abs_tolerance_yuan"])
    rel_tol = float(spec.REPORT["eps_relative_tolerance"])
    floor = float(spec.REPORT["eps_denominator_floor_yuan"])
    clip = float(spec.REPORT["eps_relative_clip"])
    tolerance = np.maximum(abs_tol, rel_tol * work["eps_old"].abs())
    work["direction"] = np.where(
        work["eps_delta"] > tolerance,
        "up",
        np.where(work["eps_delta"] < -tolerance, "down", "flat"),
    )
    work["eps_rel"] = np.clip(
        work["eps_delta"] / np.maximum(work["eps_old"].abs(), floor), -clip, clip
    )
    work["gap_days"] = work["gap_days"].astype(int)
    return work[columns].reset_index(drop=True)


def target_revision_events(level: pd.DataFrame) -> pd.DataFrame:
    """目标价配对（±1% 容差，§5.3）。"""
    columns = ["ts_code", "analyst_team", "report_date", "target_rel", "direction", "gap_days"]
    paired = _pair_series(level, "target", lookback_days=int(spec.REPORT["revision_lookback_days"]))
    if paired.empty:
        return pd.DataFrame(columns=columns)
    paired = paired[paired["target_old"] > 0].copy()
    paired["target_rel"] = paired["target_new"] / paired["target_old"] - 1.0
    tolerance = float(spec.REPORT["target_tolerance"])
    paired["direction"] = np.where(
        paired["target_rel"] > tolerance,
        "up",
        np.where(paired["target_rel"] < -tolerance, "down", "flat"),
    )
    return paired[columns].reset_index(drop=True)


def rating_revision_events(level: pd.DataFrame) -> pd.DataFrame:
    """评级分差配对（>0 上调、<0 下调、=0 持平，§5.3）。"""
    columns = ["ts_code", "analyst_team", "report_date", "rating_delta", "direction", "gap_days"]
    paired = _pair_series(level, "rating", lookback_days=int(spec.REPORT["revision_lookback_days"]))
    if paired.empty:
        return pd.DataFrame(columns=columns)
    paired = paired.copy()
    paired["rating_delta"] = paired["rating_new"] - paired["rating_old"]
    paired["direction"] = np.where(
        paired["rating_delta"] > 0,
        "up",
        np.where(paired["rating_delta"] < 0, "down", "flat"),
    )
    return paired[columns].reset_index(drop=True)


def build_events(
    raw: pd.DataFrame,
    trading_days: Sequence[str] | None = None,
    lag_trading_days: int = 0,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """从 report_rc 原始表构建全部事件表与审计。"""
    emit = _emit(log)
    frame, audit = normalize(raw)
    emit(f"report_rc 规范化：{audit['rows_before']} -> {audit['rows_after']}")
    prepared = prepare_fields(frame, trading_days, lag_trading_days)
    audit["visibility_rule"] = "report_date -> 首个可见交易日（可加 lag_trading_days）"
    audit["lag_trading_days"] = int(lag_trading_days)

    eps = eps_table(prepared)
    level = report_level_table(prepared)
    activity = activity_table(prepared)

    # EPS 主预测期（每篇研报唯一）：FY1 有效优先，否则 FY0；非主预测期不投票
    primary = eps.copy()
    if len(primary):
        year = primary["report_date"].str[:4].astype(int)
        primary["prefer"] = np.where(primary["absolute_fy"] == year + 1, 0, 1)
        primary = primary.sort_values(
            ["ts_code", "analyst_team", "report_date", "prefer"], kind="mergesort"
        ).drop_duplicates(subset=["ts_code", "analyst_team", "report_date"], keep="first")
    primary = primary[["ts_code", "analyst_team", "report_date", "absolute_fy", "eps"]]

    eps_events = eps_revision_events(primary)
    target_events = target_revision_events(level)
    rating_events = rating_revision_events(level)
    emit(
        f"事件表：eps_pair={len(eps)} primary={len(primary)} eps_rev={len(eps_events)} "
        f"target_rev={len(target_events)} rating_rev={len(rating_events)}"
    )

    audit.update(
        {
            "analyst_team_rule": "org_name|author_name（两者均非空才投票）",
            "same_day_aggregation": "target/rating 同日多研报取中位数后每分析师一票",
            "units": "eps=元/股；target=元/股；rating=factors/consensus.rating_to_score",
            "eps_primary_fy_rule": "FY1 有效优先，否则 FY0；两者皆无则无 revision 事件",
            "revision_lookback_days": int(spec.REPORT["revision_lookback_days"]),
        }
    )
    return {
        "eps": eps,
        "primary": primary,
        "level": level,
        "activity": activity,
        "eps_events": eps_events,
        "target_events": target_events,
        "rating_events": rating_events,
        "audit": audit,
        "prepared_rows": int(len(prepared)),
    }


def event_fingerprint(events: dict[str, pd.DataFrame]) -> dict[str, str]:
    """事件表内容指纹（分表登记，便于跨阶段做漂移检测）。"""
    out: dict[str, str] = {}
    for name in (
        "eps",
        "primary",
        "level",
        "activity",
        "eps_events",
        "target_events",
        "rating_events",
    ):
        frame = events[name]
        if frame is None or len(frame) == 0:
            out[name] = sha256_bytes(b"")
            continue
        ordered = frame.reindex(sorted(frame.columns), axis=1).reset_index(drop=True)
        out[name] = sha256_bytes(ordered.to_csv(index=False).encode("utf-8"))
    return out


def field_availability(report_rc: pd.DataFrame) -> dict[str, Any]:
    """字段可用性台账（R0 单位与身份核验的输入）。"""
    frame = prepare_fields(normalize(report_rc)[0])
    return {
        "rows": int(len(frame)),
        "distinct_codes": int(frame["ts_code"].nunique()),
        "report_date_min": str(frame["report_date"].min()),
        "report_date_max": str(frame["report_date"].max()),
        "named_analyst_ratio": float(frame["named_analyst"].mean()),
        "eps_available_ratio": float(frame["eps_value"].notna().mean()),
        "eps_positive_ratio": float((frame["eps_value"] > 0).mean()),
        "target_available_ratio": float(frame["target_value"].notna().mean()),
        "rating_available_ratio": float(frame["rating_value"].notna().mean()),
        "quarter_parsed_ratio": float(frame["fy_year"].notna().mean()),
        "unit_note": "EPS 元/股；目标价 元/股；不作单位猜测或换算",
    }


def analyst_coverage_matrix(
    activity: pd.DataFrame,
    primary: pd.DataFrame,
    dates: Sequence[str],
    codes: Sequence[str],
    *,
    window_days: int | None = None,
) -> dict[str, np.ndarray]:
    """逐 (T, 股票) 的 90 日分析师活跃数与报告年龄。

    Returns:
        `n_active`（有有效 FY0/FY1 主预测的分析师数）、
        `report_age`（T 减最新研报日的自然日数，含无作者报告）、
        `has_report`（窗口内是否有任何具名分析师的有效主预测）。
    """
    window = int(window_days or spec.REPORT["diffusion_window_days"])
    day_ord = pd.to_datetime(pd.Series(list(dates)).astype(str), format="%Y%m%d").map(
        pd.Timestamp.toordinal
    ).to_numpy(dtype=np.int64)
    code_index = {str(code): j for j, code in enumerate(codes)}
    n_active = np.zeros((len(dates), len(codes)), dtype=np.int16)
    report_age = np.full((len(dates), len(codes)), np.nan, dtype=np.float32)

    primary_days = primary.drop_duplicates(subset=["ts_code", "analyst_team", "report_date"]).copy()
    primary_days["_ord"] = pd.to_datetime(primary_days["report_date"], format="%Y%m%d").map(
        pd.Timestamp.toordinal
    )
    for code, part in primary_days.groupby("ts_code", sort=False):
        j = code_index.get(str(code))
        if j is None:
            continue
        active = np.zeros(len(dates), dtype=np.int16)
        for _, team_part in part.groupby("analyst_team", sort=False):
            ords = np.sort(team_part["_ord"].to_numpy(dtype=np.int64))
            left = np.searchsorted(ords, day_ord - window, side="right")
            right = np.searchsorted(ords, day_ord, side="right")
            active += (right > left).astype(np.int16)
        n_active[:, j] = active

    activity_days = activity.drop_duplicates(subset=["ts_code", "report_date"]).copy()
    activity_days["_ord"] = pd.to_datetime(activity_days["report_date"], format="%Y%m%d").map(
        pd.Timestamp.toordinal
    )
    for code, part in activity_days.groupby("ts_code", sort=False):
        j = code_index.get(str(code))
        if j is None:
            continue
        ords = np.sort(part["_ord"].to_numpy(dtype=np.int64))
        position = np.searchsorted(ords, day_ord, side="right") - 1
        ages = np.full(len(dates), np.nan, dtype=np.float32)
        valid = position >= 0
        ages[valid] = (day_ord[valid] - ords[position[valid]]).astype(np.float32)
        report_age[:, j] = ages

    return {"n_active": n_active, "report_age": report_age, "has_report": n_active > 0}
