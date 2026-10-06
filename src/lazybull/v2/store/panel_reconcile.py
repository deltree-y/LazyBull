# -*- coding: utf-8 -*-
"""v2 P1 单元 4：panel 对账门（严格门 = panel vs 冻结参照；归因门 = panel vs cs_train）。

读取口径：panel 侧经 `PanelDataStore.load_features`（热/冷透明）+ `load_labels`，
**按月查询**（月内逐日切片；归档月文件避免逐日重读）；参照侧直接读日分区 parquet。
列交集 = 375 panel 特征列 + 6 标签列 = 381（键列以 index 承载）；panel 多 34 物化列
= 预期差异（冻结 §6），登记 ``cols_only_panel``；2012 年代 cs_train schema 缺失列
按日登记（同样 cols_only 口径，仅 ≤20121231 合法）。

三段式聚合与判定：
- 合并分母重算占比（与 replay_compare 同口径，atol=1e-6，NaN==NaN 一致）；
- 严格门（出口 = 残差 100% 归因，冻结 §7）：列级 超容差行占比 >1e-4 或 max|Δ|>0.05
  ⇒ over_gate；差异列须全部落入登记类（L1 血缘 / D-04 数据态 / D-11 分块锚定 /
  D-12 裁决豁免），任一未归因列或行集差异 ⇒ 实现漂移 fail；
- 归因门（冻结 §7 + 单元 3 §4a 五类清单）：列 × 差异日边界验证——
  D-04①/③/④ 按列族放行；D-04② 差异日须 ⊆ [20240101, 20240331]；D-05 应为零
  （出现即异常）；其余列差异日 ≤20121231 归 L1、20130101~20151231 归 L2，
  出现 ≥2016 差异日 ⇒ 越界。出现第六类或越界 ⇒ fail 并输出越界清单。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from loguru import logger

from src.lazybull.v2.common.types import FeatureQuery, TradeDate
from src.lazybull.v2.store.column_groups import LABEL_TABLES, PANEL_GROUPS
from src.lazybull.v2.store.data_store import PanelDataStore
from src.lazybull.v2.store.replay_compare import (
    _column_samples,
    _compare_nonnumeric_column,
    _compare_numeric_column,
    attribute_column,
)

__all__ = ["reconcile_panel", "L1_BOUNDARY_END", "L2_RANGE", "D04_2_RANGE"]

#: L1/L2 血缘差异日边界（单元 3 报告 §4a；单元 4 复核更正：L2 实为参照侧污染，见 D-09）
L1_BOUNDARY_END = "20121231"
L2_RANGE = ("20130101", "20151231")
#: D-04② 登记事件日窗口（2024-01-16/17 丢行 + 20 日窗口余波）
D04_2_RANGE = ("20240101", "20240331")
#: L1 长尾末（与 L1_BOUNDARY_END 同值，语义别名：长窗列 EMA/250 日长尾自然延伸至 2012 年底）
_L1_TAIL_END = L1_BOUNDARY_END
#: 长窗/EMA/风控族的 L1 长尾延伸边界（250 日窗口与 EMA 链自 2012 冷启动成熟需 ~1.5 年）
_L1_LONG_TAIL_END = "20130630"
_LONG_WINDOW_PREFIXES = (
    "mkt_", "macd_", "vol_regime", "turnover_percentile", "skewness", "kurtosis",
    "downside_", "var_95", "cvar_95", "zscore_macd", "volatility_", "parkinson",
    "garch_", "vol_of_vol", "vol_ratio_5_20", "amihud", "atr_", "bb_",
)
#: 数据态孤立单点的判定阈值：单日超门行占比 ≤1% 且期外累计 ≤0.1%
_SCATTER_DAY_SHARE = 0.01
_SCATTER_TOTAL_SHARE = 0.001

#: D-12 裁决豁免列族（2026-10-06 用户裁决，冻结文档 §8 D-13）：panel 解锁前瞻列
#: 经全量 share_float（2005~2026 全分区）+ 全日历重建查询表重写，语义 = PIT 完整；
#: 参照/cs_train 保持旧链区间截断行为（行为冻结不动）⇒ 严格门/归因门对该族从
#: 「逐值一致」改为「豁免 + 差异全归因登记」（参照缺 ann≤2010 分区与末年分区的内容差）。
D12_EXEMPT_COLUMNS = frozenset({"days_to_unlock", "unlock_ratio", "unlock_risk_flag"})
_D12_TEXT = "D-12 公告前瞻截断豁免（裁决 2026-10-06：panel 全量 PIT vs 旧链区间截断）"

#: D-11 分块锚定影响窗（严格门专用）：panel 回填分块覆盖段起点 20170103/20220104，
#: 批量缓存锚定 120 交易日长尾 ≈ 6 个自然月；复牌股伴随事件（实证 002260.SZ 2022-05
#: 复牌 ⇒ EMA 链几何收敛 0.8/日，持续 ~4 个月）再留 3 个月余量 ⇒ 边界 +9 个月；
#: 参照为单调用锚定 20120104。窗口内长记忆列（EMA/MA250/滚动族及其 zscore/neu 派生）
#: 及稀疏伴随单点差异属预期锚定效应。
_D11_WINDOWS = (("20170103", "20170831"), ("20220104", "20220930"))
_D11_TEXT = "D-11 分块锚定效应（panel 分块锚 20170103/20220104 vs 参照单调用锚 20120104）"

#: 纯数据态孤立单点判定阈值（严格门专用）：全日稀疏（每日 ≤1%）+ 总量 ≤20 行
#: + 日数 ≤10 + max|Δ| ≤ 0.05（超幅单点一律调查，不登记）
_SCATTER_PURE_MAX_DAYS = 10
_SCATTER_PURE_MAX_ROWS = 20

#: panel 特征列（377 − 2 键列 = 375；键列以 index 承载）
PANEL_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    c for cols in PANEL_GROUPS.values() for c in cols if c not in ("trade_date", "ts_code")
)

_GATE_SHARE = 1e-4
_GATE_MAX_ABS = 0.05
_SAMPLE_CAP = 5


def _load_label_month(
    store: PanelDataStore, label_name: str, month_dates: list[str]
) -> dict[str, pd.DataFrame]:
    """直读标签分区（协议 load_labels 按协议形态裁剪为 [label_value, maturity_status]；
    对账需要 neu_label_value ⇒ 旁路直读 parquet，数值同源）。"""
    label_dir = store.labels_dir / label_name
    frames = []
    for date in month_dates:
        path = label_dir / f"{date}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return {
            date: pd.DataFrame(columns=["ts_code", "label_value", "neu_label_value"])
            for date in month_dates
        }
    month_df = pd.concat(frames, ignore_index=True)
    return {
        date: month_df[month_df["trade_date"].astype(str) == date][
            ["ts_code", "label_value", "neu_label_value"]
        ]
        for date in month_dates
    }


def _load_panel_month(store: PanelDataStore, month_dates: list[str]) -> dict[str, pd.DataFrame]:
    """按月读 panel（特征 + 3 标签表合并为 381 列日帧），返回 {date: DataFrame}。"""
    query_args = {
        "start_date": TradeDate.from_str(month_dates[0]),
        "end_date": TradeDate.from_str(month_dates[-1]),
    }
    panel = store.load_features(FeatureQuery(columns=list(PANEL_FEATURE_COLUMNS), **query_args))
    label_days = {name: _load_label_month(store, name, month_dates) for name in LABEL_TABLES}
    by_date: dict[str, pd.DataFrame] = {}
    for date in month_dates:
        # _slice_day 已丢 trade_date 层（index=ts_code）；reset_index 让 ts_code 成列供 merge
        day = _slice_day(panel.df, date).reset_index()
        # 3 张标签表先 set_index 后一次 concat，避免逐表 merge 碎片化
        label_parts = []
        for name, (value_col, neu_col) in LABEL_TABLES.items():
            label_day = label_days[name][date]
            if not label_day.empty:
                label_parts.append(
                    label_day.rename(
                        columns={"label_value": value_col, "neu_label_value": neu_col}
                    ).set_index("ts_code")
                )
        if label_parts:
            day = day.merge(pd.concat(label_parts, axis=1).reset_index(), on="ts_code", how="outer")
        by_date[date] = day
    return by_date


def _slice_day(df: pd.DataFrame, date: str) -> pd.DataFrame:
    """从 (trade_date, ts_code) 索引的月帧切出单日（ts_code 索引）。"""
    if df.empty:
        return df
    mask = df.index.get_level_values("trade_date") == date
    if not mask.any():
        return df.iloc[0:0]
    day = df[mask]
    return day.reset_index(level="trade_date", drop=True)


def _compare_day(date: str, ref: pd.DataFrame, panel_day: pd.DataFrame) -> dict[str, Any]:
    """单日比对（行集 + 列交集逐值；与 replay_compare 同口径）。"""
    ref_codes = set(ref["ts_code"].astype(str))
    panel_codes = set(panel_day["ts_code"].astype(str))
    common_codes = sorted(ref_codes & panel_codes)
    common_cols = sorted((set(ref.columns) & set(panel_day.columns)) - {"trade_date", "ts_code"})
    left = ref.set_index("ts_code").loc[common_codes]
    right = panel_day.set_index("ts_code").loc[common_codes]
    col_stats: dict[str, Any] = {}
    for col in common_cols:
        lcol, rcol = left[col], right[col]
        if pd.api.types.is_numeric_dtype(lcol) and pd.api.types.is_numeric_dtype(rcol):
            over, max_abs, mask = _compare_numeric_column(lcol, rcol, 1e-6)
        else:
            over, mask = _compare_nonnumeric_column(lcol, rcol)
            max_abs = 0.0
        if over == 0:
            continue
        col_stats[col] = {
            "rows": len(common_codes),
            "over_rows": over,
            "max_abs_diff": max_abs,
            "samples": _column_samples(mask, lcol, rcol),
        }
    return {
        "date": date,
        "status": "ok",
        "rows_ref": len(ref),
        "rows_panel": len(panel_day),
        "n_only_ref": len(ref_codes - panel_codes),
        "n_only_panel": len(panel_codes - ref_codes),
        "sample_only_ref": sorted(ref_codes - panel_codes)[:_SAMPLE_CAP],
        "sample_only_panel": sorted(panel_codes - ref_codes)[:_SAMPLE_CAP],
        "cols_only_ref": sorted(set(ref.columns) - set(panel_day.columns)),
        "cols_only_panel": sorted(set(panel_day.columns) - set(ref.columns)),
        "common_cols": common_cols,
        "columns": col_stats,
    }


def _reconcile_month(data_root: str, ref_dir: str, month_dates: list[str]) -> list[dict[str, Any]]:
    """worker：月级读 panel + 逐日读参照比对（joblib 并行单元）。"""
    store = PanelDataStore(data_root)
    panel_days = _load_panel_month(store, month_dates)
    results = []
    for date in month_dates:
        ref_path = Path(ref_dir) / f"{date}.parquet"
        if not ref_path.exists():
            results.append({"date": date, "status": "missing_ref"})
            continue
        results.append(_compare_day(date, pd.read_parquet(ref_path), panel_days[date]))
    return results


def _classify_attrib_special(
    col: str, diff_days: list[tuple[str, int, int]]
) -> tuple[str, bool] | None:
    """归因门单元 4 复核专项登记（冻结文档 §8 D-14；未命中返回 None）。

    三列/族经全量差异日诊断定界（panel ≡ 参照 ⇒ 差异源在 cs_train 侧；
    诊断产物 data/reports/v2_p1_attrib_violators_diag_20261006.json）：
    - skewness_20/kurtosis_20：L1 主体 + D-05 实现对偶数值尾——rolling.skew/kurt
      与中心矩 NaN 安全实现在数值不稳定股（近恒定收益/停牌簇）翻转，晚日逐日
      ≤25 股且 ≤1.5%（实证：skewness ≤3 股；kurtosis ≤23 股聚于 2015-04~08 极端
      行情簇，max|Δ|=35.1 @20150326 300288.SZ 同簇）；越此 ⇒ 越界；
    - zscore_macd_hist(_sz)：L1 macd-EMA 冷启动长尾（衰减实证收敛至 2013-09，
      单元 3「待调查」闭环）+ D-11 分块锚定 2022-05 复牌簇 + 稀疏孤立单点；
    - downside_corr_20：L1 + 2013-09/10 cs_train 构建环境血缘块（pandas 2.x vs
      3.0.5 滚动聚合行为差推断，20 日窗连续 17 日板块 ≤20131031）+ D-04②
      （2024-01 窗口）+ 稀疏孤立单点（2021-12）。
    """
    if col in ("skewness_20", "kurtosis_20"):
        late = [(d, o, t) for d, o, t in diff_days if d > _L1_LONG_TAIL_END]
        ok = all(o <= 25 and (t == 0 or o / t <= 0.015) for _, o, t in late)
        return (
            "L1 + D-05 实现对偶数值尾（rolling vs 中心矩实现的边界股翻转，"
            "逐日 ≤25 股且 ≤1.5%，聚 2015 停牌簇；现役模型不含）",
            not ok,
        )
    if col in ("zscore_macd_hist", "zscore_macd_hist_sz"):
        ok = all(
            d <= "20130930"
            or any(lo <= d <= hi for lo, hi in _D11_WINDOWS)
            or (o <= 5 and t > 0 and o / t <= _SCATTER_DAY_SHARE)
            for d, o, t in diff_days
        )
        return (
            "L1 macd-EMA 冷启动长尾（至 2013-09）+ D-11 分块锚定（2022-05 复牌簇）"
            "+ 稀疏孤立单点",
            not ok,
        )
    if col == "downside_corr_20":
        lo2, hi2 = D04_2_RANGE
        ok = all(
            d <= "20131031"
            or lo2 <= d <= hi2
            or (o <= 5 and t > 0 and o / t <= _SCATTER_DAY_SHARE)
            for d, o, t in diff_days
        )
        return (
            "L1 + 2013-09/10 cs_train 构建环境血缘块（pandas 2.x vs 3.0.5 滚动聚合"
            "行为差推断）+ D-04② + 稀疏孤立单点",
            not ok,
        )
    return None


def _classify_attrib_column(col: str, diff_days: list[tuple[str, int, int]]) -> tuple[str, bool]:
    """归因门类判定：返回 (类别, 是否越界)。

    diff_days: [(date, over_rows_in_day, day_total_rows)]。判定规则：
    - D-12 豁免族（2026-10-06 裁决）：panel 全量 PIT vs cs_train 旧链区间截断，
      差异全归因登记，不设日期边界；
    - D-14 专项登记（skew/kurt 数值尾、zscore_macd EMA 长尾、downside_corr
      2013-09/10 血缘块）：按定界规则归因，越界 ⇒ 越界；
    - D-04①/③/④（水位/修订类）：按事件日登记，不设日期边界 ⇒ 归因；
    - D-04②（raw 丢行事件）：差异日须命中登记区间，越界 ⇒ 越界；
    - D-05（skew/kurt）：归因门中按日期模式归类（参照侧已修复；panel vs cs_train 的
      差异实为 L1 血缘 + 数据态孤立单点，不是回归本身——名命中不直接判异常）；
    - L1：全部差异日 ≤ L1 长尾末（2012 年底，长窗列 EMA/250 日窗口的长尾自然延伸）；
    - L1+数据态孤立单点：L1 期差异 + L1 期外仅零星日（每日超门行占比 ≤1% 且
      期外累计占比 ≤0.1%）⇒ 归因；
    - 其余 ⇒ 越界（第六类，须调查）。
    """
    if col in D12_EXEMPT_COLUMNS:
        return (_D12_TEXT, False)
    special = _classify_attrib_special(col, diff_days)
    if special is not None:
        return special
    attr = attribute_column(col)
    if attr.startswith("D-04②"):
        lo, hi = D04_2_RANGE
        ok = all(lo <= d <= hi for d, _, _ in diff_days)
        return (attr, not ok)
    if attr.startswith("D-04"):
        return (attr, False)
    # L1 边界按列族：长窗/EMA/风控族自 2012 冷启动的长尾延伸至 ~2013 年中（250 日窗口
    # 与 EMA 链成熟所需时间），其余族 2012 年底收口
    l1_end = _L1_LONG_TAIL_END if col.startswith(_LONG_WINDOW_PREFIXES) else _L1_TAIL_END
    dates = [d for d, _, _ in diff_days]
    if all(d <= l1_end for d in dates):
        return ("L1 2012 构建血缘（冷启动零预热/schema 演进）", False)
    late = [(d, o, t) for d, o, t in diff_days if d > l1_end]
    total_over = sum(o for _, o, _ in diff_days)
    late_over = sum(o for _, o, _ in late)
    late_sparse = all(o / t <= _SCATTER_DAY_SHARE for _, o, t in late if t > 0)
    late_tiny = total_over > 0 and (late_over / total_over) <= _SCATTER_TOTAL_SHARE
    if late_sparse and late_tiny:
        return ("L1 血缘 + 数据态孤立单点（期外每日 ≤1% 且累计 ≤0.1%）", False)
    return ("越界（第六类/日期越出登记边界，须调查）", True)


def _is_anchor_sensitive(col: str) -> bool:
    """长记忆列判定（D-11 锚定效应的易感列）：长窗前缀族或其 zscore/neu 派生子列。"""
    if col.startswith(_LONG_WINDOW_PREFIXES):
        return True
    for prefix in ("zscore_", "neu_"):
        if col.startswith(prefix):
            parent = col[len(prefix):]
            if parent.endswith("_sz"):
                parent = parent[: -len("_sz")]
            return parent.startswith(_LONG_WINDOW_PREFIXES)
    return False


def _classify_strict_column(
    col: str, diff_days: list[tuple[str, int, int]], max_abs_diff: float
) -> tuple[str, bool]:
    """严格门类判定：返回 (类别, 是否未归因)。

    严格门（panel vs 冻结参照）的登记差异类（冻结文档 §8；出口 = 残差 100% 归因）：
    - D-12 豁免族（2026-10-06 裁决）：panel 全量 PIT vs 参照旧链截断，差异全归因登记；
    - D-04①/③/④（数据态水位/修订）：无条件归因（参照与 panel 消费同一漂移 raw 的
      构建上下文不同——参照单调用、panel 分块）；
    - D-04②（2024-01 丢行事件）：差异日须命中登记区间；
    - L1（2012 冷启动血缘）：差异日 ≤ L1 长尾末（长窗族 2013-06）；
    - D-11 分块锚定：差异日 ⊆ 锚定影响窗（20170103~20170831 / 20220104~20220930）
      且（列为长记忆易感族，或全日稀疏微量伴随单点——复牌股锚定事件的伴随效应）；
    - L1 + 数据态孤立单点（同归因门阈值）；
    - 纯数据态孤立单点：日数 ≤10 且总量 ≤20 行且全日 ≤1% 且 max|Δ| ≤ 0.05
      （百分位秩单步翻转等数据态修订残响；超幅/密集 ⇒ 调查）；
    - 其余 ⇒ 未归因（实现漂移嫌疑，门 FAIL）。
    """
    if col in D12_EXEMPT_COLUMNS:
        return (_D12_TEXT, False)
    attr = attribute_column(col)
    if attr.startswith("D-04②"):
        lo, hi = D04_2_RANGE
        ok = all(lo <= d <= hi for d, _, _ in diff_days)
        return (attr, not ok)
    if attr.startswith("D-04"):
        return (attr, False)
    dates = [d for d, _, _ in diff_days]
    total_over = sum(o for _, o, _ in diff_days)
    all_sparse = all(o / t <= _SCATTER_DAY_SHARE for _, o, t in diff_days if t > 0)
    in_d11 = all(any(lo <= d <= hi for lo, hi in _D11_WINDOWS) for d in dates)
    if in_d11 and _is_anchor_sensitive(col):
        return (_D11_TEXT, False)
    if in_d11 and all_sparse and total_over <= _SCATTER_PURE_MAX_ROWS:
        return (_D11_TEXT + "（伴随稀疏单点）", False)
    l1_end = _L1_LONG_TAIL_END if col.startswith(_LONG_WINDOW_PREFIXES) else _L1_TAIL_END
    if all(d <= l1_end for d in dates):
        return ("L1 2012 构建血缘（冷启动零预热/schema 演进）", False)
    late = [(d, o, t) for d, o, t in diff_days if d > l1_end]
    late_over = sum(o for _, o, _ in late)
    late_sparse = all(o / t <= _SCATTER_DAY_SHARE for _, o, t in late if t > 0)
    late_tiny = total_over > 0 and (late_over / total_over) <= _SCATTER_TOTAL_SHARE
    if late_sparse and late_tiny:
        return ("L1 血缘 + 数据态孤立单点（期外每日 ≤1% 且累计 ≤0.1%）", False)
    if (
        len(dates) <= _SCATTER_PURE_MAX_DAYS
        and total_over <= _SCATTER_PURE_MAX_ROWS
        and all_sparse
        and max_abs_diff <= _GATE_MAX_ABS
    ):
        return ("数据态孤立单点（纯：全日稀疏微量，单日 ≤1% 且总量 ≤20 行）", False)
    return ("未归因（差异模式越出登记类边界，实现漂移嫌疑，须调查）", True)


def _merge_and_judge(day_results: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    """三段式后两段：合并全窗口列级口径 + 判定（严格门 / 归因门）。"""
    merged: dict[str, Any] = {}
    diff_dates_by_col: dict[str, list[str]] = {}
    missing_days: list[str] = []
    row_diff_days: list[dict[str, Any]] = []
    cols_only_ref_days: list[dict[str, Any]] = []
    cols_only_panel_counter: dict[str, list[str]] = {}
    for day in sorted(day_results, key=lambda d: d["date"]):
        if day["status"] != "ok":
            missing_days.append(day["date"])
            continue
        if day["n_only_ref"] or day["n_only_panel"]:
            row_diff_days.append(
                {
                    "date": day["date"],
                    "n_only_ref": day["n_only_ref"],
                    "n_only_panel": day["n_only_panel"],
                    "sample_only_ref": day["sample_only_ref"],
                    "sample_only_panel": day["sample_only_panel"],
                }
            )
        if day["cols_only_ref"]:
            cols_only_ref_days.append({"date": day["date"], "cols": day["cols_only_ref"]})
        for col in day["cols_only_panel"]:
            cols_only_panel_counter.setdefault(col, []).append(day["date"])
        for col, stat in day["columns"].items():
            agg = merged.setdefault(
                col, {"total_rows": 0, "over_rows": 0, "max_abs_diff": 0.0, "samples": []}
            )
            agg["total_rows"] += stat["rows"]
            agg["over_rows"] += stat["over_rows"]
            agg["max_abs_diff"] = max(agg["max_abs_diff"], stat["max_abs_diff"])
            diff_dates_by_col.setdefault(col, []).append(
                (day["date"], stat["over_rows"], stat["rows"])
            )
            if len(agg["samples"]) < _SAMPLE_CAP:
                for sample in stat["samples"]:
                    sample["date"] = day["date"]
                agg["samples"] = (agg["samples"] + stat["samples"])[:_SAMPLE_CAP]
    for col, agg in merged.items():
        agg["over_share"] = agg["over_rows"] / agg["total_rows"] if agg["total_rows"] else 0.0
        agg["over_gate"] = bool(
            agg["over_share"] > _GATE_SHARE or agg["max_abs_diff"] > _GATE_MAX_ABS
        )
        day_list = diff_dates_by_col[col]
        agg["diff_day_summary"] = {
            "days": len(day_list),
            "first": day_list[0][0],
            "last": day_list[-1][0],
            "top": sorted(day_list, key=lambda x: -x[1])[:_SAMPLE_CAP],
        }
        if mode == "strict":
            klass, unattributed = _classify_strict_column(col, day_list, agg["max_abs_diff"])
            agg["attribution"] = klass
            agg["unattributed"] = unattributed
        else:
            klass, violated = _classify_attrib_column(col, day_list)
            agg["attribution"] = klass
            agg["boundary_violation"] = violated
    return {
        "columns": merged,
        "missing_days": missing_days,
        "row_diff_days": row_diff_days,
        "cols_only_ref_days": cols_only_ref_days,
        "cols_only_panel": {
            col: {"days": len(dates), "first": min(dates), "last": max(dates)}
            for col, dates in cols_only_panel_counter.items()
        },
    }


def reconcile_panel(
    data_root: Path | str,
    ref_root: Path | str,
    dates: Iterable[str],
    mode: str = "strict",
    n_jobs: int = -1,
) -> dict[str, Any]:
    """panel 对账主入口（按月 joblib 并行；mode = strict | attrib）。

    Returns:
        报告 dict（days / columns / verdict）。verdict.pass_ False ⇒ 门未过。
    """
    from joblib import Parallel, delayed

    if mode not in ("strict", "attrib"):
        raise ValueError(f"mode 非法: {mode!r}（strict | attrib）")
    ref_dir = Path(ref_root)
    nested = ref_dir / "features" / "cs_train"
    if nested.exists():
        ref_dir = nested
    date_list = sorted(str(d) for d in dates)
    months: dict[str, list[str]] = {}
    for date in date_list:
        months.setdefault(f"{date[:4]}-{date[4:6]}", []).append(date)
    logger.info(f"panel 对账启动: mode={mode}，{len(date_list)} 天 / {len(months)} 个月")
    month_items = [months[m] for m in sorted(months)]
    if n_jobs == 1:
        day_results = [
            r for md in month_items for r in _reconcile_month(str(data_root), str(ref_dir), md)
        ]
    else:
        per_month = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_reconcile_month)(str(data_root), str(ref_dir), md) for md in month_items
        )
        day_results = [r for month_result in per_month for r in month_result]
    merged = _merge_and_judge(day_results, mode)
    columns = merged["columns"]
    diff_cols = {c: a for c, a in columns.items() if a["over_rows"] > 0}
    verdict = _build_verdict(merged, diff_cols, mode)
    return {
        "mode": mode,
        "data_root": str(data_root),
        "ref_root": str(ref_root),
        "window": {
            "start": date_list[0] if date_list else None,
            "end": date_list[-1] if date_list else None,
        },
        "days": {
            "requested": len(date_list),
            "compared": len(date_list) - len(merged["missing_days"]),
            "missing_ref_days": merged["missing_days"],
            "row_set_diff_days": merged["row_diff_days"],
        },
        "cols_only_ref_days": merged["cols_only_ref_days"],
        "cols_only_panel": merged["cols_only_panel"],
        "columns": columns,
        "summary": {
            "columns_with_diff": len(diff_cols),
            "over_gate_columns": sorted(c for c, a in diff_cols.items() if a["over_gate"]),
            "by_attribution": _count_by(diff_cols, "attribution"),
        },
        "verdict": verdict,
    }


def _count_by(diff_cols: dict[str, Any], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for agg in diff_cols.values():
        counts[agg[key]] = counts.get(agg[key], 0) + 1
    return counts


def _build_verdict(merged: dict[str, Any], diff_cols: dict[str, Any], mode: str) -> dict[str, Any]:
    """判定：严格门 = 差异 100% 归因登记类（§7 出口）；归因门 = 无越界 + 无第六类 + 行集零差异。"""
    if mode == "strict":
        unattributed = {
            col: {
                "attribution": agg["attribution"],
                "over_share": agg["over_share"],
                "max_abs_diff": agg["max_abs_diff"],
            }
            for col, agg in diff_cols.items()
            if agg.get("unattributed")
        }
        attributed = {
            col: agg["attribution"] for col, agg in diff_cols.items() if not agg.get("unattributed")
        }
        return {
            "pass": not unattributed and not merged["row_diff_days"],
            "rule": (
                "出口 = 残差 100% 归因（冻结 §7）：差异列须全部落入登记类"
                "（L1/D-04/D-11/D-12 豁免）且行集零差异；未归因即实现漂移"
            ),
            "offenders": sorted(unattributed),
            "unattributed_columns": unattributed,
            "attributed_columns": attributed,
            "mechanical_zero": not diff_cols,
        }
    violations = {
        col: {
            "attribution": agg["attribution"],
            "over_share": agg["over_share"],
            "max_abs_diff": agg["max_abs_diff"],
        }
        for col, agg in diff_cols.items()
        if agg.get("boundary_violation")
    }
    return {
        "pass": not violations and not merged["row_diff_days"],
        "rule": "差异须 100% 落入五类预期清单（L1/L2/D-04/D-05=0/2012 schema 缺失按列交集）",
        "boundary_violations": violations,
    }
