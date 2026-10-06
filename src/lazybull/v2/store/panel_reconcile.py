# -*- coding: utf-8 -*-
"""v2 P1 单元 4：panel 对账门（严格门 = panel vs 冻结参照；归因门 = panel vs cs_train）。

读取口径：panel 侧经 `PanelDataStore.load_features`（热/冷透明）+ `load_labels`，
**按月查询**（月内逐日切片；归档月文件避免逐日重读）；参照侧直接读日分区 parquet。
列交集 = 375 panel 特征列 + 6 标签列 = 381（键列以 index 承载，trade_date/ts_code
键列噪声在比对源头剔除、不进 cols_only 登记）。列集差异的实证来源（终版验收产物
v2_p1_panel_reconcile_final_20261006.json / v2_p1_panel_vs_cstrain_final_20261006.json，
均在 data/reports/）：
CYQ5 物化列（cost_concentration 等 5 列，参照侧 cyq_perf 数据态差异，
20120104~20171229）与 2012 冷启动 schema 演进（≤20121231）——分别登记
``cols_only_panel`` / ``cols_only_ref_days`` 并纳入 verdict 结构合法性判定（fail-closed）。

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
    _KEY_COLUMNS,
    _column_samples,
    _compare_nonnumeric_column,
    _compare_numeric_column,
    _count_nan_mismatch,
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
    "mkt_",
    "macd_",
    "vol_regime",
    "turnover_percentile",
    "skewness",
    "kurtosis",
    "downside_",
    "var_95",
    "cvar_95",
    "zscore_macd",
    "volatility_",
    "parkinson",
    "garch_",
    "vol_of_vol",
    "vol_ratio_5_20",
    "amihud",
    "atr_",
    "bb_",
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

# ========== 对账门收紧常量（v2 P1 三轮评审整改 R3-04 / P1-1 / P1-2 / P1-3 / R2-1）==========
# 实证来源：终版验收产物 data/reports/v2_p1_panel_reconcile_final_20261006.json（strict）
# 与 data/reports/v2_p1_panel_vs_cstrain_final_20261006.json（attrib）。
# 哲学 = fail-closed（对照 D-13 先例）：越界即 FAIL，须调查并重新登记，不留静默放行口。

#: 结构合法性：CYQ5 物化列白名单（参照侧 cyq_perf 数据态差异；实证恰这 5 列 ×
#: 20120104~20171229 落在 cols_only_panel）
CYQ5_COLUMNS = frozenset(
    {
        "cost_concentration",
        "weight_avg_bias",
        "winner_rate",
        "winner_rate_chg_5",
        "winner_rate_chg_20",
    }
)
#: CYQ5 登记差异末日上界（实证 last=20171229）
CYQ5_LAST_DAY = "20171229"
#: 2012 冷启动 schema 演进边界：cols_only_panel / cols_only_ref_days 登记日 ≤ 此 ⇒ 合法
SCHEMA_EVOLUTION_END = "20121231"

#: D-04③/④ 登记差异窗（实证 first=20161228；上界 20260702 = 终版参照构建数据态水位）
D04_3_4_RANGE = ("20161201", "20260702")
#: D-04① 差异日下界（水位类差异只会出现在接近参照构建数据态的近期窗口；
#: lhb/block_discount 终版零差异——本规则对存量无约束、对未来漂移 fail-closed）
D04_1_DIFF_START = "20260101"
#: D-11 锚定分支超幅上限（实证窗内最大 kurtosis_20=2.326；窗内任意密度放行、超幅拒）
D11_ANCHOR_MAX_ABS = 5.0

#: D-14 专项三上限（天数 / 累计 over / max|Δ|，越任一 ⇒ 越界；括注终版产物实证值）
_D14_SKEW_MAX_LATE_DAYS = 250  # 实证 194
_D14_SKEW_MAX_LATE_OVER = 40000  # 实证总 33355
_D14_SKEW_MAX_ABS = 5.0  # 实证 3.121
_D14_KURT_MAX_LATE_DAYS = 750  # 实证 658
_D14_KURT_MAX_LATE_OVER = 40000  # 实证总 34492
_D14_KURT_MAX_ABS = 40.0  # 实证 35.13（2015 极端行情簇）
_D14_ZSCORE_MACD_MAX_DAYS = 500  # 实证 410 / 413
_D14_ZSCORE_MACD_MAX_OVER = 600000  # 实证 348151 / 471326
_D14_ZSCORE_MACD_MAX_ABS = 20.0  # 实证 14.28 / 15.29
_D14_DOWNSIDE_CORR_MAX_DAYS = 200  # 实证 147
_D14_DOWNSIDE_CORR_MAX_OVER = 300000  # 实证 255697
_D14_DOWNSIDE_CORR_MAX_ABS = 1.5  # 实证 1.116

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
            "over_nan_rows": _count_nan_mismatch(lcol, rcol),
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
        # 键列噪声在源头剔除（panel 侧键列以 index 承载，ref 侧是列 ⇒ 恒差 trade_date）
        "cols_only_ref": sorted(set(ref.columns) - set(panel_day.columns) - set(_KEY_COLUMNS)),
        "cols_only_panel": sorted(set(panel_day.columns) - set(ref.columns) - set(_KEY_COLUMNS)),
        "common_cols": common_cols,
        # 当日每个 common 列的可比行数（含零差异列；R3-04 全窗口分母的数据来源）
        "col_rows": {col: len(common_codes) for col in common_cols},
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
    col: str, diff_days: list[tuple[str, int, int, int]], max_abs_diff: float
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

    三上限（R3 评审 P1-3 整改；常量见模块顶部 _D14_*，注释括注实证值）：
    差异天数上限、累计 over 上限（skew/kurt 只计 late 日求和）、max|Δ| 上限，
    越任一 ⇒ 越界。
    """
    if col in ("skewness_20", "kurtosis_20"):
        late = [(d, o, t) for d, o, t, _ in diff_days if d > _L1_LONG_TAIL_END]
        if col == "skewness_20":
            caps = (_D14_SKEW_MAX_LATE_DAYS, _D14_SKEW_MAX_LATE_OVER, _D14_SKEW_MAX_ABS)
        else:
            caps = (_D14_KURT_MAX_LATE_DAYS, _D14_KURT_MAX_LATE_OVER, _D14_KURT_MAX_ABS)
        ok = (
            all(o <= 25 and (t == 0 or o / t <= 0.015) for _, o, t in late)
            and len(late) <= caps[0]
            and sum(o for _, o, _ in late) <= caps[1]
            and max_abs_diff <= caps[2]
        )
        return (
            "L1 + D-05 实现对偶数值尾（rolling vs 中心矩实现的边界股翻转，"
            "逐日 ≤25 股且 ≤1.5%，聚 2015 停牌簇，受天数/累计/超幅三上限约束；现役模型不含）",
            not ok,
        )
    if col in ("zscore_macd_hist", "zscore_macd_hist_sz"):
        ok = (
            all(
                d <= "20130930"
                or any(lo <= d <= hi for lo, hi in _D11_WINDOWS)
                or (o <= 5 and t > 0 and o / t <= _SCATTER_DAY_SHARE)
                for d, o, t, _ in diff_days
            )
            and len(diff_days) <= _D14_ZSCORE_MACD_MAX_DAYS
            and sum(o for _, o, *_ in diff_days) <= _D14_ZSCORE_MACD_MAX_OVER
            and max_abs_diff <= _D14_ZSCORE_MACD_MAX_ABS
        )
        return (
            "L1 macd-EMA 冷启动长尾（至 2013-09）+ D-11 分块锚定（2022-05 复牌簇）"
            "+ 稀疏孤立单点（受天数/累计/超幅三上限约束）",
            not ok,
        )
    if col == "downside_corr_20":
        lo2, hi2 = D04_2_RANGE
        ok = (
            all(
                d <= "20131031"
                or lo2 <= d <= hi2
                or (o <= 5 and t > 0 and o / t <= _SCATTER_DAY_SHARE)
                for d, o, t, _ in diff_days
            )
            and len(diff_days) <= _D14_DOWNSIDE_CORR_MAX_DAYS
            and sum(o for _, o, *_ in diff_days) <= _D14_DOWNSIDE_CORR_MAX_OVER
            and max_abs_diff <= _D14_DOWNSIDE_CORR_MAX_ABS
        )
        return (
            "L1 + 2013-09/10 cs_train 构建环境血缘块（pandas 2.x vs 3.0.5 滚动聚合"
            "行为差推断）+ D-04② + 稀疏孤立单点（受天数/累计/超幅三上限约束）",
            not ok,
        )
    return None


def _d04_compliant(attr: str, diff_days: list[tuple[str, int, int, int]]) -> bool:
    """D-04①/③/④ 放行条件校验（严格门/归因门同构；D-04② 在调用处先行处理）。

    - ③/④ 基础列（dividend_payout_ratio/cf_nm/ocf_to_profit/fund_*、margin 族）：
      ① 纯 NaN 型差异（每日 over_nan_rows == over_rows；终版实证这些列 max_abs=0）
      ② 差异日 ⊆ 登记窗 D04_3_4_RANGE（实证 first=20161228）；
    - ①（days_to_unlock/unlock_ratio/unlock_risk_flag 实际被 D-12 豁免先拦截；
      block_discount_*/lhb_* 前缀）：差异日全部 ≥ D04_1_DIFF_START（水位类差异
      只会出现在接近参照构建数据态的近期窗口）；
    - 派生连锁列（zscore_/neu_）：同族日期规则，但免除纯 NaN 要求（实证
      zscore_dividend_payout_ratio_sz max_abs=3.4）；跨列一致性在 _merge_and_judge
      合并后第二阶段检查；
    - 未知 D-04 子类 ⇒ False（fail-closed）。
    """
    derived = "派生连锁" in attr
    if attr.startswith("D-04①"):
        return all(d >= D04_1_DIFF_START for d, *_ in diff_days)
    if attr.startswith(("D-04③", "D-04④")):
        lo, hi = D04_3_4_RANGE
        if not all(lo <= d <= hi for d, *_ in diff_days):
            return False
        if derived:
            return True
        return all(o == nan for _, o, _, nan in diff_days)
    return False


def _classify_attrib_column(
    col: str, diff_days: list[tuple[str, int, int, int]], max_abs_diff: float
) -> tuple[str, bool]:
    """归因门类判定：返回 (类别, 是否越界)。

    diff_days: [(date, over_rows_in_day, day_total_rows, over_nan_rows)]。判定规则：
    - D-12 豁免族（2026-10-06 裁决）：panel 全量 PIT vs cs_train 旧链区间截断，
      差异全归因登记，不设日期边界；
    - D-14 专项登记（skew/kurt 数值尾、zscore_macd EMA 长尾、downside_corr
      2013-09/10 血缘块）：按定界规则归因，受天数/累计/超幅三上限约束，越界 ⇒ 越界；
    - D-04①/③/④（水位/修订类）：不再无条件放行——按 _d04_compliant 校验
      （③/④ 基础列纯 NaN 型 + 登记窗；① 差异日下界），越界 ⇒ 越界；
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
    special = _classify_attrib_special(col, diff_days, max_abs_diff)
    if special is not None:
        return special
    attr = attribute_column(col)
    if attr.startswith("D-04②"):
        lo, hi = D04_2_RANGE
        ok = all(lo <= d <= hi for d, *_ in diff_days)
        return (attr, not ok)
    if attr.startswith("D-04"):
        return (attr, not _d04_compliant(attr, diff_days))
    # L1 边界按列族：长窗/EMA/风控族自 2012 冷启动的长尾延伸至 ~2013 年中（250 日窗口
    # 与 EMA 链成熟所需时间），其余族 2012 年底收口
    l1_end = _L1_LONG_TAIL_END if col.startswith(_LONG_WINDOW_PREFIXES) else _L1_TAIL_END
    dates = [d for d, *_ in diff_days]
    if all(d <= l1_end for d in dates):
        return ("L1 2012 构建血缘（冷启动零预热/schema 演进）", False)
    late = [(d, o, t) for d, o, t, _ in diff_days if d > l1_end]
    total_over = sum(o for _, o, *_ in diff_days)
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
            plen = len(prefix)  # 切片界先赋值（black/flake8 E203 兼容）
            parent = col[plen:]
            if parent.endswith("_sz"):
                parent = parent[: -len("_sz")]
            return parent.startswith(_LONG_WINDOW_PREFIXES)
    return False


def _derived_parent(col: str) -> str | None:
    """zscore_X / zscore_X_sz / neu_X 派生列的母列名（非派生列返回 None）。"""
    for prefix in ("zscore_", "neu_"):
        if col.startswith(prefix):
            plen = len(prefix)  # 切片界先赋值（black/flake8 E203 兼容）
            parent = col[plen:]
            if parent.endswith("_sz"):
                parent = parent[: -len("_sz")]
            return parent
    return None


def _classify_strict_column(
    col: str, diff_days: list[tuple[str, int, int, int]], max_abs_diff: float
) -> tuple[str, bool]:
    """严格门类判定：返回 (类别, 是否未归因)。

    严格门（panel vs 冻结参照）的登记差异类（冻结文档 §8；出口 = 残差 100% 归因）：
    - D-12 豁免族（2026-10-06 裁决）：panel 全量 PIT vs 参照旧链截断，差异全归因登记；
    - D-04①/③/④（数据态水位/修订）：不再无条件放行——按 _d04_compliant 校验
      （③/④ 基础列纯 NaN 型 + 登记窗 D04_3_4_RANGE；① 差异日 ≥ D04_1_DIFF_START；
      派生连锁单列免纯 NaN 要求、跨列一致性在合并后检查），越界 ⇒ 未归因；
    - D-04②（2024-01 丢行事件）：差异日须命中登记区间；
    - L1（2012 冷启动血缘）：差异日 ≤ L1 长尾末（长窗族 2013-06）；
    - D-11 分块锚定：差异日 ⊆ 锚定影响窗（20170103~20170831 / 20220104~20220930）
      且（列为长记忆易感族且 max|Δ| ≤ D11_ANCHOR_MAX_ABS——**窗内任意密度放行、
      超幅 >5.0 拒**（实证窗内最大 kurtosis_20=2.326；zscore_macd_hist_sz 窗内
      23857/51327≈46% 密集差异合法），或全日稀疏微量伴随单点——复牌股锚定事件
      的伴随效应）；
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
        ok = all(lo <= d <= hi for d, *_ in diff_days)
        return (attr, not ok)
    if attr.startswith("D-04"):
        return (attr, not _d04_compliant(attr, diff_days))
    dates = [d for d, *_ in diff_days]
    total_over = sum(o for _, o, *_ in diff_days)
    all_sparse = all(o / t <= _SCATTER_DAY_SHARE for _, o, t, _ in diff_days if t > 0)
    in_d11 = all(any(lo <= d <= hi for lo, hi in _D11_WINDOWS) for d in dates)
    if in_d11 and _is_anchor_sensitive(col) and max_abs_diff <= D11_ANCHOR_MAX_ABS:
        return (_D11_TEXT, False)
    if in_d11 and all_sparse and total_over <= _SCATTER_PURE_MAX_ROWS:
        return (_D11_TEXT + "（伴随稀疏单点）", False)
    l1_end = _L1_LONG_TAIL_END if col.startswith(_LONG_WINDOW_PREFIXES) else _L1_TAIL_END
    if all(d <= l1_end for d in dates):
        return ("L1 2012 构建血缘（冷启动零预热/schema 演进）", False)
    late = [(d, o, t) for d, o, t, _ in diff_days if d > l1_end]
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
    diff_dates_by_col: dict[str, list[tuple[str, int, int, int]]] = {}
    col_total_rows: dict[str, int] = {}
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
        # 分母全窗口累计（含零差异日；列在某日不可比则当日不计入——R3-04）
        for col, n_rows in day.get("col_rows", {}).items():
            col_total_rows[col] = col_total_rows.get(col, 0) + n_rows
        for col, stat in day["columns"].items():
            agg = merged.setdefault(
                col, {"total_rows": 0, "over_rows": 0, "max_abs_diff": 0.0, "samples": []}
            )
            agg["over_rows"] += stat["over_rows"]
            agg["max_abs_diff"] = max(agg["max_abs_diff"], stat["max_abs_diff"])
            diff_dates_by_col.setdefault(col, []).append(
                (day["date"], stat["over_rows"], stat["rows"], stat["over_nan_rows"])
            )
            if len(agg["samples"]) < _SAMPLE_CAP:
                for sample in stat["samples"]:
                    sample["date"] = day["date"]
                agg["samples"] = (agg["samples"] + stat["samples"])[:_SAMPLE_CAP]
    for col, agg in merged.items():
        # over_share 分母 = 全窗口该列实际可比日的可比行数合计（不是有差异日的行数）
        agg["total_rows"] = col_total_rows.get(col, 0)
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
            klass, violated = _classify_attrib_column(col, day_list, agg["max_abs_diff"])
            agg["attribution"] = klass
            agg["boundary_violation"] = violated
    _check_derived_chain(merged, diff_dates_by_col, mode)
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


def _check_derived_chain(
    merged: dict[str, Any],
    diff_dates_by_col: dict[str, list[tuple[str, int, int, int]]],
    mode: str,
) -> None:
    """派生连锁跨列一致性（合并后第二阶段；D-04 派生列在单列阶段只过同族日期规则）。

    派生列（归因文本含「派生连锁」）的母列必须也在差异列集合中、且派生列差异日
    ⊆ 母列差异日（实证 zscore_cf_nm 840 天 == cf_nm 840 天）；不满足 ⇒ 改判
    未归因（strict）/ 越界（attrib）。
    """
    flag = "unattributed" if mode == "strict" else "boundary_violation"
    for col, agg in merged.items():
        if "派生连锁" not in agg.get("attribution", "") or agg.get(flag):
            continue
        parent = _derived_parent(col)
        parent_days = {d for d, *_ in diff_dates_by_col.get(parent, ())}
        own_days = {d for d, *_ in diff_dates_by_col[col]}
        if not parent_days or not own_days <= parent_days:
            agg[flag] = True
            agg["attribution"] += "（跨列一致性不满足：母列缺失或差异日越出母列，须调查）"


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


def _structural_offenders(merged: dict[str, Any]) -> list[str]:
    """结构合法性检查（fail-closed，对照 D-13 先例：越界即 FAIL，须调查并重新登记）。

    实证来源：终版验收产物 v2_p1_panel_reconcile_final_20261006.json（strict）/
    v2_p1_panel_vs_cstrain_final_20261006.json（attrib）在该规则下仍 PASS
    （missing_ref=0；strict cols_only_panel=CYQ5；attrib 另有 47 列仅在 20120104 单日）。
    - missing_ref_days 非空 ⇒ 违例（参照缺日须调查）；
    - cols_only_panel 逐列：（列 ∈ CYQ5 且 last ≤ CYQ5_LAST_DAY）或
      last ≤ SCHEMA_EVOLUTION_END（2012 冷启动 schema 演进）⇒ 合法，其余违例；
    - cols_only_ref_days（键列噪声已在比对源头剔除）：全部登记日
      ≤ SCHEMA_EVOLUTION_END ⇒ 合法，否则违例。
    """
    offenders: list[str] = []
    if merged["missing_days"]:
        offenders.append(
            f"missing_ref_days×{len(merged['missing_days'])}（首 {merged['missing_days'][0]}）"
        )
    for col, info in sorted(merged["cols_only_panel"].items()):
        legal = (col in CYQ5_COLUMNS and info["last"] <= CYQ5_LAST_DAY) or (
            info["last"] <= SCHEMA_EVOLUTION_END
        )
        if not legal:
            offenders.append(
                f"cols_only_panel:{col}（{info['first']}~{info['last']}，越出白名单边界）"
            )
    late_ref = [
        entry["date"]
        for entry in merged["cols_only_ref_days"]
        if entry["date"] > SCHEMA_EVOLUTION_END
    ]
    if late_ref:
        offenders.append(f"cols_only_ref_days:越界日 {late_ref[0]}~{late_ref[-1]}")
    return offenders


def _build_verdict(merged: dict[str, Any], diff_cols: dict[str, Any], mode: str) -> dict[str, Any]:
    """判定：严格门 = 差异 100% 归因登记类（§7 出口）；归因门 = 无越界 + 无第六类 +
    行集零差异；两种 mode 都追加结构合法性判定（missing_ref / cols_only 白名单）。"""
    structural = _structural_offenders(merged)
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
            "pass": not unattributed and not merged["row_diff_days"] and not structural,
            "rule": (
                "出口 = 残差 100% 归因（冻结 §7）：差异列须全部落入登记类"
                "（L1/D-04/D-11/D-12 豁免）且行集零差异、结构合法（missing_ref=0、"
                "cols_only 落入 CYQ5/2012 边界）；未归因或结构越界即 fail-closed"
            ),
            "offenders": sorted(unattributed) + structural,
            "unattributed_columns": unattributed,
            "attributed_columns": attributed,
            "structural_offenders": structural,
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
        "pass": not violations and not merged["row_diff_days"] and not structural,
        "rule": (
            "差异须 100% 落入五类预期清单（L1/L2/D-04/D-05=0/2012 schema 缺失按列交集）"
            "且结构合法（missing_ref=0、cols_only 落入 CYQ5/2012 边界）；越界即 fail-closed"
        ),
        "boundary_violations": violations,
        "structural_offenders": structural,
    }
