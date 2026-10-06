# -*- coding: utf-8 -*-
"""v2 P1 单元 5：列级可用起点剖面（只读；M1.5「分时段变列集训练」前置诊断）。

诊断问题（契约 §8.1-M1.5）：2012 前主要因子列覆盖率是被「0.6 缺失率门禁」拖死
（可工程解 ⇒ 立项变列集训练），还是 TuShare 源头无数据（硬约束 ⇒ 不立项）？

口径：
- 采样日 = 2005Q1~2011Q4 各季末交易日（28 个）+ 20120104 锚点；
- 截面（universe）= 当日 raw/daily 全部 ts_code；
- 日分区数据集覆盖率 = 当日分区命中 universe 的比例；
- 历史类数据集覆盖率 = universe 中存在 PIT 记录（ann_date ≤ T，pledge/fund 缺 ann_date
  回退 end_date）的比例（因子查询表前向填充语义——有记录即可用）;
- 衍生列（量价/风控/市场状态/标签）= daily 驱动，覆盖率 = 100%（250 日长窗列须
  额外满足 daily 历史 ≥ 窗口）；zscore/neu 随母列；
- 列 → 数据集映射 = 脚本内注册表（按因子族前缀/精确名，与 features/factors 代码同源）；
- 门禁仿真：列在某年覆盖率 ≥ 40% ⇒ 该年「0.6 门禁幸存」（训练入口
  max_feature_missing_ratio=0.6  ⇒ 缺失 ≤60% 即幸存，prepare.py:94）。

产物：data/reports/v2_p1_column_availability_20261006.json + 控制台摘要。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.lazybull.v2.store.column_groups import PANEL_GROUPS  # noqa: E402

RAW = ROOT / "data" / "raw"

#: 日分区数据集（覆盖率 = 当日分区命中 universe 比例）
_DAY_DATASETS = (
    "daily_basic",
    "moneyflow",
    "margin_detail",
    "moneyflow_hsgt",
    "cyq_perf",
    "stk_limit",
)

#: 历史类数据集（一次性加载 代码列 + PIT 日期列；覆盖率 = 有 PIT≤T 记录的比例）
#: pit = PIT 日期首选列（缺失按 end_date → trade_date → report_date 回退）；
#: code = 股票代码列（fund_portfolio 原始表 ts_code 为基金代码，股票代码在 symbol）。
_HISTORY_DATASETS: dict[str, dict[str, str]] = {
    "fina_indicator": {"pit": "ann_date", "code": "ts_code"},
    "forecast": {"pit": "ann_date", "code": "ts_code"},
    "express": {"pit": "ann_date", "code": "ts_code"},
    "report_rc": {"pit": "report_date", "code": "ts_code"},
    "cashflow": {"pit": "ann_date", "code": "ts_code"},
    "income": {"pit": "ann_date", "code": "ts_code"},
    "dividend": {"pit": "ann_date", "code": "ts_code"},
    "share_float": {"pit": "ann_date", "code": "ts_code"},
    "pledge_stat": {"pit": "ann_date", "code": "ts_code"},
    "fund_portfolio": {"pit": "ann_date", "code": "symbol"},
    "stk_holdernumber": {"pit": "ann_date", "code": "ts_code"},
    # 物化四族源（只登记存在性——物化列 status=deprecated，不进门禁仿真）
    "stk_holdertrade": {"pit": "ann_date", "code": "ts_code"},
    "repurchase": {"pit": "ann_date", "code": "ts_code"},
    "top10_floatholders": {"pit": "ann_date", "code": "ts_code"},
    "top_inst": {"pit": "trade_date", "code": "ts_code"},
}

#: 事件窗数据集（覆盖率 = 近 20 交易日内有 ≥1 行记录的 universe 比例——lhb_ 20 日 /
#: block_discount 10 日滚动聚合语义，同日快照口径会把稀疏事件误判为不可用）
_EVENT_WINDOW_DATASETS: dict[str, int] = {"top_list": 20, "block_trade": 20}

#: 列 → 数据集映射（精确名优先于前缀；多数据集取 min 覆盖率）
_EXACT_MAP: dict[str, tuple[str, ...]] = {
    "dividend_payout_ratio": ("dividend", "income"),
    "days_to_unlock": ("share_float",),
    "unlock_ratio": ("share_float",),
    "holder_freshness_days": ("stk_holdernumber",),
    "holder_num_chg": ("stk_holdernumber",),
    "holder_num_chg_2q": ("stk_holdernumber",),
    "rqye_rzye_ratio": ("margin_detail",),
    "fundamental_freshness_days": ("fina_indicator",),
    "consensus_freshness_days": ("report_rc",),
    "cons_revision_freshness_days": ("report_rc",),
    "express_freshness_days": ("express",),
    "forecast_freshness_days": ("forecast",),
    "dividend_freshness_days": ("dividend",),
    "dividend_days_to_ex_date": ("dividend",),
    "fund_portfolio_freshness_days": ("fund_portfolio",),
    "cashflow_freshness_days": ("cashflow",),
    "in_date": ("stock_basic",),
    "is_new_stock": ("stock_basic",),
}
_PREFIX_MAP: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("zscore_", ("__parent__",)),
    ("neu_", ("__parent__",)),
    ("cons_", ("report_rc",)),
    ("express_", ("express",)),
    ("forecast_", ("forecast",)),
    ("dividend_", ("dividend",)),
    ("dv_ttm", ("daily_basic",)),
    ("cashflow_", ("cashflow",)),
    ("cf_", ("cashflow",)),
    ("ocf", ("cashflow",)),
    ("fcf", ("cashflow",)),
    ("capex_to_ocf", ("cashflow",)),
    ("q_ocf", ("cashflow",)),
    ("fund_", ("fund_portfolio",)),
    ("winner_rate", ("cyq_perf",)),
    ("cost_concentration", ("cyq_perf",)),
    ("rzye", ("margin_detail",)),
    ("short_", ("margin_detail",)),
    ("margin_", ("margin_detail",)),
    ("north_", ("moneyflow_hsgt",)),
    ("lhb_", ("__daily__",)),  # 个股级稀疏事件，未上榜 0 填充（LhbFactorHandler）⇒ 非 NaN
    ("elg_", ("moneyflow",)),
    ("lg_", ("moneyflow",)),
    ("net_mf", ("moneyflow",)),
    ("order_imbalance", ("moneyflow",)),
    ("pledge_", ("pledge_stat",)),
    ("unlock_", ("share_float",)),
    ("block_discount_", ("block_trade",)),
    ("holder_", ("stk_holdernumber",)),
    ("ht_", ("stk_holdertrade",)),
    ("rp_", ("repurchase",)),
    ("tfh_", ("top10_floatholders",)),
    ("ti_", ("top_inst",)),
    ("sw_", ("shenwan_industry",)),
)
#: fina_indicator 家族（fundamental 族内除已映射外的其余列由 default 兜底）
_FINA_PREFIXES = (
    "roe", "roa", "grossprofit", "netprofit", "assets_turn", "inv_turn", "current_ratio",
    "debt_to_assets", "quick_ratio", "equity_yoy", "or_yoy", "profit_dedt", "q_gr_yoy",
    "int_to_talcap", "inventory", "assets_", "debt_", "gross",
)
_DAILY_BASIC_EXACT = {
    "pb", "pe_ttm", "ps_ttm", "dv_ttm", "total_mv", "circ_mv", "turnover_rate",
    "volume_ratio", "ep_ttm", "bp", "log_total_mv", "log_circ_mv",
    "pe_ttm_missing", "dv_ttm_missing", "is_loss",
}
#: daily 派生（量价/风控/市场状态/标签/可交易标记）——universe 即 daily 行 ⇒ 100%
#: 但 250 日长窗列须 daily 历史 ≥ 窗口
_LONG_WINDOW_250 = {
    "mkt_ma250_ratio", "mkt_atr_pct_ma250", "turnover_percentile", "vol_regime_percentile",
}
_STOCK_BASIC_COLS = {"in_date", "is_new_stock", "list_days"}


def _map_column(col: str) -> tuple[str, ...]:
    """列 → 源数据集元组（'__daily__' = daily 派生，'__parent__' = 随母列）。"""
    if col in _EXACT_MAP:
        return _EXACT_MAP[col]
    if col in _DAILY_BASIC_EXACT:
        return ("daily_basic",)
    if col in _STOCK_BASIC_COLS:
        return ("stock_basic",)
    for prefix, datasets in _PREFIX_MAP:
        if col.startswith(prefix):
            return datasets
    if col.startswith(_FINA_PREFIXES):
        return ("fina_indicator",)
    return ("__daily__",)


def _resolve_parent(col: str) -> str:
    for prefix in ("zscore_", "neu_"):
        if col.startswith(prefix):
            parent = col[len(prefix):]
            if parent.endswith("_sz"):
                parent = parent[: -len("_sz")]
            return parent
    return col


def _norm_dates(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("-", "", regex=False).str[:8]


def _load_history_codes(dataset: str, pit_col: str, code_col: str) -> pd.DataFrame:
    """一次性加载 (code, pit_date)（分区目录或单文件；pit 列缺失按 end_date →
    trade_date → report_date 回退；code_col 支持 symbol 等别名）。"""
    import pyarrow.parquet as pq

    def _pick_pit(names: set[str]) -> str | None:
        for c in (pit_col, "end_date", "trade_date", "report_date"):
            if c in names:
                return c
        return None

    def _pick_code(names: set[str]) -> str | None:
        for c in (code_col, "ts_code", "symbol"):
            if c in names:
                return c
        return None

    ddir = RAW / dataset
    single = RAW / f"{dataset}.parquet"
    if ddir.is_dir():
        parts = []
        for path in sorted(ddir.glob("*.parquet")):
            names = set(pq.ParquetFile(path).schema_arrow.names)
            use_pit, use_code = _pick_pit(names), _pick_code(names)
            if use_pit is None or use_code is None:
                continue
            parts.append(
                pd.read_parquet(path, columns=[use_code, use_pit]).rename(
                    columns={use_code: "code", use_pit: "pit"}
                )
            )
        if not parts:
            return pd.DataFrame(columns=["code", "pit"])
        df = pd.concat(parts, ignore_index=True)
    elif single.exists():
        names = set(pq.ParquetFile(single).schema_arrow.names)
        use_pit, use_code = _pick_pit(names), _pick_code(names)
        if use_pit is None or use_code is None:
            return pd.DataFrame(columns=["code", "pit"])
        df = pd.read_parquet(single, columns=[use_code, use_pit]).rename(
            columns={use_code: "code", use_pit: "pit"}
        )
    else:
        return pd.DataFrame(columns=["code", "pit"])
    df["code"] = df["code"].astype(str)
    df["pit"] = _norm_dates(df["pit"])
    df = df[df["pit"].str.match(r"^\d{8}$", na=False)]
    return df


def _trade_days_in_year(year: int) -> list[str]:
    days = []
    for path in sorted((RAW / "daily").glob(f"{year}-*.parquet")):
        days.append(path.stem.replace("-", ""))
    return days


def main() -> int:
    t0 = time.time()
    # 采样日：2005~2011 各季末（取该季最后一个有 daily 分区的交易日）+ 20120104
    sample_days: list[str] = []
    for year in range(2005, 2012):
        days = _trade_days_in_year(year)
        for q_end in ("0331", "0630", "0930", "1231"):
            q = [d for d in days if d <= f"{year}{q_end}"]
            if q:
                sample_days.append(q[-1])
    sample_days.append("20120104")
    logger.info(f"采样日 {len(sample_days)} 个: {sample_days[0]} ~ {sample_days[-1]}")

    # 历史数据集一次性加载
    history: dict[str, pd.DataFrame] = {}
    for ds, cfg in _HISTORY_DATASETS.items():
        history[ds] = _load_history_codes(ds, cfg["pit"], cfg["code"])
        logger.info(f"历史数据集 {ds}: {len(history[ds])} 行（code+pit）")

    # 事件窗数据集一次性加载（ts_code, trade_date）
    event_hist: dict[str, pd.DataFrame] = {}
    for ds in _EVENT_WINDOW_DATASETS:
        event_hist[ds] = _load_history_codes(ds, "trade_date", "ts_code")
        logger.info(f"事件窗数据集 {ds}: {len(event_hist[ds])} 行")

    # 交易日历（daily 分区名 ⇒ 事件窗回看对齐）
    all_trade_days = sorted(
        p.stem.replace("-", "") for p in (RAW / "daily").glob("*.parquet")
    )

    # daily 全量首末（长窗列历史深度判定）
    daily_first = "20050104"

    columns = [c for cols in PANEL_GROUPS.values() for c in cols if c not in ("trade_date", "ts_code")]
    col_ds = {c: _map_column(c) for c in columns}

    # 逐采样日：universe + 各数据集覆盖率
    day_rows: list[dict] = []
    for day in sample_days:
        daily_path = RAW / "daily" / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet"
        universe = set(
            pd.read_parquet(daily_path, columns=["ts_code"])["ts_code"].astype(str)
        )
        cov: dict[str, float] = {"__daily__": 1.0, "stock_basic": 1.0, "shenwan_industry": 1.0}
        for ds in _DAY_DATASETS:
            p = RAW / ds / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet"
            if p.exists():
                codes = set(pd.read_parquet(p, columns=["ts_code"])["ts_code"].astype(str))
                cov[ds] = len(universe & codes) / len(universe)
            else:
                cov[ds] = 0.0
        for ds, hist in history.items():
            if len(hist) == 0:
                cov[ds] = 0.0
                continue
            codes = set(hist.loc[hist["pit"] <= day, "code"])
            cov[ds] = len(universe & codes) / len(universe)
        # 事件窗数据集：近 N 交易日内有记录的 universe 比例
        import bisect

        day_idx = bisect.bisect_right(all_trade_days, day)
        for ds, win in _EVENT_WINDOW_DATASETS.items():
            hist = event_hist[ds]
            if len(hist) == 0 or day_idx == 0:
                cov[ds] = 0.0
                continue
            win_start = all_trade_days[max(0, day_idx - win)]
            mask = (hist["pit"] >= win_start) & (hist["pit"] <= day)
            codes = set(hist.loc[mask, "code"])
            cov[ds] = len(universe & codes) / len(universe)
        day_rows.append({"day": day, "universe": len(universe), "coverage": cov})
        logger.info(f"  {day}: universe={len(universe)}（{(time.time()-t0):.0f}s）")

    # 列级覆盖率（min over 源数据集；__parent__ 递归解析；250 日长窗加 daily 深度条件）
    def col_cov(col: str, day: str, cov: dict[str, float], depth: int = 0) -> float:
        if depth > 3:
            return 0.0
        if col in _LONG_WINDOW_250:
            # daily 2005-01-04 起，250 交易日前不可用
            return 1.0 if day >= "20060110" else 0.0
        datasets = col_ds[col]
        vals = []
        for ds in datasets:
            if ds == "__parent__":
                vals.append(col_cov(_resolve_parent(col), day, cov, depth + 1))
            elif ds == "__daily__":
                vals.append(1.0 if day >= daily_first else 0.0)
            else:
                vals.append(cov.get(ds, 0.0))
        return min(vals) if vals else 0.0

    col_day_cov: dict[str, dict[str, float]] = {}
    for row in day_rows:
        day = row["day"]
        for col in columns:
            col_day_cov.setdefault(col, {})[day] = round(col_cov(col, day, row["coverage"]), 4)

    # 年度聚合（年均覆盖率）+ available_from（首个覆盖率 ≥40% 的采样日）
    years = list(range(2005, 2012))
    col_year_cov: dict[str, dict[str, float]] = {}
    col_available_from: dict[str, str | None] = {}
    for col, dcov in col_day_cov.items():
        col_year_cov[col] = {}
        for year in years:
            days_y = [d for d in dcov if d.startswith(str(year))]
            if days_y:
                col_year_cov[col][f"{year}"] = round(sum(dcov[d] for d in days_y) / len(days_y), 4)
        avail = None
        for row in day_rows:
            if dcov[row["day"]] >= 0.4:
                avail = row["day"]
                break
        col_available_from[col] = avail

    # 门禁仿真：现役模型 154 列逐年幸存率
    # 模型版本升级后此剖面语义需重新标定（硬编码 v24288 列集快照）
    model_cols = json.load(open(ROOT / "data/models/stock_selection/v24288_features.json", encoding="utf-8"))
    model_cols = [c for c in model_cols if c in col_year_cov]
    gate: dict[str, dict] = {}
    for year in years:
        survived = [c for c in model_cols if col_year_cov[c].get(f"{year}", 0.0) >= 0.4]
        gate[f"{year}"] = {
            "model154_survived": len(survived),
            "model154_total": len(model_cols),
            "share": round(len(survived) / max(len(model_cols), 1), 4),
            "dropped": sorted(set(model_cols) - set(survived)),
        }

    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sample_days": sample_days,
        "day_rows": day_rows,
        "col_year_cov": col_year_cov,
        "col_available_from": col_available_from,
        "gate_simulation": gate,
        "notes": {
            "gate": "幸存 = 年均覆盖率 ≥40%（训练入口 max_feature_missing_ratio=0.6）",
            "derived": "daily 派生列 = 100%（250 日长窗自 20060110 起）；zscore/neu 随母列",
            "materialized": "物化四族（deprecated）不进门禁仿真",
        },
    }
    out_path = ROOT / "data" / "reports" / "v2_p1_column_availability_20261006.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info(f"剖面产物: {out_path}（耗时 {(time.time()-t0)/60:.1f} min）")

    print("\n=== 门禁仿真（现役模型 154 列逐年幸存）===")
    for year in years:
        g = gate[f"{year}"]
        print(f"  {year}: {g['model154_survived']}/{g['model154_total']} = {g['share']:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
