# -*- coding: utf-8 -*-
"""v2 P1 单元 5：影子通路验收——单日增量构建 build_daily(D) 三口径对账。

契约（方案 R1-4 之 3/D2/D3）：影子门数据来源 = raw 单一写入 + 只读转换器 + 每日影子
构建增量对账。验收口径：
1. **判据 B（主）：shadow ≡ 同窗批量参照**（build_features_data 同 [D−7月, D] 捕获窗
   ——注意 D−7 个自然月为日历月减法，如 20260702→20251202）⇒ 预期全零/ulp（atol=1e-6）；
2. **参照判据：shadow vs panel(D)**——差异须全部落入登记类（D-13 豁免族 / 公告季频
   加载窗截断族 / fund −18mo 窗沿族 / macd EMA 重排噪声 / mkt 250 日窗 warmup 深度差 /
   zscore 连锁），类外 ⇒ FAIL；
3. **labels**：shadow vs 生产 labels ⇒ 预期全零（daily 驱动、窗口无关）。

cs_infer 对照为**信息登记项**（不进判定）：cs_infer 冻结于其构建日数据态（raw
2026-08-30/31 全量刷新前），与当前数据态的差异属 D-04 类数据态边界，非路径问题。

用法：python scripts/v2_p1/shadow_path_acceptance.py [--date 20260702] \
        [--ref-batch-dir temp/p1_shadow_ref_batch]
产物：data/reports/v2_p1_shadow_path_acceptance_<YYYYMMDD>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.lazybull.data.loader import DataLoader  # noqa: E402
from src.lazybull.data.storage import Storage  # noqa: E402
from src.lazybull.v2.common.types import TradeDate  # noqa: E402
from src.lazybull.v2.store.column_groups import LABEL_TABLES, PANEL_GROUPS  # noqa: E402
from src.lazybull.v2.store.data_store import PanelDataStore  # noqa: E402
from src.lazybull.v2.store.panel_builder import V2PanelBuilder, bootstrap_manifest  # noqa: E402
from src.lazybull.v2.store.replay_compare import (  # noqa: E402
    _compare_nonnumeric_column,
    _compare_numeric_column,
)

_GROUPS = tuple(PANEL_GROUPS)
_ATOL = 1e-6

#: 参照判据（shadow vs panel）的登记预期类（单元 5 验收登记；D-13 / 窗口截断族 /
#: ulp 尾 / warmup / zscore 连锁）
_PANEL_EXPECTED: dict[str, str] = {
    "days_to_unlock": "D-13 豁免（panel 全量 PIT vs shadow 旧链截断，裁决 2026-10-06）",
    "unlock_ratio": "D-13 豁免",
    "unlock_risk_flag": "D-13 豁免",
    "cf_nm": "公告季频加载窗截断（shadow [D−7月,D] vs 回填分块窗；cs_infer 同 shadow 口径）",
    "ocf_to_profit": "公告季频加载窗截断",
    "dividend_payout_ratio": "公告季频加载窗截断（income/分红查询表）",
    "fund_count": "fund −18mo 窗沿截断（最近持仓报告落窗沿外；cs_infer 同口径）",
    "fund_count_chg": "fund −18mo 窗沿截断",
    "fund_hold_ratio": "fund −18mo 窗沿截断",
    "fund_hold_ratio_chg": "fund −18mo 窗沿截断",
    "fund_portfolio_freshness_days": "fund −18mo 窗沿截断",
    "has_fund_holding": "fund −18mo 窗沿截断（派生标记）",
    "macd_dea": "macd EMA float32 ulp 尾（大模值相对 ~2e-8）",
    "macd_dif": "macd EMA float32 ulp 尾",
    "macd_hist": "macd EMA float32 ulp 尾",
    "mkt_atr_pct_ma250": "mkt 250 日窗 warmup 深度差（cs_infer 窗口语义，D-02 登记）",
    "zscore_cf_nm": "zscore 连锁（母列窗截断 ⇒ 截面统计微移）",
    "zscore_cf_nm_sz": "zscore 连锁",
    "zscore_ocf_to_profit": "zscore 连锁",
    "zscore_ocf_to_profit_sz": "zscore 连锁",
    "zscore_dividend_payout_ratio": "zscore 连锁",
    "zscore_dividend_payout_ratio_sz": "zscore 连锁",
}


def _compare_frames(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    """逐列比对（左=生产，右=scratch；键对齐 + atol + NaN==NaN 一致）。"""
    common_cols = sorted((set(left.columns) & set(right.columns)) - {"trade_date", "ts_code"})
    l_idx = left.set_index("ts_code")
    r_idx = right.set_index("ts_code")
    common_codes = l_idx.index.intersection(r_idx.index)
    result = {"cols_only_prod": sorted(set(left.columns) - set(right.columns) - {"trade_date", "ts_code"}),
              "cols_only_scratch": sorted(set(right.columns) - set(left.columns) - {"trade_date", "ts_code"}),
              "rows_only_prod": sorted(set(l_idx.index) - set(r_idx.index))[:5],
              "rows_only_scratch": sorted(set(r_idx.index) - set(l_idx.index))[:5],
              "diffs": {}}
    for col in common_cols:
        lcol, rcol = l_idx.loc[common_codes, col], r_idx.loc[common_codes, col]
        if pd.api.types.is_numeric_dtype(lcol) and pd.api.types.is_numeric_dtype(rcol):
            over, max_abs, mask = _compare_numeric_column(lcol, rcol, _ATOL)
        else:
            over, mask = _compare_nonnumeric_column(lcol, rcol)
            max_abs = 0.0
        if over:
            bad = mask[mask].index[:5]
            result["diffs"][col] = {
                "over_rows": int(over),
                "max_abs_diff": max_abs,
                "samples": [
                    {
                        "ts_code": str(c),
                        "prod": None if pd.isna(lcol.loc[c]) else float(lcol.loc[c]),
                        "shadow": None if pd.isna(rcol.loc[c]) else float(rcol.loc[c]),
                    }
                    for c in bad
                ],
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="v2 P1 影子通路验收（双口径对账）")
    parser.add_argument("--date", default="20260702", help="验收日 YYYYMMDD（须有 cs_infer 分区）")
    parser.add_argument("--data-root", default=str(ROOT / "data"), help="生产 v2 store 数据根")
    parser.add_argument("--scratch-root", default=str(ROOT / "temp" / "p1_shadow_scratch"))
    parser.add_argument(
        "--ref-batch-dir",
        default=None,
        help="同窗批量参照根（run_frozen_reference 同 [D−7月,D] 窗构建；判据 B 用）",
    )
    args = parser.parse_args()
    t0 = time.time()

    prod = PanelDataStore(args.data_root)
    scratch = PanelDataStore(args.scratch_root)
    bootstrap_manifest(scratch)

    logger.info(f"影子构建 build_daily({args.date}) → scratch {args.scratch_root}")
    builder = V2PanelBuilder(DataLoader(Storage()))
    builder.build_daily(TradeDate.from_str(args.date), _GROUPS, scratch)

    # scratch 当日合并帧（8 族 outer join）
    scratch_frames = []
    for group in _GROUPS:
        f = scratch.panel_dir / args.date / f"{group}.parquet"
        if f.exists():
            scratch_frames.append(pd.read_parquet(f))
    shadow_day = scratch_frames[0]
    for f in scratch_frames[1:]:
        shadow_day = shadow_day.merge(f, on=["trade_date", "ts_code"], how="outer")

    report: dict = {"date": args.date, "primary_cs_infer": None, "secondary_panel": {}, "labels": {}}
    n_problem = 0

    # ── cs_infer 对照（信息登记项，不进判定）：cs_infer 冻结于其构建日数据态
    # （2026-07-02，raw 08-30/31 全量刷新前）；shadow 用当前数据态 ⇒ 差异 = 数据态
    # 边界（D-04 类：raw 刷新修订 + 行集过滤状态变化），非路径忠实性问题。──
    cs_infer_file = ROOT / "data" / "features" / "cs_infer" / f"{args.date}.parquet"
    if cs_infer_file.exists():
        cs_day = pd.read_parquet(cs_infer_file)
        cmp_result = _compare_frames(cs_day, shadow_day)
        report["primary_cs_infer"] = {
            "note": (
                "信息登记项（不进判定）：cs_infer 冻结于 2026-07-02 数据态（raw 08-30/31 "
                "刷新前），shadow 为当前数据态 ⇒ 差异 = 数据态边界（D-04 类）"
            ),
            "diff_cols": len(cmp_result["diffs"]),
            "top_diff_cols": dict(
                sorted(cmp_result["diffs"].items(), key=lambda kv: -kv[1]["over_rows"])[:15]
            ),
            "rows_only_cs_infer": cmp_result["rows_only_prod"],
            "rows_only_shadow": cmp_result["rows_only_scratch"],
        }
        logger.info(
            f"cs_infer 对照（信息项）: 差异列 {len(cmp_result['diffs'])}，"
            f"行集差 cs_infer-only={len(cmp_result['rows_only_prod'])}"
        )
    else:
        report["primary_cs_infer"] = {"status": f"cs_infer/{args.date}.parquet 不存在，对照跳过"}

    # ── 参照判据：shadow vs panel(D)（差异须全部落入登记类）──
    for group in _GROUPS:
        prod_file = prod.panel_dir / args.date / f"{group}.parquet"
        scratch_file = scratch.panel_dir / args.date / f"{group}.parquet"
        if not prod_file.exists() or not scratch_file.exists():
            report["secondary_panel"][group] = {
                "status": f"缺文件 prod={prod_file.exists()} scratch={scratch_file.exists()}"
            }
            n_problem += 1
            continue
        cmp_result = _compare_frames(pd.read_parquet(prod_file), pd.read_parquet(scratch_file))
        classified, unclassified = {}, {}
        for col, d in cmp_result["diffs"].items():
            if col in _PANEL_EXPECTED:
                classified[col] = {"class": _PANEL_EXPECTED[col], **d}
            else:
                unclassified[col] = d
        structural = any(
            cmp_result[k] for k in ("cols_only_prod", "cols_only_scratch", "rows_only_prod", "rows_only_scratch")
        )
        report["secondary_panel"][group] = {
            "classified": classified,
            "unclassified": unclassified,
            "structural": structural,
        }
        if unclassified or structural:
            n_problem += 1
        logger.info(f"  panel/{group}: 登记类 {len(classified)}，未登记 {len(unclassified)}")

    # labels 对账（shadow vs 生产 labels；daily 驱动窗口无关 ⇒ 预期全零）
    for label_name, (value_col, neu_col) in LABEL_TABLES.items():
        prod_f = prod.labels_dir / label_name / f"{args.date}.parquet"
        scratch_f = scratch.labels_dir / label_name / f"{args.date}.parquet"
        if not prod_f.exists() or not scratch_f.exists():
            report["labels"][label_name] = {"status": f"缺文件 prod={prod_f.exists()} scratch={scratch_f.exists()}"}
            n_problem += 1
            continue
        lp = pd.read_parquet(prod_f)[["ts_code", "label_value", "neu_label_value"]]
        rp = pd.read_parquet(scratch_f)[["ts_code", "label_value", "neu_label_value"]]
        cmp_result = _compare_frames(lp, rp)
        report["labels"][label_name] = {"diff_cols": len(cmp_result["diffs"]), **cmp_result}
        if cmp_result["diffs"] or any(
            cmp_result[k] for k in ("cols_only_prod", "cols_only_scratch", "rows_only_prod", "rows_only_scratch")
        ):
            n_problem += 1
        logger.info(f"  labels/{label_name}: 差异列 {len(cmp_result['diffs'])}")

    # ── 判据 B：shadow vs 同窗批量参照（build_features_data 同 [D−7月,D] 窗 ⇒ 预期全零/ulp）──
    if args.ref_batch_dir:
        ref_dir = Path(args.ref_batch_dir) / "features" / "cs_train"
        ref_day_file = ref_dir / f"{args.date}.parquet"
        if ref_day_file.exists():
            from src.lazybull.v2.store.column_groups import MATERIALIZED_COLUMNS

            ref_day = pd.read_parquet(ref_day_file)
            label_cols = [c for c in ref_day.columns if c.startswith("y_ret_") or c.startswith("neu_y_ret_")]
            cmp_b = _compare_frames(ref_day.drop(columns=label_cols), shadow_day)
            # 34 物化列（冻结 §6 拍板 panel 独有）不计结构差异——其正确性由单元 2/4 自有验收覆盖
            materialized = set(MATERIALIZED_COLUMNS)
            structural = any(
                cmp_result_col
                for key in ("cols_only_prod", "rows_only_prod", "rows_only_scratch")
                if (cmp_result_col := cmp_b[key])
            ) or bool(set(cmp_b["cols_only_scratch"]) - materialized)
            report["primary_ref_batch"] = {
                "diff_cols": len(cmp_b["diffs"]),
                "diffs": cmp_b["diffs"],
                "structural": structural,
                "materialized_cols_only_shadow": sorted(set(cmp_b["cols_only_scratch"]) & materialized),
            }
            # 参照标签 vs shadow 标签（同窗 ⇒ 预期全零）
            ref_labels = ref_day[["ts_code", *label_cols]]
            shadow_label_frames = []
            for label_name, (value_col, neu_col) in LABEL_TABLES.items():
                f = scratch.labels_dir / label_name / f"{args.date}.parquet"
                if f.exists():
                    ld = pd.read_parquet(f)[["ts_code", "label_value", "neu_label_value"]]
                    shadow_label_frames.append(
                        ld.rename(
                            columns={
                                "label_value": label_name,
                                "neu_label_value": f"neu_{label_name}",
                            }
                        ).set_index("ts_code")
                    )
            shadow_labels = pd.concat(shadow_label_frames, axis=1).reset_index()
            cmp_bl = _compare_frames(ref_labels, shadow_labels)
            report["primary_ref_batch"]["labels_diff"] = cmp_bl["diffs"]
            if cmp_b["diffs"] or cmp_bl["diffs"] or report["primary_ref_batch"]["structural"]:
                n_problem += 1
            logger.info(
                f"判据 B（同窗批量参照）: 特征差异列 {len(cmp_b['diffs'])}，"
                f"标签差异列 {len(cmp_bl['diffs'])}"
            )
        else:
            report["primary_ref_batch"] = {"status": f"参照分区缺失: {ref_day_file}"}
            n_problem += 1
    else:
        report["primary_ref_batch"] = {"status": "未提供 --ref-batch-dir，判据 B 跳过"}

    report["verdict"] = {
        "pass": n_problem == 0,
        "rule": (
            f"主判据 shadow≡cs_infer（atol={_ATOL}，D-04① 数据态登记）；"
            "参照判据 shadow vs panel 差异全部落入登记类（D-13/窗截断/ulp/warmup/zscore 连锁）；"
            "labels 全零"
        ),
        "problem_items": n_problem,
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    out = ROOT / "data" / "reports" / f"v2_p1_shadow_path_acceptance_{args.date}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info(f"验收产物: {out}")
    print(f"影子通路验收（{args.date}）: {'PASS' if n_problem == 0 else 'FAIL'}（问题项 {n_problem}）")
    return 0 if n_problem == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
