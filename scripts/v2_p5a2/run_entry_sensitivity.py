# -*- coding: utf-8 -*-
"""P5a-2：入场点敏感度报告（方案 §3.5 证据机器标准产物；§3.3 爬坡建仓立项门输入）。

口径（方案 F7 §3.5 写死）：以链式净值路径上每个交易日为起点入场，统计全期 CAGR /
区间 MaxDD / 总收益的**分布**——报告口径 = **最差 K 起点明细 + 中位数 / 离散度（IQR）**；
滑窗起点的有效自由度 ≈ 折数级（14 折；路径长 1,714，min_days=252 过滤后有效起点 1,462），
分位数精读是伪精度，**禁止**。

双臂：B1（现生产默认 e2online_r，主口径）与 B0（neutral，v2 迁移目标对照）。
输入 = P0 冻结的链式净值路径（baseline_freeze.md §5 登记批次）。

用法：
    python scripts/v2_p5a2/run_entry_sensitivity.py [--top-k 10] [--out-name NAME]
产物：data/reports/p5a2_entry_sensitivity_<date>.json（或 --out-name 指定名）。

通用性定位（F9 评审②-09 登记）：`ARMS` 硬编码 P0 冻结双臂（B0/B1）——本脚本服务 §3.3 立项门素材；
对任意批次 / 任意配置的通用化（`--nav` 参数）留待 M3 / P5b 需要时。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# P0 冻结批次（baseline_freeze.md §5）
ARMS = {
    "B1_e2online_r": ROOT / "data/walk_forward/batches/wf_batch_20260930_172037/raw/chain_nav_wf_20260930_172039_dff2a95a.csv",
    "B0_neutral": ROOT / "data/walk_forward/batches/wf_batch_20260930_171221/raw/chain_nav_wf_20260930_171223_f71a780c.csv",
}
ANNUAL_DAYS = 252


def load_nav(nav_path: Path) -> pd.DataFrame:
    df = pd.read_csv(nav_path)
    return df[["nav", "split_index"]].astype({"nav": float})


def entry_distribution(nav: np.ndarray, min_days: int) -> pd.DataFrame:
    """逐起点入场口径：t 日收盘入场（nav 归一到 t），持有至路径末端。

    起点须满足剩余交易日 ≥ min_days（默认 252 ≈ 一年）——过短窗口的年化是伪数字
    （末端 2 日窗口年化会把 −1% 放大成 −76%），污染分布。

    Returns:
        DataFrame：start_idx / cagr / maxdd / total_return。
    """
    n = nav.size
    rows = []
    for t in range(n - 1):
        days = n - 1 - t
        if days < min_days:
            continue
        window = nav[t:]
        total = window[-1] / window[0] - 1.0
        cagr = (1.0 + total) ** (ANNUAL_DAYS / days) - 1.0
        dd = window / np.maximum.accumulate(window) - 1.0
        rows.append({"start_idx": t, "days": days, "cagr": cagr, "maxdd": float(dd.min()),
                     "total_return": total})
    return pd.DataFrame(rows)


def summarize(dist: pd.DataFrame, splits: np.ndarray, top_k: int) -> dict:
    """中位数 / 离散度（IQR）+ 最差 K 起点明细（禁止分位数精读，方案 §3.5）。"""
    worst = dist.nsmallest(top_k, "cagr").copy()
    worst["split_index"] = [int(splits[i]) for i in worst["start_idx"]]
    return {
        "n_entries": int(len(dist)),
        "cagr_median": float(dist["cagr"].median()),
        "cagr_mean": float(dist["cagr"].mean()),
        "cagr_iqr": [float(dist["cagr"].quantile(0.25)), float(dist["cagr"].quantile(0.75))],
        "cagr_std": float(dist["cagr"].std()),
        "maxdd_median": float(dist["maxdd"].median()),
        "maxdd_worst": float(dist["maxdd"].min()),
        "total_return_median": float(dist["total_return"].median()),
        "worst_k_by_cagr": worst.to_dict(orient="records"),
        "dof_note": "有效自由度 ≈ 折数级（14 折）；滑窗起点高度重叠，分位数精读是伪精度（方案 §3.5）",
    }


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty():
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError):
                pass
    parser = argparse.ArgumentParser(description="P5a-2 入场点敏感度报告")
    parser.add_argument("--top-k", type=int, default=10, help="最差 K 起点明细（默认 10）")
    parser.add_argument("--min-days", type=int, default=252,
                        help="起点最短持有交易日（默认 252 ≈ 一年；过短窗口年化是伪数字）")
    parser.add_argument("--out-name", type=str, default=None, help="输出文件名（默认 p5a2_entry_sensitivity_<date>.json）")
    args = parser.parse_args(argv)

    out = {
        "timestamp": datetime.now().isoformat(),
        "purpose": "P5a-2 入场点敏感度（方案 §3.5 标准产物；§3.3 爬坡建仓立项门输入）",
        "config": {"top_k": args.top_k, "min_days": args.min_days},
        "arms": {k: str(v.parent.parent.name) for k, v in ARMS.items()},
        "results": {},
    }
    for arm, path in ARMS.items():
        df = load_nav(path)
        nav = df["nav"].to_numpy()
        splits = df["split_index"].to_numpy()
        dist = entry_distribution(nav, args.min_days)
        s = summarize(dist, splits, args.top_k)
        out["results"][arm] = s
        print(f"[{arm}] 起点 {s['n_entries']} 个：CAGR 中位 {s['cagr_median']:.2%}，"
              f"IQR [{s['cagr_iqr'][0]:.2%}, {s['cagr_iqr'][1]:.2%}]，std {s['cagr_std']:.2%}；"
              f"最差起点 CAGR {s['worst_k_by_cagr'][0]['cagr']:.2%}（#{s['worst_k_by_cagr'][0]['start_idx']}，"
              f"split{s['worst_k_by_cagr'][0]['split_index']}）；MaxDD 中位 {s['maxdd_median']:.2%} / "
              f"最差 {s['maxdd_worst']:.2%}")

    b0 = out["results"]["B0_neutral"]
    b1 = out["results"]["B1_e2online_r"]
    out["paired_note"] = {
        "cagr_median_delta_pp": (b1["cagr_median"] - b0["cagr_median"]) * 100,
        "note": "B1−B0 中位差仅登记（政策层已裁决退役，F8）；爬坡立项门判读用两臂较宽者",
    }

    name = (args.out_name or f"p5a2_entry_sensitivity_{datetime.now():%Y%m%d}").removesuffix(".json")
    out_path = ROOT / "data" / "reports" / f"{name}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n落盘: {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
