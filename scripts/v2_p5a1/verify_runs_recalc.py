# -*- coding: utf-8 -*-
"""P5a-1 A3 验收（正式脚本，R-08 修复：脚本 + 报告 + 校验表三件套）。

3 个历史实验批（holdertrade A2 core / repurchase / top10fh，契约 §10 登记口径）：
① runs_convert 转换为 runs 契约 schema；② runs_loader 读入（证据机器通路）；
③ 链式全周期指标重算与历史报告登记值逐项比对；④ 产出校验表 CSV + 报告 JSON。

产物落盘 data/reports/p5a1_a3_recalc_<date>.{csv,json}。

勘误登记（v0.203.1）：v0.203.0 曾误用 holdertrade **B0 基线批**
（phase4_ht_ab_20260917_base）错标 `holdertrade_A1_full` 验收；契约 §10 登记对象为
A2 core 臂（phase4b_ht_core_20260917，报告 §7.2 值 15.147% / −25.09% / 0.6275），
本次改回契约口径。

用法：
    python scripts/v2_p5a1/verify_runs_recalc.py [--out-root temp/runs_converted]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.compare.fold_subset import (  # noqa: E402
    RunArtifacts,
    chain_metrics_from_fold_returns,
    per_fold_returns,
)
from src.lazybull.v2.evidence.runs_convert import convert_wf_batch  # noqa: E402
from src.lazybull.v2.evidence.runs_loader import load_runs_batch  # noqa: E402

# 验收批（P5a-1 契约 §10 登记的 3 个历史实验；holdertrade 为 A2 core 臂）
BATCHES = {
    "holdertrade_A2_core": ROOT / "data" / "walk_forward" / "batches" / "phase4b_ht_core_20260917" / "raw",
    "repurchase_headroom": ROOT / "data" / "walk_forward" / "batches" / "phase4_rp_headroom_20260918" / "raw",
    "top10fh_concentration": ROOT / "data" / "walk_forward" / "batches" / "phase4_tfh_conc_20260919" / "raw",
}
# 三批的共同对照基线（契约 §2 baseline_ref；holdertrade A0 / repurchase B0 / top10fh B0
# 均为同一批次，见 docs/*_wf_ab_result.md 批次表）
BASELINE_REF = "phase4_ht_ab_20260917_base"
# 历史报告登记的链式全周期值（docs/*_wf_ab_result.md，4 位小数截断）
LEGACY_CHAIN = {
    # holdertrade_wf_ab_result.md §7.2（A2 core 臂）：15.147% / −25.09% / 0.6275
    "holdertrade_A2_core": {"cagr": 0.1514, "max_drawdown": -0.2509, "sharpe": 0.6275},
    "repurchase_headroom": {"cagr": 0.1435, "max_drawdown": -0.2021, "sharpe": 0.6049},
    "top10fh_concentration": {"cagr": 0.1413, "max_drawdown": -0.2802, "sharpe": 0.5978},
}
CHAIN_TOL = 1e-4  # 链式裁决口径容差（报告值 4 位小数截断）


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty():
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError):
                pass
    parser = argparse.ArgumentParser(description="P5a-1 A3 验收（3 批重算一致）")
    parser.add_argument("--out-root", default="temp/runs_converted", help="转换产物输出根（临时）")
    parser.add_argument("--keep", action="store_true", help="保留转换产物（默认跑完清理）")
    args = parser.parse_args(argv)

    out_root = ROOT / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    all_ok = True

    for label, raw_dir in BATCHES.items():
        if not raw_dir.exists():
            print(f"[{label}] 缺批次目录，跳过")
            all_ok = False
            continue
        # ① 转换（A/B 臂批登记 baseline_ref，契约 §2）
        report = convert_wf_batch(raw_dir, out_root, label, baseline_ref=BASELINE_REF)
        # ② 证据机器读入（runs_loader 通路 + 读取不变量硬校验）
        try:
            batch = load_runs_batch(out_root / label)
            load_ok = True
            load_err = ""
        except Exception as exc:  # noqa: BLE001 - 验收须捕获并记录
            load_ok = False
            load_err = str(exc)
            batch = None
        # ③ 重算链式指标
        recalc = {"cagr": None, "max_drawdown": None, "sharpe": None}
        n_folds = 0
        if load_ok and batch is not None:
            run = batch.to_run_artifacts()
            splits = sorted(run.chain["split_index"].unique().tolist())
            n_folds = len(splits)
            folds = per_fold_returns(run, splits)
            recalc = chain_metrics_from_fold_returns([folds[s] for s in splits])
        # ④ 与历史报告比对
        ref = LEGACY_CHAIN[label]
        d_cagr = abs((recalc["cagr"] or float("nan")) - ref["cagr"])
        d_maxdd = abs((recalc["max_drawdown"] or float("nan")) - ref["max_drawdown"])
        d_sharpe = abs((recalc["sharpe"] or float("nan")) - ref["sharpe"])
        chain_ok = bool(d_cagr <= CHAIN_TOL and d_maxdd <= CHAIN_TOL and d_sharpe <= CHAIN_TOL)
        batch_ok = bool(report.row_check_pass and load_ok and chain_ok)
        all_ok &= batch_ok
        rows.append({
            "批次": label,
            "转换文件数": len(report.files),
            "转换行数校验": "通过" if report.row_check_pass else "失败",
            "runs读入": "通过" if load_ok else f"失败: {load_err}",
            "折数": n_folds,
            "重算CAGR": recalc["cagr"],
            "重算MaxDD": recalc["max_drawdown"],
            "重算夏普": recalc["sharpe"],
            "ΔCAGR": d_cagr,
            "ΔMaxDD": d_maxdd,
            "Δ夏普": d_sharpe,
            "链式一致": "是" if chain_ok else "否",
            "判定": "通过" if batch_ok else "未过",
        })
        print(f"[{label}] 转换 {len(report.files)} 文件 / 读入 {'OK' if load_ok else 'FAIL'} / "
              f"链式 Δcagr={d_cagr:.2e} Δmaxdd={d_maxdd:.2e} Δsharpe={d_sharpe:.2e} "
              f"→ {'通过' if batch_ok else '未过'}")

    # 落盘：校验表 CSV + 报告 JSON（R-08 三件套）
    df = pd.DataFrame(rows)
    date_str = datetime.now().strftime("%Y%m%d")
    csv_path = ROOT / "data" / "reports" / f"p5a1_a3_recalc_{date_str}.csv"
    json_path = ROOT / "data" / "reports" / f"p5a1_a3_recalc_{date_str}.json"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "verdict": "通过" if all_ok else "未过",
        "batches": rows,
        "criteria": f"链式全周期指标重算与历史报告逐项一致（容差 {CHAIN_TOL}）",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== A3 验收：{'[通过]' if all_ok else '[未过]'} ===")
    print(f"校验表: {csv_path.relative_to(ROOT)}")
    print(f"报告: {json_path.relative_to(ROOT)}")

    # 默认清理临时转换产物（遵守临时文件规范）
    if not args.keep:
        import shutil
        shutil.rmtree(out_root, ignore_errors=True)
        print(f"临时转换产物已清理: {args.out_root}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
