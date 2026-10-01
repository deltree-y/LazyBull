# -*- coding: utf-8 -*-
"""P5a-1 功效标定 + 判据自洽性（正式入口，R-04 修复：落盘产物 + 可复现脚本）。

真实换种子批驱动（B0 = factor_ab_20260916_071157_base seeds 42,61,82；
B0′ = factor_ab_20260916_123214_noiseB0 seeds 43,62,83）。
产物落盘 data/reports/p5a1_power_calibration_<date>.json。

用法：
    python scripts/v2_p5a1/run_power_calibration.py [--n-noise 20] [--n-boot 100]
        [--shift-grid 2 5 8] [--shift-pp 2.0] [--threshold 0.8]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.compare.fold_subset import load_run  # noqa: E402
from src.lazybull.v2.evidence.criterion_calibration import (  # noqa: E402
    criterion_self_consistency_dual,
)
from src.lazybull.v2.evidence.power_calibration import power_calibration_curve  # noqa: E402

B0_DIR = ROOT / "data" / "walk_forward" / "batches" / "factor_ab_20260916_071157_base"
B0_ALT_DIR = ROOT / "data" / "walk_forward" / "batches" / "factor_ab_20260916_123214_noiseB0"


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty():
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError):
                pass
    parser = argparse.ArgumentParser(description="P5a-1 功效标定 + 判据自洽性")
    parser.add_argument("--n-noise", type=int, default=20, help="独立噪声实现数 M")
    parser.add_argument("--n-boot", type=int, default=100, help="每实现内配对重排次数")
    parser.add_argument("--shift-grid", nargs="+", type=float, default=[2.0, 5.0, 8.0])
    parser.add_argument("--shift-pp", type=float, default=2.0, help="判据自洽性平移档")
    parser.add_argument("--threshold", type=float, default=0.8, help="检出频率阈值")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    b0 = load_run(B0_DIR, label="B0")
    b0a = load_run(B0_ALT_DIR, label="B0'")
    splits = sorted(set(b0.chain["split_index"]) & set(b0a.chain["split_index"]))
    print(f"换种子批：B0(42,61,82) vs B0'(43,62,83)，共同折 {len(splits)}")

    # 功效标定曲线（频率语义）
    curve = power_calibration_curve(
        b0, splits, shift_grid_pp=args.shift_grid,
        n_boot=args.n_boot, n_noise=args.n_noise, seed=args.seed, alt_seed=b0a,
    )
    # 判据自洽性（双指标）
    consistency = criterion_self_consistency_dual(
        b0, b0a, splits, shift_pp=args.shift_pp,
        n_boot=args.n_boot, n_noise=args.n_noise, seed=args.seed,
        shift_detect_threshold=args.threshold,
    )

    # 输出
    print("\n=== 功效标定（频率语义）===")
    for p in curve:
        print(f"  +{p.shift_annual_pp:.0f}pp: ΔCAGR={p.detection_prob_cagr:.0%}，"
              f"ΔMaxDD={p.detection_prob_maxdd:.0%}（M={p.n_noise}, {p.noise_source}）")
    print("\n=== 判据自洽性 ===")
    for m, r in consistency.items():
        print(f"  [{m}] 换种子不可区分={r.seed_swap_indistinguishable} "
              f"/ +{r.shift_pp:.0f}pp 检出频率={r.shift_detection_freq:.0%} "
              f"（阈值 {r.shift_detect_threshold:.0%}）/ 通过={r.criterion_passes}")

    # 落盘（R-04）
    out = {
        "timestamp": datetime.now().isoformat(),
        "batches": {"B0": str(B0_DIR.name), "B0_alt": str(B0_ALT_DIR.name)},
        "config": {"n_noise": args.n_noise, "n_boot": args.n_boot,
                   "shift_grid": args.shift_grid, "shift_pp": args.shift_pp,
                   "threshold": args.threshold, "seed": args.seed},
        "n_folds": len(splits),
        "power_curve": [
            {"shift_annual_pp": p.shift_annual_pp,
             "detection_prob_cagr": p.detection_prob_cagr,
             "detection_prob_maxdd": p.detection_prob_maxdd,
             "noise_source": p.noise_source,
             "dof_note": p.effective_dof_note}
            for p in curve
        ],
        "criterion_self_consistency": {
            m: {"seed_swap_indistinguishable": r.seed_swap_indistinguishable,
                "seed_swap_is_real": r.seed_swap_is_real,
                "shift_pp": r.shift_pp,
                "shift_detection_freq": r.shift_detection_freq,
                "shift_detect_threshold": r.shift_detect_threshold,
                "shifted_detectable": r.shifted_detectable,
                "criterion_passes": r.criterion_passes,
                "note": r.note}
            for m, r in consistency.items()
        },
    }
    out_path = ROOT / "data" / "reports" / f"p5a1_power_calibration_{datetime.now():%Y%m%d}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已落盘: {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
