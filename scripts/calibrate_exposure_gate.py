"""terminal_loss 政策层 E2 入口：条件暴露门控的校准 + 三臂离线评估（P2-2 一阶筛选）。

用法（示例，按 WF 批次自动找台账）：

    python scripts/calibrate_exposure_gate.py \
        --batch wf_batch_20260914_084247 \
        --calibration-start 20220701 --calibration-end 20231229 \
        --eval-start 20240101 --eval-end 20251204 \
        --arms combined score regime \
        --de-exposure 0.5 --cost-bps 15

产出（全部**中文表头**，utf-8-sig 可直接用 Excel 打开）：

- ``暴露门控校准.csv``：每个臂一行，含冻结阈值（校准段分位 / 市场波动阈值 / 得分阈值）
- ``暴露门控逐日判定.csv``：一行 = 某臂某交易日（市场波动、组合平均风险概率、是否触发/首触、暴露系数）
- ``暴露门控评估.csv``：一行 = 某臂 × 口径 × 分组（含"每日净增量"一阶代理与两条对照臂）
- ``暴露门控评估_按折.csv``：同样的指标按折拆开（稳健性）

**本入口不改交易链路**：它是"要不要接引擎"的筛选门禁，不是净值对比。
"""

import argparse
import glob
import sys
from pathlib import Path
from typing import List

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lazybull.risk.terminal_loss.exposure_gate import (  # noqa: E402
    ARM_LABELS,
    ARMS,
    ExposureGateConfig,
    RollingGateConfig,
    export_exposure_table,
    read_ledger_frames,
    run_exposure_gate,
    run_rolling_gate,
    to_chinese_tables,
    write_gate_outputs,
)

ARM_BY_LABEL = {label: key for key, label in ARM_LABELS.items()}


def _arm_label(arm: str, args: argparse.Namespace) -> str:
    """与 run_*_gate 内部保持一致的臂标签（固定口径用内置标签，滚动口径加窗口后缀）。"""
    if args.threshold_mode == "rolling":
        return f"{ARM_LABELS[arm]}-滚动{args.window_days}日"
    return ARM_LABELS[arm]


def _load_trading_days(args: argparse.Namespace, daily: pd.DataFrame) -> List[str]:
    """加载判定表区间内的交易日历（用于导出系数表时补齐台账缺口）。"""
    from src.lazybull.risk.terminal_loss.dataset import load_trade_calendar

    start, end = str(daily["date"].min()), str(daily["date"].max())
    days = [str(day) for day in load_trade_calendar(args.calendar_root, start, end)]
    logger.info(f"交易日历: {start}~{end} 共 {len(days)} 日（用于补齐台账缺口）")
    return days


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="terminal_loss E2 暴露门控校准与评估（P2-2）")
    parser.add_argument(
        "--batch",
        default=None,
        help="WF 批次目录名（如 wf_batch_20260914_084247）；自动扫描 <batch>/policy_sidecar*/风险台账.csv",
    )
    parser.add_argument(
        "--ledger",
        nargs="*",
        default=None,
        help="显式指定风险台账 CSV（覆盖 --batch 扫描）",
    )
    parser.add_argument("--data-root", default="data/walk_forward/batches", help="批次根目录")
    parser.add_argument(
        "--threshold-mode",
        default="fixed",
        choices=["fixed", "rolling"],
        help="阈值口径：fixed=固定校准段阈值；rolling=滚动分位（阈值随模型水平漂移自归一）",
    )
    parser.add_argument(
        "--window-days", type=int, default=250, help="rolling 口径的滚动窗口长度（交易日）"
    )
    parser.add_argument(
        "--min-window-days", type=int, default=60, help="rolling 口径窗口内最少交易日"
    )
    parser.add_argument("--calibration-start", default=None, help="校准段起点（YYYYMMDD，fixed 口径必填）")
    parser.add_argument("--calibration-end", default=None, help="校准段终点（YYYYMMDD，fixed 口径必填）")
    parser.add_argument(
        "--eval-start", default=None, help="评估段起点（默认=校准段终点之后首个交易日，由数据决定）"
    )
    parser.add_argument("--eval-end", default="20991231", help="评估段终点（默认 20991231）")
    parser.add_argument(
        "--arms",
        nargs="*",
        default=list(ARMS),
        choices=list(ARMS),
        help="要评估的臂（combined=E2；score=纯模型分数对照；regime=纯 regime 对照）",
    )
    parser.add_argument("--regime-quantile", type=float, default=2.0 / 3.0, help="regime 阈值分位")
    parser.add_argument(
        "--score-quantile", type=float, default=0.5, help="得分阈值分位（层内，combined 臂）"
    )
    parser.add_argument("--de-exposure", type=float, default=0.5, help="触发日目标暴露系数 λ")
    parser.add_argument("--cost-bps", type=float, default=15.0, help="单边成本（基点）")
    parser.add_argument(
        "--min-layer-days", type=int, default=20, help="层内校准日数下限（低于即拒绝校准）"
    )
    parser.add_argument("--out-dir", default=None, help="输出目录（默认 <batch>/暴露门控E2）")
    parser.add_argument(
        "--calendar-root",
        default="data",
        help="交易日历所在数据根目录（导出系数表时用于补齐台账缺口，默认 data）",
    )
    parser.add_argument(
        "--export-table",
        action="store_true",
        default=False,
        help=(
            "额外导出引擎可读的两列暴露系数表（每个臂一份，`暴露系数表_<臂>.csv`），"
            "供 walk_forward.py --exposure-table 做 P2-3 shadow 回测"
        ),
    )
    return parser.parse_args()


def resolve_ledgers(args: argparse.Namespace) -> List[Path]:
    """定位风险台账：显式 --ledger 优先，否则按 --batch 扫描 policy_sidecar* 子目录。"""
    if args.ledger:
        paths = [Path(p) for p in args.ledger]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"台账文件不存在: {missing}")
        return paths
    if not args.batch:
        raise SystemExit("必须提供 --batch 或 --ledger 之一")
    pattern = str(Path(args.data_root) / args.batch / "policy_sidecar*" / "风险台账.csv")
    found = sorted(glob.glob(pattern))
    if not found:
        raise SystemExit(
            f"{pattern} 未找到风险台账；请先运行 "
            f"scripts/batch/batch_policy_sidecar.ps1 -Batch {args.batch}"
        )
    return [Path(p) for p in found]


def main() -> None:
    args = parse_args()
    ledger_paths = resolve_ledgers(args)
    logger.info(f"读取风险台账 {len(ledger_paths)} 个: {[p.parent.name for p in ledger_paths]}")

    if args.threshold_mode == "fixed":
        if not args.calibration_start or not args.calibration_end:
            raise SystemExit("--threshold-mode fixed 必须提供 --calibration-start/--calibration-end")
        eval_start = args.eval_start
        if eval_start is None:
            # 未显式给起点时，用校准段终点加一天，交由数据自然裁剪（不做交易日历外推）
            eval_start = str(int(args.calibration_end) + 1)
            logger.warning(f"未指定 --eval-start，按校准段终点次日 {eval_start} 处理")
        if not args.calibration_end < eval_start:
            raise SystemExit(
                f"校准段终点 {args.calibration_end} 必须早于评估段起点 {eval_start}（禁止重叠）"
            )
        eval_segment = (eval_start, args.eval_end)
        ledger = read_ledger_frames(ledger_paths)
        configs = [
            ExposureGateConfig(
                arm=arm,
                regime_quantile=args.regime_quantile,
                score_quantile=args.score_quantile,
                de_exposure_multiplier=args.de_exposure,
                cost_bps=args.cost_bps,
                min_layer_days=args.min_layer_days,
            )
            for arm in args.arms
        ]
        calibration, daily, (overall, by_fold) = run_exposure_gate(
            ledger,
            configs,
            (args.calibration_start, args.calibration_end),
            eval_segment,
        )
        out_suffix = ""
    else:
        if not args.eval_start:
            raise SystemExit("--threshold-mode rolling 必须提供 --eval-start（无校准段可推导）")
        eval_segment = (args.eval_start, args.eval_end)
        ledger = read_ledger_frames(ledger_paths)
        configs = [
            RollingGateConfig(
                arm=arm,
                window_days=args.window_days,
                regime_quantile=args.regime_quantile,
                score_quantile=args.score_quantile,
                de_exposure_multiplier=args.de_exposure,
                cost_bps=args.cost_bps,
                min_window_days=args.min_window_days,
                label=f"{ARM_LABELS[arm]}-滚动{args.window_days}日",
            )
            for arm in args.arms
        ]
        calibration, daily, (overall, by_fold) = run_rolling_gate(ledger, configs, eval_segment)
        out_suffix = f"_滚动{args.window_days}日"

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(args.data_root) / args.batch / f"暴露门控E2{out_suffix}"
        if args.batch
        else Path(".") / f"暴露门控E2{out_suffix}"
    )
    paths = write_gate_outputs(out_dir, calibration, daily, (overall, by_fold))
    logger.info(f"产物目录: {out_dir}")
    for name, path in paths.items():
        logger.info(f"  {name}: {path.name}")
    if args.export_table:
        mode_tag = "rolling" if args.threshold_mode == "rolling" else "fixed"
        trading_days = _load_trading_days(args, daily)
        for arm in args.arms:
            label = daily.loc[daily["arm"] == _arm_label(arm, args), "arm"].unique()
            if label.size != 1:
                raise SystemExit(f"臂 {arm} 的标签不唯一: {label.tolist()}")
            suffix = f"_{args.window_days}d" if args.threshold_mode == "rolling" else ""
            table_path = export_exposure_table(
                daily,
                out_dir / f"exposure_table_{arm}_{mode_tag}{suffix}.csv",
                arm_label=str(label[0]),
                trading_days=trading_days,
            )
            logger.info(
                f"  暴露系数表: {table_path.name}"
                f"（{len(pd.read_csv(table_path, encoding='utf-8-sig'))} 日，含缺口顺延；文件名用 ASCII"
                f"避免 PowerShell 传参转码）"
            )

    tables = to_chinese_tables(calibration, daily, (overall, by_fold))
    evaluated = tables["eval"]
    cols = ["臂", "口径", "分组", "交易日数", "触发日数", "触发占比", "收益项", "成本项", "净增量（日均口径）"]
    print("\n=== 暴露门控评估（一阶代理；正数=降暴露带来净增量）===")
    print(evaluated[cols].round(5).to_string(index=False))
    cols_kpi = ["臂", "口径", "分组", "触发日平均加权收益", "未触发日平均加权收益",
                "触发日亏损日频率", "未触发日亏损日频率", "触发日事件率", "未触发日事件率"]
    print("\n=== 触发性与风险改善 ===")
    print(evaluated[cols_kpi].round(4).to_string(index=False))
    print("\n=== 校准阈值 ===")
    print(tables["calibration"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
