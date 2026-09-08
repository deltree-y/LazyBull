# -*- coding: utf-8 -*-
"""期末异常亏损风险模型训练入口（第一阶段：标签与离线模型）

按 docs/plans/terminal_loss_risk_model_plan.md 8.1 实施：构建独立标签
（不写回 cs_train/cs_infer）→ 冻结 manifest 特征矩阵 → Train/ES 分割
（label_end_date 隔离）→ 二分类训练（logloss 早停）→ 概率质量报告
（分 h、h × σ 分位、事件率、覆盖分布）。

内存与抽样契约（方案第 6 节）：全区间全网格标签约 2 亿行不可行，脚本按
日期分块构建标签并立即关联特征；随后执行预登记抽样——每组 (股票, 日)
确定性抽取 n_h 个期限（跨组覆盖全部 h）、按交易日位置等距抽日期
（every_n），抽样参数落入模型元数据。抽样保留自然事件率，覆盖分布
随报告落盘。

已知限制（首轮登记）：pct_* 百分位母截面取自 cs_train（其行已按
y_ret_* 标签有效性过滤，分母较完整日截面窄约 5%~10%）；训练与后续
推理将使用同一母截面口径保证一致性，二阶段用持仓快照复核该偏差。

用法示例：
python scripts/train_terminal_risk_model.py \
    --start-date 20190102 --end-date 20260630 \
    --train-start 20190102 --train-end 20231231 \
    --es-start 20240102 --es-end 20240628

产物保存双模式：默认走 ModelRegistry 版本化（每次训练注册新版本 v{N}，
不覆盖历史）；--fixed-name 为固定文件名覆盖模式，供 WF 折目录等研究场景
（batch_terminal_risk_wf.ps1 已显式使用该开关）。
"""

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
from loguru import logger

# GPU 训练 + CPU numpy 输入预测时的数据结构回退告警（与主模型 walk_forward/runner.py 同处理）
warnings.filterwarnings("ignore", category=UserWarning, message=".*mismatched devices.*")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lazybull.factors.risk.volatility_factors import (  # noqa: E402
    compute_sigma_daily_panel,
)
from src.lazybull.risk.terminal_loss import (  # noqa: E402
    BASE_FEATURES,
    TERMINAL_LOSS_FEATURES,
    TERMINAL_LOSS_MODEL_TYPE,
    StageSpec,
    TerminalLossLabelConfig,
    TerminalLossModel,
    TerminalLossModelConfig,
    TerminalLossTrainConfig,
    build_performance_metrics,
    build_terminal_loss_labels,
    build_training_matrix,
    empty_training_matrix,
    evaluate_probability_quality,
    load_clean_daily_panels,
    load_cs_train_days,
    load_trade_calendar,
    save_flat_artifacts,
    save_versioned_artifacts,
    split_stages_with_label_isolation,
    subsample_dates,
    subsample_h_per_group,
    train_terminal_loss_model,
)
from src.lazybull.risk.terminal_loss.labels import (  # noqa: E402
    summarize_label_coverage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="期末异常亏损风险模型训练（第一阶段）")
    parser.add_argument("--data-root", default="data", help="数据根目录")
    parser.add_argument(
        "--output-dir", default="data/models/terminal_loss", help="模型与报告输出目录"
    )
    parser.add_argument(
        "--start-date", required=True, help="数据起点 YYYYMMDD（自动前移 sigma 预热）"
    )
    parser.add_argument("--end-date", required=True, help="数据终点 YYYYMMDD")
    parser.add_argument("--train-start", required=True, help="Train 段起点 YYYYMMDD")
    parser.add_argument("--train-end", required=True, help="Train 段终点 YYYYMMDD")
    parser.add_argument("--es-start", required=True, help="ES 段起点 YYYYMMDD")
    parser.add_argument("--es-end", required=True, help="ES 段终点 YYYYMMDD")
    parser.add_argument("--k", type=float, default=1.0, help="异常亏损波动倍数（训练前固定）")
    parser.add_argument("--h-max", type=int, default=20, help="期限网格上限")
    parser.add_argument("--sigma-window", type=int, default=20, help="sigma 日历窗口")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=3, help="树最大深度（消融实验位）")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="XGBoost 训练设备（默认 cuda，与主模型一致；GPU 不稳定时可切 cpu）",
    )
    parser.add_argument(
        "--chunk-days", type=int, default=50, help="标签构建分块的交易日数（内存控制）"
    )
    parser.add_argument(
        "--h-per-group", type=int, default=2, help="每组 (股票,日) 抽取的期限数（预登记抽样）"
    )
    parser.add_argument(
        "--every-n-days", type=int, default=3, help="交易日等距抽样间隔（预登记抽样，1=不抽）"
    )
    parser.add_argument(
        "--fixed-name",
        action="store_true",
        help="固定文件名覆盖模式（WF 折目录研究用）；"
        "默认为 ModelRegistry 版本化保存（每次训练新增版本，不覆盖）",
    )
    return parser.parse_args()


def build_matrix_chunked(args, label_config, open_panel, sigma_panel, limit_panel, calendar):
    """分块构建训练矩阵：每块标签构建后立即关联特征并抽样，控制峰值内存。

    块面板向后多切 h_max+1 日，使块内所有 h 的标签端点落在真实数据上
    （非 immature）；数据真实末端的 immature 语义保持正确。
    """
    feature_dates = [d for d in calendar if args.train_start <= d <= args.es_end]
    n = len(calendar)
    pos_of = {d: i for i, d in enumerate(calendar)}
    matrix_pieces = []
    coverage_counter = {}
    valid_event_sum, valid_count = 0, 0

    for c0 in range(0, len(feature_dates), args.chunk_days):
        chunk_dates = feature_dates[c0 : c0 + args.chunk_days]
        i0, i1 = pos_of[chunk_dates[0]], pos_of[chunk_dates[-1]]
        i_end = min(i1 + 1 + label_config.h_max + 1, n)
        sub_cal = calendar[i0:i_end]
        labels = build_terminal_loss_labels(
            open_panel.iloc[i0:i_end],
            sigma_panel.iloc[i0:i_end],
            label_config,
            limit_down_panel=limit_panel.iloc[i0:i_end],
        )
        # 块面板尾部的端点区日期也会作为 T 生成（短 h 可 valid），
        # 它们属于下一块的块日期：过滤避免误报特征缺失与重复纳入
        labels = labels[labels["trade_date"].isin(set(chunk_dates))]
        # 覆盖统计累计（块内按状态计数；valid 组累计事件数）
        counts = labels.groupby(["h", "label_status"]).size()
        for (h, status), cnt in counts.items():
            coverage_counter[(h, status)] = coverage_counter.get((h, status), 0) + int(cnt)
        valid = labels[labels["label_status"] == "valid"]
        valid_event_sum += int(valid["loss_label"].sum())
        valid_count += len(valid)

        features_by_date = load_cs_train_days(args.data_root, chunk_dates, BASE_FEATURES)
        piece = build_training_matrix(labels, features_by_date, sigma_panel)
        piece = subsample_h_per_group(piece, n_h=args.h_per_group)
        matrix_pieces.append(piece)
        logger.info(
            f"分块 [{chunk_dates[0]},{chunk_dates[-1]}]: 标签 {len(labels)} 行 → "
            f"矩阵 {len(piece)} 行（含 h 抽样 1/{args.h_per_group}）"
        )

    # 空块（如 ES 终点日全部 immature）由 empty_training_matrix 保持数值
    # dtype，避免 concat 把整列提升为 object；全空时走空矩阵由主流程报错
    matrix = (
        pd.concat(matrix_pieces, ignore_index=True) if matrix_pieces else empty_training_matrix()
    )
    matrix = subsample_dates(matrix, every_n=args.every_n_days)
    coverage_df = (
        pd.Series(coverage_counter, name="count").rename_axis(["h", "label_status"]).reset_index()
    )
    logger.info(
        f"训练矩阵（含日期抽样 1/{args.every_n_days}）: {len(matrix)} 行；"
        f"全网格 valid 事件率 {valid_event_sum / max(valid_count, 1):.4f}"
    )
    return matrix, coverage_df, valid_event_sum, valid_count


def main() -> int:
    args = parse_args()
    label_config = TerminalLossLabelConfig(
        h_max=args.h_max, sigma_window=args.sigma_window, loss_sigma_multiple=args.k
    )
    train_config = TerminalLossTrainConfig(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        device=args.device,
    )

    # sigma 预热：数据加载起点前移 sigma_window+2 个交易日
    pre_calendar = load_trade_calendar(args.data_root, "19900101", args.start_date)
    warmup_dates = pre_calendar[-(args.sigma_window + 2) :]
    data_start = warmup_dates[0] if warmup_dates else args.start_date
    logger.info(
        f"数据区间 [{data_start},{args.end_date}]（含 sigma 预热前移），"
        f"标签窗口 k={args.k}, h≤{args.h_max}, σ窗口={args.sigma_window}"
    )

    open_panel, close_panel, limit_panel, calendar = load_clean_daily_panels(
        args.data_root, data_start, args.end_date
    )
    close_long = close_panel.stack().rename("close_adj").reset_index()
    sigma_panel = compute_sigma_daily_panel(close_long, calendar, window=args.sigma_window)

    matrix, coverage_df, valid_event_sum, valid_count = build_matrix_chunked(
        args, label_config, open_panel, sigma_panel, limit_panel, calendar
    )
    if matrix.empty:
        logger.error("训练矩阵为空，终止")
        return 1

    split = split_stages_with_label_isolation(
        matrix,
        [
            StageSpec("train", args.train_start, args.train_end),
            StageSpec("es", args.es_start, args.es_end),
        ],
    )
    train_df, es_df = split.stages["train"], split.stages["es"]
    if train_df.empty or es_df.empty:
        logger.error(f"分割后 Train={len(train_df)} 行 / ES={len(es_df)} 行，存在空段，终止")
        return 1

    result = train_terminal_loss_model(
        train_df,
        es_df,
        TERMINAL_LOSS_FEATURES,
        train_config=train_config,
        label_config=label_config,
        stage_dates={
            "train": (args.train_start, args.train_end),
            "es": (args.es_start, args.es_end),
        },
    )

    # 概率质量报告：ES 段（早停参考）+ Train 段（过拟合差距诊断）
    es_report = evaluate_probability_quality(
        es_df["loss_label"].to_numpy(),
        result.classifier.predict_proba(es_df[TERMINAL_LOSS_FEATURES])[:, 1],
        es_df["h"].to_numpy(),
        sigma_values=es_df["sigma_daily_20"].to_numpy(),
    )
    train_report = evaluate_probability_quality(
        train_df["loss_label"].to_numpy(),
        result.classifier.predict_proba(train_df[TERMINAL_LOSS_FEATURES])[:, 1],
        train_df["h"].to_numpy(),
    )

    # 保存模型与报告
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_config = TerminalLossModelConfig(
        task_id=label_config.task_id,
        feature_names=TERMINAL_LOSS_FEATURES,
        train_config=result.train_config,
        label_config=result.label_config,
        metadata={
            **result.to_metadata(),
            "sampling": {
                "chunk_days": args.chunk_days,
                "h_per_group": args.h_per_group,
                "every_n_days": args.every_n_days,
            },
            "full_grid_valid_event_rate": valid_event_sum / max(valid_count, 1),
            "pct_cross_section_source": "cs_train（y_ret 标签有效域，已知限制登记）",
            "known_limitations": ["pct_* 母截面为 cs_train 过滤后域，分母较完整日截面窄约 5%~10%"],
        },
    )
    model = TerminalLossModel(model_config, result.classifier)
    report_payload = {
        "es": _report_to_json(es_report),
        "train": _report_to_json(train_report),
        "label_coverage": coverage_df.to_dict(orient="records"),
        "isolation_dropped": split.isolation_dropped,
    }
    if args.fixed_name:
        save_flat_artifacts(
            out_dir,
            model,
            report_payload,
            es_report["calibration_by_h_sigma"],
            coverage_df,
        )
    else:
        version = save_versioned_artifacts(
            out_dir,
            model,
            train_start_date=args.train_start,
            train_end_date=args.train_end,
            n_samples=int(model_config.metadata["n_train"]),
            train_params=model_config.metadata,
            performance_metrics=build_performance_metrics(es_report, train_report),
            report_payload=report_payload,
            calibration_df=es_report["calibration_by_h_sigma"],
            coverage_df=coverage_df,
        )
        logger.info(f"已注册 {TERMINAL_LOSS_MODEL_TYPE} v{version}（每次训练新增版本，不覆盖）")
    logger.info(
        f"ES 段概率质量: logloss={es_report['logloss']:.4f}, "
        f"brier={es_report['brier']:.4f}, pr_auc={es_report['pr_auc']:.4f}, "
        f"事件率={es_report['event_rate']:.4f}"
    )
    logger.info(f"输出目录: {out_dir}")
    return 0


def _report_to_json(report: dict) -> dict:
    """报告 JSON 序列化（DataFrame → records）。"""
    out = {}
    for key, value in report.items():
        if hasattr(value, "to_dict"):
            out[key] = value.to_dict(orient="records")
        else:
            out[key] = value
    return out


if __name__ == "__main__":
    raise SystemExit(main())
