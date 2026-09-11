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

pct_* 分母契约（方案 4.4）：分母取自 ``clean/daily`` 全量化重建的
**标签过滤前完整同日母截面**（``mother_section``，四个基列与特征流水线
同一实现），并在交集上与 ``cs_train`` 同名列逐值校验（超容差即报错）。
不再使用 cs_train（y_ret 标签有效域）当分母。

报告门禁（方案 8.1）：除 ES/Train 概率质量外，报告含
- 各 label_status 按 h 的占比（含 execution_blocked 计数）
- 缺失组 vs valid 组的代理画像与当日截面分位条件事件率
- endpoint_delayed 敏感性（E 之后首个有报价开盘价重算标签）
- 按预登记阈值（1%）判定的"是否必须补做敏感性"结论

产物落盘：始终经 ModelRegistry 版本化注册（每次训练新增 v{N}，永不覆盖，
研究折目录同样适用）；``--fixed-name`` 额外写固定名别名供既有工具读取。

用法示例：
python scripts/train_terminal_risk_model.py \
    --start-date 20190102 --end-date 20260630 \
    --train-start 20190102 --train-end 20231231 \
    --es-start 20240102 --es-end 20240628
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from loguru import logger

# GPU 训练 + CPU numpy 输入预测时的数据结构回退告警（与主模型 walk_forward/runner.py 同处理）
warnings.filterwarnings("ignore", category=UserWarning, message=".*mismatched devices.*")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lazybull.factors.risk.volatility_factors import (  # noqa: E402
    compute_sigma_daily_panel,
)
from src.lazybull.risk.terminal_loss import (  # noqa: E402
    AUDIT_PROXY_COLUMNS,
    BASE_FEATURES,
    TERMINAL_LOSS_FEATURES,
    TERMINAL_LOSS_MODEL_TYPE,
    LabelCoverageAccumulator,
    MotherSectionCache,
    ProxyProfileAccumulator,
    StageSpec,
    TerminalLossLabelConfig,
    TerminalLossModel,
    TerminalLossModelConfig,
    TerminalLossTrainConfig,
    build_performance_metrics,
    build_terminal_loss_labels,
    build_training_matrix,
    coverage_audit_required,
    delayed_endpoint_sensitivity,
    empty_training_matrix,
    evaluate_probability_quality,
    load_clean_daily_panels,
    load_cs_train_days,
    load_trade_calendar,
    save_terminal_loss_artifacts,
    split_stages_with_label_isolation,
    subsample_dates,
    subsample_h_per_group,
    train_terminal_loss_model,
    validate_mother_section_against_cs_train,
)
from src.lazybull.risk.terminal_loss.coverage_audit import (  # noqa: E402
    DELAYED_SEARCH_WINDOW_DAYS,
)
from src.lazybull.risk.terminal_loss.mother_section import (  # noqa: E402
    assert_mother_section_validation_ok,
    merge_mother_section_validation,
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
    parser.add_argument("--learning-rate", type=float, default=0.03, help="学习率（消融实验位）")
    parser.add_argument(
        "--early-stopping-rounds", type=int, default=30, help="ES 段 logloss 早停轮数"
    )
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--eval-metric",
        default="logloss",
        choices=["logloss", "rank_ic_daily"],
        help="早停指标：logloss（概率校准口径，默认）或 rank_ic_daily"
        "（逐日截面 Spearman 均值，与门禁 lift 同向）；两种口径是不同的超参签名，"
        "不得并入同一组比较",
    )
    # min_child_weight / scale_pos_weight 不暴露 CLI：
    # min_child_weight=1 与样本权重 1/期限网格大小 绑定（正则尺度策略 A 的设计
    # 不变量，单一 (股票,日) 组无法独自成叶）；scale_pos_weight 会破坏自然事件率
    # 口径（抽样保留自然事件率、概率校准优先）。
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
        help="额外写一套固定名别名（WF 折目录研究用）；"
        "模型始终经 ModelRegistry 版本化保存（每次训练新增版本，不覆盖）",
    )
    parser.add_argument(
        "--no-es-predictions",
        action="store_true",
        help="不落盘 ES 逐行预测（默认落盘，供门禁区间重判 block_stats 使用）",
    )
    return parser.parse_args()


def build_matrix_chunked(args, label_config, open_panel, sigma_panel, limit_panel, calendar):
    """分块构建训练矩阵：每块标签构建后立即关联特征并抽样，控制峰值内存。

    块面板向后多切 h_max+1 日（使块内所有 h 的标签端点落在真实数据上，
    非 immature），再多切 DELAYED_SEARCH_WINDOW_DAYS 日供 endpoint_delayed
    敏感性在块内完成（方案 2.4 第 6 条）。

    同时累计三项报告门禁输入（全部可跨块累加）：
    - 标签覆盖计数与 valid 组事件率（LabelCoverageAccumulator）
    - 缺失/valid 组的代理画像与当日截面分位条件事件率（ProxyProfileAccumulator）
    - endpoint_delayed 敏感性（改用 E 之后首个有报价开盘价重算标签）
    """
    feature_dates = [d for d in calendar if args.train_start <= d <= args.es_end]
    n = len(calendar)
    pos_of = {d: i for i, d in enumerate(calendar)}
    matrix_pieces = []
    coverage = LabelCoverageAccumulator()
    profile = ProxyProfileAccumulator(proxies=tuple(AUDIT_PROXY_COLUMNS))
    delayed_totals: Dict[str, int] = {}
    mother_validation_parts: List[Dict[str, Any]] = []
    # 母截面历史窗口按折准备一次（含 7 个月预热），各分块只切片：
    # 逐块重建窗口会重复加载分区与重算风控因子（实测折耗时 3~4 倍）。
    mother_cache = MotherSectionCache(
        args.data_root, feature_dates[0], feature_dates[-1]
    )

    for c0 in range(0, len(feature_dates), args.chunk_days):
        chunk_dates = feature_dates[c0 : c0 + args.chunk_days]
        i0, i1 = pos_of[chunk_dates[0]], pos_of[chunk_dates[-1]]
        i_end = min(i1 + 1 + label_config.h_max + 1 + DELAYED_SEARCH_WINDOW_DAYS, n)
        labels = build_terminal_loss_labels(
            open_panel.iloc[i0:i_end],
            sigma_panel.iloc[i0:i_end],
            label_config,
            limit_down_panel=limit_panel.iloc[i0:i_end],
        )
        # 块面板尾部的端点区日期也会作为 T 生成（短 h 可 valid），
        # 它们属于下一块的块日期：过滤避免误报特征缺失与重复纳入
        labels = labels[labels["trade_date"].isin(set(chunk_dates))]
        coverage.add(labels)
        audit_labels = labels.rename(columns={"sigma_at_t": "sigma_daily_20"})

        features_by_date = load_cs_train_days(args.data_root, chunk_dates, BASE_FEATURES)
        # pct_* 分母：标签过滤前的完整同日母截面（方案 4.4），
        # 与特征流水线同一实现，并在交集上做逐值一致性校验
        mother_by_date = mother_cache.build(chunk_dates)
        # 逐块只统计，窗口级汇总后再判定（块内只有 50 个交易日，块内占比判定
        # 会把"3 个修订日"这类极小样本误判为实现漂移）
        mother_validation_parts.append(
            validate_mother_section_against_cs_train(args.data_root, mother_by_date, chunk_dates)
        )
        proxy_long = pd.concat(
            [df.assign(trade_date=d) for d, df in mother_by_date.items()], ignore_index=True
        )
        profile.add(audit_labels, proxy_long)
        # delayed 敏感性读标签表自带的 sigma_at_t（母截面/审计帧用 sigma_daily_20 命名）
        _accumulate_delayed(
            delayed_totals,
            delayed_endpoint_sensitivity(
                labels,
                open_panel.iloc[i0:i_end],
                label_config,
                max_delay_days=DELAYED_SEARCH_WINDOW_DAYS,
            ),
        )

        piece = build_training_matrix(labels, features_by_date, sigma_panel, mother_by_date)
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
    coverage_df = coverage.to_frame()
    mother_validation = assert_mother_section_validation_ok(
        merge_mother_section_validation(mother_validation_parts)
    )
    logger.info(
        f"训练矩阵（含日期抽样 1/{args.every_n_days}）: {len(matrix)} 行；"
        f"全网格 valid 事件率 {coverage.valid_event_sum / max(coverage.valid_count, 1):.4f}"
    )
    status_share = coverage.status_share()
    audit = {
        "status_share": status_share.to_dict(orient="records"),
        "proxy_profile": profile.profile_table().to_dict(orient="records"),
        "proxy_conditional_event_rate": profile.conditional_table().to_dict(orient="records"),
        "implied_missing_event_rate": profile.implied_missing_event_rate().to_dict(
            orient="records"
        ),
        "delayed_endpoint": delayed_totals,
        "required": coverage_audit_required(status_share),
        "mother_section_validation": mother_validation,
    }
    return {
        "matrix": matrix,
        "coverage_df": coverage_df,
        "valid_event_sum": coverage.valid_event_sum,
        "valid_count": coverage.valid_count,
        "coverage_audit": audit,
    }


def _accumulate_delayed(acc: Dict[str, int], result: Dict[str, Any]) -> None:
    """累计 endpoint_delayed 敏感性计数（跨块求和）。"""
    for key in (
        "n_endpoint_missing",
        "n_evaluated",
        "n_t1_missing",
        "n_delay_unavailable",
        "n_event",
    ):
        acc[key] = acc.get(key, 0) + int(result["total"].get(key, 0))


def main() -> int:
    args = parse_args()
    label_config = TerminalLossLabelConfig(
        h_max=args.h_max, sigma_window=args.sigma_window, loss_sigma_multiple=args.k
    )
    train_config = TerminalLossTrainConfig(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        early_stopping_rounds=args.early_stopping_rounds,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        random_state=args.random_state,
        eval_metric=args.eval_metric,
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

    built = build_matrix_chunked(args, label_config, open_panel, sigma_panel, limit_panel, calendar)
    matrix = built["matrix"]
    coverage_df = built["coverage_df"]
    coverage_audit = built["coverage_audit"]
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
    es_prob = result.classifier.predict_proba(es_df[TERMINAL_LOSS_FEATURES])[:, 1]
    es_report = evaluate_probability_quality(
        es_df["loss_label"].to_numpy(),
        es_prob,
        es_df["h"].to_numpy(),
        sigma_values=es_df["sigma_daily_20"].to_numpy(),
    )
    train_report = evaluate_probability_quality(
        train_df["loss_label"].to_numpy(),
        result.classifier.predict_proba(train_df[TERMINAL_LOSS_FEATURES])[:, 1],
        train_df["h"].to_numpy(),
    )
    # ES 逐行预测落盘：门禁区间重判（block_stats）需要逐行预测，
    # 只有聚合指标无法重采样；--no-es-predictions 可跳过
    es_predictions = None
    if not args.no_es_predictions:
        es_predictions = pd.DataFrame(
            {
                "trade_date": es_df["trade_date"].to_numpy(),
                "ts_code": es_df["ts_code"].to_numpy(),
                "h": es_df["h"].to_numpy(),
                "loss_label": es_df["loss_label"].to_numpy(),
                "p_loss": es_prob,
            }
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
            "full_grid_valid_event_rate": built["valid_event_sum"] / max(built["valid_count"], 1),
            "pct_cross_section_source": (
                "clean/daily 完整同日母截面（标签过滤前，mother_section 重建，"
                "与特征流水线同一实现的四个基列）"
            ),
            "mother_section_validation": coverage_audit.get("mother_section_validation", {}),
            "coverage_audit_required": coverage_audit.get("required", {}),
            "es_predictions": None if es_predictions is None else int(len(es_predictions)),
        },
    )
    model = TerminalLossModel(model_config, result.classifier)
    report_payload = {
        "es": _report_to_json(es_report),
        "train": _report_to_json(train_report),
        "label_coverage": coverage_df.to_dict(orient="records"),
        "isolation_dropped": split.isolation_dropped,
        "coverage_audit": coverage_audit,
    }
    # 统一落盘：总是注册制版本化（v{N} 永不覆盖）；--fixed-name 额外写固定名
    # 别名供 summarize_terminal_risk_wf.py 等既有工具零改动读取
    saved = save_terminal_loss_artifacts(
        out_dir,
        model,
        fixed_name=bool(args.fixed_name),
        train_start_date=args.train_start,
        train_end_date=args.train_end,
        n_samples=int(model_config.metadata["n_train"]),
        train_params=model_config.metadata,
        performance_metrics=build_performance_metrics(es_report, train_report),
        report_payload=report_payload,
        calibration_df=es_report["calibration_by_h_sigma"],
        coverage_df=coverage_df,
        es_predictions=es_predictions,
    )
    logger.info(
        f"已注册 {TERMINAL_LOSS_MODEL_TYPE} {saved['version_str']}"
        f"（每次训练新增版本，不覆盖）" + ("；已写固定名别名" if saved["fixed_name"] else "")
    )
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
