#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Walk-forward CLI 参数构建与解析。"""

import argparse
from typing import List, Optional

from loguru import logger

from ...signals.downside_penalty import DOWNSIDE_PENALTY_COLUMNS, DOWNSIDE_PENALTY_GRID
from .training_core import (
    SEED_ENSEMBLE_KEEP_MIN_MODELS,
    SEED_ENSEMBLE_KEEP_TOP_RATIO,
)


def _normalize_selected_split_indices(raw_indices: Optional[List[int]]) -> List[int]:
    """规范化 split 下标列表：去重保序，且要求非负整数。"""
    if not raw_indices:
        return []

    normalized: List[int] = []
    seen = set()
    for raw_index in raw_indices:
        split_index = int(raw_index)
        if split_index < 0:
            raise ValueError(f"selected_split_indices 仅支持非负整数，收到: {split_index}")
        if split_index in seen:
            continue
        seen.add(split_index)
        normalized.append(split_index)
    return normalized


def build_walk_forward_parser() -> argparse.ArgumentParser:
    """构建 walk-forward 命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="Walk-forward 滚动训练")

    # Walk-forward 参数
    parser.add_argument("--split-count", type=int, required=True, help="切分数量（正整数）")
    parser.add_argument(
        "--final-date",
        type=str,
        required=True,
        help=(
            "最终日期，格式 YYYYMMDD。"
            "若启用部署训练，表示部署训练数据最后一天；"
            "若禁用部署训练，表示最后一个 split 测试结束日期"
        ),
    )
    parser.add_argument("--train-window-years", type=int, default=5, help="训练窗口年数，默认 5")
    parser.add_argument("--test-window-months", type=int, default=11, help="测试窗口月数，默认 11")
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="训练数据内部验证集比例，默认 0.1"
    )
    parser.add_argument(
        "--selected-split-indices",
        type=int,
        nargs="*",
        default=[],
        help="仅训练指定 split 下标（如 0 4 5 7 9）；留空表示训练全部 split",
    )

    # 数据参数
    parser.add_argument(
        "--label-column", type=str, default="y_ret_5", help="标签列名，默认 y_ret_5"
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        choices=["y_ret_5", "y_ret_10", "y_ret_20", "neu_y_ret_5", "neu_y_ret_10", "neu_y_ret_20"],
        help="标签选择（y_ret_5|y_ret_10|y_ret_20|neu_y_ret_5|neu_y_ret_10|neu_y_ret_20），默认 y_ret_5。优先级高于 --label-column",
    )
    parser.add_argument(
        "--neutral-label-blend-weight",
        type=float,
        default=0.0,
        help="原始收益在行业中性混合标签中的权重，范围 0~1，默认 0（保持原标签）",
    )

    # 任务类型和标签变换参数
    parser.add_argument(
        "--task",
        type=str,
        default="regression",
        choices=["regression", "classification"],
        help="任务类型（regression|classification），默认 regression",
    )
    parser.add_argument(
        "--label-transform",
        type=str,
        default="raw",
        choices=["raw", "cs_zscore"],
        help="标签变换方式（raw|cs_zscore），默认 raw。仅对 regression 任务生效",
    )
    parser.add_argument(
        "--winsorize-p",
        type=float,
        default=0.01,
        help="winsorize 参数（截断比例），默认 0.01（截断上下1%%）。仅当 label-transform=cs_zscore 时生效",
    )
    parser.add_argument(
        "--pos-quantile",
        type=float,
        default=None,
        help="分类任务正类百分比阈值（例如 0.2 表示 Top20%%），与 pos-topk 二选一",
    )
    parser.add_argument(
        "--pos-topk",
        type=int,
        default=None,
        help="分类任务正类数量阈值（例如 300 表示每日 Top300），与 pos-quantile 二选一，优先级更高",
    )
    parser.add_argument(
        "--scale-pos-weight",
        type=float,
        default=None,
        help="分类任务正类权重，None 表示自动计算为 neg/pos（默认）",
    )

    # 算法选择
    parser.add_argument(
        "--algorithm",
        type=str,
        default="xgboost",
        choices=["xgboost", "lightgbm"],
        help="训练算法（xgboost|lightgbm），默认 xgboost",
    )

    # 模型参数
    parser.add_argument("--n-estimators", type=int, default=200, help="树的数量，默认 200")
    parser.add_argument(
        "--max-depth", type=int, default=5, help="树的最大深度，默认 5（金融数据噪声大不宜过深）"
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=None,
        help="LightGBM 叶子数，默认 31。仅 LightGBM 有效，XGBoost 忽略此参数",
    )
    parser.add_argument("--learning-rate", type=float, default=0.05, help="学习率，默认 0.05")
    parser.add_argument("--subsample", type=float, default=0.8, help="样本采样比例，默认 0.8")
    parser.add_argument(
        "--colsample-bytree", type=float, default=0.8, help="特征采样比例，默认 0.8"
    )
    parser.add_argument(
        "--min-child-weight",
        type=int,
        default=100,
        help="叶节点最少样本权重和，防止过拟合，默认 100（金融数据建议 100-500）",
    )
    parser.add_argument(
        "--reg-alpha",
        type=float,
        default=0.05,
        help="L1 正则化系数，默认 0.05（建议范围 0.05-0.5）",
    )
    parser.add_argument(
        "--reg-lambda", type=float, default=1.0, help="L2 正则化系数，默认 1.0（建议范围 1.0-5.0）"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="节点分裂最小损失下降，默认 0.1（建议范围 0.0-1.0）",
    )
    parser.add_argument("--random-state", type=int, default=42, help="随机种子，默认 42")
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=200,
        help="早停轮数（验证集指标连续N轮不改善则停止），默认 200。设为 0 则禁用早停，使用固定 n_estimators",
    )
    parser.add_argument(
        "--early-stopping-metric",
        type=str,
        default="rank_ic",
        choices=["auto", "rank_ic", "rank_ic_daily"],
        help=(
            "早停监控指标：auto（mae/auc，默认指标）；rank_ic（整段 Spearman Rank IC）；"
            "rank_ic_daily（逐日截面 Spearman RankIC 均值，与 daily_rankic 评估口径一致，"
            "避免整段指标被单一事件期样本主导）。默认 rank_ic"
        ),
    )
    parser.add_argument(
        "--min-best-iteration",
        type=int,
        default=0,
        help=(
            "best_iteration 下限监控阈值，默认 0（禁用）。启用早停且 best_iteration 低于该值时"
            "告警并在 summary 中标记 best_iteration_floor_triggered，不改变模型行为"
        ),
    )
    parser.add_argument(
        "--oos-detail-metrics",
        action="store_true",
        default=False,
        help=(
            "启用每个 split 的 OOS 详细指标对比表（验证集 vs 测试集）。"
            "默认关闭，仅输出重点 TopK 面板与一行简报，减少日志噪音"
        ),
    )

    # rank-weight 参数：Top/Bottom K 样本增强权重
    parser.add_argument(
        "--rank-weight-enabled",
        action="store_true",
        default=True,
        help="启用 Top/Bottom K 样本权重增强（默认开启）",
    )
    parser.add_argument(
        "--no-rank-weight",
        action="store_false",
        dest="rank_weight_enabled",
        help="禁用 rank-weight（覆盖 --rank-weight-enabled）",
    )
    parser.add_argument(
        "--rank-weight-topk", type=int, default=30, help="每日 Top/Bottom K 样本数，默认 30"
    )
    parser.add_argument(
        "--rank-weight", type=float, default=5.0, help="Top/Bottom K 样本权重，默认 5.0"
    )
    parser.add_argument(
        "--rank-weight-topk-weight-mode",
        type=str,
        default="linear_decay",
        choices=["linear_decay", "flat"],
        help="TopK 权重分配模式：linear_decay（默认）| flat（TopK 同权）",
    )

    # 时间衰减权重
    parser.add_argument(
        "--time-decay-half-life",
        type=float,
        default=0,
        help="时间衰减半衰期（年）。0 表示禁用。例如 1.0 → 1年前样本权重=0.5，2年前=0.25",
    )

    # 目标函数
    parser.add_argument(
        "--objective",
        type=str,
        default="mse",
        choices=["mse", "lambdarank"],
        help="目标函数类型：mse（回归，默认）或 lambdarank（排序学习，直接优化股票排序）",
    )

    # 基本面因子
    parser.add_argument(
        "--enable-fundamental-features",
        action="store_true",
        help="启用基本面因子（ROE、营收增速等）作为训练特征",
    )

    # 另类数据因子
    parser.add_argument(
        "--enable-alt-features",
        action="store_true",
        help="启用另类数据因子（股东人数、业绩预告等）",
    )

    # 融资融券因子
    parser.add_argument(
        "--enable-margin-features",
        action="store_true",
        help="启用融资融券因子（融资余额变动、融券/融资比、净买入比等）",
    )

    # 筹码胜率因子（5000 积分）
    parser.add_argument(
        "--enable-cyq-features",
        action="store_true",
        help="启用筹码胜率因子（winner_rate、成本偏离、筹码集中度等）",
    )

    # 基金持仓因子（5000 积分）
    parser.add_argument(
        "--enable-fund-features",
        action="store_true",
        help="启用基金持仓因子（持股比例、基金数量及其变化）",
    )

    # 业绩快报因子（5000 积分）
    parser.add_argument(
        "--enable-express-features",
        action="store_true",
        help="启用业绩快报因子（实际营收/净利润增速、业绩惊喜等）",
    )
    parser.add_argument(
        "--feature-stability-filter",
        action="store_true",
        help="启用特征稳定性筛选（移除跨时期IC方向不一致的特征）",
    )

    parser.add_argument(
        "--factor-prune",
        action="store_true",
        help="启用因子精简（从 data/models/factor_exclude_list.json 加载排除列表）",
    )
    parser.add_argument(
        "--factor-exclude-file",
        type=str,
        default=None,
        help="因子精简清单路径；未指定时使用 data/models/factor_exclude_list.json",
    )
    parser.add_argument(
        "--freshness-strategy",
        type=str,
        default="state_keep_event_decay",
        choices=["state_keep_event_decay", "state_keep_event_no_decay", "drop_all"],
        help=(
            "freshness 处理策略：state_keep_event_decay=状态型保留+事件型衰减（默认），"
            "state_keep_event_no_decay=状态型保留+事件型不衰减（实验归因），"
            "drop_all=删除全部 freshness 特征"
        ),
    )
    parser.add_argument(
        "--event-freshness-half-life-days",
        type=float,
        default=45.0,
        help="事件型因子 freshness 衰减半衰期（天），默认 45",
    )

    # 多偏移集成
    parser.add_argument(
        "--ensemble-offsets",
        type=int,
        default=0,
        help="多偏移集成：偏移月数（0=禁用, 1=±1个月→3模型, 2=±2个月→3模型）",
    )
    parser.add_argument(
        "--ensemble-seeds",
        type=str,
        default=None,
        help="多种子 bagging：逗号分隔的随机种子列表（如 42,1,2,3,4）。"
        "默认 None=单种子（用 --random-state），与多偏移正交可叠加",
    )
    parser.add_argument(
        "--ensemble-seed-keep-top-ratio",
        type=float,
        default=SEED_ENSEMBLE_KEEP_TOP_RATIO,
        help="多种子筛选保留比例（0~1），默认 0.30",
    )
    parser.add_argument(
        "--ensemble-seed-keep-min-models",
        type=int,
        default=SEED_ENSEMBLE_KEEP_MIN_MODELS,
        help="多种子筛选最少保留模型数，默认 3",
    )

    # 因子增强（2.2）
    parser.add_argument(
        "--enable-enhanced-features",
        action="store_true",
        default=False,
        help="启用增强因子（开盘强度、日内波动结构、委托不平衡）",
    )

    # 北向资金因子
    parser.add_argument(
        "--enable-north-features",
        action="store_true",
        default=False,
        help="启用北向资金因子（moneyflow_hsgt, 市场级广播）",
    )

    # 龙虎榜因子
    parser.add_argument(
        "--enable-lhb-features",
        action="store_true",
        default=False,
        help="启用龙虎榜因子（top_list, 稀疏数据未上榜填 0）",
    )

    # 一致预期因子
    parser.add_argument(
        "--enable-consensus-features",
        action="store_true",
        default=False,
        help="启用卖方一致预期因子（report_rc, 滚动 30/60/90 日聚合）",
    )

    # 现金流质量因子（需 cashflow 接口，2000 积分）
    parser.add_argument(
        "--enable-cashflow-quality-features",
        action="store_true",
        default=False,
        help="启用现金流质量因子（需先下载 cashflow 数据）",
    )

    # 一致预期修正因子（基于已有 report_rc 构建时序修正信号）
    parser.add_argument(
        "--enable-consensus-revision-features",
        action="store_true",
        default=False,
        help="启用一致预期修正因子（EPS修正加速度/分歧度等时序信号）",
    )

    # 分红政策质量因子（需 dividend 接口，2000 积分）
    parser.add_argument(
        "--enable-dividend-policy-features",
        action="store_true",
        default=False,
        help="启用分红政策质量因子（分红稳定性/增长率/支付率/双日期事件，需先下载 dividend 数据）",
    )

    # 运行时可用性标记（结构性缺失显式化；不改特征产物）
    parser.add_argument(
        "--enable-availability-markers",
        action="store_true",
        default=False,
        help=(
            "启用运行时可用性标记因子（has_cons_coverage/has_margin_balance/"
            "has_fund_holding/has_express_data；训练与推理同时按现有列派生，不修改特征分区）"
        ),
    )

    # 股东增减持因子（需先用 --enable-holdertrade-features 构建特征）
    parser.add_argument(
        "--enable-holdertrade-features",
        action="store_true",
        default=False,
        help=("启用股东增减持因子（运行时派生；需先下载 stk_holdertrade）"),
    )
    parser.add_argument(
        "--holdertrade-feature-set",
        choices=("full", "core"),
        default="full",
        help=(
            "股东增减持列集：full=全部 8 因子列+freshness+哨兵；core=仅保留证据支持的 4 列+哨兵"
            "（超参签名维度，禁止跨取值并组比较）"
        ),
    )

    # 股票回购因子（运行时派生；需先下载 repurchase）
    parser.add_argument(
        "--enable-top-inst-features",
        action="store_true",
        default=False,
        help=(
            "启用龙虎榜机构席位因子（top_inst，运行时派生；需 raw/top_inst 年分区已下载，"
            "见 docs/top_inst_factor_health.md）"
        ),
    )
    parser.add_argument(
        "--enable-repurchase-features",
        action="store_true",
        default=False,
        help="启用股票回购因子（运行时派生；需先下载 repurchase）",
    )
    parser.add_argument(
        "--repurchase-feature-set",
        choices=("full", "headroom"),
        default="full",
        help=(
            "股票回购列集：full=4 个值列+freshness+哨兵；headroom=仅 rp_price_headroom+freshness+哨兵"
            "（超参签名维度，禁止跨取值并组比较）"
        ),
    )

    # 十大流通股东因子（运行时派生；需先下载 top10_floatholders）
    parser.add_argument(
        "--enable-top10fh-features",
        action="store_true",
        default=False,
        help="启用十大流通股东因子（运行时派生；需先下载 top10_floatholders）",
    )
    parser.add_argument(
        "--top10fh-feature-set",
        choices=("full", "concentration"),
        default="full",
        help=(
            "十大流通股东列集：full=6 个值列+freshness+哨兵；concentration=仅 tfh_concentration_chg+哨兵"
            "（超参签名维度，禁止跨取值并组比较）"
        ),
    )
    parser.add_argument(
        "--exposure-replenish",
        action="store_true",
        default=False,
        help=(
            "暴露门控**对称回补**（P2-4）：上限放开且存在未回补减仓额时，T+1 按比例把持仓补回；"
            "需与 --exposure-table 或 --exposure-policy 同时使用"
        ),
    )
    parser.add_argument(
        "--exposure-trim-tolerance",
        type=float,
        default=None,
        help="暴露门控减仓/回补共用容差（组合总值比例，默认引擎 3%%）",
    )

    # 其他参数
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="数据根目录；未指定时使用 configs/base.yaml 中的 data.* 配置",
    )
    parser.add_argument(
        "--run-log-csv",
        type=str,
        default=None,
        help="训练运行日志CSV路径，默认为 {data_root}/models/stock_selection/ml_train_runs.csv",
    )
    parser.add_argument(
        "--wf-summary-csv",
        type=str,
        default=None,
        help="walk-forward 汇总CSV路径，默认为 {data_root}/walk_forward/walk_forward_summary.csv",
    )
    parser.add_argument(
        "--export-topk-details",
        action="store_true",
        default=True,
        help="导出每个 split 的逐日 Top20/Top30 名单与预测分数（默认开启）",
    )
    parser.add_argument(
        "--no-export-topk-details",
        action="store_false",
        dest="export_topk_details",
        help="禁用逐日 Top20/Top30 名单与预测分数导出",
    )
    parser.add_argument(
        "--batch-run-id", type=str, default=None, help="批量脚本生成的批次ID，仅用于汇总追踪"
    )
    parser.add_argument(
        "--batch-period-label",
        type=str,
        default=None,
        help="批量脚本传入的时间段标签，仅用于汇总追踪",
    )

    # OOS 回测参数
    parser.add_argument(
        "--oos-backtest",
        action="store_true",
        default=True,
        help="每个 split 训练后运行 OOS 回测（默认开启）",
    )
    parser.add_argument(
        "--no-oos-backtest",
        action="store_false",
        dest="oos_backtest",
        help="禁用 OOS 回测（仅保留统计指标评估）",
    )
    parser.add_argument(
        "--oos-backtest-months",
        type=int,
        default=0,
        help="OOS 回测时长（月），默认 0 表示自动对齐 test_window_months",
    )
    parser.add_argument("--bt-top-n", type=int, default=30, help="OOS 回测持仓 Top N，默认 30")
    parser.add_argument(
        "--bt-rebalance-freq",
        type=int,
        default=None,
        help="OOS 回测调仓频率（交易日），默认从标签自动推断",
    )

    # 回测初始资金
    parser.add_argument(
        "--bt-initial-capital",
        type=float,
        default=1000000.0,
        help="OOS 回测初始资金（默认：1000000）",
    )
    parser.add_argument(
        "--bt-sell-timing",
        type=str,
        default="open",
        choices=["open", "close"],
        help="OOS 回测卖出时机：open 或 close，默认 open",
    )
    parser.add_argument(
        "--bt-exclude-st",
        action="store_true",
        default=True,
        dest="bt_exclude_st",
        help="OOS 回测排除 ST 股票（默认开启）",
    )
    parser.add_argument(
        "--bt-no-exclude-st",
        action="store_false",
        dest="bt_exclude_st",
        help="OOS 回测不排除 ST 股票",
    )
    parser.add_argument(
        "--bt-min-list-days", type=int, default=365, help="OOS 回测最少上市天数，默认 365"
    )
    parser.add_argument(
        "--bt-max-weight-per-stock",
        type=float,
        default=None,
        help="OOS 回测单股最大权重（0~1），默认不限制",
    )
    parser.add_argument(
        "--bt-max-per-industry",
        type=int,
        default=None,
        help="OOS 回测单行业最大持仓数量，默认不限制",
    )

    # 信号层下行风险惩罚（A5：路由型，排序后处理；不改训练列集）
    parser.add_argument(
        "--downside-penalty",
        type=float,
        default=0.0,
        choices=DOWNSIDE_PENALTY_GRID,
        help=(
            "下行风险惩罚强度 λ（冻结网格 0/0.25/0.5；0=关闭且与基线逐位一致）："
            "对候选排序做 score−λ×风险分位 后处理；超参签名维度，禁止跨取值并组比较。"
            "预登记见 docs/plans/downside_penalty_prereg.md"
        ),
    )
    parser.add_argument(
        "--downside-penalty-column",
        choices=tuple(DOWNSIDE_PENALTY_COLUMNS),
        default="downside_vol_20",
        help=(
            "惩罚所用风险列：downside_vol_20=主臂；cvar_95_20=稳健性对照（不得用于宣布通过）"
        ),
    )

    # OOS 回测止损参数
    parser.add_argument(
        "--bt-stop-loss-enabled", action="store_true", default=False, help="启用 OOS 回测止损功能"
    )
    parser.add_argument(
        "--bt-stop-loss-drawdown-pct",
        type=float,
        default=30.0,
        help="OOS 回测回撤止损阈值（%%），默认 30.0",
    )
    parser.add_argument(
        "--bt-stop-loss-consecutive-limit-down",
        type=int,
        default=2,
        help="OOS 回测连续跌停止损天数，默认 2",
    )

    # 分批调仓
    parser.add_argument(
        "--stagger-tranches",
        type=int,
        default=1,
        help="分批调仓批次数（默认1=不分批）。设为K时将资金分成K份错开调仓，降低时点风险",
    )

    # 政策层 E2 shadow（P2-3）
    parser.add_argument(
        "--exposure-table",
        default=None,
        help=(
            "暴露系数表 CSV 路径（两列：日期, 暴露系数；系数落于 (0,1]）。"
            "启用后 OOS 回测的买入预算基数乘上信号日的暴露系数（缩减/暂停加仓）；"
            "表为 None 时不启用（成交与净值与改动前逐位一致）。文件名建议 ASCII"
        ),
    )
    # 政策层 P2-5：在线现算（λ_t 随持仓实时计算，不读预导出表）
    parser.add_argument(
        "--exposure-policy",
        default=None,
        help=(
            "在线暴露政策 `k=v,k=v`（如 arm=combined,mode=rolling,window=250,regime_q=0.6667,"
            "score_q=0.5,lambda=0.5）；与 --exposure-table 互斥。"
            "需同时给出 --policy-model-root 与 --policy-arm-suffix"
        ),
    )
    parser.add_argument(
        "--policy-model-root",
        default=None,
        help="terminal_loss 折模型根目录（如 data/walk_forward/terminal_risk_wf）",
    )
    parser.add_argument(
        "--policy-arm-suffix",
        default=None,
        help="折目录后缀（如 _d5_v6m_fscore），决定使用哪套风险模型",
    )
    parser.add_argument(
        "--policy-coverage-start",
        default=None,
        help="政策覆盖起点（YYYYMMDD，含）；默认不限（由折 ES 区间决定是否判定）",
    )

    # 仓位管理模式
    parser.add_argument(
        "--position-sizing",
        type=str,
        default="equal",
        choices=["equal", "score", "kelly", "half_kelly"],
        help="仓位管理模式: equal=等权, score=按分数, kelly=Kelly最优, half_kelly=半Kelly",
    )
    parser.add_argument(
        "--kelly-vol-window", type=int, default=60, help="Kelly 波动率估计窗口（交易日），默认 60"
    )
    parser.add_argument(
        "--kelly-max-leverage",
        type=float,
        default=0.25,
        help="Kelly 单只股票仓位上限（占总资产），默认 0.25",
    )

    parser.add_argument(
        "--no-early-rebalance-on-empty",
        dest="enable_early_rebalance_on_empty",
        action="store_false",
        default=True,
        help="禁用空仓/持有期拖尾时的提前调仓（默认启用：仓位清空或持有期满后残留盈利延续持仓时提前触发新一轮T0）",
    )

    # 部署训练参数
    parser.add_argument(
        "--no-deploy-train",
        action="store_true",
        default=False,
        help="禁用部署模型训练（默认开启：walk-forward完成后自动训练部署模型）",
    )

    # 跳过训练、复用已有模型（仅调参回测）
    parser.add_argument(
        "--skip-training",
        action="store_true",
        default=False,
        help="跳过模型训练，直接使用已有模型做 OOS 回测（需配合 --start-model-version）",
    )
    parser.add_argument(
        "--start-model-version",
        type=int,
        default=None,
        help="skip-training 模式下第一个 split 对应的模型版本号，后续 split 依次 +1",
    )
    parser.add_argument(
        "--skip-training-eval",
        action="store_true",
        default=False,
        help=(
            "skip-training 模式下补跑 OOS 评估（逐日 Top-K 明细 + 测试指标，不注册新模型、"
            "不写训练台账）；用于复用旧模型的惩罚/政策类实验（默认跳过以保持最小开销）"
        ),
    )
    return parser


def parse_walk_forward_args(argv: Optional[List[str]] = None):
    """解析 walk-forward 参数并执行规范化与校验。"""
    parser = build_walk_forward_parser()
    args = parser.parse_args(argv)

    args.selected_split_indices = _normalize_selected_split_indices(
        getattr(args, "selected_split_indices", [])
    )
    if args.ensemble_seed_keep_top_ratio <= 0 or args.ensemble_seed_keep_top_ratio > 1:
        logger.warning(
            f"ensemble_seed_keep_top_ratio={args.ensemble_seed_keep_top_ratio} 非法，自动修正为 {SEED_ENSEMBLE_KEEP_TOP_RATIO}"
        )
        args.ensemble_seed_keep_top_ratio = SEED_ENSEMBLE_KEEP_TOP_RATIO
    if args.ensemble_seed_keep_min_models <= 0:
        logger.warning(
            f"ensemble_seed_keep_min_models={args.ensemble_seed_keep_min_models} 非法，自动修正为 {SEED_ENSEMBLE_KEEP_MIN_MODELS}"
        )
        args.ensemble_seed_keep_min_models = SEED_ENSEMBLE_KEEP_MIN_MODELS

    # 如果指定了 --label，则覆盖 --label-column
    if args.label is not None:
        args.label_column = args.label
    if not 0.0 <= args.neutral_label_blend_weight <= 1.0:
        parser.error("--neutral-label-blend-weight 必须在 0~1 之间")
    if args.neutral_label_blend_weight > 0 and args.task != "regression":
        parser.error("--neutral-label-blend-weight 仅支持 regression 任务")
    if args.neutral_label_blend_weight > 0 and args.skip_training:
        parser.error("混合标签必须重新训练，不能与 --skip-training 同时使用")
    if getattr(args, "skip_training_eval", False) and not args.skip_training:
        parser.error(
            "--skip-training-eval 仅在 --skip-training 模式下有效（复用旧模型补跑 OOS 评估）"
        )
    if args.neutral_label_blend_weight > 0 and not args.label_column.startswith("neu_y_ret_"):
        parser.error("混合标签要求 --label 使用 neu_y_ret_N")

    return args
