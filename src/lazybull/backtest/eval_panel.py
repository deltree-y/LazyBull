# -*- coding: utf-8 -*-
"""ML 信号评估面板：从 scripts/run_ml_backtest.py 下沉的评估能力。

提供按日评估 (evaluate_daily)、等数量分组 (equal_count_grouping)、
评估面板导出 (export_evaluation_panel) 及 CSV 工具函数，供测试与示例复用。
"""

import csv
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from src.lazybull.signals import MLSignal
from src.lazybull.universe import BasicUniverse

def _append_dict_to_csv(file_path: Path, row: dict, fieldnames: list = None):
    """把一个 dict 追加到 CSV（如果不存在则写 header）
    
    Args:
        file_path: 目标文件 Path
        row: 要写入的一行 dict
        fieldnames: 列顺序列表（如果 None 则使用 row.keys() 的顺序）
    """
    # 确保目录存在
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = file_path.exists()

    # 使用 utf-8-sig 以便 Excel 直接识别中文
    with open(file_path, 'a', newline='', encoding='utf-8-sig') as f:
        if fieldnames is None:
            fieldnames_local = list(row.keys())
        else:
            fieldnames_local = fieldnames
        writer = csv.DictWriter(f, fieldnames=fieldnames_local)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _append_trades_to_cumulative_file(
    trades: pd.DataFrame,
    args,
    reporter: 'Reporter',
    run_id: str,
    run_time: str
):
    """将交易记录追加到累加文件中，并计算交易盈亏
    
    Args:
        trades: 本次回测的交易记录DataFrame
        args: 命令行参数
        reporter: Reporter实例
        run_id: 回测ID
        run_time: 回测执行时间
    """
    if trades is None or len(trades) == 0:
        logger.info("本次回测无交易记录，跳过累加文件写入")
        return
    
    try:
        # 累加文件路径
        cumulative_file = Path(reporter.output_dir) / "ml_backtest_trades_runs.csv"
        
        # 准备要添加的参数列（中文列名）
        model_version_str = "最新版本" if args.model_version is None else str(args.model_version)
        
        # 定义所有字段及其顺序（先是核心参数，再是原有交易字段）
        fieldnames = [
            "回测ID",
            "回测时间",
            "开始日期",
            "结束日期",
            "模型版本",
            "TopN",
            "仓位管理",
            "调仓频率",
            "初始资金",
            "卖出时机",
            # 原有交易记录字段
            "交易日期",
            "股票代码",
            "操作",
            "成交价格",
            "成交股数",
            "成交金额",
            "交易成本",
            "买入价格",
            "收益金额",
            "收益率"
        ]
        
        # 构建买入价格字典（FIFO：先进先出）
        # 存储格式：buy_prices[股票代码] = [{'price': 买入价格, 'amount': 买入金额, 'cost': 买入成本, 'shares': 股数}, ...]
        buy_prices = defaultdict(deque)
        
        # 第一遍遍历：记录所有买入交易
        for _, trade in trades.iterrows():
            action = trade.get('action', '')
            if action == 'buy':
                stock = trade.get('stock', '')
                buy_prices[stock].append({
                    'price': trade.get('price', 0),
                    'amount': trade.get('amount', 0),
                    'cost': trade.get('cost', 0),
                    'shares': trade.get('shares', 0)
                })
        
        # 第二遍遍历：计算卖出交易的盈亏，并写入累加文件
        for _, trade in trades.iterrows():
            action = trade.get('action', '')
            stock = trade.get('stock', '')
            
            # 初始化收益字段
            buy_price_value = ''
            profit_amount = ''
            profit_pct = ''
            
            # 如果是卖出交易，计算收益
            if action == 'sell':
                # 从 FIFO 队列中获取对应的买入信息
                if stock in buy_prices and len(buy_prices[stock]) > 0:
                    buy_info = buy_prices[stock].popleft()  # 先进先出
                    
                    # 计算收益
                    # 买入成本 = 买入金额 + 买入交易成本
                    buy_cost = buy_info['amount'] + buy_info['cost']
                    
                    # 卖出金额和成本
                    sell_amount = trade.get('amount', 0)
                    sell_cost = trade.get('cost', 0)
                    
                    # 收益金额 = 卖出金额 - 买入成本 - 卖出成本
                    profit_amount_value = sell_amount - buy_cost - sell_cost
                    
                    # 收益率 = (收益金额 / 买入成本) × 100%
                    if buy_cost > 0:
                        profit_pct_value = (profit_amount_value / buy_cost) * 100
                        profit_pct = f"{profit_pct_value:.2f}%"
                    else:
                        profit_pct = "0.00%"
                    
                    # 格式化收益金额（保留2位小数）
                    profit_amount = f"{profit_amount_value:.2f}"
                    buy_price_value = buy_info['price']
                else:
                    # 没有找到对应的买入记录（理论上不应发生）
                    logger.warning(f"卖出交易未找到对应买入记录: {stock} @ {trade.get('date', '')}")
            
            # 构建完整的交易记录（参数 + 交易明细）
            trade_with_params = {
                "回测ID": run_id,
                "回测时间": run_time,
                "开始日期": args.start_date,
                "结束日期": args.end_date,
                "模型版本": model_version_str,
                "TopN": args.top_n,
                "仓位管理": args.position_sizing,
                "调仓频率": args.rebalance_freq,
                "初始资金": args.initial_capital,
                "卖出时机": args.sell_timing,
                # 原有交易字段
                "交易日期": trade.get('date', ''),
                "股票代码": stock,
                "操作": "买入" if action == 'buy' else "卖出" if action == 'sell' else action,
                "成交价格": trade.get('price', ''),
                "成交股数": trade.get('shares', ''),
                "成交金额": trade.get('amount', ''),
                "交易成本": trade.get('cost', ''),
                "买入价格": buy_price_value,
                "收益金额": profit_amount,
                "收益率": profit_pct
            }
            
            # 追加到累加文件
            _append_dict_to_csv(cumulative_file, trade_with_params, fieldnames=fieldnames)
        
        logger.info(f"本次回测 {len(trades)} 笔交易已追加到累加文件: {cumulative_file}")
        
    except Exception as ex:
        # 记录追加失败不影响回测结果输出，但记录错误信息
        logger.exception(f"写交易记录到累加文件失败: {ex}")


def equal_count_grouping(scores: pd.Series, n_groups: int = 10) -> pd.Series:
    """按预测分数等数量分组
    
    将样本按分数降序排序后，切分成近似等大小的 n_groups 组。
    前面的组可能多1个样本（如果总数不能被 n_groups 整除）。
    
    Args:
        scores: 预测分数 Series（index 为 ts_code 或其他标识）
        n_groups: 分组数量
        
    Returns:
        pd.Series: 分组标签（1 表示最高分组，n_groups 表示最低分组）
    """
    if len(scores) == 0:
        return pd.Series(dtype=int)
    
    # 按分数降序排序
    sorted_scores = scores.sort_values(ascending=False)
    n_samples = len(sorted_scores)
    
    # 计算每组的基础大小和需要额外加1的组数
    base_size = n_samples // n_groups
    extra = n_samples % n_groups
    
    # 分配组标签
    group_labels = []
    for i in range(n_groups):
        # 前 extra 组每组多1个
        group_size = base_size + (1 if i < extra else 0)
        group_labels.extend([i + 1] * group_size)
    
    # 创建分组 Series（保持原始索引顺序）
    groups = pd.Series(group_labels, index=sorted_scores.index)
    return groups


def evaluate_daily(
    date: str,
    signal: MLSignal,
    universe: list,
    features_df: pd.DataFrame,
    label_column: str,
    n_groups: int = 10,
    topk: int = None
) -> tuple:
    """评估单日的 ML 信号质量
    
    Args:
        date: 交易日期（YYYYMMDD 格式字符串）
        signal: ML 信号生成器
        universe: 股票池
        features_df: 当日特征数据（包含 label 列）
        label_column: 真实收益标签列名（如 y_ret_5）
        n_groups: 分组数量
        topk: TopK 指标的 K（若为 None 则不计算）
        
    Returns:
        (daily_metrics, group_details) 元组
        - daily_metrics: dict，日度指标
        - group_details: list of dict，每组的详细信息
    """
    # 使用 MLSignal.generate_ranked 得到排序候选
    ranked = signal.generate_ranked(
        date=pd.Timestamp(date),
        universe=universe,
        data={'features': features_df}
    )
    
    if not ranked or len(ranked) == 0:
        logger.warning(f"{date}: 无排序候选，跳过评估")
        return None, None
    
    # 转换为 DataFrame
    ranked_df = pd.DataFrame(ranked, columns=['ts_code', 'score'])
    
    # 与 label 列对齐
    if label_column not in features_df.columns:
        logger.warning(f"{date}: 特征数据中缺少标签列 {label_column}，跳过评估")
        return None, None
    
    # 合并预测分数和真实标签
    eval_df = ranked_df.merge(
        features_df[['ts_code', label_column]],
        on='ts_code',
        how='left'
    )
    
    # 过滤缺失标签的样本
    eval_df = eval_df.dropna(subset=[label_column])
    
    if len(eval_df) == 0:
        logger.warning(f"{date}: 所有样本的标签都缺失，跳过评估")
        return None, None
    
    n_samples = len(eval_df)
    
    # 计算 RankIC (Spearman)
    rank_ic = eval_df['score'].corr(eval_df[label_column], method='spearman')
    
    # 等数量分组
    eval_df['group'] = equal_count_grouping(eval_df.set_index('ts_code')['score'], n_groups).values
    
    # 计算各组平均真实收益
    group_stats = eval_df.groupby('group').agg({
        'ts_code': 'count',
        label_column: 'mean',
        'score': 'mean'
    }).rename(columns={
        'ts_code': 'count',
        label_column: 'avg_real_return',
        'score': 'avg_score'
    })
    
    # Top 组和 Bottom 组
    top_group_return = group_stats.loc[1, 'avg_real_return'] if 1 in group_stats.index else None
    bottom_group_return = group_stats.loc[n_groups, 'avg_real_return'] if n_groups in group_stats.index else None
    long_short_return = top_group_return - bottom_group_return if (top_group_return is not None and bottom_group_return is not None) else None
    
    # TopK 平均真实收益
    topk_return = None
    if topk is not None and topk > 0:
        topk_samples = eval_df.nlargest(min(topk, len(eval_df)), 'score')
        if len(topk_samples) > 0:
            topk_return = topk_samples[label_column].mean()
    
    # 日度指标
    daily_metrics = {
        '交易日期': date,
        '样本数': n_samples,
        'RankIC': rank_ic,
        'TopK平均收益': topk_return,
        'Top组平均收益': top_group_return,
        'Bottom组平均收益': bottom_group_return,
        '多空收益': long_short_return
    }
    
    # 分组明细
    group_details = []
    for group_id, row in group_stats.iterrows():
        group_details.append({
            '交易日期': date,
            '组号': group_id,
            '组内股票数': int(row['count']),
            '组内平均真实收益': row['avg_real_return'],
            '组内平均预测分数': row['avg_score']
        })
    
    return daily_metrics, group_details


def export_evaluation_panel(
    signal: MLSignal,
    universe_obj: BasicUniverse,
    features_by_date: dict,
    trading_dates: list,
    label_column: str,
    output_dir: str,
    output_name: str,
    n_groups: int = 10,
    topk: int = None,
    args = None
):
    """导出评估面板 CSV
    
    Args:
        signal: ML 信号生成器
        universe_obj: 股票池对象
        features_by_date: 按日期组织的特征数据字典 {date_str: features_df}
        trading_dates: 交易日列表（pd.Timestamp）
        label_column: 真实收益标签列名
        output_dir: 输出目录
        output_name: 输出文件名前缀
        n_groups: 分组数量
        topk: TopK 指标的 K
        args: 命令行参数（用于汇总 CSV）
    """
    logger.info("=" * 60)
    logger.info("开始导出评估面板...")
    logger.info(f"标签列: {label_column}")
    logger.info(f"分组数: {n_groups}")
    logger.info(f"TopK: {topk}")
    logger.info("=" * 60)
    
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # 准备输出文件
    daily_file = output_dir_path / f"{output_name}_eval_daily.csv"
    groups_file = output_dir_path / f"{output_name}_eval_groups.csv"
    summary_file = output_dir_path / f"{output_name}_eval_summary.csv"
    
    # 删除旧文件（如果存在）
    for f in [daily_file, groups_file]:
        if f.exists():
            f.unlink()
    
    # 遍历每个交易日进行评估
    all_daily_metrics = []
    all_group_details = []
    
    for date_ts in trading_dates:
        date_str = date_ts.strftime('%Y%m%d')
        
        # 检查是否有特征数据
        if date_str not in features_by_date:
            logger.debug(f"{date_str}: 无特征数据，跳过评估")
            continue
        
        features_df = features_by_date[date_str]
        
        # 获取当日股票池
        universe = universe_obj.get_stocks(date_ts)
        
        # 评估当日
        daily_metrics, group_details = evaluate_daily(
            date=date_str,
            signal=signal,
            universe=universe,
            features_df=features_df,
            label_column=label_column,
            n_groups=n_groups,
            topk=topk
        )
        
        if daily_metrics is not None:
            all_daily_metrics.append(daily_metrics)
        
        if group_details is not None:
            all_group_details.extend(group_details)
    
    # 写入日度评估 CSV
    if all_daily_metrics:
        daily_df = pd.DataFrame(all_daily_metrics)
        daily_df.to_csv(daily_file, index=False, encoding='utf-8-sig')
        logger.info(f"日度评估 CSV 已保存: {daily_file} ({len(daily_df)} 行)")
    else:
        logger.warning("没有日度评估数据")
    
    # 写入分组明细 CSV
    if all_group_details:
        groups_df = pd.DataFrame(all_group_details)
        groups_df.to_csv(groups_file, index=False, encoding='utf-8-sig')
        logger.info(f"分组明细 CSV 已保存: {groups_file} ({len(groups_df)} 行)")
    else:
        logger.warning("没有分组明细数据")
    
    # 计算汇总指标
    if all_daily_metrics:
        daily_df = pd.DataFrame(all_daily_metrics)
        
        # 聚合指标
        rank_ic_mean = daily_df['RankIC'].mean()
        rank_ic_std = daily_df['RankIC'].std()
        rank_ic_ir = rank_ic_mean / rank_ic_std if rank_ic_std > 1e-10 else None
        
        topk_mean = daily_df['TopK平均收益'].mean() if 'TopK平均收益' in daily_df.columns else None
        long_short_mean = daily_df['多空收益'].mean() if '多空收益' in daily_df.columns else None
        
        # 构建汇总记录
        summary_record = {
            '回测时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            '开始日期': args.start_date if args else None,
            '结束日期': args.end_date if args else None,
            '标签列': label_column,
            '模型版本': args.model_version if args and args.model_version is not None else '最新版本',
            'TopN': args.top_n if args else None,
            '仓位管理': args.position_sizing if args else None,
            '调仓频率': args.rebalance_freq if args else None,
            '初始资金': args.initial_capital if args else None,
            '卖出时机': args.sell_timing if args else None,
            '分组数': n_groups,
            'TopK': topk,
            '评估天数': len(daily_df),
            'RankIC均值': rank_ic_mean,
            'RankIC标准差': rank_ic_std,
            'RankIC_IR': rank_ic_ir,
            'TopK平均收益': topk_mean,
            '多空平均收益': long_short_mean,
        }
        
        # 追加到汇总 CSV
        summary_fieldnames = list(summary_record.keys())
        _append_dict_to_csv(summary_file, summary_record, fieldnames=summary_fieldnames)
        logger.info(f"汇总指标 CSV 已保存: {summary_file}")
        
        # 打印汇总信息
        logger.info("=" * 60)
        logger.info("评估面板汇总指标:")
        logger.info(f"  评估天数: {len(daily_df)}")
        logger.info(f"  RankIC 均值: {rank_ic_mean:.4f}")
        logger.info(f"  RankIC 标准差: {rank_ic_std:.4f}")
        logger.info(f"  RankIC IR: {rank_ic_ir:.4f}" if rank_ic_ir is not None else "  RankIC IR: N/A")
        logger.info(f"  TopK 平均收益: {topk_mean:.4f}" if topk_mean is not None else "  TopK 平均收益: N/A")
        logger.info(f"  多空平均收益: {long_short_mean:.4f}" if long_short_mean is not None else "  多空平均收益: N/A")
        logger.info("=" * 60)
