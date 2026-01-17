"""回测报告生成"""

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


class Reporter:
    """回测报告生成器"""
    
    def __init__(self, output_dir: str = "./data/reports"):
        """初始化报告生成器
        
        Args:
            output_dir: 报告输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(
        self,
        nav_curve: pd.DataFrame,
        trades: pd.DataFrame,
        output_name: str = "backtest_report"
    ) -> dict:
        """生成回测报告
        
        Args:
            nav_curve: 净值曲线DataFrame
            trades: 交易记录DataFrame
            output_name: 输出文件名（不含扩展名）
            
        Returns:
            报告统计指标字典
        """
        # 计算关键指标
        stats = self._calculate_statistics(nav_curve, trades)
        
        # 打印到控制台
        self._print_summary(stats)
        
        # 将列名转换为中文
        nav_curve_chinese = self._translate_nav_columns(nav_curve)
        trades_chinese = self._translate_trades_columns(trades)
        
        # 保存净值曲线
        nav_file = self.output_dir / f"{output_name}_nav.csv"
        nav_curve_chinese.to_csv(nav_file, index=False, encoding="utf-8-sig")
        logger.info(f"净值曲线已保存: {nav_file}")
        
        # 保存交易记录
        if not trades_chinese.empty:
            trades_file = self.output_dir / f"{output_name}_trades.csv"
            trades_chinese.to_csv(trades_file, index=False, encoding="utf-8-sig")
            logger.info(f"交易记录已保存: {trades_file}")
        
        # 保存统计指标
        stats_file = self.output_dir / f"{output_name}_stats.txt"
        self._save_stats(stats, stats_file)
        
        return stats
    
    def _translate_nav_columns(self, nav_curve: pd.DataFrame) -> pd.DataFrame:
        """将净值曲线列名转换为中文
        
        Args:
            nav_curve: 原始净值曲线DataFrame
            
        Returns:
            列名为中文的DataFrame
        """
        if nav_curve.empty:
            return nav_curve
        
        column_mapping = {
            'date': '日期',
            'portfolio_value': '组合总值',
            'capital': '可用资金',
            'market_value': '持仓市值',
            'nav': '净值',
            'return': '收益率'
        }
        
        result = nav_curve.copy()
        result = result.rename(columns=column_mapping)
        
        return result
    
    def _translate_trades_columns(self, trades: pd.DataFrame) -> pd.DataFrame:
        """将交易记录列名转换为中文
        
        Args:
            trades: 原始交易记录DataFrame
            
        Returns:
            列名为中文的DataFrame
        """
        if trades.empty:
            return trades
        
        column_mapping = {
            'date': '交易日期',
            'stock': '股票代码',
            'action': '操作',
            'price': '成交价格',
            'shares': '成交股数',
            'amount': '成交金额',
            'cost': '交易成本'
        }
        
        result = trades.copy()
        result = result.rename(columns=column_mapping)
        
        # 将 action 列的值也转换为中文
        if '操作' in result.columns:
            action_mapping = {
                'buy': '买入',
                'sell': '卖出'
            }
            result['操作'] = result['操作'].map(action_mapping).fillna(result['操作'])
        
        return result
    
    def _calculate_statistics(self, nav_curve: pd.DataFrame, trades: pd.DataFrame) -> dict:
        """计算统计指标
        
        Args:
            nav_curve: 净值曲线
            trades: 交易记录
            
        Returns:
            统计指标字典
        """
        if nav_curve.empty:
            return {}
        
        # 基础指标
        total_return = nav_curve['return'].iloc[-1]
        nav_values = nav_curve['nav'].values
        
        # 最大回撤
        cummax = pd.Series(nav_values).cummax()
        drawdown = (pd.Series(nav_values) - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # 年化收益率（简化计算）
        trading_days = len(nav_curve)
        years = trading_days / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # 波动率（年化）
        daily_returns = nav_curve['nav'].pct_change().dropna()
        volatility = daily_returns.std() * (252 ** 0.5)
        
        # 夏普比率（假设无风险利率为3%）
        risk_free_rate = 0.03
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # 交易统计
        total_trades = len(trades)
        total_cost = trades['cost'].sum() if not trades.empty else 0
        
        stats = {
            '总收益率': f"{total_return * 100:.2f}%",
            '年化收益率': f"{annual_return * 100:.2f}%",
            '最大回撤': f"{max_drawdown * 100:.2f}%",
            '波动率': f"{volatility * 100:.2f}%",
            '夏普比率': f"{sharpe_ratio:.2f}",
            '交易次数': total_trades,
            '总交易成本': f"{total_cost:.2f}元",
            '回测天数': trading_days,
            '起始净值': f"{nav_values[0]:.4f}",
            '结束净值': f"{nav_values[-1]:.4f}",
        }
        
        return stats
    
    def _print_summary(self, stats: dict) -> None:
        """打印摘要到控制台
        
        Args:
            stats: 统计指标字典
        """
        logger.info("=" * 60)
        logger.info("回测报告摘要")
        logger.info("=" * 60)
        
        for key, value in stats.items():
            logger.info(f"{key:12s}: {value}")
        
        logger.info("=" * 60)
    
    def _save_stats(self, stats: dict, file_path: Path) -> None:
        """保存统计指标到文件
        
        Args:
            stats: 统计指标字典
            file_path: 文件路径
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("回测报告统计指标\n")
            f.write("=" * 60 + "\n\n")
            
            for key, value in stats.items():
                f.write(f"{key:12s}: {value}\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        logger.info(f"统计指标已保存: {file_path}")
