"""ML 信号生成模块

基于训练好的机器学习模型生成交易信号
使用排序选股 Top N 方式
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..ml import ModelRegistry
from .base import Signal


class MLSignal(Signal):
    """ML 信号生成器
    
    基于机器学习模型预测，选择预测收益最高的 Top N 股票
    """
    
    def __init__(
        self,
        top_n: int = 30,
        model_version: Optional[int] = None,
        models_dir: str = "./data/models",
        weight_method: str = "equal",
        amount_filter_enabled: bool = True,
        amount_filter_pct: float = 20.0,
        amount_window: int = 5
    ):
        """初始化 ML 信号
        
        Args:
            top_n: 选择 Top N 只股票
            model_version: 模型版本号，None 表示使用最新版本
            models_dir: 模型目录
            weight_method: 权重分配方法，"equal" 表示等权，"score" 表示按预测分数加权
            amount_filter_enabled: 是否启用成交额过滤，默认True
            amount_filter_pct: 过滤成交额后N%的股票，默认20%
            amount_window: 计算成交额时使用的天数窗口，默认5天（使用近5日平均成交额）
        """
        super().__init__("ml_signal")
        self.top_n = top_n
        self.model_version = model_version
        self.models_dir = models_dir
        self.weight_method = weight_method
        self.amount_filter_enabled = amount_filter_enabled
        self.amount_filter_pct = amount_filter_pct
        self.amount_window = amount_window
        
        # 延迟加载模型
        self.model = None
        self.metadata = None
        self.feature_columns = None
        
        logger.info(
            f"ML 信号初始化: top_n={top_n}, model_version={model_version}, "
            f"weight_method={weight_method}, amount_filter_enabled={amount_filter_enabled}, "
            f"amount_filter_pct={amount_filter_pct}%, amount_window={amount_window}"
        )
    
    def _load_model(self) -> None:
        """加载模型（延迟加载）"""
        if self.model is None:
            registry = ModelRegistry(models_dir=self.models_dir)
            self.model, self.metadata = registry.load_model(version=self.model_version)
            self.feature_columns = self.metadata["feature_columns"]
            logger.info(
                f"模型已加载: {self.metadata['version_str']}, "
                f"特征数={self.metadata['feature_count']}"
            )
    
    def _apply_amount_filter(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """应用成交额过滤
        
        过滤掉后N%的股票（基于n天成交额）。优先使用amount字段，缺失时当作0处理。
        
        Args:
            features_df: 特征DataFrame，需包含amount或相关成交额字段
            
        Returns:
            过滤后的DataFrame
        """
        if not self.amount_filter_enabled:
            return features_df
        
        if len(features_df) == 0:
            return features_df
        
        # 检查是否有成交额字段
        amount_col = None
        if 'amount' in features_df.columns:
            amount_col = 'amount'
        elif f'amount_ma{self.amount_window}' in features_df.columns:
            amount_col = f'amount_ma{self.amount_window}'
        elif f'amount_ratio_{self.amount_window}' in features_df.columns:
            # 如果有amount_ratio，尝试反推出平均成交额
            logger.warning(f"未找到amount或amount_ma{self.amount_window}字段，尝试使用amount_ratio")
            # 这种情况下，我们无法准确过滤，给出警告后跳过
            logger.warning("无法从amount_ratio推算绝对成交额，跳过成交额过滤")
            return features_df
        
        if amount_col is None:
            logger.warning("特征数据中没有成交额字段(amount或amount_ma*)，跳过成交额过滤")
            return features_df
        
        # 成交额缺失当0处理
        features_with_amount = features_df.copy()
        features_with_amount[amount_col] = features_with_amount[amount_col].fillna(0)
        
        # 过滤成交额为0的股票也要包含在内，但它们会被排在最后
        # 计算成交额分位数阈值（基于所有股票，包括成交额为0的）
        amount_threshold_pct = self.amount_filter_pct / 100.0
        
        # 如果所有成交额都是0，则不过滤
        if (features_with_amount[amount_col] == 0).all():
            logger.warning("所有股票成交额为0，跳过成交额过滤")
            return features_df
        
        amount_threshold = features_with_amount[amount_col].quantile(amount_threshold_pct)
        
        # 过滤掉成交额后N%的股票
        before_count = len(features_with_amount)
        features_filtered = features_with_amount[features_with_amount[amount_col] > amount_threshold].copy()
        filtered_count = before_count - len(features_filtered)
        
        if filtered_count > 0:
            logger.info(
                f"成交额过滤: 从{before_count}只股票中过滤掉{self.amount_window}天成交额后{self.amount_filter_pct}%的{filtered_count}只股票, "
                f"剩余{len(features_filtered)}只 (阈值={amount_threshold:.2f})"
            )
        
        return features_filtered
    
    def generate(
        self,
        date: pd.Timestamp,
        universe: List[str],
        data: Dict
    ) -> Dict[str, float]:
        """生成 ML 信号
        
        Args:
            date: 当前日期
            universe: 股票池（股票代码列表）
            data: 数据字典，应包含 "features" 键，值为当日特征 DataFrame
            
        Returns:
            信号字典，{股票代码: 权重}
        """
        # 加载模型
        self._load_model()
        
        # 获取当日特征数据
        if "features" not in data:
            logger.warning(f"{date.date()} 没有特征数据")
            return {}
        
        features_df = data["features"]
        
        if features_df is None or len(features_df) == 0:
            logger.warning(f"{date.date()} 特征数据为空")
            return {}
        
        # 过滤股票池
        features_df = features_df[features_df['ts_code'].isin(universe)].copy()
        
        if len(features_df) == 0:
            logger.warning(f"{date.date()} 股票池没有匹配的特征数据")
            return {}
        
        # 应用成交额过滤
        features_df = self._apply_amount_filter(features_df)
        
        if len(features_df) == 0:
            logger.warning(f"{date.date()} 成交额过滤后无可选股票")
            return {}
        
        # 准备特征
        try:
            X = features_df[self.feature_columns].copy()
            X = X.fillna(0)  # 填充缺失值
        except KeyError as e:
            logger.error(f"特征列缺失: {e}")
            return {}
        
        # 预测
        predictions = self.model.predict(X)
        features_df['ml_score'] = predictions
        
        # 按预测分数排序，选择 Top N
        features_df = features_df.sort_values('ml_score', ascending=False)
        top_stocks = features_df.head(self.top_n)
        
        if len(top_stocks) == 0:
            logger.warning(f"{date.date()} 没有有效的预测结果")
            return {}
        
        # 分配权重
        if self.weight_method == "equal":
            # 等权
            weight = 1.0 / len(top_stocks)
            signals = {stock: weight for stock in top_stocks['ts_code'].tolist()}
        elif self.weight_method == "score":
            # 按预测分数加权
            total_score = top_stocks['ml_score'].sum()
            if total_score <= 0:
                # 如果所有分数都是负数或零，回退到等权
                weight = 1.0 / len(top_stocks)
                signals = {stock: weight for stock in top_stocks['ts_code'].tolist()}
            else:
                # 归一化分数为权重（使用向量化操作）
                scores = top_stocks['ml_score'].values
                stocks = top_stocks['ts_code'].values
                weights = np.maximum(0, scores) / total_score
                
                # 重新归一化确保权重和为 1
                total_weight = weights.sum()
                if total_weight > 0:
                    weights = weights / total_weight
                
                signals = dict(zip(stocks, weights))
        else:
            raise ValueError(f"不支持的权重方法: {self.weight_method}")
        
        logger.debug(
            f"ML 信号生成完成: {date.date()}, 选择 {len(signals)} 只股票, "
            f"平均预测分数={top_stocks['ml_score'].mean():.6f}"
        )
        
        return signals
    
    def generate_ranked(
        self,
        date: pd.Timestamp,
        universe: List[str],
        data: Dict
    ) -> List[tuple]:
        """生成排序后的候选股票列表（支持回填）
        
        返回所有候选股票的完整排序列表，而不仅仅是 top N。
        这样可以在 top N 中有不可交易股票时从后续候选中回填。
        
        Args:
            date: 当前日期
            universe: 股票池
            data: 数据字典，应包含 "features" 键
            
        Returns:
            排序后的 (股票代码, 预测分数) 元组列表，按分数降序排列
        """
        # 加载模型
        self._load_model()
        
        # 获取当日特征数据
        if "features" not in data:
            logger.warning(f"{date.date()} 没有特征数据")
            return []
        
        features_df = data["features"]
        
        if features_df is None or len(features_df) == 0:
            logger.warning(f"{date.date()} 特征数据为空")
            return []
        
        # 过滤股票池
        features_df = features_df[features_df['ts_code'].isin(universe)].copy()
        
        if len(features_df) == 0:
            logger.warning(f"{date.date()} 股票池没有匹配的特征数据")
            return []
        
        # 应用成交额过滤
        features_df = self._apply_amount_filter(features_df)
        
        if len(features_df) == 0:
            logger.warning(f"{date.date()} 成交额过滤后无可选股票")
            return []
        
        # 准备特征
        try:
            X = features_df[self.feature_columns].copy()
            X = X.fillna(0)  # 填充缺失值
        except KeyError as e:
            logger.error(f"特征列缺失: {e}")
            return []
        
        # 预测
        predictions = self.model.predict(X)
        features_df['ml_score'] = predictions
        
        # 按预测分数排序，返回所有候选
        features_df = features_df.sort_values('ml_score', ascending=False)
        
        # 返回 (股票代码, 分数) 元组列表
        ranked = list(zip(features_df['ts_code'].tolist(), features_df['ml_score'].tolist()))
        
        logger.info(
            f"ML排序候选生成: {date.date()}, "#候选数 {len(ranked)}, "
            f"平均预测分数[{features_df['ml_score'].mean():.3f}], "
            f"最高/最低[{features_df['ml_score'].max():.3f}/{features_df['ml_score'].min():.3f}]"
        )
        
        return ranked
    
    def generate_with_features(
        self,
        date: pd.Timestamp,
        universe: List[str],
        features_df: pd.DataFrame
    ) -> Dict[str, float]:
        """使用提供的特征数据生成信号（便捷方法）
        
        Args:
            date: 当前日期
            universe: 股票池
            features_df: 特征 DataFrame
            
        Returns:
            信号字典
        """
        data = {"features": features_df}
        return self.generate(date, universe, data)
    
    def get_model_info(self) -> Dict:
        """获取模型信息
        
        Returns:
            模型元数据字典
        """
        self._load_model()
        return self.metadata
