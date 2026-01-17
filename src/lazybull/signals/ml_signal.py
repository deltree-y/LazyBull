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
        weight_method: str = "equal"
    ):
        """初始化 ML 信号
        
        Args:
            top_n: 选择 Top N 只股票
            model_version: 模型版本号，None 表示使用最新版本
            models_dir: 模型目录
            weight_method: 权重分配方法，"equal" 表示等权，"score" 表示按预测分数加权
        """
        super().__init__("ml_signal")
        self.top_n = top_n
        self.model_version = model_version
        self.models_dir = models_dir
        self.weight_method = weight_method
        
        # 延迟加载模型
        self.model = None
        self.metadata = None
        self.feature_columns = None
        
        logger.info(
            f"ML 信号初始化: top_n={top_n}, model_version={model_version}, "
            f"weight_method={weight_method}"
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
                # 如果所有分数都是负数或零，使用等权
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
