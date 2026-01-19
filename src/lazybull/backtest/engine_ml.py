"""ML 回测引擎

扩展 BacktestEngine 以支持 ML 信号的特征数据注入
"""

from typing import Dict, Optional

import pandas as pd
from loguru import logger

from .engine import BacktestEngine


class BacktestEngineML(BacktestEngine):
    """支持 ML 信号的回测引擎
    
    通过重写 _build_signal_data 方法注入特征数据，
    其他回测逻辑（信号过滤、回填、权重归一化等）复用父类实现。
    """
    
    def __init__(self, features_by_date: Dict[str, pd.DataFrame], **kwargs):
        """初始化 ML 回测引擎
        
        Args:
            features_by_date: 按日期组织的特征数据字典，键为日期字符串（YYYYMMDD），值为特征 DataFrame
            **kwargs: 其他参数传递给父类 BacktestEngine
        """
        super().__init__(**kwargs)
        self.features_by_date = features_by_date
        
        logger.info(f"ML 回测引擎初始化: 特征数据覆盖 {len(features_by_date)} 个交易日")
    
    def _build_signal_data(self, date: pd.Timestamp) -> Optional[Dict]:
        """构建信号数据（注入 ML 特征）
        
        从 features_by_date 中获取当日特征数据。
        
        Args:
            date: 信号生成日期
            
        Returns:
            包含 "features" 键的数据字典，如果当日无特征数据则返回 None
        """
        # 转换日期格式
        date_str = date.strftime('%Y%m%d')
        
        # 获取特征数据
        features_df = self.features_by_date.get(date_str)
        
        if features_df is None or len(features_df) == 0:
            # 无特征数据，返回 None 让父类跳过该日期
            logger.warning(f"信号日 {date.date()} 没有特征数据，跳过")
            return None
        
        # 返回特征数据字典
        return {"features": features_df}
