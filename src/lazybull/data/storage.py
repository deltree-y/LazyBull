"""数据存储模块"""

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


class Storage:
    """数据存储类
    
    负责数据的持久化存储，优先使用Parquet格式
    """
    
    def __init__(self, root_path: str = "./data"):
        """初始化存储
        
        Args:
            root_path: 数据根目录
        """
        self.root_path = Path(root_path)
        self.raw_path = self.root_path / "raw"
        self.clean_path = self.root_path / "clean"
        self.features_path = self.root_path / "features"
        self.reports_path = self.root_path / "reports"
        
        # 确保目录存在
        for path in [self.raw_path, self.clean_path, self.features_path, self.reports_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"数据存储初始化完成，根目录: {self.root_path}")
    
    def save_raw(self, df: pd.DataFrame, name: str, format: str = "parquet") -> None:
        """保存原始数据
        
        Args:
            df: 数据DataFrame
            name: 文件名（不含扩展名）
            format: 文件格式，parquet/csv
        """
        self._save_data(df, self.raw_path / name, format)
    
    def save_clean(self, df: pd.DataFrame, name: str, format: str = "parquet") -> None:
        """保存清洗后数据
        
        Args:
            df: 数据DataFrame
            name: 文件名（不含扩展名）
            format: 文件格式，parquet/csv
        """
        self._save_data(df, self.clean_path / name, format)
    
    def save_features(self, df: pd.DataFrame, name: str, format: str = "parquet") -> None:
        """保存特征数据
        
        Args:
            df: 数据DataFrame
            name: 文件名（不含扩展名）
            format: 文件格式，parquet/csv
        """
        self._save_data(df, self.features_path / name, format)
    
    def save_report(self, df: pd.DataFrame, name: str, format: str = "csv") -> None:
        """保存报告数据
        
        Args:
            df: 数据DataFrame
            name: 文件名（不含扩展名）
            format: 文件格式，csv/parquet
        """
        self._save_data(df, self.reports_path / name, format)
    
    def load_raw(self, name: str, format: str = "parquet") -> Optional[pd.DataFrame]:
        """加载原始数据
        
        Args:
            name: 文件名（不含扩展名）
            format: 文件格式
            
        Returns:
            数据DataFrame，不存在返回None
        """
        return self._load_data(self.raw_path / name, format)
    
    def load_clean(self, name: str, format: str = "parquet") -> Optional[pd.DataFrame]:
        """加载清洗后数据
        
        Args:
            name: 文件名（不含扩展名）
            format: 文件格式
            
        Returns:
            数据DataFrame，不存在返回None
        """
        return self._load_data(self.clean_path / name, format)
    
    def load_features(self, name: str, format: str = "parquet") -> Optional[pd.DataFrame]:
        """加载特征数据
        
        Args:
            name: 文件名（不含扩展名）
            format: 文件格式
            
        Returns:
            数据DataFrame，不存在返回None
        """
        return self._load_data(self.features_path / name, format)
    
    def save_cs_train_day(self, df: pd.DataFrame, trade_date: str, format: str = "parquet") -> None:
        """保存单日截面训练数据
        
        Args:
            df: 数据DataFrame
            trade_date: 交易日期，格式YYYYMMDD
            format: 文件格式，parquet/csv
        """
        cs_train_path = self.features_path / "cs_train"
        cs_train_path.mkdir(parents=True, exist_ok=True)
        self._save_data(df, cs_train_path / trade_date, format)
    
    def load_cs_train_day(self, trade_date: str, format: str = "parquet") -> Optional[pd.DataFrame]:
        """加载单日截面训练数据
        
        Args:
            trade_date: 交易日期，格式YYYYMMDD
            format: 文件格式
            
        Returns:
            数据DataFrame，不存在返回None
        """
        cs_train_path = self.features_path / "cs_train"
        return self._load_data(cs_train_path / trade_date, format)
    
    def _save_data(self, df: pd.DataFrame, path: Path, format: str) -> None:
        """保存数据
        
        Args:
            df: 数据DataFrame
            path: 文件路径（不含扩展名）
            format: 文件格式
        """
        if format == "parquet":
            file_path = path.with_suffix(".parquet")
            df.to_parquet(file_path, index=False)
        elif format == "csv":
            file_path = path.with_suffix(".csv")
            df.to_csv(file_path, index=False, encoding="utf-8-sig")
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        logger.info(f"数据已保存: {file_path} ({len(df)} 条记录)")
    
    def _load_data(self, path: Path, format: str) -> Optional[pd.DataFrame]:
        """加载数据
        
        Args:
            path: 文件路径（不含扩展名）
            format: 文件格式
            
        Returns:
            数据DataFrame，不存在返回None
        """
        if format == "parquet":
            file_path = path.with_suffix(".parquet")
        elif format == "csv":
            file_path = path.with_suffix(".csv")
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        if not file_path.exists():
            logger.warning(f"文件不存在: {file_path}")
            return None
        
        try:
            if format == "parquet":
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path)
            
            logger.info(f"数据已加载: {file_path} ({len(df)} 条记录)")
            return df
        except Exception as e:
            logger.error(f"加载数据失败: {file_path}, 错误: {str(e)}")
            return None
