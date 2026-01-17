"""数据存储模块"""

from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger


class Storage:
    """数据存储类
    
    负责数据的持久化存储，优先使用Parquet格式
    支持按日期分区存储raw和clean数据
    """
    
    def __init__(self, root_path: str = "./data"):
        """初始化存储
        
        Args:
            root_path: 数据根目录
        
        注意：
            - trade_cal和stock_basic使用单文件存储（不分区）
            - daily/daily_basic/adj_factor/suspend/stk_limit等使用按日期分区存储
            - clean层数据使用按日期分区存储
            - features层数据使用按日期分区存储
        """
        self.root_path = Path(root_path)
        self.raw_path = self.root_path / "raw"
        self.clean_path = self.root_path / "clean"
        self.features_path = self.root_path / "features"
        self.reports_path = self.root_path / "reports"
        
        # 确保目录存在
        for path in [self.raw_path, self.clean_path, self.features_path, self.reports_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"数据存储初始化完成，根目录: {self.root_path}，使用partitioned存储模式")
    
    def save_raw(self, df: pd.DataFrame, name: str, format: str = "parquet", is_force: bool = False) -> None:
        """保存原始数据
        
        Args:
            df: 数据DataFrame
            name: 文件名（不含扩展名）
            format: 文件格式，parquet/csv
        """
        self._save_data(df, self.raw_path / name, format, is_force)
    
    def save_clean(self, df: pd.DataFrame, name: str, format: str = "parquet", is_force: bool = False) -> None:
        """保存清洗后数据
        
        Args:
            df: 数据DataFrame
            name: 文件名（不含扩展名）
            format: 文件格式，parquet/csv
        """
        self._save_data(df, self.clean_path / name, format, is_force)
    
    def save_features(self, df: pd.DataFrame, name: str, format: str = "parquet", is_force: bool = False) -> None:
        """保存特征数据
        
        Args:
            df: 数据DataFrame
            name: 文件名（不含扩展名）
            format: 文件格式，parquet/csv
        """
        self._save_data(df, self.features_path / name, format, is_force)
    
    def save_report(self, df: pd.DataFrame, name: str, format: str = "csv", is_force: bool = False) -> None:
        """保存报告数据
        
        Args:
            df: 数据DataFrame
            name: 文件名（不含扩展名）
            format: 文件格式，csv/parquet
        """
        self._save_data(df, self.reports_path / name, format, is_force)
    
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
    
    def save_raw_by_date(
        self,
        df: pd.DataFrame,
        name: str,
        trade_date: str,
        format: str = "parquet"
    ) -> None:
        """保存按日期分区的原始数据
        
        目录结构: data/raw/{name}/{YYYY-MM-DD}.parquet
        
        Args:
            df: 数据DataFrame
            name: 数据类型名称（如daily, daily_basic, suspend_d等）
            trade_date: 交易日期，格式YYYYMMDD或YYYY-MM-DD
            format: 文件格式，parquet/csv
        """
        # 转换日期格式为YYYY-MM-DD
        date_str = self._format_date(trade_date)
        
        # 创建分区目录
        partition_path = self.raw_path / name
        partition_path.mkdir(parents=True, exist_ok=True)
        
        # 保存数据
        self._save_data(df, partition_path / date_str, format)
    
    def load_raw_by_date(
        self,
        name: str,
        trade_date: str,
        format: str = "parquet"
    ) -> Optional[pd.DataFrame]:
        """加载按日期分区的原始数据
        
        Args:
            name: 数据类型名称
            trade_date: 交易日期，格式YYYYMMDD或YYYY-MM-DD
            format: 文件格式
            
        Returns:
            数据DataFrame，不存在返回None
        """
        # 转换日期格式
        date_str = self._format_date(trade_date)
        
        # 尝试从分区目录加载
        partition_path = self.raw_path / name / date_str
        return self._load_data(partition_path, format)
    
    def load_raw_by_date_range(
        self,
        name: str,
        start_date: str,
        end_date: str,
        format: str = "parquet"
    ) -> Optional[pd.DataFrame]:
        """加载日期范围内的原始数据
        
        Args:
            name: 数据类型名称
            start_date: 开始日期，格式YYYYMMDD或YYYY-MM-DD
            end_date: 结束日期，格式YYYYMMDD或YYYY-MM-DD
            format: 文件格式
            
        Returns:
            合并后的数据DataFrame，不存在返回None
        """
        partition_dir = self.raw_path / name
        
        if not partition_dir.exists():
            logger.warning(f"分区目录不存在: {partition_dir}")
            return None
        
        # 转换日期格式
        start_str = self._format_date(start_date)
        end_str = self._format_date(end_date)
        
        # 收集所有符合条件的文件
        dfs = []
        for file_path in sorted(partition_dir.glob(f"*.{format}")):
            date_part = file_path.stem  # 文件名（不含扩展名）
            if start_str <= date_part <= end_str:
                df = self._load_data(partition_dir / date_part, format)
                if df is not None:
                    dfs.append(df)
        
        if not dfs:
            logger.warning(f"没有找到符合日期范围的数据: {name} [{start_date}, {end_date}]")
            return None
        
        # 合并所有数据
        result = pd.concat(dfs, ignore_index=True)
        logger.info(f"加载了 {len(dfs)} 个分区文件，共 {len(result)} 条记录")
        return result
    
    def save_clean_by_date(
        self,
        df: pd.DataFrame,
        name: str,
        trade_date: str,
        format: str = "parquet"
    ) -> None:
        """保存按日期分区的清洗数据
        
        目录结构: data/clean/{name}/{YYYY-MM-DD}.parquet
        
        Args:
            df: 数据DataFrame
            name: 数据类型名称（如daily, daily_basic等）
            trade_date: 交易日期，格式YYYYMMDD或YYYY-MM-DD
            format: 文件格式，parquet/csv
        """
        # 转换日期格式为YYYY-MM-DD
        date_str = self._format_date(trade_date)
        
        # 创建分区目录
        partition_path = self.clean_path / name
        partition_path.mkdir(parents=True, exist_ok=True)
        
        # 保存数据
        self._save_data(df, partition_path / date_str, format)
    
    def load_clean_by_date(
        self,
        name: str,
        trade_date: str,
        format: str = "parquet"
    ) -> Optional[pd.DataFrame]:
        """加载按日期分区的清洗数据
        
        Args:
            name: 数据类型名称
            trade_date: 交易日期，格式YYYYMMDD或YYYY-MM-DD
            format: 文件格式
            
        Returns:
            数据DataFrame，不存在返回None
        """
        # 转换日期格式
        date_str = self._format_date(trade_date)
        
        # 尝试从分区目录加载
        partition_path = self.clean_path / name / date_str
        return self._load_data(partition_path, format)
    
    def load_clean_by_date_range(
        self,
        name: str,
        start_date: str,
        end_date: str,
        format: str = "parquet"
    ) -> Optional[pd.DataFrame]:
        """加载日期范围内的清洗数据
        
        Args:
            name: 数据类型名称
            start_date: 开始日期，格式YYYYMMDD或YYYY-MM-DD
            end_date: 结束日期，格式YYYYMMDD或YYYY-MM-DD
            format: 文件格式
            
        Returns:
            合并后的数据DataFrame，不存在返回None
        """
        partition_dir = self.clean_path / name
        
        if not partition_dir.exists():
            logger.warning(f"分区目录不存在: {partition_dir}")
            return None
        
        # 转换日期格式
        start_str = self._format_date(start_date)
        end_str = self._format_date(end_date)
        
        # 收集所有符合条件的文件
        dfs = []
        for file_path in sorted(partition_dir.glob(f"*.{format}")):
            date_part = file_path.stem  # 文件名（不含扩展名）
            if start_str <= date_part <= end_str:
                df = self._load_data(partition_dir / date_part, format)
                if df is not None:
                    dfs.append(df)
        
        if not dfs:
            logger.warning(f"没有找到符合日期范围的数据: {name} [{start_date}, {end_date}]")
            return None
        
        # 合并所有数据
        result = pd.concat(dfs, ignore_index=True)
        logger.info(f"加载了 {len(dfs)} 个分区文件，共 {len(result)} 条记录")
        return result
    
    def list_partitions(self, layer: str, name: str) -> List[str]:
        """列出某个数据类型的所有分区日期
        
        Args:
            layer: 数据层，'raw'或'clean'
            name: 数据类型名称
            
        Returns:
            日期列表（格式YYYY-MM-DD），按升序排序
        """
        if layer == "raw":
            partition_dir = self.raw_path / name
        elif layer == "clean":
            partition_dir = self.clean_path / name
        else:
            raise ValueError(f"不支持的数据层: {layer}")
        
        if not partition_dir.exists():
            return []
        
        # 收集所有.parquet文件的日期
        dates = []
        for file_path in partition_dir.glob("*.parquet"):
            date_str = file_path.stem
            # 验证日期格式（YYYY-MM-DD）
            if len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
                dates.append(date_str)
        
        # 同时检查.csv文件
        for file_path in partition_dir.glob("*.csv"):
            date_str = file_path.stem
            if len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
                if date_str not in dates:
                    dates.append(date_str)
        
        return sorted(dates)
    
    def _format_date(self, date_str: str) -> str:
        """统一日期格式为YYYY-MM-DD
        
        Args:
            date_str: 日期字符串，支持YYYYMMDD或YYYY-MM-DD
            
        Returns:
            格式化后的日期字符串YYYY-MM-DD
            
        Raises:
            ValueError: 如果日期格式无效
        """
        import re
        
        if len(date_str) == 8:  # YYYYMMDD
            # 验证格式
            if not re.match(r'^\d{8}$', date_str):
                raise ValueError(f"不支持的日期格式: {date_str}，YYYYMMDD格式应为8位数字")
            
            # 验证日期有效性
            try:
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                
                # 简单范围检查
                if not (1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31):
                    raise ValueError(f"日期值超出有效范围: {date_str}")
                    
                return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            except (ValueError, IndexError) as e:
                raise ValueError(f"无效的日期: {date_str}, 错误: {str(e)}")
                
        elif len(date_str) == 10:  # YYYY-MM-DD
            # 验证格式
            if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                raise ValueError(f"不支持的日期格式: {date_str}，YYYY-MM-DD格式应为YYYY-MM-DD")
            return date_str
        else:
            raise ValueError(f"不支持的日期格式: {date_str}，应为YYYYMMDD或YYYY-MM-DD")
    
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
    
    def check_basic_data_freshness(self, name: str, required_end_date: str) -> bool:
        """检查基础数据（trade_cal或stock_basic）是否足够新
        
        Args:
            name: 数据名称，'trade_cal'或'stock_basic'
            required_end_date: 需要的结束日期，格式YYYYMMDD
            
        Returns:
            True表示数据足够新，False表示需要更新
        """
        df = self.load_raw(name)
        if df is None:
            logger.info(f"{name} 数据不存在，需要下载")
            return False
        
        # 获取数据中的最新日期
        if name == "trade_cal":
            if 'cal_date' not in df.columns:
                logger.warning(f"{name} 缺少 cal_date 列")
                return False
            
            # 转换为字符串格式YYYYMMDD
            try:
                if pd.api.types.is_datetime64_any_dtype(df['cal_date']):
                    latest_date = df['cal_date'].max().strftime('%Y%m%d')
                else:
                    latest_date = str(df['cal_date'].max()).replace('-', '')
            except Exception as e:
                logger.warning(f"无法解析 {name} 的日期: {e}")
                return False
            
            logger.info(f"{name} 最新日期: {latest_date}, 需要日期: {required_end_date}")
            return latest_date >= required_end_date
            
        elif name == "stock_basic":
            # stock_basic不基于日期判断，而是检查是否存在
            # 可以根据数据更新频率（如每季度）来判断是否需要更新
            # 这里简化为：如果文件存在就认为足够新
            logger.info(f"{name} 数据已存在，记录数: {len(df)}")
            return True
        
        return False
    
    def is_feature_exists(self, trade_date: str, format: str = "parquet") -> bool:
        """判断特征数据是否存在
        
        Args:
            trade_date: 交易日期，格式YYYYMMDD
            format: 文件格式
            
        Returns:
            True表示存在，False表示不存在
        """
        cs_train_path = self.features_path / "cs_train"
        path = cs_train_path / trade_date
        
        if format == "parquet":
            file_path = path.with_suffix(".parquet")
        elif format == "csv":
            file_path = path.with_suffix(".csv")
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        return file_path.exists()

    def is_data_exists(self, layer: str, name: str, date: str, format: str = "parquet") -> bool:
        """判断文件是否存在
        
        Args:
            layer: 数据层，'raw'或'clean'
            name: 数据类型名称
            date: 交易日期，格式YYYYMMDD
            format: 文件格式
        """
        if layer == "raw":
            base_path = self.raw_path
        elif layer == "clean":
            base_path = self.clean_path
        else:
            raise ValueError(f"不支持的数据层: {layer}")
        path = base_path / name / self._format_date(date)
        
        if format == "parquet":
            file_path = path.with_suffix(".parquet")
        elif format == "csv":
            file_path = path.with_suffix(".csv")
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        return file_path.exists()
        
    def _save_data(self, df: pd.DataFrame, path: Path, format: str, is_force: bool = False) -> None:
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
