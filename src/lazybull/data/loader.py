"""数据加载模块"""

from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from .storage import Storage
from ..common.date_utils import to_trade_date_str, normalize_date_column


class DataLoader:
    """数据加载器
    
    提供标准化的数据加载接口
    """
    
    def __init__(self, storage: Optional[Storage] = None, verbose: bool = False):
        """初始化数据加载器
        
        Args:
            storage: 存储实例，如不提供则创建默认实例
            verbose: 是否输出详细日志
        """
        self.storage = storage or Storage(verbose=verbose)
        self.verbose = verbose
    
    def load_trade_cal(self) -> Optional[pd.DataFrame]:
        """加载交易日历
        
        Returns:
            交易日历DataFrame
        """
        df = self.storage.load_raw("trade_cal")
        if df is not None:
            # 转换日期格式
            if 'cal_date' in df.columns:
                df['cal_date'] = pd.to_datetime(df['cal_date'], format='%Y%m%d')
            if 'pretrade_date' in df.columns:
                df['pretrade_date'] = pd.to_datetime(df['pretrade_date'], format='%Y%m%d', errors='coerce')
        return df
    
    def load_stock_basic(self) -> Optional[pd.DataFrame]:
        """加载股票基本信息
        
        Returns:
            股票基本信息DataFrame
        """
        df = self.storage.load_raw("stock_basic")
        if df is not None:
            # 转换日期格式
            if 'list_date' in df.columns:
                df['list_date'] = pd.to_datetime(df['list_date'], format='%Y%m%d', errors='coerce')
        return df
    
    def load_daily(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """加载日线行情数据
        
        优先尝试从分区数据加载（如果提供了日期范围），否则加载完整数据
        
        Args:
            start_date: 开始日期，格式YYYY-MM-DD或YYYYMMDD
            end_date: 结束日期，格式YYYY-MM-DD或YYYYMMDD
            
        Returns:
            日线行情DataFrame
        """
        # 如果提供了日期范围，尝试从分区加载
        if start_date and end_date:
            # 转换日期格式
            start_str = self._normalize_date(start_date)
            end_str = self._normalize_date(end_date)
            
            # 尝试从分区加载
            df = self.storage.load_raw_by_date_range("daily", start_str, end_str)
            
            if df is not None:
                # 转换日期格式
                if 'trade_date' in df.columns:
                    # 尝试从YYYYMMDD格式转换
                    try:
                        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
                    except (ValueError, TypeError):
                        # 如果失败，可能已经是datetime格式
                        df['trade_date'] = pd.to_datetime(df['trade_date'])
                return df
        
        # 回退到加载完整数据
        df = self.storage.load_raw("daily")
        if df is None:
            return None
        
        # 确保日期类型一致（转换为 datetime 以便比较）
        if 'trade_date' in df.columns:
            df = normalize_date_column(df, 'trade_date', to_str=False)
        
        # 日期过滤（统一为 datetime 类型比较）
        if start_date:
            start_dt = pd.to_datetime(self._normalize_date(start_date))
            df = df[df['trade_date'] >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(self._normalize_date(end_date))
            df = df[df['trade_date'] <= end_dt]
        
        return df
    
    def load_daily_basic(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """加载每日指标数据
        
        优先尝试从分区数据加载（如果提供了日期范围），否则加载完整数据
        
        Args:
            start_date: 开始日期，格式YYYY-MM-DD或YYYYMMDD
            end_date: 结束日期，格式YYYY-MM-DD或YYYYMMDD
            
        Returns:
            每日指标DataFrame
        """
        # 如果提供了日期范围，尝试从分区加载
        if start_date and end_date:
            # 转换日期格式
            start_str = self._normalize_date(start_date)
            end_str = self._normalize_date(end_date)
            
            # 尝试从分区加载
            df = self.storage.load_raw_by_date_range("daily_basic", start_str, end_str)
            
            if df is not None:
                # 转换日期格式
                if 'trade_date' in df.columns:
                    try:
                        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
                    except (ValueError, TypeError):
                        df['trade_date'] = pd.to_datetime(df['trade_date'])
                return df
        
        # 回退到加载完整数据
        df = self.storage.load_raw("daily_basic")
        if df is None:
            return None
        
        # 确保日期类型一致（转换为 datetime 以便比较）
        if 'trade_date' in df.columns:
            df = normalize_date_column(df, 'trade_date', to_str=False)
        
        # 日期过滤（统一为 datetime 类型比较）
        if start_date:
            start_dt = pd.to_datetime(self._normalize_date(start_date))
            df = df[df['trade_date'] >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(self._normalize_date(end_date))
            df = df[df['trade_date'] <= end_dt]
        
        return df
    
    def _normalize_date(self, date_str: str) -> str:
        """标准化日期格式为YYYY-MM-DD
        
        Args:
            date_str: 日期字符串，支持YYYYMMDD或YYYY-MM-DD
            
        Returns:
            标准化后的日期字符串
        """
        if len(date_str) == 8:  # YYYYMMDD
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        return date_str

    def _load_quarter_partitioned_raw(
        self,
        name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lookback_years: int = 1,
        columns: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        """加载按季度分区的公告型原始数据。"""
        if not hasattr(self.storage, "list_partitions"):
            return None
        partitions = self.storage.list_partitions("raw", name)
        if not partitions:
            return None

        if start_date and end_date:
            start_norm = self._normalize_date(start_date)
            end_norm = self._normalize_date(end_date)
            start_year = pd.to_datetime(start_norm).year - lookback_years
            range_start = f"{start_year}-01-01"
            range_end = end_norm
        else:
            range_start = partitions[0]
            range_end = partitions[-1]

        return self.storage.load_raw_by_date_range(
            name,
            range_start,
            range_end,
            columns=columns,
        )
    
    def get_trading_dates(self, start_date: str, end_date: str) -> list:
        """获取指定范围内的交易日列表
        
        Args:
            start_date: 开始日期，格式YYYY-MM-DD
            end_date: 结束日期，格式YYYY-MM-DD
            
        Returns:
            交易日期列表
        """
        df = self.load_trade_cal()
        if df is None:
            logger.warning("交易日历未加载，返回空列表")
            return []
        
        # 筛选交易日
        mask = (
            (df['cal_date'] >= pd.to_datetime(start_date)) &
            (df['cal_date'] <= pd.to_datetime(end_date)) &
            (df['is_open'] == 1)
        )
        
        trading_dates = df[mask]['cal_date'].tolist()
        return sorted(trading_dates)
    
    def load_clean_daily(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """加载清洗后的日线行情数据
        
        优先尝试从分区数据加载（如果提供了日期范围），否则加载完整数据
        
        Args:
            start_date: 开始日期，格式YYYY-MM-DD或YYYYMMDD
            end_date: 结束日期，格式YYYY-MM-DD或YYYYMMDD
            
        Returns:
            日线行情DataFrame（包含复权价格列）
        """
        # 如果提供了日期范围，尝试从分区加载
        if start_date and end_date:
            # 转换日期格式
            start_str = self._normalize_date(start_date)
            end_str = self._normalize_date(end_date)
            
            # 尝试从分区加载
            df = self.storage.load_clean_by_date_range("daily", start_str, end_str)
            
            if df is not None:
                # 确保日期格式一致（YYYYMMDD字符串）
                if 'trade_date' in df.columns:
                    # 如果是 datetime，转换为 YYYYMMDD
                    if pd.api.types.is_datetime64_any_dtype(df['trade_date']):
                        df['trade_date'] = df['trade_date'].dt.strftime('%Y%m%d')
                return df
        
        # 回退到加载完整数据
        df = self.storage.load_clean("daily")
        if df is None:
            return None
        
        # 确保日期格式一致
        if 'trade_date' in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df['trade_date']):
                df['trade_date'] = df['trade_date'].dt.strftime('%Y%m%d')
        
        # 日期过滤
        if start_date:
            start_dt = self._normalize_date(start_date).replace('-', '')
            df = df[df['trade_date'] >= start_dt]
        if end_date:
            end_dt = self._normalize_date(end_date).replace('-', '')
            df = df[df['trade_date'] <= end_dt]
        
        return df
    
    def load_clean_daily_basic(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """加载清洗后的每日指标数据
        
        Args:
            start_date: 开始日期，格式YYYY-MM-DD或YYYYMMDD
            end_date: 结束日期，格式YYYY-MM-DD或YYYYMMDD
            
        Returns:
            每日指标DataFrame
        """
        # 如果提供了日期范围，尝试从分区加载
        if start_date and end_date:
            start_str = self._normalize_date(start_date)
            end_str = self._normalize_date(end_date)
            
            df = self.storage.load_clean_by_date_range("daily_basic", start_str, end_str)
            
            if df is not None:
                if 'trade_date' in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df['trade_date']):
                        df['trade_date'] = df['trade_date'].dt.strftime('%Y%m%d')
                return df
        
        # 回退到加载完整数据
        df = self.storage.load_clean("daily_basic")
        if df is None:
            return None
        
        # 确保日期格式一致
        if 'trade_date' in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df['trade_date']):
                df['trade_date'] = df['trade_date'].dt.strftime('%Y%m%d')
        
        # 日期过滤
        if start_date:
            start_dt = self._normalize_date(start_date).replace('-', '')
            df = df[df['trade_date'] >= start_dt]
        if end_date:
            end_dt = self._normalize_date(end_date).replace('-', '')
            df = df[df['trade_date'] <= end_dt]
        
        return df
    
    def load_clean_trade_cal(self) -> Optional[pd.DataFrame]:
        """加载清洗后的交易日历
        
        Returns:
            交易日历DataFrame
        """
        df = self.storage.load_clean("trade_cal")
        if df is not None:
            # 保持日期为字符串格式（YYYYMMDD）
            # clean 层已经标准化为 YYYYMMDD 字符串
            pass
        return df
    
    def load_clean_stock_basic(self) -> Optional[pd.DataFrame]:
        """加载清洗后的股票基本信息
        
        Returns:
            股票基本信息DataFrame
        """
        df = self.storage.load_clean("stock_basic")
        return df
    
    def load_clean_moneyflow(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """加载清洗后的资金流向数据
        
        Args:
            start_date: 开始日期，格式YYYY-MM-DD或YYYYMMDD
            end_date: 结束日期，格式YYYY-MM-DD或YYYYMMDD
            
        Returns:
            资金流向DataFrame
        """
        # 如果提供了日期范围，尝试从分区加载
        if start_date and end_date:
            start_str = self._normalize_date(start_date)
            end_str = self._normalize_date(end_date)
            
            df = self.storage.load_clean_by_date_range("moneyflow", start_str, end_str)
            
            if df is not None:
                if 'trade_date' in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df['trade_date']):
                        df['trade_date'] = df['trade_date'].dt.strftime('%Y%m%d')
                return df
        
        # 回退到加载完整数据
        df = self.storage.load_clean("moneyflow")
        if df is None:
            return None
        
        # 确保日期格式一致
        if 'trade_date' in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df['trade_date']):
                df['trade_date'] = df['trade_date'].dt.strftime('%Y%m%d')
        
        # 日期过滤
        if start_date:
            start_dt = self._normalize_date(start_date).replace('-', '')
            df = df[df['trade_date'] >= start_dt]
        if end_date:
            end_dt = self._normalize_date(end_date).replace('-', '')
            df = df[df['trade_date'] <= end_dt]
        
        return df
    
    def load_clean_daily_by_date(self, trade_date: str) -> Optional[pd.DataFrame]:
        """加载指定日期的清洗后日线数据
        
        Args:
            trade_date: 交易日期 YYYYMMDD
            
        Returns:
            日线数据DataFrame
        """
        # 转换日期格式 YYYYMMDD -> YYYY-MM-DD
        if len(trade_date) == 8:
            date_str = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"
        else:
            date_str = trade_date
        
        # 确保已存在清洗过的数据, 若不存在则下载创建
        from .ensure import ensure_clean_data_for_date
        from ..data.tushare_client import TushareClient
        from ..data.cleaner import DataCleaner
        loader = DataLoader(verbose=False)
        ts = TushareClient(verbose=False)
        cleaner = DataCleaner(verbose=False)
        ensure_clean_data_for_date(self.storage, loader, cleaner, ts, trade_date)
        
        df = self.storage.load_clean_by_date("daily", date_str)
        
        if df is not None and 'trade_date' in df.columns:
            # 确保日期格式一致（YYYYMMDD字符串）
            if pd.api.types.is_datetime64_any_dtype(df['trade_date']):
                df['trade_date'] = df['trade_date'].dt.strftime('%Y%m%d')
        
        return df
    
    def load_shenwan_industry(self) -> Optional[pd.DataFrame]:
        """加载申万行业分类数据

        Returns:
            申万行业分类DataFrame，包含 ts_code, sw_code, sw_name 等字段
        """
        df = self.storage.load_raw("shenwan_industry")
        if df is None:
            logger.warning(
                "未找到申万行业分类数据！\n"
            )
        return df

    def load_fina_indicator(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lookback_years: int = 1,
    ) -> Optional[pd.DataFrame]:
        """加载财务指标数据（fina_indicator）

        Returns:
            财务指标DataFrame，包含 ts_code, ann_date, end_date, roe_waa 等字段。
            不存在返回 None。
        """
        df = self._load_quarter_partitioned_raw(
            "fina_indicator",
            start_date=start_date,
            end_date=end_date,
            lookback_years=lookback_years,
        )
        if df is None:
            df = self.storage.load_raw("fina_indicator")
        if df is None:
            logger.warning(
                "未找到财务指标数据！\n"
                "请先运行: python scripts/download_fina_indicator.py"
            )
            return None

        # 日期列标准化为 YYYYMMDD 字符串
        for col in ['ann_date', 'end_date']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('-', '').str[:8]

        return df

    def load_margin_detail(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """加载融资融券明细数据（按日分区存储）"""
        if start_date and end_date:
            start_str = self._normalize_date(start_date)
            end_str = self._normalize_date(end_date)
            df = self.storage.load_raw_by_date_range("margin_detail", start_str, end_str)
            if df is not None and "trade_date" in df.columns:
                df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "").str[:8]
            return df
        df = self.storage.load_raw("margin_detail")
        return df

    def load_stk_holdernumber(self) -> Optional[pd.DataFrame]:
        """加载股东人数数据（单文件）"""
        df = self.storage.load_raw("stk_holdernumber")
        if df is None:
            logger.warning("未找到股东人数数据")
        else:
            for col in ["ann_date", "end_date"]:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace("-", "").str[:8]
        return df

    def load_forecast(self) -> Optional[pd.DataFrame]:
        """加载业绩预告数据（单文件）"""
        df = self.storage.load_raw("forecast")
        if df is None:
            logger.warning("未找到业绩预告数据")
        else:
            for col in ["ann_date", "end_date"]:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace("-", "").str[:8]
        return df

    def load_cyq_perf(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """加载筹码胜率数据（按日分区存储）"""
        if start_date and end_date:
            start_str = self._normalize_date(start_date)
            end_str = self._normalize_date(end_date)
            df = self.storage.load_raw_by_date_range("cyq_perf", start_str, end_str)
            if df is not None and "trade_date" in df.columns:
                df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "").str[:8]
            return df
        # 兼容：无日期参数时尝试加载单文件（旧格式）
        df = self.storage.load_raw("cyq_perf")
        if df is None:
            logger.warning("未找到筹码胜率数据")
        else:
            if "trade_date" in df.columns:
                df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "").str[:8]
        return df

    def load_express(self) -> Optional[pd.DataFrame]:
        """加载业绩快报数据（单文件）"""
        df = self.storage.load_raw("express")
        if df is None:
            logger.warning("未找到业绩快报数据")
        else:
            for col in ["ann_date", "end_date"]:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace("-", "").str[:8]
        return df

    def load_cashflow(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lookback_years: int = 1,
    ) -> Optional[pd.DataFrame]:
        """加载现金流量表数据（单文件）"""
        df = self._load_quarter_partitioned_raw(
            "cashflow",
            start_date=start_date,
            end_date=end_date,
            lookback_years=lookback_years,
        )
        if df is None:
            df = self.storage.load_raw("cashflow")
        if df is None:
            logger.warning("未找到现金流量表数据")
        else:
            for col in ["ann_date", "end_date", "f_ann_date"]:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace("-", "").str[:8]
        return df

    def load_moneyflow_hsgt(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """加载沪深股通资金流向（按日分区存储）"""
        if start_date and end_date:
            start_str = self._normalize_date(start_date)
            end_str = self._normalize_date(end_date)
            df = self.storage.load_raw_by_date_range(
                "moneyflow_hsgt", start_str, end_str
            )
        else:
            df = self.storage.load_raw("moneyflow_hsgt")
        if df is None:
            logger.warning("未找到北向资金数据")
        elif "trade_date" in df.columns:
            df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "").str[:8]
        return df

    def load_top_list(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """加载龙虎榜数据（按日分区存储）"""
        if start_date and end_date:
            start_str = self._normalize_date(start_date)
            end_str = self._normalize_date(end_date)
            df = self.storage.load_raw_by_date_range("top_list", start_str, end_str)
        else:
            df = self.storage.load_raw("top_list")
        if df is None:
            logger.warning("未找到龙虎榜数据")
        elif "trade_date" in df.columns:
            df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "").str[:8]
        return df

    def load_report_rc(self) -> Optional[pd.DataFrame]:
        """加载卖方一致预期研报数据（单文件）"""
        df = self.storage.load_raw("report_rc")
        if df is None:
            logger.warning("未找到一致预期研报数据")
        elif "report_date" in df.columns:
            df["report_date"] = (
                df["report_date"].astype(str).str.replace("-", "").str[:8]
            )
        return df

    def load_fund_portfolio(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """加载基金持仓数据（按季度分区存储）"""
        if start_date and end_date:
            start_str = self._normalize_date(start_date)
            end_str = self._normalize_date(end_date)
            df = self.storage.load_raw_by_date_range("fund_portfolio", start_str, end_str)
        else:
            # 兼容：无日期参数时尝试加载单文件（旧格式）
            df = self.storage.load_raw("fund_portfolio")
        if df is None:
            logger.warning("未找到基金持仓数据")
        else:
            for col in ["ann_date", "end_date"]:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace("-", "").str[:8]
        return df

    def build_stock_names_dict(self) -> Dict[str, str]:
        """从 stock_basic 构建 {ts_code: name} 股票名称字典"""
        stock_names: Dict[str, str] = {}

        stock_basic = self.load_clean_stock_basic()
        if stock_basic is None or stock_basic.empty:
            stock_basic = self.load_stock_basic()

        if stock_basic is None or stock_basic.empty:
            logger.warning("无法加载 stock_basic 数据")
            return stock_names

        if 'ts_code' not in stock_basic.columns or 'name' not in stock_basic.columns:
            logger.warning("stock_basic 数据缺少必要列（ts_code 或 name）")
            return stock_names

        for _, row in stock_basic.iterrows():
            if pd.notna(row.get('ts_code')) and pd.notna(row.get('name')) and row['name']:
                stock_names[row['ts_code']] = row['name']

        return stock_names
