"""数据加载模块"""

import warnings
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from .storage import Storage
from .loader_announcement import AnnouncementRiskLoaderMixin
from ..common.date_utils import normalize_series_to_yyyymmdd, normalize_to_yyyymmdd


class DataLoader(AnnouncementRiskLoaderMixin):
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
            # 统一返回 YYYYMMDD 字符串
            if "cal_date" in df.columns:
                df["cal_date"] = normalize_series_to_yyyymmdd(df["cal_date"])
            if "pretrade_date" in df.columns:
                df["pretrade_date"] = normalize_series_to_yyyymmdd(df["pretrade_date"])
        return df

    def load_stock_basic(self) -> Optional[pd.DataFrame]:
        """加载股票基本信息

        Returns:
            股票基本信息DataFrame
        """
        df = self.storage.load_raw("stock_basic")
        if df is not None:
            # 统一返回 YYYYMMDD 字符串
            if "list_date" in df.columns:
                df["list_date"] = normalize_series_to_yyyymmdd(df["list_date"])
        return df

    def load_daily(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
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
                if "trade_date" in df.columns:
                    df["trade_date"] = normalize_series_to_yyyymmdd(df["trade_date"])
                return df

        # 回退到加载完整数据
        df = self.storage.load_raw("daily")
        if df is None:
            return None

        # 统一日期类型为 YYYYMMDD 字符串
        if "trade_date" in df.columns:
            df["trade_date"] = normalize_series_to_yyyymmdd(df["trade_date"])

        # 日期过滤（字符串比较）
        if start_date:
            start_dt = self._normalize_date(start_date)
            df = df[df["trade_date"] >= start_dt]
        if end_date:
            end_dt = self._normalize_date(end_date)
            df = df[df["trade_date"] <= end_dt]

        return df

    def load_daily_basic(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
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
                if "trade_date" in df.columns:
                    df["trade_date"] = normalize_series_to_yyyymmdd(df["trade_date"])
                return df

        # 回退到加载完整数据
        df = self.storage.load_raw("daily_basic")
        if df is None:
            return None

        # 统一日期类型为 YYYYMMDD 字符串
        if "trade_date" in df.columns:
            df["trade_date"] = normalize_series_to_yyyymmdd(df["trade_date"])

        # 日期过滤（字符串比较）
        if start_date:
            start_dt = self._normalize_date(start_date)
            df = df[df["trade_date"] >= start_dt]
        if end_date:
            end_dt = self._normalize_date(end_date)
            df = df[df["trade_date"] <= end_dt]

        return df

    def _normalize_date(self, date_str: str) -> str:
        """标准化日期格式为 YYYYMMDD

        Args:
            date_str: 日期字符串，支持YYYYMMDD或YYYY-MM-DD

        Returns:
            标准化后的日期字符串（YYYYMMDD）
        """
        normalized = normalize_to_yyyymmdd(date_str)
        if normalized is None:
            raise ValueError(f"不支持的日期格式: {date_str}")
        return normalized

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
            start_year = pd.to_datetime(start_norm, format="%Y%m%d").year - lookback_years
            range_start = f"{start_year}-01-01"
            range_end = f"{end_norm[:4]}-{end_norm[4:6]}-{end_norm[6:8]}"
        else:
            range_start = partitions[0]
            range_end = partitions[-1]

        df = self.storage.load_raw_by_date_range(
            name,
            range_start,
            range_end,
            columns=columns,
        )
        # freshness 契约：窗口外仍有最新公告的股票保留"旧值 + 大 freshness"，
        # 而非直接 NaN —— 从窗口起点之前最近的分区补充其最新一条公告。
        # 窗口内无任何分区时同样执行（df 为 None），否则目标窗口外的股票会整体变 NaN。
        if start_date and end_date:
            extra = self._load_pre_window_latest_rows(
                name, df, range_start, partitions, columns=columns
            )
            if extra is not None and len(extra) > 0:
                if df is not None and len(df) > 0:
                    df = pd.concat([df, extra], ignore_index=True)
                else:
                    df = extra
        return df

    def _load_pre_window_latest_rows(
        self,
        name: str,
        window_df: Optional[pd.DataFrame],
        range_start: str,
        partitions: List[str],
        columns: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        """从窗口起点之前的分区，补充窗口内缺失股票的最新一条公告。

        保持 freshness 契约：对窗口外仍有历史公告的股票，保留其最近一条记录
        （对应其最新报告期），由 PIT 查询按 ann_date 对齐并输出大 freshness，
        而非在数据层直接硬缺失(NaN)。窗口内已覆盖的股票不重复加载。
        完整遍历窗口前分区（不做启发式截断），确保能找到更早的有效股票。

        Args:
            name: 数据集名称
            window_df: 窗口内已加载的数据（用于识别已覆盖股票）；可为 None/空
                （此时窗口内无数据，窗口前所有股票均视为窗口外）
            range_start: 窗口起点（YYYY-MM-DD）
            partitions: 全部分区日期列表（升序）
            columns: 仅读取指定列（与窗口加载一致）

        Returns:
            窗口外股票的最新公告数据；无则返回 None
        """
        if window_df is not None and len(window_df) > 0 and "ts_code" not in window_df.columns:
            return None
        window_codes = (
            set(window_df["ts_code"].unique())
            if window_df is not None and len(window_df) > 0
            else set()
        )
        pre_partitions = [p for p in partitions if p < range_start]
        if not pre_partitions:
            return None

        seen = set(window_codes)
        extras: List[pd.DataFrame] = []
        for part_date in reversed(pre_partitions):
            df = self.storage.load_raw_by_date(name, part_date, columns=columns)
            if df is None or len(df) == 0 or "ts_code" not in df.columns:
                continue
            new_rows = df[~df["ts_code"].isin(seen)]
            if len(new_rows) > 0:
                extras.append(new_rows)
                seen |= set(new_rows["ts_code"].unique())
        if not extras:
            return None
        return pd.concat(extras, ignore_index=True)

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
        start_str = self._normalize_date(start_date)
        end_str = self._normalize_date(end_date)
        mask = (df["cal_date"] >= start_str) & (df["cal_date"] <= end_str) & (df["is_open"] == 1)

        trading_dates = [d for d in df[mask]["cal_date"].tolist() if d is not None]
        return sorted(trading_dates)

    def load_clean_daily(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
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
                if "trade_date" in df.columns:
                    df["trade_date"] = normalize_series_to_yyyymmdd(df["trade_date"])
                return df

        # 回退到加载完整数据
        df = self.storage.load_clean("daily")
        if df is None:
            return None

        # 确保日期格式一致
        if "trade_date" in df.columns:
            df["trade_date"] = normalize_series_to_yyyymmdd(df["trade_date"])

        # 日期过滤
        if start_date:
            start_dt = self._normalize_date(start_date)
            df = df[df["trade_date"] >= start_dt]
        if end_date:
            end_dt = self._normalize_date(end_date)
            df = df[df["trade_date"] <= end_dt]

        return df

    def load_clean_daily_basic(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
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
                if "trade_date" in df.columns:
                    df["trade_date"] = normalize_series_to_yyyymmdd(df["trade_date"])
                return df

        # 回退到加载完整数据
        df = self.storage.load_clean("daily_basic")
        if df is None:
            return None

        # 确保日期格式一致
        if "trade_date" in df.columns:
            df["trade_date"] = normalize_series_to_yyyymmdd(df["trade_date"])

        # 日期过滤
        if start_date:
            start_dt = self._normalize_date(start_date)
            df = df[df["trade_date"] >= start_dt]
        if end_date:
            end_dt = self._normalize_date(end_date)
            df = df[df["trade_date"] <= end_dt]

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
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
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
                if "trade_date" in df.columns:
                    df["trade_date"] = normalize_series_to_yyyymmdd(df["trade_date"])
                return df

        # 回退到加载完整数据
        df = self.storage.load_clean("moneyflow")
        if df is None:
            return None

        # 确保日期格式一致
        if "trade_date" in df.columns:
            df["trade_date"] = normalize_series_to_yyyymmdd(df["trade_date"])

        # 日期过滤
        if start_date:
            start_dt = self._normalize_date(start_date)
            df = df[df["trade_date"] >= start_dt]
        if end_date:
            end_dt = self._normalize_date(end_date)
            df = df[df["trade_date"] <= end_dt]

        return df

    def load_clean_daily_by_date(
        self,
        trade_date: str,
        auto_ensure: bool = False,
        ensure_loader: Optional["DataLoader"] = None,
        ensure_cleaner: Optional["DataCleaner"] = None,
        ensure_client: Optional["TushareClient"] = None,
    ) -> Optional[pd.DataFrame]:
        """加载指定日期的清洗后日线数据

        Args:
            trade_date: 交易日期 YYYYMMDD

        Returns:
            日线数据DataFrame
        """
        normalized_trade_date = self._normalize_date(trade_date)

        if auto_ensure:
            from .ensure import ensure_clean_data_for_date
            from ..data.tushare_client import TushareClient
            from ..data.cleaner import DataCleaner

            _loader = ensure_loader or self
            _cleaner = ensure_cleaner or DataCleaner(verbose=False)
            _client = ensure_client or TushareClient(verbose=False)
            ensure_clean_data_for_date(
                self.storage,
                _loader,
                _cleaner,
                _client,
                normalized_trade_date,
            )

        date_str = (
            f"{normalized_trade_date[:4]}-{normalized_trade_date[4:6]}-{normalized_trade_date[6:8]}"
        )
        df = self.storage.load_clean_by_date("daily", date_str)

        if df is not None and "trade_date" in df.columns:
            df["trade_date"] = normalize_series_to_yyyymmdd(df["trade_date"])

        return df

    def load_shenwan_industry(self) -> Optional[pd.DataFrame]:
        """加载申万行业分类数据

        Returns:
            申万行业分类DataFrame，包含 ts_code, sw_code, sw_name 等字段
        """
        df = self.storage.load_raw("shenwan_industry")
        if df is None:
            logger.warning("未找到申万行业分类数据！\n")
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
                "未找到财务指标数据！\n" "请先运行: python scripts/download_fina_indicator.py"
            )
            return None

        # 日期列标准化为 YYYYMMDD 字符串
        for col in ["ann_date", "end_date"]:
            if col in df.columns:
                df[col] = normalize_series_to_yyyymmdd(df[col])

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
                df["trade_date"] = normalize_series_to_yyyymmdd(df["trade_date"])
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
                    df[col] = normalize_series_to_yyyymmdd(df[col])
        return df

    def load_forecast(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lookback_years: int = 1,
    ) -> Optional[pd.DataFrame]:
        """加载业绩预告数据（按季度 end_date 分区存储）

        Returns:
            业绩预告DataFrame，包含 ts_code, ann_date, end_date 等字段。
            不存在返回 None。
        """
        df = self._load_quarter_partitioned_raw(
            "forecast",
            start_date=start_date,
            end_date=end_date,
            lookback_years=lookback_years,
        )
        if df is None:
            logger.warning("未找到业绩预告数据")
        else:
            for col in ["ann_date", "end_date"]:
                if col in df.columns:
                    df[col] = normalize_series_to_yyyymmdd(df[col])
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
                df["trade_date"] = normalize_series_to_yyyymmdd(df["trade_date"])
            return df
        # 兼容：无日期参数时尝试加载单文件（旧格式）
        df = self.storage.load_raw("cyq_perf")
        if df is None:
            logger.warning("未找到筹码胜率数据")
        else:
            if "trade_date" in df.columns:
                df["trade_date"] = normalize_series_to_yyyymmdd(df["trade_date"])
        return df

    def load_express(self) -> Optional[pd.DataFrame]:
        """加载业绩快报数据（按季度 end_date 分区存储，兼容旧单文件自动迁移）"""
        df = self._load_quarter_partitioned_raw("express")
        # 旧单文件仍存在（含"部分分区 + 旧单文件"混合态）时先迁移合并，避免漏读旧数据；
        # 迁移不可用（空文件/缺分区列等异常旧文件）时保留已有分区数据，不遮蔽
        if self.storage.load_raw("express") is not None:
            migrated = self.storage.migrate_raw_single_file_to_partitions(
                "express",
                partition_date_col="end_date",
                dedup_cols=["ts_code", "end_date", "ann_date"],
            )
            if migrated is not None:
                df = migrated
        if df is None:
            logger.warning("未找到业绩快报数据")
        else:
            for col in ["ann_date", "end_date"]:
                if col in df.columns:
                    df[col] = normalize_series_to_yyyymmdd(df[col])
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
                    df[col] = normalize_series_to_yyyymmdd(df[col])
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
            df = self.storage.load_raw_by_date_range("moneyflow_hsgt", start_str, end_str)
        else:
            df = self.storage.load_raw("moneyflow_hsgt")
        if df is None:
            logger.warning("未找到北向资金数据")
        elif "trade_date" in df.columns:
            df["trade_date"] = normalize_series_to_yyyymmdd(df["trade_date"])
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
            df["trade_date"] = normalize_series_to_yyyymmdd(df["trade_date"])
        return df

    def load_report_rc(self) -> Optional[pd.DataFrame]:
        """加载卖方一致预期研报数据（按年 report_date 分区存储）"""
        partitions = self.storage.list_partitions("raw", "report_rc")
        if not partitions:
            logger.warning("未找到一致预期研报数据")
            return None
        dfs: List[pd.DataFrame] = []
        for p in partitions:
            df = self.storage.load_raw_by_date("report_rc", p)
            if df is not None and len(df) > 0:
                dfs.append(df)
        if not dfs:
            logger.warning("未找到一致预期研报数据")
            return None
        # 抑制 pandas 对 concat 含 all-NA 列时的 FutureWarning：
        # report_rc 按年分区存储，部分分区存在整列全 NA 的情况，
        # 该警告对结果无影响，仅屏蔽避免刷屏（与 storage.py 统一模式一致）。
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=".*DataFrame concatenation with empty or all-NA entries.*",
            )
            result = pd.concat(dfs, ignore_index=True)
        if "report_date" in result.columns:
            result["report_date"] = normalize_series_to_yyyymmdd(result["report_date"])
        return result

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
                    df[col] = normalize_series_to_yyyymmdd(df[col])
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

        if "ts_code" not in stock_basic.columns or "name" not in stock_basic.columns:
            logger.warning("stock_basic 数据缺少必要列（ts_code 或 name）")
            return stock_names

        for _, row in stock_basic.iterrows():
            if pd.notna(row.get("ts_code")) and pd.notna(row.get("name")) and row["name"]:
                stock_names[row["ts_code"]] = row["name"]

        return stock_names
