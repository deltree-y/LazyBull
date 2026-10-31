"""停牌日历工具类

提供统一的停牌判断接口，基于 raw/suspend 数据

v2 迁移复制件（P2a-T1，行为冻结）：与旧模块 ``common/suspend_calendar.py``
双源并存至 v2 切换。P4-T3 E4 起仅依赖宿主注入的窄读取协议，不再 import
或缺省创建旧 ``data.storage.Storage``；缓存、异常与 S/R 判定语义不变。
"""

from typing import Dict, Iterable, Optional, Protocol, Tuple

import pandas as pd
from loguru import logger


class SuspendDataReader(Protocol):
    """停牌日历所需的最小只读能力（由回测/纸面宿主注入）。"""

    def load_raw_by_date(self, dataset: str, trade_date: str) -> Optional[pd.DataFrame]:
        """按日读取原始数据集；分区不存在时返回 None。"""
        ...


def get_suspend_calendar(
    data_storage: SuspendDataReader,
) -> Tuple["SuspendCalendar", SuspendDataReader]:
    """以宿主注入的读取能力构建停牌日历；缺失注入 fail-fast。"""
    if data_storage is None:
        raise ValueError("停牌日历需要宿主注入 SuspendDataReader，禁止缺省创建 Storage")
    return SuspendCalendar(data_storage), data_storage


class SuspendCalendar:
    """停牌日历工具类

    提供统一的停牌判断接口，基于 raw/suspend 数据

    判定规则：
    - 当日存在记录且 suspend_type == 'S' => 停牌 True
    - 当日存在记录且 suspend_type == 'R' => 非停牌 False
    - 当日无记录 => 非停牌 False

    严格模式：
    - 当日 suspend 数据文件缺失或无法加载 => 抛出异常
    """

    def __init__(self, storage: SuspendDataReader):
        """初始化停牌日历

        Args:
            storage: 宿主注入的窄只读协议对象，用于读取 raw/suspend 数据
        """
        self.storage = storage
        # 按 trade_date 缓存已加载的停牌数据
        self._cache: Dict[str, Optional[pd.DataFrame]] = {}

    def _load_suspend_data(self, trade_date: str) -> Optional[pd.DataFrame]:
        """加载指定交易日的停牌数据

        Args:
            trade_date: 交易日期，格式 YYYYMMDD 或 YYYY-MM-DD

        Returns:
            停牌数据 DataFrame，如果文件不存在则返回 None

        Raises:
            FileNotFoundError: 严格模式下，文件不存在时抛出
            Exception: 数据加载失败时抛出
        """
        # 检查缓存
        if trade_date in self._cache:
            return self._cache[trade_date]

        try:
            # 使用 Storage 的 load_raw_by_date 方法加载按日期分区的 suspend 数据
            df = self.storage.load_raw_by_date("suspend", trade_date)

            # 严格模式：如果数据文件不存在，抛出异常
            if df is None:
                error_msg = f"停牌数据文件缺失：trade_date={trade_date}，无法判断停牌状态"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            # 缓存数据
            self._cache[trade_date] = df
            logger.debug(f"已加载停牌数据：trade_date={trade_date}，记录数={len(df)}")
            return df

        except FileNotFoundError:
            # 严格模式：文件不存在，抛出异常
            raise
        except Exception as e:
            # 其他加载错误，抛出异常
            error_msg = f"加载停牌数据失败：trade_date={trade_date}，错误={str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    def is_suspended(self, ts_code: str, trade_date: str) -> bool:
        """判断指定股票在指定日期是否停牌

        Args:
            ts_code: 股票代码
            trade_date: 交易日期，格式 YYYYMMDD 或 YYYY-MM-DD

        Returns:
            True 表示停牌，False 表示非停牌

        Raises:
            FileNotFoundError: 严格模式下，suspend 数据文件缺失时抛出
            Exception: 数据加载失败时抛出
        """
        # 加载停牌数据
        df = self._load_suspend_data(trade_date)

        # 查找该股票的记录
        if df is not None and len(df) > 0:
            stock_records = df[df["ts_code"] == ts_code]

            if len(stock_records) > 0:
                # 存在记录，检查 suspend_type
                suspend_type = stock_records.iloc[0]["suspend_type"]
                if suspend_type == "S":
                    # 停牌
                    return True
                elif suspend_type == "R":
                    # 复牌
                    return False
                else:
                    # 未知类型，保守处理为非停牌
                    logger.warning(
                        f"未知的 suspend_type: {suspend_type}, "
                        f"ts_code={ts_code}, trade_date={trade_date}"
                    )
                    return False

        # 无记录，默认为非停牌
        return False

    def get_status_reason(self, ts_code: str, trade_date: str) -> str:
        """获取停牌状态描述（用于日志）

        Args:
            ts_code: 股票代码
            trade_date: 交易日期，格式 YYYYMMDD 或 YYYY-MM-DD

        Returns:
            状态描述字符串："停牌"/"复牌"/"无记录"

        Raises:
            FileNotFoundError: 严格模式下，suspend 数据文件缺失时抛出
            Exception: 数据加载失败时抛出
        """
        # 加载停牌数据
        df = self._load_suspend_data(trade_date)

        # 查找该股票的记录
        if df is not None and len(df) > 0:
            stock_records = df[df["ts_code"] == ts_code]

            if len(stock_records) > 0:
                # 存在记录，检查 suspend_type
                suspend_type = stock_records.iloc[0]["suspend_type"]
                if suspend_type == "S":
                    return "停牌"
                elif suspend_type == "R":
                    return "复牌"
                else:
                    return f"未知状态({suspend_type})"

        # 无记录
        return "无记录"

    def batch_is_suspended(self, ts_codes: Iterable[str], trade_date: str) -> Dict[str, bool]:
        """批量判断多个股票在指定日期是否停牌

        Args:
            ts_codes: 股票代码列表
            trade_date: 交易日期，格式 YYYYMMDD 或 YYYY-MM-DD

        Returns:
            {ts_code: is_suspended} 字典，True 表示停牌，False 表示非停牌

        Raises:
            FileNotFoundError: 严格模式下，suspend 数据文件缺失时抛出
            Exception: 数据加载失败时抛出
        """
        # 加载停牌数据（只加载一次）
        df = self._load_suspend_data(trade_date)

        result = {}
        for ts_code in ts_codes:
            # 查找该股票的记录
            if df is not None and len(df) > 0:
                stock_records = df[df["ts_code"] == ts_code]

                if len(stock_records) > 0:
                    # 存在记录，检查 suspend_type
                    suspend_type = stock_records.iloc[0]["suspend_type"]
                    if suspend_type == "S":
                        # 停牌
                        result[ts_code] = True
                    elif suspend_type == "R":
                        # 复牌
                        result[ts_code] = False
                    else:
                        # 未知类型，保守处理为非停牌
                        logger.warning(
                            f"未知的 suspend_type: {suspend_type}, "
                            f"ts_code={ts_code}, trade_date={trade_date}"
                        )
                        result[ts_code] = False
                else:
                    # 无记录，默认为非停牌
                    result[ts_code] = False
            else:
                # 无记录，默认为非停牌
                result[ts_code] = False

        return result
