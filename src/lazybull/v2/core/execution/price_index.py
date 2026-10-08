"""回测价格索引与取价（P2a-T4，engine.py 函数级拆分件）。

来源：旧 ``backtest/engine.py`` 的 ``_prepare_price_index`` /
``_get_trade_price`` / ``_get_pnl_price`` / ``_get_trade_price_open`` /
``_get_pnl_price_open``（T0 规划 §3.3）。方法转函数、显式入参替代 self：
四套价格序列封装为 ``PriceIndexes`` 值对象显式传递，价格口径与全部
回退分支（缺 close_adj / open / open_adj / 全 NaN）逐字保留。
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from loguru import logger


@dataclass
class PriceIndexes:
    """四套价格序列（MultiIndex: (trade_date, ts_code)）。

    Attributes:
        trade_price_index: 收盘成交价格（不复权 close）
        pnl_price_index: 收盘绩效价格（后复权 close_adj）
        trade_price_open_index: 开盘成交价格（不复权 open）
        pnl_price_open_index: 开盘绩效价格（后复权 open_adj）
    """

    trade_price_index: pd.Series
    pnl_price_index: pd.Series
    trade_price_open_index: pd.Series
    pnl_price_open_index: pd.Series


def prepare_price_index(price_data: pd.DataFrame) -> PriceIndexes:
    """准备价格索引（使用 MultiIndex，替代嵌套字典）

    构建四套价格序列：
    - trade_price_index: 收盘成交价格（不复权 close）
    - pnl_price_index: 收盘绩效价格（后复权 close_adj）
    - trade_price_open_index: 开盘成交价格（不复权 open）
    - pnl_price_open_index: 开盘绩效价格（后复权 open_adj）

    Args:
        price_data: 价格数据，需包含 ts_code, trade_date, close, open（可选），close_adj（可选），open_adj（可选）

    Returns:
        四套价格序列值对象
    """
    logger.info("开始准备价格索引...")

    # 检查必需列
    if "close" not in price_data.columns:
        raise ValueError("价格数据缺少 'close' 列，无法进行回测")

    # 转换日期列为 datetime（向量化操作，避免 iterrows）
    if not pd.api.types.is_datetime64_any_dtype(price_data["trade_date"]):
        # 创建副本以避免修改原始数据
        price_data = price_data.copy()
        price_data["trade_date"] = pd.to_datetime(price_data["trade_date"])

    # 构建收盘成交价格索引（不复权 close）
    trade_price_df = price_data[["trade_date", "ts_code", "close"]].copy()
    trade_price_df.set_index(["trade_date", "ts_code"], inplace=True)
    trade_price_index = trade_price_df["close"]

    # 构建收盘绩效价格索引（后复权 close_adj）
    if "close_adj" in price_data.columns:
        pnl_price_df = price_data[["trade_date", "ts_code", "close_adj"]].copy()
        pnl_price_df.set_index(["trade_date", "ts_code"], inplace=True)
        pnl_price_index = pnl_price_df["close_adj"]
        logger.info("价格索引构建完成: 收盘成交价格=close, 收盘绩效价格=close_adj")
    else:
        # 如果缺少 close_adj，回退到 close
        logger.warning("价格数据缺少 'close_adj' 列，绩效价格将使用 'close' 列（不复权）")
        pnl_price_index = trade_price_index.copy()
        logger.info("价格索引构建完成: 收盘成交价格=close, 收盘绩效价格=close（退化）")

    # 构建开盘成交价格索引（不复权 open）
    if "open" in price_data.columns:
        # 过滤掉NaN值，只保留有效的开盘价
        open_data = price_data[["trade_date", "ts_code", "open"]].copy()
        open_data = open_data[open_data["open"].notna()]

        if len(open_data) > 0:
            open_data.set_index(["trade_date", "ts_code"], inplace=True)
            trade_price_open_index = open_data["open"]
            logger.info(f"开盘价格索引构建完成: 开盘成交价格=open, 共{len(open_data)}条记录")
        else:
            logger.warning("价格数据的 'open' 列全部为NaN，开盘价格将使用收盘价格代替")
            trade_price_open_index = trade_price_index.copy()
    else:
        logger.warning("价格数据缺少 'open' 列，开盘价格将使用收盘价格代替")
        trade_price_open_index = trade_price_index.copy()

    # 构建开盘绩效价格索引（后复权 open_adj）
    if "open_adj" in price_data.columns:
        # 过滤掉NaN值，只保留有效的开盘绩效价格
        open_adj_data = price_data[["trade_date", "ts_code", "open_adj"]].copy()
        open_adj_data = open_adj_data[open_adj_data["open_adj"].notna()]

        if len(open_adj_data) > 0:
            open_adj_data.set_index(["trade_date", "ts_code"], inplace=True)
            pnl_price_open_index = open_adj_data["open_adj"]
            logger.info(
                f"开盘绩效价格索引构建完成: 开盘绩效价格=open_adj, 共{len(open_adj_data)}条记录"
            )
        else:
            # 如果open_adj全部为NaN，尝试使用open
            if "open" in price_data.columns:
                logger.warning(
                    "价格数据的 'open_adj' 列全部为NaN，开盘绩效价格将使用 'open' 列（不复权）"
                )
                pnl_price_open_index = trade_price_open_index.copy()
            else:
                logger.warning(
                    "价格数据缺少 'open' 和 'open_adj' 列，开盘绩效价格将使用收盘绩效价格代替"
                )
                pnl_price_open_index = pnl_price_index.copy()
    else:
        # 如果缺少 open_adj，回退到 open 或 close_adj
        if "open" in price_data.columns:
            # 如果有 open 但没有 open_adj，使用 open
            logger.warning("价格数据缺少 'open_adj' 列，开盘绩效价格将使用 'open' 列（不复权）")
            pnl_price_open_index = trade_price_open_index.copy()
        else:
            # 如果连 open 都没有，使用 close_adj
            logger.warning("价格数据缺少 'open_adj' 列，开盘绩效价格将使用收盘绩效价格代替")
            pnl_price_open_index = pnl_price_index.copy()

    return PriceIndexes(
        trade_price_index=trade_price_index,
        pnl_price_index=pnl_price_index,
        trade_price_open_index=trade_price_open_index,
        pnl_price_open_index=pnl_price_open_index,
    )


def get_trade_price(price_indexes: PriceIndexes, date: pd.Timestamp, stock: str) -> Optional[float]:
    """获取收盘成交价格（不复权 close）

    Args:
        price_indexes: 四套价格序列
        date: 日期
        stock: 股票代码

    Returns:
        成交价格，如果不存在则返回 None
    """
    try:
        return price_indexes.trade_price_index.loc[(date, stock)]
    except KeyError:
        return None


def get_pnl_price(price_indexes: PriceIndexes, date: pd.Timestamp, stock: str) -> Optional[float]:
    """获取收盘绩效价格（后复权 close_adj）

    Args:
        price_indexes: 四套价格序列
        date: 日期
        stock: 股票代码

    Returns:
        绩效价格，如果不存在则返回 None
    """
    try:
        return price_indexes.pnl_price_index.loc[(date, stock)]
    except KeyError:
        return None


def get_trade_price_open(
    price_indexes: PriceIndexes, date: pd.Timestamp, stock: str
) -> Optional[float]:
    """获取开盘成交价格（不复权 open）

    如果开盘价格不存在，返回 None。调用者应处理降级策略（如使用收盘价）。

    Args:
        price_indexes: 四套价格序列
        date: 日期
        stock: 股票代码

    Returns:
        开盘成交价格，如果不存在则返回 None
    """
    try:
        return price_indexes.trade_price_open_index.loc[(date, stock)]
    except KeyError:
        return None


def get_pnl_price_open(
    price_indexes: PriceIndexes, date: pd.Timestamp, stock: str
) -> Optional[float]:
    """获取开盘绩效价格（后复权 open_adj）

    如果开盘绩效价格不存在，返回 None。调用者应处理降级策略（如使用收盘绩效价格）。

    Args:
        price_indexes: 四套价格序列
        date: 日期
        stock: 股票代码

    Returns:
        开盘绩效价格，如果不存在则返回 None
    """
    try:
        return price_indexes.pnl_price_open_index.loc[(date, stock)]
    except KeyError:
        return None
