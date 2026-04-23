"""TuShare数据接口客户端"""

import os
import threading
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import tushare as ts
from loguru import logger

from ..common.config import get_tushare_settings


def _is_rate_limit_error(err_msg: str, keywords: List[str]) -> bool:
    """判断异常消息是否为 TuShare 限流错误。

    仅对限流错误做长等待; 其他错误(如 token 无效、接口不存在、网络短暂抖动)
    只做 retry_delay 的短等, 避免浪费时间。
    """
    if not err_msg:
        return False
    msg_lower = err_msg.lower()
    return any(kw in msg_lower for kw in keywords)


class TushareClient:
    """TuShare Pro API 客户端

    封装 TuShare 接口调用, 提供:
    - 线程安全的令牌桶限频 (全局 QPS 严格受 rate_limit 限制, 支持多线程共享)
    - 限流感知的重试: 仅对限流错误使用 retry_rate_limit_sleep 长等, 其他错误短等
    """

    def __init__(
        self,
        token: Optional[str] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        rate_limit: Optional[int] = None,
        verbose: bool = True,
    ):
        """初始化 TuShare 客户端

        Args:
            token: TuShare token, 如不提供则从环境变量 TS_TOKEN 读取
            max_retries: 最大重试次数
            retry_delay: 非限流错误的重试基础延迟 (秒)
            rate_limit: 每分钟请求上限
            verbose: 是否输出详细日志
        """
        # 获取 token
        self.token = token or os.getenv("TS_TOKEN")
        if not self.token:
            raise ValueError(
                "未找到TuShare token！\n"
                "请设置环境变量 TS_TOKEN 或创建 .env 文件。\n"
                "获取token: https://tushare.pro/register"
            )

        # 设置 token
        ts.set_token(self.token)
        self.pro = ts.pro_api()

        # 配置参数
        defaults = get_tushare_settings()
        self.max_retries = max_retries if max_retries is not None else defaults["max_retries"]
        self.retry_delay = retry_delay if retry_delay is not None else defaults["retry_delay"]
        self.rate_limit = rate_limit if rate_limit is not None else defaults["rate_limit"]
        self._rate_limit_keywords: List[str] = defaults["rate_limit_error_keywords"]
        self._retry_rate_limit_sleep: float = defaults["retry_rate_limit_sleep"]

        # 参数验证
        if self.rate_limit <= 0:
            raise ValueError(f"rate_limit 必须大于0, 当前值: {self.rate_limit}")

        # 线程安全的限频控制: 令牌桶思路, 通过锁保护 _last_request_time
        # 多线程调用时, 所有线程共享同一限频队列, 真实 QPS 不会超过 rate_limit
        self._request_interval = 60.0 / self.rate_limit
        self._last_request_time = 0.0
        self._rate_limit_lock = threading.Lock()

        if verbose:
            logger.info(
                f"TuShare客户端初始化成功, 限频: {self.rate_limit}次/分钟 "
                f"(线程安全令牌桶), 限流重试等待: {self._retry_rate_limit_sleep}s"
            )
        self.verbose = verbose

    def _rate_limit_wait(self, override_interval: Optional[float] = None) -> None:
        """执行限频等待 (线程安全)。

        多线程调用时, 所有调用排队经过同一把锁, 保证全局 QPS 受 rate_limit 严格控制。
        并发提速来自"请求已发出、等待响应"期间其他线程可以继续排队, 网络等待并行化。

        Args:
            override_interval: 若提供, 则本次等待使用此最小间隔 (秒), 用于官方
                无明示限频的接口 (如 top_list) 局部提速。
        """
        interval = override_interval if override_interval is not None else self._request_interval
        with self._rate_limit_lock:
            elapsed = time.time() - self._last_request_time
            if elapsed < interval:
                time.sleep(interval - elapsed)
            self._last_request_time = time.time()

    def query(
        self,
        api_name: str,
        fields: Optional[str] = None,
        skip_rate_limit: bool = False,
        rate_limit_override: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """调用 TuShare API

        Args:
            api_name: API 名称
            fields: 返回字段, 逗号分隔
            skip_rate_limit: 是否跳过限频等待 (适用于无限频要求的接口)
            rate_limit_override: 本次调用的临时限频 (次/分钟)
            **kwargs: API 参数

        Returns:
            查询结果 DataFrame
        """
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                if not skip_rate_limit:
                    override = None
                    if rate_limit_override is not None and rate_limit_override > 0:
                        override = 60.0 / rate_limit_override
                    self._rate_limit_wait(override_interval=override)

                logger.debug(f"调用API: {api_name}, 参数: {kwargs}")
                df = self.pro.query(api_name, fields=fields, **kwargs)
                logger.debug(f"API {api_name} 返回 {len(df)} 条记录")
                return df

            except Exception as e:
                last_err = e
                err_msg = str(e)
                is_rl = _is_rate_limit_error(err_msg, self._rate_limit_keywords)
                logger.warning(
                    f"API调用失败 ({attempt + 1}/{self.max_retries}): {api_name}, "
                    f"{'[限流]' if is_rl else '[其它]'} {err_msg}"
                )

                if attempt < self.max_retries - 1:
                    # 限流错误: 长等, 让服务端限流窗口过去
                    # 其它错误: 短等 (固定 retry_delay, 不做指数退避, 避免雪球)
                    wait = self._retry_rate_limit_sleep if is_rl else self.retry_delay
                    time.sleep(wait)
                else:
                    logger.error(f"API调用最终失败: {api_name}")
                    raise

        # 理论不会到这里; 兜底抛出最后一个错误
        if last_err is not None:
            raise last_err
        return pd.DataFrame()
    
    def get_trade_cal(
        self,
        start_date: str = None,
        end_date: str = None,
        exchange: str = "SSE"
    ) -> pd.DataFrame:
        """获取交易日历
        
        Args:
            start_date: 开始日期，格式YYYYMMDD（不指定则获取全部数据）
            end_date: 结束日期，格式YYYYMMDD（不指定则获取全部数据）
            exchange: 交易所，SSE上交所/SZSE深交所
            
        Returns:
            交易日历DataFrame
        """
        # 构建查询参数
        kwargs = {"exchange": exchange}
        if start_date is not None:
            kwargs["start_date"] = start_date
        if end_date is not None:
            kwargs["end_date"] = end_date
            
        return self.query(
            "trade_cal",
            fields="exchange,cal_date,is_open,pretrade_date",
            **kwargs
        )
    
    def get_stock_basic(
        self,
        list_status: str = "L",
        fields: Optional[str] = None
    ) -> pd.DataFrame:
        """获取股票列表
        
        Args:
            list_status: 上市状态，L上市/D退市/P暂停上市
            fields: 返回字段
            
        Returns:
            股票列表DataFrame
        """
        if fields is None:
            fields = "ts_code,symbol,name,area,industry,market,list_date"
        
        return self.query("stock_basic", fields=fields, list_status=list_status)
    
    def get_daily(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """获取日线行情
        
        Args:
            ts_code: 股票代码
            trade_date: 交易日期
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            日线行情DataFrame
        """
        return self.query(
            "daily",
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date
        )
    
    def get_daily_basic(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """获取每日指标（PE、PB等）
        
        Args:
            ts_code: 股票代码
            trade_date: 交易日期
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            每日指标DataFrame
        """
        return self.query(
            "daily_basic",
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date
        )
    
    def get_adj_factor(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """获取复权因子
        
        Args:
            ts_code: 股票代码
            trade_date: 交易日期
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            复权因子DataFrame，包含 ts_code, trade_date, adj_factor 等字段
        """
        return self.query(
            "adj_factor",
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date
        )
    
    def get_suspend_d(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        suspend_type: Optional[str] = None
    ) -> pd.DataFrame:
        """获取停复牌信息
        
        注意：此API已更新参数，旧版本使用suspend_date/resume_date的代码需要迁移
        
        Args:
            ts_code: 股票代码，支持多个股票
            trade_date: 交易日期，格式YYYYMMDD
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
            suspend_type: 停复牌类型，S=停牌，R=复牌
            
        Returns:
            停复牌信息DataFrame，包含以下字段：
            - ts_code: 股票代码
            - trade_date: 停复牌日期
            - suspend_timing: 盘中停复牌时段（如有）
            - suspend_type: S=停牌，R=复牌
            
        Examples:
            >>> # 获取某日所有停牌股票
            >>> client.get_suspend_d(trade_date='20230315', suspend_type='S')
            >>> # 获取某个时间段某只股票的停复牌记录
            >>> client.get_suspend_d(ts_code='000001.SZ', start_date='20230101', end_date='20230331')
        """
        return self.query(
            "suspend_d",
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            suspend_type=suspend_type
        )
    
    def get_stk_limit(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """获取每日涨跌停价格
        
        Args:
            ts_code: 股票代码
            trade_date: 交易日期
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            涨跌停价格DataFrame，包含 up_limit, down_limit 等字段
        """
        return self.query(
            "stk_limit",
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date
        )
    
    def get_namechange(
        self,
        ts_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """获取股票名称变更历史
        
        用于判断ST状态等
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            名称变更历史DataFrame
        """
        return self.query(
            "namechange",
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )
    
    def get_moneyflow(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """获取个股资金流向
        
        Args:
            ts_code: 股票代码
            trade_date: 交易日期
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            资金流向DataFrame，包含以下字段：
            - ts_code: 股票代码
            - trade_date: 交易日期
            - buy_sm_vol: 小单买入量（手）
            - buy_sm_amount: 小单买入金额（万元）
            - sell_sm_vol: 小单卖出量（手）
            - sell_sm_amount: 小单卖出金额（万元）
            - buy_md_vol: 中单买入量（手）
            - buy_md_amount: 中单买入金额（万元）
            - sell_md_vol: 中单卖出量（手）
            - sell_md_amount: 中单卖出金额（万元）
            - buy_lg_vol: 大单买入量（手）
            - buy_lg_amount: 大单买入金额（万元）
            - sell_lg_vol: 大单卖出量（手）
            - sell_lg_amount: 大单卖出金额（万元）
            - buy_elg_vol: 特大单买入量（手）
            - buy_elg_amount: 特大单买入金额（万元）
            - sell_elg_vol: 特大单卖出量（手）
            - sell_elg_amount: 特大单卖出金额（万元）
            - net_mf_vol: 净流入量（手）
            - net_mf_amount: 净流入额（万元）
        """
        return self.query(
            "moneyflow",
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date
        )
    
    def get_index_classify(
        self,
        level: str = "L1",
        src: str = "SW2021",
        **kwargs
    ) -> pd.DataFrame:
        """获取指数基本信息
        
        Args:
            level: 行业级别，L1=一级/L2=二级/L3=三级
            src: 申万分类版本，SW2021=申万2021版/SW2014=申万2014版
            **kwargs: 其他参数
            
        Returns:
            指数基本信息DataFrame，包含以下字段：
            - index_code: 指数代码
            - index_name: 指数名称
            - level: 行业级别
            - industry_code: 行业代码
            - parent_code: 父级代码
            - src: 分类来源
        """
        return self.query(
            "index_classify",
            level = level,
            src = src,
            **kwargs
        )
    
    def get_index_classify(
        self,
        level: str = "L1",
        src: str = "SW2021",
        **kwargs
    ) -> pd.DataFrame:
        """获取申万行业分类
        
        Args:
            level: 行业级别，L1=一级/L2=二级/L3=三级
            src: 申万分类版本，SW2021=申万2021版/SW2014=申万2014版
            **kwargs: 其他参数
            
        Returns:
            申万行业分类DataFrame，包含以下字段：
            - index_code: 指数代码
            - industry_name: 行业名称
            - level: 行业级别
            - industry_code: 行业代码
            - parent_code: 父级代码
            - src: 分类来源
        """
        return self.query(
            "index_classify",
            level=level,
            src=src,
            **kwargs
        )
    
    def get_realtime_quote(self, ts_codes: str) -> pd.DataFrame:
        """获取实时行情

        Args:
            ts_codes: 股票代码，多个以逗号分隔，如 '000001.SZ,000002.SZ'

        Returns:
            实时行情DataFrame，包含 ts_code, name, price, pre_close, open,
            high, low, volume, amount, time 等字段
        """
        #return self.query("realtime_quote", ts_code=ts_codes)
        return ts.realtime_quote(ts_code=ts_codes)

    def get_index_member(
        self,
        l1_code: str = None,
        l2_code: str = None,
        l3_code: str = None,
        **kwargs
    ) -> pd.DataFrame:
        """获取指数成分股
        
        Args:
            l1_code: 一级行业代码
            l2_code: 二级行业代码
            l3_code: 三级行业代码
            **kwargs: 其他参数
            
        Returns:
            指数成分股DataFrame，包含以下字段：
            - l1_code: 一级行业代码
            - l1_name: 一级行业名称
            - l2_code: 二级行业代码
            - l2_name: 二级行业名称
            - l3_code: 三级行业代码
            - l3_name: 三级行业名称
            - ts_code: 成分股代码
            - ts_name: 成分股名称
            - in_date: 加入日期
            - out_date: 退出日期
            - is_new: 是否最新成分股，1=是，0=否
        """
        return self.query(
            "index_member_all",
            l1_code=l1_code,
            l2_code=l2_code,
            l3_code=l3_code,
            **kwargs
        )

    def get_fina_indicator(
        self,
        ts_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        fields: Optional[str] = None
    ) -> pd.DataFrame:
        """获取财务指标数据（fina_indicator）

        注意：此 API 只支持按单个股票查询，每次最多返回 100 条记录。
        需要 2000 积分权限。

        Args:
            ts_code: 股票代码（必须，单只股票，如 '000001.SZ'）
            start_date: 报告期开始日期，格式 YYYYMMDD
            end_date: 报告期结束日期，格式 YYYYMMDD
            fields: 返回字段，逗号分隔

        Returns:
            财务指标 DataFrame
        """
        if fields is None:
            fields = (
                "ts_code,ann_date,end_date,"
                "roe_waa,or_yoy,netprofit_yoy,debt_to_assets,q_gr_yoy"
            )

        return self.query(
            "fina_indicator",
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields=fields,
        )

    def get_fina_indicator_by_date(
        self,
        ann_date: Optional[str] = None,
        period: Optional[str] = None,
        fields: Optional[str] = None,
    ) -> pd.DataFrame:
        """按公告日或报告期获取全市场财务指标（fina_indicator_vip，5000 积分）

        Args:
            ann_date: 公告日期，格式 YYYYMMDD（与 period 二选一）
            period: 报告期，格式 YYYYMMDD，如 20231231（与 ann_date 二选一）
            fields: 返回字段，逗号分隔

        Returns:
            全市场财务指标 DataFrame
        """
        if fields is None:
            fields = (
                "ts_code,ann_date,end_date,"
                "roe_waa,or_yoy,netprofit_yoy,debt_to_assets,q_gr_yoy"
            )
        kwargs: dict = {"fields": fields}
        if ann_date is not None:
            kwargs["ann_date"] = ann_date
        if period is not None:
            kwargs["period"] = period
        return self.query("fina_indicator_vip", **kwargs)

    def get_forecast_by_date(
        self,
        ann_date: Optional[str] = None,
        period: Optional[str] = None,
    ) -> pd.DataFrame:
        """按公告日或报告期获取全市场业绩预告（forecast_vip，5000 积分）

        Args:
            ann_date: 公告日期，格式 YYYYMMDD（与 period 二选一）
            period: 报告期，格式 YYYYMMDD，如 20231231（与 ann_date 二选一）

        Returns:
            全市场业绩预告 DataFrame
        """
        kwargs: dict = {}
        if ann_date is not None:
            kwargs["ann_date"] = ann_date
        if period is not None:
            kwargs["period"] = period
        return self.query("forecast_vip", **kwargs)

    def get_cyq_perf(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取筹码胜率数据（cyq_perf，5000 积分）

        支持两种查询方式：
        1. 按 trade_date 获取全市场当日数据（推荐，单次获取所有股票）
        2. 按 ts_code + start_date/end_date 获取单只股票历史数据

        Args:
            ts_code: 股票代码（可选）
            trade_date: 交易日期，格式 YYYYMMDD（可选，与 ts_code 二选一）
            start_date: 开始日期，格式 YYYYMMDD（配合 ts_code 使用）
            end_date: 结束日期，格式 YYYYMMDD（配合 ts_code 使用）

        Returns:
            筹码胜率 DataFrame
        """
        kwargs = {}
        if ts_code is not None:
            kwargs["ts_code"] = ts_code
        if trade_date is not None:
            kwargs["trade_date"] = trade_date
        if start_date is not None:
            kwargs["start_date"] = start_date
        if end_date is not None:
            kwargs["end_date"] = end_date
        return self.query("cyq_perf", **kwargs)

    def get_express_vip(
        self,
        ann_date: Optional[str] = None,
        period: Optional[str] = None,
    ) -> pd.DataFrame:
        """按公告日/报告期获取全市场业绩快报（express_vip，5000 积分）

        Args:
            ann_date: 公告日期，格式 YYYYMMDD
            period: 报告期，格式 YYYYMMDD（如 20231231）

        Returns:
            业绩快报 DataFrame
        """
        kwargs = {}
        if ann_date is not None:
            kwargs["ann_date"] = ann_date
        if period is not None:
            kwargs["period"] = period
        return self.query("express_vip", **kwargs)

    def get_fund_portfolio(
        self,
        ts_code: Optional[str] = None,
        period: Optional[str] = None,
        ann_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取公募基金持仓数据（fund_portfolio，5000 积分）

        Args:
            ts_code: 基金代码（按单只基金查询）
            period: 报告期，格式 YYYYMMDD（如 20231231）
            ann_date: 公告日期，格式 YYYYMMDD

        Returns:
            基金持仓 DataFrame
        """
        kwargs = {}
        if ts_code is not None:
            kwargs["ts_code"] = ts_code
        if period is not None:
            kwargs["period"] = period
        if ann_date is not None:
            kwargs["ann_date"] = ann_date
        return self.query("fund_portfolio", **kwargs)

    def get_stk_holdernumber(
        self,
        ts_code: Optional[str] = None,
        ann_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取股东人数数据

        支持多种查询方式：
        1. 按 ann_date 获取当日公告的全市场数据
        2. 按 start_date/end_date 获取一段时间内全市场数据（单次限3000条）
        3. 按 ts_code 获取单只股票历史数据

        Args:
            ts_code: 股票代码（可选）
            ann_date: 公告日期，格式 YYYYMMDD（可选）
            start_date: 开始日期，格式 YYYYMMDD（可选）
            end_date: 结束日期，格式 YYYYMMDD（可选）

        Returns:
            股东人数 DataFrame
        """
        kwargs: dict = {}
        if ts_code is not None:
            kwargs["ts_code"] = ts_code
        if ann_date is not None:
            kwargs["ann_date"] = ann_date
        if start_date is not None:
            kwargs["start_date"] = start_date
        if end_date is not None:
            kwargs["end_date"] = end_date
        return self.query("stk_holdernumber", **kwargs)

    def get_moneyflow_hsgt(
        self,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取沪深股通资金流向（moneyflow_hsgt，2000 积分）

        市场级日度数据，返回沪股通/深股通当日整体买卖与净流入，
        用作北向资金宏观因子（广播到全部 ts_code）。

        Args:
            trade_date: 交易日期，格式 YYYYMMDD（可选）
            start_date: 开始日期，格式 YYYYMMDD（可选）
            end_date: 结束日期，格式 YYYYMMDD（可选）

        Returns:
            DataFrame，主要字段：
            - trade_date: 交易日期
            - ggt_ss: 港股通（上海）
            - ggt_sz: 港股通（深圳）
            - hgt: 沪股通（亿元）
            - sgt: 深股通（亿元）
            - north_money: 北向资金净流入（亿元）
            - south_money: 南向资金净流入（亿元）
        """
        kwargs: dict = {}
        if trade_date is not None:
            kwargs["trade_date"] = trade_date
        if start_date is not None:
            kwargs["start_date"] = start_date
        if end_date is not None:
            kwargs["end_date"] = end_date
        return self.query("moneyflow_hsgt", **kwargs)

    def get_top_list(
        self,
        trade_date: Optional[str] = None,
        ts_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取龙虎榜个股明细（top_list，2000 积分）

        Args:
            trade_date: 交易日期，格式 YYYYMMDD（可选）
            ts_code: 股票代码（可选）
            start_date: 开始日期，格式 YYYYMMDD（可选）
            end_date: 结束日期，格式 YYYYMMDD（可选）

        Returns:
            DataFrame，主要字段：
            - trade_date, ts_code, name, close
            - pct_change: 涨跌幅
            - turnover_rate: 换手率
            - amount: 总成交额
            - l_sell/l_buy: 龙虎榜卖/买入额
            - l_amount: 龙虎榜成交额
            - net_amount: 龙虎榜净买入额
            - net_rate: 龙虎榜净买入额占比
            - amount_rate: 龙虎榜成交额占比
            - float_values: 当日流通市值
            - reason: 上榜理由
        """
        kwargs: dict = {}
        if trade_date is not None:
            kwargs["trade_date"] = trade_date
        if ts_code is not None:
            kwargs["ts_code"] = ts_code
        if start_date is not None:
            kwargs["start_date"] = start_date
        if end_date is not None:
            kwargs["end_date"] = end_date
        # 官方未明示限频, 局部放宽到 1000 次/分钟 (60ms/次), 加速历史批量下载
        return self.query("top_list", rate_limit_override=60, **kwargs)

    def get_report_rc(
        self,
        ts_code: Optional[str] = None,
        report_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取卖方研报一致预期（report_rc，2000 积分）

        Args:
            ts_code: 股票代码（可选）
            report_date: 研报日期，格式 YYYYMMDD（可选）
            start_date: 报告日期起（可选）
            end_date: 报告日期止（可选）

        Returns:
            DataFrame，主要字段：
            - ts_code, name
            - report_date: 研报日期
            - report_title, report_type
            - classify, org_name, author_name
            - quarter: 预测季度
            - op_rt: 预测营收增长率
            - op_pr: 预测营收
            - tp: 预测净利润
            - np: 预测净利润
            - eps: 每股收益预测
            - pe/rd/roe/ev_ebitda: 估值/收益指标
            - rating: 评级
            - max_price, min_price: 预测价格区间
        """
        kwargs: dict = {}
        if ts_code is not None:
            kwargs["ts_code"] = ts_code
        if report_date is not None:
            kwargs["report_date"] = report_date
        if start_date is not None:
            kwargs["start_date"] = start_date
        if end_date is not None:
            kwargs["end_date"] = end_date
        return self.query("report_rc", **kwargs)
