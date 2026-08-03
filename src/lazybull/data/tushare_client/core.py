# -*- coding: utf-8 -*-
"""TushareClient 核心：令牌桶限频、限流重试、常量与工具函数。"""

import os
import re
import threading
import time
from typing import Dict, List, Optional

import pandas as pd
import tushare as ts
from loguru import logger

from ...common.config import get_tushare_settings

FINA_INDICATOR_DEFAULT_FIELDS = (
    "ts_code,ann_date,end_date,"
    "roe_waa,roe_dt,roa,or_yoy,netprofit_yoy,"
    "profit_dedt,q_gr_yoy,equity_yoy,"
    "grossprofit_margin,netprofit_margin,"
    "debt_to_assets,current_ratio,quick_ratio,"
    "q_ocf_to_sales,int_to_talcap,assets_turn,inv_turn"
)

# 接口级每分钟限频（Tushare 8000 积分）。
# 部分接口有独立限频（低于全局 rate_limit），如 cyq_perf=100 次/分钟。
# 未知接口回退全局 rate_limit；收到限流错误时会自动解析"频率超限(X次/分钟)"并动态更新。
_API_RATE_LIMITS_DEFAULT: Dict[str, int] = {
    "cyq_perf": 150,
    "margin_detail": 250,
    "top_list": 450,  # 官方限频 500 次/分钟, 客户端侧留 10% 余量避免被限流
    "report_rc": 200,  # 官方限频约 200~300 次/分钟, 客户端侧保守取值避免长期高并发被拒
}
# 确定性业务错误 (参数错误/单次查询超限等): 重试必然再失败, 直接抛以节省请求量
_ERR_NO_RETRY_KEYWORDS = ("查询数据失败", "请确认参数")
# 限流错误中提取接口频次，如 "抱歉，您访问接口(cyq_perf)频率超限(200次/分钟)..."
_RATE_LIMIT_MSG_FREQ = re.compile(r"频率超限\s*\(\s*(\d+)\s*次/分钟")


def _is_rate_limit_error(err_msg: str, keywords: List[str]) -> bool:
    """判断异常消息是否为 TuShare 限流错误。

    仅对限流错误做长等待; 其他错误(如 token 无效、接口不存在、网络短暂抖动)
    只做 retry_delay 的短等, 避免浪费时间。
    """
    if not err_msg:
        return False
    msg_lower = err_msg.lower()
    return any(kw in msg_lower for kw in keywords)


class ClientCoreMixin:
    """TushareClient 核心 mixin：限频与查询。"""

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

        # 线程安全的限频控制: 按接口分桶令牌桶, 每个接口独立限频
        # 不同接口(如 cyq_perf=200/分钟) 限频不同, 分桶避免低限频接口拖累全局,
        # 也避免高并发下低限频接口被限流
        self._request_interval = 60.0 / self.rate_limit
        self._api_rate_limits: Dict[str, int] = dict(_API_RATE_LIMITS_DEFAULT)
        self._rate_limit_locks: Dict[str, threading.Lock] = {}
        self._last_request_time_by_api: Dict[str, float] = {}

        if verbose:
            logger.info(
                f"TuShare客户端初始化成功, 限频: {self.rate_limit}次/分钟 "
                f"(线程安全令牌桶), 限流重试等待: {self._retry_rate_limit_sleep}s"
            )
        self.verbose = verbose

    def _request_interval_for(self, api_name: str) -> float:
        """按接口返回限频最小间隔 (秒)。未知接口回退全局 rate_limit。"""
        rl = self._api_rate_limits.get(api_name, self.rate_limit)
        return 60.0 / rl

    def _rate_limit_wait(self, api_name: str, override_interval: Optional[float] = None) -> None:
        """执行按接口限频等待 (线程安全, 每个接口独立令牌桶)。

        多线程调用时, 同一接口的所有调用排队经过该接口的锁, 保证接口级 QPS 受
        该接口限频严格约束; 不同接口互不拖累 (如 cyq_perf=200/分钟 不影响全局 500/分钟)。
        并发提速来自"请求已发出、等待响应"期间其他线程可以继续排队, 网络等待并行化。

        Args:
            api_name: 接口名 (限频分桶键)
            override_interval: 若提供, 则本次等待使用此最小间隔 (秒), 用于临时
                放宽/收紧某次调用的限频 (默认 None 走接口级/全局令牌桶)。
        """
        interval = (
            override_interval
            if override_interval is not None
            else self._request_interval_for(api_name)
        )
        lock = self._rate_limit_locks.setdefault(api_name, threading.Lock())
        with lock:
            last = self._last_request_time_by_api.get(api_name, 0.0)
            elapsed = time.time() - last
            if elapsed < interval:
                time.sleep(interval - elapsed)
            self._last_request_time_by_api[api_name] = time.time()

    def query(
        self,
        api_name: str,
        fields: Optional[str] = None,
        skip_rate_limit: bool = False,
        rate_limit_override: Optional[int] = None,
        **kwargs,
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
                    self._rate_limit_wait(api_name, override_interval=override)

                logger.debug(f"调用API: {api_name}, 参数: {kwargs}")
                df = self.pro.query(api_name, fields=fields, **kwargs)
                logger.debug(f"API {api_name} 返回 {len(df)} 条记录")
                return df

            except Exception as e:
                last_err = e
                err_msg = str(e)
                # 确定性业务错误 (如参数错误/单次查询超限): 重试必失败, 直接抛
                # 省去 2/3 的重复请求, 避免高并发下放大请求风暴;
                # 属预期正常分流 (如大年份超限后走二分), 用 debug 避免黄色噪音
                if any(k in err_msg for k in _ERR_NO_RETRY_KEYWORDS):
                    logger.debug(f"API调用失败 (确定性错误, 不重试): {api_name}, {err_msg}")
                    raise
                is_rl = _is_rate_limit_error(err_msg, self._rate_limit_keywords)
                if is_rl:
                    # 从错误信息解析接口频次, 自适应更新接口级限频 (如 cyq_perf=200/分钟)
                    freq_match = _RATE_LIMIT_MSG_FREQ.search(err_msg)
                    if freq_match:
                        freq = int(freq_match.group(1))
                        if freq > 0 and freq != self._api_rate_limits.get(api_name):
                            self._api_rate_limits[api_name] = freq
                            logger.warning(f"接口 {api_name} 限频自动适配为 {freq} 次/分钟")
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
