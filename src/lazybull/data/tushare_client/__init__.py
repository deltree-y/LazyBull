# -*- coding: utf-8 -*-
"""TuShare Pro API 客户端（tushare_client 子包门面）。"""

from .alt import ClientAltMixin
from .basic import ClientBasicMixin
from .core import (
    FINA_INDICATOR_DEFAULT_FIELDS,
    ClientCoreMixin,
    _API_RATE_LIMITS_DEFAULT,
    _RATE_LIMIT_MSG_FREQ,
    _is_rate_limit_error,
    ts,
)
from .daily import ClientDailyMixin
from .fundamental import ClientFundamentalMixin


class TushareClient(
    ClientCoreMixin, ClientBasicMixin, ClientDailyMixin, ClientFundamentalMixin, ClientAltMixin
):
    """TuShare Pro API 客户端（mixin 组合）。"""


__all__ = [
    "FINA_INDICATOR_DEFAULT_FIELDS",
    "TushareClient",
    "ClientCoreMixin",
    "ClientBasicMixin",
    "ClientDailyMixin",
    "ClientFundamentalMixin",
    "ClientAltMixin",
    "_API_RATE_LIMITS_DEFAULT",
    "_RATE_LIMIT_MSG_FREQ",
    "_is_rate_limit_error",
    "ts",
]
