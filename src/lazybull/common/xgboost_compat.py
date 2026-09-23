# -*- coding: utf-8 -*-
"""XGBoost 反序列化告警抑制（公共单一入口）。

背景：模型对象经 joblib/pickle 反序列化时，XGBoost 从 pickle 层
（``pickle.py`` 的 ``setstate``，无法通过 ``module="xgboost"`` 过滤）发出
若干已知无害的 UserWarning：

1. 跨版本 pickle 反序列化告警（``loading a serialized model``）——
   旧版本 xgboost 保存的 joblib 模型在新版本加载时提示重导出；
2. GPU 训练模型在无 GPU 环境加载时的设备回退告警
   （``grow_gpu_hist`` updater 更换 / ``No visible GPU is found`` /
   ``Device is changed from GPU to CPU``）。

两类告警均不影响推理结果（模型照常载入、CPU 上树结构一致），
但会污染纸面交易与回测日志。本模块按消息正则**精确屏蔽**上述模式，
其余 UserWarning 不受影响。

调用方（单一来源，禁止各自复制正则）：

- ``ml/model_registry.py``（选股模型加载，保留旧名 ``_suppress_xgboost_pickle_warning``）；
- ``risk/terminal_loss/policy_sidecar.py``（折模型默认加载器）；
- ``risk/terminal_loss/model.py``（``TerminalLossModel.load``）。
"""

import warnings
from contextlib import contextmanager
from typing import Iterator

#: 已知无害的 XGBoost 反序列化告警消息正则（re.match 从头匹配）
XGBOOST_PICKLE_WARNING_PATTERNS = (
    ".*loading a serialized model.*",
    ".*Changing updater from.*grow_gpu_hist.*",
    ".*No visible GPU is found.*",
    ".*Device is changed from GPU to CPU.*",
)


@contextmanager
def suppress_xgboost_pickle_warning() -> Iterator[None]:
    """抑制 XGBoost 反序列化链路的已知告警（见模块 docstring）。"""
    with warnings.catch_warnings():
        for pattern in XGBOOST_PICKLE_WARNING_PATTERNS:
            warnings.filterwarnings("ignore", category=UserWarning, message=pattern)
        yield
