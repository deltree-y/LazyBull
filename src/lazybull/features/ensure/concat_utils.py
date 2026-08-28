# -*- coding: utf-8 -*-
"""ensure 子包：pd.concat 的 FutureWarning 屏蔽辅助（只屏蔽告警，不改数据）。"""

import warnings
from typing import List

import pandas as pd

# pandas 对 concat 中 empty/all-NA entries 的 FutureWarning: 这是未来 dtype 推断
# 行为变更的提示, 不影响当前结果。不为此对数据做任何剔除/补列处理 (否则会破坏
# raw 层 schema, 导致训练用到列、预测时列被删), 仅屏蔽该告警。
# 与 scripts/raw_download/periodic.py 的 _concat_no_warning、
# src/lazybull/data/loader.py 的 _concat_no_all_na_warning 为同一既定模式。
# 注意: filterwarnings 的 message 用 re.match 从头匹配, 需带 "The behavior of " 前缀。
_CONCAT_ALL_NA_WARNING = (
    r"The behavior of DataFrame concatenation with empty or all-NA entries is deprecated"
)


def _concat_no_warning(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """原样 concat, 仅屏蔽 pandas 的 empty/all-NA entries FutureWarning。"""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=_CONCAT_ALL_NA_WARNING,
            category=FutureWarning,
        )
        return pd.concat(frames, ignore_index=True)
