# -*- coding: utf-8 -*-
"""模型列集审计常量。"""

FEATURE_FILE_GLOB = "*_features.json"
DEFAULT_MODEL_DIR = "data/models/stock_selection"
DEFAULT_LAST_VERSIONS = 40

# 频次表状态（中文展示口径，判定规则见 columns.frequency_table）
STATUS_ALWAYS = "常驻"
STATUS_NEW = "新增"
STATUS_REMOVED = "已移除"
STATUS_INTERMITTENT = "间断出现"
