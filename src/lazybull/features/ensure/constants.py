# -*- coding: utf-8 -*-
"""ensure 子包常量：warmup 窗口与因子下载阈值。"""

# 与 build_clean_features.py 保持一致：
# - 过去约 7 个月用于覆盖 120 交易日 warmup
# - 向后扩展 1 个月保持离线构建口径（require_label=False 时不会因标签使用未来数据）
FEATURE_DATA_HISTORY_MONTHS = 7
FEATURE_DATA_FUTURE_MONTHS = 1
HISTORICAL_DATA_MONTHS = FEATURE_DATA_HISTORY_MONTHS
# 最多检查最近 N 个交易日 clean 分区，确保 warmup 期间缺口可被自动补齐
MAX_HISTORICAL_DAYS = 180

# 因子数据最低记录数阈值，低于此值视为数据不足，触发全量下载
# 这些因子是 point-in-time 查询，需要全量历史才有意义
_MIN_FINA_RECORDS = 1000       # 财务指标：全量应有 10 万+ 条
_MIN_HOLDER_RECORDS = 500      # 股东人数：全量应有数万条
_MIN_FORECAST_RECORDS = 500    # 业绩预告：全量应有数万条
_MIN_EXPRESS_RECORDS = 500        # 业绩快报：全量应有数万条
_MIN_REPORT_RC_RECORDS = 1000     # 一致预期研报：全量应有数万条
_MIN_CASHFLOW_RECORDS = 1000      # 现金流量表：全量应有数万条
_FINA_REQUIRED_RAW_COLS = [
    "q_gr_yoy",
    "q_ocf_to_sales",
    "int_to_talcap",
    "inv_turn",
    "update_flag",
]
