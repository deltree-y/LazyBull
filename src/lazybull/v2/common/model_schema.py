# -*- coding: utf-8 -*-
"""模型相关列族 schema 描述（唯一来源；P3-T2 上移，R2-02 / §16.1-A）。

落地内容（单一来源，消费方改经本模块引用，factors → common 合规）：

- 现金流质量列族：`CASHFLOW_QUALITY_SCHEMA_VERSION` / `CASHFLOW_QUALITY_VERSION_COL`
  / `cashflow_quality_live_columns`（原 `factors/cashflow_quality.py:43-44/:347-357`）；
- 分红政策列族：`DIVIDEND_POLICY_SCHEMA_VERSION` / `DIVIDEND_POLICY_VERSION_COL`
  （原 `factors/dividend.py:56-57`）；
- 一致预期修正列族：`CONSENSUS_REVISION_SCHEMA_VERSION` / `CONSENSUS_REVISION_VERSION_COL`
  （原 `factors/consensus_revision.py:54-55`；与 R2-02 同为「列族版本常量」schema 描述，
  同模式扩展上移——v2/train 禁 import factors，哨兵校验与 attach 函数需要无 factors 来源）；
- `read_*_schema_version` 三纯函数（原 `ml/train_core/constants.py:318-355`；
  推理共享常量，§2.2 处置表）。

common 层禁令（F14）：本模块不得 import 任何业务包（store/core/train/sleeves/factors/
universe/hosts/evidence/report/quality）；零项目内依赖。
"""

from typing import List

# ── 现金流质量列族 ─────────────────────────────────────────────
# 现金流质量因子 schema 哨兵：v3 = 依赖修订事件驱动 TTM + 确定性版本去重。
# handler 对当日全截面恒写当前版本号（含无数据股票），训练入口校验
# 哨兵列缺失、NaN 或版本不符必须失败；语义重做时递增哨兵值。
CASHFLOW_QUALITY_SCHEMA_VERSION = 3
CASHFLOW_QUALITY_VERSION_COL = "cashflow_quality_schema_v2"


def cashflow_quality_live_columns(feature_columns: List[str]) -> List[str]:
    """从训练门禁后的 feature_columns 中提取实际入模的现金流质量列（含 _sz 变体）。"""
    base = {
        "zscore_ocf_to_revenue",
        "zscore_ocf_to_profit",
        "zscore_fcf_yield",
        "zscore_capex_to_ocf",
        "cashflow_freshness_days",
    }
    sz = {f"{c}_sz" for c in base if c != "cashflow_freshness_days"}
    return sorted(c for c in feature_columns if c in base or c in sz)


# ── 分红政策列族 ───────────────────────────────────────────────
# 分红政策因子 schema 哨兵：handler 对当日全截面恒写当前版本号（含无数据股票），
# 训练入口校验哨兵列缺失、NaN 或版本不符必须失败；语义重做时递增哨兵值。
DIVIDEND_POLICY_SCHEMA_VERSION = 3
DIVIDEND_POLICY_VERSION_COL = "dividend_schema_v1"

# ── 一致预期修正列族 ───────────────────────────────────────────
CONSENSUS_REVISION_VERSION_COL = "cons_revision_schema_v2"
CONSENSUS_REVISION_SCHEMA_VERSION = 2


# ── train_params schema 版本读取（纯函数）─────────────────────
def read_cashflow_quality_schema_version(train_params: dict) -> int:
    """安全读取现金流质量 schema 版本；缺失或异常内容返回 -1。"""
    raw = (train_params or {}).get("cashflow_quality_schema_version")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return -1


def read_cons_revision_schema_version(train_params: dict) -> int:
    """安全读取修正 schema 版本；缺失或异常内容（如 "v2" 字符串）返回 -1。"""
    raw = (train_params or {}).get("cons_revision_schema_version")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return -1


def read_dividend_policy_schema_version(train_params: dict) -> int:
    """安全读取分红政策 schema 版本；缺失或异常内容返回 -1。"""
    raw = (train_params or {}).get("dividend_policy_schema_version")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return -1
