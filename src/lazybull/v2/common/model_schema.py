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

from typing import Any, List, Mapping, Optional, Sequence, Tuple

from loguru import logger

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


# ── 选股过滤列（就绪门 / 训练侧共用单一来源）──────────────────
#: 选股过滤列（MLSignal 选股过滤消费列，不在模型 feature_columns 内）。原唯一
#: 来源 = ``v2/hosts/paper/daily_assembly.py``；上移本中立层（P4 后维护项 B
#: 方案，reviewer2 复审 P1-A-2 处置）：训练侧采集过滤列训练期缺失率与宿主
#: 就绪门判据共用同一常量，且 train 包不 import hosts（orchestrator 分层契约）。
#: ``daily_assembly`` 保留兼容 re-export。
SELECTION_FILTER_COLUMNS: Tuple[str, ...] = ("amount_ma20", "total_mv", "sw_l1_code")


# ── 逐列训练期缺失率分布（feature_missing_rate）schema v2 ──────
#: schema 版本 2（2026-10-31 用户裁决 B′）：``train_params["feature_missing_rate"]``
#: = dict[列名 → **逐日缺失率分布摘要**]：
#: ``{p50, p90, p95, p99, max, mean, n_days, p99_unstable}``（0~1 小数）。
#: 就绪门判据（分级、不阻断，见 ``coverage_gate``）= 当日率 ≤ p95 ⇒ pass；
#: p95 < 当日率 ≤ p99 ⇒ warning；当日率 > p99 ⇒ error；三档均不阻断预测。
#: 分母口径两侧必须同域（决策：主板域；见 ``feature_missing_rate_stat.row_domain``）。
#: 键集注册侧精确 = 最终注册列 ∪ ``SELECTION_FILTER_COLUMNS``；v1（单值 μ）与
#: 缺键均视为 ``metadata_schema_unsupported``/``metadata_missing``（不隐式回退）。
FEATURE_MISSING_RATE_SCHEMA_VERSION = 2

#: 分级分位（全列统一，禁止按列配置；见设计 v4 §2）
FEATURE_MISSING_RATE_ALERT_QUANTILE = 0.95
FEATURE_MISSING_RATE_ERROR_QUANTILE = 0.99

#: p99 稳定性样本下限：训练窗交易日 < 500 时 p99 退化为 p95 并标注
#: ``p99_unstable=true``（1454 日窗下 p99≈第 15 个最大值，短窗时尾部不可靠）
FEATURE_MISSING_RATE_MIN_DAYS_FOR_P99 = 500

#: 读侧/门侧 coverage 校验错误码（语义见设计文档；B′：三类均降为 error 日志）
FEATURE_MISSING_RATE_ERR_METADATA_MISSING = "metadata_missing"
FEATURE_MISSING_RATE_ERR_SCHEMA_UNSUPPORTED = "metadata_schema_unsupported"
FEATURE_MISSING_RATE_ERR_METADATA_INVALID = "metadata_invalid"

_DIST_FIELDS = ("mean", "p50", "p90", "p95", "p99", "max")


def _as_rate(value: Any, *, field: str, column: str) -> float:
    """值域校验（0~1 小数；bool/NaN/±Inf 拒绝）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"feature_missing_rate 非法（列 {column} 字段 {field} 非数值）")
    value_f = float(value)
    if value_f != value_f or value_f in (float("inf"), float("-inf")):
        raise ValueError(f"feature_missing_rate 非法（列 {column} 字段 {field} NaN/Inf）")
    if not (0.0 <= value_f <= 1.0):
        raise ValueError(f"feature_missing_rate 非法（列 {column} 字段 {field}={value_f}，须 0~1）")
    return value_f


def summarize_daily_missing_rates(
    daily_rates: Mapping[str, Sequence[float]],
    *,
    alert_quantile: float = FEATURE_MISSING_RATE_ALERT_QUANTILE,
    error_quantile: float = FEATURE_MISSING_RATE_ERROR_QUANTILE,
    min_days_for_p99: int = FEATURE_MISSING_RATE_MIN_DAYS_FOR_P99,
) -> dict:
    """逐列逐日缺失率序列 → 分布摘要（训练采集与补录工具共用，单一实现）。

    - ``p95``/``p99`` 用线性插值分位（``numpy.percentile`` 缺省口径，确定性）；
    - ``n_days < min_days_for_p99`` ⇒ p99 退化为 p95 并标 ``p99_unstable=True``
      （短训练窗尾部估计不可靠，用户决策 A3）；
    - 返回原生 ``dict[str, dict]``（JSON 可序列化，禁止 NumPy 标量）。
    """
    import numpy as np

    summary: dict = {}
    for column, values in daily_rates.items():
        series = np.asarray([float(v) for v in values], dtype=float)
        series = series[~np.isnan(series)]
        n_days = int(series.size)
        if n_days == 0:
            raise ValueError(f"列 {column} 无逐日缺失率样本（训练帧为空？）")
        p95 = float(np.percentile(series, alert_quantile * 100.0))
        unstable = n_days < int(min_days_for_p99)
        p99 = p95 if unstable else float(np.percentile(series, error_quantile * 100.0))
        summary[column] = {
            "mean": float(series.mean()),
            "p50": float(np.percentile(series, 50.0)),
            "p90": float(np.percentile(series, 90.0)),
            "p95": p95,
            "p99": p99,
            "max": float(series.max()),
            "n_days": n_days,
            "p99_unstable": bool(unstable),
        }
    return summary


#: 结构性漂移单列豁免（B′ 决策 a，2026-11-01 用户批准）：可选块
#: ``train_params["feature_missing_rate_drift_exempt"]`` = 登记对象——豁免列表必须
#: 携带登记时间/理由/证据/复核条件（禁止裸列名，防"静默白名单"）。语义 = 该列
#: 分级 error **封顶为 warning**（不参与聚合升级计数）；全空（100%）不豁免。
DRIFT_EXEMPT_KEY = "feature_missing_rate_drift_exempt"
_DRIFT_EXEMPT_REQUIRED = ("columns", "registered_at", "review_condition")


def read_drift_exempt(train_params: Optional[Mapping]) -> frozenset:
    """读取单列豁免清单（结构不合法 ⇒ 返回空集 + 告警，不阻断交易）。

    只有 ``columns``（字符串列表）+ ``registered_at`` + ``review_condition`` 三要素
    齐备且非空时才生效；任何形式缺失/非法均视为未登记（豁免不会静默生效）。
    """
    raw = (train_params or {}).get(DRIFT_EXEMPT_KEY)
    if not isinstance(raw, Mapping):
        return frozenset()
    columns = raw.get("columns")
    missing = [
        key for key in _DRIFT_EXEMPT_REQUIRED if not isinstance(raw.get(key), (str, list, tuple))
    ]
    # all-or-nothing（reviewer3 P1-4）：任一列非字符串/空串、审计字段非字符串 ⇒ 整块不生效
    bad_columns = (
        not isinstance(columns, (list, tuple))
        or not columns
        or any(not isinstance(col, str) or not col.strip() for col in columns)
    )
    bad_audit = any(
        not isinstance(raw.get(key), str) or not str(raw.get(key)).strip()
        for key in ("registered_at", "review_condition")
    )
    if missing or bad_columns or bad_audit:
        logger.warning(
            f"{DRIFT_EXEMPT_KEY} 登记结构不合法（缺 {missing or '三要素'} / 非法列或审计字段）"
            "——整块不启用（防部分生效/静默白名单）"
        )
        return frozenset()
    return frozenset(col.strip() for col in columns)


def _parse_distribution_entry(column: Any, value: Any) -> Optional[dict]:
    """单列分布摘要校验（读侧拆分件；非法 ⇒ None）。

    全字段一致校验（reviewer3 P1-4）：六字段全部为有限 [0,1] 数值、完整分位序
    ``p50 ≤ p90 ≤ p95 ≤ p99 ≤ max``、``n_days`` 为**整数**（拒绝 2.9/字符串/bool）、
    ``p99_unstable`` 必须为 bool、``mean`` 不越界。
    """
    if not isinstance(column, str) or not column.strip():
        return None
    if not isinstance(value, Mapping):
        return None
    try:
        stats = {
            field: _as_rate(value.get(field), field=field, column=column) for field in _DIST_FIELDS
        }
    except (TypeError, ValueError):
        return None
    n_days_raw = value.get("n_days")
    if not isinstance(n_days_raw, int) or isinstance(n_days_raw, bool) or n_days_raw < 1:
        return None
    unstable = value.get("p99_unstable", False)
    if not isinstance(unstable, bool):
        return None
    ordered = [stats["p50"], stats["p90"], stats["p95"], stats["p99"], stats["max"]]
    if any(left > right for left, right in zip(ordered, ordered[1:])):
        return None
    stats["n_days"] = n_days_raw
    stats["p99_unstable"] = unstable
    return stats


def read_feature_missing_rate(
    train_params: Optional[Mapping],
) -> Tuple[Optional[dict], Optional[str]]:
    """读取并校验 ``feature_missing_rate`` 分布摘要（读侧纯函数，门禁/工具共用）。

    返回 (distribution dict, error_code)，两者互斥——合法 ⇒ (dict, None)；
    非法 ⇒ (None, code)。三码均为**告警语义**（不阻断，见 B′ 决策 A1），
    但必须显式可见、禁止静默回退：

    - ``metadata_missing``：train_params 无该键（旧模型未补录）；
    - ``metadata_schema_unsupported``：版本键缺失或非 v2（含 v1 单值形态）；
    - ``metadata_invalid``：非 dict / 空 dict / 非字符串键 / 字段缺省或非数值 /
      bool / NaN / ±Inf / 越界 / 分位序不满足 ``p50≤p90≤p95≤p99≤max`` /
      ``n_days`` 非整数或 <1 / ``p99_unstable`` 非 bool。
    """
    params = dict(train_params or {})
    if "feature_missing_rate" not in params:
        return None, FEATURE_MISSING_RATE_ERR_METADATA_MISSING
    version_raw = params.get("feature_missing_rate_schema_version")
    # 严格整数（reviewer3 P1-4）：拒绝 2.9 / "2" / True / inf —— 不做隐式取整或宽容转换
    if not isinstance(version_raw, int) or isinstance(version_raw, bool):
        return None, FEATURE_MISSING_RATE_ERR_SCHEMA_UNSUPPORTED
    if version_raw != FEATURE_MISSING_RATE_SCHEMA_VERSION:
        return None, FEATURE_MISSING_RATE_ERR_SCHEMA_UNSUPPORTED
    raw = params["feature_missing_rate"]
    if not isinstance(raw, Mapping) or not raw:
        return None, FEATURE_MISSING_RATE_ERR_METADATA_INVALID
    distribution: dict = {}
    for key, value in raw.items():
        stats = _parse_distribution_entry(key, value)
        if stats is None:
            return None, FEATURE_MISSING_RATE_ERR_METADATA_INVALID
        distribution[key] = stats
    return distribution, None


def build_feature_missing_rate_entry(
    candidate_distributions: Mapping[str, Mapping],
    final_feature_columns: Sequence[str],
) -> dict:
    """注册前投影：键集**精确** = 最终注册列 ∪ ``SELECTION_FILTER_COLUMNS``。

    两条注册路径（v2 trainer / v2 deploy）与补录工具共用。候选缺键（最终列不在
    候选集）⇒ ValueError（写入侧异常，禁止注册无分布基线的模型）；值域/分位序
    校验同读侧口径。返回原生可序列化 dict。
    """
    final_keys = list(dict.fromkeys([*final_feature_columns, *SELECTION_FILTER_COLUMNS]))
    projected: dict = {}
    missing_keys: list = []
    for col in final_keys:
        if col not in candidate_distributions:
            missing_keys.append(col)
            continue
        candidate = candidate_distributions[col]
        coverage, error = read_feature_missing_rate(
            {
                "feature_missing_rate_schema_version": FEATURE_MISSING_RATE_SCHEMA_VERSION,
                "feature_missing_rate": {col: candidate},
            }
        )
        if error is not None:
            raise ValueError(f"feature_missing_rate 分布非法（列 {col}）: {error}")
        projected[col] = coverage[col]
    if missing_keys:
        raise ValueError(
            f"feature_missing_rate 注册投影缺键（最终列不在候选统计集，写入侧异常）: "
            f"{sorted(set(missing_keys))}"
        )
    extra = sorted(set(candidate_distributions) - set(final_keys))
    if extra:
        logger.info(
            f"feature_missing_rate 注册投影剔除候选多余键 {len(extra)} 个（不写入 metadata）"
        )
    return projected
