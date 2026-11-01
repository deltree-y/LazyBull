# -*- coding: utf-8 -*-
"""就绪门门 4 逐列缺失率**分级判据**（B′，2026-10-31 用户裁决）。

从 ``daily_assembly.py`` 拆出（复杂度契约 §4.4：新增功能对应新增文件）。

判据（全列统一分位，禁止按列配置）
==================================
基线 = 该列逐日截面缺失率分布的 **p95 / p99**（训练期静态分布，可由近 N 日
滚动窗口在线自校准）：

- 当日率 ≤ p95 ⇒ **pass**
- p95 < 当日率 ≤ p99 ⇒ **warning**（日志 + 结构化证据）
- 当日率 > p99 ⇒ **error**（日志 + 结构化证据）
- 当日率 = 100% ⇒ **error**（标记 ``all_nan``）

三档均**不阻断日跑**（B′ 决策 A1：单列质量波动不再整日停牌；注意信号层仍对
全空列硬拒预测，"不阻断"≠"当日必有信号"）。

唯一阻断形态 = **聚合升级**（决策 A2）：逐成员 error 列数 ≥
``max(15, ceil(20% × 该成员判定列数))`` ⇒ 登记缺口（``ok=False``）——防"大面积源
故障但只打日志、照常交易"。

分母同域（决策）
================
训练期分布与当日率**必须同一行域**（主极域 = 训练帧主板过滤后）；
``domain_codes`` 缺失 ⇒ 记 ``coverage_domain_missing`` error 且**不判定**
（无法同域比较，禁止静默退回全行制造口径漂移）。域内行数不足
``MIN_DOMAIN_ROWS`` 时只留证、不升级（极端停牌日方差放大）。

其余两类信息（列不在当日帧、metadata 缺失/非法）同样降为 error 日志
（决策 A1），不阻断。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import AbstractSet, Any, Mapping, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from src.lazybull.v2.common.model_schema import (
    FEATURE_MISSING_RATE_ALERT_QUANTILE,
    FEATURE_MISSING_RATE_ERR_METADATA_INVALID,
    FEATURE_MISSING_RATE_ERR_METADATA_MISSING,
    FEATURE_MISSING_RATE_ERR_SCHEMA_UNSUPPORTED,
    FEATURE_MISSING_RATE_ERROR_QUANTILE,
    read_feature_missing_rate,
)

__all__ = [
    "AGG_ERROR_FRACTION",
    "AGG_MIN_ERROR_COLUMNS",
    "COVERAGE_ERR_ALL_NAN",
    "COVERAGE_ERR_DOMAIN_EMPTY",
    "COVERAGE_ERR_DOMAIN_MISSING",
    "COVERAGE_ERR_EXEMPT_UNKNOWN",
    "COVERAGE_ERR_KEY_MISSING",
    "COVERAGE_ERR_RATE_EXCEEDED",
    "COVERAGE_ERR_RATE_INVALID",
    "COVERAGE_ERR_ROW_DOMAIN",
    "COVERAGE_ERR_ROW_DOMAIN_MISSING",
    "COVERAGE_ERR_UNCOVERED",
    "EXPECTED_ROW_DOMAIN",
    "COVERAGE_RATE_FLOOR",
    "COVERAGE_SEVERITY_ERROR",
    "COVERAGE_SEVERITY_PASS",
    "COVERAGE_SEVERITY_WARN",
    "CoverageMember",
    "CoverageOutcome",
    "MIN_DOMAIN_ROWS",
    "ROLLING_MIN_DAYS",
    "ROLLING_WINDOW_DAYS",
    "classify_severity",
    "evaluate_coverage",
]

#: 分级语义（三档均不阻断；仅聚合升级阻断）
COVERAGE_SEVERITY_PASS = "pass"
COVERAGE_SEVERITY_WARN = "warning"
COVERAGE_SEVERITY_ERROR = "error"

#: 告警/错误归因码（与 model_schema 读侧三码共同构成可行动清单）
COVERAGE_ERR_KEY_MISSING = "coverage_key_missing"
COVERAGE_ERR_RATE_EXCEEDED = "rate_exceeded"
COVERAGE_ERR_ALL_NAN = "all_nan"
COVERAGE_ERR_UNCOVERED = "uncovered"
COVERAGE_ERR_DOMAIN_MISSING = "coverage_domain_missing"
COVERAGE_ERR_DOMAIN_EMPTY = "coverage_domain_empty"
COVERAGE_ERR_RATE_INVALID = "coverage_rate_invalid"
COVERAGE_ERR_ROW_DOMAIN = "coverage_row_domain_mismatch"
COVERAGE_ERR_ROW_DOMAIN_MISSING = "coverage_row_domain_missing"
COVERAGE_ERR_EXEMPT_UNKNOWN = "coverage_exempt_unknown_column"

#: 训练侧/补录声明的行域（两侧同域契约的唯一合法值）
EXPECTED_ROW_DOMAIN = "train_filtered_main_board"

#: 聚合升级（决策 A2；2026-11-01 用户裁决改为 max(15, 20%)）：
#: error 列数 ≥ max(15, ceil(20% × 判定列数)) ⇒ 阻断（防大面积源故障照常交易）。
#: 157 判定列 ⇒ 阈值 32；明显放宽（原 max(8,10%) ⇒ 16），单日 13 列全空类
#: 事件将不再阻断——该放宽为用户显式裁决，观察期须复核。
AGG_MIN_ERROR_COLUMNS = 15
AGG_ERROR_FRACTION = 0.20

#: 滚动分位（决策 B3）：近 60 交易日窗口、最少 20 日；不足 ⇒ 回落训练期静态分布
ROLLING_WINDOW_DAYS = 60
ROLLING_MIN_DAYS = 20

#: 域内行数下限：低于此值只留证不升级（极端停牌日分母过小方差放大）
MIN_DOMAIN_ROWS = 1000

#: 绝对率下限（B′ 决策 b，2026-10-31 用户裁决，**显式评审项**）：近零基线列的
#: 训练期分位可能低至 0.04%~1.0%（`fund_count`/`zscore_profit_dedt`/
#: `zscore_q_gr_yoy`/`forecast_type_score` 等），当日任何微小缺失即超阈 ⇒ 慢性
#: error 噪音。分级阈值统一下限为 ``max(分位, 本值)``：低于 2% 的差异视为不可
#: 测量的噪声。取值依据 = 76 日对照敏感性拐点（F=0.5%/1%/2%/3%/5% ⇒ 超 20%
#: 交易日列数 11/8/6/6/6，2% 起稳定）；调整须重跑 76 日对照并登记。**只压缩
#: error/warning 阈值，不改变聚合升级、全空硬告警与同域规则**。
COVERAGE_RATE_FLOOR = 0.02

#: 成员级 metadata 错误文案（三类均 error 日志，不阻断）
MEMBER_ERR_GAP_TEXT: dict[str, str] = {
    FEATURE_MISSING_RATE_ERR_METADATA_MISSING: (
        "模型 metadata 缺 feature_missing_rate 分布（B′：仅告警不阻断；请补录/重训以恢复基线）"
        ": 成员 {member_id}"
    ),
    FEATURE_MISSING_RATE_ERR_SCHEMA_UNSUPPORTED: (
        "feature_missing_rate schema 版本不符（v2 分布摘要方可分级；仅告警不阻断）：成员 {member_id}"
    ),
    FEATURE_MISSING_RATE_ERR_METADATA_INVALID: (
        "feature_missing_rate 分布非法（字段/值域/分位序异常；仅告警不阻断）：成员 {member_id}"
    ),
}


@dataclass(frozen=True)
class CoverageMember:
    """门 4 判定的成员判定面（EnsembleSignal A/B 逐成员判定）。

    单模型缺省形态 = 单一成员（由 ``assemble_daily_data`` 以 feature_columns +
    train_params 构造）；多成员由宿主装配层（factory）从各子模型 metadata 派生
    （成员标识 = 模型 ``version_str``）。判定列集 = **该成员自己的**
    manifest/来源分类列集 − 该成员 freshness 闭包（``member_rate_columns``）。
    """

    member_id: str
    columns: Sequence[str]
    train_params: Mapping[str, Any]
    #: 结构性漂移单列豁免（B′ 决策 a）：该列 error 封顶 warning，不计数不阻断
    drift_exempt: frozenset = frozenset()


@dataclass
class CoverageOutcome:
    """门 4 判定产出（分级 + 结构化证据；gaps 仅聚合升级时非空）。"""

    rates: dict[str, float] = field(default_factory=dict)
    missing_columns: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    verdicts: list[dict[str, Any]] = field(default_factory=list)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    members_echo: list[dict[str, Any]] = field(default_factory=list)
    extra_keys: list[str] = field(default_factory=list)
    member_aggregates: list[dict] = field(default_factory=list)
    blocked: bool = False
    blocked_reason: str = ""
    rate_denominator: str = ""
    n_rows_used: int = 0
    n_rows_total: int = 0
    error_columns: int = 0
    warning_columns: int = 0
    error_threshold: int = 0


def _alert(
    severity: str,
    code: str,
    message: str,
    *,
    member: str | None = None,
    column: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    entry = {
        "severity": severity,
        "code": code,
        "message": message,
        "member": member,
        "column": column,
    }
    entry.update(extra)
    return entry


def main_board_codes(stock_basic: Any) -> AbstractSet[str] | None:
    """从 ``stock_basic`` 构建判定域（主板；与训练帧 ``row_domain`` 同域）。

    缺 ``market`` 列/空表 ⇒ None（门侧记 ``coverage_domain_missing`` 且不判定）。
    """
    columns = getattr(stock_basic, "columns", [])
    if stock_basic is None or "market" not in columns:
        return None
    board = stock_basic[stock_basic["market"] == "主板"]
    codes = frozenset(board["ts_code"].astype(str).tolist())
    return codes or None


def _restrict_to_domain(day_frame: pd.DataFrame, domain_codes: AbstractSet[str]) -> pd.DataFrame:
    """按域股票集合限定当日判定行（两侧同域的唯一实现点）。"""
    if "ts_code" not in day_frame.columns:
        raise ValueError("当日帧缺 ts_code 列，无法按域限定判定行（禁止静默用全行）")
    mask = day_frame["ts_code"].astype(str).isin(domain_codes)
    return day_frame[mask]


def _rolling_baseline(
    column: str,
    rate_history: Mapping[str, Mapping[str, float]] | None,
    current_date: str | None = None,
) -> tuple[float, float, str, int] | None:
    """近 N 日滚动分位（在线自校准，决策 B3）。

    **as-of 严格早于判定日**（reviewer1 P1-B / reviewer3 P1-2）：``current_date``
    给定时只取 ``day < current_date`` 的样本——禁止用当日（同日重跑/幂等重跑）或
    未来日（历史回放）参与自身阈值。窗口样本 ≥ ``ROLLING_MIN_DAYS`` 才启用；
    返回 (p95, p99, "rolling", n)。滚动窗为显式窗口长度，不做 500 日 p99 降级
    （该规则只适用于训练期静态分布）。
    """
    if not rate_history:
        return None
    days = sorted(str(day) for day in rate_history)
    if current_date is not None:
        days = [day for day in days if day < str(current_date)]
    days = days[-ROLLING_WINDOW_DAYS:]
    values = [
        float(rate_history[day][column]) for day in days if column in (rate_history[day] or {})
    ]
    if len(values) < ROLLING_MIN_DAYS:
        return None
    series = np.asarray(values, dtype=float)
    p95 = float(np.percentile(series, FEATURE_MISSING_RATE_ALERT_QUANTILE * 100.0))
    p99 = float(np.percentile(series, FEATURE_MISSING_RATE_ERROR_QUANTILE * 100.0))
    return p95, p99, "rolling", int(series.size)


def classify_severity(
    rate: float, p95: float, p99: float, floor: float = COVERAGE_RATE_FLOOR
) -> str:
    """分级唯一实现（生产判定与影子/滚动复算共用，防两套口径漂移）。

    阈值下限 = ``max(分位, floor)``（决策 b）；``rate >= 1.0`` 恒为 error（全空）。
    """
    if rate >= 1.0:
        return COVERAGE_SEVERITY_ERROR
    if rate > max(float(p99), float(floor)):
        return COVERAGE_SEVERITY_ERROR
    if rate > max(float(p95), float(floor)):
        return COVERAGE_SEVERITY_WARN
    return COVERAGE_SEVERITY_PASS


def _judge_member(
    rate_frame: pd.DataFrame,
    member: CoverageMember,
    rate_columns: Sequence[str],
    distribution: Mapping[str, Mapping],
    rate_history: Mapping[str, Mapping[str, float]] | None,
    outcome: CoverageOutcome,
    current_date: str | None = None,
) -> int:
    """单成员逐列分级（返回该成员 error 列数）。"""
    errors = 0
    for col in rate_columns:
        if col not in rate_frame.columns:
            outcome.missing_columns.append(col)
            # P2-2：整列缺席计 error（比高缺失更严重的源故障，聚合保险丝不得失明）
            errors += 1
            outcome.alerts.append(
                _alert(
                    COVERAGE_SEVERITY_ERROR,
                    COVERAGE_ERR_UNCOVERED,
                    f"判定列不在当日组装结果（整列缺席，计 error；信号层将自行跳过该列）: {col}",
                    member=member.member_id,
                    column=col,
                )
            )
            continue
        rate = float(rate_frame[col].isna().mean())
        if not math.isfinite(rate) or not (0.0 <= rate <= 1.0):
            # P1-3：非有限/越界率不得映射为 pass（NaN 比较恒 False 会静默放行）
            errors += 1
            outcome.alerts.append(
                _alert(
                    COVERAGE_SEVERITY_ERROR,
                    COVERAGE_ERR_RATE_INVALID,
                    f"当日缺失率非有限或越界（不可判定，计 error）: {col} rate={rate!r}",
                    member=member.member_id,
                    column=col,
                    rate=rate,
                )
            )
            continue
        outcome.rates[col] = rate
        stats = distribution.get(col)
        if stats is None:
            outcome.alerts.append(
                _alert(
                    COVERAGE_SEVERITY_ERROR,
                    COVERAGE_ERR_KEY_MISSING,
                    f"判定列在模型分布基线中无记录（列集合不匹配，仅告警不阻断）: {col}",
                    member=member.member_id,
                    column=col,
                    rate=rate,
                )
            )
            continue

        rolling = _rolling_baseline(col, rate_history, current_date)
        if rolling is not None:
            p95, p99, baseline_source, baseline_days = rolling
        else:
            p95, p99 = float(stats["p95"]), float(stats["p99"])
            baseline_source = "static"
            baseline_days = int(stats.get("n_days") or 0)

        severity = classify_severity(rate, p95, p99)
        warn_threshold = max(p95, COVERAGE_RATE_FLOOR)
        error_threshold = max(p99, COVERAGE_RATE_FLOOR)
        drift_exempt = col in member.drift_exempt
        if severity == COVERAGE_SEVERITY_PASS:
            code = None
        elif rate >= 1.0:
            # 全空不豁免（数据断链硬信号）
            code = COVERAGE_ERR_ALL_NAN
            errors += 1
        elif drift_exempt and severity == COVERAGE_SEVERITY_ERROR:
            # 决策 a：结构性漂移列 error 封顶 warning，不参与聚合升级计数
            severity = COVERAGE_SEVERITY_WARN
            code = COVERAGE_ERR_RATE_EXCEEDED
        else:
            code = COVERAGE_ERR_RATE_EXCEEDED
            if severity == COVERAGE_SEVERITY_ERROR:
                errors += 1

        outcome.verdicts.append(
            {
                "member": member.member_id,
                "column": col,
                "rate": rate,
                "p95": p95,
                "p99": p99,
                "warn_threshold": warn_threshold,
                "error_threshold": error_threshold,
                "rate_floor": COVERAGE_RATE_FLOOR,
                "baseline_source": baseline_source,
                "baseline_days": baseline_days,
                "drift_exempt": drift_exempt,
                "severity": severity,
                "error_code": code,
            }
        )
        if severity != COVERAGE_SEVERITY_PASS:
            outcome.alerts.append(
                _alert(
                    severity,
                    code or COVERAGE_ERR_RATE_EXCEEDED,
                    f"{col} 当日缺失率 {rate:.1%} vs 阈值 "
                    f"warn={warn_threshold:.1%}/error={error_threshold:.1%}"
                    f"（{baseline_source}，{baseline_days} 日；下限 {COVERAGE_RATE_FLOOR:.0%}）",
                    member=member.member_id,
                    column=col,
                    rate=rate,
                    p95=p95,
                    p99=p99,
                    warn_threshold=warn_threshold,
                    error_threshold=error_threshold,
                    baseline_source=baseline_source,
                    drift_exempt=drift_exempt,
                )
            )
    return errors


def evaluate_coverage(
    day_frame: pd.DataFrame | None,
    member_faces: Sequence[tuple[CoverageMember, Sequence[str]]],
    *,
    domain_codes: AbstractSet[str] | None = None,
    rate_history: Mapping[str, Mapping[str, float]] | None = None,
    current_date: str | None = None,
) -> CoverageOutcome:
    """门 4（B′）：逐列分级判定；仅聚合升级阻断（逐成员判定）。

    ``member_faces`` = [(CoverageMember, member_rate_columns)]（判定列集由
    ``daily_assembly`` 按成员分类/freshness 闭包计算）。当日帧为空 ⇒ 跳过判定
    （缺口由门 1 登记）。``domain_codes`` 缺失 ⇒ 记 error 且不判定（禁止跨域比较）；
    域内 0 行 ⇒ ``coverage_domain_empty`` error 且不判定。``current_date`` 用于
    滚动基线 as-of 裁窗（严格早于判定日）。
    """
    outcome = CoverageOutcome()
    if day_frame is None or day_frame.empty:
        outcome.alerts.append(
            _alert(
                COVERAGE_SEVERITY_ERROR,
                COVERAGE_ERR_UNCOVERED,
                "当日 panel 截面为空，缺失率判定跳过（见门 1 分区存在性）",
            )
        )
        return outcome
    if not domain_codes:
        outcome.rate_denominator = "unavailable"
        outcome.alerts.append(
            _alert(
                COVERAGE_SEVERITY_ERROR,
                COVERAGE_ERR_DOMAIN_MISSING,
                "缺少判定域（主板股票集合）：无法与训练期分布同域比较，本次跳过缺失率判定"
                "（禁止静默退回全行比较）",
            )
        )
        return outcome

    rate_frame = _prepare_rate_frame(day_frame, domain_codes, outcome)
    if rate_frame is None:
        return outcome
    blocked_reasons = _judge_members(rate_frame, member_faces, rate_history, current_date, outcome)
    _finalize_aggregate(outcome, blocked_reasons)
    return outcome


def _prepare_rate_frame(
    day_frame: pd.DataFrame, domain_codes: AbstractSet[str], outcome: CoverageOutcome
) -> pd.DataFrame | None:
    """域裁剪 + 空域/小分母守卫（返回可判定帧；None = 不判定）。"""
    rate_frame = _restrict_to_domain(day_frame, domain_codes)
    outcome.rate_denominator = "main_board_rows"
    outcome.n_rows_used = int(len(rate_frame))
    outcome.n_rows_total = int(len(day_frame))
    if outcome.n_rows_used == 0:
        # P1-3：空交集不得逐个判 pass（NaN 比较恒 False 会静默放行）
        outcome.alerts.append(
            _alert(
                COVERAGE_SEVERITY_ERROR,
                COVERAGE_ERR_DOMAIN_EMPTY,
                "判定域与当日帧无交集（域内 0 行）：缺失率不可判定（计 error，不生成逐列判定）",
            )
        )
        return None
    if outcome.n_rows_used < MIN_DOMAIN_ROWS:
        outcome.alerts.append(
            _alert(
                COVERAGE_SEVERITY_WARN,
                "coverage_domain_rows_insufficient",
                f"域内行数 {outcome.n_rows_used} < {MIN_DOMAIN_ROWS}（极端停牌/数据缺口？）"
                "——分母过小，判定只留证不升级",
            )
        )

    return rate_frame


def _judge_members(
    rate_frame: pd.DataFrame,
    member_faces: Sequence[tuple[CoverageMember, Sequence[str]]],
    rate_history: Mapping[str, Mapping[str, float]] | None,
    current_date: str | None,
    outcome: CoverageOutcome,
) -> list:
    """逐成员判定 + 逐成员失配/错误证据；返回阻断原因列表。"""
    blocked_reasons: list[str] = []
    for member, rate_columns in member_faces:
        distribution, error_code = read_feature_missing_rate(member.train_params)
        outcome.members_echo.append(
            {
                "member_id": member.member_id,
                "n_columns": len(list(member.columns)),
                "n_distribution_keys": len(distribution) if distribution is not None else None,
                "distribution_error": error_code,
            }
        )
        if error_code is not None:
            outcome.alerts.append(
                _alert(
                    COVERAGE_SEVERITY_ERROR,
                    error_code,
                    MEMBER_ERR_GAP_TEXT[error_code].format(member_id=member.member_id),
                    member=member.member_id,
                )
            )
            continue
        assert distribution is not None
        # P2-2（reviewer3）：行域声明与门侧实际域绑定；声明不一致 ⇒ 该成员不判定
        stat = (member.train_params or {}).get("feature_missing_rate_stat")
        declared_domain = stat.get("row_domain") if isinstance(stat, Mapping) else None
        if declared_domain != EXPECTED_ROW_DOMAIN:
            if declared_domain is None:
                outcome.alerts.append(
                    _alert(
                        COVERAGE_SEVERITY_WARN,
                        COVERAGE_ERR_ROW_DOMAIN_MISSING,
                        f"metadata 未声明 row_domain（应为 {EXPECTED_ROW_DOMAIN}）：按现状继续判定，"
                        "建议补录声明（missing_stat）",
                        member=member.member_id,
                    )
                )
            else:
                outcome.alerts.append(
                    _alert(
                        COVERAGE_SEVERITY_ERROR,
                        COVERAGE_ERR_ROW_DOMAIN,
                        f"metadata 行域 {declared_domain!r} ≠ 门侧同域契约 {EXPECTED_ROW_DOMAIN!r}"
                        "：拒绝用该成员分布判定（禁止跨域套用）",
                        member=member.member_id,
                    )
                )
                continue
        # 豁免列必须落在该成员分布键集内（reviewer3 P1-4）；任一未知 ⇒ 豁免整块不生效
        exempt_set = set(member.drift_exempt)
        unknown_exempt = sorted(exempt_set - set(distribution))
        if unknown_exempt:
            outcome.alerts.append(
                _alert(
                    COVERAGE_SEVERITY_ERROR,
                    COVERAGE_ERR_EXEMPT_UNKNOWN,
                    f"豁免登记含分布外的未知列 {unknown_exempt}：整块豁免不生效（all-or-nothing）",
                    member=member.member_id,
                )
            )
            exempt_set = set()
        effective_member = member
        if exempt_set != set(member.drift_exempt):
            effective_member = type(member)(
                member_id=member.member_id,
                columns=member.columns,
                train_params=member.train_params,
                drift_exempt=frozenset(exempt_set),
            )
        member_errors = _judge_member(
            rate_frame,
            effective_member,
            rate_columns,
            distribution,
            rate_history,
            outcome,
            current_date,
        )
        member_judged = max(len(list(rate_columns)), 1)
        member_threshold = max(
            AGG_MIN_ERROR_COLUMNS, int(math.ceil(AGG_ERROR_FRACTION * member_judged))
        )
        member_blocked = (
            outcome.n_rows_used >= MIN_DOMAIN_ROWS and member_errors >= member_threshold
        )
        outcome.member_aggregates.append(
            {
                "member_id": member.member_id,
                "error_columns": member_errors,
                "n_judged_columns": member_judged,
                "error_threshold": member_threshold,
                "blocked": member_blocked,
            }
        )
        if member_blocked:
            blocked_reasons.append(
                f"成员 {member.member_id}: {member_errors} 列 ≥ 阈值 {member_threshold}"
            )
        extra = sorted(set(distribution) - set(rate_columns))
        if extra:
            outcome.extra_keys.extend(extra)

    return blocked_reasons


def _finalize_aggregate(outcome: CoverageOutcome, blocked_reasons: Sequence[str]) -> None:
    """聚合计数/证据与阻断落盘（分子分母逐成员同集合）。"""
    outcome.extra_keys = sorted(set(outcome.extra_keys))
    if outcome.extra_keys:
        # 合法超集（derived/freshness 等不参与判定的 metadata 键）：留证不告警
        logger.debug(f"coverage 多余键（合法超集，证据留证）: {outcome.extra_keys}")

    # 计数口径与分母同集合（reviewer3 P1-1）：分子取逐成员 error 的最大值，
    # 阈值取逐成员阈值最大值；不再用成员并集列数配"全体验证条目"分子
    outcome.error_columns = max(
        (item["error_columns"] for item in outcome.member_aggregates), default=0
    )
    outcome.error_threshold = max(
        (item["error_threshold"] for item in outcome.member_aggregates),
        default=max(AGG_MIN_ERROR_COLUMNS, 1),
    )
    outcome.warning_columns = sum(
        1 for v in outcome.verdicts if v["severity"] == COVERAGE_SEVERITY_WARN
    )
    if blocked_reasons:
        outcome.blocked = True
        outcome.blocked_reason = (
            "数据质量聚合升级（决策 A2）："
            + "；".join(blocked_reasons)
            + f"（逐成员阈值 = max({AGG_MIN_ERROR_COLUMNS}, "
            f"{AGG_ERROR_FRACTION:.0%}×该成员判定列数)）——大面积数据源故障，拒绝生成信号"
        )
        outcome.gaps.append(outcome.blocked_reason)
    if outcome.warning_columns or outcome.error_columns:
        logger.warning(
            f"就绪门缺失率分级：error {outcome.error_columns} 列 / warning {outcome.warning_columns} 列"
            f"（阈值 {outcome.error_threshold}；三档均不阻断，聚合超限才阻断）"
        )
