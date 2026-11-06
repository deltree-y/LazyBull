# -*- coding: utf-8 -*-
"""P6 报告状态词、验证级别与降级原因码（设计 §5.1，机器可读唯一来源）。

设计原文口径：
- ``available``＝来源和门禁有效；``empty_valid``＝已核实为空；
  ``degraded``＝来源与身份可核对但存在已登记降级项（必须带 reason_code，且不得被
  趋势/可比性判断当作 available）；``not_available``＝缺文件/缺列/不支持/不可比/暂无来源；
  ``stale``＝保留有效旧值与原日期；``invalid``＝损坏、冲突、哈希或契约违例；
- ``recovered_generation_count_missing`` 是 reason_code，不是第七种状态；
- ``projection_only`` 是独立 verification_level，不是 report_status。

本模块只提供状态构造与校验的**纯函数**，不做任何 I/O；所有投影/审计模块共用同一
组状态词与原因码，禁止各自复制字面量。
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

__all__ = [
    "REASON_CODES",
    "STATUS_WORDS",
    "VERIFICATION_LEVELS",
    "StatusError",
    "degradation_entry",
    "is_usable",
    "make_status",
    "require_known_reason",
]

#: 六种报告状态（闭集，禁止新增第七种状态）
STATUS_WORDS: tuple[str, ...] = (
    "available",
    "empty_valid",
    "degraded",
    "not_available",
    "stale",
    "invalid",
)

#: 内容验证级别（与 report_status 正交）
VERIFICATION_LEVELS: tuple[str, ...] = ("authority_verified", "projection_only")

#: 降级原因码注册表（键即 reason_code，值即中文口径说明）
REASON_CODES: dict[str, str] = {
    "account_projection_missing": "权威账户投影文件缺失，无法与权威比对",
    "account_start_date_conflict": "extra 与顶层 account_start_date 同时存在且不一致",
    "annualization_input_invalid": "初始资金或总资产非正，年化不可计算",
    "annualization_start_after_nav": "账户起始日晚于 NAV 日，年化不可计算",
    "authority_mismatch": "投影内容与权威不一致（零修复，不猜值）",
    "authority_missing": "权威 engine_state.bin 缺失",
    "not_authority_verifiable": "当前日权威成交不可核对（不得宣称完成内容认证）",
    "authority_unreadable": "权威字节存在但反序列化/形态校验失败",
    "chain_metric_nonfinite": "链式指标输出含非有限值",
    "chain_rows_empty": "链式净值无数据行",
    "chain_rows_unparsable": "链式净值行日期不可解析",
    "chain_source_missing": "链式净值来源缺失",
    "config_missing": "配置文件缺失或非 dict 形态",
    "count_mismatch": "记录计数与文件行数/权威行数矛盾",
    "date_regression": "日期序列非严格递增（权威代或链路日期倒退）",
    "duplicate_chain_date": "链式净值同日重复行",
    "instruction_future_ambiguous": "存在多个未来执行日指令文件（跨代歧义）",
    "instruction_missing_future": "记录声明有指令但缺未来执行日文件",
    "instruction_source_date_mismatch": "未来指令行 source_date 不等于当前权威日",
    "invalid_json": "JSON 文件损坏或顶层类型不符",
    "missing_precondition": "缺少计算前置字段",
    "multiple_quality_value": "同一数据集摘要行出现多个不同值",
    "nav_before_account_date": "NAV 最新行早于账户日（保留旧值并标 stale）",
    "nav_row_missing": "当前日有成交但缺同日 NAV 行",
    "nav_row_unparsable": "NAV 行 trade_date 不可解析",
    "nav_source_missing": "NAV 持久化来源缺失",
    "no_persisted_actual_model": "当前 v2 代不持久化实跑模型（禁止回填）",
    "no_readonly_resolution": "latest 模型解析为只读不可用（禁止 Registry 回退写）",
    "no_source": "该字段当前没有合法来源",
    "nonfinite_chain_nav": "链式净值含 NaN/Inf/非正 NAV，整臂拒绝出指标",
    "nonfinite_value": "数值非有限",
    "pending_not_list": "pending 队列顶层不是 list",
    "quality_value_not_numeric": "质量摘要计数无法严格解析为非负整数",
    "rebalance_not_dict": "调仓状态顶层不是 dict",
    "recovered_generation_count_missing": "恢复代缺计数字段，但内容可由权威核对",
    "recovery_counts_missing": "恢复代缺计数字段且无法由权威核对",
    "run_record_pre_switch_shape": "切换前记录形态（同时缺 run_kind 与 engine）",
    "run_record_missing": "缺当前权威代的 run record 文件",
    "total_value_nonpositive": "NAV 总资产非正",
    "unstable_snapshot": "读取期间输入身份变化，重试后仍不稳定",
    "unsupported": "来源不支持只读访问（如远端 SMB）",
    "untrusted_root": "根或文件不满足受信本地条件（symlink/junction/硬链接/逃逸/远程）",
    "write_outside_whitelist": "写入白名单外路径",
    "bundle_seal_mismatch": "sealed bundle 封存哈希不一致或缺失",
    "bundle_field_missing": "sealed bundle 缺少契约字段",
    "bundle_chain_broken": "sealed bundle 哈希链断裂",
    "bundle_evidence_class": "sealed bundle 未声明 calibration_only/未标非前向证据",
    "bundle_path_escape": "sealed bundle 内路径逃逸",
    "manifest_seal_mismatch": "manifest 自封存哈希不一致",
    "manifest_schema_unsupported": "manifest schema 版本不支持",
    "output_sha_mismatch": "产物字节哈希与 manifest 登记不一致",
    "projection_sha_mismatch": "projection 哈希与 manifest 登记不一致",
    "input_drift": "输入身份在读取后发生变化",
}

#: 必须携带 reason_code 的状态（其余状态允许为空）
_REASON_REQUIRED: frozenset[str] = frozenset({"degraded", "not_available", "invalid", "stale"})


class StatusError(ValueError):
    """状态词/原因码/验证级别越界（契约闭集违反）。"""


def require_known_reason(reason_code: str) -> str:
    """校验原因码在注册表内；未知原因码直接报错（禁止自由文本冒充 reason）。"""
    if reason_code not in REASON_CODES:
        raise StatusError(f"未知降级原因码：{reason_code!r}（必须先在 REASON_CODES 登记）")
    return reason_code


def make_status(
    status: str,
    *,
    reason_code: str | None = None,
    source_ref: str | None = None,
    detail: str | None = None,
    verification_level: str = "projection_only",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """构造统一状态行；状态词/验证级别/原因码全部按闭集校验。"""
    if status not in STATUS_WORDS:
        raise StatusError(f"未知报告状态：{status!r}（闭集 {STATUS_WORDS}）")
    if verification_level not in VERIFICATION_LEVELS:
        raise StatusError(f"未知验证级别：{verification_level!r}（闭集 {VERIFICATION_LEVELS}）")
    if reason_code is not None:
        require_known_reason(reason_code)
    if status in _REASON_REQUIRED and reason_code is None:
        raise StatusError(f"状态 {status} 必须携带 reason_code")
    row: dict[str, Any] = {
        "report_status": status,
        "degradation_reason_code": reason_code,
        "verification_level": verification_level,
        "source_ref": source_ref,
        "detail": detail,
    }
    if extra:
        for key, value in extra.items():
            if key in row:
                raise StatusError(f"状态行扩展键与既有字段冲突：{key!r}")
            row[key] = value
    return row


def is_usable(row: Mapping[str, Any]) -> bool:
    """是否可作为“有效值”参与下游展示/可比性判断（degraded 不算 available）。"""
    return row.get("report_status") in ("available", "empty_valid")


def degradation_entry(
    row: Mapping[str, Any],
    *,
    field_path: str,
) -> dict[str, Any]:
    """把一条非可用状态行转登记项（manifest ``degradations[]`` 的唯一构造口径）。"""
    return {
        "field_path": field_path,
        "report_status": row.get("report_status"),
        "reason_code": row.get("degradation_reason_code"),
        "source_ref": row.get("source_ref"),
        "detail": row.get("detail"),
    }


def collect_degradations(
    entries: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """按 field_path 稳定排序并去重，保证 manifest 输出确定性。"""
    seen: dict[str, dict[str, Any]] = {}
    for entry in entries:
        key = str(entry.get("field_path"))
        if key not in seen:
            seen[key] = dict(entry)
    return [seen[key] for key in sorted(seen)]
