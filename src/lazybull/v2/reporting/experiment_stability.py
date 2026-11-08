# -*- coding: utf-8 -*-
"""跨期稳定性完整等价类（设计 §5.3.2，逐条对应七条判据）。

口径：

1. **单一分类**：``DYNAMIC_EXACT``（动态身份）与 ``STAT_OUTPUT``（统计/训练输出、``key_*``/
   ``KEY_*``）之外的字段默认进入策略等价类（含新增开关与 ``stock_domain``）；
2. **外层 stable_identity** = ``config_without_dynamic`` + host/实现 + 数据态 + 代码态 +
   数据水位；``group_key = canonical_json_sha256(stable_identity)``，该键不冒充 batch 指纹；
3. **打分前置硬断言**：缺 ``综合得分``/五个指标列/可用 PARAM_COLS 一律 ``missing_precondition``，
   **禁止调用**旧函数；组内 ``综合得分`` 不足两条有限值同样降级（不产生数值 stability_score）；
4. **旧函数分组投影**：对函数实际 ``group_cols ∩ DYNAMIC_EXACT`` 用统一占位值（``batch_run_id``
   不得二次拆组），``wf_run_id`` 保留以维持 ``run_id_list`` 追溯；只改分组键、不改任何指标；
5. **有效分组** = stable_identity ∧ 投影后 group_cols；每个分区独立调用
   ``build_period_stability_table``，列名唯一来源仍是 ``scripts/compare/aggregate.py``；
6. 无前置 / 期数不足由 P6 产生**显式状态行**（不冒充函数原输出、不私造 stability_score）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.compare.aggregate import (
    _build_period_label,
    build_period_stability_table,
    stability_group_columns,
)
from scripts.compare.constants import COL_NAMES, PARAM_COLS
from scripts.compare.scoring import compute_composite_score
from src.lazybull.v2.reporting import experiment_csv, status
from src.lazybull.v2.store.store_identity import canonical_json_sha256

__all__ = [
    "DYNAMIC_EXACT",
    "DYNAMIC_PLACEHOLDER",
    "REQUIRED_METRIC_COLUMNS",
    "STAT_OUTPUT",
    "STABLE_IDENTITY_FIELDS",
    "StabilityRecord",
    "build_stability_input",
    "build_stability_rows",
    "classify_field",
    "stable_identity_of",
]

#: 动态身份字段（可从策略等价类排除；与设计 §5.3.2-1 逐字一致）
DYNAMIC_EXACT: frozenset[str] = frozenset(
    {
        "wf_run_id",
        "batch_run_id",
        "batch_id",
        "run_id",
        "batch_period_label",
        "registered_at",
        "created_at",
        "split_index",
        "split_count",
        "wf_start_date",
        "wf_end_date",
        "final_date",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "val_start",
        "val_end",
        "start_model_version",
        "model_version",
        "bt_start",
        "bt_end",
    }
)
#: ``bt_metrics`` 的已知输出键（``ml/walk_forward/backtest.py::merge_backtest_metrics``）
_BT_OUTPUTS: tuple[str, ...] = (
    "bt_total_return",
    "bt_annual_return",
    "bt_max_drawdown",
    "bt_volatility",
    "bt_sharpe",
    "bt_calmar",
    "bt_trading_days",
    "bt_win_rate",
)
#: ``test_daily`` 的已知输出键（``ml/train_core/eval.py`` → ``summary.py`` 原键并入）。
#: 生产默认 TopK 集合 = ``{20,30,100,300}``（旧 runner / v2 orchestrator），另兼容自定义 K：
#: 仅当键**完整匹配** ``top<正整数>_return_(mean|std)`` 才归统计输出（不是宽前缀，R3-05-R2）。
_DAILY_OUTPUTS: tuple[str, ...] = (
    "daily_rankic_mean",
    "daily_rankic_std",
    "daily_rankic_ir",
)
_DAILY_OUTPUT_PATTERN = re.compile(r"^top[1-9]\d*_return_(?:mean|std)$")
#: 统计/训练输出字段（不是策略参数）。
#: 只按**明确列举的已知输出键**排除；``bt_top_n``/``bt_rebalance_freq``/
#: ``test_window_months``/``val_ratio`` 等真实策略参数必须留在等价类内（R3-05-R1），
#: 不得用整段 ``bt_*``/``test_*``/``val_*`` 前缀粗删。
STAT_OUTPUT: frozenset[str] = frozenset(
    {
        *_BT_OUTPUTS,
        *_DAILY_OUTPUTS,
        "train_samples",
        "val_samples",
        "test_samples",
        "best_iteration",
        "best_iteration_floor_triggered",
        "val_rankic_ir",
    }
)
#: 统计输出的列名前缀（未穷举的 ``key_*``/``KEY_*`` 与诊断列同属非策略参数；
#: ``bt_*/test_*/val_*`` 不作前缀剔除）
STAT_PREFIXES: tuple[str, ...] = ("key_", "KEY_", "diagnostic_")
#: 外层稳定身份字段（跨代码态/数据态明确不并组）
STABLE_IDENTITY_FIELDS: tuple[str, ...] = (
    "host_mode",
    "implementation",
    "data_state_id",
    "git_commit",
    "git_dirty",
    "data_daily_latest",
    "data_cs_train_latest",
    "data_dividend_coverage",
)
#: 动态身份统一占位值（只用于旧函数分组键，不写入任何指标）
DYNAMIC_PLACEHOLDER = "p6_dynamic"
#: 打分前置必需指标列（``COL_NAMES`` 唯一来源）
REQUIRED_METRIC_COLUMNS: tuple[str, ...] = (
    "综合得分",
    COL_NAMES["chain_cagr"],
    COL_NAMES["chain_max_drawdown"],
    COL_NAMES["chain_sharpe"],
    COL_NAMES["oos_cross_split_ir"],
    COL_NAMES["bt_win_rate"],
)
#: 中文列名 → 英文键（用于判断旧函数实际分组列是否动态身份）
_REVERSE_NAMES: dict[str, str] = {value: key for key, value in COL_NAMES.items()}
#: 函数输出列 → P6 列（``COL_NAMES`` 反向映射；未登记的中文键保留原样）
_OUTPUT_MAP: dict[str, str] = dict(_REVERSE_NAMES)
_OUTPUT_MAP["最新运行时间"] = "latest_run_time"


@dataclass
class StabilityRecord:
    """稳定性输入的一行（= 一次 run/批次的原值与元数据）。"""

    arm_id: str
    run_id: str
    config: dict[str, Any]
    host_mode: str | None
    implementation: str
    data_state_id: str | None
    watermarks: dict[str, Any]
    metrics: dict[str, Any]
    batch_id: str
    batch_run_id: str
    batch_period_label: str
    split_count: int
    final_date: str | None
    model_version: str | None
    extra: dict[str, Any] = field(default_factory=dict)


def classify_field(name: str) -> str:
    """字段单一分类：``dynamic`` / ``stat`` / ``stable``（未知默认 stable）。"""
    key = _REVERSE_NAMES.get(str(name), str(name))
    if key in DYNAMIC_EXACT:
        return "dynamic"
    if (
        key in STAT_OUTPUT
        or _DAILY_OUTPUT_PATTERN.fullmatch(str(name)) is not None
        or any(str(name).startswith(prefix) for prefix in STAT_PREFIXES)
    ):
        return "stat"
    return "stable"


def _canonical(value: Any) -> str:
    """canonical JSON（复用 store_identity 口径，禁止自造序列化）。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _jsonable(value: Any) -> Any:
    """JSON 可序列化归一（numpy 标量→Python 标量、NaN→None、嵌套结构递归）。

    等价类键必须能在两台机器上逐字节一致：numpy 标量在 canonical JSON 里不可序列化，
    这里显式归一，**不改变取值语义**（只改类型表示）。
    """
    if value is None:
        return None
    if isinstance(value, (bool,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y%m%d")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (int, str)):
        return value
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def stable_identity_of(record: StabilityRecord) -> dict[str, Any]:
    """外层稳定身份（策略配置去掉动态/统计字段 + host/实现/数据态/代码态/水位）。"""
    config = {
        str(key): _jsonable(value)
        for key, value in record.config.items()
        if classify_field(str(key)) == "stable"
    }
    identity: dict[str, Any] = {"config_without_dynamic": config}
    for name in STABLE_IDENTITY_FIELDS:
        if name in ("host_mode", "implementation", "data_state_id"):
            identity[name] = getattr(record, name)
        else:
            identity[name] = _jsonable(record.watermarks.get(name))
    return identity


def _group_key(identity: Mapping[str, Any]) -> str:
    """稳定等价类键（canonical JSON SHA-256；不冒充 batch 配置指纹）。"""
    return canonical_json_sha256(dict(identity))


def build_stability_input(
    records: Sequence[StabilityRecord], *, with_composite: bool = True
) -> pd.DataFrame:
    """构造旧管线输入（中文列名，PARAM_COLS 用 ``COL_NAMES`` 映射后的名称）。

    ``with_composite=True`` 时调用既有 ``compute_composite_score`` 补 ``综合得分``（唯一实现，
    本模块不复制公式）；显式传 ``False`` 用于验证“缺前置即禁止打分”的负例路径。
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {
            COL_NAMES["wf_run_id"]: record.run_id,
            COL_NAMES["batch_run_id"]: record.batch_run_id,
            COL_NAMES["batch_period_label"]: record.batch_period_label,
            COL_NAMES["split_count"]: record.split_count,
            COL_NAMES["final_date"]: record.final_date,
            COL_NAMES["chain_cagr"]: record.metrics.get("chain_cagr"),
            COL_NAMES["chain_max_drawdown"]: record.metrics.get("chain_max_drawdown"),
            COL_NAMES["chain_sharpe"]: record.metrics.get("chain_sharpe"),
            COL_NAMES["oos_cross_split_ir"]: record.metrics.get("oos_cross_split_ir"),
            COL_NAMES["bt_win_rate"]: record.metrics.get("bt_win_rate"),
            COL_NAMES["start_model_version"]: record.model_version,
        }
        row.update({str(key): value for key, value in record.extra.items()})
        for key, value in record.config.items():
            column = COL_NAMES.get(str(key), str(key))
            if column in row:
                continue
            row[column] = value
        rows.append(row)
    frame = pd.DataFrame(rows)
    if with_composite and not frame.empty:
        frame["综合得分"] = compute_composite_score(frame)
    return frame


def _status_row(
    word: str,
    reason: str,
    *,
    detail: str,
    group_key: str,
    config_json: str,
    record: StabilityRecord | None = None,
    period_count: int = 0,
    run_ids: Sequence[str] = (),
    partition_arm_count: int = 0,
) -> dict[str, Any]:
    """显式状态行（字段与合法行同构，值一律 ``None``，不写 0 冒充）。"""
    row: dict[str, Any] = {
        "group_key": group_key,
        "config_json": config_json,
        "period_count": period_count,
        "period_labels": None,
        "run_id_list": " | ".join(str(item) for item in run_ids) if run_ids else None,
        "stability_score": None,
        "score_mean": None,
        "score_std": None,
        "score_min": None,
        "score_max": None,
        "chain_cagr_mean": None,
        "chain_cagr_std": None,
        "chain_cagr_min": None,
        "chain_max_drawdown_mean": None,
        "chain_max_drawdown_worst": None,
        "oos_cross_split_ir_mean": None,
        "oos_cross_split_ir_std": None,
        "bt_win_rate_mean": None,
        "bt_win_rate_min": None,
        "chain_sharpe_mean": None,
        "chain_sharpe_std": None,
        "latest_run_time": None,
        "group_columns": None,
        "group_values": None,
        "partition_arm_count": int(partition_arm_count),
        "report_status": word,
        "degradation_reason_code": reason,
    }
    if record is not None:
        row["run_id_list"] = record.run_id
    return row


def _precondition_problems(frame: pd.DataFrame) -> list[str]:
    """打分前置硬断言：缺列即 ``missing_precondition``（禁止调用旧函数）。"""
    problems: list[str] = [
        f"缺列 {column}" for column in REQUIRED_METRIC_COLUMNS if column not in frame.columns
    ]
    if not stability_group_columns(frame):
        problems.append("无可用的 PARAM_COLS 分组列（旧函数会返回空表）")
    return problems


def _project_dynamic(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """对旧函数实际分组列中的动态身份做统一占位（只改分组键，不改指标）。"""
    group_cols = stability_group_columns(frame)
    projected = frame.copy()
    dynamic: list[str] = []
    for column in group_cols:
        if classify_field(column) == "dynamic":
            projected[column] = DYNAMIC_PLACEHOLDER
            dynamic.append(column)
    return projected, dynamic


def _group_tuples(frame: pd.DataFrame, columns: Sequence[str]) -> list[tuple[Any, ...]]:
    """有效分组键集合（保持首次出现顺序；NaN 参与比较）。"""
    seen: list[tuple[Any, ...]] = []
    for _, item in frame.iterrows():
        key = tuple("" if pd.isna(item[column]) else item[column] for column in columns)
        if key not in seen:
            seen.append(key)
    return seen


def _row_key(row: Mapping[str, Any], columns: Sequence[str]) -> tuple[Any, ...]:
    """从函数输出行取同一分组键（缺失列按空串参与）。"""
    values = []
    for column in columns:
        value = row.get(column)
        values.append("" if value is None or pd.isna(value) else value)
    return tuple(values)


def _group_mask(projected: pd.DataFrame, group_cols: Sequence[str], values: Sequence[Any]) -> Any:
    """单个有效分组在投影帧上的布尔掩码（空值按空串比较）。"""
    mask = pd.Series(True, index=projected.index)
    for column, value in zip(group_cols, values):
        series = projected[column]
        if value == "":
            mask &= series.isna() | (series.astype(str) == "")
        else:
            mask &= series == value
    return mask


def _partition_rows(
    partition: pd.DataFrame,
    identity: Mapping[str, Any],
    *,
    group_key: str,
    run_ids: Sequence[str],
    compute_composite: bool,
    partition_arm_count: int,
) -> tuple[list[dict[str, Any]], set[str]]:
    """单个稳定等价类的行集（合法行 + 显式状态行）与动态分组列。

    ``compute_composite=True`` 时**在本分区内**调用 ``compute_composite_score``：等价类不因
    报告里出现无关臂而改变得分（A-06）；显式传入 frame 时保持原样（缺列即前置失败）。
    """
    if compute_composite and "综合得分" not in partition.columns:
        partition = partition.copy()
        partition["综合得分"] = compute_composite_score(partition)
    # 状态行与合法行使用同一份 canonical 身份载荷（显式缺列 frame 的 API 分支亦然，R3-12-R2）
    partition_payload = {"stable_identity": dict(identity), "group": {}}
    partition_key = canonical_json_sha256(partition_payload)
    config_json = _canonical(partition_payload)
    problems = _precondition_problems(partition)
    if problems:
        return [
            _status_row(
                "not_available",
                "missing_precondition",
                detail="；".join(problems),
                group_key=partition_key,
                config_json=config_json,
                period_count=int(len(partition)),
                run_ids=run_ids,
                partition_arm_count=partition_arm_count,
            )
        ], set()
    projected, dynamic = _project_dynamic(partition)
    group_cols = stability_group_columns(projected)
    result = build_period_stability_table(projected)
    produced = (
        [_row_key(item, group_cols) for _, item in result.iterrows()] if not result.empty else []
    )
    rows: list[dict[str, Any]] = []
    for group_values in _group_tuples(projected, group_cols):
        rows.extend(
            _group_rows(
                projected,
                result,
                group_cols,
                group_values,
                identity,
                produced=produced,
                partition_arm_count=len(partition),
            )
        )
    return rows, set(dynamic)


def _group_rows(
    projected: pd.DataFrame,
    result: pd.DataFrame,
    group_cols: Sequence[str],
    group_values: Sequence[Any],
    identity: Mapping[str, Any],
    *,
    produced: Sequence[tuple[Any, ...]],
    partition_arm_count: int = 0,
) -> list[dict[str, Any]]:
    """单个有效分组的行集：合法行或一条显式状态行（不足两期不在函数输出中）。"""
    members = projected[_group_mask(projected, group_cols, group_values)]
    group_key_text = canonical_json_sha256(
        {"stable_identity": identity, "group": dict(zip(group_cols, group_values))}
    )
    # 与 group_key 同一份完整 canonical 身份（含 host/实现/数据态/代码态/水位），
    # 使 ``hash(json.loads(config_json)) == group_key`` 可独立复算（R3-12）
    config_text = _canonical(
        {"stable_identity": dict(identity), "group": dict(zip(group_cols, group_values))}
    )
    scores = pd.to_numeric(members.get("综合得分"), errors="coerce").dropna()
    # 期间数按**唯一实现的期间标签**去重计数：同窗口重跑不得冒充多期（R3-04）
    period_labels = [_build_period_label(row) for _, row in members.iterrows()]
    unique_periods = list(dict.fromkeys(period_labels))
    if group_values not in set(produced):
        return [
            _status_row(
                "not_available",
                "insufficient_periods",
                detail=f"组内仅 {len(unique_periods)} 期（旧函数跳过 ≤1 期分组）",
                group_key=group_key_text,
                config_json=config_text,
                period_count=int(len(unique_periods)),
                run_ids=members[COL_NAMES["wf_run_id"]].astype(str).tolist(),
                partition_arm_count=partition_arm_count,
            )
        ]
    if len(scores) < 2:
        return [
            _status_row(
                "not_available",
                "insufficient_periods",
                detail=f"组内综合得分仅 {len(scores)} 条有限值（不产生 stability_score）",
                group_key=group_key_text,
                config_json=config_text,
                period_count=int(len(unique_periods)),
                run_ids=members[COL_NAMES["wf_run_id"]].astype(str).tolist(),
                partition_arm_count=partition_arm_count,
            )
        ]
    matched = result[[_row_key(item, group_cols) == group_values for _, item in result.iterrows()]]
    return _map_output(
        matched, group_key_text, config_text, group_cols, partition_arm_count
    )


def build_stability_rows(
    records: Sequence[StabilityRecord], *, frame: pd.DataFrame | None = None
) -> dict[str, Any]:
    """按完整等价类分区生成稳定性行 + 显式状态行（返回行集与登记信息）。"""
    compute_composite = frame is None
    material = (
        build_stability_input(records, with_composite=False) if compute_composite else frame.copy()
    )
    if material.empty:
        # 空输入状态行也使用与合法行同一份 canonical 身份载荷：hash(config_json)==group_key（R3-12-R1）
        empty_identity = {"stable_identity": {"reason": "no_input_rows"}, "group": {}}
        row = _status_row(
            "not_available",
            "missing_precondition",
            detail="无稳定性输入行",
            group_key=canonical_json_sha256(empty_identity),
            config_json=_canonical(empty_identity),
        )
        return {
            "status": row,
            "rows": [row],
            "partitions": 0,
            "expected_row_count": 1,
            "dynamic_columns": [],
        }
    if len(records) != len(material):
        raise ValueError(
            f"稳定性输入行数 {len(material)} 与记录数 {len(records)} 不一致（禁止错位分组）"
        )
    by_identity: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        by_identity.setdefault(_group_key(stable_identity_of(record)), []).append(index)

    rows: list[dict[str, Any]] = []
    dynamic_all: set[str] = set()
    expected_total = 0
    for group_key, indices in by_identity.items():
        # **分区局部**重建输入：等价类的列集只由本分区记录的 config 决定，不因报告里加入
        # 无关臂而多出 NaN 列（否则 group_key 会随无关臂漂移，A-06 不变性）
        partition = (
            build_stability_input([records[index] for index in indices], with_composite=False)
            if compute_composite
            else material.iloc[indices].reset_index(drop=True)
        )
        identity = stable_identity_of(records[indices[0]])
        expected_total += _expected_partition_rows(partition, compute_composite)
        partition_rows, dynamic = _partition_rows(
            partition,
            identity,
            group_key=group_key,
            run_ids=[records[index].run_id for index in indices],
            compute_composite=compute_composite,
            partition_arm_count=len(indices),
        )
        rows.extend(partition_rows)
        dynamic_all.update(dynamic)
    generated = status.make_status(
        "available",
        detail=f"{len(by_identity)} 个稳定等价类、{len(rows)} 行（合法行与旧函数逐项对账）",
    )
    return {
        "status": generated,
        "rows": rows,
        "partitions": len(by_identity),
        # 源侧独立期望行数（不取自 rows 列表长度；多一条键唯一行也会被行数门拦住）
        "expected_row_count": expected_total,
        "dynamic_columns": sorted(dynamic_all),
    }


def _expected_partition_rows(partition: pd.DataFrame, compute_composite: bool) -> int:
    """单个分区的**源侧**期望行数：前置缺失 1 行；否则 = 有效分组数。"""
    material = partition
    if compute_composite and "综合得分" not in material.columns:
        material = material.copy()
        material["综合得分"] = compute_composite_score(material)
    if _precondition_problems(material):
        return 1
    projected, _dynamic = _project_dynamic(material)
    return len(_group_tuples(projected, stability_group_columns(projected)))


def _map_output(
    frame: pd.DataFrame,
    group_key: str,
    config_json: str,
    group_cols: Sequence[str],
    partition_arm_count: int = 0,
) -> list[dict[str, Any]]:
    """函数输出行 → P6 稳定性行（列名映射自 ``COL_NAMES``）。"""
    rows: list[dict[str, Any]] = []
    allowed = set(experiment_csv.STABILITY_COLUMNS)
    for _, item in frame.iterrows():
        row: dict[str, Any] = {"group_key": group_key, "config_json": config_json}
        for column, value in item.items():
            target = _OUTPUT_MAP.get(str(column), str(column))
            if target not in allowed:
                # 旧函数的分组列不进入固定列序；取值在同行的 group_values 中完整陈列
                continue
            row[target] = None if value is None or (isinstance(value, float) and np.isnan(value)) else value
        row["group_columns"] = " | ".join(group_cols)
        row["partition_arm_count"] = int(partition_arm_count)
        row["group_values"] = _canonical(
            {column: (None if pd.isna(item[column]) else item[column]) for column in group_cols}
        )
        row["report_status"] = "available"
        row["degradation_reason_code"] = None
        rows.append(row)
    return rows
