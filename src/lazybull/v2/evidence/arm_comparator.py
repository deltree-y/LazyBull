"""P6.5 T4 经济工件逐位比较器（两臂 runs + 物理/合并虚拟视图；**禁止容差**）。

契约依据：
- 设计 ``docs/plans/v2/p65_merger_reconciliation_design_20261108.md`` §8.4 第 5~9 条
  （唯一生产者、隔离前置、生成身份绑定、固定输入与真正证据分清单、比较对象、
  实际文件集与缺失语义、Parquet/JSON 规范化字节、float64 位模式与 dtype/键序/NaN）；
- ``docs/contracts/runs_artifact_contract.md`` §1/§2/§7/§9（列集与读入不变量）；
- T3 交付的 ``hosts/virtual_arm_runs.py``（隔离根 + generation 身份 + 缺失语义分类）。

判定口径（冻结，不得放宽）：
- **内存浮点按 float64 位模式**比较（``struct.pack(">d")``），同时比较 dtype、键序与
  NaN 位置；不使用 ``pytest.approx``、不使用 I1/I3 恒等式容差、不做"看起来相近"兜底；
- **runs 以规范化字节**比较：Parquet 解码后按**契约列序锚点**（契约列在前且保持契约
  顺序，``summary`` 的非契约列按左臂列序追加）重编码（无压缩、无索引）取 SHA-256；
  JSON 去掉排除字段后按 ``sort_keys`` 规范序列化取 SHA-256；字节不等即差异；
- 排除字段**仅** ``created_at`` / ``arms``，运行前冻结（``EXCLUDED_META_FIELDS``）；
  任何事后增删都抛 ``ComparisonInputError``（"排除表越界"拒绝面）；
- 缺失语义：必需文件/折级 ``_meta`` 必须两侧存在；``attribution`` /
  ``holdings_snapshot`` 单侧缺失即失败、两侧同源缺失登记 ``not_compared/source_absent``；
  ``topk_detail`` 登记 ``not_compared/contract_optional_missing``，**不得**写成逐位一致；
- **折集合闭合**：折身份取自 ``summary``/``chain_nav`` 的 ``split_index``（不取自目录
  枚举），实际折目录必须与契约折集合完全一致，且每折 ``trades``/``daily``/``_meta``
  两侧存在——"两臂同时缺整折"不得空遍历通过（外部评审 T4-Codex-01）；
- **配置覆盖独立核对**：``batch_meta.config`` 必须与调用方冻结的键值清单逐键相等、
  ``config_fingerprint`` 必须等于由 config 重算的指纹（``fingerprint_keys`` 单一来源），
  不得用"config 键 ⊆ summary 列"自证覆盖（外部评审 T4-Codex-02）；
- **规范化字节可落盘**：传入 ``capture_dir`` 时把两侧实际参与比较的字节写成文件，
  证据包据此自含"SHA-256 + 本体"，不在落盘时重算第二套编码（外部评审 T4-Codex-04）；
- 两臂身份（``batch_meta.arms`` 与 generation 臂标签）在比较前先校验：A 产物复制给 B、
  篡改根映射、复用 generation 全部在写入层与比较器层同时失败。

边界：本模块只读（不写 writer、不改产物），差异不修不掩盖；恒等式容差门（I1/I3）与
runs 批次内部自校验（``1e-3``/``1e-6``）**不属于**本比较器口径。
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from dataclasses import field as _dc_field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from src.lazybull.v2.core.accounting.g4_nav import float64_bits
from src.lazybull.v2.evidence.fingerprint_keys import config_fingerprint_keys, fingerprint
from src.lazybull.v2.evidence.runs_schema import (
    ATTR_CONTRACT_COLS,
    CHAIN_NAV_CONTRACT_COLS,
    DAILY_CONTRACT_COLS,
    SNAP_CONTRACT_COLS,
    SUMMARY_REQUIRED_COLS,
    TRADES_CONTRACT_COLS,
)
from src.lazybull.v2.hosts.self_split_arms import (
    ArmView,
    assert_arms_are_distinct,
    assert_view_constructions_distinct,
)
from src.lazybull.v2.hosts.virtual_arm_runs import (
    ARM_SINGLE,
    ARM_SPLIT,
    REQUIRED_FOLD_FILES,
    DualArmRunsWriter,
)

#: 本模块唯一阶段标识（进入全部错误上下文）
STAGE = "P6.5"
#: 运行前冻结的 ``batch_meta`` 排除字段（至多两项；事后增删即停止）
EXCLUDED_META_FIELDS: Tuple[str, ...] = ("created_at", "arms")
#: 排除理由（冻结，供证据登记）
EXCLUSION_REASONS: Mapping[str, str] = {
    "created_at": "写墙钟时间非经济事实；两臂必然不同，且不参与任何经济指标派生",
    "arms": "臂标签是不同构造标签（single_reference/virtual_split），按契约不作逐位证据",
}
#: 规范化 Parquet 编码参数（无压缩、无索引；同一实现两侧共用）
_PARQUET_KWARGS: Mapping[str, Any] = {"index": False, "compression": "none"}
#: 逐文件行身份列（差异定位与行序校验；缺失则退化为纯行号比较）
_KEY_COLUMNS: Mapping[str, Tuple[str, ...]] = {
    "summary.parquet": ("split_index",),
    "chain_nav.parquet": ("split_index", "date"),
    "trades.parquet": ("trade_date", "action", "ts_code"),
    "daily.parquet": ("trade_date",),
    "holdings_snapshot.parquet": ("trade_date", "ts_code"),
    "attribution.parquet": ("signal_date", "planned_ts_code", "actual_ts_code"),
}
#: ``batch_meta.json`` 契约 §2 必需键（缺失即契约违反；两臂同步缺键不得读成一致）
_BATCH_META_REQUIRED_KEYS: Tuple[str, ...] = (
    "schema_version",
    "batch_id",
    "created_at",
    "host_mode",
    "source",
    "code_state",
    "data_state",
    "config",
    "config_fingerprint",
    "baseline_ref",
    "arms",
    "meta_notes",
)
#: 逐文件**契约必需列**（缺失即契约违反；比较器不做"左右一致即通过"）
_REQUIRED_COLUMNS: Mapping[str, Tuple[str, ...]] = {
    "summary.parquet": tuple(SUMMARY_REQUIRED_COLS),
    "chain_nav.parquet": tuple(CHAIN_NAV_CONTRACT_COLS),
    "trades.parquet": tuple(TRADES_CONTRACT_COLS),
    "daily.parquet": tuple(DAILY_CONTRACT_COLS),
    "holdings_snapshot.parquet": tuple(SNAP_CONTRACT_COLS),
    "attribution.parquet": tuple(ATTR_CONTRACT_COLS),
}
#: 契约列集=全列集的文件（列**顺序**也属契约；summary 不在此列，其列序由配置派生）
_FULL_COLUMN_FILES = frozenset(
    {
        "chain_nav.parquet",
        "trades.parquet",
        "daily.parquet",
        "holdings_snapshot.parquet",
        "attribution.parquet",
    }
)
#: 逐文件比较顺序（批次级 + 折级；缺失项按状态跳过）
_FOLD_FILE_NAMES: Tuple[str, ...] = (
    "trades.parquet",
    "daily.parquet",
    "_meta.json",
    "attribution.parquet",
    "holdings_snapshot.parquet",
)
#: 折目录命名（writer 契约：``folds/split{index:02d}``；与 ``summary.split_index`` 对应）
_FOLD_DIR_TEMPLATE = "split{index:02d}"

_STATUS_PRESENT = "present"


class ComparisonInputError(ValueError):
    """比较输入/身份/排除表非法（**零副作用**拒绝）。"""


class ArmComparisonError(RuntimeError):
    """两臂经济工件存在差异（终态语义；禁止降级为 warning 或近似比较）。"""


def _error(message: str) -> Exception:
    return ComparisonInputError(f"阶段={STAGE} 两臂比较 {message}")


@dataclass(frozen=True, slots=True)
class FieldDifference:
    """单条差异（字段路径 + 左右值 + 两侧 float64 位模式 + 来源文件/行）。"""

    field_path: str
    left: Any
    right: Any
    left_bits: str
    right_bits: str
    source: str
    row: Optional[int] = None

    def describe(self) -> str:
        """单行差异描述（含位模式与来源行）。"""
        row_tag = "" if self.row is None else f" 行={self.row}"
        return (
            f"{self.source}{row_tag} {self.field_path}：左={self.left!r}"
            f"（位={self.left_bits or '<非浮点>'}） 右={self.right!r}"
            f"（位={self.right_bits or '<非浮点>'}）"
        )


@dataclass(frozen=True, slots=True)
class ArmDiffReport:
    """两臂比较结果（差异清单 + 文件状态 + 规范化 SHA-256 + 排除表 + 规范化字节落盘）。"""

    differences: Tuple[FieldDifference, ...]
    file_status: Tuple[Tuple[str, str], ...]
    file_sha256: Tuple[Tuple[str, str, str], ...]
    excluded_fields: Tuple[str, ...]
    compared_summary_columns: Tuple[str, ...]
    fingerprint_key_count: int
    #: 落盘的规范化字节（``(runs 相对路径, 左臂文件, 右臂文件)``；未开启落盘时为空）
    normalized_files: Tuple[Tuple[str, str, str], ...] = ()
    #: 折集合闭合登记（``expected``/``actual``/``missing_files``）
    fold_closure: Mapping[str, Any] = _dc_field(default_factory=dict)
    #: 单臂 config 由 ``fingerprint_keys.fingerprint`` 重算的指纹（应与记录值相同）
    config_fingerprint_recomputed: str = ""

    @property
    def ok(self) -> bool:
        """是否零差异（``not_compared`` 项不计为差异，但也不得表述为逐位一致）。"""
        return not self.differences

    def not_compared(self) -> Tuple[str, ...]:
        """登记为 ``not_compared`` 的文件（证据措辞用；不得写成"逐位一致"）。"""
        return tuple(path for path, status in self.file_status if status.startswith("not_compared"))

    def describe_differences(self, limit: int = 10) -> str:
        """差异摘要（前 ``limit`` 条；用于异常消息与证据）。"""
        lines = [item.describe() for item in self.differences[:limit]]
        tail = ""
        if len(self.differences) > limit:
            tail = f"；另有 {len(self.differences) - limit} 条差异未列出"
        return "\n".join(lines) + tail

    def assert_clean(self, *, context: str = "") -> None:
        """零差异断言（失败即抛 ``ArmComparisonError``，列出全部差异）。"""
        if self.differences:
            raise ArmComparisonError(
                f"[{STAGE}][G4] 两臂经济工件存在 {len(self.differences)} 处差异"
                f"（容差=0，禁止近似比较）{context}：\n{self.describe_differences()}"
            )


# ────────────────────────── 排除表 ──────────────────────────


def assert_exclusion_table(fields: Sequence[str]) -> Tuple[str, ...]:
    """排除表冻结校验：只允许 ``created_at``/``arms``，顺序固定、不得增删。"""
    actual = tuple(str(item) for item in fields)
    if actual != EXCLUDED_META_FIELDS:
        extra = sorted(set(actual) - set(EXCLUDED_META_FIELDS))
        missing = sorted(set(EXCLUDED_META_FIELDS) - set(actual))
        raise _error(
            f"排除字段表必须为 {list(EXCLUDED_META_FIELDS)}，实际 {list(actual)}"
            f"（多出={extra} 缺少={missing}）；排除表运行前冻结，事后增删即停止"
        )
    return actual


# ────────────────────────── 标量与容器 ──────────────────────────


def _is_float(value: Any) -> bool:
    """是否按 float64 位模式比较（int/bool/str/None 不算浮点）。"""
    if isinstance(value, float):
        return True
    if isinstance(value, (int, bool, str, bytes)) or value is None:
        return False
    return hasattr(value, "__float__")


def _compare_scalar(path: str, left: Any, right: Any, source: str) -> List[FieldDifference]:
    """标量比较：浮点按位模式（NaN 位模式也必须一致），其余按 ``!=``。"""
    if _is_float(left) or _is_float(right):
        left_value = float(left)
        right_value = float(right)
        left_bits = float64_bits(left_value)
        right_bits = float64_bits(right_value)
        if left_bits != right_bits:
            return [FieldDifference(path, left_value, right_value, left_bits, right_bits, source)]
        return []
    if left != right:
        return [FieldDifference(path, left, right, "", "", source)]
    return []


def compare_scalars(
    path: str,
    left: Any,
    right: Any,
    *,
    source: str = "<内存>",
    row: Optional[int] = None,
) -> Tuple[FieldDifference, ...]:
    """导出：两个标量逐位比较（浮点按 float64 位模式；NaN 位模式也必须一致）。"""
    return tuple(
        FieldDifference(
            item.field_path,
            item.left,
            item.right,
            item.left_bits,
            item.right_bits,
            item.source,
            row,
        )
        for item in _compare_scalar(path, left, right, source)
    )


def compare_value_trees(
    left: Any, right: Any, *, source: str = "<内存>", path: str = ""
) -> Tuple[FieldDifference, ...]:
    """递归比较嵌套 Mapping/序列（键序、长度、逐元素位模式）。"""
    diffs: List[FieldDifference] = []
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        left_keys = list(left)
        right_keys = list(right)
        if left_keys != right_keys:
            diffs.append(
                FieldDifference(
                    f"{path}.<keys>" if path else "<keys>",
                    left_keys,
                    right_keys,
                    "",
                    "",
                    source,
                )
            )
        for key in left_keys:
            if key in right:
                child = f"{path}.{key}" if path else str(key)
                diffs.extend(compare_value_trees(left[key], right[key], source=source, path=child))
        return tuple(diffs)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            diffs.append(
                FieldDifference(
                    f"{path}.<len>" if path else "<len>", len(left), len(right), "", "", source
                )
            )
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            diffs.extend(
                compare_value_trees(left_item, right_item, source=source, path=f"{path}[{index}]")
            )
        return tuple(diffs)
    return tuple(_compare_scalar(path or "<value>", left, right, source))


# ────────────────────────── 帧比较 ──────────────────────────


def normalized_frame_bytes(frame: pd.DataFrame, columns: Sequence[str]) -> bytes:
    """规范化 Parquet 字节（契约列序 + 无压缩 + 无索引；证据 SHA-256 的输入）。"""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise _error(f"规范化失败：帧缺少契约列 {missing}")
    buffer = io.BytesIO()
    frame.loc[:, list(columns)].to_parquet(buffer, **_PARQUET_KWARGS)
    return buffer.getvalue()


def normalized_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """规范化 JSON 字节（``sort_keys`` + 紧凑分隔符；证据 SHA-256 的输入）。"""
    return json.dumps(
        dict(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    ).encode("utf-8")


def _is_missing(value: Any) -> bool:
    """缺失判定（None/NaN/NA；容器与字符串不参与缺失语义）。"""
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, str, bytes)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # pragma: no cover - 非常规标量
        return False


def _cell_difference(
    path: str, left: Any, right: Any, source: str, row: int
) -> List[FieldDifference]:
    """单元格比较（NaN 位置：缺失只与缺失相等；浮点按位模式）。"""
    left_missing = _is_missing(left)
    right_missing = _is_missing(right)
    if left_missing or right_missing:
        if left_missing and right_missing:
            return []
        return [
            FieldDifference(
                path,
                None if left_missing else left,
                None if right_missing else right,
                "" if left_missing or not _is_float(left) else float64_bits(float(left)),
                "" if right_missing or not _is_float(right) else float64_bits(float(right)),
                source,
                row,
            )
        ]
    return [
        FieldDifference(
            item.field_path,
            item.left,
            item.right,
            item.left_bits,
            item.right_bits,
            item.source,
            row,
        )
        for item in _compare_scalar(path, left, right, source)
    ]


def _key_token(values: Any) -> Tuple[Any, ...]:
    """行身份键的可比较 token（缺失统一为哨兵，避免 NaN/NA 造成伪差异或 ``bool`` 异常）。"""
    tokens: List[Any] = []
    for value in values:
        if _is_missing(value):
            tokens.append(("<NA>",))
            continue
        if _is_float(value):
            tokens.append(("f", float64_bits(float(value))))
            continue
        try:
            hash(value)
        except TypeError:  # pragma: no cover - 非常规不可哈希标量
            tokens.append(("s", str(value)))
            continue
        tokens.append((type(value).__name__, value))
    return tuple(tokens)


def compare_frames(
    name: str,
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    columns: Sequence[str],
    key_columns: Sequence[str],
    source_left: str,
    source_right: str,
) -> Tuple[FieldDifference, ...]:
    """逐列逐行比较两个契约帧（列序/dtype/行数/键序/NaN 位置/浮点位模式）。

    Args:
        name: 逻辑名（进入字段路径，如 ``summary``）
        columns: 契约列序（两侧都必须具备；列序不同即差异）
        key_columns: 行身份列（差异定位与行序校验；缺失则退化为纯行号比较）
    """
    diffs: List[FieldDifference] = []
    if list(left.columns) != list(columns):
        diffs.append(
            FieldDifference(
                f"{name}.<columns>", list(left.columns), list(columns), "", "", source_left
            )
        )
    if list(right.columns) != list(columns):
        diffs.append(
            FieldDifference(
                f"{name}.<columns>", list(columns), list(right.columns), "", "", source_right
            )
        )
    contract_cols = [
        column for column in columns if column in left.columns and column in right.columns
    ]
    for column in contract_cols:
        left_dtype = str(left[column].dtype)
        right_dtype = str(right[column].dtype)
        if left_dtype != right_dtype:
            diffs.append(
                FieldDifference(
                    f"{name}.{column}.<dtype>", left_dtype, right_dtype, "", "", source_left
                )
            )
    if len(left) != len(right):
        diffs.append(FieldDifference(f"{name}.<rows>", len(left), len(right), "", "", source_left))
    shared_rows = min(len(left), len(right))
    keys_present = [column for column in key_columns if column in contract_cols]
    if keys_present and len(left) and len(right):
        left_keys = [
            _key_token(left[column].iloc[row] for column in keys_present)
            for row in range(len(left))
        ]
        right_keys = [
            _key_token(right[column].iloc[row] for column in keys_present)
            for row in range(len(right))
        ]
        if left_keys != right_keys:
            diffs.append(
                FieldDifference(
                    f"{name}.<key_order>", left_keys[:5], right_keys[:5], "", "", source_left
                )
            )
    for row in range(shared_rows):
        for column in contract_cols:
            diffs.extend(
                _cell_difference(
                    f"{name}.{column}",
                    left[column].iloc[row],
                    right[column].iloc[row],
                    f"{source_left} vs {source_right}",
                    row,
                )
            )
    return tuple(diffs)


# ────────────────────────── 经济视图（物理 + 合并虚拟） ──────────────────────────


def assert_distinct_arm_views(left: ArmView, right: ArmView) -> None:
    """比较前置：两臂必须是不同构造（臂标签冻结 + 袖数 1 vs 2），否则是自我比较。

    ``compare_assemblies`` 与 ``compare_arm_views`` 都先过本门：即使调用方伪造
    ``ArmView``（把 single 直接当 split 传入、或把两臂袖集合改成一样），比较器也
    不会给出"零差异"的伪证据（T4 复审 B-R1-01）。判定逻辑委托装配层唯一实现
    ``self_split_arms.assert_view_constructions_distinct``，只把错误类型收敛为本模块
    的 ``ComparisonInputError``（比较器输入拒绝面）。
    """
    try:
        assert_view_constructions_distinct(left, right)
    except ValueError as error:
        raise ComparisonInputError(f"阶段={STAGE} 两臂比较 {error}") from error


def compare_arm_views(left: ArmView, right: ArmView) -> Tuple[FieldDifference, ...]:
    """两臂经济事实逐位比较（物理 + 合并虚拟视图；不比较袖级归属）。

    比较项（设计 §8.4 第 6 条）：物理成交（股数/价/费/来源 lot/事件键/目标）、物理现金、
    每来源 lot 股数、物理货币总值、合并虚拟视图的现金和/来源股数和/货币总值/
    **先合并总值再单次派生的 NAV**。袖级现金与袖级归属属**不同构造**，不作逐位比较
    （单袖 400 vs 双子袖 200+200 是合法差异，由 I2/I3 与 G4 分别校验）。

    Raises:
        ComparisonInputError: 两臂不是不同构造（同一臂/袖数相同/臂标签非冻结值）——
            这是**比较器层**的自我比较防护，与写入层 generation 身份互为双保险。
    """
    assert_distinct_arm_views(left, right)
    diffs: List[FieldDifference] = []
    source = f"{left.arm} vs {right.arm}"
    if left.settlement_dates != right.settlement_dates:
        diffs.append(
            FieldDifference(
                "physical.settlement_dates",
                left.settlement_dates,
                right.settlement_dates,
                "",
                "",
                source,
            )
        )
    diffs.extend(
        _compare_scalar(
            "physical.initial_capital",
            left.physical_initial_capital,
            right.physical_initial_capital,
            source,
        )
    )
    diffs.extend(_compare_scalar("physical.cash", left.physical_cash, right.physical_cash, source))
    diffs.extend(
        _compare_scalar(
            "physical.total_value", left.physical_total_value, right.physical_total_value, source
        )
    )
    if left.physical_positions != right.physical_positions:
        diffs.append(
            FieldDifference(
                "physical.positions",
                left.physical_positions,
                right.physical_positions,
                "",
                "",
                source,
            )
        )
    if left.physical_source_lots != right.physical_source_lots:
        diffs.append(
            FieldDifference(
                "physical.source_lots",
                left.physical_source_lots,
                right.physical_source_lots,
                "",
                "",
                source,
            )
        )
    if len(left.fills) != len(right.fills):
        diffs.append(
            FieldDifference(
                "physical.fills.<count>", len(left.fills), len(right.fills), "", "", source
            )
        )
    for index, (left_fill, right_fill) in enumerate(zip(left.fills, right.fills)):
        for field in (
            "ts_code",
            "order_target",
            "requested_shares",
            "shares",
            "price",
            "fee",
            "cash_before",
            "cash_after",
            "fill_key",
            "source_lot_id",
        ):
            diffs.extend(
                _compare_scalar(
                    f"physical.fills[{index}].{field}",
                    getattr(left_fill, field),
                    getattr(right_fill, field),
                    source,
                )
            )
    diffs.extend(
        _compare_scalar(
            "virtual.cash_sum", left.merged_virtual_cash, right.merged_virtual_cash, source
        )
    )
    diffs.extend(
        _compare_scalar(
            "virtual.merged_total_value", left.merged_total_value, right.merged_total_value, source
        )
    )
    diffs.extend(_compare_scalar("virtual.merged_nav", left.merged_nav, right.merged_nav, source))
    if left.virtual_source_lots != right.virtual_source_lots:
        diffs.append(
            FieldDifference(
                "virtual.source_lots",
                left.virtual_source_lots,
                right.virtual_source_lots,
                "",
                "",
                source,
            )
        )
    return tuple(diffs)


# ────────────────────────── runs 两臂树 ──────────────────────────


def _fold_dirs(writer: DualArmRunsWriter, batch_id: str) -> Tuple[str, ...]:
    """折目录名（两侧文件集已由 ``artifact_status`` 校验一致）。"""
    root = writer.out_dir(ARM_SINGLE, batch_id) / "folds"
    if not root.is_dir():  # pragma: no cover - writer 必写至少一折
        return ()
    return tuple(sorted(item.name for item in root.iterdir() if item.is_dir()))


def _assert_batch_meta_keys(payload: Mapping[str, Any], side: str) -> None:
    """``batch_meta`` 契约 §2 必需键门（两臂同步缺键不得读成"一致"）。"""
    missing = [key for key in _BATCH_META_REQUIRED_KEYS if key not in payload]
    if missing:
        raise _error(
            f"{side}/batch_meta.json 缺少 runs 契约 §2 必需键 {missing}；"
            "比较器不得在两臂同步缺键时给出零差异"
        )


class _NormalizedCapture:
    """把两侧**实际参与比较**的规范化字节落盘（证据自含：SHA 与本体同源）。

    名称规则：``<runs 相对路径>`` 中的 ``/`` → ``__``，两侧后缀 ``.single.bin`` /
    ``.split.bin``。落盘的字节就是本次比较用的字节，不在落盘时重算第二套编码。
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.files: List[Tuple[str, str, str]] = []

    def record(self, relative: str, left: bytes, right: bytes) -> None:
        """写入一对规范化字节并登记相对路径（相对捕获目录的父目录）。"""
        stem = relative.replace("/", "__")
        left_name = f"{stem}.single.bin"
        right_name = f"{stem}.split.bin"
        (self.directory / left_name).write_bytes(left)
        (self.directory / right_name).write_bytes(right)
        prefix = self.directory.name
        self.files.append((relative, f"{prefix}/{left_name}", f"{prefix}/{right_name}"))


def _compare_json_file(
    left_path: Any,
    right_path: Any,
    relative: str,
    *,
    excluded_fields: Sequence[str],
    side_labels: Tuple[str, str] = ("single", "split"),
    capture: Optional[_NormalizedCapture] = None,
) -> Tuple[List[FieldDifference], Tuple[str, str]]:
    """JSON 文件比较：去排除字段后规范字节（不等则逐键定位差异）。"""
    left_payload = json.loads(left_path.read_text(encoding="utf-8"))
    right_payload = json.loads(right_path.read_text(encoding="utf-8"))
    if relative == "batch_meta.json":
        _assert_batch_meta_keys(left_payload, side_labels[0])
        _assert_batch_meta_keys(right_payload, side_labels[1])
        for field in excluded_fields:
            left_payload.pop(field, None)
            right_payload.pop(field, None)
    left_bytes = normalized_json_bytes(left_payload)
    right_bytes = normalized_json_bytes(right_payload)
    if capture is not None:
        capture.record(relative, left_bytes, right_bytes)
    source = f"single/{relative} vs split/{relative}"
    diffs: List[FieldDifference] = []
    if left_bytes != right_bytes:
        diffs.extend(compare_value_trees(left_payload, right_payload, source=source))
        if not diffs:  # pragma: no cover - 规范字节不等必有结构差异
            diffs.append(
                FieldDifference(
                    f"{relative}.<bytes>",
                    left_bytes[:24].hex(),
                    right_bytes[:24].hex(),
                    "",
                    "",
                    source,
                )
            )
    return diffs, (
        hashlib.sha256(left_bytes).hexdigest(),
        hashlib.sha256(right_bytes).hexdigest(),
    )


def _assert_required_columns(
    name: str, left_columns: Sequence[str], right_columns: Sequence[str]
) -> None:
    """契约必需列门：两臂都必须具备该文件的契约必需列（缺列即契约违反）。

    比较器的列集取自产物自身，若不额外校验必需列，"两臂同步缺同一列"会被读成
    一致；本门把列集锚定到 ``evidence/runs_schema.py`` 的契约常量（单一来源）。
    """
    required = _REQUIRED_COLUMNS.get(f"{name}.parquet")
    if not required:
        return
    for side, columns in (("single", left_columns), ("split", right_columns)):
        missing = [column for column in required if column not in columns]
        if missing:
            raise _error(
                f"{side}/{name}.parquet 缺少 runs 契约必需列 {missing}；"
                "比较器不得在两臂同步缺列时给出零差异"
            )


def _assert_contract_order(
    name: str, left_columns: Sequence[str], right_columns: Sequence[str]
) -> None:
    """契约列序门：**契约列集就是全列集**的文件，列序必须等于契约顺序。

    ``summary`` 不在本门范围：它的契约只固定必需列**集合**（列序由配置派生，契约未
    规定顺序），故只由 ``compare_frames`` 比较两臂列序是否一致。trades/daily/
    chain_nav/holdings_snapshot/attribution 的列集与顺序均由 ``evidence/runs_schema.py``
    冻结，两臂**同步**重排列序在此被打红（复审判定 A-R3-03）。
    """
    file_name = f"{name}.parquet"
    if file_name not in _FULL_COLUMN_FILES:
        return
    contract = list(_REQUIRED_COLUMNS[file_name])
    for side, columns in (("single", left_columns), ("split", right_columns)):
        if list(columns) != contract:
            raise _error(
                f"{side}/{file_name} 契约列序被改写：实际={list(columns)} 期望={contract}；"
                '列序属 runs 契约，比较器不得以"两臂一致"放行'
            )


def _column_anchor(file_name: str, columns: Sequence[str]) -> List[str]:
    """规范字节重编码列序：契约列在前（**契约顺序**），其余列按左臂列序追加。

    只用于让两侧的字节编码确定性可比；列序本身的判定由 ``_assert_contract_order``
    与 ``compare_frames`` 的列序比较负责，不靠锚点吞掉顺序差异。
    """
    contract = _REQUIRED_COLUMNS.get(file_name)
    if not contract:
        return list(columns)
    front = [column for column in contract if column in columns]
    return front + [column for column in columns if column not in front]


def _compare_parquet_file(
    left_path: Any,
    right_path: Any,
    relative: str,
    *,
    capture: Optional[_NormalizedCapture] = None,
) -> Tuple[List[FieldDifference], Tuple[str, str], List[str], int]:
    """Parquet 文件比较：逐列位模式 + 契约列序门 + 规范化字节（可落盘）。

    Returns:
        ``(差异, (左SHA, 右SHA), 列序, 指纹键数)``；指纹键数仅 summary 有意义（其余 0）。
    """
    left_frame = pd.read_parquet(left_path)
    right_frame = pd.read_parquet(right_path)
    file_name = relative.rsplit("/", 1)[-1]
    name = file_name.replace(".parquet", "")
    _assert_required_columns(name, left_frame.columns, right_frame.columns)
    _assert_contract_order(name, left_frame.columns, right_frame.columns)
    columns = list(left_frame.columns)
    diffs = list(
        compare_frames(
            name,
            left_frame,
            right_frame,
            columns=columns,
            key_columns=_KEY_COLUMNS.get(relative.rsplit("/", 1)[-1], ()),
            source_left=f"single/{relative}",
            source_right=f"split/{relative}",
        )
    )
    anchor = _column_anchor(file_name, columns)
    left_bytes = normalized_frame_bytes(left_frame, anchor)
    right_bytes = normalized_frame_bytes(right_frame, anchor)
    if capture is not None:
        capture.record(relative, left_bytes, right_bytes)
    if left_bytes != right_bytes and not diffs:  # pragma: no cover - 字节不等必有单元格差异
        diffs.append(
            FieldDifference(
                f"{relative}.<bytes>",
                left_bytes[:24].hex(),
                right_bytes[:24].hex(),
                "",
                "",
                f"single/{relative} vs split/{relative}",
            )
        )
    fingerprint_count = (
        len(config_fingerprint_keys(columns)) if relative == "summary.parquet" else 0
    )
    return (
        diffs,
        (hashlib.sha256(left_bytes).hexdigest(), hashlib.sha256(right_bytes).hexdigest()),
        columns,
        fingerprint_count,
    )


def _assert_config_keys_compared(
    writer: DualArmRunsWriter,
    batch_id: str,
    summary_columns: Sequence[str],
    *,
    required_config_keys: Sequence[str] = (),
    expected_config: Optional[Mapping[str, Any]] = None,
    dynamic_config_keys: Sequence[str] = (),
) -> Tuple[Mapping[str, Any], str]:
    """指纹配置键独立核对门（外部评审 T4-Codex-02）。

    与"键 ⊆ summary 列"的自证式覆盖不同，本门以**调用方冻结的独立清单**为准：

    1. ``required_config_keys`` 必须全部出现在两臂 ``batch_meta.config`` 中，且都进 summary
       列（否则等于没比）；
    2. 给了 ``expected_config`` 时，config 必须与之逐键**值相等**，且除
       ``dynamic_config_keys`` 外不得有其他键（"记录值 = 预期清单"，不是"记录值 ⊆ 列"）；
    3. ``config_fingerprint`` 必须等于由 config **重算**的指纹（单一来源
       ``fingerprint_keys.fingerprint``），记录值与实际参数不可分裂。

    Returns:
        ``(单臂 config, 重算指纹)``。

    Raises:
        ComparisonInputError: 缺键/多键/值不符/指纹不自洽。
    """
    if not summary_columns:  # pragma: no cover - summary 为必需文件
        raise _error("summary 列集为空，无法证明配置键逐项比较")
    frozen = tuple(str(key) for key in required_config_keys)
    dynamic = set(str(key) for key in dynamic_config_keys)
    observed: Dict[str, Dict[str, Any]] = {}
    fingerprints: Dict[str, str] = {}
    for arm in (ARM_SINGLE, ARM_SPLIT):
        payload = json.loads(
            (writer.out_dir(arm, batch_id) / "batch_meta.json").read_text(encoding="utf-8")
        )
        config = payload.get("config")
        if not isinstance(config, dict):
            raise _error(f"臂={arm} batch_meta.config 缺失或非 Mapping")
        observed[arm] = dict(config)
        recomputed = fingerprint(dict(config))
        recorded = str(payload.get("config_fingerprint", ""))
        if recorded != recomputed:
            raise _error(
                f"臂={arm} config_fingerprint 与 config 重算值不符（记录={recorded} "
                f"重算={recomputed}）；config 记录值与实际参数不得分裂"
            )
        fingerprints[arm] = recomputed
    missing = sorted(set(frozen) - set(observed[ARM_SINGLE]))
    if missing:
        raise _error(
            f"独立冻结的配置键清单未出现在 batch_meta.config：{missing}；"
            '比较器不得以"记录值 ⊆ summary 列"自证覆盖（两者同源）'
        )
    if expected_config is not None:
        target = {str(key): value for key, value in expected_config.items()}
        for arm, config in observed.items():
            decorated = {key: config.get(key) for key in target if key not in dynamic}
            if decorated != target:
                diff_keys = sorted(key for key in target if decorated.get(key) != target[key])
                raise _error(
                    f"臂={arm} batch_meta.config 与冻结清单不符，差异键={diff_keys}；"
                    "配置记录必须与宿主实际参数一致"
                )
            if extra := sorted(set(config) - set(target) - dynamic):
                raise _error(
                    f"臂={arm} batch_meta.config 出现冻结清单外的新键 {extra}；"
                    "新增配置键必须先入冻结清单再比较"
                )
    if uncovered := sorted(set(frozen) - set(summary_columns)):
        raise _error(
            f"独立冻结的配置键未被 summary 列覆盖：{uncovered}（summary 列 "
            f"{len(summary_columns)} 个）；比较器不得跳过任何配置键"
        )
    unresolved = sorted(set(observed[ARM_SINGLE]) - set(summary_columns) - dynamic)
    if unresolved:
        raise _error(
            f"batch_meta.config 键未被 summary 列覆盖：{unresolved}；比较器不得跳过任何配置键"
        )
    return observed[ARM_SINGLE], fingerprints[ARM_SINGLE]


def _expected_fold_dirs(writer: DualArmRunsWriter, batch_id: str) -> Tuple[str, ...]:
    """**契约折集合**：由 summary 与 chain_nav 的 ``split_index`` 唯一确定。

    runs 契约要求折级文件按折存在（任务 §2.2）；左臂目录枚举在"两臂同时缺整折"时
    会空遍历，因此折身份不取自目录，而取自批次级文件里记录的折号（外部评审
    T4-Codex-01）。
    """
    indices: set = set()
    for arm in (ARM_SINGLE, ARM_SPLIT):
        out_dir = writer.out_dir(arm, batch_id)
        for name in ("summary.parquet", "chain_nav.parquet"):
            frame = pd.read_parquet(out_dir / name, columns=["split_index"])
            indices |= {int(value) for value in frame["split_index"].unique()}
    if not indices:
        raise _error("summary/chain_nav 未记录任何折号，无法确定契约折集合")
    return tuple(_FOLD_DIR_TEMPLATE.format(index=index) for index in sorted(indices))


def _assert_fold_closure(
    writer: DualArmRunsWriter, batch_id: str, status: Mapping[str, str]
) -> Dict[str, Any]:
    """折集合闭合门：实际折目录 = 契约折集合，且每折必需文件两侧存在。

    Returns:
        登记项 ``{"expected": [...], "actual": [...], "missing_files": [...]}``。

    Raises:
        ComparisonInputError: 缺整折/多折/折级必需文件缺失。
    """
    expected = _expected_fold_dirs(writer, batch_id)
    actual = _fold_dirs(writer, batch_id)
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise _error(
            f"折集合与批次记录不符：契约折={list(expected)} 实际折={list(actual)}"
            f"（缺={missing} 多={extra}）；遗漏整折不得比遗漏折内文件更容易通过"
        )
    missing_files: List[str] = []
    for fold in expected:
        for name in REQUIRED_FOLD_FILES:
            relative = f"folds/{fold}/{name}"
            if status.get(relative) != _STATUS_PRESENT:
                missing_files.append(relative)
    if missing_files:
        raise _error(
            f"折级必需文件缺失 {missing_files}；"
            "任务 §2.2 要求折级 trades/daily/_meta 两侧必须存在"
        )
    return {"expected": list(expected), "actual": list(actual), "missing_files": []}


def _comparison_files(status: Mapping[str, str], folds: Sequence[str]) -> List[str]:
    """待比较文件清单（批次三件 + 契约折集合内实际存在的折级文件）。"""
    files = ["batch_meta.json", "summary.parquet", "chain_nav.parquet"]
    for fold in folds:
        for name in _FOLD_FILE_NAMES:
            relative = f"folds/{fold}/{name}"
            if relative in status:
                files.append(relative)
    return files


def compare_runs_trees(
    writer: DualArmRunsWriter,
    batch_id: str,
    *,
    excluded_fields: Sequence[str] = EXCLUDED_META_FIELDS,
    required_config_keys: Sequence[str] = (),
    expected_config: Optional[Mapping[str, Any]] = None,
    dynamic_config_keys: Sequence[str] = (),
    capture_dir: Optional[Path] = None,
) -> ArmDiffReport:
    """比较两臂 runs 树（身份绑定 + 缺失语义 + 折闭合 + 逐文件规范字节）。

    Args:
        writer/batch_id: 两臂写入器与批次键。
        excluded_fields: 排除表（只允许 ``created_at``/``arms``）。
        required_config_keys: **独立冻结**的指纹配置键清单（不得由 summary 列反推）。
        expected_config: 冻结的配置键值清单（动态键除外，逐键值相等）。
        dynamic_config_keys: 值随运行变化的键（例如 ``batch_id``）。
        capture_dir: 给定则把两侧实际比较的规范化字节落盘到该目录。

    Raises:
        ComparisonInputError: 排除表越界、折集合不闭合、配置清单不符。
        ArmIsolationError: 两臂身份绑定/文件集/哈希失败（T3 写入层守卫，先于比较）。
    """
    excluded = assert_exclusion_table(excluded_fields)
    for arm in (ARM_SINGLE, ARM_SPLIT):
        writer.load_generation(arm=arm, batch_id=batch_id)
    status = writer.artifact_status(batch_id=batch_id)
    fold_closure = _assert_fold_closure(writer, batch_id, status)
    files = _comparison_files(status, fold_closure["expected"])
    capture = _NormalizedCapture(capture_dir) if capture_dir is not None else None
    diffs: List[FieldDifference] = []
    hashes: List[Tuple[str, str, str]] = []
    summary_columns: List[str] = []
    fingerprint_counts: List[int] = []
    for relative in files:
        state = status.get(relative)
        if state is None:  # pragma: no cover - 文件集一致时不可达
            raise _error(f"{relative} 未出现在两臂文件集中")
        if state != _STATUS_PRESENT:
            continue
        left_path = writer.out_dir(ARM_SINGLE, batch_id) / relative
        right_path = writer.out_dir(ARM_SPLIT, batch_id) / relative
        if relative.endswith(".json"):
            file_diffs, pair = _compare_json_file(
                left_path, right_path, relative, excluded_fields=excluded, capture=capture
            )
            fingerprint_count = 0
        else:
            file_diffs, pair, columns, fingerprint_count = _compare_parquet_file(
                left_path, right_path, relative, capture=capture
            )
            if relative == "summary.parquet":
                summary_columns = columns
        diffs.extend(file_diffs)
        hashes.append((relative, pair[0], pair[1]))
        fingerprint_counts.append(fingerprint_count)
    _, recomputed = _assert_config_keys_compared(
        writer,
        batch_id,
        summary_columns,
        required_config_keys=required_config_keys,
        expected_config=expected_config,
        dynamic_config_keys=dynamic_config_keys,
    )
    return ArmDiffReport(
        differences=tuple(diffs),
        file_status=tuple(sorted(status.items())),
        file_sha256=tuple(hashes),
        excluded_fields=excluded,
        compared_summary_columns=tuple(summary_columns),
        fingerprint_key_count=max(fingerprint_counts) if fingerprint_counts else 0,
        normalized_files=tuple(capture.files) if capture is not None else (),
        fold_closure=dict(fold_closure),
        config_fingerprint_recomputed=recomputed,
    )


def compare_assemblies(
    assembly: Any,
    *,
    required_config_keys: Sequence[str] = (),
    expected_config: Optional[Mapping[str, Any]] = None,
    dynamic_config_keys: Sequence[str] = (),
    capture_dir: Optional[Path] = None,
) -> ArmDiffReport:
    """自拆装配的整体比较：两臂经济视图 + runs 树（**唯一 G4 判定入口**）。

    先过装配层自我比较防护（臂标签/袖数/generation 身份/信号指纹），再过比较器层
    视图身份门与写入层 generation 绑定；任一门失败都不会返回"零差异"。
    """
    assert_arms_are_distinct(assembly)
    diffs = list(compare_arm_views(assembly.single, assembly.split))
    runs_report = compare_runs_trees(
        assembly.writer,
        assembly.batch_id,
        required_config_keys=required_config_keys,
        expected_config=expected_config,
        dynamic_config_keys=dynamic_config_keys,
        capture_dir=capture_dir,
    )
    diffs.extend(runs_report.differences)
    return ArmDiffReport(
        differences=tuple(diffs),
        file_status=runs_report.file_status,
        file_sha256=runs_report.file_sha256,
        excluded_fields=runs_report.excluded_fields,
        compared_summary_columns=runs_report.compared_summary_columns,
        fingerprint_key_count=runs_report.fingerprint_key_count,
        normalized_files=runs_report.normalized_files,
        fold_closure=runs_report.fold_closure,
        config_fingerprint_recomputed=runs_report.config_fingerprint_recomputed,
    )


__all__ = (
    "ArmComparisonError",
    "ArmDiffReport",
    "ComparisonInputError",
    "EXCLUDED_META_FIELDS",
    "EXCLUSION_REASONS",
    "FieldDifference",
    "assert_exclusion_table",
    "compare_arm_views",
    "compare_assemblies",
    "compare_frames",
    "compare_runs_trees",
    "compare_scalars",
    "compare_value_trees",
    "normalized_frame_bytes",
    "normalized_json_bytes",
)
