"""P6.5 T5 独立同代状态信封（Q17=B：**不改** P4 ``EngineState`` 键集/字段/blob 语义）。

契约依据：
- 设计 ``docs/plans/v2/p65_merger_reconciliation_design_20261108.md`` §4.5（快照与恢复
  语义、五项状态所有权表）、§7.2（失败模式与错误上下文）、§10.1 G5/G7；
- ``docs/contracts/netting_freeze.md`` F25 §4/§5、``docs/contracts/protocols.md``
  F13 §3.3/§3.4（虚拟账为运行时概念、无借用守卫）。

五项状态所有权（**同一原子提交点**、**先全量校验再统一写入**）：

| 所有者 | 内容 | 恢复依据 |
|---|---|---|
| ``p4_engine_state`` | P4 基础 blob（物理现金/持仓/trades、幂等键与最后成交日、演化字段） | 既有 ``EngineState`` 校验 |
| ``physical_lot_view`` | 物理来源 lot 伴随视图（来源 lot ID、股数、成本、批次顺序、事件水位、初始身份） | 信封直接恢复；**不得从虚拟侧反推** |
| ``virtual_ledger`` | 各袖现金、来源/取得双层 lot、费用、占用、结算身份与批次结果 | 同一信封同代恢复 |
| ``settlement_identity`` | 物理/内部结算幂等键、结算事件序号、最后价源标识与日期、链级批次指纹 | 防重复结算与跨日错价 |
| ``schema_generation`` | 信封版本、运行 ID、提交点、四项文件哈希、上一代提交点与信封哈希 | 先全量校验再写 |

编码纪律（本模块为唯一实现）：
- 规范化 JSON：``sort_keys`` + 紧凑分隔符 + ``ensure_ascii`` + 末尾单换行，字节可复核；
- **payload 中禁止裸 float**：所有 float64 一律经 ``f64()`` 编码为 ``{"$f64": "<hex>"}``
  （``float.hex`` 位精确、``-0.0`` 与次正规数均无损往返），因此任何一处十进制近似
  都无法混入同代比较；
- P4 blob 保持既有 ``pickle`` 协议 5 整 blob 字节（``engine_state.dumps_blob``），
  以 base64 内联，SHA-256 直接对该字节计算——"逐位一致"以字节相等为判据。

失败语义（**零副作用**）：
- ``StateEnvelopeError``：输入/结构/身份/水位校验拒绝（**不生成新信封**、不改上一代
  信封哈希、不改任何内存状态）；
- ``StateEnvelopeIntegrityError``：哈希不符、代际不符、提交点倒退、幂等键集收缩等
  完整性问题（``RuntimeError`` 语义：终态，禁止续写）；
- ``StateEnvelopeCorruptError``：磁盘信封不可解析/半提交代残留（终态；只能从提交点
  之前的完整信封新建实例重放）。

与 P4 的关系（硬边界）：本模块只**读** ``EngineState``（``state_to_blob`` /
``state_from_blob`` / ``restore_state``），不改其键集、字段与校验语义；P6.5 扩展状态
全部由本信封承载。本模块不做文件 IO（原子落盘见 ``accounting/envelope_store.py``），
不读配置、不读真实行情。
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

#: 信封 schema 版本（五项所有权或编码规则变更必须递增）
ENVELOPE_SCHEMA_VERSION = 1
#: 信封记录类型（磁盘形态身份；与 schema 版本互相独立校验）
ENVELOPE_RECORD_KIND = "v2_p65_state_envelope"
#: 本模块唯一阶段标识（进入全部错误上下文）
STAGE = "P6.5"

#: 五项状态所有者（集合封闭：缺一/多一即拒绝）
OWNER_P4_ENGINE_STATE = "p4_engine_state"
OWNER_PHYSICAL_LOT_VIEW = "physical_lot_view"
OWNER_VIRTUAL_LEDGER = "virtual_ledger"
OWNER_SETTLEMENT_IDENTITY = "settlement_identity"
OWNER_SCHEMA_GENERATION = "schema_generation"
OWNER_NAMES: Tuple[str, ...] = (
    OWNER_P4_ENGINE_STATE,
    OWNER_PHYSICAL_LOT_VIEW,
    OWNER_VIRTUAL_LEDGER,
    OWNER_SETTLEMENT_IDENTITY,
    OWNER_SCHEMA_GENERATION,
)
#: 承载"状态搬运"的四项（第五项是标记，自身哈希由其余四项派生）
STATE_OWNER_NAMES: Tuple[str, ...] = OWNER_NAMES[:4]
#: 规范化 JSON 媒介类型
MEDIA_TYPE_CANONICAL_JSON = "application/json;lazybull-canonical-v1"
#: P4 blob 媒介类型（固定 pickle 协议 5）
MEDIA_TYPE_PICKLE5 = "application/x-python-pickle-5"
#: 允许的媒介类型（拒绝未知编码）
_MEDIA_TYPES = (MEDIA_TYPE_CANONICAL_JSON, MEDIA_TYPE_PICKLE5)


class StateEnvelopeError(ValueError):
    """信封输入/结构/身份/水位校验拒绝（**零副作用**：不产生任何新信封）。"""


class StateEnvelopeIntegrityError(RuntimeError):
    """信封完整性/代际破缺（终态：禁止续写，只能从提交点前的完整信封重放）。"""


class StateEnvelopeCorruptError(StateEnvelopeIntegrityError):
    """磁盘信封不可解析或为半提交代残留（终态；禁止静默回退到旧代）。"""


def _error(message: str) -> Exception:
    """带阶段上下文的信封输入拒绝。"""
    return StateEnvelopeError(f"阶段={STAGE} 状态信封 {message}")


def _integrity(message: str) -> Exception:
    """带阶段上下文的信封完整性破缺。"""
    return StateEnvelopeIntegrityError(f"阶段={STAGE} 状态信封 {message}")


def f64(value: Any) -> Dict[str, str]:
    """float64 → 位精确编码（``float.hex``；拒绝 NaN/Inf，避免位比较失去意义）。"""
    number = float(value)
    if not math.isfinite(number):
        raise _error(f"float64 编码拒绝非有限值 {value!r}（NaN/Inf 不参与同代比较）")
    return {"$f64": number.hex()}


def unf64(node: Any, *, field: str = "") -> float:
    """位精确编码 → float64（形态非法即拒绝；不接受裸 float/数字字符串）。"""
    if not isinstance(node, Mapping) or set(node.keys()) != {"$f64"}:
        raise _error(f"float64 解码失败（{field or '<未命名>'}）：期望 {{'$f64': '<hex>'}}，实际 {node!r}")
    raw = node["$f64"]
    if not isinstance(raw, str):
        raise _error(f"float64 解码失败（{field or '<未命名>'}）：hex 必须为 str，实际 {type(raw).__name__}")
    try:
        return float.fromhex(raw)
    except ValueError as error:
        raise _error(
            f"float64 解码失败（{field or '<未命名>'}）：{raw!r} 不是合法 hex 浮点（{error}）"
        ) from error


def _assert_no_raw_float(node: Any, path: str = "$") -> None:
    """payload 禁止裸 float（位精确纪律：必须显式 ``f64()``）。"""
    if isinstance(node, float):
        raise _error(
            f"规范化 payload 含未编码 float（{path}={node!r}）；"
            "所有 float64 必须经 f64() 编码为 {'$f64': '<hex>'}"
        )
    if isinstance(node, Mapping):
        for key, value in node.items():
            if not isinstance(key, str):
                raise _error(f"规范化 payload 的键必须为 str（{path} 下出现 {type(key).__name__}）")
            _assert_no_raw_float(value, f"{path}.{key}")
        return
    if isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            _assert_no_raw_float(value, f"{path}[{index}]")


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """规范化 JSON 字节（``sort_keys`` + 紧凑 + ASCII + 末尾换行；唯一实现）。"""
    if not isinstance(payload, Mapping):
        raise _error(f"规范化序列化需要 Mapping，实际 {type(payload).__name__}")
    _assert_no_raw_float(payload)
    text = json.dumps(
        dict(payload), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return (text + "\n").encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    """字节 SHA-256（信封/所有者/证据共用的唯一摘要口径）。"""
    return hashlib.sha256(payload).hexdigest()


def require_envelope_run_id(value: Any) -> str:
    """信封层运行标识校验（与证据层 ``require_ascii_id`` **同一规则集**）。

    规则：非空、无首尾空白、ASCII、不含 ``/`` ``\\`` ``..`` ``:``、不得为 ``.``/``..``。
    本模块位于 ``v2/core``（不得反向依赖 ``v2/hosts``），规则一致性由测试逐例锁定
    （两侧对同一批输入必须给出相同的接受/拒绝结论）。
    """
    raw = str(value if value is not None else "")
    text = raw.strip()
    if not text:
        raise _error("run_id 不得为空")
    if raw != text:
        raise _error(f"run_id 不得含首尾空白，实际 {value!r}")
    if not text.isascii():
        raise _error(f"run_id 必须为 ASCII：{text!r}")
    if any(separator in text for separator in ("/", "\\", "..", ":")):
        raise _error(f"run_id 不得包含路径分隔符/上跳：{text!r}")
    if text in {".", ".."}:
        raise _error(f"run_id 不得为当前/父目录标记：{text!r}")
    return text


@dataclass(frozen=True, slots=True)
class EnvelopeOwner:
    """单个状态所有者的落盘形态（媒介类型 + 字节 + SHA-256）。"""

    name: str
    media_type: str
    sha256: str
    byte_size: int
    payload_b64: str

    def payload_bytes(self) -> bytes:
        """所有者原始字节（base64 解码；解码失败即拒绝）。"""
        try:
            return base64.b64decode(self.payload_b64.encode("ascii"), validate=True)
        except (ValueError, UnicodeEncodeError) as error:  # pragma: no cover - 由构造侧保证
            raise _error(f"所有者 {self.name} 的 payload 不是合法 base64：{error}") from error

    def document(self) -> Dict[str, Any]:
        """磁盘形态（键集封闭）。"""
        return {
            "media_type": self.media_type,
            "sha256": self.sha256,
            "bytes": self.byte_size,
            "payload_b64": self.payload_b64,
        }


@dataclass(frozen=True, slots=True)
class StateEnvelope:
    """同代状态信封（五项所有权 + 同代标记；``envelope_sha256`` 为**文件字节**摘要）。

    ``envelope_sha256`` 只由 ``EnvelopeStore`` 在原子落盘/读取时赋予：文件自身不含
    自身摘要（避免自引用），上一代摘要记录在下一代与证据 manifest 中。
    """

    run_id: str
    commit_point: int
    trade_date: str
    previous_commit_point: int
    previous_envelope_sha256: str
    owners: Mapping[str, EnvelopeOwner]
    schema_version: int = ENVELOPE_SCHEMA_VERSION
    envelope_sha256: str = ""

    @property
    def generation_token(self) -> str:
        """同代标记（运行 ID + 提交点；进入错误上下文与证据键）。"""
        return f"{self.run_id}#{self.commit_point:06d}"

    def owner(self, name: str) -> EnvelopeOwner:
        """按名取所有者（缺即拒绝）。"""
        owner = self.owners.get(name)
        if owner is None:
            raise _error(f"信封 {self.generation_token} 缺少所有者 {name!r}")
        return owner

    def payload(self, name: str) -> bytes:
        """所有者原始字节。"""
        return self.owner(name).payload_bytes()

    def with_envelope_sha(self, digest: str) -> "StateEnvelope":
        """返回赋予文件摘要后的副本（只由存储层调用）。"""
        from dataclasses import replace

        return replace(self, envelope_sha256=digest)


def _owner_hashes(owners: Mapping[str, EnvelopeOwner]) -> Dict[str, str]:
    """四项状态所有者哈希（标记自身不在其中，避免自引用）。"""
    return {name: owners[name].sha256 for name in STATE_OWNER_NAMES}


def build_marker_owner(
    *,
    run_id: str,
    commit_point: int,
    trade_date: str,
    previous_commit_point: int,
    previous_envelope_sha256: str,
    owners: Mapping[str, EnvelopeOwner],
) -> EnvelopeOwner:
    """构造第五项所有者（schema 与同代标记；含四项文件哈希与上一代身份）。"""
    payload = {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "record_kind": "v2_p65_schema_generation_marker",
        "envelope_schema_version": ENVELOPE_SCHEMA_VERSION,
        "run_id": run_id,
        "commit_point": int(commit_point),
        "trade_date": trade_date,
        "previous_commit_point": int(previous_commit_point),
        "previous_envelope_sha256": previous_envelope_sha256,
        "owner_file_sha256": _owner_hashes(owners),
        "owner_count": len(STATE_OWNER_NAMES),
        "note": (
            "五项所有权同一原子提交点；本项只记录版本/同代身份与四项文件哈希，"
            "信封自身摘要由存储层对文件字节计算并记入下一代与证据 manifest"
        ),
    }
    payload_bytes = canonical_json_bytes(payload)
    return EnvelopeOwner(
        name=OWNER_SCHEMA_GENERATION,
        media_type=MEDIA_TYPE_CANONICAL_JSON,
        sha256=sha256_bytes(payload_bytes),
        byte_size=len(payload_bytes),
        payload_b64=base64.b64encode(payload_bytes).decode("ascii"),
    )


def make_owner(name: str, payload: bytes, media_type: str) -> EnvelopeOwner:
    """由原始字节构造所有者（摘要与长度同时登记，形成互相校验）。"""
    if name not in STATE_OWNER_NAMES:
        raise _error(f"未知状态所有者 {name!r}；合法集合={list(STATE_OWNER_NAMES)}")
    if media_type not in _MEDIA_TYPES:
        raise _error(f"所有者 {name} 的媒介类型非法 {media_type!r}；合法={list(_MEDIA_TYPES)}")
    if not isinstance(payload, (bytes, bytearray)) or not payload:
        raise _error(f"所有者 {name} 的 payload 必须为非空 bytes")
    raw = bytes(payload)
    return EnvelopeOwner(
        name=name,
        media_type=media_type,
        sha256=sha256_bytes(raw),
        byte_size=len(raw),
        payload_b64=base64.b64encode(raw).decode("ascii"),
    )


def _is_hex_digest(value: Any) -> bool:
    """64 位小写十六进制摘要判定。"""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _validate_envelope_request(
    *,
    run_id: str,
    commit_point: int,
    trade_date: str,
    previous_commit_point: int,
    previous_envelope_sha256: str,
    state_owners: Mapping[str, EnvelopeOwner],
) -> None:
    """构造期输入校验（提交点连续性/交易日形态/上一代摘要/四项键集）。"""
    require_envelope_run_id(run_id)
    if isinstance(commit_point, bool) or not isinstance(commit_point, int):
        raise _error(f"提交点必须为 int，实际 {type(commit_point).__name__}={commit_point!r}")
    if commit_point < 1:
        raise _error(f"提交点必须从 1 起（0 = 无信封），实际 {commit_point}")
    if not isinstance(trade_date, str) or not trade_date.isdigit() or len(trade_date) != 8:
        raise _error(f"信封交易日必须为 YYYYMMDD 字符串，实际 {trade_date!r}")
    if isinstance(previous_commit_point, bool) or not isinstance(previous_commit_point, int):
        raise _error(
            f"上一代提交点必须为 int（无上一代用 0），实际 {previous_commit_point!r}"
        )
    if previous_commit_point != commit_point - 1:
        raise _error(
            f"提交点必须严格连续：本代 {commit_point} 要求上一代 {commit_point - 1}，"
            f"实际 {previous_commit_point}（禁止跳号或倒退）"
        )
    if not isinstance(previous_envelope_sha256, str):
        raise _error(
            f"上一代信封摘要必须为 str，实际 {type(previous_envelope_sha256).__name__}"
        )
    if previous_commit_point == 0 and previous_envelope_sha256:
        raise _error("无上一代（提交点 0）时上一代信封摘要必须为空")
    if previous_commit_point > 0 and not _is_hex_digest(previous_envelope_sha256):
        raise _error(
            f"上一代信封摘要必须为 64 位小写十六进制，实际 {previous_envelope_sha256!r}"
        )
    names = set(state_owners.keys())
    expected = set(STATE_OWNER_NAMES)
    if names != expected:
        raise _error(
            f"四项状态所有者集合不符：缺 {sorted(expected - names) or '无'} / "
            f"多 {sorted(names - expected) or '无'}"
        )
    for name in STATE_OWNER_NAMES:
        if state_owners[name].name != name:
            raise _error(f"所有者键 {name!r} 与其名称 {state_owners[name].name!r} 不一致")


def build_envelope(
    *,
    run_id: str,
    commit_point: int,
    trade_date: str,
    previous_commit_point: int,
    previous_envelope_sha256: str,
    state_owners: Mapping[str, EnvelopeOwner],
) -> StateEnvelope:
    """由四项状态所有者构造完整信封（第五项标记在此生成）。

    全部可预见校验在任何写入之前完成；构造失败不产生任何对象/文件。
    """
    _validate_envelope_request(
        run_id=run_id,
        commit_point=commit_point,
        trade_date=trade_date,
        previous_commit_point=previous_commit_point,
        previous_envelope_sha256=previous_envelope_sha256,
        state_owners=state_owners,
    )
    owners: Dict[str, EnvelopeOwner] = dict(state_owners)
    owners[OWNER_SCHEMA_GENERATION] = build_marker_owner(
        run_id=run_id,
        commit_point=commit_point,
        trade_date=trade_date,
        previous_commit_point=previous_commit_point,
        previous_envelope_sha256=previous_envelope_sha256,
        owners=owners,
    )
    envelope = StateEnvelope(
        run_id=run_id,
        commit_point=commit_point,
        trade_date=trade_date,
        previous_commit_point=previous_commit_point,
        previous_envelope_sha256=previous_envelope_sha256,
        owners=MappingProxyType({name: owners[name] for name in OWNER_NAMES}),
    )
    validate_envelope(envelope)
    return envelope


def document(envelope: StateEnvelope) -> Dict[str, Any]:
    """信封 → 磁盘文档（键集封闭；不含自身摘要，避免自引用）。"""
    return {
        "schema_version": int(envelope.schema_version),
        "record_kind": ENVELOPE_RECORD_KIND,
        "run_id": envelope.run_id,
        "commit_point": int(envelope.commit_point),
        "trade_date": envelope.trade_date,
        "previous": {
            "commit_point": int(envelope.previous_commit_point),
            "envelope_sha256": envelope.previous_envelope_sha256,
        },
        "owners": {name: envelope.owner(name).document() for name in OWNER_NAMES},
    }


def envelope_bytes(envelope: StateEnvelope) -> bytes:
    """信封 → 规范化文件字节（存储层原子落盘与重算摘要的唯一来源）。"""
    return canonical_json_bytes(document(envelope))


def parse_document(payload: Mapping[str, Any]) -> StateEnvelope:
    """磁盘文档 → 信封（结构 + 哈希全量校验；任何不符即拒绝）。"""
    if not isinstance(payload, Mapping):
        raise StateEnvelopeCorruptError(
            f"阶段={STAGE} 状态信封文档必须为对象，实际 {type(payload).__name__}"
        )
    expected_keys = {
        "schema_version",
        "record_kind",
        "run_id",
        "commit_point",
        "trade_date",
        "previous",
        "owners",
    }
    if set(payload.keys()) != expected_keys:
        raise StateEnvelopeCorruptError(
            f"阶段={STAGE} 状态信封文档键集不符：缺 "
            f"{sorted(expected_keys - set(payload.keys())) or '无'} / 多 "
            f"{sorted(set(payload.keys()) - expected_keys) or '无'}"
        )
    if payload["record_kind"] != ENVELOPE_RECORD_KIND:
        raise StateEnvelopeCorruptError(
            f"阶段={STAGE} 状态信封记录类型不符：期望 {ENVELOPE_RECORD_KIND!r}，"
            f"实际 {payload['record_kind']!r}"
        )
    if payload["schema_version"] != ENVELOPE_SCHEMA_VERSION:
        raise StateEnvelopeCorruptError(
            f"阶段={STAGE} 状态信封 schema 版本不符：期望 {ENVELOPE_SCHEMA_VERSION}，"
            f"实际 {payload['schema_version']!r}"
        )
    previous = payload["previous"]
    if not isinstance(previous, Mapping) or set(previous.keys()) != {
        "commit_point",
        "envelope_sha256",
    }:
        raise StateEnvelopeCorruptError(
            f"阶段={STAGE} 状态信封 previous 形态非法：{previous!r}"
        )
    raw_owners = payload["owners"]
    if not isinstance(raw_owners, Mapping):
        raise StateEnvelopeCorruptError(
            f"阶段={STAGE} 状态信封 owners 必须为对象，实际 {type(raw_owners).__name__}"
        )
    if set(raw_owners.keys()) != set(OWNER_NAMES):
        raise StateEnvelopeCorruptError(
            f"阶段={STAGE} 状态信封所有者集合不符：缺 "
            f"{sorted(set(OWNER_NAMES) - set(raw_owners.keys())) or '无'} / 多 "
            f"{sorted(set(raw_owners.keys()) - set(OWNER_NAMES)) or '无'}"
        )
    owners: Dict[str, EnvelopeOwner] = {}
    for name in OWNER_NAMES:
        entry = raw_owners[name]
        if not isinstance(entry, Mapping) or set(entry.keys()) != {
            "media_type",
            "sha256",
            "bytes",
            "payload_b64",
        }:
            raise StateEnvelopeCorruptError(
                f"阶段={STAGE} 状态信封所有者 {name!r} 形态非法：{entry!r}"
            )
        if entry["media_type"] not in _MEDIA_TYPES:
            raise StateEnvelopeCorruptError(
                f"阶段={STAGE} 状态信封所有者 {name!r} 媒介类型非法：{entry['media_type']!r}"
            )
        if not isinstance(entry["bytes"], int) or isinstance(entry["bytes"], bool):
            raise StateEnvelopeCorruptError(
                f"阶段={STAGE} 状态信封所有者 {name!r} 的 bytes 必须为 int"
            )
        owners[name] = EnvelopeOwner(
            name=name,
            media_type=entry["media_type"],
            sha256=entry["sha256"],
            byte_size=entry["bytes"],
            payload_b64=entry["payload_b64"],
        )
    envelope = StateEnvelope(
        run_id=payload["run_id"],
        commit_point=payload["commit_point"],
        trade_date=payload["trade_date"],
        previous_commit_point=previous["commit_point"],
        previous_envelope_sha256=previous["envelope_sha256"],
        owners=MappingProxyType(owners),
    )
    validate_envelope(envelope)
    return envelope


def _check_owner_hash(owner: EnvelopeOwner, *, context: str = "") -> None:
    """单所有者字节/长度/摘要三重校验（任一不符即完整性破缺）。

    ``context`` 由调用方传入同代标识（阶段 / 信封版本 / run_id / 提交点 / 交易日），
    错误信息据此可直接定位到代——信封层的拒绝也必须带上下文（任务 §2.2）。
    """
    label = f"阶段={STAGE} 状态信封 v{ENVELOPE_SCHEMA_VERSION}"
    if context:
        label = f"{label} 代={context}"
    payload = owner.payload_bytes()
    digest = sha256_bytes(payload)
    if digest != owner.sha256:
        raise StateEnvelopeIntegrityError(
            f"{label} 所有者 {owner.name} 哈希不符：登记 {owner.sha256} "
            f"实算 {digest}（字节数 登记 {owner.byte_size} 实算 {len(payload)}）"
        )
    if len(payload) != owner.byte_size:
        raise StateEnvelopeIntegrityError(
            f"{label} 所有者 {owner.name} 字节数不符：登记 {owner.byte_size} "
            f"实算 {len(payload)}"
        )


def validate_marker(envelope: StateEnvelope) -> Dict[str, Any]:
    """第五项标记校验（版本/运行 ID/提交点/交易日/上一代/四项哈希逐项相等）。"""
    marker_bytes = envelope.payload(OWNER_SCHEMA_GENERATION)
    try:
        marker = json.loads(marker_bytes.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StateEnvelopeIntegrityError(
            f"阶段={STAGE} 信封 {envelope.generation_token} 的同代标记不可解析：{error}"
        ) from error
    if not isinstance(marker, Mapping):
        raise StateEnvelopeIntegrityError(
            f"阶段={STAGE} 信封 {envelope.generation_token} 的同代标记必须为对象"
        )
    expected = {
        "envelope_schema_version": envelope.schema_version,
        "run_id": envelope.run_id,
        "commit_point": envelope.commit_point,
        "trade_date": envelope.trade_date,
        "previous_commit_point": envelope.previous_commit_point,
        "previous_envelope_sha256": envelope.previous_envelope_sha256,
        "owner_file_sha256": _owner_hashes(envelope.owners),
        "owner_count": len(STATE_OWNER_NAMES),
    }
    drift = {
        key: (marker.get(key), value)
        for key, value in expected.items()
        if marker.get(key) != value
    }
    if drift:
        raise StateEnvelopeIntegrityError(
            f"阶段={STAGE} 信封 {envelope.generation_token} 的同代标记与信封头部不符"
            f"（实际, 期望）={drift}"
        )
    return marker


def _validate_previous_link(envelope: StateEnvelope, previous: StateEnvelope) -> None:
    """与上一代的代际衔接校验（提交点 +1、摘要相等、运行 ID 相同）。"""
    if envelope.commit_point != previous.commit_point + 1:
        raise _error(
            f"提交点不连续：上一代 {previous.commit_point}，本代 {envelope.commit_point}"
            f"（必须严格 +1）"
        )
    if envelope.previous_commit_point != previous.commit_point:
        raise _error(
            f"上一代提交点登记不符：信封记 {envelope.previous_commit_point}，"
            f"实际上一代 {previous.commit_point}"
        )
    if not previous.envelope_sha256:
        raise _error("上一代信封缺少文件摘要（禁止用未落盘信封作为恢复起点）")
    if envelope.previous_envelope_sha256 != previous.envelope_sha256:
        raise _error(
            f"上一代信封摘要不符：信封记 {envelope.previous_envelope_sha256}，"
            f"实际 {previous.envelope_sha256}"
        )
    if envelope.run_id != previous.run_id:
        raise _error(
            f"同代链运行 ID 不一致：上一代 {previous.run_id!r}，本代 {envelope.run_id!r}"
        )


def validate_envelope(
    envelope: StateEnvelope,
    *,
    expected_run_id: Optional[str] = None,
    previous: Optional[StateEnvelope] = None,
) -> StateEnvelope:
    """信封全量校验（**纯校验、零写入**；任何拒绝都不改任何状态）。

    Args:
        envelope: 待校验信封
        expected_run_id: 期望运行 ID（给出时必须相等，防止跨运行拼接）
        previous: 期望的上一代信封（给出时提交点/摘要必须严格衔接）

    Raises:
        StateEnvelopeError: 结构/身份/水位不符（输入拒绝）。
        StateEnvelopeIntegrityError: 哈希或代际破缺（完整性终态）。
    """
    if not isinstance(envelope, StateEnvelope):
        raise _error(f"信封类型非法：{type(envelope).__name__}")
    if envelope.schema_version != ENVELOPE_SCHEMA_VERSION:
        raise _error(
            f"信封 schema 版本不符：期望 {ENVELOPE_SCHEMA_VERSION}，实际 {envelope.schema_version!r}"
        )
    if tuple(envelope.owners.keys()) != OWNER_NAMES:
        raise _error(
            f"信封 {envelope.generation_token} 所有者顺序/集合不符："
            f"期望 {list(OWNER_NAMES)}，实际 {list(envelope.owners.keys())}"
        )
    require_envelope_run_id(envelope.run_id)
    if expected_run_id is not None and envelope.run_id != expected_run_id:
        raise _error(
            f"信封运行 ID 不符：期望 {expected_run_id!r}，实际 {envelope.run_id!r}"
            f"（禁止跨运行拼接状态）"
        )
    if envelope.commit_point < 1:
        raise _error(f"信封提交点必须从 1 起，实际 {envelope.commit_point}")
    for name in OWNER_NAMES:
        _check_owner_hash(envelope.owner(name), context=envelope.generation_token)
    validate_marker(envelope)
    if envelope.commit_point > 1 and not envelope.previous_envelope_sha256:
        raise _error(
            f"信封 {envelope.generation_token} 缺少上一代摘要（提交点 > 1 必须衔接上一代）"
        )
    if previous is not None:
        _validate_previous_link(envelope, previous)
    return envelope


def assert_no_watermark_regression(
    *, current: Mapping[str, Any], previous: Mapping[str, Any], label: str
) -> None:
    """水位单调性（幂等键集只增不减、最后日期不倒退；**纯校验**）。

    Args:
        current: 本代水位投影（``applied_keys`` / ``settlement_keys`` / ``last_fill_date``）
        previous: 上一代同形投影
        label: 错误上下文（阶段/提交点/日期）
    """
    for key_name in ("applied_keys", "settlement_keys"):
        current_keys = tuple(current.get(key_name, ()))
        previous_keys = tuple(previous.get(key_name, ()))
        if len(set(current_keys)) != len(current_keys):
            raise _error(f"{label}：{key_name} 含重复键（幂等键集不得重复）")
        missing = [key for key in previous_keys if key not in current_keys]
        if missing:
            raise StateEnvelopeIntegrityError(
                f"{label}：{key_name} 水位倒退——上一代 {len(previous_keys)} 个键中有 "
                f"{len(missing)} 个在本代消失（示例={missing[:3]}）；"
                "幂等键集只增不减，禁止跨代重用水位"
            )
        if list(current_keys[: len(previous_keys)]) != list(previous_keys):
            # 允许追加但禁止重排：键序是同一重放流的确定性产物
            raise StateEnvelopeIntegrityError(
                f"{label}：{key_name} 键序相对上一代发生重排（前缀不相等）；"
                "同代水位必须是同一重放流的确定性追加"
            )
    current_last = current.get("last_fill_date")
    previous_last = previous.get("last_fill_date")
    if previous_last is not None:
        if current_last is None:
            raise StateEnvelopeIntegrityError(
                f"{label}：last_fill_date 水位倒退（上一代 {previous_last!r}，本代清空）"
            )
        if _as_sort_key(str(current_last)) < _as_sort_key(str(previous_last)):
            raise StateEnvelopeIntegrityError(
                f"{label}：last_fill_date 倒退（上一代 {previous_last!r}，本代 {current_last!r}）"
            )


def _as_sort_key(text: str) -> Tuple[int, ...]:
    """日期/时间戳字符串的自然序键（非数字按原串比较）。"""
    digits = [int(char) for char in text if char.isdigit()]
    return tuple(digits) if digits else (0,)


def assert_no_state_shrink(
    *, current: Mapping[str, Any], previous: Mapping[str, Any], label: str
) -> None:
    """跨代状态不得收缩（结算/链级身份只增不减、事件序号单调）。

    口径登记（T5-A-R1-03）：``last_event_sequence`` 是**派生值**
    （= 结算身份数 + 批次身份数），不是独立计数器；它的作用是与上一代比较"是否单调
    不减"，以及在同一代内证明"身份集合未被绕过"——不承担"独立发现事件回退"的职责
    （真正的回退由键集包含关系判定）。
    """
    for key_name in ("batch_ids", "settlement_keys", "chain_batch_ids"):
        current_keys = tuple(current.get(key_name, ()))
        missing = [key for key in tuple(previous.get(key_name, ())) if key not in current_keys]
        if missing:
            raise StateEnvelopeIntegrityError(
                f"{label}：{key_name} 跨代收缩（缺失示例={missing[:3]}）；"
                "结算/链级身份只增不减，禁止跨代重用水位"
            )
    current_sequence = int(current.get("last_event_sequence", 0))
    previous_sequence = int(previous.get("last_event_sequence", 0))
    if current_sequence < previous_sequence:
        raise StateEnvelopeIntegrityError(
            f"{label}：结算事件序号倒退（上一代 {previous_sequence}，本代 {current_sequence}）"
        )


def owner_payload_document(owner: EnvelopeOwner) -> Dict[str, Any]:
    """JSON 所有者 → 解析后的文档（媒介类型不是规范化 JSON 即拒绝）。"""
    if owner.media_type != MEDIA_TYPE_CANONICAL_JSON:
        raise _error(
            f"所有者 {owner.name} 的媒介类型为 {owner.media_type!r}，不能按 JSON 解析"
        )
    try:
        return json.loads(owner.payload_bytes().decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StateEnvelopeIntegrityError(
            f"阶段={STAGE} 所有者 {owner.name} 的规范化 JSON 不可解析：{error}"
        ) from error


def require_key_set(payload: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    """键集封闭校验（缺键/多键即拒绝）。"""
    actual = set(payload.keys())
    wanted = set(expected)
    if actual != wanted:
        raise _error(
            f"{label} 键集不符：缺 {sorted(wanted - actual) or '无'} / "
            f"多 {sorted(actual - wanted) or '无'}"
        )


__all__ = (
    "ENVELOPE_RECORD_KIND",
    "ENVELOPE_SCHEMA_VERSION",
    "MEDIA_TYPE_CANONICAL_JSON",
    "MEDIA_TYPE_PICKLE5",
    "OWNER_NAMES",
    "OWNER_P4_ENGINE_STATE",
    "OWNER_PHYSICAL_LOT_VIEW",
    "OWNER_SCHEMA_GENERATION",
    "OWNER_SETTLEMENT_IDENTITY",
    "OWNER_VIRTUAL_LEDGER",
    "STATE_OWNER_NAMES",
    "EnvelopeOwner",
    "STAGE",
    "StateEnvelope",
    "StateEnvelopeCorruptError",
    "StateEnvelopeError",
    "StateEnvelopeIntegrityError",
    "assert_no_state_shrink",
    "assert_no_watermark_regression",
    "build_envelope",
    "build_marker_owner",
    "canonical_json_bytes",
    "document",
    "envelope_bytes",
    "f64",
    "make_owner",
    "owner_payload_document",
    "parse_document",
    "require_envelope_run_id",
    "require_key_set",
    "sha256_bytes",
    "unf64",
    "validate_envelope",
    "validate_marker",
)
