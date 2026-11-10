# -*- coding: utf-8 -*-
"""P6.5 T5 独立同代状态信封：五项所有权、规范化编码、原子提交、校验先行、零副作用。

覆盖（任务 §3.1/§3.5）：
- 五项所有权齐备且同代（P4 blob / 物理 lot 伴随视图 / 虚拟账 / 结算与估值身份 /
  schema 与同代标记），物理侧成本/批次顺序/事件水位/初始身份**不从虚拟侧反推**；
- 规范化序列化：float64 位精确（``float.hex``）、payload 禁止裸 float、键排序 + ASCII；
- 原子提交：一次性校验 → 临时名 → 原子替换；禁止覆盖、禁止跳号/倒退；
- 校验先行：schema/版本/run_id/提交点/哈希/水位任一不符即拒绝，且**不生成新信封**、
  上一代信封 SHA 不变（逐字节比较）；
- 与 P4 的关系：``EngineState`` 键集与 blob 往返语义零改动。
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Dict

import pytest

from src.lazybull.v2.core.accounting.envelope_store import (
    EnvelopeStore,
    EnvelopeStoreError,
)
from src.lazybull.v2.core.accounting.engine_state import (
    dumps_blob,
    export_state,
    loads_blob,
    state_from_blob,
    state_to_blob,
)
from src.lazybull.v2.core.accounting.state_envelope import (
    ENVELOPE_SCHEMA_VERSION,
    OWNER_NAMES,
    OWNER_P4_ENGINE_STATE,
    OWNER_PHYSICAL_LOT_VIEW,
    OWNER_SCHEMA_GENERATION,
    OWNER_SETTLEMENT_IDENTITY,
    OWNER_VIRTUAL_LEDGER,
    EnvelopeOwner,
    StateEnvelope,
    StateEnvelopeCorruptError,
    StateEnvelopeError,
    StateEnvelopeIntegrityError,
    build_envelope,
    canonical_json_bytes,
    f64,
    make_owner,
    parse_document,
    require_envelope_run_id,
    sha256_bytes,
    unf64,
)
from src.lazybull.v2.core.accounting.state_owners import (
    assert_fresh_instance,
    build_identity_owner,
    build_p4_owner,
    build_physical_owner,
    build_virtual_owner,
    decode_owners,
)
from src.lazybull.v2.hosts.recovery_runtime import (
    RecoverableRuntime,
    RecoveryScenario,
)
from src.lazybull.v2.hosts.virtual_arm_runs import require_ascii_id

_RUN_ID = "p65t5-envelope"


def _runtime(tmp_path: Path, *, days: int = 2, run_id: str = _RUN_ID) -> RecoverableRuntime:
    """构造一个已跑到 ``days`` 日（含提交）的可恢复运行时。"""
    scenario = RecoveryScenario()
    runtime = RecoverableRuntime(scenario=scenario, root=tmp_path / "chain", run_id=run_id)
    for index in range(days):
        runtime.run_and_commit(index)
    return runtime


def _owners_of(runtime: RecoverableRuntime) -> Dict[str, EnvelopeOwner]:
    """当前时刻的五项中前四项所有者（第五项由 build_envelope 生成）。"""
    host = runtime.host
    date = runtime.scenario.date_list()[runtime._next_day - 1]  # noqa: SLF001 - 测试读内部状态
    states = host.virtual_ledger.snapshot()
    source = host._build_price_source(date, states=states)  # noqa: SLF001
    return {
        OWNER_P4_ENGINE_STATE: build_p4_owner(host),
        OWNER_PHYSICAL_LOT_VIEW: build_physical_owner(host.physical_lot_view),
        OWNER_VIRTUAL_LEDGER: build_virtual_owner(host.virtual_ledger),
        OWNER_SETTLEMENT_IDENTITY: build_identity_owner(
            host=host,
            ledger=host.virtual_ledger,
            price_source_id=source.source_id,
            price_source_as_of=str(source.as_of),
        ),
    }


def _build(
    runtime: RecoverableRuntime,
    *,
    commit_point: int,
    previous_commit_point: int,
    previous_sha: str,
    run_id: str = _RUN_ID,
    trade_date: str = "",
) -> StateEnvelope:
    """用当前运行时的四项状态构造一代信封（不落盘）。

    ``trade_date`` 缺省取当前已跑到的最后一日（必须与结算身份的价源日期相同，否则
    同代一致性校验会拒绝——这正是信封要拦的跨日错价）。
    """
    stamp = trade_date or runtime.scenario.trade_dates[runtime._next_day - 1]  # noqa: SLF001
    return build_envelope(
        run_id=run_id,
        commit_point=commit_point,
        trade_date=stamp,
        previous_commit_point=previous_commit_point,
        previous_envelope_sha256=previous_sha,
        state_owners=_owners_of(runtime),
    )


# ────────────────────────── 五项所有权 ──────────────────────────


def test_envelope_carries_exactly_five_owners() -> None:
    """五类状态所有权齐备：集合与顺序封闭，缺一/多一即拒绝。"""
    assert OWNER_NAMES == (
        OWNER_P4_ENGINE_STATE,
        OWNER_PHYSICAL_LOT_VIEW,
        OWNER_VIRTUAL_LEDGER,
        OWNER_SETTLEMENT_IDENTITY,
        OWNER_SCHEMA_GENERATION,
    )


def test_same_generation_commit_writes_five_owners(tmp_path: Path) -> None:
    """同代落盘：一个提交点文件内含五项、逐项哈希可复算、当日交易日一致。"""
    runtime = _runtime(tmp_path, days=1)
    stored = runtime.store.load(1)
    document = json.loads(stored.path.read_bytes().decode("ascii"))
    assert set(document["owners"]) == set(OWNER_NAMES)
    assert document["commit_point"] == 1
    assert document["trade_date"] == runtime.scenario.trade_dates[0]
    for name in OWNER_NAMES:
        payload = base64.b64decode(document["owners"][name]["payload_b64"])
        assert sha256_bytes(payload) == document["owners"][name]["sha256"]
        assert len(payload) == document["owners"][name]["bytes"]
    assert stored.file_sha256 == sha256_bytes(stored.path.read_bytes())


def test_physical_owner_records_cost_order_events_and_origin(tmp_path: Path) -> None:
    """物理伴随视图落盘成本/批次顺序/事件水位/初始冻结输入身份（不从虚拟侧反推）。"""
    runtime = _runtime(tmp_path, days=1)
    document = decode_owners(runtime.store.load(1).envelope)
    physical = document.physical
    assert [code for code, _lot in physical.order] == [
        runtime.scenario.ts_code,
        runtime.scenario.ts_code,
    ]
    assert [lot_id for _code, lot_id in physical.order] == [
        seed[0] for seed in runtime.scenario.physical_lots
    ]
    assert [shares for _c, _l, shares, _cost in physical.lots] == [
        int(seed[1]) for seed in runtime.scenario.physical_lots
    ]
    assert [cost for _c, _l, _s, cost in physical.lots] == [
        float(seed[2]) for seed in runtime.scenario.physical_lots
    ]
    assert physical.origin_fingerprint == runtime.host.physical_lot_view.origin_fingerprint()
    assert physical.events == ()


def test_physical_owner_rejects_reordered_batches(tmp_path: Path) -> None:
    """批次顺序被重排（内容不变）必须拒绝：顺序是所有权的一部分。"""
    runtime = _runtime(tmp_path, days=1)
    owner = build_physical_owner(runtime.host.physical_lot_view)
    payload = json.loads(owner.payload_bytes().decode("ascii"))
    payload["batch_order"] = list(reversed(payload["batch_order"]))
    tampered = make_owner(
        OWNER_PHYSICAL_LOT_VIEW, canonical_json_bytes(payload), owner.media_type
    )
    with pytest.raises(StateEnvelopeError, match="batch_order"):
        from src.lazybull.v2.core.accounting.state_owners import decode_physical_owner

        decode_physical_owner(tampered)


def test_settlement_identity_cross_checks_p4_and_virtual(tmp_path: Path) -> None:
    """结算/估值身份与 P4 账本键、虚拟账批次键必须逐项一致（同代一致性）。"""
    runtime = _runtime(tmp_path, days=2)
    restored = decode_owners(runtime.store.load(2).envelope)
    p4_ledger = restored.p4_state.ledger
    assert tuple(p4_ledger["applied_keys"]) == restored.identity.physical_fill_keys
    assert restored.identity.internal_settlement_keys == restored.virtual.settlement_keys()
    assert restored.identity.batch_ids == restored.virtual.batch_ids()
    assert restored.identity.price_source_as_of == runtime.scenario.trade_dates[1]
    assert restored.identity.last_event_sequence == (
        len(restored.virtual.settlement_keys()) + len(restored.virtual.batch_ids())
    )


# ────────────────────────── 规范化编码 ──────────────────────────


@pytest.mark.parametrize(
    "value",
    [0.0, -0.0, 1.0, 1.0 + 2**-52, 5e-324, 2.2250738585072014e-308, 1.7976931348623157e308],
)
def test_float64_hex_roundtrip_is_bitwise_exact(value: float) -> None:
    """float64 编解码位精确（含 -0.0、次正规数、1 ULP 差、极值）。"""
    encoded = f64(value)
    decoded = unf64(encoded)
    assert decoded == value or (value == 0.0 and str(decoded) == str(value))
    assert f64(decoded) == encoded


def test_payload_rejects_raw_float_and_non_finite() -> None:
    """payload 禁止裸 float（位精确纪律）；NaN/Inf 一律拒绝。"""
    with pytest.raises(StateEnvelopeError, match="未编码 float"):
        canonical_json_bytes({"x": 1.5})
    with pytest.raises(StateEnvelopeError, match="非有限"):
        f64(float("nan"))
    with pytest.raises(StateEnvelopeError, match="非有限"):
        f64(float("inf"))


def test_canonical_json_is_sorted_ascii_and_newline_terminated() -> None:
    """规范化 JSON：键排序 + ASCII 转义 + 紧凑分隔 + 末尾单换行。"""
    payload = {"b": 1, "a": [f64(1.0), "中文"], "c": None}
    payload_bytes = canonical_json_bytes(payload)
    text = payload_bytes.decode("ascii")
    assert text.endswith("}\n")
    assert '"a"' in text and text.index('"a"') < text.index('"b"') < text.index('"c"')
    assert "\\u4e2d" in text  # ensure_ascii：中文一律转义，字节与区域设置无关
    assert canonical_json_bytes(payload) == payload_bytes


def test_parse_document_rejects_version_and_kind_mismatch(tmp_path: Path) -> None:
    """schema 版本 / 记录类型不符：不可解析即终态（不得静默兼容）。"""
    runtime = _runtime(tmp_path, days=1)
    document = json.loads(runtime.store.load(1).path.read_bytes().decode("ascii"))
    bad_version = dict(document, schema_version=ENVELOPE_SCHEMA_VERSION + 1)
    with pytest.raises(StateEnvelopeCorruptError, match="schema 版本"):
        parse_document(bad_version)
    bad_kind = dict(document, record_kind="something_else")
    with pytest.raises(StateEnvelopeCorruptError, match="记录类型"):
        parse_document(bad_kind)


# ────────────────────────── 校验先行与零副作用 ──────────────────────────


def _snapshot(root: Path) -> Dict[str, str]:
    """目录内全部文件的内容摘要（零副作用判据）。"""
    return {
        path.relative_to(root).as_posix(): sha256_bytes(path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_commit_rejects_gap_and_regression_without_side_effects(tmp_path: Path) -> None:
    """提交点跳号/倒退必须拒绝，且不生成任何新文件、既有信封摘要不变。"""
    runtime = _runtime(tmp_path, days=1)
    store = runtime.store
    before = _snapshot(store.root)
    with pytest.raises(EnvelopeStoreError, match="提交点必须严格连续"):
        store.commit(
            _build(
                runtime,
                commit_point=3,
                previous_commit_point=2,
                previous_sha="0" * 64,
            )
        )
    assert _snapshot(store.root) == before


def test_cross_generation_watermark_regression_is_rejected(tmp_path: Path) -> None:
    """跨代水位收缩必须被写入路径拒绝（不是只留给测试的工具函数）。"""
    runtime = _runtime(tmp_path, days=3)
    store = runtime.store
    # 回到提交点 2 的宿主状态，却试图写第 4 代：同代内一致性全通过，只有跨代比较能发现
    runtime.restart(2)
    before = _snapshot(store.root)
    from src.lazybull.v2.hosts.recovery_runtime import RecoveryError

    with pytest.raises((RecoveryError, StateEnvelopeError), match="提交日序不符"):
        runtime.commit_day(3)
    assert _snapshot(store.root) == before
    # 直接经 EnvelopeStore 提交"旧代内容 + 新提交点"也必须被跨代守卫拒绝
    envelope = _build(
        runtime,
        commit_point=4,
        previous_commit_point=3,
        previous_sha=store.load(3).file_sha256,
    )
    with pytest.raises(StateEnvelopeIntegrityError, match="跨代收缩|水位倒退"):
        store.commit(envelope)
    assert _snapshot(store.root) == before


def test_chain_fingerprint_rewrite_across_generations_is_rejected(tmp_path: Path) -> None:
    """同一链级批次键的请求指纹跨代被改写 ⇒ 拒绝（id 集合相同也不行）。"""
    runtime = _runtime(tmp_path, days=2)
    store = runtime.store
    stored = store.load(2)
    envelope = stored.envelope
    identity_owner = envelope.owner(OWNER_SETTLEMENT_IDENTITY)
    payload = json.loads(identity_owner.payload_bytes().decode("ascii"))
    assert payload["chain_batch_fingerprints"], "第 2 代应已登记链级批次指纹"
    batch_id, fingerprint = payload["chain_batch_fingerprints"][0]
    payload["chain_batch_fingerprints"][0] = [batch_id, "deadbeef" * 8]
    forged_owner = make_owner(
        OWNER_SETTLEMENT_IDENTITY,
        canonical_json_bytes(payload),
        identity_owner.media_type,
    )
    rewritten = build_envelope(
        run_id=envelope.run_id,
        commit_point=3,
        trade_date=envelope.trade_date,
        previous_commit_point=2,
        previous_envelope_sha256=stored.file_sha256,
        state_owners={
            OWNER_P4_ENGINE_STATE: envelope.owner(OWNER_P4_ENGINE_STATE),
            OWNER_PHYSICAL_LOT_VIEW: envelope.owner(OWNER_PHYSICAL_LOT_VIEW),
            OWNER_VIRTUAL_LEDGER: envelope.owner(OWNER_VIRTUAL_LEDGER),
            OWNER_SETTLEMENT_IDENTITY: forged_owner,
        },
    )
    before = _snapshot(store.root)
    with pytest.raises(StateEnvelopeIntegrityError, match="请求指纹跨代变化"):
        store.commit(rewritten)
    assert _snapshot(store.root) == before
    assert fingerprint != "deadbeef" * 8


def test_rejected_commit_creates_no_chain_root(tmp_path: Path) -> None:
    """被拒绝的提交不得创建链根目录（零副作用含目录）。"""
    runtime = _runtime(tmp_path, days=1)
    missing_root = tmp_path / "never-created"
    store = EnvelopeStore(missing_root, run_id=_RUN_ID)
    with pytest.raises(EnvelopeStoreError, match="尚无任何信封|提交点必须严格连续"):
        store.commit(
            _build(
                runtime,
                commit_point=2,
                previous_commit_point=1,
                previous_sha="0" * 64,
            )
        )
    assert not missing_root.exists()


def test_commit_refuses_to_overwrite_existing_generation(tmp_path: Path) -> None:
    """同提交点不得再写一次（禁止覆盖既有信封）。"""
    runtime = _runtime(tmp_path, days=1)
    store = runtime.store
    path = store.path_of(1)
    original = path.read_bytes()
    envelope = _build(runtime, commit_point=1, previous_commit_point=0, previous_sha="")
    with pytest.raises(EnvelopeStoreError, match="已存在"):
        store.commit(envelope)
    assert path.read_bytes() == original


def test_owner_hash_tamper_is_integrity_error(tmp_path: Path) -> None:
    """任一所有者字节被改动 ⇒ 哈希破缺（五项逐项覆盖）。"""
    runtime = _runtime(tmp_path, days=1)
    base_document = json.loads(runtime.store.load(1).path.read_bytes().decode("ascii"))
    for name in OWNER_NAMES:
        tampered = json.loads(json.dumps(base_document))
        payload = base64.b64decode(tampered["owners"][name]["payload_b64"])
        tampered["owners"][name]["payload_b64"] = base64.b64encode(
            payload + b" "
        ).decode("ascii")
        with pytest.raises(StateEnvelopeIntegrityError, match="哈希不符"):
            parse_document(tampered)


def test_marker_must_match_header(tmp_path: Path) -> None:
    """同代标记与信封头部不符（交易日被改写）⇒ 完整性破缺。"""
    runtime = _runtime(tmp_path, days=1)
    document = json.loads(runtime.store.load(1).path.read_bytes().decode("ascii"))
    marker = json.loads(
        base64.b64decode(document["owners"][OWNER_SCHEMA_GENERATION]["payload_b64"]).decode(
            "ascii"
        )
    )
    marker["trade_date"] = "20991231"
    marker_bytes = canonical_json_bytes(marker)
    document["owners"][OWNER_SCHEMA_GENERATION]["payload_b64"] = base64.b64encode(
        marker_bytes
    ).decode("ascii")
    document["owners"][OWNER_SCHEMA_GENERATION]["sha256"] = sha256_bytes(marker_bytes)
    document["owners"][OWNER_SCHEMA_GENERATION]["bytes"] = len(marker_bytes)
    with pytest.raises(StateEnvelopeIntegrityError, match="同代标记"):
        parse_document(document)


def test_run_id_mismatch_between_store_and_envelope(tmp_path: Path) -> None:
    """跨运行拼接：信封 run_id 与链不符即拒绝。"""
    runtime = _runtime(tmp_path, days=1)
    other = EnvelopeStore(runtime.store.root, run_id="p65t5-other")
    with pytest.raises(StateEnvelopeIntegrityError, match="运行 ID"):
        other.load(1)


def test_envelope_rejects_wrong_run_id_at_build() -> None:
    """构造期 run_id 非法（空/首尾空白/路径分隔符/上跳标记）一律拒绝。"""
    for bad in ("", None, " ", " lead", "trail ", "a/b", "a\\b", ".", "..", "a..b", "a:b"):
        with pytest.raises(StateEnvelopeError):
            require_envelope_run_id(bad)


def test_run_id_rule_matches_evidence_layer() -> None:
    """信封层与证据层 run_id 规则逐例一致（单一语义、两处实现由测试锁定）。"""
    cases = [
        "ok_id_1",
        "UPPER",
        "with-dash",
        "with.dot",
        "a b",
        "a/b",
        "a\\b",
        ".",
        "..",
        "中文",
        "",
        " lead",
        "trail ",
        "colon:1",
        "star*",
    ]
    for value in cases:
        envelope_ok = True
        evidence_ok = True
        try:
            require_envelope_run_id(value)
        except (StateEnvelopeError, ValueError):
            envelope_ok = False
        try:
            require_ascii_id(value, "run_id")
        except ValueError:
            evidence_ok = False
        assert envelope_ok == evidence_ok, f"run_id 规则不一致：{value!r}"


def test_watermark_regression_helpers() -> None:
    """水位单调性：键集只增不减、顺序不得重排、最后成交日不得倒退。"""
    from src.lazybull.v2.core.accounting.state_envelope import (
        assert_no_state_shrink,
        assert_no_watermark_regression,
    )

    previous = {
        "applied_keys": ["k1"],
        "settlement_keys": ["s1"],
        "batch_ids": ["b1"],
        "last_fill_date": "2026-01-06 00:00:00",
        "last_event_sequence": 2,
    }
    grown = {
        "applied_keys": ["k1", "k2"],
        "settlement_keys": ["s1", "s2"],
        "batch_ids": ["b1", "b2"],
        "last_fill_date": "2026-01-09 00:00:00",
        "last_event_sequence": 4,
    }
    assert_no_watermark_regression(current=grown, previous=previous, label="t")
    assert_no_state_shrink(current=grown, previous=previous, label="t")
    shrunk = dict(grown, applied_keys=["k2"])
    with pytest.raises(StateEnvelopeIntegrityError, match="水位倒退"):
        assert_no_watermark_regression(current=shrunk, previous=previous, label="t")
    with pytest.raises(StateEnvelopeIntegrityError, match="收缩"):
        assert_no_state_shrink(current=dict(grown, batch_ids=["b2"]), previous=previous, label="t")
    reordered = dict(grown, applied_keys=["k2", "k1"])
    with pytest.raises(StateEnvelopeIntegrityError, match="重排"):
        assert_no_watermark_regression(current=reordered, previous=previous, label="t")
    with pytest.raises(StateEnvelopeIntegrityError, match="序号倒退"):
        assert_no_state_shrink(
            current=dict(grown, last_event_sequence=1), previous=previous, label="t"
        )
    # 链级批次身份缺键分支（A-R3-02）：上一代已登记、本代缺失即收缩
    with pytest.raises(StateEnvelopeIntegrityError, match="chain_batch_ids"):
        assert_no_state_shrink(
            current=dict(grown, chain_batch_ids=[]),
            previous=dict(previous, chain_batch_ids=["p65t5-d1"]),
            label="t",
        )


# ────────────────────────── 与 P4 的关系 ──────────────────────────


def test_commit_does_not_change_engine_state_contract(tmp_path: Path) -> None:
    """信封落盘不改变 ``EngineState`` 键集与 blob 往返语义（P4 契约零改动）。"""
    runtime = _runtime(tmp_path, days=1)
    before = set(state_to_blob(export_state(runtime.host))["fields"])
    runtime.run_and_commit(1)
    after = set(state_to_blob(export_state(runtime.host))["fields"])
    assert before == after
    assert not any("envelope" in name for name in after)
    assert set(after) == before


def test_p4_blob_roundtrip_is_byte_identical(tmp_path: Path) -> None:
    """P4 blob 的 export→restore→export 字节一致（既有契约，本任务不得削弱）。"""
    runtime = _runtime(tmp_path, days=2)
    blob = state_to_blob(export_state(runtime.host))
    first = dumps_blob(blob)
    again = dumps_blob(state_to_blob(state_from_blob(loads_blob(first))))
    assert first == again


def test_restore_target_must_be_fresh_instance(tmp_path: Path) -> None:
    """恢复目标必须是新建实例：已演化实例（有成交/持仓/结算身份）一律拒绝。"""
    runtime = _runtime(tmp_path, days=2)
    with pytest.raises(StateEnvelopeError, match="不是新建实例"):
        assert_fresh_instance(runtime.host)
    fresh = RecoverableRuntime(
        scenario=runtime.scenario, root=tmp_path / "other", run_id="fresh", seed_initial_state=False
    )
    fresh.host.virtual_ledger.open_account("A", 1000.0)
    with pytest.raises(StateEnvelopeError, match="已开户"):
        assert_fresh_instance(fresh.host)


# ────────────────────────── 跨所有者经济一致性（T5-Codex-C1） ──────────────────────────


def test_mixed_generation_physical_owner_is_rejected(tmp_path: Path) -> None:
    """物理视图取自旧代、其余取本代 ⇒ 写盘前拒绝（五项必须同一套经济事实）。

    原实现只核对身份键/价源：给旧视图重算外层哈希与同代 marker 后即可与"别的代"的
    P4/虚拟账拼成可提交信封，落盘后恢复起点即 I2 破缺（第 3 日真实买入 100 股，
    旧视图仍为 400 股而 P4 已 500 股）。
    """
    runtime = _runtime(tmp_path, days=3)
    store = runtime.store
    stale = store.load(1).envelope.owner(OWNER_PHYSICAL_LOT_VIEW)
    runtime.run_day(3)  # D3 买入 100 股：本代物理视图与 cp1 不同
    owners = _owners_of(runtime)
    owners[OWNER_PHYSICAL_LOT_VIEW] = stale
    mixed = build_envelope(
        run_id=_RUN_ID,
        commit_point=4,
        trade_date=runtime.scenario.trade_dates[3],  # noqa: SLF001 - 测试读内部状态
        previous_commit_point=3,
        previous_envelope_sha256=store.load(3).file_sha256,
        state_owners=owners,
    )
    before = _snapshot(store.root)
    with pytest.raises(StateEnvelopeError, match="来源批次股数不一致|物理来源 lot 视图"):
        store.commit(mixed)
    assert _snapshot(store.root) == before
    assert store.latest_commit_point() == 3


def test_source_lot_misallocation_is_rejected(tmp_path: Path) -> None:
    """代码总量不变、来源之间一增一减 ⇒ 仍拒绝（I2 的粒度是来源 lot，不是代码汇总）。

    R1-01 路径 A 的定点负例：确定性宿主里把同一代码的两个来源批次各改 1 股（净零），
    物理总量与 P4 仍相等、现金也不用动，但按 ``(ts_code, source_lot_id)`` 对账即破缺。
    """
    runtime = RecoverableRuntime(
        scenario=RecoveryScenario(), root=tmp_path / "chain", run_id=_RUN_ID
    )
    runtime.run_day(0)
    view = runtime.host.physical_lot_view
    lots = view._lots  # noqa: SLF001 - 故障注入需要改写来源批次股数
    keys = sorted(key for key in lots if lots[key] > 0)
    assert len(keys) >= 2, "合成场景应至少有两个来源批次"
    left, right = keys[0], keys[1]
    assert left[0] == right[0], "同一代码的两个来源批次才构成净零错配"
    lots[left] += 1
    lots[right] -= 1
    before = _snapshot(runtime.store.root)
    with pytest.raises(StateEnvelopeError, match="来源批次股数不一致"):
        runtime.commit_day(0)
    assert _snapshot(runtime.store.root) == before
    assert runtime.store.latest_commit_point() == 0


def test_balanced_cash_with_stale_valuation_is_rejected(tmp_path: Path) -> None:
    """两侧现金同步 +1（I1 仍成立）但当日日终估值陈旧 ⇒ 拒绝（R1-01 路径 B）。"""
    runtime = RecoverableRuntime(
        scenario=RecoveryScenario(), root=tmp_path / "chain", run_id=_RUN_ID
    )
    runtime.run_day(0)
    runtime.host.current_capital += 1.0
    runtime.host.virtual_ledger._books["A"].cash += 1.0  # noqa: SLF001 - 同步改虚拟侧
    before = _snapshot(runtime.store.root)
    with pytest.raises(StateEnvelopeError, match="当日日终快照 capital"):
        runtime.commit_day(0)
    assert _snapshot(runtime.store.root) == before
    assert runtime.store.latest_commit_point() == 0


def test_rewritten_valuation_single_field_is_rejected(tmp_path: Path) -> None:
    """只改写日终快照的 capital（总值不动）⇒ 快照自洽性被绑定并拒绝。"""
    runtime = RecoverableRuntime(
        scenario=RecoveryScenario(), root=tmp_path / "chain", run_id=_RUN_ID
    )
    runtime.run_day(0)
    runtime.host.current_capital += 1.0
    runtime.host.virtual_ledger._books["A"].cash += 1.0  # noqa: SLF001
    runtime.host.portfolio_values[-1]["capital"] += 1.0
    before = _snapshot(runtime.store.root)
    with pytest.raises(StateEnvelopeError, match="不自洽"):
        runtime.commit_day(0)
    assert _snapshot(runtime.store.root) == before
    assert runtime.store.latest_commit_point() == 0


def test_nan_snapshot_capital_is_rejected(tmp_path: Path) -> None:
    """日终快照现金为 NaN ⇒ 拒绝（R2-01：不能靠不等式隐式拒绝 NaN）。

    ``abs(NaN - cash) > tol`` 恒为 False，若不先验有限性，NaN 现金会被当作有效
    日终事实提交、解码并从信封恢复。
    """
    runtime = RecoverableRuntime(
        scenario=RecoveryScenario(), root=tmp_path / "chain", run_id=_RUN_ID
    )
    runtime.run_day(0)
    runtime.host.portfolio_values[-1]["capital"] = float("nan")
    before = _snapshot(runtime.store.root)
    with pytest.raises(StateEnvelopeError, match="非有限"):
        runtime.commit_day(0)
    assert _snapshot(runtime.store.root) == before
    assert runtime.store.latest_commit_point() == 0


def test_nan_snapshot_market_value_is_rejected(tmp_path: Path) -> None:
    """日终快照市值为 NaN ⇒ 拒绝（三项统一验有限，不遗漏任一字段）。"""
    runtime = RecoverableRuntime(
        scenario=RecoveryScenario(), root=tmp_path / "chain", run_id=_RUN_ID
    )
    runtime.run_day(0)
    runtime.host.portfolio_values[-1]["market_value"] = float("inf")
    before = _snapshot(runtime.store.root)
    with pytest.raises(StateEnvelopeError, match="非有限"):
        runtime.commit_day(0)
    assert _snapshot(runtime.store.root) == before
    assert runtime.store.latest_commit_point() == 0


def test_post_gate_cash_mutation_is_rejected(tmp_path: Path) -> None:
    """恒等式门通过之后改写物理现金 ⇒ 提交仍被拒（不是"过了门就永久可信"）。"""
    runtime = RecoverableRuntime(
        scenario=RecoveryScenario(), root=tmp_path / "chain", run_id=_RUN_ID
    )
    runtime.run_day(0)
    runtime.host.current_capital += 1.0
    before = _snapshot(runtime.store.root)
    with pytest.raises(StateEnvelopeError, match="P4 现金"):
        runtime.commit_day(0)
    assert _snapshot(runtime.store.root) == before
    assert runtime.store.latest_commit_point() == 0


# ────────────────────────── 只有临时文件的半提交现场（T5-Codex-C2） ──────────────────────────


def test_temp_only_residue_is_reported_not_silently_ignored(tmp_path: Path) -> None:
    """rename 前退出（只有 temp、没有 final）必须被读侧识别为半提交代。"""
    runtime = _runtime(tmp_path, days=1)
    store = runtime.store
    store.temp_path_of(2).write_bytes(b'{"incomplete":')
    before = _snapshot(store.root)
    with pytest.raises(StateEnvelopeCorruptError, match="只有临时文件"):
        store.load_latest()
    with pytest.raises(StateEnvelopeCorruptError, match="只有临时文件"):
        store.assert_chain_intact()
    # 显式读取破缺前完整代仍然可用（恢复入口不受影响）
    assert store.load(1).commit_point == 1
    assert _snapshot(store.root) == before
