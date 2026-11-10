# -*- coding: utf-8 -*-
"""P6.5 T4 经济工件逐位比较器：字段覆盖、位模式/dtype/键序/NaN、缺失语义、排除表冻结。

覆盖：
- 排除表仅 ``created_at``/``arms`` 且运行前冻结（增、删、改序全部拒绝）；
- 浮点按 float64 位模式（1 ULP 即差异）、NaN 位模式、NaN 位置、dtype、列序、键序；
- 规范化字节（Parquet 无压缩无索引 / JSON ``sort_keys``）确定且可区分；
- 两臂经济视图**逐字段覆盖**（逐项单独变异必须各报一条差异，路径可定位）；
- 差异清单含字段路径/左右值/位模式/来源文件与行；``assert_clean`` 汇总抛出；
- runs 文件级篡改（summary 单值 1 ULP）必须被捕获（先重签 generation 排除哈希噪声）；
- 比较器源码不得使用 ``approx``/``isclose``（禁止容差兜底）。

全部为合成输入：不读真实配置/行情。
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import shutil
import struct
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from src.lazybull.v2.evidence import arm_comparator
from src.lazybull.v2.evidence.fingerprint_keys import fingerprint
from src.lazybull.v2.evidence.arm_comparator import (
    EXCLUDED_META_FIELDS,
    EXCLUSION_REASONS,
    ArmComparisonError,
    ArmDiffReport,
    ComparisonInputError,
    FieldDifference,
    assert_exclusion_table,
    compare_arm_views,
    compare_assemblies,
    compare_frames,
    compare_runs_trees,
    compare_scalars,
    compare_value_trees,
    normalized_frame_bytes,
    normalized_json_bytes,
)
from src.lazybull.v2.hosts.self_split_arms import (
    SelfSplitScenario,
    run_self_split_arms,
)
from src.lazybull.v2.hosts.virtual_arm_runs import (
    ARM_SINGLE,
    ARM_SPLIT,
    ArmIsolationError,
    file_hashes,
    generation_identity,
)

_CONFIG = {
    "bt_top_n": 1,
    "label_column": "syn",
    "stock_domain": "main",
    "wf_start_date": "20260105",
    "wf_end_date": "20260107",
}


@pytest.fixture(scope="module")
def assembly(tmp_path_factory):
    base = tmp_path_factory.mktemp("t4cmp")
    return run_self_split_arms(
        scenario=SelfSplitScenario(),
        run_id="t4cmp",
        base_dir=base,
        writer_config=_CONFIG,
        code_state={"git_commit": "syn", "git_dirty": False},
        data_state={"data_state_id": "syn_t4cmp"},
    )


# ────────────────────────── 排除表冻结 ──────────────────────────


def test_exclusion_table_is_frozen_to_two_fields() -> None:
    assert EXCLUDED_META_FIELDS == ("created_at", "arms")
    assert set(EXCLUSION_REASONS) == set(EXCLUDED_META_FIELDS)
    assert assert_exclusion_table(EXCLUDED_META_FIELDS) == EXCLUDED_META_FIELDS
    with pytest.raises(ComparisonInputError, match="多出"):
        assert_exclusion_table(("created_at", "arms", "run_id"))
    with pytest.raises(ComparisonInputError, match="缺少"):
        assert_exclusion_table(("created_at",))
    with pytest.raises(ComparisonInputError, match="冻结"):
        assert_exclusion_table(("arms", "created_at"))


# ────────────────────────── 位模式 / dtype / 键序 / NaN ──────────────────────────


def test_one_ulp_difference_is_a_difference() -> None:
    value = 1.00395
    nudged = value + 2.0**-52
    diffs = compare_scalars("nav", value, nudged, source="syn")
    assert len(diffs) == 1
    assert diffs[0].field_path == "nav"
    assert diffs[0].left_bits != diffs[0].right_bits
    assert diffs[0].left == value and diffs[0].right == nudged
    assert compare_scalars("nav", value, value) == ()


def test_signed_zero_and_nan_payload_are_bitwise_differences() -> None:
    """``+0.0`` vs ``-0.0``（`==` 相等但位模式不同）与不同 NaN 载荷必须是差异。"""
    diffs = compare_scalars("x", 0.0, -0.0)
    assert len(diffs) == 1
    assert diffs[0].left_bits == "0000000000000000"
    assert diffs[0].right_bits == "8000000000000000"
    assert 0.0 == -0.0  # 数值相等，位模式不同：只有位比较能区分

    payload_nan = struct.unpack(">d", struct.pack(">Q", 0x7FF8000000000001))[0]
    assert payload_nan != payload_nan  # NaN 语义
    nan_diffs = compare_scalars("x", payload_nan, float("nan"))
    assert len(nan_diffs) == 1 and nan_diffs[0].left_bits != nan_diffs[0].right_bits

    left = pd.DataFrame({"v": [0.0]})
    right = pd.DataFrame({"v": [-0.0]})
    frame_diffs = compare_frames(
        "f",
        left,
        right,
        columns=["v"],
        key_columns=(),
        source_left="L",
        source_right="R",
    )
    assert [item.field_path for item in frame_diffs] == ["f.v"]
    assert frame_diffs[0].left_bits != frame_diffs[0].right_bits


def test_nan_bit_patterns_and_positions_are_compared() -> None:
    """NaN 只与同位模式 NaN 相等；NaN 位置差异必须是差异（不做 NaN==NaN 放行）。"""
    nan = float("nan")
    assert compare_scalars("x", nan, nan) == ()
    diffs = compare_scalars("x", nan, 1.0)
    assert len(diffs) == 1 and diffs[0].left_bits != diffs[0].right_bits
    assert compare_scalars("x", 1.0, 1) == ()  # 数值相等即通过（dtype 由帧比较负责）
    assert len(compare_scalars("x", "a", 1)) == 1


def test_value_tree_compares_key_order_and_nesting() -> None:
    left = {"a": 1, "b": {"c": 2.0}}
    right = {"b": {"c": 2.0}, "a": 1}
    diffs = compare_value_trees(left, right, source="meta")
    assert len(diffs) == 1
    assert diffs[0].field_path == "<keys>"
    assert diffs[0].left == ["a", "b"] and diffs[0].right == ["b", "a"]

    nested = compare_value_trees(
        {"a": {"c": 1.0}}, {"a": {"c": math.nextafter(1.0, math.inf)}}, source="meta"
    )
    assert len(nested) == 1
    assert nested[0].field_path == "a.c"


def test_frame_comparison_covers_dtype_order_nan_and_values() -> None:
    columns = ("trade_date", "ts_code", "amount")
    left = pd.DataFrame(
        {
            "trade_date": ["20260106"],
            "ts_code": ["600000.SH"],
            "amount": [1.0],
        }
    )
    right = left.copy()
    assert (
        compare_frames(
            "trades",
            left,
            right,
            columns=list(columns),
            key_columns=("trade_date", "ts_code"),
            source_left="L",
            source_right="R",
        )
        == ()
    )

    # dtype 不同
    typed = left.copy()
    typed["amount"] = typed["amount"].astype("float32")
    diffs = compare_frames(
        "trades",
        left,
        typed,
        columns=list(columns),
        key_columns=(),
        source_left="L",
        source_right="R",
    )
    assert any(item.field_path == "trades.amount.<dtype>" for item in diffs)

    # 值位模式不同
    nudged = left.copy()
    nudged["amount"] = [math.nextafter(1.0, math.inf)]
    diffs = compare_frames(
        "trades",
        left,
        nudged,
        columns=list(columns),
        key_columns=("trade_date", "ts_code"),
        source_left="L",
        source_right="R",
    )
    assert [item.field_path for item in diffs] == ["trades.amount"]
    assert diffs[0].row == 0 and diffs[0].left_bits != diffs[0].right_bits

    # NaN 位置
    missing = left.copy()
    missing["amount"] = [None]
    diffs = compare_frames(
        "trades",
        left,
        missing,
        columns=list(columns),
        key_columns=(),
        source_left="L",
        source_right="R",
    )
    paths = [item.field_path for item in diffs]
    assert "trades.amount" in paths
    value_diff = next(item for item in diffs if item.field_path == "trades.amount")
    assert value_diff.right is None and value_diff.left == 1.0

    # 键序被改写
    reordered = pd.concat([left, left]).reset_index(drop=True)
    reordered.loc[0, "ts_code"] = "000001.SZ"
    base_two = pd.concat([left, left]).reset_index(drop=True)
    base_two.loc[0, "ts_code"] = "600000.SH"
    diffs = compare_frames(
        "trades",
        base_two,
        reordered,
        columns=list(columns),
        key_columns=("trade_date", "ts_code"),
        source_left="L",
        source_right="R",
    )
    assert any(item.field_path == "trades.<key_order>" for item in diffs)


def test_normalized_bytes_are_deterministic_and_discriminating() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", None]})
    first = normalized_frame_bytes(frame, ["a", "b"])
    second = normalized_frame_bytes(frame.copy(), ["a", "b"])
    assert first == second
    nudged = pd.DataFrame({"a": [math.nextafter(1.0, math.inf), 2.0], "b": ["x", None]})
    assert normalized_frame_bytes(nudged, ["a", "b"]) != first
    with pytest.raises(ComparisonInputError, match="缺少契约列"):
        normalized_frame_bytes(frame, ["a", "b", "c"])

    assert normalized_json_bytes({"b": 1, "a": 2}) == normalized_json_bytes({"a": 2, "b": 1})
    assert normalized_json_bytes({"a": 1}) != normalized_json_bytes({"a": 2})


# ────────────────────────── 经济视图逐字段覆盖 ──────────────────────────


def _mutations(arm) -> list[tuple[str, object]]:
    """逐字段变异清单（每项只改一个字段，路径必须能被比较器定位）。"""
    fill = arm.fills[0]
    return [
        ("physical.cash", replace(arm, physical_cash=arm.physical_cash - 0.01)),
        (
            "physical.total_value",
            replace(arm, physical_total_value=arm.physical_total_value + 0.01),
        ),
        ("physical.positions", replace(arm, physical_positions=(("600000.SH", 399),))),
        ("physical.source_lots", replace(arm, physical_source_lots=(("600000.SH", "L9", 400),))),
        (
            "physical.fills.<count>",
            replace(arm, fills=(fill, replace(fill, fill_key="extra"))),
        ),
        ("physical.fills[0].shares", replace(arm, fills=(replace(fill, shares=300),))),
        (
            "physical.fills[0].price",
            replace(arm, fills=(replace(fill, price=math.nextafter(10.0, math.inf)),)),
        ),
        ("physical.fills[0].fee", replace(arm, fills=(replace(fill, fee=6.0),))),
        ("physical.fills[0].order_target", replace(arm, fills=(replace(fill, order_target=300),))),
        (
            "physical.fills[0].requested_shares",
            replace(arm, fills=(replace(fill, requested_shares=300),)),
        ),
        (
            "physical.fills[0].cash_before",
            replace(arm, fills=(replace(fill, cash_before=99_999.0),)),
        ),
        ("physical.fills[0].cash_after", replace(arm, fills=(replace(fill, cash_after=95_994.0),))),
        ("physical.fills[0].fill_key", replace(arm, fills=(replace(fill, fill_key="k2"),))),
        (
            "physical.fills[0].source_lot_id",
            replace(arm, fills=(replace(fill, source_lot_id="PHY:k2"),)),
        ),
        ("virtual.cash_sum", replace(arm, merged_virtual_cash=arm.merged_virtual_cash - 0.01)),
        (
            "virtual.merged_total_value",
            replace(arm, merged_total_value=arm.merged_total_value + 1.0),
        ),
        (
            "virtual.merged_nav",
            replace(arm, merged_nav=math.nextafter(arm.merged_nav, math.inf)),
        ),
        ("virtual.source_lots", replace(arm, virtual_source_lots=(("600000.SH", "L9", 400),))),
    ]


def test_every_economic_field_is_covered_by_the_comparator(assembly) -> None:
    """逐字段变异：每个经济字段都必须各报一条可定位差异（覆盖而非"整体相等"）。"""
    assert compare_arm_views(assembly.single, assembly.split) == ()
    for expected_path, mutated in _mutations(assembly.split):
        diffs = compare_arm_views(assembly.single, mutated)
        paths = [item.field_path for item in diffs]
        assert expected_path in paths, (expected_path, paths)
        item = next(entry for entry in diffs if entry.field_path == expected_path)
        assert item.source == "single_reference vs virtual_split"
        assert item.left != item.right


def test_report_aggregates_all_differences_and_raises() -> None:
    diffs = (
        FieldDifference(
            "a", 1.0, math.nextafter(1.0, math.inf), "3ff0000000000000", "3ff0000000000001", "f"
        ),
        FieldDifference("b", 2, 3, "", "", "g", 7),
    )
    report = ArmDiffReport(
        differences=diffs,
        file_status=(("x", "present"),),
        file_sha256=(),
        excluded_fields=EXCLUDED_META_FIELDS,
        compared_summary_columns=(),
        fingerprint_key_count=0,
    )
    assert report.ok is False
    text = report.describe_differences()
    assert "3ff0000000000000" in text and "行=7" in text
    with pytest.raises(ArmComparisonError, match="2 处差异"):
        report.assert_clean()


# ────────────────────────── 文件级篡改 ──────────────────────────


def _resign(writer, arm: str, batch_id: str) -> None:
    """重签 generation 使树哈希自洽（把比较焦点限制在字节/位差异上）。"""
    path = writer.generation_path(arm)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["file_hashes"] = [
        [relative, digest] for relative, digest in file_hashes(Path(document["out_dir"]))
    ]
    document["generation_id"] = generation_identity(document)
    path.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def test_summary_single_value_1ulp_tampering_is_caught(tmp_path: Path) -> None:
    assembly = run_self_split_arms(
        scenario=SelfSplitScenario(),
        run_id="t4cmp2",
        base_dir=tmp_path,
        writer_config=_CONFIG,
        code_state={"git_commit": "syn", "git_dirty": False},
        data_state={"data_state_id": "syn_t4cmp2"},
    )
    compare_runs_trees(assembly.writer, assembly.batch_id).assert_clean()

    path = assembly.writer.out_dir(ARM_SINGLE, assembly.batch_id) / "summary.parquet"
    frame = pd.read_parquet(path)
    frame["bt_sharpe"] = [math.nextafter(value, math.inf) for value in frame["bt_sharpe"]]
    frame.to_parquet(path, index=False, compression="none")
    _resign(assembly.writer, ARM_SINGLE, assembly.batch_id)

    report = compare_runs_trees(assembly.writer, assembly.batch_id)
    assert not report.ok
    paths = [item.field_path for item in report.differences]
    assert "summary.bt_sharpe" in paths
    item = next(entry for entry in report.differences if entry.field_path == "summary.bt_sharpe")
    assert item.row == 0 and item.left_bits != item.right_bits
    assert "single/summary.parquet" in item.source
    with pytest.raises(ArmComparisonError):
        report.assert_clean()


def test_fold_meta_json_tampering_is_caught(tmp_path: Path) -> None:
    assembly = run_self_split_arms(
        scenario=SelfSplitScenario(),
        run_id="t4cmp3",
        base_dir=tmp_path,
        writer_config=_CONFIG,
        code_state={"git_commit": "syn", "git_dirty": False},
        data_state={"data_state_id": "syn_t4cmp3"},
    )
    path = (
        assembly.writer.out_dir(ARM_SPLIT, assembly.batch_id) / "folds" / "split00" / "_meta.json"
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    document["topk_detail.missing"] = False
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    _resign(assembly.writer, ARM_SPLIT, assembly.batch_id)

    report = compare_runs_trees(assembly.writer, assembly.batch_id)
    assert report.differences
    assert any("topk_detail.missing" in item.field_path for item in report.differences)


def test_comparator_uses_no_tolerance_shortcuts() -> None:
    """比较器源码不得出现 approx/isclose（禁止容差兜底）；容差常量不得来自恒等式。"""
    source = inspect.getsource(arm_comparator)
    for forbidden in ("approx(", "isclose(", "rel_tol", "abs_tol"):
        assert forbidden not in source, forbidden
    assert "CASH_TOLERANCE" not in source and "NAV_TOLERANCE" not in source


def test_frame_key_comparison_handles_missing_keys() -> None:
    """键列含 NaN/pd.NA 时不得产生伪差异，也不得抛 ``bool`` 异常。"""
    columns = ("signal_date", "planned_ts_code", "actual_ts_code", "pred_score")
    left = pd.DataFrame(
        {
            "signal_date": ["20260105", None],
            "planned_ts_code": ["600000.SH", "000001.SZ"],
            "actual_ts_code": [pd.NA, None],
            "pred_score": [0.9, 0.5],
        }
    )
    right = left.copy()
    assert (
        compare_frames(
            "attribution",
            left,
            right,
            columns=list(columns),
            key_columns=("signal_date", "planned_ts_code", "actual_ts_code"),
            source_left="L",
            source_right="R",
        )
        == ()
    )
    changed = left.copy()
    changed["actual_ts_code"] = ["600000.SH", None]
    diffs = compare_frames(
        "attribution",
        left,
        changed,
        columns=list(columns),
        key_columns=("signal_date", "planned_ts_code", "actual_ts_code"),
        source_left="L",
        source_right="R",
    )
    paths = [item.field_path for item in diffs]
    assert "attribution.<key_order>" in paths
    assert "attribution.actual_ts_code" in paths


def test_non_empty_attribution_frame_is_compared_bitwise(tmp_path: Path) -> None:
    """两臂同写 1 行 attribution：完全相同 → 零差异；改 1 ULP → 必须失败。"""
    assembly = run_self_split_arms(
        scenario=SelfSplitScenario(),
        run_id="t4cmp4",
        base_dir=tmp_path,
        writer_config=_CONFIG,
        code_state={"git_commit": "syn", "git_dirty": False},
        data_state={"data_state_id": "syn_t4cmp4"},
    )
    frame = pd.DataFrame(
        {
            "wf_run_id": [assembly.batch_id],
            "split_index": [0],
            "model_version": [6501],
            "signal_date": ["20260105"],
            "ranking_date": ["20260105"],
            "execution_date": ["20260106"],
            "execution_stage": ["t1"],
            "tranche_idx": [0],
            "planned_ts_code": ["600000.SH"],
            "actual_ts_code": ["600000.SH"],
            "planned_rank": [1],
            "actual_rank": [1],
            "pred_score": [0.9],
            "target_weight": [0.04],
            "status": ["filled"],
            "reason": [None],
            "buy_price": [10.0],
            "signal_price": [10.0],
            "signal_to_buy_return": [0.0],
        }
    )
    for arm in (ARM_SINGLE, ARM_SPLIT):
        target = (
            assembly.writer.out_dir(arm, assembly.batch_id)
            / "folds"
            / "split00"
            / "attribution.parquet"
        )
        frame.to_parquet(target, index=False, compression="none")
        _resign(assembly.writer, arm, assembly.batch_id)

    report = compare_runs_trees(assembly.writer, assembly.batch_id)
    report.assert_clean()
    assert dict(report.file_status)["folds/split00/attribution.parquet"] == "present"
    assert "folds/split00/attribution.parquet" not in report.not_compared()

    split_path = (
        assembly.writer.out_dir(ARM_SPLIT, assembly.batch_id)
        / "folds"
        / "split00"
        / "attribution.parquet"
    )
    doctored = frame.copy()
    doctored["pred_score"] = [math.nextafter(0.9, math.inf)]
    doctored.to_parquet(split_path, index=False, compression="none")
    _resign(assembly.writer, ARM_SPLIT, assembly.batch_id)
    report = compare_runs_trees(assembly.writer, assembly.batch_id)
    assert not report.ok
    item = next(
        entry for entry in report.differences if entry.field_path == "attribution.pred_score"
    )
    assert item.row == 0 and item.left_bits != item.right_bits


def test_config_key_outside_summary_columns_is_rejected(tmp_path: Path) -> None:
    """A-R1-01 反例：batch_meta.config 出现 summary 列之外的键 ⇒ 覆盖门必须拒绝。"""
    assembly = run_self_split_arms(
        scenario=SelfSplitScenario(),
        run_id="t4cmp5",
        base_dir=tmp_path,
        writer_config=_CONFIG,
        code_state={"git_commit": "syn", "git_dirty": False},
        data_state={"data_state_id": "syn_t4cmp5"},
    )
    for arm in (ARM_SINGLE, ARM_SPLIT):
        path = assembly.writer.out_dir(arm, assembly.batch_id) / "batch_meta.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["config"]["phantom_key_not_in_summary"] = 1
        document["config_fingerprint"] = fingerprint(document["config"])
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        _resign(assembly.writer, arm, assembly.batch_id)
    with pytest.raises(ComparisonInputError, match="未被 summary 列覆盖"):
        compare_runs_trees(assembly.writer, assembly.batch_id)


def test_reordered_contract_columns_are_rejected(tmp_path: Path) -> None:
    """两臂同步重排**契约列序**仍是差异（列序锚定 runs_schema 契约顺序）。"""
    assembly = run_self_split_arms(
        scenario=SelfSplitScenario(),
        run_id="t4cmp8",
        base_dir=tmp_path,
        writer_config=_CONFIG,
        code_state={"git_commit": "syn", "git_dirty": False},
        data_state={"data_state_id": "syn_t4cmp8"},
    )
    for arm in (ARM_SINGLE, ARM_SPLIT):
        path = (
            assembly.writer.out_dir(arm, assembly.batch_id) / "folds" / "split00" / "trades.parquet"
        )
        frame = pd.read_parquet(path)
        columns = list(frame.columns)
        columns[0], columns[1] = columns[1], columns[0]
        frame.loc[:, columns].to_parquet(path, index=False, compression="none")
        _resign(assembly.writer, arm, assembly.batch_id)
    with pytest.raises(ComparisonInputError, match="契约列序被改写"):
        compare_runs_trees(assembly.writer, assembly.batch_id)


def test_missing_batch_meta_contract_key_is_rejected(tmp_path: Path) -> None:
    """两臂同步删除 batch_meta 契约 §2 必需键（config_fingerprint）不得读成"一致"。"""
    assembly = run_self_split_arms(
        scenario=SelfSplitScenario(),
        run_id="t4cmp7",
        base_dir=tmp_path,
        writer_config=_CONFIG,
        code_state={"git_commit": "syn", "git_dirty": False},
        data_state={"data_state_id": "syn_t4cmp7"},
    )
    for arm in (ARM_SINGLE, ARM_SPLIT):
        path = assembly.writer.out_dir(arm, assembly.batch_id) / "batch_meta.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document.pop("config_fingerprint")
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        _resign(assembly.writer, arm, assembly.batch_id)
    with pytest.raises(ComparisonInputError, match="缺少 runs 契约 §2 必需键"):
        compare_runs_trees(assembly.writer, assembly.batch_id)


def test_missing_contract_column_is_rejected(tmp_path: Path) -> None:
    """两臂同步缺契约必需列（test_start）不得被读成"一致"。"""
    assembly = run_self_split_arms(
        scenario=SelfSplitScenario(),
        run_id="t4cmp6",
        base_dir=tmp_path,
        writer_config=_CONFIG,
        code_state={"git_commit": "syn", "git_dirty": False},
        data_state={"data_state_id": "syn_t4cmp6"},
    )
    for arm in (ARM_SINGLE, ARM_SPLIT):
        path = assembly.writer.out_dir(arm, assembly.batch_id) / "summary.parquet"
        frame = pd.read_parquet(path).drop(columns=["test_start"])
        frame.to_parquet(path, index=False, compression="none")
        _resign(assembly.writer, arm, assembly.batch_id)
    with pytest.raises(ComparisonInputError, match="缺少 runs 契约必需列"):
        compare_runs_trees(assembly.writer, assembly.batch_id)


def test_full_assembly_comparison_is_clean(assembly) -> None:
    """同一装配：视图 + runs 全字段零差异（唯一 G4 判定入口）。"""
    report = compare_assemblies(assembly)
    report.assert_clean()
    assert report.excluded_fields == EXCLUDED_META_FIELDS
    assert all(left == right for _relative, left, right in report.file_sha256)
    assert report.compared_summary_columns
    assert "bt_sharpe" in report.compared_summary_columns
    assert report.fingerprint_key_count > 0


def _batch_meta_path(assembly, arm: str) -> Path:
    return assembly.writer.out_dir(arm, assembly.batch_id) / "batch_meta.json"


def test_missing_entire_fold_is_rejected(tmp_path: Path) -> None:
    """两臂同时删掉**整折**目录（再重签 generation）不得空遍历通过（Codex-01）。"""
    assembly = run_self_split_arms(
        scenario=SelfSplitScenario(),
        run_id="t4fold1",
        base_dir=tmp_path,
        writer_config=_CONFIG,
        code_state={"git_commit": "syn", "git_dirty": False},
        data_state={"data_state_id": "syn_t4fold1"},
    )
    for arm in (ARM_SINGLE, ARM_SPLIT):
        shutil.rmtree(assembly.writer.out_dir(arm, assembly.batch_id) / "folds")
        _resign(assembly.writer, arm, assembly.batch_id)
    with pytest.raises(ComparisonInputError, match="折集合与批次记录不符"):
        compare_runs_trees(assembly.writer, assembly.batch_id)


def test_extra_fold_directory_is_rejected(tmp_path: Path) -> None:
    """两臂同时多出契约折集合之外的折目录必须失败（缺/多都不放行）。"""
    assembly = run_self_split_arms(
        scenario=SelfSplitScenario(),
        run_id="t4fold2",
        base_dir=tmp_path,
        writer_config=_CONFIG,
        code_state={"git_commit": "syn", "git_dirty": False},
        data_state={"data_state_id": "syn_t4fold2"},
    )
    for arm in (ARM_SINGLE, ARM_SPLIT):
        out = assembly.writer.out_dir(arm, assembly.batch_id) / "folds"
        shutil.copytree(out / "split00", out / "split01")
        _resign(assembly.writer, arm, assembly.batch_id)
    with pytest.raises(ComparisonInputError, match="折集合与批次记录不符"):
        compare_runs_trees(assembly.writer, assembly.batch_id)


def test_missing_fold_required_file_is_rejected(tmp_path: Path) -> None:
    """整折内必需文件两侧同缺（_meta.json）必须失败，不得记为一致。"""
    assembly = run_self_split_arms(
        scenario=SelfSplitScenario(),
        run_id="t4fold3",
        base_dir=tmp_path,
        writer_config=_CONFIG,
        code_state={"git_commit": "syn", "git_dirty": False},
        data_state={"data_state_id": "syn_t4fold3"},
    )
    for arm in (ARM_SINGLE, ARM_SPLIT):
        path = assembly.writer.out_dir(arm, assembly.batch_id) / "folds/split00/_meta.json"
        path.unlink()
        _resign(assembly.writer, arm, assembly.batch_id)
    with pytest.raises(ArmIsolationError):
        compare_runs_trees(assembly.writer, assembly.batch_id)


def test_config_item_removed_on_both_arms_is_rejected(tmp_path: Path) -> None:
    """两臂同步删 config 键且重算指纹：独立冻结清单必须打红（Codex-02）。"""
    assembly = run_self_split_arms(
        scenario=SelfSplitScenario(),
        run_id="t4cfg1",
        base_dir=tmp_path,
        writer_config=_CONFIG,
        code_state={"git_commit": "syn", "git_dirty": False},
        data_state={"data_state_id": "syn_t4cfg1"},
    )
    frozen = json.loads(_batch_meta_path(assembly, ARM_SINGLE).read_text(encoding="utf-8"))[
        "config"
    ]
    frozen.pop("batch_id")
    for arm in (ARM_SINGLE, ARM_SPLIT):
        path = _batch_meta_path(assembly, arm)
        document = json.loads(path.read_text(encoding="utf-8"))
        document["config"].pop("bt_top_n")
        document["config_fingerprint"] = fingerprint(document["config"])
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        _resign(assembly.writer, arm, assembly.batch_id)
    with pytest.raises(ComparisonInputError, match="独立冻结的配置键清单未出现在"):
        compare_runs_trees(
            assembly.writer,
            assembly.batch_id,
            required_config_keys=sorted(frozen),
            expected_config=frozen,
            dynamic_config_keys=("batch_id",),
        )


def test_config_fingerprint_mismatch_is_rejected(tmp_path: Path) -> None:
    """config 被改写但指纹未重算：记录值与实际参数分裂必须拒绝（Codex-02）。"""
    assembly = run_self_split_arms(
        scenario=SelfSplitScenario(),
        run_id="t4cfg2",
        base_dir=tmp_path,
        writer_config=_CONFIG,
        code_state={"git_commit": "syn", "git_dirty": False},
        data_state={"data_state_id": "syn_t4cfg2"},
    )
    for arm in (ARM_SINGLE, ARM_SPLIT):
        path = _batch_meta_path(assembly, arm)
        document = json.loads(path.read_text(encoding="utf-8"))
        document["config"]["bt_top_n"] = 7
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        _resign(assembly.writer, arm, assembly.batch_id)
    with pytest.raises(ComparisonInputError, match="config_fingerprint 与 config 重算值不符"):
        compare_runs_trees(assembly.writer, assembly.batch_id)


def test_config_value_drift_from_frozen_list_is_rejected(tmp_path: Path) -> None:
    """两臂同步把 config 值改成清单外取值（并重算指纹）：值不符必须拒绝。"""
    assembly = run_self_split_arms(
        scenario=SelfSplitScenario(),
        run_id="t4cfg3",
        base_dir=tmp_path,
        writer_config=_CONFIG,
        code_state={"git_commit": "syn", "git_dirty": False},
        data_state={"data_state_id": "syn_t4cfg3"},
    )
    frozen = json.loads(_batch_meta_path(assembly, ARM_SINGLE).read_text(encoding="utf-8"))[
        "config"
    ]
    frozen.pop("batch_id")
    for arm in (ARM_SINGLE, ARM_SPLIT):
        path = _batch_meta_path(assembly, arm)
        document = json.loads(path.read_text(encoding="utf-8"))
        document["config"]["bt_top_n"] = 7
        document["config_fingerprint"] = fingerprint(document["config"])
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        _resign(assembly.writer, arm, assembly.batch_id)
    with pytest.raises(ComparisonInputError, match="与冻结清单不符"):
        compare_runs_trees(
            assembly.writer,
            assembly.batch_id,
            required_config_keys=sorted(frozen),
            expected_config=frozen,
            dynamic_config_keys=("batch_id",),
        )


def test_normalized_capture_files_match_reported_sha(assembly, tmp_path: Path) -> None:
    """落盘的规范化字节必须与报告 SHA 一致，且能逐条定位（Codex-04）。"""
    capture = tmp_path / "normalized"
    report = compare_runs_trees(assembly.writer, assembly.batch_id, capture_dir=capture)
    assert report.normalized_files
    assert {item[0] for item in report.normalized_files} == {item[0] for item in report.file_sha256}
    expected = {item[0]: (item[1], item[2]) for item in report.file_sha256}
    for relative, left_rel, right_rel in report.normalized_files:
        left_file = tmp_path / left_rel
        right_file = tmp_path / right_rel
        assert left_file.is_file() and right_file.is_file()
        assert (
            hashlib.sha256(left_file.read_bytes()).hexdigest(),
            hashlib.sha256(right_file.read_bytes()).hexdigest(),
        ) == expected[relative]
