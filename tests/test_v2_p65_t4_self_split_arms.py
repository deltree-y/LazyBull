# -*- coding: utf-8 -*-
"""P6.5 T4 自拆双臂装配：黄金值、两臂逐位一致、隔离/自我比较负例、新键继续拒绝。

覆盖：
- §2.2 黄金值逐值（D1 成交 400@10 / 费用 5 / 现金 95,995 / 总值 99,995 / NAV 0.99995；
  D2 A 48,000/50,200、B 47,995/50,195、合并 100,395、NAV 1.00395）且由执行链产出；
- 两臂物理与 runs 全字段逐位一致（比较器零差异、SHa 相等、排除字段仅 created_at/arms）；
- 两臂是**不同构造**（子袖数 1 vs 2、信号指纹不同、generation 身份不同、根为兄弟目录）；
- 负例：``out_dir`` 写前存在、A 产物复制给 B、篡改根映射、复用 generation、
  单侧缺失、自拆多成交、费用二次计算，在写入层与比较器层均失败；
- T3 收口 §6 P3 交接 ``T3-B-R3-03``：执行期失败后**同实例换新键**继续必须被拒。

全部为合成输入：不写 data/runs，不读真实配置/行情。
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from src.lazybull.universe.base import BasicUniverse
from src.lazybull.v2.common.cost import CostModel
from src.lazybull.v2.core.accounting.nav import calculate_nav
from src.lazybull.v2.core.accounting.physical_lot_view import PhysicalLotView
from src.lazybull.v2.core.accounting.virtual_ledger import VirtualLedger
from src.lazybull.v2.core.decision.virtual_merger import MultiSleevePortfolioMerger
from src.lazybull.v2.core.signal.base import EqualWeightSignal
from src.lazybull.v2.evidence.arm_comparator import (
    EXCLUDED_META_FIELDS,
    ComparisonInputError,
    compare_arm_views,
    compare_assemblies,
)
from src.lazybull.v2.hosts.order_chain import OrderChainError, OrderChainInvariantError
from src.lazybull.v2.hosts.virtual_native_engine import ChainFailedTerminatedError
from src.lazybull.v2.hosts.self_split_arms import (
    ARM_SINGLE,
    ARM_SPLIT,
    SPLIT_SLEEVE_IDS,
    SelfSplitArmError,
    SelfSplitScenario,
    assert_arms_are_distinct,
    run_self_split_arms,
)
from src.lazybull.v2.hosts.virtual_arm_runs import ArmIsolationError, DualArmRunsWriter
from src.lazybull.v2.hosts.virtual_native_engine import VirtualNativeMergeBacktestEngineML

_X = "600000.SH"
_D0 = pd.Timestamp("2026-01-05")
_D1 = pd.Timestamp("2026-01-06")
_D2 = pd.Timestamp("2026-01-07")
_CONFIG = {
    "bt_top_n": 1,
    "label_column": "syn",
    "stock_domain": "main",
    "wf_start_date": "20260105",
    "wf_end_date": "20260107",
}
_CODE_STATE = {"git_commit": "syn", "git_dirty": False}
_DATA_STATE = {"data_state_id": "syn_t4", "sources": {"daily": "2026-01-07"}}


def _run(
    base_dir: Path, *, run_id: str = "t4arms", scenario: SelfSplitScenario | None = None
) -> object:
    return run_self_split_arms(
        scenario=scenario or SelfSplitScenario(),
        run_id=run_id,
        base_dir=base_dir,
        writer_config=_CONFIG,
        code_state=_CODE_STATE,
        data_state=_DATA_STATE,
    )


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> tuple[object, Path]:
    """一次性跑出两臂模板（逐用例复制树，避免反复跑全链）。"""
    base = tmp_path_factory.mktemp("t4arms_template")
    return _run(base), base


def _copy_assembly(assembly, source_base: Path, target_base: Path) -> object:
    """把两臂树复制到新根并重建 writer（用于篡改/缺失/自我比较负例）。

    注意：generation manifest 记录的是**绝对根**，跨根复制后必须重写 manifest 才能
    继续用同一 ``ArmView``；负例因此改为"在原根上就地篡改"，本函数只用于
    "同根不同 run_id"的写前存在断言。
    """
    shutil.copytree(source_base, target_base, dirs_exist_ok=True)
    writer = DualArmRunsWriter(run_id=assembly.run_id, base_dir=target_base, config=dict(_CONFIG))
    return replace(assembly, writer=writer)


# ────────────────────────── 黄金值（由执行链产出） ──────────────────────────


def test_self_split_golden_values_come_from_the_execution_chain(template) -> None:
    assembly, _base = template
    assert assembly.daily_physical_values == (
        (str(_D0), 100_000.0),
        (str(_D1), 99_995.0),
        (str(_D2), 100_395.0),
    )
    d1_nav = assembly.daily_physical_values[1][1] / 100_000.0
    assert d1_nav == 0.99995

    single = assembly.single
    split = assembly.split
    for arm in (single, split):
        assert arm.arm in (ARM_SINGLE, ARM_SPLIT)
        assert arm.physical_initial_capital == 100_000.0
        assert arm.physical_cash == 95_995.0
        assert arm.physical_positions == ((_X, 400),)
        assert arm.physical_total_value == 100_395.0
        assert arm.merged_virtual_cash == 95_995.0
        assert arm.merged_total_value == 100_395.0
        assert arm.merged_nav == 1.00395
        assert arm.virtual_source_lots == arm.physical_source_lots
        assert len(arm.fills) == 1
        fill = arm.fills[0]
        assert (fill.ts_code, fill.order_target, fill.requested_shares, fill.shares) == (
            _X,
            400,
            400,
            400,
        )
        assert fill.price == 10.0
        assert fill.fee == 5.0
        assert fill.cash_before == 100_000.0
        assert fill.cash_after == 95_995.0
        assert fill.fill_key == "20260106|buy|600000.SH|0"
        assert fill.source_lot_id == f"PHY:{fill.fill_key}"
        assert arm.settlement_dates == ("20260106", "20260107")
        # 成交日志由执行器落账（不是比较器前手工指定）
        assert len(arm.trades) == 1
        assert arm.trades[0]["shares"] == 400

    # 自拆臂逐袖（单个子袖不要求等于单袖，合计必须相等）
    assert single.virtual_cash_by_sleeve == (("solo", 95_995.0),)
    assert single.virtual_total_value_by_sleeve == (("solo", 100_395.0),)
    assert split.virtual_cash_by_sleeve == (("A", 48_000.0), ("B", 47_995.0))
    assert split.virtual_total_value_by_sleeve == (("A", 50_200.0), ("B", 50_195.0))
    assert calculate_nav(50_200.0, 100_000.0) == 0.502
    assert calculate_nav(50_195.0, 100_000.0) == 0.50195
    assert calculate_nav(50_200.0, 100_000.0) + calculate_nav(50_195.0, 100_000.0) == (
        1.0039500000000001
    )

    # D1（成交日收盘）逐袖黄金值：现金/股数/总值/贡献 NAV 四列全覆盖
    assert single.trade_day_sleeve_facts == (("solo", 95_995.0, 400, 99_995.0, 0.99995),)
    assert split.trade_day_sleeve_facts == (
        ("A", 48_000.0, 200, 50_000.0, 0.5),
        ("B", 47_995.0, 200, 49_995.0, 0.49995),
    )
    d1_contrib_sum = split.trade_day_sleeve_facts[0][4] + split.trade_day_sleeve_facts[1][4]
    assert d1_contrib_sum == 0.99995
    assert d1_contrib_sum == single.trade_day_sleeve_facts[0][4]


def test_arms_are_distinct_constructions_and_roots_are_siblings(template) -> None:
    assembly, _base = template
    assert_arms_are_distinct(assembly)
    assert assembly.single.sleeve_ids == ("solo",)
    assert assembly.split.sleeve_ids == tuple(sorted(SPLIT_SLEEVE_IDS))
    single_gen = assembly.single.generation
    split_gen = assembly.split.generation
    assert single_gen.generation_id != split_gen.generation_id
    assert single_gen.signal_fingerprint != split_gen.signal_fingerprint
    assert single_gen.config_fingerprint == split_gen.config_fingerprint
    roots = assembly.writer.roots
    assert roots[ARM_SINGLE].name == roots[ARM_SPLIT].name == "runs_root"
    assert roots[ARM_SINGLE].parent.parent == roots[ARM_SPLIT].parent.parent
    assert roots[ARM_SINGLE].parent.name == "single"
    assert roots[ARM_SPLIT].parent.name == "split"


def test_two_arms_are_bitwise_identical_under_frozen_exclusion_table(template) -> None:
    assembly, _base = template
    report = compare_assemblies(assembly)
    report.assert_clean()
    assert report.ok is True
    assert report.differences == ()
    assert report.excluded_fields == EXCLUDED_META_FIELDS == ("created_at", "arms")
    for relative, left_sha, right_sha in report.file_sha256:
        assert left_sha == right_sha, relative
    status = dict(report.file_status)
    assert status["batch_meta.json"] == "present"
    assert status["summary.parquet"] == "present"
    assert status["folds/split00/trades.parquet"] == "present"
    # 同源缺失与契约可选缺失只登记 not_compared（不得写成逐位一致）
    assert status["folds/split00/attribution.parquet"] == "not_compared/source_absent"
    assert status["folds/split00/topk_detail.parquet"] == ("not_compared/contract_optional_missing")
    assert report.not_compared() == (
        "folds/split00/attribution.parquet",
        "folds/split00/topk_detail.parquet",
    )
    assert report.fingerprint_key_count >= len(_CONFIG)


def test_batch_meta_excludes_only_time_and_arm_label(template) -> None:
    """排除投影后 batch_meta 逐字节相同；未排除字段（config/fingerprint/血缘）必须全等。"""
    assembly, _base = template
    metas = {}
    for arm in (ARM_SINGLE, ARM_SPLIT):
        payload = json.loads(
            (assembly.writer.out_dir(arm, assembly.batch_id) / "batch_meta.json").read_text(
                encoding="utf-8"
            )
        )
        metas[arm] = payload
    assert metas[ARM_SINGLE]["arms"] == [ARM_SINGLE]
    assert metas[ARM_SPLIT]["arms"] == [ARM_SPLIT]
    for field in ("config", "config_fingerprint", "code_state", "data_state", "source"):
        assert metas[ARM_SINGLE][field] == metas[ARM_SPLIT][field]
    assert metas[ARM_SINGLE]["config_fingerprint"] == metas[ARM_SPLIT]["config_fingerprint"]


# ────────────────────────── 写入层负例 ──────────────────────────


def test_pre_existing_out_dir_fails_before_writing(tmp_path: Path) -> None:
    """``out_dir`` 写前存在必须失败（零覆盖、零副作用）。"""
    base = tmp_path / "arms"
    target = base / "t4pre" / "single" / "runs_root" / "t4pre-b0"
    target.mkdir(parents=True)
    with pytest.raises(ArmIsolationError, match="写前 out_dir 已存在"):
        _run(base, run_id="t4pre")
    assert list(target.iterdir()) == []
    assert not (base / "t4pre" / "split" / "runs_root" / "t4pre-b0").exists()


def test_reusing_run_id_fails_on_second_run(tmp_path: Path) -> None:
    """同一 ``run_id`` 二次装配必须被写前存在断言拒绝（根目录保留首次产物）。"""
    base = tmp_path / "second"
    first = _run(base, run_id="t4reuse")
    with pytest.raises(ArmIsolationError, match="写前 out_dir 已存在"):
        run_self_split_arms(
            scenario=SelfSplitScenario(),
            run_id="t4reuse",
            base_dir=base,
            writer_config=_CONFIG,
            code_state=_CODE_STATE,
            data_state=_DATA_STATE,
        )
    assert first.writer.out_dir(ARM_SPLIT, first.batch_id).is_dir()


# ────────────────────────── 比较器层负例（自我比较防护） ──────────────────────────


def test_copying_arm_a_artifacts_to_arm_b_fails_at_comparator(tmp_path: Path) -> None:
    """A 产物整体复制给 B：比较器在身份绑定阶段即失败（不是"文件集一致"假通过）。"""
    assembly = _run(tmp_path, run_id="t4copied")
    source = Path(assembly.single.generation.out_dir)
    target = Path(assembly.writer.out_dir(ARM_SPLIT, assembly.batch_id))
    shutil.copytree(source, target, dirs_exist_ok=True)
    with pytest.raises(ArmIsolationError, match="batch_meta.arms"):
        compare_assemblies(assembly)


def test_tampered_root_mapping_fails_at_comparator(tmp_path: Path) -> None:
    assembly = _run(tmp_path, run_id="t4tamper")
    path = assembly.writer.generation_path(ARM_SINGLE)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["runs_root"] = str(tmp_path / "elsewhere")
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ArmIsolationError, match="根映射被篡改"):
        compare_assemblies(assembly)


def test_reused_generation_fails_at_comparator(tmp_path: Path) -> None:
    assembly = _run(tmp_path, run_id="t4reuse_gen")
    shutil.copyfile(
        assembly.writer.generation_path(ARM_SINGLE),
        assembly.writer.generation_path(ARM_SPLIT),
    )
    with pytest.raises(ArmIsolationError, match="身份不符"):
        compare_assemblies(assembly)


def test_one_sided_missing_artifact_fails_at_comparator(tmp_path: Path) -> None:
    """单侧缺文件：即使把 generation 改写为与当前树自洽，缺失语义仍必须失败。"""
    assembly = _run(tmp_path, run_id="t4missing")
    (
        Path(assembly.single.generation.out_dir) / "folds" / "split00" / "holdings_snapshot.parquet"
    ).unlink()
    _resign_generation(assembly.writer, ARM_SINGLE, assembly.batch_id)
    with pytest.raises(ArmIsolationError, match="单侧缺失"):
        compare_assemblies(assembly)


def _resign_generation(writer: DualArmRunsWriter, arm: str, batch_id: str) -> None:
    """把 generation 重签为与当前树自洽（模拟"manifest 与树一致但产物缺失"）。"""
    from src.lazybull.v2.hosts.virtual_arm_runs import file_hashes, generation_identity

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


def test_extra_physical_fill_is_detected(template) -> None:
    """自拆多产生一笔物理成交必须被比较器捕获（含差异路径与来源）。"""
    assembly, _base = template
    mutated = replace(
        assembly, split=replace(assembly.split, fills=assembly.split.fills + assembly.split.fills)
    )
    diffs = compare_arm_views(mutated.single, mutated.split)
    assert diffs
    assert diffs[0].field_path == "physical.fills.<count>"
    assert diffs[0].left == 1 and diffs[0].right == 2
    report = compare_assemblies_no_runs(mutated)
    assert report.ok is False


def compare_assemblies_no_runs(assembly) -> object:
    """只比较内存视图（不触碰 runs 树；用于变异用例）。"""
    from src.lazybull.v2.evidence.arm_comparator import ArmDiffReport

    return ArmDiffReport(
        differences=compare_arm_views(assembly.single, assembly.split),
        file_status=(),
        file_sha256=(),
        excluded_fields=EXCLUDED_META_FIELDS,
        compared_summary_columns=(),
        fingerprint_key_count=0,
    )


def test_double_charged_fee_is_detected(template) -> None:
    """费用二次计算（费用/现金同时被二次扣减）必须被比较器捕获。"""
    assembly, _base = template
    fill = assembly.single.fills[0]
    doubled = replace(
        assembly,
        single=replace(
            assembly.single,
            fills=(
                replace(
                    fill,
                    fee=fill.fee * 2.0,
                    cash_after=fill.cash_after - fill.fee,
                ),
            ),
            physical_cash=assembly.single.physical_cash - fill.fee,
        ),
    )
    diffs = compare_arm_views(doubled.single, doubled.split)
    paths = {item.field_path for item in diffs}
    assert "physical.fills[0].fee" in paths
    assert "physical.cash" in paths
    fee_diff = next(item for item in diffs if item.field_path == "physical.fills[0].fee")
    assert fee_diff.left == 10.0 and fee_diff.right == 5.0
    assert fee_diff.left_bits != fee_diff.right_bits


# ────────────────────────── T3-B-R3-03 交接 ──────────────────────────


_X2 = "600519.SH"


def _add_quote(host: VirtualNativeMergeBacktestEngineML, code: str, price: float) -> None:
    """补齐当日行情（两股同日有效价；用于排除"无价即拒"的伪拒绝）。"""
    rows = [
        {
            "ts_code": item,
            "trade_date": _D1,
            "close": price if item == code else 10.0,
            "close_adj": price if item == code else 10.0,
            "open": price if item == code else 10.0,
            "open_adj": price if item == code else 10.0,
        }
        for item in (_X, code)
    ]
    host._prepare_price_index(pd.DataFrame(rows))


def _host_for_new_key_retry() -> VirtualNativeMergeBacktestEngineML:
    cost_model = CostModel(
        commission_rate=0.0003, min_commission=5.0, stamp_tax=0.0005, slippage=0.0
    )
    ledger = VirtualLedger(cost_model=cost_model)
    ledger.open_account("A", 5_000.0)
    ledger.open_account("B", 5_000.0)
    host = VirtualNativeMergeBacktestEngineML(
        features_by_date={},
        universe=BasicUniverse(
            pd.DataFrame({"ts_code": [_X], "name": ["甲"], "industry": ["I"]}), verbose=False
        ),
        signal=EqualWeightSignal(top_n=1),
        initial_capital=10_000.0,
        cost_model=cost_model,
        verbose=False,
        portfolio_merger=MultiSleevePortfolioMerger(("A", "B"), {"A": 0.5, "B": 0.5}),
        virtual_ledger=ledger,
        physical_lot_view=PhysicalLotView(),
        run_id="T4R3",
    )
    host.current_capital = 10_000.0
    host.positions = {}
    host._emit_immediate_log = lambda *args, **kwargs: None
    host._prepare_price_index(
        pd.DataFrame(
            {
                "ts_code": [_X],
                "trade_date": [_D1],
                "close": [10.0],
                "close_adj": [10.0],
                "open": [10.0],
                "open_adj": [10.0],
            }
        )
    )
    return host


def test_mid_execution_failure_blocks_new_key_on_same_instance() -> None:
    """T3-B-R3-03：执行期失败后同实例**换新键**继续必须被物理基数门拒绝。

    边界（如实登记）：**新实例**从破缺前信封重放（含物理幂等水位与来源视图恢复）
    属 T5「多 lot 状态信封恢复」范围，T4 不实现也不冒充；本用例只证明"同一实例不得
    换个键继续"，与 T5 的恢复语义不冲突。
    """
    host = _host_for_new_key_retry()
    real_buy = host._buy_stock_direct

    def boom(*args, **kwargs):
        real_buy(*args, **kwargs)
        raise RuntimeError("执行器中途抛出（物理已部分成交）")

    host._buy_stock_direct = boom  # type: ignore[assignment]
    with pytest.raises(OrderChainInvariantError, match="执行链中途异常"):
        host.execute_order_chain(_D1, {"A": {_X: 100}, "B": {_X: 100}}, batch_id="F1")
    assert host.positions[_X]["shares"] == 200  # 物理合并目标 200 已成交、虚拟侧未提交
    cash_after_failure = host.current_capital
    trades_after_failure = list(host.trades)

    host._buy_stock_direct = real_buy  # type: ignore[assignment]
    # 执行后失败已是实例终态（外部评审 P2）：换新键在物理基数门之前即被终态守卫拒绝
    with pytest.raises(ChainFailedTerminatedError):
        host.execute_order_chain(_D1, {"A": {_X: 200}, "B": {_X: 200}}, batch_id="F2")
    assert host.current_capital == cash_after_failure
    assert host.trades == trades_after_failure
    assert host.positions[_X]["shares"] == 200


def test_assembly_rejects_duplicate_signal_fingerprint(tmp_path: Path) -> None:
    """自我比较防护（装配层）：两臂信号指纹相同必须拒绝。"""
    assembly = _run(tmp_path / "fp", run_id="t4fp")
    forged = replace(
        assembly,
        single=replace(
            assembly.single,
            generation=replace(
                assembly.single.generation,
                signal_fingerprint=assembly.split.generation.signal_fingerprint,
            ),
        ),
    )
    with pytest.raises(SelfSplitArmError, match="信号指纹相同"):
        assert_arms_are_distinct(forged)


def test_comparator_rejects_forged_single_view_as_split(tmp_path: Path) -> None:
    """比较器层防护：把 single 视图直接当 split 传入，不得给出"零差异"伪证据。"""
    assembly = _run(tmp_path / "forged", run_id="t4forged")
    forged_view = replace(
        assembly, split=replace(assembly.single, generation=assembly.split.generation)
    )
    with pytest.raises(SelfSplitArmError, match="臂标签相同"):
        compare_assemblies(forged_view)
    # 即使绕过装配层（直接给视图），比较器自己的身份门也必须拒绝
    with pytest.raises(ComparisonInputError, match="臂标签相同"):
        compare_arm_views(assembly.single, replace(assembly.single))
    with pytest.raises(ComparisonInputError, match="同一视图对象"):
        compare_arm_views(assembly.single, assembly.single)
    with pytest.raises(ComparisonInputError, match="两臂必须为不同构造"):
        compare_arm_views(
            assembly.single, replace(assembly.split, sleeve_ids=assembly.single.sleeve_ids)
        )


def test_split_side_pre_existing_out_dir_leaves_no_single_artifacts(tmp_path: Path) -> None:
    """T4 复审 B-R1-02：第二臂写前不可写时，第一臂不得留下任何产物（零副作用）。"""
    base = tmp_path / "arms"
    batch = "t4pre2-b0"
    (base / "t4pre2" / "split" / "runs_root" / batch).mkdir(parents=True)
    with pytest.raises(ArmIsolationError, match="写前 out_dir 已存在"):
        _run(base, run_id="t4pre2")
    assert not (base / "t4pre2" / "single").exists()
    assert not (base / "t4pre2" / "single" / "generation.json").exists()


def test_post_run_gate_failure_leaves_no_artifacts(tmp_path: Path, monkeypatch) -> None:
    """写后断言（G4 位模式门）失败也不得留下任何产物：三条断言已前置到写盘之前。"""
    from src.lazybull.v2.core.accounting.g4_nav import G4NavMismatchError
    from src.lazybull.v2.hosts import self_split_arms as arms_module

    def boom(*args, **kwargs):
        raise G4NavMismatchError("模拟 G4 位模式不一致")

    monkeypatch.setattr(arms_module, "assert_g4_nav_bitwise_equal", boom)
    with pytest.raises(G4NavMismatchError, match="模拟 G4 位模式不一致"):
        _run(tmp_path, run_id="t4post")
    assert not (tmp_path / "t4post").exists()


def test_preexisting_run_root_is_rolled_back_to_snapshot(tmp_path: Path, monkeypatch) -> None:
    """预有运行根 + 第二臂写出失败：既有文件保留，本次新增产物全部撤销（Codex-03）。"""
    base = tmp_path / "arms"
    root = base / "t4rb"
    root.mkdir(parents=True)
    marker = root / "preserve.txt"
    marker.write_text("keep", encoding="utf-8")
    real_write = DualArmRunsWriter.write_arm
    calls = {"count": 0}

    def flaky(self, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("模拟第二臂写出失败")
        return real_write(self, *args, **kwargs)

    monkeypatch.setattr(DualArmRunsWriter, "write_arm", flaky)
    with pytest.raises(OSError, match="模拟第二臂写出失败"):
        _run(base, run_id="t4rb")
    assert marker.read_text(encoding="utf-8") == "keep"
    assert sorted(item.name for item in root.iterdir()) == ["preserve.txt"]


def test_preexisting_empty_run_root_is_preserved(tmp_path: Path, monkeypatch) -> None:
    """预有空运行根：失败后根仍在（不得删除既有目录），且无本次产物残留。"""
    base = tmp_path / "arms"
    root = base / "t4rb2"
    root.mkdir(parents=True)
    real_write = DualArmRunsWriter.write_arm
    calls = {"count": 0}

    def flaky(self, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("模拟第二臂写出失败")
        return real_write(self, *args, **kwargs)

    monkeypatch.setattr(DualArmRunsWriter, "write_arm", flaky)
    with pytest.raises(OSError):
        _run(base, run_id="t4rb2")
    assert root.is_dir()
    assert list(root.iterdir()) == []


def test_new_host_continuing_broken_state_is_rejected() -> None:
    """Codex-07：物理已成交、虚拟未提交后，**新宿主接续同一破缺态**必须被拒绝。

    三类输入：原股票（物理基数门）、另一股票（合成边界无价即拒）、空目标
    （预演现金守恒）。拒绝均发生在物理执行之前（零副作用）。T5 的"从破缺前信封
    合法恢复后重放"不在本条范围。
    """
    host = _host_for_new_key_retry()
    real_buy = host._buy_stock_direct

    def boom(*args, **kwargs):
        real_buy(*args, **kwargs)
        raise RuntimeError("执行器中途抛出（物理已成交、虚拟未提交）")

    host._buy_stock_direct = boom  # type: ignore[assignment]
    with pytest.raises(OrderChainInvariantError, match="执行链中途异常"):
        host.execute_order_chain(_D1, {"A": {_X: 100}, "B": {_X: 100}}, batch_id="F1")
    shares = host.positions[_X]["shares"]
    cash = host.current_capital
    assert shares == 200

    fresh = _host_for_new_key_retry()
    fresh.positions = {_X: {"shares": shares, "cost": 10.0, "open_date": _D1}}
    fresh.current_capital = cash
    from src.lazybull.v2.core.accounting.physical_lot_view import PhysicalLot

    fresh.physical_lot_view = PhysicalLotView(
        (PhysicalLot(ts_code=_X, source_lot_id="L1", shares=shares),)
    )
    with pytest.raises(OrderChainError, match="虚拟袖合计当前持仓"):
        fresh.execute_order_chain(_D1, {"A": {_X: 100}, "B": {_X: 100}}, batch_id="N1")

    other = _host_for_new_key_retry()
    other.positions = {_X: {"shares": shares, "cost": 10.0, "open_date": _D1}}
    other.current_capital = cash
    other.physical_lot_view = PhysicalLotView(
        (PhysicalLot(ts_code=_X, source_lot_id="L1", shares=shares),)
    )
    _add_quote(other, _X2, 10.0)  # 两股同日有效行情：排除"无价提前拦截"的伪拒绝
    with pytest.raises(OrderChainError, match="预演结算拒绝"):
        other.execute_order_chain(_D1, {"A": {_X2: 100}, "B": {_X2: 100}}, batch_id="N2")
    # 零副作用：无新增成交、持仓与现金保持破缺态原值
    assert other.trades == []
    assert set(other.positions) == {_X}
    assert other.current_capital == cash

    # 健康对照：同一输入在**未破缺**宿主上可以正常买入另一股 200 股
    healthy = _host_for_new_key_retry()
    _add_quote(healthy, _X2, 10.0)
    result = healthy.execute_order_chain(_D1, {"A": {_X2: 100}, "B": {_X2: 100}}, batch_id="H1")
    assert result.applied is True
    assert healthy.positions[_X2]["shares"] == 200

    empty = _host_for_new_key_retry()
    empty.positions = {_X: {"shares": shares, "cost": 10.0, "open_date": _D1}}
    empty.current_capital = cash
    empty.physical_lot_view = PhysicalLotView(
        (PhysicalLot(ts_code=_X, source_lot_id="L1", shares=shares),)
    )
    with pytest.raises(OrderChainError, match="预演结算拒绝"):
        empty.execute_order_chain(_D1, {"A": {}, "B": {}}, batch_id="N3")


def test_multi_stock_scenario_is_rejected(tmp_path: Path) -> None:
    """多股票配额分摊未定义：装配层必须显式拒绝（不得把单股票股数复制到每只股票）。"""
    scenario = SelfSplitScenario(
        prices=(
            (_D0, {_X: 10.0, "000001.SZ": 10.0}),
            (_D1, {_X: 10.0, "000001.SZ": 10.0}),
            (_D2, {_X: 11.0, "000001.SZ": 11.0}),
        ),
        signals=((_X, 1.0), ("000001.SZ", 1.0)),
    )
    with pytest.raises(SelfSplitArmError, match="只支持单股票组合配额"):
        _run(tmp_path / "multi", run_id="t4multi", scenario=scenario)
    assert not (tmp_path / "multi" / "t4multi").exists()
