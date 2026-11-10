# -*- coding: utf-8 -*-
"""P6.5 T3 runs 双臂物理隔离与 generation 绑定（机制与守卫；比较归 T4）。

覆盖：两臂根为不同兄弟目录且不嵌套；写前 ``out_dir`` 不存在（不依赖
``exist_ok``）；generation 记录臂标签/解析根/out_dir/输入哈希/写前不存在断言/
写后文件哈希；共根、嵌套、写前存在、A 产物复制给 B、篡改根映射、复用 generation、
单侧缺文件、同源缺失登记 ``not_compared`` 全部可打红。

全部为合成结果帧：不写 ``data/runs``，不读真实配置/行情，不做经济字段比较。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.lazybull.v2.hosts.virtual_arm_runs import (
    ARM_SINGLE,
    ARM_SPLIT,
    ArmIsolationError,
    DualArmRunsWriter,
    assert_isolated_roots,
    file_hashes,
    generation_identity,
)

_CONFIG = {
    "bt_top_n": 20,
    "label_column": "neu_y_ret_20",
    "enable_fundamental": True,
    "stock_domain": "main",
    "wf_start_date": "20240102",
    "wf_end_date": "20241231",
}
_DATA_STATE = {"data_state_id": "syn_t3", "sources": {"daily": "2024-07-02"}}
_CODE_STATE = {"git_commit": "syn", "git_dirty": False}
_SIGNALS_SINGLE = {"signals": {"600000.SH": 1.0}}
_SIGNALS_SPLIT = {"signals": {"600000.SH": 0.5}, "sleeves": {"A": 0.5, "B": 0.5}}


def _nav(dates: list[str], values: list[float]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp(value) for value in dates],
            "portfolio_value": values,
            "capital": [value * 0.4 for value in values],
            "market_value": [value * 0.6 for value in values],
        }
    )
    frame["nav"] = frame["portfolio_value"] / 1_000_000.0
    frame["return"] = frame["nav"] - 1.0
    return frame


def _fold(
    split: int = 0,
    *,
    dates: tuple[str, str] = ("2024-01-02", "2024-01-03"),
    with_attribution: bool = True,
    with_snapshot: bool = True,
) -> dict:
    nav = _nav(list(dates), [1_000_000.0, 1_010_000.0])
    trades = pd.DataFrame(
        [
            {
                "date": pd.Timestamp(dates[0]),
                "signal_date": pd.Timestamp(dates[0]) - pd.Timedelta(days=1),
                "stock": "600000.SH",
                "action": "buy",
                "price": 10.0,
                "shares": 100,
                "amount": 1000.0,
                "cost": 5.0,
                "buy_date": pd.Timestamp(dates[0]),
                "buy_price": 10.0,
                "buy_type": "rebalance",
                "buy_reason": "signal",
            }
        ]
    )
    attribution = (
        pd.DataFrame(
            [
                {
                    "signal_date": pd.Timestamp(dates[0]) - pd.Timedelta(days=1),
                    "ranking_date": pd.Timestamp(dates[0]) - pd.Timedelta(days=1),
                    "execution_date": pd.Timestamp(dates[0]),
                    "execution_stage": "t1",
                    "tranche_idx": 0,
                    "planned_stock": "600000.SH",
                    "actual_stock": "600000.SH",
                    "planned_rank": 1,
                    "actual_rank": 1,
                    "pred_score": 0.9,
                    "target_weight": 0.05,
                    "status": "filled",
                    "reason": None,
                    "buy_price": 10.0,
                    "signal_price": 9.9,
                    "signal_to_buy_return": 0.01,
                }
            ]
        )
        if with_attribution
        else pd.DataFrame()
    )
    snapshot = (
        pd.DataFrame(
            [
                {
                    "date": pd.Timestamp(dates[0]),
                    "ts_code": "600000.SH",
                    "shares": 100,
                    "market_value": 1000.0,
                    "weight": 0.001,
                    "portfolio_value": 1_000_000.0,
                    "buy_date": pd.Timestamp(dates[0]),
                    "signal_date": pd.Timestamp(dates[0]) - pd.Timedelta(days=1),
                    "holding_days": 0,
                    "planned_exit_date": None,
                    "remaining_intervals": None,
                }
            ]
        )
        if with_snapshot
        else pd.DataFrame()
    )
    return {
        "split_index": split,
        "model_version": 24008,
        "train_start": "20180101",
        "train_end": "20231229",
        "test_start": "20240102",
        "test_end": "20240628",
        "metrics": {
            "bt_total_return": 0.01,
            "bt_annual_return": 0.02,
            "bt_max_drawdown": -0.01,
            "bt_volatility": 0.1,
            "bt_sharpe": 1.0,
            "bt_calmar": 2.0,
            "bt_trading_days": 2,
            "bt_start": "20240102",
            "bt_end": "20240702",
            "bt_top_n": 20,
        },
        "nav_curve": nav,
        "trades": trades,
        "attribution": attribution,
        "holdings_snapshot": snapshot,
    }


def _writer(tmp_path: Path, *, run_id: str = "t3run") -> DualArmRunsWriter:
    return DualArmRunsWriter(run_id=run_id, base_dir=tmp_path, config=_CONFIG)


def _write_both(
    tmp_path: Path,
    *,
    batch_id: str = "b1",
    with_attribution: bool = True,
    with_snapshot: bool = True,
) -> tuple[DualArmRunsWriter, object, object]:
    writer = _writer(tmp_path)
    results = [_fold(with_attribution=with_attribution, with_snapshot=with_snapshot)]
    single = writer.write_arm(
        arm=ARM_SINGLE,
        results=results,
        batch_id=batch_id,
        code_state=_CODE_STATE,
        data_state=_DATA_STATE,
        signals=_SIGNALS_SINGLE,
        created_at="2026-11-09T21:00:00",
    )
    split = writer.write_arm(
        arm=ARM_SPLIT,
        results=[_fold(with_attribution=with_attribution, with_snapshot=with_snapshot)],
        batch_id=batch_id,
        code_state=_CODE_STATE,
        data_state=_DATA_STATE,
        signals=_SIGNALS_SPLIT,
        created_at="2026-11-09T21:00:01",
    )
    return writer, single, split


# ────────────────────────── 隔离与身份绑定 ──────────────────────────


def test_roots_are_distinct_siblings_and_generation_is_bound(tmp_path: Path) -> None:
    writer, single, split = _write_both(tmp_path)

    roots = writer.roots
    assert roots[ARM_SINGLE] != roots[ARM_SPLIT]
    assert roots[ARM_SINGLE].parent.parent == roots[ARM_SPLIT].parent.parent
    assert roots[ARM_SINGLE].name == roots[ARM_SPLIT].name == "runs_root"
    assert roots[ARM_SINGLE].parent.name == "single"
    assert roots[ARM_SPLIT].parent.name == "split"

    assert single.arm == ARM_SINGLE and split.arm == ARM_SPLIT
    assert single.pre_write_absent is True and split.pre_write_absent is True
    assert single.signal_fingerprint != split.signal_fingerprint

    loaded_single = writer.load_generation(arm=ARM_SINGLE, batch_id="b1")
    loaded_split = writer.load_generation(arm=ARM_SPLIT, batch_id="b1")
    assert loaded_single == single
    assert loaded_split == split
    assert loaded_single.pre_write_parent_entries == ()
    assert "b1" not in loaded_single.pre_write_parent_entries
    assert loaded_single.generation_id == generation_identity(single.to_document())
    assert loaded_single.file_hashes == file_hashes(Path(single.out_dir))
    assert loaded_single.config_fingerprint == loaded_split.config_fingerprint
    assert len(loaded_single.config_fingerprint) == 16


def test_arm_batch_meta_carries_only_its_own_label(tmp_path: Path) -> None:
    writer, single, split = _write_both(tmp_path)
    for generation in (single, split):
        meta = json.loads(
            (Path(generation.out_dir) / "batch_meta.json").read_text(encoding="utf-8")
        )
        assert meta["arms"] == [generation.arm]
        assert meta["batch_id"] == "b1"


def test_same_root_and_nested_roots_fail(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    with pytest.raises(ArmIsolationError, match="两臂根相同"):
        assert_isolated_roots({ARM_SINGLE: shared, ARM_SPLIT: shared})
    with pytest.raises(ArmIsolationError, match="嵌套"):
        assert_isolated_roots({ARM_SINGLE: shared, ARM_SPLIT: shared / "inner"})
    with pytest.raises(ArmIsolationError, match="嵌套"):
        assert_isolated_roots({ARM_SINGLE: shared / "inner", ARM_SPLIT: shared})
    with pytest.raises(ArmIsolationError, match="臂集合"):
        assert_isolated_roots({ARM_SINGLE: tmp_path / "a"})


def test_pre_existing_out_dir_fails_before_writing(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    target = writer.out_dir(ARM_SINGLE, "b1")
    target.mkdir(parents=True)
    with pytest.raises(ArmIsolationError, match="写前 out_dir 已存在"):
        writer.write_arm(
            arm=ARM_SINGLE,
            results=[_fold()],
            batch_id="b1",
            code_state=_CODE_STATE,
            data_state=_DATA_STATE,
            signals=_SIGNALS_SINGLE,
            created_at="2026-11-09T21:00:00",
        )
    assert list(target.iterdir()) == []


def test_second_write_for_same_arm_is_rejected_by_generation(tmp_path: Path) -> None:
    writer, _single, _split = _write_both(tmp_path)
    with pytest.raises(ArmIsolationError, match="写前 out_dir 已存在"):
        writer.write_arm(
            arm=ARM_SINGLE,
            results=[_fold()],
            batch_id="b1",
            code_state=_CODE_STATE,
            data_state=_DATA_STATE,
            signals=_SIGNALS_SINGLE,
            created_at="2026-11-09T21:00:02",
        )
    with pytest.raises(ArmIsolationError, match="generation 已存在"):
        writer.write_arm(
            arm=ARM_SINGLE,
            results=[_fold()],
            batch_id="b2",
            code_state=_CODE_STATE,
            data_state=_DATA_STATE,
            signals=_SIGNALS_SINGLE,
            created_at="2026-11-09T21:00:03",
        )
    assert not writer.out_dir(ARM_SINGLE, "b2").exists()


def test_copying_arm_a_artifacts_into_arm_b_fails(tmp_path: Path) -> None:
    writer, single, split = _write_both(tmp_path)
    shutil.copytree(single.out_dir, split.out_dir, dirs_exist_ok=True)
    with pytest.raises(ArmIsolationError, match="batch_meta.arms"):
        writer.load_generation(arm=ARM_SPLIT, batch_id="b1")


def test_reusing_other_arm_generation_fails(tmp_path: Path) -> None:
    writer, _single, _split = _write_both(tmp_path)
    source = writer.generation_path(ARM_SINGLE)
    target = writer.generation_path(ARM_SPLIT)
    shutil.copyfile(source, target)
    with pytest.raises(ArmIsolationError, match="身份不符"):
        writer.load_generation(arm=ARM_SPLIT, batch_id="b1")


def test_tampered_root_mapping_fails(tmp_path: Path) -> None:
    writer, single, _split = _write_both(tmp_path)
    path = writer.generation_path(ARM_SINGLE)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["runs_root"] = str(tmp_path / "elsewhere")
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ArmIsolationError, match="根映射被篡改"):
        writer.load_generation(arm=ARM_SINGLE, batch_id="b1")


def test_tampered_generation_id_fails(tmp_path: Path) -> None:
    writer, _single, _split = _write_both(tmp_path)
    path = writer.generation_path(ARM_SINGLE)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["signal_fingerprint"] = "0" * 16
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ArmIsolationError, match="generation_id"):
        writer.load_generation(arm=ARM_SINGLE, batch_id="b1")


def test_artifact_status_rejects_copied_arm_before_classification(tmp_path: Path) -> None:
    """文件集分类前必须通过臂身份绑定：复制来的产物不得被当成两臂一致。"""
    writer, single, split = _write_both(tmp_path)
    shutil.copytree(single.out_dir, split.out_dir, dirs_exist_ok=True)
    with pytest.raises(ArmIsolationError, match="batch_meta.arms"):
        writer.artifact_status(batch_id="b1")


def test_generation_pre_write_snapshot_cannot_claim_batch_existed(tmp_path: Path) -> None:
    """写前目录快照若自相矛盾（已含 batch_id），manifest 必须被拒。"""
    writer, single, _split = _write_both(tmp_path)
    path = writer.generation_path(ARM_SINGLE)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["pre_write_parent_entries"] = ["b1"]
    document["generation_id"] = generation_identity(document)
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ArmIsolationError, match="写前目录快照已含同名条目"):
        writer.load_generation(arm=ARM_SINGLE, batch_id="b1")


def test_generation_hash_list_shape_is_validated(tmp_path: Path) -> None:
    writer, _single, _split = _write_both(tmp_path)
    path = writer.generation_path(ARM_SINGLE)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["file_hashes"] = "not-a-list"
    # 同步重算身份：证明形态校验独立于 generation_id 校验仍然生效
    document["generation_id"] = generation_identity(document)
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ArmIsolationError, match="file_hashes 形态非法"):
        writer.load_generation(arm=ARM_SINGLE, batch_id="b1")


def test_tampered_artifact_bytes_fail_hash_binding(tmp_path: Path) -> None:
    writer, single, _split = _write_both(tmp_path)
    target = Path(single.out_dir) / "summary.parquet"
    target.write_bytes(target.read_bytes() + b"tamper")
    with pytest.raises(ArmIsolationError, match="文件哈希不符"):
        writer.load_generation(arm=ARM_SINGLE, batch_id="b1")


# ────────────────────────── 可缺产物语义 ──────────────────────────


def test_optional_artifacts_present_are_reported(tmp_path: Path) -> None:
    writer, _single, _split = _write_both(tmp_path)
    status = writer.artifact_status(batch_id="b1")
    assert status["summary.parquet"] == "present"
    assert status["folds/split00/trades.parquet"] == "present"
    assert status["folds/split00/attribution.parquet"] == "present"
    assert status["folds/split00/holdings_snapshot.parquet"] == "present"
    assert status["folds/split00/topk_detail.parquet"] == "not_compared/contract_optional_missing"


def test_one_sided_missing_optional_file_fails(tmp_path: Path) -> None:
    writer, single, _split = _write_both(tmp_path)
    (Path(single.out_dir) / "folds" / "split00" / "attribution.parquet").unlink()
    with pytest.raises(ArmIsolationError, match="单侧缺失"):
        writer.artifact_status(batch_id="b1")


def test_one_sided_missing_required_file_fails(tmp_path: Path) -> None:
    writer, single, _split = _write_both(tmp_path)
    (Path(single.out_dir) / "folds" / "split00" / "daily.parquet").unlink()
    with pytest.raises(ArmIsolationError, match="必需产物缺失"):
        writer.artifact_status(batch_id="b1")


def test_source_absent_optional_artifacts_are_registered(tmp_path: Path) -> None:
    writer, _single, _split = _write_both(tmp_path, with_attribution=False, with_snapshot=False)
    status = writer.artifact_status(batch_id="b1")
    assert status["folds/split00/attribution.parquet"] == "not_compared/source_absent"
    assert status["folds/split00/holdings_snapshot.parquet"] == "not_compared/source_absent"
    assert status["folds/split00/daily.parquet"] == "present"


def test_unknown_arm_and_unsafe_ids_are_rejected(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    with pytest.raises(ArmIsolationError, match="未知臂标签"):
        writer.out_dir("other_arm", "b1")
    with pytest.raises(ArmIsolationError, match="路径分隔符"):
        writer.out_dir(ARM_SINGLE, "../escape")
    with pytest.raises(ArmIsolationError, match="ASCII"):
        DualArmRunsWriter(run_id="运行1", base_dir=tmp_path, config=_CONFIG)
    with pytest.raises(ArmIsolationError, match="config"):
        DualArmRunsWriter(run_id="ok", base_dir=tmp_path, config={})


def test_writer_rejection_leaves_no_partial_artifacts(tmp_path: Path) -> None:
    """外部评审 P2：writer 契约拒绝必须零残留，修正后同批次可重试。"""
    writer = _writer(tmp_path)
    bad = _fold()
    bad["trades"]["unknown_column"] = 1
    with pytest.raises(ValueError):
        writer.write_arm(
            arm=ARM_SINGLE,
            results=[bad],
            batch_id="b1",
            code_state=_CODE_STATE,
            data_state=_DATA_STATE,
            signals=_SIGNALS_SINGLE,
            created_at="2026-11-10T11:30:00",
        )
    target = writer.out_dir(ARM_SINGLE, "b1")
    assert not target.exists()
    assert not writer.roots[ARM_SINGLE].exists()

    generation = writer.write_arm(
        arm=ARM_SINGLE,
        results=[_fold()],
        batch_id="b1",
        code_state=_CODE_STATE,
        data_state=_DATA_STATE,
        signals=_SIGNALS_SINGLE,
        created_at="2026-11-10T11:30:01",
    )
    assert target.exists()
    assert generation.pre_write_absent is True


def test_padded_ids_are_rejected_before_any_write(tmp_path: Path) -> None:
    """外部评审 R1-01：首尾空白 ID 不得被 strip 接受而制造两套路径身份。"""
    writer = _writer(tmp_path)
    kwargs = dict(
        arm=ARM_SINGLE,
        results=[_fold()],
        code_state=_CODE_STATE,
        data_state=_DATA_STATE,
        signals=_SIGNALS_SINGLE,
        created_at="2026-11-10T00:00:00",
    )
    with pytest.raises(ArmIsolationError, match="首尾空白"):
        writer.write_arm(batch_id=" b1", **kwargs)
    assert list(tmp_path.rglob("*")) == []
    with pytest.raises(ArmIsolationError, match="首尾空白"):
        writer.out_dir(ARM_SINGLE, "b1 ")
    with pytest.raises(ArmIsolationError, match="首尾空白"):
        DualArmRunsWriter(run_id=" t3run", base_dir=tmp_path, config=_CONFIG)

    generation = writer.write_arm(batch_id="b1", **kwargs)
    assert generation.batch_id == "b1"
    assert (writer.out_dir(ARM_SINGLE, "b1") / "summary.parquet").is_file()
