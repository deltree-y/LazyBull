# -*- coding: utf-8 -*-
"""P6.5 T4 证据包：三件套 JSON、manifest 自含字段、SHA-256 自洽、禁覆盖。

覆盖：
- 证据根下产出 zero_turnover / allocation_conservation / self_split / self_split_diff /
  normalized_sha256 / manifest 与两臂 generation manifest（不写 data/runs/）；
- manifest 含 HEAD/脏标记、输入、配置、费率、容差、排除表、runs writer 身份、文件哈希、命令；
- manifest 内 ``file_sha256`` 与磁盘逐文件一致（自含可复核）；
- 已存在目录拒绝（禁覆盖）；run_id 必须 ASCII；
- not_compared 项只登记、不得写成逐位一致。

全部为合成输入 + 临时目录：不读真实配置/行情，不污染 data/ 与 data/runs/。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.lazybull.v2.evidence import p65_t4_evidence
from src.lazybull.v2.hosts.self_split_arms import SelfSplitArmError
from src.lazybull.v2.hosts.virtual_arm_runs import ARM_SPLIT
from src.lazybull.v2.evidence.p65_t4_evidence import (
    EVIDENCE_SCHEMA,
    EvidenceVerdictError,
    P65EvidenceError,
    collect_t4_evidence,
    default_run_id,
)

_CODE_STATE = {"git_commit": "0" * 40, "git_dirty": True}


def _collect(root: Path, *, run_id: str = "v2p65_t4_20260101_test0001"):
    return collect_t4_evidence(
        root=root,
        run_id=run_id,
        code_state=_CODE_STATE,
        command="python scripts/v2_p65/run_t4_acceptance.py --run-id <test>",
        scenario_notes={"boundary": "合成对账边界"},
    )


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("t4evidence")
    return _collect(root)


def test_evidence_files_and_verdicts(bundle) -> None:
    root = bundle.root
    for name in (
        "zero_turnover.json",
        "allocation_conservation.json",
        "self_split.json",
        "self_split_diff.json",
        "normalized_sha256.json",
        "manifest.json",
    ):
        assert (root / name).is_file(), name
    arms = root / "arms" / f"{bundle.run_id}-arms"
    assert (arms / "single" / "generation.json").is_file()
    assert (arms / "split" / "generation.json").is_file()
    assert (
        arms / "single" / "runs_root" / f"{bundle.run_id}-arms-b0" / "batch_meta.json"
    ).is_file()

    verdict = dict(bundle.verdict)
    assert verdict["zero_turnover"] == "PASS"
    assert verdict["allocation_conservation"] == "PASS"
    assert verdict["self_split"] == "PASS"
    assert verdict["self_split_bitwise"] == "PASS"
    assert verdict["not_compared"] == [
        "folds/split00/attribution.parquet",
        "folds/split00/topk_detail.parquet",
    ]
    assert "阶段出口" in str(verdict["boundary"])


def test_zero_turnover_golden_values(bundle) -> None:
    payload = json.loads((bundle.root / "zero_turnover.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == EVIDENCE_SCHEMA
    assert payload["physical"]["fills"] == 0
    assert payload["physical"]["fees"] == 0.0
    assert payload["physical"]["total_value"] == 104_800.0
    assert payload["physical"]["nav"] == 1.048
    assert payload["virtual"]["A"]["cash"] == 51_200.0
    assert payload["virtual"]["A"]["shares"] == 300
    assert payload["virtual"]["A"]["total_value"] == 54_800.0
    assert payload["virtual"]["A"]["nav_contrib"] == 0.548
    assert payload["virtual"]["B"]["cash"] == 48_800.0
    assert payload["virtual"]["B"]["total_value"] == 50_000.0
    assert payload["virtual"]["B"]["nav_contrib"] == 0.5
    assert payload["identities"]["i3_left"] == 1.048
    assert all(payload["assertions"].values())


def test_allocation_conservation_golden_values(bundle) -> None:
    payload = json.loads((bundle.root / "allocation_conservation.json").read_text(encoding="utf-8"))
    assert payload["physical"]["shares"] == 200
    assert payload["physical"]["fee"] == 5.0
    assert payload["physical"]["cash"] == 7_995.0
    assert payload["virtual"]["A"]["allocated_shares"] == 50
    assert payload["virtual"]["A"]["fee"] == 0.0
    assert payload["virtual"]["A"]["total_value"] == 2_500.0
    assert payload["virtual"]["B"]["allocated_shares"] == 150
    assert payload["virtual"]["B"]["fee"] == 5.0
    assert payload["virtual"]["B"]["total_value"] == 7_495.0
    assert payload["virtual"]["A"]["nav_contrib"] == 0.25
    assert payload["virtual"]["B"]["nav_contrib"] == 0.7495
    assert payload["voided_intents"] == [["A", "600000.SH", 50], ["B", "600000.SH", 150]]
    assert all(payload["assertions"].values())


def test_manifest_is_self_contained_and_hashes_match(bundle) -> None:
    manifest = json.loads((bundle.root / "manifest.json").read_text(encoding="utf-8"))
    for key in (
        "schema_version",
        "stage",
        "task",
        "run_id",
        "code_state",
        "environment",
        "command",
        "inputs",
        "runs_writer",
        "file_sha256",
        "verdict",
    ):
        assert key in manifest, key
    assert manifest["code_state"]["git_commit"] == "0" * 40
    assert manifest["code_state"]["git_dirty"] is True
    assert manifest["inputs"]["exclusion_table"]["fields"] == ["created_at", "arms"]
    assert manifest["inputs"]["exclusion_table"]["source_module"].endswith("arm_comparator.py")
    assert len(manifest["inputs"]["exclusion_table"]["source_sha256"]) == 64
    assert manifest["inputs"]["exclusion_table"]["declared_at_import"] is True
    assert manifest["inputs"]["tolerances"]["g4_nav"] == 0
    assert manifest["inputs"]["tolerances"]["arm_comparison"] == 0
    assert manifest["inputs"]["rates"]["commission_rate"] == 0.0003
    assert manifest["runs_writer"]["module"].endswith("runs_writer.py")
    assert len(manifest["runs_writer"]["sha256"]) == 64

    import hashlib

    recorded = {item[0]: item[1] for item in manifest["file_sha256"]}
    on_disk = {
        path.relative_to(bundle.root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in bundle.root.rglob("*")
        if path.is_file() and path.relative_to(bundle.root).as_posix() != "manifest.json"
    }
    assert recorded == on_disk


def test_self_split_evidence_registers_g4_and_generations(bundle) -> None:
    payload = json.loads((bundle.root / "self_split.json").read_text(encoding="utf-8"))
    gate = payload["g4_gate"]
    assert gate["tolerance"] == 0
    assert gate["bitwise_equal"] is True
    assert gate["single_arm_nav"] == 1.00395
    assert gate["split_merged_nav"] == 1.00395
    assert gate["single_arm_nav_bits"] == gate["split_merged_nav_bits"]
    single, split = payload["arms"]
    assert single["sleeve_ids"] == ["solo"]
    assert split["sleeve_ids"] == ["A", "B"]
    assert single["generation"]["runs_root"] != split["generation"]["runs_root"]
    assert single["generation"]["pre_write_absent"] is True
    assert split["generation"]["pre_write_absent"] is True
    assert len(single["generation"]["file_hashes"]) == len(split["generation"]["file_hashes"])


def test_existing_evidence_directory_is_never_overwritten(tmp_path: Path) -> None:
    _collect(tmp_path, run_id="v2p65_t4_20260101_overwrite")
    with pytest.raises(P65EvidenceError, match="禁止覆盖既有证据"):
        _collect(tmp_path, run_id="v2p65_t4_20260101_overwrite")


def test_failed_assembly_leaves_no_directory(tmp_path: Path, monkeypatch) -> None:
    """装配期失败必须清理本次创建的目录（证据根不留半成品；A-R1-04 口径）。"""
    before = sorted(item.name for item in tmp_path.iterdir())

    def boom(*args, **kwargs):
        raise P65EvidenceError("模拟装配期断言失败")

    monkeypatch.setattr(p65_t4_evidence, "build_zero_turnover_evidence", boom)
    with pytest.raises(P65EvidenceError, match="模拟装配期断言失败"):
        _collect(tmp_path, run_id="v2p65_t4_20260101_fail")
    after = sorted(item.name for item in tmp_path.iterdir())
    assert after == before  # 调用前后条目集完全相同（不是恒真的 glob 断言）
    assert not (tmp_path / "v2p65_t4_20260101_fail").exists()


def _resign_arm(writer, arm: str, batch_id: str) -> None:
    """重签 generation 使树哈希自洽（把差异焦点限制在数据位模式上）。"""
    import hashlib

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
    assert len(hashlib.sha256(path.read_bytes()).hexdigest()) == 64


def test_failed_verdict_from_real_1ulp_diff_is_persisted(tmp_path: Path, monkeypatch) -> None:
    """真实 1 ULP 差异驱动 FAIL：差异清单（位模式/来源行）必须落盘并抛退出码 3 的异常。"""
    import math

    import pandas as pd

    real_runner = p65_t4_evidence.run_self_split_arms

    def doctored_runner(**kwargs):
        assembly = real_runner(**kwargs)
        # 在 split 臂 summary 上制造**真实** 1 ULP 差异（不是塞字符串）
        path = assembly.writer.out_dir(ARM_SPLIT, assembly.batch_id) / "summary.parquet"
        frame = pd.read_parquet(path)
        frame["bt_sharpe"] = [math.nextafter(value, math.inf) for value in frame["bt_sharpe"]]
        frame.to_parquet(path, index=False, compression="none")
        _resign_arm(assembly.writer, ARM_SPLIT, assembly.batch_id)
        return assembly

    monkeypatch.setattr(p65_t4_evidence, "run_self_split_arms", doctored_runner)
    with pytest.raises(EvidenceVerdictError, match="候选门未通过") as excinfo:
        _collect(tmp_path, run_id="v2p65_t4_20260101_failverdict")

    root = excinfo.value.root
    diff = json.loads((root / "self_split_diff.json").read_text(encoding="utf-8"))
    assert diff["verdict"] == "FAIL" and diff["difference_count"] >= 1
    first = diff["differences"][0]
    assert "summary.bt_sharpe" in first
    assert "位=" in first and "行=0" in first and "single/summary.parquet" in first
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["verdict"]["self_split_bitwise"] == "FAIL"
    assert manifest["verdict"]["difference_count"] >= 1


def test_acceptance_log_is_registered_in_manifest(tmp_path: Path) -> None:
    """驱动自身的 UTF-8 日志路径必须进 manifest（可复核证据来源）。"""
    bundle = collect_t4_evidence(
        root=tmp_path,
        run_id="v2p65_t4_20260101_logreg",
        code_state=_CODE_STATE,
        command="python scripts/v2_p65/run_t4_acceptance.py",
        scenario_notes={"acceptance_log": "logs/p65_t4_acceptance_x.log"},
    )
    manifest = json.loads((bundle.root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scenario_notes"]["acceptance_log"] == "logs/p65_t4_acceptance_x.log"


def test_run_id_must_be_ascii() -> None:
    with pytest.raises(P65EvidenceError, match="ASCII"):
        collect_t4_evidence(
            root=Path("unused"),
            run_id="运行1",
            code_state=_CODE_STATE,
            command="x",
        )


def test_default_run_id_is_ascii_and_date_tagged() -> None:
    run_id = default_run_id(date="20261110", salt="t4")
    assert run_id.isascii()
    assert run_id.startswith("v2p65_t4_20261110_")
    assert default_run_id(date="20261110", salt="t4") == run_id
    assert default_run_id(date="20261110", salt="other") != run_id
    with pytest.raises(P65EvidenceError, match="YYYYMMDD"):
        default_run_id(date="2026-11-10")


def test_invalid_run_id_leaves_no_directory_outside_root(tmp_path: Path) -> None:
    """越界/分隔符 run_id 必须在任何 mkdir 之前拒绝（Codex-05）。"""
    evidence_root = tmp_path / "evidence"
    before = sorted(item.name for item in tmp_path.rglob("*"))
    for bad in ("../outside/child/run", "a/b", "a\\b", "abs:1", " pad "):
        with pytest.raises(P65EvidenceError):
            collect_t4_evidence(
                root=evidence_root,
                run_id=bad,
                code_state=_CODE_STATE,
                command="x",
            )
    assert not evidence_root.exists()
    assert sorted(item.name for item in tmp_path.rglob("*")) == before
    assert not (tmp_path / "outside").exists()


def test_cli_rejects_dot_run_id_before_creating_log(tmp_path: Path) -> None:
    """CLI 对 ``run_id='.'`` 必须在**建日志之前**拒绝且日志目录零新增（复审 R1-02）。"""
    from src.lazybull.common.config import get_logs_dir
    from scripts.v2_p65 import run_t4_acceptance as driver

    logs_dir = Path(get_logs_dir())
    before = sorted(item.name for item in logs_dir.glob("p65_t4_acceptance_*.log"))
    code = driver.main(["--root", str(tmp_path / "evidence"), "--run-id", "."])
    after = sorted(item.name for item in logs_dir.glob("p65_t4_acceptance_*.log"))
    assert code == driver.EXIT_SETUP_ERROR
    assert after == before
    assert not (tmp_path / "evidence").exists()


def test_normalized_bytes_are_stored_and_locatable(bundle) -> None:
    """证据包必须保存被比较的规范化字节本体，且摘要可逐条定位（Codex-04）。"""
    document = json.loads((bundle.root / "normalized_sha256.json").read_text(encoding="utf-8"))
    assert document["files"]
    manifest = json.loads((bundle.root / "manifest.json").read_text(encoding="utf-8"))
    registered = {item[0]: item[1] for item in manifest["file_sha256"]}
    for entry in document["files"]:
        for side in ("single", "split"):
            relative = entry[f"{side}_path"]
            path = bundle.root / relative
            assert path.is_file(), relative
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == entry[f"{side}_sha256"]
            assert registered.get(relative) == digest
        assert entry["single_sha256"] == entry["split_sha256"]


def test_config_check_freezes_complete_list(bundle) -> None:
    """配置覆盖：完整冻结键值清单 + 重算指纹（Codex-02）。"""
    diff = json.loads((bundle.root / "self_split_diff.json").read_text(encoding="utf-8"))
    check = diff["config_check"]
    assert set(check["expected"]) | set(check["dynamic_keys"]) == set(
        check["observed"]["single_reference"]
    )
    assert check["observed"]["single_reference"] == check["observed"]["virtual_split"]
    assert check["recomputed_matches_recorded"] is True
    assert check["fingerprint_key_count"] == len(check["fingerprint_keys"]) == 14
    fold = diff["fold_closure"]
    assert fold["expected"] == fold["actual"] == ["split00"]
    assert fold["missing_files"] == []
    manifest = json.loads((bundle.root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["inputs"]["expected_batch_config"] == check["expected"]
    host = manifest["inputs"]["host_parameters"]
    assert host["scenario"]["physical_initial_capital"] == 100000.0
    assert host["scenario"]["window_days"] == 2
    assert host["equal_across_arms"] is True and host["matches_frozen"] is True
    assert manifest["verdict"]["fold_count"] == 1


def test_recorded_host_parameters_equal_instances(bundle) -> None:
    """记录的宿主参数必须等于两臂实例回读值（复审 R1-01）。"""
    self_split = json.loads((bundle.root / "self_split.json").read_text(encoding="utf-8"))
    host = self_split["host_parameters"]
    frozen = host["frozen"]
    single = host["observed"]["single_reference"]
    split = host["observed"]["virtual_split"]
    assert single == split == frozen
    assert single["holding_period"] == 5 and single["rebalance_freq"] == 5
    assert single["sell_timing"] == "open"
    # 场景侧只保留窗口跨度，不再把 runs 配置字段 holding_days 当宿主持有期
    assert host["scenario"]["window_days"] == 2
    assert "holding_days" not in host["scenario"]


def test_real_host_parameter_drift_is_rejected(tmp_path: Path, monkeypatch) -> None:
    """只改工厂产出的**真实宿主参数**、冻结表不变 ⇒ 证据拒绝且零残留。"""
    from src.lazybull.v2.hosts import self_split_arms as arms_module

    real_new_host = arms_module._new_host

    def drifted(*args, **kwargs):
        host, ledger = real_new_host(*args, **kwargs)
        host.holding_period = 4  # 只改实例生效值，不动冻结表
        return host, ledger

    monkeypatch.setattr(arms_module, "_new_host", drifted)
    evidence_root = tmp_path / "evidence"
    with pytest.raises(SelfSplitArmError, match="宿主实际参数与冻结表不符"):
        collect_t4_evidence(
            root=evidence_root,
            run_id="v2p65_t4_20260101_drift",
            code_state=_CODE_STATE,
            command="x",
        )
    assert not (evidence_root / "v2p65_t4_20260101_drift").exists()
    assert list(evidence_root.iterdir()) == []
