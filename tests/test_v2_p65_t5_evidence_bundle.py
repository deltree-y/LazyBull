# -*- coding: utf-8 -*-
"""P6.5 T5 证据包（G5~G7）：自含性、哈希可复核、身份字段、禁覆盖、版本一致性。

覆盖（任务 §3.3/§3.4/§5）：
- 证据根含 ``recovery_matrix.json`` / ``state_envelopes/``（两代及以上 + 索引）/
  ``pytest_targeted.txt`` / ``pytest_full.txt`` / ``complexity.json`` / ``manifest.json``；
- manifest 身份字段齐备（HEAD/脏标记/输入/费率/容差/排除表/runs writer/文件哈希/命令）
  且登记的文件哈希与磁盘一致；
- ``recovery_matrix.json`` 与信封索引互相可核（提交点链、上一代摘要衔接）；
- 失败路径也落盘（G5 FAIL 时证据完整保留并抛判定异常），既有证据目录禁覆盖；
- 版本：``pyproject.toml`` 与 ``CHANGELOG.md`` 版本一致，``README.md`` 不混入 changelog。
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

import pytest

from src.lazybull.v2.core.accounting.state_envelope import OWNER_NAMES, sha256_bytes
from src.lazybull.v2.evidence import p65_t5_evidence as evidence
from src.lazybull.v2.evidence.p65_t5_evidence import (
    P65T5EvidenceError,
    T5EvidenceVerdictError,
    collect_t5_evidence,
    default_run_id,
)
from src.lazybull.v2.hosts.recovery_cases import RecoveryMatrix

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUN_ID = "v2p65_t5_test_20261110_abcd1234"


def _fixture_log(tmp_path: Path, name: str, text: str = "1 passed in 0.01s\n") -> Path:
    """假回归日志（测试专用，不代表真实运行）。"""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """构造一份完整证据包（模块级复用；全量日志用测试夹具提供的假日志）。"""
    base = tmp_path_factory.mktemp("p65_t5_evidence")
    root = base / "evidence"
    targeted = _fixture_log(base, "targeted.log", "410 passed in 13.0s\n")
    full = _fixture_log(base, "full.log", "5928 passed in 2100s\n")
    complexity = _fixture_log(base, "complexity.log", "新增硬超限 0 条\n")
    complexity_src = _fixture_log(base, "complexity_src.log", "新增硬超限 0 条\n")
    flake8 = _fixture_log(base, "flake8.log", "")
    command = evidence.reproduce_command(
        root=root,
        run_id=_RUN_ID,
        pytest_targeted=targeted,
        complexity=complexity,
        complexity_src=complexity_src,
        pytest_full=full,
        flake8=flake8,
    )
    return collect_t5_evidence(
        root=root,
        run_id=_RUN_ID,
        code_state={"git_commit": "cecd163", "git_dirty": True},
        command=command,
        pytest_targeted=targeted,
        complexity_log=complexity,
        complexity_src_log=complexity_src,
        pytest_full=full,
        flake8_log=flake8,
        scenario_notes={"boundary": "合成对账边界"},
    )


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_bundle_contains_required_files(bundle: Any) -> None:
    """G5~G7 要求的产物全部在场（缺一即不构成证据包）。"""
    root = bundle.root
    for name in (
        "recovery_matrix.json",
        "pytest_targeted.txt",
        "pytest_full.txt",
        "complexity.json",
        "complexity_src.json",
        "manifest.json",
    ):
        assert (root / name).is_file(), f"缺少证据文件 {name}"
    index = _read_json(root / "state_envelopes" / "index.json")
    assert len(index["entries"]) >= 2
    for entry in index["entries"]:
        assert (root / entry["path"]).is_file()
    assert runner_has_expected_files(root)


def runner_has_expected_files(root: Path) -> bool:
    """包内至少含参照链与五个场景链（每条链的完整信封目录）。"""
    chains = {path.name for path in (root / "envelope_chains").iterdir() if path.is_dir()}
    return {
        "reference",
        "multi_source_lot",
        "mid_commit_point",
        "transfer_after_resume",
        "post_fill_watermark",
        "half_commit_crash",
        "identity_breach",
    } <= chains


def test_manifest_identity_fields_are_complete(bundle: Any) -> None:
    """G7：manifest 必须记录 HEAD/脏标记/输入/费率/容差/排除表/runs writer/命令/哈希。"""
    manifest = _read_json(bundle.root / "manifest.json")
    assert manifest["schema_version"] == evidence.T5_EVIDENCE_SCHEMA
    assert manifest["task"] == "T5"
    assert manifest["stage"] == "P6.5"
    assert manifest["code_state"] == {"git_commit": "cecd163", "git_dirty": True}
    assert manifest["command"].startswith("python scripts/v2_p65/run_t5_acceptance.py")
    assert manifest["environment"]["python"] and manifest["environment"]["pandas"]
    inputs = manifest["inputs"]
    for key in (
        "scenario",
        "config",
        "rates",
        "tolerances",
        "exclusion_table",
        "recovery_parameters",
    ):
        assert key in inputs, f"manifest.inputs 缺少 {key}"
    assert inputs["config"]["read"] is False
    assert "不读取 configs" in inputs["config"]["reason"]
    # "零配置读取"必须由**全新进程探针**证明，不是固定声明（T5-Codex-C5）
    assert inputs["config"]["probe"]["ok"] is True
    assert inputs["config"]["probe"]["kind"] == "fresh_process_block_load_config"
    assert inputs["rates"] == {
        "commission_rate": 0.0003,
        "min_commission": 5.0,
        "stamp_tax": 0.0005,
        "slippage": 0.0,
    }
    assert inputs["tolerances"]["recovery_state_projection"] == 0
    assert inputs["tolerances"]["i1_cash_absolute"] == 1e-6
    assert inputs["exclusion_table"]["used_by_t5"] is False
    assert inputs["exclusion_table"]["source_sha256"]
    assert "p4_engine_state" in inputs["scenario"] or "trade_dates" in inputs["scenario"]
    assert manifest["runs_writer"]["wrote_runs"] is False
    assert manifest["runs_writer"]["sha256"]
    # 代码身份必须自含（git_dirty 下 HEAD 不足以标识版本）
    assert set(manifest["code_surface"]) == set(evidence.CODE_SURFACE)
    for relative, digest in manifest["code_surface"].items():
        assert sha256_bytes((_REPO_ROOT / relative).read_bytes()) == digest
    # 复杂度双口径分列入包
    calibers = [entry["caliber"] for entry in manifest["regression"]["complexity"]]
    assert calibers == ["src", "src+scripts"]
    for entry in manifest["regression"]["complexity"]:
        assert sha256_bytes((bundle.root / entry["path"]).read_bytes()) == entry["sha256"]
    # 命令必须可逐字复现（含日志参数）
    assert "--pytest-targeted" in manifest["command"]
    assert "--complexity" in manifest["command"]
    assert "--complexity-src" in manifest["command"]
    contract = manifest["envelope_contract"]
    assert contract["schema_version"] >= 1
    assert tuple(contract["owner_names"]) == OWNER_NAMES
    assert contract["commit_point_chain"] == [1, 2, 3, 4, 5]
    recovery = manifest["recovery"]
    assert recovery["scenario_count"] == 6
    assert recovery["compare_rule"].startswith("float64 位模式")
    assert "信封字节不等" in recovery["envelope_byte_boundary"]
    assert manifest["evidence_complete"] is True
    assert manifest["verdict"]["recovery_matrix"] == "PASS"
    assert manifest["verdict"]["failed_cases"] == []
    assert manifest["verdict"]["pytest_full"] == "RUN"


def test_manifest_file_hashes_match_disk(bundle: Any) -> None:
    """manifest 登记的每个文件摘要都能用磁盘字节重算复核。"""
    manifest = _read_json(bundle.root / "manifest.json")
    assert manifest["file_sha256"], "文件哈希清单不得为空"
    for relative, digest in manifest["file_sha256"]:
        path = bundle.root / relative
        assert path.is_file(), f"{relative} 不在证据包内"
        assert sha256_bytes(path.read_bytes()) == digest
    assert "manifest.json" not in {relative for relative, _digest in manifest["file_sha256"]}


def test_recovery_matrix_and_envelopes_cross_check(bundle: Any) -> None:
    """矩阵结论、信封索引与磁盘三方一致（摘要可互相定位）。"""
    matrix = _read_json(bundle.root / "recovery_matrix.json")
    index = _read_json(bundle.root / "state_envelopes" / "index.json")
    assert matrix["verdict"] == "PASS"
    assert matrix["case_count"] == 6
    assert matrix["failed_cases"] == []
    assert matrix["reference_digest_sha256"]
    for case in matrix["cases"]:
        assert case["verdict"] == "PASS"
        assert case["bitwise_equal"] is True
        assert case["difference_count"] == 0
        assert case["differences"] == []
        assert len(case["envelope_sha256"]) == 5
        for _point, digest in case["envelope_sha256"]:
            assert re.fullmatch(r"[0-9a-f]{64}", digest)
    points = [entry["commit_point"] for entry in index["entries"]]
    assert points == [1, 2, 3, 4, 5]
    for previous, current in zip(index["entries"], index["entries"][1:]):
        assert current["previous_envelope_sha256"] == previous["sha256"]
        assert current["run_id"] == previous["run_id"]
        for name in OWNER_NAMES:
            assert re.fullmatch(r"[0-9a-f]{64}", current["owner_sha256"][name])
    reference = dict(
        (point, digest) for point, digest in matrix["reference_envelopes"]
    )
    for entry in index["entries"]:
        assert reference[entry["commit_point"]] == entry["sha256"]


def test_pending_full_is_declared_not_faked(tmp_path: Path) -> None:
    """全量未跑：包内写"待跑声明"并标记 PENDING（不得冒充已运行）。"""
    base = tmp_path / "pending"
    base.mkdir()
    bundle = collect_t5_evidence(
        root=base / "evidence",
        run_id="v2p65_t5_pending_20261110_0000ffff",
        code_state={"git_commit": "cecd163", "git_dirty": True},
        command="python scripts/v2_p65/run_t5_acceptance.py",
        pytest_targeted=_fixture_log(base, "t.log"),
        complexity_log=_fixture_log(base, "c.log"),
    )
    assert bundle.complete is False
    manifest = _read_json(bundle.root / "manifest.json")
    assert manifest["evidence_complete"] is False
    assert manifest["verdict"]["pytest_full"] == "PENDING"
    declaration = json.loads((bundle.root / "pytest_full.txt").read_text(encoding="utf-8"))
    assert declaration["status"] == "PENDING"
    assert "禁止运行全量 pytest" in declaration["rule"]


def test_existing_evidence_root_is_rejected(bundle: Any) -> None:
    """既有证据目录禁覆盖（同 run_id 再跑即拒绝，且不改动既有内容）。"""
    before = {
        path.relative_to(bundle.root).as_posix(): sha256_bytes(path.read_bytes())
        for path in sorted(bundle.root.rglob("*"))
        if path.is_file()
    }
    with pytest.raises(P65T5EvidenceError, match="禁止覆盖"):
        collect_t5_evidence(
            root=bundle.root.parent,
            run_id=_RUN_ID,
            code_state={"git_commit": "x", "git_dirty": False},
            command="x",
            pytest_targeted=bundle.root / "pytest_targeted.txt",
            complexity_log=bundle.root / "complexity.json",
        )
    after = {
        path.relative_to(bundle.root).as_posix(): sha256_bytes(path.read_bytes())
        for path in sorted(bundle.root.rglob("*"))
        if path.is_file()
    }
    assert after == before


def test_illegal_run_id_creates_no_directory(tmp_path: Path) -> None:
    """非法 run_id 在建目录之前拒绝（证据根之外零残留）。"""
    root = tmp_path / "evidence"
    for bad in ("../escape", "a/b", ".", "..", "", " ", "中文"):
        with pytest.raises(P65T5EvidenceError, match="run_id 非法"):
            collect_t5_evidence(
                root=root,
                run_id=bad,
                code_state={},
                command="x",
                pytest_targeted=tmp_path / "missing.log",
                complexity_log=tmp_path / "missing.log",
            )
    assert not root.exists()


def test_missing_regression_log_is_rejected(tmp_path: Path) -> None:
    """回归日志缺失即拒绝（证据不得登记未发生的运行），目录零残留。"""
    root = tmp_path / "evidence"
    with pytest.raises(P65T5EvidenceError, match="日志不存在"):
        collect_t5_evidence(
            root=root,
            run_id="v2p65_t5_missing_20261110_11112222",
            code_state={},
            command="x",
            pytest_targeted=tmp_path / "missing.log",
            complexity_log=tmp_path / "missing.log",
        )
    assert not (root / "v2p65_t5_missing_20261110_11112222").exists()


def test_failed_recovery_keeps_evidence_and_raises_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """G5 FAIL：证据仍完整落盘（保留现场），并抛判定异常映射退出码 3。"""
    real = evidence.run_recovery_matrix

    def failing(**kwargs: Any) -> RecoveryMatrix:
        matrix = real(**kwargs)
        broken = replace(
            matrix.cases[0],
            equal=False,
            verdict="FAIL",
            differences=("reference.physical.cash: 参照='0x1.8p+16' 恢复='0x1.8p+16+1ulp'",),
        )
        return replace(matrix, cases=(broken,) + matrix.cases[1:])

    monkeypatch.setattr(evidence, "run_recovery_matrix", failing)
    base = tmp_path / "fail"
    base.mkdir()
    with pytest.raises(T5EvidenceVerdictError) as failure:
        collect_t5_evidence(
            root=base / "evidence",
            run_id="v2p65_t5_fail_20261110_33334444",
            code_state={"git_commit": "x"},
            command="x",
            pytest_targeted=_fixture_log(base, "t.log"),
            complexity_log=_fixture_log(base, "c.log"),
        )
    target = failure.value.root
    assert (target / "recovery_matrix.json").is_file()
    matrix = _read_json(target / "recovery_matrix.json")
    assert matrix["verdict"] == "FAIL"
    assert matrix["failed_cases"] == ["multi_source_lot"]
    assert matrix["cases"][0]["differences"]
    manifest = _read_json(target / "manifest.json")
    assert manifest["verdict"]["recovery_matrix"] == "FAIL"
    assert manifest["verdict"]["failed_cases"] == ["multi_source_lot"]


def test_default_run_id_contains_stage_date_and_content_salt() -> None:
    """run_id 形态：阶段 + 日期 + 短内容指纹（ASCII，可用于证据根单段目录）。"""
    run_id = default_run_id(date="20261110", salt="abcdef")
    assert re.fullmatch(r"v2p65_t5_20261110_[0-9a-f]{8}", run_id)
    with pytest.raises(P65T5EvidenceError, match="YYYYMMDD"):
        default_run_id(date="2026-11-10")


def test_content_salt_covers_delivery_surface() -> None:
    """内容指纹必须覆盖真实交付面（8 新增 + 5 修改，共 13 文件），缺失即拒绝。"""
    salt = evidence.content_salt()
    assert re.fullmatch(r"[0-9a-f]{8}", salt)
    assert evidence.content_salt(root=_REPO_ROOT) == salt
    for relative in (
        "src/lazybull/v2/hosts/recovery_cases.py",
        "src/lazybull/v2/hosts/recovery_runtime.py",
        "src/lazybull/v2/hosts/recovery_state_digest.py",
        "src/lazybull/v2/core/accounting/state_consistency.py",
        "src/lazybull/v2/core/accounting/physical_lot_view.py",
        "src/lazybull/v2/hosts/order_chain.py",
        "src/lazybull/v2/hosts/virtual_native_engine.py",
        "src/lazybull/common/cost.py",
        "src/lazybull/v2/common/cost.py",
    ):
        assert relative in evidence.CODE_SURFACE
    # 8 新增交付模块 + 5 被修改模块：清单不得漏掉任何一个被改动的文件
    assert len(evidence.CODE_SURFACE) == 13
    assert len(set(evidence.CODE_SURFACE)) == 13
    with pytest.raises(P65T5EvidenceError, match="文件缺失"):
        evidence.content_salt(("src/lazybull/v2/hosts/not_exists.py",))


def test_content_salt_changes_with_delivery_file(tmp_path: Path) -> None:
    """交付面文件内容变化 ⇒ 指纹变化（run_id 才能标识代码版本）。"""
    root = tmp_path / "repo"
    (root / "a").mkdir(parents=True)
    (root / "a" / "one.py").write_text("x = 1\n", encoding="utf-8")
    (root / "a" / "two.py").write_text("y = 2\n", encoding="utf-8")
    files = ("a/one.py", "a/two.py")
    before = evidence.content_salt(files, root=root)
    (root / "a" / "two.py").write_text("y = 3\n", encoding="utf-8")
    assert evidence.content_salt(files, root=root) != before


def test_version_consistency_and_readme_purity() -> None:
    """版本收口：pyproject 与 CHANGELOG 一致；README 不混入 changelog 内容。"""
    pyproject = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version_match = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)
    assert version_match is not None
    version = version_match.group(1)
    changelog = (_REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    heading = re.search(r"^## \[([0-9]+\.[0-9]+\.[0-9]+)\] - (\d{4}-\d{2}-\d{2})$", changelog, re.M)
    assert heading is not None
    assert heading.group(1) == version, "CHANGELOG 首条版本必须等于 pyproject 版本"
    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert f"## [{version}]" not in readme
    assert not re.search(r"^#{2,3} \[[0-9]+\.[0-9]+\.[0-9]+\]", readme, re.MULTILINE)
    assert "### Added（" not in readme and "### Changed（" not in readme


# ────────────────────────── 运行中破缺必须留证（T5-Codex-C4） ──────────────────────────


def test_identity_failure_mid_run_retains_prior_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """真实校验器在运行中抛异常时，破缺前信封与现场必须保留并写失败记录。"""
    from src.lazybull.v2.core.accounting.reconciliation import ReconciliationError
    from src.lazybull.v2.hosts.recovery_runtime import RecoverableRuntime

    original = RecoverableRuntime.run_day
    injected: Dict[str, bool] = {"done": False}

    def wrapper(self: Any, day_index: int) -> Any:
        result = original(self, day_index)
        if day_index == 1 and not injected["done"]:
            injected["done"] = True
            date = self.scenario.date_list()[day_index]
            self.host.current_capital += 0.01  # 破坏 I1（现金）
            self.host._run_identity_gate(  # noqa: SLF001 - 真实校验器抛异常
                date=date,
                portfolio_value=self.host._calculate_portfolio_value(date),
            )
        return result

    monkeypatch.setattr(RecoverableRuntime, "run_day", wrapper)
    base = tmp_path / "failure"
    root = base / "evidence"
    root.mkdir(parents=True)
    targeted = _fixture_log(base, "t.log", "1 passed\n")
    complexity = _fixture_log(base, "c.log", "0\n")
    run_id = "v2p65_t5_fail_20261110_11112222"
    with pytest.raises(ReconciliationError):
        collect_t5_evidence(
            root=root,
            run_id=run_id,
            code_state={"git_commit": "cecd163", "git_dirty": True},
            command="python scripts/v2_p65/run_t5_acceptance.py --probe",
            pytest_targeted=targeted,
            complexity_log=complexity,
        )
    target = root / run_id
    assert target.is_dir(), "破缺现场不得被整根删除"
    failure = json.loads((target / "failure.json").read_text(encoding="utf-8"))
    assert failure["kind"] == "failure"
    assert failure["error_type"] == "ReconciliationError"
    assert failure["input_rejection"] is False
    assert "I1" in failure["message"] or "现金" in failure["message"]
    retained = [path for path, _digest in failure["retained_files"]]
    assert any(path.endswith(".json") and path.startswith("envelope_chains/") for path in retained)
    # 破缺前完整信封仍在，且没有伪造成"有效证据包"
    assert (target / "envelope_chains" / "reference").is_dir()
    assert not (target / "manifest.json").exists()


# ────────────────────────── 合成宿主零配置读取（T5-Codex-C5） ──────────────────────────


def test_synthetic_host_does_not_read_default_config() -> None:
    """全新解释器 + 拦截 Config.load_config：合成宿主完整构造链不得读默认配置。"""
    probe = evidence._run_config_probe()  # noqa: SLF001 - 证据层同一探针
    assert probe["ok"] is True, probe
    assert "PROBE_CONFIG_READ" not in probe["stderr_tail"]
    assert probe["returncode"] == 0


def test_config_probe_is_concurrency_safe() -> None:
    """两个独立探针并发必须都成功（R1-02：各自独立临时目录，互不删除对方链）。"""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: evidence._run_config_probe(), range(3)))  # noqa: SLF001
    errors = [
        (item["returncode"], item["stderr_tail"][-200:]) for item in results if not item["ok"]
    ]
    assert not errors, errors
