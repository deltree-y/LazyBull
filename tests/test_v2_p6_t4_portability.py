# -*- coding: utf-8 -*-
"""P6 T4 跨平台准备：路径字节一致 / NFC / CRLF-LF / 时区-locale / Linux 验收包。

P6-T4-03：Windows 与 POSIX 路径在同一 root 下产出**逐字节相同**的 root-relative 引用；
``Path.as_uri``（``file:///D:/...``）形态一律拒绝；Linux 包为固定文件集 + POSIX 路径 +
期望哈希 + 复跑命令（真实 Linux 执行门归 T5）。
"""

from __future__ import annotations

import json
import os
import unicodedata
from pathlib import Path

import pytest

from src.lazybull.v2.reporting import generate, trend_contract, trend_csv, trend_portability
from src.lazybull.v2.reporting.csv_core import CsvContractError
from tests import p6_t4_env


def test_p6_t4_03_paths_are_byte_identical_across_platforms() -> None:
    """Windows/POSIX 路径压成同一条 root-relative POSIX 引用（字节一致）。"""
    windows = trend_portability.relative_posix("D:/repo/data/runs/x", "D:/repo")
    posix = trend_portability.relative_posix("/repo/data/runs/x", "/repo")
    assert windows == posix == "data/runs/x"
    assert trend_portability.portable_source_ref("src0", "data\\runs\\x") == "src0:data/runs/x"
    # NFC：NFD 输入归一为 NFC（字节一致，不出现两种码位）
    nfd = unicodedata.normalize("NFD", "data/runs/é")
    assert trend_portability.portable_source_ref("src0", nfd) == (
        "src0:" + unicodedata.normalize("NFC", "data/runs/é")
    )
    # Path.as_uri 形态（Windows 盘符/URI）不得进入 canonical 输出
    uri = Path("D:/repo/data").as_uri()
    assert uri.startswith("file:///")
    with pytest.raises(ValueError):
        trend_portability.relative_posix(uri, "D:/repo")
    with pytest.raises(ValueError):
        trend_portability.assert_portable_ref("src0:../escape")
    with pytest.raises(ValueError):
        trend_portability.assert_portable_ref("src0:/abs/path")


def test_p6_t4_crlf_bom_are_rejected_by_csv_contract() -> None:
    """CRLF/BOM 负例：两份趋势 CSV 只接受 UTF-8 无 BOM + LF + 单尾换行。"""
    header = ",".join(trend_contract.RUN_INDEX_COLUMNS)
    good = f"{header}\n"
    assert trend_csv.verify_run_index(good)["row_count"] == 0
    with pytest.raises(CsvContractError):
        trend_csv.verify_run_index(f"{header}\r\n")
    with pytest.raises(CsvContractError):
        trend_csv.verify_run_index("\ufeff" + good)
    with pytest.raises(CsvContractError):
        trend_csv.verify_run_index(good.rstrip("\n"))
    trend_header = ",".join(trend_contract.TREND_COLUMNS)
    with pytest.raises(CsvContractError):
        trend_csv.verify_trend(f"{trend_header}\r\n")


def test_p6_t4_linux_package_fixed_fileset_and_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux 验收包：固定四文件 + POSIX 路径 + 期望哈希 + 复跑命令；时区/locale 不影响字节。"""
    files = {"report.html": "a" * 64, "run_index.csv": "b" * 64, "trend.csv": "c" * 64}
    commands = [
        "cd <repo_root>",
        "python scripts/p6_report.py generate --kind trend --evidence-class research "
        "--source runs:data/runs/<batch>",
    ]
    package = trend_portability.build_linux_package(
        report_id="p6_trend_20261106T090000Z_0123456789ab",
        generated_at="20261106T090000Z",
        evidence_class="research",
        files=files,
        commands=commands,
        corpus_sha256="d" * 64,
    )
    assert tuple(sorted(package)) == tuple(sorted(trend_portability.LINUX_PACKAGE_FILES))
    expected = json.loads(package["expected_hashes.json"].decode("utf-8"))
    assert expected["files"] == files
    assert expected["root_placeholder"] == "<repo_root>"
    assert expected["commands"] == commands
    assert any("<repo_root>" in item for item in expected["commands"])
    shell = package["acceptance.sh"].decode("utf-8")
    assert "\r" not in shell and shell.startswith("#!/usr/bin/env sh\n")
    assert shell.isascii()  # POSIX shell 只用 ASCII，避免 Linux 终端编码差异
    assert all(name in shell for name in files)
    manifest = json.loads(package["package_manifest.json"].decode("utf-8"))
    assert manifest["files"] and manifest["record_sha256"]
    assert "D:" not in package["expected_hashes.json"].decode("utf-8")
    # 时区/locale 环境变化不改变 canonical 包字节（无墙钟、无 locale 格式化）
    monkeypatch.setenv("TZ", "Asia/Shanghai")
    monkeypatch.setenv("LC_ALL", "C")
    again = trend_portability.build_linux_package(
        report_id="p6_trend_20261106T090000Z_0123456789ab",
        generated_at="20261106T090000Z",
        evidence_class="research",
        files=files,
        commands=commands,
        corpus_sha256="d" * 64,
    )
    assert again == package
    monkeypatch.setenv("TZ", "UTC")
    assert os.environ["TZ"] == "UTC"


def test_p6_t4_03_generation_is_cwd_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**同一绝对来源根**在两个 cwd 下生成同一份字节（root_id 锚定仓库根，非 cwd）。"""
    corpus = tmp_path / "corpus"
    p6_t4_env.build_corpus(corpus, upto=1)
    outputs: dict[str, dict[str, bytes]] = {}
    for name, cwd in (("alpha", tmp_path), ("beta", Path.cwd().parent)):
        monkeypatch.chdir(cwd)
        result = generate.generate_trend(
            p6_t4_env.research_request(
                tmp_path / f"reports_{name}", corpus, attempt=f"cwd_{name}"
            )
        )
        outputs[name] = dict(result["files"])
    assert outputs["alpha"] == outputs["beta"]


def test_p6_t4_03_generation_is_tz_locale_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """生成链路与 TZ/locale 无关：固定 generated_at 后两种环境逐字节一致。

    注意（T4-R2-B2-03）：Windows 无 ``time.tzset``，同进程改 ``TZ`` 不改变本地时区；
    本用例在 Windows 上实际验证的是“生成结果与进程环境变量/locale 设置无关”，
    TZ 语义只在 POSIX 生效（T5 双平台复算时补子进程对照）。
    """
    corpus = tmp_path / "corpus"
    p6_t4_env.build_corpus(corpus, upto=1)
    monkeypatch.setenv("TZ", "UTC")
    monkeypatch.setenv("LC_ALL", "C")
    first = generate.generate_trend(
        p6_t4_env.research_request(tmp_path / "reports_a", corpus, attempt="env_a")
    )
    monkeypatch.setenv("TZ", "Asia/Shanghai")
    monkeypatch.setenv("LC_ALL", "C.UTF-8")
    second = generate.generate_trend(
        p6_t4_env.research_request(tmp_path / "reports_b", corpus, attempt="env_b")
    )
    assert first["files"] == second["files"]
