# -*- coding: utf-8 -*-
"""P6 T0.5 环境与依赖完整（P6-T05-06）。

覆盖：pyproject/requirements 声明一致、Jinja2 上下限与缺失 fail-closed、自动转义/严格未定义、
生产包禁止 import Playwright、验收环境缺项判无效、docs 冻结件与代码同源。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.reporting import contracts, deps, perf_gate  # noqa: E402

DOCS_FREEZE = ROOT / "docs" / "contracts" / "p6_t05_contract_freeze_20261105.json"


def test_p6_t05_06_declared_dependencies_match_freeze() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

    jinja = re.search(r'(?m)^jinja2\s*=\s*"([^"]+)"', pyproject)
    assert jinja is not None and jinja.group(1) == ">=3.1,<4.0"
    assert re.search(r"(?mi)^Jinja2\s*>=\s*3\.1\s*,\s*<\s*4\.0\s*$", requirements)

    verification = re.search(
        r"(?ms)^\[tool\.poetry\.group\.verification\.dependencies\]\s*(.*)$", pyproject
    )
    assert verification is not None
    assert re.search(r'(?m)^playwright\s*=\s*">=1\.47,<2\.0"', verification.group(1))
    assert re.search(r"(?mi)^playwright\s*>=\s*1\.47\s*,\s*<\s*2\.0\s*$", requirements)

    import_pattern = re.compile(r"(?m)^\s*(?:from|import)\s+playwright")
    for path in sorted((ROOT / "src" / "lazybull" / "v2" / "reporting").glob("*.py")):
        assert import_pattern.search(path.read_text(encoding="utf-8")) is None, path


def test_p6_t05_06_jinja2_gate_and_environment() -> None:
    module = deps.load_jinja2()
    assert deps.version_in_bounds(
        module.__version__, deps.JINJA2_MIN_VERSION, deps.JINJA2_MAX_EXCLUSIVE
    )
    assert deps.JINJA2_MIN_VERSION == "3.1"
    assert deps.JINJA2_MAX_EXCLUSIVE == "4.0"
    assert deps.parse_version("3.1.6") == ((3, 1, 6), 1, 0)
    assert deps.parse_version("3.1.6.post1") == ((3, 1, 6), 2, 1)
    assert deps.parse_version("3.1rc1") == ((3, 1), 0, 1)
    # C1（评审 R2-T05 复核项）：预发布后缀不得冒充正式版越过下限
    assert deps.version_in_bounds("3.1rc1", "3.1", "4.0") is False
    assert deps.version_in_bounds("3.1.dev0", "3.1", "4.0") is False
    assert deps.version_in_bounds("3.1", "3.1", "4.0") is True
    assert deps.version_in_bounds("3.1.6.post1", "3.1", "4.0") is True
    assert deps.version_in_bounds("4.0", "3.1", "4.0") is False
    # C1 续（评审 R2-T05 复核项）：上限同数字段的预发布版不得冒充 <4.0 放行
    assert deps.version_in_bounds("4.0rc1", "3.1", "4.0") is False
    assert deps.version_in_bounds("4.0.dev0", "3.1", "4.0") is False
    assert deps.version_in_bounds("3.9rc1", "3.1", "4.0") is True  # 非上限同段预发布按区间判定
    with pytest.raises(deps.ReportingDependencyError):
        deps.parse_version("unknown")
    with pytest.raises(deps.ReportingDependencyError):
        deps.parse_version("3.1.notaversion")

    environment = deps.build_environment()
    assert environment.autoescape
    assert "&lt;script&gt;" in environment.from_string("{{ value }}").render(
        value="<script>alert(1)</script>"
    )
    with pytest.raises(Exception):
        environment.from_string("{{ missing_value }}").render()


def test_p6_t05_06_missing_jinja2_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "jinja2", None)
    with pytest.raises(deps.ReportingDependencyError, match="Jinja2"):
        deps.load_jinja2()


def test_p6_t05_06_environment_validator_and_freeze_consistency() -> None:
    complete = {
        "cpu_model": "Intel64 Family 6",
        "cpu_cores": 8,
        "ram_bytes": 16 * 1024**3,
        "os": "Windows",
        "os_build": "10.0.26100",
        "python_version": "3.12.10",
        "jinja2_version": "3.1.6",
        "playwright_version": "1.64.1",
        "chromium_version": "156.0.8078.4",
        "power_plan": "Balanced",
    }
    assert perf_gate.validate_environment(complete)["passed"] is True

    missing = dict(complete)
    missing.pop("power_plan")
    result = perf_gate.validate_environment(missing)
    assert result["passed"] is False and result["missing"] == ["power_plan"]

    incomplete = dict(complete, power_plan="")
    assert perf_gate.validate_environment(incomplete)["passed"] is False

    generation_subset = {
        key: value
        for key, value in complete.items()
        if key not in {"playwright_version", "chromium_version"}
    }
    assert (
        perf_gate.validate_environment(generation_subset, require_browser=False)["passed"] is True
    )
    assert perf_gate.validate_environment(generation_subset)["passed"] is False

    record = json.loads(DOCS_FREEZE.read_text(encoding="utf-8"))
    assert record["contract"] == contracts.as_frozen_payload()
    assert record["contract_sha256"] == contracts.frozen_contract_sha256()
    assert record["schema"] == "p6_t05_freeze_record/v1"


def test_p6_t05_06_environment_rejects_null_and_typed_garbage() -> None:
    """R1-2/R2-T05-04：键在但值为 None 必判无效；数值严格正整数（拒布尔/字符串/浮点/零负）。"""
    complete = {
        "cpu_model": "Intel64 Family 6",
        "cpu_cores": 8,
        "ram_bytes": 16 * 1024**3,
        "os": "Windows",
        "os_build": "10.0.26100",
        "python_version": "3.12.10",
        "jinja2_version": "3.1.6",
        "playwright_version": "1.64.1",
        "chromium_version": "156.0.8078.4",
        "power_plan": "Balanced",
    }
    for field in ("cpu_model", "cpu_cores", "ram_bytes", "power_plan"):
        result = perf_gate.validate_environment(dict(complete, **{field: None}))
        assert result["passed"] is False and field in result["invalid"], (field, result)
    for bad in (True, "8", 8.0, 0, -1):
        result = perf_gate.validate_environment(dict(complete, cpu_cores=bad))
        assert result["passed"] is False and "cpu_cores" in result["invalid"], (bad, result)
    # reviewer2 null_environment_probe 场景：全部键在但值全 None ⇒ 环境门必须失败
    all_null = {key: None for key in complete}
    assert perf_gate.validate_environment(all_null)["passed"] is False
