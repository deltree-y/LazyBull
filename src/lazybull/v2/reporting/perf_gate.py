# -*- coding: utf-8 -*-
"""P6 性能门与验收环境门（设计 §7.1/§7.2）。

阈值语义（逐条写死，禁止取均值或使用热缓存冒充冷启动）：
- 单页体积 ``<= 3_000_000`` 字节（含 CSS/SVG/图片/字体/注释）；
- 冷/热打开各 5 次，每次 ``< 2_000`` ms；
- 生成 5 次，每次 ``< 10_000`` ms。

阈值负例通过**注入合成度量记录**触发（``evaluate_metric`` / ``judge_*``），不制造真实慢页面。
本模块不导入 Playwright，也不写盘。
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

from src.lazybull.v2.reporting import contracts

__all__ = [
    "COLD_REPEATS",
    "GENERATION_REPEATS",
    "HOT_REPEATS",
    "MAX_GENERATION_MS",
    "MAX_OPEN_MS",
    "MAX_REPORT_BYTES",
    "MeasurementError",
    "evaluate_metric",
    "judge_generation",
    "judge_generation_series",
    "judge_open",
    "judge_open_series",
    "judge_size",
    "scan_external_references",
    "scan_request_urls",
    "validate_environment",
    "verify_acceptance_versions",
]

MAX_REPORT_BYTES = contracts.THRESHOLDS["max_report_bytes"]
MAX_OPEN_MS = contracts.THRESHOLDS["max_open_ms"]
MAX_GENERATION_MS = contracts.THRESHOLDS["max_generation_ms"]
COLD_REPEATS = contracts.THRESHOLDS["cold_repeats"]
HOT_REPEATS = contracts.THRESHOLDS["hot_repeats"]
GENERATION_REPEATS = contracts.THRESHOLDS["generation_repeats"]

#: 请求白名单：本地文件、内联数据与本地伪协议不会发起网络访问
_ALLOWED_REQUEST_SCHEMES = frozenset({"file", "data", "about", "blob"})
#: 非抓取的命名空间声明（HTML/SVG 规范 URI，不产生请求）
_ALLOWED_NAMESPACE_URLS = frozenset({"http://www.w3.org/2000/svg", "http://www.w3.org/1999/xhtml"})
_SCHEME_URL = re.compile(r"(?i)\b(?:https?|wss?|ftp)://[^\s\"'<>)]*")
_RESOURCE_ATTR = re.compile(r"(?i)\b(?:src|href|action|formaction)\s*=\s*([\"'])(.*?)\1")
_CSS_IMPORT = re.compile(r"(?i)@import\s+(?:url\()?[\"']?((?:https?:)?//[^\"')\s]+)")
_CSS_REMOTE_URL = re.compile(r"(?i)url\(\s*[\"']?((?:https?:)?//[^\"')\s]+)")

#: 验收环境必需字段（设计 §7.2：缺项无效）
REQUIRED_ENVIRONMENT_FIELDS: tuple[str, ...] = (
    "cpu_model",
    "cpu_cores",
    "ram_bytes",
    "os",
    "os_build",
    "python_version",
    "jinja2_version",
    "playwright_version",
    "chromium_version",
    "power_plan",
)
_INVALID_ENVIRONMENT_TEXT = frozenset({"", "unknown", "unavailable", "n/a", "none"})


class MeasurementError(ValueError):
    """度量记录不合法（缺字段、非数值、非有限、次数不符）。"""


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MeasurementError(f"{field} 必须是数值：{value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise MeasurementError(f"{field} 必须有限：{value!r}")
    return number


def _non_negative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MeasurementError(f"{field} 必须是整数：{value!r}")
    if value < 0:
        raise MeasurementError(f"{field} 不能为负：{value!r}")
    return value


def judge_size(byte_size: int) -> dict[str, Any]:
    """单页体积门：``<= 3_000_000`` 字节通过（压缩体积不参与判定）。"""
    size = _non_negative_int(byte_size, field="byte_size")
    return {
        "kind": "size",
        "value": size,
        "threshold": MAX_REPORT_BYTES,
        "rule": "<=",
        "passed": size <= MAX_REPORT_BYTES,
    }


def judge_open(elapsed_ms: float, *, mode: str) -> dict[str, Any]:
    """单次打开门：``< 2_000`` ms 通过；``mode`` 只能是 cold/hot。"""
    if mode not in {"cold", "hot"}:
        raise MeasurementError(f"打开测量 mode 只能是 cold/hot：{mode!r}")
    elapsed = _finite_number(elapsed_ms, field=f"{mode}.elapsed_ms")
    return {
        "kind": "open",
        "mode": mode,
        "value": elapsed,
        "threshold": MAX_OPEN_MS,
        "rule": "<",
        "passed": elapsed < MAX_OPEN_MS,
    }


def judge_open_series(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """冷/热打开序列门：必须恰为 5 冷 + 5 热，且每次 ``< 2_000`` ms。"""
    materials = list(records)
    judged = [
        judge_open(record.get("elapsed_ms"), mode=str(record.get("mode", "")))
        for record in materials
    ]
    cold = [entry for entry in judged if entry["mode"] == "cold"]
    hot = [entry for entry in judged if entry["mode"] == "hot"]
    if len(cold) != COLD_REPEATS or len(hot) != HOT_REPEATS:
        raise MeasurementError(
            "打开测量必须恰为 "
            f"{COLD_REPEATS} 冷 + {HOT_REPEATS} 热：实际 {len(cold)} 冷 + {len(hot)} 热"
        )
    return {
        "kind": "open_series",
        "cold_count": len(cold),
        "hot_count": len(hot),
        "expected_cold": COLD_REPEATS,
        "expected_hot": HOT_REPEATS,
        "max_ms": max(entry["value"] for entry in judged),
        "counts_ok": True,
        "passed": all(entry["passed"] for entry in judged),
        "records": judged,
    }


def judge_generation(elapsed_ms: float) -> dict[str, Any]:
    """单次生成门：``< 10_000`` ms 通过。"""
    elapsed = _finite_number(elapsed_ms, field="generation.elapsed_ms")
    return {
        "kind": "generation",
        "value": elapsed,
        "threshold": MAX_GENERATION_MS,
        "rule": "<",
        "passed": elapsed < MAX_GENERATION_MS,
    }


def judge_generation_series(values: Sequence[float]) -> dict[str, Any]:
    """生成序列门：必须恰 5 次，且每次 ``< 10_000`` ms。"""
    materials = list(values)
    if len(materials) != GENERATION_REPEATS:
        raise MeasurementError(
            f"生成测量必须恰为 {GENERATION_REPEATS} 次：实际 {len(materials)} 次"
        )
    judged = [judge_generation(value) for value in materials]
    return {
        "kind": "generation_series",
        "repeat": len(judged),
        "expected_repeat": GENERATION_REPEATS,
        "max_ms": max(entry["value"] for entry in judged),
        "counts_ok": True,
        "passed": all(entry["passed"] for entry in judged),
        "records": judged,
    }


def evaluate_metric(record: Mapping[str, Any]) -> dict[str, Any]:
    """注入合成度量记录的统一定罪入口（T0.5 阈值负例唯一通道）。

    ``record`` 形态：``{"kind": "size|open|generation", "value": <数值>, "mode": "cold|hot"}``。
    """
    kind = record.get("kind")
    if kind == "size":
        return judge_size(record.get("value"))  # type: ignore[arg-type]
    if kind == "open":
        return judge_open(record.get("value"), mode=str(record.get("mode", "")))
    if kind == "generation":
        return judge_generation(record.get("value"))
    raise MeasurementError(f"未知度量记录 kind：{kind!r}")


def scan_external_references(text: str) -> list[str]:
    """扫描 HTML 中的远程资源引用；返回排序去重后的命中列表（空表示通过）。

    白名单只含 HTML/SVG 命名空间声明；``data:`` 内联资源与本地固定文件名不在此列。
    """
    findings: set[str] = set()
    for match in _SCHEME_URL.finditer(text):
        url = match.group(0).rstrip(".,;")
        if url not in _ALLOWED_NAMESPACE_URLS:
            findings.add(url)
    for match in _RESOURCE_ATTR.finditer(text):
        value = match.group(2).strip()
        lowered = value.lower()
        if lowered.startswith(("http://", "https://", "//", "ws://", "wss://", "ftp://")):
            if value not in _ALLOWED_NAMESPACE_URLS:
                findings.add(value)
    for regex in (_CSS_IMPORT, _CSS_REMOTE_URL):
        for match in regex.finditer(text):
            findings.add(match.group(1))
    return sorted(findings)


def scan_request_urls(urls: Iterable[str]) -> list[str]:
    """请求清单门：只允许本地伪协议；任何 http(s)/ws(s)/ftp 请求都列为违规。"""
    offenders: set[str] = set()
    for url in urls:
        scheme = urlsplit(str(url)).scheme.lower()
        if scheme not in _ALLOWED_REQUEST_SCHEMES:
            offenders.add(str(url))
    return sorted(offenders)


def validate_environment(
    record: Mapping[str, Any], *, require_browser: bool = True
) -> dict[str, Any]:
    """验收环境完整性门：CPU 型号/核数/RAM/OS/Python/Jinja2/Playwright/Chromium/电源模式缺项无效。

    ``require_browser=False`` 用于纯生成测量（不启动浏览器），此时不要求 Playwright/Chromium 两项。
    """
    required = [
        field
        for field in REQUIRED_ENVIRONMENT_FIELDS
        if require_browser or field not in {"playwright_version", "chromium_version"}
    ]
    missing = sorted(field for field in required if field not in record)
    invalid: list[str] = []
    text_fields = ["cpu_model", "os", "os_build", "python_version", "jinja2_version", "power_plan"]
    if require_browser:
        text_fields.extend(["playwright_version", "chromium_version"])
    for field in text_fields:
        if field not in record:
            continue  # 缺键已计入 missing；此处只管"键在但值不合法"（评审 R1-2/R2-T05-04）
        value = record[field]
        if value is None:
            invalid.append(field)
            continue
        text = str(value).strip().lower()
        if text in _INVALID_ENVIRONMENT_TEXT:
            invalid.append(field)
        elif field == "power_plan" and text == "not_applicable_non_windows":
            continue
    for field in ("cpu_cores", "ram_bytes"):
        if field not in record:
            continue
        value = record[field]
        # 严格正整数：拒绝 None/布尔/字符串/浮点与零负值（评审 R2-T05-04 整改要求）
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            invalid.append(field)
    return {
        "kind": "environment",
        "missing": missing,
        "invalid": sorted(set(invalid)),
        "passed": not missing and not invalid,
    }


def verify_acceptance_versions(
    installed: Mapping[str, Any], *, names: Sequence[str] | None = None
) -> dict[str, Any]:
    """验收精确版本门：Jinja2/Playwright/Chromium 与冻结值逐项相等。

    ``names`` 可限定本次实际使用的依赖（纯生成测量只核对 Jinja2）。
    """
    selected = tuple(names) if names is not None else tuple(contracts.ACCEPTANCE_VERSIONS)
    unknown = sorted(set(selected) - set(contracts.ACCEPTANCE_VERSIONS))
    if unknown:
        raise MeasurementError(f"未知验收依赖：{unknown}")
    mismatch: dict[str, dict[str, Any]] = {}
    for name in selected:
        expected = contracts.ACCEPTANCE_VERSIONS[name]
        actual = installed.get(name)
        if str(actual) != expected:
            mismatch[name] = {"expected": expected, "actual": actual}
    return {"kind": "acceptance_versions", "mismatch": mismatch, "passed": not mismatch}
