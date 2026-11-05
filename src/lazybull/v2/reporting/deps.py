# -*- coding: utf-8 -*-
"""P6 报告生成依赖门（Jinja2 兼容上下限 + 缺失 fail-closed）。

设计 §6.7：``pyproject.toml`` 与 ``requirements.txt`` 同步声明 Jinja2 兼容上下限；验收环境锁定
精确版本；Jinja2 缺失或版本不满足时**直接失败**，禁止静默退化为字符串拼接。

本模块**禁止 import Playwright**：Playwright/Chromium 属于独立验收侧依赖，只能由
``scripts/v2_p6/measure_report.py`` 在函数内部导入。
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from types import ModuleType
from typing import Any

from src.lazybull.v2.reporting.contracts import ACCEPTANCE_VERSIONS, DEPENDENCY_BOUNDS

__all__ = [
    "JINJA2_ACCEPTANCE_VERSION",
    "JINJA2_MAX_EXCLUSIVE",
    "JINJA2_MIN_VERSION",
    "ReportingDependencyError",
    "build_environment",
    "load_jinja2",
    "parse_version",
    "version_in_bounds",
]

#: Jinja2 运行时兼容上下限（与冻结契约同源）
JINJA2_MIN_VERSION = DEPENDENCY_BOUNDS["jinja2"]["minimum"]
JINJA2_MAX_EXCLUSIVE = DEPENDENCY_BOUNDS["jinja2"]["maximum_exclusive"]
#: 验收环境精确版本
JINJA2_ACCEPTANCE_VERSION = ACCEPTANCE_VERSIONS["jinja2"]

_VERSION_TOKEN = re.compile(
    r"^(?P<numbers>\d+(?:\.\d+)*)(?:(?P<sep>[._-])(?P<suffix>[a-zA-Z].*)|(?P<glued>[a-zA-Z].*))?$"
)
#: 后缀档位：预发布（a/b/rc/dev/pre 等）= 0 < 正式版（无后缀）= 1 < post = 2
_PRERELEASE_KIND = re.compile(r"^(?P<kind>post|pre|dev|a|alpha|b|beta|rc|c)(?P<seq>\d*)$")


class ReportingDependencyError(RuntimeError):
    """P6 报告生成依赖缺失或版本越界（fail-closed，不提供降级路径）。"""


def parse_version(text: str) -> tuple[tuple[int, ...], int, int]:
    """解析版本号为（数字段, 后缀档位, 后缀序号）；无法解析即报错（不猜测、不补零）。

    预发布后缀（如 ``3.1rc1``/``3.1.dev0``）必须小于同数字正式版（评审 C1：
    只取前导数字会让 ``3.1rc1`` 冒充 ``3.1`` 越过下限）。
    """
    match = _VERSION_TOKEN.match(str(text).strip())
    if match is None:
        raise ReportingDependencyError(f"无法解析依赖版本号：{text!r}")
    numbers = tuple(int(part) for part in match.group("numbers").split("."))
    if not numbers:
        raise ReportingDependencyError(f"依赖版本号为空：{text!r}")
    raw_suffix = (match.group("suffix") or match.group("glued") or "").lstrip("._-").lower()
    if raw_suffix == "":
        return numbers, 1, 0
    ranked = _PRERELEASE_KIND.match(raw_suffix)
    if ranked is None:
        raise ReportingDependencyError(f"无法解析依赖版本后缀：{text!r}")
    if ranked.group("kind") == "post":
        return numbers, 2, int(ranked.group("seq") or 0)
    return numbers, 0, int(ranked.group("seq") or 0)


def _pad(parts: tuple[int, ...], width: int) -> tuple[int, ...]:
    return parts + (0,) * max(0, width - len(parts))


def _version_key(parsed: tuple[tuple[int, ...], int, int], width: int) -> tuple[int, ...]:
    numbers, rank, sequence = parsed
    return (*_pad(numbers, width), rank, sequence)


def version_in_bounds(version: str, minimum: str, maximum_exclusive: str) -> bool:
    """版本是否落在 ``[minimum, maximum_exclusive)``；预发布语义与 packaging specifier 对齐。

    除按 PEP 440 排序（预发布 < 正式版 < post）外：上限为无预发布正式版时，其同数字段
    预发布版（``4.0rc1`` 之于 ``<4.0``）不算落在区间内（评审 C1 续）。
    """
    current = parse_version(version)
    lower = parse_version(minimum)
    upper = parse_version(maximum_exclusive)
    width = max(len(item[0]) for item in (current, lower, upper))
    if not (
        _version_key(lower, width) <= _version_key(current, width) < _version_key(upper, width)
    ):
        return False
    upper_numbers, upper_rank, _ = upper
    current_numbers, current_rank, _ = current
    if (
        upper_rank == 1
        and current_rank == 0
        and _pad(current_numbers, width) == _pad(upper_numbers, width)
    ):
        return False
    return True


def load_jinja2() -> ModuleType:
    """导入 Jinja2 并校验兼容上下限；缺失或越界直接抛 ``ReportingDependencyError``。"""
    try:
        module = importlib.import_module("jinja2")
    except ImportError as exc:  # pragma: no cover - 由测试注入缺失场景覆盖
        raise ReportingDependencyError(
            "P6 报告生成依赖 Jinja2 缺失：请安装 "
            f"'jinja2>={JINJA2_MIN_VERSION},<{JINJA2_MAX_EXCLUSIVE}'；"
            "禁止退化为字符串拼接（设计 §6.7）"
        ) from exc
    version = str(getattr(module, "__version__", "") or "")
    if not version or not version_in_bounds(version, JINJA2_MIN_VERSION, JINJA2_MAX_EXCLUSIVE):
        raise ReportingDependencyError(
            "Jinja2 版本越界："
            f"实际 {version or 'unknown'}，要求 >={JINJA2_MIN_VERSION},<{JINJA2_MAX_EXCLUSIVE}"
        )
    return module


def build_environment(*, template_dir: Path | str | None = None) -> Any:
    """构造确定性 Jinja2 环境：严格未定义变量、自动转义、禁自动重载与模板缓存。

    ``template_dir`` 只允许显式传入固定模板目录；不提供任意目录搜索路径。
    """
    jinja2 = load_jinja2()
    loader = None
    if template_dir is not None:
        directory = Path(template_dir)
        if not directory.is_dir():
            raise ReportingDependencyError(f"模板目录不存在或不是目录：{directory}")
        loader = jinja2.FileSystemLoader(str(directory))
    return jinja2.Environment(
        loader=loader,
        autoescape=jinja2.select_autoescape(default=True, default_for_string=True),
        undefined=jinja2.StrictUndefined,
        auto_reload=False,
        cache_size=0,
        keep_trailing_newline=True,
        newline_sequence="\n",
        trim_blocks=True,
        lstrip_blocks=True,
    )
