# -*- coding: utf-8 -*-
"""P6 T0.5 验收环境采集（设计 §7.2；缺项由 ``perf_gate.validate_environment`` 判无效）。

只做只读探测：CPU/核数/RAM/OS 构建/Python/Jinja2/Playwright/Chromium/电源模式。
Chromium 版本由打开测量传入；生成测量不要求浏览器，可省略该两项。
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
if __name__ == "__main__":  # 验收进程禁写 __pycache__（设计 §2.5）；pytest 导入时不改全局
    sys.dont_write_bytecode = True
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.reporting import deps, validation  # noqa: E402

__all__ = ["capture_environment", "power_plan"]


def power_plan() -> str:
    """Windows 电源模式；非 Windows 记录明确的非适用标记（不伪造档位）。

    子进程按字节捕获后逐级解码（UTF-8 → GBK → replace 降级），不依赖机器
    控制台代码页（text=True 用 locale 解码会在代码页切换时抛 UnicodeDecodeError）。
    """
    if os.name != "nt":
        return "not_applicable_non_windows"
    try:
        completed = subprocess.run(
            ["powercfg", "/getactivescheme"],
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    raw = completed.stdout or completed.stderr or b""
    text = ""
    for encoding in ("utf-8", "gbk"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if not text:
        text = raw.decode("utf-8", errors="replace")
    return text.strip() or "unavailable"


def _cpu_memory() -> tuple[str, int, int]:
    try:
        import psutil

        cores = int(psutil.cpu_count(logical=False) or 0)
        ram_bytes = int(psutil.virtual_memory().total)
    except (ImportError, OSError, ValueError):
        cores, ram_bytes = 0, 0
    model = platform.processor().strip() or platform.machine().strip() or "unknown"
    return model, cores, ram_bytes


def capture_environment(chromium_version: str | None = None) -> dict[str, Any]:
    """采集验收环境；``chromium_version`` 为 None 时不记录浏览器两项（生成测量专用）。"""
    model, cores, ram_bytes = _cpu_memory()
    os_release = platform.release()
    os_build = platform.version()
    if os.name == "nt":
        win = platform.win32_ver()
        if len(win) > 1 and win[1]:
            os_release, os_build = win[0], win[1]
    record: dict[str, Any] = {
        "cpu_model": model,
        "cpu_cores": cores,
        "ram_bytes": ram_bytes,
        "os": platform.system() or "unknown",
        "os_build": os_build or "unknown",
        "os_release": os_release or "unknown",
        "python_version": platform.python_version(),
        "jinja2_version": getattr(deps.load_jinja2(), "__version__", ""),
        "power_plan": power_plan(),
        "captured_at_utc": validation.utc_stamp(),
    }
    try:
        record["playwright_version"] = importlib.metadata.version("playwright")
    except importlib.metadata.PackageNotFoundError:
        record["playwright_version"] = "not_installed"
    if chromium_version is not None:
        record["chromium_version"] = chromium_version
    return record
