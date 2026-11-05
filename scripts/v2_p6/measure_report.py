# -*- coding: utf-8 -*-
"""P6 T0.5 打开时间测量入口（设计 §7.2：5 冷 + 5 热，每次 < 2,000ms）。

冷 = 新 Chromium 进程 + 本次 validation 内全新 profile；热 = 独立持久 profile 预热后 reload。
资源协议（设计 §6.6）：除被测 HTML 的 file 导航（含热 reload）外零请求、零 WebSocket、
无本地子资源依赖。观测挂载在 **context 级**，覆盖被测页、popup/new page 及其首批请求
（评审 R2-T05-01 续），动态拼接 URL 亦不漏；任何子资源（含本地 ``file:``）与任何
WebSocket 都记为违规。原始观测逐 context 落盘（设计 §7.2），与推导 gate 分离。
写域（设计 §2.5）：profile 显式绑定本次 validation；Playwright/Chromium 可控临时目录
经 ``TMP/TEMP/TMPDIR`` 重定向进 validation，并以运行期子进程 ``--user-data-dir`` 命令行
证据、profile 落盘文件数与「OS TEMP 无新增 playwright 条目」三方复核。

Playwright 只在本脚本函数内部导入；``src/lazybull/v2/reporting/**`` 禁止依赖它。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
if __name__ == "__main__":  # 验收进程禁写 __pycache__（设计 §2.5）；pytest 导入时不改全局
    sys.dont_write_bytecode = True
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p6 import env_probe  # noqa: E402
from src.lazybull.v2.reporting import contracts, perf_gate, validation  # noqa: E402

__all__ = ["MeasureSetupError", "main", "measure"]

SCHEMA = "p6_t05_performance/v3"
#: CLI 退出码约定（记录进原始 JSON 供对照；实际退出码由执行层保存，评审 R2-T05-03 续）
_EXIT_CODE_CONVENTION = {"pass": 0, "fail": 3, "setup_error": 2}
#: 打开测量允许写的 reports 根（模块常量：测试可注入 tmp 根，禁止真实生产根写入）
_REPORTS_ROOT = (ROOT / contracts.REPORTS_ROOT).resolve()
#: OS TEMP 中 Playwright 可控临时目录的公共前缀（playwright-artifacts-* 等）
_PLAYWRIGHT_TEMP_PREFIX = "playwright-"
_NAVIGATION_SCRIPT = """
() => {
  const entry = performance.getEntriesByType('navigation')[0];
  if (entry && entry.loadEventEnd > 0) { return entry.loadEventEnd; }
  return performance.now();
}
"""


class MeasureSetupError(RuntimeError):
    """测量环境不满足（Playwright 缺失、输出越界、HTML 不存在、profile 不全新）。"""


def _require_sync_playwright() -> Any:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - 环境缺失分支由人工验收覆盖
        raise MeasureSetupError(
            "Playwright 缺失：请安装验收侧依赖 'playwright<2.0' 并执行 "
            "'python -m playwright install chromium'（生产生成器不得导入 Playwright）"
        ) from exc
    return sync_playwright


def _navigation_ms(page: Any) -> float:
    return float(page.evaluate(_NAVIGATION_SCRIPT))


def _new_run_record(mode: str, index: int) -> dict[str, Any]:
    """单个 context 的原始观测容器（覆盖该 context 全部页面/popup 的资源事件）。"""
    return {
        "mode": mode,
        "index": index,
        "navigations": [],
        "requests": [],
        "websockets": [],
        "failures": [],
        "pages": [],
    }


def _attach_context(context: Any, record: dict[str, Any]) -> None:
    """context 级观测（评审 R2-T05-01 续）：popup 与其首批请求一并捕获，无需按页挂卸。

    注意：Playwright 的 ``websocket`` 事件只在 page 级存在——在每个新页（含 popup）
    创建时挂载 page 级 WebSocket 监听（context.on("page") 对 new_page() 亦触发）。
    """

    def _on_request(request: Any) -> None:
        record["requests"].append(
            {
                "url": request.url,
                "method": request.method,
                "resource_type": request.resource_type,
                "is_navigation": bool(request.is_navigation_request()),
            }
        )

    def _on_websocket(socket: Any) -> None:
        record["websockets"].append({"url": socket.url})

    def _on_failed(request: Any) -> None:
        record["failures"].append(
            {"url": request.url, "error": str(request.failure) if request.failure else "unknown"}
        )

    def _on_page(page: Any) -> None:
        record["pages"].append(str(page.url))
        page.on("websocket", _on_websocket)

    context.on("request", _on_request)
    context.on("requestfailed", _on_failed)
    context.on("page", _on_page)


def _measure_navigation(
    page: Any,
    url: str,
    *,
    reload: bool,
    role: str,
    index: int,
    record: dict[str, Any],
) -> None:
    """单次导航计时；资源观测已在 context 级挂载。"""
    if reload:
        page.reload(wait_until="load", timeout=60_000)
    else:
        page.goto(url, wait_until="load", timeout=60_000)
    record["navigations"].append(
        {"index": index, "role": role, "navigation_url": url, "elapsed_ms": _navigation_ms(page)}
    )


def _dir_file_count(root: Path) -> int:
    return sum(1 for path in root.rglob("*") if path.is_file())


def _process_user_data_dirs() -> list[str]:
    """运行期证据：本进程树中 Chromium 子进程的 ``--user-data-dir`` 参数值。"""
    try:
        import psutil
    except ImportError as exc:  # pragma: no cover - psutil 为项目依赖，缺失分支兜底
        return [f"<unavailable: ImportError {exc}>"]
    try:
        children = psutil.Process().children(recursive=True)
    except OSError as exc:
        return [f"<unavailable: {exc.__class__.__name__}>"]
    dirs: set[str] = set()
    for child in children:
        try:
            cmdline = child.cmdline()
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
        for argument in cmdline:
            if argument.startswith("--user-data-dir="):
                dirs.add(argument.split("=", 1)[1])
    return sorted(dirs)


def _playwright_temp_entries(tmp_root: Path) -> set[str]:
    """OS TEMP 中 Playwright 可控临时条目名快照（playwright-artifacts-* 等）。"""
    return {path.name for path in tmp_root.glob(f"{_PLAYWRIGHT_TEMP_PREFIX}*")}


def _launch(profile_dir: Path, playwright: Any) -> Any:
    """持久上下文：浏览器 profile 绑定本次 validation；可控临时目录经环境变量重定向。"""
    return playwright.chromium.launch_persistent_context(
        str(profile_dir), headless=True, viewport={"width": 1280, "height": 900}
    )


def _cold_run_once(
    playwright: Any, url: str, index: int, profile_root: Path
) -> tuple[dict[str, Any], str, list[str]]:
    """单次冷测量：全新 profile + 新进程；profile 预存在即拒绝（冷语义守卫）。"""
    profile_dir = profile_root / f"cold_{index}"
    if profile_dir.exists():
        raise MeasureSetupError(f"冷启动 profile 必须全新，已存在：{profile_dir}")
    profile_dir.mkdir()
    context = _launch(profile_dir, playwright)
    record = _new_run_record("cold", index)
    try:
        _attach_context(context, record)
        page = context.new_page()
        _measure_navigation(page, url, reload=False, role="measured", index=0, record=record)
        browser = context.browser
        version = browser.version if browser is not None else ""
        user_data_dirs = _process_user_data_dirs()
    finally:
        context.close()
    record["profile_dir"] = profile_dir.as_posix()
    record["profile_files"] = _dir_file_count(profile_dir)
    return record, version, user_data_dirs


def _cold_runs(
    playwright: Any, url: str, repeat: int, profile_root: Path
) -> tuple[list[dict[str, Any]], str, list[str]]:
    records: list[dict[str, Any]] = []
    user_data_dirs: list[str] = []
    chromium_version = ""
    for index in range(repeat):
        record, version, dirs = _cold_run_once(playwright, url, index, profile_root)
        records.append(record)
        user_data_dirs.extend(dirs)
        chromium_version = chromium_version or version
    return records, chromium_version, user_data_dirs


def _hot_runs(
    playwright: Any, url: str, repeat: int, profile_root: Path
) -> tuple[list[dict[str, Any]], str, list[str]]:
    """热测量：独立持久 profile，预热导航后连续 reload；全部导航纳入计时与资源观测。"""
    profile_dir = profile_root / "hot"
    if profile_dir.exists():
        raise MeasureSetupError(f"热测量 profile 必须全新，已存在：{profile_dir}")
    profile_dir.mkdir()
    context = _launch(profile_dir, playwright)
    record = _new_run_record("hot_series", 0)
    try:
        _attach_context(context, record)
        page = context.new_page()
        _measure_navigation(page, url, reload=False, role="warmup", index=0, record=record)
        for index in range(repeat):
            _measure_navigation(
                page, url, reload=True, role="measured", index=index + 1, record=record
            )
        browser = context.browser
        chromium_version = browser.version if browser is not None else ""
        user_data_dirs = _process_user_data_dirs()
    finally:
        context.close()
    record["profile_dir"] = profile_dir.as_posix()
    record["profile_files"] = _dir_file_count(profile_dir)
    return [record], chromium_version, user_data_dirs


def _timing_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """抽取计时序列：仅 measured 导航（warmup 只观测资源不计打开耗时），恰 5 冷 + 5 热。"""
    timing: list[dict[str, Any]] = []
    for record in records:
        mode = "cold" if record["mode"] == "cold" else "hot"
        for navigation in record["navigations"]:
            if navigation["role"] == "measured":
                timing.append({"mode": mode, "elapsed_ms": navigation["elapsed_ms"]})
    return timing


def _classify_offenders(records: list[dict[str, Any]], allowed_uri: str) -> list[str]:
    """§6.6 协议分类：仅允许被测 HTML 的导航请求；其余请求与一切 WebSocket 均违规。"""
    offenders: set[str] = set()
    for record in records:
        for entry in record["requests"]:
            if entry["is_navigation"] and entry["url"] == allowed_uri:
                continue
            offenders.add(entry["url"])
        for socket in record["websockets"]:
            offenders.add(socket["url"])
    return sorted(offenders)


def _judge_write_domain(
    process_dirs: list[str],
    os_temp_new: set[str],
    target: Path,
    profile_root: Path,
    tmp_root: Path,
) -> dict[str, Any]:
    """浏览器写域证据门（设计 §2.5，评审 R2-T05-02 续）。

    三方证据：运行期 ``--user-data-dir`` 全部位于本次 validation、profile 有实际落盘、
    OS TEMP 在测量期间无新增 Playwright 条目（可控临时目录已重定向进 validation）。
    """
    inside = [item for item in process_dirs if Path(item).is_relative_to(target)]
    outside = [item for item in process_dirs if not Path(item).is_relative_to(target)]
    profile_files = _dir_file_count(profile_root)
    return {
        "kind": "browser_write_domain",
        "process_user_data_dirs": process_dirs,
        "inside_validation": sorted(inside),
        "outside_validation": sorted(outside),
        "profile_files_total": profile_files,
        "redirected_tmp_root": tmp_root.as_posix(),
        "os_temp_new_entries": sorted(os_temp_new),
        "passed": (
            bool(process_dirs) and not outside and profile_files > 0 and not os_temp_new
        ),
    }


def _assert_new_validation_out(out: Path) -> Path:
    reports_root = _REPORTS_ROOT
    target = out.resolve()
    if target.exists():
        raise MeasureSetupError(f"输出目录已存在，拒绝覆盖：{target}")
    validations_root = reports_root / "validations"
    try:
        relative = target.relative_to(validations_root)
    except ValueError as exc:
        raise MeasureSetupError(
            f"打开测量只允许写 {validations_root} 之下：{target}"
        ) from exc
    evidence_class = relative.parts[0] if relative.parts else ""
    if evidence_class not in contracts.EVIDENCE_CLASSES:
        raise MeasureSetupError(
            f"validation 根形态不合法（evidence_class 须 ∈ "
            f"{sorted(contracts.EVIDENCE_CLASSES)}）：{target}"
        )
    return target


def measure(
    html: Path | str,
    out: Path | str,
    *,
    repeat: int = contracts.THRESHOLDS["cold_repeats"],
) -> dict[str, Any]:
    """执行 5 冷 + 5 热测量，原始观测与判定写入新 validation 目录。"""
    if repeat != contracts.THRESHOLDS["cold_repeats"]:
        raise MeasureSetupError(
            f"冷/热测量次数固定为 {contracts.THRESHOLDS['cold_repeats']}，收到 {repeat}"
        )
    html_path = Path(html).resolve()
    if not html_path.is_file():
        raise MeasureSetupError(f"HTML 不存在：{html_path}")
    target = _assert_new_validation_out(Path(out))
    html_text = html_path.read_text(encoding="utf-8")
    html_sha_before = validation.sha256_file(html_path)
    byte_size = html_path.stat().st_size
    validation.ensure_new_dir(target)
    profile_root = target / "browser_profiles"
    validation.ensure_new_dir(profile_root)
    tmp_root = target / "tmp"
    validation.ensure_new_dir(tmp_root)

    real_tmp = Path(tempfile.gettempdir())
    os_temp_before = _playwright_temp_entries(real_tmp)
    saved = {key: os.environ.get(key) for key in ("TMP", "TEMP", "TMPDIR")}
    try:
        # 评审 R2-T05-02 续：driver 缺省在 os.tmpdir() 下建 playwright-artifacts-*，
        # 启动前将可控临时目录整体重定向进本次 validation，运行后核对 OS TEMP 零新增。
        for key in ("TMP", "TEMP", "TMPDIR"):
            os.environ[key] = str(tmp_root)
        sync_playwright = _require_sync_playwright()
        with sync_playwright() as playwright:
            cold, cold_version, cold_dirs = _cold_runs(
                playwright, html_path.as_uri(), repeat, profile_root
            )
            hot, hot_version, hot_dirs = _hot_runs(
                playwright, html_path.as_uri(), repeat, profile_root
            )
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    os_temp_new = _playwright_temp_entries(real_tmp) - os_temp_before

    environment = env_probe.capture_environment(cold_version or hot_version)
    open_gate = perf_gate.judge_open_series(_timing_records([*cold, *hot]))
    size_gate = perf_gate.judge_size(byte_size)
    external_refs = perf_gate.scan_external_references(html_text)
    environment_gate = perf_gate.validate_environment(environment)
    versions_gate = perf_gate.verify_acceptance_versions(
        {
            "jinja2": environment.get("jinja2_version"),
            "playwright": environment.get("playwright_version"),
            "chromium": environment.get("chromium_version"),
        }
    )
    offenders = _classify_offenders([*cold, *hot], html_path.as_uri())
    request_total = sum(len(record["requests"]) for record in [*cold, *hot])
    navigation_total = sum(
        1 for record in [*cold, *hot] for entry in record["requests"] if entry["is_navigation"]
    )
    websocket_total = sum(len(record["websockets"]) for record in [*cold, *hot])
    write_domain_gate = _judge_write_domain(
        sorted({*cold_dirs, *hot_dirs}), os_temp_new, target, profile_root, tmp_root
    )
    html_sha_after = validation.sha256_file(html_path)

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "html": {
            "path": str(html_path),
            "sha256_before": html_sha_before,
            "sha256_after": html_sha_after,
            "bytes": byte_size,
            "unchanged": html_sha_before == html_sha_after,
            "external_references": external_refs,
        },
        "url": html_path.as_uri(),
        "observations": {"cold": cold, "hot": hot},
        "exit_code_convention": dict(_EXIT_CODE_CONVENTION),
        "gates": {
            "size": size_gate,
            "open": open_gate,
            "requests": {
                "total_requests": request_total,
                "navigation_requests": navigation_total,
                "websocket_count": websocket_total,
                "offenders": offenders,
                "passed": not offenders,
            },
            "environment": environment_gate,
            "acceptance_versions": versions_gate,
            "browser_write_domain": write_domain_gate,
        },
        "environment": environment,
        "protocol_reference_only": False,
    }
    payload["verdict"] = (
        "pass"
        if all(
            (
                size_gate["passed"],
                open_gate["passed"],
                not offenders,
                not external_refs,
                environment_gate["passed"],
                versions_gate["passed"],
                write_domain_gate["passed"],
                payload["html"]["unchanged"],
            )
        )
        else "fail"
    )
    validation.write_json_new(target / "performance.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P6 T0.5 打开时间测量（5 冷 + 5 热）")
    parser.add_argument("--html", required=True, help="待测的固定 HTML 文件")
    parser.add_argument("--repeat", type=int, default=contracts.THRESHOLDS["cold_repeats"])
    parser.add_argument(
        "--out", required=True, help="新的 validation 目录（须位于 data/reports/p6/validations/）"
    )
    args = parser.parse_args(argv)
    try:
        payload = measure(args.html, args.out, repeat=args.repeat)
    except MeasureSetupError as exc:
        print(f"[P6-T0.5] 打开测量未执行：{exc}", file=sys.stderr)
        return 2
    print(json.dumps({"verdict": payload["verdict"], "out": str(args.out)}, ensure_ascii=False))
    return 0 if payload["verdict"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
