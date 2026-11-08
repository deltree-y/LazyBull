# -*- coding: utf-8 -*-
"""Jinja2 渲染入口（设计 §6.1 / §6.7：自包含、零外部依赖、确定性）。

硬口径（P6 T2）：

- 依赖门：Jinja2 缺失或版本越界由 ``deps.load_jinja2`` fail-closed（禁止字符串拼接降级）；
- 环境固定：``StrictUndefined`` + 自动转义 + 关闭自动重载/模板缓存 + LF；
- 来源文本只经 ``p6text`` 过滤器（NFC + 转义 ``< > & U+2028 U+2029``）进入**文本节点**；
- SVG 只允许经 ``p6svg`` 注入，且必须先通过结构校验（本模块内部生成，禁止来源文本进入）；
- 渲染结果接受 ``html_check`` 自检（CSP / 禁止标签 / 事件属性 / 远程引用）；不通过即抛错，
  **不发布**——安全门在发布前 fail-closed，而不是等验收脚本才发现。
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from markupsafe import Markup

from src.lazybull import PROJECT_ROOT
from src.lazybull.v2.reporting import deps, html_check, security, text_safety

__all__ = [
    "ENTRYPOINT_PATH",
    "GENERATOR_FILES",
    "SHARED_GENERATOR_FILES",
    "TEMPLATE_DIR",
    "TEMPLATE_NAME",
    "TEMPLATE_NAMES",
    "build_environment",
    "entrypoint_identity",
    "generator_file_sha256",
    "render_dashboard_bytes",
    "render_dashboard_html",
    "render_experiment_bytes",
    "render_report_bytes",
    "template_sha256",
]

#: 模板目录（包内固定位置，不提供任意搜索路径）
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
#: 驾驶舱模板名（保留常量以兼容 T2 调用方）
TEMPLATE_NAME = "dashboard.html.j2"
#: 报告类型 → 模板名（闭集：未登记的类型 fail-closed）
TEMPLATE_NAMES: dict[str, str] = {
    "dashboard": TEMPLATE_NAME,
    "experiment": "experiment.html.j2",
    "trend": "trend.html.j2",
}
#: generator 记录里必须逐文件登记 SHA 的生成器文件（相对 reporting 包根）。
#: 覆盖**决定输出字节**的全部环节：projection 生成链（authority/cockpit/config_view/
#: instructions/quality/status/snapshot）→ 契约/验收门 → 视图/图元/安全 → 渲染 → manifest → 发布。
#: T3 补齐 ``contracts.py``/``validation.py``/``perf_gate.py``（T2 遗留 R6-1）与实验面模块。
GENERATOR_FILES: tuple[str, ...] = (
    "authority.py",
    "cockpit.py",
    "config_view.py",
    "contracts.py",
    "csv_core.py",
    "dashboard_sections.py",
    "dashboard_view.py",
    "deps.py",
    "evidence.py",
    "experiment_csv.py",
    "experiment_guard.py",
    "experiment_metrics.py",
    "experiment_projection.py",
    "experiment_sources.py",
    "experiment_stability.py",
    "experiment_view.py",
    "generate.py",
    "html_check.py",
    "instructions.py",
    "manifest.py",
    "perf_gate.py",
    "publish.py",
    "quality.py",
    "render.py",
    "security.py",
    "snapshot.py",
    "status.py",
    "svg.py",
    "text_safety.py",
    "trend_contract.py",
    "trend_csv.py",
    "trend_metrics.py",
    "trend_observe.py",
    "trend_portability.py",
    "trend_projection.py",
    "trend_sources.py",
    "trend_svg.py",
    "trend_view.py",
    "validation.py",
    "view_primitives.py",
    "templates/dashboard.html.j2",
    "templates/experiment.html.j2",
    "templates/trend.html.j2",
)
#: T3 实际消费的共享唯一实现（repo-relative）：决定指标/CSV 字节，必须逐文件登记 SHA。
#: dirty 标志不能替代该清单（R3-10）。
SHARED_GENERATOR_FILES: tuple[str, ...] = (
    "src/lazybull/v2/common/chain_metrics.py",
    "src/lazybull/v2/evidence/fingerprint_keys.py",
    "src/lazybull/v2/evidence/runs_loader.py",
    "src/lazybull/v2/evidence/runs_schema.py",
    "scripts/compare/aggregate.py",
    # 评分权重与列映射直接决定 stability 数值（R3-10-R1）
    "scripts/compare/constants.py",
    "scripts/compare/fold_subset.py",
    "scripts/compare/report.py",
    "scripts/compare/scoring.py",
    # T4 趋势面实际消费的宿主唯一实现（纸面年化/配置展平决定 production 指标与身份）
    "src/lazybull/v2/hosts/paper/config_adapter.py",
    "src/lazybull/v2/hosts/paper/performance.py",
)
#: 生成入口脚本（repo 相对路径；SHA 单独登记在 ``generator.entrypoint``）
ENTRYPOINT_PATH = "scripts/p6_report.py"
_PACKAGE_ROOT = Path(__file__).resolve().parent


def _text_filter(value: Any) -> Markup:
    """模板过滤器 ``p6text``：NFC + 文本转义，返回 Markup 防止二次转义。"""
    return Markup(text_safety.escape_text(value))


def _assert_svg(value: Any) -> str:
    """SVG 注入前结构校验：必须是单个 ``<svg>`` 元素，且 DOM 无脚本/事件/可执行 URL。

    校验走 DOM 解析而非字符串包含判定：来源文本可能被转义成 ``&lt;img src=...&gt;``
    这类**文本节点**，字符串包含判定会把它误判为可执行片段（也会掩盖真正的属性注入）。
    """
    text = str(value or "")
    stripped = text.strip()
    if not stripped.startswith("<svg") or not stripped.endswith("</svg>"):
        raise ValueError("SVG 注入必须是单个 <svg>...</svg> 元素")
    report = html_check.scan_fragment(stripped)
    if not report["ok"]:
        raise ValueError(f"SVG 注入含可执行/外链结构：{report['offenders']}")
    return stripped


def _svg_filter(value: Any) -> Markup:
    """模板过滤器 ``p6svg``：仅用于本模块生成的静态 SVG（先校验后放行）。"""
    return Markup(_assert_svg(value))


def build_environment() -> Any:
    """构造确定性 Jinja2 环境并注册 ``p6text``/``p6svg`` 两个过滤器。"""
    env = deps.build_environment(template_dir=TEMPLATE_DIR)
    env.filters["p6text"] = _text_filter
    env.filters["p6svg"] = _svg_filter
    return env


def _render_html(view: Mapping[str, Any], template_name: str) -> str:
    """按模板渲染 HTML（UTF-8 / LF / 结尾单换行 + **强制**发布前安全自检）。

    自检不可关闭：``html_check.scan_html`` 不通过即抛错、不返回字节（契约要求
    “任一命中即抛错，不发布”）。
    """
    if not isinstance(view, Mapping):
        raise ValueError(f"视图模型必须是对象：{type(view).__name__}")
    payload = dict(view)
    # CSP 是常量而非来源文本：标为 safe，避免自动转义把引号写成实体（浏览器解码后虽等价，
    # 但原始字节与判定器/审计的期望值应当一致）
    payload["csp"] = Markup(security.CSP_CONTENT)
    template = build_environment().get_template(template_name)
    html = template.render(**payload)
    if "\r" in html:
        raise ValueError("渲染结果含 CR（必须 LF 换行）")
    if not html.endswith("\n"):
        html += "\n"
    report = html_check.scan_html(html)
    if not report["ok"]:
        raise ValueError(f"渲染结果未通过安全自检：{report['offenders']}")
    return html


def render_dashboard_html(view: Mapping[str, Any]) -> str:
    """渲染驾驶舱 HTML（固定模板 + 强制安全自检）。"""
    return _render_html(view, TEMPLATE_NAME)


def render_report_bytes(view: Mapping[str, Any], *, report_kind: str) -> bytes:
    """按报告类型渲染为 UTF-8 字节（未登记类型 fail-closed，不退化为驾驶舱模板）。"""
    template_name = TEMPLATE_NAMES.get(str(report_kind))
    if template_name is None:
        raise ValueError(f"未登记的报告类型：{report_kind!r}（合法 {sorted(TEMPLATE_NAMES)}）")
    return _render_html(view, template_name).encode("utf-8")


def render_experiment_bytes(view: Mapping[str, Any]) -> bytes:
    """渲染实验报告为 UTF-8 字节（与驾驶舱共用安全自检与 LF 口径）。"""
    return render_report_bytes(view, report_kind="experiment")


def render_dashboard_bytes(view: Mapping[str, Any]) -> bytes:
    """渲染为 UTF-8 字节（无 BOM；字节一致性的判定对象）。"""
    return render_dashboard_html(view).encode("utf-8")


def template_sha256(name: str = TEMPLATE_NAME) -> str:
    """模板文件字节 SHA-256（manifest generator 字段逐文件登记）。"""
    path = TEMPLATE_DIR / name
    return sha256(path.read_bytes()).hexdigest()


def generator_file_sha256() -> dict[str, str]:
    """生成器/模板逐文件 SHA-256（设计 §8.1：dirty 不替代文件哈希）。"""
    digests: dict[str, str] = {}
    for relative in GENERATOR_FILES:
        path = _PACKAGE_ROOT / relative
        if not path.is_file():
            raise ValueError(f"generator 文件缺失，无法登记 SHA：{relative}")
        digests[f"src/lazybull/v2/reporting/{relative}"] = sha256(path.read_bytes()).hexdigest()
    for relative in SHARED_GENERATOR_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise ValueError(f"共享生成器文件缺失，无法登记 SHA：{relative}")
        digests[relative] = sha256(path.read_bytes()).hexdigest()
    return digests


def entrypoint_identity() -> dict[str, str]:
    """生成入口脚本身份（``scripts/p6_report.py`` 的 repo 相对路径 + 字节 SHA）。"""
    path = PROJECT_ROOT / ENTRYPOINT_PATH
    if not path.is_file():
        return {"path": ENTRYPOINT_PATH, "sha256": "<missing>"}
    return {"path": ENTRYPOINT_PATH, "sha256": sha256(path.read_bytes()).hexdigest()}
