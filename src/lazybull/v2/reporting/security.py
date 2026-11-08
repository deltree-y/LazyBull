# -*- coding: utf-8 -*-
"""报告安全基线常量（设计 §6：CSP、禁止标签、属性闭集、链接白名单）。

唯一来源：渲染器（``render``）用``CSP_CONTENT`` 写 meta，判定器（``html_check``）用同一
常量核对——两侧引用同一常量，禁止各自复制字符串（复制即会漂移成“自己判自己”）。

``ALLOWED_ATTRIBUTES`` 是**属性闭集**：本报告族的 HTML/SVG 只允许出现这些属性
（小写形态，与 ``HTMLParser`` 归一后一致）。未登记属性一律判违规——这条把
“来源文本被塞进属性位”这一类注入（``hidden``/``data-x``/夹带引号）变成结构上不可能，
而不是依赖转义是否覆盖引号。
"""

from __future__ import annotations

__all__ = [
    "ALLOWED_ATTRIBUTES",
    "ALLOWED_NAMESPACE_URLS",
    "CSP_CONTENT",
    "FORBIDDEN_TAGS",
    "MAX_SNAPSHOT_NODES",
    "URL_ATTRIBUTES",
]

#: 设计 §6.3 的最低 CSP（T2 冻结；放宽须重走安全门并重新取证）
CSP_CONTENT = (
    "default-src 'none'; style-src 'unsafe-inline'; img-src data:; font-src data:; "
    "base-uri 'none'; form-action 'none'"
)

#: 可执行 / 外链 / 交互类标签（出现即违规）
FORBIDDEN_TAGS: frozenset[str] = frozenset(
    {
        "applet",
        "audio",
        "base",
        "button",
        "embed",
        "form",
        "frame",
        "frameset",
        "iframe",
        "input",
        "link",
        "object",
        "script",
        "select",
        "source",
        "textarea",
        "track",
        "video",
    }
)

#: 链接/资源类属性（只允许文内锚点与固定 companion 文件名）
URL_ATTRIBUTES: frozenset[str] = frozenset(
    {"href", "src", "action", "formaction", "poster", "data", "xlink:href", "ping"}
)

#: 属性闭集（小写；``HTMLParser`` 已把 ``viewBox`` 归一为 ``viewbox``）
ALLOWED_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "aria-label",
        "charset",
        "class",
        "content",
        "cx",
        "cy",
        "data-status",
        "fill",
        "height",
        "href",
        "http-equiv",
        "id",
        "lang",
        "points",
        "r",
        "role",
        "text-anchor",
        "width",
        "x",
        "xmlns",
        "y",
        "viewbox",
    }
)

#: 允许出现在属性值里的命名空间声明（不产生请求；其余含 ``//`` 的值一律违规）
ALLOWED_NAMESPACE_URLS: frozenset[str] = frozenset(
    {"http://www.w3.org/2000/svg", "http://www.w3.org/1999/xhtml"}
)

#: DOM 快照节点上限（超出即标 truncated，避免证据文件被压力页撑爆）
MAX_SNAPSHOT_NODES = 4_000
