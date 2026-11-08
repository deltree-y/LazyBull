# -*- coding: utf-8 -*-
"""HTML DOM 安全扫描（设计 §6：CSP、零外部请求、只写文本节点）。

用途：P6-T2-01（恶意字段 DOM 快照）与“自包含 / 零外部请求”门的**机器可读判定器**，
被 pytest 合成夹具与验收驱动共用（同一实现，禁止两套口径）。

判定口径（``scan_html`` 返回 ``ok`` 为全部条件同时成立）：

1. 无 ``script``/``iframe``/``object``/``embed``/``applet``/``link``/``base``/``form``
   等可执行或外链标签；``meta`` 只允许**纯 charset** 或 **且仅一个** CSP；
2. **属性闭集**：属性名必须落在 ``security.ALLOWED_ATTRIBUTES`` 内（未登记即违规）；
   同元素**重复属性**即违规（浏览器保留第一个、dict 保留最后一个，二者分歧会被利用）；
3. 无 ``on*`` 事件属性、无 ``style`` 属性（CSS 只允许内联 ``<style>`` 块）；
4. 链接/资源属性只允许文内锚点 ``#...`` 与固定 companion 文件名；任何属性值含 ``//``
   且不是白名单命名空间声明即违规（覆盖 ``content="0;url=//host"`` 这类协议相对 URL）；
5. 全文无 http(s)/ws(s)/ftp 远程引用、无 CSS ``@import``/``url(http...)``；
6. 存在且**恰一个** CSP meta，内容与 ``security.CSP_CONTENT`` 完全一致。

扫描器本身只做解析，不改写 HTML、不写盘。
"""

from __future__ import annotations

from html.parser import HTMLParser
from typing import Any, Sequence

from src.lazybull.v2.reporting import perf_gate, security, text_safety

__all__ = [
    "CSP_CONTENT",
    "FORBIDDEN_TAGS",
    "MAX_SNAPSHOT_NODES",
    "URL_ATTRIBUTES",
    "scan_fragment",
    "scan_html",
    "text_snapshot",
]

#: CSP 唯一来源在 ``security``；此处转导出，保证判定器与渲染器同源
CSP_CONTENT = security.CSP_CONTENT
#: 禁止标签（转导出）
FORBIDDEN_TAGS = security.FORBIDDEN_TAGS
#: 链接/资源属性（转导出）
URL_ATTRIBUTES = security.URL_ATTRIBUTES
#: DOM 快照节点上限（转导出）
MAX_SNAPSHOT_NODES = security.MAX_SNAPSHOT_NODES

#: HTML 空元素（不进入文本节点的父元素栈）
_VOID_TAGS: frozenset[str] = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
     "param", "source", "track", "wbr"}
)


class _DomParser(HTMLParser):
    """最小 DOM 收集器：标签序列、逐标签属性表（含重复属性）、文本节点。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[str] = []
        self.elements: list[tuple[str, dict[str, str], tuple[str, ...]]] = []
        self.nodes: list[dict[str, str]] = []
        self._stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: Sequence[tuple[str, str | None]]) -> None:
        self._record(tag, attrs)
        if tag not in _VOID_TAGS:
            self._stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: Sequence[tuple[str, str | None]]) -> None:
        self._record(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        while self._stack:
            if self._stack.pop() == tag:
                break

    def handle_data(self, data: str) -> None:
        if not data or not data.strip():
            return
        if len(self.nodes) < MAX_SNAPSHOT_NODES:
            self.nodes.append(
                {
                    "tag": self._stack[-1] if self._stack else "#document",
                    "text": text_safety.normalize_text(data.strip()),
                }
            )

    def _record(self, tag: str, attrs: Sequence[tuple[str, str | None]]) -> None:
        """记录标签与属性；同一元素内重复属性名单独留证（不静默取其一）。"""
        self.tags.append(tag)
        material: dict[str, str] = {}
        duplicates: list[str] = []
        for name, value in attrs:
            key = str(name).lower()
            if key in material:
                duplicates.append(key)
                continue
            material[key] = "" if value is None else str(value)
        self.elements.append((tag, material, tuple(duplicates)))


def text_snapshot(html: str) -> list[dict[str, str]]:
    """DOM 文本节点快照（P6-T2-01 证据形态；节点已 NFC）。"""
    parser = _DomParser()
    parser.feed(html)
    parser.close()
    return parser.nodes


def _meta_problem(attrs: dict[str, str]) -> str | None:
    """``meta`` 只允许**纯 charset** 或 CSP；``refresh``/``name`` 等一律违规。

    先判 ``http-equiv``：只写 ``charset`` 不能豁免 ``refresh``（浏览器按 refresh 跳转，
    等于一次外部导航）。
    """
    http_equiv = attrs.get("http-equiv", "").strip().lower()
    if http_equiv == "content-security-policy":
        return None
    if http_equiv:
        return f"<meta> 非 CSP http-equiv：{http_equiv}"
    if set(attrs) == {"charset"}:
        return None
    return f"<meta> 非 charset/CSP 用途：{attrs}"


def _url_problem(tag: str, name: str, value: str) -> str | None:
    """链接/资源属性白名单：文内锚点或固定 companion 文件名。"""
    if name not in URL_ATTRIBUTES:
        return None
    text = value.strip()
    if text.startswith("#") or text in text_safety.COMPANION_ALLOWED:
        return None
    return f"{tag}@{name}={text!r} 不是文内锚点或固定 companion 文件名"


def _value_problem(tag: str, name: str, value: str) -> str | None:
    """任意属性值里的协议相对 URL（``//host``）与未登记命名空间。"""
    if "//" not in value:
        return None
    if value.strip() in security.ALLOWED_NAMESPACE_URLS:
        return None
    return f"{tag}@{name}={value!r} 含协议相对/远程地址"


def _classify_attribute(
    tag: str,
    name: str,
    value: str,
    *,
    events: set[str],
    styles: list[str],
    urls: list[str],
    unknown: set[str],
) -> None:
    """单个属性的类别判定（事件 / style / 未登记 / 链接越界 / 协议相对 URL）。"""
    if name.startswith("on"):
        events.add(f"{tag}@{name}")
    elif name == "style":
        styles.append(tag)
    elif name not in security.ALLOWED_ATTRIBUTES:
        unknown.add(f"{tag}@{name}")
    else:
        url_problem = _url_problem(tag, name, value)
        if url_problem:
            urls.append(url_problem)
    value_problem = _value_problem(tag, name, value)
    if value_problem:
        urls.append(value_problem)


def _element_problems(
    tag: str,
    attrs: dict[str, str],
    duplicates: Sequence[str],
    *,
    events: set[str],
    styles: list[str],
    urls: list[str],
    unknown: set[str],
) -> None:
    """单元素属性判定：重复属性 + 值层引号/实体 + 逐属性分类（嵌套保持浅层）。"""
    for name in duplicates:
        urls.append(f"{tag}@{name} 重复出现（浏览器与解析器取值不一致）")
    csp_meta = (
        tag == "meta"
        and attrs.get("http-equiv", "").strip().lower() == "content-security-policy"
    )
    for name, value in attrs.items():
        _classify_attribute(
            tag, name, value, events=events, styles=styles, urls=urls, unknown=unknown
        )
        if csp_meta and name == "content":
            continue  # 冻结 CSP 常量本身含单引号（唯一豁免）
        if any(token in value for token in ('"', "'", "&#")):
            urls.append(f"{tag}@{name} 属性值含引号/字符实体（禁止属性位承载来源文本）")


def _attribute_problems(
    elements: Sequence[tuple[str, dict[str, str], tuple[str, ...]]],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """逐元素属性判定：``(事件属性, style 属性, URL/重复问题, meta 问题, CSP 内容)``。"""
    events: set[str] = set()
    styles: list[str] = []
    urls: list[str] = []
    metas: list[str] = []
    unknown: set[str] = set()
    csp: list[str] = []
    for tag, attrs, duplicates in elements:
        _element_problems(
            tag,
            attrs,
            duplicates,
            events=events,
            styles=styles,
            urls=urls,
            unknown=unknown,
        )
        if tag != "meta":
            continue
        problem = _meta_problem(attrs)
        if problem:
            metas.append(problem)
        if attrs.get("http-equiv", "").strip().lower() == "content-security-policy":
            csp.append(attrs.get("content", ""))
    urls.extend(f"未登记属性 {item}" for item in sorted(unknown))
    return sorted(events), styles, urls, metas, csp


def scan_fragment(markup: str) -> dict[str, Any]:
    """片段级 DOM 安全判定（禁止标签/未登记属性/事件/style/URL），用于内联 SVG 注入前。"""
    parser = _DomParser()
    parser.feed(markup)
    parser.close()
    forbidden = sorted({tag for tag in parser.tags if tag in FORBIDDEN_TAGS})
    events, styles, urls, metas, _csp = _attribute_problems(parser.elements)
    offenders: list[str] = [f"禁止标签 <{tag}>" for tag in forbidden]
    if events:
        offenders.append(f"事件属性：{events}")
    if styles:
        offenders.append(f"style 属性：{sorted(set(styles))}")
    offenders.extend(urls)
    offenders.extend(metas)
    return {
        "schema": "p6_html_fragment_scan/v1",
        "tags": sorted(set(parser.tags)),
        "forbidden_tags": forbidden,
        "event_attributes": events,
        "url_problems": urls,
        "offenders": offenders,
        "ok": not offenders,
    }


def scan_html(html: str, *, csp: str = CSP_CONTENT) -> dict[str, Any]:
    """解析并判定 HTML 安全性（返回机器可读报告，``ok`` 为总判据）。"""
    parser = _DomParser()
    parser.feed(html)
    parser.close()
    forbidden = sorted({tag for tag in parser.tags if tag in FORBIDDEN_TAGS})
    events, styles, urls, metas, csp_contents = _attribute_problems(parser.elements)
    external_refs = perf_gate.scan_external_references(html)
    offenders: list[str] = [f"禁止标签 <{tag}>" for tag in forbidden]
    if events:
        offenders.append(f"事件属性：{events}")
    if styles:
        offenders.append(f"style 属性（CSS 只允许内联 <style> 块）：{sorted(set(styles))}")
    offenders.extend(urls)
    offenders.extend(metas)
    if external_refs:
        offenders.append(f"远程引用：{external_refs}")
    if not csp_contents:
        offenders.append("缺 Content-Security-Policy meta")
    elif csp_contents != [csp]:
        offenders.append(f"CSP 与冻结内容不一致（须恰一个且逐字相同）：{csp_contents}")
    return {
        "schema": "p6_html_scan/v1",
        "tags": sorted(set(parser.tags)),
        "tag_count": len(parser.tags),
        "forbidden_tags": forbidden,
        "event_attributes": events,
        "url_problems": urls,
        "meta_problems": metas,
        "external_references": external_refs,
        "csp": csp_contents,
        "csp_match": csp_contents == [csp],
        "text_nodes": len(parser.nodes),
        "text_snapshot_truncated": len(parser.nodes) >= MAX_SNAPSHOT_NODES,
        "offenders": offenders,
        "ok": not offenders,
    }
