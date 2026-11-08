# -*- coding: utf-8 -*-
"""P6 T2 注入、CSP 与零外部请求（注入与 CSP 3 项，设计 §6.2~§6.6）。

关键证据 P6-T2-01：把恶意 HTML/SVG/script/URL 放进**每个来源字段**后，DOM 中它们只能
作为**文本节点**出现——没有任何 script/事件属性/URL 可用位；同时校验 CSP 判定器不是
空转（改宽/删除 CSP 必须被判失败）。
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.reporting import (  # noqa: E402
    html_check,
    perf_gate,
    publish,
    render,
    security,
    text_safety,
)
from tests.p6_t1_env import build_paper_root, build_projection  # noqa: E402
from tests.p6_t2_env import (  # noqa: E402
    MALICIOUS_TEXTS,
    build_view,
    malicious_projection,
    render_projection,
)


def _real_html(tmp_path: Path) -> str:
    built = build_paper_root(tmp_path / "paper")
    projection, _snapshot, _audit, _info = build_projection(
        built["root"], tmp_path / "staging", attempt_id="attempt_security"
    )
    return render_projection(projection)


def test_p6_t2_01_malicious_fields_render_as_text_only(tmp_path: Path) -> None:
    """P6-T2-01：恶意字段 DOM 快照——只作文本节点，无 script/事件/URL 可用位。"""
    projection = malicious_projection()
    html = render_projection(projection)
    scan = html_check.scan_html(html)
    assert scan["ok"], scan["offenders"]
    assert scan["forbidden_tags"] == []
    assert scan["event_attributes"] == []
    assert scan["url_problems"] == []

    nodes = html_check.text_snapshot(html)
    texts = [node["text"] for node in nodes]
    for payload in MALICIOUS_TEXTS:
        normalized = unicodedata.normalize("NFC", payload)
        assert any(normalized in text for text in texts), f"注入串未作为文本出现：{payload}"
    # U+2028/U+2029 必须以字符引用落地（否则源文本可伪造换行/断句）
    assert "&#8232;" in html and "&#8233;" in html
    assert "\u2028" not in html and "\u2029" not in html
    # NFC：分解形态必须被合成为单个字符
    assert "nfc-é-decomposed" in html
    assert "nfc-e\u0301-decomposed" not in html
    # 逐项：可执行片段不会以“标签/属性”形式出现（只允许作为已转义文本）
    assert "<script" not in html.lower()
    assert "<iframe" not in html.lower()
    assert "<img" not in html.lower()
    assert "<svg><script" not in html.lower()
    assert "javascript:alert" not in html.lower()
    # 原始尖括号形态必须已转义（文本里只保留实体）
    assert "&lt;script&gt;" in html
    assert 'onclick=alert(1)' in html  # 文本可见（不隐藏来源），但不是属性
    assert "https://" not in html  # URL 方案字面量已中和（可见文本不变）

    # 渲染侧自检在注入场景下同样通过（发布前 fail-closed，不是验收脚本才拦）
    assert render.render_dashboard_html(build_view(projection))


def test_p6_t2_05_csp_exact_and_gate_not_vacuous(tmp_path: Path) -> None:
    """CSP 精确匹配冻结内容；改宽/删除 CSP 必须被判失败（判定器非空转）。"""
    html = _real_html(tmp_path)
    scan = html_check.scan_html(html)
    assert scan["ok"], scan["offenders"]
    assert scan["csp"] == [security.CSP_CONTENT]
    assert scan["csp_match"] is True
    assert security.CSP_CONTENT == (
        "default-src 'none'; style-src 'unsafe-inline'; img-src data:; font-src data:; "
        "base-uri 'none'; form-action 'none'"
    )

    widened = html.replace("default-src 'none'", "default-src *")
    widened_scan = html_check.scan_html(widened)
    assert widened_scan["ok"] is False
    assert any("CSP" in item for item in widened_scan["offenders"])

    removed = re.sub(r"<meta http-equiv=\"Content-Security-Policy\"[^>]*>\n", "", html)
    removed_scan = html_check.scan_html(removed)
    assert removed_scan["ok"] is False
    assert "缺 Content-Security-Policy meta" in removed_scan["offenders"]

    # 事件属性 / 禁止标签 / 外链的判定器负例（防止“只测正常路径”）
    for tampered in (
        html.replace("</section>", '<script>alert(1)</script></section>', 1),
        html.replace("<h2>", '<h2 onclick="alert(1)">', 1),
        html.replace('href="projection.json"', 'href="https://evil.example/p.json"'),
    ):
        assert html_check.scan_html(tampered)["ok"] is False


def test_p6_t2_06_zero_external_requests_and_fixed_companions(tmp_path: Path) -> None:
    """零 HTTP(S)/WebSocket 引用；链接只允许文内锚点与固定 companion 文件名。"""
    html = _real_html(tmp_path)
    assert perf_gate.scan_external_references(html) == []
    hrefs = re.findall(r'href="([^"]*)"', html)
    assert hrefs, "至少应有 companion 链接与目录锚点"
    assert all(item.startswith("#") or item in text_safety.COMPANION_FILES for item in hrefs)
    companions = [item for item in hrefs if not item.startswith("#")]
    assert sorted(set(companions)) == sorted(text_safety.COMPANION_FILES)
    assert all(item.endswith(".json") for item in companions)
    # 本版不内联任何 base64 图片/字体（体积与来源治理）
    assert "data:image" not in html and "@font-face" not in html
    # 除 SVG 命名空间声明外，全文不得出现任何 scheme URL
    without_namespace = html.replace("http://www.w3.org/2000/svg", "")
    assert "http://" not in without_namespace and "https://" not in without_namespace
    assert not perf_gate.scan_request_urls(["file:///tmp/report.html", "about:blank"])
    assert perf_gate.scan_request_urls(
        ["https://evil.example/x", "wss://evil.example/ws", "file:///tmp/report.html"]
    ) == ["https://evil.example/x", "wss://evil.example/ws"]

    # 恶意 URL 只作文本，不得成为任何 href；可见文本保留原值（不隐藏来源）
    malicious = render_projection(malicious_projection())
    malicious_hrefs = re.findall(r'href="([^"]*)"', malicious)
    assert all("evil.example" not in item for item in malicious_hrefs)
    malicious_texts = [node["text"] for node in html_check.text_snapshot(malicious)]
    assert any("https://evil.example/payload" in text for text in malicious_texts)
    assert "https://" not in malicious  # 方案字面量已中和为字符引用

    # SVG 注入面的结构校验（禁止脚本/事件/外链进入 SVG）
    for bad_svg in (
        "<svg onload=alert(1)></svg>",
        '<svg><script>alert(1)</script></svg>',
        '<svg href="https://evil.example"></svg>',
        '<svg aria-label="x"><rect onclick="alert(1)" width="1"/></svg>',
        "<div>not-svg</div>",
    ):
        with pytest.raises(ValueError):
            render._assert_svg(bad_svg)


def test_p6_t2_21_attribute_allowlist_and_meta_rules(tmp_path: Path) -> None:
    """属性闭集 + 重复属性 + meta 规则 + 协议相对 URL（判定器不得空转）。"""
    html = _real_html(tmp_path)
    scan = html_check.scan_html(html)
    assert scan["ok"], scan["offenders"]

    # 属性逃逸：未登记属性一律违规（把“来源进属性”变成结构上不可能）
    for fragment in (
        '<polyline hidden points="1,1"/>',
        '<circle data-x="y" r="1"/>',
        '<rect onclick="alert(1)" width="1"/>',
        '<text class="a" class="b">x</text>',  # 重复属性：浏览器取第一个、解析器取最后一个
    ):
        report = html_check.scan_fragment(fragment)
        assert report["ok"] is False, fragment
        assert report["offenders"]

    # 整页级：属性闭集在真实页面上同样生效
    for tampered in (
        html.replace("<h2>", '<h2 hidden>', 1),
        html.replace("<h2>", '<h2 data-x="y">', 1),
        html.replace("<h2>", '<h2 class="a" class="b">', 1),
    ):
        assert html_check.scan_html(tampered)["ok"] is False

    # meta 规则：refresh 不得被 charset 短路；多重/放宽 CSP 必须被拒
    head = (
        '<html><head><meta charset="utf-8">'
        f'<meta http-equiv="Content-Security-Policy" content="{security.CSP_CONTENT}">'
        "</head><body>"
    )
    assert html_check.scan_html(head + "<p>x</p></body></html>")["ok"] is True
    refresh = head.replace(
        '<meta charset="utf-8">',
        '<meta charset="utf-8" http-equiv="refresh" content="0;url=//evil.example/x">',
    )
    assert html_check.scan_html(refresh + "<p>x</p></body></html>")["ok"] is False
    doubled = head + (
        f'<meta http-equiv="Content-Security-Policy" content="{security.CSP_CONTENT}">'
        "<p>x</p></body></html>"
    )
    assert html_check.scan_html(doubled)["ok"] is False
    protocol_relative = head + '<script src="//evil.example/x.js"></script></body></html>'
    assert html_check.scan_html(protocol_relative)["ok"] is False

    # 属性值层：引号/字符实体不得出现在属性值里（CSP 常量是唯一豁免）
    for fragment in (
        '<p class="a&#39;b">x</p>',
        '<p class="a&quot;b">x</p>',
        '<text aria-label="x\'y">z</text>',
        '<text aria-label="x&#x27;y">z</text>',  # 十六进制实体同样反解后再判
        '<p class="a&amp;#x27;b">x</p>',  # 双重编码：解析值是字面 `&#x27;`
    ):
        assert html_check.scan_fragment(fragment)["ok"] is False, fragment
    csp_meta = (
        '<meta http-equiv="Content-Security-Policy" content="'
        f"{security.CSP_CONTENT}\">"
    )
    assert html_check.scan_fragment(csp_meta)["ok"] is True

    # companion 闭集：页面链接必须都能落进 run 文件集（否则是死链）
    assert set(text_safety.COMPANION_FILES) <= set(publish.RUN_FILENAMES)
