# -*- coding: utf-8 -*-
"""P6 T2 自包含与字节形态（自包含 1 项，设计 §6.1 / §6.2 / §8.2）。

单页 HTML 必须：UTF-8 无 BOM、LF 换行、结尾单换行；CSS/SVG 内联、系统字体、
无 `<script>`/`<link>`/`@import`/外部字体，任何本地子资源都不依赖。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.reporting import html_check, text_safety  # noqa: E402
from tests.p6_t1_env import build_paper_root, build_projection  # noqa: E402
from tests.p6_t2_env import render_projection  # noqa: E402


def test_p6_t2_07_self_contained_single_file(tmp_path: Path) -> None:
    """单文件自包含：内联 CSS/SVG、系统字体、无脚本/外链、UTF-8 无 BOM + LF。"""
    built = build_paper_root(tmp_path / "paper")
    projection, _snapshot, _audit, _info = build_projection(
        built["root"], tmp_path / "staging", attempt_id="attempt_selfcontained"
    )
    html = render_projection(projection)
    payload = html.encode("utf-8")

    assert not payload.startswith(b"\xef\xbb\xbf"), "禁止 UTF-8 BOM"
    assert b"\r" not in payload, "必须 LF 换行"
    assert payload.endswith(b"\n")
    assert payload.decode("utf-8") == html
    assert html.startswith("<!DOCTYPE html>")

    scan = html_check.scan_html(html)
    assert scan["ok"], scan["offenders"]
    assert "script" not in scan["tags"]
    assert "link" not in scan["tags"]
    assert scan["tags"].count("style") == 1
    assert "svg" in scan["tags"]
    assert "@import" not in html
    assert "@font-face" not in html
    assert "url(http" not in html.lower()
    # 系统字体栈 + 无远程字体
    assert "-apple-system" in html and "Microsoft YaHei" in html
    assert html.count("<svg") >= 1
    # 只有固定 companion 链接，渲染不依赖任何本地子资源
    for name in text_safety.COMPANION_FILES:
        assert f'href="{name}"' in html
    assert "<img" not in html.lower()
    assert "<style>" in html and "</style>" in html
    # 结尾单换行（不是空行堆叠）
    assert not html.endswith("\n\n")
