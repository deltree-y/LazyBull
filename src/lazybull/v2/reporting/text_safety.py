# -*- coding: utf-8 -*-
"""来源文本安全口径（设计 §6.4：Unicode NFC + 转义 + 只写文本节点）。

硬口径（P6 T2）：

- 一切**来源文本**（账户/持仓/指令/pending/调仓/配置/质量/降级明细）先做 Unicode NFC，
  再转义 ``< > &`` 与行分隔符 ``U+2028``/``U+2029``；
- 来源文本**只允许写进文本节点**，不得进入属性、脚本、URL 或可执行命令；本模块因此
  只提供文本口径，不提供“属性安全插值”入口（禁止以此为由把来源写进属性）；属性位只允许
  **闭集常量**，由 ``security.ALLOWED_ATTRIBUTES`` 在发布前判定器里强制（未登记属性即 fail）；
- 引号不做 HTML 文本转义（文本节点无需），但**也不提供**用于属性的变体，避免误用；
- **换行归一（T2-R3-06）**：``CRLF``/``CR`` 在文本层折叠为 ``LF``（页面字节契约是全页 LF；
  不归一会让来源里合法的 Windows 换行导致整页生成失败），归一不改源文件与权威身份比较；
- **已知边界（T2-R2-10 登记，T2-R3-06 修订）**：条款只覆盖契约点名的 5 个字符 + scheme 字面量
  + 换行归一。其余 C0/C1 控制字符与双向覆盖字符（``U+202E`` 等）原样进入文本节点——这是
  “可见文本等价、不隐藏来源”的取向所决定的；``html_check.text_snapshot`` 原样收录，
  证据审阅时按原文判读。

本模块是纯函数、无 I/O，也不 import Jinja2（模板过滤器在 ``render`` 内注册）。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "COMPANION_ALLOWED",
    "COMPANION_FILES",
    "EXPERIMENT_COMPANION_FILES",
    "TREND_COMPANION_FILES",
    "companion_links",
    "escape_text",
    "normalize_text",
    "plain_text",
]

#: 驾驶舱 companion（固定文件名集合；CSV 归 T3 的实验面）
COMPANION_FILES: tuple[str, ...] = ("projection.json", "report_manifest.json")
#: 实验面 companion = 两份只读 JSON + 五份完整 CSV（文件名与 ``experiment_csv`` 同源）
EXPERIMENT_COMPANION_FILES: tuple[str, ...] = (
    "arm_metrics.csv",
    "fold_metrics.csv",
    "chain_nav.csv",
    "execution_attribution.csv",
    "stability.csv",
    "projection.json",
    "report_manifest.json",
)
#: 趋势面 companion = 两份只读 JSON + 两份完整 CSV（文件名与 ``trend_csv`` 同源）
TREND_COMPANION_FILES: tuple[str, ...] = (
    "run_index.csv",
    "trend.csv",
    "projection.json",
    "report_manifest.json",
)
#: 允许出现在 ``href`` 的固定文件名（三类报告面合集；仍为闭集，不含用户可控 URL）
COMPANION_ALLOWED: frozenset[str] = (
    frozenset(COMPANION_FILES)
    | frozenset(EXPERIMENT_COMPANION_FILES)
    | frozenset(TREND_COMPANION_FILES)
)
#: companion 固定说明（链接与说明都由本模块给出，禁止模板自造文件名）
_COMPANION_NOTES: dict[str, str] = {
    "projection.json": "本页渲染所用的 projection 原文（canonical JSON，只读）",
    "report_manifest.json": "自封存 manifest（含输入/投影/产出哈希链）",
    "arm_metrics.csv": "臂指标全值（含 Δ 与状态列；非抽样）",
    "fold_metrics.csv": "逐臂逐折指标全值（TopK 仅旧 WF 有来源）",
    "chain_nav.csv": "链式净值全行（原行序，行键=臂+折+源行号）",
    "execution_attribution.csv": "执行归因全行（契约 19 列原值 + 原 status/reason）",
    "stability.csv": "跨期稳定性全行（含显式状态行）",
    "run_index.csv": "完整 run 索引（index_mode=full，永不抽样；含 invalid 行）",
    "trend.csv": "五类指标全值（行键=run_key+metric_name；缺失留状态行，不插值）",
}
#: 转义字符表（``&`` 必须最先替换，否则会二次转义）
_ESCAPES: tuple[tuple[str, str], ...] = (
    ("&", "&amp;"),
    ("<", "&lt;"),
    (">", "&gt;"),
    ("\u2028", "&#8232;"),
    ("\u2029", "&#8233;"),
)
#: 方案字面量（``scheme:``）：文本节点里把冒号写成字符引用，**渲染文本不变**，
#: 但原始 HTML 字节不再含可被“远程引用扫描”命中的 scheme URL（来源文本仍可见）。
_SCHEME_LITERAL = re.compile(r"(?i)\b(?:https?|wss?|ftp)://|\b(?:javascript|data|file|vbscript):")


def _neutralize_scheme(match: re.Match[str]) -> str:
    """把方案字面量的冒号替换为字符引用（可见文本不变，扫描器不再命中）。"""
    return match.group(0).replace(":", "&#58;")


def neutralize_schemes(text: str) -> str:
    """在**已转义**文本上中和方案字面量（只影响字节形态，不改可见文本）。"""
    return _SCHEME_LITERAL.sub(_neutralize_scheme, text)


def normalize_text(value: Any) -> str:
    """来源文本规整为 NFC 字符串；``None`` 归一为空串（不产生 ``"None"`` 文本）。

    **换行归一**：``CRLF``/``CR`` 一律折叠为 ``LF``（T2-R3-06）。页面字节契约是全页 LF，
    若不在此归一，来源里合法的 Windows 换行会让整个报告在发布前检查处失败；归一发生在
    文本层、不改源文件，也不影响 T1 权威身份比较。
    """
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text)


def escape_text(value: Any) -> str:
    """NFC + HTML 文本转义（``< > &`` 与 U+2028/U+2029）+ 方案字面量中和。

    ``scheme:`` 的冒号会写成 ``&#58;``：浏览器渲染出的可见文本与来源完全一致，
    但原始字节里不再存在 ``https://`` 一类可被“零远程引用”扫描命中的串。
    """
    text = normalize_text(value)
    for source, target in _ESCAPES:
        text = text.replace(source, target)
    return neutralize_schemes(text)


def plain_text(value: Any) -> str:
    """供**非 HTML 场景**（CSV/日志/审计）使用的 NFC 文本；不转义但剔除控制换行。"""
    text = normalize_text(value)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def companion_links(files: Sequence[str] | None = None) -> list[dict[str, str]]:
    """固定 companion 链接清单（文件名固定、顺序固定；不含任何用户可控 URL）。"""
    names = tuple(COMPANION_FILES if files is None else files)
    unknown = [name for name in names if name not in _COMPANION_NOTES]
    if unknown:
        raise ValueError(f"未登记的 companion 文件：{unknown}")
    return [{"name": name, "href": name, "note": _COMPANION_NOTES[name]} for name in names]


def iter_source_texts(payload: Any) -> Iterable[str]:
    """递归抽取载荷中的全部字符串（测试/审计用：确认来源文本都已 NFC）。"""
    if isinstance(payload, str):
        yield payload
    elif isinstance(payload, Mapping):
        for key in payload:
            yield from iter_source_texts(payload[key])
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            yield from iter_source_texts(item)
