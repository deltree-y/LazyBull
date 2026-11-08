# -*- coding: utf-8 -*-
"""跨平台路径规范与 canonical Linux 验收包（P6 T4）。

硬口径（设计 §8.2 / §8.3 / §9）：

- 所有对外引用一律 ``root_id:relative_posix``：**NFC + POSIX 分隔符 + 无绝对路径**；
  Windows 盘符、UNC、``Path.as_uri`` 形态（``file:///D:/...``）一律拒绝；
- 本模块**不使用** ``Path.as_uri``/``resolve`` 生成对外文本（它们泄漏盘符与平台差异）；
- Linux 验收包是**固定文件集**（sh / expected_hashes / package_manifest / README），
  路径全部用 ``<repo_root>`` 占位，期望哈希逐文件登记，复跑命令逐条登记；
- 本模块只产出字节，不执行 Linux（真实 Linux 门归 T5）。
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.lazybull.v2.reporting.snapshot import ROOT_ID_PATTERN
from src.lazybull.v2.store.store_identity import canonical_json_sha256

__all__ = [
    "LINUX_PACKAGE_FILES",
    "PACKAGE_SCHEMA",
    "POSIX_ROOT_PLACEHOLDER",
    "assert_portable_ref",
    "build_linux_package",
    "package_bytes",
    "portable_source_ref",
    "relative_posix",
]

#: 仓库根占位符（canonical 输出只允许出现该形态，不写真实绝对路径）
POSIX_ROOT_PLACEHOLDER = "<repo_root>"
#: 验收包 schema
PACKAGE_SCHEMA = "p6_trend_linux_package/v1"
#: 验收包固定文件集（缺一即失败；顺序固定）
LINUX_PACKAGE_FILES: tuple[str, ...] = (
    "acceptance.sh",
    "expected_hashes.json",
    "package_manifest.json",
    "README.md",
)
_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")
_URI_PATTERN = re.compile(r"(?i)^[a-z][a-z0-9+.\-]*://|^file:")


def _nfc(text: str) -> str:
    """Unicode NFC（跨平台文件名归一；不改磁盘字节）。"""
    return unicodedata.normalize("NFC", str(text))


def _slash(text: str) -> str:
    """分隔符归一（Windows ``\\`` → POSIX ``/``），不解析软链接。"""
    return _nfc(str(text).replace("\\", "/"))


def _reject_unsafe(text: str, *, what: str) -> None:
    """拒绝绝对路径/URI/点段（对外引用只允许 root-relative POSIX）。"""
    if not text:
        raise ValueError(f"{what} 为空")
    if _URI_PATTERN.match(text):
        raise ValueError(f"{what} 不得是 URI（禁止 Path.as_uri 形态）：{text!r}")
    if text.startswith("/") or _DRIVE_PATTERN.match(text):
        raise ValueError(f"{what} 不得是绝对路径（必须 root-relative）：{text!r}")
    if "\\" in text:
        raise ValueError(f"{what} 含反斜杠（必须 POSIX 分隔符）：{text!r}")
    parts = text.split("/")
    if any(part in ("..", "") for part in parts) or text in (".", "./"):
        raise ValueError(f"{what} 含空段/父目录段：{text!r}")
    if _nfc(text) != text:
        raise ValueError(f"{what} 不是 NFC 归一形态：{text!r}")


def relative_posix(path: str | Path, root: str | Path) -> str:
    """把绝对/相对路径压成相对 ``root`` 的 POSIX 文本（跨平台字节一致）。

    Windows 与 POSIX 形态在同一 ``root`` 下必须得到**逐字节相同**的结果：
    ``D:/repo/data/runs/x`` 与 ``/repo/data/runs/x``（root 分别为 ``D:/repo`` 与
    ``/repo``）都返回 ``data/runs/x``。
    """
    target = _slash(str(path)).rstrip("/")
    base = _slash(str(root)).rstrip("/")
    if target == base:
        raise ValueError(f"路径与根相同，无可引用相对部分：{path!r}")
    prefix = f"{base}/"
    if not target.startswith(prefix):
        raise ValueError(f"路径不在声明根之下：{path!r}（root={root!r}）")
    relative = target[len(prefix):]
    _reject_unsafe(relative, what="相对路径")
    return relative


def assert_portable_ref(text: str) -> str:
    """校验并返回 ``root_id:relative`` 引用（``.`` 表示根自身）。"""
    material = str(text)
    if ":" not in material:
        raise ValueError(f"引用必须是 root_id:relative：{material!r}")
    root_id, relative = material.split(":", 1)
    if not ROOT_ID_PATTERN.match(root_id):
        raise ValueError(f"root_id 形态非法：{root_id!r}")
    if relative != ".":
        _reject_unsafe(relative, what="引用相对路径")
    return material


def portable_source_ref(root_id: str, relative: str | Path | None = None) -> str:
    """构造对外引用（NFC/POSIX；空相对路径写 ``.``）。"""
    text = "" if relative is None else _slash(str(relative)).strip("/")
    if not text:
        text = "."
    return assert_portable_ref(f"{root_id}:{text}")


def _shell_check_script(files: Mapping[str, str]) -> str:
    """POSIX shell 校验脚本（ASCII；逐文件比对 ``sha256sum``）。"""
    lines = [
        "#!/usr/bin/env sh",
        "# P6 T4 canonical Linux acceptance (generated; do not edit)",
        "set -eu",
        'if [ "$#" -ne 1 ]; then',
        '  echo "usage: sh acceptance.sh <report_dir>" >&2',
        "  exit 2",
        "fi",
        'REPORT_DIR="$1"',
        "fail=0",
        "check() {",
        '  if [ ! -f "$REPORT_DIR/$1" ]; then echo "MISSING $1" >&2; fail=1; return; fi',
        '  actual=$(sha256sum "$REPORT_DIR/$1" | cut -d" " -f1)',
        '  if [ "$actual" != "$2" ]; then echo "HASH_MISMATCH $1" >&2; fail=1;',
        '  else echo "OK $1"; fi',
        "}",
    ]
    for name, digest in sorted(files.items()):
        lines.append(f'check "{name}" "{digest}"')
    lines.extend(['if [ "$fail" -ne 0 ]; then echo "VERDICT FAIL"; exit 3; fi',
                  'echo "VERDICT PASS"'])
    return "\n".join(lines) + "\n"


def _expected_payload(
    *,
    report_id: str,
    generated_at: str,
    evidence_class: str,
    files: Mapping[str, str],
    commands: Sequence[str],
    corpus_sha256: str | None,
    steps: Sequence[str] = (),
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """``expected_hashes.json`` 载荷（路径只以占位符/文件名出现）。"""
    payload: dict[str, Any] = {
        "schema": PACKAGE_SCHEMA,
        "report_id": str(report_id),
        "generated_at": str(generated_at),
        "evidence_class": str(evidence_class),
        "root_placeholder": POSIX_ROOT_PLACEHOLDER,
        "files": {str(name): str(digest) for name, digest in sorted(files.items())},
        "corpus": {"content_sha256": corpus_sha256},
        "commands": [str(item) for item in commands],
        "steps": [str(item) for item in steps],
        "note": (
            "本包只做**逐文件哈希比对**（acceptance.sh），不在包内复算生成；"
            "复跑命令为仓库相对 POSIX，`<repo_root>`/`<out>` 为占位符；"
            "真实 Linux 全链生成与双平台哈希门归 T5。"
        ),
    }
    if previous is not None:
        payload["previous"] = dict(previous)
    return payload


def _readme_text() -> str:
    """验收包说明（固定文本；只写命令与边界，不写实测数字）。"""
    return (
        "# P6 T4 canonical Linux 验收包\n"
        "\n"
        "用途：在 Linux 上复核本报告 run 的字节确定性与写域边界（Windows 侧已生成）。\n"
        "\n"
        "## 复跑命令（在仓库根执行；`<repo_root>` 为占位符）\n"
        "\n"
        "```sh\n"
        "cd <repo_root>\n"
        "cp -r <report_dir> /tmp/p6_trend_check\n"
        "sh acceptance.sh /tmp/p6_trend_check\n"
        "```\n"
        "\n"
        "## 判定\n"
        "\n"
        "- `expected_hashes.json` 的 `files` 为逐文件期望哈希（报告 HTML/CSV/JSON/manifest）；\n"
        "- `acceptance.sh` 逐文件比对，任一缺失或不等即 `VERDICT FAIL`（退出码 3）；\n"
        "- `package_manifest.json` 自封存本包文件哈希（`record_sha256` = 去自身后 canonical JSON 哈希）。\n"
        "\n"
        "## 边界\n"
        "\n"
        "- 本包不执行生成、不访问网络、不写仓库；真实 Linux 全链生成与双平台哈希门归 T5；\n"
        "- 路径一律 root-relative POSIX；出现盘符/绝对路径即为生成缺陷。\n"
    )


def package_bytes(payload: Mapping[str, Any]) -> bytes:
    """canonical JSON 字节（UTF-8/LF、sort_keys、allow_nan=False）。"""
    import json

    text = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2,
                      allow_nan=False)
    return (text + "\n").encode("utf-8")


def build_linux_package(
    *,
    report_id: str,
    generated_at: str,
    evidence_class: str,
    files: Mapping[str, str],
    commands: Sequence[str],
    corpus_sha256: str | None = None,
    steps: Sequence[str] = (),
    previous: Mapping[str, Any] | None = None,
) -> dict[str, bytes]:
    """生成 canonical Linux 验收包（固定四文件；返回文件名 → 字节）。"""
    if not files:
        raise ValueError("验收包必须登记至少一个产物哈希")
    for name in files:
        _reject_unsafe(str(name), what="产物文件名")
        if "/" in str(name):
            raise ValueError(f"产物名必须是单段固定名：{name!r}")
    expected = _expected_payload(
        report_id=report_id,
        generated_at=generated_at,
        evidence_class=evidence_class,
        files=files,
        commands=commands,
        corpus_sha256=corpus_sha256,
        steps=steps,
        previous=previous,
    )
    expected_bytes = package_bytes(expected)
    shell_bytes = _shell_check_script(files).encode("utf-8")
    readme_bytes = _readme_text().encode("utf-8")
    manifest_payload = {
        "schema": PACKAGE_SCHEMA,
        "report_id": str(report_id),
        "generated_at": str(generated_at),
        "files": [
            {"path": name, "sha256": _sha256(payload)}
            for name, payload in sorted(
                {
                    "acceptance.sh": shell_bytes,
                    "expected_hashes.json": expected_bytes,
                    "README.md": readme_bytes,
                }.items()
            )
        ],
    }
    manifest_payload["record_sha256"] = canonical_json_sha256(manifest_payload)
    package = {
        "acceptance.sh": shell_bytes,
        "expected_hashes.json": expected_bytes,
        "package_manifest.json": package_bytes(manifest_payload),
        "README.md": readme_bytes,
    }
    if tuple(sorted(package)) != tuple(sorted(LINUX_PACKAGE_FILES)):
        raise ValueError(f"验收包文件集与固定清单不一致：{sorted(package)}")
    return package


def _sha256(payload: bytes) -> str:
    from hashlib import sha256

    return sha256(payload).hexdigest()
