# -*- coding: utf-8 -*-
"""P6-T1-16 精确写审计器（设计 §2.5）。

契约（逐条对应设计表）：
- 按命令分别断言：``generate`` 只允许本次 ``.staging/<attempt>/**`` 与最终**新建**
  ``<evidence_class>/<kind>/<report_id>/**``；``verify`` 无任何写入；``reproduce`` 只允许
  新 ``reproductions/<new_id>/**``；``measure``/``real_golden`` 只允许本次
  ``validations/<evidence_class>/<validation_id>/**``；
- **不允许**以 ``logs/**``、``temp/**``、``__pycache__/**``、``.pytest_cache/**`` 作兜底
  写例外（即使被加进白名单也一律判违规；段比较做大小写/尾点/首尾空白归一）；
- 判定语义 = “白名单外差异全空 ∧ 白名单内差异**恰为**本命令允许集合”；
  ``expected_paths`` **必填**（缺参即 TypeError），杜绝“默认只做单向判定”；
- “**新**目录”硬判据：给出 ``bindings`` 时占位符按字面量替换；``generate`` 下本次
  ``<report_id>`` 目录内 ``before`` 已有键出现 modified/removed 一律 fail（历史不覆盖）；
- 重解析点（symlink/junction）出现在受审树内即 fail（禁止借 junction 把写入洗到树外）。

审计由两层组成：``ReadonlyGuard``（运行期写拦截，越界即抛）与 ``audit_command``
（前后全树哈希差异判定）。运行期层用 **CPython 审计事件钩子**（``sys.addaudithook``）
而非逐个 patch 属性——后者永远漏（``Path.open``/``io.open``/``os.open``/``os.link``/
``os.utime``/``os.truncate``/``shutil.*``/子进程均覆盖）；钩子常驻但仅在存在活动 guard
时才执法，无活动 guard 即为空操作。

**诚实边界**（不得省略）：
- 钩子对“绕过 CPython 解释器”的写入（C 扩展直接 ``fopen``、外部进程、``ctypes``）不可见；
- **guard 之前**已持有的文件描述符（`open(fd, "wb")` 的 ``path`` 实参是 ``int``）无法映射到路径，
  当前实现放行——属同一类边界；
- 因此运行期层是**尽力而为**；判定性结论只来自 ``audit_command`` 的前后全树哈希，
  任何对外声明都不得把“guard 无违规”单独当作零写证据；
- ``audit_command`` 对**任何带占位符的命令**（`generate`/`measure`/`real_golden`/`reproduce`）
  强制绑定全部占位符（缺一即 ``ValueError``），并由绑定后的字面量自动推导“必须新建”的目录前缀，
  杜绝“忘传参数即静默放弃硬判据”；``verify`` 白名单为空故不受影响。
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import posixpath
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Iterable, Mapping, Sequence

from src.lazybull.v2.reporting import contracts, status, validation
from src.lazybull.v2.reporting.snapshot import is_reparse_point

__all__ = [
    "FORBIDDEN_PREFIXES",
    "FORBIDDEN_SEGMENTS",
    "ReadonlyGuard",
    "ReadonlyViolationError",
    "TreeScan",
    "audit_command",
    "compile_whitelist",
    "diff_trees",
    "is_forbidden",
    "match_allowed",
    "scan_tree",
    "snapshot_tree",
    "write_audit",
]

#: 永不允许作为写例外的路径前缀（设计 §2.5 明令禁止兜底）
FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "logs/",
    "temp/",
    "__pycache__/",
    ".pytest_cache/",
)
#: 前缀对应的路径段（大小写/尾点/空白归一后比较）
FORBIDDEN_SEGMENTS: tuple[str, ...] = (
    "logs",
    "temp",
    "__pycache__",
    ".pytest_cache",
)
#: 审计事件 → 动作名（CPython 审计钩子的可执法写入类事件）
_WRITE_EVENTS: dict[str, int] = {
    "os.mkdir": 1,
    "os.rmdir": 1,
    "os.remove": 1,
    "os.rename": 2,
    "os.replace": 2,
    "os.link": 2,
    "os.symlink": 2,
    "os.truncate": 1,
    "os.utime": 1,
    "os.chmod": 1,
    "os.chown": 1,
    "shutil.copyfile": 2,
    "shutil.copymode": 2,
    "shutil.copystat": 2,
    "shutil.move": 2,
    "shutil.rmtree": 1,
}
#: ``bindings`` 取值必须是字面量（禁止 ``*``/``?``/分隔符等模式元字符）
_BINDING_LITERAL_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]+$")
#: ``os.open`` 的写标志位（O_WRONLY / O_RDWR / O_CREAT / O_TRUNC / O_APPEND）
_OS_WRITE_FLAGS = 0
for _flag in ("O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND"):
    _OS_WRITE_FLAGS |= int(getattr(os, _flag, 0))


class ReadonlyViolationError(RuntimeError):
    """命令试图写入白名单外路径（运行期拦截，命令必须失败）。"""


def _normalize_segment(part: str) -> str:
    """段归一：大小写折叠 + 去首尾空白 + 去 Windows 尾点。"""
    return str(part).strip().casefold().rstrip(" .")


def is_forbidden(relative: str) -> bool:
    """路径是否命中禁止兜底段（``logs``/``temp``/``__pycache__``/``.pytest_cache``）。

    对大小写、首尾空白与 Windows 尾点归一后比较（``Logs``、``logs.``、``" logs"`` 同判）。
    """
    parts = str(relative).replace("\\", "/").strip("/").split("/")
    forbidden = {_normalize_segment(item) for item in FORBIDDEN_SEGMENTS}
    return any(_normalize_segment(part) in forbidden for part in parts)


def _has_dot_segment(relative: str) -> bool:
    """路径是否含 ``.``/``..``/空段（禁止参与白名单判定）。"""
    parts = str(relative).replace("\\", "/").split("/")
    return any(part in ("", ".", "..") for part in parts)


def compile_whitelist(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    """把设计里的 ``<占位符>``/``**``/``*`` 模式编译为正则（``/`` 分隔，锚定全串）。

    - 结尾 ``/**`` 同时匹配其**目录自身**（新建 ``<report_id>`` 目录是允许写的必要步骤）；
    - ``<占位符>`` 不得匹配点段（``..``），模式自身含 ``.``/``..``/空段直接 ``ValueError``。
    """
    compiled: list[re.Pattern[str]] = []
    for raw in patterns:
        text = str(raw).replace("\\", "/")
        if text.startswith("/") or re.match(r"^[A-Za-z]:", text):
            raise ValueError(f"白名单模式必须是相对 posix 路径：{raw!r}")
        if _has_dot_segment(text):
            raise ValueError(f"白名单模式不得含 . / .. / 空段：{raw!r}")
        text = text.strip("/")
        trailing = text.endswith("/**")
        body = text[: -len("/**")] if trailing else text
        parts: list[str] = []
        index = 0
        while index < len(body):
            if body.startswith("**", index):
                parts.append(".*")
                index += 2
                continue
            char = body[index]
            if char == "*":
                parts.append("[^/]*")
                index += 1
                continue
            if char == "<":
                end = body.find(">", index)
                if end < 0:
                    raise ValueError(f"白名单模式占位符未闭合：{raw!r}")
                parts.append(r"(?!\.{1,2}(?:/|$))[^/]+")
                index = end + 1
                continue
            parts.append(re.escape(char))
            index += 1
        suffix = "(?:/.*)?" if trailing else ""
        compiled.append(re.compile("^" + "".join(parts) + suffix + "$"))
    return compiled


def bind_patterns(patterns: Iterable[str], bindings: Mapping[str, str] | None) -> list[str]:
    """把 ``<占位符>`` 按 ``bindings`` **字面量**替换（未提供的占位符保持通配）。

    Raises:
        ValueError: 绑定值含 ``*``/``?``/``.``/``..``/空段等模式元字符——否则 ``*`` 会被
            当成通配符，把白名单与“新目录”判据一起放宽（fail-open）。
    """
    resolved: list[str] = []
    for pattern in patterns:
        text = str(pattern)
        referenced = _placeholders(text)  # 只校验本模式真正引用的占位符（共享绑定表可透传）
        for name, value in (bindings or {}).items():
            if str(name) not in referenced:
                continue
            literal = str(value).replace("\\", "/").strip("/")
            if _has_dot_segment(literal):
                raise ValueError(f"bindings[{name}] 含 . / .. / 空段：{value!r}")
            if not _BINDING_LITERAL_PATTERN.match(literal):
                raise ValueError(
                    f"bindings[{name}] 必须是字面量（仅字母/数字/下划线/连字符/点）：{value!r}"
                    "——含通配符会把白名单与“新目录”判据一起放宽"
                )
            if literal.startswith(".") or literal.endswith("."):
                raise ValueError(
                    f"bindings[{name}] 不得以点开头或结尾：{value!r}"
                    "（Windows 会剥离尾点，导致白名单永不匹配）"
                )
            text = text.replace(f"<{name}>", literal)
        resolved.append(text)
    return resolved


def _placeholders(pattern: str) -> set[str]:
    """模式中的 ``<占位符>`` 名称集合。"""
    return set(re.findall(r"<([^<>/]+)>", str(pattern)))


def derive_new_prefixes(
    *,
    command: str,
    patterns: Sequence[str],
    bindings: Mapping[str, str] | None,
) -> tuple[list[str], list[str]]:
    """由 ``bindings`` 绑值后的字面量推导“必须新建”的目录前缀。

    Returns:
        ``(前缀清单, 未能绑定的模式清单)``。**任一带占位符的模式未绑定即 ``ValueError``**
        （``generate``/``reproduce``/``measure``/``real_golden`` 同款语义：否则“新目录”判据与
        白名单都退化为通配，属于 fail-open）。白名单为空的命令（``verify``）不受影响。
    """
    prefixes: list[str] = []
    unbound: list[str] = []
    for pattern in patterns:
        bound = bind_patterns([pattern], bindings)[0]
        if "<" in bound:
            unbound.append(str(pattern))
            continue
        prefixes.append(bound[: -len("/**")] if bound.endswith("/**") else bound)
    if unbound:
        required: set[str] = set()
        for pattern in patterns:
            required |= _placeholders(pattern)
        missing = sorted(required - set(bindings or {}))
        raise ValueError(
            f"{command} 写审计必须绑定全部占位符 {sorted(required)}，缺 {missing}"
            "（否则“新目录”判据与白名单都退化为通配）"
        )
    return sorted(set(prefixes)), sorted(set(unbound))


def match_allowed(relative: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    """路径是否命中任一白名单模式（禁止段与点段优先判否）。"""
    text = str(relative).replace("\\", "/")
    if text.startswith("/") or re.match(r"^[A-Za-z]:", text):
        return False
    if _has_dot_segment(text):
        return False
    normalized = posixpath.normpath(text).lstrip("/")
    if _has_dot_segment(normalized) or is_forbidden(normalized):
        return False
    return any(pattern.match(normalized) for pattern in patterns)


@dataclass(frozen=True)
class TreeScan:
    """一次整树扫描结果：文件身份表 + **目录集合** + 重解析点 + 越界符号链接。

    ``dirs`` 是“新目录”判据的**必需证据**：空目录在文件表里没有任何身份，
    只靠文件无法区分“全新目录”与“已存在的空目录”。
    """

    files: dict[str, dict[str, Any]] = field(default_factory=dict)
    reparse: list[str] = field(default_factory=list)
    dirs: set[str] = field(default_factory=set)


class FileTree(dict):
    """文件身份表（dict 语义）+ 同一时刻的目录集合（``dirs`` 属性）。"""

    def __init__(self, files: Mapping[str, Any], dirs: Iterable[str]) -> None:
        super().__init__(files)
        self.dirs: frozenset[str] = frozenset(dirs)


def _sha256_file(path: Path) -> str:
    """文件字节 SHA-256（分块读，避免大文件驻留）。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_tree(
    roots: Mapping[str, Path | str], *, excludes: Sequence[str] = ()
) -> TreeScan:
    """整树扫描（``root_id:relative`` → ``{sha256,size}``）+ 重解析点清单。

    重解析点一律登记（不穿透、不静默跳过）：借 junction 写到树外的字节必须使审计 fail。
    """
    scan = TreeScan()
    for root_id, root in roots.items():
        base = Path(root)
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            relative = path.relative_to(base).as_posix()
            if any(fnmatch.fnmatchcase(relative, pattern) for pattern in excludes):
                continue
            if is_reparse_point(path):
                scan.reparse.append(f"{root_id}:{relative}")
                continue
            if path.is_dir():
                scan.dirs.add(f"{root_id}:{relative}")
                continue
            if not path.is_file():
                continue
            scan.files[f"{root_id}:{relative}"] = {
                "sha256": _sha256_file(path),
                "size": path.stat().st_size,
            }
    return scan


def snapshot_tree(
    roots: Mapping[str, Path | str], *, excludes: Sequence[str] = ()
) -> FileTree:
    """文件身份快照 + 目录集合（``FileTree`` 兼作 dict；重解析点见 ``scan_tree``）。"""
    scan = scan_tree(roots, excludes=excludes)
    return FileTree(scan.files, scan.dirs)


def diff_trees(
    before: Mapping[str, Mapping[str, Any]], after: Mapping[str, Mapping[str, Any]]
) -> dict[str, list[str]]:
    """前后快照差异：新增 / 修改 / 删除（键为 ``root_id:relative``）。"""
    added = sorted(key for key in after if key not in before)
    removed = sorted(key for key in before if key not in after)
    modified = sorted(
        key
        for key in before
        if key in after and before[key].get("sha256") != after[key].get("sha256")
    )
    return {"added": added, "modified": modified, "removed": removed}


def _relative_of(key: str) -> str:
    """``root_id:relative`` → 相对路径（去掉根标识）。"""
    return str(key).split(":", 1)[1] if ":" in str(key) else str(key)


def _resolve_output_root(output_root: str | None) -> str:
    """校验显式输出根（必填）：只有该根下的差异可能进白名单，其他根一律 fail。"""
    text = str(output_root or "").strip()
    if not text:
        raise ValueError(
            "output_root 不能为空：必须显式指定唯一输出工作树根（否则输入根写入会被"
            "同名相对路径洗白成输出根写入）"
        )
    return text


def _classify(
    changed: Sequence[str],
    compiled: Sequence[re.Pattern[str]],
    *,
    output_root: str,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """把差异键分为 ``(白名单内, 白名单外, 禁止段, 非输出根)``。

    只有 ``output_root`` 下的差异才可能落进白名单；其他根的任何变化一律判违规
    （R2-20261106-03：丢弃 root_id 会让输入根写入借同名相对路径进白名单）。
    """
    inside: list[str] = []
    outside: list[str] = []
    forbidden: list[str] = []
    foreign: list[str] = []
    for key in changed:
        root_id, _, relative = str(key).partition(":")
        relative = relative if _ else str(key)
        if is_forbidden(relative):
            forbidden.append(relative)
        elif root_id != output_root:
            foreign.append(key)
        elif match_allowed(relative, compiled):
            inside.append(relative)
        else:
            outside.append(relative)
    return inside, outside, forbidden, foreign


def _dir_evidence(before: Mapping[str, Any], prefixes: Sequence[str]) -> Iterable[str]:
    """取“扫描前目录集合”证据；前缀非空而无证据即拒绝判定（fail-closed）。"""
    existing = getattr(before, "dirs", None)
    if prefixes and existing is None:
        raise ValueError(
            "缺少“扫描前目录集合”证据：new_prefixes 非空时必须传"
            " scan_tree(...)/snapshot_tree(...) 的结果（空目录在文件表里没有身份，"
            "默认“不存在”会把复用旧空目录判成 pass）"
        )
    return existing or ()


def _new_dir_violations(
    *,
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
    new_prefixes: Sequence[str],
    existing_dirs: Iterable[str],
) -> list[str]:
    """“新目录”判据：绑定前缀在 ``before`` 已存在（**含空目录**）或被改写即违规。

    R2-20261106-07（复审）：文件表不含空目录身份，必须另行提供 ``scan_tree(...).dirs``
    作为“扫描前目录集合”证据；缺失即拒绝判定（禁止默认“不存在”）。
    """
    violations: list[str] = []
    prefixes = [str(item).strip("/") for item in new_prefixes]
    if not prefixes:  # 白名单里的 /** 一律指“本次新建目录”（与命令无关）
        return violations
    before_dirs = {_relative_of(item) for item in existing_dirs}
    for prefix in prefixes:
        if any(
            item == prefix or item.startswith(f"{prefix}/") for item in before_dirs
        ):
            violations.append(f"{prefix}（目录已存在，非新建目录）")
    for key in sorted(before):
        relative = _relative_of(key)
        prefix = next(
            (item for item in prefixes if relative == item or relative.startswith(f"{item}/")),
            None,
        )
        if prefix is None:
            continue
        current = after.get(key)
        if current is None:
            violations.append(f"{relative}（新目录内文件被删除）")
        elif current.get("sha256") != before[key].get("sha256"):
            violations.append(f"{relative}（已有报告被就地覆盖）")
    return violations


def audit_command(
    *,
    command: str,
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
    expected_paths: Sequence[str],
    allowed_patterns: Sequence[str] | None = None,
    bindings: Mapping[str, str] | None = None,
    new_prefixes: Sequence[str] = (),
    reparse_paths: Sequence[str] = (),
    extra_violations: Sequence[Mapping[str, Any]] = (),
    output_root: str,
) -> dict[str, Any]:
    """按命令判定写入差异；``verdict=pass`` 需全部硬条件同时成立。

    必填项：``expected_paths``（本次应出现的差异集合，``verify`` 传 ``[]``）、
    ``output_root``（唯一允许写入的根标识，其他根任何变化一律 fail）。
    其余：``bindings`` 占位符字面量、``new_prefixes`` 必须新建前缀（由 bindings 推导）、
    ``reparse_paths`` 重解析点、``extra_violations`` 运行期违规（均非空即 fail）。
    """
    raw_patterns = (
        list(contracts.WRITE_WHITELIST.get(command, ()))
        if allowed_patterns is None
        else list(allowed_patterns)
    )
    derived_prefixes, unbound = derive_new_prefixes(
        command=command, patterns=raw_patterns, bindings=bindings
    )
    patterns = bind_patterns(raw_patterns, bindings)
    compiled = compile_whitelist(patterns)
    prefixes = sorted(set(derived_prefixes) | {str(item).strip("/") for item in new_prefixes})
    diff = diff_trees(before, after)
    changed = sorted(diff["added"] + diff["modified"] + diff["removed"])
    root_id = _resolve_output_root(output_root)
    inside, outside, forbidden, foreign = _classify(changed, compiled, output_root=root_id)
    expected = sorted(str(item) for item in expected_paths)
    unexpected_inside = sorted(set(inside) - set(expected))
    missing_expected = sorted(set(expected) - set(inside))
    covered = _new_dir_violations(
        before=before,
        after=after,
        new_prefixes=prefixes,
        existing_dirs=_dir_evidence(before, prefixes),
    )
    violations = [dict(item) for item in extra_violations]
    blocked = (
        outside
        or forbidden
        or foreign
        or unexpected_inside
        or missing_expected
        or covered
        or list(reparse_paths)
        or violations
    )
    return _audit_report(
        command=command,
        verdict="fail" if blocked else "pass",
        details={
            "output_root": root_id,
            "allowed_patterns": patterns,
            "bindings": dict(bindings or {}),
            "new_prefixes": prefixes,
            "unbound_patterns": unbound,
            "diff": diff,
            "changed": changed,
            "inside_whitelist": inside,
            "outside_whitelist": outside,
            "foreign_root_paths": foreign,
            "forbidden_paths": forbidden,
            "expected_paths": expected,
            "unexpected_inside": unexpected_inside,
            "missing_expected": missing_expected,
            "overwrote_existing": covered,
            "reparse_paths": [str(item) for item in reparse_paths],
            "runtime_violations": violations,
        },
    )


def _audit_report(
    *, command: str, verdict: str, details: Mapping[str, Any]
) -> dict[str, Any]:
    """装配审计报告（状态行 + 全量判定明细）。"""
    return status.make_status(
        "available" if verdict == "pass" else "invalid",
        reason_code=None if verdict == "pass" else "write_outside_whitelist",
        detail=(
            "白名单外零差异、白名单内恰为预期集合、新目录未被覆盖、无运行期违规"
            if verdict == "pass"
            else "写域判定失败（见 diff/violations）"
        ),
        extra={"command": command, "verdict": verdict, **dict(details)},
    )


def write_audit(path: Path | str, report: Mapping[str, Any]) -> Path:
    """落 ``readonly_audit.json``（不覆盖、UTF-8/LF、allow_nan=False）。"""
    return validation.write_json_new(path, report)


# ---------------------------------------------------------------------------
# 运行期拦截（CPython 审计事件钩子）
# ---------------------------------------------------------------------------

#: 活动 guard 栈（钩子常驻，无 guard 时为空操作）
_ACTIVE_GUARDS: list["ReadonlyGuard"] = []
_HOOK_INSTALLED = False


def _audit_hook(event: str, args: tuple[Any, ...]) -> None:
    """CPython 审计钩子：把写入类事件交给**栈内全部** guard 执法（任一违规即抛）。"""
    if not _ACTIVE_GUARDS:
        return
    for guard in tuple(_ACTIVE_GUARDS):
        guard.on_event(event, args)


def _install_hook() -> None:
    """安装常驻钩子（不可卸载；无活动 guard 即空操作）。"""
    global _HOOK_INSTALLED
    if not _HOOK_INSTALLED:
        sys.addaudithook(_audit_hook)
        _HOOK_INSTALLED = True


class ReadonlyGuard:
    """运行期写拦截：工作树内白名单外写入/建目录/改名/链接/时间戳/socket 一律抛错。

    只对 ``worktree_root`` 之内（按**未解析**路径判断，防止 junction 洗白）的路径执法；
    工作树之外（系统临时目录、浏览器缓存等）不干预。工作树内禁止段无论白名单如何配置
    都直接判违规。

    实现用 ``sys.addaudithook``（覆盖 ``open``/``os.open``/``os.link``/``os.utime``/
    ``os.truncate``/``os.remove``/``os.rename``/``shutil.*``/``subprocess.Popen`` 等），
    不做逐属性 patch（后者必然遗漏 ``Path.open``/``io.open`` 一类入口）。
    """

    def __init__(
        self,
        *,
        allowed_patterns: Iterable[str],
        worktree_root: Path | str,
        bindings: Mapping[str, str] | None = None,
        deny_network: bool = False,
        deny_subprocess: bool = True,
    ) -> None:
        self.allowed_patterns = bind_patterns(list(allowed_patterns), bindings)
        raw_root = Path(os.path.abspath(Path(worktree_root)))
        resolved_root = raw_root.resolve()
        if raw_root != resolved_root:
            raise ReadonlyViolationError(
                f"worktree_root 经重解析点/别名（未解析 {raw_root} ≠ 解析后 {resolved_root}），"
                "拒绝以可能失效的执法根启动"
            )
        self.worktree_root = resolved_root
        self.deny_network = deny_network
        self.deny_subprocess = deny_subprocess
        self.violations: list[dict[str, Any]] = []
        self._compiled = compile_whitelist(self.allowed_patterns)

    # -- 路径判定 ---------------------------------------------------------
    def _relative(self, target: Any) -> tuple[str | None, bool]:
        """返回 ``(工作树内相对路径, 是否经重解析点逃逸)``（工作树外返回 ``(None, False)``）。"""
        try:
            raw = Path(os.fspath(target))
        except TypeError:
            return None, False
        absolute = Path(os.path.abspath(raw))
        inside_raw = self._raw_relative(absolute)
        if inside_raw is None:
            return None, False
        try:
            resolved = raw.resolve()
        except OSError:
            return inside_raw, False
        if self._raw_relative(resolved) is None:
            return inside_raw, True
        return resolved.relative_to(self.worktree_root).as_posix(), False

    def _raw_relative(self, path: Path) -> str | None:
        """未解析路径是否位于工作树内（不做 symlink 解引用）。"""
        try:
            return path.relative_to(self.worktree_root).as_posix()
        except ValueError:
            return None

    def check(self, target: Any, *, action: str) -> None:
        """写入前执法：越界/重解析/禁止段一律登记后抛错（命令必须失败）。"""
        relative, escaped = self._relative(target)
        if relative is None:
            return
        if escaped:
            self.violations.append({"action": action, "path": relative, "kind": "reparse_escape"})
            raise ReadonlyViolationError(
                f"只读命令经重解析点写到工作树外（{action}）：{relative}"
            )
        if is_forbidden(relative) or not match_allowed(relative, self._compiled):
            self.violations.append({"action": action, "path": relative})
            raise ReadonlyViolationError(
                f"只读命令试图写入白名单外路径（{action}）：{relative}"
            )

    # -- 审计事件映射 -----------------------------------------------------
    def on_event(self, event: str, args: tuple[Any, ...]) -> None:
        """审计事件 → 目标路径（写类事件才执法）。"""
        if event == "open":
            self._on_open(args)
            return
        if event == "os.open":
            self._on_os_open(args)
            return
        count = _WRITE_EVENTS.get(event)
        if count is not None:
            for item in args[:count]:
                self.check(item, action=event)
            return
        if event in ("socket.connect", "socket.bind") and self.deny_network:
            self.violations.append({"action": event, "path": "<socket>"})
            raise ReadonlyViolationError("只读命令禁止网络连接/绑定")
        if event == "subprocess.Popen" and self.deny_subprocess:
            executable = args[0] if args else "<subprocess>"
            self.violations.append({"action": event, "path": str(executable)})
            raise ReadonlyViolationError(
                f"只读命令禁止派生外部进程（无法审计其写域）：{executable}"
            )

    def _on_open(self, args: tuple[Any, ...]) -> None:
        """``open(path, mode, flags)``：mode/flags 任一含写意图即执法。"""
        if len(args) < 3:
            return
        target, mode, flags = args[0], args[1], args[2]
        writing = isinstance(mode, str) and any(flag in mode for flag in ("w", "a", "x", "+"))
        if not writing and isinstance(flags, int) and flags & _OS_WRITE_FLAGS:
            writing = True
        if writing:
            self.check(target, action="open")

    def _on_os_open(self, args: tuple[Any, ...]) -> None:
        """``os.open(path, flags, mode)``：flags 含写标志位即执法。"""
        if len(args) < 2:
            return
        target, flags = args[0], args[1]
        if isinstance(flags, int) and flags & _OS_WRITE_FLAGS:
            self.check(target, action="os.open")

    # -- 生命周期 ---------------------------------------------------------
    def __enter__(self) -> "ReadonlyGuard":
        _install_hook()
        _ACTIVE_GUARDS.append(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if self in _ACTIVE_GUARDS:
            _ACTIVE_GUARDS.remove(self)
        return False
