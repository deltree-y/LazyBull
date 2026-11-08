# -*- coding: utf-8 -*-
"""受信根判定与单读字节私有稳定快照（设计 §2.3）。

协议（逐条对应设计 §2.3）：
1. **受信根判定**：只接受本地目录；目录检 symlink/junction/reparse point 与解析后逃逸，
   **不使用目录 st_nlink**（POSIX 正常目录随子目录数增长）；文件检 ``st_nlink>1`` 硬链接、
   symlink/reparse；UNC/SMB 路径一律拒绝。
2. **单读字节私有快照**：每个输入只从原路径读一次字节，立即写入
   ``.staging/<attempt>/snapshot/<root_id>/<relative>``，并以**副本字节**计算 SHA-256；
   后续解析、loader 与复算只读副本。
3. **二次身份核验**：全部读取后对原路径集与逐文件字节重新计算身份，路径/哈希任一变化即
   丢弃本次 staging 并重试；最多重试 2 次，仍变化则 ``unstable_snapshot`` 非零退出。
4. 不取 P4 写锁、不调用恢复/ensure/runner/state_service；本模块只读原根、只写 staging。

孤儿 staging 永不发布、不参与索引或证据链（是否清理另行授权）。
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from src.lazybull.v2.reporting import status

__all__ = [
    "FILE_ATTRIBUTE_REPARSE_POINT",
    "ROOT_ID_PATTERN",
    "SnapshotError",
    "SnapshotReceipt",
    "StableSnapshot",
    "UnstableSnapshotError",
    "UntrustedRootError",
    "assert_trusted_dir",
    "is_reparse_point",
    "assert_trusted_file",
    "assert_trusted_root",
    "capture_stable_snapshot",
    "is_remote_path",
]

#: Windows 重解析点（junction / mount point / symlink）属性位
FILE_ATTRIBUTE_REPARSE_POINT = 0x400
#: 根标识形态（进入 ``root_id:relative`` 路径键与 manifest ``root_map``）
ROOT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,31}$")
#: 最多重试次数（设计 §2.3.6：最多重试 2 次）
MAX_RETRIES = 2


class SnapshotError(RuntimeError):
    """稳定快照协议违反（受信根不合格 / 读取期间身份漂移）。"""


class UntrustedRootError(SnapshotError):
    """根或文件不满足受信本地条件（symlink/junction/硬链接/逃逸/远程）。"""


class UnstableSnapshotError(SnapshotError):
    """读取期间输入身份变化，重试后仍不稳定（``invalid/unstable_snapshot``）。"""


def is_remote_path(text: str) -> bool:
    """UNC/SMB 路径判定（``\\\\host\\share`` 与 ``//host/share`` 均视为远程）。"""
    stripped = str(text).strip()
    return stripped.startswith("\\\\") or stripped.startswith("//")


def _reparse_point(path: Path) -> bool:
    """是否重解析点（symlink / junction / mount point）。"""
    if path.is_symlink():
        return True
    isjunction = getattr(os.path, "isjunction", None)
    if callable(isjunction) and isjunction(path):
        return True
    try:
        attributes = getattr(os.lstat(path), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attributes & FILE_ATTRIBUTE_REPARSE_POINT)


#: 公共别名（写审计/封存校验复用同一实现，禁止第二套受信判定）
is_reparse_point = _reparse_point


def _under(target: Path, base: Path) -> bool:
    """解析后路径是否位于 base 之下（含相等）。"""
    try:
        target.resolve().relative_to(base.resolve())
    except (ValueError, OSError):
        return False
    return True


def assert_trusted_root(
    root: Path | str,
    *,
    root_id: str,
    base: Path | str | None = None,
) -> dict[str, Any]:
    """校验受信根并返回**非 canonical 审计侧档**（解析路径与身份只进审计，不进报告）。

    Args:
        root: 待校验根目录。
        root_id: 根标识（进入 ``root_id:relative`` 路径键）。
        base: 声明的作用域根；给出时要求解析后仍在其下（逃逸即拒绝）。

    Raises:
        UntrustedRootError: 远程路径 / 不存在 / 非目录 / symlink-junction-reparse / 逃逸。
    """
    if not ROOT_ID_PATTERN.match(str(root_id)):
        raise UntrustedRootError(f"root_id 形态非法：{root_id!r}（要求 ^[a-z][a-z0-9_]{{1,31}}$）")
    text = str(root)
    if is_remote_path(text):
        raise UntrustedRootError(f"受信根拒绝远程/SMB 路径：{text}（root_id={root_id}）")
    path = Path(root)
    if not path.exists():
        raise UntrustedRootError(f"受信根不存在：{text}")
    if not path.is_dir():
        raise UntrustedRootError(f"受信根不是目录：{text}")
    if _reparse_point(path):
        raise UntrustedRootError(f"受信根是 symlink/junction/reparse point：{text}")
    if base is not None and not _under(path, Path(base)):
        raise UntrustedRootError(f"受信根解析后逃逸出声明作用域：{text}（base={base}）")
    resolved = path.resolve()
    if is_remote_path(str(resolved)):
        raise UntrustedRootError(f"受信根解析后仍为远程路径：{resolved}")
    return {
        "root_id": root_id,
        "resolved": str(resolved),
        "scope_base": None if base is None else str(Path(base).resolve()),
    }


def assert_trusted_dir(path: Path | str, *, root: Path | str) -> None:
    """目录受信校验：**只**检 symlink/junction/reparse 与逃逸，不检 ``st_nlink``。"""
    target = Path(path)
    if not target.is_dir():
        raise UntrustedRootError(f"待读目录不存在或非目录：{target}")
    if _reparse_point(target):
        raise UntrustedRootError(f"待读目录是 symlink/junction/reparse point：{target}")
    if not _under(target, Path(root)):
        raise UntrustedRootError(f"待读目录逃逸出受信根：{target}（root={root}）")


def assert_trusted_file(path: Path | str, *, root: Path | str) -> None:
    """文件受信校验：symlink/junction/reparse 与 ``st_nlink>1`` 硬链接一律拒绝。"""
    target = Path(path)
    if not target.is_file():
        raise UntrustedRootError(f"待读文件不存在或非普通文件：{target}")
    if _reparse_point(target):
        raise UntrustedRootError(f"待读文件是 symlink/junction/reparse point：{target}")
    if not _under(target, Path(root)):
        raise UntrustedRootError(f"待读文件逃逸出受信根：{target}（root={root}）")
    links = int(getattr(os.lstat(target), "st_nlink", 1) or 1)
    if links > 1:
        raise UntrustedRootError(
            f"待读文件是多硬链接（st_nlink={links}），拒绝读取：{target}"
        )


def _validate_relative(relative: str) -> str:
    """相对路径必须是 POSIX 相对键（禁绝对、禁 ``..``、禁空段）。"""
    text = str(relative).replace("\\", "/").strip()
    if not text or text.startswith("/") or re.match(r"^[A-Za-z]:", text):
        raise SnapshotError(f"输入相对路径非法（须为 POSIX 相对路径）：{relative!r}")
    if any(part in ("", ".", "..") for part in text.split("/")):
        raise SnapshotError(f"输入相对路径含非法段：{relative!r}")
    return text


def _list_live(root: Path, relative: str, *, suffix: str = "") -> list[str]:
    """枚举受信目录下的普通文件（升序）；目录缺失返回空表（缺失与空目录区分由调用方判）。"""
    directory = root / relative
    if not directory.is_dir():
        return []
    assert_trusted_dir(directory, root=root)
    return sorted(
        entry.name
        for entry in directory.iterdir()
        if entry.is_file() and not _reparse_point(entry) and entry.name.endswith(suffix)
    )


@dataclass(frozen=True)
class SnapshotReceipt:
    """单个输入文件的稳定身份（root_id:relative + 副本字节 SHA-256）。"""

    root_id: str
    relative: str
    sha256: str
    size: int

    @property
    def path_key(self) -> str:
        """manifest ``inputs[].path`` 的唯一形态：``root_id:relative_posix``。"""
        return f"{self.root_id}:{self.relative}"

    def as_dict(self) -> dict[str, Any]:
        """canonical 输入身份行（不含 mtime/绝对路径）。"""
        return {"path": self.path_key, "sha256": self.sha256, "size": self.size}


@dataclass
class StableSnapshot:
    """一次尝试的私有快照：原根只读一次，副本承载后续全部解析。"""

    roots: dict[str, Path]
    staging_dir: Path
    attempt_id: str
    receipts: dict[str, SnapshotReceipt] = field(default_factory=dict)
    listings: dict[str, tuple[str, tuple[str, ...]]] = field(default_factory=dict)
    root_audit: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: 请求过但**当时不存在**的路径（``root_id:relative``）：二次核验时不得变成存在
    missing: dict[str, str] = field(default_factory=dict)

    @property
    def snapshot_root(self) -> Path:
        """副本根（``<staging>/<attempt>/snapshot``）。"""
        return self.staging_dir / self.attempt_id / "snapshot"

    def copy_path(self, root_id: str, relative: str) -> Path:
        """副本落盘路径（后续解析/loader 只读此处）。"""
        return self.snapshot_root / root_id / _validate_relative(relative)

    def root_copy(self, root_id: str) -> Path:
        """某根的副本目录（可交给 ``PaperStateStore(root_path=...)``）。"""
        return self.snapshot_root / root_id

    def exists(self, root_id: str, relative: str) -> bool:
        """副本中是否存在（缺失与否的唯一判据，不使用 loader 的默认空值）。"""
        return self.copy_path(root_id, relative).is_file()

    def capture(self, root_id: str, relative: str) -> bool:
        """单读原始字节 → 写副本 → **以副本字节**计算 SHA-256；缺失返回 False。"""
        if root_id not in self.roots:
            raise SnapshotError(f"未声明的根：{root_id!r}（已声明 {sorted(self.roots)}）")
        rel = _validate_relative(relative)
        source = Path(self.roots[root_id]) / rel
        if not source.exists():
            # R2-20261106-05：缺失也要登记身份，否则读完后新出现的文件不会被身份核验看见
            self.missing[f"{root_id}:{rel}"] = "absent_at_capture"
            return False
        assert_trusted_file(source, root=self.roots[root_id])
        payload = source.read_bytes()
        target = self.copy_path(root_id, rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        copy_bytes = target.read_bytes()
        if len(copy_bytes) != len(payload):
            raise SnapshotError(
                f"副本字节长度与原始不符（{len(copy_bytes)} ≠ {len(payload)}）：{target}"
            )
        digest = sha256(copy_bytes).hexdigest()
        self.receipts[f"{root_id}:{rel}"] = SnapshotReceipt(root_id, rel, digest, len(copy_bytes))
        return True

    def capture_many(self, root_id: str, relatives: Iterable[str]) -> dict[str, bool]:
        """批量捕获（逐文件单读）；返回 relative → 是否存在。"""
        return {str(rel): self.capture(root_id, str(rel)) for rel in relatives}

    def list_dir(self, root_id: str, relative: str, *, suffix: str = "") -> list[str]:
        """枚举目录（文件名升序）；结果登记后参与二次身份核验。

        ``relative`` 允许为空串表示**根本身**（旧 WF 运行目录的候选文件就在根下）；
        空串只用于枚举，``capture`` 仍要求非空 POSIX 相对路径。
        """
        if root_id not in self.roots:
            raise SnapshotError(f"未声明的根：{root_id!r}")
        rel = "" if not str(relative).strip() else _validate_relative(relative)
        names = _list_live(Path(self.roots[root_id]), rel, suffix=suffix)
        self.listings[f"{root_id}:{rel}"] = (suffix, tuple(names))
        return list(names)

    def list_dirs(self, root_id: str, relative: str) -> list[str]:
        """枚举**子目录**名（升序）；同样登记进 listings 参与二次身份核验。

        runs 契约产物按 ``folds/split<N>/`` 分目录存放，而 :meth:`list_dir` 只列文件；
        为保持“每个输入只读一次 + 身份可核验”的同一协议，这里只枚举目录名（不穿透、
        不跟随重解析点），随后由调用方对每个目录内的固定文件名逐个 ``capture``。
        """
        if root_id not in self.roots:
            raise SnapshotError(f"未声明的根：{root_id!r}")
        rel = "" if not str(relative).strip() else _validate_relative(relative)
        directory = Path(self.roots[root_id]) / rel
        if not directory.is_dir():
            self.listings[f"{root_id}:{rel}"] = ("dirs", ())
            return []
        assert_trusted_dir(directory, root=self.roots[root_id])
        names = sorted(
            entry.name for entry in directory.iterdir() if entry.is_dir() and not _reparse_point(entry)
        )
        self.listings[f"{root_id}:{rel}"] = ("dirs", tuple(names))
        return names

    def read_bytes(self, root_id: str, relative: str) -> bytes | None:
        """读副本字节（不触碰原根）；未捕获或缺失返回 None。

        已登记回执的路径在读出后**复核副本 SHA-256**：副本被后续代码改写即抛
        ``SnapshotError``（禁止用被污染的副本承载投影）。
        """
        target = self.copy_path(root_id, relative)
        if not target.is_file():
            return None
        payload = target.read_bytes()
        receipt = self.receipts.get(f"{root_id}:{_validate_relative(relative)}")
        if receipt is not None and sha256(payload).hexdigest() != receipt.sha256:
            raise SnapshotError(f"副本字节与回执不一致（副本被改写）：{target}")
        return payload

    def verify_copy(self) -> list[str]:
        """一次性复核全部副本字节与回执一致；返回问题清单（空 = 全部一致）。"""
        problems: list[str] = []
        for key, receipt in sorted(self.receipts.items()):
            target = self.copy_path(receipt.root_id, receipt.relative)
            if not target.is_file():
                problems.append(f"{key}: 副本消失")
                continue
            actual = sha256(target.read_bytes()).hexdigest()
            if actual != receipt.sha256:
                problems.append(f"{key}: 副本字节变化 {receipt.sha256[:12]}→{actual[:12]}")
        return problems

    def read_json(self, root_id: str, relative: str) -> tuple[Any, str | None]:
        """读副本 JSON；返回 ``(对象, 失败原因码)``（坏 JSON/缺文件不抛异常）。"""
        if not self.exists(root_id, relative):
            return None, None
        raw = self.read_bytes(root_id, relative)
        if raw is None:
            return None, None
        try:
            return json.loads(raw.decode("utf-8")), None
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, "invalid_json"

    def verify_live(self) -> list[str]:
        """对原路径集与逐文件字节重新计算身份；返回漂移描述（空 = 稳定）。"""
        drift: list[str] = []
        for key, receipt in sorted(self.receipts.items()):
            source = Path(self.roots[receipt.root_id]) / receipt.relative
            if not source.is_file():
                drift.append(f"{key}: 读取后消失")
                continue
            try:
                assert_trusted_file(source, root=self.roots[receipt.root_id])
            except UntrustedRootError as exc:
                drift.append(f"{key}: 读取后不再受信（{exc}）")
                continue
            actual = sha256(source.read_bytes()).hexdigest()
            if actual != receipt.sha256:
                drift.append(f"{key}: 字节变化 {receipt.sha256[:12]}→{actual[:12]}")
        for key in sorted(self.missing):
            root_id, relative = key.split(":", 1)
            if (Path(self.roots[root_id]) / relative).exists():
                drift.append(f"{key}: 读取期间由缺失变为存在")
        for key, (suffix, names) in sorted(self.listings.items()):
            root_id, relative = key.split(":", 1)
            if suffix == "dirs":
                directory = Path(self.roots[root_id]) / relative
                live = (
                    sorted(
                        entry.name
                        for entry in directory.iterdir()
                        if entry.is_dir() and not _reparse_point(entry)
                    )
                    if directory.is_dir()
                    else []
                )
            else:
                live = _list_live(Path(self.roots[root_id]), relative, suffix=suffix)
            if tuple(live) != names:
                drift.append(f"{key}: 目录内容变化 {len(names)}→{len(live)} 项")
        return drift

    def discard(self) -> None:
        """丢弃本次 staging（漂移/失败现场按纪律保留由调用方决定，此处仅用于重试前清理）。"""
        target = self.staging_dir / self.attempt_id
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)


def _new_snapshot(
    roots: Mapping[str, Path | str],
    staging_root: Path | str,
    attempt_id: str,
    *,
    scope_base: Path | str | None,
) -> StableSnapshot:
    """构造一次尝试的快照对象（先验受信根，再建 staging 目录）。"""
    if not ROOT_ID_PATTERN.match(str(attempt_id)):
        raise SnapshotError(f"attempt_id 形态非法：{attempt_id!r}")
    staging_dir = Path(staging_root)
    if _reparse_point(staging_dir):
        raise UntrustedRootError(f"staging 根是重解析点（symlink/junction），拒绝写入：{staging_dir}")
    target = staging_dir / attempt_id
    if target.exists():
        raise SnapshotError(f"staging attempt 目录已存在，拒绝覆盖：{target}")
    resolved: dict[str, Path] = {}
    audit: dict[str, dict[str, Any]] = {}
    for root_id, root in roots.items():
        resolved[str(root_id)] = Path(root)
        audit[str(root_id)] = assert_trusted_root(root, root_id=str(root_id), base=scope_base)
    snapshot = StableSnapshot(roots=resolved, staging_dir=staging_dir, attempt_id=attempt_id)
    snapshot.root_audit = audit
    return snapshot


def capture_stable_snapshot(
    *,
    roots: Mapping[str, Path | str],
    staging_root: Path | str,
    attempt_ids: Sequence[str],
    reader: Callable[[StableSnapshot], None],
    max_retries: int = MAX_RETRIES,
    scope_base: Path | str | None = None,
) -> tuple[StableSnapshot, dict[str, Any]]:
    """按 ``reader`` 采集一次稳定快照；漂移即丢弃 staging 并以新 attempt 重试。

    Args:
        roots: ``root_id -> 根目录``（全部先过受信根判定）。
        staging_root: ``data/reports/p6/.staging`` 一类目录（本函数只写其下 attempt 子目录）。
        attempt_ids: 依次可用的 attempt 标识（长度须 >= ``max_retries + 1``）。
        reader: 采集回调（对快照对象调用 ``capture``/``list_dir``）。
        max_retries: 最多重试次数（设计 §2.3.6 = 2）。
        scope_base: 声明作用域根（提供时所有根必须解析在其下）。

    Returns:
        ``(稳定快照, 审计档)``；审计档含逐次尝试与漂移清单。

    Raises:
        UnstableSnapshotError: 重试耗尽仍漂移（``invalid/unstable_snapshot``）。
    """
    ids = [str(item) for item in attempt_ids]
    if max_retries < 0:
        raise SnapshotError(f"max_retries 不得为负：{max_retries}")
    if len(ids) < max_retries + 1:
        raise SnapshotError(
            f"attempt_ids 数量不足：需要 {max_retries + 1} 个，实际 {len(ids)}"
        )
    attempts: list[dict[str, Any]] = []
    for index, attempt_id in enumerate(ids[: max_retries + 1]):
        snapshot = _new_snapshot(roots, staging_root, attempt_id, scope_base=scope_base)
        reader(snapshot)
        drift = snapshot.verify_live()
        attempts.append({"attempt_id": attempt_id, "drift": drift, "stable": not drift})
        if not drift:
            audit = {
                "attempts": attempts,
                "retries": index,
                "stable": True,
                "inputs": len(snapshot.receipts),
                "roots": {root_id: dict(item) for root_id, item in snapshot.root_audit.items()},
            }
            return snapshot, audit
        snapshot.discard()
    last = attempts[-1]["drift"]
    row = status.make_status("invalid", reason_code="unstable_snapshot", detail=str(last))
    raise UnstableSnapshotError(
        f"读取期间输入身份持续漂移（{row['report_status']}/"
        f"{row['degradation_reason_code']}），重试 {max_retries} 次后仍不一致：{last}"
    )
