# -*- coding: utf-8 -*-
"""P5b 试车场 T2 产物布局、内容身份与目录搬运原语。

契约来源：设计 `docs/plans/v2/p5b_testbed_design_20261101.md` v4 §4.3（族级事务 / 根系
内容身份 / 部分 roots 提升）与 §7.1（产物根布局）。

布局口径（§7.1 + §4.3 失败现场）：

```text
<campaign_root>/
├── registered/                     # T1 注册与 D0 fresh（本模块只读，不写）
├── arms/<arm_id>/                  # 逻辑可见代（PaperStateStore 根；提升目标）
├── staging/<date>/<attempt_id>/    # 本 attempt 的暂存根（驱动发生地）
├── backups/<date>/<attempt_id>/    # 父代备份（可验证 = 内容身份复算一致）
├── failed/<date>/<attempt_id>/     # 失败现场（已提升的新 root / 未发布分片）
├── ledger/<arm_id>/<date>.json     # 逐日分片（T2 只写单日分片，长期分片账本归 T3）
├── ledger/<arm_id>/manifest.json   # 分片清单（幂等补齐入口）
├── journal/<date>.json             # 族级事务 journal（唯一发布点载体）
├── reports/                        # 运行报告
└── _index.json                     # 读侧投影（唯一已发布代视图）
```

内容身份（§4.3「各臂旧/新 root 的 gen_id/content_sha」）：把臂根下全部常规文件的
``{相对路径: {size, sha256}}`` 做 canonical JSON 哈希；排除写锁与原子写残留
（``state/.write.lock`` / ``*.tmp``）——它们是瞬时基础设施，不是逻辑状态。

写域边界：本模块不写 ``data/paper`` / ``data/runs`` / ``data/models``；所有目录操作由
调用方（runner/family_txn）在 campaign 根内传入绝对路径。
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from src.lazybull.v2.hosts.testbed.paths import TestbedError
from src.lazybull.v2.store.store_identity import canonical_json_sha256

__all__ = [
    "ARMS_DIR",
    "BACKUPS_DIR",
    "CampaignInvalidError",
    "CampaignLayout",
    "FAILED_DIR",
    "FamilyTxnError",
    "IGNORED_RELATIVE",
    "INDEX_FILENAME",
    "JOURNAL_DIR",
    "LEDGER_DIR",
    "REPORTS_DIR",
    "STAGING_DIR",
    "copy_root",
    "file_identities",
    "gen_id",
    "move_root",
    "remove_root",
    "root_identity",
]

#: 逻辑可见代目录（PaperStateStore 根）
ARMS_DIR = "arms"
#: 暂存代目录（``staging/<date>/<attempt_id>/<arm_id>``）
STAGING_DIR = "staging"
#: 父代备份目录（``backups/<date>/<attempt_id>/<arm_id>``）
BACKUPS_DIR = "backups"
#: 失败现场目录（``failed/<date>/<attempt_id>/...``）
FAILED_DIR = "failed"
#: 逐日分片目录（``ledger/<arm_id>/<date>.json``）
LEDGER_DIR = "ledger"
#: 事务 journal 目录（``journal/<date>.json``）
JOURNAL_DIR = "journal"
#: 运行报告目录
REPORTS_DIR = "reports"
#: 读侧已发布代投影文件名
INDEX_FILENAME = "_index.json"
#: 不纳入内容身份的瞬时文件（相对臂根，正斜杠）
IGNORED_RELATIVE = ("state/.write.lock",)
#: 不纳入内容身份的后缀（原子写残留）
_IGNORED_SUFFIXES = (".tmp",)


class FamilyTxnError(TestbedError):
    """族级事务状态机错误（状态非法 / 现场与 journal 不一致）。"""

    __test__ = False


class CampaignInvalidError(FamilyTxnError):
    """父代备份不可验证或现场拼接风险：当前 campaign 失效（禁止猜测）。"""

    __test__ = False


def gen_id(generation: int) -> str:
    """代际标识（``g0000`` = D0 fresh；每个已发布虚拟日 +1）。"""
    value = int(generation)
    if value < 0:
        raise FamilyTxnError(f"generation 必须 >= 0，实际 {generation!r}")
    return f"g{value:04d}"


def _ignored(relative: str) -> bool:
    if relative in IGNORED_RELATIVE:
        return True
    return relative.endswith(_IGNORED_SUFFIXES)


def _sha256_of_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """分块哈希（不把大 parquet 整块读进内存；内容身份口径不变）。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identities(root: Path) -> Dict[str, Dict[str, Any]]:
    """目录下全部常规文件的内容身份（``{相对路径: {size, sha256}}``，按路径排序）。"""
    base = Path(root)
    if not base.is_dir():
        raise FamilyTxnError(f"内容身份要求目录存在: {base}")
    identities: Dict[str, Dict[str, Any]] = {}
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(base).as_posix()
        if _ignored(relative):
            continue
        identities[relative] = {
            "size": path.stat().st_size,
            "sha256": _sha256_of_file(path),
        }
    return identities


def root_identity(root: Path) -> Dict[str, Any]:
    """根内容身份：``{file_count, content_sha, files}``（``files`` 供分片取证）。

    Raises:
        FamilyTxnError: 目录缺失或为空（空根不能冒充一代账本）。
    """
    identities = file_identities(root)
    if not identities:
        raise FamilyTxnError(f"根内容身份为空（禁止空根冒充一代）: {root}")
    return {
        "file_count": len(identities),
        "content_sha": canonical_json_sha256(identities),
        "files": identities,
    }


def copy_root(src: Path, dst: Path) -> None:
    """复制一整个代根（排除写锁与 ``*.tmp`` 瞬时文件）；目标已存在即拒绝。"""
    source, target = Path(src), Path(dst)
    if not source.is_dir():
        raise FamilyTxnError(f"复制源必须是目录: {source}")
    if target.exists():
        raise FamilyTxnError(f"复制目标已存在（禁止覆盖代根）: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns(*_IGNORED_SUFFIXES, "state/.write.lock", ".write.lock"),
    )


def move_root(src: Path, dst: Path) -> None:
    """目录级重命名搬运（同一文件系统内原子；目标存在即拒绝）。"""
    source, target = Path(src), Path(dst)
    if not source.exists():
        raise FamilyTxnError(f"搬运源不存在: {source}")
    if target.exists():
        raise FamilyTxnError(f"搬运目标已存在（禁止覆盖）: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(source, target)
    except OSError as exc:
        raise FamilyTxnError(
            f"目录搬运失败（跨文件系统或占用）: {source} → {target}（{exc}）"
        ) from exc


def remove_root(path: Path) -> None:
    """删除目录树（缺失视为已清理）；失败显式上报，不静默吞掉。"""
    target = Path(path)
    try:
        shutil.rmtree(target)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise FamilyTxnError(f"目录清理失败（需人工核对）: {target}（{exc}）") from exc


def _clean_tmp(path: Path) -> None:
    """删除目录下的恢复临时根（``*.restore-tmp``）。"""
    if not path.is_dir():
        return
    for leftover in sorted(path.glob("*.restore-tmp")):
        remove_root(leftover)


@dataclass(frozen=True)
class CampaignLayout:
    """campaign 产物布局（T2 新增目录的唯一定义点）。"""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())

    # ---- 根目录 ----
    @property
    def arms_root(self) -> Path:
        return self.root / ARMS_DIR

    @property
    def staging_root(self) -> Path:
        return self.root / STAGING_DIR

    @property
    def backups_root(self) -> Path:
        return self.root / BACKUPS_DIR

    @property
    def failed_root(self) -> Path:
        return self.root / FAILED_DIR

    @property
    def ledger_root(self) -> Path:
        return self.root / LEDGER_DIR

    @property
    def journal_root(self) -> Path:
        return self.root / JOURNAL_DIR

    @property
    def reports_root(self) -> Path:
        return self.root / REPORTS_DIR

    @property
    def index_path(self) -> Path:
        return self.root / INDEX_FILENAME

    # ---- 单元路径 ----
    def arm_root(self, arm_id: str) -> Path:
        return self.arms_root / arm_id

    def attempt_dir(self, base: Path, virtual_date: str, attempt_id: str) -> Path:
        return Path(base) / virtual_date / attempt_id

    def staged_arm_root(self, virtual_date: str, attempt_id: str, arm_id: str) -> Path:
        return self.attempt_dir(self.staging_root, virtual_date, attempt_id) / arm_id

    def backup_arm_root(self, virtual_date: str, attempt_id: str, arm_id: str) -> Path:
        return self.attempt_dir(self.backups_root, virtual_date, attempt_id) / arm_id

    def failed_arm_root(
        self, virtual_date: str, attempt_id: str, arm_id: str, *, slot: str = "roots"
    ) -> Path:
        return self.attempt_dir(self.failed_root, virtual_date, attempt_id) / slot / arm_id

    def journal_path(self, virtual_date: str) -> Path:
        return self.journal_root / f"{virtual_date}.json"

    def shard_path(self, arm_id: str, virtual_date: str) -> Path:
        return self.ledger_root / arm_id / f"{virtual_date}.json"

    def manifest_path(self, arm_id: str) -> Path:
        return self.ledger_root / arm_id / "manifest.json"

    # ---- 清理 ----
    def clean_restore_leftovers(self, arm_id: str) -> None:
        """清理该臂恢复残留（``arms/<arm_id>/*.restore-tmp``）。"""
        _clean_tmp(self.arm_root(arm_id))

    def staging_attempt_dir(self, virtual_date: str, attempt_id: str) -> Path:
        return self.attempt_dir(self.staging_root, virtual_date, attempt_id)


def layout_for(campaign_root: str | Path) -> CampaignLayout:
    """构造布局（不建目录；目录由写入路径按需创建）。"""
    return CampaignLayout(Path(campaign_root))


def summarise_identities(
    arms: Iterable[str], identities: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """臂 → 内容身份摘要（``{arm_id: {gen_id, content_sha}}`` 形态的组装助手）。"""
    return {
        arm_id: {
            "gen_id": identities[arm_id]["gen_id"],
            "content_sha": identities[arm_id]["content_sha"],
        }
        for arm_id in arms
    }


def optional_identity(root: Path) -> Optional[Dict[str, Any]]:
    """根存在则返回内容身份，不存在返回 None（现场判定的统一入口）。"""
    if not Path(root).is_dir():
        return None
    return root_identity(root)
