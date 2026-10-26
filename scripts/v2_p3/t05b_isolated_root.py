"""P3 T0.5b 隔离运行根构建与生产模型目录守卫（§7.3 / §13.3.4）。

隔离运行根：``data/frozen_reference/v2_p3/<run>/root/``
- 只读接入 = ``clean/ raw/ features/ labels/`` 四目录（Windows 目录联接）；
- ``models/ walk_forward/ reports/ paper/`` 为全新目录，**禁止联接 models**（防写穿生产）；
  P3 T5 起 ``build_isolated_root`` 支持 ``extra_fresh_dirs`` 追加全新目录（如 ``runs``，
  默认 ``()`` 与 spike 现状逐位一致）；
- 生产 ``data/models/stock_selection/`` 守卫：运行前后文件清单 + 关键索引文件内容哈希
  比对，任一变化即停止（§7.4-3）。
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from loguru import logger

# 只读接入的四目录（§13.3.4；labels 缺表返回空故必须显式接入）
READONLY_LINK_DIRS = ("clean", "raw", "features", "labels")
# 全新目录（禁止联接；models 禁止联接为硬契约）
FRESH_DIRS = ("models", "walk_forward", "reports", "paper")

# 守卫全量内容哈希的关键索引文件（写穿即判漂移）
GUARD_HASH_FILES = ("ml_train_runs.csv", "latest_model_version.txt", "model_registry.json")


def _ensure_link(link: Path, target: Path) -> None:
    """创建目录联接（Windows）/符号链接（POSIX）；已存在则跳过。"""
    if link.exists():
        return
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True
        )
    else:
        link.symlink_to(target)
    logger.info(f"目录联接: {link} -> {target}")


def _normalize_extra_fresh_dirs(extra_fresh_dirs: Sequence[str]) -> List[str]:
    """归一化追加全新目录名（P3 T5：如 ``runs``——防默认解析写穿生产 ``data/runs``）。

    空名 / 与只读接入目录或既有全新目录重名 ⇒ ValueError；去重保序。
    """
    normalized: List[str] = []
    for name in extra_fresh_dirs:
        if not isinstance(name, str) or not name.strip() or name.strip() != name:
            raise ValueError(f"追加全新目录名必须为非空且前后无空白的字符串: {name!r}")
        if name in READONLY_LINK_DIRS or name in FRESH_DIRS:
            raise ValueError(f"追加全新目录名与既有目录冲突: {name!r}")
        if name not in normalized:
            normalized.append(name)
    return normalized


def build_isolated_root(
    run_dir: Path,
    production_data_root: Path,
    link_fn: Callable[[Path, Path], None] = _ensure_link,
    verify_links: bool = True,
    extra_fresh_dirs: Sequence[str] = (),
) -> Path:
    """在 ``run_dir`` 下构建隔离运行根 ``root/`` 并返回其路径。

    Args:
        run_dir: 本次运行的隔离目录（root 建于其下）；
        production_data_root: 生产数据根（如 ``data/``）；
        link_fn: 目录联接实现（测试可注入替身避免真实 mklink）；
        verify_links: 建成后断言四只读目录确为联接且指向生产根（S5：
        防既有真实空目录导致 ``_ensure_link`` 静默跳过、读到空数据）；
        注入替身 link_fn 的测试须显式置 False。
        extra_fresh_dirs: 追加全新目录名（P3 T5：臂运行根需 ``runs/`` 全新目录
        兜底，防任何默认解析写穿生产 ``data/runs``）；默认 ``()`` ⇒ 行为与
        spike 现状逐位一致。

    Raises:
        FileExistsError: root 已存在且非空（禁止复用，隔离根整体可删后重建）；
        ValueError: 生产数据根位于 run_dir 内（方向错误，防写穿）。
    """
    run_dir = Path(run_dir).resolve()
    production_data_root = Path(production_data_root).resolve()
    root = run_dir / "root"
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"隔离运行根已存在且非空，禁止复用: {root}")
    if _is_relative_to(production_data_root, run_dir):
        raise ValueError(f"生产数据根不得位于隔离运行目录内: {production_data_root}")

    fresh_dirs = list(FRESH_DIRS) + _normalize_extra_fresh_dirs(extra_fresh_dirs)
    root.mkdir(parents=True, exist_ok=True)
    for name in READONLY_LINK_DIRS:
        target = production_data_root / name
        if not target.is_dir():
            raise FileNotFoundError(f"生产数据目录缺失，无法只读接入: {target}")
        link_fn(root / name, target)
    for name in fresh_dirs:
        (root / name).mkdir(parents=True, exist_ok=True)
    for name in fresh_dirs:
        candidate = root / name
        if candidate.is_symlink() or _is_junction(candidate):
            raise ValueError(f"隔离根全新目录禁止联接: {candidate}")
    if verify_links:
        assert_readonly_links(root, production_data_root)
    logger.info(
        f"隔离运行根就绪: {root}（只读接入 {READONLY_LINK_DIRS}，全新 {tuple(fresh_dirs)}）"
    )
    return root


def assert_readonly_links(root: Path, production_data_root: Path) -> None:
    """断言四只读目录均为联接（junction/symlink）且解析指向生产根对应目录（S5）。

    既有真实目录（``_ensure_link`` 静默跳过的情形）或指向错误目标一律拒绝。
    """
    root = Path(root)
    production_data_root = Path(production_data_root).resolve()
    for name in READONLY_LINK_DIRS:
        candidate = root / name
        target = production_data_root / name
        is_link = candidate.is_symlink() or _is_junction(candidate)
        if not is_link:
            raise ValueError(f"只读接入目录必须是联接而非真实目录: {candidate}")
        if candidate.resolve() != target.resolve():
            raise ValueError(f"只读接入目录指向错误目标: {candidate} -> {candidate.resolve()}")


def collect_consumption_snapshot(
    files: List[Path],
    base_dir: Path,
    out_path: Path,
    *,
    missing_files: Optional[List[Path]] = None,
) -> List[Dict[str, object]]:
    """消费集合物理身份快照（§13.3.4 唯一硬门；复用 §5-1 store_identity 机制）并落盘。

    只枚举本次运行实际消费的文件；集合外的正常日更追加不参与比对（不失败）。

    ``missing_files``（R2-T5AE-R1-02）：采集时**不存在但属于本次候选**的路径，以
    ``{"path": ..., "missing": true}`` 记录追加登记（无内容身份）——评分/运行期间
    该候选出现即被闭合核验判为「快照后新增实读候选」。默认 ``None`` ⇒ 落盘格式与
    现状逐位一致。
    """
    from src.lazybull.v2.store.store_identity import collect_file_identities

    records = collect_file_identities(files, base_dir=base_dir)
    for path in missing_files or []:
        records.append({"path": str(path), "missing": True})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"消费集合身份快照: {len(records)} 个文件 -> {out_path}")
    return records


def verify_consumption_snapshot(snapshot_path: Path, base_dir: Path) -> List[str]:
    """按落盘清单重算并返回漂移路径；空列表 = 消费集合运行前后逐位一致。"""
    import json

    from src.lazybull.v2.store.store_identity import verify_file_identities

    expected = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    return verify_file_identities(expected, base_dir=base_dir)


def verify_consumption_snapshot_closed(
    snapshot_path: Path, base_dir: Path, *, rescan_paths: Optional[List[Path]] = None
) -> List[str]:
    """消费集合闭合核验（R2-T5AE-R1-02）：清单身份 + 缺失态 + 快照后新增检测。

    在 ``verify_consumption_snapshot``（逐文件内容身份）之上追加两条硬门：

    - **缺失候选态**：快照中 ``missing: true`` 的候选（采集时不存在）当前仍须
      缺失；出现 ⇒ 漂移（快照后新增实读候选，读取无跑前身份）；
    - **重枚举闭合**（``rescan_paths``）：评分/运行后重新枚举的现存消费候选，
      其显示路径必须 ⊆ 快照正常记录；多出 ⇒ 漂移（快照后新增实读候选）。

    三种情形（快照后新增 / 替换 / 已有变更）分别由重枚举闭合、清单身份哈希、
    清单身份哈希捕获。旧格式快照（无 ``missing`` 记录）向后兼容。漂移消息与
    清单身份漂移同列返回，调用方统一按非空 ⇒ FAIL 处置。
    """
    import json

    from src.lazybull.v2.store.store_identity import (
        _display_path,
        verify_file_identities,
    )

    base_dir = Path(base_dir)
    records = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"消费快照顶层须为记录列表: {snapshot_path}")
    present = [record for record in records if not record.get("missing")]
    missing = [record for record in records if record.get("missing")]
    drift = verify_file_identities(present, base_dir=base_dir)
    for record in missing:
        raw_path = Path(str(record["path"]))
        path = raw_path if raw_path.is_absolute() else base_dir / raw_path
        if path.exists():
            drift.append(f"快照后新增实读候选（采集时缺失现已出现）: {record['path']}")
    if rescan_paths is not None:
        registered = {str(record["path"]) for record in present}
        for path in rescan_paths:
            display = str(_display_path(Path(path), base_dir))
            if display not in registered:
                drift.append(f"快照后新增实读候选（重枚举发现）: {display}")
    return drift


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _is_junction(path: Path) -> bool:
    return bool(os.path.isjunction(str(path))) if hasattr(os.path, "isjunction") else False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_production_models_guard(production_data_root: Path) -> Dict[str, object]:
    """生产选股模型目录守卫快照：文件清单（大小 + mtime_ns）+ 关键索引文件内容哈希。"""
    base = Path(production_data_root) / "models" / "stock_selection"
    if not base.is_dir():
        raise FileNotFoundError(f"生产选股模型目录不存在: {base}")
    entries: Dict[str, List[int]] = {}
    for dirpath, _dirnames, filenames in os.walk(base):
        for filename in filenames:
            path = Path(dirpath) / filename
            stat = path.stat()
            rel = path.relative_to(base).as_posix()
            entries[rel] = [int(stat.st_size), int(stat.st_mtime_ns)]
    hashes = {}
    for name in GUARD_HASH_FILES:
        candidate = base / name
        if candidate.exists():
            hashes[name] = _sha256_file(candidate)
    logger.info(f"生产模型目录守卫快照: {base}（{len(entries)} 个文件）")
    return {"base": str(base), "entries": entries, "hashes": hashes}


def verify_production_models_guard(snapshot: Dict[str, object]) -> List[str]:
    """重算守卫快照并返回漂移描述清单；空列表表示生产模型目录零变化。"""
    base = Path(str(snapshot["base"]))
    before_entries: Dict[str, List[int]] = dict(snapshot["entries"])  # type: ignore[arg-type]
    before_hashes: Dict[str, str] = dict(snapshot["hashes"])  # type: ignore[arg-type]
    drift: List[str] = []

    after_entries: Dict[str, List[int]] = {}
    for dirpath, _dirnames, filenames in os.walk(base):
        for filename in filenames:
            path = Path(dirpath) / filename
            stat = path.stat()
            rel = path.relative_to(base).as_posix()
            after_entries[rel] = [int(stat.st_size), int(stat.st_mtime_ns)]

    for rel in sorted(set(before_entries) | set(after_entries)):
        if rel not in before_entries:
            drift.append(f"新增文件: {rel}")
        elif rel not in after_entries:
            drift.append(f"删除文件: {rel}")
        elif before_entries[rel] != after_entries[rel]:
            drift.append(f"文件变化: {rel}")

    for name, before_hash in before_hashes.items():
        candidate = base / name
        if not candidate.exists():
            continue  # 缺失已在文件清单比对中登记
        after_hash = _sha256_file(candidate)
        if after_hash != before_hash:
            drift.append(f"索引文件内容变化: {name}")
    return drift
