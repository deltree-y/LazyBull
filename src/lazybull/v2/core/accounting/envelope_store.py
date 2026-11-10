"""P6.5 T5 状态信封的原子落盘与同代链（文件 IO 唯一入口）。

契约依据：设计 §4.5（同一原子提交点落盘、先全量校验再统一写入）、§10.1 G5
（``state_envelopes/`` 两代信封本体 + SHA）。

落盘规则：
- 一代 = 一个文件 ``<root>/envelope-<提交点:06d>.json``（规范化 JSON 字节，末位单换行）；
- **原子替换**：先写同目录临时名 ``.envelope-<提交点:06d>.json.tmp``（``O_EXCL``），
  ``flush`` + ``fsync`` 后 ``os.replace`` 到最终名；单文件替换因此"要么旧、要么新"，
  不存在半成品最终文件；**已知边界**：只对文件 fsync、不对父目录 fsync（Windows 无
  portability 对应调用），因此"掉电后 rename 是否已持久"由文件系统决定，本模块的
  保证是"进程内可见状态要么旧、要么新"，掉电现场按半提交代处理（可检出、可隔离）；
- **跨代水位门（写入必过）**：`commit` 在写盘前对**当前代与上一代**同时调用
  `assert_no_watermark_regression` 与 `assert_no_state_shrink`（幂等键集只增不减、键序
  不重排、最后成交日不倒退、事件序号单调、链级批次身份只增不减），并额外比对**链级
  批次指纹值**（同一 `batch_id` 的请求指纹跨代变化即"另一套事实"，拒绝）；
- **禁止覆盖**：最终名已存在即拒绝（不删除、不改写既有代）；
- **单写者前提**：`target.exists()` 检查与 `os.replace` 之间存在 TOCTOU 窗口，本模块
  只保证**单进程单写者**下的原子性（T5-A-R2-03 登记）；并发写同一链不受支持——链根
  按运行 ID 隔离，跨进程并发写入须由调用方另行加锁；
- **信封摘要** = 最终文件字节的 SHA-256（文件不含自身摘要，避免自引用）；摘要记入
  下一代 ``previous_envelope_sha256`` 与证据 manifest；
- **零副作用**：任何校验失败都不产生新文件、不改动既有文件与目录内容。

半提交代（崩溃现场）语义：
- 崩溃可能留下临时名残留或（磁盘/掉电下）截断的最终名；两者都是"半提交代"，
  ``load`` 一律抛 ``StateEnvelopeCorruptError``（终态），``load_latest`` **不静默回退**
  到旧代；
- 恢复唯一路径：``quarantine_incomplete(提交点)`` 把残骸**移动**到 ``<root>/crashed/``
  （保留现场、禁止覆盖），再由调用方从**提交点之前**的完整信封新建实例重放；
- 存在半提交代残骸时，同提交点的再次 ``commit`` 被拒绝（"不能用已推进的内存水位
  重复提交"）。

边界：本模块只管字节与目录，不解释状态内容（五项语义见 ``state_owners``）；不写
``data/runs/``、不读配置、不读真实行情。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple

from src.lazybull.v2.core.accounting.state_envelope import (
    STAGE,
    StateEnvelope,
    StateEnvelopeCorruptError,
    StateEnvelopeError,
    StateEnvelopeIntegrityError,
    assert_no_state_shrink,
    assert_no_watermark_regression,
    envelope_bytes,
    parse_document,
    sha256_bytes,
    validate_envelope,
)
from src.lazybull.v2.core.accounting.state_owners import decode_owners

#: 提交点文件前缀/后缀（磁盘命名唯一来源）
ENVELOPE_FILE_PREFIX = "envelope-"
ENVELOPE_FILE_SUFFIX = ".json"
#: 临时名后缀（半提交代检测）
TEMP_SUFFIX = ".tmp"
#: 崩溃现场隔离目录名（保留残骸，禁止覆盖）
CRASHED_DIR_NAME = "crashed"


class EnvelopeStoreError(StateEnvelopeError):
    """存储层输入/路径/状态拒绝（**零副作用**）。"""


def _error(message: str) -> Exception:
    """带阶段上下文的存储层拒绝。"""
    return EnvelopeStoreError(f"阶段={STAGE} 状态信封存储 {message}")


@dataclass(frozen=True, slots=True)
class StoredEnvelope:
    """已落盘的信封（文件路径 + 文件字节摘要 + 赋予摘要后的信封对象）。"""

    envelope: StateEnvelope
    path: Path
    file_sha256: str
    byte_size: int

    @property
    def commit_point(self) -> int:
        """提交点。"""
        return self.envelope.commit_point

    @property
    def generation_token(self) -> str:
        """同代标记。"""
        return self.envelope.generation_token


def _commit_point_of(name: str) -> Optional[int]:
    """文件名 → 提交点（不匹配命名规则返回 None）。"""
    if not (name.startswith(ENVELOPE_FILE_PREFIX) and name.endswith(ENVELOPE_FILE_SUFFIX)):
        return None
    digits = name[len(ENVELOPE_FILE_PREFIX) : -len(ENVELOPE_FILE_SUFFIX)]
    if not digits.isdigit():
        return None
    return int(digits)


def _commit_point_of_temp(name: str) -> Optional[int]:
    """临时文件名 → 提交点（``.envelope-000002.json.tmp``；不匹配返回 None）。"""
    body = name[1:] if name.startswith(".") else name
    if not body.endswith(TEMP_SUFFIX):
        return None
    return _commit_point_of(body[: -len(TEMP_SUFFIX)])


def _commit_point_of_crash_note(name: str) -> Optional[int]:
    """崩溃现场 note 文件名 → 提交点（``.envelope-000002.json.tmp.<8hex>.tmp.note.json``）。"""
    marker = f"{ENVELOPE_FILE_PREFIX}"
    start = name.find(marker)
    if start < 0:
        return None
    rest = name[start + len(marker) :]
    digits = "".join(char for char in rest[:8] if char.isdigit())
    if len(digits) < 6:
        return None
    return int(digits[:6])


class EnvelopeStore:
    """单运行的同代信封链（一个目录 = 一条链；跨运行目录互不可见）。"""

    def __init__(self, root: Path, *, run_id: str) -> None:
        from src.lazybull.v2.core.accounting.state_envelope import require_envelope_run_id

        self._root = Path(root)
        self._run_id = require_envelope_run_id(run_id)

    @property
    def root(self) -> Path:
        """链根目录。"""
        return self._root

    @property
    def run_id(self) -> str:
        """链运行标识。"""
        return self._run_id

    def path_of(self, commit_point: int) -> Path:
        """提交点 → 最终文件路径。"""
        self._require_commit_point(commit_point)
        return self._root / f"{ENVELOPE_FILE_PREFIX}{commit_point:06d}{ENVELOPE_FILE_SUFFIX}"

    def temp_path_of(self, commit_point: int) -> Path:
        """提交点 → 临时文件路径（半提交代检测用）。"""
        self._require_commit_point(commit_point)
        return self._root / (
            f".{ENVELOPE_FILE_PREFIX}{commit_point:06d}{ENVELOPE_FILE_SUFFIX}{TEMP_SUFFIX}"
        )

    @staticmethod
    def _require_commit_point(commit_point: Any) -> int:
        """提交点严格校验（非 bool 正整数）。"""
        if isinstance(commit_point, bool) or not isinstance(commit_point, int) or commit_point < 1:
            raise _error(f"提交点必须为 >=1 的 int，实际 {commit_point!r}")
        return commit_point

    # ────────────────────────── 读取 ──────────────────────────

    def commit_points(self) -> Tuple[int, ...]:
        """磁盘上存在的提交点（含损坏文件；升序）。"""
        if not self._root.is_dir():
            return ()
        points: List[int] = []
        for entry in self._root.iterdir():
            if not entry.is_file():
                continue
            point = _commit_point_of(entry.name)
            if point is not None:
                points.append(point)
        return tuple(sorted(points))

    def temp_files(self) -> Tuple[Path, ...]:
        """残留临时文件（半提交代现场；升序）。"""
        if not self._root.is_dir():
            return ()
        return tuple(
            sorted(
                entry
                for entry in self._root.iterdir()
                if entry.is_file() and entry.name.endswith(TEMP_SUFFIX)
            )
        )

    def latest_commit_point(self) -> int:
        """磁盘上最大提交点（0 = 无信封）。"""
        points = self.commit_points()
        return points[-1] if points else 0

    def temp_only_points(self) -> Tuple[int, ...]:
        """只有临时名、没有最终名的提交点（**半提交现场**；升序）。

        真实写盘顺序是"临时文件 → fsync → rename"，因此 rename 之前进程退出最自然的
        现场就是"只有 temp、没有 final"。这类现场必须被读侧识别为半提交代，不能当成
        链尾完整（T5-Codex-C2）。
        """
        finals = set(self.commit_points())
        points = {
            point
            for point in (_commit_point_of_temp(path.name) for path in self.temp_files())
            if point is not None and point not in finals
        }
        return tuple(sorted(points))

    def quarantine_note_points(self) -> Tuple[int, ...]:
        """已登记的崩溃现场（``crashed/*.note.json``）对应的提交点（升序去重）。"""
        directory = self.crashed_dir()
        if not directory.is_dir():
            return ()
        points = {
            point
            for point in (
                _commit_point_of_crash_note(entry.name)
                for entry in directory.iterdir()
                if entry.is_file() and entry.name.endswith(".note.json")
            )
            if point is not None
        }
        return tuple(sorted(points))

    def quarantine_note_files(self) -> Tuple[Path, ...]:
        """崩溃现场 note 文件（升序；用于"实例是否在隔离之后重建"的判定）。"""
        directory = self.crashed_dir()
        if not directory.is_dir():
            return ()
        return tuple(
            sorted(
                entry
                for entry in directory.iterdir()
                if entry.is_file() and entry.name.endswith(".note.json")
            )
        )

    def quarantine_note_files_of(self, commit_point: int) -> Tuple[Path, ...]:
        """某提交点的崩溃现场 note 文件（升序）。"""
        point = self._require_commit_point(commit_point)
        return tuple(
            entry
            for entry in self.quarantine_note_files()
            if _commit_point_of_crash_note(entry.name) == point
        )

    def load(self, commit_point: int) -> StoredEnvelope:
        """读取并全量校验单代信封（损坏/缺失即拒绝，**不做任何回退**）。"""
        point = self._require_commit_point(commit_point)
        path = self.path_of(point)
        if not path.is_file():
            raise _error(
                f"提交点 {point} 的信封不存在（路径 {path}）；"
                f"链上已有提交点={list(self.commit_points())}"
            )
        raw = path.read_bytes()
        digest = sha256_bytes(raw)
        try:
            document = json.loads(raw.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise StateEnvelopeCorruptError(
                f"阶段={STAGE} 信封文件 {path.name} 不可解析（提交点 {point}，"
                f"字节 {len(raw)}，摘要 {digest}）：{error}；"
                "半提交代只能隔离现场后从提交点之前重放"
            ) from error
        envelope = parse_document(document)
        if envelope.run_id != self._run_id:
            raise StateEnvelopeIntegrityError(
                f"阶段={STAGE} 信封 {envelope.generation_token} 的运行 ID 与链不符"
                f"（链={self._run_id!r}，文件={envelope.run_id!r}）"
            )
        if envelope.commit_point != point:
            raise StateEnvelopeIntegrityError(
                f"阶段={STAGE} 信封文件名提交点 {point} 与内容提交点 "
                f"{envelope.commit_point} 不一致"
            )
        stored = StoredEnvelope(
            envelope=replace(envelope, envelope_sha256=digest),
            path=path,
            file_sha256=digest,
            byte_size=len(raw),
        )
        previous = None
        if point > 1:
            previous_path = self.path_of(point - 1)
            if not previous_path.is_file():
                raise StateEnvelopeIntegrityError(
                    f"阶段={STAGE} 信封链断裂：提交点 {point} 存在但上一代 {point - 1} 缺失"
                )
            previous = self.load(point - 1).envelope
        validate_envelope(stored.envelope, expected_run_id=self._run_id, previous=previous)
        return stored

    def load_latest(self) -> StoredEnvelope:
        """读取最大提交点信封（**不静默回退**：最大代损坏或存在半提交现场即拒绝）。

        T5-Codex-C2：只扫最终名的旧实现会把"只有临时文件"的现场当成链尾完整代；
        这里显式检查临时残留，把"链尾之后还有未完成的代"作为损坏/不完整错误报出。
        """
        points = self.commit_points()
        point = points[-1] if points else 0
        residual = self.temp_only_points()
        if residual and (point == 0 or residual[-1] > point):
            raise StateEnvelopeCorruptError(
                f"阶段={STAGE} 信封链 {self._root} 的提交点 {residual[-1]} 只有临时文件、"
                f"没有最终文件（半提交现场；已有完整提交点={list(points)}）；"
                "禁止把现场当作完整链尾——先 quarantine_incomplete 隔离，再从之前的完整信封重放"
            )
        if point == 0:
            raise _error(f"链 {self._root} 尚无任何信封（提交点=0）")
        return self.load(point)

    def scan(self) -> Tuple[Tuple[int, str], ...]:
        """逐提交点扫描状态（``ok`` / 错误类别；证据与诊断用，不改任何文件）。"""
        report: List[Tuple[int, str]] = []
        for point in self.commit_points():
            try:
                self.load(point)
            except StateEnvelopeCorruptError:
                report.append((point, "corrupt"))
            except (StateEnvelopeIntegrityError, StateEnvelopeError):
                report.append((point, "invalid"))
            else:
                report.append((point, "ok"))
        return tuple(report)

    # ────────────────────────── 写入 ──────────────────────────

    def commit(self, envelope: StateEnvelope) -> StoredEnvelope:
        """原子落盘一代信封（校验先行；失败零副作用）。

        Raises:
            EnvelopeStoreError: 提交点非连续/已存在/半提交代残留/写入失败。
            StateEnvelopeError: 信封内容校验拒绝。
            StateEnvelopeIntegrityError: 与上一代代际不符。
        """
        if not isinstance(envelope, StateEnvelope):
            raise _error(f"commit 需要 StateEnvelope，实际 {type(envelope).__name__}")
        validate_envelope(envelope, expected_run_id=self._run_id)
        # 写入前必须证明"这份信封可恢复"：五项联合解码 + 跨项同代一致性全部先行，
        # 否则会把不可恢复的信封落盘（违反"先全量校验再统一写入"）
        restored = decode_owners(envelope)
        point = self._require_commit_point(envelope.commit_point)
        # 覆盖/残骸先于连续性判定：两者都是"同一提交点不得再写一次"的硬门
        target = self.path_of(point)
        if target.exists():
            raise _error(
                f"提交点 {point} 的信封已存在（{target.name}）；禁止覆盖既有信封"
                "（半提交代也同样是既存文件：先隔离现场再从提交点之前重放）"
            )
        temp = self.temp_path_of(point)
        if temp.exists():
            raise _error(
                f"提交点 {point} 存在半提交代残留（{temp.name}）；"
                "禁止用已推进的内存水位重复提交——必须先 quarantine_incomplete 隔离"
                "现场，再从提交点之前的完整信封新建实例重放"
            )
        latest = self.latest_commit_point()
        if point != latest + 1:
            raise _error(
                f"提交点必须严格连续：链上最新 {latest}，本次 {point}"
                f"（期望 {latest + 1}）；禁止跳号、倒退或跨代重用水位"
            )
        previous = self.load(point - 1).envelope if point > 1 else None
        validate_envelope(envelope, expected_run_id=self._run_id, previous=previous)
        if previous is not None:
            self._assert_generation_advance(
                envelope=envelope, previous=previous, restored=restored
            )
        payload = envelope_bytes(envelope)
        self._root.mkdir(parents=True, exist_ok=True)
        try:
            with open(temp, "xb") as handle:  # 独占创建：并发/残留都不会静默覆盖
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, target)
        except BaseException:
            # 零副作用：临时文件不留、最终文件不产生、既有代不改动
            try:
                if temp.exists():
                    temp.unlink()
            except OSError:  # pragma: no cover - 清理失败不掩盖原始异常
                pass
            raise
        digest = sha256_bytes(payload)
        if sha256_bytes(target.read_bytes()) != digest:  # pragma: no cover - 落盘后自检
            raise StateEnvelopeIntegrityError(
                f"阶段={STAGE} 信封 {point} 落盘后摘要不符（期望 {digest}）；"
                "按半提交代处理，禁止继续提交"
            )
        return StoredEnvelope(
            envelope=replace(envelope, envelope_sha256=digest),
            path=target,
            file_sha256=digest,
            byte_size=len(payload),
        )

    @staticmethod
    def _watermark_projection(restored: Any) -> Mapping[str, Any]:
        """解码结果 → 跨代水位投影（含链级批次身份；``last_fill_date`` 归一为字符串）。"""
        watermark = dict(restored.identity.watermark())
        watermark["chain_batch_ids"] = [
            batch_id for batch_id, _fingerprint in restored.identity.chain_batch_fingerprints
        ]
        return watermark

    def _assert_generation_advance(
        self, *, envelope: StateEnvelope, previous: StateEnvelope, restored: Any
    ) -> None:
        """跨代水位校验（写入路径的**必过关**，不是只给测试用的工具函数）。

        与上一代逐项比较：幂等键集只增不减、键序不得重排、最后成交日不得倒退、
        结算事件序号单调不减、**链级批次身份只增不减**。这挡住"把旧代状态写成新提交点"
        （``restart(n)`` 后不跑第 n+1 日直接 ``commit``）——该场景下所有同代内一致性
        校验都会通过，只有跨代比较能发现。
        """
        previous_restored = decode_owners(previous)
        current_watermark = self._watermark_projection(restored)
        previous_watermark = self._watermark_projection(previous_restored)
        label = (
            f"阶段={STAGE} 信封 {previous.generation_token}→{envelope.generation_token} "
            f"日期={previous.trade_date}→{envelope.trade_date}"
        )
        assert_no_watermark_regression(
            current=current_watermark, previous=previous_watermark, label=label
        )
        assert_no_state_shrink(
            current=current_watermark, previous=previous_watermark, label=label
        )
        # 链级身份不只看 id 集合，还要看**指纹值**：同一批次键的请求变了即"另一套事实"
        # （T5-A-R2-01）——否则只改指纹就能绕过"同一身份不得承载两套请求"。
        previous_fingerprints = dict(previous_restored.identity.chain_batch_fingerprints)
        current_fingerprints = dict(restored.identity.chain_batch_fingerprints)
        changed = sorted(
            batch_id
            for batch_id, fingerprint in previous_fingerprints.items()
            if current_fingerprints.get(batch_id, fingerprint) != fingerprint
        )
        if changed:
            raise StateEnvelopeIntegrityError(
                f"{label}：链级批次身份 {changed[:3]} 的请求指纹跨代变化；"
                "同一批次键不得承载两套事实（禁止借重算指纹改写历史事件）"
            )

    # ────────────────────────── 崩溃现场 ──────────────────────────

    def crashed_dir(self) -> Path:
        """崩溃现场隔离目录（``<root>/crashed``）。"""
        return self._root / CRASHED_DIR_NAME

    def quarantine_incomplete(self, commit_point: int, *, reason: str = "") -> Tuple[Path, ...]:
        """把某提交点的半提交代残骸**移动**到 ``crashed/``（保留现场、禁止覆盖）。

        Args:
            commit_point: 半提交代提交点
            reason: 隔离原因（写入残骸同名的 ``.note.json``，供复核）

        Returns:
            被移动的目标路径（按临时名、最终名顺序；无残骸返回空元组）。

        Raises:
            EnvelopeStoreError: 残骸目标已存在（禁止覆盖既有现场）或移动失败。
        """
        point = self._require_commit_point(commit_point)
        candidates = [self.temp_path_of(point), self.path_of(point)]
        existing = [path for path in candidates if path.is_file()]
        if not existing:
            raise _error(f"提交点 {point} 没有可隔离的残骸（临时名与最终名都不存在）")
        target_dir = self.crashed_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
        moved: List[Path] = []
        for path in existing:
            raw = path.read_bytes()
            digest = sha256_bytes(raw)
            target = target_dir / f"{path.name}.{digest[:8]}{TEMP_SUFFIX}"
            if target.exists():
                raise _error(
                    f"崩溃现场目标已存在（{target.name}）；禁止覆盖既有残骸证据"
                )
            os.replace(path, target)
            moved.append(target)
            note = {
                "commit_point": point,
                "source_name": path.name,
                "bytes": len(raw),
                "sha256": digest,
                "reason": reason or "<未登记>",
                "rule": "半提交代只能隔离现场，并从提交点之前的完整信封新建实例重放",
            }
            note_path = target_dir / f"{target.name}.note.json"
            note_path.write_bytes(
                (
                    json.dumps(note, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
                ).encode("utf-8")
            )
        return tuple(moved)

    def assert_chain_intact(self) -> Tuple[int, ...]:
        """整链完整性断言（逐个提交点全量校验；半提交现场即抛）。

        T5-Codex-C2：临时残留（无论是否已有同名最终名）一律视为不完整现场——
        有临时名而无最终名是"rename 前退出"，两者并存是"rename 后残留"。
        """
        residual = self.temp_only_points()
        if residual:
            raise StateEnvelopeCorruptError(
                f"阶段={STAGE} 信封链 {self._root} 存在只有临时文件的提交点 {list(residual)}"
                f"（已有完整提交点={list(self.commit_points())}）；半提交现场必须先隔离再判定整链"
            )
        points = self.commit_points()
        for point in points:
            self.load(point)
        if points and points != tuple(range(1, points[-1] + 1)):
            raise _error(f"提交点不连续：{list(points)}（必须从 1 起的连续段）")
        leftovers = [path.name for path in self.temp_files()]
        if leftovers:
            raise StateEnvelopeCorruptError(
                f"阶段={STAGE} 信封链 {self._root} 残留临时文件 {leftovers[:3]}；"
                "临时文件只允许出现在隔离现场，禁止作为整链完整的一部分"
            )
        return points


def store_for(root: Path, *, run_id: str) -> EnvelopeStore:
    """构造存储（构造期只校验运行标识，不触碰磁盘）。"""
    return EnvelopeStore(Path(root), run_id=run_id)


def envelope_file_document(path: Path) -> Mapping[str, Any]:
    """只读解析磁盘信封文件（不做完整校验；证据层登记用）。"""
    try:
        return json.loads(Path(path).read_bytes().decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StateEnvelopeCorruptError(
            f"阶段={STAGE} 信封文件 {Path(path).name} 不可解析：{error}"
        ) from error


__all__ = (
    "CRASHED_DIR_NAME",
    "ENVELOPE_FILE_PREFIX",
    "ENVELOPE_FILE_SUFFIX",
    "EnvelopeStore",
    "EnvelopeStoreError",
    "StoredEnvelope",
    "TEMP_SUFFIX",
    "envelope_file_document",
    "store_for",
)
