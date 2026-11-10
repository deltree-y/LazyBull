"""P6.5 T3 runs 双臂物理隔离与 generation 身份绑定（仅机制与守卫；比较归 T4）。

契约依据：设计 ``docs/plans/v2/p65_merger_reconciliation_design_20261108.md``
§8.4（Q9=B：既有 writer、两臂独立根、写前不存在、生成身份绑定）与 T3 任务 §3.3。

职责与边界：
- 复用 ``hosts/backtest/runs_writer.py::write_runs_batch``（**不新增第二套投影**）；
  每臂根为 ``<base>/<run_id>/single/runs_root`` 与 ``<base>/<run_id>/split/runs_root``
  （不同兄弟目录、互不嵌套）；
- 写出前断言 ``out_dir = runs_root/<batch_id>`` **不存在**（不依赖 writer 的
  ``exist_ok=True``）；同一臂的 generation 已存在即拒绝（禁止复用/覆盖身份）；
- 每根外独立 generation manifest：臂标签、解析后的根/`out_dir`、输入信号与 config
  哈希、写前不存在断言、写后文件哈希与内容派生的 ``generation_id``；
- 自我比较防护：两臂同根/嵌套、写前目录已存在、A 产物整体复制给 B（``arms``
  标签不符）、篡改根映射、复用 generation，全部在写前或校验期失败；
- 可缺产物语义：``attribution``/``holdings_snapshot`` 单侧缺失即失败；两侧同源
  缺失登记 ``not_compared/source_absent``；``topk_detail`` 登记
  ``not_compared/contract_optional_missing``（writer 契约本就不写该文件）。

不实现（T4）：两臂经济字段逐位比较、自拆臂装配、G4 双 NAV 门。本模块对经济
字段只做"文件存在性 + 身份绑定"，不得据此宣称 G4 或逐位一致。
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: 单袖参照臂标签（不同构造标签，不作逐位证据）
ARM_SINGLE = "single_reference"
#: 虚拟自拆臂标签
ARM_SPLIT = "virtual_split"
#: 两臂根目录名（兄弟、不嵌套）
ARM_DIRS: Mapping[str, str] = {ARM_SINGLE: "single", ARM_SPLIT: "split"}
#: 本模块产物来源标识
SOURCE = "v2_p65_t3_synthetic_split"
#: runs 契约的批次级必需文件（缺一即失败）
REQUIRED_BATCH_FILES: Tuple[str, ...] = (
    "batch_meta.json",
    "summary.parquet",
    "chain_nav.parquet",
)
#: 折级必需文件
REQUIRED_FOLD_FILES: Tuple[str, ...] = ("trades.parquet", "daily.parquet", "_meta.json")
#: 折级可缺文件（单侧缺失即失败；两侧同源缺失登记 not_compared）
OPTIONAL_FOLD_FILES: Tuple[str, ...] = ("attribution.parquet", "holdings_snapshot.parquet")
#: 契约明确不写的可选项（两侧同源缺失，登记为 contract_optional_missing）
CONTRACT_OPTIONAL_FOLD_FILES: Tuple[str, ...] = ("topk_detail.parquet",)

_STATUS_PRESENT = "present"
_STATUS_SOURCE_ABSENT = "not_compared/source_absent"
_STATUS_CONTRACT_OPTIONAL = "not_compared/contract_optional_missing"

_GENERATION_KEYS = (
    "schema_version",
    "run_id",
    "arm",
    "batch_id",
    "runs_root",
    "out_dir",
    "source",
    "signal_fingerprint",
    "config_fingerprint",
    "pre_write_absent",
    "pre_write_parent_entries",
    "file_hashes",
    "created_at",
    "generation_id",
)


class ArmIsolationError(ValueError):
    """两臂身份/路径/写前状态非法（**零副作用**拒绝）。"""


def _error(message: str) -> Exception:
    """带阶段上下文的双臂拒绝。"""
    return ArmIsolationError(f"阶段=P6.5 T3 runs {message}")


def _require_ascii_id(value: Any, field: str) -> str:
    """路径身份：非空 ASCII 且不含路径分隔符（防止根映射被字符串改写）。

    **不接受首尾空白**：``.strip()`` 曾使预检/清理使用 ``b1``、而 writer 收到
    ``" b1"`` 并向另一目录写出（外部评审 R1-01），这里直接拒绝以保证预检、写出、
    generation 与清理使用同一路径身份。
    """
    raw = str(value if value is not None else "")
    text = raw.strip()
    if not text:
        raise _error(f"{field} 不得为空")
    if raw != text:
        raise _error(
            f"{field} 不得含首尾空白（实际 {value!r}）；路径身份必须在预检/写出/清理间一致"
        )
    if not text.isascii():
        raise _error(f"{field} 必须为 ASCII：{text!r}")
    if any(sep in text for sep in ("/", "\\", "..", ":")):
        raise _error(f"{field} 不得包含路径分隔符/上跳：{text!r}")
    if text in {".", ".."}:
        # 单段身份必须能作为**目录名**使用：``.`` 会解析回根自身（复审 R1-02），
        # 使"预检通过、深层再拒"之间产生日志等副作用，故在此一并拒绝。
        raise _error(f"{field} 不得为当前/父目录标记：{text!r}")
    return text


def require_ascii_id(value: Any, field: str) -> str:
    """公开入口：单段路径身份校验（预检/写出/清理/证据根共用**同一规则**）。

    证据层在 ``mkdir`` 之前调用本函数，避免"证据根之外先建目录、深层写入层才拒绝"
    的写域越界（外部评审 T4-Codex-05）。
    """
    return _require_ascii_id(value, field)


def _canonical(payload: Mapping[str, Any]) -> str:
    """规范化序列化（generation_id 与文件内容哈希的唯一口径）。"""
    return json.dumps(
        dict(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    )


def file_hashes(root: Path) -> Tuple[Tuple[str, str], ...]:
    """目录下全部文件的内容哈希（相对路径 POSIX，排序；写后绑定用）。"""
    root = Path(root)
    if not root.is_dir():
        raise _error(f"产物目录不存在：{root}")
    hashes: List[Tuple[str, str]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes.append((path.relative_to(root).as_posix(), digest))
    return tuple(hashes)


def assert_isolated_roots(roots: Mapping[str, Path]) -> None:
    """两臂根必须不同、互不嵌套（同根或嵌套都会造成自我比较/覆盖）。"""
    if set(roots) != set(ARM_DIRS):
        raise _error(f"臂集合必须为 {sorted(ARM_DIRS)}，实际 {sorted(roots)}")
    resolved = {arm: Path(path).resolve() for arm, path in roots.items()}
    left, right = resolved[ARM_SINGLE], resolved[ARM_SPLIT]
    if left == right:
        raise _error(f"两臂根相同：{left}（自我比较防护，拒绝共用 out_dir）")
    if left in right.parents or right in left.parents:
        raise _error(f"两臂根嵌套：{left} 与 {right}（必须为不同兄弟目录）")


@dataclass(frozen=True, slots=True)
class ArmGeneration:
    """单臂 generation manifest（内容派生身份；``created_at`` 不入比较）。"""

    run_id: str
    arm: str
    batch_id: str
    runs_root: str
    out_dir: str
    signal_fingerprint: str
    config_fingerprint: str
    pre_write_absent: bool
    #: 写前 ``runs_root`` 的实测条目快照（证明当时不含 ``batch_id`` 目录）
    pre_write_parent_entries: Tuple[str, ...]
    file_hashes: Tuple[Tuple[str, str], ...]
    generation_id: str
    created_at: str

    def to_document(self) -> Dict[str, Any]:
        """JSON 文档（键集封闭；``file_hashes`` 为二元组列表）。"""
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "arm": self.arm,
            "batch_id": self.batch_id,
            "runs_root": self.runs_root,
            "out_dir": self.out_dir,
            "source": SOURCE,
            "signal_fingerprint": self.signal_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "pre_write_absent": self.pre_write_absent,
            "pre_write_parent_entries": list(self.pre_write_parent_entries),
            "file_hashes": [[path, digest] for path, digest in self.file_hashes],
            "created_at": self.created_at,
            "generation_id": self.generation_id,
        }


def generation_identity(document: Mapping[str, Any]) -> str:
    """内容派生的 generation 身份（排除 ``generation_id`` 与 ``created_at``）。"""
    payload = {
        key: value for key, value in document.items() if key not in ("generation_id", "created_at")
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


class DualArmRunsWriter:
    """两臂 runs 写出器（复用 ``write_runs_batch``；仅机制与身份绑定）。"""

    def __init__(self, *, run_id: str, base_dir: Path, config: Mapping[str, Any]) -> None:
        if not isinstance(config, Mapping) or not config:
            raise _error("批次 config 必须为非空 Mapping（两臂固定输入）")
        self.run_id = _require_ascii_id(run_id, "run_id")
        self.base_dir = Path(base_dir).resolve()
        self.config = dict(config)
        self._roots = {
            arm: (self.base_dir / self.run_id / arm_dir / "runs_root").resolve()
            for arm, arm_dir in ARM_DIRS.items()
        }
        assert_isolated_roots(self._roots)
        self._arm_dirs = {
            arm: (self.base_dir / self.run_id / arm_dir).resolve()
            for arm, arm_dir in ARM_DIRS.items()
        }

    @property
    def roots(self) -> Mapping[str, Path]:
        """两臂 runs 根（只读投影）。"""
        return dict(self._roots)

    def out_dir(self, arm: str, batch_id: str) -> Path:
        """``runs_root/<batch_id>``（解析后）。"""
        if arm not in self._roots:
            raise _error(f"未知臂标签 {arm!r}；合法臂={sorted(self._roots)}")
        return (self._roots[arm] / _require_ascii_id(batch_id, "batch_id")).resolve()

    def generation_path(self, arm: str) -> Path:
        """根外 generation manifest 路径（每臂一份）。"""
        if arm not in self._arm_dirs:
            raise _error(f"未知臂标签 {arm!r}；合法臂={sorted(self._arm_dirs)}")
        return self._arm_dirs[arm] / "generation.json"

    # ────────────────────────── 写出 ──────────────────────────

    def assert_writable(self, *, arm: str, batch_id: str) -> None:
        """写前预检（**只读**）：out_dir 不存在、runs_root 无同名条目、generation 不存在。

        供多臂编排在**写任何一臂之前**做全量预检，避免"第一臂已落盘、第二臂才拒绝"的
        半副作用（T4 复审 B-R2 口径）。``write_arm`` 内部同样调用本预检，两处判定
        同源，不产生第二套规则。
        """
        self._preflight_write(arm=arm, batch_id=batch_id)

    def _preflight_write(self, *, arm: str, batch_id: str) -> Tuple[Path, Path, Tuple[str, ...]]:
        """写前不存在断言（返回 target/generation 路径与 runs_root 写前条目快照）。"""
        target = self.out_dir(arm, batch_id)
        generation_file = self.generation_path(arm)
        if target.exists():
            raise _error(
                f"臂={arm} 批次={batch_id}：写前 out_dir 已存在 {target}；"
                "禁止覆盖（不得依赖 writer 的 exist_ok），请使用新批次键"
            )
        parent = target.parent
        parent_entries = (
            tuple(sorted(item.name for item in parent.iterdir())) if parent.is_dir() else ()
        )
        if batch_id in parent_entries:
            raise _error(
                f"臂={arm} 批次={batch_id}：写前 runs_root 已含同名条目（实测 {parent}）；"
                "禁止覆盖"
            )
        if generation_file.exists():
            raise _error(
                f"臂={arm}：generation 已存在 {generation_file}；"
                "禁止复用/覆盖生成身份（每次运行每臂只允许一个 generation）"
            )
        return target, generation_file, parent_entries

    def write_arm(
        self,
        *,
        arm: str,
        results: Sequence[Mapping[str, object]],
        batch_id: str,
        code_state: Mapping[str, object],
        data_state: Mapping[str, object],
        signals: Mapping[str, object],
        created_at: str,
        extra_meta_notes: Optional[Mapping[str, str]] = None,
    ) -> ArmGeneration:
        """写出一臂并冻结 generation（写前不存在断言；失败零副作用）。

        writer 的契约校验可能发生在创建目录之后：任何异常都必须撤销本次新建的
        ``target``/``generation`` 与空父目录，否则半成品会污染证据根并占用批次身份
        （修正输入后无法重试）。写前已断言二者不存在，故清理不会删除既有内容。
        """
        from src.lazybull.v2.hosts.backtest.runs_writer import write_runs_batch

        # 入口统一路径身份：预检、writer、generation 与清理必须使用同一个 batch_id
        batch_id = _require_ascii_id(batch_id, "batch_id")
        target, generation_file, parent_entries = self._preflight_write(arm=arm, batch_id=batch_id)
        root_existed = self._roots[arm].is_dir()
        try:
            written = write_runs_batch(
                list(results),
                batch_id=batch_id,
                runs_root=self._roots[arm],
                config=self.config,
                code_state=dict(code_state),
                data_state=dict(data_state),
                arms=(arm,),
                source=SOURCE,
                extra_meta_notes=dict(extra_meta_notes) if extra_meta_notes else None,
            )
            if Path(written).resolve() != target:
                raise _error(
                    f"臂={arm} 批次={batch_id}：writer 返回目录 {written} != 预期 {target}"
                )
            self._assert_batch_binding(arm=arm, batch_id=batch_id, out_dir=target)
            hashes = file_hashes(target)
            document = {
                "schema_version": 1,
                "run_id": self.run_id,
                "arm": arm,
                "batch_id": batch_id,
                "runs_root": str(self._roots[arm]),
                "out_dir": str(target),
                "source": SOURCE,
                "signal_fingerprint": _payload_fingerprint(signals),
                "config_fingerprint": _payload_fingerprint(self.config),
                "pre_write_absent": True,
                "pre_write_parent_entries": list(parent_entries),
                "file_hashes": [[path, digest] for path, digest in hashes],
                "created_at": str(created_at),
            }
            document["generation_id"] = generation_identity(document)
            generation_file.write_text(
                json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
        except BaseException:
            self._rollback_failed_write(
                target=target, generation_file=generation_file, root_existed=root_existed
            )
            raise
        return _generation_from_document(document)

    @staticmethod
    def _rollback_failed_write(*, target: Path, generation_file: Path, root_existed: bool) -> None:
        """撤销写出失败产生的半成品；仅删除本次新建内容（写前已断言不存在）。"""
        if generation_file.exists():
            generation_file.unlink()
        if target.exists():
            shutil.rmtree(target)
        if root_existed:
            return
        for path in (target.parent, target.parent.parent, target.parent.parent.parent):
            try:
                path.rmdir()
            except OSError:
                break

    # ────────────────────────── 校验 ──────────────────────────

    def load_generation(self, *, arm: str, batch_id: str) -> ArmGeneration:
        """读取并校验 generation（根映射、内容哈希、臂标签三者必须全部自洽）。"""
        path = self.generation_path(arm)
        if not path.is_file():
            raise _error(f"臂={arm}：generation 缺失 {path}")
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, Mapping) or set(document) != set(_GENERATION_KEYS):
            raise _error(
                f"臂={arm}：generation 键集非法（期望 {sorted(_GENERATION_KEYS)}，"
                f"实际 {sorted(document) if isinstance(document, Mapping) else type(document)}）"
            )
        expected_out = self.out_dir(arm, batch_id)
        if document["run_id"] != self.run_id or document["arm"] != arm:
            raise _error(
                f"臂={arm}：generation 身份不符（run_id={document['run_id']!r} "
                f"arm={document['arm']!r}）；禁止复用其它运行/臂的生成身份"
            )
        if document["batch_id"] != batch_id:
            raise _error(f"臂={arm}：generation 批次={document['batch_id']!r} != 请求 {batch_id!r}")
        if document["runs_root"] != str(self._roots[arm]) or document["out_dir"] != str(
            expected_out
        ):
            raise _error(
                f"臂={arm}：generation 根映射被篡改（runs_root={document['runs_root']!r} "
                f"out_dir={document['out_dir']!r}）；解析后根必须为 {self._roots[arm]}"
            )
        if document["generation_id"] != generation_identity(document):
            raise _error(f"臂={arm}：generation_id 与内容不符（manifest 被篡改）")
        if document["pre_write_absent"] is not True:
            raise _error(f"臂={arm}：generation 未登记写前不存在断言")
        entries = document["pre_write_parent_entries"]
        if not isinstance(entries, (list, tuple)) or any(
            not isinstance(item, str) for item in entries
        ):
            raise _error(f"臂={arm}：generation 写前目录快照形态非法（必须为字符串列表）")
        if batch_id in entries:
            raise _error(
                f"臂={arm} 批次={batch_id}：写前目录快照已含同名条目（{list(entries)}）；"
                "manifest 与写前不存在断言自相矛盾"
            )
        self._assert_batch_binding(arm=arm, batch_id=batch_id, out_dir=expected_out)
        self._assert_tree_hashes(arm=arm, out_dir=expected_out, recorded=document["file_hashes"])
        return _generation_from_document(document)

    def _assert_batch_binding(self, *, arm: str, batch_id: str, out_dir: Path) -> None:
        """``batch_meta`` 必须自证臂标签与本批次身份（A 产物复制给 B 的判据）。"""
        meta_path = out_dir / "batch_meta.json"
        if not meta_path.is_file():
            raise _error(f"臂={arm} 批次={batch_id}：缺少 batch_meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("batch_id") != batch_id:
            raise _error(
                f"臂={arm} 批次={batch_id}：batch_meta.batch_id={meta.get('batch_id')!r} 不符"
            )
        arms = list(meta.get("arms") or [])
        if arms != [arm]:
            raise _error(
                f"臂={arm} 批次={batch_id}：batch_meta.arms={arms} != [{arm!r}]；"
                "疑似把另一臂产物复制到本臂（自我比较/身份冒用）"
            )

    @staticmethod
    def _assert_tree_hashes(*, arm: str, out_dir: Path, recorded: Any) -> None:
        """写后文件哈希必须与当前树逐文件一致（复制/篡改/单侧缺文件均可检出）。"""
        if not isinstance(recorded, (list, tuple)) or any(
            not isinstance(item, (list, tuple)) or len(item) != 2 for item in recorded
        ):
            raise _error(f"臂={arm}：generation file_hashes 形态非法（必须为二元组列表）")
        current = file_hashes(out_dir)
        recorded_pairs = tuple((str(path), str(digest)) for path, digest in recorded)
        if current != recorded_pairs:
            missing = sorted(set(dict(recorded_pairs)) - set(dict(current)))
            extra = sorted(set(dict(current)) - set(dict(recorded_pairs)))
            raise _error(
                f"臂={arm}：产物树与 generation 文件哈希不符（缺失={missing} 多出={extra}）；"
                "禁止用复制/篡改后的产物冒充本臂结果"
            )

    # ────────────────────────── 产物语义（比较归 T4） ──────────────────────────

    def artifact_status(self, *, batch_id: str) -> Mapping[str, str]:
        """两臂实际文件集与缺失语义（**不做经济比较**）。

        分类前置：两臂 ``batch_meta`` 身份绑定必须先通过（臂标签/批次身份），
        否则"文件集合一致"可能只是把另一臂产物整体复制过来的假一致；逐文件哈希
        由 ``load_generation`` 负责（本方法只做存在性分类，以便独立打红缺失语义）。
        """
        for arm in (ARM_SINGLE, ARM_SPLIT):
            self._assert_batch_binding(
                arm=arm, batch_id=batch_id, out_dir=self.out_dir(arm, batch_id)
            )
        single = self._collect_files(self.out_dir(ARM_SINGLE, batch_id))
        split = self._collect_files(self.out_dir(ARM_SPLIT, batch_id))
        return _classify_files(single, split)

    @staticmethod
    def _collect_files(out_dir: Path) -> Mapping[str, bool]:
        """目录下合同文件的存在性投影（批次级 + 逐折）。"""
        if not out_dir.is_dir():
            raise _error(f"产物目录不存在：{out_dir}")
        presence: Dict[str, bool] = {}
        for name in REQUIRED_BATCH_FILES:
            presence[name] = (out_dir / name).is_file()
        fold_root = out_dir / "folds"
        fold_dirs = (
            sorted(item for item in fold_root.iterdir() if item.is_dir())
            if fold_root.is_dir()
            else []
        )
        for fold in fold_dirs:
            for name in REQUIRED_FOLD_FILES + OPTIONAL_FOLD_FILES + CONTRACT_OPTIONAL_FOLD_FILES:
                presence[f"folds/{fold.name}/{name}"] = (fold / name).is_file()
        return presence


def _payload_fingerprint(payload: Mapping[str, Any]) -> str:
    """输入信号/配置的规范化指纹（16 位；与证据层同口径 sha256 截断）。"""
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()[:16]


def _classify_files(single: Mapping[str, bool], split: Mapping[str, bool]) -> Mapping[str, str]:
    """按契约语义分类：必需缺失/单侧可缺缺失即失败，同源缺失登记 not_compared。"""
    if set(single) != set(split):
        only_single = sorted(set(single) - set(split))
        only_split = sorted(set(split) - set(single))
        raise _error(
            f"两臂文件集合不一致（单侧缺文件不得记为一致）：仅 single 有={only_single}，"
            f"仅 split 有={only_split}"
        )
    status: Dict[str, str] = {}
    for rel in sorted(single):
        name = rel.rsplit("/", 1)[-1]
        left, right = single[rel], split[rel]
        if name in CONTRACT_OPTIONAL_FOLD_FILES:
            if left or right:
                raise _error(
                    f"{rel}：writer 契约不写 topk_detail，单侧出现即契约违反（"
                    f"single={left} split={right}）"
                )
            status[rel] = _STATUS_CONTRACT_OPTIONAL
            continue
        if name in OPTIONAL_FOLD_FILES:
            if left != right:
                raise _error(
                    f"{rel}：可缺产物单侧缺失（single={left} split={right}）；"
                    "单侧缺失必须失败，不得记为一致"
                )
            status[rel] = _STATUS_PRESENT if left else _STATUS_SOURCE_ABSENT
            continue
        if not (left and right):
            raise _error(f"{rel}：必需产物缺失（single={left} split={right}）；不得记为一致")
        status[rel] = _STATUS_PRESENT
    return status


def _generation_from_document(document: Mapping[str, Any]) -> ArmGeneration:
    """JSON 文档 → 冻结值对象（键集已校验）。"""
    hashes = tuple((str(path), str(digest)) for path, digest in document["file_hashes"])
    return ArmGeneration(
        run_id=str(document["run_id"]),
        arm=str(document["arm"]),
        batch_id=str(document["batch_id"]),
        runs_root=str(document["runs_root"]),
        out_dir=str(document["out_dir"]),
        signal_fingerprint=str(document["signal_fingerprint"]),
        config_fingerprint=str(document["config_fingerprint"]),
        pre_write_absent=bool(document["pre_write_absent"]),
        pre_write_parent_entries=tuple(str(item) for item in document["pre_write_parent_entries"]),
        file_hashes=hashes,
        generation_id=str(document["generation_id"]),
        created_at=str(document["created_at"]),
    )


def snapshot_tree_paths(root: Path) -> Tuple[str, ...]:
    """目录下全部条目（文件 + 目录）的相对路径快照（不存在即空）。"""
    root = Path(root)
    if not root.exists():
        return ()
    return tuple(sorted(item.relative_to(root).as_posix() for item in root.rglob("*")))


def remove_new_tree_paths(root: Path, snapshot: Sequence[str], *, remove_root: bool) -> None:
    """只撤销快照之后新增的条目；既有文件与目录原样保留（**不删已有根**）。

    Args:
        root: 运行根（``<base>/<run_id>``）。
        snapshot: 写前条目快照（``snapshot_tree_paths`` 的结果）。
        remove_root: 根原本不存在时连根一起删；原本存在时保留根与既有内容。
    """
    root = Path(root)
    if not root.exists():
        return
    before = set(snapshot)
    current = sorted(snapshot_tree_paths(root), key=lambda item: item.count("/"), reverse=True)
    for relative in current:
        if relative in before:
            continue
        path = root / relative
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:  # pragma: no cover - 目录内仍有既有文件时保留
                continue
    if remove_root and root.exists():
        shutil.rmtree(root, ignore_errors=True)


__all__: Tuple[str, ...] = (
    "ARM_SINGLE",
    "ARM_SPLIT",
    "ARM_DIRS",
    "ArmGeneration",
    "ArmIsolationError",
    "DualArmRunsWriter",
    "assert_isolated_roots",
    "file_hashes",
    "generation_identity",
    "remove_new_tree_paths",
    "require_ascii_id",
    "snapshot_tree_paths",
)
