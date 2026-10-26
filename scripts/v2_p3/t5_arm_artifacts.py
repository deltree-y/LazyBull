# -*- coding: utf-8 -*-
"""P3 T5 三臂统一批驱动——产物布局与纯函数件（§19.1-A；验收专用，不入生产链路）。

职责（无副作用扩散，落盘路径仅此模块登记）：

- **布局契约**（``data/frozen_reference/v2_p3/<run_name>/``）：
  ``arm_meta.json`` / ``before_guard.json`` / ``consumption_snapshot.json`` /
  ``after_guard.json`` / ``consumption_verify.json`` /
  ``manifest_baseline.json`` + ``manifest_verify.json``（R2-T5A-02，仅臂③启用
  扩展消费集合时落盘：panel 热/冷区 + labels 三表入字节门，manifest 族走
  相关性 diff 语义）/
  ``root/``（隔离运行根：junctions clean/raw/features/labels + 全新
  models/walk_forward/reports/paper/runs）/
  ``folds/splitNN/{model_version.txt, boundary.json, fingerprint.json}`` /
  ``batch_fingerprint.json`` / ``runs/<batch_id>/`` / ``chain_metrics.json``；
  double-run 模式 = ``run1/run2`` 子目录各自完整布局，基线快照共享于 run_dir 顶层；
- 逐折模型版本解析（隔离根台账 ``ml_train_runs.csv`` 按 ``split_index``，
  替代 latest 指针单值口径）与 skip-training 起始版本反解（编排器
  ``model_version = start_model_version + split_index`` 算术口径的逆运算 +
  全折一致性断言）；
- 全批汇总指纹（§13.3.3：逐折指纹按折号升序 LF 拼接的 sha256）；
- 链式指标（``v2/common/chain_metrics.py`` 唯一实现，禁止自建公式）；
- 承接④ 运行时派生族 derive 调用点静态扫描 + 白名单核验（F03-③；
  AST 扫描先例 = ``tests/test_v2_common_shared_moves.py``）；
- B0' 冻结落盘（Q12-A/§13.3.6；P2-3：runs 批收益实体随冻结复制，
  manifest 相对路径全量可解析断言含幂等复用分支）与双跑指纹视图（复用
  ``t05b_evidence._finalize_comparison`` 四门判定，禁止复制语义）；
- 打分侧消费核对（R2-T5A-02 后半：臂③打分 store 实际读取 receipt
  ⊆ 消费集合断言，``assert_score_receipt_within_consumption``）。
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger

# 独立运行时仓库根不入 sys.path，按薄入口惯例显式引导
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.v2_p3.t05b_baselines import (  # noqa: E402
    _MANIFEST_FAMILY_LABEL,
    _acquire_baselines,
    _sha256_file_bytes,
    enumerate_consumption_files,
    verify_manifest_baseline,
)
from scripts.v2_p3.t05b_fingerprint import environment_identity, sha256_text  # noqa: E402
from scripts.v2_p3.t05b_isolated_root import (  # noqa: E402
    FRESH_DIRS,
    build_isolated_root,
    verify_consumption_snapshot_closed,
    verify_production_models_guard,
)
from scripts.v2_p3.t05b_spike_runner import (  # noqa: E402
    _assert_identity_compatible,
    collect_model_identity,
)

#: 三臂封闭集合（§19.1-A）
ARM_LEGACY_CS_TRAIN = "legacy_cs_train"
ARM_V2_CS_TRAIN_VERIFY = "v2_cs_train_verify"
ARM_V2_PANEL = "v2_panel"
ARMS = (ARM_LEGACY_CS_TRAIN, ARM_V2_CS_TRAIN_VERIFY, ARM_V2_PANEL)

#: T5 冻结参照根（沿 spike 同根）
T5_BASE_DIR = Path("data/frozen_reference/v2_p3")
#: B0' 冻结目录名（臂①全 14 折时追加冻结，Q12-A/§13.3.6）
B0_PRIME_DIR_NAME = "b0_prime"
#: WF 现役折数（§13.3.3；全折判定以此为口径）
FULL_SPLIT_COUNT = 14
#: 臂隔离根追加的全新目录（防任何默认解析写穿生产 ``data/runs``）
ARM_EXTRA_FRESH_DIRS = ("runs",)

#: 逐折产物五件套（R2-T5A-R1-05：核心三件套 + 预测行两套件；齐备才算完成；
#: 核心三件部分存在 = 中断/篡改痕迹 ⇒ 拒绝；预测行两件缺失 ⇒ 判不完整走显式
#: 重建（重打分），其字节 sha256 与指纹登记不一致 ⇒ 拒绝 = 证据篡改语义）
FOLD_FILES = (
    "model_version.txt",
    "boundary.json",
    "fingerprint.json",
    "predictions.parquet",
    "signal_topk.parquet",
)
#: 核心三件套（FOLD_FILES 前段；续跑「部分存在 ⇒ 拒绝」的原语义子集）
_FOLD_CORE_FILES = FOLD_FILES[:3]
#: 预测行两套件：文件名 → fingerprint.json 登记 sha256 键（续跑核验口径）
FOLD_PREDICTION_SHA_KEYS: Dict[str, str] = {
    "predictions.parquet": "predictions_sha256",
    "signal_topk.parquet": "signal_topk_sha256",
}

#: arm_meta.json 身份键（续跑时逐项相等核验，漂移 ⇒ 拒绝覆盖）
_ARM_META_IDENTITY_KEYS = (
    "arm",
    "producer_identity",
    "sleeve_identity",
    "input_snapshot_digest",
    "split_count",
    "final_date",
    "test_window_months",
    "selected_split_indices",
)


def fold_dir(arm_dir: Path, split_index: int) -> Path:
    """折产物目录（``folds/splitNN``，NN = 两位零填充折号）。"""
    return Path(arm_dir) / "folds" / f"split{split_index:02d}"


def fold_name(split_index: int) -> str:
    return f"split{split_index:02d}"


# ---------------------------------------------------------------------------
# 隔离根与基线（复用 t05b 语义；缓存探针换为臂布局）
# ---------------------------------------------------------------------------


def _single_run_layer_cache_exists(layer_dir: Path) -> bool:
    """单层臂布局「已有运行缓存」判定（逐折指纹 / 批指纹 / 链式指标 / runs 批次目录）。"""
    layer_dir = Path(layer_dir)
    if (layer_dir / "batch_fingerprint.json").exists() or (
        layer_dir / "chain_metrics.json"
    ).exists():
        return True
    if list(layer_dir.glob("folds/split*/fingerprint.json")):
        return True
    runs_dir = layer_dir / "runs"
    return runs_dir.is_dir() and any(runs_dir.iterdir())


def arm_run_cache_exists(run_dir: Path) -> bool:
    """臂布局「已有运行缓存」判定（_acquire_baselines 的 cache_probe）。

    **单跑/双跑两种布局并查**（R2-T5A-11：双跑布局下缓存在 ``run1/``、``run2/``
    子目录，外层探针漏查会导致「有缓存缺基线」时先重建快照而非拒绝——违反
    §14.10「先于任何快照写入」）：当前层 ∪ run1/run2 子层任一层有缓存即判有。
    """
    run_dir = Path(run_dir)
    layers = (run_dir, run_dir / "run1", run_dir / "run2")
    return any(_single_run_layer_cache_exists(layer) for layer in layers)


def acquire_arm_baselines(
    run_dir: Path,
    production_data_root: Path,
    skip_existing: bool,
    arm: str = ARM_LEGACY_CS_TRAIN,
) -> Dict[str, object]:
    """臂运行基线获取——``_acquire_baselines`` 原语义复用（R2-T05B-04/§14.10 不退化）。

    「有缓存但基线缺失 ⇒ 拒绝」独立于 ``skip_existing`` 且先于任何快照写入；
    缓存探针换为臂布局（``arm_run_cache_exists``，单/双跑并查，R2-T5A-11）。

    R2-T5A-02：``arm == v2_panel`` ⇒ 消费集合扩展到 panel 热/冷区 + labels 三表
    （字节门）并加采 manifest 族相关性基线；其余臂（默认）与 spike 现状逐位一致。

    R2-T5AE-02：所有 T5 臂追加 ``raw/moneyflow_hsgt`` 闭市日判定输入（字节门），
    防「改 raw 即改 excluded_days、守卫仍零漂移」的覆盖门敞口。
    """
    return _acquire_baselines(
        Path(run_dir),
        Path(production_data_root),
        skip_existing,
        cache_probe=arm_run_cache_exists,
        include_panel_labels=(arm == ARM_V2_PANEL),
        include_moneyflow_hsgt=True,
    )


def assert_fresh_dirs_real(root: Path, names: Sequence[str] = ()) -> None:
    """续跑复用隔离根时断言全新目录均为真实目录（非联接，防写穿生产）。"""
    import os

    for name in tuple(FRESH_DIRS) + tuple(names):
        candidate = Path(root) / name
        if not candidate.exists():
            continue
        is_junction = (
            bool(os.path.isjunction(str(candidate))) if hasattr(os.path, "isjunction") else False
        )
        if candidate.is_symlink() or is_junction:
            raise ValueError(f"隔离根全新目录禁止联接（续跑复核）: {candidate}")


def ensure_arm_root(
    arm_dir: Path,
    production_data_root: Path,
    *,
    link_fn: Optional[Callable[[Path, Path], None]] = None,
    verify_links: bool = True,
) -> Path:
    """臂隔离根获取：首跑新建（含 runs/ 全新目录）；续跑复用并复核只读联接与全新目录。

    ``link_fn`` / ``verify_links`` 透传 ``build_isolated_root``（测试注入替身避免真实
    mklink；续跑复用路径在 ``verify_links=False`` 时跳过只读联接断言）。
    """
    from scripts.v2_p3.t05b_isolated_root import assert_readonly_links

    arm_dir = Path(arm_dir)
    root = arm_dir / "root"
    if root.exists():
        if verify_links:
            assert_readonly_links(root, Path(production_data_root))
        assert_fresh_dirs_real(root, ARM_EXTRA_FRESH_DIRS)
        (root / "runs").mkdir(exist_ok=True)
        logger.info(f"续跑复用隔离根（只读联接与全新目录复核通过）: {root}")
        return root
    build_kwargs: Dict[str, object] = {"extra_fresh_dirs": ARM_EXTRA_FRESH_DIRS}
    if link_fn is not None:
        build_kwargs["link_fn"] = link_fn
        build_kwargs["verify_links"] = verify_links
    return build_isolated_root(  # type: ignore[arg-type]
        arm_dir, production_data_root, **build_kwargs
    )


def runs_batch_state(runs_dir: Path, selected_split_indices: Sequence[int]) -> str:
    """runs 批次目录状态：``empty`` / ``complete``；异常形态 ⇒ RuntimeError（禁止猜测）。

    - 多批次目录 ⇒ 拒绝（批次目录须唯一，防混批）；
    - 目录存在但缺 batch_meta/summary ⇒ 疑似写入中断，拒绝并指引人工清理；
    - 批次折集 != 本次选定折集 ⇒ 拒绝（配置漂移应已被生产者身份拦截，此处兜底）。
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir() or not any(runs_dir.iterdir()):
        return "empty"
    candidates = sorted(d for d in runs_dir.iterdir() if d.is_dir())
    if len(candidates) != 1:
        raise RuntimeError(
            f"runs/ 下批次目录不唯一，禁止混批: {[d.name for d in candidates]}（{runs_dir}）"
        )
    batch_dir = candidates[0]
    meta_path = batch_dir / "batch_meta.json"
    summary_path = batch_dir / "summary.parquet"
    if not (meta_path.exists() and summary_path.exists()):
        raise RuntimeError(
            f"runs 批次目录不完整（疑似写入中断，非有效证据）: {batch_dir}；"
            "请人工删除该目录后带 --skip-existing 续跑"
        )
    splits = {
        int(value)
        for value in pd.read_parquet(summary_path, columns=["split_index"])["split_index"]
    }
    expected = {int(split) for split in selected_split_indices}
    if splits != expected:
        raise RuntimeError(
            f"runs 批次折集 {sorted(splits)} 与本次选定 {sorted(expected)} 不一致，"
            "禁止混用；请换新 --run-name 重新取证"
        )
    return "complete"


# ---------------------------------------------------------------------------
# 生产者身份与 arm_meta
# ---------------------------------------------------------------------------


def build_arm_producer_identity(
    arm: str,
    config_digest: str,
    input_snapshot_digest: str,
    run_index: Optional[int] = None,
) -> Dict[str, object]:
    """臂生产者身份（沿 spike 口径：代码态 + 环境 + 配置摘要 + 输入基线摘要 + 臂名）。"""
    from src.lazybull.v2.store.store_identity import workspace_identity

    identity: Dict[str, object] = {
        "arm": arm,
        "source_digest": workspace_identity(_REPO_ROOT)["source_digest"],
        "environment_identity": environment_identity(),
        "config_digest": config_digest,
        "input_snapshot_digest": input_snapshot_digest,
    }
    if run_index is not None:
        identity["run_index"] = int(run_index)
    return identity


def arm_config_digest(payload: Mapping[str, object]) -> str:
    """臂配置摘要 = canonical JSON（键序无关）的 sha256。"""
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_arm_meta(arm_dir: Path, meta: Mapping[str, object]) -> Path:
    """arm_meta.json 落盘；已存在 ⇒ 身份键逐项相等核验（漂移拒绝覆盖，续跑幂等）。"""
    path = Path(arm_dir) / "arm_meta.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        mismatches = [key for key in _ARM_META_IDENTITY_KEYS if existing.get(key) != meta.get(key)]
        if mismatches:
            raise RuntimeError(
                f"arm_meta.json 已存在且身份键不一致，禁止覆盖（append-only）: {mismatches}"
            )
        logger.info(f"arm_meta 身份一致，沿用（续跑幂等）: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(meta), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"arm_meta 落盘: {path}")
    return path


# ---------------------------------------------------------------------------
# 逐折版本解析与折产物
# ---------------------------------------------------------------------------


def resolve_fold_model_version(root: Path, split_index: int) -> int:
    """逐折模型版本 = 隔离根台账 ``ml_train_runs.csv`` 按 ``split_index`` 过滤的末行版本。

    缺行 ⇒ ValueError（该折未成功注册，须先续训；禁止回退 latest 指针单值口径）。
    """
    csv_path = Path(root) / "models" / "stock_selection" / "ml_train_runs.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"隔离根内未发现训练运行台账: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    rows = df[df["split_index"].astype(str) == str(split_index)]
    if rows.empty:
        raise ValueError(f"台账缺 split {split_index} 记录（该折未成功注册，须先续训）: {csv_path}")
    return int(rows.iloc[-1]["model_version"])


def resolve_skip_start_model_version(root: Path, selected_split_indices: Sequence[int]) -> int:
    """skip-training 起始版本反解：``start = 各折版本 - split_index`` 必须全折唯一。

    编排器 skip 路径按 ``model_version = start_model_version + split_index`` 算术寻址
    （orchestrator.py:803）——逐折版本非等差 ⇒ 版本寻址不成立，拒绝并列出各折版本。
    """
    starts: Dict[int, int] = {}
    for split in selected_split_indices:
        version = resolve_fold_model_version(root, split)
        starts.setdefault(version - int(split), version)
    if len(starts) != 1:
        raise ValueError(
            "逐折模型版本不满足 start+split_index 等差寻址（skip-training 回放拒绝）: "
            f"start 候选 = {sorted(starts)}"
        )
    return next(iter(starts))


def load_fold_record_for_resume(
    arm_dir: Path, root: Path, split_index: int, producer_identity: Mapping[str, object]
) -> Optional[Dict[str, object]]:
    """折级续跑复用（先核验后复用，R2-T05B-04 口径沿用到折粒度；R2-T5A-R1-05 五件套）。

    续跑三态语义（逐折五件套口径）：

    - **全缺 ⇒ None**（待执行）；
    - **核心三件（model_version.txt/boundary.json/fingerprint.json）部分存在 ⇒
      RuntimeError**（中断/篡改痕迹，禁止静默重建）；
    - **预测行两套件（predictions.parquet/signal_topk.parquet）缺失 ⇒ None**
      （判不完整走显式重建 = 重打分，复现形态「删 predictions 后续跑仍 PASS」消除）；
    - **预测行文件字节 sha256 != 指纹登记的 predictions_sha256/signal_topk_sha256，
      或文件存在但指纹未登记 ⇒ ValueError**（证据篡改/不一致语义，禁止复用）；
    - 五件齐备且哈希一致 ⇒ 核验生产者身份 + 隔离模型字节身份后返回复用记录。
    """
    directory = fold_dir(arm_dir, split_index)
    existing = [name for name in FOLD_FILES if (directory / name).exists()]
    if not existing:
        return None
    core_existing = [name for name in _FOLD_CORE_FILES if (directory / name).exists()]
    if len(core_existing) != len(_FOLD_CORE_FILES):
        raise RuntimeError(
            f"折 {fold_name(split_index)} 产物部分存在（{existing}），疑似中断/篡改痕迹，"
            f"禁止静默重建；请清理 {directory} 或换新 --run-name"
        )
    fingerprint = json.loads((directory / "fingerprint.json").read_text(encoding="utf-8"))
    _assert_identity_compatible(fingerprint.get("producer_identity") or {}, dict(producer_identity))
    saved_model_identity = fingerprint.get("model_identity")
    if not saved_model_identity:
        raise ValueError(f"折 {fold_name(split_index)} 指纹缺模型身份记录，禁止复用")
    model_version = int((directory / "model_version.txt").read_text(encoding="utf-8").strip())
    if int(fingerprint.get("model_version", -1)) != model_version:
        raise ValueError(
            f"折 {fold_name(split_index)} model_version.txt 与指纹内版本不一致，禁止复用"
        )
    current_model_identity = collect_model_identity(Path(root), model_version)
    if current_model_identity != saved_model_identity:
        raise ValueError(f"折 {fold_name(split_index)} 隔离根模型字节与保存值不一致，禁止复用")
    # R2-T5A-R1-05：预测行两套件核验（身份/模型核验通过后；缺失 ⇒ 显式重建，漂移 ⇒ 拒绝）
    missing_predictions = []
    for name, sha_key in FOLD_PREDICTION_SHA_KEYS.items():
        candidate = directory / name
        registered = fingerprint.get(sha_key)
        if not candidate.exists():
            missing_predictions.append(name)
            continue
        if not registered:
            raise ValueError(
                f"折 {fold_name(split_index)} {name} 存在但指纹未登记 {sha_key}，"
                "证据不一致，禁止复用（R2-T5A-R1-05）"
            )
        actual = _sha256_file_bytes(candidate)
        if actual != registered:
            raise ValueError(
                f"折 {fold_name(split_index)} {name} 字节哈希与指纹登记不一致"
                f"（{actual[:16]}… != {str(registered)[:16]}…），证据篡改语义，"
                "禁止复用（R2-T5A-R1-05）"
            )
    if missing_predictions:
        logger.warning(
            f"折 {fold_name(split_index)} 预测行产物缺失 {missing_predictions}，"
            "判不完整走显式重建（重打分，R2-T5A-R1-05）"
        )
        return None
    boundary = json.loads((directory / "boundary.json").read_text(encoding="utf-8"))
    logger.info(f"折 {fold_name(split_index)} 产物复用（生产者 + 模型字节 + 预测行哈希核验通过）")
    return {
        "split_index": int(split_index),
        "model_version": model_version,
        "boundary": boundary,
        "fingerprint": fingerprint,
        "reused": True,
    }


def write_fold_artifacts(
    arm_dir: Path,
    split_index: int,
    model_version: int,
    boundary: Mapping[str, object],
    fingerprint: Mapping[str, object],
) -> Path:
    """逐折核心三件套落盘（顺序 = model_version → boundary → fingerprint，末件为完成标记）。

    预测行两套件（predictions.parquet/signal_topk.parquet）由打分工具（
    ``score_model_fingerprint`` 持久化段）先于本函数落盘，五件套齐备口径见
    ``FOLD_FILES`` / ``load_fold_record_for_resume``（R2-T5A-R1-05）。
    """
    directory = fold_dir(arm_dir, split_index)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model_version.txt").write_text(f"{int(model_version)}\n", encoding="utf-8")
    (directory / "boundary.json").write_text(
        json.dumps(dict(boundary), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (directory / "fingerprint.json").write_text(
        json.dumps(dict(fingerprint), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return directory


# ---------------------------------------------------------------------------
# 全批汇总指纹与链式指标
# ---------------------------------------------------------------------------


def build_batch_fingerprint_doc(fold_fingerprints: Mapping[int, str]) -> Dict[str, object]:
    """全批汇总指纹（§13.3.3）：逐折指纹按折号升序 LF 拼接（含结尾 LF）的 sha256。

    逐折值 = ``folds/splitNN/fingerprint.json`` 的 ``fingerprint``（sha256 hex）；
    汇总拼接口径与 S1 序列化同族（LF 连接、UTF-8、sha256）。
    """
    if not fold_fingerprints:
        raise ValueError("逐折指纹为空，禁止产出空批次汇总指纹（R2-T05B-03 同口径）")
    ordered = sorted(int(split) for split in fold_fingerprints)
    folds = {fold_name(split): str(fold_fingerprints[split]) for split in ordered}
    joined = "\n".join(folds[name] for name in ordered_names(folds)) + "\n"
    return {
        "spec": (
            "全批汇总指纹 = 逐折指纹（sha256 hex 文本）按折号升序以 LF 拼接"
            "（含结尾 LF）的 sha256（§13.3.3 逐折 + 全批汇总规格）"
        ),
        "folds": folds,
        "batch_fingerprint": sha256_text(joined),
    }


def ordered_names(folds: Mapping[str, object]) -> List[str]:
    """折名按折号升序（splitNN 零填充 ⇒ 字典序即数值序，显式排序防误用）。"""
    return sorted(folds, key=lambda name: int(name.replace("split", "")))


def collect_and_write_batch_fingerprint(arm_dir: Path) -> Dict[str, object]:
    """从落盘逐折指纹汇总全批指纹并写 ``batch_fingerprint.json``（含内部自洽口径）。"""
    arm_dir = Path(arm_dir)
    fold_fps: Dict[int, str] = {}
    for directory in sorted((arm_dir / "folds").glob("split*")):
        fingerprint_path = directory / "fingerprint.json"
        if not fingerprint_path.exists():
            continue
        doc = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        fold_fps[int(directory.name.replace("split", ""))] = str(doc["fingerprint"])
    batch_doc = build_batch_fingerprint_doc(fold_fps)
    (arm_dir / "batch_fingerprint.json").write_text(
        json.dumps(batch_doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        f"全批汇总指纹落盘: {arm_dir / 'batch_fingerprint.json'}"
        f"（{len(fold_fps)} 折，{batch_doc['batch_fingerprint'][:16]}…）"
    )
    return batch_doc


def compute_and_write_chain_metrics(arm_dir: Path) -> Dict[str, object]:
    """链式指标 = 本臂 runs 批 chain_nav 经 ``v2/common/chain_metrics`` 唯一实现计算。

    批次目录解析复用 ``t5_compare_gates._resolve_runs_batch_dir``（runs/ 下须唯一）；
    禁止自建年化/回撤公式。
    """
    from scripts.v2_p3.t5_compare_gates import _resolve_runs_batch_dir
    from src.lazybull.v2.common.chain_metrics import calculate_chain_metrics

    arm_dir = Path(arm_dir)
    batch_dir = _resolve_runs_batch_dir(arm_dir)
    chain_nav = pd.read_parquet(batch_dir / "chain_nav.parquet")
    metrics = calculate_chain_metrics(chain_nav)
    doc: Dict[str, object] = {
        **metrics,
        "source_batch": batch_dir.name,
        "implementation": "src.lazybull.v2.common.chain_metrics.calculate_chain_metrics",
    }
    (arm_dir / "chain_metrics.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    logger.info(f"链式指标落盘: {arm_dir / 'chain_metrics.json'}（来源批次 {batch_dir.name}）")
    return doc


# ---------------------------------------------------------------------------
# 运行后守卫复核（漂移清单落盘；漂移 ⇒ 调用方判 FAIL）
# ---------------------------------------------------------------------------


def verify_after_guards(arm_dir: Path, baselines: Mapping[str, object]) -> Dict[str, List[str]]:
    """运行后复核：生产模型守卫 + 消费集合闭合核验（漂移清单落盘）。

    R2-T5A-02：基线含 manifest 基线（臂③）时追加 manifest 相关性 diff 复核，
    漂移清单落 ``manifest_verify.json`` 并随 ``manifest_drift`` 键返回
    （消费集合唯一硬门的相关性语义段；verdict 接线由 runner 侧完成）。

    R2-T5AE-R1-02：消费集合核验升级为**闭合核验**——逐文件内容身份之外，
    运行后重枚举消费集合（arm 感知口径与基线采集同源）检测「快照后新增实读
    候选」；``missing`` 候选态记录同门核验。
    """
    arm_dir = Path(arm_dir)
    guard_before = baselines["guard_before"]
    guard_drift = verify_production_models_guard(guard_before)  # type: ignore[arg-type]
    # panel 扩展口径 = manifest 基线存在性（include_panel_labels 的直接产物；
    # 双跑布局 run_dir 顶层无 arm_meta.json，改由基线承载，单/双跑一致）
    include_panel_labels = bool(baselines.get("manifest_baseline_path"))
    rescanned_by_dir = enumerate_consumption_files(
        Path(str(baselines["production_data_root"])),  # type: ignore[arg-type]
        include_panel_labels=include_panel_labels,
        include_moneyflow_hsgt=True,
    )
    rescan_files = [
        path
        for label, files in rescanned_by_dir.items()
        if label != _MANIFEST_FAMILY_LABEL
        for path in files
    ]
    consumption_drift = verify_consumption_snapshot_closed(
        baselines["consumption_snapshot_path"],  # type: ignore[arg-type]
        base_dir=baselines["base_dir"],  # type: ignore[arg-type]
        rescan_paths=rescan_files,
    )
    (arm_dir / "after_guard.json").write_text(
        json.dumps({"drift": guard_drift}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (arm_dir / "consumption_verify.json").write_text(
        json.dumps({"drift": consumption_drift}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest_drift: List[str] = []
    if baselines.get("manifest_baseline_path"):
        manifest_drift = verify_manifest_baseline(
            baselines["manifest_baseline_path"],  # type: ignore[arg-type]
            baselines["production_data_root"],  # type: ignore[arg-type]
        )
        (arm_dir / "manifest_verify.json").write_text(
            json.dumps({"drift": manifest_drift}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if guard_drift:
        logger.error(f"生产模型目录漂移（§7.4-3，停止并回滚）: {guard_drift[:5]}")
    if consumption_drift:
        logger.error(f"消费集合漂移（§13.3.4 唯一硬门）: {consumption_drift[:5]}")
    if manifest_drift:
        logger.error(f"manifest 族相关性漂移（§13.3.4 唯一硬门扩展段）: {manifest_drift[:5]}")
    return {
        "guard_drift": guard_drift,
        "consumption_drift": consumption_drift,
        "manifest_drift": manifest_drift,
    }


# ---------------------------------------------------------------------------
# 打分侧消费核对（R2-T5A-02 后半：臂③打分 store 实际读取 ⊆ 消费集合）
# ---------------------------------------------------------------------------


def collect_score_read_receipt_files(score_store: object) -> List[str]:
    """打分 store 会话级实际读取文件清单（显示路径去重排序；无读取 ⇒ 空列表）。

    数据源 = ``PanelDataStore.read_receipts``（T3 会话级累积，覆盖逐折多次读取；
    单槽 ``last_read_receipt`` 只保留末次读取，不满足全折覆盖口径）。清单单一
    来源 = 读取动作本身（StoreReadReceipt.files），不另造冷热区解析规则。
    """
    receipts = getattr(score_store, "read_receipts", None) or ()
    paths = {str(record["path"]) for receipt in receipts for record in receipt.files}
    return sorted(paths)


def _resolve_display_path(display: str, base: Path) -> str:
    """显示路径解析为绝对路径规范形（receipt 相对 store root / 快照相对 base_dir）。"""
    raw = Path(display)
    resolved = raw if raw.is_absolute() else (Path(base) / raw)
    return resolved.resolve().as_posix()


def assert_score_receipt_within_consumption(
    score_store: object,
    consumption_snapshot_path: Path,
    *,
    base_dir: Path,
    production_data_root: Path,
) -> Dict[str, object]:
    """臂③打分实际读取 ⊆ 消费集合断言（R2-T5A-02 后半：唯一硬门覆盖打分侧真实消费）。

    receipt 显示路径按 store root（= 生产数据根）解析、消费集合清单按
    ``base_dir`` 解析，两侧以解析后绝对路径集合比对；任一越界 ⇒ RuntimeError
    （列出越界文件，禁止静默放行）。无读取 receipt（全折复用未打分）⇒ 空通过
    并在报告登记。返回复核报告（调用方可并入证据/复核清单）。
    """
    records = json.loads(Path(consumption_snapshot_path).read_text(encoding="utf-8"))
    consumption = {_resolve_display_path(str(record["path"]), Path(base_dir)) for record in records}
    receipt_files = collect_score_read_receipt_files(score_store)
    root = Path(production_data_root)
    outside = [
        display
        for display in receipt_files
        if _resolve_display_path(display, root) not in consumption
    ]
    report: Dict[str, object] = {
        "check": "score_receipt_within_consumption（R2-T5A-02 打分侧消费核对）",
        "receipt_count": len(getattr(score_store, "read_receipts", None) or ()),
        "receipt_files": len(receipt_files),
        "consumption_files": len(consumption),
        "outside_files": outside,
        "pass": not outside,
    }
    if not receipt_files:
        report["note"] = "打分 store 无读取 receipt（全折复用/未打分），集合核对空通过"
    if outside:
        raise RuntimeError(
            f"打分侧实际读取越出消费集合（§13.3.4 唯一硬门，R2-T5A-02）: {outside[:10]}"
            f"{'...' if len(outside) > 10 else ''}（共 {len(outside)} 个）"
        )
    logger.info(
        f"打分侧消费核对通过: {len(receipt_files)} 个实际读取文件 ⊆ 消费集合"
        f"（{len(consumption)} 个）"
    )
    return report


# ---------------------------------------------------------------------------
# 承接④：运行时派生族 derive 调用点静态扫描与白名单核验（F03-③）
# ---------------------------------------------------------------------------

#: 运行时派生族 → factors 包 derive 函数名（五族全列——AGENTS.md「四族」口径之外
#: 的 top_inst 经同一 DI 通道派生，漂移检出必须同覆盖；函数名唯一来源 = factors 各模块）
RUNTIME_DERIVE_FAMILY_FUNCS: Dict[str, Tuple[str, ...]] = {
    "holdertrade": ("derive_holdertrade_columns",),
    "repurchase": ("derive_repurchase_columns",),
    "top10_floatholders": ("derive_top10fh_columns",),
    "top_inst": ("derive_top_inst_columns",),
    "availability": ("derive_availability_markers", "ensure_availability_markers"),
}
_ALL_FAMILY_FUNC_NAMES = {name for names in RUNTIME_DERIVE_FAMILY_FUNCS.values() for name in names}
_FUNC_TO_FAMILY = {
    name: family for family, names in RUNTIME_DERIVE_FAMILY_FUNCS.items() for name in names
}

#: 已登记 import 引用点白名单（file|funcname；T3 守卫已落位置，漂移 ⇒ 拒绝）
DERIVE_IMPORT_SITE_WHITELIST: Tuple[str, ...] = (
    "src/lazybull/v2/core/signal/ml_signal.py|ensure_availability_markers",
    "src/lazybull/v2/hosts/backtest/replay.py|derive_holdertrade_columns",
    "src/lazybull/v2/hosts/backtest/replay.py|derive_repurchase_columns",
    "src/lazybull/v2/hosts/backtest/replay.py|derive_top10fh_columns",
    "src/lazybull/v2/hosts/backtest/replay.py|derive_top_inst_columns",
    "src/lazybull/v2/store/panel_builder.py|derive_availability_markers",
)

#: 已登记调用点白名单（file|symbol 多重集合；DI 通道调用点全列）
DERIVE_CALL_SITE_WHITELIST: Tuple[str, ...] = (
    # 信号侧 availability（T3 replay 防双写关联位置）
    "src/lazybull/v2/core/signal/ml_signal.py|ensure_availability_markers",
    # 回测侧 cs_train 验收路径四族共享内层调用（_derive_feature_family 形参）
    "src/lazybull/v2/hosts/backtest/replay.py|derive_fn",
    # P1 物化期 availability（构建期物化，非运行时派生；同一实现登记在案）
    "src/lazybull/v2/store/panel_builder.py|derive_availability_markers",
    # 训练侧 availability（RuntimeDerivers.availability DI 通道）
    "src/lazybull/v2/train/core/prepare_families.py|derivers.availability",
    # 训练侧四族 derive_fn DI 通道（holdertrade/repurchase/top10fh/top_inst 四处段）
    "src/lazybull/v2/train/core/prepare_families.py|deriver.derive_fn",
    "src/lazybull/v2/train/core/prepare_families.py|deriver.derive_fn",
    "src/lazybull/v2/train/core/prepare_families.py|deriver.derive_fn",
    "src/lazybull/v2/train/core/prepare_families.py|deriver.derive_fn",
    # 评估侧 availability_ensurer + 四族共享循环（split_training._derive_eval_families）
    "src/lazybull/v2/train/walk_forward/split_training.py|availability_ensurer",
    "src/lazybull/v2/train/walk_forward/split_training.py|deriver.derive_fn",
)

#: DI 通道调用形态（Name 调用 = 参数化注入函数；Attribute 调用 = 容器/deriver 成员）
_DI_NAME_CALLS = frozenset({"derive_fn", "availability_ensurer"})
_DI_ATTR_CALLS = frozenset({"derive_fn", "availability"})


def _call_symbol(func: ast.expr) -> Optional[str]:
    """调用符号归一：Name → 裸名；``<name>.<attr>`` → 点式；复杂受体 → ``*.<attr>``。"""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name):
            return f"{func.value.id}.{func.attr}"
        return f"*.{func.attr}"
    return None


def _scan_one_file(py_file: Path, rel: str) -> Tuple[List[str], List[str], List[Dict[str, object]]]:
    """单文件扫描：factors 派生函数 import 绑定 + 直接/DI 调用点。"""
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    bindings: Dict[str, str] = {}  # 本地名 → factors 函数名
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "src.lazybull.factors"
        ):
            for alias in node.names:
                if alias.name in _ALL_FAMILY_FUNC_NAMES:
                    bindings[alias.asname or alias.name] = alias.name
    import_sites = [f"{rel}|{func_name}" for func_name in sorted(bindings.values())]
    call_sites: List[str] = []
    details: List[Dict[str, object]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        symbol = _call_symbol(node.func)
        if symbol is None:
            continue
        family: Optional[str] = None
        if isinstance(node.func, ast.Name) and node.func.id in bindings:
            family = _FUNC_TO_FAMILY[bindings[node.func.id]]
            symbol = bindings[node.func.id]  # 归一为 factors 函数名（别名无关化）
        elif isinstance(node.func, ast.Attribute) and node.func.attr in _ALL_FAMILY_FUNC_NAMES:
            family = _FUNC_TO_FAMILY[node.func.attr]
            symbol = node.func.attr
        elif symbol in _DI_NAME_CALLS or symbol.rsplit(".", 1)[-1] in _DI_ATTR_CALLS:
            family = "di_channel"
        if family is None:
            continue
        call_sites.append(f"{rel}|{symbol}")
        details.append({"file": rel, "line": node.lineno, "symbol": symbol, "family": family})
    return import_sites, call_sites, details


def scan_runtime_derive_call_sites(v2_root: Optional[Path] = None) -> Dict[str, object]:
    """静态扫描 ``src/lazybull/v2/**`` 运行时派生族 derive 全调用点（承接④）。

    两维度：``import_sites``（factors 五族 derive 函数 import 引用点）与
    ``call_sites``（直接调用 + DI 通道调用 ``*.derive_fn`` / ``derivers.availability`` /
    ``availability_ensurer`` / 内层 ``derive_fn``）。路径为仓库相对（v2_root 在
    仓库外时退化为相对 v2_root——仅供扫描器自验测试）。
    """
    root = Path(v2_root) if v2_root is not None else _REPO_ROOT / "src" / "lazybull" / "v2"
    import_sites: List[str] = []
    call_sites: List[str] = []
    details: List[Dict[str, object]] = []
    for py_file in sorted(root.rglob("*.py")):
        try:
            rel = py_file.relative_to(_REPO_ROOT).as_posix()
        except ValueError:
            rel = py_file.relative_to(root).as_posix()
        file_imports, file_calls, file_details = _scan_one_file(py_file, rel)
        import_sites += file_imports
        call_sites += file_calls
        details += file_details
    return {
        "import_sites": sorted(import_sites),
        "call_sites": sorted(call_sites),
        "details": sorted(details, key=lambda item: (str(item["file"]), int(item["line"]))),
    }


def assert_derive_call_sites_registered(scan: Mapping[str, object]) -> None:
    """白名单核验（多重集合逐位相等）：任一新增/消失调用点或 import 引用点 ⇒ ValueError。"""
    problems: List[str] = []
    for key, whitelist in (
        ("import_sites", DERIVE_IMPORT_SITE_WHITELIST),
        ("call_sites", DERIVE_CALL_SITE_WHITELIST),
    ):
        actual = sorted(str(site) for site in scan[key])  # type: ignore[index]
        expected = sorted(whitelist)
        if actual == expected:
            continue
        added = sorted(set(actual) - set(expected))
        removed = sorted(set(expected) - set(actual))
        count_diff = len(actual) - len(expected)
        problems.append(f"{key} 漂移: 新增={added} 消失={removed} 计数差={count_diff}")
    if problems:
        raise ValueError(
            "运行时派生族 derive 调用点漂移（承接④ F03-③ 核验失败——"
            "新增调用点须先登记白名单并过评审）: " + "；".join(problems)
        )


def collect_derive_call_site_report(v2_root: Optional[Path] = None) -> Dict[str, object]:
    """承接④证据段：扫描 + 白名单核验（漂移即失败）+ 调用点清单返回。"""
    scan = scan_runtime_derive_call_sites(v2_root)
    assert_derive_call_sites_registered(scan)
    return {
        "whitelist_match": True,
        "import_sites": scan["import_sites"],
        "call_sites": scan["call_sites"],
        "details": scan["details"],
        "guard_note": (
            "T3 运行时守卫逐项引用：训练侧适配层校验（data_adapter._validate_runtime_families）"
            " / 评估侧 DI 守卫（split_training._derive_eval_families 缺注入即报错）"
            " / 信号侧 replay 防双写（replay.py panel 模式 lookup 拒绝 + ml_signal "
            "availability 单点）"
        ),
    }


# ---------------------------------------------------------------------------
# B0' 冻结（Q12-A/§13.3.6：臂①全 14 折时追加）
# ---------------------------------------------------------------------------


def _assert_b0_prime_manifest_paths(target: Path, manifest: Mapping[str, object]) -> None:
    """manifest 所列全部相对路径在 ``b0_prime/`` 下必须可解析（P2-3 复核断言）。

    覆盖逐折 fingerprint/boundary/daily_nav 与 runs 批 chain_nav；幂等复用分支
    同样先过本断言（证据残缺 ⇒ 拒绝复用，防「清单指向空位」的伪冻结）。
    """
    target = Path(target)
    missing: List[str] = []
    folds = manifest.get("folds", {})
    for name, entry in sorted(folds.items()):  # type: ignore[union-attr]
        for key in ("fingerprint_path", "boundary_path", "daily_nav_path"):
            rel = str((entry or {}).get(key, ""))  # type: ignore[union-attr]
            if not rel or not (target / rel).is_file():
                missing.append(f"folds.{name}.{key}={rel or '<缺失键>'}")
    runs_batch = manifest.get("runs_batch", {})
    chain_nav_rel = str((runs_batch or {}).get("chain_nav_path", ""))  # type: ignore[union-attr]
    if not chain_nav_rel or not (target / chain_nav_rel).is_file():
        missing.append(f"runs_batch.chain_nav_path={chain_nav_rel or '<缺失键>'}")
    if missing:
        raise RuntimeError(
            f"B0' 冻结产物不完整（manifest 所列路径不可解析）: {missing}（{target}）；"
            "请人工清理 b0_prime/ 后重跑冻结并经评审登记"
        )


def freeze_b0_prime(
    arm_dir: Path,
    *,
    run_name: str,
    producer_identity: Mapping[str, object],
    config_digest: str,
) -> Path:
    """臂①产物冻结为 B0'（``data/frozen_reference/v2_p3/b0_prime/``）。

    内容 = chain_metrics.json + batch_fingerprint.json + 逐折 fingerprint/boundary 副本 +
    **runs 批收益实体副本**（``runs/<批次>/chain_nav.parquet`` + ``summary.parquet`` +
    逐折 ``folds/<名>/daily.parquet``——P2-3：manifest 收益路径相对 ``b0_prime/``
    必须可解析）+ ``freeze_manifest.json``（配置指纹 + 逐折收益路径指针 + 链式指标 +
    数据态/代码态身份 + 来源 run_name）。已冻结 ⇒ 来源 run_name 与批指纹逐项一致
    **且 manifest 所列路径全部可解析**才幂等复用，否则拒绝覆盖（append-only）。
    """
    arm_dir = Path(arm_dir)
    target = arm_dir.parent / B0_PRIME_DIR_NAME
    manifest_path = target / "freeze_manifest.json"
    batch_doc = json.loads((arm_dir / "batch_fingerprint.json").read_text(encoding="utf-8"))
    chain_doc = json.loads((arm_dir / "chain_metrics.json").read_text(encoding="utf-8"))
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("source_run_name") == run_name
            and existing.get("batch_fingerprint") == batch_doc["batch_fingerprint"]
        ):
            _assert_b0_prime_manifest_paths(target, existing)
            logger.info(f"B0' 已冻结且身份一致，幂等复用: {target}")
            return target
        raise RuntimeError(
            f"B0' 已冻结且身份不匹配（{existing.get('source_run_name')} vs {run_name}），"
            "禁止覆盖（append-only）；如需重冻请先人工清理并经评审登记"
        )

    from scripts.v2_p3.t5_compare_gates import _resolve_runs_batch_dir

    batch_dir = _resolve_runs_batch_dir(arm_dir)
    batch_meta = json.loads((batch_dir / "batch_meta.json").read_text(encoding="utf-8"))
    (target / "folds").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(arm_dir / "chain_metrics.json", target / "chain_metrics.json")
    shutil.copyfile(arm_dir / "batch_fingerprint.json", target / "batch_fingerprint.json")
    # runs 批收益实体复制（P2-3：manifest 相对路径在 b0_prime/ 下可解析）
    runs_target = target / "runs" / batch_dir.name
    runs_target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(batch_dir / "chain_nav.parquet", runs_target / "chain_nav.parquet")
    shutil.copyfile(batch_dir / "summary.parquet", runs_target / "summary.parquet")

    folds_manifest: Dict[str, object] = {}
    for name in ordered_names(batch_doc["folds"]):
        source_fold = fold_dir(arm_dir, int(name.replace("split", "")))
        target_fold = target / "folds" / name
        target_fold.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_fold / "fingerprint.json", target_fold / "fingerprint.json")
        shutil.copyfile(source_fold / "boundary.json", target_fold / "boundary.json")
        runs_fold = runs_target / "folds" / name
        runs_fold.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(batch_dir / "folds" / name / "daily.parquet", runs_fold / "daily.parquet")
        model_version = int((source_fold / "model_version.txt").read_text(encoding="utf-8"))
        folds_manifest[name] = {
            "model_version": model_version,
            "fingerprint_path": f"folds/{name}/fingerprint.json",
            "boundary_path": f"folds/{name}/boundary.json",
            "daily_nav_path": f"runs/{batch_dir.name}/folds/{name}/daily.parquet",
        }
    manifest = {
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "source_run_name": run_name,
        "arm": ARM_LEGACY_CS_TRAIN,
        "config_digest": config_digest,
        "producer_identity": dict(producer_identity),
        "batch_fingerprint": batch_doc["batch_fingerprint"],
        "chain_metrics": chain_doc,
        "folds": folds_manifest,
        "runs_batch": {
            "batch_id": batch_dir.name,
            "chain_nav_path": f"runs/{batch_dir.name}/chain_nav.parquet",
            "source_arm_dir": str(arm_dir),
        },
        "data_state_id": (batch_meta.get("data_state") or {}).get("data_state_id"),
        "code_state": batch_meta.get("code_state"),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    _assert_b0_prime_manifest_paths(target, manifest)
    logger.info(f"B0' 冻结完成: {target}（来源 {run_name}，批次 {batch_dir.name}）")
    return target


# ---------------------------------------------------------------------------
# 双跑比对（阶段 B：同码双跑四门，复用 t05b_evidence._finalize_comparison）
# ---------------------------------------------------------------------------


def build_batch_fingerprint_view(arm_dir: Path) -> Dict[str, object]:
    """臂级产物 → spike 指纹视图（复用 ``_finalize_comparison`` 四门判定的输入形态）。

    映射：集成指纹 = 全批汇总指纹；逐种子诊断 = 逐折逐种子拍平（标签 ``splitNN:seed``）；
    覆盖 = 逐折打分日拍平（``splitNN:YYYYMMDD``）；行数 = 逐折行数求和。
    逐折文件与批汇总 ``folds`` 映射的内部自洽先核验（输入身份校验）。
    """
    arm_dir = Path(arm_dir)
    batch_doc = json.loads((arm_dir / "batch_fingerprint.json").read_text(encoding="utf-8"))
    folds_map = batch_doc["folds"]
    per_seed: List[Dict[str, object]] = []
    score_days: List[str] = []
    excluded_days: List[str] = []
    n_rows = 0
    for name in ordered_names(folds_map):
        fingerprint_path = arm_dir / "folds" / name / "fingerprint.json"
        doc = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        if str(doc["fingerprint"]) != str(folds_map[name]):
            raise ValueError(
                f"批次汇总 folds.{name} 与逐折指纹文件不自洽: "
                f"汇总={folds_map[name]} vs 逐折={doc['fingerprint']}"
            )
        for item in doc.get("per_seed_diagnostic") or []:
            per_seed.append(
                {"seed": f"{name}:{item['seed']}", "fingerprint": str(item["fingerprint"])}
            )
        score_days.extend(f"{name}:{day}" for day in doc.get("score_days") or [])
        excluded_days.extend(f"{name}:{day}" for day in doc.get("excluded_days") or [])
        n_rows += int(doc.get("n_rows") or 0)
    return {
        "fingerprint": str(batch_doc["batch_fingerprint"]),
        "per_seed_diagnostic": per_seed,
        "score_days": score_days,
        "excluded_days": excluded_days,
        "n_rows": n_rows,
    }


def compare_double_run(
    run1_dir: Path,
    run2_dir: Path,
    guard_drift: Sequence[str],
    consumption_drift: Sequence[str],
) -> Dict[str, object]:
    """双跑四门比对：集成指纹（= 全批汇总指纹）逐位 / 逐种子诊断 / 覆盖日集 / 守卫+消费集合。

    判定核复用 ``t05b_evidence._finalize_comparison``（import 复用，禁止复制语义）；
    键名 ``ensemble_fingerprint_run{1,2}`` 在本口径下 = 两臂全批汇总指纹。
    """
    from scripts.v2_p3.t05b_evidence import _finalize_comparison

    views = [build_batch_fingerprint_view(Path(d)) for d in (run1_dir, run2_dir)]
    comparison = _finalize_comparison(
        [{"fingerprint": views[0]}, {"fingerprint": views[1]}],
        list(guard_drift),
        list(consumption_drift),
    )
    comparison["run1_dir"] = str(run1_dir)
    comparison["run2_dir"] = str(run2_dir)
    comparison["gate_note"] = (
        "ensemble_fingerprint_* = 各 run 全批汇总指纹（逐折逐位由汇总指纹覆盖）；"
        "per_seed = 逐折逐种子拍平比对；coverage = 逐折打分日拍平比对"
    )
    return comparison
