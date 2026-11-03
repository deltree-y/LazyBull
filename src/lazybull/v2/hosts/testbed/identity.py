# -*- coding: utf-8 -*-
"""P5b 试车场身份采集（input / config / model / code / window-state）。

复用 P4 既有能力，不新造口径：

- 配置解析与消费白名单：``hosts/paper/config_adapter``
  （``PAPER_CONFIG_TO_V2`` / ``resolve_engine`` / ``resolve_model_version``）；
- 配置指纹：``v2/evidence/fingerprint_keys.fingerprint``（口径冻结的 16 位短指纹）；
- 文件/代码/环境身份：``v2/store/store_identity``
  （``canonical_json_sha256`` / ``file_identity`` / ``workspace_identity``）；
- 模型产物指纹载荷与 P4 门 1 同形（``{suffix: {bytes, sha256}}``），跨阶段可比。

评审处置（R1-T1-03/08、R2-T1-11/12/13/14/15/16）：

- 数据态拆两档——``window_state_id``（窗口内容，fail-closed 漂移门）与
  ``global_watermark``（全库水位，仅登记，不参与门禁，避免生产日更误报）；
- 窗口身份覆盖实读数据集：``clean/daily`` 强制、``features/cs_infer`` 存在即纳入；
- 代码态拆两档——``execution_digest``（试车场 + 复用件的执行闭包，漂移门）与
  ``workspace``（全仓 git/源码摘要，仅登记）；
- 模型身份登记 ``model_type`` / ``train_end_date`` 并断言训练截止严格早于 D0。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from src.lazybull.v2.evidence.fingerprint_keys import fingerprint
from src.lazybull.v2.hosts.paper.config_adapter import (
    PAPER_CONFIG_TO_V2,
    resolve_engine,
    resolve_model_version,
)
from src.lazybull.v2.hosts.testbed.clock import normalize_day
from src.lazybull.v2.hosts.testbed.paths import TestbedError
from src.lazybull.v2.store.store_identity import (
    canonical_json_sha256,
    collect_file_identities,
    environment_identity,
    file_identity,
    workspace_identity,
)

__all__ = [
    "EXECUTION_CLOSURE",
    "MODEL_ARTIFACT_SUFFIXES",
    "collect_code_identity",
    "collect_config_identity",
    "collect_data_state_identity",
    "collect_input_identity",
    "collect_model_identity",
    "identity_summary",
]

#: 模型产物三件套（与 P4 门 1 身份同后缀）
MODEL_ARTIFACT_SUFFIXES = ("model.joblib", "metadata.json", "features.json")
#: 执行闭包（漂移门作用面）：试车场自身 + 被复用的纸面宿主/身份件 + 状态库语义依赖
EXECUTION_CLOSURE = (
    "src/lazybull/v2/hosts/testbed",
    "src/lazybull/v2/hosts/paper",
    "src/lazybull/v2/core/accounting/engine_state.py",
    "src/lazybull/v2/core/execution/live_schedule.py",
    "src/lazybull/v2/store/store_identity.py",
    "src/lazybull/v2/evidence/fingerprint_keys.py",
)
#: 窗口内必须逐日物化的清洗数据集（行情/估值最小面，fail-closed）
_REQUIRED_WINDOW_DATASETS = ("clean/daily",)
#: 窗口内存在即纳入身份的数据集（缺失不阻断，但存在即受漂移门约束）
_OPTIONAL_WINDOW_DATASETS = ("features/cs_infer",)
#: v2 PaperHost 实际消费的 panel 分区（``features/panel/<day>/core.parquet``）；
#: 存在即必须纳入窗口身份——否则合法重写 panel 后仍可沿用旧 campaign 身份发布
_PANEL_DATASET = "features/panel"
#: 冷区 panel 归档（store 可能优先读取；月份目录布局）
_PANEL_ARCHIVE = "features/panel_archive"
#: 全库水位登记的数据集（仅登记，不参与门禁）
_WATERMARK_DATASETS = ("raw/daily", "clean/daily", "features/cs_infer")


def _jsonable(value: Any) -> Any:
    """递归把配置值转成可 canonical 序列化的形态（未知类型按 str 登记）。"""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def collect_config_identity(flat: Mapping[str, Any]) -> Dict[str, Any]:
    """有效执行配置身份（P4 白名单口径 + 冻结指纹 + 引擎解析结果）。

    ``exposure_policy`` / ``exposure_replenish`` 属已退役家族（Q5 穿透不消费），此处
    仍并入身份：它们不改变执行行为，但可以暴露"同一 campaign 内有人改了政策键"的
    意图差异，故按身份登记而非按消费白名单剔除（登记差异，非漏闸）。
    """
    execution = {key: flat[key] for key in PAPER_CONFIG_TO_V2 if key in flat}
    execution["engine"] = resolve_engine(flat)
    execution["exposure_policy"] = flat.get("exposure_policy")
    execution["exposure_replenish"] = flat.get("exposure_replenish")
    execution = _jsonable(execution)
    return {
        "execution": execution,
        "fingerprint": fingerprint(execution),
        "config_sha256": canonical_json_sha256(_jsonable(dict(flat))),
    }


def collect_model_identity(
    flat: Mapping[str, Any], *, models_dir: str, d0_date: str
) -> Dict[str, Any]:
    """模型身份：显式版本 + 三件套内容哈希 + 训练截止证据。

    试车场臂**必须显式冻结** ``model_version``（禁止 registry latest）；模型
    ``metadata.json`` 必须给出 ``train_end_date`` 且严格早于 D0（设计 §4.1 尾段）。
    """
    raw_version = flat.get("model_version")
    if raw_version is None or str(raw_version).strip() == "":
        raise TestbedError(
            "试车场臂必须显式冻结 model_version（禁止 registry latest 语义）："
            f"实际 {raw_version!r}"
        )
    version = resolve_model_version(raw_version, models_dir)
    root = Path(models_dir)
    artifacts: Dict[str, Any] = {}
    files: list[Dict[str, Any]] = []
    missing: list[str] = []
    for suffix in MODEL_ARTIFACT_SUFFIXES:
        path = root / f"v{version}_{suffix}"
        if not path.is_file():
            missing.append(str(path))
            continue
        record = file_identity(path, base_dir=root)
        files.append(record)
        # 载荷键与 P4 门 1 同形，保证跨阶段指纹可比
        artifacts[suffix] = {"bytes": record["size"], "sha256": record["sha256"]}
    if missing:
        raise FileNotFoundError(f"试车场臂模型产物缺失（v{version}）: {missing}")
    metadata_path = root / f"v{version}_metadata.json"
    metadata = _read_metadata(metadata_path)
    model_type = metadata.get("model_type")
    if not model_type:
        raise TestbedError(f"模型 v{version} 的 metadata 缺 model_type: {metadata_path}")
    train_start_day = _metadata_day(
        _required_metadata(metadata, "train_start_date", metadata_path), name="train_start_date"
    )
    train_end_day = _metadata_day(
        _required_metadata(metadata, "train_end_date", metadata_path), name="train_end_date"
    )
    if train_start_day > train_end_day:
        raise TestbedError(
            f"模型训练区间非法（start 晚于 end）: {train_start_day} > {train_end_day}"
        )
    d0_day = normalize_day(d0_date, name="d0_date")
    if train_end_day >= d0_day:
        raise TestbedError(
            f"模型训练截止必须严格早于 D0（设计 §4.1）: train_end_date={train_end_day} "
            f">= d0_date={d0_day}"
        )
    return {
        "version": int(version),
        "models_dir": str(root.resolve()).replace("\\", "/"),
        "artifacts": artifacts,
        "files": files,
        "content_fingerprint": fingerprint(artifacts),
        "model_type": str(model_type),
        "train_start_date": train_start_day,
        "train_end_date": train_end_day,
        "trained_before_d0": True,
    }


def _required_metadata(metadata: Mapping[str, Any], key: str, path: Path) -> Any:
    value = metadata.get(key)
    if value is None or str(value).strip() == "":
        raise TestbedError(f"模型 metadata 缺必需字段 {key}: {path}")
    return value


def _metadata_day(value: Any, *, name: str) -> str:
    """metadata 日期 → YYYYMMDD：先去除 ``-``/``/`` 分隔符，再做真实日期校验。"""
    raw = str(value).strip().replace("-", "").replace("/", "")
    return normalize_day(raw, name=name)


def _read_metadata(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TestbedError(f"模型 metadata 读取失败（损坏/非法 JSON）: {path}（{exc}）") from exc
    if not isinstance(payload, dict):
        raise TestbedError(f"模型 metadata 顶层必须是对象: {path}")
    return payload


def _stem_day(stem: str) -> Optional[str]:
    """分区文件名 stem → YYYYMMDD（非日期形态返回 None，不参与水位）。"""
    digits = stem.replace("-", "")
    return digits if len(digits) == 8 and digits.isdigit() else None


def _latest_partition(directory: Path) -> Optional[str]:
    """目录内最新分区日期（YYYYMMDD；忽略非日期文件）。"""
    if not directory.is_dir():
        return None
    partitions = [path for path in directory.iterdir() if path.suffix == ".parquet"]
    days = [day for day in (_stem_day(path.stem) for path in partitions) if day is not None]
    return max(days) if days else None


def _dash(day: str) -> str:
    return f"{day[:4]}-{day[4:6]}-{day[6:]}"


def _partition_path(data_root: Path, dataset: str, day: str) -> Path:
    name = _dash(day) if dataset.startswith("clean/") or dataset.startswith("raw/") else day
    return data_root / dataset / f"{name}.parquet"


def collect_data_state_identity(*, data_root: str, virtual_days: Sequence[str]) -> Dict[str, Any]:
    """数据态身份：窗口内容（门禁）+ 全库水位（仅登记）。

    ``window_state_id`` 只由窗口内实际分区的物理身份构成，窗口外新增分区不改变
    它（避免生产日更把 campaign 冻死）；``global_watermark`` 单独登记以便报告。
    """
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"试车场数据根不存在: {root}")
    days = [str(day) for day in virtual_days]
    if not days:
        raise TestbedError("数据态身份需要至少一个虚拟日（D0）")
    window_files: list[Dict[str, Any]] = []
    panel_present = (root / _PANEL_DATASET).is_dir()
    for day in days:
        for dataset in _REQUIRED_WINDOW_DATASETS:
            path = _partition_path(root, dataset, day)
            if not path.is_file():
                raise FileNotFoundError(f"虚拟日 {day} 缺必需清洗分区: {path}（禁止跳日或填旧值）")
            window_files.append(file_identity(path, base_dir=root))
        for dataset in _OPTIONAL_WINDOW_DATASETS:
            path = _partition_path(root, dataset, day)
            if path.is_file():
                window_files.append(file_identity(path, base_dir=root))
        # v2 宿主实际消费 panel：按**实际读域**登记，而不是只看热区单个 core 文件
        # （R2-P5B-01：其他特征族 fundamental/sector... 与冷区 archive 都可能被读取）
        if panel_present:
            hot = sorted((root / _PANEL_DATASET / day).glob("*.parquet"))
            if not hot:
                raise FileNotFoundError(
                    f"虚拟日 {day} 缺 panel 分区: {root / _PANEL_DATASET / day}"
                    "（宿主实际消费该输入，禁止缺日或填旧值）"
                )
            for path in hot:
                window_files.append(file_identity(path, base_dir=root))
            # 冷区回退：store 可能按月份目录读取 archive 分区（YYYY-MM 与 YYYYMM 两种命名都探测）
            month = f"{day[:4]}-{day[4:6]}"
            for candidate in (
                root / _PANEL_ARCHIVE / month / "core.parquet",
                root / _PANEL_ARCHIVE / f"{day[:6]}" / "core.parquet",
            ):
                if candidate.is_file():
                    window_files.append(file_identity(candidate, base_dir=root))
    watermark = {dataset: _latest_partition(root / dataset) for dataset in _WATERMARK_DATASETS}
    return {
        "window_files": window_files,
        "window_day_count": len(days),
        "window_state_id": canonical_json_sha256({"window_files": window_files})[:16],
        "panel_covered": panel_present,
        "global_watermark": watermark,
        "global_watermark_note": "仅登记，不参与 campaign 漂移门禁",
    }


def collect_code_identity(repo_root: str) -> Dict[str, Any]:
    """代码态身份：执行闭包（门禁）+ 全仓工作树（仅登记）。"""
    root = Path(repo_root)
    sources: list[Path] = []
    for entry in EXECUTION_CLOSURE:
        path = root / entry
        if path.is_dir():
            sources.extend(sorted(path.rglob("*.py")))
        elif path.is_file():
            sources.append(path)
        else:
            raise FileNotFoundError(f"执行闭包文件缺失: {path}")
    execution_files = collect_file_identities(sorted(sources), base_dir=root)
    workspace = workspace_identity(root)
    return {
        "execution_digest": canonical_json_sha256(execution_files),
        "execution_file_count": len(execution_files),
        "git_commit": workspace["git_commit"],
        "git_dirty": workspace["git_dirty"],
        "workspace_source_digest": workspace["source_digest"],
        "workspace_source_file_count": workspace["source_file_count"],
        "environment": workspace["environment"],
    }


def collect_input_identity(
    *,
    data_root: str,
    row_domain: str,
    d0_date: str,
    virtual_days: Sequence[str],
    window_state_id: str,
) -> Dict[str, Any]:
    """输入面身份：只读历史输入根 + 行域声明 + 虚拟窗口 + 窗口数据态引用。"""
    payload = {
        "data_root": str(Path(data_root).resolve()).replace("\\", "/"),
        "row_domain": str(row_domain),
        "d0_date": str(d0_date),
        "virtual_days": [str(day) for day in virtual_days],
        "window_state_id": str(window_state_id),
    }
    return {**payload, "input_sha256": canonical_json_sha256(payload)}


def identity_summary(identity: Mapping[str, Any]) -> Dict[str, Any]:
    """身份压成可比较的哈希摘要（campaign/arm 共用，缺段显式 None）。"""
    config = identity.get("config") or {}
    model = identity.get("model") or {}
    code = identity.get("code") or {}
    window = identity.get("window_state") or {}
    source = identity.get("input") or {}
    return {
        "input": source.get("input_sha256"),
        "config": config.get("fingerprint"),
        "model": model.get("content_fingerprint"),
        "code": code.get("execution_digest"),
        "window_state": window.get("window_state_id"),
        "environment": code.get("environment") or environment_identity(),
    }
