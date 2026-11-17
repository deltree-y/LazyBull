"""固定缓冲换仓研究的数据适配与只读缓存，不进入生产链路。

完整合法排名由冻结模型重新推理一次；两臂消费同一缓存。主矩阵 scores 的
行域不同，禁止用它补排名。阶段研究工具按 CLAUDE §7.8 豁免复杂度。
"""

# 复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.8）。

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.v2_p3.t05b_fingerprint import normalize_signal_topk_rows
from scripts.v2_p7.t25.params import (
    T25Params,
    _load_model_identity,
    build_params,
    compute_input_identity,
)
from scripts.v2_p7.t25.signal_layer import RankedProvider, compare_frames
from scripts.v2_p7.t25.target_layer import VarianceBook

SOURCE = Path("data/paper_shadow/p7_testbed/p7_t25_hk_20261112_r6")
STOCK_BASIC_SHA = "fd460335771274e2f7c5008d8be43b7dc2bc734d3518bac75339498181356d64"
TOPK_COLUMNS = ["trade_date", "rank", "ts_code", "score"]
CACHE_SCHEMA = "p7_buffer_path_inputs/v1"


@dataclass
class ResearchInputs:
    """两条研究路径共享的冻结输入；不含候选收益。"""

    params: T25Params
    calendar: tuple[str, ...]
    signal_days: tuple[str, ...]
    schedule: dict[str, int]
    ranked: dict[str, list[tuple[str, float]]]
    variance_book: VarianceBook
    loader: ReadOnlyDailyLoader
    identity: dict[str, Any]
    validation: dict[str, Any]


class ReadOnlyDailyLoader:
    """只允许读取登记日历中的 clean 行情，禁止自动补齐。"""

    def __init__(self, loader: Any, calendar: Sequence[str]) -> None:
        self._loader = loader
        self._days = frozenset(calendar)

    def load_clean_daily_by_date(
        self, trade_date: str, auto_ensure: bool = False
    ) -> pd.DataFrame | None:
        """按业务日读取；任何扩域或隐式写入请求直接拒绝。"""
        if auto_ensure:
            raise ValueError("研究行情只读，禁止 auto_ensure=True")
        day = str(trade_date)
        if day not in self._days:
            raise ValueError(f"行情日不在已登记扩展日历：{day}")
        frame = self._loader.load_clean_daily_by_date(day, auto_ensure=False)
        if frame is None or frame.empty:
            raise ValueError(f"已登记行情分区为空：{day}")
        if "trade_date" not in frame or not frame.trade_date.astype(str).eq(day).all():
            raise ValueError(f"行情行日期与请求不一致：{day}")
        if "ts_code" not in frame or frame.ts_code.duplicated().any():
            raise ValueError(f"行情股票键缺失或重复：{day}")
        return frame


def _sha(path: Path) -> str:
    """流式计算文件字节身份。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """读取 UTF-8 对象，不猜测或补齐元信息。"""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"元信息不是对象：{path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """只在新缓存根内写确定性 JSON。"""
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _json_value(value: Any) -> Any:
    """统一整数对象键在 JSON 落盘前后的表示。"""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _check_spec(spec_sha256: str) -> None:
    """正式准备和复用都必须绑定冻结规格的完整 SHA。"""
    if len(spec_sha256) != 64 or any(c not in "0123456789abcdef" for c in spec_sha256):
        raise ValueError("spec_sha256 必须是冻结规格的 64 位小写 SHA256")


def _code_identity(root: Path) -> dict[str, Any]:
    """覆盖推理、选择过滤、行情、方差及适配依赖，排除无关研究入口。"""
    paths = set((root / "src/lazybull").rglob("*.py"))
    paths.update((root / "scripts/v2_p3").rglob("*.py"))
    paths.update((root / "scripts/v2_p7/t25").rglob("*.py"))
    paths.update(
        root / name
        for name in (
            "scripts/v2_p7/buffer_path_data.py",
            "scripts/v2_p7/common.py",
            "configs/base.yaml",
        )
    )
    files = {path.relative_to(root).as_posix(): _sha(path) for path in sorted(paths)}
    environment = {
        name: importlib.metadata.version(name) for name in ("numpy", "pandas", "xgboost")
    }
    environment["python"] = platform.python_version()
    return {"files": files, "environment": environment}


def validate_input_identity(current: Mapping[str, Any], reference: Mapping[str, Any]) -> None:
    """逐组核对 r6 数据身份，仅继承已裁决的 stock_basic 特定字节。"""
    for label, value in (("当前", current), ("r6", reference)):
        if value.get("clean_meta_mismatch"):
            raise ValueError(f"{label} 元数据存在未接受漂移")
        for group in ("cs_train", "clean_daily"):
            if value[group].get("frozen_manifest_mismatch"):
                raise ValueError(f"{label} {group} 存在未接受漂移")
        for accepted in value.get("accepted_input_drift", []):
            if (
                accepted.get("path") != "data/clean/stock_basic.parquet"
                or accepted.get("current") != STOCK_BASIC_SHA
            ):
                raise ValueError(f"{label} 出现不在 r6 裁决内的输入漂移")
        if value["clean_meta"]["data/clean/stock_basic.parquet"]["sha256"] != STOCK_BASIC_SHA:
            raise ValueError(f"{label} stock_basic 不是 r6 已接受字节")
    for group in ("cs_train", "clean_daily"):
        for key in ("digest", "partitions", "frozen_manifest_checked"):
            if current[group][key] != reference[group][key]:
                raise ValueError(f"当前 {group}.{key} 与 r6 不一致")
    if _json_value(current["clean_meta"]) != _json_value(reference["clean_meta"]):
        raise ValueError("当前 clean 元数据身份与 r6 不一致")


def _reference_topk(
    root: Path, params: T25Params
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    """核验已保存辅证的 SHA、折模型、日域与合法缺日计数。"""
    source = root / SOURCE
    manifest = _read_json(source / "manifest.json")
    scores = _read_json(source / "scores.json")
    registry = _read_json(source / "fold_registry.json")
    records = {int(row["split_index"]): row for row in registry["folds"]}
    if set(records) != {fold.split_index for fold in params.folds}:
        raise ValueError("r6 折登记与当前参数不一致")
    paths = [source / name for name in ("manifest.json", "scores.json", "fold_registry.json")]
    frames: list[pd.DataFrame] = []
    empty_days: list[str] = []
    closure_days: list[str] = []
    for fold in params.folds:
        folder = source / "folds" / f"split{fold.split_index:02d}"
        meta_path, topk_path = folder / "fold.json", folder / "signal_topk.parquet"
        metadata = _read_json(meta_path)
        registered = records[fold.split_index]
        paths.extend([meta_path, topk_path])
        if (
            int(metadata["model_version"]) != fold.model_version
            or int(registered["model_version"]) != fold.model_version
        ):
            raise ValueError(f"split{fold.split_index:02d} 模型身份不一致")
        digest = _sha(topk_path)
        if (
            digest != metadata["signal_topk_sha256"]
            or digest != registered["scores"]["signal_topk_sha256"]
        ):
            raise ValueError(f"split{fold.split_index:02d} Top20 原文件 SHA 不符")
        frame = pd.read_parquet(topk_path, columns=TOPK_COLUMNS)
        if len(frame) != int(metadata["signal_topk_rows"]):
            raise ValueError(f"split{fold.split_index:02d} Top20 行数不符")
        span = set(fold.signal_days)
        closures = set(str(day) for day in metadata["closure_days"])
        present = set(frame.trade_date.astype(str))
        if not closures <= span or not present <= span or closures & present:
            raise ValueError(f"split{fold.split_index:02d} Top20/闭市日期越域或相交")
        absent = span - closures - present
        if len(absent) != int(metadata["signal_topk_empty_days"]):
            raise ValueError(f"split{fold.split_index:02d} 缺排名日不符合原空日登记")
        if frame.duplicated(["trade_date", "ts_code"]).any():
            raise ValueError(f"split{fold.split_index:02d} Top20 股票键重复")
        for day, group in frame.groupby("trade_date", sort=False):
            if group["rank"].tolist() != list(range(1, len(group) + 1)) or len(group) > 20:
                raise ValueError(f"{day} Top20 名次不连续或超限")
        frames.append(frame)
        closure_days.extend(closures)
        empty_days.extend(absent)
    if sorted(closure_days) != sorted(scores["stock_connect_closure_days"]):
        raise ValueError("逐折闭市日与 r6 总登记不一致")
    if not set(scores["empty_ranking_days"]) <= set(empty_days + closure_days):
        raise ValueError("r6 目标层空日与完整信号辅证不一致")
    reference = pd.concat(frames, ignore_index=True).sort_values(["trade_date", "rank"])
    info = {
        "input_identity": manifest["input_identity"],
        "closure_days": sorted(closure_days),
        "reference_empty_days": sorted(empty_days),
    }
    return (
        reference.reset_index(drop=True),
        info,
        {path.relative_to(root).as_posix(): _sha(path) for path in paths},
    )


def validate_rankings(
    ranked: Mapping[str, Sequence[tuple[str, float]]],
    reference: pd.DataFrame,
    signal_days: Sequence[str],
    closure_days: Sequence[str],
    reference_empty_days: Sequence[str],
) -> dict[str, Any]:
    """逐位核对辅证 Top20；内核并列次序保留并独立登记。"""
    if set(ranked) != set(signal_days) or len(set(signal_days)) != len(signal_days):
        raise ValueError("全排名日域与冻结信号日域不一致")
    rows: list[tuple[str, int, str, float]] = []
    ties: list[dict[str, Any]] = []
    empty: list[str] = []
    for day in signal_days:
        values = [(str(code), float(score)) for code, score in ranked[day]]
        if len({code for code, _ in values}) != len(values):
            raise ValueError(f"{day} 全排名股票键重复")
        if any(not code or not math.isfinite(score) for code, score in values):
            raise ValueError(f"{day} 全排名含空股票或非有限分数")
        if any(left[1] < right[1] for left, right in zip(values, values[1:])):
            raise ValueError(f"{day} 内核全排名不是分数降序")
        if not values:
            empty.append(day)
        canonical = sorted(values, key=lambda pair: (-pair[1], pair[0]))[:20]
        if [code for code, _ in values[:20]] != [code for code, _ in canonical]:
            ties.append(
                {
                    "trade_date": day,
                    "kernel_top20": [code for code, _ in values[:20]],
                    "auxiliary_top20": [code for code, _ in canonical],
                }
            )
        if day in closure_days:
            if values:
                raise ValueError(f"{day} 原闭市缺排名日出现新合法候选，禁止静默接受")
            continue
        rows.extend((day, rank, code, score) for rank, (code, score) in enumerate(canonical, 1))
    current = pd.DataFrame(normalize_signal_topk_rows(rows), columns=TOPK_COLUMNS).astype(
        {"trade_date": str, "rank": np.int64, "ts_code": str, "score": np.float64}
    )
    comparison = compare_frames(reference.reset_index(drop=True), current)
    if not comparison["identical"]:
        raise ValueError(f"合法排名 Top20 与冻结辅证不逐位一致：{comparison}")
    if set(empty) != set(closure_days) | set(reference_empty_days):
        raise ValueError("完整排名空日与冻结闭市/合法空日登记不一致")
    return {
        "top20_auxiliary_bitwise_exact": True,
        "top20_rows": len(current),
        "signal_days": len(signal_days),
        "ranked_rows": sum(len(value) for value in ranked.values()),
        "closure_days": sorted(closure_days),
        "empty_ranking_days": sorted(empty),
        "ordinary_empty_ranking_days": sorted(set(empty) - set(closure_days)),
        "kernel_top20_order_exact": not ties,
        "kernel_tie_differences": ties,
        "warmup_boundary": "沿用 T25 首个 OOS 日起表，不额外加入预热日",
    }


def _capture_identity(
    root: Path, params: T25Params, reference: Mapping[str, Any], files: Mapping[str, str], spec: str
) -> dict[str, Any]:
    """采集当前身份并严格对照 r6，不消费任何候选收益。"""
    current = compute_input_identity(
        params, accepted_drift=(f"data/clean/stock_basic.parquet:{STOCK_BASIC_SHA}",)
    )
    validate_input_identity(current, reference["input_identity"])
    observed = {name: _sha(root / name) for name in files}
    if observed != dict(files):
        raise ValueError("读取期间 r6 辅证输入字节发生变化")
    return _json_value(
        {
            "spec_sha256": spec,
            "data": current,
            "models": _load_model_identity(params.data_root, params.precheck_dir, params.folds),
            "reference_files": observed,
            "code": _code_identity(root),
            "calendar": list(params.extended_days),
            "signal_days": list(params.signal_days),
            "schedule": dict(params.v20_schedule),
        }
    )


def _runtime_inputs(
    params: T25Params, ranked: dict[str, list[tuple[str, float]]], identity: dict, validation: dict
) -> ResearchInputs:
    """两臂共用只读行情和同一方差表，方差查询由既有实现按 <=T 截断。"""
    from src.lazybull.data.loader import DataLoader
    from src.lazybull.data.storage import Storage

    loader = ReadOnlyDailyLoader(
        DataLoader(Storage(root_path=str(params.data_root))), params.extended_days
    )
    variance = VarianceBook(params.extended_days, loader.load_clean_daily_by_date)
    return ResearchInputs(
        params=params,
        calendar=tuple(params.extended_days),
        signal_days=tuple(params.signal_days),
        schedule=dict(params.v20_schedule),
        ranked=ranked,
        variance_book=variance,
        loader=loader,
        identity=identity,
        validation=validation,
    )


def prepare_inputs(repo_root: Path, cache_dir: Path, *, spec_sha256: str) -> ResearchInputs:
    """显式创建全窗共享输入；不训练，目标目录存在即拒绝，最终 manifest 才表示完成。"""
    started = time.perf_counter()
    root, cache = Path(repo_root).resolve(), Path(cache_dir).resolve()
    _check_spec(spec_sha256)
    if cache.exists():
        raise FileExistsError(f"缓存目录已存在，拒绝覆盖：{cache}")
    params = build_params(root)
    reference, info, files = _reference_topk(root, params)
    identity = _capture_identity(root, params, info, files, spec_sha256)
    cache.mkdir(parents=True, exist_ok=False)
    _write_json(cache / "input_identity.json", identity)
    predictor = RankedProvider(params)
    inference_started = time.perf_counter()
    ranked = {day: list(predictor(day)) for day in params.signal_days}
    inference_seconds = time.perf_counter() - inference_started
    validation = validate_rankings(
        ranked, reference, params.signal_days, info["closure_days"], info["reference_empty_days"]
    )
    inputs = _runtime_inputs(params, ranked, identity, validation)
    if _capture_identity(root, params, info, files, spec_sha256) != identity:
        raise ValueError("准备输入期间数据、模型规格或代码身份发生变化")
    rows = [
        (day, rank, code, score)
        for day in params.signal_days
        for rank, (code, score) in enumerate(ranked[day], 1)
    ]
    frame = pd.DataFrame(rows, columns=TOPK_COLUMNS).astype(
        {"trade_date": str, "rank": np.int64, "ts_code": str, "score": np.float64}
    )
    frame.to_parquet(cache / "ranked.parquet", engine="pyarrow", compression="snappy", index=False)
    validation["inference_seconds"] = inference_seconds
    validation["preparation_seconds"] = time.perf_counter() - started
    _write_json(cache / "validation.json", validation)
    _write_json(
        cache / "manifest.json",
        {
            "schema": CACHE_SCHEMA,
            "completed": True,
            "spec_sha256": spec_sha256,
            "outputs": {
                name: _sha(cache / name)
                for name in ("input_identity.json", "ranked.parquet", "validation.json")
            },
        },
    )
    return inputs


def load_inputs(repo_root: Path, cache_dir: Path, *, spec_sha256: str) -> ResearchInputs:
    """复用已完成缓存；重核输入、代码、规格与辅证，禁止自动回退重新推理。"""
    root, cache = Path(repo_root).resolve(), Path(cache_dir).resolve()
    _check_spec(spec_sha256)
    manifest = _read_json(cache / "manifest.json")
    if manifest.get("schema") != CACHE_SCHEMA or manifest.get("completed") is not True:
        raise ValueError("缓存未完成或 schema 不符")
    if manifest.get("spec_sha256") != spec_sha256:
        raise ValueError("缓存冻结规格 SHA 不符")
    expected_names = {"input_identity.json", "ranked.parquet", "validation.json"}
    if set(manifest.get("outputs", {})) != expected_names:
        raise ValueError("缓存产物集合不完整")
    for name, digest in manifest["outputs"].items():
        if _sha(cache / name) != digest:
            raise ValueError(f"缓存产物字节变化：{name}")
    identity = _read_json(cache / "input_identity.json")
    params = build_params(root)
    reference, info, files = _reference_topk(root, params)
    if _capture_identity(root, params, info, files, spec_sha256) != identity:
        raise ValueError("缓存输入、代码或规格身份已变化，禁止复用")
    frame = pd.read_parquet(cache / "ranked.parquet", columns=TOPK_COLUMNS)
    if not set(frame.trade_date.astype(str)) <= set(params.signal_days):
        raise ValueError("缓存排名包含冻结信号日历之外的日期")
    ranked: dict[str, list[tuple[str, float]]] = {day: [] for day in params.signal_days}
    for day, group in frame.groupby("trade_date", sort=False):
        if group["rank"].tolist() != list(range(1, len(group) + 1)):
            raise ValueError(f"{day} 缓存全排名名次不连续")
        ranked[str(day)] = [
            (str(code), float(score)) for code, score in zip(group.ts_code, group.score)
        ]
    validation = validate_rankings(
        ranked, reference, params.signal_days, info["closure_days"], info["reference_empty_days"]
    )
    saved_validation = _read_json(cache / "validation.json")
    if any(saved_validation.get(key) != value for key, value in validation.items()):
        raise ValueError("缓存覆盖/核验登记与当前复算不一致")
    validation.update({"cache_reused": True, "original_timing": saved_validation})
    inputs = _runtime_inputs(params, ranked, identity, validation)
    if _capture_identity(root, params, info, files, spec_sha256) != identity:
        raise ValueError("复用输入期间数据或代码身份发生变化")
    return inputs
