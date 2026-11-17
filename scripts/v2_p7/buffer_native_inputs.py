"""固定续持原生复验的只读输入适配，不训练、不计算候选收益。

全段统一读取官方 panel；cs_infer 只按纸面宿主规则补 freshness。
历史 panel 的 label_filtered 行域原样登记，禁止伪称完整 structural 截面。
"""

# 复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.8）。

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from loguru import logger

START = "20260203"
END = "20261029"
MODEL_VERSION = 24022
SCHEMA = "p7_buffer_native_rankings/v1"
PREREG = Path("docs/experiments/p7_buffer_native_prereg_20261116.md")


def _sha(path: Path) -> str:
    """按实际文件字节取身份，缺文件不静默跳过。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _file_record(root: Path, path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(root).as_posix(), "sha256": _sha(path)}


def _calendar(root: Path) -> list[str]:
    frame = pd.read_parquet(root / "data/clean/trade_cal.parquet")
    selected = frame[(frame.exchange == "SSE") & (frame.is_open.astype(int) == 1)]
    return sorted(selected.cal_date.astype(str).unique().tolist())


def _training_dates(root: Path) -> list[dict[str, Any]]:
    """只读训练日志中的身份和日期，不读取模型评估收益。"""
    path = root / "data/models/stock_selection/ml_train_runs.csv"
    columns = pd.read_csv(path, nrows=0).columns
    selected = [
        name
        for name in columns
        if name in {"model_version", "timestamp", "split_index", "wf_run_id"}
        or name.endswith("_date")
    ]
    frame = pd.read_csv(path, usecols=selected, low_memory=False)
    frame = frame[frame.model_version.isin([24021, MODEL_VERSION])].astype(object)
    return frame.where(pd.notna(frame), None).to_dict("records")


def _industry_audit(root: Path) -> dict[str, Any]:
    """区分生产静态行业映射与另存的 PIT 研究表，不替换任何输入。"""
    relative = [
        "data/raw/shenwan_industry.parquet",
        "data/clean/sw_industry_pit.parquet",
        "data/clean/sw_industry_pit/pit_20261112.parquet",
        "data/clean/sw_industry_pit/code_table_20261112.parquet",
    ]
    files = []
    for name in relative:
        path = root / name
        if path.is_file():
            table = pq.ParquetFile(path)
            files.append(
                {
                    **_file_record(root, path),
                    "rows": table.metadata.num_rows,
                    "columns": table.schema_arrow.names,
                }
            )
    return {
        "production_mapping": "DataLoader.load_shenwan_industry -> raw/shenwan_industry",
        "historical_pit_proven": False,
        "limitation": "当前行业映射未按信号日筛选历史有效区间；另存 PIT 表不能证明生产输入已使用它。",
        "files": files,
    }


def audit_holdout(repo_root: Path) -> dict[str, Any]:
    """返回冻结整段的物理覆盖、模型时点、行域与输入 SHA；不读取标签值。"""
    root = repo_root.resolve()
    calendar = _calendar(root)
    days = [day for day in calendar if START <= day <= END]
    if not days or days[0] != START or days[-1] != END:
        raise ValueError(f"新窗口日历不完整：{START}–{END}")
    manifest_path = root / "data/features/manifest.json"
    manifest = _json(manifest_path)
    paths = {
        root / "data/clean/trade_cal.parquet",
        root / "data/clean/stock_basic.parquet",
        manifest_path,
        root / "data/models/stock_selection/model_registry.json",
        root / "data/models/stock_selection/ml_train_runs.csv",
        root / "data/paper/config.yaml",
        root / "configs/base.yaml",
    }
    paths.update(path for path in (root / "data/features").glob("manifest*") if path.is_file())
    wal = root / "data/features/manifest_wal.jsonl"
    absent_paths = [] if wal.exists() else [wal.relative_to(root).as_posix()]
    models = {}
    for version in (24021, MODEL_VERSION):
        prefix = root / "data/models/stock_selection"
        meta_path = prefix / f"v{version}_metadata.json"
        meta = _json(meta_path)
        model_paths = [meta_path, prefix / meta["model_file"], prefix / meta["features_file"]]
        paths.update(model_paths)
        models[str(version)] = {
            key: meta[key]
            for key in ("train_start_date", "train_end_date", "created_at", "label_column")
        }
        models[str(version)]["files"] = [_file_record(root, path) for path in model_paths]
    coverage = []
    for day in days:
        clean = root / "data/clean/daily" / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet"
        panel = sorted((root / "data/features/panel" / day).glob("*.parquet"))
        if not clean.is_file() or len(panel) != 8:
            raise ValueError(f"{day} 行情或八组 panel 缺失；禁止 cs_train 回退")
        paths.add(clean)
        paths.update(panel)
        domains = {
            manifest["partitions"][path.relative_to(root / "data/features").as_posix()][
                "row_domain"
            ]
            for path in panel
        }
        expected = "label_filtered" if day <= "20260702" else "structural"
        if domains != {expected}:
            raise ValueError(f"{day} panel 行域与冻结分段不一致：{domains}")
        infer = root / "data/features/cs_infer" / f"{day}.parquet"
        if infer.is_file():
            paths.add(infer)
        else:
            absent_paths.append(infer.relative_to(root).as_posix())
        coverage.append(
            {
                "day": day,
                "row_domain": expected,
                "cs_infer_exists": infer.is_file(),
                "cs_train_exists": (root / "data/features/cs_train" / f"{day}.parquet").is_file(),
            }
        )
    industry = _industry_audit(root)
    paths.update(root / item["path"] for item in industry["files"])
    return {
        "schema": SCHEMA,
        "start": START,
        "end": END,
        "days": days,
        "day_count": len(days),
        "model_version": MODEL_VERSION,
        "models": models,
        "training_dates": _training_dates(root),
        "coverage": coverage,
        "row_domain_segments": manifest["row_domain_segments"],
        "labels_sealed_through": {
            key: value.get("sealed_through") for key, value in manifest.get("labels", {}).items()
        },
        "industry": industry,
        "evidence_grade": "H；候选未用历史窗口，非前向、非全项目盲留出",
        "other_research_usage": [
            "P7 P1 与 C1 历史研究使用至 20261029",
            "P5b T6 使用 20260703–20261029；P4 影子使用 20260717–20260730",
            "F24 已将 20250612–20260702 登记为非 virgin 的模型刷新 OOS 段",
        ],
        "input_files": [_file_record(root, path) for path in sorted(paths)],
        "absent_input_paths": absent_paths,
    }


class HoldoutRankedProvider:
    """24022 固定模型；复用内核选择，官方 panel 与 freshness 输入路径。"""

    def __init__(self, root: Path) -> None:
        from scripts.v2_p3.t05b_fingerprint import _configure_signal_selection
        from scripts.v2_p7.t25.signal_layer import _signal_selection, _signal_universe_config
        from src.lazybull.data.loader import DataLoader
        from src.lazybull.data.storage import Storage
        from src.lazybull.universe import BasicUniverse
        from src.lazybull.v2.core.signal.ml_signal import MLSignal
        from src.lazybull.v2.hosts.paper import daily_assembly as assembly
        from src.lazybull.v2.store.data_store import PanelDataStore

        self.root = root
        self.loader = DataLoader(Storage(root_path=str(root / "data")))
        self.store = PanelDataStore(root / "data")
        self.assembly = assembly
        self.signal = MLSignal(
            top_n=20,
            model_version=MODEL_VERSION,
            models_dir=str(root / "data/models/stock_selection"),
            verbose=False,
        )
        # MLSignal 延迟加载；官方列来源检查必须使用已加载模型的实际列集。
        self.signal._load_model()
        self.selection = dict(_signal_selection())
        self.universe_config = dict(_signal_universe_config())
        _configure_signal_selection(self.signal, self.selection)
        basic = self.loader.load_clean_stock_basic()
        if basic is None or basic.empty:
            raise ValueError("缺少 clean/stock_basic；不允许回退 raw")
        self.universe = BasicUniverse(stock_basic=basic, verbose=False, **self.universe_config)
        registered = set(self.store.manifest.snapshot()["columns"])
        classification = assembly._classify_columns(self.signal.feature_columns, registered)
        gaps, uncovered = assembly._check_column_sources(classification, (), frozenset())
        if uncovered:
            raise ValueError(f"模型列来源不完整：{gaps}")
        self.freshness = assembly._required_freshness_columns(
            self.signal.feature_columns, self.signal.metadata.get("train_params", {})
        )
        self.columns = assembly._store_query_columns(classification, registered, self.freshness)
        missing_filters = set(assembly.SELECTION_FILTER_COLUMNS) - set(self.columns)
        if missing_filters:
            raise ValueError(f"panel 缺选择过滤列：{sorted(missing_filters)}")
        self.day_records: list[dict[str, Any]] = []

    def _features(self, day: str) -> pd.DataFrame:
        from src.lazybull.v2.common.types import FeatureQuery, TradeDate

        result = self.store.load_features(
            FeatureQuery(
                columns=self.columns,
                start_date=TradeDate.from_str(day),
                end_date=TradeDate.from_str(day),
            )
        )
        frames = self.assembly._split_panel_frame_by_date(result.df)
        frame = frames.get(day)
        if frame is None or frame.empty:
            raise ValueError(f"{day} 官方 panel 为空；禁止 cs_train 或其他日期回退")
        infer = self.assembly._load_cs_infer_freshness_frame(self.loader, day, self.freshness)
        frame, present = self.assembly._merge_cs_infer_freshness(frame, infer, self.freshness)
        missing, gaps = self.assembly._check_freshness_columns(self.freshness, frame, present)
        if missing:
            raise ValueError(f"{day} freshness 输入不完整：{gaps}")
        if frame.ts_code.duplicated().any() or not frame.trade_date.astype(str).eq(day).all():
            raise ValueError(f"{day} panel 股票键重复或日期越界")
        return frame

    def __call__(self, day: str) -> list[tuple[str, float]]:
        if not START <= day <= END:
            raise ValueError(f"排名日不在冻结窗口：{day}")
        frame = self._features(day)
        quote = self.loader.load_clean_daily_by_date(day, auto_ensure=False)
        if quote is None or quote.empty or not quote.trade_date.astype(str).eq(day).all():
            raise ValueError(f"{day} clean 行情缺失或日期不符")
        codes = self.universe.get_stocks(pd.Timestamp(day), quote_data=quote)
        ranked = self.signal.generate_ranked(pd.Timestamp(day), codes, {"features": frame.copy()})
        result = [(str(code), float(score)) for code, score in ranked]
        if (
            len({code for code, _ in result}) != len(result)
            or any(not math.isfinite(score) for _, score in result)
            or any(left[1] < right[1] for left, right in zip(result, result[1:]))
        ):
            raise ValueError(f"{day} 内核排名含重复股票或非有限分数")
        record: dict[str, Any] = {
            "day": day,
            "feature_rows": len(frame),
            "ranked_rows": len(result),
        }
        if not result:
            section = frame[frame.ts_code.isin(codes)]
            selected, _, _ = self.signal._apply_selection_filters(section.copy())
            all_nan = [
                col
                for col in self.signal.feature_columns
                if col in selected and selected[col].isna().all()
            ]
            record.update(
                empty_reason=(
                    "universe_empty"
                    if not codes
                    else "selection_empty" if selected.empty else "kernel_refused"
                ),
                all_nan_model_columns=all_nan,
            )
        self.day_records.append(record)
        return result


def _verify_files(root: Path, records: list[dict[str, Any]]) -> None:
    for item in records:
        if _sha(root / item["path"]) != item["sha256"]:
            raise ValueError(f"输入在本次推理期间漂移：{item['path']}")


def build_holdout_rankings(
    repo_root: Path, output: Path, *, spec_sha256: str
) -> dict[str, list[tuple[str, float]]]:
    """一次全窗推理保存共享缓存；拒覆盖，规格/实际输入字节漂移即拒发布。"""
    root = repo_root.resolve()
    if _sha(root / PREREG) != spec_sha256:
        raise ValueError("新窗口预登记 SHA 不匹配")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    audit = audit_holdout(root)
    _write_json(output / "audit.json", audit)
    code_identity = _file_record(root, root / "scripts/v2_p7/buffer_native_inputs.py")
    provider = HoldoutRankedProvider(root)
    rankings: dict[str, list[tuple[str, float]]] = {}
    for index, day in enumerate(audit["days"], 1):
        rankings[day] = provider(day)
        if index % 20 == 0 or index == len(audit["days"]):
            logger.info(f"新窗口排名输入：{index}/{len(audit['days'])} 日，最后 {day}")
    _verify_files(root, audit["input_files"] + [code_identity])
    appeared = [name for name in audit["absent_input_paths"] if (root / name).exists()]
    if appeared:
        raise ValueError(f"原先缺席的可消费输入在推理期间出现：{appeared}")
    if _sha(root / PREREG) != spec_sha256:
        raise ValueError("推理期间预登记发生变化")
    frame = pd.DataFrame(
        [
            (day, rank, code, score)
            for day, rows in rankings.items()
            for rank, (code, score) in enumerate(rows, 1)
        ],
        columns=["trade_date", "rank", "ts_code", "score"],
    )
    frame.to_parquet(output / "ranked.parquet", index=False)
    validation = {
        "complete": True,
        "days": len(rankings),
        "ranked_rows": len(frame),
        "empty_days": [row for row in provider.day_records if row["ranked_rows"] == 0],
        "day_records": provider.day_records,
        "preparation_seconds": time.perf_counter() - started,
        "selection": provider.selection,
        "universe": provider.universe_config,
        "input_files_verified_after": True,
        "feature_source": "official_panel_with_official_cs_infer_freshness_merge",
        "cs_train_fallback": False,
    }
    _write_json(output / "validation.json", validation)
    _write_json(
        output / "manifest.json",
        {
            "schema": SCHEMA,
            "complete": True,
            "spec_sha256": spec_sha256,
            "model_version": MODEL_VERSION,
            "source_code": code_identity,
            "files": {
                name: _sha(output / name)
                for name in ("audit.json", "ranked.parquet", "validation.json")
            },
        },
    )
    return rankings
