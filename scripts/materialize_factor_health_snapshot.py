#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""物化"体检快照"并驱动因子体检/诊断（运行时派生列的离线扫描入口）。

**为什么需要它**：股东增减持（stk_holdertrade）/ 股票回购（repurchase）等家族列按契约
**运行时派生**，不写入生产 `features/cs_train`；而 `analyze_factor_health.py` /
`analyze_factor_diagnosis.py` 只读分区。本脚本把「特征清单 + 运行时派生列」物化成
**独立临时数据根**（不触碰生产数据），再把工具的 `data.root` 指向该根运行，从而复用同一套体检/诊断口径。

做三件事：
1. 物化：按采样间隔取 `--start~--end` 的分区，只保留体检实际读取的列
   （特征清单 + 标签 + 市场波动 + 规模代理 + 键列）+ 派生的运行时列，写入 `<out-root>/features/cs_train`；
2. 搭根：`clean` / `raw` / `models` 以 Windows 目录联接指向生产（只读复用），产出 `feature_file.json`；
3. 运行：在同一进程内把 `data.root` 指向临时根，依次跑体检与诊断（产物落 `<out-root>/reports/`）。

示例：
    python scripts/materialize_factor_health_snapshot.py \
        --start 20200101 --end 20260702 --every 3 \
        --with-holdertrade --out-root temp/ht_health_root_20260917

    python scripts/materialize_factor_health_snapshot.py \
        --start 20200101 --end 20260702 --every 3 \
        --with-repurchase --out-root temp/rp_health_root_20260918
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger

from scripts.factor_health.analysis import resolve_latest_feature_file
from src.lazybull.common.config import (
    get_config,
    get_data_root,
    get_logs_dir,
    get_stock_selection_models_root,
)
from src.lazybull.common.logger import setup_logger
from src.lazybull.data import DataLoader, Storage

#: 体检/诊断读取的支撑列（见 scripts/factor_health/scan.py 与 constants.py）
SUPPORT_COLS = ["trade_date", "ts_code", "neu_y_ret_20", "mkt_vol_20", "zscore_size"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="物化体检快照（含运行时派生列）并驱动体检/诊断")
    parser.add_argument("--start", default="20200101", help="起始日期 YYYYMMDD")
    parser.add_argument("--end", default="99999999", help="结束日期 YYYYMMDD")
    parser.add_argument("--every", type=int, default=3, help="分区采样间隔（交易日）")
    parser.add_argument("--out-root", required=True, help="临时数据根目录（禁止指向生产 data/）")
    parser.add_argument(
        "--feature-file",
        default="",
        help="特征清单 JSON（默认取最新注册模型）；运行时列会追加进清单",
    )
    parser.add_argument(
        "--with-holdertrade",
        action="store_true",
        help="派生股东增减持列（stk_holdertrade，运行时派生家族）",
    )
    parser.add_argument(
        "--with-repurchase",
        action="store_true",
        help="派生股票回购列（repurchase，运行时派生家族）",
    )
    parser.add_argument("--skip-health", action="store_true", help="只物化，不跑体检")
    parser.add_argument(
        "--skip-usage",
        action="store_true",
        help="体检跳过模型使用度统计（模型落盘格式与工具不匹配时使用，使用度需另行统计）",
    )
    parser.add_argument("--skip-diagnosis", action="store_true", help="只物化/体检，不跑诊断")
    parser.add_argument(
        "--model-versions",
        default="",
        help="使用度统计的模型版本（如 24008-24021）；缺省用工具默认",
    )
    parser.add_argument(
        "--usage-model-count", type=int, default=0, help="使用度统计版本数（0=工具默认）"
    )
    parser.add_argument(
        "--health-out", default="", help="体检产物目录（默认 <out-root>/reports/...）"
    )
    parser.add_argument(
        "--diagnosis-out", default="", help="诊断产物目录（默认 <out-root>/reports/...）"
    )
    return parser


def _resolve_feature_file(raw: str) -> Path:
    if raw:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"特征清单不存在: {path}")
        return path
    model_dir = Path(get_stock_selection_models_root())
    return resolve_latest_feature_file(model_dir)


def _ensure_link(link: Path, target: Path) -> None:
    if link.exists():
        return
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True
        )
    else:
        link.symlink_to(target)
    logger.info(f"目录联接: {link} -> {target}")


def _runtime_family_specs(enabled: Dict[str, bool]) -> Dict[str, Dict[str, object]]:
    """构造运行时派生家族规格（列清单 / raw 加载 / 查询表构建 / 就地派生 / 支撑列）。

    新增运行时家族时在此登记即可（脚本主体不再出现家族分支）。
    """
    specs: Dict[str, Dict[str, object]] = {}
    if enabled.get("holdertrade"):
        from src.lazybull.factors.holdertrade import (
            available_holdertrade_columns,
            build_holdertrade_lookup_by_date,
            derive_holdertrade_columns,
        )

        specs["holdertrade"] = {
            "columns": available_holdertrade_columns(),
            "load_raw": lambda loader: loader.load_stk_holdertrade(),
            "build_lookup": build_holdertrade_lookup_by_date,
            "derive": derive_holdertrade_columns,
            "extra_cols": [],
            "download_name": "stk_holdertrade",
        }
    if enabled.get("repurchase"):
        from src.lazybull.factors.repurchase import (
            available_repurchase_columns,
            build_repurchase_lookup_by_date,
            derive_repurchase_columns,
        )

        specs["repurchase"] = {
            "columns": available_repurchase_columns(),
            "load_raw": lambda loader: loader.load_repurchase(),
            "build_lookup": build_repurchase_lookup_by_date,
            "derive": derive_repurchase_columns,
            # 派生需要当日价格与流通市值（VWAP = amount × 10 ÷ vol）；这是**读取支撑列**，
            # 派生后不写入快照（体检工具不读它们），见 materialize 的 frame[available] 裁剪
            "extra_cols": ["circ_mv", "amount", "vol"],
            "download_name": "repurchase",
        }
    return specs


def materialize(args: argparse.Namespace) -> Dict[str, object]:
    """物化分区与运行时列，返回元信息。"""
    storage = Storage()
    src_dir = Path(storage.root_path) / "features" / "cs_train"
    if not src_dir.exists():
        raise FileNotFoundError(f"生产特征分区不存在: {src_dir}")

    out_root = Path(args.out_root)
    cs_dir = out_root / "features" / "cs_train"
    cs_dir.mkdir(parents=True, exist_ok=True)

    feature_file = _resolve_feature_file(args.feature_file)
    features: List[str] = list(json.loads(feature_file.read_text(encoding="utf-8")))
    specs = _runtime_family_specs(
        {
            "holdertrade": bool(args.with_holdertrade),
            "repurchase": bool(args.with_repurchase),
        }
    )
    runtime_cols: List[str] = []
    for spec in specs.values():
        for col in spec["columns"]:  # type: ignore[union-attr]
            if col not in features:
                features.append(col)
                runtime_cols.append(col)
    # 运行时派生列不在分区中，扫描前必须从清单剔除（否则 read_parquet 缺列报错）
    features = [c for c in features if c in _partition_columns(src_dir, args.start, args.end)]
    features += runtime_cols

    (out_root / "feature_file.json").write_text(
        json.dumps(features, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    files = sorted(p for p in src_dir.glob("*.parquet") if args.start <= p.stem <= args.end)
    files = files[:: max(int(args.every), 1)]
    if not files:
        raise FileNotFoundError(f"{src_dir} 在 {args.start}~{args.end} 无分区")
    schema = _schema_names(files)
    base_cols = [c for c in dict.fromkeys(SUPPORT_COLS + features) if c in schema]
    # 运行时派生列**不在分区 schema 中**（由派生写入帧）：不进 read_cols，但必须进快照列集
    write_cols = base_cols + [c for c in runtime_cols if c not in base_cols]
    read_cols = list(base_cols)
    for spec in specs.values():
        for col in spec["extra_cols"]:  # type: ignore[union-attr]
            if col in schema and col not in read_cols:
                read_cols.append(col)
    logger.info(
        f"物化 {len(files)} 个分区（{files[0].stem}~{files[-1].stem}），"
        f"列 {len(write_cols)}（运行时列 {len(runtime_cols)}，家族 {sorted(specs)}）"
    )

    lookups: Dict[str, object] = {}
    if specs:
        loader = DataLoader(storage)
        dates = [p.stem for p in files]
        for name, spec in specs.items():
            raw = spec["load_raw"](loader)  # type: ignore[operator]
            if raw is None or len(raw) == 0:
                raise ValueError(
                    f"启用运行时家族 {name} 但缺少对应 raw 数据"
                    f"（{spec.get('download_name', name)}: 请先运行 "
                    f"python scripts/download_raw.py --download {spec.get('download_name', name)}）"
                )
            lookups[name] = spec["build_lookup"](raw, dates)  # type: ignore[operator]
            logger.info(f"运行时家族 {name}: 查询表 {len(lookups[name])} 个交易日")  # type: ignore[arg-type]

    started = time.perf_counter()
    total_bytes = 0
    for index, path in enumerate(files, start=1):
        frame = pd.read_parquet(path, columns=read_cols)
        for name, spec in specs.items():
            cols = spec["columns"]  # type: ignore[assignment]
            spec["derive"](frame, lookups[name], wanted=cols)  # type: ignore[operator]
            blank = [c for c in cols if frame[c].isna().all()]  # type: ignore[union-attr]
            if blank:
                raise RuntimeError(f"{path.stem} {name} 运行时列全空: {blank}")
        if len(read_cols) != len(write_cols):
            frame = frame[write_cols]  # 派生支撑列不落盘
        target = cs_dir / f"{path.stem}.parquet"
        frame.to_parquet(target, index=False)
        total_bytes += target.stat().st_size
        if index % 100 == 0 or index == len(files):
            logger.info(f"  物化进度 {index}/{len(files)}，用时 {time.perf_counter()-started:.0f}s")
    logger.info(f"物化完成: {len(files)} 分区, {total_bytes/1e9:.2f} GB")
    return {"files": len(files), "bytes": total_bytes, "features": features}


def _schema_names(files: List[Path]) -> set:
    import pyarrow.parquet as pq

    names = set()
    for path in (files[0], files[-1]):
        names.update(pq.ParquetFile(path).schema.names)
    return names


def _partition_columns(src_dir: Path, start: str, end: str) -> set:
    files = sorted(p for p in src_dir.glob("*.parquet") if start <= p.stem <= end)
    if not files:
        return set()
    return _schema_names(files)


def main() -> None:
    args = build_parser().parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logger(
        log_level="INFO", log_file=str(Path(get_logs_dir()) / f"health_snapshot_{timestamp}.log")
    )

    out_root = Path(args.out_root).resolve()
    production_root = Path(get_data_root()).resolve()
    if production_root in out_root.parents or out_root == production_root:
        raise ValueError(f"--out-root 不得位于生产数据根内: {out_root}")

    meta = materialize(args)

    _ensure_link(out_root / "clean", production_root / "clean")
    _ensure_link(out_root / "raw", production_root / "raw")
    _ensure_link(out_root / "models", production_root / "models")
    (out_root / "reports").mkdir(parents=True, exist_ok=True)
    for sub in ("factor_health", "factor_diagnosis"):
        (out_root / "reports" / sub).mkdir(parents=True, exist_ok=True)

    feature_file = out_root / "feature_file.json"
    health_out = (
        Path(args.health_out) if args.health_out else out_root / "reports" / "factor_health"
    )
    diagnosis_out = (
        Path(args.diagnosis_out)
        if args.diagnosis_out
        else out_root / "reports" / "factor_diagnosis"
    )

    # 工具通过 data.root 解析分区与股票池；同进程改写全局配置即可，无需改动生产配置
    get_config().set("data.root", str(out_root))

    if not args.skip_health:
        from scripts.analyze_factor_health import main as health_main

        argv = [
            "analyze_factor_health.py",
            "--start",
            args.start,
            "--end",
            args.end,
            "--every",
            "1",  # 快照已是采样后的子集
            "--feature-file",
            str(feature_file),
            "--out",
            str(health_out),
        ]
        if args.model_versions:
            argv += ["--model-versions", args.model_versions]
        if args.skip_usage:
            argv += ["--skip-models"]
        old_argv = sys.argv
        sys.argv = argv
        try:
            health_main()
        finally:
            sys.argv = old_argv
        logger.info(f"体检产物: {health_out}")

    if not args.skip_diagnosis:
        from scripts.analyze_factor_diagnosis import main as diagnosis_main

        argv = [
            "analyze_factor_diagnosis.py",
            "--health-dir",
            str(health_out),
            "--start",
            args.start,
            "--end",
            args.end,
            "--every",
            "1",
            "--out",
            str(diagnosis_out),
        ]
        if args.usage_model_count:
            argv += ["--usage-model-count", str(args.usage_model_count)]
        old_argv = sys.argv
        sys.argv = argv
        try:
            diagnosis_main()
        finally:
            sys.argv = old_argv
        logger.info(f"诊断产物: {diagnosis_out}")

    logger.info(
        f"快照数据根: {out_root}（{meta['files']} 分区，{meta['bytes']/1e9:.2f} GB，"
        f"特征 {len(meta['features'])} 列）"
    )


if __name__ == "__main__":
    main()
