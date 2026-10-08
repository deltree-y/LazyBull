# -*- coding: utf-8 -*-
"""验收门 1（数据身份门）：重放输入完整读取清单的哈希采集与校验。

规划 §5 门 1（P2A2-01 + V2R-03）：重放输入内容身份 = **完整读取清单**
（不止特征与模型）——cs_train 分区 + clean/daily 行情 + 交易日历 +
stock_basic + 停牌输入 + **有效配置快照（必填，R2-T7-02）** + 折模型文件
（v24008~v24021）。采集哈希清单 fixture，开跑前逐字节校验；同水位修订 ⇒
拒绝开跑。配置快照 = 驱动 config + 成本设置（``get_cost_settings``，与引擎
``CostModel`` 同一读取源），由 ``replay_b0.build_manifest_config_snapshot``
同源组装；缺失或变化均计入校验差异（禁止打印提示放行）。

反例语义（门 1 点名三条，测试锚定）：
- 仅代码态字段变化 ⇒ manifest 不含代码态 ⇒ 通过；
- 水位不变但清单内文件内容改变 ⇒ 拒绝；
- 特征/模型不变但行情/日历/基础表被改 ⇒ 拒绝。

窗口内容抽查：``spot_check_frozen_trades`` 从 B0 冻结 trades CSV 抽样，
按 (ts_code, trade_date) 查当前 clean/daily，断言成交价 ∈ {open, close,
open_adj, close_adj}（容差 1e-6）。

**机器时间纪律**：采集/校验读取真实生产数据，属机器时间任务，不进 pytest
（pytest 以 tmp_path 合成文件集驱动 collect/validate 纯逻辑）。

用法：
    python scripts/v2_p2a/input_manifest.py collect --data-root ./data \
        --out logs/b0_input_manifest_20261008.json --with-driver-config
    python scripts/v2_p2a/input_manifest.py validate --data-root ./data \
        --manifest logs/b0_input_manifest_20261008.json --with-driver-config
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger

# 脚本直接运行时把仓库根插入 sys.path（--with-driver-config 需导入 replay_b0；
# 同 scripts/v2_p1、v2_p5a1 既有先例）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

#: B0 折模型版本区间（skip-training 复用折模型，baseline_freeze §2）
MODEL_VERSION_RANGE = range(24008, 24022)  # v24008 ~ v24021

#: 清单覆盖窗口（B0 折测试窗并集，YYYYMMDD；采集时记录，校验时按同一窗口重算清单）
DEFAULT_WINDOW = ("20181126", "20260702")

#: 有效配置快照在清单中的伪路径键（canonical JSON 哈希，不占文件系统）
CONFIG_SNAPSHOT_KEY = "<config>/driver_config.json"

_CHUNKSIZE = 1024 * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_CHUNKSIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_config(config: Dict[str, object]) -> str:
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _iter_partition_files(directory: Path, date_min: str, date_max: str) -> List[Path]:
    """目录内窗口命中的分区文件（支持 YYYYMMDD 与 YYYY-MM-DD 两种 stem）。"""
    if not directory.exists():
        return []
    lo = date_min.replace("-", "")
    hi = date_max.replace("-", "")
    hits = []
    for path in sorted(directory.iterdir()):
        if path.suffix != ".parquet":
            continue
        digits = path.stem.replace("-", "")
        if len(digits) == 8 and digits.isdigit() and lo <= digits <= hi:
            hits.append(path)
    return hits


def list_manifest_files(
    data_root: Path,
    window: Tuple[str, str] = DEFAULT_WINDOW,
    model_versions: Sequence[int] = tuple(MODEL_VERSION_RANGE),
) -> List[Path]:
    """重放完整读取清单（相对 data_root 的文件集；缺失文件也计入——缺失即拒绝）。"""
    data_root = Path(data_root)
    date_min, date_max = window
    files: List[Path] = []
    # cs_train 分区（B0 折测试窗并集内实际存在的分区）
    files += _iter_partition_files(data_root / "features" / "cs_train", date_min, date_max)
    # clean/daily 行情分区（同窗口）
    files += _iter_partition_files(data_root / "clean" / "daily", date_min, date_max)
    # 交易日历 + stock_basic（clean 优先、raw 兜底——与重放读取顺序一致）
    for rel in (
        Path("clean") / "trade_cal.parquet",
        Path("raw") / "trade_cal.parquet",
        Path("clean") / "stock_basic.parquet",
        Path("raw") / "stock_basic.parquet",
    ):
        p = data_root / rel
        if p.exists():
            files.append(p)
    # 停牌输入（raw/suspend 分区，同窗口）
    files += _iter_partition_files(data_root / "raw" / "suspend", date_min, date_max)
    # 折模型文件（model.joblib/metadata/features.json）
    models_root = data_root / "models" / "stock_selection"
    for version in model_versions:
        for suffix in ("model.joblib", "metadata.json", "features.json"):
            p = models_root / f"v{version}_{suffix}"
            if p.exists():
                files.append(p)
    return files


def collect_manifest(
    data_root: Path,
    out_path: Path,
    *,
    config: Optional[Dict[str, object]] = None,
    window: Tuple[str, str] = DEFAULT_WINDOW,
    model_versions: Sequence[int] = tuple(MODEL_VERSION_RANGE),
) -> Dict[str, object]:
    """采集哈希清单并落盘（相对路径 → {sha256, size}）。"""
    data_root = Path(data_root)
    files = list_manifest_files(data_root, window, model_versions)
    entries: Dict[str, Dict[str, object]] = {}
    for path in files:
        rel = path.relative_to(data_root).as_posix()
        entries[rel] = {"sha256": _sha256_file(path), "size": path.stat().st_size}
    if config is not None:
        canonical = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
        entries[CONFIG_SNAPSHOT_KEY] = {
            "sha256": _sha256_config(config),
            "size": len(canonical.encode("utf-8")),
            "content": canonical,  # 内嵌 canonical 内容供复核（校验时逐键定位差异）
        }
    manifest: Dict[str, object] = {
        "schema_version": 1,
        "created_at": pd.Timestamp.now().isoformat(),
        "window": list(window),
        "model_versions": list(model_versions),
        "files": entries,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"输入哈希清单已采集: {out_path}（{len(entries)} 项）")
    return manifest


def _diff_config_section(
    prefix: str,
    recorded_sec: Dict[str, object],
    actual_sec: Dict[str, object],
    diffs: List[str],
) -> None:
    """单层配置段逐键比对（供嵌套段如 cost_settings 复用；结果追加进 diffs）。"""
    for key in sorted(set(recorded_sec) | set(actual_sec)):
        if key not in recorded_sec:
            diffs.append(f"  配置多键: {prefix}{key}")
        elif key not in actual_sec:
            diffs.append(f"  配置缺键: {prefix}{key}")
        elif recorded_sec[key] != actual_sec[key]:
            diffs.append(
                f"  配置错值: {prefix}{key}: 登记={recorded_sec[key]!r}"
                f" vs 重算={actual_sec[key]!r}"
            )


def _diff_config_key(
    key: str,
    recorded_config: Dict[str, object],
    config: Dict[str, object],
    diffs: List[str],
) -> None:
    """单键比对（嵌套段委派 _diff_config_section；结果追加进 diffs）。"""
    recorded_v = recorded_config.get(key)
    actual_v = config.get(key)
    if isinstance(recorded_v, dict) and isinstance(actual_v, dict):
        _diff_config_section(f"{key}.", recorded_v, actual_v, diffs)
    elif key not in recorded_config:
        diffs.append(f"  配置多键: {key}")
    elif key not in config:
        diffs.append(f"  配置缺键: {key}")
    elif recorded_v != actual_v:
        diffs.append(f"  配置错值: {key}: 登记={recorded_v!r} vs 重算={actual_v!r}")


def _diff_config_snapshot(
    recorded_entry: Dict[str, object], config: Dict[str, object]
) -> List[str]:
    """配置快照逐键比对（哈希不等时按内嵌 canonical 内容定位差异键）。"""
    actual_hash = _sha256_config(config)
    if actual_hash == recorded_entry["sha256"]:
        return []
    diffs = [f"配置快照哈希不符: 登记 {recorded_entry['sha256']} vs 重算 {actual_hash}"]
    recorded_content = recorded_entry.get("content")
    if not recorded_content:
        return diffs
    recorded_config = json.loads(str(recorded_content))
    for key in sorted(set(recorded_config) | set(config)):
        _diff_config_key(key, recorded_config, config, diffs)
    return diffs


def validate_manifest(
    data_root: Path,
    manifest_path: Path,
    *,
    config: Optional[Dict[str, object]] = None,
) -> List[str]:
    """重算比对清单；返回差异清单（空 = 通过）。

    差异三类：缺失（清单有而磁盘无）/ 多出（磁盘有而清单无，同口径清单重算）/
    哈希不符（同水位修订）。调用方按非空 ⇒ 非零退出处理。

    有效配置快照（R2-T7-02 修复后口径）：**必填**——清单缺快照键即差异；
    快照存在时调用方必须给出来源一致的 config 重算比对（缺失/变化均计入差异），
    不得仅打印提示放行。
    """
    data_root = Path(data_root)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    recorded: Dict[str, Dict[str, object]] = manifest["files"]
    window = tuple(manifest.get("window") or DEFAULT_WINDOW)
    model_versions = tuple(manifest.get("model_versions") or list(MODEL_VERSION_RANGE))

    diffs: List[str] = []
    # 当前同口径清单（缺/多判定基准）
    current_files = {
        p.relative_to(data_root).as_posix(): p
        for p in list_manifest_files(data_root, window, model_versions)  # type: ignore[arg-type]
    }
    recorded_files = {k for k in recorded if k != CONFIG_SNAPSHOT_KEY}
    for rel in sorted(recorded_files - set(current_files)):
        diffs.append(f"缺失: {rel}")
    for rel in sorted(set(current_files) - recorded_files):
        diffs.append(f"多出: {rel}")
    for rel in sorted(recorded_files & set(current_files)):
        actual = _sha256_file(current_files[rel])
        if actual != recorded[rel]["sha256"]:
            diffs.append(f"哈希不符: {rel}")
    # 有效配置快照：必填 + 同源重算比对（R2-T7-02）
    recorded_snapshot = recorded.get(CONFIG_SNAPSHOT_KEY)
    if recorded_snapshot is None:
        diffs.append(f"缺失: 有效配置快照键 {CONFIG_SNAPSHOT_KEY}（R2-T7-02：必填）")
    elif config is None:
        diffs.append("配置快照校验需要来源一致的 config（调用方未提供，禁止放行）")
    else:
        diffs.extend(_diff_config_snapshot(recorded_snapshot, config))
    return diffs


def validate_config_snapshot(config: Dict[str, object], manifest_path: Path) -> Optional[str]:
    """驱动 config 与清单内配置快照比对；一致返回 None，否则返回差异说明。"""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    recorded = manifest["files"].get(CONFIG_SNAPSHOT_KEY)
    if recorded is None:
        return "清单无配置快照键"
    diffs = _diff_config_snapshot(recorded, config)
    return "; ".join(diffs) if diffs else None


def spot_check_frozen_trades(
    frozen_dir: Path,
    loader,
    n: int = 20,
    seed: int = 42,
) -> List[str]:
    """窗口内容抽查：冻结 trades 抽样成交价 vs 当前 clean/daily 交叉验证。

    Args:
        frozen_dir: B0 冻结批次 raw 目录（含 walk_forward_trades_*.csv）
        loader: 具备 ``load_clean_daily(start, end)`` 的读取对象
        n: 抽样行数
        seed: 抽样随机种子（可复现）

    Returns:
        违例清单（空 = 通过）；抽不到样本属异常（返回非空说明）
    """
    frozen_dir = Path(frozen_dir)
    trades_files = sorted(frozen_dir.glob("walk_forward_trades_*.csv"))
    if not trades_files:
        return [f"冻结目录无 trades CSV: {frozen_dir}"]
    frames = [pd.read_csv(p, encoding="utf-8-sig") for p in trades_files]
    trades = pd.concat(frames, ignore_index=True)
    if trades.empty:
        return ["冻结 trades 为空"]
    sample = trades.sample(n=min(n, len(trades)), random_state=seed)

    violations: List[str] = []
    for _, row in sample.iterrows():
        ts_code = str(row["stock"])
        trade_date = str(row["date"]).replace("-", "")[:8]
        price = float(row["price"])
        daily = loader.load_clean_daily(trade_date, trade_date)
        if daily is None or len(daily) == 0:
            violations.append(f"{ts_code}@{trade_date}: clean/daily 分区缺失")
            continue
        day_row = daily[daily["ts_code"] == ts_code]
        if day_row.empty:
            violations.append(f"{ts_code}@{trade_date}: clean/daily 无该股行")
            continue
        candidates = [
            float(day_row.iloc[0][col])
            for col in ("open", "close", "open_adj", "close_adj")
            if col in day_row.columns and pd.notna(day_row.iloc[0][col])
        ]
        if not any(abs(price - c) <= 1e-6 for c in candidates):
            violations.append(
                f"{ts_code}@{trade_date}: 成交价 {price} 不在 {{open, close, open_adj, "
                f"close_adj}}={candidates}（容差 1e-6）"
            )
    return violations


def _driver_config_snapshot(data_root: str) -> Dict[str, object]:
    """门 1 有效配置快照（同源组装：replay_b0 单一来源；延迟导入避免拖重）。"""
    from scripts.v2_p2a.replay_b0 import (
        _load_splits_for_snapshot,
        build_manifest_config_snapshot,
    )

    return build_manifest_config_snapshot(_load_splits_for_snapshot(data_root))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="验收门 1：B0 重放输入哈希清单")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("collect", "validate"):
        p = sub.add_parser(name)
        p.add_argument("--data-root", required=True)
        p.add_argument(
            "--with-driver-config",
            action="store_true",
            help="接入有效配置快照（驱动 config + 成本设置，replay_b0 同源组装；"
            "B0 验收正式口径必须开启）",
        )
        if name == "collect":
            p.add_argument("--out", required=True, help="清单输出路径（建议 logs/ 下）")
        else:
            p.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)

    config = _driver_config_snapshot(args.data_root) if args.with_driver_config else None
    if args.command == "collect":
        collect_manifest(Path(args.data_root), Path(args.out), config=config)
        return 0
    diffs = validate_manifest(Path(args.data_root), Path(args.manifest), config=config)
    if diffs:
        logger.error(f"输入清单校验失败（{len(diffs)} 项差异）:")
        for d in diffs:
            logger.error(f"  {d}")
        return 1
    logger.info("输入清单校验通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
