# -*- coding: utf-8 -*-
"""Walk-forward 数据态血缘（git 版本 + 数据水位）。

背景：同一配置在不同日期复跑，中间的数据修复/增量下载/代码提交会让结果
不可比（实测同配置 CAGR 波动可达 4pp，远大于多数配置间差异）。

本模块在每次 walk-forward 运行落盘时采集两类信息：
  1. 代码态：git commit（含工作区是否干净）；
  2. 数据态：关键 raw 数据集最新分区、cs_train 最新分区、dividend 覆盖状态摘要。

完整快照写入 data_state_{wf_run_id}.json（与 summary 同目录），
摘要列并入 summary CSV；对比工具按数据态 ID 分组，跨数据态的指标差异
只允许提示、不允许直接比较。
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

# 数据态快照覆盖的关键 raw 数据集（离线训练主链路依赖，目录列举成本可忽略）
_DATA_STATE_RAW_DATASETS = (
    "daily",
    "adj_factor",
    "daily_basic",
    "moneyflow",
    "stk_limit",
    "suspend",
    "stock_st",
    "margin_detail",
)

_DATA_STATE_FILE_TEMPLATE = "data_state_{wf_run_id}.json"

# 计算数据态 ID 时排除的易变字段（运行标识与采集时间不参与指纹）
_STATE_ID_EXCLUDED_KEYS = ("collected_at", "wf_run_id", "batch_run_id")

# 项目根目录（src/lazybull/ml/walk_forward/data_state.py -> 项目根）
_PROJECT_ROOT = Path(__file__).resolve().parents[4]

_GIT_TIMEOUT_SECONDS = 10


def _git_state() -> Dict[str, Any]:
    """读取当前 git commit 与工作区脏标记；git 不可用时降级为 None。"""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
        if commit.returncode != 0:
            return {"git_commit": None, "git_dirty": None}
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
        if status.returncode != 0:
            return {"git_commit": commit.stdout.strip(), "git_dirty": None}
        return {
            "git_commit": commit.stdout.strip(),
            "git_dirty": bool(status.stdout.strip()),
        }
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug(f"采集 git 数据态失败（不影响训练）: {exc}")
        return {"git_commit": None, "git_dirty": None}


def _latest_partition_date(directory: Path) -> Optional[str]:
    """返回目录内最新分区日期（YYYY-MM-DD 文件名），无分区时返回 None。"""
    if not directory.exists():
        return None
    dates = [
        path.stem
        for path in directory.iterdir()
        if path.suffix in (".parquet", ".csv")
        and len(path.stem) == 10
        and path.stem[4] == "-"
        and path.stem[7] == "-"
    ]
    return max(dates) if dates else None


def collect_data_state(
    data_root: Optional[str],
    wf_run_id: Optional[str] = None,
    batch_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """采集当前代码态与数据态快照。任何单一信息源失败都不阻断训练。"""
    from src.lazybull.data.storage import Storage

    state: Dict[str, Any] = {
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "wf_run_id": wf_run_id,
        "batch_run_id": batch_run_id,
    }
    state.update(_git_state())

    raw_latest: Dict[str, Optional[str]] = {}
    features_latest: Optional[str] = None
    dividend_coverage: Optional[Dict[str, int]] = None
    try:
        storage = Storage(root_path=str(data_root)) if data_root else Storage()
        for name in _DATA_STATE_RAW_DATASETS:
            raw_latest[name] = _latest_partition_date(storage.raw_path / name)
        features_latest = _latest_partition_date(storage.features_path / "cs_train")
        # 延迟导入：dividend_raw 依赖 tushare 客户端，避免拖重模块加载
        from src.lazybull.data.dividend_raw import summarize_dividend_coverage

        dividend_coverage = summarize_dividend_coverage(storage)
    except Exception as exc:
        logger.warning(f"采集数据水位失败（不影响训练）: {exc}")
    state["raw_latest_partitions"] = raw_latest
    state["features_cs_train_latest"] = features_latest
    state["dividend_coverage"] = dividend_coverage
    return state


def compute_data_state_id(state: Dict[str, Any]) -> str:
    """对数据态快照计算稳定短指纹（排除运行标识与采集时间）。"""
    payload = {k: v for k, v in state.items() if k not in _STATE_ID_EXCLUDED_KEYS}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()[:8]


def data_state_summary_columns(state: Dict[str, Any]) -> Dict[str, Any]:
    """将数据态快照压缩为 summary CSV 摘要列（完整水位见 data_state JSON）。"""
    dividend = state.get("dividend_coverage")
    if dividend:
        dividend_text = ",".join(f"{k}={dividend[k]}" for k in sorted(dividend))
    else:
        dividend_text = None
    daily_latest = (state.get("raw_latest_partitions") or {}).get("daily")
    return {
        "data_state_id": compute_data_state_id(state),
        "git_commit": state.get("git_commit"),
        "git_dirty": state.get("git_dirty"),
        "data_daily_latest": daily_latest,
        "data_cs_train_latest": state.get("features_cs_train_latest"),
        "data_dividend_coverage": dividend_text,
    }


def write_data_state_file(output_dir: Path, state: Dict[str, Any]) -> Optional[Path]:
    """将数据态快照写入 summary 同目录，便于人工核对完整水位。"""
    wf_run_id = state.get("wf_run_id") or "unknown"
    path = Path(output_dir) / _DATA_STATE_FILE_TEMPLATE.format(wf_run_id=wf_run_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(state, file, ensure_ascii=False, indent=2, default=str)
        return path
    except OSError as exc:
        logger.warning(f"写入数据态血缘文件失败（不影响训练结果）: {path}，{exc}")
        return None
