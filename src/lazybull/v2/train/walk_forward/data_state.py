# -*- coding: utf-8 -*-
"""Walk-forward 数据态血缘（git 版本 + 数据水位 + P3-T3 增强输入身份）。

P3-T3 搬运改写自 src.lazybull.ml.walk_forward.data_state（2026-10-22）：

- **旧口径逐字保留**（collect_data_state / compute_data_state_id /
  data_state_summary_columns / write_data_state_file 及私有件）——
  replay_b0 等 P2a/P2b 工具改指后行为不变；唯一改指点 = ``_PROJECT_ROOT``
  深度修正（parents[4] → parents[5]，包路径多一级，登记）；
- **§5 改造 additive 新增**（旧函数不动）：
  ① 实际消费文件身份——store 侧只消费读取回执（``StoreReadReceipt``，R7 单一来源，
  不复制冷热区解析规则）；非 store 输入（clean / trade_cal / stock_basic /
  cs_train_verify 分区）经 ``store_identity.file_identity`` 现算；
  ② 运行前采集、运行后复核（``verify_consumed_file_identities``，漂移 ⇒
  ``InputIdentityDriftError``；含「manifest 登记值不变而物理文件变更」拒绝——
  物理核验为准，登记值仅核验通过后摘要展示）；
  ③ 环境身份（``collect_environment_identity``：复用 store_identity 四库版本 +
  xgboost 版本 + 线程环境变量，§5-3）；
  ④ **新旧批次 data_state_id 不可比**（§5-4）：增强口径
  （``compute_enhanced_data_state_id``，sha256 前 8 位，输入身份 + 环境身份参与）
  与旧口径（md5 前 8 位）并存；``scripts/compare/data_state.py`` 按 id 分组告警的
  机制不变（新旧混入天然分组告警，契约保留）；P3 回放基线改用显式文件哈希清单
  （receipt + 现算清单落盘）。
- 产物形态不变：``data_state_{wf_run_id}.json`` 落 summary 同目录，旧键集合不变，
  增强身份段 additive 附加（``input_files`` / ``environment_identity`` 键）。

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
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger

from src.lazybull.v2.store.store_identity import (
    InputIdentityDriftError,
    environment_identity,
    file_identity,
    verify_file_identities,
)

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

# 计算数据态 ID 时排除的易变字段（运行标识、采集时间与指纹本身不参与指纹）
_STATE_ID_EXCLUDED_KEYS = ("collected_at", "wf_run_id", "batch_run_id", "data_state_id")

# 项目根目录（src/lazybull/v2/train/walk_forward/data_state.py -> 项目根；
# P3-T3 改指点：包路径较旧件多一级，parents[4] → parents[5]）
_PROJECT_ROOT = Path(__file__).resolve().parents[5]

_GIT_TIMEOUT_SECONDS = 10

#: 环境身份采集的线程相关环境变量（§5-3）
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


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


def _is_partition_stem(stem: str) -> bool:
    """识别分区文件名：raw/clean 层为 YYYY-MM-DD（10 位带横线），features 层为 YYYYMMDD（8 位纯数字）。"""
    if len(stem) == 10 and stem[4] == "-" and stem[7] == "-":
        return stem[:4].isdigit() and stem[5:7].isdigit() and stem[8:].isdigit()
    if len(stem) == 8 and stem.isdigit():
        return True
    return False


def _latest_partition_date(directory: Path) -> Optional[str]:
    """返回目录内最新分区日期（按磁盘上的原始文件名记录），无分区时返回 None。"""
    if not directory.exists():
        return None
    dates = [
        path.stem
        for path in directory.iterdir()
        if path.suffix in (".parquet", ".csv") and _is_partition_stem(path.stem)
    ]
    # 同一目录内分区格式一致（raw 带横线 / features 纯数字），字典序即为时间序
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
    """将数据态快照写入 summary 同目录，便于人工核对完整水位。

    落盘内容额外附带 data_state_id，使 JSON 自描述；指纹计算仍基于原始 state
    （排除运行标识与采集时间），附带字段不影响指纹。
    """
    wf_run_id = state.get("wf_run_id") or "unknown"
    path = Path(output_dir) / _DATA_STATE_FILE_TEMPLATE.format(wf_run_id=wf_run_id)
    payload = {**state, "data_state_id": compute_data_state_id(state)}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, default=str)
        return path
    except OSError as exc:
        logger.warning(f"写入数据态血缘文件失败（不影响训练结果）: {path}，{exc}")
        return None


# ---------------------------------------------------------------------------
# §5 增强输入身份（P3-T3 additive 新增；旧口径函数逐字保留于上方）
# ---------------------------------------------------------------------------


def collect_environment_identity() -> Dict[str, Any]:
    """环境身份（§5-3）：四库版本（复用 store_identity）+ xgboost 版本 + 线程设置。"""
    env: Dict[str, Any] = dict(environment_identity())
    try:
        import xgboost

        env["xgboost"] = xgboost.__version__
    except ImportError:
        env["xgboost"] = "unavailable"
    for var in _THREAD_ENV_VARS:
        env[f"env.{var}"] = os.environ.get(var, "")
    return env


def collect_consumed_file_identities(
    receipts: Sequence[Any],
    extra_files: Sequence[Path | str] = (),
    *,
    base_dir: Path | str | None = None,
) -> List[Dict[str, Any]]:
    """汇总实际消费文件身份（§5-1；R7：store 侧只消费 receipt，不复制解析规则）。

    Args:
        receipts: ``StoreReadReceipt`` 序列（其 files 已是以 store root 为基准的
            ``file_identity`` 记录）；多个 receipt 的重叠文件按路径去重——同路径
            同 sha256 保留一条，同路径不同 sha256 ⇒ 两次读取间漂移，拒绝；
        extra_files: 非 store 输入文件路径（clean / trade_cal / stock_basic /
            cs_train_verify 分区等），经 ``file_identity`` 现算；
        base_dir: extra_files 显示路径基准（与 receipt 的 store root 一致口径）。

    Returns:
        按显示路径排序的身份清单（``[{"path", "sha256", "size"}, ...]``）。
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for receipt in receipts:
        if receipt is None:
            continue
        for record in receipt.files:
            path = str(record["path"])
            existing = merged.get(path)
            if existing is not None:
                if (existing["sha256"], existing["size"]) != (record["sha256"], record["size"]):
                    raise InputIdentityDriftError(
                        f"同一文件在多次读取间漂移: {path}（ receipt 间 sha256 不一致）"
                    )
                continue
            merged[path] = dict(record)
    for path in extra_files:
        record = file_identity(path, base_dir=base_dir)
        display = record["path"]
        existing = merged.get(display)
        if existing is not None:
            if (existing["sha256"], existing["size"]) != (record["sha256"], record["size"]):
                raise InputIdentityDriftError(
                    f"非 store 输入与 receipt 记录冲突: {display}（sha256 不一致）"
                )
            continue
        merged[display] = record
    return [merged[path] for path in sorted(merged)]


def verify_consumed_file_identities(
    records: Sequence[Dict[str, Any]],
    *,
    base_dir: Path | str | None = None,
) -> None:
    """运行后复核实际消费文件（§5-1：任何漂移 ⇒ 失败；物理核验为准）。

    覆盖「manifest 登记值不变而物理文件变更」用例——核验以文件物理内容
    （sha256 + size）重算为准，不采信任何登记值。
    """
    mismatches = verify_file_identities(records, base_dir=base_dir)
    if mismatches:
        raise InputIdentityDriftError(
            f"运行后复核实际消费文件漂移: {mismatches[:5]}"
            f"{'...' if len(mismatches) > 5 else ''}（共 {len(mismatches)} 个）"
        )


def compute_enhanced_data_state_id(
    state: Dict[str, Any],
    consumed_files: Sequence[Dict[str, Any]],
    environment: Dict[str, Any],
    extra_sections: Optional[Dict[str, Any]] = None,
) -> str:
    """增强口径数据态 ID（§5：实际消费文件身份 + 环境身份参与，sha256 前 8 位）。

    **与旧口径（``compute_data_state_id``，md5 前 8 位）新旧批次不可比**（§5-4 登记）。
    复核处置（reviewer1 复核 P2-1）：``extra_sections``（水位/manifest 指纹/
    代码态摘要）**纳入** canonical payload——同文件集但不同 manifest 指纹/代码态
     ⇒ 不同 ID。
    """
    payload = {
        "base_state": {k: v for k, v in state.items() if k not in _STATE_ID_EXCLUDED_KEYS},
        "input_files": [{"path": r["path"], "sha256": r["sha256"]} for r in consumed_files],
        "environment_identity": dict(environment),
        "extra_sections": dict(extra_sections or {}),
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def write_enhanced_data_state_file(
    output_dir: Path,
    state: Dict[str, Any],
    consumed_files: Sequence[Dict[str, Any]],
    environment: Dict[str, Any],
    extra_sections: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """增强口径落盘（产物形态不变：同文件名模板；旧键集合不变，增强段 additive）。

    附加键：``input_files``（实际消费文件身份清单）/ ``environment_identity`` /
    ``data_state_id`` = 增强口径指纹（``compute_enhanced_data_state_id``）;
    ``extra_sections``（P3-T3 评审处置 reviewer1-P3-6:水位三对象 / manifest 指纹 /
    代码态摘要等附加段,键名不得与既有键冲突)。
    """
    wf_run_id = state.get("wf_run_id") or "unknown"
    path = Path(output_dir) / _DATA_STATE_FILE_TEMPLATE.format(wf_run_id=wf_run_id)
    payload = {
        **state,
        "input_files": list(consumed_files),
        "environment_identity": dict(environment),
    }
    for key, value in (extra_sections or {}).items():
        if key in payload:
            raise ValueError(f"extra_sections 键与既有键冲突: {key!r}")
        payload[key] = value
    # 复核处置（reviewer1 复核 P2-1）:ID 计算含 extras(extras 先入、ID 后算)
    payload["data_state_id"] = compute_enhanced_data_state_id(
        state, consumed_files, environment, extra_sections
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, default=str)
        return path
    except OSError as exc:
        logger.warning(f"写入数据态血缘文件失败（不影响训练结果）: {path}，{exc}")
        return None
