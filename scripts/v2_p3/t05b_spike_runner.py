"""P3 T0.5b 旧路径确定性 spike 编排器（§7.1-0 / §7.4-2）。

同码、同数据、同配置、同环境**连跑两次**（各 1 折 × 3 种子 42/61/82，首折 split 0），
两侧冻结矩阵预测指纹**逐位一致**才放行 G-code 正式启动；**不通过 ⇒ 停止**
（判据变更须用户批准，不得留到 T5）。

- 隔离运行根：``data/frozen_reference/v2_p3/<run>/run{1,2}/root/``（§13.3.4）；
- 生产配置 = ``batch_walk_forward.ps1`` 现役主链路口径（xgboost / neu_y_ret_20 /
  regression / cs_zscore / 500 树 / 早停 50·rank_ic_daily / rank_weight 100·topk120 /
  seeds 42,61,82 / freshness state_keep_event_decay·120 / 域 main），
  ``--no-deploy-train``（Q10-B：P3 回放显式关闭 deploy 训练）；
  **与批处理生产口径的差异清单（评审处置补登）**：① 折子集
  ``--selected-split-indices 0``；② ``--no-deploy-train``；③ 输出根指隔离根；
  ④ 未传 ``--batch-run-id`` / ``--batch-period-label``（输出追踪参数，不影响数值路径）；
- 生产 ``data/models/stock_selection/`` 守卫：运行前后比对，任一变化即停止并回滚（§7.4-3），
  守卫原始快照落盘 ``before_guard.json``（可独立重放比较）；
- **消费集合前后内容哈希硬门（§13.3.4 唯一硬门）**：运行前枚举实际消费文件
  （cs_train 训练~测试窗分区 + clean/stock_basic + trade_cal + clean/daily 测试窗分区）
  采集物理身份落盘 ``consumption_snapshot.json``，双跑后重算核验，漂移即 FAIL；
- 单折耗时与首折边界日期清单随证据 JSON 落盘（§13.3.2 标定登记）；
  **train 段双口径登记**：外层训练窗（台账 start_date/end_date）与**实际训练子集**
  （训练日志 split.py 切分行，已扣 label_delta 间隔）分列，禁止混用（M1/R2-T05B-02）。

用法（机器任务，预计 >10min，须先获用户授权，§7.4-9）：

    python scripts/v2_p3/t05b_spike_runner.py
    python scripts/v2_p3/t05b_spike_runner.py --run-name t05b_spike_20261020 --skip-existing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
from loguru import logger

# 独立运行时仓库根不入 sys.path，按薄入口惯例显式引导
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.v2_p3.t05b_baselines import (  # noqa: E402
    _acquire_baselines,
    _normalize_partition_day,
)
from scripts.v2_p3.t05b_evidence import (  # noqa: E402
    _build_evidence,
    _finalize_comparison,
    _log_verdict,
)
from scripts.v2_p3.t05b_fingerprint import (  # noqa: E402
    environment_identity,
    load_main_board_codes,
    score_model_fingerprint,
)
from scripts.v2_p3.t05b_isolated_root import (  # noqa: E402
    build_isolated_root,
    verify_consumption_snapshot,
    verify_production_models_guard,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SPIKE_BASE_DIR = Path("data/frozen_reference/v2_p3")


def build_spike_walk_forward_args(
    root: Path,
    summary_csv: Path,
    selected_split_indices: Optional[Sequence[str]] = ("0",),
    split_count: int = 14,
    final_date: str = "20260105",
    test_window_months: int = 6,
) -> List[str]:
    """spike 用 walk-forward 参数（镜像批处理生产口径；单折 + 关闭 deploy）。

    与 ``batch_walk_forward.ps1`` 主链路差异（登记）：仅折子集
    （``--selected-split-indices 0``）、``--no-deploy-train``（Q10-B）、
    输出根指向隔离运行根；训练数值路径参数逐项一致。

    P3 T5 泛化：``selected_split_indices`` 参数化——默认 ``("0",)`` 与 spike 现状
    逐位一致；传 ``None`` 时不带该 flag（正式臂①全 14 折，CLI 留空 = 全折语义）；
    传多值序列 ⇒ 多下标透传（臂折子集续跑）。其余参数镜像不变。

    P3 T5 阶段 A 评审处置（P2-1 ≡ R2-T5A-08）：``split_count`` / ``final_date`` /
    ``test_window_months`` 提升为形参——默认值 14 / "20260105" / 6 保持 spike
    现状逐位不变；臂①调用处自 ``ArmRunContext`` 透传（用户参数与真实命令挂钩，
    不再硬编码静默无效）。
    """
    split_flags: List[str] = []
    if selected_split_indices is not None:
        split_flags = ["--selected-split-indices", *[str(i) for i in selected_split_indices]]
    return [
        "--algorithm",
        "xgboost",
        "--split-count",
        str(int(split_count)),
        "--final-date",
        str(final_date),
        "--train-window-years",
        "6",
        "--test-window-months",
        str(int(test_window_months)),
        "--val-ratio",
        "0.2",
        "--label",
        "neu_y_ret_20",
        "--neutral-label-blend-weight",
        "0",
        "--task",
        "regression",
        "--label-transform",
        "cs_zscore",
        "--objective",
        "mse",
        "--n-estimators",
        "500",
        "--max-depth",
        "5",
        "--num-leaves",
        "63",
        "--learning-rate",
        "0.03",
        "--subsample",
        "0.8",
        "--colsample-bytree",
        "0.3",
        "--min-child-weight",
        "200",
        "--reg-alpha",
        "0.03",
        "--reg-lambda",
        "7",
        "--gamma",
        "0.5",
        "--rank-weight-topk",
        "120",
        "--rank-weight",
        "100",
        "--rank-weight-topk-weight-mode",
        "linear_decay",
        "--data-root",
        str(root),
        "--early-stopping-rounds",
        "50",
        "--early-stopping-metric",
        "rank_ic_daily",
        "--min-best-iteration",
        "30",
        "--time-decay-half-life",
        "0",
        "--freshness-strategy",
        "state_keep_event_decay",
        "--event-freshness-half-life-days",
        "120",
        *split_flags,
        "--enable-fundamental-features",
        "--enable-alt-features",
        "--enable-margin-features",
        "--enable-cyq-features",
        "--enable-fund-features",
        "--enable-express-features",
        "--enable-north-features",
        "--enable-lhb-features",
        "--enable-consensus-features",
        "--enable-enhanced-features",
        "--enable-cashflow-quality-features",
        "--ensemble-seeds",
        "42,61,82",
        "--ensemble-seed-keep-top-ratio",
        "1",
        "--ensemble-seed-keep-min-models",
        "3",
        "--oos-backtest",
        "--oos-backtest-months",
        "0",
        "--bt-top-n",
        "20",
        "--bt-initial-capital",
        "1000000",
        "--bt-sell-timing",
        "open",
        "--bt-min-list-days",
        "365",
        "--stock-domain",
        "main",
        "--bt-rebalance-freq",
        "20",
        "--bt-max-weight-per-stock",
        "0.15",
        "--stagger-tranches",
        "2",
        "--position-sizing",
        "half_kelly",
        "--kelly-vol-window",
        "60",
        "--kelly-max-leverage",
        "0.2",
        "--no-deploy-train",
        "--wf-summary-csv",
        str(summary_csv),
    ]


def _read_latest_model_version(root: Path) -> int:
    version_file = root / "models" / "stock_selection" / "latest_model_version.txt"
    if not version_file.exists():
        raise FileNotFoundError(f"隔离根内未发现模型版本指针: {version_file}")
    text = version_file.read_text(encoding="utf-8").strip()
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"模型版本指针内容无法解析: {version_file}（{text!r}）") from exc


def _date_str(value: object) -> Optional[str]:
    """运行日志日期单元格归一为 YYYYMMDD 字符串（容忍 20241211.0 浮点形态）。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return str(int(float(str(value))))
    except (TypeError, ValueError):
        return None


def _int_or_none(value: object) -> Optional[int]:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


_TRAIN_BOUNDARY_RE = re.compile(
    r"训练集:\s*(\d{8})\s*至\s*(\d{8})（(\d+)\s*个交易日，(\d+)\s*条样本）"
)
_VAL_BOUNDARY_RE = re.compile(
    r"验证集:\s*(\d{8})\s*至\s*(\d{8})（(\d+)\s*个交易日，(\d+)\s*条样本）"
)
# 逐折训练段锚点（ml/v2 两侧 split_training 同一行文本：``开始训练 Split {i}``）
_FOLD_TRAIN_MARKER_RE = re.compile(r"开始训练 Split (\d+)")
# deploy 段锚点（旧 runner.py:658 / v2 orchestrator 同文；--no-deploy-train 时不出现）
_DEPLOY_TRAIN_MARKER = "开始部署模型训练"


def _slice_fold_log_section(text: str, split_index: int) -> str:
    """从全量训练日志切出指定折的段落（P3 T5 逐折边界泛化）。

    段起点 = ``开始训练 Split {split_index}``；段终点 = 下一个 ``开始训练 Split``（异号）
    或 deploy 段锚点或文末。锚点缺失 ⇒ ValueError（禁止静默回退全量解析——多折日志
    全量解析的三种子一致性假设不成立，会误判不一致）。

    兼容路径：整份日志**无任何折锚点**（T0.5b 合成夹具 / 单折截断日志）⇒ 按全量
    文本解析；若实为多折内容，折间边界不同会触发三种子一致性拒绝（fail-loud，
    无静默缩域）。
    """
    matches = list(_FOLD_TRAIN_MARKER_RE.finditer(text))
    if not matches:
        return text
    start = None
    for match in matches:
        if int(match.group(1)) == split_index:
            start = match.start()
            break
    if start is None:
        raise ValueError(f"训练日志中未找到 split {split_index} 的训练段锚点（开始训练 Split）")
    end = len(text)
    for match in matches:
        if match.start() > start and int(match.group(1)) != split_index:
            end = match.start()
            break
    deploy_at = text.find(_DEPLOY_TRAIN_MARKER, start)
    if deploy_at != -1 and deploy_at < end:
        end = deploy_at
    return text[start:end]


def parse_actual_train_boundary(
    log_path: Path, split_index: Optional[int] = None
) -> Dict[str, object]:
    """从训练日志解析**实际训练子集**边界（M1/R2-T05B-02 处置）。

    来源 = ``split.py:155-161`` 切分日志行（``prepare.py:980-981`` 同口径统计），
    三种子各记录一次、必须全部一致；日志须为 UTF-8 单一编码（M2 修复后形态）。

    P3 T5 泛化：``split_index=None``（默认）= 现状全量解析（spike 单折日志，
    逐位一致）；传整数 ⇒ 先切该折段落再解析（臂全 14 折日志逐折登记）。
    """
    text = Path(log_path).read_text(encoding="utf-8")
    if split_index is not None:
        text = _slice_fold_log_section(text, split_index)
    train_matches = _TRAIN_BOUNDARY_RE.findall(text)
    if not train_matches:
        raise ValueError(f"训练日志中未找到实际训练子集切分行: {log_path}")
    if len(set(train_matches)) != 1:
        raise ValueError(
            f"三种子实际训练子集边界不一致，禁止登记: {sorted(set(train_matches))}（{log_path}）"
        )
    train_start, train_end, n_dates, n_samples = train_matches[0]
    result: Dict[str, object] = {
        "train_actual_start": train_start,
        "train_actual_end": train_end,
        "train_actual_n_dates": int(n_dates),
        "train_actual_samples": int(n_samples),
    }
    val_matches = _VAL_BOUNDARY_RE.findall(text)
    if len(set(val_matches)) == 1:
        val_start, val_end, val_n_dates, val_samples = val_matches[0]
        result["val_raw_log_start"] = val_start
        result["val_raw_log_end"] = val_end
        result["val_raw_log_n_dates"] = int(val_n_dates)
        result["val_raw_log_samples"] = int(val_samples)
    return result


def read_split_boundary(root: Path, log_path: Path, split_index: int) -> Dict[str, object]:
    """逐折边界日期清单（§13.3.2 标定登记；M1/R2-T05B-02 双口径分列；P3 T5 逐折泛化）。

    - 外层训练窗（``outer_train_*``）：台账 ``start_date/end_date/trade_days_count``
      （split_training.py:602-603 写入口径 = 完整外层窗口，含 val-raw 段）；
    - **实际训练子集**（``train_actual_*``）：训练日志切分行（已扣 label_delta 间隔，
      §13.3.2 约定的 prepare.py:980-995 统计口径）；与 val-raw 不重叠；
    - val-raw / ES / calibration / embargo：台账既有列；
    - 样本三层口径（S4）：``total_samples``（过滤前）/ ``samples_after_filter``（过滤后）/
      ``train_samples + val_raw_samples``（切分后）；``val_samples`` 台账列实为 ES 段样本
      （与 ``val_es_samples`` 同值，命名易误读，登记为 ``val_es_samples_alias``）。
    """
    csv_path = root / "models" / "stock_selection" / "ml_train_runs.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"隔离根内未发现训练运行日志: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    split_rows = df[df["split_index"].astype(str) == str(split_index)]
    if split_rows.empty:
        raise ValueError(f"训练运行日志缺 split {split_index} 记录: {csv_path}")
    row = split_rows.iloc[-1]
    boundary: Dict[str, object] = {
        "outer_train_start": _date_str(row.get("start_date")),
        "outer_train_end": _date_str(row.get("end_date")),
        "test_start": _date_str(row.get("test_start_date")),
        "test_end": _date_str(row.get("test_end_date")),
        "val_raw_start": _date_str(row.get("val_raw_start_date")),
        "val_raw_end": _date_str(row.get("val_raw_end_date")),
        "val_raw_n_dates": _int_or_none(row.get("val_raw_n_dates")),
        "val_raw_samples": _int_or_none(row.get("val_raw_samples")),
        "val_es_start": _date_str(row.get("val_es_start_date")),
        "val_es_end": _date_str(row.get("val_es_end_date")),
        "val_es_n_dates": _int_or_none(row.get("val_es_n_dates")),
        "val_es_samples": _int_or_none(row.get("val_es_samples")),
        "val_calib_start": _date_str(row.get("val_calib_start_date")),
        "val_calib_end": _date_str(row.get("val_calib_end_date")),
        "val_calib_n_dates": _int_or_none(row.get("val_calib_n_dates")),
        "val_calib_samples": _int_or_none(row.get("val_calib_samples")),
        "val_embargo_start": _date_str(row.get("val_embargo_start_date")),
        "val_embargo_end": _date_str(row.get("val_embargo_end_date")),
        "val_embargo_n_dates": _int_or_none(row.get("val_embargo_n_dates")),
        "val_embargo_samples": _int_or_none(row.get("val_embargo_samples")),
        "train_samples": _int_or_none(row.get("train_samples")),
        "val_es_samples_alias": _int_or_none(row.get("val_samples")),
        "total_samples": _int_or_none(row.get("total_samples")),
        "samples_after_filter": _int_or_none(row.get("samples_after_filter")),
        "trade_days_count": _int_or_none(row.get("trade_days_count")),
        "best_iteration": _int_or_none(row.get("best_iteration")),
        "n_jobs": _int_or_none(row.get("n_jobs")),
    }
    actual = parse_actual_train_boundary(log_path, split_index=split_index)
    # 日志 val-raw 与台账 val-raw 必须一致（交叉校验，任一不符即拒绝登记）
    if actual.get("val_raw_log_start") is not None:
        if (
            actual["val_raw_log_start"] != boundary["val_raw_start"]
            or actual["val_raw_log_end"] != boundary["val_raw_end"]
            or actual["val_raw_log_samples"] != boundary["val_raw_samples"]
        ):
            raise ValueError(f"日志 val-raw 与台账不一致，禁止登记: {actual} vs {boundary}")
    boundary.update(actual)
    if boundary["train_actual_end"] is not None and boundary["val_raw_start"] is not None:
        if not str(boundary["train_actual_end"]) < str(boundary["val_raw_start"]):
            raise ValueError(
                f"实际训练子集截止 {boundary['train_actual_end']} 跨入 val-raw 起点"
                f" {boundary['val_raw_start']}，禁止登记"
            )
    return boundary


def read_split0_boundary(root: Path, log_path: Path) -> Dict[str, object]:
    """首折边界（原签名包装；P3 T5 起实现泛化为 ``read_split_boundary`` 逐折读取）。"""
    return read_split_boundary(root, log_path, 0)


def _run_walk_forward(root: Path, log_path: Path, args: Optional[List[str]] = None) -> float:
    """执行一次 spike 训练（子进程，日志落 logs/），返回墙钟秒数。

    日志编码（M2）：子进程强制 ``PYTHONIOENCODING=utf-8``；父进程以二进制句柄
    写 header 并立即 flush，保证整文件 UTF-8 单一编码、header 位于文件首部。

    P3 T5 泛化：``args=None``（默认）= spike 现状（内部构建 spike 参数，逐位一致）；
    显式传参 ⇒ 调用方构建（T5 臂①全 14 折 / 折子集续跑口径，summary 名按 run_name）。
    """
    if args is None:
        summary_csv = root / "walk_forward" / "raw" / "walk_forward_summary_spike.csv"
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        args = build_spike_walk_forward_args(root, summary_csv)
    cmd = [
        sys.executable,
        "scripts/walk_forward.py",
        *args,
    ]
    logger.info(f"spike 训练命令: {' '.join(cmd)}")
    logger.info(f"spike 训练日志: {log_path}")
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    started = time.monotonic()
    with open(log_path, "ab") as log_fh:
        # S10：header 不写前导换行，保证日志首行即 header
        header = (
            f"===== spike 训练启动 {datetime.now().isoformat(timespec='seconds')} =====\n"
            + " ".join(cmd)
            + "\n"
        )
        log_fh.write(header.encode("utf-8"))
        log_fh.flush()
        completed = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(f"spike 训练失败（exit={completed.returncode}），日志: {log_path}")
    return elapsed


# 空文本 sha256（R2-T05B-03：空批次指纹禁止放行）
_EMPTY_SHA256 = hashlib.sha256("".encode("utf-8")).hexdigest()


def spike_config_digest() -> str:
    """spike 配置摘要（路径占位归一化，run1/run2 可比）。"""
    args = build_spike_walk_forward_args(Path("<ROOT>"), Path("<SUMMARY>"))
    return hashlib.sha256(json.dumps(args, ensure_ascii=False).encode("utf-8")).hexdigest()


def list_expected_oos_days(data_root: Path, test_start: str, test_end: str) -> List[str]:
    """独立预期打分日 = trade_cal ``is_open=1`` ∩ [test_start, test_end]（N3：
    预期日来源独立于已存在分区清单；clean 日历优先、raw 兜底）。"""
    from src.lazybull.data.loader import DataLoader
    from src.lazybull.data.storage import Storage

    storage = Storage(root_path=str(data_root))
    loader = DataLoader(storage)
    cal = loader.load_clean_trade_cal()
    if cal is None or len(cal) == 0:
        cal = loader.load_trade_cal()
    if cal is None or len(cal) == 0:
        raise ValueError(f"无法加载交易日历: {data_root}")
    mask = (
        (cal["is_open"] == 1)
        & (cal["cal_date"].astype(str) >= test_start)
        & (cal["cal_date"].astype(str) <= test_end)
    )
    days = sorted(str(day) for day in cal.loc[mask, "cal_date"].tolist())
    if not days:
        raise ValueError(f"交易日历在 [{test_start}, {test_end}] 内无开市日: {data_root}")
    return days


def resolve_score_days(expected_days: List[str], features_root: Path) -> List[str]:
    """预期打分日 ∩ cs_train 实际分区：缺失即失败并列出缺失日（N3 补验口径）。"""
    cs_train_dir = Path(features_root) / "cs_train"
    if not cs_train_dir.is_dir():
        raise FileNotFoundError(f"cs_train 分区目录不存在: {cs_train_dir}")
    existing = set()
    for path in cs_train_dir.glob("*.parquet"):
        day = _normalize_partition_day(path.stem)
        if len(day) == 8 and day.isdigit():
            existing.add(day)
    missing = [day for day in expected_days if day not in existing]
    if missing:
        suffix = "…" if len(missing) > 10 else ""
        raise RuntimeError(
            f"预期打分日缺 cs_train 分区（{len(missing)} 日）: {missing[:10]}{suffix}（N3）"
        )
    return list(expected_days)


def build_producer_identity(run_index: int, input_snapshot_digest: str) -> Dict[str, object]:
    """单次运行的生产者身份（R2-T05B-04）：代码态 + 环境 + 配置摘要 + **输入基线摘要**。"""
    from src.lazybull.v2.store.store_identity import workspace_identity

    return {
        "run_index": run_index,
        "source_digest": workspace_identity(REPO_ROOT)["source_digest"],
        "environment_identity": environment_identity(),
        "config_digest": spike_config_digest(),
        "input_snapshot_digest": input_snapshot_digest,
    }


def _assert_identity_compatible(saved: Dict[str, object], current: Dict[str, object]) -> None:
    """续跑复用前逐项核验生产者身份；任一不匹配即拒绝（R2-T05B-04 补验口径）。"""
    mismatches = []
    for key in ("source_digest", "config_digest", "input_snapshot_digest"):
        if saved.get(key) != current.get(key):
            mismatches.append(f"{key}: {saved.get(key)} != {current.get(key)}")
    saved_env = saved.get("environment_identity") or {}
    current_env = current.get("environment_identity") or {}
    if saved_env != current_env:
        mismatches.append("environment_identity 不一致")
    if mismatches:
        raise ValueError(
            "续跑指纹的生产者身份与当前不一致，禁止复用（R2-T05B-04）；"
            "请换新 --run-name 重新取证: " + "; ".join(mismatches)
        )


_MODEL_SIDECAR_SUFFIXES = ("_model.joblib", "_metadata.json", "_features.json")


def collect_model_identity(root: Path, model_version: int) -> Dict[str, str]:
    """隔离根内模型及必需 sidecar 的物理身份（R2-T05B-04 复核余项：绑定模型字节）。"""
    from src.lazybull.v2.store.store_identity import file_identity

    base = Path(root) / "models" / "stock_selection"
    identity: Dict[str, str] = {}
    for suffix in _MODEL_SIDECAR_SUFFIXES:
        candidate = base / f"v{model_version}{suffix}"
        if not candidate.exists():
            raise FileNotFoundError(f"模型 sidecar 缺失: {candidate}")
        identity[candidate.name] = file_identity(candidate)["sha256"]
    return identity


def _assert_resume_run_compatible(
    saved: Dict[str, object], current_identity: Dict[str, object], root: Path
) -> None:
    """续跑单 run 复用前核验：生产者身份 + 隔离模型字节身份（R2-T05B-04 复核余项）。"""
    _assert_identity_compatible(
        saved.get("fingerprint", {}).get("producer_identity", {}), current_identity
    )
    saved_model_identity = saved.get("model_identity")
    if not saved_model_identity:
        raise ValueError("续跑指纹缺模型身份记录，禁止复用（R2-T05B-04）")
    current_model_identity = collect_model_identity(root, int(saved["model_version"]))
    if current_model_identity != saved_model_identity:
        raise ValueError("隔离根模型文件身份与保存值不一致，禁止复用（R2-T05B-04）")


def _assert_coverage(
    fingerprint: Dict[str, object], expected_days: List[str], run_index: int
) -> None:
    """覆盖完整性门（R2-T05B-03 + N3）：打分日清单 == 独立预期日清单 + 非空批次。"""
    if fingerprint.get("score_days") != list(expected_days):
        raise RuntimeError(
            f"run{run_index} 打分日清单与独立预期日不一致"
            f"（{fingerprint.get('n_days')}/{len(expected_days)} 日），禁止缩域后放行（N3）"
        )
    if fingerprint["n_rows"] <= 0 or fingerprint["fingerprint"] == _EMPTY_SHA256:
        raise RuntimeError(f"run{run_index} 空预测批次，禁止产出通过指纹（R2-T05B-03）")


def _execute_run(
    run_index: int,
    run_dir: Path,
    production_data_root: Path,
    logs_dir: Path,
    run_name: str,
    producer_identity: Dict[str, object],
    skip_existing: bool,
    expected_n_jobs: Optional[int],
) -> Dict[str, object]:
    """单跑执行：续跑复用（先核验）或 训练 → 边界 → 预期日 → 打分 → 覆盖门 → 落盘。"""
    sub_dir = run_dir / f"run{run_index}"
    root = sub_dir / "root"
    fingerprint_path = sub_dir / "fingerprint.json"
    if skip_existing and fingerprint_path.exists():
        saved = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        _assert_resume_run_compatible(saved, producer_identity, root)
        saved["reused"] = True
        logger.info(f"run{run_index} 指纹复用（生产者 + 模型身份核验通过）: {fingerprint_path}")
        return saved

    root = build_isolated_root(sub_dir, production_data_root)
    log_path = logs_dir / f"v2_p3_t05b_spike_{run_name}_run{run_index}.log"
    wall_seconds = _run_walk_forward(root, log_path)
    logger.info(f"run{run_index} 训练完成，墙钟 {wall_seconds/60:.1f} 分钟")

    model_version = _read_latest_model_version(root)
    boundary = read_split0_boundary(root, log_path)
    if boundary["n_jobs"] != expected_n_jobs:
        raise RuntimeError(
            f"台账 n_jobs={boundary['n_jobs']} != 登记 cpu_count={expected_n_jobs}，"
            "线程固定口径被破坏（reviewer1-S8），停止"
        )
    expected_days = list_expected_oos_days(
        root, str(boundary["test_start"]), str(boundary["test_end"])
    )
    score_days = resolve_score_days(expected_days, root / "features")
    main_board_codes = load_main_board_codes(root)
    fingerprint = score_model_fingerprint(
        models_dir=root / "models" / "stock_selection",
        model_version=model_version,
        data_root=root,
        score_days=score_days,
        main_board_codes=main_board_codes,
        producer_identity=producer_identity,
    )
    _assert_coverage(fingerprint, expected_days, run_index)
    run_evidence: Dict[str, object] = {
        "run_index": run_index,
        "reused": False,
        "wall_seconds": round(wall_seconds, 1),
        "model_version": model_version,
        "model_identity": collect_model_identity(root, model_version),
        "split0_boundary": boundary,
        "n_score_days": len(score_days),
        "score_day_first": score_days[0],
        "score_day_last": score_days[-1],
        "fingerprint": fingerprint,
        "train_log": str(log_path),
    }
    sub_dir.mkdir(parents=True, exist_ok=True)
    fingerprint_path.write_text(
        json.dumps(run_evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return run_evidence


def run_spike(
    run_name: str,
    production_data_root: Path,
    skip_existing: bool = False,
    spike_base_dir: Path = SPIKE_BASE_DIR,
) -> Dict[str, object]:
    """spike 主流程：基线获取（新建/续跑核验）→ 双跑 → 指纹比对 → 复核 → 证据落盘。"""
    from src.lazybull.common.config import get_logs_dir

    production_data_root = Path(production_data_root)
    run_dir = Path(spike_base_dir) / run_name
    logs_dir = Path(get_logs_dir())
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    baselines = _acquire_baselines(run_dir, production_data_root, skip_existing)
    evidence = _build_evidence(
        run_name, production_data_root, baselines, spike_config_digest(), REPO_ROOT
    )
    expected_n_jobs = os.cpu_count()

    runs_evidence: List[Dict[str, object]] = []
    for run_index in (1, 2):
        producer_identity = build_producer_identity(
            run_index, str(baselines["input_snapshot_digest"])
        )
        runs_evidence.append(
            _execute_run(
                run_index,
                run_dir,
                production_data_root,
                logs_dir,
                run_name,
                producer_identity,
                skip_existing,
                expected_n_jobs,
            )
        )

    logger.info("生产模型目录守卫复核（运行后）...")
    guard_before = baselines["guard_before"]
    guard_drift = verify_production_models_guard(guard_before)  # type: ignore[arg-type]
    logger.info("消费集合物理身份复核（运行后）...")
    consumption_drift = verify_consumption_snapshot(
        baselines["consumption_snapshot_path"],  # type: ignore[arg-type]
        base_dir=baselines["base_dir"],  # type: ignore[arg-type]
    )

    evidence["runs"] = runs_evidence
    evidence["comparison"] = _finalize_comparison(runs_evidence, guard_drift, consumption_drift)
    evidence["finished_at"] = datetime.now().isoformat(timespec="seconds")

    evidence_path = run_dir / "spike_evidence.json"
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"spike 证据落盘: {evidence_path}")
    _log_verdict(evidence["comparison"])  # type: ignore[arg-type]
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P3 T0.5b 旧路径确定性 spike（机器任务，须先获授权）"
    )
    parser.add_argument(
        "--run-name",
        default=f"t05b_spike_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="spike 运行名（默认时间戳）；复用同名目录可配合 --skip-existing 续跑",
    )
    parser.add_argument("--production-data-root", default="./data", help="生产数据根")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="已完成 run 的指纹存在时复用（中断续跑用）",
    )
    args = parser.parse_args()

    evidence = run_spike(
        run_name=args.run_name,
        production_data_root=Path(args.production_data_root),
        skip_existing=args.skip_existing,
    )
    verdict = str(evidence["comparison"]["verdict"])  # type: ignore[index]
    if verdict != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
