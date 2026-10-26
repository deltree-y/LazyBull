# -*- coding: utf-8 -*-
"""P3 T5 三臂统一批驱动（t5_arm_runner）与组装/产物件单元测试（§19.8 arm runner 节）。

**全部合成/mock，禁止真实训练与真实生产数据**。覆盖：

- 参数镜像 parity（臂① args == spike 生产口径去掉首折限制；默认逐位不变）；
- 逐折版本解析（台账 split_index 口径）与 skip 起始版本反解（等差断言正反例）；
- 逐折边界泛化（逐折段落切分 / 包装等价 / 交叉校验拒绝 / 锚点缺失拒绝）；
- 折级断点续跑（齐备+身份一致复用 / 模型字节篡改拒绝 / 部分产物拒绝 /
  基线缺失拒绝独立于续跑开关且先于任何快照写入）；
- 承接① sleeve_identity 双注入接线；承接②断言正反例；承接③启动门正反例
  （集合不等 / 序指纹不符拒绝；禁止关闭族子集比对）；
- 承接④ 四族 derive 调用点扫描（真实 v2 树 == 白名单 / 漂移检出 / 合成新点捕获）；
- double-run 四门比对接线（合成 executor 同码两遍 PASS / run2 篡改 FAIL）；
- B0' 冻结落盘结构 / 幂等复用 / 身份不符拒绝；
- 布局契约字段完整性（run_arm 单跑合成端到端）；
- runs_batch_writer 适配闭包（真实 write_runs_batch 落盘 + extra_meta_notes 透传 +
  缺净值曲线拒绝）。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v2_p3.t5_arm_artifacts import (  # noqa: E402
    ARM_LEGACY_CS_TRAIN,
    ARM_V2_CS_TRAIN_VERIFY,
    ARM_V2_PANEL,
    B0_PRIME_DIR_NAME,
    FULL_SPLIT_COUNT,
    acquire_arm_baselines,
    arm_config_digest,
    assert_derive_call_sites_registered,
    build_batch_fingerprint_doc,
    collect_derive_call_site_report,
    compare_double_run,
    load_fold_record_for_resume,
    resolve_fold_model_version,
    resolve_skip_start_model_version,
    runs_batch_state,
    scan_runtime_derive_call_sites,
    write_arm_meta,
)
from scripts.v2_p3.t5_arm_assembly import (  # noqa: E402
    VALUE_DECLARATION_ORDER_FINGERPRINT,
    assemble_arm_orchestrator,
    assert_feature_declaration_gate,
    assert_holding_period_matches_label,
    build_backtest_host_params,
    build_runs_batch_config,
    build_runs_batch_writer,
    build_signal_factory,
)
from scripts.v2_p3.t5_arm_runner import (  # noqa: E402
    ArmExecutor,
    ArmRunContext,
    _normalize_selected,
    build_arm_config_payload,
    run_arm,
)
from scripts.v2_p3.t05b_fingerprint import fingerprint_prediction_rows  # noqa: E402
from scripts.v2_p3.t05b_spike_runner import (  # noqa: E402
    build_spike_walk_forward_args,
    parse_actual_train_boundary,
    read_split0_boundary,
    read_split_boundary,
)

# ---------------------------------------------------------------------------
# 合成夹具（口径复制自 test_v2_p3_t05b_spike.py 先例；本文件自包含）
# ---------------------------------------------------------------------------

_DAYS: List[str] = pd.bdate_range("20250102", periods=60).strftime("%Y%m%d").tolist()
_TRAIN_BOUNDARY = ("20200107", "20220930", 600, 60000)  # 训练集切分行（train_actual）
_VAL_RAW = ("20221003", "20221230", 60, 6000)  # val-raw（台账与日志须一致）


def _fold_test_window(split_index: int) -> Tuple[str, str]:
    return _DAYS[2 * split_index], _DAYS[2 * split_index + 1]


def _ledger_row(split_index: int, model_version: int) -> Dict[str, object]:
    test_start, test_end = _fold_test_window(split_index)
    return {
        "split_index": split_index,
        "model_version": model_version,
        "start_date": 20200102,
        "end_date": 20221231,
        "test_start_date": int(test_start),
        "test_end_date": int(test_end),
        "val_raw_start_date": int(_VAL_RAW[0]),
        "val_raw_end_date": int(_VAL_RAW[1]),
        "val_raw_n_dates": _VAL_RAW[2],
        "val_raw_samples": _VAL_RAW[3],
        "val_es_start_date": 20221003,
        "val_es_end_date": 20221130,
        "val_es_n_dates": 40,
        "val_es_samples": 4000,
        "val_calib_start_date": 20221201,
        "val_calib_end_date": 20221230,
        "val_calib_n_dates": 20,
        "val_calib_samples": 2000,
        "val_embargo_start_date": None,
        "val_embargo_end_date": None,
        "val_embargo_n_dates": 0,
        "val_embargo_samples": 0,
        "train_samples": _TRAIN_BOUNDARY[3],
        "val_samples": 4000,
        "total_samples": 70000,
        "samples_after_filter": 66000,
        "trade_days_count": 700,
        "best_iteration": 490,
        "n_jobs": 4,
    }


def _write_ledger(root: Path, splits: Sequence[int], *, version_offset: int = 0) -> None:
    """合成台账（ml_train_runs.csv；版本 = split_index + 1 + offset）。"""
    models = root / "models" / "stock_selection"
    models.mkdir(parents=True, exist_ok=True)
    csv_path = models / "ml_train_runs.csv"
    rows = [_ledger_row(split, split + 1 + version_offset) for split in splits]
    if csv_path.exists():
        existing = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
        rows_df = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    else:
        rows_df = pd.DataFrame(rows)
    rows_df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def _write_model_files(root: Path, splits: Sequence[int], *, version_offset: int = 0) -> None:
    """合成模型三件套（字节内容确定 = 版本函数，续跑核验/篡改检出依赖）。"""
    models = root / "models" / "stock_selection"
    models.mkdir(parents=True, exist_ok=True)
    for split in splits:
        version = split + 1 + version_offset
        for suffix in ("_model.joblib", "_metadata.json", "_features.json"):
            (models / f"v{version}{suffix}").write_bytes(f"model-bytes-v{version}".encode())


def _fold_log_section(split_index: int, *, mismatch_seed: bool = False) -> str:
    """单折日志段（锚点 + 三种子切分行；mismatch_seed=True 时第三种子边界错位）。"""
    lines = [f"00:00:01 | INFO | 开始训练 Split {split_index}\n"]
    for seed_idx in range(3):
        end = _TRAIN_BOUNDARY[1] if not (mismatch_seed and seed_idx == 2) else "20221001"
        lines.append(
            "00:00:02 | INFO |   训练集: "
            f"{_TRAIN_BOUNDARY[0]} 至 {end}（{_TRAIN_BOUNDARY[2]} 个交易日，"
            f"{_TRAIN_BOUNDARY[3]} 条样本）\n"
        )
        lines.append(
            "00:00:02 | INFO |   验证集: "
            f"{_VAL_RAW[0]} 至 {_VAL_RAW[1]}（{_VAL_RAW[2]} 个交易日，"
            f"{_VAL_RAW[3]} 条样本）\n"
        )
    return "".join(lines)


def _write_log(log_path: Path, splits: Sequence[int]) -> None:
    with open(log_path, "a", encoding="utf-8") as fh:
        for split in splits:
            fh.write(_fold_log_section(split))


def _make_production_root(tmp_path: Path) -> Path:
    """合成生产数据根：消费集合契约全目录 + 域/日历单文件 + 生产模型目录。"""
    prod = tmp_path / "data"
    # 闭市日判定输入目录（raw/moneyflow_hsgt）随 R2-T5AE-02 起纳入消费集合硬门
    for sub in ("features/cs_train", "clean/daily", "raw/suspend", "raw/moneyflow_hsgt"):
        directory = prod / sub
        directory.mkdir(parents=True, exist_ok=True)
    for day in _DAYS:
        pd.DataFrame(
            {"trade_date": [day], "ts_code": ["000001.SH"], "is_st": [0], "x": [1.0]}
        ).to_parquet(prod / "features" / "cs_train" / f"{day}.parquet", index=False)
    pd.DataFrame({"trade_date": [_DAYS[0]], "close": [10.0]}).to_parquet(
        prod / "clean" / "daily" / f"{_DAYS[0][:4]}-{_DAYS[0][4:6]}-{_DAYS[0][6:]}.parquet",
        index=False,
    )
    pd.DataFrame({"trade_date": [_DAYS[0]]}).to_parquet(
        prod / "raw" / "suspend" / f"{_DAYS[0][:4]}-{_DAYS[0][4:6]}-{_DAYS[0][6:]}.parquet",
        index=False,
    )
    pd.DataFrame(
        {"trade_date": [_DAYS[0]], "hgt": [1.0], "sgt": [1.0], "north_money": [2.0]}
    ).to_parquet(
        prod / "raw" / "moneyflow_hsgt" / f"{_DAYS[0][:4]}-{_DAYS[0][4:6]}-{_DAYS[0][6:]}.parquet",
        index=False,
    )
    pd.DataFrame({"ts_code": ["000001.SH"], "market": ["主板"]}).to_parquet(
        prod / "clean" / "stock_basic.parquet", index=False
    )
    pd.DataFrame({"cal_date": _DAYS, "is_open": [1] * len(_DAYS)}).to_parquet(
        prod / "clean" / "trade_cal.parquet", index=False
    )
    pd.DataFrame({"cal_date": _DAYS, "is_open": [1] * len(_DAYS)}).to_parquet(
        prod / "raw" / "trade_cal.parquet", index=False
    )
    (prod / "models" / "stock_selection").mkdir(parents=True, exist_ok=True)
    (prod / "models" / "stock_selection" / "sentinel.txt").write_text("prod", encoding="utf-8")
    (prod / "labels").mkdir(parents=True, exist_ok=True)  # 只读接入四目录契约（labels 必接）
    return prod


def _copytree_link(link: Path, target: Path) -> None:
    """测试用目录联接替身（合成小目录真实复制，避免 mklink）。"""
    shutil.copytree(target, link)


def _synthetic_nav(split_index: int) -> pd.DataFrame:
    days = pd.bdate_range(_fold_test_window(split_index)[0], periods=3)
    return pd.DataFrame(
        {
            "date": days,
            "portfolio_value": [1.0, 1.001, 1.002],
            "capital": [1.0, 1.001, 1.002],
            "market_value": [0.0, 0.0, 0.0],
            "nav": [1.0, 1.001, 1.002],
            "return": [0.0, 0.001, 0.001],
        }
    )


def _write_synthetic_runs_batch(ctx: ArmRunContext, splits: Sequence[int]) -> None:
    """经**真实** ``write_runs_batch`` 适配闭路落合成 runs 批（写后自校验一并过）。"""
    results = []
    for split in splits:
        test_start, test_end = _fold_test_window(split)
        results.append(
            {
                "split_index": split,
                "model_version": split + 1,
                "train_start": "20200102",
                "train_end": "20221231",
                "test_start": test_start,
                "test_end": test_end,
                "bt_metrics": {
                    "bt_total_return": 0.002,
                    "bt_max_drawdown": -0.001,
                    "bt_sharpe": 1.0,
                    "bt_start": test_start,
                    "bt_end": test_end,
                },
                "_nav_curve": _synthetic_nav(split),
                "_trades": pd.DataFrame(),
                "_execution_attribution": pd.DataFrame(),
                "_holdings_snapshot": pd.DataFrame(),
            }
        )
    writer = build_runs_batch_writer(
        runs_root=ctx.runs_root,
        config={"label_column": "neu_y_ret_20", "algorithm": "xgboost", "stock_domain": "main"},
        code_state={"git_commit": "test", "git_dirty": False},
        arms=[ctx.arm],
        source=f"v2_p3_t5_{ctx.arm}",
    )
    writer(
        results=results,
        wf_run_id=f"wf_synthetic_{ctx.run_name}_{ctx.arm_dir.name}",
        data_state_payload={"data_state_id": "ds_synthetic"},
        extra_meta_notes={"sleeve_id": "value", "spec_version": "1"},
    )


class _ExecutorState:
    def __init__(self) -> None:
        self.train_calls: List[Tuple[List[int], bool]] = []
        self.replay_calls = 0
        self.score_calls: List[int] = []
        # R2-T5A-06 接线核验：逐折记录 (persist_predictions, predictions_path, anchor)
        self.score_persist: List[Tuple[object, object, object]] = []
        # 信号选择参数映射传参核验（signal_selection，三臂统一）
        self.score_selection: List[object] = []


def _emulate_prediction_persistence(
    result: Dict[str, object],
    kwargs: Dict[str, object],
    rows: List[Tuple[str, str, float]],
    score_days: Sequence[str],
) -> None:
    """模拟真实持久化契约：predictions.parquet + signal_topk.parquet 落盘 +
    相对锚点 POSIX 登记（与 ``score_model_fingerprint`` 持久化段同形态；
    续跑五件套核验依赖真实落盘字节）。
    """
    anchor = kwargs.get("predictions_anchor")
    pred_raw = kwargs.get("predictions_path")
    if not (kwargs.get("persist_predictions") and pred_raw and anchor):
        return
    anchor_path = Path(str(anchor))
    pred = Path(str(pred_raw))
    pred.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=["trade_date", "ts_code", "score"])
    frame.to_parquet(pred, index=False)
    result["predictions_path"] = pred.resolve().relative_to(anchor_path.resolve()).as_posix()
    result["predictions_sha256"] = hashlib.sha256(pred.read_bytes()).hexdigest()
    result["predictions_rows"] = len(rows)
    selection = kwargs.get("signal_selection")
    if selection is None:
        return
    top_n = int(selection["top_n"])  # type: ignore[index]
    topk = frame.sort_values("score", ascending=False).head(top_n)  # 确定性合成 top-k
    topk_path = pred.with_name("signal_topk.parquet")
    topk.to_parquet(topk_path, index=False)
    result["signal_topk_path"] = topk_path.resolve().relative_to(anchor_path.resolve()).as_posix()
    result["signal_topk_sha256"] = hashlib.sha256(topk_path.read_bytes()).hexdigest()
    result["signal_topk_rows"] = len(topk)
    result["signal_topk_days"] = list(score_days)
    result["signal_topk_empty_days"] = []


def _make_synthetic_executor(state: _ExecutorState, *, tamper_run2: bool = False) -> ArmExecutor:
    """合成执行缝：训练 = 台账/模型/日志合成；回放 = 真实版本反解 + 合成 runs 批；
    打分 = 确定性合成指纹（tamper_run2 时 run2 指纹加盐，双跑 FAIL 用例）。"""

    def train_fn(ctx: ArmRunContext, subset: Tuple[int, ...], write_batch: bool):
        state.train_calls.append((list(subset), write_batch))
        _write_ledger(ctx.root, subset)
        _write_model_files(ctx.root, subset)
        _write_log(ctx.log_path, subset)
        if write_batch:
            _write_synthetic_runs_batch(ctx, subset)
        return {"mode": "synthetic_train", "trained_splits": list(subset)}

    def replay_fn(ctx: ArmRunContext):
        state.replay_calls += 1
        start = resolve_skip_start_model_version(ctx.root, ctx.selected_split_indices)
        _write_synthetic_runs_batch(ctx, ctx.selected_split_indices)
        return {"mode": "synthetic_replay", "start_model_version": start}

    def score_fn(
        *,
        models_dir: Path,
        model_version: int,
        data_root: Path,
        score_days: Sequence[str],
        main_board_codes: object,
        producer_identity: Optional[Dict[str, object]] = None,
        **kwargs: object,
    ) -> Dict[str, object]:
        state.score_calls.append(model_version)
        state.score_persist.append(
            (
                kwargs.get("persist_predictions"),
                kwargs.get("predictions_path"),
                kwargs.get("predictions_anchor"),
            )
        )
        state.score_selection.append(kwargs.get("signal_selection"))
        bump = 0.0
        if tamper_run2 and (producer_identity or {}).get("run_index") == 2:
            bump = 1000.0
        rows = [
            (day, "000001.SH", model_version * 1.0 + idx + bump)
            for day in score_days
            for idx in range(2)
        ]
        fingerprint = fingerprint_prediction_rows(rows)
        result: Dict[str, object] = {
            "model_version": model_version,
            "fingerprint": fingerprint,
            "per_seed_diagnostic": [
                {"seed": seed, "n_rows": len(rows), "fingerprint": fingerprint}
                for seed in (42, 61, 82)
            ],
            "score_days": list(score_days),
            "n_days": len(score_days),
            "n_rows": len(rows),
            "producer_identity": dict(producer_identity or {}),
        }
        # R2-T5A-06/R1-05 契约仿真：真实落盘 + 相对锚点登记 + signal_topk 增键
        _emulate_prediction_persistence(result, kwargs, rows, list(score_days))
        return result

    return ArmExecutor(train_fn=train_fn, replay_fn=replay_fn, score_fn=score_fn)


def _run_kwargs(prod: Path, tmp_path: Path, state: _ExecutorState, **over: object):
    kwargs = {
        "arm": ARM_V2_CS_TRAIN_VERIFY,
        "run_name": "t5_test",
        "production_data_root": prod,
        "split_count": 3,
        "final_date": "20260105",
        "base_dir": tmp_path / "frozen",
        "logs_dir": tmp_path / "logs",
        "executor": _make_synthetic_executor(state),
        "link_fn": _copytree_link,
        "verify_links": False,
    }
    kwargs.update(over)
    return kwargs


# ---------------------------------------------------------------------------
# 1. 参数镜像 parity（臂① = spike 生产口径去首折限制）
# ---------------------------------------------------------------------------


class TestArgsMirrorParity:
    def test_default_bit_identical_to_spike(self):
        """默认值调用与 spike 现状逐位一致（含 --selected-split-indices 0）。"""
        args = build_spike_walk_forward_args(Path("<ROOT>"), Path("<SUMMARY>"))
        assert "--selected-split-indices" in args
        idx = args.index("--selected-split-indices")
        assert args[idx + 1] == "0"

    def test_none_omits_split_flag_only(self):
        """None ⇒ 去掉首折限制 flag，其余逐项一致且相对顺序不变（正式臂①全 14 折）。"""
        spike = build_spike_walk_forward_args(Path("<ROOT>"), Path("<SUMMARY>"))
        full = build_spike_walk_forward_args(
            Path("<ROOT>"), Path("<SUMMARY>"), selected_split_indices=None
        )
        assert "--selected-split-indices" not in full
        idx = spike.index("--selected-split-indices")
        drop = {idx, idx + 1}
        expected = [item for i, item in enumerate(spike) if i not in drop]
        assert full == expected
        # 生产口径关键项逐项在位
        for key in ("--label", "neu_y_ret_20", "--ensemble-seeds", "42,61,82", "--bt-top-n", "20"):
            assert key in full

    def test_subset_passthrough(self):
        args = build_spike_walk_forward_args(
            Path("<ROOT>"), Path("<SUMMARY>"), selected_split_indices=("0", "4", "5")
        )
        idx = args.index("--selected-split-indices")
        assert [args[idx + 1], args[idx + 2], args[idx + 3]] == ["0", "4", "5"]


# ---------------------------------------------------------------------------
# 2. 逐折版本解析 + skip 起始版本反解
# ---------------------------------------------------------------------------


class TestFoldVersionResolution:
    def test_resolve_per_fold(self, tmp_path):
        root = tmp_path / "root"
        _write_ledger(root, [0, 1, 2])
        assert resolve_fold_model_version(root, 0) == 1
        assert resolve_fold_model_version(root, 2) == 3

    def test_missing_fold_rejected(self, tmp_path):
        root = tmp_path / "root"
        _write_ledger(root, [0, 1])
        with pytest.raises(ValueError, match="须先续训"):
            resolve_fold_model_version(root, 2)

    def test_missing_ledger_rejected(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="台账"):
            resolve_fold_model_version(tmp_path / "root", 0)

    def test_skip_start_version(self, tmp_path):
        root = tmp_path / "root"
        _write_ledger(root, [0, 1, 2])
        assert resolve_skip_start_model_version(root, [0, 1, 2]) == 1
        # 子集同样反解一致（v(s) = 1 + s）
        assert resolve_skip_start_model_version(root, [1, 2]) == 1

    def test_skip_start_non_arithmetic_rejected(self, tmp_path):
        root = tmp_path / "root"
        _write_ledger(root, [0, 1])
        _write_ledger(root, [2], version_offset=10)  # split2 ⇒ v13，破坏等差
        with pytest.raises(ValueError, match="等差寻址"):
            resolve_skip_start_model_version(root, [0, 1, 2])


# ---------------------------------------------------------------------------
# 3. 逐折边界泛化
# ---------------------------------------------------------------------------


class TestPerFoldBoundary:
    def test_per_fold_read(self, tmp_path):
        root = tmp_path / "root"
        _write_ledger(root, [0, 1])
        log_path = tmp_path / "train.log"
        _write_log(log_path, [0, 1])
        b0 = read_split_boundary(root, log_path, 0)
        b1 = read_split_boundary(root, log_path, 1)
        assert b0["test_start"] == _fold_test_window(0)[0]
        assert b1["test_start"] == _fold_test_window(1)[0]
        assert b0["train_actual_start"] == _TRAIN_BOUNDARY[0]
        assert b1["val_raw_samples"] == _VAL_RAW[3]

    def test_wrapper_equivalent(self, tmp_path):
        root = tmp_path / "root"
        _write_ledger(root, [0])
        log_path = tmp_path / "train.log"
        _write_log(log_path, [0])
        assert read_split0_boundary(root, log_path) == read_split_boundary(root, log_path, 0)

    def test_marker_absent_rejected(self, tmp_path):
        """台账有该折但日志无折锚点 ⇒ 拒绝（禁止全量解析误判）。"""
        root = tmp_path / "root"
        _write_ledger(root, [0, 1, 5])
        log_path = tmp_path / "train.log"
        _write_log(log_path, [0, 1])
        with pytest.raises(ValueError, match="锚点"):
            read_split_boundary(root, log_path, 5)

    def test_markerless_log_compat(self, tmp_path):
        """无任何折锚点（T0.5b 合成形态）⇒ 全量解析兼容路径。"""
        log_path = tmp_path / "run.log"
        log_path.write_text(
            _fold_log_section(0).replace("开始训练 Split 0\n", ""), encoding="utf-8"
        )
        actual = parse_actual_train_boundary(log_path, split_index=0)
        assert actual["train_actual_start"] == _TRAIN_BOUNDARY[0]

    def test_seed_disagree_within_fold_rejected(self, tmp_path):
        log_path = tmp_path / "train.log"
        log_path.write_text(_fold_log_section(0, mismatch_seed=True), encoding="utf-8")
        with pytest.raises(ValueError, match="边界不一致"):
            parse_actual_train_boundary(log_path, split_index=0)


# ---------------------------------------------------------------------------
# 4. 折级断点续跑
# ---------------------------------------------------------------------------


def _completed_fold_env(tmp_path: Path) -> Tuple[Path, Path, Dict[str, object]]:
    """一套已完成折产物的合成环境；返回 (arm_dir, root, producer_identity)。"""
    prod = _make_production_root(tmp_path)
    state = _ExecutorState()
    evidence = run_arm(**_run_kwargs(prod, tmp_path, state, skip_existing=False))
    assert evidence["verdict"] == "PASS"
    arm_dir = tmp_path / "frozen" / "t5_test"
    fingerprint = json.loads(
        (arm_dir / "folds" / "split00" / "fingerprint.json").read_text(encoding="utf-8")
    )
    return arm_dir, arm_dir / "root", fingerprint["producer_identity"]


class TestFoldResume:
    def test_reuse_when_identity_consistent(self, tmp_path):
        arm_dir, root, producer = _completed_fold_env(tmp_path)
        record = load_fold_record_for_resume(arm_dir, root, 0, producer)
        assert record is not None and record["reused"] is True
        assert record["model_version"] == 1

    def test_missing_fold_returns_none(self, tmp_path):
        arm_dir, root, producer = _completed_fold_env(tmp_path)
        assert load_fold_record_for_resume(arm_dir, root, 9, producer) is None

    def test_partial_artifacts_rejected(self, tmp_path):
        arm_dir, root, producer = _completed_fold_env(tmp_path)
        (arm_dir / "folds" / "split00" / "boundary.json").unlink()
        with pytest.raises(RuntimeError, match="部分存在"):
            load_fold_record_for_resume(arm_dir, root, 0, producer)

    def test_tampered_model_bytes_rejected(self, tmp_path):
        arm_dir, root, producer = _completed_fold_env(tmp_path)
        target = root / "models" / "stock_selection" / "v1_model.joblib"
        target.write_bytes(b"tampered")
        with pytest.raises(ValueError, match="模型字节"):
            load_fold_record_for_resume(arm_dir, root, 0, producer)

    def test_tampered_producer_identity_rejected(self, tmp_path):
        arm_dir, root, producer = _completed_fold_env(tmp_path)
        drifted = dict(producer, config_digest="0" * 64)
        with pytest.raises(ValueError, match="生产者身份"):
            load_fold_record_for_resume(arm_dir, root, 0, drifted)

    def test_full_resume_reuses_everything(self, tmp_path):
        """端到端续跑：全部折复用，训练/回放/打分零调用，verdict PASS。"""
        prod = _make_production_root(tmp_path)
        state1 = _ExecutorState()
        first = run_arm(**_run_kwargs(prod, tmp_path, state1, skip_existing=False))
        assert first["verdict"] == "PASS"
        state2 = _ExecutorState()
        second = run_arm(**_run_kwargs(prod, tmp_path, state2, skip_existing=True))
        assert second["verdict"] == "PASS"
        assert state2.train_calls == [] and state2.replay_calls == 0 and state2.score_calls == []
        folds = second["execution"]["folds"]  # type: ignore[index]
        assert all(fold["reused"] for fold in folds)
        assert second["baselines"]["resumed"] is True  # type: ignore[index]

    def test_baseline_missing_rejected_before_snapshot_write(self, tmp_path):
        """有缓存但基线缺失 ⇒ 拒绝（独立于 --skip-existing）且先于任何快照写入。"""
        prod = _make_production_root(tmp_path)
        run_dir = tmp_path / "frozen" / "t5_test"
        cached = run_dir / "folds" / "split00"
        cached.mkdir(parents=True)
        (cached / "fingerprint.json").write_text("{}", encoding="utf-8")
        for skip in (False, True):
            with pytest.raises(RuntimeError, match="基线缺失"):
                acquire_arm_baselines(run_dir, prod, skip)
            assert not (run_dir / "before_guard.json").exists()
            assert not (run_dir / "consumption_snapshot.json").exists()

    def test_baseline_drift_rejected_on_resume(self, tmp_path):
        """续跑基线即时复核：生产模型目录漂移 ⇒ 拒绝。"""
        prod = _make_production_root(tmp_path)
        state1 = _ExecutorState()
        run_arm(**_run_kwargs(prod, tmp_path, state1, skip_existing=False))
        (prod / "models" / "stock_selection" / "sentinel.txt").write_text(
            "drifted", encoding="utf-8"
        )
        with pytest.raises(RuntimeError, match="续跑基线漂移"):
            acquire_arm_baselines(tmp_path / "frozen" / "t5_test", prod, True)

    # -- R2-T5A-R1-05：五件套续跑三态（缺失重建 / 哈希漂移拒绝 / 齐备复用） --

    def test_missing_predictions_triggers_rebuild(self, tmp_path):
        """删 predictions.parquet 续跑 ⇒ 该折判不完整走显式重建（score_fn 再调）。

        复现形态「删 predictions 后 --skip-existing 仍 PASS」必须消失：重建后
        五件套再生，verdict PASS（显式重打分，非静默复用）。
        """
        prod = _make_production_root(tmp_path)
        state1 = _ExecutorState()
        first = run_arm(**_run_kwargs(prod, tmp_path, state1, skip_existing=False))
        assert first["verdict"] == "PASS"
        target = tmp_path / "frozen" / "t5_test" / "folds" / "split00" / "predictions.parquet"
        assert target.exists()
        target.unlink()
        state2 = _ExecutorState()
        second = run_arm(**_run_kwargs(prod, tmp_path, state2, skip_existing=True))
        assert second["verdict"] == "PASS"
        assert state2.train_calls == [] and state2.replay_calls == 0
        assert state2.score_calls == [1]  # 仅 split00（v1）重打分
        folds = second["execution"]["folds"]  # type: ignore[index]
        assert [fold["reused"] for fold in folds] == [False, True, True]
        assert target.exists()  # 重建再生

    def test_missing_signal_topk_triggers_rebuild(self, tmp_path):
        """缺 signal_topk.parquet 同样判不完整走重建。"""
        arm_dir, root, producer = _completed_fold_env(tmp_path)
        (arm_dir / "folds" / "split01" / "signal_topk.parquet").unlink()
        assert load_fold_record_for_resume(arm_dir, root, 1, producer) is None
        # 齐备折不受影响（五件 + 哈希一致 ⇒ 复用）
        record = load_fold_record_for_resume(arm_dir, root, 0, producer)
        assert record is not None and record["reused"] is True

    def test_tampered_predictions_hash_rejected(self, tmp_path):
        """predictions.parquet 字节漂移 ⇒ 拒绝（证据篡改语义）。"""
        arm_dir, root, producer = _completed_fold_env(tmp_path)
        target = arm_dir / "folds" / "split00" / "predictions.parquet"
        target.write_bytes(b"tampered")
        with pytest.raises(ValueError, match="篡改"):
            load_fold_record_for_resume(arm_dir, root, 0, producer)

    def test_tampered_signal_topk_hash_rejected(self, tmp_path):
        """signal_topk.parquet 字节漂移 ⇒ 同样拒绝。"""
        arm_dir, root, producer = _completed_fold_env(tmp_path)
        target = arm_dir / "folds" / "split00" / "signal_topk.parquet"
        target.write_bytes(b"tampered")
        with pytest.raises(ValueError, match="篡改"):
            load_fold_record_for_resume(arm_dir, root, 0, producer)


# ---------------------------------------------------------------------------
# 5. 承接① sleeve_identity 双注入接线（组装面）
# ---------------------------------------------------------------------------


def _bare_root(tmp_path: Path) -> Path:
    """组装测试用裸根（仅域解析所需 stock_basic）。"""
    root = tmp_path / "iso"
    (root / "clean").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ts_code": ["000001.SH"], "market": ["主板"]}).to_parquet(
        root / "clean" / "stock_basic.parquet", index=False
    )
    return root


class TestAssemblySleeveWiring:
    def test_double_injection_cs_train(self, tmp_path):
        """臂②组装：host 与 trainer 同值注入（承接①），回测参数生产口径逐项。"""
        assembly = assemble_arm_orchestrator(
            ARM_V2_CS_TRAIN_VERIFY,
            _bare_root(tmp_path),
            tmp_path / "runs",
            run_name="t5_asm",
            code_state={"git_commit": "t", "git_dirty": False},
        )
        expected = ("value", 1)
        assert assembly.host.sleeve_identity == expected
        assert assembly.trainer.sleeve_identity == expected
        assert assembly.source_mode == "cs_train_verify"
        assert assembly.store is None
        bt = assembly.host.backtest
        assert bt.bt_top_n == 20 and bt.stagger_tranches == 2
        assert bt.position_sizing == "half_kelly"
        assert bt.kelly_vol_window == 60 and bt.kelly_max_leverage == 0.2
        assert bt.bt_max_weight_per_stock == 0.15
        assert bt.bt_initial_capital == 1_000_000.0
        assert bt.bt_sell_timing == "open" and bt.bt_exclude_st is True
        assert bt.bt_min_list_days == 365
        assert bt.bt_rebalance_freq == assembly.spec.holding_period_days == 20
        assert assembly.host.runs_batch_writer is not None
        assert assembly.host.export_topk_details is True
        assert assembly.config.label_name == "neu_y_ret_20"
        evidence = assembly.gate_evidence
        assert evidence["feature_declaration_gate"]["order_fingerprint"] == (
            VALUE_DECLARATION_ORDER_FINGERPRINT
        )
        assert evidence["deploy_train"]["enabled"] is False

    def test_panel_arm_assembly(self, tmp_path):
        """臂③组装：panel 源 + PanelDataStore + acceptance_purpose None。"""
        from src.lazybull.v2.store.data_store import PanelDataStore

        assembly = assemble_arm_orchestrator(
            ARM_V2_PANEL,
            _bare_root(tmp_path),
            tmp_path / "runs",
            run_name="t5_asm_panel",
            code_state={"git_commit": "t", "git_dirty": False},
        )
        assert assembly.source_mode == "panel"
        assert isinstance(assembly.store, PanelDataStore)
        assert assembly.trainer.sleeve_identity == assembly.host.sleeve_identity

    def test_signal_factory_binds_spec_and_domain(self, tmp_path):
        """信号工厂：spec 信号参数 + 域市值边界逐项（MLSignal 构造即断言，不加载模型）。"""
        from src.lazybull.v2.sleeves._registry import get_sleeve_spec

        spec = get_sleeve_spec("value")
        factory = build_signal_factory(spec, tmp_path / "models")
        signal = factory(20)
        assert signal.top_n == 20
        assert signal.min_amount_ma20 == 50000.0
        assert signal.min_total_mv == 500000.0 and signal.max_total_mv == 15000000.0
        assert signal.exclude_financial is True
        assert signal.downside_penalty == 0.0


# ---------------------------------------------------------------------------
# 6. 承接②持有期断言正反例 / 承接③启动门正反例
# ---------------------------------------------------------------------------


class TestHoldingPeriodAssert:
    def test_real_spec_passes(self):
        from src.lazybull.v2.sleeves._registry import get_sleeve_spec

        assert assert_holding_period_matches_label(get_sleeve_spec("value")) == 20

    def test_mismatch_rejected(self):
        stub = SimpleNamespace(label_name="neu_y_ret_20", holding_period_days=10)
        with pytest.raises(ValueError, match="承接②"):
            assert_holding_period_matches_label(stub)

    def test_unparseable_label_rejected(self):
        stub = SimpleNamespace(label_name="neu_y_ret", holding_period_days=20)
        with pytest.raises(ValueError, match="承接②"):
            assert_holding_period_matches_label(stub)


class TestFeatureDeclarationGate:
    def test_real_spec_passes(self):
        from src.lazybull.v2.sleeves._registry import get_sleeve_spec

        evidence = assert_feature_declaration_gate(get_sleeve_spec("value"))
        assert evidence["declared_column_count"] == 375
        assert evidence["order_fingerprint"] == VALUE_DECLARATION_ORDER_FINGERPRINT

    def test_set_mismatch_rejected(self):
        """集合不等（缺一列）⇒ 拒绝；按关闭族清单子集比对的退化形态同样拒绝。"""
        from src.lazybull.v2.sleeves._registry import get_sleeve_spec

        spec = get_sleeve_spec("value")
        stub = SimpleNamespace(feature_columns=tuple(spec.feature_columns[:-1]))
        with pytest.raises(ValueError, match="承接③启动门拒绝"):
            assert_feature_declaration_gate(stub)

    def test_order_mismatch_rejected(self):
        """集合相等但内部换位 ⇒ 序指纹不符拒绝。"""
        from src.lazybull.v2.sleeves._registry import get_sleeve_spec

        spec = get_sleeve_spec("value")
        columns = list(spec.feature_columns)
        columns[0], columns[1] = columns[1], columns[0]
        stub = SimpleNamespace(feature_columns=tuple(columns))
        with pytest.raises(ValueError, match="有序声明指纹"):
            assert_feature_declaration_gate(stub)


# ---------------------------------------------------------------------------
# 7. 承接④ 四族 derive 调用点静态扫描
# ---------------------------------------------------------------------------


class TestDeriveCallSiteScan:
    def test_real_tree_matches_whitelist(self):
        """真实 v2 树扫描 == 已登记白名单（T3 守卫位置；漂移即失败）。"""
        scan = scan_runtime_derive_call_sites()
        assert_derive_call_sites_registered(scan)  # 不抛即通过
        assert len(scan["call_sites"]) == 10
        assert len(scan["import_sites"]) == 6

    def test_report_collects_details(self):
        report = collect_derive_call_site_report()
        assert report["whitelist_match"] is True
        files = {item["file"] for item in report["details"]}
        assert "src/lazybull/v2/train/core/prepare_families.py" in files
        assert "src/lazybull/v2/train/walk_forward/split_training.py" in files

    def test_drift_detected_on_tampered_scan(self):
        scan = scan_runtime_derive_call_sites()
        tampered = dict(scan)
        tampered["call_sites"] = list(scan["call_sites"]) + [
            "src/lazybull/v2/train/core/prepare_families.py|deriver.derive_fn"
        ]
        with pytest.raises(ValueError, match="调用点漂移"):
            assert_derive_call_sites_registered(tampered)

    def test_synthetic_new_call_site_captured(self, tmp_path, monkeypatch):
        """合成新调用点 ⇒ 扫描捕获正确符号与族归属（白名单外新增必现形）。"""
        from scripts.v2_p3 import t5_arm_artifacts

        # 固定为合成仓外扫描场景，兼容 --basetemp 位于真实仓库 temp/ 内。
        monkeypatch.setattr(t5_arm_artifacts, "_REPO_ROOT", tmp_path / "synthetic_repo")
        v2_root = tmp_path / "v2"
        v2_root.mkdir()
        (v2_root / "evil.py").write_text(
            "from src.lazybull.factors.holdertrade import derive_holdertrade_columns\n"
            "def f(df, src):\n"
            "    return derive_holdertrade_columns(df, src)\n",
            encoding="utf-8",
        )
        scan = scan_runtime_derive_call_sites(v2_root)
        assert scan["import_sites"] == ["evil.py|derive_holdertrade_columns"]
        assert scan["call_sites"] == ["evil.py|derive_holdertrade_columns"]
        assert scan["details"][0]["family"] == "holdertrade"


# ---------------------------------------------------------------------------
# 8. runs_batch_writer 适配闭包
# ---------------------------------------------------------------------------


class TestRunsBatchWriterAdapter:
    def _results(self) -> List[Dict[str, object]]:
        return [
            {
                "split_index": 0,
                "model_version": 1,
                "train_start": "20200102",
                "train_end": "20221231",
                "test_start": _DAYS[0],
                "test_end": _DAYS[1],
                "bt_metrics": {"bt_total_return": 0.002, "bt_start": _DAYS[0], "bt_end": _DAYS[1]},
                "_nav_curve": _synthetic_nav(0),
                "_trades": pd.DataFrame(),
                "_execution_attribution": pd.DataFrame(),
                "_holdings_snapshot": pd.DataFrame(),
            }
        ]

    def test_extra_meta_notes_passthrough(self, tmp_path):
        writer = build_runs_batch_writer(
            runs_root=tmp_path / "runs",
            config={"label_column": "neu_y_ret_20"},
            code_state={"git_commit": "t", "git_dirty": False},
            arms=[ARM_V2_CS_TRAIN_VERIFY],
            source="v2_p3_t5_test",
        )
        batch_dir = writer(
            results=self._results(),
            wf_run_id="wf_adapter_test",
            data_state_payload={"data_state_id": "ds_test"},
            extra_meta_notes={"sleeve_id": "value", "spec_version": "1"},
        )
        batch_meta = json.loads((batch_dir / "batch_meta.json").read_text(encoding="utf-8"))
        assert batch_meta["meta_notes"]["sleeve_id"] == "value"
        assert batch_meta["meta_notes"]["spec_version"] == "1"
        assert batch_meta["arms"] == [ARM_V2_CS_TRAIN_VERIFY]

    def test_missing_nav_curve_rejected(self, tmp_path):
        writer = build_runs_batch_writer(
            runs_root=tmp_path / "runs",
            config={},
            code_state={"git_commit": "t", "git_dirty": False},
            arms=["x"],
            source="t",
        )
        results = self._results()
        results[0]["_nav_curve"] = None
        with pytest.raises(ValueError, match="净值曲线"):
            writer(
                results=results,
                wf_run_id="wf_adapter_bad",
                data_state_payload={"data_state_id": "ds_test"},
            )


# ---------------------------------------------------------------------------
# 9. 配置摘要 / arm_meta / 批指纹 / 批次目录状态
# ---------------------------------------------------------------------------


class TestConfigDigestAndMeta:
    def test_legacy_payload_no_split_flag(self):
        payload = build_arm_config_payload(ARM_LEGACY_CS_TRAIN, 14, "20260105", 6, range(14))
        assert "--selected-split-indices" not in payload["walk_forward_args"]
        assert payload["v2_replay"]["skip_training"] is True
        assert payload["v2_replay"]["acceptance_purpose"] == "g_code_parity"

    def test_v2_payload_spec_driven(self):
        payload = build_arm_config_payload(ARM_V2_PANEL, 14, "20260105", 6, range(14))
        config = payload["runs_batch_config"]
        # P3-1 第三通道撤销：runs config 不写袖子键（身份只走模型 metadata + batch_meta）
        assert "sleeve_id" not in config and "spec_version" not in config
        assert config["feature_source"] == "panel"
        assert config["bt_top_n"] == 20 and config["bt_rebalance_freq"] == 20

    def test_digest_stable_and_subset_sensitive(self):
        args_a = (14, "20260105", 6, tuple(range(14)))
        digest_a = arm_config_digest(build_arm_config_payload(ARM_V2_CS_TRAIN_VERIFY, *args_a))
        digest_b = arm_config_digest(build_arm_config_payload(ARM_V2_CS_TRAIN_VERIFY, *args_a))
        assert digest_a == digest_b
        digest_c = arm_config_digest(
            build_arm_config_payload(ARM_V2_CS_TRAIN_VERIFY, 14, "20260105", 6, (0,))
        )
        assert digest_c != digest_a

    def test_backtest_params_production_overrides(self):
        """BacktestHostParams 显式生产口径逐项（默认值 ≠ 生产，§19.0-4 清单）。"""
        from src.lazybull.v2.sleeves._registry import get_sleeve_spec

        spec = get_sleeve_spec("value")
        bt = build_backtest_host_params(spec)
        assert bt.bt_top_n == 20 and bt.stagger_tranches == 2
        assert bt.position_sizing == "half_kelly"
        assert bt.kelly_vol_window == 60 and bt.kelly_max_leverage == 0.2
        assert bt.bt_max_weight_per_stock == 0.15
        assert bt.bt_initial_capital == 1_000_000.0
        assert bt.bt_sell_timing == "open"
        assert bt.bt_exclude_st is True and bt.bt_min_list_days == 365
        assert bt.bt_rebalance_freq == 20  # 承接②：= spec.holding_period_days
        config = build_runs_batch_config(spec, bt, "cs_train_verify")
        assert config["n_estimators"] == 500 and config["algorithm"] == "xgboost"
        assert config["ensemble_seeds"] == "42,61,82"
        assert config["deploy_train"] is False

    def test_arm_meta_resume_and_drift(self, tmp_path):
        meta = {
            "arm": ARM_V2_CS_TRAIN_VERIFY,
            "producer_identity": {"config_digest": "x"},
            "sleeve_identity": None,
            "input_snapshot_digest": "y",
            "split_count": 3,
            "final_date": "20260105",
            "test_window_months": 6,
            "selected_split_indices": [0, 1, 2],
        }
        write_arm_meta(tmp_path, meta)
        write_arm_meta(tmp_path, meta)  # 幂等沿用
        drifted = dict(meta, split_count=4)
        with pytest.raises(RuntimeError, match="身份键不一致"):
            write_arm_meta(tmp_path, drifted)


class TestBatchFingerprintAndState:
    def test_batch_fingerprint_spec(self):
        fp0 = fingerprint_prediction_rows([("20250102", "000001.SH", 1.0)])
        fp1 = fingerprint_prediction_rows([("20250104", "000001.SH", 2.0)])
        doc = build_batch_fingerprint_doc({1: fp1, 0: fp0})
        assert list(doc["folds"]) == ["split00", "split01"]  # 折号升序
        joined = "\n".join([fp0, fp1]) + "\n"
        assert doc["batch_fingerprint"] == hashlib.sha256(joined.encode()).hexdigest()

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="空批次"):
            build_batch_fingerprint_doc({})

    def test_runs_batch_state(self, tmp_path):
        runs = tmp_path / "runs"
        assert runs_batch_state(runs, [0]) == "empty"
        # 不完整目录（缺 batch_meta/summary）⇒ 拒绝
        bad = runs / "wf_x"
        bad.mkdir(parents=True)
        with pytest.raises(RuntimeError, match="不完整"):
            runs_batch_state(runs, [0])
        # 多目录 ⇒ 拒绝
        (runs / "wf_y").mkdir()
        with pytest.raises(RuntimeError, match="不唯一"):
            runs_batch_state(runs, [0])

    def test_runs_batch_split_mismatch(self, tmp_path):
        prod = _make_production_root(tmp_path)
        state = _ExecutorState()
        run_arm(**_run_kwargs(prod, tmp_path, state, selected_split_indices=[0]))
        # 折集不符（批次只含 split0，本次选定 0/1）⇒ 拒绝混用
        with pytest.raises(RuntimeError, match="折集"):
            runs_batch_state(tmp_path / "frozen" / "t5_test" / "runs", [0, 1])


# ---------------------------------------------------------------------------
# 10. double-run 四门比对接线
# ---------------------------------------------------------------------------


def _make_fp_arm_dir(base: Path, bump: float = 0.0, days: Optional[List[str]] = None) -> Path:
    """手工构造臂级指纹产物（compare_double_run 直接单测用）。"""
    days = days or ["20250102"]
    fold_fps: Dict[int, str] = {}
    for split in (0, 1):
        rows = [(day, "000001.SH", 1.0 + split + bump) for day in days]
        fp = fingerprint_prediction_rows(rows)
        directory = base / "folds" / f"split{split:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "fingerprint.json").write_text(
            json.dumps(
                {
                    "fingerprint": fp,
                    "n_rows": len(rows),
                    "score_days": days,
                    "per_seed_diagnostic": [
                        {"seed": seed, "n_rows": len(rows), "fingerprint": fp}
                        for seed in (42, 61, 82)
                    ],
                }
            ),
            encoding="utf-8",
        )
        fold_fps[split] = fp
    doc = build_batch_fingerprint_doc(fold_fps)
    (base / "batch_fingerprint.json").write_text(json.dumps(doc), encoding="utf-8")
    return base


class TestDoubleRunGates:
    def test_compare_pass(self, tmp_path):
        run1 = _make_fp_arm_dir(tmp_path / "run1")
        run2 = _make_fp_arm_dir(tmp_path / "run2")
        comparison = compare_double_run(run1, run2, [], [])
        assert comparison["verdict"] == "PASS"
        assert comparison["ensemble_fingerprint_match"] is True
        assert comparison["per_seed_fingerprint_match"] is True
        assert comparison["coverage_match"] is True

    def test_compare_fingerprint_tamper_fail(self, tmp_path):
        run1 = _make_fp_arm_dir(tmp_path / "run1")
        run2 = _make_fp_arm_dir(tmp_path / "run2", bump=1000.0)
        comparison = compare_double_run(run1, run2, [], [])
        assert comparison["verdict"] == "FAIL"
        assert comparison["ensemble_fingerprint_match"] is False

    def test_compare_coverage_tamper_fail(self, tmp_path):
        run1 = _make_fp_arm_dir(tmp_path / "run1", days=["20250102"])
        run2 = _make_fp_arm_dir(tmp_path / "run2", days=["20250103"])
        comparison = compare_double_run(run1, run2, [], [])
        assert comparison["verdict"] == "FAIL"
        assert comparison["coverage_match"] is False

    def test_compare_guard_drift_fail(self, tmp_path):
        run1 = _make_fp_arm_dir(tmp_path / "run1")
        run2 = _make_fp_arm_dir(tmp_path / "run2")
        comparison = compare_double_run(run1, run2, ["文件变化: x"], [])
        assert comparison["verdict"] == "FAIL"

    def test_end_to_end_double_run_pass(self, tmp_path):
        """run_arm 双跑端到端（合成 executor 同码两遍）：四门全过 + run1/run2 完整布局。"""
        prod = _make_production_root(tmp_path)
        state = _ExecutorState()
        evidence = run_arm(
            **_run_kwargs(
                prod,
                tmp_path,
                state,
                double_run=True,
                selected_split_indices=[0],
                run_name="t5_double",
            )
        )
        assert evidence["verdict"] == "PASS"
        comparison = evidence["comparison"]
        assert comparison["ensemble_fingerprint_match"] is True
        assert len(state.train_calls) == 2  # run1/run2 各训一次
        for idx in (1, 2):
            sub = tmp_path / "frozen" / "t5_double" / f"run{idx}"
            for name in (
                "arm_meta.json",
                "batch_fingerprint.json",
                "chain_metrics.json",
                "after_guard.json",
                "consumption_verify.json",
            ):
                assert (sub / name).exists(), name
            assert (sub / "folds" / "split00" / "fingerprint.json").exists()
            assert (sub / "root" / "runs").is_dir()

    def test_end_to_end_double_run_tamper_fail(self, tmp_path):
        prod = _make_production_root(tmp_path)
        state = _ExecutorState()
        evidence = run_arm(
            **_run_kwargs(
                prod,
                tmp_path,
                state,
                double_run=True,
                selected_split_indices=[0],
                run_name="t5_double_bad",
                executor=_make_synthetic_executor(state, tamper_run2=True),
            )
        )
        assert evidence["verdict"] == "FAIL"


# ---------------------------------------------------------------------------
# 11. B0' 冻结落盘结构
# ---------------------------------------------------------------------------


class TestB0PrimeFreeze:
    def _run_legacy_full(self, tmp_path: Path) -> Dict[str, object]:
        prod = _make_production_root(tmp_path)
        state = _ExecutorState()
        return run_arm(
            **_run_kwargs(
                prod,
                tmp_path,
                state,
                arm=ARM_LEGACY_CS_TRAIN,
                split_count=FULL_SPLIT_COUNT,
                run_name="t5_legacy_full",
            )
        )

    def test_freeze_structure(self, tmp_path):
        evidence = self._run_legacy_full(tmp_path)
        assert evidence["verdict"] == "PASS"
        b0p = tmp_path / "frozen" / B0_PRIME_DIR_NAME
        assert evidence["b0_prime"] == str(b0p)
        assert (b0p / "chain_metrics.json").exists()
        assert (b0p / "batch_fingerprint.json").exists()
        manifest = json.loads((b0p / "freeze_manifest.json").read_text(encoding="utf-8"))
        assert manifest["source_run_name"] == "t5_legacy_full"
        assert manifest["arm"] == ARM_LEGACY_CS_TRAIN
        assert manifest["config_digest"] == evidence["config_digest"]
        assert manifest["data_state_id"] == "ds_synthetic"
        assert manifest["code_state"] == {"git_commit": "test", "git_dirty": False}
        assert len(manifest["folds"]) == FULL_SPLIT_COUNT
        fold0 = manifest["folds"]["split00"]
        assert fold0["model_version"] == 1
        assert fold0["daily_nav_path"].endswith("folds/split00/daily.parquet")
        assert manifest["runs_batch"]["chain_nav_path"].endswith("chain_nav.parquet")
        assert manifest["chain_metrics"]["trading_days"] is not None
        for split in range(FULL_SPLIT_COUNT):
            fold_dir = b0p / "folds" / f"split{split:02d}"
            assert (fold_dir / "fingerprint.json").exists()
            assert (fold_dir / "boundary.json").exists()

    def test_freeze_idempotent_and_conflict_rejected(self, tmp_path):
        self._run_legacy_full(tmp_path)
        b0p = tmp_path / "frozen" / B0_PRIME_DIR_NAME
        manifest_path = b0p / "freeze_manifest.json"
        before = manifest_path.read_text(encoding="utf-8")
        # 幂等：同源重冻 ⇒ 复用不改写
        prod = tmp_path / "data"
        state = _ExecutorState()
        run_arm(
            **_run_kwargs(
                prod,
                tmp_path,
                state,
                arm=ARM_LEGACY_CS_TRAIN,
                split_count=FULL_SPLIT_COUNT,
                run_name="t5_legacy_full",
                skip_existing=True,
            )
        )
        assert manifest_path.read_text(encoding="utf-8") == before
        # 身份不符 ⇒ 拒绝覆盖
        manifest = json.loads(before)
        manifest["batch_fingerprint"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(RuntimeError, match="禁止覆盖"):
            run_arm(
                **_run_kwargs(
                    prod,
                    tmp_path,
                    state,
                    arm=ARM_LEGACY_CS_TRAIN,
                    split_count=FULL_SPLIT_COUNT,
                    run_name="t5_legacy_full",
                    skip_existing=True,
                )
            )

    def test_no_freeze_for_subset_or_other_arms(self, tmp_path):
        prod = _make_production_root(tmp_path)
        state = _ExecutorState()
        evidence = run_arm(
            **_run_kwargs(
                prod,
                tmp_path,
                state,
                arm=ARM_LEGACY_CS_TRAIN,
                split_count=FULL_SPLIT_COUNT,
                selected_split_indices=[0, 1],
                run_name="t5_legacy_subset",
            )
        )
        assert evidence["verdict"] == "PASS"
        assert evidence["b0_prime"] is None
        assert not (tmp_path / "frozen" / B0_PRIME_DIR_NAME).exists()


# ---------------------------------------------------------------------------
# 12. 布局契约字段完整性（run_arm 单跑端到端合成驱动）
# ---------------------------------------------------------------------------


class TestLayoutContract:
    def test_layout_completeness(self, tmp_path):
        prod = _make_production_root(tmp_path)
        state = _ExecutorState()
        evidence = run_arm(**_run_kwargs(prod, tmp_path, state))
        assert evidence["verdict"] == "PASS"
        arm_dir = tmp_path / "frozen" / "t5_test"

        # 顶层文件
        for name in (
            "arm_meta.json",
            "before_guard.json",
            "consumption_snapshot.json",
            "after_guard.json",
            "consumption_verify.json",
            "batch_fingerprint.json",
            "chain_metrics.json",
        ):
            assert (arm_dir / name).exists(), name

        # arm_meta 字段
        arm_meta = json.loads((arm_dir / "arm_meta.json").read_text(encoding="utf-8"))
        assert arm_meta["arm"] == ARM_V2_CS_TRAIN_VERIFY
        for key in ("source_digest", "environment_identity", "config_digest"):
            assert key in arm_meta["producer_identity"]
        assert arm_meta["sleeve_identity"] == {"sleeve_id": "value", "spec_version": 1}
        assert arm_meta["input_snapshot_digest"]
        assert arm_meta["deploy_train"]["enabled"] is False

        # 隔离根：junctions 替身 + 全新目录（含 runs/）
        root = arm_dir / "root"
        for name in (
            "clean",
            "raw",
            "features",
            "labels",
            "models",
            "walk_forward",
            "reports",
            "paper",
            "runs",
        ):
            assert (root / name).is_dir(), name

        # 逐折五件套（R2-T5A-R1-05：三件套 + 预测行两套件真实落盘）
        for split in range(3):
            directory = arm_dir / "folds" / f"split{split:02d}"
            for name in (
                "model_version.txt",
                "boundary.json",
                "fingerprint.json",
                "predictions.parquet",
                "signal_topk.parquet",
            ):
                assert (directory / name).exists(), name
            assert (directory / "model_version.txt").read_text().strip() == str(split + 1)
            boundary = json.loads((directory / "boundary.json").read_text(encoding="utf-8"))
            assert boundary["test_start"] == _fold_test_window(split)[0]
            fingerprint = json.loads((directory / "fingerprint.json").read_text(encoding="utf-8"))
            assert fingerprint["model_identity"]
            assert fingerprint["producer_identity"]["arm"] == ARM_V2_CS_TRAIN_VERIFY
            # R2-T5A-06 锚点接线：fingerprint.json 登记为相对臂目录 POSIX 形态
            assert fingerprint["predictions_path"] == f"folds/split{split:02d}/predictions.parquet"
            assert fingerprint["predictions_rows"] == 4
            # 信号 top-k 产物增键（signal_selection 接线）
            assert fingerprint["signal_topk_path"] == f"folds/split{split:02d}/signal_topk.parquet"
            assert fingerprint["signal_topk_rows"] == 4
            assert fingerprint["signal_topk_days"] == list(_fold_test_window(split))
            assert fingerprint["signal_topk_empty_days"] == []

        # 批指纹 / runs 批 / 链式指标
        batch_doc = json.loads((arm_dir / "batch_fingerprint.json").read_text(encoding="utf-8"))
        assert set(batch_doc["folds"]) == {"split00", "split01", "split02"}
        assert len(batch_doc["batch_fingerprint"]) == 64
        runs_dirs = [d for d in (arm_dir / "runs").iterdir() if d.is_dir()]
        assert len(runs_dirs) == 1
        assert (runs_dirs[0] / "batch_meta.json").exists()
        assert (runs_dirs[0] / "chain_nav.parquet").exists()
        chain = json.loads((arm_dir / "chain_metrics.json").read_text(encoding="utf-8"))
        for key in ("cagr", "max_drawdown", "sharpe"):
            assert key in chain
        assert chain["implementation"].endswith("chain_metrics.calculate_chain_metrics")

        # 训练路径（P3-5 统一产批）：臂②训练不写批，runs 批由全折回放统一产出
        assert state.train_calls == [([0, 1, 2], False)]
        assert state.replay_calls == 1
        # R2-T5A-06：逐折打分统一携带预测行持久化接线（fold 目录 predictions.parquet）
        assert len(state.score_persist) == 3
        for split, (persist, pred_path, anchor) in enumerate(state.score_persist):
            assert persist is True
            pred = Path(str(pred_path))
            assert pred.name == "predictions.parquet"
            assert pred.parent.name == f"split{split:02d}" and pred.parent.parent.name == "folds"
            assert Path(str(anchor)) == arm_dir
        # signal_selection 三臂统一映射（spec 信号参数 + 域市值边界同源）
        expected_selection = {
            "top_n": 20,
            "min_amount_ma20": 50000.0,
            "exclude_financial": True,
            "min_total_mv": 500000.0,
            "max_total_mv": 15000000.0,
        }
        assert state.score_selection == [expected_selection] * 3
        # 承接④证据段 + 非 panel 臂打分消费核对 skipped 登记（R2-T5A-02 后半）
        assert evidence["derive_call_sites"]["whitelist_match"] is True
        receipt_check = evidence["execution"]["score_receipt_check"]  # type: ignore[index]
        assert receipt_check["skipped"] is True

    def test_legacy_arm_flow_uses_replay(self, tmp_path):
        """臂①：训练不写批 + skip 回放产出批次（start_model_version 反解 = 1）。"""
        prod = _make_production_root(tmp_path)
        state = _ExecutorState()
        evidence = run_arm(
            **_run_kwargs(prod, tmp_path, state, arm=ARM_LEGACY_CS_TRAIN, run_name="t5_legacy")
        )
        assert evidence["verdict"] == "PASS"
        assert state.train_calls == [([0, 1, 2], False)]
        assert state.replay_calls == 1
        replay = evidence["execution"]["replay"]  # type: ignore[index]
        assert replay["start_model_version"] == 1
        arm_meta = json.loads(
            (tmp_path / "frozen" / "t5_legacy" / "arm_meta.json").read_text(encoding="utf-8")
        )
        assert arm_meta["sleeve_identity"] is None  # 旧臂无 spec 身份（回放批次经 batch_meta 携带）

    def test_resume_after_mid_training_crash(self, tmp_path):
        """折级续跑：首跑中断（fold 0 已注册）⇒ 续跑补训 [1,2]（不写批）+ 全折回放 + 全折打分。"""
        prod = _make_production_root(tmp_path)
        crash_state = _ExecutorState()

        def crashing_train(ctx, subset, write_batch):
            _write_ledger(ctx.root, [0])
            _write_model_files(ctx.root, [0])
            _write_log(ctx.log_path, [0])
            raise RuntimeError("模拟中断")

        crashed_executor = ArmExecutor(
            train_fn=crashing_train,
            replay_fn=_make_synthetic_executor(crash_state).replay_fn,
            score_fn=_make_synthetic_executor(crash_state).score_fn,
        )
        with pytest.raises(RuntimeError, match="模拟中断"):
            run_arm(**_run_kwargs(prod, tmp_path, crash_state, executor=crashed_executor))

        state2 = _ExecutorState()
        evidence = run_arm(**_run_kwargs(prod, tmp_path, state2, skip_existing=True))
        assert evidence["verdict"] == "PASS"
        assert state2.train_calls == [([1, 2], False)]  # 补训缺折不写批
        assert state2.replay_calls == 1  # 全折回放统一产批
        assert sorted(state2.score_calls) == [1, 2, 3]  # 三折均首次打分
        folds = evidence["execution"]["folds"]  # type: ignore[index]
        assert [fold["reused"] for fold in folds] == [False, False, False]

    def test_normalize_selected(self):
        assert _normalize_selected(None, 3) == (0, 1, 2)
        assert _normalize_selected([2, 0, 0], 3) == (0, 2)
        with pytest.raises(ValueError, match="越界"):
            _normalize_selected([3], 3)


# ---------------------------------------------------------------------------
# 13. 臂③ 打分接线（panel 模式 _score_fold；stub store，不触真实 PanelDataStore）
# ---------------------------------------------------------------------------


class TestPanelScoreWiring:
    def test_score_fold_panel_kwargs(self, tmp_path, monkeypatch):
        from scripts.v2_p3 import t5_arm_executors as executors_mod

        class _StubStore:
            def load_features(self, query):
                index = pd.MultiIndex.from_tuples(
                    [(day, "000001.SH") for day in _DAYS], names=["trade_date", "ts_code"]
                )
                return SimpleNamespace(df=pd.DataFrame(index=index))

        captured: Dict[str, object] = {}

        def score_fn(**kwargs):
            captured.update(kwargs)
            days = list(kwargs["score_days"])
            rows = [(day, "000001.SH", 1.0) for day in days]
            return {
                "model_version": kwargs["model_version"],
                "fingerprint": fingerprint_prediction_rows(rows),
                "per_seed_diagnostic": [],
                "score_days": days,
                "n_days": len(days),
                "n_rows": len(rows),
            }

        monkeypatch.setattr(
            executors_mod, "_load_model_feature_columns", lambda models_dir, version: ["f1", "f2"]
        )
        root = tmp_path / "arm" / "root"
        _write_ledger(root, [0])
        _write_model_files(root, [0])
        log_path = tmp_path / "train.log"
        _write_log(log_path, [0])
        # 隔离根直建（本用例不经 ensure_arm_root）：复制生产 features/clean
        prod = _make_production_root(tmp_path)
        shutil.copytree(prod / "features", root / "features")
        shutil.copytree(prod / "clean", root / "clean")

        ctx = ArmRunContext(
            arm=ARM_V2_PANEL,
            run_name="t5_panel_unit",
            arm_dir=tmp_path / "arm",
            root=root,
            production_data_root=prod,
            split_count=3,
            final_date="20260105",
            test_window_months=6,
            selected_split_indices=(0,),
            log_path=log_path,
            replay_log_path=tmp_path / "replay.log",
        )
        record = executors_mod._score_fold(
            ctx, 0, {"arm": ARM_V2_PANEL}, score_fn, {"000001.SH"}, _StubStore()
        )
        assert captured["source_mode"] == "panel"
        assert isinstance(captured["store"], _StubStore)
        assert captured["feature_columns"] == ["f1", "f2"]
        # R2-T5A-06 锚点接线：persist 三件套传参（锚点 = 臂目录）
        assert captured["persist_predictions"] is True
        assert captured["predictions_anchor"] == tmp_path / "arm"
        assert Path(str(captured["predictions_path"])).name == "predictions.parquet"
        assert record["model_version"] == 1
        assert (tmp_path / "arm" / "folds" / "split00" / "fingerprint.json").exists()
