# -*- coding: utf-8 -*-
"""P3 T5 阶段 A 评审处置回归（t5_arm_runner / t5_arm_assembly / 袖子身份贯通）。

**全部合成/mock，禁止真实训练与真实生产数据**。逐条对应评审意见：

- R2-T5A-01：默认相对 ``base_dir`` 端到端可执行（入口统一 ``resolve()`` 绝对路径，
  下游不再做相对/绝对混合比较）；
- P2-1 ≡ R2-T5A-08：臂①窗口三参数（split_count/final_date/test_window_months）提升为
  ``build_spike_walk_forward_args`` 形参（默认值与 spike 现状逐位不变），真实命令与
  配置摘要均自 ctx 透传；
- P2-4（§13.3.8）：臂③运行前 source_digest 前置校验（``--compare-arm-dir``）正反例
  （先于任何训练/快照动作）；
- R2-T5A-10：host-only 单侧携带 ⇒ 模型注册 metadata 与 runs ``batch_meta`` 两出口
  同值（真实落盘验证）；trainer-only 同理；双侧不等仍拒绝；bind 写入点校验；
- P3-1：runs 批次 config 不含袖子键（第三通道撤销）；
- P2-5：复杂度锚定（``assemble_arm_orchestrator`` ≤150 行 / runner 文件 ≤800 行）；
- S2-T5A-01：``spec.top_n`` / ``spec.signal_impl`` 消费（替换 spec 值即生效；
  签名不兼容 / 解析失败拒绝）；
- R2-T5A-06：``_score_fold`` 预测行持久化接线（``persist_predictions=True`` +
  ``predictions_path=<fold目录>/predictions.parquet`` + ``predictions_anchor=臂目录``
  逐折传参；端到端 fingerprint.json 登记相对 POSIX 路径）；
- R2-T5A-02 集成接线：``acquire_arm_baselines(arm=...)`` 臂感知（臂③消费集合扩展）、
  臂③打分 receipt ⊆ 消费集合核对（越界 ⇒ verdict FAIL 不抛断）、``manifest_drift``
  并入单/双跑判定、双跑 ``manifest_verify.json`` 同步落盘。

夹具口径复制自 ``tests/test_v2_p3_t5_arm_runner.py`` 与
``tests/test_v2_p3_t5_sleeve_identity.py``（本仓测试文件自包含约定，不跨测试模块导入）。
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v2_p3.t5_arm_artifacts import (  # noqa: E402
    ARM_LEGACY_CS_TRAIN,
    ARM_V2_CS_TRAIN_VERIFY,
    ARM_V2_PANEL,
    resolve_skip_start_model_version,
)
from scripts.v2_p3.t5_arm_assembly import (  # noqa: E402
    build_backtest_host_params,
    build_runs_batch_config,
    build_runs_batch_writer,
    build_signal_factory,
    resolve_signal_class,
)
from scripts.v2_p3.t5_arm_runner import (  # noqa: E402
    ArmExecutor,
    ArmRunContext,
    _score_fold,
    build_arm_config_payload,
    run_arm,
)
from scripts.v2_p3.t05b_fingerprint import fingerprint_prediction_rows  # noqa: E402
from scripts.v2_p3.t05b_spike_runner import build_spike_walk_forward_args  # noqa: E402

# ---------------------------------------------------------------------------
# 合成夹具（口径复制自 tests/test_v2_p3_t5_arm_runner.py；本文件自包含）
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


def _fold_log_section(split_index: int) -> str:
    """单折日志段（锚点 + 三种子切分行）。"""
    lines = [f"00:00:01 | INFO | 开始训练 Split {split_index}\n"]
    for _seed_idx in range(3):
        lines.append(
            "00:00:02 | INFO |   训练集: "
            f"{_TRAIN_BOUNDARY[0]} 至 {_TRAIN_BOUNDARY[1]}（{_TRAIN_BOUNDARY[2]} 个交易日，"
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
        # 逐折记录 (persist_predictions, predictions_path, predictions_anchor)
        self.score_persist: List[Tuple[object, object, object]] = []
        # 信号选择参数映射传参核验（signal_selection，三臂统一）
        self.score_selection: List[object] = []


def _register_predictions(
    result: Dict[str, object],
    kwargs: Dict[str, object],
    rows: List[Tuple[str, str, float]],
    score_days: Sequence[str],
) -> None:
    """R2-T5A-06/R1-05 契约仿真：predictions.parquet + signal_topk.parquet 真实落盘 +
    相对锚点 POSIX 登记（与 ``score_model_fingerprint`` 持久化段同形态）。"""
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


def _make_synthetic_executor(state: _ExecutorState) -> ArmExecutor:
    """合成执行缝：训练 = 台账/模型/日志合成；回放 = 真实版本反解 + 合成 runs 批；
    打分 = 确定性合成指纹（记录持久化接线传参供断言）。"""

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
        rows = [
            (day, "000001.SH", model_version * 1.0 + idx) for day in score_days for idx in range(2)
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
        _register_predictions(result, kwargs, rows, list(score_days))
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
# 1. R2-T5A-01：默认相对 base_dir 端到端可执行（入口统一 resolve 绝对路径）
# ---------------------------------------------------------------------------


class TestRelativeBaseDir:
    def test_relative_base_dir_end_to_end(self, tmp_path, monkeypatch):
        """相对 base_dir/production_data_root/logs_dir ⇒ 全流程可执行（修复前必抛
        「隔离根解析与上下文不一致」——build_isolated_root 返回绝对路径 vs ctx 相对）。"""
        monkeypatch.chdir(tmp_path)
        _make_production_root(tmp_path)  # 创建于 tmp_path/data（相对 ./data 同指）
        state = _ExecutorState()
        evidence = run_arm(
            arm=ARM_V2_CS_TRAIN_VERIFY,
            run_name="t5_rel",
            production_data_root=Path("./data"),
            split_count=3,
            final_date="20260105",
            base_dir=Path("frozen_rel"),  # 相对路径（默认 T5_BASE_DIR 同形态）
            logs_dir=Path("logs_rel"),
            executor=_make_synthetic_executor(state),
            link_fn=_copytree_link,
            verify_links=False,
        )
        assert evidence["verdict"] == "PASS"
        arm_dir = tmp_path / "frozen_rel" / "t5_rel"
        assert (arm_dir / "t5_arm_evidence.json").exists()
        assert (arm_dir / "root").is_dir()
        assert state.train_calls == [([0, 1, 2], False)] and state.replay_calls == 1


# ---------------------------------------------------------------------------
# 2. P2-1 ≡ R2-T5A-08：臂①窗口三参数透传（非默认参数下 args 逐项 == ctx 值）
# ---------------------------------------------------------------------------


class TestLegacyWindowArgsPassthrough:
    def test_real_legacy_train_args_follow_ctx(self, tmp_path, monkeypatch):
        """``_real_legacy_train`` 真实命令 args 逐项 == ctx（stub 掉子进程执行）。"""
        from scripts.v2_p3 import t5_arm_executors as executors_mod

        captured: Dict[str, object] = {}

        def fake_run(root: Path, log_path: Path, args: Optional[List[str]] = None) -> float:
            captured["args"] = list(args or [])
            return 0.0

        monkeypatch.setattr(executors_mod, "_run_walk_forward", fake_run)
        ctx = ArmRunContext(
            arm=ARM_LEGACY_CS_TRAIN,
            run_name="t5_ctx",
            arm_dir=tmp_path / "arm",
            root=tmp_path / "arm" / "root",
            production_data_root=tmp_path / "data",
            split_count=7,
            final_date="20250101",
            test_window_months=3,
            selected_split_indices=tuple(range(7)),
            log_path=tmp_path / "train.log",
            replay_log_path=tmp_path / "replay.log",
        )
        executors_mod._real_legacy_train(ctx, tuple(range(7)), False)
        args = captured["args"]
        assert args[args.index("--split-count") + 1] == "7"
        assert args[args.index("--final-date") + 1] == "20250101"
        assert args[args.index("--test-window-months") + 1] == "3"
        assert "--selected-split-indices" not in args  # 全折不带折子集 flag（生产语义）

    def test_spike_args_non_default_window_params(self):
        """形参扩展：非默认三参数逐项透传（默认值逐位不变由 spike 套件锚定）。"""
        args = build_spike_walk_forward_args(
            Path("<ROOT>"),
            Path("<SUMMARY>"),
            selected_split_indices=("1", "2"),
            split_count=9,
            final_date="20251231",
            test_window_months=4,
        )
        assert args[args.index("--split-count") + 1] == "9"
        assert args[args.index("--final-date") + 1] == "20251231"
        assert args[args.index("--test-window-months") + 1] == "4"
        idx = args.index("--selected-split-indices")
        assert [args[idx + 1], args[idx + 2]] == ["1", "2"]

    def test_config_payload_mirror_carries_window_params(self):
        """配置摘要的臂①参数镜像同样随入参透传（与用户参数不再脱钩）。"""
        payload = build_arm_config_payload(ARM_LEGACY_CS_TRAIN, 9, "20251231", 4, range(9))
        args = payload["walk_forward_args"]
        assert args[args.index("--split-count") + 1] == "9"
        assert args[args.index("--final-date") + 1] == "20251231"
        assert args[args.index("--test-window-months") + 1] == "4"


# ---------------------------------------------------------------------------
# 3. P2-4（§13.3.8）：臂③运行前 source_digest 前置校验（--compare-arm-dir）
# ---------------------------------------------------------------------------


class TestCompareArmDigestGate:
    def test_panel_arm_requires_compare_dir(self, tmp_path):
        """臂③缺 --compare-arm-dir ⇒ 拒绝（先于任何快照/目录写入）。"""
        prod = _make_production_root(tmp_path)
        state = _ExecutorState()
        with pytest.raises(ValueError, match="--compare-arm-dir"):
            run_arm(**_run_kwargs(prod, tmp_path, state, arm=ARM_V2_PANEL))
        assert not (tmp_path / "frozen" / "t5_test").exists()

    def test_missing_arm_meta_rejected(self, tmp_path):
        prod = _make_production_root(tmp_path)
        state = _ExecutorState()
        empty_dir = tmp_path / "arm2_run"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="arm_meta"):
            run_arm(**_run_kwargs(prod, tmp_path, state, compare_arm_dir=empty_dir))
        assert not (tmp_path / "frozen" / "t5_test").exists()

    def test_digest_mismatch_rejected_before_any_write(self, tmp_path):
        """臂② source_digest 与当前代码态不一致 ⇒ 拒绝启动并提示重跑臂②。"""
        prod = _make_production_root(tmp_path)
        state = _ExecutorState()
        compare_dir = tmp_path / "arm2_run"
        compare_dir.mkdir()
        (compare_dir / "arm_meta.json").write_text(
            json.dumps({"producer_identity": {"source_digest": "0" * 64}}), encoding="utf-8"
        )
        with pytest.raises(RuntimeError, match="重跑臂②"):
            run_arm(**_run_kwargs(prod, tmp_path, state, compare_arm_dir=compare_dir))
        # 先于任何训练/快照动作：run 目录与基线快照均未产生
        assert not (tmp_path / "frozen" / "t5_test").exists()
        assert state.train_calls == [] and state.replay_calls == 0 and state.score_calls == []

    def test_digest_match_passes_end_to_end(self, tmp_path):
        """正例：先跑臂②（合成）⇒ 第二次以 compare_arm_dir 指向其 run 目录 ⇒ PASS。"""
        prod = _make_production_root(tmp_path)
        state1 = _ExecutorState()
        first = run_arm(**_run_kwargs(prod, tmp_path, state1, run_name="t5_arm2"))
        assert first["verdict"] == "PASS"
        arm2_meta = json.loads(
            (tmp_path / "frozen" / "t5_arm2" / "arm_meta.json").read_text(encoding="utf-8")
        )
        state2 = _ExecutorState()
        second = run_arm(
            **_run_kwargs(
                prod,
                tmp_path,
                state2,
                run_name="t5_arm3",
                compare_arm_dir=tmp_path / "frozen" / "t5_arm2",
            )
        )
        assert second["verdict"] == "PASS"
        params = second["params"]  # type: ignore[index]
        assert params["compare_source_digest"] == (arm2_meta["producer_identity"]["source_digest"])


# ---------------------------------------------------------------------------
# 4. R2-T5A-10：袖子身份贯通（host-only / trainer-only 两出口同值，真实落盘）
# ---------------------------------------------------------------------------

# 合成数据夹具（口径复制自 tests/test_v2_p3_t5_sleeve_identity.py；本文件自包含）

_CODES = [f"{i:06d}.SH" for i in range(1, 25)]  # 24 只主板合成股
_BASE_COLUMNS = [
    "neu_ret_1",
    "neu_ret_20",
    "neu_ret_5",
    "alpha_industry_20",
    "alpha_industry_5",
    "ind_ret_avg",
    "ind_momentum_rank",
    "zscore_ma_deviation_20",
    "zscore_acceleration",
    "zscore_macd_hist",
    "bb_pct",
    "zscore_turnover_rate",
    "vol_ratio_20",
    "vol_burst_20",
    "zscore_amount_ma20",
    "zscore_net_mf_amount",
    "zscore_elg_net_amount_sum_20",
    "lg_net_amount_sum_5",
    "zscore_volatility_20",
    "zscore_volatility_5",
    "amplitude",
    "zscore_bb_width",
    "upper_shadow",
    "lower_shadow",
    "spec_score",
    "rsi_14",
    "kdj_j",
    "zscore_size",
    "zscore_bp",
    "zscore_dv_ttm",
    "zscore_pe_ttm",
    "is_loss",
    "list_days",
    "mkt_adv_dec_ratio",
    "mkt_ret_avg_20",
    "mkt_turnover_std",
    "mkt_vol_20",
]
_LABEL_COLS = ["y_ret_5", "y_ret_10", "y_ret_20", "neu_y_ret_5", "neu_y_ret_10", "neu_y_ret_20"]
_FILTER_COLS = ["is_st", "is_suspended", "is_limit_up"]
_TRAIN_DAYS = 120  # 外层训练窗交易日数


def _panel_days() -> List[str]:
    return pd.bdate_range("20240102", periods=_TRAIN_DAYS + 40).strftime("%Y%m%d").tolist()


def _cell_value(day_idx: int, code_idx: int, col_idx: int) -> float:
    """确定性特征值。"""
    return round((day_idx + 1) * 0.01 + (code_idx + 1) * 0.001 + (col_idx + 1) * 0.0001, 6)


def _frame_rows(day: str, day_idx: int) -> Dict[str, list]:
    payload: Dict[str, list] = {
        "trade_date": [day] * len(_CODES),
        "ts_code": list(_CODES),
    }
    for col_idx, col in enumerate(_BASE_COLUMNS):
        payload[col] = [_cell_value(day_idx, code_idx, col_idx) for code_idx in range(len(_CODES))]
    for col in _FILTER_COLS:
        payload[col] = [0] * len(_CODES)
    for label_idx, col in enumerate(_LABEL_COLS):
        payload[col] = [
            round(0.02 * (label_idx + 1) + day_idx * 0.001 + code_idx * 0.0001, 6)
            for code_idx in range(len(_CODES))
        ]
    return payload


def _write_base_inputs(root: Path) -> None:
    calendar = _panel_days()
    (root / "clean").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"cal_date": calendar, "is_open": [1] * len(calendar)}).to_parquet(
        root / "clean" / "trade_cal.parquet", index=False
    )
    pd.DataFrame({"ts_code": _CODES, "market": ["主板"] * len(_CODES)}).to_parquet(
        root / "clean" / "stock_basic.parquet", index=False
    )
    daily = root / "clean" / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    for day in calendar:
        pd.DataFrame({"trade_date": [day], "ts_code": [_CODES[0]], "close": [1.0]}).to_parquet(
            daily / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet", index=False
        )


def _build_panel_root(root: Path):
    """panel store 填充（全部族 + 六标签表；readiness 协议列序强校验）。"""
    from src.lazybull.v2.common.types import TradeDate
    from src.lazybull.v2.store.column_groups import LABEL_TABLES, PANEL_GROUPS
    from src.lazybull.v2.store.data_store import PanelDataStore

    _write_base_inputs(root)
    store = PanelDataStore(root)
    store.manifest.register_columns(
        {
            **{col: {"group": "core", "source": "test"} for col in _BASE_COLUMNS},
            **{col: {"group": "core", "source": "test"} for col in _FILTER_COLS},
        }
    )
    store.manifest.save()
    for day_idx, day in enumerate(_panel_days()):
        rows = _frame_rows(day, day_idx)
        for group in PANEL_GROUPS:
            if group == "core":
                core_cols = ("trade_date", "ts_code") + tuple(_BASE_COLUMNS) + tuple(_FILTER_COLS)
                payload = {col: rows[col] for col in core_cols}
            else:
                # readiness 要求八族当日齐备（非 core 族仅键列）
                payload = {"trade_date": rows["trade_date"], "ts_code": rows["ts_code"]}
            store.append_features(TradeDate.from_str(day), group, pd.DataFrame(payload))
        for table_idx, table in enumerate(LABEL_TABLES):
            raw_col = _LABEL_COLS[table_idx]  # y_ret_5/10/20
            neu_col = _LABEL_COLS[table_idx + 3]  # neu_y_ret_5/10/20
            store.append_labels(
                table,
                pd.DataFrame(
                    {
                        "ts_code": rows["ts_code"],
                        "trade_date": rows["trade_date"],
                        "label_value": rows[raw_col],
                        "neu_label_value": rows[neu_col],
                        "maturity_status": ["sealed"] * len(rows["ts_code"]),
                    },
                    columns=[
                        "ts_code",
                        "trade_date",
                        "label_value",
                        "neu_label_value",
                        "maturity_status",
                    ],  # readiness 协议列序强校验
                ),
            )
    return store


def _wf_config():
    from src.lazybull.v2.common.types import ModelConfig

    return ModelConfig(
        model_family="stock_selection",
        feature_columns=tuple(_BASE_COLUMNS + _FILTER_COLS),
        label_name="neu_y_ret_20",
        hyperparams={
            "algorithm": "xgboost",
            "n_estimators": 20,
            "max_depth": 3,
            "early_stopping_rounds": 10,
            "early_stopping_metric": "rank_ic",
            "rank_weight_enabled": False,
            "random_state": 42,
        },
        train_window_months=72,  # 生产口径 6 年;合成日历 160 日 ⇒ 起点对齐日历首日
        ensemble_seeds=(42, 61, 82),
        label_spec={"task": "regression", "label_transform": "raw", "version": 1},
        split_spec={"val_ratio": 0.2, "version": 1},
        # 全 False 配方（合成数据仅基础列;开族 ⇒ prepare 缺列 fail-fast）
        feature_recipe={
            "enable_fundamental_features": False,
            "max_feature_missing_ratio": 0.6,
            "version": 1,
        },
        domain="main",
    )


def _minimal_config():
    """轻量配置（守卫/注册接缝测试用,三 spec 带 version 满足 F10 版本门）。"""
    from src.lazybull.v2.common.types import ModelConfig

    return ModelConfig(
        model_family="stock_selection",
        feature_columns=("f1", "f2"),
        label_name="neu_y_ret_20",
        hyperparams={"algorithm": "xgboost", "random_state": 42},
        train_window_months=72,
        label_spec={"task": "regression", "label_transform": "raw", "version": 1},
        split_spec={"val_ratio": 0.2, "version": 1},
        feature_recipe={"max_feature_missing_ratio": 0.6, "version": 1},
        domain="main",
    )


def _stub_backtest_runner(**kwargs) -> Dict[str, object]:
    """宿主注入的回测 stub：合成净值/空成交产物（空折账本契约合法）。"""
    bt_start = kwargs["bt_start"]
    dates = pd.bdate_range(bt_start, periods=3)
    nav = pd.DataFrame(
        {
            "date": dates,
            "portfolio_value": [1.0, 1.005, 1.011],
            "capital": [1.0, 1.005, 1.011],
            "market_value": [0.0, 0.0, 0.0],
            "nav": [1.0, 1.005, 1.011],
            "return": [0.0, 0.005, 0.011],
        }
    )
    return {
        "bt_total_return": 0.011,
        "bt_annual_return": 0.011,
        "bt_max_drawdown": -0.001,
        "bt_volatility": 0.01,
        "bt_sharpe": 1.0,
        "bt_calmar": 11.0,
        "bt_trading_days": 3,
        "bt_start": bt_start,
        "bt_end": kwargs["bt_end"],
        "bt_top_n": kwargs["bt_top_n"],
        "_nav_curve": nav,
        "_trades": pd.DataFrame(),
        "_execution_attribution": pd.DataFrame(),
        "_holdings_snapshot": pd.DataFrame(),
    }


def _run_single_fold_wf(
    tmp_path: Path,
    *,
    host_identity: Optional[tuple],
    trainer_identity: Optional[tuple],
) -> Tuple[str, Path, Path]:
    """单折全流程（真实训练注册 + stub 回测 + 真实 write_runs_batch 落盘）。

    host 与 trainer 身份**分列注入**（R2-T5A-10 单侧携带场景构造）；返回
    (wf_run_id, output_root, runs_root) 供两出口落盘读回断言。
    """
    from src.lazybull.data.loader import DataLoader
    from src.lazybull.data.storage import Storage
    from src.lazybull.v2.common.model_registry import ModelRegistry
    from src.lazybull.v2.hosts.backtest.runs_writer import write_runs_batch
    from src.lazybull.v2.train.frame_cache import TrainingFrameCache
    from src.lazybull.v2.train.trainer import V2Trainer
    from src.lazybull.v2.train.walk_forward.orchestrator import (
        OrchestratorHostConfig,
        V2WalkForwardOrchestrator,
    )

    root = tmp_path / "iso"
    store = _build_panel_root(root)
    output_root = tmp_path / "out"
    runs_root = tmp_path / "runs"
    days = _panel_days()
    registry = ModelRegistry(models_dir=str(output_root / "models" / "stock_selection"))
    trainer = V2Trainer(
        registry=registry,
        main_board_codes=set(_CODES),
        source_mode="panel",
        frame_cache=TrainingFrameCache(),
        sleeve_identity=trainer_identity,
    )
    storage = Storage(root_path=str(root))
    loader = DataLoader(storage)

    def writer(results, wf_run_id, data_state_payload, *, extra_meta_notes=None):
        mapped = []
        for result in results:
            mapped.append(
                {
                    "split_index": result["split_index"],
                    "model_version": result["model_version"],
                    "train_start": result["train_start"],
                    "train_end": result["train_end"],
                    "test_start": result["test_start"],
                    "test_end": result["test_end"],
                    "metrics": dict(result.get("bt_metrics") or {}),
                    "nav_curve": result["_nav_curve"],
                    "trades": result["_trades"],
                    "attribution": result["_execution_attribution"],
                    "holdings_snapshot": result["_holdings_snapshot"],
                }
            )
        return write_runs_batch(
            mapped,
            batch_id=wf_run_id,
            runs_root=Path(runs_root),
            config={
                "label_column": "neu_y_ret_20",
                "algorithm": "xgboost",
                "bt_top_n": 2,
                "bt_rebalance_freq": 20,
                "stock_domain": "main",
                "feature_source": "panel",
            },
            code_state={"git_commit": "test", "git_dirty": False},
            data_state=data_state_payload,
            source="p3_t5_r1_sleeve_identity_acceptance",
            extra_meta_notes=extra_meta_notes,
        )

    host = OrchestratorHostConfig(
        data_root=str(root),
        output_root=str(output_root),
        backtest_runner=_stub_backtest_runner,
        signal_factory=lambda top_n: None,  # 回测启用必须注入工厂（P1-3 守卫口径）
        runs_batch_writer=writer,
        sleeve_identity=host_identity,
    )
    orch = V2WalkForwardOrchestrator(
        host,
        registry=registry,
        source_mode="panel",
        storage=storage,
        loader=loader,
    )
    wf_run_id = orch.run(_wf_config(), 1, days[-1], trainer, store, test_window_months=1)
    return wf_run_id, output_root, runs_root


class TestSleeveIdentityPropagation:
    """R2-T5A-10：run() 解析后的有效身份在训练前同步给 trainer（两出口同值）。"""

    def test_host_only_identity_reaches_both_exits(self, tmp_path):
        """host-only 单侧携带 ⇒ 同 run 注册 metadata 与 runs batch_meta 均含同值。"""
        wf_run_id, output_root, runs_root = _run_single_fold_wf(
            tmp_path, host_identity=("host_side", 2), trainer_identity=None
        )
        metadata = json.loads(
            (output_root / "models" / "stock_selection" / "v1_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        assert metadata["train_params"]["sleeve_id"] == "host_side"
        assert metadata["train_params"]["sleeve_spec_version"] == 2
        batch_meta = json.loads(
            (runs_root / wf_run_id / "batch_meta.json").read_text(encoding="utf-8")
        )
        assert batch_meta["meta_notes"]["sleeve_id"] == "host_side"
        assert batch_meta["meta_notes"]["spec_version"] == "2"

    def test_trainer_only_identity_reaches_both_exits(self, tmp_path):
        """trainer-only 单侧携带 ⇒ 两出口同值（batch_meta 经编排器解析值携带）。"""
        wf_run_id, output_root, runs_root = _run_single_fold_wf(
            tmp_path, host_identity=None, trainer_identity=("t_side", 4)
        )
        metadata = json.loads(
            (output_root / "models" / "stock_selection" / "v1_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        assert metadata["train_params"]["sleeve_id"] == "t_side"
        assert metadata["train_params"]["sleeve_spec_version"] == 4
        batch_meta = json.loads(
            (runs_root / wf_run_id / "batch_meta.json").read_text(encoding="utf-8")
        )
        assert batch_meta["meta_notes"]["sleeve_id"] == "t_side"
        assert batch_meta["meta_notes"]["spec_version"] == "4"

    def test_mismatch_still_rejected(self, tmp_path):
        """双侧非 None 且不等 ⇒ run() 开头 ValueError（fail-fast 语义保持）。"""
        from src.lazybull.v2.train.trainer import V2Trainer
        from src.lazybull.v2.train.walk_forward.orchestrator import (
            OrchestratorHostConfig,
            V2WalkForwardOrchestrator,
        )

        host = OrchestratorHostConfig(
            data_root=str(tmp_path),
            output_root=str(tmp_path / "out"),
            sleeve_identity=("sleeve_b", 1),
        )
        orch = V2WalkForwardOrchestrator(host, registry=MagicMock(), source_mode="panel")
        trainer = V2Trainer(
            registry=MagicMock(),
            main_board_codes=set(),
            sleeve_identity=("sleeve_a", 1),
        )
        with pytest.raises(ValueError, match="袖子身份"):
            orch.run(_minimal_config(), 1, "20240630", trainer, None, test_window_months=1)

    def test_bind_write_point_validation(self):
        """bind 写入点：None 不写 / 同值幂等 / 异值拒绝 / 非法值同 validate 口径。"""
        from src.lazybull.v2.train.trainer import V2Trainer

        trainer = V2Trainer(registry=MagicMock(), main_board_codes=set())
        trainer.bind_sleeve_identity(None)  # None ⇒ 不写
        assert trainer.sleeve_identity is None
        trainer.bind_sleeve_identity(("value", 1))
        trainer.bind_sleeve_identity(("value", 1))  # 同值幂等
        assert trainer.sleeve_identity == ("value", 1)
        with pytest.raises(ValueError, match="禁止覆盖"):
            trainer.bind_sleeve_identity(("other", 1))
        fresh = V2Trainer(registry=MagicMock(), main_board_codes=set())
        with pytest.raises(ValueError, match="sleeve_identity"):
            fresh.bind_sleeve_identity(("a", 0))  # 非法值同构造期校验口径


# ---------------------------------------------------------------------------
# 5. P3-1：runs 批次 config 不含袖子键（第三通道撤销）
# ---------------------------------------------------------------------------


class TestRunsConfigNoSleeveKeys:
    def test_config_has_no_sleeve_keys(self):
        from src.lazybull.v2.sleeves._registry import get_sleeve_spec

        spec = get_sleeve_spec("value")
        backtest = build_backtest_host_params(spec)
        for source_mode in ("cs_train_verify", "panel"):
            config = build_runs_batch_config(spec, backtest, source_mode)
            assert "sleeve_id" not in config
            assert "spec_version" not in config


# ---------------------------------------------------------------------------
# 6. P2-5：复杂度锚定（拆分成果防回退）
# ---------------------------------------------------------------------------


class TestComplexityAnchors:
    def test_assemble_function_within_hard_limit(self):
        source = (ROOT / "scripts" / "v2_p3" / "t5_arm_assembly.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "assemble_arm_orchestrator":
                found = True
                assert node.end_lineno - node.lineno + 1 <= 150  # 契约 §4.4 硬上限
        assert found, "assemble_arm_orchestrator 未找到"

    def test_runner_file_within_soft_limit(self):
        lines = (
            (ROOT / "scripts" / "v2_p3" / "t5_arm_runner.py")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        assert len(lines) <= 800  # 契约 §4.4 文件行数软上限

    def test_compare_gates_file_within_hard_limit(self):
        """t5_compare_gates.py 文件行数 < 1200 硬上限（reviewer1 P3-N1 锚点）。"""
        lines = (
            (ROOT / "scripts" / "v2_p3" / "t5_compare_gates.py")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        assert len(lines) < 1200

    def test_soft_overflow_files_registered(self):
        """软超限登记锚点（reviewer1 P3-N1；契约 §4.4 拆分义务在案）。

        登记值（本锚点建立时）：``t5_arm_artifacts.py`` 约 970 行、
        ``t5_attribution_sections.py`` 约 928 行——两文件均超 800 软上限，
        拆分义务已登记在案；后续拆分落地 ≤800 时本锚点失败，届时同步收紧为
        ``<= 800`` 断言。
        """
        for name in ("t5_arm_artifacts.py", "t5_attribution_sections.py"):
            lines = (ROOT / "scripts" / "v2_p3" / name).read_text(encoding="utf-8").splitlines()
            assert len(lines) > 800, f"{name} 已回落至软上限内，请同步收紧本登记锚点为 <= 800"


# ---------------------------------------------------------------------------
# 7. S2-T5A-01：spec.top_n / spec.signal_impl 消费（替换 spec 值即生效）
# ---------------------------------------------------------------------------


def _spec_stub(**over: object) -> SimpleNamespace:
    base: Dict[str, object] = {
        "label_name": "neu_y_ret_20",
        "holding_period_days": 20,
        "top_n": 20,
        "domain": "main",
        "signal_impl": "src.lazybull.v2.core.signal.ml_signal.MLSignal",
        "min_amount_ma20": 50000.0,
        "exclude_financial": True,
    }
    base.update(over)
    return SimpleNamespace(**base)


class TestSpecSignalConsumption:
    def test_top_n_consumed_from_spec(self):
        """替换 spec.top_n ⇒ BacktestHostParams 与信号工厂产物同步生效。"""
        assert build_backtest_host_params(_spec_stub(top_n=5)).bt_top_n == 5
        signal = build_signal_factory(_spec_stub(top_n=5), ROOT / "nonexistent_models")(5)
        assert signal.top_n == 5

    def test_real_spec_top_n_unchanged(self):
        """现役 value 规格（top_n=20）⇒ 与修复前逐位一致。"""
        from src.lazybull.v2.sleeves._registry import get_sleeve_spec

        spec = get_sleeve_spec("value")
        assert spec.top_n == 20
        assert build_backtest_host_params(spec).bt_top_n == 20

    def test_signal_impl_substitution_takes_effect(self, tmp_path, monkeypatch):
        """替换 spec.signal_impl ⇒ importlib 解析的类被真实构造（替换即生效）。"""
        (tmp_path / "fake_signal_mod.py").write_text(
            "class FakeSignal:\n"
            "    def __init__(self, top_n=1, model_version=None, models_dir=None,\n"
            "                 min_amount_ma20=0.0, min_total_mv=0.0, max_total_mv=0.0,\n"
            "                 exclude_financial=False, verbose=False,\n"
            "                 downside_penalty=0.0, downside_penalty_column='x'):\n"
            "        self.top_n = top_n\n"
            "        self.models_dir = models_dir\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        stub = _spec_stub(signal_impl="fake_signal_mod.FakeSignal")
        signal = build_signal_factory(stub, tmp_path / "models")(7)
        assert type(signal).__name__ == "FakeSignal"
        assert signal.top_n == 7

    def test_signal_impl_incompatible_signature_rejected(self, tmp_path, monkeypatch):
        """构造签名不兼容（缺 kwarg 表面、无 **kwargs）⇒ 拒绝（禁止静默回退 MLSignal）。"""
        (tmp_path / "bad_signal_mod.py").write_text(
            "class BadSignal:\n" "    def __init__(self, top_n):\n" "        self.top_n = top_n\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        with pytest.raises(ValueError, match="不兼容"):
            resolve_signal_class("bad_signal_mod.BadSignal")

    def test_signal_impl_position_only_param_rejected(self, tmp_path, monkeypatch):
        """bind 预验证强化（reviewer2 §3 P3）：top_n 位置仅限 ⇒ 工厂 kwarg 不可绑定 ⇒ 拒绝。"""
        (tmp_path / "posonly_signal_mod.py").write_text(
            "class PosOnlySignal:\n"
            "    def __init__(self, top_n, /, **kwargs):\n"
            "        self.top_n = top_n\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        with pytest.raises(ValueError, match="不兼容"):
            resolve_signal_class("posonly_signal_mod.PosOnlySignal")

    def test_signal_impl_extra_required_param_rejected(self, tmp_path, monkeypatch):
        """bind 预验证强化：额外必填参数（工厂不供给）⇒ 缺参绑定失败 ⇒ 拒绝。"""
        (tmp_path / "extra_signal_mod.py").write_text(
            "class ExtraRequiredSignal:\n"
            "    def __init__(self, top_n, extra_required, **kwargs):\n"
            "        self.top_n = top_n\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        with pytest.raises(ValueError, match="不兼容"):
            resolve_signal_class("extra_signal_mod.ExtraRequiredSignal")

    def test_signal_impl_unresolvable_rejected(self):
        with pytest.raises(ValueError, match="解析失败"):
            resolve_signal_class("no.such.module.ClassName")

    def test_real_spec_signal_impl_resolves(self):
        """现役 value 规格 signal_impl ⇒ 解析为 v2 MLSignal（签名兼容断言通过）。"""
        from src.lazybull.v2.core.signal.ml_signal import MLSignal
        from src.lazybull.v2.sleeves._registry import get_sleeve_spec

        assert resolve_signal_class(get_sleeve_spec("value").signal_impl) is MLSignal


# ---------------------------------------------------------------------------
# 8. R2-T5A-06：_score_fold 预测行持久化接线（桩住工具函数断言传参）
# ---------------------------------------------------------------------------


class TestScoreFoldPersistWiring:
    def test_persist_kwargs_passed_per_fold(self, tmp_path):
        """所有臂统一传 persist_predictions=True + <fold目录>/predictions.parquet + 锚点。"""
        captured: Dict[str, object] = {}

        def score_fn(**kwargs: object) -> Dict[str, object]:
            captured.update(kwargs)
            days = list(kwargs["score_days"])  # type: ignore[arg-type]
            rows = [(day, "000001.SH", 1.0) for day in days]
            return {
                "model_version": kwargs["model_version"],
                "fingerprint": fingerprint_prediction_rows(rows),
                "per_seed_diagnostic": [],
                "score_days": days,
                "n_days": len(days),
                "n_rows": len(rows),
            }

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
            arm=ARM_V2_CS_TRAIN_VERIFY,
            run_name="t5_persist_unit",
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
        record = _score_fold(ctx, 0, {"arm": ARM_V2_CS_TRAIN_VERIFY}, score_fn, {"000001.SH"}, None)
        assert captured["persist_predictions"] is True
        assert captured["predictions_path"] == (
            tmp_path / "arm" / "folds" / "split00" / "predictions.parquet"
        )
        # 锚点接线（R2-T5A-06 集成轮）：相对臂目录 POSIX 登记口径的入口
        assert captured["predictions_anchor"] == tmp_path / "arm"
        assert record["model_version"] == 1
        assert (tmp_path / "arm" / "folds" / "split00" / "fingerprint.json").exists()


# ---------------------------------------------------------------------------
# 9. R2-T5A-02 集成接线：消费门臂感知 + 打分 receipt 核对 + manifest_drift 入判定
# ---------------------------------------------------------------------------

_FROZEN_END = "20260702"


def _write_manifest(features_dir: Path) -> None:
    """合成有效 manifest（口径复制自 test_v2_p3_t5_r1_consumption_fixes.py，自包含约定）。"""
    doc = {
        "manifest_version": "1",
        "updated_at": "2026-07-02T00:00:00",
        "columns": {
            "close": {
                "group": "core",
                "source": "daily",
                "definition_version": "v1",
                "available_from": None,
                "backfilled_at": None,
                "status": "active",
            }
        },
        "partitions": {
            f"panel/{_DAYS[0]}/core.parquet": {
                "sha256_16": "a" * 16,
                "rows": 2,
                "written_at": "2026-06-01T00:00:00",
                "zone": "hot",
                "row_domain": "label_filtered",
            },
        },
        "dependencies": {"raw_datasets": [], "factor_functions": {}},
        "labels": {
            name: {
                "partitions": {
                    f"{_DAYS[0]}.parquet": {
                        "sha256_16": "c" * 16,
                        "rows": 2,
                        "written_at": "2026-06-01T00:00:00",
                        "row_domain": "label_filtered",
                    }
                },
                "sealed_through": _FROZEN_END,
            }
            for name in ("y_ret_5", "y_ret_10", "y_ret_20")
        },
        "row_domain_segments": {
            "label_filtered_through": _FROZEN_END,
            "structural_from": "20260703",
        },
    }
    (features_dir / "manifest.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (features_dir / "manifest.json.bak").write_bytes((features_dir / "manifest.json").read_bytes())
    (features_dir / "manifest_wal.jsonl").write_bytes(b"")  # 空 WAL = 无待处理态


def _make_panel_production_root(tmp_path: Path) -> Path:
    """``_make_production_root`` + panel 热区 / labels 三表 / manifest 族（臂③扩展枚举）。"""
    prod = _make_production_root(tmp_path)
    # panel 热区：窗内日（正例 receipt 目标）+ 冻结段外 20260703（不入消费集合，反例目标）
    for day in _DAYS[:3]:
        directory = prod / "features" / "panel" / day
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "core.parquet").write_bytes(f"hot-{day}".encode())
    directory = prod / "features" / "panel" / "20260703"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "core.parquet").write_bytes(b"hot-out")
    # panel 冷区（归档目录存在性防御；窗内 2025-06）
    archive = prod / "features" / "panel_archive" / "2025-06"
    archive.mkdir(parents=True, exist_ok=True)
    (archive / "core.parquet").write_bytes(b"arc-2025-06")
    # labels 三表（窗内非零覆盖）
    for name in ("y_ret_5", "y_ret_10", "y_ret_20"):
        directory = prod / "labels" / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{_DAYS[0]}.parquet").write_bytes(f"lab-{name}".encode())
    _write_manifest(prod / "features")
    return prod


class _FakePanelStore:
    """PanelDataStore 替身：load_features 返回全日域合成帧；receipt 清单预置/可追加。

    ``manifest`` 属性委托真实 PanelDataStore（打补丁前捕获真类）——manifest 基线
    快照/复核（``t1_reconcile_adapter._manifest_snapshot``）与打分 store 创建共用
    同一模块属性，全局补丁下必须保住 manifest 真实语义。
    """

    def __init__(self, receipts: Sequence[object] = ()) -> None:
        self._receipts = list(receipts)
        self._real: object = None

    def bind(self, real_cls: object, root: Path) -> "_FakePanelStore":
        self._real = real_cls(root)  # type: ignore[operator]
        return self

    @property
    def manifest(self) -> object:
        if self._real is None:
            raise RuntimeError("_FakePanelStore 未绑定真实实例（harness 先调 bind）")
        return self._real.manifest  # type: ignore[attr-defined]

    @property
    def read_receipts(self) -> Tuple[object, ...]:
        return tuple(self._receipts)

    def add_receipt(self, receipt: object) -> None:
        self._receipts.append(receipt)

    def load_features(self, query: object) -> SimpleNamespace:
        index = pd.MultiIndex.from_tuples(
            [(day, "000001.SH") for day in _DAYS], names=["trade_date", "ts_code"]
        )
        return SimpleNamespace(df=pd.DataFrame(index=index))


def _receipt(paths: Sequence[str]) -> SimpleNamespace:
    """合成 StoreReadReceipt 形态（files = 显示路径记录，相对 store root 口径）。"""
    files = tuple({"path": p, "sha256": "0" * 64, "size": 1} for p in paths)
    return SimpleNamespace(
        kind="features",
        start_date=_DAYS[0],
        end_date=_DAYS[-1],
        files=files,
        collected_at="2026-10-24T00:00:00",
    )


def _fabricate_compare_dir(tmp_path: Path) -> Path:
    """合成臂② run 目录（arm_meta.json 携带当前代码态 source_digest，过 P2-4 前置门）。"""
    from src.lazybull.v2.store.store_identity import workspace_identity

    compare_dir = tmp_path / "arm2_run"
    compare_dir.mkdir(exist_ok=True)
    digest = workspace_identity(ROOT)["source_digest"]
    (compare_dir / "arm_meta.json").write_text(
        json.dumps({"producer_identity": {"source_digest": digest}}), encoding="utf-8"
    )
    return compare_dir


def _run_panel_arm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: _ExecutorState,
    store: _FakePanelStore,
    **over: object,
) -> Dict[str, object]:
    """臂③合成端到端：PanelDataStore 与特征列读取桩住（禁止真实 store/训练）。"""
    from scripts.v2_p3 import t5_arm_executors as executors_mod
    from src.lazybull.v2.store import data_store as data_store_mod

    real_store_cls = data_store_mod.PanelDataStore  # 打补丁前捕获真类（manifest 委托用）
    monkeypatch.setattr(
        data_store_mod, "PanelDataStore", lambda root: store.bind(real_store_cls, root)
    )
    monkeypatch.setattr(
        executors_mod, "_load_model_feature_columns", lambda models_dir, version: ["f1"]
    )
    prod = _make_panel_production_root(tmp_path)
    kwargs = _run_kwargs(
        prod,
        tmp_path,
        state,
        arm=ARM_V2_PANEL,
        compare_arm_dir=_fabricate_compare_dir(tmp_path),
        **over,
    )
    return run_arm(**kwargs)  # type: ignore[arg-type]


class TestPanelArmIntegration:
    def test_panel_arm_end_to_end_pass(self, tmp_path, monkeypatch):
        """臂③全链路 PASS：消费门扩展生效 + receipt 核对通过 + manifest 复核落盘 +
        fingerprint.json 登记相对臂目录 predictions_path。"""
        store = _FakePanelStore(receipts=[_receipt([f"features/panel/{_DAYS[0]}/core.parquet"])])
        state = _ExecutorState()
        evidence = _run_panel_arm(tmp_path, monkeypatch, state, store, run_name="t5_panel")
        assert evidence["verdict"] == "PASS"
        # 消费门臂感知：扩展目录入计数（臂③枚举生效）
        counts = evidence["baselines"]["consumption_counts"]  # type: ignore[index]
        assert counts["features/panel"] == 3 and counts["labels"] == 3
        # 打分 receipt 核对：实际读取 ⊆ 消费集合，报告入证据
        check = evidence["execution"]["score_receipt_check"]  # type: ignore[index]
        assert check["pass"] is True and check["receipt_files"] == 1
        # manifest 复核段落盘（臂③基线启用）
        arm_dir = tmp_path / "frozen" / "t5_panel"
        assert (arm_dir / "manifest_verify.json").exists()
        guards = evidence["guards"]  # type: ignore[index]
        assert guards["manifest_drift"] == []
        # 锚点接线端到端：fingerprint.json 登记相对臂目录 POSIX 路径 + 五件套落盘
        fold_dir = arm_dir / "folds" / "split00"
        fingerprint = json.loads((fold_dir / "fingerprint.json").read_text(encoding="utf-8"))
        assert fingerprint["predictions_path"] == "folds/split00/predictions.parquet"
        assert fingerprint["signal_topk_path"] == "folds/split00/signal_topk.parquet"
        assert fingerprint["signal_topk_rows"] == 4
        for name in (
            "model_version.txt",
            "boundary.json",
            "fingerprint.json",
            "predictions.parquet",
            "signal_topk.parquet",
        ):
            assert (fold_dir / name).exists(), name
        # signal_selection 传参核验（spec + 域市值同源）
        assert state.score_selection[0] == {
            "top_n": 20,
            "min_amount_ma20": 50000.0,
            "exclude_financial": True,
            "min_total_mv": 500000.0,
            "max_total_mv": 15000000.0,
        }

    def test_panel_arm_receipt_outside_fails_verdict(self, tmp_path, monkeypatch):
        """打分读取越出消费集合（冻结段外分区）⇒ 不抛断流程，verdict FAIL + 报告入证据。"""
        store = _FakePanelStore(
            receipts=[
                _receipt(
                    [
                        f"features/panel/{_DAYS[0]}/core.parquet",
                        "features/panel/20260703/core.parquet",  # 冻结段外，未入消费集合
                    ]
                )
            ]
        )
        state = _ExecutorState()
        evidence = _run_panel_arm(tmp_path, monkeypatch, state, store, run_name="t5_panel_bad")
        assert evidence["verdict"] == "FAIL"
        check = evidence["execution"]["score_receipt_check"]  # type: ignore[index]
        assert check["pass"] is False
        assert "20260703" in str(check["error"])

    def test_manifest_drift_fails_verdict(self, tmp_path, monkeypatch):
        """manifest 窗内登记运行期改写 ⇒ manifest_drift 入判定，verdict FAIL。"""
        store = _FakePanelStore(receipts=[_receipt([f"features/panel/{_DAYS[0]}/core.parquet"])])
        state = _ExecutorState()
        base_train = _make_synthetic_executor(state).train_fn

        def tampering_train(ctx: ArmRunContext, subset: Tuple[int, ...], write_batch: bool):
            result = base_train(ctx, subset, write_batch)
            manifest_path = ctx.production_data_root / "features" / "manifest.json"
            doc = json.loads(manifest_path.read_text(encoding="utf-8"))
            doc["partitions"][f"panel/{_DAYS[0]}/core.parquet"]["sha256_16"] = "f" * 16
            manifest_path.write_text(
                json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return result

        executor = ArmExecutor(
            train_fn=tampering_train,
            replay_fn=_make_synthetic_executor(state).replay_fn,
            score_fn=_make_synthetic_executor(state).score_fn,
        )
        evidence = _run_panel_arm(
            tmp_path, monkeypatch, state, store, run_name="t5_panel_mdf", executor=executor
        )
        assert evidence["verdict"] == "FAIL"
        guards = evidence["guards"]  # type: ignore[index]
        assert any(_DAYS[0] in item for item in guards["manifest_drift"])

    def test_double_run_manifest_verify_copied(self, tmp_path, monkeypatch):
        """臂③双跑 PASS：run1/run2 各自同步 manifest_verify.json（共享基线内容一致）。"""
        store = _FakePanelStore(receipts=[_receipt([f"features/panel/{_DAYS[0]}/core.parquet"])])
        state = _ExecutorState()
        evidence = _run_panel_arm(
            tmp_path,
            monkeypatch,
            state,
            store,
            run_name="t5_panel_double",
            double_run=True,
            selected_split_indices=[0],
        )
        assert evidence["verdict"] == "PASS"
        for idx in (1, 2):
            sub = tmp_path / "frozen" / "t5_panel_double" / f"run{idx}"
            assert (sub / "manifest_verify.json").exists()
            sub_check = evidence["runs"][idx - 1]["score_receipt_check"]  # type: ignore[index]
            assert sub_check["pass"] is True

    def test_double_run_receipt_outside_fails(self, tmp_path, monkeypatch):
        """双跑打分 receipt 越界 ⇒ drift 并入比对输入，comparison verdict FAIL。"""
        store = _FakePanelStore(receipts=[_receipt(["features/panel/20260703/core.parquet"])])
        state = _ExecutorState()
        evidence = _run_panel_arm(
            tmp_path,
            monkeypatch,
            state,
            store,
            run_name="t5_panel_double_bad",
            double_run=True,
            selected_split_indices=[0],
        )
        assert evidence["verdict"] == "FAIL"
        comparison = evidence["comparison"]  # type: ignore[index]
        assert any("越出消费集合" in item for item in comparison["consumption_drift"])
