# -*- coding: utf-8 -*-
"""v2 P3 T5 数据影响归因交付物测试（全合成数据，不读取真实生产数据/产物）。

覆盖（P3 规划 §7.5 / §13.3.7 证据清单映射）：
- a 节：逐折 × 消费列差异计数 + F11 词表分类（D-12 / D-04③ 纯 NaN 闭合；
  未分类/越界列 ⇒ 拒绝项 FAIL）；词表边界单元级（L1 / L1 长尾 / L1+孤立单点 /
  D-04③ 纯 NaN 型约束 / D-05 专项三上限 / D-12 豁免）；
- 两臂消费列集不一致 ⇒ FAIL（含列序不一致）；
- b 节：score_days 对称差 / n_rows 差 / 指纹一致标记；
- c/d 节：键集对称差计数 + 交集行价格/数量最大绝对差；
- e 节：chain_nav max|Δ| 与末日 Δ + 逐折 daily 汇总差；
- f 节判定表接线：带内 / ΔCAGR 超带（有/无归因闭合）/ ΔMaxDD 劣化（有/无归因）；
- g 节：source_digest 不一致 ⇒ FAIL；字段缺失 ⇒ FAIL 前置；
- JSON/MD 落盘字段完整性 + CLI 冒烟（合成输入 exit 码 0/1）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v2_p3 import t5_attribution_sections as sections  # noqa: E402
from scripts.v2_p3.run_data_impact_attribution import build_report, main  # noqa: E402
from scripts.v2_p3.t5_attribution_report import render_markdown  # noqa: E402
from scripts.v2_p3.t05b_fingerprint import fingerprint_prediction_rows  # noqa: E402
from scripts.v2_p3.t05b_isolated_root import collect_consumption_snapshot  # noqa: E402
from src.lazybull.v2.common.chain_metrics import calculate_chain_metrics  # noqa: E402
from src.lazybull.v2.common.types import TradeDate  # noqa: E402
from src.lazybull.v2.store.column_groups import group_of_column  # noqa: E402
from src.lazybull.v2.store.data_store import PanelDataStore  # noqa: E402

_DAYS = ["20260105", "20260106"]
_CODES = ["600000.SH", "000001.SZ"]
_FOLDS = ("split00", "split01")
_CONSUMED_COLUMNS = ("ret_1", "days_to_unlock", "dividend_payout_ratio")

_OBSERVER = {
    "source_digest": "observer-digest",
    "git_commit": "obs0000",
    "git_dirty": True,
    "environment": {"python": "3.12.0"},
}
_ARM_META_DIGEST = "arm-shared-digest"


def _default_chain_metrics() -> dict[str, Any]:
    """基线链式指标 = 默认 chain_nav（nav_last_delta=0）经唯一实现派生。

    R2-T5A-R1-07②：臂 chain_metrics.json 必须与其 runs 批 chain_nav 重算一致
    （容差 1e-9），故夹具一律派生、禁止手填指标值。
    """
    return calculate_chain_metrics(_default_chain_nav_frame(0.0))


def _default_chain_nav_frame(nav_last_delta: float) -> pd.DataFrame:
    rows = [
        {"date": "20260105", "nav": 1.0, "split_index": 0},
        {"date": "20260106", "nav": 1.01, "split_index": 0},
        {"date": "20260106", "nav": 1.01, "split_index": 1},
        {"date": "20260107", "nav": 1.02 + nav_last_delta, "split_index": 1},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 合成夹具：生产数据根（panel store + cs_train 分区）
# ---------------------------------------------------------------------------


def _register_columns(store: PanelDataStore, columns: list[str]) -> None:
    store.manifest.register_columns(
        {
            col: {
                "group": group_of_column(col) if _is_panel_column(col) else "core",
                "source": "test",
            }
            for col in columns
        }
    )
    store.manifest.save()


def _is_panel_column(col: str) -> bool:
    try:
        group_of_column(col)
    except KeyError:
        return False
    return True


def _build_data_root(
    tmp_path: Path,
    *,
    columns: tuple[str, ...] = _CONSUMED_COLUMNS,
    panel_overrides: dict[str, list[float]] | None = None,
    cs_train_overrides: dict[str, list[float]] | None = None,
    open_days: tuple[str, ...] = tuple(_DAYS),
    closed_days: tuple[str, ...] = (),
) -> Path:
    """构建合成数据根：panel 八族中实际用到的族 + cs_train 日分区 + 交易日历。

    默认差形（闭合语义）：days_to_unlock 值差（D-12 豁免）+
    dividend_payout_ratio 纯 NaN 型差（D-04③ 登记窗内）。
    交易日历（clean/trade_cal.parquet）= a 节完整覆盖前置的独立预期日域来源
    （R2-T5A-04）：``open_days`` 为 is_open=1，``closed_days`` 为 is_open=0。
    """
    data_root = tmp_path / "data_root"
    store = PanelDataStore(data_root)
    _register_columns(store, list(columns))
    panel_overrides = panel_overrides or {}
    cs_train_overrides = cs_train_overrides or {}
    for day in _DAYS:
        _append_panel_day(store, day, columns, panel_overrides)
        _write_cs_train_day(data_root, day, columns, cs_train_overrides)
    _write_trade_calendar(data_root, open_days, closed_days)
    return data_root


def _write_trade_calendar(
    data_root: Path, open_days: tuple[str, ...], closed_days: tuple[str, ...]
) -> None:
    clean_dir = data_root / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"cal_date": day, "is_open": 1} for day in open_days] + [
        {"cal_date": day, "is_open": 0} for day in closed_days
    ]
    pd.DataFrame(rows).to_parquet(clean_dir / "trade_cal.parquet", index=False)


def _default_values(col: str) -> list[float]:
    return {
        "ret_1": [0.01, 0.02],
        "days_to_unlock": [100.0, 100.0],
        "dividend_payout_ratio": [0.3, 0.3],
    }.get(col, [1.0, 1.0])


def _append_panel_day(
    store: PanelDataStore,
    day: str,
    columns: tuple[str, ...],
    overrides: dict[str, list[float]],
) -> None:
    by_group: dict[str, dict[str, list[float]]] = {}
    for col in columns:
        group = group_of_column(col) if _is_panel_column(col) else "core"
        values = overrides.get(col)
        if values is None:
            values = _default_values(col)
            if col == "days_to_unlock":
                values = [200.0, 200.0]  # D-12 值差（panel 全量 PIT 语义侧）
        by_group.setdefault(group, {})[col] = list(values)
    for group, col_values in by_group.items():
        payload: dict[str, list] = {
            "trade_date": [day] * len(_CODES),
            "ts_code": list(_CODES),
            **col_values,
        }
        store.append_features(TradeDate.from_str(day), group, pd.DataFrame(payload))


def _write_cs_train_day(
    data_root: Path,
    day: str,
    columns: tuple[str, ...],
    overrides: dict[str, list[float]],
) -> None:
    cs_train_dir = data_root / "features" / "cs_train"
    cs_train_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, list] = {"trade_date": [day] * len(_CODES), "ts_code": list(_CODES)}
    for col in columns:
        values = overrides.get(col)
        if values is None:
            values = _default_values(col)
            if col == "dividend_payout_ratio":
                values = [0.3, float("nan")]  # D-04③ 纯 NaN 型差
        payload[col] = list(values)
    pd.DataFrame(payload).to_parquet(cs_train_dir / f"{day}.parquet", index=False)


# ---------------------------------------------------------------------------
# 合成夹具：臂目录（folds + 模型登记根 + runs 批次 + 身份文件）
# ---------------------------------------------------------------------------


def _build_arm(
    root: Path,
    name: str,
    *,
    data_root: Path,
    digest: str | None = _ARM_META_DIGEST,
    columns: tuple[str, ...] = _CONSUMED_COLUMNS,
    trade_price: float = 10.0,
    extra_trade_row: bool = False,
    holdings_shares: float = 100.0,
    nav_last_delta: float = 0.0,
    daily_nav_delta: float = 0.0,
    predictions_rows: list[dict[str, Any]] | None = None,
) -> Path:
    """构建最小臂 run 目录（两臂布局同构；差异经参数注入）。

    跨路契约证据自洽（R2-T5A-R1-03/05/07 夹具口径）：
    - 预测行落盘 ``folds/splitNN/predictions.parquet``（按键升序），fingerprint.json 登记
      ``predictions_path/predictions_sha256/predictions_rows`` + **行集文本真指纹**
      + 派生 ``n_rows``/``score_days``；
    - ``chain_metrics.json`` = 本臂 runs 批 chain_nav 经唯一实现**派生**（禁手填）；
    - ``consumption_snapshot.json`` 真实采集 data_root 的 cs_train + panel 分区，
      ``arm_meta.producer_identity.input_snapshot_digest`` = 快照文件 sha256（摘要绑定）。
    """
    arm = root / name
    for idx, fold in enumerate(_FOLDS):
        fold_dir = arm / "folds" / fold
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "model_version.txt").write_text(str(idx + 1), encoding="utf-8")
        boundary = {
            "outer_train_start": _DAYS[0],
            "outer_train_end": _DAYS[1],
            "test_start": _DAYS[1],
            "test_end": _DAYS[1],
            "outer_train_samples": 4,
            "test_samples": 2,
        }
        (fold_dir / "boundary.json").write_text(json.dumps(boundary), encoding="utf-8")
        pred_frame = _write_predictions(fold_dir, predictions_rows)
        fingerprint = {
            "model_version": idx + 1,
            "source_mode": "cs_train_verify" if name == "arm2" else "panel",
            "score_days": sorted({str(d) for d in pred_frame["trade_date"]}),
            "n_rows": len(pred_frame),
            "fingerprint": _prediction_frame_fingerprint(pred_frame),
            "per_seed_diagnostic": [],
            "predictions_path": f"folds/{fold}/predictions.parquet",
            "predictions_sha256": _sha256_file(fold_dir / "predictions.parquet"),
            "predictions_rows": len(pred_frame),
        }
        (fold_dir / "fingerprint.json").write_text(json.dumps(fingerprint), encoding="utf-8")
    _write_models_registry(arm, columns)
    metrics = _write_runs_batch(
        arm,
        name,
        trade_price=trade_price,
        extra_trade_row=extra_trade_row,
        holdings_shares=holdings_shares,
        nav_last_delta=nav_last_delta,
        daily_nav_delta=daily_nav_delta,
    )
    (arm / "chain_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    snapshot_digest = _write_consumption_binding(arm, data_root)
    _write_arm_meta(arm, digest, snapshot_digest)
    before_guard = {
        "base": str(Path(data_root).resolve() / "models" / "stock_selection"),
        "entries": {},
        "hashes": {},
    }
    (arm / "before_guard.json").write_text(json.dumps(before_guard), encoding="utf-8")
    return arm


def _write_consumption_binding(arm: Path, data_root: Path) -> str:
    """真实采集消费快照（cs_train 日分区 + panel 分区全量）并返回快照文件 sha256。

    基目录 = 生产数据根的父级（``t05b_baselines`` 口径，与归因 runner 复核同基）。
    """
    snapshot_path = arm / "consumption_snapshot.json"
    files = sorted((Path(data_root) / "features" / "cs_train").glob("*.parquet"))
    files.extend(
        path
        for layer in ("clean", "raw")
        if (path := Path(data_root) / layer / "trade_cal.parquet").is_file()
    )
    panel_root = Path(data_root) / "features" / "panel"
    if panel_root.is_dir():
        files.extend(sorted(panel_root.rglob("*.parquet")))
    base_dir = Path(data_root).resolve().parent
    collect_consumption_snapshot(files, base_dir, snapshot_path)
    return _sha256_file(snapshot_path)


def _default_predictions_rows() -> list[dict[str, Any]]:
    """默认预测行（两臂一致 ⇒ b 节零差异）：按键升序、-0.0 已归一。"""
    return [
        {"trade_date": "20260106", "ts_code": "000001.SZ", "score": 0.11},
        {"trade_date": "20260106", "ts_code": "600000.SH", "score": 0.22},
    ]


def _write_predictions(fold_dir: Path, rows: list[dict[str, Any]] | None) -> pd.DataFrame:
    frame = pd.DataFrame(rows if rows is not None else _default_predictions_rows())
    frame = frame.astype({"trade_date": str, "ts_code": str, "score": float})
    frame = frame.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    frame.to_parquet(fold_dir / "predictions.parquet", index=False)
    return frame


def _prediction_frame_fingerprint(frame: pd.DataFrame) -> str:
    """行集文本真指纹（``t05b_fingerprint.fingerprint_prediction_rows`` 同口径）。"""
    rows = [
        (str(trade_date), str(ts_code), float(score))
        for trade_date, ts_code, score in frame[["trade_date", "ts_code", "score"]].itertuples(
            index=False
        )
    ]
    return fingerprint_prediction_rows(rows)


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_models_registry(arm: Path, columns: tuple[str, ...]) -> None:
    models_dir = arm / "root" / "models" / "stock_selection"
    models_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(len(_FOLDS)):
        (models_dir / f"v{idx + 1}_features.json").write_text(
            json.dumps(list(columns)), encoding="utf-8"
        )


def _write_arm_meta(arm: Path, digest: str | None, input_snapshot_digest: str) -> None:
    """臂 arm_meta.json（t5_arm_runner 实际落盘口径：producer_identity 嵌套，P3-2；
    ``input_snapshot_digest`` = 消费快照文件 sha256，R2-T5A-R1-03③ 摘要绑定）。"""
    producer: dict[str, Any] = {
        "arm": arm.name,
        "environment_identity": {"python": "3.12.0", "pandas": "3.0.0"},
        "config_digest": f"cfg-{arm.name}",
        "input_snapshot_digest": input_snapshot_digest,
    }
    if digest is not None:
        producer["source_digest"] = digest
    arm_meta: dict[str, Any] = {"arm": arm.name, "producer_identity": producer}
    (arm / "arm_meta.json").write_text(json.dumps(arm_meta), encoding="utf-8")


def _write_runs_batch(
    arm: Path,
    name: str,
    *,
    trade_price: float,
    extra_trade_row: bool,
    holdings_shares: float,
    nav_last_delta: float,
    daily_nav_delta: float,
) -> dict[str, Any]:
    """写 runs 批并返回 chain_nav 派生指标（chain_metrics.json 的唯一合法来源）。"""
    batch = arm / "runs" / f"batch_{name}"
    batch.mkdir(parents=True, exist_ok=True)
    batch_meta = {
        "batch_id": f"batch_{name}",
        "code_state": {"git_commit": "abc1234", "git_dirty": True},
        "data_state": {"data_state_id": f"ds-{name}", "sources": {"daily": "2026-01-06"}},
    }
    (batch / "batch_meta.json").write_text(json.dumps(batch_meta), encoding="utf-8")
    nav_frame = _default_chain_nav_frame(nav_last_delta)
    nav_frame.to_parquet(batch / "chain_nav.parquet", index=False)
    for idx, fold in enumerate(_FOLDS):
        fold_dir = batch / "folds" / fold
        fold_dir.mkdir(parents=True, exist_ok=True)
        _write_trades(fold_dir, trade_price, extra_trade_row and idx == 0)
        _write_daily(fold_dir, daily_nav_delta)
        _write_holdings(fold_dir, holdings_shares)
    return calculate_chain_metrics(nav_frame)


def _write_trades(fold_dir: Path, price: float, extra_row: bool) -> None:
    rows = [
        {
            "trade_date": "20260106",
            "ts_code": "000001.SZ",
            "action": "buy",
            "price": price,
            "shares": 100.0,
            "amount": price * 100.0,
        }
    ]
    if extra_row:
        rows.append(
            {
                "trade_date": "20260107",
                "ts_code": "600000.SH",
                "action": "sell",
                "price": price,
                "shares": 50.0,
                "amount": price * 50.0,
            }
        )
    pd.DataFrame(rows).to_parquet(fold_dir / "trades.parquet", index=False)


def _write_daily(fold_dir: Path, nav_delta: float) -> None:
    rows = [
        {"trade_date": _DAYS[offset], "nav": 1.0 + nav_value + nav_delta, "total_value": 1e6}
        for offset, nav_value in enumerate((0.0, 0.01))
    ]
    pd.DataFrame(rows).to_parquet(fold_dir / "daily.parquet", index=False)


def _write_holdings(fold_dir: Path, shares: float) -> None:
    rows = [
        {
            "trade_date": "20260106",
            "ts_code": "000001.SZ",
            "shares": shares,
            "market_value": shares * 10.0,
            "weight": 0.2,
        }
    ]
    pd.DataFrame(rows).to_parquet(fold_dir / "holdings_snapshot.parquet", index=False)


def _build_b0p(
    root: Path,
    *,
    chain_metrics: dict[str, Any] | None = None,
    digest: str | None = _ARM_META_DIGEST,
) -> Path:
    """构建合成 B0' 冻结目录（真实布局：freeze_manifest.json 为身份唯一来源，P3-2）。"""
    b0p = root / "b0_prime"
    b0p.mkdir(parents=True, exist_ok=True)
    metrics = dict(_default_chain_metrics() if chain_metrics is None else chain_metrics)
    (b0p / "chain_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    producer: dict[str, Any] = {
        "arm": "legacy_cs_train",
        "environment_identity": {"python": "3.12.0", "pandas": "3.0.0"},
    }
    if digest is not None:
        producer["source_digest"] = digest
    manifest = {
        "frozen_at": "2026-10-23T00:00:00",
        "source_run_name": "arm1_run",
        "arm": "legacy_cs_train",
        "config_digest": "cfg-b0p",
        "producer_identity": producer,
        "batch_fingerprint": "bfp-b0p",
        "chain_metrics": metrics,
        "folds": {},
        "runs_batch": {
            "batch_id": "batch_arm1",
            "chain_nav_path": "runs/batch_arm1/chain_nav.parquet",
        },
        "data_state_id": "ds-b0p",
        "code_state": {"git_commit": "abc1234", "git_dirty": True},
    }
    (b0p / "freeze_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return b0p


@pytest.fixture()
def synth_env(tmp_path: Path) -> dict[str, Path]:
    """默认闭合环境：两臂同 digest/列集/产物，a 节差异全部落入 D-12/D-04③ 登记类。"""
    data_root = _build_data_root(tmp_path)
    arm2 = _build_arm(tmp_path, "arm2", data_root=data_root)
    arm3 = _build_arm(tmp_path, "arm3", data_root=data_root)
    b0p = _build_b0p(tmp_path)
    return {"data_root": data_root, "arm2": arm2, "arm3": arm3, "b0p": b0p}


def _run_report(env: dict[str, Path]) -> dict[str, Any]:
    return build_report(
        env["arm2"],
        env["arm3"],
        env["b0p"],
        env["data_root"],
        run_id="test_run",
        observer_identity=_OBSERVER,
    )


# ---------------------------------------------------------------------------
# a 节：逐折 × 消费列差异计数 + 词表分类
# ---------------------------------------------------------------------------


class TestSectionA:
    def test_closed_with_vocab_classification(self, synth_env):
        """D-12 值差 + D-04③ 纯 NaN 差 ⇒ 全部词表闭合，总判定 PASS。"""
        report = _run_report(synth_env)
        assert report["verdict"]["verdict"] == "PASS", report["verdict"]["errors"]
        section_a = report["sections"]["a_column_input_diffs"]
        assert section_a["closed"] is True
        assert section_a["rejected_columns"] == {}
        assert len(section_a["folds"]) == 2
        fold0 = section_a["folds"][0]
        assert fold0["consumed_columns"] == list(_CONSUMED_COLUMNS)
        assert fold0["days_compared"] == 2
        # 完整覆盖前置（R2-T5A-04）：预期日域 = 两日开市，非开市剔除 0
        assert fold0["expected_days_count"] == 2
        assert fold0["excluded_non_open_days"] == 0
        assert fold0["coverage_gap_count"] == 0
        unlock = fold0["columns"]["days_to_unlock"]
        assert unlock["over_rows"] == 4  # 2 股 × 2 日
        assert unlock["total_rows"] == 4
        assert unlock["attribution"].startswith("D-12")
        assert unlock["boundary_violation"] is False
        payout = fold0["columns"]["dividend_payout_ratio"]
        assert payout["over_rows"] == 2  # 每日 1 行 NaN 型差
        assert payout["attribution"].startswith("D-04③")
        assert payout["boundary_violation"] is False
        assert "ret_1" not in fold0["columns"]  # 零差异列不入差异列集
        assert "ret_1" in fold0["identical_columns"]

    def test_unclassified_column_fails(self, tmp_path):
        """未分类列（晚窗密集值差、不属任何登记类）⇒ 拒绝项，a 节不闭合 ⇒ FAIL。"""
        data_root = _build_data_root(
            tmp_path,
            columns=(*_CONSUMED_COLUMNS, "mystery_col"),
            panel_overrides={"mystery_col": [9.0, 9.0]},
            cs_train_overrides={"mystery_col": [1.0, 1.0]},
        )
        arm2 = _build_arm(
            tmp_path, "arm2", data_root=data_root, columns=(*_CONSUMED_COLUMNS, "mystery_col")
        )
        arm3 = _build_arm(
            tmp_path, "arm3", data_root=data_root, columns=(*_CONSUMED_COLUMNS, "mystery_col")
        )
        b0p = _build_b0p(tmp_path)
        report = build_report(arm2, arm3, b0p, data_root, observer_identity=_OBSERVER)
        assert report["verdict"]["verdict"] == "FAIL"
        section_a = report["sections"]["a_column_input_diffs"]
        assert section_a["closed"] is False
        assert any("mystery_col" in key for key in section_a["rejected_columns"])

    def test_column_set_mismatch_fails(self, synth_env):
        """两臂消费列集不一致 ⇒ FAIL（禁止取交集降级）。"""
        arm3_models = synth_env["arm3"] / "root" / "models" / "stock_selection"
        for idx in range(len(_FOLDS)):
            (arm3_models / f"v{idx + 1}_features.json").write_text(
                json.dumps([*_CONSUMED_COLUMNS, "extra_col"]), encoding="utf-8"
            )
        report = _run_report(synth_env)
        assert report["verdict"]["verdict"] == "FAIL"
        assert any("消费列集不一致" in message for message in report["verdict"]["errors"])

    def test_column_order_mismatch_fails(self, synth_env):
        """列集相同但列序不一致 ⇒ FAIL（模型输入列序即语义）。"""
        arm3_models = synth_env["arm3"] / "root" / "models" / "stock_selection"
        reversed_columns = list(reversed(_CONSUMED_COLUMNS))
        for idx in range(len(_FOLDS)):
            (arm3_models / f"v{idx + 1}_features.json").write_text(
                json.dumps(reversed_columns), encoding="utf-8"
            )
        report = _run_report(synth_env)
        assert report["verdict"]["verdict"] == "FAIL"
        assert any("列序不一致" in message for message in report["verdict"]["errors"])


class TestClassificationBoundaries:
    """词表分类边界（ColumnImpactAggregator 直喂 _compare_day 形态单日结果）。"""

    def _finalize(
        self,
        col: str,
        diff_days: list[tuple[str, int, int, int]],
        *,
        max_abs_diff: float = 0.1,
        cols_only_panel: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        aggregator = sections.ColumnImpactAggregator()
        for date, over, total, nan_over in diff_days:
            aggregator.add_day(
                {
                    "date": date,
                    "n_only_ref": 0,
                    "n_only_panel": 0,
                    "sample_only_ref": [],
                    "sample_only_panel": [],
                    "cols_only_ref": [],
                    "cols_only_panel": list(cols_only_panel),
                    "col_rows": {col: total},
                    "columns": {
                        col: {
                            "rows": total,
                            "over_rows": over,
                            "over_nan_rows": nan_over,
                            "max_abs_diff": max_abs_diff,
                            "samples": [],
                        }
                    },
                }
            )
        return aggregator.finalize()["columns"][col]

    def _finalize_structure(self, *, date: str, cols_only_panel: tuple[str, ...]) -> dict[str, Any]:
        aggregator = sections.ColumnImpactAggregator()
        aggregator.add_day(
            {
                "date": date,
                "n_only_ref": 0,
                "n_only_panel": 0,
                "sample_only_ref": [],
                "sample_only_panel": [],
                "cols_only_ref": [],
                "cols_only_panel": list(cols_only_panel),
                "col_rows": {},
                "columns": {},
            }
        )
        return aggregator.finalize()

    def test_l1_boundary(self):
        entry = self._finalize("bp", [("20121220", 50, 100, 0)])
        assert entry["attribution"].startswith("L1") and entry["boundary_violation"] is False

    def test_l1_long_tail_window_column(self):
        entry = self._finalize("mkt_volatility_20", [("20130315", 50, 100, 0)])
        assert entry["attribution"].startswith("L1") and entry["boundary_violation"] is False

    def test_l2_period_dense_violation(self):
        entry = self._finalize("bp", [("20140105", 50, 100, 0)])
        assert entry["boundary_violation"] is True

    def test_l1_scatter_after_tail(self):
        entry = self._finalize("bp", [("20121220", 2000, 3000, 0), ("20140105", 1, 3000, 0)])
        assert "孤立单点" in entry["attribution"] and entry["boundary_violation"] is False

    def test_d04_3_nan_only_compliant(self):
        entry = self._finalize("dividend_payout_ratio", [("20200101", 5, 100, 5)])
        assert entry["attribution"].startswith("D-04③") and entry["boundary_violation"] is False

    def test_d04_3_value_diff_violated(self):
        entry = self._finalize("dividend_payout_ratio", [("20200101", 5, 100, 0)])
        assert entry["boundary_violation"] is True

    def test_d12_exempt_any_day(self):
        entry = self._finalize("days_to_unlock", [("20260105", 100, 100, 0)])
        assert entry["attribution"].startswith("D-12") and entry["boundary_violation"] is False

    def test_d05_sparse_ok_dense_violated(self):
        sparse = self._finalize("skewness_20", [("20150326", 3, 300, 0)])
        assert sparse["boundary_violation"] is False
        dense = self._finalize("skewness_20", [("20150326", 30, 300, 0)])
        assert dense["boundary_violation"] is True

    def test_cyq5_cols_only_panel_registered(self):
        """CYQ5 物化列（末日 ≤ 20171229）属已登记 schema 演进，不算结构破坏。"""
        cols = (
            "cost_concentration",
            "weight_avg_bias",
            "winner_rate",
            "winner_rate_chg_5",
            "winner_rate_chg_20",
        )
        result = self._finalize_structure(date="20171229", cols_only_panel=cols)
        assert result["closed"] is True
        assert result["cols_only_panel"] == {}
        assert len(result["registered_cols_only_panel"]) == 5

    def test_cyq5_cols_only_panel_after_last_day_violates(self):
        """CYQ5 列若越出 20171229 末日照样判破坏（fail-closed）。"""
        result = self._finalize_structure(date="20180102", cols_only_panel=("cost_concentration",))
        assert result["closed"] is False
        assert "cost_concentration" in result["cols_only_panel"]

    def test_non_cyq_cols_only_panel_violates(self):
        """非 CYQ5 列不在白名单，即使日期早于 20171229 也判破坏。"""
        result = self._finalize_structure(date="20171229", cols_only_panel=("bp",))
        assert result["closed"] is False
        assert "bp" in result["cols_only_panel"]

    def test_schema_evolution_2012_registered(self):
        """2012 冷启动 schema 演进（末日 ≤ 20121231）属已登记结构差异。"""
        result = self._finalize_structure(date="20121231", cols_only_panel=("bp",))
        assert result["closed"] is True
        assert "bp" in result["registered_cols_only_panel"]

    def test_d11_sparse_single_point_uses_strict_fallback(self):
        """D-11 复牌窗内的稀疏伴随单点（如 002260.SZ 20220505 amplitude 0 vs NaN）
        按 strict 分类器已登记的 D-11 伴随单点放行，不判越界。"""
        entry = self._finalize("amplitude", [("20220505", 1, 3872, 0)], max_abs_diff=0.0)
        assert entry["attribution"].startswith("D-11")
        assert entry["attribution_mode"] == "strict_d11_fallback"
        assert entry["boundary_violation"] is False

    def test_d11_dense_point_still_violates(self):
        """D-11 窗内密集差异仍越界（只放行稀疏伴随单点，不放开密集差异）。"""
        entry = self._finalize("amplitude", [("20220505", 100, 3872, 0)], max_abs_diff=0.0)
        assert entry["boundary_violation"] is True

    def test_non_d11_sparse_point_not_fallback(self):
        """D-11 窗外即使单行稀疏也不走 strict 兜底（仅放开 D-11 伴随单点）。"""
        entry = self._finalize("amplitude", [("20230101", 1, 3872, 0)], max_abs_diff=0.0)
        assert entry["boundary_violation"] is True

    def test_d11_anchor_main_branch_dense_not_fallback(self):
        """D-11 长记忆主分支（窗内任意密度）不得由归因门 fallback 放行。"""
        entry = self._finalize("volatility_20", [("20220505", 100, 100, 0)], max_abs_diff=1.0)
        assert entry["boundary_violation"] is True
        assert entry["attribution_mode"] == "attrib"

    def test_d11_anchor_main_branch_skew_not_fallback(self):
        """D-11 主分支不得覆盖 skewness 专项逐日 25 行上限。"""
        entry = self._finalize("skewness_20", [("20220505", 26, 3000, 0)], max_abs_diff=1.0)
        assert entry["boundary_violation"] is True
        assert entry["attribution_mode"] == "attrib"

    def test_row_diff_breaks_closure(self):
        aggregator = sections.ColumnImpactAggregator()
        aggregator.add_day(
            {
                "date": "20260105",
                "n_only_ref": 1,
                "n_only_panel": 0,
                "sample_only_ref": ["000001.SZ"],
                "sample_only_panel": [],
                "cols_only_ref": [],
                "cols_only_panel": [],
                "col_rows": {},
                "columns": {},
            }
        )
        assert aggregator.finalize()["closed"] is False


# ---------------------------------------------------------------------------
# b~e 节：影响计数
# ---------------------------------------------------------------------------


class TestImpactSections:
    def test_fingerprint_impact_counts(self, tmp_path):
        """b 节：行集不一致 ⇒ 指纹/n_rows 差 + 键集对称差逐折登记（登记证据不作硬失败）。"""
        data_root = _build_data_root(tmp_path)
        arm2 = _build_arm(tmp_path, "arm2", data_root=data_root)
        arm3 = _build_arm(
            tmp_path,
            "arm3",
            data_root=data_root,
            predictions_rows=[
                {"trade_date": "20260106", "ts_code": "000001.SZ", "score": 0.11},
                {"trade_date": "20260106", "ts_code": "300750.SZ", "score": 0.15},
                {"trade_date": "20260106", "ts_code": "600000.SH", "score": 0.22},
            ],
        )
        b0p = _build_b0p(tmp_path)
        report = build_report(arm2, arm3, b0p, data_root, observer_identity=_OBSERVER)
        assert report["verdict"]["verdict"] == "PASS", report["verdict"]["errors"]
        fold0 = report["sections"]["b_prediction_impact"]["folds"][0]
        assert fold0["fingerprint_match"] is False
        assert fold0["score_days_sym_diff_count"] == 0
        assert fold0["n_rows_delta"] == 1
        # 预测行逐值归因（R2-T5A-06）：臂③同日多一只入选 ⇒ 键对称差 1、交集逐值零差异
        assert fold0["predictions"]["key_sym_diff_count"] == 1
        assert fold0["predictions"]["keys_only_arm3"] == 1
        assert fold0["predictions"]["value_diff_rows"] == 0
        note = report["sections"]["b_prediction_impact"]["caliber_note"]
        assert "predictions.parquet" in note and "%.17g" in note

    def test_trade_impact_counts(self, tmp_path):
        """c 节：键集对称差（臂③多一行卖）+ 交集行价格最大绝对差。"""
        data_root = _build_data_root(tmp_path)
        arm2 = _build_arm(tmp_path, "arm2", data_root=data_root)
        arm3 = _build_arm(
            tmp_path, "arm3", data_root=data_root, trade_price=10.5, extra_trade_row=True
        )
        b0p = _build_b0p(tmp_path)
        report = build_report(arm2, arm3, b0p, data_root, observer_identity=_OBSERVER)
        assert report["verdict"]["verdict"] == "PASS", report["verdict"]["errors"]
        folds = report["sections"]["c_trade_impact"]["folds"]
        fold0 = next(f for f in folds if f["fold"] == "split00")
        assert fold0["key_sym_diff_count"] == 1
        assert fold0["keys_only_arm3"] == 1
        assert fold0["intersection_rows"] == 1
        assert fold0["value_max_abs_diff"]["price"]["max_abs_diff"] == pytest.approx(0.5)
        fold1 = next(f for f in folds if f["fold"] == "split01")
        assert fold1["key_sym_diff_count"] == 0
        assert fold1["value_max_abs_diff"]["price"]["max_abs_diff"] == pytest.approx(0.5)

    def test_holdings_impact_counts(self, tmp_path):
        """d 节：持仓数量差（同键集零对称差，交集 shares/market_value 差）。"""
        data_root = _build_data_root(tmp_path)
        arm2 = _build_arm(tmp_path, "arm2", data_root=data_root)
        arm3 = _build_arm(tmp_path, "arm3", data_root=data_root, holdings_shares=120.0)
        b0p = _build_b0p(tmp_path)
        report = build_report(arm2, arm3, b0p, data_root, observer_identity=_OBSERVER)
        assert report["verdict"]["verdict"] == "PASS", report["verdict"]["errors"]
        fold0 = report["sections"]["d_holdings_impact"]["folds"][0]
        assert fold0["key_sym_diff_count"] == 0
        assert fold0["value_max_abs_diff"]["shares"]["max_abs_diff"] == pytest.approx(20.0)
        assert fold0["value_max_abs_diff"]["market_value"]["max_abs_diff"] == pytest.approx(200.0)

    def test_nav_impact_counts(self, tmp_path):
        """e 节：chain_nav max|Δ| 与末日 Δ + 逐折 daily max|Δnav|。"""
        data_root = _build_data_root(tmp_path)
        arm2 = _build_arm(tmp_path, "arm2", data_root=data_root)
        arm3 = _build_arm(
            tmp_path, "arm3", data_root=data_root, nav_last_delta=0.05, daily_nav_delta=0.001
        )
        b0p = _build_b0p(tmp_path)
        report = build_report(arm2, arm3, b0p, data_root, observer_identity=_OBSERVER)
        assert report["verdict"]["verdict"] == "PASS", report["verdict"]["errors"]
        chain = report["sections"]["e_nav_impact"]["chain_nav"]
        assert chain["max_abs_diff"] == pytest.approx(0.05)
        assert chain["final_fold_last_date_diff"] == pytest.approx(0.05)
        split0 = next(f for f in chain["folds"] if f["split_index"] == 0)
        assert split0["max_abs_diff"] == pytest.approx(0.0)
        daily0 = report["sections"]["e_nav_impact"]["daily_folds"][0]
        assert daily0["value_max_abs_diff"]["nav"]["max_abs_diff"] == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# f 节：判定表接线（带内 / 超带有无归因 / ΔMaxDD 劣化有无归因）
# ---------------------------------------------------------------------------


class TestGdataWiring:
    def test_within_band_pass(self, synth_env):
        section_f = _run_report(synth_env)["sections"]["f_gdata_judgment_table"]
        assert section_f["verdict"] == "PASS"
        assert section_f["cagr_judgment"]["branch"] == "within_band"
        assert section_f["inputs"]["maxdd_attribution_closed"] is True
        assert section_f["inputs"]["cagr_attribution_closed"] is True

    def test_overshoot_with_attribution_closed_pass(self, tmp_path):
        """ΔCAGR 超带（nav 深跌派生）+ a 节闭合 ⇒ 判定表放行并登记，总判定 PASS。"""
        data_root = _build_data_root(tmp_path)
        arm2 = _build_arm(tmp_path, "arm2", data_root=data_root)
        arm3 = _build_arm(tmp_path, "arm3", data_root=data_root, nav_last_delta=-0.5)
        b0p = _build_b0p(tmp_path)
        report = build_report(arm2, arm3, b0p, data_root, observer_identity=_OBSERVER)
        assert report["verdict"]["verdict"] == "PASS", report["verdict"]["errors"]
        section_f = report["sections"]["f_gdata_judgment_table"]
        assert section_f["cagr_judgment"]["branch"] == "overshoot_attributed"
        assert any("超带" in text for text in section_f["registrations"])

    def test_overshoot_without_attribution_fails(self, tmp_path):
        """ΔCAGR 超带 + a 节未闭合（未分类列）⇒ 判定表 FAIL，总判定 FAIL。"""
        data_root = _build_data_root(
            tmp_path,
            columns=(*_CONSUMED_COLUMNS, "mystery_col"),
            panel_overrides={"mystery_col": [9.0, 9.0]},
            cs_train_overrides={"mystery_col": [1.0, 1.0]},
        )
        arm2 = _build_arm(
            tmp_path, "arm2", data_root=data_root, columns=(*_CONSUMED_COLUMNS, "mystery_col")
        )
        arm3 = _build_arm(
            tmp_path,
            "arm3",
            data_root=data_root,
            columns=(*_CONSUMED_COLUMNS, "mystery_col"),
            nav_last_delta=-0.5,
        )
        b0p = _build_b0p(tmp_path)
        report = build_report(arm2, arm3, b0p, data_root, observer_identity=_OBSERVER)
        assert report["verdict"]["verdict"] == "FAIL"
        section_f = report["sections"]["f_gdata_judgment_table"]
        assert section_f["verdict"] == "FAIL"
        assert section_f["cagr_judgment"]["branch"] == "overshoot_unattributed"
        assert section_f["inputs"]["cagr_attribution_closed"] is False

    def test_maxdd_deterioration_wiring(self, tmp_path):
        """ΔMaxDD 劣化（nav 下跌派生）：a 节闭合 ⇒ 放行并登记；未闭合 ⇒ FAIL。"""
        data_root = _build_data_root(tmp_path / "case1")
        arm2 = _build_arm(tmp_path / "case1", "arm2", data_root=data_root)
        arm3 = _build_arm(tmp_path / "case1", "arm3", data_root=data_root, nav_last_delta=-0.05)
        b0p = _build_b0p(tmp_path / "case1")
        report = build_report(arm2, arm3, b0p, data_root, observer_identity=_OBSERVER)
        assert report["verdict"]["verdict"] == "PASS", report["verdict"]["errors"]
        section_f = report["sections"]["f_gdata_judgment_table"]
        assert section_f["maxdd_judgment"]["branch"] == "deterioration_attributed"
        assert any("ΔMaxDD" in text for text in section_f["registrations"])

        data_root2 = _build_data_root(
            tmp_path / "case2",
            columns=(*_CONSUMED_COLUMNS, "mystery_col"),
            panel_overrides={"mystery_col": [9.0, 9.0]},
            cs_train_overrides={"mystery_col": [1.0, 1.0]},
        )
        root2 = tmp_path / "case2"
        arm2b = _build_arm(
            root2, "arm2", data_root=data_root2, columns=(*_CONSUMED_COLUMNS, "mystery_col")
        )
        arm3b = _build_arm(
            root2,
            "arm3",
            data_root=data_root2,
            columns=(*_CONSUMED_COLUMNS, "mystery_col"),
            nav_last_delta=-0.05,
        )
        b0pb = _build_b0p(root2)
        report2 = build_report(arm2b, arm3b, b0pb, data_root2, observer_identity=_OBSERVER)
        assert report2["verdict"]["verdict"] == "FAIL"
        assert (
            report2["sections"]["f_gdata_judgment_table"]["maxdd_judgment"]["branch"]
            == "deterioration_unattributed"
        )


# ---------------------------------------------------------------------------
# g 节：身份附录
# ---------------------------------------------------------------------------


class TestIdentityAppendix:
    def test_digest_mismatch_fails(self, tmp_path):
        """臂③ source_digest ≠ 臂② ⇒ 同代码态前提破坏 ⇒ FAIL。"""
        data_root = _build_data_root(tmp_path)
        arm2 = _build_arm(tmp_path, "arm2", data_root=data_root)
        arm3 = _build_arm(tmp_path, "arm3", data_root=data_root, digest="arm3-different-digest")
        b0p = _build_b0p(tmp_path)
        report = build_report(arm2, arm3, b0p, data_root, observer_identity=_OBSERVER)
        assert report["verdict"]["verdict"] == "FAIL"
        section_g = report["sections"]["g_identity_appendix"]
        assert section_g["source_digest_match"] is False
        assert any("source_digest 不一致" in message for message in report["verdict"]["errors"])

    def test_missing_digest_field_fails(self, tmp_path):
        """arm_meta.json 缺 source_digest 字段 ⇒ FAIL 前置。"""
        data_root = _build_data_root(tmp_path)
        arm2 = _build_arm(tmp_path, "arm2", data_root=data_root)
        arm3 = _build_arm(tmp_path, "arm3", data_root=data_root, digest=None)
        b0p = _build_b0p(tmp_path)
        report = build_report(arm2, arm3, b0p, data_root, observer_identity=_OBSERVER)
        assert report["verdict"]["verdict"] == "FAIL"
        assert any("缺 source_digest 字段" in message for message in report["verdict"]["errors"])

    def test_identity_fields_recorded(self, synth_env):
        """两侧代码态 + 环境身份 + 数据态 + 消费快照复核 + 本脚本身份齐全（P3-2/R2-T5A-04）。"""
        report = _run_report(synth_env)
        section_g = report["sections"]["g_identity_appendix"]
        assert section_g["source_digest_match"] is True
        assert section_g["arm2"]["source_digest"] == _ARM_META_DIGEST
        assert section_g["arm2"]["git_commit"] == "abc1234"  # 读 runs batch_meta.code_state
        assert section_g["arm2"]["git_dirty"] is True
        assert section_g["arm2"]["environment"] == {"python": "3.12.0", "pandas": "3.0.0"}
        assert section_g["arm3"]["environment"] == {"python": "3.12.0", "pandas": "3.0.0"}
        assert section_g["arm2"]["data_state_id"] == "ds-arm2"
        assert section_g["arm3"]["data_state_id"] == "ds-arm3"
        assert section_g["arm2"]["consumption_snapshot_sha256"] is not None
        assert section_g["arm2"]["consumption_snapshot_verified"] is True
        assert section_g["arm3"]["consumption_snapshot_verified"] is True
        # B0' 身份读 freeze_manifest.json（producer_identity + code_state + data_state_id）
        assert section_g["b0p"]["source_digest"] == _ARM_META_DIGEST
        assert section_g["b0p"]["git_commit"] == "abc1234"
        assert section_g["b0p"]["environment"] == {"python": "3.12.0", "pandas": "3.0.0"}
        assert section_g["b0p"]["data_state_id"] == "ds-b0p"
        assert section_g["script"]["source_digest"] == "observer-digest"

    def test_producer_identity_layout_supported(self, synth_env):
        """workspace_identity 旧布局兜底可解析（runner↔归因跨工具契约回归锁）。

        现役落盘口径 = ``producer_identity``（默认夹具）；本用例改写为仅
        ``workspace_identity`` 嵌套 source_digest/environment + 顶层
        ``input_snapshot_digest`` 的旧布局，断言 ``extract_source_digest`` /
        ``extract_input_snapshot_digest`` / ``_arm_identity`` 的兜底键位链不回退。
        """
        for key in ("arm2", "arm3"):
            path = synth_env[key] / "arm_meta.json"
            meta = json.loads(path.read_text(encoding="utf-8"))
            producer = meta.pop("producer_identity")
            meta["input_snapshot_digest"] = producer["input_snapshot_digest"]
            meta["workspace_identity"] = {
                "source_digest": producer["source_digest"],
                "environment": producer["environment_identity"],
            }
            path.write_text(json.dumps(meta), encoding="utf-8")
        report = _run_report(synth_env)
        section_g = report["sections"]["g_identity_appendix"]
        assert report["verdict"]["verdict"] == "PASS", report["verdict"]["errors"]
        assert section_g["source_digest_match"] is True
        assert section_g["arm2"]["source_digest"] == _ARM_META_DIGEST
        assert section_g["arm2"]["environment"] == {"python": "3.12.0", "pandas": "3.0.0"}


# ---------------------------------------------------------------------------
# 落盘完整性与 CLI 冒烟
# ---------------------------------------------------------------------------


class TestOutputs:
    def test_json_md_completeness(self, synth_env):
        """JSON 七节齐全 + MD 必含节标题与放行依据 + 证据链接指针。"""
        report = _run_report(synth_env)
        assert set(report["sections"]) == {
            "a_column_input_diffs",
            "b_prediction_impact",
            "c_trade_impact",
            "d_holdings_impact",
            "e_nav_impact",
            "f_gdata_judgment_table",
            "g_identity_appendix",
        }
        md = render_markdown(report)
        for heading in (
            "## 放行依据",
            "## a. 逐折 × 消费列输入差异",
            "## b. 预测影响",
            "## c. 成交影响",
            "## d. 持仓影响",
            "## e. 净值影响",
            "## f. 链式指标判定表",
            "## g. 身份附录",
            "## 证据链接",
        ):
            assert heading in md, heading
        assert "归因闭合声明" in md
        assert "禁止混用" in md

    def test_cli_smoke_pass(self, synth_env, monkeypatch, tmp_path):
        """CLI 冒烟：合成全闭合输入 ⇒ exit 0，JSON/MD 双落盘且字段齐全。"""
        monkeypatch.setattr(
            "src.lazybull.v2.store.store_identity.workspace_identity", lambda root: _OBSERVER
        )
        out_json = tmp_path / "out" / "report.json"
        out_md = tmp_path / "out" / "report.md"
        exit_code = main(
            [
                "--arm2-dir",
                str(synth_env["arm2"]),
                "--arm3-dir",
                str(synth_env["arm3"]),
                "--b0p-dir",
                str(synth_env["b0p"]),
                "--production-data-root",
                str(synth_env["data_root"]),
                "--run-id",
                "smoke_test",
                "--out-json",
                str(out_json),
                "--out-md",
                str(out_md),
            ]
        )
        assert exit_code == 0
        doc = json.loads(out_json.read_text(encoding="utf-8"))
        assert doc["verdict"]["verdict"] == "PASS"
        assert doc["run_id"] == "smoke_test"
        assert doc["evidence"]["json_path"] == str(out_json)
        assert out_md.read_text(encoding="utf-8").startswith("# v2 P3 T5 数据影响归因报告")

    def test_cli_smoke_fail_exit1(self, synth_env, monkeypatch, tmp_path):
        """CLI 冒烟：代码态不一致 ⇒ verdict FAIL ⇒ exit 1（JSON 仍落盘）。"""
        monkeypatch.setattr(
            "src.lazybull.v2.store.store_identity.workspace_identity", lambda root: _OBSERVER
        )
        arm_meta_path = synth_env["arm3"] / "arm_meta.json"
        arm_meta = json.loads(arm_meta_path.read_text(encoding="utf-8"))
        arm_meta["producer_identity"]["source_digest"] = "tampered-digest"
        arm_meta_path.write_text(json.dumps(arm_meta), encoding="utf-8")
        out_json = tmp_path / "fail" / "report.json"
        exit_code = main(
            [
                "--arm2-dir",
                str(synth_env["arm2"]),
                "--arm3-dir",
                str(synth_env["arm3"]),
                "--b0p-dir",
                str(synth_env["b0p"]),
                "--production-data-root",
                str(synth_env["data_root"]),
                "--run-id",
                "smoke_fail",
                "--out-json",
                str(out_json),
                "--out-md",
                str(tmp_path / "fail" / "report.md"),
            ]
        )
        assert exit_code == 1
        doc = json.loads(out_json.read_text(encoding="utf-8"))
        assert doc["verdict"]["verdict"] == "FAIL"
        assert doc["sections"]["g_identity_appendix"]["source_digest_match"] is False
