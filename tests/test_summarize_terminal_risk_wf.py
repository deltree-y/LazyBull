"""terminal_risk WF 汇总工具测试（合成折目录，不依赖真实数据）

覆盖：消融后缀解析、超参签名（meta 权威）、分组聚合调参分、历史台账追加、
跨 batch 历史最优比较（含旧 schema 兼容）。
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_terminal_risk_wf import (
    BASELINE_LABEL,
    HISTORY_RUN_COLUMNS,
    HISTORY_TABLE_COLUMNS,
    TUNING_COLUMNS,
    UNREGISTERED_SIGNATURE,
    append_history_records,
    build_history_records,
    build_history_table,
    build_param_signature,
    build_tuning_table,
    collect_fold_rows,
    main,
    parse_experiment_suffix,
)

_TRAIN_CFG = {
    "max_depth": 3,
    "learning_rate": 0.03,
    "n_estimators": 500,
    "early_stopping_rounds": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": 42,
    "eval_metric": "logloss",
}
_LABEL_CFG = {
    "task_id": "terminal_vol_scaled_loss",
    "h_min": 1,
    "h_max": 20,
    "sigma_window": 20,
    "loss_sigma_multiple": 1.0,
}
_SAMPLING_CFG = {"chunk_days": 50, "h_per_group": 2, "every_n_days": 3}


def _write_fold(
    wf_root: Path,
    name: str,
    pr_auc: float,
    event_rate: float,
    mean_pred: float = None,
    depth: int = 3,
) -> None:
    """写一个假折目录（report + 模型元数据，字段口径与训练产物一致）。"""
    fold_dir = wf_root / name
    fold_dir.mkdir(parents=True)
    es_section = {
        "n": 1000,
        "event_rate": event_rate,
        "logloss": 0.3,
        "brier": 0.1,
        "pr_auc": pr_auc,
    }
    if mean_pred is not None:
        es_section["mean_pred"] = mean_pred
    with open(fold_dir / "terminal_loss_report.json", "w", encoding="utf-8") as f:
        json.dump({"es": es_section, "train": {"event_rate": event_rate}}, f)
    meta = {
        "train_config": {**_TRAIN_CFG, "max_depth": depth},
        "label_config": _LABEL_CFG,
        "metadata": {
            "stage_dates": {
                "train": ["20230101", "20231231"],
                "es": ["20240101", "20240630"],
            },
            "best_iteration": 123,
            "n_train": 5000,
            "n_es": 1000,
            "sampling": _SAMPLING_CFG,
        },
    }
    with open(fold_dir / "terminal_loss_model.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)


def _summary_row(fold: str, lift: float, pred_bias: float = 0.0) -> dict:
    """直接构造窄 summary 行（同时覆盖缺超参列的容错路径）。"""
    return {"fold": fold, "lift": lift, "pred_bias": pred_bias}


def test_parse_experiment_suffix():
    """后缀解析：baseline / _d* / _d*_lr* / _w*y / _v*m / _em* / _s* 与未知尾缀兜底。"""
    empty = {
        "suffix": "",
        "depth": None,
        "learning_rate": None,
        "train_window_years": None,
        "val_months": None,
        "eval_metric": None,
        "seed": None,
    }
    assert parse_experiment_suffix("2022H2") == empty
    assert parse_experiment_suffix("2022H2_d3") == {
        **empty,
        "suffix": "_d3",
        "depth": 3,
    }
    parsed = parse_experiment_suffix("2022H2_d3_lr0.05")
    assert parsed["suffix"] == "_d3_lr0.05"
    assert parsed["depth"] == 3
    assert parsed["learning_rate"] == pytest.approx(0.05)
    # 训练窗口年数与随机种子后缀（v0.108.5 新增消融位）
    assert parse_experiment_suffix("2022H2_w5y")["train_window_years"] == 5
    assert parse_experiment_suffix("2022H2_w5y")["suffix"] == "_w5y"
    assert parse_experiment_suffix("2022H2_s7")["seed"] == 7
    # 早停段月数后缀（v0.109.0 新增：早停段与评估段分离）
    assert parse_experiment_suffix("2022H2_v6m")["val_months"] == 6
    assert parse_experiment_suffix("2022H2_v6m")["suffix"] == "_v6m"
    # 早停指标后缀（v0.109.0 新增消融位；meta 的 `em=` 才是权威身份）
    assert parse_experiment_suffix("2022H2_v6m_emrank_ic")["eval_metric"] == "rank_ic"
    assert parse_experiment_suffix("2022H2_v6m_emrank_ic")["suffix"] == "_v6m_emrank_ic"
    combined = parse_experiment_suffix("2022H2_d5_lr0.04_w5y_v6m_emlogloss_s7")
    assert combined["suffix"] == "_d5_lr0.04_w5y_v6m_emlogloss_s7"
    assert (
        combined["depth"],
        combined["train_window_years"],
        combined["val_months"],
        combined["eval_metric"],
        combined["seed"],
    ) == (5, 5, 6, "logloss", 7)
    # 未知尾缀按无后缀处理（同后缀目录仍聚同组），保证新后缀类型不中断汇总
    assert parse_experiment_suffix("2022H2_x9")["suffix"] == ""


def test_build_param_signature():
    """超参签名：完整配置产出预期串；缺任一键或训练起止日不可解析返回 None。"""
    meta = {
        "train_config": _TRAIN_CFG,
        "label_config": _LABEL_CFG,
        "metadata": {
            "sampling": _SAMPLING_CFG,
            "stage_dates": {"train": ["20230101", "20231231"], "es": ["20240101", "20240630"]},
        },
    }
    # valm=0：v0.109.0 之前的旧协议（早停即评估），与新协议批次天然分属不同组
    assert build_param_signature(meta) == (
        "d=3|lr=0.03|nest=500|esr=30|sub=0.8|col=0.8|lam=1.0|s=42|em=logloss"
        "|k=1.0|hmax=20|sigw=20|hpg=2|end=3|wy=1|valm=0"
    )
    assert build_param_signature({}) is None
    broken = {
        "train_config": {k: v for k, v in _TRAIN_CFG.items() if k != "reg_lambda"},
        "label_config": _LABEL_CFG,
        "metadata": {"sampling": _SAMPLING_CFG},
    }
    assert build_param_signature(broken) is None
    # 缺训练起止日（无法算窗口年数）→ 返回 None，独立成组不与其他签名混组
    no_dates = {
        "train_config": _TRAIN_CFG,
        "label_config": _LABEL_CFG,
        "metadata": {"sampling": _SAMPLING_CFG},
    }
    assert build_param_signature(no_dates) is None


def test_param_signature_separates_seed_and_window():
    """随机种子与训练窗口年数都是签名维度：多种子/多窗口不得并入同一组。"""

    def _meta(seed: int, train_start: str) -> dict:
        return {
            "train_config": {**_TRAIN_CFG, "random_state": seed},
            "label_config": _LABEL_CFG,
            "metadata": {
                "sampling": _SAMPLING_CFG,
                "stage_dates": {
                    "train": [train_start, "20240630"],
                    "es": ["20240701", "20241231"],
                },
            },
        }

    base = build_param_signature(_meta(42, "20210701"))
    other_seed = build_param_signature(_meta(7, "20210701"))
    longer_window = build_param_signature(_meta(42, "20190701"))
    assert "|s=42|" in base and base.endswith("|wy=3|valm=0")
    assert "|s=7|" in other_seed and other_seed.endswith("|wy=3|valm=0")
    assert "|s=42|" in longer_window and longer_window.endswith("|wy=5|valm=0")
    assert len({base, other_seed, longer_window}) == 3


def test_param_signature_separates_val_segment():
    """早停段月数是签名维度：旧协议（valm=0）不得与新协议批次并组比较。"""

    def _meta(val) -> dict:
        stage_dates = {"train": ["20230101", "20231231"], "es": ["20250101", "20250630"]}
        if val is not None:
            stage_dates["val"] = val
        return {
            "train_config": _TRAIN_CFG,
            "label_config": _LABEL_CFG,
            "metadata": {"sampling": _SAMPLING_CFG, "stage_dates": stage_dates},
        }

    assert build_param_signature(_meta(None)).endswith("|valm=0")
    assert build_param_signature(_meta(["20240701", "20241231"])).endswith("|valm=6")
    # 非法 Val 区间（起止倒置）→ 整条签名返回 None（数据问题独立成组，
    # 不得静默当成旧协议 valm=0）
    assert build_param_signature(_meta(["20241231", "20240701"])) is None


def test_build_tuning_table_score_and_gate():
    """分组聚合：调参分手算期望、门禁按组判定、排序降序、None lift 剔除。"""
    summary = pd.DataFrame(
        [
            _summary_row("2022H2", 1.2, 0.01),
            _summary_row("2023H1", 1.8, 0.02),
            _summary_row("2022H2_d3", 1.5, -0.01),
            _summary_row("2023H1_d3", 1.05, -0.02),
            # 事件率 0 折 lift 为 None：剔除聚合并计入 n_folds 差额
            _summary_row("2024H1_d3", None, 0.0),
        ]
    )
    table = build_tuning_table(summary, lift_min_threshold=1.1)

    assert list(table.columns) == TUNING_COLUMNS
    assert len(table) == 2

    base = table[table["suffix"] == BASELINE_LABEL].iloc[0]
    base_geo = (1.2 * 1.8) ** 0.5
    assert base["n_folds"] == 2
    assert base["n_folds_valid_lift"] == 2
    assert base["lift_geo_mean"] == pytest.approx(base_geo)
    assert base["lift_min"] == pytest.approx(1.2)
    assert base["tuning_score"] == pytest.approx(0.5 * (base_geo + 1.2))
    assert bool(base["gate_pass"])  # lift_min 1.2 >= 1.1 门禁通过
    # 窄 summary 缺超参列 → 容错为未登记签名（带后缀保持组间区分）
    assert base["param_signature"].startswith(UNREGISTERED_SIGNATURE)

    d3 = table[table["suffix"] == "_d3"].iloc[0]
    d3_geo = (1.5 * 1.05) ** 0.5
    assert d3["n_folds"] == 3
    assert d3["n_folds_valid_lift"] == 2
    assert d3["lift_geo_mean"] == pytest.approx(d3_geo)
    assert d3["lift_min"] == pytest.approx(1.05)
    assert d3["tuning_score"] == pytest.approx(0.5 * (d3_geo + 1.05))
    assert not bool(d3["gate_pass"])  # lift_min 1.05 < 1.1 门禁未通过

    # baseline 调参分更高 → rank=1 且排序在前
    assert table.iloc[0]["suffix"] == BASELINE_LABEL
    assert table.iloc[0]["rank"] == 1


def test_tuning_table_splits_same_suffix_by_signature(tmp_path):
    """同后缀不同超参签名（目录残留旧消融折）必须拆行，禁止混比。"""
    _write_fold(tmp_path, "2022H2", pr_auc=0.12, event_rate=0.1, depth=3)
    _write_fold(tmp_path, "2023H1", pr_auc=0.18, event_rate=0.1, depth=4)

    summary = collect_fold_rows(tmp_path)
    table = build_tuning_table(summary, lift_min_threshold=1.1)
    # 同 (baseline) 后缀、两个 depth → 两行，签名不同
    assert len(table) == 2
    assert set(table["depth"]) == {3, 4}
    assert table["param_signature"].nunique() == 2
    # depth 从折 meta 读取（权威），不依赖后缀
    row_d4 = table[table["depth"] == 4].iloc[0]
    assert row_d4["suffix"] == BASELINE_LABEL
    assert "d=4|" in row_d4["param_signature"]


def test_append_history_records_rounds(tmp_path):
    """台账追加：连续两次调用累计两轮行；二次写入不重复 BOM、读回列名正常。"""
    history_csv = tmp_path / "tuning_history.csv"
    table = build_tuning_table(
        pd.DataFrame([_summary_row("2022H2", 1.3), _summary_row("2022H2_d3", 1.5)]),
        lift_min_threshold=1.1,
    )
    records = build_history_records(table, "wf", 1.1)
    assert list(records.columns) == HISTORY_RUN_COLUMNS
    assert append_history_records(history_csv, records) == 2
    assert append_history_records(history_csv, records) == 2

    history = pd.read_csv(history_csv, encoding="utf-8-sig")
    assert list(history.columns) == HISTORY_RUN_COLUMNS
    assert len(history) == 4
    assert set(history["suffix"]) == {BASELINE_LABEL, "_d3"}
    # 追加行不携带 BOM：第二次写入的行能作为数据行读回（列数一致无脏列）
    assert history["tuning_score"].notna().all()


def test_history_table_best_and_refresh(tmp_path):
    """历史最优：旧台账高分签名胜出；当次达到历史最好时标记刷新。"""
    history_csv = tmp_path / "tuning_history.csv"

    # 历史运行：_d3 签名调参分 1.5（时间戳 2026-09-01）
    old_table = build_tuning_table(
        pd.DataFrame([_summary_row("2022H2_d3", 1.5)]), lift_min_threshold=1.1
    )
    old_records = build_history_records(old_table, "wf", 1.1, timestamp="2026-09-01T00:00:00")
    assert append_history_records(history_csv, old_records) == 1

    # 当次运行：baseline 1.2 → 未达历史最优，最优行 is_current_best=False
    cur_table = build_tuning_table(
        pd.DataFrame([_summary_row("2022H2", 1.2)]), lift_min_threshold=1.1
    )
    cur_records = build_history_records(cur_table, "wf", 1.1, timestamp="2026-09-10T00:00:00")
    table = build_history_table(history_csv, cur_records)
    assert list(table.columns) == HISTORY_TABLE_COLUMNS
    best = table.iloc[0]
    assert best["score_best"] == pytest.approx(1.5)
    assert not bool(best["is_current_best"])  # pandas 往返后为 numpy.bool_
    assert best["runs"] == 1
    # 当次 baseline 组签名未登记但与 _d3 组区分开（不因同为未登记而合并）
    assert table.iloc[-1]["param_signature"].startswith(UNREGISTERED_SIGNATURE)

    # 台账落盘后（含当次行），baseline 组 runs=2 且中位数出现
    append_history_records(history_csv, cur_records)
    # 追加模式去重：台账已含当次行时再拼同 timestamp 内存记录不得双计
    table_dedup = build_history_table(history_csv, cur_records)
    dedup_base = table_dedup[table_dedup["suffix"] == BASELINE_LABEL].iloc[0]
    assert dedup_base["runs"] == 1
    table2 = build_history_table(
        history_csv,
        build_history_records(cur_table, "wf", 1.1, timestamp="2026-09-11T00:00:00"),
    )
    base_row = table2[table2["suffix"] == BASELINE_LABEL].iloc[0]
    assert base_row["runs"] == 2
    assert base_row["score_median"] == pytest.approx(1.2)

    # 当次 baseline 提高到 1.6 → 刷新历史最优且最优行标记为当次
    win_table = build_tuning_table(
        pd.DataFrame([_summary_row("2022H2", 1.6)]), lift_min_threshold=1.1
    )
    win_records = build_history_records(win_table, "wf", 1.1, timestamp="2026-09-12T00:00:00")
    table3 = build_history_table(history_csv, win_records)
    assert table3.iloc[0]["score_best"] == pytest.approx(1.6)
    assert bool(table3.iloc[0]["is_current_best"])


def test_history_table_legacy_schema_compat(tmp_path):
    """旧台账（无 param_signature 列）读回不崩，签名回退 (legacy) 不与真签名合并。"""
    history_csv = tmp_path / "tuning_history.csv"
    legacy = pd.DataFrame(
        [
            {
                "timestamp": "2026-09-01T00:00:00",
                "wf_root": "wf",
                "lift_min_threshold": 1.1,
                "suffix": "_d3",
                "depth": 3,
                "learning_rate": 0.03,
                "n_folds": 8,
                "n_folds_valid_lift": 8,
                "tuning_score": 1.4,
                "lift_geo_mean": 1.5,
                "lift_min": 1.3,
                "lift_std": 0.1,
                "pred_bias_mean": 0.01,
                "gate_pass": True,
            }
        ]
    )
    legacy.to_csv(history_csv, index=False, encoding="utf-8-sig")

    cur_table = build_tuning_table(
        pd.DataFrame([_summary_row("2022H2", 1.2)]), lift_min_threshold=1.1
    )
    cur_records = build_history_records(cur_table, "wf", 1.1, timestamp="2026-09-10T00:00:00")
    table = build_history_table(history_csv, cur_records)
    # 旧行独立成组（legacy 签名），当次行独立成组（未登记签名），两组互不合并
    assert len(table) == 2
    assert table.iloc[0]["param_signature"].startswith("(legacy)")
    assert table.iloc[0]["score_best"] == pytest.approx(1.4)
    # gate_pass 经 CSV 字符串往返后仍正确聚成通过率
    assert table.iloc[0]["gate_pass_rate"] == pytest.approx(1.0)


def test_collect_and_tuning_table_end_to_end(tmp_path):
    """假折目录端到端：lift 口径（pr_auc/事件率）与分组、(baseline) 标签。"""
    _write_fold(tmp_path, "2022H2", pr_auc=0.12, event_rate=0.1, mean_pred=0.11)
    _write_fold(tmp_path, "2022H2_d3", pr_auc=0.15, event_rate=0.1, mean_pred=0.09)

    summary = collect_fold_rows(tmp_path)
    assert len(summary) == 2
    by_fold = summary.set_index("fold")
    assert by_fold.loc["2022H2", "lift"] == pytest.approx(1.2)
    assert by_fold.loc["2022H2", "pred_bias"] == pytest.approx(0.11 - 0.1)
    assert by_fold.loc["2022H2_d3", "lift"] == pytest.approx(1.5)
    # 超参签名与 depth 从折 meta 提取（权威口径）
    assert by_fold.loc["2022H2_d3", "depth"] == 3
    assert by_fold.loc["2022H2_d3", "learning_rate"] == pytest.approx(0.03)
    assert by_fold.loc["2022H2", "param_signature"].startswith("d=3|lr=0.03")

    table = build_tuning_table(summary, lift_min_threshold=1.1)
    suffixes = set(table["suffix"])
    assert suffixes == {BASELINE_LABEL, "_d3"}
    d3 = table[table["suffix"] == "_d3"].iloc[0]
    assert d3["tuning_score"] == pytest.approx(0.5 * (1.5 + 1.5))
    assert bool(d3["gate_pass"])


def test_collect_fold_rows_falls_back_to_versioned_artifacts(tmp_path):
    """仅注册制产物（无固定名别名）的折仍须被汇总，不得静默跳过。"""
    fold_dir = tmp_path / "2024H1"
    fold_dir.mkdir(parents=True)
    with open(fold_dir / "v2_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "es": {"n": 500, "event_rate": 0.1, "logloss": 0.3, "brier": 0.1, "pr_auc": 0.2},
                "train": {"event_rate": 0.1},
            },
            f,
        )
    with open(fold_dir / "v2_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "version": 2,
                "train_params": {
                    "train_config": _TRAIN_CFG,
                    "label_config": _LABEL_CFG,
                    "sampling": _SAMPLING_CFG,
                    "stage_dates": {
                        "train": ["20230101", "20231231"],
                        "es": ["20240101", "20240630"],
                    },
                    "best_iteration": 77,
                    "n_train": 4000,
                    "n_es": 500,
                },
                "performance_metrics": {"lift": 2.0},
            },
            f,
        )
    summary = collect_fold_rows(tmp_path)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["fold"] == "2024H1"
    assert row["lift"] == pytest.approx(2.0)
    # best_iteration 从注册表元数据读出（v0.108.0 起才登记进 metadata）
    assert row["best_iteration"] == 77
    assert row["train_start"] == "20230101"
    # train_params 形状归一后签名与折 sidecar 一致
    assert row["param_signature"].startswith("d=3|lr=0.03")
    assert row["depth"] == 3


def test_collect_fold_rows_skips_dir_without_artifacts(tmp_path):
    """既无固定名也无版本化产物的目录：跳过并记录，不抛异常。"""
    (tmp_path / "empty_fold").mkdir(parents=True)
    _write_fold(tmp_path, "2022H2", pr_auc=0.12, event_rate=0.1)
    summary = collect_fold_rows(tmp_path)
    assert list(summary["fold"]) == ["2022H2"]


def test_color_helpers(monkeypatch):
    """彩色输出：非终端不加色码（管道/重定向干净），终端包裹 ANSI 转义。"""
    import scripts.summarize_terminal_risk_wf as mod

    monkeypatch.setattr(mod, "_COLOR_ENABLED", False)
    assert mod._c("文本", "1;92") == "文本"
    df = pd.DataFrame({"a": [1, 2]})
    assert "\033[" not in mod._highlight_first_row(df, "1;92")

    monkeypatch.setattr(mod, "_COLOR_ENABLED", True)
    assert mod._c("文本", "1;92") == "\033[1;92m文本\033[0m"
    highlighted = mod._highlight_first_row(df, "1;92")
    assert "\033[1;92m" in highlighted
    # 仅首个数据行高亮，其余行保持纯文本
    assert highlighted.count("\033[1;92m") == 1


def test_main_no_history_flag(tmp_path, monkeypatch):
    """main() 端到端：--no-history 时不生成台账，summary/tuning_scores 正常落盘。"""
    _write_fold(tmp_path, "2022H2", pr_auc=0.12, event_rate=0.1, mean_pred=0.11)
    monkeypatch.setattr(
        sys,
        "argv",
        ["summarize_terminal_risk_wf.py", "--wf-root", str(tmp_path), "--no-history"],
    )
    assert main() == 0
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "tuning_scores.csv").exists()
    assert not (tmp_path / "tuning_history.csv").exists()


def test_main_appends_history_by_default(tmp_path, monkeypatch):
    """main() 端到端：默认追加台账，重复运行累计两轮（跨 batch 纵向对比）。"""
    _write_fold(tmp_path, "2022H2", pr_auc=0.12, event_rate=0.1, mean_pred=0.11)
    argv = ["summarize_terminal_risk_wf.py", "--wf-root", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    assert main() == 0
    assert main() == 0
    history = pd.read_csv(tmp_path / "tuning_history.csv", encoding="utf-8-sig")
    assert len(history) == 2
    assert history["suffix"].tolist() == [BASELINE_LABEL, BASELINE_LABEL]
    assert history["param_signature"].nunique() == 1
