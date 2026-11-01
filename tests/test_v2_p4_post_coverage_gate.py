# -*- coding: utf-8 -*-
"""P4 后维护项 B 方案实施锁定测试（就绪门逐列训练期缺失率判据）。

覆盖设计文档（docs/plans/v2/p4_post_feature_coverage_gate_design_20261031.md v3）
测试矩阵的注册侧与宿主侧新增项：

- 注册键集：``feature_missing_rate`` 三键 additive 写入（trainer 路径单元），
  键集精确 = 最终注册列 ∪ 选股过滤列（缺键 / μ≥1.0 / 缺 data_stats ⇒ 拒绝）；
- deploy 入口 offsets>0 前置拒绝（N1：入口即 ValueError，非注册末端报错）；
- 常量分层（reviewer2 复审 P1-A-2）：train 包不 import hosts；
  ``SELECTION_FILTER_COLUMNS`` 单一来源 = common/model_schema；
- 宿主结构化证据（P1-C-1）：``runs/readiness_<date>.json`` 通过日/失败日均落盘，
  写盘失败不阻断主链路（旁路纪律），gap 产物含 coverage 字段。
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.lazybull.v2.common.model_schema import (
    FEATURE_MISSING_RATE_SCHEMA_VERSION,
    SELECTION_FILTER_COLUMNS,
    build_feature_missing_rate_entry,
    read_feature_missing_rate,
)
from src.lazybull.v2.common.types import ModelConfig
from src.lazybull.v2.train.trainer import V2Trainer
from src.lazybull.v2.train.walk_forward.deploy_training import execute_deploy_training
from src.lazybull.v2.train.walk_forward.training_core import TrainFoldParams

# ---------------------------------------------------------------------------
# 注册侧（trainer 路径单元）
# ---------------------------------------------------------------------------

_STAT_BASIS = {
    "coverage_n_samples": 100,
    "n_samples_after_filter": 100,
    "n_dropped_label_nan": 0,
    "actual_data_end": "20240809",
    "n_trade_days": 50,
    "frame": "train_pooled_pre_split",
    "denominator": "label_available_rows",
    "row_domain": "train_filtered",
}


def _dist(mu: float) -> dict:
    """B′ 分布摘要夹具（单阈值形态 p95=p99=max=mu）。"""
    return {
        "mean": mu,
        "p50": mu,
        "p90": mu,
        "p95": mu,
        "p99": mu,
        "max": mu,
        "n_days": 30,
        "p99_unstable": True,
    }


def _data_stats(candidate: dict) -> dict:
    wrapped = {
        col: (value if isinstance(value, dict) else _dist(value))
        for col, value in candidate.items()
    }
    return {
        "feature_missing_rate_distribution": wrapped,
        "feature_missing_rate_stat_basis": dict(_STAT_BASIS),
    }


def _capture_train_params(data_stats, feature_columns=("f1", "f2")) -> dict:
    registry = MagicMock()
    registry.register_model.return_value = 1
    trainer = V2Trainer(registry=registry, main_board_codes=set())
    trainer._register_fold(
        config=ModelConfig(
            model_family="stock_selection",
            feature_columns=("f1", "f2"),
            label_name="y",
            hyperparams={},
            train_window_months=72,
            split_spec={"version": 1},
        ),
        model=SimpleNamespace(),  # 非 EnsembleModel ⇒ 单模型分支
        params=TrainFoldParams(),
        feature_columns=list(feature_columns),
        train_params={"learning_rate": 0.05},
        performance_metrics={},
        train_start="20240102",
        val_end="20240809",
        X_train_len=80,
        X_val_len=20,
        ensemble_offsets=0,
        ensemble_seeds=[42],
        data_stats=data_stats,
    )
    return registry.register_model.call_args.kwargs["train_params"]


class TestRegistrationCoverage:
    def test_register_writes_three_keys_with_exact_keyset(self):
        candidate = {"f1": 0.1, "f2": 0.2, **{col: 0.0 for col in SELECTION_FILTER_COLUMNS}}
        params = _capture_train_params(_data_stats(candidate))
        # 三键 additive 写入；注册外层窗口入 stat
        assert params["feature_missing_rate_schema_version"] == FEATURE_MISSING_RATE_SCHEMA_VERSION
        assert {col: stats["p95"] for col, stats in params["feature_missing_rate"].items()} == {
            "f1": 0.1,
            "f2": 0.2,
            **{col: 0.0 for col in SELECTION_FILTER_COLUMNS},
        }
        # v2 结构：每列是分布摘要 dict
        assert isinstance(params["feature_missing_rate"]["f1"], dict)
        stat = params["feature_missing_rate_stat"]
        assert stat["window"] == ["20240102", "20240809"]
        assert stat["window_source"] == "registration"
        assert stat["source"] == "train_time"
        assert stat["coverage_n_samples"] == 100
        assert (
            stat["coverage_n_samples"]
            == stat["n_samples_after_filter"] - stat["n_dropped_label_nan"]
        )
        # 原有键不被覆盖（additive 语义）
        assert params["learning_rate"] == 0.05

    def test_candidate_extra_keys_dropped_from_registration(self):
        # 注册侧精确键集：候选多余键（门禁/稳定性筛选移除列）不写入 metadata
        candidate = {
            "f1": 0.1,
            "f2": 0.2,
            "removed_by_gate": 0.3,
            **{col: 0.0 for col in SELECTION_FILTER_COLUMNS},
        }
        params = _capture_train_params(_data_stats(candidate))
        assert "removed_by_gate" not in params["feature_missing_rate"]

    def test_missing_data_stats_rejects_registration(self):
        with pytest.raises(ValueError, match="feature_missing_rate"):
            _capture_train_params(None)

    def test_missing_candidate_key_rejects_registration(self):
        # 最终列 f2 不在候选统计集（override 重引入缺席列形态）⇒ 写入侧异常
        candidate = {"f1": 0.1, **{col: 0.0 for col in SELECTION_FILTER_COLUMNS}}
        with pytest.raises(ValueError, match="注册投影缺键"):
            _capture_train_params(_data_stats(candidate))

    def test_invalid_distribution_rejects_registration(self):
        # B′：分位序非法（p95 > p99）⇒ 写入侧异常（读侧同口径校验）
        bad = _dist(0.3)
        bad["p95"] = 0.5
        candidate = {
            "f1": bad,
            "f2": _dist(0.2),
            **{col: _dist(0.0) for col in SELECTION_FILTER_COLUMNS},
        }
        with pytest.raises(ValueError, match="分布非法"):
            _capture_train_params(_data_stats(candidate))


class TestDeployEntryOffsetsRejection:
    def test_offsets_positive_rejected_at_entry(self):
        # N1：入口即 ValueError（小时级训练白跑后末端报错形态不可接受）
        params = TrainFoldParams(ensemble_offsets=1)
        with pytest.raises(ValueError, match="ensemble_offsets>0"):
            execute_deploy_training(
                "20240809",
                "wf_unit",
                frame_provider=lambda s, e: None,
                registry=MagicMock(),
                params=params,
                train_window_years=6,
                main_board_codes=set(),
                topk_values=[20],
                trade_cal=None,
                run_log_csv="unused.csv",
            )


class TestModelSchemaHelpers:
    def test_read_feature_missing_rate_roundtrip(self):
        coverage, error = read_feature_missing_rate(
            {
                "feature_missing_rate": {"a": _dist(0.1)},
                "feature_missing_rate_schema_version": 2,
            }
        )
        assert error is None and coverage["a"]["p95"] == 0.1

    def test_v1_payload_is_schema_unsupported(self):
        # A5：v1（单值 μ）不再被接受 ⇒ metadata_schema_unsupported（不隐式兼容）
        coverage, error = read_feature_missing_rate(
            {"feature_missing_rate": {"a": 0.1}, "feature_missing_rate_schema_version": 1}
        )
        assert coverage is None and error == "metadata_schema_unsupported"

    def test_build_entry_rejects_bool_and_string_values(self):
        with pytest.raises(ValueError):
            build_feature_missing_rate_entry({"a": True}, ["a"])  # type: ignore[dict-item]
        with pytest.raises(ValueError):
            build_feature_missing_rate_entry({"a": "0.1"}, ["a"])  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# 常量分层（reviewer2 复审 P1-A-2）：train 不 import hosts；单一来源中立层
# ---------------------------------------------------------------------------


class TestLayering:
    def test_train_package_does_not_import_hosts(self):
        # orchestrator.py 分层契约："train 包不 import universe / core / hosts / factors"
        train_dir = Path(__file__).resolve().parents[1] / "src" / "lazybull" / "v2" / "train"
        offenders = []
        for py in train_dir.rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "v2.hosts" in line or "hosts.paper" in line or "hosts.backtest" in line:
                    offenders.append(f"{py.name}: {stripped}")
        assert offenders == [], f"train 包出现 hosts import（分层契约违规）: {offenders}"

    def test_selection_filter_columns_single_source(self):
        # 单一来源 = common/model_schema；daily_assembly 兼容 re-export 同值
        from src.lazybull.v2.hosts.paper import daily_assembly

        assert SELECTION_FILTER_COLUMNS == ("amount_ma20", "total_mv", "sw_l1_code")
        assert daily_assembly.SELECTION_FILTER_COLUMNS == SELECTION_FILTER_COLUMNS


# ---------------------------------------------------------------------------
# 宿主结构化证据（reviewer2 复审 P1-C-1）：通过日/失败日均持久化
# ---------------------------------------------------------------------------


class TestReadinessEvidence:
    def test_success_day_writes_readiness_record(self, tmp_path):
        from tests.test_v2_p4_t2_host import _CALENDAR_STRS, _drive, _make_host

        result = _drive(_make_host(tmp_path), _CALENDAR_STRS[0])
        assert result.driven is True
        path = tmp_path / "paper" / "runs" / f"readiness_{_CALENDAR_STRS[0]}.json"
        assert path.exists(), "通过日结构化就绪证据未落盘（P1-C-1）"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["kind"] == "readiness_evidence"
        assert payload["ok"] is True
        assert payload["coverage_quantiles"] == pytest.approx([0.95, 0.99])
        # 宿主夹具无 stock_basic ⇒ 判定域缺失：门侧记 error 且不判定（同域纪律），
        # 证据仍需完整可查（域对齐后的逐列 rates/verdicts 由 daily_assembly 套件覆盖）
        assert payload["rate_denominator"] == "unavailable"
        assert any(a["code"] == "coverage_domain_missing" for a in payload["coverage_alerts"])

    def test_failure_day_writes_both_records(self, tmp_path):
        from tests.test_v2_p4_t2_host import _CALENDAR_STRS, _make_host

        day = _CALENDAR_STRS[0]
        host = _make_host(tmp_path, panel_skip_days={day})
        host.run_t0(day)
        result = host.last_day_result
        assert result is not None and result.driven is False
        runs = tmp_path / "paper" / "runs"
        gap = json.loads((runs / f"gap_{day}.json").read_text(encoding="utf-8"))
        readiness = json.loads((runs / f"readiness_{day}.json").read_text(encoding="utf-8"))
        # gap 兼容产物补结构化字段（column_missing_rates + coverage 字段）
        assert gap["kind"] == "readiness_gap"
        assert "column_missing_rates" in gap
        assert "coverage_verdicts" in gap
        assert readiness["ok"] is False

    def test_write_failure_does_not_block_main_chain(self, tmp_path, monkeypatch):
        from src.lazybull.v2.hosts.paper import readiness_evidence as evidence_module
        from tests.test_v2_p4_t2_host import _CALENDAR_STRS, _drive, _make_host

        # 选择性拦截：仅 readiness 证据产物的原子替换失败（状态存储等其余
        # 原子写不受影响——旁路纪律只作用于证据链本身）
        real_replace = evidence_module.os.replace

        def _selective_replace(src, dst):
            if "readiness_" in str(dst):
                raise OSError("synthetic write failure")
            return real_replace(src, dst)

        monkeypatch.setattr(evidence_module.os, "replace", _selective_replace)
        # 旁路纪律：证据写盘失败 ⇒ 告警不阻断主链路，账户状态不受影响
        result = _drive(_make_host(tmp_path), _CALENDAR_STRS[0])
        assert result.driven is True
        assert not (tmp_path / "paper" / "runs" / f"readiness_{_CALENDAR_STRS[0]}.json").exists()


# ---------------------------------------------------------------------------
# 闭合项（reviewer3 P2-4/P2-5）：推送结构化摘要 + coordinator 模型列同源
# ---------------------------------------------------------------------------


class TestPushQualitySummary:
    def _report(self, blocked: bool, reason: str = ""):
        from src.lazybull.v2.hosts.paper.daily_assembly import ReadinessReport

        alerts = [
            {
                "severity": "error",
                "code": "rate_exceeded",
                "message": "x",
                "member": "v24022",
                "column": "cons_target_upside",
                "rate": 0.67,
            },
            {
                "severity": "warning",
                "code": "rate_exceeded",
                "message": "y",
                "member": "v24022",
                "column": "zscore_pe_ttm",
                "rate": 0.27,
            },
        ]
        return ReadinessReport(
            date="20261026",
            ok=not blocked,
            row_count=5211,
            row_count_band=(4682.7, 5723.3),
            gaps=[reason] if reason else [],
            uncovered_columns=[],
            derived_family_note="",
            column_missing_rates={},
            coverage_alerts=alerts,
            coverage_blocked=blocked,
            coverage_error_threshold=15,
            coverage_blocked_reason=reason,
        )

    def test_summary_contains_threshold_and_top_columns(self):
        from src.lazybull.v2.hosts.paper.readiness_evidence import append_quality_summary

        line = append_quality_summary("正文", self._report(False))
        assert "error 1 列" in line and "warning 1 列" in line
        assert "阈值 15 列" in line
        assert "cons_target_upside=67.0%" in line
        assert "聚合阻断" not in line

    def test_summary_contains_blocked_reason(self):
        from src.lazybull.v2.hosts.paper.readiness_evidence import append_quality_summary

        line = append_quality_summary("正文", self._report(True, "成员 v24022: 20 列 ≥ 阈值 15"))
        assert "聚合阻断" in line and "20 列 ≥ 阈值 15" in line


class TestCoordinatorModelColumns:
    def test_config_ab_members_resolved(self, tmp_path, monkeypatch):
        import scripts.v2_p4.daily_data_coordinator as coord

        models = tmp_path / "models" / "stock_selection"
        models.mkdir(parents=True)
        (models / "v1_features.json").write_text(json.dumps(["a", "shared"]), encoding="utf-8")
        (models / "v2_features.json").write_text(json.dumps(["shared", "b"]), encoding="utf-8")
        monkeypatch.setattr(coord, "_data_root", lambda: tmp_path)
        monkeypatch.setattr(coord, "_read_paper_model_versions", lambda: (1, 2))
        members, sources = coord._resolve_model_members()
        assert members == {1: ["a", "shared"], 2: ["shared", "b"]}
        assert sources == {1: "config_a", 2: "config_b"}
        columns, label = coord._resolve_model_columns()
        assert columns == ["a", "shared", "b"]
        assert label == "config_a:v1,config_b:v2"

    def test_missing_member_features_raises(self, tmp_path, monkeypatch):
        import scripts.v2_p4.daily_data_coordinator as coord

        models = tmp_path / "models" / "stock_selection"
        models.mkdir(parents=True)
        (models / "v1_features.json").write_text(json.dumps(["a"]), encoding="utf-8")
        monkeypatch.setattr(coord, "_data_root", lambda: tmp_path)
        monkeypatch.setattr(coord, "_read_paper_model_versions", lambda: (1, 9))
        with pytest.raises(RuntimeError, match="v9"):
            coord._resolve_model_members()
