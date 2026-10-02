# -*- coding: utf-8 -*-
"""P5a-2 偿还 F7 待办的脚本改造测试：run_power_calibration 输出后缀参数化 + 分节输出。

只测纯函数（build_report_dict / resolve_out_path / argparse 默认值），不跑真标定。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.v2_p5a1.run_power_calibration import build_report_dict, main, resolve_out_path


def _args(sections="both", out_name=None, purpose=None):
    return SimpleNamespace(
        n_noise=20, n_boot=100, shift_grid=[2.0, 5.0], shift_pp=2.0,
        threshold=0.8, seed=42, sections=sections, out_name=out_name, purpose=purpose,
    )


def _curve_point():
    return SimpleNamespace(
        shift_annual_pp=2.0, detection_prob_cagr=0.05, detection_prob_maxdd=0.0,
        n_noise=20, n_boot=100, noise_source="real_seed_swap_resample",
        effective_dof_note="note",
    )


def _consistency_result():
    return SimpleNamespace(
        seed_swap_indistinguishable=True, seed_swap_is_real=True, shift_pp=2.0,
        shift_detection_freq=0.05, shift_detect_threshold=0.8,
        shifted_detectable=False, criterion_passes=False, note="n",
    )


class TestResolveOutPath:
    def test_default_uses_date_pattern(self):
        p = resolve_out_path(None)
        assert p.name.startswith("p5a1_power_calibration_") and p.suffix == ".json"

    def test_custom_name_prevents_same_day_overwrite(self):
        p = resolve_out_path("p5a1_power_calibration_densegrid_20261001")
        assert p.name == "p5a1_power_calibration_densegrid_20261001.json"


class TestBuildReportDict:
    def test_both_sections_default(self):
        out = build_report_dict(_args(), [_curve_point()], {"cagr": _consistency_result()}, 14)
        assert "power_curve" in out and "criterion_self_consistency" in out
        assert out["config"]["sections"] == "both"
        assert "purpose" not in out

    def test_curve_only(self):
        out = build_report_dict(_args(sections="curve"), [_curve_point()], {}, 14)
        assert "power_curve" in out and "criterion_self_consistency" not in out

    def test_consistency_only(self):
        out = build_report_dict(_args(sections="consistency"), [], {"cagr": _consistency_result()}, 14)
        assert "power_curve" not in out
        assert out["criterion_self_consistency"]["cagr"]["criterion_passes"] is False

    def test_purpose_written_when_given(self):
        out = build_report_dict(_args(purpose="一次性修订窗口（B1）：密网格补测"), [_curve_point()], {}, 14)
        assert out["purpose"] == "一次性修订窗口（B1）：密网格补测"


class TestArgparse:
    def test_defaults_keep_legacy_behavior(self):
        from scripts.v2_p5a1.run_power_calibration import _build_parser
        args = _build_parser().parse_args([])
        assert args.sections == "both"
        assert args.out_name is None and args.purpose is None
        assert args.shift_grid == [2.0, 5.0, 8.0] and args.shift_pp == 2.0

    def test_out_name_json_suffix_stripped(self):
        assert resolve_out_path("x.json").name == "x.json"
        assert resolve_out_path("x").name == "x.json"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
