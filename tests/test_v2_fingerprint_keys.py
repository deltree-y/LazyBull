# -*- coding: utf-8 -*-
"""配置指纹键清单对账测试（runs 契约附录 A / baseline_freeze 附录 A 的代码承载）。

对账语义：解析 `docs/contracts/baseline_freeze.md` 附录 A 的 127 键快照，
逐键断言 `fingerprint_keys` 模块的分类（入指纹 / 排除）与文档排除清单一致——
文档与代码任一侧漂移都会在此失败（fail-safe 方向）。
"""

import re
from pathlib import Path

import pytest

from src.lazybull.v2.evidence.fingerprint_keys import (
    EXCLUDE_EXACT,
    EXCLUDE_PREFIX,
    config_fingerprint_keys,
    fingerprint,
    is_fingerprint_key,
)

ROOT = Path(__file__).resolve().parents[1]
FREEZE_DOC = ROOT / "docs" / "contracts" / "baseline_freeze.md"

#: 快照中 KEY_* 族键（文档快照剥离了 KEY_ 前缀）→ 转换后形态 key_* 小写
_KEY_FAMILY = {
    "Top20_list", "Top30_list", "Top20_hit_rate", "Top20_avg_return_median",
    "Top20_lift_mean", "Top30_hit_rate", "Top30_avg_return_median", "Top30_lift_mean",
}
#: 快照中的非 ASCII 诊断列（KEY_说明）：转换层 ASCII 过滤丢弃，不参与指纹分类
_NON_ASCII_KEYS = {"说明"}

#: 快照 127 键中预期排除的键（转换后形态；文档附录 A 排除清单的逐键展开）
_EXPECTED_EXCLUDED = (
    {"wf_run_id", "batch_run_id", "batch_period_label"}  # 运行标识（registered_at 不在快照）
    | {"data_state_id", "git_commit", "git_dirty",
       "data_daily_latest", "data_cs_train_latest", "data_dividend_coverage"}  # 数据/代码态
    | {"bt_total_return", "bt_annual_return", "bt_max_drawdown", "bt_volatility",
       "bt_sharpe", "bt_calmar", "bt_trading_days", "bt_start", "bt_end"}  # bt_* 统计列
    | {"train_samples", "val_samples", "test_samples",
       "best_iteration", "best_iteration_floor_triggered"}  # 训练产出
    | {"key_" + k.lower() for k in _KEY_FAMILY}  # 信号层统计（KEY_* 映射列）
)


def _snapshot_keys() -> list:
    """从 baseline_freeze.md 附录 A 提取 127 键快照（代码块，逗号分隔）。"""
    text = FREEZE_DOC.read_text(encoding="utf-8")
    appendix = text[text.index("附录 A"):]
    block = re.search(r"```\n(.*?)```", appendix, re.DOTALL)
    assert block, "baseline_freeze.md 附录 A 缺快照代码块"
    return [k.strip() for k in block.group(1).replace("\n", " ").split(",") if k.strip()]


def _to_converted_form(key: str) -> str:
    """快照键 → 转换后 summary 列名（KEY_ 族补 key_ 前缀并小写，其余原样）。"""
    return "key_" + key.lower() if key in _KEY_FAMILY else key


class TestFingerprintKeys:
    def test_snapshot_has_127_keys(self):
        """快照键数 = 文档自称的 127（漂移即失败）。"""
        assert len(_snapshot_keys()) == 127

    def test_per_key_classification_matches_doc(self):
        """逐键对账：快照每键的分类与文档排除清单一致。"""
        for raw_key in _snapshot_keys():
            if raw_key in _NON_ASCII_KEYS:
                continue
            key = _to_converted_form(raw_key)
            expected = key not in _EXPECTED_EXCLUDED
            assert is_fingerprint_key(key) == expected, (
                f"键 {raw_key}（转换后 {key}）分类与文档不符: "
                f"预期入指纹={expected}，实际={is_fingerprint_key(key)}"
            )

    def test_bt_config_keys_stay_in_fingerprint(self):
        """fail-safe 核心：bt_* 前缀的配置键一律入指纹（仅统计列排除）。"""
        for key in ("bt_top_n", "bt_rebalance_freq", "bt_initial_capital", "bt_sell_timing",
                    "bt_exclude_st", "bt_min_list_days", "bt_max_weight_per_stock",
                    "bt_max_per_industry", "bt_stop_loss_enabled"):
            assert is_fingerprint_key(key), f"配置键 {key} 必须入指纹"

    def test_new_key_defaults_to_fingerprint(self):
        """fail-safe 方向：未知新键默认入指纹。"""
        assert is_fingerprint_key("some_future_config_key")

    def test_exclude_lists_match_module_surface(self):
        """模块清单自洽：EXCLUDE_EXACT 无 key_* 前缀成员，前缀清单只有 key_。"""
        assert EXCLUDE_PREFIX == ("key_",)
        assert not any(k.startswith("key_") for k in EXCLUDE_EXACT)

    def test_doc_mentions_all_exact_keys(self):
        """双向同步：文档附录 A 文本覆盖 EXCLUDE_EXACT 每个键名。"""
        text = FREEZE_DOC.read_text(encoding="utf-8")
        appendix = text[text.index("附录 A"):]
        for key in sorted(EXCLUDE_EXACT):
            assert key in appendix, f"baseline_freeze.md 附录 A 未登记排除键 {key}"

    def test_config_fingerprint_keys_filters(self):
        cols = ["bt_top_n", "bt_total_return", "key_top20_hit_rate", "wf_run_id", "top_n_new"]
        assert config_fingerprint_keys(cols) == ["bt_top_n", "top_n_new"]

    def test_fingerprint_stable_and_sensitive(self):
        """指纹口径冻结：同 dict 同指纹；改任一键值指纹变化。"""
        cfg = {"a": 1, "b": "x"}
        assert fingerprint(cfg) == fingerprint({"b": "x", "a": 1})  # 键序无关
        assert fingerprint(cfg) != fingerprint({"a": 2, "b": "x"})
        assert len(fingerprint(cfg)) == 16


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
