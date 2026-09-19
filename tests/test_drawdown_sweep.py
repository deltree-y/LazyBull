"""回撤侧总扫描分析（`scripts/compare/drawdown_sweep.py`）单元测试。

只使用合成数据（不依赖真实配置与生产数据），锁定以下口径：

1. 链式净值 `date` 为折内序号 ⇒ 必须按汇总的 `bt_start/bt_end` + 交易日历还原真实日期轴；
2. 窗口 = **与评估段相交的整折**，切片归一化到起点（与 R-004 §7/§8 登记口径一致）；
3. 日差归因按信号期（λ<1）/ 信号期外（λ=1）分组，折边界重复日按前一折保留；
4. 因果洁净判定（窗口外逐位一致）必须严格按 round(12) 比较；
5. 逐折判定取汇总 CSV 的全折口径（M2 要求窗口内折改善 ≥ 3/4）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from compare.drawdown_sweep import (  # noqa: E402
    _to_date_str,
    day_diff_attribution,
    fold_pass_summary,
    foreign_window_identity,
    load_arm_nav,
    load_exposure_table,
    per_fold_metrics,
    trade_summary,
)

CAL = pd.Series([f"2024{m:02d}{d:02d}" for m in (1, 2, 3, 4) for d in range(1, 6)])  # 20 个交易日


def _write_arm(root: Path, name: str, navs, splits, summary_rows, trades=None) -> Path:
    batch = root / name
    raw = batch / "raw"
    raw.mkdir(parents=True)
    pd.DataFrame({"date": list(range(len(navs))), "nav": navs, "split_index": splits}).to_csv(
        raw / "chain_nav_wf_x.csv", index=False
    )
    pd.DataFrame(summary_rows).to_csv(raw / "walk_forward_summary_0101_0001.csv", index=False)
    if trades is not None:
        for split, frame in trades.items():
            frame.to_csv(raw / f"walk_forward_trades_wf_x_split{split:02d}.csv", index=False)
    return batch


def _summary_rows(start_offsets):
    """按折给出行情窗口（用 CAL 的切片构造 bt_start/bt_end）。"""
    rows = []
    for split, (lo, hi) in enumerate(start_offsets):
        rows.append(
            {
                "split_index": split,
                "bt_start": CAL[lo],
                "bt_end": CAL[hi],
                "bt_total_return": 0.01 * (split + 1),
                "bt_max_drawdown": -0.10 - 0.01 * split,
                "bt_sharpe": 0.5 + split,
            }
        )
    return rows


@pytest.fixture()
def synthetic(tmp_path):
    """两折窗口的最小合成批次：折窗口 [0,9] 与 [9,18]（含**折边界重复日**）。"""
    rows = _summary_rows([(0, 9), (9, 18)])
    navs = [1.0 + 0.01 * i for i in range(20)]
    base = _write_arm(tmp_path, "arm_base", navs, [0] * 10 + [1] * 10, rows)
    arm = _write_arm(
        tmp_path, "arm_arm", [1.0 + 0.012 * i for i in range(20)], [0] * 10 + [1] * 10, rows
    )
    return tmp_path, base, arm


def test_to_date_str_normalizes_variants():
    assert _to_date_str("2024-01-31") == "20240131"
    assert _to_date_str("20240131") == "20240131"
    assert _to_date_str(pd.Timestamp("2024-01-31")) == "20240131"
    assert _to_date_str(None) == ""


def test_load_arm_nav_reconstructs_calendar_dates(synthetic):
    _, base, _ = synthetic
    # 窗口只与第一折相交 ⇒ 切片仅含该整折（10 行）
    full, window = load_arm_nav(base, CAL, "20240101", "20240201")
    assert list(full["date"][:3]) == ["20240101", "20240102", "20240103"]
    assert len(full) == 20
    assert len(window) == 10
    assert window["nav"].iloc[0] == pytest.approx(1.0)
    # 窗口与两折都相交 ⇒ 两折全含（20 行），折边界重复日保留（不去重）
    _, both = load_arm_nav(base, CAL, "20240101", "20240405")
    assert len(both) == 20
    assert both["date"].duplicated().sum() == 1


def test_load_arm_nav_detects_row_count_mismatch(tmp_path):
    rows = _summary_rows([(0, 9)])
    batch = _write_arm(tmp_path, "arm_bad", [1.0] * 5, [0] * 5, rows)  # 行数不足
    with pytest.raises(ValueError, match="交易日数"):
        load_arm_nav(batch, CAL)


def test_load_exposure_table_defaults_missing_days_to_one(tmp_path):
    path = tmp_path / "table.csv"
    pd.DataFrame({"日期": ["20240101", "20240102"], "暴露系数": [0.5, 1.0]}).to_csv(
        path, index=False
    )
    table = load_exposure_table(path)
    assert table is not None
    assert table["20240101"] == 0.5
    assert table.get("20240103", 1.0) == 1.0


def test_day_diff_attribution_groups_by_signal_window(synthetic, tmp_path):
    _, base, arm = synthetic
    _, base_window = load_arm_nav(base, CAL, "20240101", "20240405")
    _, arm_window = load_arm_nav(arm, CAL, "20240101", "20240405")
    # 前 10 个交易日 λ=0.5（信号期），其后 λ=1.0（信号期外）
    table_path = tmp_path / "t.csv"
    rows = [{"日期": CAL[i], "暴露系数": 0.5 if i < 10 else 1.0} for i in range(20)]
    pd.DataFrame(rows).to_csv(table_path, index=False)
    table = load_exposure_table(table_path)
    detail, summary = day_diff_attribution(base_window, arm_window, table)
    inside = summary["信号期内（λ<1）_日数"]
    outside = summary["信号期外（λ=1）_日数"]
    # 20 行 - 每折首行收益 NaN - 1 个折边界重复日 = 18 个有效日差
    assert inside == 9
    assert outside == 9
    assert summary["日差合计_pp"] == pytest.approx(detail["日差_pp"].sum())
    assert summary["日差合计_pp"] > 0  # 臂收益更高


def test_day_diff_attribution_requires_same_dates(synthetic):
    _, base, arm = synthetic
    _, base_window = load_arm_nav(base, CAL, "20240101", "20240405")
    _, arm_window = load_arm_nav(arm, CAL, "20240101", "20240405")
    with pytest.raises(ValueError, match="交易日集合完全一致"):
        day_diff_attribution(base_window.iloc[:5], arm_window, None)


def test_foreign_window_identity_detects_perturbation(synthetic):
    _, base, arm = synthetic
    base_full, _ = load_arm_nav(base, CAL, "20240101", "20240405")
    arm_full, _ = load_arm_nav(arm, CAL, "20240101", "20240405")
    identity = foreign_window_identity(base_full, arm_full)
    assert identity["一致"] is False
    assert foreign_window_identity(base_full, base_full.copy())["一致"] is True


def test_per_fold_metrics_uses_summary(synthetic):
    _, base, _ = synthetic
    folds = per_fold_metrics(base)
    assert list(folds.columns) == ["折序号", "起", "止", "收益", "最大回撤", "夏普"]
    assert folds["收益"].tolist() == [0.01, 0.02]


def test_fold_pass_summary_requires_three_of_four():
    folds = pd.DataFrame(
        [
            {"臂": "A0", "折序号": 10, "最大回撤": -0.20, "收益": 0.1},
            {"臂": "A0", "折序号": 11, "最大回撤": -0.20, "收益": 0.1},
            {"臂": "A0", "折序号": 12, "最大回撤": -0.20, "收益": 0.1},
            {"臂": "A0", "折序号": 13, "最大回撤": -0.20, "收益": 0.1},
            {"臂": "A1", "折序号": 10, "最大回撤": -0.10, "收益": 0.2},
            {"臂": "A1", "折序号": 11, "最大回撤": -0.15, "收益": -0.1},
            {"臂": "A1", "折序号": 12, "最大回撤": -0.25, "收益": 0.2},
            {"臂": "A1", "折序号": 13, "最大回撤": -0.05, "收益": 0.2},
        ]
    )
    result = fold_pass_summary(folds, "A0").set_index("臂")
    assert result.loc["A1", "窗口内折数"] == 4
    assert result.loc["A1", "MaxDD 改善折数"] == 3
    assert result.loc["A1", "改善要求"] == 3
    assert result.loc["A1", "逐折收益为正折数"] == 3


def test_trade_summary_classifies_action_types():
    trades = pd.DataFrame(
        [
            {"action": "buy", "amount": 1000.0, "cost": 1.0, "buy_type": "plan"},
            {"action": "buy", "amount": 500.0, "cost": 0.5, "buy_type": "risk_replenish"},
            {"action": "sell", "amount": 800.0, "cost": 0.8, "sell_type": "risk_trim"},
            {"action": "sell", "amount": 300.0, "cost": 0.3, "sell_type": "stop_loss"},
        ]
    )
    stats = trade_summary(trades)
    assert stats["买入笔数"] == 2
    assert stats["窗口买入金额"] == 1500.0
    assert stats["减仓笔数"] == 1 and stats["减仓金额"] == 800.0
    assert stats["回补笔数"] == 1 and stats["回补金额"] == 500.0
    assert stats["止损笔数"] == 1 and stats["止损金额"] == 300.0


def test_trade_summary_handles_legacy_frames_without_buy_type():
    """旧臂（回补上线前）成交表没有 buy_type 列 ⇒ 回补计为 0，不得报错。"""
    trades = pd.DataFrame(
        [
            {"action": "buy", "amount": 100.0, "cost": 0.1},
            {"action": "sell", "amount": 50.0, "cost": 0.05},
        ]
    )
    stats = trade_summary(trades)
    assert stats["回补笔数"] == 0
    assert stats["减仓笔数"] == 0
    assert stats["止损笔数"] == 0
    assert stats["窗口买入金额"] == 100.0
