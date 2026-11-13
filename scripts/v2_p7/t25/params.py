# -*- coding: utf-8 -*-
"""P7 T2.5 冻结参数与输入身份（任务书 §3/§4 唯一口径）。

原则
====
- 全部参数只从**冻结输入**派生：锚批次 ``summary.parquet``、输入清单、
  T0.5 ``precheck`` 已核验附件；本模块不另立阈值；
- 任一 SHA 前缀不符、折表与 T0.5 ``fold_check`` 不逐值一致、信号日历不连续
  ⇒ fail-fast，且校验发生在任何 ``mkdir``/写字节之前（失败路径零副作用）；
- 信号域只取 14 折 ``test_start~test_end`` 的交易日并集；``bt_start/bt_end``
  **只进差异登记**，永不进入信号日历（任务书 §3「禁止」条）。

相位日历
========
``V20`` = 冻结信号域上的 ``compute_tranche_schedule(20, 2)``（相位 0）；
``V10`` = 同一排期整体后移 ``PHASE_OFFSET_DAYS`` 个交易日（超出日历尾部即丢弃）。
相位只移动建仓/再平衡日历，不改模型、分数、Top-N、20 日到期与两档拆分。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from scripts.v2_p7.common import REPO_ROOT, sha256_file

#: 锚批次（A 信号来源；只读，不继承其交易身份）
ANCHOR_REL = "data/runs/p2a_b0replay_20261008_144307_d79f3f"
ANCHOR_SUMMARY_SHA_PREFIX = "e47aea6d"

#: 冻结输入清单（模型内容 SHA 的唯一来源）
INPUT_MANIFEST_REL = "data/frozen_reference/p2a_b0_input_manifest_20261008_withconfig.json"
INPUT_MANIFEST_SHA_PREFIX = "c5eb587c"

#: T0.5 已核验附件根
PRECHECK_REL = "data/paper_shadow/p7_testbed/p7_t05_20261111_r6/precheck"

#: 测试床工作根（本 task 唯一写域，经 common.assert_targets_within_write_roots 校验）
TESTBED_REL = "data/paper_shadow/p7_testbed"

#: 冻结策略参数（任务书 §4；不得在本 task 重新选择）
TOP_N = 20
HOLDING_DAYS = 20
REBALANCE_FREQ = 20
STAGGER_TRANCHES = 2
PHASE_OFFSET_DAYS = 10
RUNOFF_REQUIRED_DAYS = 20
LOT_SIZE = 100

#: 袖初资（设计 §2.1：VV-H/VV-S 物理各 1,000,000；V20/V10 虚拟现金 700,000/300,000）
PHYSICAL_CAPITAL = 1_000_000.0
SLEEVE_CAPITAL: Mapping[str, float] = {"v20": 700_000.0, "v10": 300_000.0}
ARMS: Tuple[str, str] = ("v20", "v10")

#: 自然意图预检中**不作阻断项**的场景（用户 2026-11-12 裁决）。
#: S1 = 两袖对同一股票股数完全相等（严格等量抵消）：在冻结 half_kelly 下权重是分数的非常数函数，
#: 两袖初资/深度不同 ⇒ 单槽名义只在巧合下相等，属刀刃情形；净额化真正消费的是 S2/S3 与 S4/S5。
#: S1 缺位仍如实登记（`missing_scenarios`），并按 T4 的构造黄金例单独覆盖，不冒充自然观测。
NON_BLOCKING_SCENARIOS: Tuple[str, ...] = ("S1",)

#: 任务书 §4 冻结的信号域端点（逐值断言，防止日历漂移）
SIGNAL_DOMAIN_FIRST = "20181126"
SIGNAL_DOMAIN_LAST = "20260105"
SIGNAL_DOMAIN_DAYS = 1725
RUNOFF_START = "20260106"
RUNOFF_END = "20260202"

#: 再平衡次数（T0.5 已核验；仅作交叉校验）
V20_REBALANCE_COUNT = 173
V10_REBALANCE_COUNT = 172

DEFAULT_RUN_ID = "p7_t25_20261111_r1"

#: T0.5 precheck 必读附件（逐字节 SHA 登记）
PRECHECK_FILES: Tuple[str, ...] = (
    "fold_check.json",
    "model_identity.json",
    "phase_calendar.json",
    "panel_coverage.json",
    "source_watermarks.json",
)


class ParamsError(RuntimeError):
    """冻结输入缺件、字节不符或与冻结口径不一致（fail-fast，零副作用）。"""


@dataclass(frozen=True)
class FoldRow:
    """单折冻结身份（test 域为信号域；bt 域只作差异登记）。"""

    split_index: int
    model_version: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    bt_start: str
    bt_end: str
    signal_days: Tuple[str, ...]

    @property
    def signal_start(self) -> str:
        return self.signal_days[0]

    @property
    def signal_end(self) -> str:
        return self.signal_days[-1]

    def payload(self) -> Dict[str, Any]:
        return {
            "split_index": self.split_index,
            "model_version": self.model_version,
            "train_window": [self.train_start, self.train_end],
            "test_window": [self.test_start, self.test_end],
            "bt_window": [self.bt_start, self.bt_end],
            "test_bt_diff": {
                "start_days": _date_gap(self.bt_start, self.test_start),
                "end_days": _date_gap(self.bt_end, self.test_end),
            },
            "signal_days": len(self.signal_days),
            "signal_span": [self.signal_start, self.signal_end],
            "signal_days_sha256": days_digest(self.signal_days),
        }


@dataclass(frozen=True)
class T25Params:
    """T2.5 全部冻结输入身份与派生日历（manifest 直接序列化本对象）。"""

    repo_root: Path
    data_root: Path
    anchor_dir: Path
    anchor_summary_sha256: str
    input_manifest_path: Path
    input_manifest_sha256: str
    precheck_dir: Path
    precheck_sha256: Mapping[str, str]
    folds: Tuple[FoldRow, ...]
    signal_days: Tuple[str, ...]
    runoff_days: Tuple[str, ...]
    extended_days: Tuple[str, ...]
    sleeve_top_n: Mapping[str, int]
    sleeve_capital: Mapping[str, float]
    design_deviation: Tuple[str, ...]
    v20_schedule: Mapping[str, int]
    v10_schedule: Mapping[str, int]
    model_identity: Mapping[int, Mapping[str, str]]
    phase_reference: Mapping[str, Any]
    data_watermarks: Mapping[str, Any]

    # ---- 派生口径 ----
    @property
    def fold_by_split(self) -> Dict[int, FoldRow]:
        return {fold.split_index: fold for fold in self.folds}

    def model_version_of_day(self, day: str) -> int:
        """信号日所属折的当期模型版本（模型只在 ``test_start`` 信号点切换）。"""
        for fold in self.folds:
            if fold.signal_days[0] <= day <= fold.signal_days[-1]:
                return fold.model_version
        raise ParamsError(f"信号日不在任何折的 test 域内：{day}")

    def fold_of_day(self, day: str) -> FoldRow:
        for fold in self.folds:
            if fold.signal_days[0] <= day <= fold.signal_days[-1]:
                return fold
        raise ParamsError(f"信号日不在任何折的 test 域内：{day}")

    def current_model_version(self, day: str) -> int:
        """当期模型版本；runoff 日沿用末折模型（信号停止后不再切换）。"""
        for fold in self.folds:
            if fold.signal_days[0] <= day <= fold.signal_days[-1]:
                return fold.model_version
        if day > self.signal_days[-1]:
            return self.folds[-1].model_version
        raise ParamsError(f"日期不在信号域内：{day}")

    def current_split_index(self, day: str) -> int:
        """当期折号；runoff 日归末折（只作登记，不产生新信号）。"""
        for fold in self.folds:
            if fold.signal_days[0] <= day <= fold.signal_days[-1]:
                return fold.split_index
        if day > self.signal_days[-1]:
            return self.folds[-1].split_index
        raise ParamsError(f"日期不在信号域内：{day}")

    def schedule(self, arm: str) -> Mapping[str, int]:
        if arm == "v20":
            return self.v20_schedule
        if arm == "v10":
            return self.v10_schedule
        raise ParamsError(f"未知袖：{arm}")

    def params_payload(self) -> Dict[str, Any]:
        """manifest 用参数快照（相对路径，保证跨机器可读）。"""
        return {
            "anchor_run": ANCHOR_REL,
            "anchor_summary_sha256": self.anchor_summary_sha256,
            "input_manifest": INPUT_MANIFEST_REL,
            "input_manifest_sha256": self.input_manifest_sha256,
            "precheck_dir": PRECHECK_REL,
            "precheck_sha256": dict(self.precheck_sha256),
            "top_n": TOP_N,
            "sizing_basis": {
                "basis": "half_kelly",
                "frozen_anchor": {
                    "bt_max_weight_per_stock": 0.15,
                    "position_sizing": "half_kelly",
                    "kelly_vol_window": 60,
                    "kelly_max_leverage": 0.2,
                },
                "implementation": "compute_kelly_weights(half) → cap_and_normalize_weights；"
                "预算 = 袖 NAV（成交日）× 本批资金比例 × 批内权重",
                "status": "已按用户 2026-11-12 裁决接入冻结 sizing",
            },
            "holding_days": HOLDING_DAYS,
            "rebalance_freq": REBALANCE_FREQ,
            "stagger_tranches": STAGGER_TRANCHES,
            "phase_offset_trading_days": PHASE_OFFSET_DAYS,
            "runoff_days": RUNOFF_REQUIRED_DAYS,
            "lot_size": LOT_SIZE,
            "physical_capital": PHYSICAL_CAPITAL,
            "sleeve_top_n": dict(self.sleeve_top_n),
            "sleeve_capital": dict(self.sleeve_capital),
            "frozen_defaults": {
                "sleeve_top_n": {arm: TOP_N for arm in ARMS},
                "sleeve_capital": dict(SLEEVE_CAPITAL),
            },
            "design_deviation": list(self.design_deviation),
            "signal_domain": [self.signal_days[0], self.signal_days[-1]],
            "signal_days": len(self.signal_days),
            "runoff_window": [self.runoff_days[0], self.runoff_days[-1]],
            "extended_days": len(self.extended_days),
            "v20_rebalances": len(self.v20_schedule),
            "v10_rebalances": len(self.v10_schedule),
            "model_versions": [fold.model_version for fold in self.folds],
            "model_identity": {
                str(version): dict(files) for version, files in self.model_identity.items()
            },
            "data_watermarks": dict(self.data_watermarks),
        }


def days_digest(days: Sequence[str]) -> str:
    """交易日序列的稳定摘要（逐日换行拼接后 SHA256）。"""
    import hashlib

    payload = "\n".join(str(day) for day in days).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _date_gap(later: str, earlier: str) -> int:
    """两个 YYYYMMDD 的自然日差（仅为差异登记，不参与日历计算）。"""
    from datetime import datetime

    if not later or not earlier:
        return 0
    left = datetime.strptime(str(later), "%Y%m%d")
    right = datetime.strptime(str(earlier), "%Y%m%d")
    return int((left - right).days)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ParamsError(f"冻结输入缺件：{path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:  # pragma: no cover - 防御分支
        raise ParamsError(f"冻结输入不是合法 JSON：{path}（{error}）") from error


def load_trade_days(data_root: Path, start: str, end: str) -> List[str]:
    """交易日历 ∩ [start, end]（clean 日历优先，raw 兜底；与打分预期日同源）。"""
    from src.lazybull.data.loader import DataLoader
    from src.lazybull.data.storage import Storage

    loader = DataLoader(Storage(root_path=str(data_root)))
    calendar = loader.load_clean_trade_cal()
    if calendar is None or len(calendar) == 0:
        calendar = loader.load_trade_cal()
    if calendar is None or len(calendar) == 0:
        raise ParamsError(f"无法加载交易日历：{data_root}")
    mask = (
        (calendar["is_open"] == 1)
        & (calendar["cal_date"].astype(str) >= str(start))
        & (calendar["cal_date"].astype(str) <= str(end))
    )
    days = sorted(str(day) for day in calendar.loc[mask, "cal_date"].tolist())
    if not days:
        raise ParamsError(f"交易日历在 [{start}, {end}] 内无开市日")
    return days


def shift_schedule(
    schedule: Mapping[str, int],
    calendar: Sequence[str],
    offset: int,
) -> Dict[str, int]:
    """把排期整体后移 ``offset`` 个交易日（超出日历尾部的日期丢弃，不环绕）。"""
    if offset < 0:
        raise ParamsError(f"相位偏移必须非负：{offset}")
    index_of = {day: pos for pos, day in enumerate(calendar)}
    shifted: Dict[str, int] = {}
    for day, tranche in schedule.items():
        pos = index_of.get(str(day))
        if pos is None:
            raise ParamsError(f"排期日不在信号日历内：{day}")
        target = pos + offset
        if target >= len(calendar):
            continue
        shifted[str(calendar[target])] = int(tranche)
    return shifted


def build_phase_calendars(calendar: Sequence[str]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """构建 V20（相位 0）与 V10（整体后移）排期（内核 ``compute_tranche_schedule``）。"""
    from src.lazybull.v2.common.rules.stagger import compute_tranche_schedule

    v20 = {
        str(day): int(tranche)
        for day, tranche in compute_tranche_schedule(
            list(calendar), REBALANCE_FREQ, STAGGER_TRANCHES
        ).items()
    }
    v10 = shift_schedule(v20, calendar, PHASE_OFFSET_DAYS)
    _assert_phase_identity(v20, v10)
    return v20, v10


def _assert_phase_identity(v20: Mapping[str, int], v10: Mapping[str, int]) -> None:
    """相位冻结口径自检：次数、首末、无环绕与错峰平移。"""
    if len(v20) != V20_REBALANCE_COUNT:
        raise ParamsError(f"V20 再平衡次数不符：{len(v20)} != {V20_REBALANCE_COUNT}")
    if len(v10) != V10_REBALANCE_COUNT:
        raise ParamsError(f"V10 再平衡次数不符：{len(v10)} != {V10_REBALANCE_COUNT}")
    if sorted(v20.values()).count(0) != V20_REBALANCE_COUNT - len(v20) // 2:
        raise ParamsError("V20 两档拆分计数不符（tranche 0）")


def load_fold_table(anchor_dir: Path, precheck_dir: Path, data_root: Path) -> Tuple[FoldRow, ...]:
    """读取 14 折并逐值与 T0.5 ``fold_check`` 对账（缺件/不符即 fail-fast）。"""
    import pandas as pd

    summary_path = anchor_dir / "summary.parquet"
    if not summary_path.exists():
        raise ParamsError(f"锚批次 summary 缺件：{summary_path}")
    summary = pd.read_parquet(summary_path)
    required = {"split_index", "model_version", "test_start", "test_end"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ParamsError(f"锚批次 summary 缺列：{missing}")
    if len(summary) != 14:
        raise ParamsError(f"锚批次 summary 折数必须为 14：{len(summary)}")

    precheck = _read_json(precheck_dir / "fold_check.json")
    if not precheck.get("frozen_folds_match") or not precheck.get("test_domain_contiguous"):
        raise ParamsError("T0.5 fold_check 未通过（frozen_folds_match/test_domain_contiguous）")
    rows = {int(item["split_index"]): item for item in precheck["rows"]}
    coverage = _read_json(precheck_dir / "panel_coverage.json")
    if not coverage.get("signal_days_complete"):
        raise ParamsError("T0.5 panel_coverage 未通过（signal_days_complete=false）")
    expected_days = {
        int(item["split_index"]): int(item["expected_signal_days"]) for item in coverage["rows"]
    }
    folds: List[FoldRow] = []
    for _, record in summary.sort_values("split_index").iterrows():
        split_index = int(record["split_index"])
        check = rows.get(split_index)
        if check is None:
            raise ParamsError(f"T0.5 fold_check 缺 split{split_index:02d}")
        model_version = int(record["model_version"])
        test_start, test_end = str(record["test_start"]), str(record["test_end"])
        for key, value in (
            ("model_version", model_version),
            ("test_start", test_start),
            ("test_end", test_end),
        ):
            if str(check[key]) != str(value):
                raise ParamsError(
                    f"split{split_index:02d} {key} 与 T0.5 不符：{check[key]} != {value}"
                )
        signal_days = tuple(load_trade_days(data_root, test_start, test_end))
        if len(signal_days) != expected_days.get(split_index):
            raise ParamsError(
                f"split{split_index:02d} 交易日数与 T0.5 不符：{len(signal_days)} != "
                f"{expected_days.get(split_index)}"
            )
        folds.append(
            FoldRow(
                split_index=split_index,
                model_version=model_version,
                train_start=str(record.get("train_start", "")),
                train_end=str(record.get("train_end", "")),
                test_start=test_start,
                test_end=test_end,
                bt_start=str(check["bt_start"]),
                bt_end=str(check["bt_end"]),
                signal_days=signal_days,
            )
        )
    _assert_signal_domain(folds, data_root)
    return tuple(folds)


def _assert_signal_domain(folds: Sequence[FoldRow], data_root: Path) -> None:
    """信号域 = 14 折 test 域并集，必须连续、无重复、端点与冻结值一致。"""
    flat: List[str] = []
    for fold in folds:
        flat.extend(fold.signal_days)
    if len(flat) != len(set(flat)):
        raise ParamsError("14 折 test 域存在重复交易日")
    if len(flat) != SIGNAL_DOMAIN_DAYS:
        raise ParamsError(f"信号域交易日数不符：{len(flat)} != {SIGNAL_DOMAIN_DAYS}")
    if flat[0] != SIGNAL_DOMAIN_FIRST or flat[-1] != SIGNAL_DOMAIN_LAST:
        raise ParamsError(f"信号域端点不符：{flat[0]}~{flat[-1]}")
    calendar = load_trade_days(data_root, SIGNAL_DOMAIN_FIRST, SIGNAL_DOMAIN_LAST)
    if calendar != flat:
        raise ParamsError("14 折 test 域并集与连续交易日历不一致（存在缺口或插入日）")


def _sleeve_specs(
    counterpart_top_n: Optional[int],
    counterpart_capital: Optional[float],
) -> Tuple[Dict[str, int], Dict[str, float], Tuple[str, ...]]:
    """袖口径：基准袖固定冻结，对手袖可设计并逐项登记偏离。

    对手袖（``v10``）不对应任何真实策略，是**构造对手盘**：其深度/初资用于产生
    跨袖反向指令流。真实性约束只作用于信号、分数、价格、持仓、交易日历与整手，
    不作用于对手袖参数。
    """
    top_n = {arm: int(TOP_N) for arm in ARMS}
    capital = {arm: float(SLEEVE_CAPITAL[arm]) for arm in ARMS}
    deviation: List[str] = []
    if counterpart_top_n is not None:
        value = int(counterpart_top_n)
        if value < STAGGER_TRANCHES:
            raise ParamsError(f"对手袖 top_n 不得小于分批档数：{value}")
        top_n["v10"] = value
        if value != TOP_N:
            deviation.append(
                f"对手袖 top_n={value}（冻结 {TOP_N}）：构造对手盘深度，用于产生跨袖反向指令"
            )
    if counterpart_capital is not None:
        value = float(counterpart_capital)
        if value <= 0:
            raise ParamsError(f"对手袖初资必须为正：{value}")
        capital["v10"] = value
        if abs(value - float(SLEEVE_CAPITAL["v10"])) > 1e-9:
            slot = capital["v20"] / top_n["v20"]
            deviation.append(
                f"对手袖初资={value:.0f}（冻结 {SLEEVE_CAPITAL['v10']:.0f}），"
                f"基准袖单槽名义={slot:.0f}/槽"
            )
    return top_n, capital, tuple(deviation)


def build_params(
    repo_root: Path = REPO_ROOT,
    counterpart_top_n: Optional[int] = None,
    counterpart_capital: Optional[float] = None,
) -> T25Params:
    """组装并校验全部 T2.5 冻结输入（任何不符即 fail-fast）。

    Args:
        repo_root: 仓库根
        counterpart_top_n: 对手袖（``v10``）总目标持仓数；缺省取冻结值 ``TOP_N``。
            对手袖是**构造对手盘**（不对应任何真实策略），口径可按需设计；
            偏离冻结值时在 ``design_deviation`` 中逐项登记。
        counterpart_capital: 对手袖虚拟初资；缺省取冻结值 ``SLEEVE_CAPITAL['v10']``。
    """
    root = Path(repo_root).resolve()
    data_root = root / "data"
    anchor_dir = root / ANCHOR_REL
    precheck_dir = root / PRECHECK_REL

    summary_path = anchor_dir / "summary.parquet"
    if not summary_path.exists():
        raise ParamsError(f"锚批次 summary 缺件：{summary_path}")
    summary_sha = sha256_file(summary_path)
    if not summary_sha.startswith(ANCHOR_SUMMARY_SHA_PREFIX):
        raise ParamsError(f"锚批次 summary SHA 前缀不符：{summary_sha[:16]}")

    manifest_path = root / INPUT_MANIFEST_REL
    if not manifest_path.exists():
        raise ParamsError(f"输入清单缺件：{manifest_path}")
    manifest_sha = sha256_file(manifest_path)
    if not manifest_sha.startswith(INPUT_MANIFEST_SHA_PREFIX):
        raise ParamsError(f"输入清单 SHA 前缀不符：{manifest_sha[:16]}")

    precheck_sha: Dict[str, str] = {}
    for name in PRECHECK_FILES:
        path = precheck_dir / name
        if not path.exists():
            raise ParamsError(f"T0.5 预检附件缺件：{path}")
        precheck_sha[name] = sha256_file(path)

    folds = load_fold_table(anchor_dir, precheck_dir, data_root)
    calendar = [day for fold in folds for day in fold.signal_days]
    v20, v10 = build_phase_calendars(calendar)
    _assert_phase_reference(precheck_dir, v20, v10, calendar)

    runoff_days = tuple(load_trade_days(data_root, RUNOFF_START, RUNOFF_END))
    if len(runoff_days) != RUNOFF_REQUIRED_DAYS:
        raise ParamsError(f"runoff 交易日数不符：{len(runoff_days)} != {RUNOFF_REQUIRED_DAYS}")

    model_identity = _load_model_identity(data_root, precheck_dir, folds)
    watermarks = _load_watermarks(precheck_dir)
    sleeve_top_n, sleeve_capital, deviation = _sleeve_specs(counterpart_top_n, counterpart_capital)

    return T25Params(
        repo_root=root,
        data_root=data_root,
        anchor_dir=anchor_dir,
        anchor_summary_sha256=summary_sha,
        input_manifest_path=manifest_path,
        input_manifest_sha256=manifest_sha,
        precheck_dir=precheck_dir,
        precheck_sha256=precheck_sha,
        folds=folds,
        signal_days=tuple(calendar),
        runoff_days=runoff_days,
        extended_days=tuple(calendar) + runoff_days,
        sleeve_top_n=sleeve_top_n,
        sleeve_capital=sleeve_capital,
        design_deviation=deviation,
        v20_schedule=v20,
        v10_schedule=v10,
        model_identity=model_identity,
        phase_reference=_read_json(precheck_dir / "phase_calendar.json"),
        data_watermarks=watermarks,
    )


def _assert_phase_reference(
    precheck_dir: Path,
    v20: Mapping[str, int],
    v10: Mapping[str, int],
    calendar: Sequence[str],
) -> None:
    """相位日历与 T0.5 ``phase_calendar`` 逐项对账（次数/首末/平移/无环绕）。"""
    reference = _read_json(precheck_dir / "phase_calendar.json")
    _assert_phase_counts(reference, v20, v10)
    _assert_phase_ring(reference, v20, v10, calendar)


def _assert_phase_counts(
    reference: Mapping[str, Any], v20: Mapping[str, int], v10: Mapping[str, int]
) -> None:
    """冻结口径常量、再平衡次数与首末日逐项对账。"""
    if int(reference["holding_days"]) != HOLDING_DAYS:
        raise ParamsError("T0.5 持有期与冻结口径不符")
    if int(reference["stagger_tranches"]) != STAGGER_TRANCHES:
        raise ParamsError("T0.5 错峰档数与冻结口径不符")
    if int(reference["phase_offset_trading_days"]) != PHASE_OFFSET_DAYS:
        raise ParamsError("T0.5 相位偏移与冻结口径不符")
    if len(v20) != int(reference["v20_rebalance_dates"]):
        raise ParamsError("V20 再平衡次数与 T0.5 不符")
    if len(v10) != int(reference["v10_rebalance_dates"]):
        raise ParamsError("V10 再平衡次数与 T0.5 不符")
    if [min(v20), max(v20)] != list(reference["v20_first_last"]):
        raise ParamsError("V20 首末再平衡日与 T0.5 不符")
    if [min(v10), max(v10)] != list(reference["v10_first_last"]):
        raise ParamsError("V10 首末再平衡日与 T0.5 不符")


def _assert_phase_ring(
    reference: Mapping[str, Any],
    v20: Mapping[str, int],
    v10: Mapping[str, int],
    calendar: Sequence[str],
) -> None:
    """平移环、无环绕与首日档位标签（``v10_shifted_matches_v20``）。"""
    del reference  # 环检查只用排期本身，保留参数以便调用点对称
    index_of = {day: pos for pos, day in enumerate(calendar)}
    for day, tranche in v20.items():
        pos = index_of[day] + PHASE_OFFSET_DAYS
        if pos < len(calendar) and v10.get(calendar[pos]) != tranche:
            raise ParamsError(f"V10 平移环在 {day} 处破缺（v10_shifted_matches_v20）")
    if any(index_of[day] < PHASE_OFFSET_DAYS for day in v10):
        raise ParamsError("V10 出现环绕（早于相位偏移位置的再平衡日）")
    if index_of[min(v10)] - index_of[min(v20)] != PHASE_OFFSET_DAYS:
        raise ParamsError("V10 首个再平衡日与 V20 的相位差不为冻结偏移")
    if v10[min(v10)] != v20[min(v20)]:
        raise ParamsError("V10 首个再平衡日的档位标签与 V20 不一致（平移应保档位）")


def _load_model_identity(
    data_root: Path,
    precheck_dir: Path,
    folds: Sequence[FoldRow],
) -> Dict[int, Dict[str, str]]:
    """逐折模型字节身份：本地重算 SHA 并与 T0.5 ``model_identity`` 逐项对账。"""
    from scripts.v2_p3.t05b_spike_runner import collect_model_identity

    reference = _read_json(precheck_dir / "model_identity.json")
    if not reference.get("identity_match"):
        raise ParamsError("T0.5 model_identity 未通过（identity_match=false）")
    expected: Dict[str, str] = {}
    for row in reference["rows"]:
        expected[str(row["file"])] = str(row["sha256"])
    identity: Dict[int, Dict[str, str]] = {}
    for fold in folds:
        version = fold.model_version
        files = collect_model_identity(data_root, version)
        for name, digest in files.items():
            key = f"models/stock_selection/{name}"
            if expected.get(key) != digest:
                raise ParamsError(f"模型字节与 T0.5 不符：{key}")
        identity[version] = dict(files)
    return identity


def compute_input_identity(params: T25Params, accepted_drift: Sequence[str] = ()) -> Dict[str, Any]:
    """本 run **实际读入**的数据内容身份（逐日 SHA + 与冻结输入清单对账）。

    覆盖推理与目标/估值实际打开的源：``features/cs_train``（全部信号日）、
    ``clean/daily``（含 runoff 在内的全部扩展日）、``clean/trade_cal``、``clean/stock_basic``。
    与冻结清单比对时只比**清单里有的键**，逐条登记不一致项（不做覆盖率门槛）。

    ``accepted_drift`` 形如 ``data/clean/stock_basic.parquet:<sha256>``：只有调用方**显式**
    声明并要求接受该新身份时，该项才从 ``frozen_manifest_mismatch`` 移入
    ``accepted_input_drift``（仍逐条登记，不静默放行）。
    """
    manifest = _read_json(params.input_manifest_path).get("files", {})
    accepted_set = {str(item) for item in accepted_drift}

    def _expected(rel: str) -> str:
        key = rel.removeprefix("data/")
        entry = manifest.get(key) or manifest.get(rel) or {}
        return str(entry.get("sha256") or "")

    def _digest(rel_paths: Sequence[str]) -> Dict[str, Any]:
        rows: Dict[str, str] = {}
        mismatched: List[str] = []
        accepted: List[Dict[str, str]] = []
        size_bytes = 0
        for rel in rel_paths:
            path = params.data_root.parent / rel
            if not path.exists():
                raise ParamsError(f"本 run 应读入的数据缺件：{path}")
            rows[rel] = sha256_file(path)
            size_bytes += path.stat().st_size
            expected = _expected(rel)
            if expected and expected != rows[rel]:
                if f"{rel}:{rows[rel]}" in accepted_set:
                    accepted.append({"path": rel, "frozen": expected, "current": rows[rel]})
                else:
                    mismatched.append(rel)
        return {
            "partitions": len(rel_paths),
            "size_mb": round(size_bytes / 1e6, 1),
            "digest": days_digest([f"{rel}:{rows[rel]}" for rel in rel_paths]),
            "frozen_manifest_checked": sum(
                1
                for rel in rel_paths
                if (manifest.get(rel.removeprefix("data/")) or {}).get("sha256")
            ),
            "frozen_manifest_mismatch": mismatched,
            "accepted_input_drift": accepted,
        }

    cs_train = [f"data/features/cs_train/{day}.parquet" for day in params.signal_days]
    daily = [
        f"data/clean/daily/{day[:4]}-{day[4:6]}-{day[6:]}.parquet" for day in params.extended_days
    ]
    meta_paths = ["data/clean/trade_cal.parquet", "data/clean/stock_basic.parquet"]
    meta_rows: Dict[str, Any] = {}
    meta_mismatch: List[str] = []
    meta_accepted: List[Dict[str, str]] = []
    for rel in meta_paths:
        path = params.data_root.parent / rel
        if not path.exists():
            raise ParamsError(f"本 run 应读入的数据缺件：{path}")
        digest = sha256_file(path)
        expected = _expected(rel)
        status = "match" if expected and expected == digest else "changed"
        if expected and expected != digest:
            if f"{rel}:{digest}" in accepted_set:
                meta_accepted.append({"path": rel, "frozen": expected, "current": digest})
                status = "accepted_drift"
            else:
                meta_mismatch.append(rel)
        meta_rows[rel] = {"sha256": digest, "frozen_sha256": expected, "status": status}
    cs_train_rows = _digest(cs_train)
    daily_rows = _digest(daily)
    return {
        "schema": "p7_t25_input_identity/v1",
        "cs_train": cs_train_rows,
        "clean_daily": daily_rows,
        "clean_meta": meta_rows,
        "clean_meta_mismatch": meta_mismatch,
        "accepted_input_drift": (
            list(meta_accepted)
            + list(cs_train_rows["accepted_input_drift"])
            + list(daily_rows["accepted_input_drift"])
        ),
        "note": "SHA 在 run 内对实际读入文件重算；不一致项逐条登记，只有显式 accept 才放行。",
    }


def _load_watermarks(precheck_dir: Path) -> Dict[str, Any]:
    """数据水位（clean 行情 / 特征面板 / 源截止日），只作身份登记。"""
    source = _read_json(precheck_dir / "source_watermarks.json")
    panel = _read_json(precheck_dir / "panel_coverage.json")
    return {
        "common_data_cutoff": source.get("common_data_cutoff"),
        "clean_daily_range": source["sources"]["clean_daily"]["range"],
        "clean_daily_partitions": source["sources"]["clean_daily"]["partition_count"],
        "cs_train_range": panel.get("cs_train_range"),
        "cs_train_partitions": panel.get("cs_train_partitions"),
        "cs_infer_range": panel.get("cs_infer_range"),
        "signal_days_complete": panel.get("signal_days_complete"),
        "signal_days_missing_total": panel.get("signal_days_missing_total"),
    }
