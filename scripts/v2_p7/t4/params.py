# -*- coding: utf-8 -*-
"""P7 T4 冻结参数、T2.5 输入身份与受审旧面清单。

原则
====
- 本 task **不以 B0 存活为输入**，也不引用 B0 线任何结果；唯一输入是 T2.5 已收口
  证据根（只读引用）与 Q3=A 费用裁决（``docs/contracts/netting_freeze.md`` F26 §P7 注记）；
- 全部参数只从**冻结输入**派生：T2.5 ``manifest.json`` / ``intent_precheck.json`` /
  ``targets/calendar.json``；任一缺件、SHA 不符或计数不符 => fail-fast，
  校验发生在任何 ``mkdir``/写字节之前（失败路径零副作用）；
- 受审旧面（``virtual_merger.py`` / ``virtual_ledger.py`` / ``reconciliation.py`` /
  ``order_chain.py`` / ``virtual_native_engine.py`` / P5b ``paths|journal|registration|
  family_txn.py`` / ``scripts/v2_p7/t25/**``）只登记 SHA，**不读写**其内容以外的状态。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from scripts.v2_p7.common import REPO_ROOT, sha256_file

#: T2.5 已收口证据根（只读引用；不得改写）
T25_ROOT_REL = "data/paper_shadow/p7_testbed/p7_t25_hk_20261112_r6"

#: 测试床工作根（本 task 唯一写域）
TESTBED_REL = "data/paper_shadow/p7_testbed"

#: 默认 run 身份（整改一律新 run_id 重取，不覆盖任何完成根）
DEFAULT_RUN_ID = "p7_t4_20261112_r1"

#: T2.5 冻结的**全窗**（1,745 日 = 信号域 1,725 + runoff 20）行情摘要（C4-03）。
#: 上游 ``manifest.input_identity.clean_daily.digest`` 必须等于本值，否则视为上游被改。
FROZEN_MARKET_DAILY_DIGEST = "806ea4fd9b85f4ea5147d88f09023c851f1724ce711d410c6426061746e76404"

#: 行情/日历身份算法版本（纳入 params 身份；算法变更必须显式改版本）
MARKET_IDENTITY_SCHEMA = "p7_t4_market_identity/v1"

#: 两相位虚拟袖（V20 相位 0；V10 整体后移）
ARMS: Tuple[str, str] = ("v20", "v10")

#: 设计 §2.1 冻结：物理初资 1,000,000；V20/V10 虚拟现金 700,000/300,000
PHYSICAL_CAPITAL = 1_000_000.0
SLEEVE_CAPITAL: Mapping[str, float] = {"v20": 700_000.0, "v10": 300_000.0}

#: 整手与持期（T2.5 已冻结，本 task 不改）
LOT_SIZE = 100
HOLDING_DAYS = 20

#: T2.5 冻结的原始意图计数（T4 必须**逐笔复算一致**；S1 恒为自然 0）
FROZEN_INTENT_COUNTS: Mapping[str, int] = {"S1": 0, "S2": 20, "S3": 56, "S4": 695, "S5": 695}

#: T2.5 机械栏登记状态（S1 缺位为已登记缺口，由构造黄金例覆盖）
FROZEN_NATURAL_STATUS = "observed_with_registered_gap"

#: 冻结成本参数（``configs/base.yaml`` -> ``get_cost_settings()`` 的当值快照）
FROZEN_COST_SETTINGS: Mapping[str, float] = {
    "commission_rate": 0.0001954,
    "min_commission": 5.0,
    "stamp_tax": 0.0005,
    "slippage": 0.0005,
}

#: 受审旧面（禁改；只登记 SHA + 最近提交）
LEGACY_FILES: Tuple[str, ...] = (
    "src/lazybull/v2/core/decision/virtual_merger.py",
    "src/lazybull/v2/core/accounting/virtual_ledger.py",
    "src/lazybull/v2/core/accounting/reconciliation.py",
    "src/lazybull/v2/hosts/order_chain.py",
    "src/lazybull/v2/hosts/virtual_native_engine.py",
    "src/lazybull/v2/hosts/testbed/paths.py",
    "src/lazybull/v2/hosts/testbed/journal.py",
    "src/lazybull/v2/hosts/testbed/registration.py",
    "src/lazybull/v2/hosts/testbed/family_txn.py",
)

#: T2.5 已收口代码面（``scripts/v2_p7/t25/**``；确需改动先停并申请）
LEGACY_T25_GLOB = "scripts/v2_p7/t25"

#: 旧证据已登记逐件源码 SHA 的交叉校验来源（只做一致性核对，不新增口径）
LEGACY_EVIDENCE_MANIFESTS: Tuple[str, ...] = (
    "data/frozen_reference/v2_p65/v2p65_t5_20261110_7d3ce51f/manifest.json",
    "data/frozen_reference/v2_p65/v2p65_t5_20261110_b22ba220/manifest.json",
)

#: 状态机场景标签（``docs/contracts/netting_freeze.md`` §3）
SCENARIOS: Tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5", "S6")
REVERSE_SCENARIOS: Tuple[str, ...] = ("S1", "S2", "S3")
MERGE_SCENARIOS: Tuple[str, ...] = ("S4", "S5")

#: 成交阻断原因（市场可达性 => 跨日顺延；资金不足 => 日终作废，归 T5 授权/借用）
MARKET_BLOCK_REASONS: Tuple[str, ...] = (
    "no_valid_open",
    "row_absent",
    "suspended",
    "limit_up_blocked",
    "limit_down_blocked",
)
FUNDING_BLOCK_REASONS: Tuple[str, ...] = ("insufficient_cash",)

#: T4 状态信封 generation 起点（同一 run 内逐交易日单调递增）
INITIAL_GENERATION = 1


class ParamsError(RuntimeError):
    """冻结输入缺件、字节不符或与冻结口径不一致（fail-fast，零副作用）。"""


@dataclass(frozen=True)
class T25Input:
    """T2.5 证据根输入身份（逐件 SHA；T4 只读引用）。"""

    root: Path
    manifest_sha256: str
    precheck_sha256: str
    calendar_sha256: str
    plans_sha256: Mapping[str, str]
    holdings_sha256: Mapping[str, str]
    daily_sha256: Mapping[str, str]
    #: 上游 SHA 核验结果（C4-03：不是只采集）
    input_verification: Mapping[str, Any]
    #: 上游登记的输入内容身份（``manifest.input_identity``；行情/日历对账基准）
    input_identity: Mapping[str, Any]
    source_digest: str
    natural_counts: Mapping[str, int]
    natural_status: str
    signal_domain: Tuple[str, str]
    runoff_window: Tuple[str, str]
    calendar_days: Tuple[str, ...]
    #: ``(split_index, signal_start, signal_end, model_version)``（逐折信号域与模型）
    folds: Tuple[Tuple[int, str, str, int], ...] = ()

    def payload(self) -> Dict[str, Any]:
        return {
            "root": T25_ROOT_REL,
            "manifest_sha256": self.manifest_sha256,
            "intent_precheck_sha256": self.precheck_sha256,
            "calendar_sha256": self.calendar_sha256,
            "plans_sha256": dict(self.plans_sha256),
            "holdings_sha256": dict(self.holdings_sha256),
            "daily_sha256": dict(self.daily_sha256),
            "input_identity": dict(self.input_identity),
            "source_digest": self.source_digest,
            "natural_counts": dict(self.natural_counts),
            "natural_status": self.natural_status,
            "signal_domain": list(self.signal_domain),
            "runoff_window": list(self.runoff_window),
            "calendar_days": len(self.calendar_days),
            "folds": [
                {
                    "split_index": split,
                    "signal_start": start,
                    "signal_end": end,
                    "model_version": model,
                }
                for split, start, end, model in self.folds
            ],
        }


@dataclass(frozen=True)
class T4Params:
    """T4 全部冻结输入身份与派生口径（manifest 直接序列化本对象）。"""

    repo_root: Path
    data_root: Path
    t25: T25Input
    calendar: Tuple[str, ...]
    runoff_days: Tuple[str, ...]
    extended_days: Tuple[str, ...]
    arms: Tuple[str, ...]
    physical_capital: float
    sleeve_capital: Mapping[str, float]
    lot_size: int
    cost_settings: Mapping[str, float]
    legacy_sha: Mapping[str, Any]
    #: 本 run **实际全窗读入**的行情/日历内容身份（C4-03；纳入 params 身份）
    market_identity: Mapping[str, Any]

    @property
    def signal_end(self) -> str:
        return self.calendar[-1]

    @property
    def calendar_last(self) -> str:
        return self.extended_days[-1]

    def split_of(self, day: str) -> int:
        """日期所属折号（runoff 日归末折；只作指标分组与折界登记）。"""
        for split, start, end, _model in self.t25.folds:
            if str(start) <= str(day) <= str(end):
                return int(split)
        return int(self.t25.folds[-1][0])

    def model_of(self, day: str) -> int:
        """当期模型版本（模型只在各折 ``test_start`` 切换；runoff 沿用末折）。"""
        for _split, start, end, model in self.t25.folds:
            if str(start) <= str(day) <= str(end):
                return int(model)
        return int(self.t25.folds[-1][3])

    def params_payload(self) -> Dict[str, Any]:
        return {
            "t25_input": self.t25.payload(),
            "arms": list(self.arms),
            "physical_capital": self.physical_capital,
            "sleeve_capital": dict(self.sleeve_capital),
            "lot_size": self.lot_size,
            "holding_days": HOLDING_DAYS,
            "cost_settings": dict(self.cost_settings),
            "calendar_days": len(self.calendar),
            "runoff_days": len(self.runoff_days),
            "extended_days": len(self.extended_days),
            "scenarios": list(SCENARIOS),
            "market_block_reasons": list(MARKET_BLOCK_REASONS),
            "funding_block_reasons": list(FUNDING_BLOCK_REASONS),
            "legacy_files": list(LEGACY_FILES),
            "frozen_intent_counts": dict(FROZEN_INTENT_COUNTS),
            "market_identity": dict(self.market_identity),
            "q3_ruling": "Q3=A：S5 实际费用按 a_i/A 比例分摊 + 终端袖一次闭合",
        }


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ParamsError(f"冻结输入缺件：{path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:  # pragma: no cover - 防御分支
        raise ParamsError(f"冻结输入不是合法 JSON：{path}（{error}）") from error


def _require_file(path: Path, label: str) -> str:
    if not path.exists():
        raise ParamsError(f"{label} 缺件：{path}")
    return sha256_file(path)


def _verify_against_upstream(
    root: Path,
    expected_sha: Mapping[str, str],
    plans_sha: Mapping[str, str],
    holdings_sha: Mapping[str, str],
    daily_sha: Mapping[str, str],
    calendar_path: Path,
) -> Dict[str, Any]:
    """逐件对照上游 ``manifest.outputs`` 登记的 SHA，不一致即拒绝（C4-03）。"""
    by_path = {str(item).lstrip("./"): sha for item, sha in expected_sha.items()}
    by_name = {Path(item).name: sha for item, sha in expected_sha.items()}
    result: Dict[str, Any] = {"checked": 0, "unverified": 0, "mismatch": [], "unverified_files": []}

    def _expected_for(rel_path: str) -> str:
        return by_path.get(rel_path) or by_name.get(Path(rel_path).name) or ""

    pending = [
        (f"targets/{label}_{arm}.parquet", actual)
        for label, mapping in (
            ("plans", plans_sha),
            ("holdings", holdings_sha),
            ("daily", daily_sha),
        )
        for arm, actual in mapping.items()
    ]
    pending.append(("intent_precheck.json", sha256_file(root / "intent_precheck.json")))
    pending.append(("manifest.json", sha256_file(root / "manifest.json")))
    pending.append(("targets/calendar.json", sha256_file(calendar_path)))
    for name, actual in pending:
        expected = _expected_for(name)
        if not expected:
            result["unverified"] += 1
            result["unverified_files"].append(name)
            continue
        result["checked"] += 1
        if expected != actual:
            result["mismatch"].append({"file": name, "expected": expected, "actual": actual})
    if result["mismatch"]:
        first = result["mismatch"][0]
        raise ParamsError(
            "T2.5 输入与上游登记的 SHA 不一致（未显式接受漂移即阻断发布）："
            f"{first['file']} expected={first['expected'][:12]} actual={first['actual'][:12]}"
            f"（共 {len(result['mismatch'])} 件）"
        )
    return result


def _expected_input_identity(expected: Mapping[str, Any]) -> Dict[str, Any]:
    """从上游 ``input_identity`` 提取三类输入的冻结身份（逐类给出对账基准）。"""
    daily = expected.get("clean_daily") or {}
    meta = expected.get("clean_meta") or {}
    rows: Dict[str, Any] = {}
    digest = str(daily.get("digest") or "")
    if digest:
        rows["clean_daily"] = {
            "mode": "digest",
            "expected": digest,
            "partitions": int(daily.get("partitions") or 0),
        }
    for path, label in (
        ("data/clean/trade_cal.parquet", "clean_trade_cal"),
        ("data/clean/stock_basic.parquet", "clean_stock_basic"),
    ):
        item = meta.get(path) or {}
        if item.get("sha256"):
            rows[label] = {
                "mode": "sha256",
                "expected": str(item["sha256"]),
                "frozen": str(item.get("frozen_sha256") or ""),
                "upstream_status": str(item.get("status") or ""),
            }
    return rows


def _verify_market_identity(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    accept_drift: bool,
) -> Dict[str, Any]:
    """把本 run 实际读入的行情/日历身份与上游冻结身份对账（C4-03）。

    未登记漂移且未显式 ``accept_drift`` => ``ParamsError``（不得发布）。
    上游已接受的漂移（``accepted_input_drift``）按“继承接受”登记，不静默放过。
    """
    expected_rows = _expected_input_identity(expected)
    if not expected_rows:
        raise ParamsError("上游 manifest 未登记 input_identity，无法对账行情/日历（C4-03）")
    actual = {
        "clean_daily": str(payload["clean_daily"]["digest"]),
        "clean_trade_cal": str(payload["clean_trade_cal"]["sha256"]),
        "clean_stock_basic": str(payload["clean_stock_basic"]["sha256"]),
    }
    paths = {
        "clean_trade_cal": "data/clean/trade_cal.parquet",
        "clean_stock_basic": "data/clean/stock_basic.parquet",
    }
    inherited = {
        str(row.get("path") or ""): row for row in expected.get("accepted_input_drift") or ()
    }
    checks: List[Dict[str, Any]] = []
    accepted: List[Dict[str, Any]] = []
    mismatch: List[Dict[str, Any]] = []
    for label in ("clean_daily", "clean_trade_cal", "clean_stock_basic"):
        frozen = expected_rows.get(label)
        if not frozen:
            checks.append({"input": label, "status": "unregistered"})
            continue
        row: Dict[str, Any] = {
            "input": label,
            "mode": frozen["mode"],
            "expected": frozen["expected"],
            "actual": actual[label],
            "match": actual[label] == frozen["expected"],
        }
        if row["match"]:
            row["status"] = "match"
        elif label in paths and inherited.get(paths[label], {}).get("current") == actual[label]:
            row["status"] = "accepted_drift_inherited"
            accepted.append(
                {"input": label, "path": paths[label], "actual": actual[label], "by": "T2.5"}
            )
        elif accept_drift:
            row["status"] = "accepted_drift_now"
            accepted.append(
                {
                    "input": label,
                    "path": paths.get(label, ""),
                    "actual": actual[label],
                    "by": "T4",
                }
            )
        else:
            row["status"] = "mismatch"
            mismatch.append(row)
        checks.append(row)
    expected_partitions = int(expected_rows["clean_daily"]["partitions"])
    actual_partitions = int(payload["clean_daily"]["partitions"])
    if expected_partitions and expected_partitions != actual_partitions:
        mismatch.append(
            {
                "input": "clean_daily.partitions",
                "expected": expected_partitions,
                "actual": actual_partitions,
            }
        )
    if mismatch:
        first = mismatch[0]
        raise ParamsError(
            "行情/日历内容漂移（未显式接受即阻断发布）："
            f"{first['input']} expected={str(first['expected'])[:12]} "
            f"actual={str(first['actual'])[:12]}（共 {len(mismatch)} 项）"
        )
    return {
        "schema": "p7_t4_market_identity_verification/v1",
        "ok": True,
        "checks": checks,
        "accepted_drift": accepted,
        "mismatch": [],
    }


def market_identity(
    repo_root: Path,
    days: Sequence[str],
    expected: Optional[Mapping[str, Any]] = None,
    *,
    accept_drift: bool = False,
) -> Dict[str, Any]:
    """本 run **实际读入**的完整窗口行情与日历内容身份（C4-03）：逐日 SHA + 聚合摘要。

    与 T2.5 冻结清单同算法（``days_digest`` = 逐行 ``rel:sha`` 换行拼接后 SHA256），
    因此可直接与上游 ``input_identity.clean_daily.digest`` 对账；漂移须显式接受。
    窗口必须是**宿主真实运行**的全窗（``params.extended_days``，含 runoff），
    只覆盖信号域会漏掉 runoff 的 20 个交易日。
    """
    import hashlib

    root = Path(repo_root)
    window = tuple(str(day) for day in days)
    if not window:
        raise ParamsError("行情身份窗口为空")
    if list(window) != sorted(window):
        raise ParamsError("行情身份窗口必须按交易日升序（禁止重排后摘要不一致）")
    # clean/daily 的分区文件名是 YYYY-MM-DD（与 YYYYMMDD 交易日键不同）
    rels = [
        "data/clean/daily/{}".format(f"{day[:4]}-{day[4:6]}-{day[6:8]}.parquet") for day in window
    ]
    missing = [rel for rel in rels if not (root / rel).exists()]
    if missing:
        # C4-03 硬门：本 run 声称读入的行情缺件即阻断，禁止静默少读
        raise ParamsError(f"行情缺件 {len(missing)} 件（示例 {missing[0]}）；禁止把缺件当 0 行读入")
    rows: Dict[str, str] = {rel: sha256_file(root / rel) for rel in rels}
    lines = [f"{rel}:{rows[rel]}" for rel in rels]
    daily_digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    calendar_rel = "data/clean/trade_cal.parquet"
    basic_rel = "data/clean/stock_basic.parquet"
    for rel in (calendar_rel, basic_rel):
        if not (root / rel).exists():
            raise ParamsError(f"行情身份缺件：{rel}（日历/证券主表也必须可核验）")
    payload: Dict[str, Any] = {
        "schema": MARKET_IDENTITY_SCHEMA,
        "scope": {
            "first_day": window[0],
            "last_day": window[-1],
            "days": len(window),
        },
        "clean_daily": {"partitions": len(rows), "digest": daily_digest, "missing": []},
        "clean_trade_cal": {"path": calendar_rel, "sha256": sha256_file(root / calendar_rel)},
        "clean_stock_basic": {"path": basic_rel, "sha256": sha256_file(root / basic_rel)},
        "algorithm": "sha256('\\n'.join('rel:sha256'))，与 T2.5 days_digest 同式",
    }
    if expected is not None:
        payload["verification"] = _verify_market_identity(
            payload, expected, accept_drift=accept_drift
        )
    return payload


def _upstream_output_shas(manifest: Mapping[str, Any]) -> Dict[str, str]:
    """从 T2.5 manifest 的 ``outputs`` 提取已登记的文件 SHA（相对 T2.5 根目录）。"""
    shas: Dict[str, str] = {}
    outputs = manifest.get("outputs")
    if isinstance(outputs, Mapping):
        # T2.5 的实际结构：``{相对路径: sha256}``
        for path, sha in outputs.items():
            if path and sha:
                shas[str(path).lstrip("./")] = str(sha)
        return shas
    for item in outputs or ():
        if not isinstance(item, Mapping):
            continue
        path = str(item.get("path") or "")
        sha = str(item.get("sha256") or item.get("sha") or "")
        if path and sha:
            shas[path.lstrip("./")] = sha
    return shas


def load_t25_input(repo_root: Path) -> T25Input:
    """读取并校验 T2.5 证据根（自然计数/状态/日历/逐件 SHA 一次落定）。"""
    root = Path(repo_root) / T25_ROOT_REL
    if not root.exists():
        raise ParamsError(f"T2.5 证据根不存在：{root}")
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    expected_sha = _upstream_output_shas(manifest)
    if bool(manifest.get("alpha_consumed")):
        raise ParamsError("T2.5 证据根标记 alpha_consumed=true，与 T4 零 α 消耗前提冲突")
    mechanical = manifest.get("mechanical_line") or {}
    status = str(mechanical.get("natural_intent_status") or "")
    if status != FROZEN_NATURAL_STATUS:
        raise ParamsError(f"T2.5 机械栏状态不是冻结值：{status!r}")
    natural = manifest.get("natural_observations") or {}
    counts = {str(key): int(value) for key, value in (natural.get("counts") or {}).items()}
    for scenario, expected in FROZEN_INTENT_COUNTS.items():
        actual = int(counts.get(scenario, -1))
        if actual != int(expected):
            raise ParamsError(f"T2.5 自然计数与冻结值不符：{scenario}={actual} != {expected}")

    precheck_path = root / "intent_precheck.json"
    precheck = _read_json(precheck_path)
    if precheck.get("quantity_basis") != "half_kelly_signal_day_weight_at_t1_open_nav":
        raise ParamsError("T2.5 quantity_basis 与冻结口径不符")

    calendar_path = root / "targets" / "calendar.json"
    calendar_doc = _read_json(calendar_path)
    plan = manifest.get("params") or {}
    signal_domain = tuple(str(item) for item in (plan.get("signal_domain") or ()))
    runoff_window = tuple(str(item) for item in (plan.get("runoff_window") or ()))
    if len(signal_domain) != 2 or len(runoff_window) != 2:
        raise ParamsError("T2.5 manifest 的 signal_domain/runoff_window 不完整")

    plans_sha: Dict[str, str] = {}
    holdings_sha: Dict[str, str] = {}
    daily_sha: Dict[str, str] = {}
    input_verification: Dict[str, Any] = {"checked": 0, "mismatch": []}
    for arm in ARMS:
        plans_sha[arm] = _require_file(
            root / "targets" / f"plans_{arm}.parquet", f"T2.5 plans_{arm}"
        )
        holdings_sha[arm] = _require_file(
            root / "targets" / f"holdings_{arm}.parquet", f"T2.5 holdings_{arm}"
        )
        daily_sha[arm] = _require_file(
            root / "targets" / f"daily_{arm}.parquet", f"T2.5 daily_{arm}"
        )

    # C4-03：逐件对照上游登记 SHA（核验而非只采集）
    input_verification = _verify_against_upstream(
        root, expected_sha, plans_sha, holdings_sha, daily_sha, calendar_path
    )

    code_state = manifest.get("code_state") or {}
    folds = _fold_spans(manifest)
    input_identity = manifest.get("input_identity") or {}
    if not input_identity:
        raise ParamsError("T2.5 manifest 未登记 input_identity；C4-03 无对账基准，禁止放行")
    upstream_daily = str((input_identity.get("clean_daily") or {}).get("digest") or "")
    if upstream_daily != FROZEN_MARKET_DAILY_DIGEST:
        raise ParamsError(
            "上游登记的 clean/daily 全窗冻结摘要与冻结基准不符（上游被改或口径变更）："
            f"{upstream_daily[:12]} != {FROZEN_MARKET_DAILY_DIGEST[:12]}"
        )
    return T25Input(
        root=root,
        manifest_sha256=sha256_file(manifest_path),
        precheck_sha256=sha256_file(precheck_path),
        input_verification=input_verification,
        input_identity=dict(input_identity),
        calendar_sha256=sha256_file(calendar_path),
        plans_sha256=plans_sha,
        holdings_sha256=holdings_sha,
        daily_sha256=daily_sha,
        source_digest=str(code_state.get("source_digest") or ""),
        natural_counts=counts,
        natural_status=status,
        signal_domain=(signal_domain[0], signal_domain[1]),
        runoff_window=(runoff_window[0], runoff_window[1]),
        calendar_days=tuple(c for c in (calendar_doc.get("v20") or {}).get("dates") or ()),
        folds=folds,
    )


def _fold_spans(manifest: Mapping[str, Any]) -> Tuple[Tuple[int, str, str, int], ...]:
    """从 T2.5 ``manifest.folds`` 提取逐折信号域与模型版本（只读派生，不另立口径）。"""
    spans: List[Tuple[int, str, str, int]] = []
    for item in manifest.get("folds") or ():
        span = item.get("signal_span") or item.get("test_window") or ()
        if len(span) != 2:
            raise ParamsError(f"T2.5 折记录缺少 signal_span/test_window：{item}")
        spans.append(
            (int(item["split_index"]), str(span[0]), str(span[1]), int(item["model_version"]))
        )
    if not spans:
        raise ParamsError("T2.5 证据根未登记逐折信号域（folds 为空）")
    spans.sort()
    return tuple(spans)


def load_calendar(repo_root: Path, t25: T25Input) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """扩展日历 = T2.5 信号域交易日 + 末折 runoff 交易日（T2.5 同源口径）。

    只用 ``clean/trade_cal``（T2.5 同一 loader），不引入第二套日历实现。
    """
    from scripts.v2_p7.t25.params import load_trade_days

    data_root = Path(repo_root) / "data"
    calendar = tuple(load_trade_days(data_root, t25.signal_domain[0], t25.signal_domain[1]))
    runoff = tuple(load_trade_days(data_root, t25.runoff_window[0], t25.runoff_window[1]))
    if not calendar:
        raise ParamsError("信号域交易日为空")
    if not runoff:
        raise ParamsError("runoff 交易日为空")
    overlap = sorted(set(calendar) & set(runoff))
    if overlap:
        raise ParamsError(f"runoff 与信号域重叠：{overlap[:3]}")
    return calendar, runoff


def assert_cost_settings(settings: Mapping[str, float], accepted: bool = False) -> None:
    """成本参数与冻结快照一致性（漂移须**显式接受**并留痕，不静默放行）。"""
    problems = [
        f"{key}={settings.get(key)!r} != {value!r}"
        for key, value in FROZEN_COST_SETTINGS.items()
        if abs(float(settings.get(key, float("nan"))) - float(value)) > 1e-12
    ]
    if problems and not accepted:
        raise ParamsError(f"成本参数漂移（未显式接受）：{'；'.join(problems)}")


def legacy_identity(repo_root: Path) -> Dict[str, Any]:
    """受审旧面身份：逐件 SHA + 最近提交 + 与旧证据登记值的交叉核对。

    ``order_chain.py`` / ``virtual_native_engine.py`` 在 P6.5 T5 证据里已有逐件 SHA，
    本函数逐项比对（``old_evidence_match``）；其余文件旧证据未登记逐件 SHA，
    只登记当前字节与"最近提交未变"，不伪造一致性。
    """
    root = Path(repo_root)
    rows: Dict[str, Any] = {}
    for rel in LEGACY_FILES:
        path = root / rel
        if not path.exists():
            raise ParamsError(f"受审旧面缺件：{rel}")
        rows[rel] = _file_identity(root, rel)
    for rel in _iter_t25_files(root):
        rows[rel] = _file_identity(root, rel)
    crosscheck = _crosscheck_old_evidence(root, rows)
    return {
        "files": rows,
        "file_count": len(rows),
        "old_evidence_crosscheck": crosscheck,
        "old_evidence_crosscheck_all_match": (
            all(item["match"] for item in crosscheck.values()) if crosscheck else None
        ),
        "note": (
            "旧证据仅对 order_chain.py / virtual_native_engine.py 登记过逐件源码 SHA；"
            "其余受审文件以「当前 SHA + 最近提交 + 工作区干净」三项留痕，"
            "不声称与未登记的旧值比对。"
        ),
    }


def _file_identity(root: Path, rel: str) -> Dict[str, Any]:
    """单文件身份：SHA + 字节 + 最近提交 + 工作区状态（``modified`` 才算改动）。"""
    path = root / rel
    status = _git_status(root, rel)
    return {
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
        "last_commit": _git_last_commit(root, rel),
        "git_status": status,
        "modified_in_worktree": None if status is None else status not in ("", "??"),
        "worktree_clean": None if status is None else status == "",
    }


def _iter_t25_files(root: Path) -> Sequence[str]:
    """``scripts/v2_p7/t25/**`` 的源码文件（排除 ``__pycache__`` 与漂移声明）。"""
    base = root / LEGACY_T25_GLOB
    if not base.exists():
        return ()
    return tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in base.rglob("*.py")
            if "__pycache__" not in path.parts
        )
    )


def _git_last_commit(root: Path, rel: str) -> Optional[str]:
    """该文件最近一次提交哈希（``git log -1 --format=%H``；不可用返回 None）。"""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", rel],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - 环境缺 git
        return None
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    return text or None


def _git_status(root: Path, rel: str) -> Optional[str]:
    """该文件相对 HEAD 的 porcelain 状态码（空串=干净，``??``=未跟踪）。"""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", rel],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - 环境缺 git
        return None
    if result.returncode != 0:
        return None
    line = result.stdout.strip()
    return "" if not line else line.split(maxsplit=1)[0]


def _crosscheck_old_evidence(root: Path, rows: Mapping[str, Any]) -> Dict[str, Any]:
    """与旧证据已登记的逐件源码 SHA 交叉核对（只核对存在登记的条目）。"""
    recorded: Dict[str, str] = {}
    for rel in LEGACY_EVIDENCE_MANIFESTS:
        path = root / rel
        if not path.exists():
            continue
        payload = _read_json(path)
        for key, value in (payload.get("code_surface") or {}).items():
            recorded.setdefault(str(key), str(value))
    out: Dict[str, Any] = {}
    for rel, frozen in sorted(recorded.items()):
        current = str((rows.get(rel) or {}).get("sha256") or "")
        if not current:
            continue
        out[rel] = {
            "old_evidence_sha256": frozen,
            "current_sha256": current,
            "match": current == frozen,
            "evidence": [item for item in LEGACY_EVIDENCE_MANIFESTS if _has_key(root / item, rel)],
        }
    return out


def _has_key(path: Path, rel: str) -> bool:
    if not path.exists():
        return False
    payload = _read_json(path)
    return rel in (payload.get("code_surface") or {})


def build_params(
    repo_root: Path = REPO_ROOT,
    accept_cost_drift: bool = False,
    accept_input_drift: bool = False,
) -> T4Params:
    """组装并校验全部 T4 冻结输入（任何不符即 fail-fast，零副作用）。

    ``accept_input_drift``：行情/日历内容相对上游冻结身份漂移时的**唯一**显式放行开关；
    放行记录写入 ``market_identity.verification.accepted_drift`` 并进入 ``params_digest``。
    """
    from src.lazybull.common.config import get_cost_settings

    root = Path(repo_root).resolve()
    data_root = root / "data"
    t25 = load_t25_input(root)
    calendar, runoff = load_calendar(root, t25)
    settings = get_cost_settings()
    assert_cost_settings(settings, accepted=accept_cost_drift)
    cost = {
        key: float(settings[key])
        for key in ("commission_rate", "min_commission", "stamp_tax", "slippage")
    }
    identity = market_identity(
        root,
        calendar + runoff,
        expected=t25.input_identity,
        accept_drift=accept_input_drift,
    )
    return T4Params(
        repo_root=root,
        data_root=data_root,
        t25=t25,
        calendar=calendar,
        runoff_days=runoff,
        extended_days=calendar + runoff,
        arms=ARMS,
        physical_capital=PHYSICAL_CAPITAL,
        sleeve_capital=dict(SLEEVE_CAPITAL),
        lot_size=LOT_SIZE,
        cost_settings=cost,
        legacy_sha=legacy_identity(root),
        market_identity=identity,
    )


def params_digest(params: T4Params) -> str:
    """参数快照的稳定摘要（续跑/重放身份核验）。"""
    import hashlib

    text = json.dumps(params.params_payload(), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
