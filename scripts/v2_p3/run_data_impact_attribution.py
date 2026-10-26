# -*- coding: utf-8 -*-
"""P3 T5 F11 数据影响归因交付物（P3 规划 §7.5 指定脚本；§13.3.7 证据清单）。

G-data 门 = 数据面切换影响：臂②（新实现 × cs_train 验证源）vs 臂③（新实现 × panel），
唯一变量 = 数据源。归因范围 = 折窗口日期（boundary.json 外层训练窗 ∪ test 窗）×
模型实际消费列（``v{N}_features.json`` 登记列）；词表 = F11 组定（L1/L2 + D-04/D-05/D-12，
唯一来源 ``v2/store/reconcile_rules.py`` + ``panel_reconcile._classify_attrib_column``，
一律 import 不复制）。

报告七节（§13.3.7 硬要求）：a 逐折 × 消费列输入差异计数 + 词表分类（未分类即拒绝项
FAIL；**完整覆盖前置**：预期日域 = trade_cal ``is_open=1`` ∩ boundary 窗口，
``days_compared == 预期日数`` 且非零才可闭合，R2-T5A-04；跨折单日结果缓存 =
命名空间（两臂快照摘要 + FORMAT_VERSION + 口径摘要）隔离 + 条目内容身份校验
（R2-T5A-R1-04），panel 月帧 LRU 复用（P3-N4）；**实读绑定核验**：实际读取文件
（cs_train 日分区台账 + panel receipt）⊆ 两臂快照登记集合且逐文件哈希一致、
快照文件 sha256 == arm_meta ``input_snapshot_digest``（R2-T5A-R1-03））；
b 预测影响（逐折读两臂 ``predictions.parquet``：键集对称差 + 交集键 ``%.17g`` 逐值
diff + 指纹/n_rows/score_days 元数据 + **行集文本指纹自洽/日集越窗/空产物拒绝**
（R2-T5A-R1-05）；``signal_topk.parquet`` 产出时登记 rows/days/empty_days 展示）；
c 成交 / d 持仓 / e 净值影响（两臂 runs 批键集对称差 + 交集数值差；文件缺失 ⇒ FAIL，
与「文件在但 0 行 = 合法空」分开登记，R2-T5A-04；**净值有限性前置** + **派生指标
一致性核对**（chain_nav 重算 vs chain_metrics.json，容差 1e-9，R2-T5A-R1-07））；
f 链式指标判定表（臂③ vs B0'，``t5_compare_gates.judge_gdata_table`` 复用，
**判定比较与归因比较是两份独立证据、禁止混用**）；g 身份附录（两臂
``source_digest`` 一致断言 + ``arm_meta.json`` 的 ``producer_identity.environment_identity``
环境身份 + B0' ``freeze_manifest.json`` 代码态/数据态 + 两臂 ``consumption_snapshot.json``
启动前置 + 本脚本身份，缺失/漂移 ⇒ FAIL 前置，R2-T5A-04/P3-2）。

f 节归因闭合并联接线（R2-T5A-05 重设计）：``judge_gdata_table`` 的
``maxdd_attribution_closed`` 与 ``cagr_attribution_closed`` 两个归因出口由**同一闭合源**
驱动 = **归因闭合 = a 节闭合 ∧ 输出级一致性**（``evaluate_output_consistency``：
规则① 零输入差异 ⇒ 零输出差异；规则② 逐折对应——输出劣化折须该折 a 节有已登记
消费列差异，§13.3.6 ΔMaxDD 逐折归因闭合按规则②落地），**不得直接 = a 节 closed**。

用法（只读消费臂产物与生产数据；exit 0/1 按 verdict）：

    python scripts/v2_p3/run_data_impact_attribution.py \
        --arm2-dir data/frozen_reference/v2_p3/<臂② run> \
        --arm3-dir data/frozen_reference/v2_p3/<臂③ run> \
        --b0p-dir data/frozen_reference/v2_p3/b0_prime
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from loguru import logger

# 独立运行时仓库根不入 sys.path，按薄入口惯例显式引导
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.v2_p3 import t5_attribution_io as io  # noqa: E402
from scripts.v2_p3 import t5_attribution_sections as sections  # noqa: E402
from scripts.v2_p3.t5_attribution_report import (  # noqa: E402
    render_markdown,
    write_report_json,
    write_report_md,
)
from scripts.v2_p3.t5_compare_gates import (  # noqa: E402
    judge_gdata_table,
    load_chain_metrics,
)

#: f 节归因闭合并联接线说明（判定表两个归因出口 = 归因闭合同一来源；R2-T5A-05 新口径）
_WIRING_NOTE = (
    "maxdd_attribution_closed 与 cagr_attribution_closed 并联同一输入 = 本报告归因闭合标记"
    "（= a 节闭合 ∧ 输出级一致性：规则① 零输入差异 ⇒ 零输出差异；规则② 逐折对应——"
    "任一折输出劣化须该折 a 节有已登记消费列差异（词表分类在册），§13.3.6 ΔMaxDD 逐折归因"
    "闭合按规则②落地）；判定表两个「归因出口」由同一闭合源驱动，不直接等 a 节 closed，"
    "不另行独立归因"
)

#: a 节数值与词表口径说明（落盘图例）
_A_LEGEND = (
    "数值口径 atol=1e-6、NaN==NaN（panel_reconcile._compare_day 复用）；词表分类 = "
    "panel_reconcile._classify_attrib_column（内部即 reconcile_rules.attribute_column "
    "F11 词表唯一来源：L1/L2 + D-04①~④/D-05/D-12）；未分类/越界 ⇒ 拒绝项 FAIL；"
    "样例键口径：cs_train_value=cs_train 侧值、panel_value=panel 侧值；"
    "完整覆盖前置（R2-T5A-04）：预期日域 = trade_cal is_open=1 ∩ boundary 窗口，"
    "days_compared == 预期日数且非零才可闭合，窗口内非开市日显式登记 excluded 计数"
)

#: 总判定规则文本（落盘 rule 字段）
_VERDICT_RULE = (
    "PASS 条件（全部满足）：① 前置错误为零（折集/边界/模型消费列/预测行产物/runs 批次/"
    "chain_metrics/arm_meta/freeze_manifest/consumption_snapshot 齐全；快照非空清单、"
    "摘要绑定（快照 sha256 == input_snapshot_digest）与实读绑定（实读 ⊆ 登记集合且逐文件"
    "哈希一致）通过；预测行证据自洽（行集文本指纹/行数/日集/非空）；净值有限性前置与"
    "派生指标一致性核对（容差 1e-9）通过；两臂消费列集与 boundary 窗口一致，"
    "batch_meta/fingerprint 非空且身份字段齐备）；② 归因闭合 = a 节闭合（差异列 100% F11 "
    "词表分类、无拒绝项、完整覆盖）∧ 输出级一致性（规则① 零输入差异 ⇒ 零输出差异；"
    "规则② 逐折对应：输出劣化折须该折 a 节有已登记消费列差异，R2-T5A-05）；③ 臂②③ "
    "source_digest 一致（§13.3.8 同代码态前提）；④ f 节 G-data 判定表 PASS（归因闭合标记"
    " = ② 的归因闭合）。b~e 节为数据面切换影响的登记证据，不单独设门（G-data 门 = f 节判定表）"
)


# ---------------------------------------------------------------------------
# 分节采集
# ---------------------------------------------------------------------------


def _open_stores(production_data_root: Path) -> tuple[Any, Any]:
    """打开生产数据根只读消费端（panel store + cs_train storage）。"""
    from src.lazybull.data.storage import Storage
    from src.lazybull.v2.store.data_store import PanelDataStore

    return PanelDataStore(production_data_root), Storage(root_path=str(production_data_root))


def _column_mismatch_message(fold: str, inputs: io.FoldInputs) -> str:
    only2 = sorted(set(inputs.columns_arm2) - set(inputs.columns_arm3))
    only3 = sorted(set(inputs.columns_arm3) - set(inputs.columns_arm2))
    if not only2 and not only3:
        detail = "列序不一致（集合相同，模型输入列序即语义，禁止放行）"
    else:
        detail = f"仅臂② {only2[:5]} / 仅臂③ {only3[:5]}"
    return f"折 {fold} 两臂消费列集不一致 ⇒ FAIL（{detail}）"


def _assemble_fold_inputs(
    arm2_dir: Path, arm3_dir: Path, fold_names: Sequence[str]
) -> tuple[dict[str, io.FoldInputs], list[str]]:
    """逐折装配归因输入：模型版本/消费列/窗口读取 + 两臂消费列集一致性校验。"""
    inputs_by_fold: dict[str, io.FoldInputs] = {}
    errors: list[str] = []
    for name in fold_names:
        try:
            inputs = io.load_fold_inputs(arm2_dir, arm3_dir, name)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"折 {name} 输入装配失败: {exc}")
            continue
        if inputs.columns_arm2 != inputs.columns_arm3:
            errors.append(_column_mismatch_message(name, inputs))
            continue
        inputs_by_fold[name] = inputs
    return inputs_by_fold, errors


def _attribute_one_fold(
    calendar: Any,
    store: Any,
    storage: Any,
    inputs: io.FoldInputs,
    cache: sections.DayCompareCache | None,
    month_cache: sections.PanelMonthCache | None,
    ledger: io.ReadLedger | None,
) -> dict[str, Any]:
    """a 节单折编排：预期日域/剔除计数 → 覆盖门驱动逐日比较（缓存/LRU 纯加速）。"""
    expected = io.expected_open_days(calendar, inputs.window)
    excluded = io.count_non_open_days(calendar, inputs.window)
    result = sections.attribute_fold_inputs(
        store,
        storage,
        inputs,
        expected_days=expected,
        excluded_non_open_count=excluded,
        cache=cache,
        month_cache=month_cache,
        ledger=ledger,
    )
    result["calendar_source"] = calendar.attrs.get("source_path")
    result["calendar_source_sha256"] = calendar.attrs.get("source_sha256")
    return result


def _build_cache_namespace(arm2_dir: Path, arm3_dir: Path) -> str:
    """缓存命名空间（R2-T5A-R1-04①）= 两臂 ``input_snapshot_digest`` +
    ``DayCompareCache.FORMAT_VERSION`` + 比较口径摘要（atol 等）的合成哈希前 16 位。

    数据态（快照）变化 ⇒ 命名空间变化 ⇒ 旧缓存目录自然隔离；规则/实现变更经
    FORMAT_VERSION 递增义务（写死在 DayCompareCache docstring）进入同一合成。
    """
    from src.lazybull.v2.store.reconcile_rules import COMPARE_ATOL

    digests: list[str] = []
    for arm_dir in (arm2_dir, arm3_dir):
        meta = io.load_arm_meta(arm_dir)
        digest = io.extract_input_snapshot_digest(meta) if meta is not None else None
        digests.append(digest or "<missing>")
    caliber = f"atol={COMPARE_ATOL};nan_eq=1;neg_zero_norm=1"
    text = "\n".join([sections.DayCompareCache.FORMAT_VERSION, *digests, caliber])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _collect_section_a(
    arm2_dir: Path,
    arm3_dir: Path,
    production_data_root: Path,
    fold_names: Sequence[str],
    *,
    cache: sections.DayCompareCache | None = None,
    month_cache: sections.PanelMonthCache | None = None,
    snapshot_base_dir: Path | None = None,
    read_ledger: io.ReadLedger | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """a 节：逐折装配输入（消费列一致性校验）→ 预期日域覆盖门 → 窗口逐日两源比较。

    完整覆盖前置（R2-T5A-04）：交易日历加载失败 ⇒ 覆盖不可核验 ⇒ 错误并跳过 a 节
    （闭合不可得）；逐折预期日域 = trade_cal ``is_open=1`` ∩ boundary 窗口。
    收尾做实读绑定核验（R2-T5A-R1-03②：实读 ⊆ 两臂快照登记集合且逐文件哈希一致；
    登记基目录 = 臂锚定生产根父级，经 ``snapshot_base_dir`` 传入）。
    """
    folds: list[dict[str, Any]] = []
    inputs_by_fold, errors = _assemble_fold_inputs(arm2_dir, arm3_dir, fold_names)
    if not inputs_by_fold:
        return folds, errors
    ledger = read_ledger if read_ledger is not None else io.ReadLedger()
    try:
        calendar = io.load_trade_calendar(production_data_root, ledger=ledger)
    except (FileNotFoundError, ValueError, OSError, KeyError) as exc:
        errors.append(f"交易日历加载失败（a 节完整覆盖前置不可核验 ⇒ FAIL）: {exc}")
        return folds, errors
    try:
        store, storage = _open_stores(production_data_root)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        errors.append(f"生产数据根打开失败（a 节无法执行）: {exc}")
        return folds, errors
    store.reset_read_receipts()  # 实读清单按本次归因会话起算（§13.3.5 P2 口径）
    for name, inputs in inputs_by_fold.items():
        try:
            folds.append(
                _attribute_one_fold(calendar, store, storage, inputs, cache, month_cache, ledger)
            )
        except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
            errors.append(f"折 {name} a 节归因失败: {exc}")
    try:
        actual_reads = _collect_actual_reads(store, ledger, production_data_root)
        errors.extend(
            _verify_read_ledger_binding(arm2_dir, arm3_dir, actual_reads, snapshot_base_dir)
        )
    except (OSError, ValueError) as exc:
        errors.append(f"实读台账合成失败（R2-T5A-R2-02）: {exc}")
    return folds, errors


def _validate_fingerprint_doc(fold: str, label: str, doc: Mapping[str, Any] | None) -> list[str]:
    """逐折 fingerprint.json 必需证据校验（R2-T5A-04/06）：空对象/缺 fingerprint 字段/
    缺 predictions_* 增键 ⇒ 错误清单（调用方据此 FAIL，禁止 None==None 误判一致）。"""
    if doc is None:
        return [f"折 {fold} 指纹文件缺失（{label}）"]
    if not doc:
        return [f"折 {fold} 指纹文件为空对象（{{}}，{label}）——产物不完整 ⇒ FAIL"]
    errors: list[str] = []
    if not doc.get("fingerprint"):
        errors.append(f"折 {fold} 指纹文件缺 fingerprint 字段（{label}）——产物不完整 ⇒ FAIL")
    missing = io.missing_predictions_keys(doc)
    if missing:
        errors.append(
            f"折 {fold} 指纹文件缺 predictions_* 增键 {missing}（{label}）"
            "——预测行必需证据不完整（R2-T5A-06）⇒ FAIL"
        )
    return errors


def _collect_section_b(
    arm2_dir: Path, arm3_dir: Path, fold_names: Sequence[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    """b 节：逐折 fingerprint 元数据对比 + 两臂 predictions.parquet 键集/逐值对比。

    必需证据门禁（R2-T5A-04/06 + R2-T5A-R1-05 自洽升级）：指纹文件缺失/空对象/
    缺 fingerprint 字段/缺 predictions_* 增键/预测行产物缺失或与登记身份漂移/
    行集文本指纹不自洽/日集越窗/空产物 ⇒ FAIL，不得跳过。``signal_topk.parquet``
    若产出（fingerprint 含 signal_topk_* 键）则登记 rows/days/empty_days 展示字段
    （判定归 gates，本节不设门）。
    """
    folds: list[dict[str, Any]] = []
    errors: list[str] = []
    for name in fold_names:
        doc2 = io.load_fingerprint_doc(arm2_dir, name)
        doc3 = io.load_fingerprint_doc(arm3_dir, name)
        doc_errors = _validate_fingerprint_doc(name, "臂②", doc2) + _validate_fingerprint_doc(
            name, "臂③", doc3
        )
        if doc_errors:
            errors.extend(doc_errors)
            continue
        assert doc2 is not None and doc3 is not None  # 校验门已排 None
        try:
            boundary = io.load_boundary_window(arm3_dir, name)
            pred2 = io.load_fold_predictions(arm2_dir, name, doc2, boundary=boundary)
            pred3 = io.load_fold_predictions(arm3_dir, name, doc3, boundary=boundary)
        except (FileNotFoundError, ValueError, OSError, KeyError) as exc:
            errors.append(f"折 {name} 预测行产物读取/核验失败（必需证据 ⇒ FAIL）: {exc}")
            continue
        entry = sections.compare_fingerprint_docs(name, doc2, doc3)
        entry["predictions"] = sections.compare_prediction_frames(pred2, pred3)
        entry["signal_topk_arm2"] = io.load_fold_signal_topk(arm2_dir, name, doc2)
        entry["signal_topk_arm3"] = io.load_fold_signal_topk(arm3_dir, name, doc3)
        folds.append(entry)
    return folds, errors


def _collect_fold_runs_artifacts(
    batch2: Path,
    batch3: Path,
    fold: str,
    result: dict[str, Any],
    errors: list[str],
) -> None:
    """单折 runs 产物对比：trades/daily/holdings 任一缺失 ⇒ 错误（R2-T5A-04：
    文件缺失 = FAIL，与「文件在但 0 行 = 合法无成交/无持仓」分开登记）。"""
    pairs = (
        ("trades", "c_trade_folds", sections.compare_trade_frames),
        ("daily", "e_daily_folds", sections.compare_daily_frames),
        ("holdings_snapshot", "d_holdings_folds", sections.compare_holdings_frames),
    )
    for artifact, target, compare in pairs:
        frame2 = io.load_fold_artifact(batch2, fold, artifact)
        frame3 = io.load_fold_artifact(batch3, fold, artifact)
        if frame2 is None or frame3 is None:
            errors.append(
                f"折 {fold} {artifact} 文件缺失（臂②={'有' if frame2 is not None else '无'} / "
                f"臂③={'有' if frame3 is not None else '无'}）——必需输出证据缺失 ⇒ FAIL"
                "（文件在但 0 行属合法空，另行登记，不在此项）"
            )
            continue
        try:
            entry = compare(frame2, frame3)
        except (ValueError, KeyError) as exc:
            errors.append(f"折 {fold} {artifact} 比较失败: {exc}")
            continue
        # 合法空登记（文件在但 0 行）：与文件缺失分列入册
        result[target].append(
            {
                "fold": fold,
                "empty_arm2": bool(len(frame2) == 0),
                "empty_arm3": bool(len(frame3) == 0),
                **entry,
            }
        )


def _validate_batch_meta(batch_dir: Path, meta: Mapping[str, Any], label: str) -> list[str]:
    """batch_meta.json 必需身份校验（R2-T5A-04）：空对象/缺 data_state_id ⇒ 错误清单。"""
    if not meta:
        return [f"runs 批次 batch_meta.json 为空对象（{{}}，{label}）: {batch_dir} ⇒ FAIL"]
    data_state = meta.get("data_state")
    data_state_id = data_state.get("data_state_id") if isinstance(data_state, Mapping) else None
    if not data_state_id:
        return [
            f"runs 批次 batch_meta.json 缺 data_state.data_state_id（{label}）: "
            f"{batch_dir} ⇒ FAIL"
        ]
    return []


def _collect_runs_sections(
    arm2_dir: Path, arm3_dir: Path, fold_names: Sequence[str]
) -> tuple[dict[str, Any], list[str]]:
    """c/d/e 节：两臂 runs 批成交/持仓/净值对比 + batch_meta（数据态附录输入）。"""
    result: dict[str, Any] = {
        "c_trade_folds": [],
        "d_holdings_folds": [],
        "e_chain": None,
        "e_daily_folds": [],
        "batch2_dir": None,
        "batch3_dir": None,
        "batch2_meta": None,
        "batch3_meta": None,
    }
    errors: list[str] = []
    try:
        batch2 = io.resolve_runs_batch_dir(arm2_dir)
        batch3 = io.resolve_runs_batch_dir(arm3_dir)
    except (FileNotFoundError, ValueError) as exc:
        errors.append(f"runs 批次目录解析失败: {exc}")
        return result, errors
    result["batch2_dir"] = str(batch2)
    result["batch3_dir"] = str(batch3)
    for label, key, batch in (("臂②", "batch2_meta", batch2), ("臂③", "batch3_meta", batch3)):
        try:
            meta = io.load_batch_meta(batch)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"runs 批次 batch_meta 读取失败（{label}）: {exc}")
            continue
        result[key] = meta
        errors.extend(_validate_batch_meta(batch, meta, label))
    try:
        nav2 = io.load_chain_nav(batch2)
        nav3 = io.load_chain_nav(batch3)
        result["e_chain"] = sections.compare_chain_navs(nav2, nav3)
    except (FileNotFoundError, ValueError) as exc:
        errors.append(f"链式净值比较失败: {exc}")
    else:
        errors.extend(_check_derived_chain_metrics(arm2_dir, nav2, "臂②"))
        errors.extend(_check_derived_chain_metrics(arm3_dir, nav3, "臂③"))
    for name in fold_names:
        _collect_fold_runs_artifacts(batch2, batch3, name, result, errors)
    return result, errors


def _check_derived_chain_metrics(arm_dir: Path, nav: Any, label: str) -> list[str]:
    """派生指标一致性核对（R2-T5A-R1-07②）：runs 批 chain_nav 经
    ``v2/common/chain_metrics.calculate_chain_metrics``（唯一实现，import 不复制）
    重算链式指标，与臂 ``chain_metrics.json`` 登记值逐键比对（容差 1e-9），
    不一致 ⇒ 错误清单（调用方记 FAIL）。
    """
    from src.lazybull.v2.common.chain_metrics import calculate_chain_metrics

    try:
        registered = load_chain_metrics(arm_dir)
    except (FileNotFoundError, ValueError) as exc:
        return [f"{label} chain_metrics 读取失败（派生指标核对无法执行）: {exc}"]
    recomputed = calculate_chain_metrics(nav)
    errors: list[str] = []
    for key in ("total_return", "cagr", "max_drawdown", "volatility", "sharpe", "trading_days"):
        want = recomputed.get(key)
        got = registered.get(key)
        if want is None and got is None:
            continue
        if got is None or want is None:
            errors.append(
                f"{label} 派生指标核对失败：{key} 登记 {got} vs 重算 {want}（R2-T5A-R1-07②）"
            )
            continue
        if abs(float(got) - float(want)) > 1e-9:
            errors.append(
                f"{label} 派生指标核对失败：{key} 登记 {got} vs 重算 {want} "
                f"（|Δ| > 1e-9，R2-T5A-R1-07②）"
            )
    return errors


# ---------------------------------------------------------------------------
# f/g 节
# ---------------------------------------------------------------------------


def build_judgment_section(
    arm3_dir: Path,
    b0p_dir: Path,
    attribution_closed: bool,
    errors: list[str],
    *,
    closure_detail: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """f 节：读臂③与 B0' chain_metrics.json，调 ``judge_gdata_table`` 生成判定节。

    归因闭合并联接线（R2-T5A-05 重设计口径）：``maxdd_attribution_closed`` 与
    ``cagr_attribution_closed`` 两个出口共用 ``attribution_closed`` 单一来源，且该输入
    **不得直接 = a 节 closed**——新口径为 ``归因闭合 = a 节闭合 ∧ 输出级一致性``
    （规则① 零输入差异 ⇒ 零输出差异；规则② 逐折对应：输出劣化折须该折 a 节有
    已登记消费列差异，§13.3.6 ΔMaxDD 逐折归因闭合按规则②落地；合成方 =
    ``sections.evaluate_output_consistency``，明细经 ``closure_detail`` 随节落盘）。
    失败入 ``errors`` 并返回 None（报告节缺失，总判定 FAIL）。
    """
    try:
        arm3_chain = load_chain_metrics(arm3_dir)
        b0p_chain = load_chain_metrics(b0p_dir)
    except (FileNotFoundError, ValueError) as exc:
        errors.append(f"chain_metrics 读取失败: {exc}")
        return None
    try:
        table = judge_gdata_table(
            arm3_chain,
            b0p_chain,
            maxdd_attribution_closed=attribution_closed,
            cagr_attribution_closed=attribution_closed,
        )
    except ValueError as exc:
        errors.append(f"G-data 判定表输入契约违反: {exc}")
        return None
    return {
        "attribution_closed": attribution_closed,
        "attribution_closure_detail": closure_detail,
        "wiring_note": _WIRING_NOTE,
        **table,
    }


def _resolve_consumption_anchor(arm_dir: Path) -> Path | None:
    """消费快照登记锚点 = 臂 ``before_guard.json`` 记录的生产数据根（R2-T5A-R1-03：
    快照显示路径的采集基目录只能来自臂运行时的真实生产根——以守卫快照的
    ``base``（``<生产根>/models/stock_selection``）反解，禁止用归因传入根顶替）。
    """
    doc = io.load_json_object(Path(arm_dir) / "before_guard.json")
    if doc is None:
        return None
    base = doc.get("base")
    if not base:
        return None
    return Path(str(base)).resolve().parent.parent


def _consumption_anchor_base(arm2_dir: Path, arm3_dir: Path) -> tuple[Path | None, list[str]]:
    """快照登记基目录 = 两臂锚定生产根的父级（t05b_baselines 采集口径）；两臂须同根。"""
    errors: list[str] = []
    anchors: dict[str, Path] = {}
    for label, arm_dir in (("arm2", arm2_dir), ("arm3", arm3_dir)):
        anchor = _resolve_consumption_anchor(arm_dir)
        if anchor is None:
            errors.append(
                f"{label} 缺 before_guard.json 消费锚点（R2-T5A-R1-03 实读绑定前置输入）: "
                f"{arm_dir}"
            )
        else:
            anchors[label] = anchor
    if "arm2" not in anchors or "arm3" not in anchors:
        return None, errors
    if anchors["arm2"] != anchors["arm3"]:
        errors.append(
            f"两臂消费锚点根不一致（R2-T5A-R1-03）: {anchors['arm2']} vs {anchors['arm3']}"
        )
        return None, errors
    return anchors["arm2"].parent, errors


def _verify_consumption_snapshots(
    arm2_dir: Path, arm3_dir: Path, base_dir: Path | None
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """两臂消费快照启动前置（R2-T5A-04 + R2-T5A-R1-03③ 摘要绑定）：

    复用 ``t05b_isolated_root`` collect/verify 语义——快照文件缺失/非清单形态/
    **空清单** ⇒ FAIL 前置；按落盘清单重算，任一漂移 ⇒ FAIL；并核验
    ``sha256(快照文件) == arm_meta.producer_identity.input_snapshot_digest``（摘要不等
    ⇒ FAIL）。复核基目录 = 臂 ``before_guard.json`` 锚定的生产根父级（R1-03：
    禁止按归因传入根解析）；锚点缺失 ⇒ 跳过解析依赖项（前置错误已登记）。
    """
    from scripts.v2_p3.t05b_isolated_root import verify_consumption_snapshot

    records: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for label, arm_dir in (("arm2", arm2_dir), ("arm3", arm3_dir)):
        snapshot_path = Path(arm_dir) / "consumption_snapshot.json"
        entry: dict[str, Any] = {"verified": False, "drift_count": None, "drift_samples": []}
        records[label] = entry
        if not snapshot_path.is_file():
            errors.append(f"缺消费快照 consumption_snapshot.json（FAIL 前置）: {arm_dir}")
            continue
        try:
            doc = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"消费快照读取失败（FAIL 前置）: {snapshot_path}（{exc}）")
            continue
        if not isinstance(doc, list):
            errors.append(f"消费快照顶层须为清单形态（FAIL 前置）: {snapshot_path}")
            continue
        if not doc:
            errors.append(
                f"消费快照为空清单（FAIL 前置：无登记集合可供实读绑定，R2-T5A-R1-03）: "
                f"{snapshot_path}"
            )
            continue
        errors.extend(_check_snapshot_digest_binding(arm_dir, snapshot_path, entry))
        if base_dir is None:
            continue  # 锚点缺失（前置错误已登记）：跳过路径解析依赖的漂移复核
        drift = verify_consumption_snapshot(snapshot_path, base_dir=base_dir)
        entry["verified"] = not drift
        entry["drift_count"] = len(drift)
        entry["drift_samples"] = drift[:5]
        if drift:
            errors.append(
                f"{label} 消费快照当前态复核漂移 {len(drift)} 项（FAIL；样例 {drift[:5]}）"
            )
    return records, errors


def _check_snapshot_digest_binding(
    arm_dir: Path, snapshot_path: Path, entry: dict[str, Any]
) -> list[str]:
    """摘要绑定（R2-T5A-R1-03③）：快照文件 sha256 == arm_meta input_snapshot_digest。"""
    meta = io.load_arm_meta(arm_dir)
    digest = io.extract_input_snapshot_digest(meta) if meta is not None else None
    actual = io.file_sha256(snapshot_path)
    entry["input_snapshot_digest"] = digest
    entry["snapshot_file_sha256"] = actual
    if digest is None:
        return [
            f"arm_meta.json 缺 producer_identity.input_snapshot_digest（FAIL 前置，"
            f"R2-T5A-R1-03③ 摘要绑定输入）: {arm_dir}"
        ]
    if actual != digest:
        return [
            f"消费快照文件 sha256 与 arm_meta input_snapshot_digest 不等（摘要绑定失败，"
            f"R2-T5A-R1-03③）: {snapshot_path} 实得 {str(actual)[:16]}… vs 登记 {digest[:16]}…"
        ]
    return []


def _collect_actual_reads(
    store: Any, ledger: io.ReadLedger, production_data_root: Path
) -> dict[str, str]:
    """实读清单合成（规范化绝对路径 → 读取时点 sha256）：cs_train 台账 + panel receipt。"""
    for record in store.consumed_files:
        raw = Path(str(record["path"]))
        path = raw if raw.is_absolute() else Path(production_data_root) / raw
        ledger.record(path, digest=str(record["sha256"]))
    return ledger.records


def _verify_read_ledger_binding(
    arm2_dir: Path,
    arm3_dir: Path,
    actual_reads: Mapping[str, str],
    base_dir: Path | None,
) -> list[str]:
    """实读绑定核验（R2-T5A-R1-03②）：实读集合 ⊆ 两臂快照登记集合（并集）且
    逐文件当前内容哈希 == 快照登记哈希。登记路径按**臂锚定基目录**（
    ``before_guard.json`` 反解的生产根父级）解析、双方 ``os.path.normcase(resolve())``
    同基规范化后按绝对路径比对——归因实读根被 ``other_data_root`` 顶替时实读路径
    全部越界 ⇒ FAIL（路径不绑定）；锚点缺失 ⇒ 跳过（前置错误已登记）。
    """
    if base_dir is None:
        return []
    registered: dict[str, str] = {}
    for arm_dir in (arm2_dir, arm3_dir):
        snapshot_path = Path(arm_dir) / "consumption_snapshot.json"
        try:
            doc = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(doc, list):
            continue
        for record in doc:
            raw = Path(str(record["path"]))
            path = raw if raw.is_absolute() else base_dir / raw
            registered[io.ReadLedger.normalize(path)] = str(record.get("sha256"))
    errors: list[str] = []
    for norm_path, sha in sorted(actual_reads.items()):
        if norm_path not in registered:
            errors.append(f"实读越界（不在两臂消费快照登记集合内，R2-T5A-R1-03②）: {norm_path}")
        elif registered[norm_path] != sha:
            errors.append(f"实读内容漂移（当前哈希 != 快照登记，R2-T5A-R1-03②）: {norm_path}")
    return errors


def _nested_mapping(doc: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    nested = doc.get(key)
    return nested if isinstance(nested, Mapping) else {}


def _arm_identity(
    arm_meta: Mapping[str, Any],
    digest: str | None,
    arm_dir: Path,
    batch_meta: Mapping[str, Any] | None,
    consumption_check: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """单侧臂身份附录条目（代码态 + 环境 + 数据态 + 消费快照指针与当前态复核）。

    P3-2 键位：环境身份 = ``arm_meta.producer_identity.environment_identity``
    （t5_arm_runner 落盘口径；兜底 workspace_identity.environment / 顶层
    environment）；git_commit/git_dirty = runs ``batch_meta.code_state``（兜底
    workspace_identity/顶层 git 键）；数据态 = ``batch_meta.data_state``。
    """
    producer = _nested_mapping(arm_meta, "producer_identity")
    workspace = _nested_mapping(arm_meta, "workspace_identity")
    code_state = _nested_mapping(batch_meta or {}, "code_state")
    data_state = _nested_mapping(batch_meta or {}, "data_state")
    snapshot_path = Path(arm_dir) / "consumption_snapshot.json"
    return {
        "source_digest": digest,
        "git_commit": code_state.get(
            "git_commit", workspace.get("git_commit", arm_meta.get("git_commit"))
        ),
        "git_dirty": code_state.get(
            "git_dirty", workspace.get("git_dirty", arm_meta.get("git_dirty"))
        ),
        "environment": producer.get(
            "environment_identity", workspace.get("environment", arm_meta.get("environment"))
        ),
        "data_state_id": data_state.get("data_state_id"),
        "consumption_snapshot": str(snapshot_path),
        "consumption_snapshot_sha256": io.file_sha256(snapshot_path),
        "consumption_snapshot_verified": (
            consumption_check.get("verified") if consumption_check else None
        ),
        "consumption_drift_count": (
            consumption_check.get("drift_count") if consumption_check else None
        ),
        "consumption_drift_samples": (
            consumption_check.get("drift_samples") if consumption_check else []
        ),
        "consumption_post_verified": (
            consumption_check.get("post_verified") if consumption_check else None
        ),
        "consumption_post_drift_count": (
            consumption_check.get("post_drift_count") if consumption_check else None
        ),
        "consumption_post_drift_samples": (
            consumption_check.get("post_drift_samples") if consumption_check else []
        ),
    }


def _b0p_identity(b0p_dir: Path, errors: list[str]) -> dict[str, Any] | None:
    """B0' 身份附录条目：从 ``freeze_manifest.json`` 读代码态/数据态（P3-2；
    现恒 None 的修复——B0' 无 arm_meta.json，manifest 是唯一身份来源）。"""
    manifest = io.load_freeze_manifest(b0p_dir)
    if manifest is None:
        errors.append(f"缺 freeze_manifest.json（FAIL 前置，g 节 B0' 身份唯一来源）: {b0p_dir}")
        return None
    if not manifest:
        errors.append(f"freeze_manifest.json 为空对象（{{}}，FAIL 前置）: {b0p_dir}")
        return None
    producer = _nested_mapping(manifest, "producer_identity")
    code_state = _nested_mapping(manifest, "code_state")
    digest = producer.get("source_digest")
    if not digest:
        errors.append(
            f"freeze_manifest.json 缺 producer_identity.source_digest（FAIL 前置）: {b0p_dir}"
        )
    return {
        "source_digest": str(digest) if digest else None,
        "git_commit": code_state.get("git_commit"),
        "git_dirty": code_state.get("git_dirty"),
        "environment": producer.get("environment_identity"),
        "data_state_id": manifest.get("data_state_id"),
        "frozen_at": manifest.get("frozen_at"),
        "source_run_name": manifest.get("source_run_name"),
    }


def _missing_identity_fields(identity: Mapping[str, Any]) -> list[str]:
    return [
        key
        for key in ("source_digest", "git_commit", "git_dirty", "environment", "data_state_id")
        if identity.get(key) is None
    ]


def _build_identity_appendix(
    arm2_dir: Path,
    arm3_dir: Path,
    b0p_dir: Path,
    batch2_meta: Mapping[str, Any] | None,
    batch3_meta: Mapping[str, Any] | None,
    observer_identity: Mapping[str, Any] | None,
    *,
    consumption_checks: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """g 节：两臂 ``source_digest`` 一致断言（§13.3.8）+ 环境/数据态 + 本脚本身份。

    臂③运行前须验证当前 source_digest 与臂②保存摘要一致——本报告从两臂
    arm_meta.json 重读两侧保存摘要并再断言；文件或字段缺失 ⇒ FAIL 前置。
    B0' 身份从 ``freeze_manifest.json`` 读（P3-2）；两臂环境身份从
    ``producer_identity.environment_identity`` 读；必需身份字段（source_digest /
    git_commit / git_dirty / environment / data_state_id）缺失 ⇒ FAIL 前置。
    """
    errors: list[str] = []
    checks = consumption_checks or {}
    arms: dict[str, dict[str, Any] | None] = {}
    for key, arm_dir, batch_meta in (
        ("arm2", arm2_dir, batch2_meta),
        ("arm3", arm3_dir, batch3_meta),
    ):
        meta = io.load_arm_meta(arm_dir)
        if meta is None:
            errors.append(f"缺 arm_meta.json（FAIL 前置，§13.3.8 断言输入）: {arm_dir}")
            arms[key] = None
            continue
        digest = io.extract_source_digest(meta)
        if digest is None:
            errors.append(f"arm_meta.json 缺 source_digest 字段（FAIL 前置）: {arm_dir}")
        identity = _arm_identity(meta, digest, arm_dir, batch_meta, checks.get(key))
        missing = _missing_identity_fields(identity)
        if missing:
            errors.append(f"{key} 身份字段缺失 {missing}（FAIL 前置，P3-2）: {arm_dir}")
        arms[key] = identity
    digest2 = (arms["arm2"] or {}).get("source_digest")
    digest3 = (arms["arm3"] or {}).get("source_digest")
    digest_match = digest2 is not None and digest2 == digest3
    if digest2 is not None and digest3 is not None and not digest_match:
        errors.append(
            f"臂② 与 臂③ source_digest 不一致（{digest2[:16]}… vs {digest3[:16]}…），"
            "G-data 同代码态前提破坏（§13.3.8）"
        )
    b0p_identity = _b0p_identity(b0p_dir, errors)
    if b0p_identity is not None:
        missing_b0p = _missing_identity_fields(b0p_identity)
        if missing_b0p:
            errors.append(f"b0p 身份字段缺失 {missing_b0p}（FAIL 前置，P3-2）: {b0p_dir}")
    if observer_identity is None:
        errors.append("缺本脚本观察身份（workspace_identity）输入（FAIL 前置）")
        script_identity: dict[str, Any] = {"source_digest": None}
    else:
        script_identity = {
            key: observer_identity.get(key)
            for key in ("source_digest", "git_commit", "git_dirty", "environment")
        }
    appendix = {
        "arm2": arms["arm2"],
        "arm3": arms["arm3"],
        "b0p": b0p_identity,
        "script": script_identity,
        "source_digest_match": digest_match,
        "assertion_note": (
            "臂③运行前 source_digest == 臂②保存摘要（§13.3.8 同代码态前提）；"
            "本报告从两臂 arm_meta.json 重读保存摘要再断言，缺失字段即 FAIL 前置；"
            "B0' 代码态/数据态读 freeze_manifest.json，两臂环境身份读 arm_meta.json 的 "
            "producer_identity.environment_identity（P3-2）；两臂 consumption_snapshot.json "
            "当前态复核缺失/漂移即 FAIL（R2-T5A-04）"
        ),
    }
    return appendix, errors


# ---------------------------------------------------------------------------
# 汇总与编排
# ---------------------------------------------------------------------------


def _summarize_section_a(
    folds: Sequence[Mapping[str, Any]], cache_stats: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """a 节汇总：闭合标记 + 差异列计数 + 拒绝项 + 结构破坏计数 + 缓存命中率登记。"""
    rejected: dict[str, Any] = {}
    diff_column_count = 0
    structural_break_count = 0
    registered_structural_count = 0
    for fold in folds:
        diff_column_count += len(fold["columns"])
        for col, entry in fold["columns"].items():
            if entry["boundary_violation"]:
                rejected[f"{fold['fold']}:{col}"] = entry["attribution"]
        structural_break_count += (
            len(fold["row_diff_days"])
            + len(fold["cols_only_cs_train"])
            + len(fold["cols_only_panel"])
            + len(fold["missing_cs_train_days"])
            + len(fold["missing_panel_days"])
            + len(fold.get("duplicate_key_days", []))
        )
        registered_structural_count += len(fold.get("registered_cols_only_panel", {}))
    return {
        "closed": bool(folds) and all(fold["closed"] for fold in folds),
        "diff_column_count": diff_column_count,
        "rejected_columns": rejected,
        "structural_break_count": structural_break_count,
        "registered_structural_count": registered_structural_count,
        "legend": _A_LEGEND,
        "cache_stats": dict(cache_stats) if cache_stats is not None else None,
        "folds": list(folds),
    }


def build_report(
    arm2_dir: Path,
    arm3_dir: Path,
    b0p_dir: Path,
    production_data_root: Path,
    *,
    run_id: str | None = None,
    observer_identity: Mapping[str, Any] | None = None,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """归因报告编排：前置核验 → a~e 节采集 → 输出级一致性 → f 判定表（闭合接线）→ g 身份附录。

    ``observer_identity`` = 本脚本执行环境的 ``workspace_identity`` 快照（CLI 采集；
    缺省/None ⇒ FAIL 前置）。``cache_dir`` = a 节跨折单日结果缓存目录（P2-7；
    None ⇒ 仅进程内缓存不落盘）。全部输入只读；报告本体不落盘（落盘在 main）。
    """
    arm2_dir, arm3_dir = Path(arm2_dir), Path(arm3_dir)
    b0p_dir, production_data_root = Path(b0p_dir), Path(production_data_root)
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    errors: list[str] = []
    fold_names, fold_errors = io.discover_fold_names(arm2_dir, arm3_dir)
    errors.extend(fold_errors)
    if not fold_names:
        errors.append(f"两臂折目录均为空（folds/splitNN 未发现）: {arm2_dir} / {arm3_dir}")

    anchor_base, consumption_checks = _run_consumption_prechecks(arm2_dir, arm3_dir, errors)
    read_ledger = io.ReadLedger()
    cache, month_cache = _prepare_a_caches(arm2_dir, arm3_dir, cache_dir)
    a_folds, a_errors = _collect_section_a(
        arm2_dir,
        arm3_dir,
        production_data_root,
        fold_names,
        cache=cache,
        month_cache=month_cache,
        snapshot_base_dir=anchor_base,
        read_ledger=read_ledger,
    )
    errors.extend(a_errors)
    b_folds, b_errors = _collect_section_b(arm2_dir, arm3_dir, fold_names)
    errors.extend(b_errors)
    runs, runs_errors = _collect_runs_sections(arm2_dir, arm3_dir, fold_names)
    errors.extend(runs_errors)

    section_a = _summarize_section_a(a_folds, _a_cache_stats(cache, month_cache))
    consistency, attribution_closed = _evaluate_attribution_closure(
        section_a, a_folds, b_folds, runs, errors
    )
    section_f = build_judgment_section(
        arm3_dir, b0p_dir, attribution_closed, errors, closure_detail=consistency
    )
    section_g, g_errors = _build_identity_appendix(
        arm2_dir,
        arm3_dir,
        b0p_dir,
        runs["batch2_meta"],
        runs["batch3_meta"],
        observer_identity,
        consumption_checks=consumption_checks,
    )
    errors.extend(g_errors)
    _run_consumption_postchecks(arm2_dir, arm3_dir, read_ledger, consumption_checks, errors)
    for arm_key in ("arm2", "arm3"):
        identity = section_g.get(arm_key)
        if identity is not None:
            identity.update(
                {
                    f"consumption_{key}": consumption_checks[arm_key].get(key)
                    for key in ("post_verified", "post_drift_count", "post_drift_samples")
                }
            )
    return _assemble_report(
        arm2_dir,
        arm3_dir,
        b0p_dir,
        production_data_root,
        run_id=run_id,
        errors=errors,
        section_a=section_a,
        b_folds=b_folds,
        runs=runs,
        section_f=section_f,
        section_g=section_g,
        attribution_closed=attribution_closed,
    )


def _run_consumption_prechecks(
    arm2_dir: Path, arm3_dir: Path, errors: list[str]
) -> tuple[Path | None, dict[str, dict[str, Any]]]:
    """消费态核验启动前置（R2-T5A-04/R1-03）：锚点解析（before_guard.json 反解生产根）
    → 两臂 consumption_snapshot.json 缺失/空清单/摘要绑定/当前态复核。"""
    anchor_base, anchor_errors = _consumption_anchor_base(arm2_dir, arm3_dir)
    errors.extend(anchor_errors)
    checks, snapshot_errors = _verify_consumption_snapshots(arm2_dir, arm3_dir, anchor_base)
    errors.extend(snapshot_errors)
    return anchor_base, checks


def _prepare_a_caches(
    arm2_dir: Path, arm3_dir: Path, cache_dir: Path | None
) -> tuple[sections.DayCompareCache, sections.PanelMonthCache]:
    """a 节加速设施装配：命名空间单日比较缓存（R1-04）+ panel 月帧 LRU（P3-N4）。"""
    cache = sections.DayCompareCache(
        cache_dir, namespace=_build_cache_namespace(arm2_dir, arm3_dir)
    )
    return cache, sections.PanelMonthCache()


def _run_consumption_postchecks(
    arm2_dir: Path,
    arm3_dir: Path,
    ledger: io.ReadLedger,
    checks: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    """全部分节结束后复核唯一实读集合及快照摘要，缓存命中也不能洗白运行中漂移。

    在 a~g 全部分节完成后，只重哈希实际读取文件（含日历及 panel receipt），
    不重复扫描快照保守超集。
    身份附录分别保存启动与结束核验结果，禁止用启动 verified 冒充结束状态。
    """
    try:
        read_errors = ledger.verify_current()
    except OSError as exc:
        read_errors = [f"运行后实读身份复核失败（R2-T5A-R2-02）: {exc}"]
    errors.extend(read_errors)
    for label, arm_dir in (("arm2", arm2_dir), ("arm3", arm3_dir)):
        entry = checks.setdefault(label, {})
        try:
            snapshot_errors = _check_snapshot_digest_binding(
                arm_dir, arm_dir / "consumption_snapshot.json", {}
            )
        except (OSError, ValueError) as exc:
            snapshot_errors = [f"运行后消费快照摘要复核失败（{label}）: {exc}"]
        errors.extend(snapshot_errors)
        post_errors = read_errors + snapshot_errors
        entry["post_verified"] = bool(ledger.records) and not post_errors
        entry["post_drift_count"] = len(post_errors)
        entry["post_drift_samples"] = post_errors[:5]


def _a_cache_stats(
    cache: sections.DayCompareCache, month_cache: sections.PanelMonthCache
) -> dict[str, Any]:
    """a 节缓存命中率登记（CLI/报告口径：单日比较 + panel 月帧两组）。"""
    return {"day_compare": cache.stats(), "panel_month": month_cache.stats()}


def _evaluate_attribution_closure(
    section_a: Mapping[str, Any],
    a_folds: Sequence[Mapping[str, Any]],
    b_folds: Sequence[Mapping[str, Any]],
    runs: Mapping[str, Any],
    errors: list[str],
) -> tuple[dict[str, Any], bool]:
    """归因闭合合成（R2-T5A-05）= a 节闭合 ∧ 输出级一致性；不一致逐条入 errors。"""
    consistency = sections.evaluate_output_consistency(
        a_folds,
        b_folds,
        runs["c_trade_folds"],
        runs["d_holdings_folds"],
        runs["e_chain"],
        runs["e_daily_folds"],
    )
    attribution_closed = bool(section_a["closed"] and consistency["consistent"])
    if not consistency["consistent"]:
        errors.append("输出级一致性核验未过（R2-T5A-05，归因不闭合 ⇒ FAIL）：")
        violations = (
            consistency["zero_input_zero_output_violations"]
            + consistency["unattributed_output_violations"]
        )
        errors.extend(f"  {message}" for message in violations)
    return consistency, attribution_closed


def _assemble_report(
    arm2_dir: Path,
    arm3_dir: Path,
    b0p_dir: Path,
    production_data_root: Path,
    *,
    run_id: str,
    errors: list[str],
    section_a: dict[str, Any],
    b_folds: list[dict[str, Any]],
    runs: Mapping[str, Any],
    section_f: dict[str, Any] | None,
    section_g: dict[str, Any],
    attribution_closed: bool,
) -> dict[str, Any]:
    """组装报告 dict（七节 + 总判定 + 证据指针；落盘在 main）。"""
    verdict_pass = (
        not errors
        and attribution_closed
        and section_g["source_digest_match"]
        and section_f is not None
        and section_f["verdict"] == "PASS"
    )
    return {
        "tool": "run_data_impact_attribution",
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "arm2_dir": str(arm2_dir),
        "arm3_dir": str(arm3_dir),
        "b0p_dir": str(b0p_dir),
        "production_data_root": str(production_data_root),
        "verdict": {
            "verdict": "PASS" if verdict_pass else "FAIL",
            "rule": _VERDICT_RULE,
            "errors": errors,
        },
        "sections": {
            "a_column_input_diffs": section_a,
            "b_prediction_impact": {
                "caliber_note": sections.PREDICTION_CALIBER_NOTE,
                "folds": b_folds,
            },
            "c_trade_impact": {
                "key_semantics": "trade_date + ts_code + action（方向）；同日同股同方向多笔"
                "先按键聚合（shares/amount 求和，price=Σamount/Σshares 推导），"
                "重复键分组计数随节登记（P3-3 口径）",
                "folds": runs["c_trade_folds"],
            },
            "d_holdings_impact": {
                "key_semantics": "trade_date + ts_code；同键多行先按键聚合"
                "（shares/market_value/weight 求和），重复键分组计数随节登记（P3-3 口径）",
                "folds": runs["d_holdings_folds"],
            },
            "e_nav_impact": {
                "chain_nav": runs["e_chain"],
                "daily_folds": runs["e_daily_folds"],
            },
            "f_gdata_judgment_table": section_f,
            "g_identity_appendix": section_g,
        },
        "evidence": {
            "json_path": None,
            "arm2_runs_batch": runs["batch2_dir"],
            "arm3_runs_batch": runs["batch3_dir"],
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--arm2-dir", required=True, help="臂② run 目录（新实现 × cs_train 验证源）"
    )
    parser.add_argument("--arm3-dir", required=True, help="臂③ run 目录（新实现 × panel）")
    parser.add_argument("--b0p-dir", required=True, help="B0' 冻结目录（含 chain_metrics.json）")
    parser.add_argument(
        "--production-data-root", default="data", help="生产数据根（默认 ./data；只读消费）"
    )
    parser.add_argument("--run-id", default=None, help="报告 RUN_ID（默认当前时间戳）")
    parser.add_argument("--out-json", default=None, help="数据 JSON 输出路径（默认按 RUN_ID）")
    parser.add_argument("--out-md", default=None, help="报告 MD 输出路径（默认按日期）")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口：JSON（机器可读全量）+ MD（人读摘要）双落盘；verdict != PASS ⇒ exit 1。"""
    args = _build_parser().parse_args(argv)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = (
        Path(args.out_json)
        if args.out_json
        else Path("data") / "reports" / f"v2_p3_data_impact_{run_id}.json"
    )
    out_md = (
        Path(args.out_md)
        if args.out_md
        else Path("docs") / "reports" / f"v2_p3_data_impact_{run_id}.md"
    )
    from src.lazybull.v2.store.store_identity import workspace_identity

    observer_identity = workspace_identity(_REPO_ROOT)
    cache_dir = out_json.parent / "attribution_cache"  # P2-7：缓存目录落产物旁
    report = build_report(
        Path(args.arm2_dir),
        Path(args.arm3_dir),
        Path(args.b0p_dir),
        Path(args.production_data_root),
        run_id=run_id,
        observer_identity=observer_identity,
        cache_dir=cache_dir,
    )
    cache_stats = report["sections"]["a_column_input_diffs"].get("cache_stats") or {}
    day_stats = cache_stats.get("day_compare") or {}
    month_stats = cache_stats.get("panel_month") or {}
    logger.info(
        f"a 节缓存（P2-7/P3-N4）：单日比较命中率 {day_stats.get('hit_rate', 0.0):.1%}"
        f"（hits={day_stats.get('hits')} / misses={day_stats.get('misses')}，"
        f"命名空间 {day_stats.get('namespace')}），panel 月帧命中率 "
        f"{month_stats.get('hit_rate', 0.0):.1%}（hits={month_stats.get('hits')} / "
        f"misses={month_stats.get('misses')}），目录 {cache_dir}"
    )
    report["evidence"]["json_path"] = str(out_json)
    write_report_json(report, out_json)
    write_report_md(render_markdown(report), out_md)
    verdict = report["verdict"]
    if verdict["verdict"] != "PASS":
        for message in verdict["errors"]:
            logger.error(f"  {message}")
    print(f"verdict: {verdict['verdict']} -> {out_json} / {out_md}")
    return 0 if verdict["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
