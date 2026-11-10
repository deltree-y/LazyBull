"""P6.5 T2 内部结算批次编排：S1 转账 + S4 合并买入 + 结算后 I1/I2 回验。

契约依据：``docs/contracts/netting_freeze.md`` F25 §4/§5、``docs/contracts/protocols.md``
F13 §3.3、设计 ``docs/plans/v2/p65_merger_reconciliation_design_20261108.md``
§4.3/§7.2/§8.3。

职责边界：
- 结算事实结构、Q18 股数分摊与 Q13 费用瀑布在 ``settlement_kernel``（纯函数内核）；
  本模块只做**批次应用编排**；
- 写点唯一在 ``VirtualLedger``：``ledger._commit_settlement`` 之前一切修改只落在
  候选副本上，任何拒绝/破缺都**零提交**；
- 物理成交仍是显式注入事实，唯一成交入账口是 T3 的 ``Ledger.apply_fills``；
- ``virtual_ledger`` 对本模块与内核的公开名做重导出（T1/T2 导入路径不变）。
- 行预算：本模块与内核已按硬限拆分；任何新增须先评估 ``settlement_kernel`` 归属。

检测面（如实登记，不得读成更强结论）：
- 结算后门 = **合计层**：I1（Σ 虚拟现金 vs 物理现金）、I2（按来源键汇总股数）与守恒
  守卫（Σ 来源股数、逐袖负现金）。袖间股数/费用**置换**（合计不变）对 I1/I2 不可见，
  其正确性只由 Q13/Q18 黄金值单测与 `consumed_lots` 证据保证；逐袖借用占用归 I1 门；
- 伴随视图注入顺序义务：物理成交由验收器在调用 ``apply_settlement`` **之前**注入
  ``PhysicalLotView``（T3 的 ``Ledger.apply_fills`` 职责）。结算拒绝只保证**账本零提交**，
  不回滚调用方已注入的视图；调用方须以"先构造候选视图、结算成功后再替换"的方式使用；
- 内部 API 面：编排函数经 ``ledger`` 的包内受保护成员（``_books`` / ``_copy_books`` /
  ``_fifo_*`` / ``_assert_*`` / ``_poison`` / ``_commit_settlement``）访问账本；该面是
  accounting 包内部契约，不是公开协议（公开协议仍只有 ``Ledger``/``PortfolioMerger``）；
- 输入绑定：每条意图的 ``current_shares`` 必须等于账本当前虚拟持仓（容差 0 股）。目标
  股数不是增量，陈旧快照会放大增量并造成**超目标分配**，而 I1/I2 的合计守恒对此不可见
  （C65-T2-01）；有实际增量的意图股票都必须有同日价格，零成交短路不豁免缺价拒绝
  （C65-T2-02）。
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.lazybull.v2.core.accounting.physical_lot_view import PhysicalLotView
from src.lazybull.v2.core.accounting.settlement_kernel import (  # noqa: F401（重导出）
    FeeWaterfall,
    InternalSettlementBatch,
    PhysicalFillFact,
    SettlementResult,
    ShareAllocation,
    SleeveIntent,
    TransferFact,
    _close_terminal_fee,
    _distribute_remainder,
    _error,
    _left_to_right,
    allocate_fees,
    allocate_shares,
)

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查（避免与 virtual_ledger 循环导入）
    from src.lazybull.v2.core.accounting.virtual_ledger import (
        InternalTransfer,
        VirtualLedger,
        _SleeveBook,
    )

#: 本模块唯一阶段标识（进入全部错误上下文）
_STAGE = "P6.5"


def apply_settlement(
    ledger: "VirtualLedger",
    batch: InternalSettlementBatch,
    *,
    price_source: Any,
    physical_view: PhysicalLotView,
    identity_context: Any,
    physical_positions: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> SettlementResult:
    """应用内部结算批次（S1 内部转账 + S4 合并买入分摊），并回验 I1/I2。

    物理事实（成交价、费用、结算后物理现金、物理来源 lot）必须由验收器
    **显式注入**：本函数不产生市场指令、不写生产账本，只做虚拟投影与对账。
    一切修改先落在候选副本上，任何拒绝/破缺都保持**零提交**。

    Args:
        ledger: 目标虚拟账本（同包内部写点；只经 ``_commit_settlement`` 提交）
        batch: 内部结算批次（原始意图 + 物理成交/费用 + 转账 + 幂等身份）
        price_source: 当日同一价源（成交价/转账价必须逐股相等）
        physical_view: **已注入本批物理成交**的来源 lot 伴随视图（I2 右值）
        identity_context: 恒等式上下文（日期/宿主/运行 ID/袖集合/价源标识）
        physical_positions: 引擎聚合持仓（可选；提供时先与伴随视图对账）

    Returns:
        ``SettlementResult``（``applied=False`` = 同身份同内容重复提交）。

    Raises:
        VirtualLedgerError: 输入非法/超意图成交/零成交有费用/费用精确和不成立/
            同 ``batch_id`` 内容不同。全部在提交前完成，**零副作用**。
        ReconciliationError: 结算后 I1/I2 破缺（**零提交**；实例置终态）。
        VirtualLedgerConservationError: 落账后自检发现守恒破缺（零提交；置终态）。
    """
    from src.lazybull.v2.core.accounting.virtual_ledger import (
        PriceSource,
        VirtualLedgerConservationError,
    )

    ledger._assert_writable()
    # T2-P3-7：类型校验必须先于 ``batch.batch_id`` 属性访问，否则非法类型会以
    # AttributeError 漂移出结算拒绝面（错误不含阶段/批次上下文）。
    if not isinstance(batch, InternalSettlementBatch):
        raise _error(
            f"阶段={_STAGE} 结算输入必须为 InternalSettlementBatch，实际 "
            f"{type(batch).__name__}（拒绝面必须在任何属性访问前完成）"
        )
    if not isinstance(price_source, PriceSource):
        raise _error(
            f"阶段={_STAGE} 批次={batch.batch_id}：价格源必须为 PriceSource，实际 "
            f"{type(price_source).__name__}"
        )
    if batch.batch_id in ledger._settlements:
        raise _error(
            f"阶段={_STAGE} 批次={batch.batch_id}：该身份已被内部转账占用；"
            "同一结算键不得承载两套事实（转账/批次）"
        )
    fingerprint = batch.fingerprint()
    recorded = ledger._batches.get(batch.batch_id)
    if recorded is not None:
        if recorded[0] == fingerprint:
            return replace(recorded[1], applied=False)
        raise _error(
            f"阶段={_STAGE} 批次={batch.batch_id} 日期={batch.trade_date}："
            "内部结算批次幂等键冲突（已登记但内容不同）；禁止同一身份承载不同结算事实"
        )
    # 幂等短路之后再查派生键：同一批次的重放不得因自己上次登记的派生键被拒
    _assert_derived_ids_free(ledger, batch)
    _validate_settlement_inputs(ledger, batch, price_source, identity_context)
    if not isinstance(physical_view, PhysicalLotView):
        raise _error(
            f"阶段={_STAGE} 批次={batch.batch_id}：结算需要显式注入物理 lot 伴随视图，"
            f"实际 {type(physical_view).__name__}"
        )
    candidate = ledger._copy_books()
    shares_before = ledger._source_share_totals()
    _assert_intent_coverage(batch)
    transfers, consumed = _apply_batch_transfers(ledger, candidate, batch, physical_view)
    allocations, waterfalls, voided = _apply_batch_fills(ledger, candidate, batch)
    expected_shares = dict(shares_before)
    for fill in batch.physical_fills:
        key = (fill.ts_code, fill.source_lot_id)
        expected_shares[key] = expected_shares.get(key, 0) + fill.shares
    label = f"阶段={_STAGE} 批次={batch.batch_id} 日期={batch.trade_date}"
    try:
        ledger._assert_cash_total(
            context=label,
            expected=batch.physical_cash_after,
            books=candidate,
            phrase="结算后 Σ 虚拟现金与物理现金不一致（差应为 0，物理费用已在各袖费用内）",
        )
        ledger._assert_source_shares(context=label, expected=expected_shares, books=candidate)
        ledger._assert_no_negative_cash(context=label, books=candidate)
    except VirtualLedgerConservationError as error:
        ledger._poison(str(error))
        raise
    identity_results = _run_settlement_identity_gate(
        ledger,
        batch=batch,
        books=candidate,
        physical_view=physical_view,
        identity_context=identity_context,
        physical_positions=physical_positions,
    )
    result = SettlementResult(
        batch_id=batch.batch_id,
        applied=True,
        share_allocations=allocations,
        fee_waterfalls=waterfalls,
        transfers=transfers,
        consumed_lots=consumed,
        cash_total_after=ledger._cash_total(candidate),
        source_shares_after=tuple(
            (code, lot_id, shares)
            for (code, lot_id), shares in sorted(ledger._source_share_totals(candidate).items())
        ),
        zero_fill=(
            not batch.physical_fills
            and not batch.transfers
            and any(intent.delta_shares != 0 for intent in batch.intents)
        ),
        voided_intents=voided,
        identity_results=identity_results,
        positions_reconciled=physical_positions is not None,
    )
    ledger._commit_settlement(
        batch=batch,
        fingerprint=fingerprint,
        books=candidate,
        result=result,
        source_id=price_source.source_id,
    )
    return result


def _assert_derived_ids_free(ledger: "VirtualLedger", batch: InternalSettlementBatch) -> None:
    """派生转账子键（``<batch>#T<i>``）不得与既有结算身份冲突（双向零副作用）。

    A-R4-01/B-R4-01：外部先以 ``S1#T0`` 登记转账、批次再提交时，提交不得静默覆盖
    既有幂等指纹；发现占用即拒绝，保持零提交。
    """
    for index in range(len(batch.transfers)):
        derived = f"{batch.batch_id}#T{index}"
        if derived in ledger._settlements or derived in ledger._batches:
            raise _error(
                f"阶段={_STAGE} 批次={batch.batch_id} 日期={batch.trade_date}："
                f"派生转账子键 {derived} 已被既有结算身份占用；"
                "同一结算键不得承载两套事实（禁止覆盖既有幂等指纹）"
            )


def _validate_settlement_inputs(
    ledger: "VirtualLedger",
    batch: InternalSettlementBatch,
    price_source: Any,
    identity_context: Any,
) -> None:
    """结算输入校验（价格同日同源、袖集合一致、引用袖已开户、当前股数与账本一致）。"""
    label = f"阶段={_STAGE} 批次={batch.batch_id} 日期={batch.trade_date}"
    _validate_batch_context(ledger, batch, price_source, identity_context, label)
    _assert_current_shares_match_ledger(ledger, batch, label)
    _assert_prices_match(price_source, batch, label)


def _validate_batch_context(
    ledger: "VirtualLedger",
    batch: InternalSettlementBatch,
    price_source: Any,
    identity_context: Any,
    label: str,
) -> None:
    """同日同价源、上下文日期/袖集合、引用袖开户（C65-T2-01 前的结构校验）。"""
    if price_source.as_of != batch.trade_date:
        raise _error(
            f"{label}：价格源交易日 {price_source.as_of} 与结算日不一致（必须同日同价源）"
        )
    # T2-P3-3：上下文标注的价源标识必须与本次结算实际使用的价源一致，否则
    # 证据会把结算定价标注到另一套价源上（标注口）。
    context_source = getattr(identity_context, "price_source_id", None)
    if context_source != price_source.source_id:
        raise _error(
            f"{label}：恒等式上下文价源标识 {context_source!r} != 结算价源 "
            f"{price_source.source_id!r}；证据不得标注未参与定价的价源"
        )
    context_date = getattr(identity_context, "trade_date", None)
    if context_date != batch.trade_date:
        raise _error(f"{label}：恒等式上下文日期 {context_date} 与结算日不一致")
    context_sleeves = tuple(sorted(getattr(identity_context, "sleeve_ids", ())))
    if context_sleeves != tuple(sorted(ledger._books)):
        raise _error(
            f"{label}：恒等式上下文袖集合={list(context_sleeves)} 与账本已开户袖="
            f"{sorted(ledger._books)} 不一致（证据不得标注未校验的袖）"
        )
    referenced = {intent.sleeve_id for intent in batch.intents}
    for fact in batch.transfers:
        referenced.add(fact.from_sleeve)
        referenced.add(fact.to_sleeve)
    missing = sorted(referenced - set(ledger._books))
    if missing:
        raise _error(f"{label}：引用了未开户袖 {missing}；已开户={sorted(ledger._books)}")


def _assert_prices_match(price_source: Any, batch: InternalSettlementBatch, label: str) -> None:
    """转账价/成交价必须等于同日价源；有增量的意图股票必须有价（C65-T2-02）。"""
    from src.lazybull.v2.core.accounting.virtual_ledger import VirtualLedgerError

    for fact in batch.transfers:
        source_price = price_source.price_of(fact.ts_code)
        if fact.price != source_price:
            raise _error(
                f"{label} 股票={fact.ts_code}：内部转账价 {fact.price!r} != 价源 "
                f"{price_source.source_id} 价格 {source_price!r}（F25 §5 定价契约）"
            )
    for fill in batch.physical_fills:
        source_price = price_source.price_of(fill.ts_code)
        if fill.price != source_price:
            raise _error(
                f"{label} 股票={fill.ts_code}：物理成交价 {fill.price!r} != 价源 "
                f"{price_source.source_id} 价格 {source_price!r}"
                "（F25 §5 同日同价源定价契约）"
            )
    # C65-T2-02（P2）：零成交短路不豁免价格拒绝面——凡有实际增量的意图股票都必须有
    # 同日有效价格（F25 §4 停牌/无价 fail-fast；短路只免除费用模型与落 lot）。
    incremental_codes = sorted(
        {intent.ts_code for intent in batch.intents if intent.delta_shares != 0}
    )
    for code in incremental_codes:
        try:
            price_source.price_of(code)
        except VirtualLedgerError as error:
            raise _error(
                f"{label} 股票={code}：{error}；零成交短路不豁免缺价拒绝（禁止登记为合法零成交）"
            ) from error


def _assert_current_shares_match_ledger(
    ledger: "VirtualLedger", batch: InternalSettlementBatch, label: str
) -> None:
    """意图声明的当前股数必须等于账本当前虚拟持仓（C65-T2-01 / P1）。

    目标股数不是增量：增量必须由**当前账本事实**解析（F13 §3.4）。陈旧快照会放大
    增量并造成超目标分配，而 I1/I2 的合计守恒对此完全不可见（`a_i` 与物理成交一致）。
    因此这里逐条对账、容差 0 股、不符即零副作用拒绝。
    """
    for intent in batch.intents:
        actual = ledger.state_of(intent.sleeve_id).shares_of(intent.ts_code)
        if intent.current_shares != actual:
            raise _error(
                f"{label} 袖={intent.sleeve_id} 股票={intent.ts_code}：意图声明当前股数 "
                f"{intent.current_shares} != 账本当前虚拟持仓 {actual}"
                f"（左-右={intent.current_shares - actual} 股，容差=0 股）；"
                "增量必须由当前账本持仓解析，禁止用陈旧快照扩大增量"
            )


def _assert_intent_coverage(batch: InternalSettlementBatch) -> None:
    """意图覆盖与 S1 一致性：等量反向、总量与逐袖归属匹配、无事实不得静默。

    A-R1-01/A-R2-02：每条 ``delta≠0`` 的意图必须被物理成交或内部转账覆盖；卖出/反向
    意图缺事实即拒绝（全零成交短路只允许全体买入增量）；S1 只接受一对等量反向。
    """
    label = f"阶段={_STAGE} 批次={batch.batch_id} 日期={batch.trade_date}"
    fill_codes = {fill.ts_code for fill in batch.physical_fills}
    transfer_codes = {fact.ts_code for fact in batch.transfers}
    for fact in batch.transfers:
        if fact.ts_code in fill_codes:
            raise _error(
                f"{label} 股票={fact.ts_code}：同一股票既登记内部转账又登记物理成交；"
                "S1 抵消部分零物理指令，拒绝混用"
            )
    for code in sorted(transfer_codes):
        _assert_transfer_code(label, batch, code)
    # 意图覆盖：既无成交也无转账的股票只允许"全体买入增量"（全零成交短路）；
    # 卖出/反向意图缺事实即拒绝，禁止声明了抵消却不落任何虚拟账。
    for code in sorted({item.ts_code for item in batch.intents}):
        if code in fill_codes or code in transfer_codes:
            continue
        uncovered = [
            item for item in batch.intents if item.ts_code == code and item.delta_shares != 0
        ]
        illegal = [item for item in uncovered if item.side != "buy" or item.delta_shares < 0]
        if illegal:
            detail = {item.sleeve_id: (item.side, item.delta_shares) for item in illegal}
            raise _error(
                f"{label} 股票={code}：意图缺结算事实（既无物理成交也无内部转账）——"
                f"{detail}；卖出/反向意图必须由 S1 转账事实覆盖，禁止静默不落账"
            )


def _collect_intent_directions(
    label: str, batch: InternalSettlementBatch, code: str
) -> Tuple[List[SleeveIntent], int, set, set]:
    """单只股票的意图方向解析：符号一致、等量反向、且严格一对袖（S1 契约）。"""
    relevant = [item for item in batch.intents if item.ts_code == code]
    if not relevant:
        raise _error(f"{label} 股票={code}：内部转账缺原始意图事实（禁止无来源转账）")
    sell_total = 0
    buy_total = 0
    sell_ids: set = set()
    buy_ids: set = set()
    for item in relevant:
        if item.delta_shares == 0:
            # 零增量意图不参与方向解析（F25 §3 未要求拒绝；S4 路径同样放过，A-R4-02）
            continue
        delta = item.delta_shares
        if item.side == "sell" and delta >= 0:
            raise _error(
                f"{label} 股票={code} 袖={item.sleeve_id}：卖出意图解析增量 {delta} 股 ≥ 0"
                f"（目标={item.target_shares} 当前={item.current_shares}）"
            )
        if item.side == "buy" and delta <= 0:
            raise _error(
                f"{label} 股票={code} 袖={item.sleeve_id}：买入意图解析增量 {delta} 股 ≤ 0"
                f"（目标={item.target_shares} 当前={item.current_shares}）"
            )
        if delta < 0:
            sell_total += -delta
            sell_ids.add(item.sleeve_id)
        elif delta > 0:
            buy_total += delta
            buy_ids.add(item.sleeve_id)
    if sell_total != buy_total:
        raise _error(
            f"{label} 股票={code}：状态机=S2/S3 同股反向但净残差 {buy_total - sell_total} 股"
            f"（卖={sell_total} 买={buy_total}）；P6.5 只覆盖 S1 等量反向与 S4 同向买入，"
            "净残差顺延归 P7"
        )
    if len(sell_ids) > 1 or len(buy_ids) > 1:
        raise _error(
            f"{label} 股票={code}：状态机=S1 净残差为 0 但涉及 {len(sell_ids)} 卖 / "
            f"{len(buy_ids)} 买；S1 只定义一对等量反向，多卖/多买属未登记推广，拒绝"
        )
    return relevant, sell_total, sell_ids, buy_ids


def _summarize_transfers(
    label: str,
    batch: InternalSettlementBatch,
    code: str,
    sell_ids: set,
    buy_ids: set,
) -> Tuple[Dict[str, int], Dict[str, int], int]:
    """单只股票的转账方向与逐袖合计（转出/转入按袖）。"""
    out_by_sleeve: Dict[str, int] = {}
    in_by_sleeve: Dict[str, int] = {}
    transfer_total = 0
    for fact in batch.transfers:
        if fact.ts_code != code:
            continue
        if fact.from_sleeve not in sell_ids or fact.to_sleeve not in buy_ids:
            raise _error(
                f"{label} 股票={code}：转账 {fact.from_sleeve}→{fact.to_sleeve} 与意图方向"
                f"不符（卖方={sorted(sell_ids)} 买方={sorted(buy_ids)}）"
            )
        out_by_sleeve[fact.from_sleeve] = out_by_sleeve.get(fact.from_sleeve, 0) + fact.shares
        in_by_sleeve[fact.to_sleeve] = in_by_sleeve.get(fact.to_sleeve, 0) + fact.shares
        transfer_total += fact.shares
    return out_by_sleeve, in_by_sleeve, transfer_total


def _assert_per_sleeve_attribution(
    label: str,
    code: str,
    relevant: Sequence[SleeveIntent],
    out_by_sleeve: Mapping[str, int],
    in_by_sleeve: Mapping[str, int],
) -> None:
    """逐袖归属零差（A-R1-02）：I2 只按来源键汇总，袖间置换不能替代本断言。"""
    for item in relevant:
        delta = item.delta_shares
        if delta == 0:
            continue
        if delta < 0 and out_by_sleeve.get(item.sleeve_id, 0) != -delta:
            raise _error(
                f"{label} 股票={code} 袖={item.sleeve_id}：转出 "
                f"{out_by_sleeve.get(item.sleeve_id, 0)} 股 != 应抵消 {-delta} 股（差="
                f"{out_by_sleeve.get(item.sleeve_id, 0) + delta} 股 容差=0 股 单位=股）"
            )
        if delta > 0 and in_by_sleeve.get(item.sleeve_id, 0) != delta:
            raise _error(
                f"{label} 股票={code} 袖={item.sleeve_id}：转入 "
                f"{in_by_sleeve.get(item.sleeve_id, 0)} 股 != 应受让 {delta} 股（差="
                f"{in_by_sleeve.get(item.sleeve_id, 0) - delta} 股 容差=0 股 单位=股）"
            )


def _assert_transfer_code(label: str, batch: InternalSettlementBatch, code: str) -> None:
    """单只股票：方向/总量/逐袖归属三层校验（等量反向）。"""
    relevant, sell_total, sell_ids, buy_ids = _collect_intent_directions(label, batch, code)
    out_by_sleeve, in_by_sleeve, transfer_total = _summarize_transfers(
        label, batch, code, sell_ids, buy_ids
    )
    if transfer_total != sell_total:
        raise _error(
            f"{label} 股票={code}：内部转账合计 {transfer_total} 股 != 抵消股数 "
            f"{sell_total} 股（差={transfer_total - sell_total} 股 容差=0 股 单位=股）"
        )
    _assert_per_sleeve_attribution(label, code, relevant, out_by_sleeve, in_by_sleeve)


def _fifo_allocations_in(
    ledger: "VirtualLedger", book: "_SleeveBook", sleeve_id: str, ts_code: str, shares: int
) -> Tuple[Any, ...]:
    """候选账上的 FIFO 扣减计划（复用账本侧唯一规则 ``_plan_fifo_allocations``）。"""
    return ledger._plan_fifo_allocations(book, sleeve_id, ts_code, shares)


def _apply_batch_transfers(
    ledger: "VirtualLedger",
    candidate: Mapping[str, "_SleeveBook"],
    batch: InternalSettlementBatch,
    physical_view: PhysicalLotView,
) -> Tuple[Tuple["InternalTransfer", ...], Tuple[Tuple[str, str, int], ...]]:
    """在候选账上应用内部转账（FIFO 消耗），返回事实与消耗批次证据。"""
    from src.lazybull.v2.core.accounting.virtual_ledger import InternalTransfer

    applied: List[Any] = []
    consumed_evidence: List[Tuple[str, str, int]] = []
    for index, fact in enumerate(batch.transfers):
        seller = candidate[fact.from_sleeve]
        buyer = candidate[fact.to_sleeve]
        allocations = _fifo_allocations_in(
            ledger, seller, fact.from_sleeve, fact.ts_code, fact.shares
        )
        transfer = InternalTransfer(
            settlement_id=f"{batch.batch_id}#T{index}",
            trade_date=batch.trade_date,
            ts_code=fact.ts_code,
            from_sleeve=fact.from_sleeve,
            to_sleeve=fact.to_sleeve,
            price=fact.price,
            allocations=allocations,
        )
        consumed = ledger._resolve_allocations(seller, transfer)
        ledger._assert_lot_dates(transfer, consumed)
        amount = transfer.total_shares * transfer.price
        if buyer.cash - amount < 0.0:
            raise _error(
                f"阶段={_STAGE} 批次={batch.batch_id} 日期={batch.trade_date} "
                f"股票={fact.ts_code}：虚拟买入现金不足（无借用）——袖 {fact.to_sleeve} "
                f"现金 {buyer.cash!r} < 应付 {amount!r}（差={buyer.cash - amount!r} 元）"
            )
        ledger._assert_acquisition_ids_free(buyer, transfer)
        physical_view.validate_transfer(
            ts_code=transfer.ts_code,
            allocations=[(lot.source_lot_id, shares) for lot, shares in consumed],
        )
        ledger._apply_transfer(transfer, seller, buyer, consumed, amount)
        applied.append(transfer)
        consumed_evidence.extend(
            (transfer.settlement_id, lot.virtual_lot_id, shares) for lot, shares in consumed
        )
    return tuple(applied), tuple(consumed_evidence)


def _fill_increments(
    batch: InternalSettlementBatch, fill: PhysicalFillFact
) -> Dict[str, int]:
    """单笔成交对应的各袖原始增量（只接受全体同向买入）。"""
    intents = [item for item in batch.intents if item.ts_code == fill.ts_code]
    if not intents:
        raise _error(
            f"阶段={_STAGE} 批次={batch.batch_id} 日期={batch.trade_date} "
            f"股票={fill.ts_code}：物理成交缺原始意图事实"
        )
    increments: Dict[str, int] = {}
    for item in intents:
        if item.delta_shares == 0:
            # T2-P3-2：零增量意图不参与方向解析（与 S1 路径 ``_collect_intent_directions``
            # 同口径）；否则"持有袖目标=当前"会以 side=sell/delta=0 误拒合法批次。
            continue
        if item.side != "buy" or item.delta_shares < 0:
            raise _error(
                f"阶段={_STAGE} 批次={batch.batch_id} 状态机=S2/S3/S5 股票={fill.ts_code} "
                f"袖={item.sleeve_id}：方向={item.side} 解析增量 {item.delta_shares} 股；"
                "S4 只接受全体同向买入，反向残差/同向卖出归 P7"
            )
        increments[item.sleeve_id] = item.delta_shares
    return increments


def _apply_one_fill(
    ledger: "VirtualLedger",
    candidate: Mapping[str, "_SleeveBook"],
    batch: InternalSettlementBatch,
    fill: PhysicalFillFact,
) -> Tuple[Tuple[ShareAllocation, ...], FeeWaterfall, Tuple[Tuple[str, str, int], ...]]:
    """单笔买入成交：Q18 股数 + Q13 费用 + 虚拟现金/取得批次（只消费最终费用）。

    Returns:
        ``(股数分摊, 费用瀑布, 未成交作废意图)``——后者为 Q22 的日终作废证据。
    """
    from src.lazybull.v2.core.accounting.virtual_ledger import VirtualLedgerError, VirtualLot

    label = f"阶段={_STAGE} 批次={batch.batch_id} 日期={batch.trade_date} 股票={fill.ts_code}"
    increments = _fill_increments(batch, fill)
    try:
        share_items = allocate_shares(increments, fill.shares)
        intent_amounts = {sid: shares * fill.price for sid, shares in increments.items()}
        allocated_amounts = {
            entry.sleeve_id: entry.allocated_shares * fill.price for entry in share_items
        }
        waterfall = allocate_fees(
            intent_amounts=intent_amounts,
            allocated_amounts=allocated_amounts,
            physical_fee=fill.fee,
            buy_cost=ledger._buy_cost,
        )
    except VirtualLedgerError as error:
        # 内核错误补批次上下文（A-R4-06）：Q18/Q13 纯函数不持有批次身份
        raise VirtualLedgerError(f"{label}：{error}") from error
    for entry in share_items:
        if entry.allocated_shares == 0:
            continue
        book = candidate[entry.sleeve_id]
        fee = waterfall.fee_of(entry.sleeve_id)
        cash = entry.allocated_shares * fill.price + fee
        if book.cash - cash < 0.0:
            raise _error(
                f"阶段={_STAGE} 批次={batch.batch_id} 日期={batch.trade_date} "
                f"股票={fill.ts_code} 袖={entry.sleeve_id}：费用后虚拟现金不足"
                f"（无借用）——现金 {book.cash!r} < 应付 {cash!r} 元"
            )
        book.cash -= cash
        book.lots.append(
            VirtualLot(
                virtual_lot_id=f"{batch.batch_id}#F#{fill.ts_code}#{entry.sleeve_id}",
                ts_code=fill.ts_code,
                source_lot_id=fill.source_lot_id,
                shares=entry.allocated_shares,
                acquired_date=batch.trade_date,
                acquired_price=fill.price,
                acquired_fee=fee,
            )
        )
    voided = tuple(
        (entry.sleeve_id, fill.ts_code, entry.intent_shares - entry.allocated_shares)
        for entry in share_items
        if entry.allocated_shares < entry.intent_shares
    )
    return share_items, waterfall, voided


def _apply_batch_fills(
    ledger: "VirtualLedger",
    candidate: Mapping[str, "_SleeveBook"],
    batch: InternalSettlementBatch,
) -> Tuple[
    Tuple[ShareAllocation, ...],
    Tuple[FeeWaterfall, ...],
    Tuple[Tuple[str, str, int], ...],
]:
    """在候选账上应用 S4 合并买入（Q18 股数 + Q13 费用 + 取得批次）。

    Returns:
        ``(全部分摊, 全部瀑布, 未成交作废意图)``；无成交时作废意图 = 未被内部转账覆盖的
        买入增量（S1 已由转账完成，**不得**登记为作废）。
    """
    allocations: List[ShareAllocation] = []
    waterfalls: List[FeeWaterfall] = []
    voided: List[Tuple[str, str, int]] = []
    for fill in batch.physical_fills:
        share_items, waterfall, fill_voided = _apply_one_fill(ledger, candidate, batch, fill)
        allocations.extend(share_items)
        waterfalls.append(waterfall)
        voided.extend(fill_voided)
    # Q22 逐股票作废：未被成交、也未被内部转账覆盖的买入增量日终作废。
    # 该口径对"混合批次中某股零成交"同样成立（A-R3-01：不得因其他股有成交而静默丢弃）。
    filled_codes = {fill.ts_code for fill in batch.physical_fills}
    covered_codes = {fact.ts_code for fact in batch.transfers}
    for intent in batch.intents:
        if intent.delta_shares <= 0:
            continue
        if intent.ts_code in filled_codes or intent.ts_code in covered_codes:
            continue
        voided.append((intent.sleeve_id, intent.ts_code, intent.delta_shares))
    _assert_physical_cash_delta(batch)
    return tuple(allocations), tuple(waterfalls), tuple(sorted(voided))


def _assert_physical_cash_delta(batch: InternalSettlementBatch) -> None:
    """物理现金减少额必须等于 Σ(成交股数×成交价 + 物理费用)（结算事实自洽前置）。"""
    from src.lazybull.v2.core.accounting.virtual_ledger import _cash_tolerance

    expected = 0.0
    for fill in batch.physical_fills:
        expected += fill.shares * fill.price
        expected += fill.fee
    actual = batch.physical_cash_before - batch.physical_cash_after
    tolerance = _cash_tolerance(max(abs(actual), abs(expected)))
    if abs(actual - expected) > tolerance:
        raise _error(
            f"阶段={_STAGE} 批次={batch.batch_id} 日期={batch.trade_date}：物理现金"
            f"减少额 {actual!r} 元 != Σ(成交额+费用) {expected!r} 元（差="
            f"{actual - expected!r} 元 容差={tolerance} 单位=元）；"
            "结算事实自相矛盾，结算前拒绝"
        )


def _run_settlement_identity_gate(
    ledger: "VirtualLedger",
    *,
    batch: InternalSettlementBatch,
    books: Mapping[str, "_SleeveBook"],
    physical_view: PhysicalLotView,
    identity_context: Any,
    physical_positions: Optional[Mapping[str, Mapping[str, Any]]],
) -> Tuple[Any, ...]:
    """结算后回验 I1/I2（T1 登记的 T2 义务；破缺即抛且保持零提交）。

    走 ``reconciliation.check_settlement_identities``（与逐日门同一实现：失败路径汇总
    全部破缺编号），避免同一语义出现两条互不相同的分支（A-R2-08）；成功侧把
    ``IdentityResult`` 元组回传给结算结果（A-R4-04）。
    """
    from src.lazybull.v2.core.accounting.reconciliation import (
        ReconciliationError,
        check_settlement_identities,
    )

    try:
        return check_settlement_identities(
            physical_cash=batch.physical_cash_after,
            physical_view=physical_view,
            virtual_states=ledger._states_of(books),
            context=identity_context,
            physical_positions=physical_positions,
        )
    except ReconciliationError as error:
        ledger._poison(f"结算后恒等式破缺：{error}")
        raise
