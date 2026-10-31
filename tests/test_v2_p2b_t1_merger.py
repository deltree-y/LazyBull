# -*- coding: utf-8 -*-
"""P2b-T1 单袖合并器测试（A1 直通 + S6 单袖透传 + core 内部过渡端口定义）。

验收口径（实施规划 ``docs/plans/v2/p2b_implementation_plan_20261016.md`` §5-T1
任务卡 / §6 门 6 / §7 合成测试矩阵；净额化冻结 F3 A1/S6；协议 F9）：

- A 组（A1 直通强等价，D2）：返回对象身份（``merged is targets[sleeve_id]``）、
  权重映射对象身份与插入顺序、``Weight`` 值对象身份逐键锁定；不读取
  ``current``（爆炸哨兵）；不修改 ``targets``/目标对象/权重映射/当前账户
  （前后快照比对）；非精确权重和与 NaN/零值原样透传（**仅证明不改写输入，
  不构成真实交易合法输入证明**，D2 末段）。
- B 组（结构防越界，D1）：空目标映射与多目标映射分别拒绝、映射键与
  ``target.sleeve_id`` 不一致拒绝，错误含 sleeve 集合与阶段信息；
  **单目标内空权重映射属直通情形**（与空目标映射严格区分），原样透传。
- C 组（S6）：空指令序列与同袖多指令返回原序列及原指令对象（**调用前快照**
  逐位比对，R4-T1-01 处置：同列表自引用断言无法捕获原地重排）；混合
  ``source_sleeve``（反向/同向/自拆）fail-fast，错误含 sleeve 集合与阶段。
  S6 证据为单元级；真实接线证据归 T2/T3 分别登记（门 6）。
- D 组（结构兼容与过渡件删除）：``SingleSleevePortfolioMerger`` 对
  ``PortfolioMerger`` 协议 runtime_checkable 结构兼容 + 完整签名等同
  （参数名/种类/注解/**顺序**/返回，范式同 test_v2_p2a_t2_protocols.py）；
  P4-T3 起反向断言旧 ``LegacyTargetMergePort`` 与适配器文件不存在；core 模块
  不得 import hosts（F11 §4.10 依赖方向；轻量锁覆盖完整路径/父包/裸/相对
  四类导入形态并附逃逸形反例，残余边界归 T3 门 2 AST 工具）。
- E 组（反变异自证）：原地重排、原地改浮点、偷偷归一化、复制新对象、读取
  ``current``、多目标放行、混合袖放行、**原地反转指令序列**八类变异必须被
  对应断言捕获，且先锚定变异确实生效（防变异空转的假通过）。

全部输入为合成值对象，不依赖真实配置与真实数据。
"""

import ast
import inspect
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, get_type_hints

import pandas as pd
import pytest

from src.lazybull.v2.common.protocols.core import PortfolioMerger
from src.lazybull.v2.common.types import (
    Account,
    Money,
    Order,
    OrderReason,
    OrderSide,
    OrderType,
    Shares,
    TargetPortfolio,
    TradeDate,
    TSCode,
    Weight,
)
from src.lazybull.v2.core.decision import portfolio_merger as merger_mod
from src.lazybull.v2.core.decision.portfolio_merger import SingleSleevePortfolioMerger

_TSC_A = TSCode("600000.SH")
_TSC_B = TSCode("000001.SZ")
_TSC_C = TSCode("300750.SZ")
_TSC_D = TSCode("601318.SH")
_D0 = TradeDate.from_str("20260105")

_POK = inspect.Parameter.POSITIONAL_OR_KEYWORD
_KW_ONLY = inspect.Parameter.KEYWORD_ONLY
_EMPTY = inspect.Parameter.empty


def _full_signature(func) -> tuple:
    """完整有序签名解析（范式同 test_v2_p2a_t2_protocols.py）。

    每个参数描述 = (name, kind, default, annotation)，以有序元组返回——参数
    顺序是签名的一部分；get_type_hints 解析 ``from __future__ import
    annotations`` 下的字符串注解。
    """
    hints = get_type_hints(func)
    sig = inspect.signature(func)
    params = tuple(
        (name, param.kind, param.default, hints.get(name)) for name, param in sig.parameters.items()
    )
    return params, hints.get("return")


def _account() -> Account:
    """合成账户快照（空持仓；初始资金语义由 T2 适配器测试另行锚定）。"""
    return Account(cash=Money(500000.0), positions={}, total_value=Money(1000000.0), nav=1.0)


class _ExplosiveCurrent:
    """任意属性读取即抛错的 current 哨兵（锁定 D2「不读取 current」）。"""

    def __getattr__(self, name: str):
        raise RuntimeError(f"merge 不得读取 current.{name}")


def _target(sleeve_id: str = "value", weights: Optional[Mapping[TSCode, Weight]] = None):
    if weights is None:
        weights = {_TSC_A: Weight(0.6), _TSC_B: Weight(0.4)}
    return TargetPortfolio(sleeve_id=sleeve_id, target_weights=weights, as_of_date=_D0)


def _non_exact_weights() -> Dict[TSCode, Weight]:
    """非精确权重和（Σ=1.3）+ NaN + 零值；插入序刻意为非字典序。"""
    return {
        _TSC_A: Weight(0.9),
        _TSC_B: Weight(0.4),
        _TSC_C: Weight(float("nan")),
        _TSC_D: Weight(0.0),
    }


def _order(ts_code: str, side: OrderSide, sleeve: str) -> Order:
    return Order(
        ts_code=TSCode(ts_code),
        side=side,
        target_shares=Shares(100),
        order_type=OrderType.MARKET_OPEN,
        reason=OrderReason.REBALANCE,
        source_sleeve=sleeve,
    )


def _assert_exact_passthrough(merger, target: TargetPortfolio, current) -> None:
    """直通强等价断言组（对象/映射/顺序/值对象身份 + 输入零改写），正例与反变异复用。"""
    weights = target.target_weights
    original_items = list(weights.items())
    merged = merger.merge({"value": target}, current)
    assert merged is target, "返回对象必须是映射内唯一目标对象本身"
    assert merged.target_weights is weights, "权重映射必须保持原映射对象"
    assert list(merged.target_weights.items()) == original_items, "权重内容/顺序不得改变"
    for code, weight in original_items:
        assert merged.target_weights[code] is weight, "Weight 值对象不得被替换"
    assert list(weights.items()) == original_items, "输入权重映射不得被改写"


def _reject_check(fn) -> bool:
    """fn() 必须以 ValueError 失败（fail-fast）；返回是否如期拒绝（供反变异复用）。"""
    try:
        fn()
    except ValueError:
        return True
    return False


def _assert_multi_target_rejected(merger) -> None:
    targets = {"growth": _target("growth"), "value": _target("value")}
    try:
        merger.merge(targets, _account())
    except ValueError:
        return
    raise AssertionError("多目标映射必须被拒绝")


def _assert_mixed_sleeves_rejected(merger) -> None:
    """混合 source_sleeve（反向/同向/自拆）必须 fail-fast，错误含 sleeve 集合与阶段。"""
    cases = {
        "反向抵消（A 卖 X + B 买 X）": [
            _order("600000.SH", OrderSide.SELL, "value"),
            _order("600000.SH", OrderSide.BUY, "growth"),
        ],
        "同向合并（两袖同买）": [
            _order("600000.SH", OrderSide.BUY, "value"),
            _order("000001.SZ", OrderSide.BUY, "growth"),
        ],
        "自拆两子袖": [
            _order("600000.SH", OrderSide.BUY, "value_a"),
            _order("600000.SH", OrderSide.SELL, "value_b"),
        ],
    }
    for label, orders in cases.items():
        try:
            merger.net_across_sleeves(orders)
        except ValueError as exc:
            msg = str(exc)
        else:
            raise AssertionError(f"{label} 必须被拒绝")
        assert "阶段=P2b" in msg, f"{label} 错误缺阶段信息"
        for sleeve in {order.source_sleeve for order in orders}:
            assert sleeve in msg, f"{label} 错误缺 sleeve 集合成员 {sleeve}"


# ══════════════════ A. A1 直通：身份、顺序、副作用 ══════════════════


def test_merge_returns_identical_object_and_preserves_order():
    """对象身份 + 权重映射身份 + 插入顺序（非字典序）+ Weight 值对象身份。"""
    weights = _non_exact_weights()
    target = _target("value", weights)
    _assert_exact_passthrough(SingleSleevePortfolioMerger(), target, _account())
    # 锚定：插入序确实不是字典序（防顺序断言空转）
    assert list(weights) != sorted(weights)


def test_merge_does_not_read_current():
    """D2：merge 不读取 current——爆炸哨兵下照常直通。"""
    target = _target()
    merged = SingleSleevePortfolioMerger().merge({"value": target}, _ExplosiveCurrent())
    assert merged is target


def test_merge_does_not_mutate_inputs():
    """D2：不修改 targets 映射、目标对象、权重映射与当前账户（前后快照比对）。"""
    weights = _non_exact_weights()
    target = _target("value", weights)
    targets = {"value": target}
    account = _account()
    snapshot_items = list(weights.items())

    merged = SingleSleevePortfolioMerger().merge(targets, account)

    assert merged is target
    assert list(targets) == ["value"] and targets["value"] is target
    assert list(weights.items()) == snapshot_items
    assert target.sleeve_id == "value" and target.as_of_date == _D0
    assert account == _account(), "当前账户不得被修改"


def test_merge_non_exact_sum_nan_zero_passthrough():
    """非精确权重和 / NaN / 零值原样透传：不求和、不重归一、不清洗。

    仅证明直通器不改写输入；不构成其作为真实交易合法输入的证明（D2 末段）。
    """
    weights = _non_exact_weights()
    target = _target("value", weights)
    merged = SingleSleevePortfolioMerger().merge({"value": target}, _account())
    assert merged.target_weights is weights
    assert merged.target_weights[_TSC_A].value == 0.9
    assert math.isnan(merged.target_weights[_TSC_C].value)
    assert merged.target_weights[_TSC_D].value == 0.0
    assert abs(sum(w.value for k, w in merged.target_weights.items() if k != _TSC_C) - 1.3) < 1e-12


# ══════════════════ B. A1 越界拒绝（结构防越界） ══════════════════


def test_merge_empty_target_mapping_rejected():
    """空目标映射拒绝（错误含阶段信息与 sleeve 集合）。"""
    with pytest.raises(ValueError) as exc:
        SingleSleevePortfolioMerger().merge({}, _account())
    msg = str(exc.value)
    assert "阶段=P2b" in msg
    assert "sleeve 集合=[]" in msg


def test_merge_multi_target_mapping_rejected():
    """多目标映射拒绝（错误含排序后 sleeve 集合与阶段信息）。"""
    merger = SingleSleevePortfolioMerger()
    _assert_multi_target_rejected(merger)
    targets = {"growth": _target("growth"), "value": _target("value")}
    with pytest.raises(ValueError) as exc:
        merger.merge(targets, _account())
    msg = str(exc.value)
    assert "阶段=P2b" in msg
    assert "['growth', 'value']" in msg  # 排序后 sleeve 集合
    assert "P6.5" in msg


def test_merge_key_sleeve_mismatch_rejected():
    """映射键与 target.sleeve_id 不一致拒绝（错误同时含两者与阶段信息）。"""
    with pytest.raises(ValueError) as exc:
        SingleSleevePortfolioMerger().merge({"value": _target("other")}, _account())
    msg = str(exc.value)
    assert "阶段=P2b" in msg
    assert "'value'" in msg and "'other'" in msg


def test_merge_single_target_empty_weights_passthrough():
    """单目标内 target_weights 为空属直通情形（与空目标映射严格区分），原样透传。"""
    weights: Dict[TSCode, Weight] = {}
    target = _target("value", weights)
    merged = SingleSleevePortfolioMerger().merge({"value": target}, _account())
    assert merged is target
    assert merged.target_weights is weights


# ══════════════════ C. S6 单袖指令透传与混合袖拒绝 ══════════════════


def _assert_net_identity(merger) -> None:
    """同袖多指令透传强等价断言（**调用前快照**基准逐位比对，正例与反变异复用）。

    R4-T1-01 处置：`result is orders` 成立后再比 `result[i] is orders[i]` 是同列表
    自引用、原地重排/替换可穿透；必须对调用前快照逐位断言身份与顺序。
    """
    orders = [
        _order("600000.SH", OrderSide.SELL, "value"),
        _order("000001.SZ", OrderSide.BUY, "value"),
        _order("300750.SZ", OrderSide.SELL, "value"),
    ]
    original = tuple(orders)  # 调用前快照
    result = merger.net_across_sleeves(orders)
    assert result is orders, "必须返回原序列对象"
    assert len(result) == len(original), "指令数不得改变"
    assert all(result[i] is original[i] for i in range(len(original))), "指令对象身份/顺序不得改变"


def test_net_empty_sequence_identity():
    """S6：空指令序列原样返回（list/tuple 两形态均保持对象身份）。"""
    merger = SingleSleevePortfolioMerger()
    for orders in ([], ()):
        assert merger.net_across_sleeves(orders) is orders


def test_net_single_sleeve_multi_orders_identity():
    """S6：同袖多指令返回原序列及原指令对象（不复制、不重排、买/卖混合方向不干预）。"""
    _assert_net_identity(SingleSleevePortfolioMerger())


def test_net_mixed_sleeves_rejected():
    """S6 边界：混合 source_sleeve（反向/同向/自拆）均 fail-fast。"""
    _assert_mixed_sleeves_rejected(SingleSleevePortfolioMerger())


# ══════════════════ D. 协议结构兼容与过渡端口锁定 ══════════════════


def test_protocol_structural_compatibility():
    """PortfolioMerger 结构兼容：runtime_checkable 成员存在 + 完整签名等同。

    签名等同覆盖参数名/种类/注解/参数顺序/返回注解——协议或实现任一侧
    漂移（含注解漂移与参数换序）即报警。
    """
    merger = SingleSleevePortfolioMerger()
    assert isinstance(merger, PortfolioMerger)
    assert _full_signature(SingleSleevePortfolioMerger.merge) == _full_signature(
        PortfolioMerger.merge
    )
    assert _full_signature(SingleSleevePortfolioMerger.net_across_sleeves) == _full_signature(
        PortfolioMerger.net_across_sleeves
    )


def test_legacy_transition_symbols_removed():
    """P4-T3：旧内部端口、dict 适配方法与本文件路径均不再存在。"""
    assert not hasattr(merger_mod, "LegacyTargetMergePort")
    adapter = Path(merger_mod.__file__).parents[2] / "hosts" / "backtest" / "merger_adapter.py"
    assert not adapter.exists()


def test_merge_and_net_call_counts_track_real_calls():
    """计数仅在结构校验后的真实 merge/S6 调用递增，拒绝路径不计数。"""
    merger = SingleSleevePortfolioMerger()
    target = TargetPortfolio("value", {TSCode("A"): Weight(1.0)}, _D0)
    account = Account(Money(0.0), {}, Money(0.0), 1.0)
    assert merger.merge({"value": target}, account) is target
    orders = [
        Order(
            TSCode("A"),
            OrderSide.BUY,
            Shares(100),
            OrderType.MARKET_OPEN,
            OrderReason.REBALANCE,
            "value",
        )
    ]
    assert merger.net_across_sleeves(orders) is orders
    assert (merger.merge_call_count, merger.net_call_count) == (1, 1)
    with pytest.raises(ValueError):
        merger.merge({}, account)
    assert merger.merge_call_count == 1


def _hosts_import_offenders(src: str) -> list:
    """AST 扫描源码中的 hosts 包导入（轻量锁，T3 门 2 AST 工具上线前的过渡件）。

    按路径段 ``hosts`` 判定（不做子串匹配，``hostsfile`` 等合法名不误伤）；
    覆盖四类形态：完整路径 import/from、父包导入（``from src.lazybull.v2
    import hosts``）、裸导入（``import hosts``）与相对导入（``from ... import
    hosts`` / ``from ...hosts.backtest import runtime``）。
    """
    offenders = []
    tree = ast.parse(src)

    def _has_hosts_segment(dotted: str) -> bool:
        return "hosts" in dotted.split(".")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _has_hosts_segment(alias.name):
                    offenders.append(f"L{node.lineno} import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _has_hosts_segment(module):
                offenders.append(f"L{node.lineno} from {module} import ...")
            for alias in node.names:
                if alias.name == "hosts" or alias.name.startswith("hosts."):
                    offenders.append(f"L{node.lineno} from {module!r} import {alias.name}")
    return offenders


def test_core_module_does_not_import_hosts():
    """F11 §4.10 依赖方向：core 模块不得 import hosts（轻量锁，覆盖四类导入形态）。

    残余边界（如实登记）：仅锁本模块直接 import；跨模块间接引入、动态 import
    与全 core 面扫描由 T3 门 2 AST 工具承担（规划 D9/门 2）。
    """
    offenders = _hosts_import_offenders(Path(merger_mod.__file__).read_text(encoding="utf-8"))
    assert offenders == [], f"core 不得 import hosts: {offenders}"


def test_hosts_import_scanner_catches_escape_forms():
    """hosts 导入扫描反例（reviewer2 R2-02 处置）：四类逃逸形态必捕获、合法导入不误报。"""
    escapes = {
        "完整路径 import": "import src.lazybull.v2.hosts.backtest.runtime\n",
        "完整路径 from": "from src.lazybull.v2.hosts.backtest import runtime\n",
        "父包导入": "from src.lazybull.v2 import hosts\n",
        "裸导入": "import hosts\n",
        "相对导入别名": "from ... import hosts\n",
        "相对导入子模块": "from ...hosts.backtest import runtime\n",
    }
    for label, src in escapes.items():
        assert _hosts_import_offenders(src), f"逃逸形态未捕获: {label}"
    negatives = {
        "合法相对导入": "from ..common import types\n",
        "合法父包导入": "from src.lazybull.v2.common import types\n",
        "名含 hosts 子串但非路径段": "import hostsfile\n",
        "普通第三方导入": "import pandas as pd\n",
    }
    for label, src in negatives.items():
        assert _hosts_import_offenders(src) == [], f"负例误报: {label}"


# ══════════════════ E. 反变异自证 ══════════════════


class _ReorderMutant(SingleSleevePortfolioMerger):
    """原地重排权重映射（保留对象身份，仅破坏插入顺序）。"""

    def merge(self, targets, current):
        target = next(iter(targets.values()))
        weights = target.target_weights
        reordered = dict(sorted(weights.items(), key=lambda kv: kv[0].value))
        weights.clear()
        weights.update(reordered)
        return target


class _FloatShiftMutant(SingleSleevePortfolioMerger):
    """原地改浮点（逐键替换为偏移 1e-15 的新 Weight）。"""

    def merge(self, targets, current):
        target = next(iter(targets.values()))
        weights = target.target_weights
        for code in list(weights):
            weights[code] = Weight(weights[code].value + 1e-15)
        return target


class _RenormalizeMutant(SingleSleevePortfolioMerger):
    """偷偷归一化（对 Σ≠1 输入顺手求和重归一）。"""

    def merge(self, targets, current):
        target = next(iter(targets.values()))
        weights = target.target_weights
        total = sum(w.value for w in weights.values())
        for code in list(weights):
            weights[code] = Weight(weights[code].value / total)
        return target


class _CopyMutant(SingleSleevePortfolioMerger):
    """返回新构造的 TargetPortfolio（破坏对象身份与映射身份）。"""

    def merge(self, targets, current):
        target = next(iter(targets.values()))
        return TargetPortfolio(
            sleeve_id=target.sleeve_id,
            target_weights=dict(target.target_weights),
            as_of_date=target.as_of_date,
        )


class _CurrentReadMutant(SingleSleevePortfolioMerger):
    """违规读取 current（D2 禁止）。"""

    def merge(self, targets, current):
        _ = current.total_value
        return next(iter(targets.values()))


class _BypassMultiMutant(SingleSleevePortfolioMerger):
    """多目标放行（绕过结构防越界检查，直接取首个）。"""

    def merge(self, targets, current):
        return next(iter(targets.values()))


class _LenientNetMutant(SingleSleevePortfolioMerger):
    """混合袖静默放行且返回副本（绕过 S6 fail-fast + 破坏序列身份）。"""

    def net_across_sleeves(self, orders):
        return list(orders)


class _ReverseNetMutant(SingleSleevePortfolioMerger):
    """原地反转指令序列后返回原列表（R4-T1-01：同列表自引用身份断言无法捕获的变异）。"""

    def net_across_sleeves(self, orders):
        orders.reverse()
        return orders


def test_anti_mutation_passthrough_variants_detected():
    """直通面四类变异（重排/改浮点/归一化/复制）必须被强等价断言组捕获。

    生效锚定为逐变异具体效果观察（R2-01 处置：旧版对独立构造参照做身份
    比较恒真、锚定空转；现改为按变异类别观察其特有破坏）。
    """
    anchors = {
        # 重排：插入序被改写为非原序（原序 = _non_exact_weights 的 A,B,C,D）
        _ReorderMutant: lambda weights, result, target: list(weights)
        != [_TSC_A, _TSC_B, _TSC_C, _TSC_D],
        # 改浮点：A 权重值被偏移（0.9+1e-15 ≠ 0.9）
        _FloatShiftMutant: lambda weights, result, target: weights[_TSC_A].value != 0.9,
        # 归一化：Σ 含 NaN 时 A 权重被改写为 NaN（≠ 0.9）
        _RenormalizeMutant: lambda weights, result, target: weights[_TSC_A].value != 0.9,
        # 复制：返回对象不再是原目标
        _CopyMutant: lambda weights, result, target: result is not target,
    }
    for mutant_cls, anchor in anchors.items():
        # 锚定：变异确实生效（防变异空转的假通过）
        probe_weights = _non_exact_weights()
        probe_target = _target("value", probe_weights)
        probe_result = mutant_cls().merge({"value": probe_target}, _account())
        assert anchor(
            probe_weights, probe_result, probe_target
        ), f"{mutant_cls.__name__} 变异未生效"
        # 强等价断言组必须拒绝
        with pytest.raises(AssertionError):
            _assert_exact_passthrough(
                mutant_cls(), _target("value", _non_exact_weights()), _account()
            )


def test_anti_mutation_current_read_detected():
    """读取 current 变异：爆炸哨兵下必须抛 RuntimeError（正例不抛由 A 组锁定）。"""
    with pytest.raises(RuntimeError):
        _CurrentReadMutant().merge({"value": _target()}, _ExplosiveCurrent())


def test_anti_mutation_boundary_leniency_detected():
    """越界放行两类变异（多目标放行/混合袖放行）必须被对应拒绝断言捕获。"""
    assert not _reject_check(
        lambda: _BypassMultiMutant().merge(
            {"growth": _target("growth"), "value": _target("value")}, _account()
        )
    ), "多目标放行变异未被识别"
    with pytest.raises(AssertionError):
        _assert_multi_target_rejected(_BypassMultiMutant())
    with pytest.raises(AssertionError):
        _assert_mixed_sleeves_rejected(_LenientNetMutant())
    # 序列身份变异（返回副本）同样被 S6 身份断言捕获
    orders = [_order("600000.SH", OrderSide.BUY, "value")]
    assert _LenientNetMutant().net_across_sleeves(orders) is not orders


def test_anti_mutation_net_inplace_reverse_detected():
    """原地反转并返回原列表的变异必须被快照基准断言捕获（R4-T1-01 复收条件）。"""
    # 锚定：变异确实生效（原序列被原地反转，首位置对象改变）
    probe = [
        _order("600000.SH", OrderSide.SELL, "value"),
        _order("000001.SZ", OrderSide.BUY, "value"),
    ]
    first = probe[0]
    result = _ReverseNetMutant().net_across_sleeves(probe)
    assert result is probe and result[0] is not first, "反转变异未生效"
    # 快照基准的强等价断言必须拒绝
    with pytest.raises(AssertionError):
        _assert_net_identity(_ReverseNetMutant())
