# -*- coding: utf-8 -*-
"""P2a T2 协议层补齐验收测试（契约 docs/contracts/protocols.md §2~§4）。

验收口径 = 类型检查，证据分三类（R1-T2-01 / R2-T2-01 处置，2026-10-07）：
1. 成员存在性：runtime_checkable isinstance（只证明成员存在，不检签名）；
2. 完整签名锁定：get_type_hints 解析注解（参数类型/返回类型/参数种类/默认值/
   属性返回类型/**参数顺序**）逐项对照契约预期表——对源码级漂移敏感（含注解
   漂移与参数换序，TestSourceMutationRegression）；
3. 实现侧签名等同校验：带完整注解的合成 stub 全部 18 项方法接口通过等同比对；
   缺参数 / 错误参数类型 / 错误返回类型 / **参数换序**四类反例被拒绝。

参数顺序是签名的一部分（R2-T2-R2-01：协议方法允许按位置传参，dict 相等
不比较插入顺序 ⇒ 规范化签名必须保留有序参数序列）。

口径边界（如实登记）：等同比对 ≠ mypy 级别子类型宽窄化兼容——协议锁定的
目的是防抄写漂移与接口走样，等同是最强且无歧义的口径；mypy strict +
import-linter 属契约 §9 落档清单欠账，另行登记。

全部输入为合成 stub，不依赖真实配置与真实数据。
"""

import ast
import inspect
from pathlib import Path
from typing import Mapping, Sequence, get_type_hints

import pandas as pd

from src.lazybull.v2.common.protocols import (
    BacktestHost,
    DecisionMaker,
    Executor,
    FundScheduler,
    Ledger,
    PaperHost,
    PortfolioMerger,
    Signal,
    Trainer,
    WalkForwardOrchestrator,
)
from src.lazybull.v2.common.protocols.store import DataStore
from src.lazybull.v2.common.types import (
    Account,
    ExecutionContext,
    Fill,
    ModelConfig,
    Money,
    Order,
    PanelFrame,
    RankedCandidates,
    TargetPortfolio,
    TradeDate,
    VirtualAccount,
)

_PROTOCOLS_DIR = (
    Path(__file__).parents[1] / "src" / "lazybull" / "v2" / "common" / "protocols"
)

_POK = inspect.Parameter.POSITIONAL_OR_KEYWORD
_EMPTY = inspect.Parameter.empty
_NONE_TYPE = type(None)
# 协议方法的 self 项：位置或关键字传参、无默认值、无注解
_SELF = ("self", _POK, _EMPTY, None)


def _full_signature(func) -> tuple:
    """完整有序签名解析：(参数描述有序元组, 返回注解)。

    每个参数描述 = (name, kind, default, annotation)，以有序元组序列返回——
    参数顺序是签名的一部分（按位置传参语义），换序 ⇒ 元组不等。
    get_type_hints 解析 `from __future__ import annotations` 下的字符串注解；
    注解以类型对象比对（typing 泛型 / UnionType 实现了 __eq__）。
    """
    hints = get_type_hints(func)
    sig = inspect.signature(func)
    params = tuple(
        (name, param.kind, param.default, hints.get(name))
        for name, param in sig.parameters.items()
    )
    return params, hints.get("return")


def _annotation_compatible(stub_func, proto_func) -> bool:
    """stub 与协议方法的完整有序签名等同比对（含参数顺序）。"""
    return _full_signature(stub_func) == _full_signature(proto_func)


class TestProtocolSurface:
    """12 协议落位与 __init__ 导出（成员存在性证据之一）。"""

    def test_init_all_exports(self):
        import src.lazybull.v2.common.protocols as pkg

        assert sorted(pkg.__all__) == sorted(
            [
                "BacktestHost",
                "DataStore",
                "DecisionMaker",
                "Executor",
                "FeatureBuilder",
                "FundScheduler",
                "Ledger",
                "PaperHost",
                "PortfolioMerger",
                "Signal",
                "Trainer",
                "WalkForwardOrchestrator",
            ]
        )
        for name in pkg.__all__:
            assert getattr(pkg, name) is not None

    def test_protocol_classes_are_runtime_checkable(self):
        for proto in (
            Signal,
            Trainer,
            WalkForwardOrchestrator,
            DecisionMaker,
            Executor,
            Ledger,
            FundScheduler,
            PortfolioMerger,
            BacktestHost,
            PaperHost,
        ):
            assert getattr(proto, "_is_runtime_protocol", False), proto.__name__


def _property_hints(proto, name: str) -> dict:
    """property 返回类型解析（get_type_hints 作用于 fget）。"""
    prop = inspect.getattr_static(proto, name)
    return get_type_hints(prop.fget)


class TestFullSignatureLock:
    """完整签名锁定：契约 §2~§4 的 20 项接口（18 方法 + 2 属性）逐项对照预期表。

    预期表从契约文本独立转录（不回读协议源码），参数以有序元组表达——
    协议文件的注解 / 参数 / 默认值 / 参数顺序发生任何漂移都会在此失败。
    """

    # ---- §2.1 Signal ----

    def test_signal_properties(self):
        assert isinstance(inspect.getattr_static(Signal, "sleeve_id"), property)
        assert isinstance(
            inspect.getattr_static(Signal, "holding_period_days"), property
        )
        assert _property_hints(Signal, "sleeve_id")["return"] is str
        assert _property_hints(Signal, "holding_period_days")["return"] is int

    def test_signal_generate_ranked(self):
        assert _full_signature(Signal.generate_ranked) == (
            (
                _SELF,
                ("date", _POK, _EMPTY, TradeDate),
                ("features", _POK, _EMPTY, PanelFrame),
                ("context", _POK, _EMPTY, ExecutionContext),
            ),
            RankedCandidates,
        )

    # ---- §2.2 Trainer / WalkForwardOrchestrator ----

    def test_trainer_train_fold(self):
        assert _full_signature(Trainer.train_fold) == (
            (
                _SELF,
                ("config", _POK, _EMPTY, ModelConfig),
                ("train_start", _POK, _EMPTY, TradeDate),
                ("train_end", _POK, _EMPTY, TradeDate),
                ("val_start", _POK, _EMPTY, TradeDate),
                ("val_end", _POK, _EMPTY, TradeDate),
                ("store", _POK, _EMPTY, DataStore),
            ),
            str,
        )

    def test_walk_forward_orchestrator_run(self):
        assert _full_signature(WalkForwardOrchestrator.run) == (
            (
                _SELF,
                ("config", _POK, _EMPTY, ModelConfig),
                ("split_count", _POK, _EMPTY, int),
                ("final_date", _POK, _EMPTY, TradeDate),
                ("trainer", _POK, _EMPTY, Trainer),
                ("store", _POK, _EMPTY, DataStore),
            ),
            str,
        )

    # ---- §3.1 DecisionMaker ----

    def test_decision_maker(self):
        assert _full_signature(DecisionMaker.decide) == (
            (
                _SELF,
                ("date", _POK, _EMPTY, TradeDate),
                ("merged_target", _POK, _EMPTY, TargetPortfolio),
                ("current_account", _POK, _EMPTY, Account),
                ("market_state", _POK, _EMPTY, pd.DataFrame),
                ("context", _POK, _EMPTY, ExecutionContext),
            ),
            Sequence[Order],
        )
        assert _full_signature(DecisionMaker.apply_constraints) == (
            (
                _SELF,
                ("target", _POK, _EMPTY, TargetPortfolio),
                ("current", _POK, _EMPTY, Account),
            ),
            TargetPortfolio,
        )

    # ---- §3.2 Executor ----

    def test_executor_execute_daily(self):
        assert _full_signature(Executor.execute_daily) == (
            (
                _SELF,
                ("date", _POK, _EMPTY, TradeDate),
                ("orders", _POK, _EMPTY, Sequence[Order]),
                ("account", _POK, _EMPTY, Account),
                ("context", _POK, _EMPTY, ExecutionContext),
            ),
            Sequence[Fill],
        )

    # ---- §3.3 Ledger ----

    def test_ledger(self):
        assert _full_signature(Ledger.get_account) == ((_SELF,), Account)
        assert _full_signature(Ledger.get_virtual_account) == (
            (
                _SELF,
                ("sleeve_id", _POK, _EMPTY, str),
            ),
            VirtualAccount,
        )
        assert _full_signature(Ledger.apply_fills) == (
            (
                _SELF,
                ("fills", _POK, _EMPTY, Sequence[Fill]),
            ),
            _NONE_TYPE,
        )
        assert _full_signature(Ledger.snapshot) == (
            (
                _SELF,
                ("date", _POK, _EMPTY, TradeDate),
            ),
            _NONE_TYPE,
        )

    # ---- §3.4 FundScheduler / PortfolioMerger ----

    def test_fund_scheduler(self):
        assert _full_signature(FundScheduler.available_budget) == (
            (
                _SELF,
                ("sleeve_id", _POK, _EMPTY, str),
                ("date", _POK, _EMPTY, TradeDate),
                ("virtual_accounts", _POK, _EMPTY, Mapping[str, VirtualAccount]),
            ),
            Money,
        )
        assert _full_signature(FundScheduler.record_borrowing) == (
            (
                _SELF,
                ("borrower", _POK, _EMPTY, str),
                ("lender", _POK, _EMPTY, str),
                ("amount", _POK, _EMPTY, Money),
                ("date", _POK, _EMPTY, TradeDate),
            ),
            _NONE_TYPE,
        )
        assert _full_signature(FundScheduler.settle_due) == (
            (
                _SELF,
                ("date", _POK, _EMPTY, TradeDate),
                ("virtual_accounts", _POK, _EMPTY, Mapping[str, VirtualAccount]),
            ),
            Sequence[Order],
        )

    def test_portfolio_merger(self):
        assert _full_signature(PortfolioMerger.merge) == (
            (
                _SELF,
                ("targets", _POK, _EMPTY, Mapping[str, TargetPortfolio]),
                ("current", _POK, _EMPTY, Account),
            ),
            TargetPortfolio,
        )
        assert _full_signature(PortfolioMerger.net_across_sleeves) == (
            (
                _SELF,
                ("orders", _POK, _EMPTY, Sequence[Order]),
            ),
            Sequence[Order],
        )

    # ---- §4 宿主层 ----

    def test_hosts(self):
        assert _full_signature(BacktestHost.run) == (
            (
                _SELF,
                ("start_date", _POK, _EMPTY, TradeDate),
                ("end_date", _POK, _EMPTY, TradeDate),
                ("sleeves", _POK, _EMPTY, Mapping[str, Signal]),
                ("merger", _POK, _EMPTY, PortfolioMerger),
                ("decision_maker", _POK, _EMPTY, DecisionMaker),
                ("executor", _POK, _EMPTY, Executor),
                ("ledger", _POK, _EMPTY, Ledger),
                ("store", _POK, _EMPTY, DataStore),
            ),
            _NONE_TYPE,
        )
        assert _full_signature(PaperHost.run_t0) == (
            (
                _SELF,
                ("date", _POK, _EMPTY, TradeDate),
            ),
            Sequence[Order],
        )
        assert _full_signature(PaperHost.run_t1) == (
            (
                _SELF,
                ("date", _POK, _EMPTY, TradeDate),
                ("fills", _POK, _EMPTY, Sequence[Fill]),
            ),
            _NONE_TYPE,
        )


# ========== 合成 stub（完整注解，正例） ==========


class _StubSignal:
    @property
    def sleeve_id(self) -> str:
        return "stub_sleeve"

    @property
    def holding_period_days(self) -> int:
        return 20

    def generate_ranked(
        self,
        date: TradeDate,
        features: PanelFrame,
        context: ExecutionContext,
    ) -> RankedCandidates:
        raise NotImplementedError


class _StubTrainer:
    def train_fold(
        self,
        config: ModelConfig,
        train_start: TradeDate,
        train_end: TradeDate,
        val_start: TradeDate,
        val_end: TradeDate,
        store: DataStore,
    ) -> str:
        return "v1"


class _StubOrchestrator:
    def run(
        self,
        config: ModelConfig,
        split_count: int,
        final_date: TradeDate,
        trainer: Trainer,
        store: DataStore,
    ) -> str:
        return "wf_run_1"


class _StubDecisionMaker:
    def decide(
        self,
        date: TradeDate,
        merged_target: TargetPortfolio,
        current_account: Account,
        market_state: pd.DataFrame,
        context: ExecutionContext,
    ) -> Sequence[Order]:
        return []

    def apply_constraints(
        self,
        target: TargetPortfolio,
        current: Account,
    ) -> TargetPortfolio:
        return target


class _StubExecutor:
    def execute_daily(
        self,
        date: TradeDate,
        orders: Sequence[Order],
        account: Account,
        context: ExecutionContext,
    ) -> Sequence[Fill]:
        return []


class _StubLedger:
    def get_account(self) -> Account:
        raise NotImplementedError

    def get_virtual_account(self, sleeve_id: str) -> VirtualAccount:
        raise NotImplementedError

    def apply_fills(self, fills: Sequence[Fill]) -> None:
        return None

    def snapshot(self, date: TradeDate) -> None:
        return None


class _StubFundScheduler:
    def available_budget(
        self,
        sleeve_id: str,
        date: TradeDate,
        virtual_accounts: Mapping[str, VirtualAccount],
    ) -> Money:
        return Money(0.0)

    def record_borrowing(
        self,
        borrower: str,
        lender: str,
        amount: Money,
        date: TradeDate,
    ) -> None:
        return None

    def settle_due(
        self,
        date: TradeDate,
        virtual_accounts: Mapping[str, VirtualAccount],
    ) -> Sequence[Order]:
        return []


class _StubMerger:
    def merge(
        self,
        targets: Mapping[str, TargetPortfolio],
        current: Account,
    ) -> TargetPortfolio:
        raise NotImplementedError

    def net_across_sleeves(self, orders: Sequence[Order]) -> Sequence[Order]:
        return []


class _StubBacktestHost:
    def run(
        self,
        start_date: TradeDate,
        end_date: TradeDate,
        sleeves: Mapping[str, Signal],
        merger: PortfolioMerger,
        decision_maker: DecisionMaker,
        executor: Executor,
        ledger: Ledger,
        store: DataStore,
    ) -> None:
        return None


class _StubPaperHost:
    def run_t0(self, date: TradeDate) -> Sequence[Order]:
        return []

    def run_t1(self, date: TradeDate, fills: Sequence[Fill]) -> None:
        return None


# ========== 合成 stub（反例：签名漂移） ==========


class _WrongSignalMissingParam:
    """反例：缺参数（context 缺失）。"""

    def generate_ranked(
        self,
        date: TradeDate,
        features: PanelFrame,
    ) -> RankedCandidates:
        raise NotImplementedError


class _WrongSignalParamType:
    """反例：错误参数类型（features: str）。"""

    def generate_ranked(
        self,
        date: TradeDate,
        features: str,
        context: ExecutionContext,
    ) -> RankedCandidates:
        raise NotImplementedError


class _WrongSignalReturnType:
    """反例：错误返回类型（-> None）。"""

    def generate_ranked(
        self,
        date: TradeDate,
        features: PanelFrame,
        context: ExecutionContext,
    ) -> None:
        return None


class _ReorderedSignal:
    """反例：参数换序（features 先于 date，类型跟随各自参数）——
    按位置传参会把交易日绑定到 features，调用立即失败。"""

    def generate_ranked(
        self,
        features: PanelFrame,
        date: TradeDate,
        context: ExecutionContext,
    ) -> RankedCandidates:
        raise NotImplementedError


class _WrongPaperHostMissingParam:
    """反例：缺参数（date 缺失）。"""

    def run_t0(self) -> Sequence[Order]:
        return []


class _WrongPaperHostParamType:
    """反例：错误参数类型（date: str）。"""

    def run_t0(self, date: str) -> Sequence[Order]:
        return []


class _WrongPaperHostReturnType:
    """反例：错误返回类型（-> None）。"""

    def run_t0(self, date: TradeDate) -> None:
        return None


class _ReorderedPaperHost:
    """反例：参数换序（fills 先于 date）。"""

    def run_t1(self, fills: Sequence[Fill], date: TradeDate) -> None:
        return None


class TestAnnotationCompat:
    """实现侧签名等同校验：正例 18 项方法接口全覆盖 + 四类反例被拒绝。"""

    _POSITIVE_CASES = [
        (_StubSignal.generate_ranked, Signal.generate_ranked),
        (_StubTrainer.train_fold, Trainer.train_fold),
        (_StubOrchestrator.run, WalkForwardOrchestrator.run),
        (_StubDecisionMaker.decide, DecisionMaker.decide),
        (_StubDecisionMaker.apply_constraints, DecisionMaker.apply_constraints),
        (_StubExecutor.execute_daily, Executor.execute_daily),
        (_StubLedger.get_account, Ledger.get_account),
        (_StubLedger.get_virtual_account, Ledger.get_virtual_account),
        (_StubLedger.apply_fills, Ledger.apply_fills),
        (_StubLedger.snapshot, Ledger.snapshot),
        (_StubFundScheduler.available_budget, FundScheduler.available_budget),
        (_StubFundScheduler.record_borrowing, FundScheduler.record_borrowing),
        (_StubFundScheduler.settle_due, FundScheduler.settle_due),
        (_StubMerger.merge, PortfolioMerger.merge),
        (_StubMerger.net_across_sleeves, PortfolioMerger.net_across_sleeves),
        (_StubBacktestHost.run, BacktestHost.run),
        (_StubPaperHost.run_t0, PaperHost.run_t0),
        (_StubPaperHost.run_t1, PaperHost.run_t1),
    ]

    def test_positive_stubs_signature_identical(self):
        """正例：18 方法全部有序签名等同（含 2 属性返回类型，见 TestFullSignatureLock）。"""
        assert len(self._POSITIVE_CASES) == 18
        for stub_func, proto_func in self._POSITIVE_CASES:
            assert _annotation_compatible(stub_func, proto_func), (
                f"{stub_func.__qualname__} 与协议签名不等同"
            )

    def test_negative_missing_param_rejected(self):
        assert not _annotation_compatible(
            _WrongSignalMissingParam.generate_ranked, Signal.generate_ranked
        )
        assert not _annotation_compatible(
            _WrongPaperHostMissingParam.run_t0, PaperHost.run_t0
        )

    def test_negative_param_type_rejected(self):
        assert not _annotation_compatible(
            _WrongSignalParamType.generate_ranked, Signal.generate_ranked
        )
        assert not _annotation_compatible(
            _WrongPaperHostParamType.run_t0, PaperHost.run_t0
        )

    def test_negative_return_type_rejected(self):
        assert not _annotation_compatible(
            _WrongSignalReturnType.generate_ranked, Signal.generate_ranked
        )
        assert not _annotation_compatible(
            _WrongPaperHostReturnType.run_t0, PaperHost.run_t0
        )

    def test_negative_param_order_rejected(self):
        """参数换序必须被公共等同比对拒绝（R2-T2-R2-01）——类型各自正确也不行。"""
        assert not _annotation_compatible(
            _ReorderedSignal.generate_ranked, Signal.generate_ranked
        )
        assert not _annotation_compatible(
            _ReorderedPaperHost.run_t1, PaperHost.run_t1
        )


class TestSourceMutationRegression:
    """协议源码漂移回归（R2-T2-R2-01 复收条件 1）。

    对协议源码做文本级变异（不修改文件、内存 exec 变异副本），
    公共锁定器必须对注解漂移与参数换序都失败。
    """

    @staticmethod
    def _mutated_protocol_method(source: str, mutations: list):
        """读协议源码 → 应用 (old, new) 替换序列 → exec → 返回变异后的 Signal 类。"""
        for old, new in mutations:
            source = source.replace(old, new)
        namespace: dict = {}
        exec(compile(source, "<mutated protocol>", "exec"), namespace)  # noqa: S102
        return namespace

    def test_signal_param_order_mutation_detected(self):
        """Signal.generate_ranked 源码参数换序（date/features 互换）⇒ 锁定失败。"""
        src = (_PROTOCOLS_DIR / "signal.py").read_text(encoding="utf-8")
        old_block = "        date: TradeDate,\n        features: PanelFrame,\n"
        new_block = "        features: PanelFrame,\n        date: TradeDate,\n"
        assert old_block in src, "源码参数块格式与预期不符，需同步更新本变异测试"
        mutated = self._mutated_protocol_method(src, [(old_block, new_block)])
        mutated_sig = _full_signature(mutated["Signal"].generate_ranked)
        assert mutated_sig != _full_signature(Signal.generate_ranked)

    def test_hosts_param_order_mutation_detected(self):
        """PaperHost.run_t1 源码参数换序（date/fills 互换）⇒ 锁定失败。"""
        src = (_PROTOCOLS_DIR / "hosts.py").read_text(encoding="utf-8")
        old_block = (
            "    def run_t1(self, date: TradeDate, fills: Sequence[Fill]) -> None:"
        )
        new_block = (
            "    def run_t1(self, fills: Sequence[Fill], date: TradeDate) -> None:"
        )
        assert old_block in src, "源码签名行格式与预期不符，需同步更新本变异测试"
        mutated = self._mutated_protocol_method(src, [(old_block, new_block)])
        mutated_sig = _full_signature(mutated["PaperHost"].run_t1)
        assert mutated_sig != _full_signature(PaperHost.run_t1)

    def test_signal_annotation_mutation_detected(self):
        """Signal.generate_ranked 注解漂移（features: str）⇒ 锁定失败。"""
        src = (_PROTOCOLS_DIR / "signal.py").read_text(encoding="utf-8")
        assert "features: PanelFrame" in src
        mutated = self._mutated_protocol_method(
            src, [("features: PanelFrame", "features: str")]
        )
        mutated_sig = _full_signature(mutated["Signal"].generate_ranked)
        assert mutated_sig != _full_signature(Signal.generate_ranked)


class TestStructuralSubtyping:
    """成员存在性证据：runtime_checkable isinstance 只查成员存在、不检签名——
    其类型校验局限由 TestFullSignatureLock / TestAnnotationCompat 兜住。"""

    def test_stubs_satisfy_protocols(self):
        assert isinstance(_StubSignal(), Signal)
        assert isinstance(_StubTrainer(), Trainer)
        assert isinstance(_StubOrchestrator(), WalkForwardOrchestrator)
        assert isinstance(_StubDecisionMaker(), DecisionMaker)
        assert isinstance(_StubExecutor(), Executor)
        assert isinstance(_StubLedger(), Ledger)
        assert isinstance(_StubFundScheduler(), FundScheduler)
        assert isinstance(_StubMerger(), PortfolioMerger)
        assert isinstance(_StubBacktestHost(), BacktestHost)
        assert isinstance(_StubPaperHost(), PaperHost)

    def test_missing_method_fails_check(self):
        class _NoRanked:
            sleeve_id = "x"
            holding_period_days = 20

        assert not isinstance(_NoRanked(), Signal)

        class _NoApplyFills:
            def get_account(self):
                return None

            def get_virtual_account(self, sleeve_id):
                return None

            def snapshot(self, date):
                return None

        assert not isinstance(_NoApplyFills(), Ledger)


class TestDependencyDirection:
    """契约 §7：协议层只依赖 common.types + 协议包内 + 标准库 / pandas。"""

    _ALLOWED_ROOTS = (
        "src.lazybull.v2.common.types",
        "src.lazybull.v2.common.protocols",
    )
    _ALLOWED_STDLIB_OR_THIRD = (
        "pandas",
        "typing",
        "__future__",
        "dataclasses",
        "datetime",
        "enum",
    )

    def test_no_implementation_side_imports(self):
        files = sorted(_PROTOCOLS_DIR.glob("*.py"))
        expected = {"__init__.py", "store.py", "signal.py", "core.py", "hosts.py"}
        assert {f.name for f in files} >= expected
        for f in files:
            tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
            for node in ast.walk(tree):
                targets = []
                if isinstance(node, ast.Import):
                    targets = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    targets = [node.module]
                for mod in targets:
                    self._check_module(f.name, mod)

    def _check_module(self, filename: str, mod: str) -> None:
        if mod.startswith("src.lazybull"):
            # 精确段匹配：root 或 root.<子模块>，防 types_evil 类前缀放行（O2）
            allowed = any(
                mod == root or mod.startswith(root + ".")
                for root in self._ALLOWED_ROOTS
            )
            assert allowed, f"{filename}: 协议层禁止依赖实现侧模块（{mod}）"
        else:
            root = mod.split(".")[0]
            assert root in self._ALLOWED_STDLIB_OR_THIRD, (
                f"{filename}: 协议层依赖白名单外模块（{mod}）"
            )


class TestF8ErratumBoundaryText:
    """F8 勘误（随方案 F11 同批生效）在 core.py 边界声明的落实。"""

    def test_core_boundary_no_take_profit_hook(self):
        """勘误 1/3：钩子括注不得含止盈语义；钩子序列以代码调用点枚举为准。"""
        text = (_PROTOCOLS_DIR / "core.py").read_text(encoding="utf-8")
        module_doc = ast.get_docstring(ast.parse(text)) or ""
        # 旧文本「到期 / 止损 / 止盈 / …」钩子括注不得复活
        assert "止损 / 止盈" not in module_doc
        assert "黄金调用点清单" in module_doc, "勘误 3：钩子序列以代码调用点枚举为准（T5 固化）"
        assert "F8 勘误" in module_doc

    def test_trim_replenish_skip_queue_registered(self):
        """延迟订单最小语义：减仓 / 回补不进延迟队列特例必须登记在协议层。"""
        text = (_PROTOCOLS_DIR / "core.py").read_text(encoding="utf-8")
        assert "不进延迟队列" in text
