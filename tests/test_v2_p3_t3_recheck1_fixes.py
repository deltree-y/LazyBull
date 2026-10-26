# -*- coding: utf-8 -*-
"""P3-T3 复核轮(2026-10-22)测试缺口补齐。

覆盖条目(括号 = 复核意见编号):
1. TestOosGuardPassthrough —— reviewer1 复核 P1-1 负例:``ReplayContractError``
   穿透 split 级容错(不得吞成 bt_metrics={});``_frozen_through`` 收紧到测试窗前
   ⇒ run 期拒绝;一般 ``RuntimeError`` ⇒ 容错语义保持(bt_metrics={} 且 run 完成);
2. TestPanelFamilySwitchRejected —— P2-2 负例:panel 模式五族开关任一开 ⇒
   run 期 ValueError(物化直读接线未完成);全关 ⇒ 放行;
3. TestDomainConsistency —— P2-7 负例:训练域 ≠ 宿主域 ⇒ run 期 ValueError;
4. TestNormalizeSelectedSplitIndices —— P2-8:split 下标归一化单元(去重保序 /
   负值拒绝 / 空入参);
5. TestEvalFamilyInjectionPrecheck —— codex-F04:cs_train_verify 模式族启用而
   派生注入不齐 ⇒ 训练注册前 ValueError(trainer 协议入口零调用佐证);
6. TestReplayGuardPermanent —— codex-F03 末节(R06):``_assert_replay_cs_train_reachable``
   正反例固化(tests/ 放行;purpose 非法 / 仓外调用方 / 生产脚本调用方 ⇒ 拒绝);
7. TestManifestRegistrationDrift —— codex-F01 真实负例:readiness 核验后、
   load_features 读取前篡改 core 分区(manifest 登记保持旧指纹)⇒
   ``load_training_frame`` 抛 InputIdentityDriftError(manifest 比对层检出);
8. TestExtrasEnterDataStateId —— reviewer1 复核 P2-1:extra_sections 入增强
   data_state_id(指纹不同 ⇒ ID 不同;None 与 {} 同 ID);summary 摘要列与 JSON
   内 id 同口径;
9. TestExtraSectionsIsolation —— reviewer1 复核 P2-3:附加段逐段独立 try,
   水位解析失败不连坐 label_sealed_through / manifest_fingerprint / code_state;
   全失败 ⇒ 各段缺失且无 KeyError、不上抛。

夹具复用 tests/test_v2_train_wf_orchestrator_e2e.py 的合成数据构造器——经
importlib 加载为共享夹具模块(pytest 同目录 import 依赖收集次序,不采用)。
全部 tmp_path 合成数据;不触碰真实生产数据。
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

from src.lazybull.v2.common.model_registry import ModelRegistry
from src.lazybull.v2.common.types import ModelConfig, ReplayContractError, TradeDate
from src.lazybull.v2.store.column_groups import LABEL_TABLES
from src.lazybull.v2.store.data_store import PanelDataStore
from src.lazybull.v2.store.store_identity import InputIdentityDriftError
from src.lazybull.v2.train.data_adapter import (
    SOURCE_CS_TRAIN_VERIFY,
    SOURCE_PANEL,
    AdapterConfig,
    load_training_frame,
)
from src.lazybull.v2.train.frame_cache import TrainingFrameCache
from src.lazybull.v2.train.trainer import V2Trainer
from src.lazybull.v2.train.walk_forward.backtest import BacktestHostParams
from src.lazybull.v2.train.walk_forward.data_state import (
    compute_enhanced_data_state_id,
    write_enhanced_data_state_file,
)
from src.lazybull.v2.train.walk_forward.identity_collect import (
    collect_identity_extra_sections,
)
from src.lazybull.v2.train.walk_forward.orchestrator import (
    OrchestratorHostConfig,
    V2WalkForwardOrchestrator,
    _normalize_selected_split_indices,
)

# ---------------------------------------------------------------------------
# 共享夹具:importlib 加载 e2e 文件模块(避免 pytest 同目录 import 次序依赖)
# ---------------------------------------------------------------------------


def _load_e2e_fixtures():
    path = Path(__file__).with_name("test_v2_train_wf_orchestrator_e2e.py")
    spec = importlib.util.spec_from_file_location("p3_t3_recheck1_e2e_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e2e = _load_e2e_fixtures()


class _SentinelHalt(Exception):
    """阶段截停哨兵:抛出即证明前置守卫已放行(只断言不被本条守卫拒绝)。"""


class _CountingTrainerStub:
    """训练协议入口计数桩:佐证守卫拒绝路径零调用 train_fold / make_frame_provider。"""

    def __init__(self) -> None:
        self.train_fold_calls = 0
        self.frame_provider_calls = 0
        self.frame_cache = SimpleNamespace(clear=lambda: None)

    def train_fold(self, *args: Any, **kwargs: Any) -> str:
        self.train_fold_calls += 1
        raise AssertionError("守卫拒绝路径不得进入训练协议入口 train_fold")

    def make_frame_provider(self, *args: Any, **kwargs: Any) -> None:
        self.frame_provider_calls += 1
        raise AssertionError("守卫拒绝路径不得构造帧供给")


def _config_variant(
    *,
    recipe_over: Optional[Dict[str, Any]] = None,
    domain: Optional[str] = None,
) -> ModelConfig:
    """e2e ``_config()`` 变体构造(frozen 配置不可就地改,整棵重建)。

    三 spec 注入 ``version=1``(codex-F02:非空 spec 必须带版本号,否则
    ``_register_fold`` 拒绝;e2e ``_config()`` 原文不带版本号,本文件自带版本化
    变体以免与在途 e2e 夹具形态耦合)。
    """
    base = e2e._config()
    recipe = {"version": 1, **dict(base.feature_recipe), **(recipe_over or {})}
    label_spec = {"version": 1, **dict(base.label_spec)}
    split_spec = {"version": 1, **dict(base.split_spec)}
    return ModelConfig(
        model_family=base.model_family,
        feature_columns=tuple(base.feature_columns),
        label_name=base.label_name,
        hyperparams=dict(base.hyperparams),
        train_window_months=base.train_window_months,
        ensemble_seeds=tuple(base.ensemble_seeds),
        label_spec=label_spec,
        feature_recipe=recipe,
        split_spec=split_spec,
        domain=domain if domain is not None else base.domain,
    )


# ---------------------------------------------------------------------------
# 1. reviewer1 复核 P1-1 负例:守卫契约穿透 vs 一般容错
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shared_panel_env(tmp_path_factory):
    """模块级共享 panel 根(store 只读消费,三类用例复用;回执逐 run 复位)。"""
    root = tmp_path_factory.mktemp("oos_guard_panel")
    store = e2e._build_panel_root(root)
    return root, store


class TestOosGuardPassthrough:
    """OOS 回测异常分层:ReplayContractError 穿透,RuntimeError 容错置 {}。"""

    def _make_orchestrator(
        self,
        tmp_path: Path,
        root: Path,
        *,
        backtest_runner: Any,
        frozen_through: Optional[str] = None,
    ):
        output_root = tmp_path / "out"
        registry = ModelRegistry(models_dir=str(output_root / "models" / "stock_selection"))
        trainer = V2Trainer(
            registry=registry,
            main_board_codes=e2e._main_codes(),
            source_mode=SOURCE_PANEL,
            frame_cache=TrainingFrameCache(),
        )
        storage, loader = e2e._old_path_objects(root)
        host = OrchestratorHostConfig(
            data_root=str(root),
            output_root=str(output_root),
            backtest_runner=backtest_runner,
            signal_factory=lambda top_n: None,  # P1-3 守卫:回测启用必须注入工厂
        )
        orch = V2WalkForwardOrchestrator(
            host,
            registry=registry,
            source_mode=SOURCE_PANEL,
            storage=storage,
            loader=loader,
            frozen_through=frozen_through,
        )
        return orch, trainer

    def test_replay_contract_error_from_runner_propagates(self, tmp_path, shared_panel_env):
        """backtest_runner 抛 ReplayContractError ⇒ run 必须抛(不被吞成 bt_metrics={})。"""
        root, store = shared_panel_env

        def runner_boom(**kwargs: Any) -> Dict[str, Any]:
            raise ReplayContractError("OOS回测(panel): 21/22 个交易日缺 core 分区")

        orch, trainer = self._make_orchestrator(tmp_path, root, backtest_runner=runner_boom)
        with pytest.raises(ReplayContractError, match="个交易日缺"):
            orch.run(_config_variant(), 1, e2e._days()[-1], trainer, store, test_window_months=1)

    def test_frozen_through_before_test_window_rejected(self, tmp_path, shared_panel_env):
        """_frozen_through 收紧到测试窗前 ⇒ run 抛 ReplayContractError(回放冻结纪律)。"""
        root, store = shared_panel_env
        runner_calls: List[Dict[str, Any]] = []

        def runner_spy(**kwargs: Any) -> Dict[str, Any]:
            runner_calls.append(kwargs)
            return e2e._stub_backtest_runner(**kwargs)

        # 合成测试窗 2024-06 起;冻结上界收紧到 20240101 ⇒ 任一 bt_end 均越界
        orch, trainer = self._make_orchestrator(
            tmp_path, root, backtest_runner=runner_spy, frozen_through="20240101"
        )
        with pytest.raises(ReplayContractError, match="冻结上界"):
            orch.run(_config_variant(), 1, e2e._days()[-1], trainer, store, test_window_months=1)
        # 冻结窗守卫先于 runner 调用(证明拒绝点位置)
        assert runner_calls == []

    def test_generic_runner_failure_keeps_tolerance(self, tmp_path, shared_panel_env, monkeypatch):
        """backtest_runner 抛一般 RuntimeError ⇒ run 正常完成且该折 bt_metrics={}。"""
        root, store = shared_panel_env

        def runner_boom(**kwargs: Any) -> Dict[str, Any]:
            raise RuntimeError("引擎内部失败")

        orch, trainer = self._make_orchestrator(tmp_path, root, backtest_runner=runner_boom)
        from src.lazybull.v2.train.walk_forward import orchestrator as orch_mod

        captured: Dict[str, Any] = {}
        original = orch_mod.write_walk_forward_summary

        def spy(results, output_path, args, wf_run_id):
            captured["results"] = results
            return original(results, output_path, args, wf_run_id)

        monkeypatch.setattr(orch_mod, "write_walk_forward_summary", spy)
        wf_run_id = orch.run(
            _config_variant(), 1, e2e._days()[-1], trainer, store, test_window_months=1
        )
        assert wf_run_id.startswith("wf_")  # 容错语义:run 正常完成
        assert captured["results"][0]["bt_metrics"] == {}  # 该折回测指标置空


# ---------------------------------------------------------------------------
# 2. P2-2 负例:panel 模式五族开关拒绝
# ---------------------------------------------------------------------------


class TestPanelFamilySwitchRejected:
    """panel 模式五族开关任一开 ⇒ run 期 ValueError;全关 ⇒ 放行到下一阶段。"""

    @pytest.mark.parametrize(
        "switch",
        [
            "enable_holdertrade_features",
            "enable_repurchase_features",
            "enable_top10fh_features",
            "enable_top_inst_features",
            "enable_availability_markers",
        ],
    )
    def test_each_family_switch_rejected_in_panel_mode(self, tmp_path, switch):
        host = OrchestratorHostConfig(data_root=str(tmp_path), output_root=str(tmp_path / "out"))
        orch = V2WalkForwardOrchestrator(
            host,
            registry=ModelRegistry(models_dir=str(tmp_path / "m")),
            source_mode=SOURCE_PANEL,
        )
        config = _config_variant(recipe_over={switch: True})
        with pytest.raises(ValueError, match="物化直读接线未完成"):
            orch.run(config, 1, "20241231", _CountingTrainerStub(), None)

    def test_all_switches_off_pass_guard(self, tmp_path, monkeypatch):
        """五族全关 ⇒ 本条放行(哨兵截停于 _open_run,证明已越过 P2-2/P2-7/P1-3)。"""
        host = OrchestratorHostConfig(
            data_root=str(tmp_path),
            output_root=str(tmp_path / "out"),
            signal_factory=lambda top_n: None,
        )
        orch = V2WalkForwardOrchestrator(
            host,
            registry=ModelRegistry(models_dir=str(tmp_path / "m")),
            source_mode=SOURCE_PANEL,
        )

        def _halt(self, *args: Any, **kwargs: Any):
            raise _SentinelHalt()

        monkeypatch.setattr(V2WalkForwardOrchestrator, "_open_run", _halt)
        with pytest.raises(_SentinelHalt):
            orch.run(_config_variant(), 1, "20241231", _CountingTrainerStub(), None)


# ---------------------------------------------------------------------------
# 3. P2-7 负例:训练域与宿主域一致性
# ---------------------------------------------------------------------------


class TestDomainConsistency:
    """params.stock_domain ≠ host.domain ⇒ run 期 ValueError;一致 ⇒ 放行。"""

    def test_domain_mismatch_rejected(self, tmp_path):
        host = OrchestratorHostConfig(
            data_root=str(tmp_path),
            output_root=str(tmp_path / "out"),
            domain="main",
        )
        orch = V2WalkForwardOrchestrator(
            host,
            registry=ModelRegistry(models_dir=str(tmp_path / "m")),
            source_mode=SOURCE_PANEL,
        )
        config = _config_variant(domain="main_gem")
        with pytest.raises(ValueError, match="训练域.*不一致"):
            orch.run(config, 1, "20241231", _CountingTrainerStub(), None)

    def test_domain_match_passes_guard(self, tmp_path, monkeypatch):
        host = OrchestratorHostConfig(
            data_root=str(tmp_path),
            output_root=str(tmp_path / "out"),
            domain="main",
            signal_factory=lambda top_n: None,
        )
        orch = V2WalkForwardOrchestrator(
            host,
            registry=ModelRegistry(models_dir=str(tmp_path / "m")),
            source_mode=SOURCE_PANEL,
        )

        def _halt(self, *args: Any, **kwargs: Any):
            raise _SentinelHalt()

        monkeypatch.setattr(V2WalkForwardOrchestrator, "_open_run", _halt)
        with pytest.raises(_SentinelHalt):
            orch.run(_config_variant(), 1, "20241231", _CountingTrainerStub(), None)


# ---------------------------------------------------------------------------
# 4. P2-8:selected_split_indices 归一化单元
# ---------------------------------------------------------------------------


class TestNormalizeSelectedSplitIndices:
    def test_dedup_preserves_first_seen_order(self):
        assert _normalize_selected_split_indices([3, 1, 3, 2]) == [3, 1, 2]

    def test_negative_index_rejected(self):
        with pytest.raises(ValueError, match="非负整数"):
            _normalize_selected_split_indices([-1])

    def test_empty_and_none_yield_empty(self):
        assert _normalize_selected_split_indices([]) == []
        assert _normalize_selected_split_indices(None) == []


# ---------------------------------------------------------------------------
# 5. codex-F04:cs_train_verify 族启用 ⇒ 派生注入齐备性校验(训练注册前)
# ---------------------------------------------------------------------------


class TestEvalFamilyInjectionPrecheck:
    """族启用缺 lookup/deriver/出列清单/ensurer ⇒ ValueError 且训练协议入口零调用。"""

    def _make_orchestrator(
        self,
        tmp_path: Path,
        *,
        runtime_derivers: Any = None,
        runtime_lookups: Optional[Dict[str, Any]] = None,
        family_output_cols: Optional[Dict[str, List[str]]] = None,
        availability_ensurer: Any = None,
    ) -> V2WalkForwardOrchestrator:
        root = tmp_path / "cs"
        e2e._write_base_inputs(root)  # 日历 + stock_basic + clean/daily(轻量,无 cs_train)
        storage, loader = e2e._old_path_objects(root)
        host = OrchestratorHostConfig(
            data_root=str(root),
            output_root=str(tmp_path / "out"),
            backtest=BacktestHostParams(oos_backtest=False),
            family_output_cols=family_output_cols,
            availability_ensurer=availability_ensurer,
        )
        return V2WalkForwardOrchestrator(
            host,
            registry=ModelRegistry(models_dir=str(tmp_path / "m")),
            source_mode=SOURCE_CS_TRAIN_VERIFY,
            acceptance_purpose="g_code_parity",
            storage=storage,
            loader=loader,
            runtime_derivers=runtime_derivers,
            runtime_lookups=runtime_lookups,
        )

    @pytest.mark.parametrize(
        "runtime_lookups,runtime_derivers,family_output_cols,missing_hint",
        [
            (
                None,
                SimpleNamespace(holdertrade=object()),
                {"holdertrade": ["ht_col_a"]},
                "runtime_lookups",
            ),
            (
                {"holdertrade_lookup": object()},
                None,
                {"holdertrade": ["ht_col_a"]},
                "runtime_derivers",
            ),
            (
                {"holdertrade_lookup": object()},
                SimpleNamespace(holdertrade=object()),
                None,
                "family_output_cols",
            ),
        ],
        ids=["缺lookup", "缺deriver", "缺出列清单"],
    )
    def test_incomplete_injection_rejected_before_training(
        self, tmp_path, runtime_lookups, runtime_derivers, family_output_cols, missing_hint
    ):
        """derivers/lookups 齐但 family_output_cols=None 等缺项 ⇒ 训练注册前拒绝。"""
        orch = self._make_orchestrator(
            tmp_path,
            runtime_lookups=runtime_lookups,
            runtime_derivers=runtime_derivers,
            family_output_cols=family_output_cols,
        )
        trainer = _CountingTrainerStub()
        config = _config_variant(recipe_over={"enable_holdertrade_features": True})
        with pytest.raises(ValueError, match="派生注入不齐备") as exc_info:
            orch.run(config, 1, e2e._days()[-1], trainer, None, test_window_months=1)
        assert missing_hint in str(exc_info.value)
        # 训练注册前拒绝:训练协议入口零调用
        assert trainer.train_fold_calls == 0
        assert trainer.frame_provider_calls == 0

    def test_availability_incomplete_injection_rejected(self, tmp_path):
        """enable_availability_markers=True 缺 derivers.availability / ensurer ⇒ 拒绝。"""
        orch = self._make_orchestrator(tmp_path, runtime_derivers=SimpleNamespace())
        trainer = _CountingTrainerStub()
        config = _config_variant(recipe_over={"enable_availability_markers": True})
        with pytest.raises(ValueError, match="派生注入不齐备") as exc_info:
            orch.run(config, 1, e2e._days()[-1], trainer, None, test_window_months=1)
        assert "availability" in str(exc_info.value)
        assert trainer.train_fold_calls == 0

    def test_complete_injection_passes_guard(self, tmp_path, monkeypatch):
        """注入齐备 ⇒ 不触发本守卫(哨兵截停于 _run_splits,证明已越过 F04)。"""
        orch = self._make_orchestrator(
            tmp_path,
            runtime_lookups={"holdertrade_lookup": object()},
            runtime_derivers=SimpleNamespace(holdertrade=object()),
            family_output_cols={"holdertrade": ["ht_col_a"]},
        )
        trainer = _CountingTrainerStub()

        def _halt(self, *args: Any, **kwargs: Any):
            raise _SentinelHalt()

        monkeypatch.setattr(V2WalkForwardOrchestrator, "_run_splits", _halt)
        config = _config_variant(recipe_over={"enable_holdertrade_features": True})
        with pytest.raises(_SentinelHalt):
            orch.run(config, 1, e2e._days()[-1], trainer, None, test_window_months=1)
        assert trainer.train_fold_calls == 0


# ---------------------------------------------------------------------------
# 6. codex-F03 末节(R06):replay cs_train 验收源调用方白名单守卫
# ---------------------------------------------------------------------------


class TestReplayGuardPermanent:
    """守卫正反例固化:tests/ 放行;purpose 非法 / 仓外 / 生产脚本调用方 ⇒ 拒绝。"""

    def test_tests_caller_allowed(self):
        from src.lazybull.v2.hosts.backtest import replay as replay_mod

        # 本文件位于 tests/ 白名单:真实调用栈直接放行(不抛即通过)
        replay_mod._assert_replay_cs_train_reachable("regression_edge")

    def test_purpose_required(self):
        from src.lazybull.v2.hosts.backtest import replay as replay_mod

        with pytest.raises(ReplayContractError, match="验收用途"):
            replay_mod._assert_replay_cs_train_reachable(None)
        with pytest.raises(ReplayContractError, match="验收用途"):
            replay_mod._assert_replay_cs_train_reachable("paper_live")

    def test_out_of_repo_caller_rejected(self, monkeypatch):
        """仓外路径(规范化 ⇒ None)⇒ fail-closed 拒绝。"""
        from src.lazybull.v2.hosts.backtest import replay as replay_mod

        monkeypatch.setattr(replay_mod, "_repo_relative_path", lambda filename: None)
        with pytest.raises(ReplayContractError, match="不在仓库内"):
            replay_mod._assert_replay_cs_train_reachable("regression_edge")

    def test_production_script_caller_rejected(self, monkeypatch):
        """生产脚本路径(scripts/paper_trade.py,不在白名单前缀)⇒ 拒绝。"""
        from src.lazybull.v2.hosts.backtest import replay as replay_mod

        monkeypatch.setattr(
            replay_mod, "_repo_relative_path", lambda filename: "scripts/paper_trade.py"
        )
        with pytest.raises(ReplayContractError, match="验收入口"):
            replay_mod._assert_replay_cs_train_reachable("regression_edge")

    def test_internal_frames_skipped_first_external_governs(self, monkeypatch):
        """「首个外部帧」语义:内部引擎帧(v2/train)跳过,首个外部帧落白名单 ⇒ 放行。"""
        from src.lazybull.v2.hosts.backtest import replay as replay_mod

        calls: List[str] = []

        def fake_rel(filename: str) -> str:
            calls.append(filename)
            if len(calls) == 1:
                return "src/lazybull/v2/train/walk_forward/orchestrator.py"  # 内部帧 ⇒ 跳过
            return "tests/test_v2_p3_t3_recheck1_fixes.py"  # 首个外部帧 ⇒ 白名单

        monkeypatch.setattr(replay_mod, "_repo_relative_path", fake_rel)
        replay_mod._assert_replay_cs_train_reachable("regression_edge")  # 不抛即放行
        assert len(calls) == 2  # 内部帧被跳过后才判定外部帧


# ---------------------------------------------------------------------------
# 7. codex-F01 真实负例:manifest 登记不变而物理文件变更
# ---------------------------------------------------------------------------

_CODES25 = [f"{i:06d}.SH" for i in range(1, 5)]  # 4 只合成股
_FEATURE_COLS25 = ["f_alpha", "f_beta"]


class TestManifestRegistrationDrift:
    """25 日合成 store:篡改 core 分区(manifest 保持旧指纹)⇒ 读取期比对检出。"""

    def _build_25d_store(self, root: Path) -> tuple:
        """轻量 25 日 panel store(八族分区 + 三标签表,readiness 齐备口径)。"""
        days = pd.bdate_range("20240102", periods=25).strftime("%Y%m%d").tolist()
        (root / "clean").mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"cal_date": days, "is_open": [1] * len(days)}).to_parquet(
            root / "clean" / "trade_cal.parquet", index=False
        )
        daily = root / "clean" / "daily"
        daily.mkdir(parents=True, exist_ok=True)
        for day in days:
            pd.DataFrame(
                {"trade_date": [day], "ts_code": [_CODES25[0]], "close": [1.0]}
            ).to_parquet(daily / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet", index=False)

        store = PanelDataStore(root)
        store.manifest.register_columns(
            {col: {"group": "core", "source": "test"} for col in _FEATURE_COLS25}
        )
        store.manifest.save()
        from src.lazybull.v2.store.column_groups import PANEL_GROUPS

        for day_idx, day in enumerate(days):
            core_payload: Dict[str, list] = {
                "trade_date": [day] * len(_CODES25),
                "ts_code": list(_CODES25),
            }
            for col_idx, col in enumerate(_FEATURE_COLS25):
                core_payload[col] = [
                    round(day_idx * 0.01 + code_idx * 0.001 + col_idx * 0.0001, 6)
                    for code_idx in range(len(_CODES25))
                ]
            for group in PANEL_GROUPS:
                if group == "core":
                    payload = core_payload
                else:
                    # readiness 要求八族当日齐备(非 core 族仅键列)
                    payload = {
                        "trade_date": core_payload["trade_date"],
                        "ts_code": core_payload["ts_code"],
                    }
                store.append_features(TradeDate.from_str(day), group, pd.DataFrame(payload))
            for table_idx, table in enumerate(LABEL_TABLES):
                store.append_labels(
                    table,
                    pd.DataFrame(
                        {
                            "ts_code": core_payload["ts_code"],
                            "trade_date": core_payload["trade_date"],
                            "label_value": [round(0.02 * (table_idx + 1) + day_idx * 0.001, 6)]
                            * len(_CODES25),
                            "neu_label_value": [round(0.01 * (table_idx + 1) + day_idx * 0.001, 6)]
                            * len(_CODES25),
                            "maturity_status": ["sealed"] * len(_CODES25),
                        },
                        columns=[
                            "ts_code",
                            "trade_date",
                            "label_value",
                            "neu_label_value",
                            "maturity_status",
                        ],  # readiness 协议列序强校验
                    ),
                )
        return store, days

    def test_manifest_registration_drift_detected(self, tmp_path, monkeypatch):
        store, days = self._build_25d_store(tmp_path / "store25")
        config = AdapterConfig(
            start_date=days[0],
            end_date=days[-1],
            feature_columns=tuple(_FEATURE_COLS25),
            label_column="neu_y_ret_20",
            source_mode=SOURCE_PANEL,
        )
        # 对照:未篡改 ⇒ 正常读出(25 日 × 4 股)
        frame = load_training_frame(config, store=store)
        assert len(frame.df) == 25 * len(_CODES25)

        core_path = store.panel_dir / days[0] / "core.parquet"

        def _tamper() -> None:
            df = pd.read_parquet(core_path)
            df.iloc[0, df.columns.get_loc("f_alpha")] = df["f_alpha"].iloc[0] + 1.0
            df.to_parquet(core_path, index=False)  # manifest 登记保持旧指纹

        # 篡改时点:readiness 核验后、load_features 读取前(包装 load_features 首调插桩;
        # 直接先篡改再调第二次会被 readiness 的登记指纹校验以 LabelRefreshRecoveryError
        # 先行拦截,触不到本条 load_features 读后 manifest 比对层)
        original_load = store.load_features
        tampered = False

        def tampering_load(query):
            nonlocal tampered
            if not tampered:
                tampered = True
                _tamper()
            return original_load(query)

        monkeypatch.setattr(store, "load_features", tampering_load)
        with pytest.raises(InputIdentityDriftError, match="manifest 登记指纹不一致"):
            load_training_frame(config, store=store)
        assert tampered  # 篡改确已发生(检出来自 manifest 比对,而非空转)


# ---------------------------------------------------------------------------
# 8. reviewer1 复核 P2-1:extra_sections 入增强 data_state_id
# ---------------------------------------------------------------------------


class TestExtrasEnterDataStateId:
    def _base_state(self) -> Dict[str, Any]:
        return {
            "collected_at": "2026-10-22T00:00:00",
            "wf_run_id": "wf_extras_unit",
            "batch_run_id": None,
            "git_commit": "abc1234",
            "git_dirty": False,
            "raw_latest_partitions": {"daily": "2024-06-28"},
            "features_cs_train_latest": "20240628",
            "dividend_coverage": {"covered": 10},
        }

    def _consumed(self) -> List[Dict[str, Any]]:
        return [{"path": "features/panel/20240102/core.parquet", "sha256": "a" * 64, "size": 128}]

    def _env(self) -> Dict[str, Any]:
        return {"python": "3.12.0", "xgboost": "3.4.0", "env.OMP_NUM_THREADS": ""}

    def test_manifest_fingerprint_difference_changes_id(self):
        """同 state/文件清单/环境,extras 含不同 manifest_fingerprint ⇒ 两个增强 id 不同。"""
        state, consumed, env = self._base_state(), self._consumed(), self._env()
        extras_a = {"manifest_fingerprint": "fp_a", "watermarks": {"input_watermark": "20240628"}}
        extras_b = {"manifest_fingerprint": "fp_b", "watermarks": {"input_watermark": "20240628"}}
        id_a = compute_enhanced_data_state_id(state, consumed, env, extras_a)
        id_b = compute_enhanced_data_state_id(state, consumed, env, extras_b)
        assert id_a != id_b

    def test_none_and_empty_extras_yield_same_id(self):
        """extras=None 与 extras={} ⇒ 同 ID(空段等价语义)。"""
        state, consumed, env = self._base_state(), self._consumed(), self._env()
        assert compute_enhanced_data_state_id(
            state, consumed, env, None
        ) == compute_enhanced_data_state_id(state, consumed, env, {})

    def test_written_json_id_matches_compute(self, tmp_path):
        """落盘 JSON 内 data_state_id == compute_enhanced_data_state_id(extras 先入后算)。"""
        state, consumed, env = self._base_state(), self._consumed(), self._env()
        extras = {
            "manifest_fingerprint": "fp_x",
            "watermarks": {"input_watermark": "20240628"},
            "code_state": {"git_commit": "abc1234", "git_dirty": False, "source_digest": "sd"},
        }
        path = write_enhanced_data_state_file(tmp_path, state, consumed, env, extras)
        assert path is not None
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["data_state_id"] == compute_enhanced_data_state_id(
            state, consumed, env, extras
        )
        assert payload["manifest_fingerprint"] == "fp_x"  # 附加段 additive 落盘

    def test_summary_column_matches_json_id(self, tmp_path):
        """summary 摘要列与 JSON 内 id 同口径(经 _collect_data_state_fields 视图通道)。"""
        from src.lazybull.v2.train.walk_forward import summary as summary_mod

        consumed, env = self._consumed(), self._env()
        extras = {"manifest_fingerprint": "fp_y", "watermarks": {"input_watermark": "20240628"}}
        args = SimpleNamespace(
            data_root=str(tmp_path / "data"),
            batch_run_id=None,
            _data_state_extras={
                "consumed_files": consumed,
                "environment": env,
                "extra_sections": extras,
            },
        )
        output_path = str(tmp_path / "out" / "summary.csv")
        summary_cols = summary_mod._collect_data_state_fields(args, "wf_sum_unit", output_path)
        payload = json.loads(
            (tmp_path / "out" / "data_state_wf_sum_unit.json").read_text(encoding="utf-8")
        )
        assert summary_cols["data_state_id"] == payload["data_state_id"]


# ---------------------------------------------------------------------------
# 9. reviewer1 复核 P2-3:附加段逐段隔离(单段失败不连坐)
# ---------------------------------------------------------------------------


class TestExtraSectionsIsolation:
    def test_watermark_failure_does_not_cascade(self, tmp_path, monkeypatch):
        """resolve_input_watermark 抛错 ⇒ watermarks 段仍含 label_sealed_through、
        manifest_fingerprint / code_state 段齐备(不连坐)。"""
        from src.lazybull.v2.store import store_identity as si_mod
        from src.lazybull.v2.train import data_adapter as da_mod

        def _raise_watermark(root: Any) -> str:
            raise RuntimeError("水位解析爆炸")

        monkeypatch.setattr(da_mod, "resolve_input_watermark", _raise_watermark)
        monkeypatch.setattr(
            si_mod,
            "workspace_identity",
            lambda root: {"git_commit": "c", "git_dirty": False, "source_digest": "d"},
        )

        class _FakeManifest:
            def snapshot(self) -> Dict[str, Any]:
                return {"labels": {"y_ret_20": {"sealed_through": "20240131"}}}

            def content_fingerprint(self) -> str:
                return "fp0123456789abcdef"

        class _FakeStore:
            manifest = _FakeManifest()

        sections = collect_identity_extra_sections(str(tmp_path), _FakeStore())
        assert "watermarks" in sections
        assert sections["watermarks"]["label_sealed_through"] == {"y_ret_20": "20240131"}
        assert "input_watermark" not in sections["watermarks"]  # 失败段缺失而非连坐
        assert sections["manifest_fingerprint"] == "fp0123456789abcdef"
        assert sections["code_state"] == {
            "git_commit": "c",
            "git_dirty": False,
            "source_digest": "d",
        }

    def test_all_sections_fail_independently(self, tmp_path, monkeypatch):
        """全段失败 ⇒ 各段缺失但无 KeyError、无连坐、不上抛。"""
        from src.lazybull.v2.store import store_identity as si_mod
        from src.lazybull.v2.train import data_adapter as da_mod

        def _raise_any(root: Any) -> Any:
            raise RuntimeError("段采集爆炸")

        monkeypatch.setattr(da_mod, "resolve_input_watermark", _raise_any)
        monkeypatch.setattr(si_mod, "workspace_identity", _raise_any)

        class _BrokenManifest:
            def snapshot(self) -> Dict[str, Any]:
                raise RuntimeError("snapshot 爆炸")

            def content_fingerprint(self) -> str:
                raise RuntimeError("指纹爆炸")

        class _BrokenStore:
            manifest = _BrokenManifest()

        sections = collect_identity_extra_sections(str(tmp_path), _BrokenStore())
        # 键保留、值降级 None(reviewer1 复核 P2-3 处置后口径:全段失败不连坐、
        # 不上抛;code_state 键恒存在——消费方无需判存在性,与 collect_data_state
        # 的 git 降级 None 同口径)
        assert set(sections) == {"code_state"}
        assert sections["code_state"] == {
            "git_commit": None,
            "git_dirty": None,
            "source_digest": None,
        }


# ---------------------------------------------------------------------------
# 复核二处置(2026-10-22):P1-1 剩余/F05、F02 残余、P2-2 剩余 DI 转发
# ---------------------------------------------------------------------------


class TestModelConfigDeepFreeze:
    """codex 复核二 F02:配置深冻结永久正反例(代理底层脱钩/嵌套拒改/版本号/空映射)。"""

    def test_proxy_underlying_detached(self):
        """MappingProxy 入参同样脱钩:改代理底层原字典不影响已构造配置。"""
        from types import MappingProxyType

        raw = {"val_ratio": 0.2}
        config = ModelConfig(
            model_family="stock_selection",
            feature_columns=("a",),
            label_name="y",
            hyperparams={},
            train_window_months=72,
            split_spec=MappingProxyType(raw),
        )
        raw["val_ratio"] = 0.9  # 代理底层原字典篡改
        assert config.split_spec["val_ratio"] == 0.2

    def test_nested_structures_immutable(self):
        """嵌套 dict/list 递归不可变:就地修改 ⇒ TypeError/AttributeError。"""
        config = ModelConfig(
            model_family="stock_selection",
            feature_columns=("a",),
            label_name="y",
            hyperparams={},
            train_window_months=72,
            label_spec={"version": 1, "nested": {"x": 1}, "lst": [1, 2]},
        )
        with pytest.raises(TypeError):
            config.label_spec["nested"]["x"] = 9  # type: ignore[index]
        with pytest.raises(AttributeError):
            config.label_spec["lst"].append(3)  # type: ignore[attr-defined]

    def test_spec_version_gate(self, tmp_path):
        """版本号门:非空缺 version ⇒ 拒绝;空映射与 None ⇒ 放行(F10 默认合法)。"""
        from src.lazybull.v2.train.trainer import V2Trainer
        from src.lazybull.v2.train.walk_forward.training_core import TrainFoldParams

        registry = ModelRegistry(models_dir=str(tmp_path / "m"))
        trainer = V2Trainer.__new__(V2Trainer)
        trainer._registry = registry

        def _reg(spec):
            trainer._register_fold(
                config=ModelConfig(
                    model_family="stock_selection",
                    feature_columns=("a",),
                    label_name="y",
                    hyperparams={},
                    train_window_months=72,
                    split_spec=spec,
                ),
                model=SimpleNamespace(),
                params=TrainFoldParams(),
                feature_columns=["a"],
                train_params={},
                performance_metrics={},
                train_start="20240102",
                val_end="20240809",
                X_train_len=1,
                X_val_len=1,
                ensemble_offsets=0,
                ensemble_seeds=[42],
            )

        with pytest.raises(ValueError, match="缺版本号"):
            _reg({"val_ratio": 0.2})
        # 空映射与 None:F10 合法默认,不拒(复核二修正后口径)
        registry.register_model = lambda **kwargs: 1  # type: ignore[assignment]
        _reg({})
        _reg(None)
        _reg({"val_ratio": 0.2, "version": 1})  # 非空带版本号 ⇒ 放行


class TestEvalDIForwardSpy:
    """reviewer1 复核二 P2-2 剩余:评估侧 DI 转发 spy(family_output_cols /
    availability_ensurer 必须实际抵达 evaluate_test_window)。"""

    def test_host_injection_reaches_eval(self, tmp_path, monkeypatch):
        root = tmp_path / "iso"
        store = e2e._build_panel_root(root)
        output_root = tmp_path / "out"
        days = e2e._days()
        registry = ModelRegistry(models_dir=str(output_root / "models" / "stock_selection"))
        trainer = V2Trainer(
            registry=registry,
            main_board_codes=e2e._main_codes(),
            source_mode="panel",
            frame_cache=TrainingFrameCache(),
        )
        storage, loader = e2e._old_path_objects(root)
        sentinel_cols = {"holdertrade": ["ht_x"]}
        sentinel_ensurer = object()
        host = OrchestratorHostConfig(
            data_root=str(root),
            output_root=str(output_root),
            backtest_runner=e2e._stub_backtest_runner,
            signal_factory=lambda top_n: None,
            family_output_cols=sentinel_cols,
            availability_ensurer=sentinel_ensurer,
        )
        orch = V2WalkForwardOrchestrator(
            host, registry=registry, source_mode="panel", storage=storage, loader=loader
        )
        captured: Dict[str, Any] = {}
        from src.lazybull.v2.train.walk_forward import split_training as st_mod

        original = st_mod.evaluate_test_window

        def spy(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return original(**kwargs)

        monkeypatch.setattr(st_mod, "evaluate_test_window", spy)
        orch.run(e2e._config(), 1, days[-1], trainer, store, test_window_months=1)
        assert captured, "评估入口未被调用"
        assert captured["family_output_cols"] is sentinel_cols
        assert captured["availability_ensurer"] is sentinel_ensurer


class TestRealStoreContractBoundary:
    """codex 复核二 F05:真实 store 边界的列登记/PIT 守卫必须穿透为
    ReplayContractError(不允许只让 runner 人工抛出);PIT 满足的正常读取放行。"""

    def test_unregistered_column_wraps_contract_error(self, tmp_path):
        store = e2e._build_panel_root(tmp_path / "root")
        from src.lazybull.v2.hosts.backtest import replay as replay_mod

        with pytest.raises(ReplayContractError, match="特征查询契约失败"):
            replay_mod._load_replay_features_panel(
                store,
                ["20240102"],
                "20240102",
                "20240102",
                None,
                None,
                None,
                None,
                feature_columns=["recheck2_missing_column"],
            )

    def test_pit_violation_wraps_contract_error(self, tmp_path):
        root = tmp_path / "root"
        store = e2e._build_panel_root(root)
        # 登记一列带 available_from 晚于查询窗 ⇒ PIT 越界
        store.manifest.register_columns(
            {
                "recheck2_pit_column": {
                    "group": "core",
                    "source": "test",
                    "available_from": "20260201",
                }
            }
        )
        from src.lazybull.v2.hosts.backtest import replay as replay_mod

        with pytest.raises(ReplayContractError, match="特征查询契约失败"):
            replay_mod._load_replay_features_panel(
                store,
                ["20240102"],
                "20240102",
                "20240102",
                None,
                None,
                None,
                None,
                feature_columns=["recheck2_pit_column"],
            )

    def test_pit_satisfied_normal_read_ok(self, tmp_path):
        """正例:available_from 已满足 ⇒ 正常读取(不被守卫误伤;分区无该列时
        reindex 补 NaN,不追加写——append 同日同族不同内容会被指纹冲突拒绝)。"""
        root = tmp_path / "root"
        store = e2e._build_panel_root(root)
        store.manifest.register_columns(
            {
                "recheck2_ok_column": {
                    "group": "core",
                    "source": "test",
                    "available_from": "20200101",
                }
            }
        )
        days = e2e._days()
        day = days[0]
        from src.lazybull.v2.hosts.backtest import replay as replay_mod

        out = replay_mod._load_replay_features_panel(
            store,
            [day],
            day,
            day,
            None,
            None,
            None,
            None,
            feature_columns=["recheck2_ok_column"],
        )
        assert day in out
