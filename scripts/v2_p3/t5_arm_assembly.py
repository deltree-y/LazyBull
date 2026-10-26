# -*- coding: utf-8 -*-
"""P3 T5 臂②/③ spec 驱动薄入口组装（§19.1-B 13 步）+ 臂① v2 内核回放组装。

组装纪律（验收专用，不入生产链路；cs_train_verify 调用方白名单 = scripts/v2_p3/）：

- 训练配置**零硬编码**：``get_sleeve_spec("value")`` → ``spec.build_model_config()``
  （承接④ spec 消费接线，codex-F03）；
- 承接①：``sleeve_identity=(spec.sleeve_id, spec.spec_version)`` 同值双注入
  ``V2Trainer`` 与 ``OrchestratorHostConfig``（编排器 run 期守卫双侧不一致 ⇒
  ValueError；metadata train_params / runs batch_meta 两路 additive）；
  **P3-1（第三通道撤销）**：runs 批次 config **不写** ``sleeve_id``/``spec_version``
  （身份只走模型 metadata ``train_params`` + runs ``batch_meta`` 的
  ``extra_meta_notes`` 两条已登记通道，config 进配置指纹，禁止第三通道）；
- 承接②：``BacktestHostParams.bt_rebalance_freq`` 显式 = ``spec.holding_period_days``
  且断言与标签名数字一致（``neu_y_ret_20`` → 20，推断口径与
  ``orchestrator.py:507-512`` 相同）；
- 承接③启动门：``spec.feature_columns`` 集合 == ``PANEL_GROUPS`` 派生列集
  （运行时复算，**全集相等**——禁止按关闭族清单子集比对）且有序声明 sha256 ==
  T4 锁定指纹（``5827da6d…``，见 ``sleeves/value/feature_columns.py`` 注释与
  ``tests/test_v2_sleeves_value.py:179-186``）；失败即拒绝启动；
- S2-T5A-01：``BacktestHostParams.bt_top_n`` 消费 ``spec.top_n``；信号类消费
  ``spec.signal_impl``（importlib 解析限定名 → 类，断言解析成功且构造签名与既有
  MLSignal kwarg 表面兼容），替换 spec 值即生效（不再硬编码 20 / MLSignal）；
- ``BacktestHostParams`` 显式生产口径覆盖（默认值 ≠ 生产，§19.0-4；对照
  ``scripts/batch/batch_walk_forward.ps1:353-368``）；
- ``deploy_train=False``（Q10-B：P3 回放显式关闭 deploy 训练，理由登记进证据）；
- ``runs_batch_writer`` = 真实 ``write_runs_batch`` 适配闭包（键映射对齐
  ``runs_writer._RESULT_REQUIRED_KEYS``；``extra_meta_notes`` 可选 kwarg 透传）；
- 臂①补充（§19.1-B 末段）：旧 runner 不写 runs 批次 ⇒ 训练完成后由本组装以
  ``skip_training=True + start_model_version`` 驱动编排器（cs_train_verify，
  ``g_code_parity``）对臂①模型跑 v2 内核 OOS 回测，产出与臂②③同构证据。
"""

from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

# 独立运行时仓库根不入 sys.path，按薄入口惯例显式引导
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.v2_p3.t5_arm_artifacts import ARM_V2_PANEL  # noqa: E402

#: 承接③ T4 全序指纹锁（reviewer2-R2-T4-S01；复算口径 =
#: ``sha256("\n".join(VALUE_FEATURE_COLUMNS).encode()).hexdigest()``）
VALUE_DECLARATION_ORDER_FINGERPRINT = (
    "5827da6dd899b693eb62c384c40f92114ab8d783f34fed8eecf20c598e8807fd"
)

#: cs_train 验证源验收用途（臂①回放 / 臂②；data_adapter.ACCEPTANCE_PURPOSES 登记值）
ACCEPTANCE_G_CODE_PARITY = "g_code_parity"

#: deploy 关闭理由（Q10-B，登记进证据）
DEPLOY_OFF_REASON = "Q10-B：P3 回放显式关闭 deploy 训练，生产部署训练留 P4 验证"


def assert_holding_period_matches_label(spec: Any) -> int:
    """承接②：持有期一致性断言（P3-8），返回标签名推断的持有期天数。

    ``spec.holding_period_days`` 必须与标签名数字一致（``neu_y_ret_20`` → 20；
    推断口径与编排器 ``orchestrator.py:507-512`` 相同：标签名首个数字段）。
    """
    match = re.search(r"(\d+)", str(spec.label_name))
    if match is None:
        raise ValueError(f"标签名无法推断持有期数字（承接②断言失败）: {spec.label_name!r}")
    inferred = int(match.group(1))
    if inferred != int(spec.holding_period_days):
        raise ValueError(
            f"承接②断言失败：spec.holding_period_days={spec.holding_period_days} "
            f"与标签名 {spec.label_name!r} 推断值 {inferred} 不一致"
            "（bt_rebalance_freq 必须由 holding_period_days 显式供给且与标签同口径）"
        )
    return inferred


def assert_feature_declaration_gate(spec: Any) -> Dict[str, object]:
    """承接③启动门：候选超集集合一致 + 有序声明指纹 == T4 锁定值；失败即拒绝启动。

    - 集合：``spec.feature_columns`` == ``PANEL_GROUPS`` 派生列集（运行时复算，
      **全集相等**——禁止按关闭族清单子集比对，§13.3.1 修订注记）；
    - 序：有序声明 sha256 == ``VALUE_DECLARATION_ORDER_FINGERPRINT``（T4 锁定，
      不读生产分区）。
    """
    from src.lazybull.v2.store.column_groups import PANEL_GROUPS

    declared = list(spec.feature_columns)
    derived = {c for cols in PANEL_GROUPS.values() for c in cols} - {"trade_date", "ts_code"}
    if set(declared) != derived:
        only_declared = sorted(set(declared) - derived)[:5]
        only_derived = sorted(derived - set(declared))[:5]
        raise ValueError(
            f"承接③启动门拒绝：候选超集 != PANEL_GROUPS 派生列集（全集比对）——"
            f"仅声明侧 {only_declared} / 仅派生侧 {only_derived}"
        )
    digest = hashlib.sha256("\n".join(declared).encode()).hexdigest()
    if digest != VALUE_DECLARATION_ORDER_FINGERPRINT:
        raise ValueError(
            f"承接③启动门拒绝：有序声明指纹 {digest[:16]}… != T4 锁定 "
            f"{VALUE_DECLARATION_ORDER_FINGERPRINT[:16]}…（声明列序漂移，禁止启动）"
        )
    logger.info(f"承接③启动门通过：{len(declared)} 列集合一致 + 有序指纹 {digest[:16]}…")
    return {
        "declared_column_count": len(declared),
        "panel_groups_derived_count": len(derived),
        "order_fingerprint": digest,
        "locked_fingerprint": VALUE_DECLARATION_ORDER_FINGERPRINT,
    }


def build_backtest_host_params(spec: Any) -> Any:
    """``BacktestHostParams`` 生产口径显式覆盖（默认值 ≠ 生产，§19.0-4 清单逐项）。

    承接②：``bt_rebalance_freq`` 显式 = ``spec.holding_period_days``（本函数内
    先断言与标签名一致）；S2-T5A-01：``bt_top_n`` 消费 ``spec.top_n``（不再硬编码
    20）。对照 ``batch_walk_forward.ps1:353-368`` 现役口径。
    """
    from src.lazybull.v2.train.walk_forward.backtest import BacktestHostParams

    holding_days = assert_holding_period_matches_label(spec)
    return BacktestHostParams(
        oos_backtest=True,
        oos_backtest_months=0,  # 0 = 编排器对齐 test_window_months（生产同口径）
        bt_top_n=int(spec.top_n),
        bt_rebalance_freq=holding_days,
        bt_initial_capital=1_000_000.0,
        bt_sell_timing="open",
        bt_exclude_st=True,
        bt_min_list_days=365,
        bt_max_weight_per_stock=0.15,
        bt_max_per_industry=None,
        bt_stop_loss_enabled=False,
        position_sizing="half_kelly",
        kelly_vol_window=60,
        kelly_max_leverage=0.2,
        stagger_tranches=2,
        enable_early_rebalance_on_empty=True,
    )


#: 信号工厂构造 kwarg 表面（与既有 ``MLSignal.__init__`` 签名逐项对齐；
#: S2-T5A-01 签名兼容断言的唯一口径）
_SIGNAL_FACTORY_KWARGS = (
    "top_n",
    "model_version",
    "models_dir",
    "min_amount_ma20",
    "min_total_mv",
    "max_total_mv",
    "exclude_financial",
    "verbose",
    "downside_penalty",
    "downside_penalty_column",
)


def resolve_signal_class(signal_impl: str) -> type:
    """``spec.signal_impl`` 限定名 → 信号类（importlib 解析；S2-T5A-01 消费接线）。

    断言：解析成功且构造签名与既有 MLSignal kwarg 表面兼容——
    reviewer2 §3 P3 建议强化为 ``inspect.signature(cls).bind(**完整工厂 kwargs)``
    预验证（位置仅限参数形如 ``def __init__(self, top_n, /, **kwargs)`` 与
    额外必填参数两反例同样检出拒绝）；失败即 ValueError（fail-fast，禁止静默
    回退 MLSignal）。解析先例见 ``tests/test_v2_sleeves_value.py:206-229``。
    """
    import importlib
    import inspect

    module_name, class_name = str(signal_impl).rsplit(".", 1)
    try:
        cls = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"signal_impl 解析失败: {signal_impl!r}（{exc}）") from exc
    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"signal_impl {signal_impl!r} 构造签名不可检查（{exc}），"
            "无法断言与 MLSignal kwarg 表面兼容，拒绝"
        ) from exc
    try:
        signature.bind(**dict.fromkeys(_SIGNAL_FACTORY_KWARGS))
    except TypeError as exc:
        raise ValueError(
            f"signal_impl {signal_impl!r} 构造签名与 MLSignal kwarg 表面不兼容"
            f"（bind 预验证失败: {exc}）；构造必须接受 {list(_SIGNAL_FACTORY_KWARGS)}"
            " 全关键字形态（位置仅限参数 / 额外必填参数同样拒绝）"
        ) from exc
    return cls


def build_signal_factory(spec: Any, models_dir: Path) -> Callable[[int], Any]:
    """跨折持久化信号工厂（宿主构造；绑定隔离模型根 + spec 信号参数 + 域市值边界）。

    域市值边界解析自 ``universe/domains.py`` 单一来源（规格不复制域市值）；
    ``model_version=None`` ⇒ 逐折按版本回读（e2e 先例）；S2-T5A-01：信号类经
    ``spec.signal_impl`` importlib 解析（``resolve_signal_class`` 兼容断言先行），
    不再硬编码 MLSignal。
    """
    from src.lazybull.universe.domains import resolve_stock_domain

    domain_cfg = resolve_stock_domain(spec.domain)
    signal_cls = resolve_signal_class(spec.signal_impl)

    def factory(top_n: int) -> Any:
        return signal_cls(
            top_n=top_n,
            model_version=None,
            models_dir=str(models_dir),
            min_amount_ma20=spec.min_amount_ma20,
            min_total_mv=float(domain_cfg["min_total_mv_wan"]),
            max_total_mv=float(domain_cfg["max_total_mv_wan"]),
            exclude_financial=spec.exclude_financial,
            verbose=False,
            downside_penalty=0.0,
            downside_penalty_column="downside_vol_20",
        )

    return factory


def build_runs_batch_config(spec: Any, backtest: Any, source_mode: str) -> Dict[str, object]:
    """runs 批次配置字典（spec + 回测参数唯一来源，零硬编码字面量）。

    G-code ② 要求两臂 summary 配置键逐项一致 ⇒ 两臂必须经本函数同源构建
    （臂①回放与臂②同为 cs_train_verify；臂③ panel 单列不参与 G-code ②）。

    P3-1（第三通道撤销）：config **不写** ``sleeve_id``/``spec_version``——袖子身份
    只走模型 metadata ``train_params`` + runs ``batch_meta`` 的 ``extra_meta_notes``
    两条已登记通道；config 进配置指纹，第三通道会把身份漂移误报为配置漂移。
    """
    hp = spec.hyperparams
    label_spec = spec.label_spec
    return {
        "label_column": spec.label_name,
        "task": label_spec["task"],
        "label_transform": label_spec["label_transform"],
        "neutral_label_blend_weight": label_spec["neutral_label_blend_weight"],
        "algorithm": hp["algorithm"],
        "objective": hp["objective"],
        "n_estimators": hp["n_estimators"],
        "max_depth": hp["max_depth"],
        "learning_rate": hp["learning_rate"],
        "min_child_weight": hp["min_child_weight"],
        "colsample_bytree": hp["colsample_bytree"],
        "subsample": hp["subsample"],
        "reg_alpha": hp["reg_alpha"],
        "reg_lambda": hp["reg_lambda"],
        "gamma": hp["gamma"],
        "early_stopping_rounds": hp["early_stopping_rounds"],
        "early_stopping_metric": hp["early_stopping_metric"],
        "min_best_iteration": hp["min_best_iteration"],
        "rank_weight": hp["rank_weight"],
        "rank_weight_topk": hp["rank_weight_topk"],
        "rank_weight_topk_weight_mode": hp["rank_weight_topk_weight_mode"],
        "freshness_strategy": hp["freshness_strategy"],
        "event_freshness_half_life_days": hp["event_freshness_half_life_days"],
        "ensemble_seeds": ",".join(str(seed) for seed in spec.ensemble_seeds),
        "bt_top_n": backtest.bt_top_n,
        "bt_rebalance_freq": backtest.bt_rebalance_freq,
        "bt_sell_timing": backtest.bt_sell_timing,
        "bt_initial_capital": backtest.bt_initial_capital,
        "bt_min_list_days": backtest.bt_min_list_days,
        "bt_max_weight_per_stock": backtest.bt_max_weight_per_stock,
        "position_sizing": backtest.position_sizing,
        "kelly_vol_window": backtest.kelly_vol_window,
        "kelly_max_leverage": backtest.kelly_max_leverage,
        "stagger_tranches": backtest.stagger_tranches,
        "stock_domain": spec.domain,
        "feature_source": source_mode,
        "deploy_train": False,
    }


def build_runs_batch_writer(
    *,
    runs_root: Path,
    config: Dict[str, object],
    code_state: Dict[str, object],
    arms: List[str],
    source: str,
) -> Callable[..., Optional[Path]]:
    """runs 批次写出器适配闭包（orchestrator 调点签名 → 真实 ``write_runs_batch``）。

    键映射对齐 ``runs_writer._RESULT_REQUIRED_KEYS``（编排器下划线键 → 契约键）；
    ``extra_meta_notes`` 为编排器 additive 通道（承接①：袖子身份并入 batch_meta），
    缺省 None 时透传 None（与内核默认形态一致）；折缺净值曲线 ⇒ 出口前显式拒绝。
    """

    def writer(
        results: List[Dict[str, Any]],
        wf_run_id: str,
        data_state_payload: Dict[str, Any],
        extra_meta_notes: Optional[Dict[str, str]] = None,
    ) -> Optional[Path]:
        from src.lazybull.v2.hosts.backtest.runs_writer import write_runs_batch

        mapped = []
        for result in results:
            nav_curve = result.get("_nav_curve")
            if nav_curve is None or len(nav_curve) == 0:
                raise ValueError(
                    f"split {result.get('split_index')} 缺 OOS 回测净值曲线，"
                    "runs 批次出口拒绝（契约三态=必须报错；先排查该折回测失败原因）"
                )
            mapped.append(
                {
                    "split_index": result["split_index"],
                    "model_version": result["model_version"],
                    "train_start": result["train_start"],
                    "train_end": result["train_end"],
                    "test_start": result["test_start"],
                    "test_end": result["test_end"],
                    "metrics": dict(result.get("bt_metrics") or {}),
                    "nav_curve": nav_curve,
                    "trades": result.get("_trades"),
                    "attribution": result.get("_execution_attribution"),
                    "holdings_snapshot": result.get("_holdings_snapshot"),
                }
            )
        return write_runs_batch(
            mapped,
            batch_id=wf_run_id,
            runs_root=Path(runs_root),
            config=dict(config),
            code_state=dict(code_state),
            data_state=data_state_payload,
            baseline_ref=None,
            arms=arms,
            source=source,
            extra_meta_notes=extra_meta_notes,
        )

    return writer


@dataclass
class ArmAssembly:
    """臂组装产物（编排器 + 训练器 + 配套引用与启动门证据）。"""

    spec: Any
    config: Any
    host: Any
    orchestrator: Any
    trainer: Any
    store: Any
    registry: Any
    backtest_params: Any
    source_mode: str
    gate_evidence: Dict[str, object]


@dataclass
class _StorageAndDomain:
    """步骤 4~7 组装产物（storage/loader/域/注册表/store + 承接②回执）。"""

    storage: Any
    loader: Any
    markets: tuple
    main_board_codes: Any
    backtest_params: Any
    registry: Any
    store: Any
    holding_period_evidence: Dict[str, object]


def _assemble_storage_and_domain(root: Path, spec: Any, is_panel: bool) -> _StorageAndDomain:
    """步骤 4~7：storage/loader → 域解析（+承接②回执）→ 注册表 → store。"""
    from src.lazybull.data.loader import DataLoader
    from src.lazybull.data.storage import Storage
    from src.lazybull.universe.domains import domain_market_whitelist
    from src.lazybull.v2.common.model_registry import ModelRegistry
    from src.lazybull.v2.train.walk_forward.training_core import _build_main_board_codes

    # 步骤 4：storage/loader（两臂均需 loader；cs_train_verify 另需 storage）
    storage = Storage(root_path=str(root))
    loader = DataLoader(storage)

    # 步骤 5：域解析（universe/domains.py 单一来源；trainer 与编排器同源同 markets）
    #         + 承接②断言（在 build_backtest_host_params 内先行，此处取回执值登记）
    markets = tuple(domain_market_whitelist(spec.domain))
    stock_basic = loader.load_clean_stock_basic()
    if stock_basic is None:
        stock_basic = loader.load_stock_basic()
    main_board_codes = _build_main_board_codes(stock_basic, markets=markets)
    backtest_params = build_backtest_host_params(spec)
    holding_period_evidence = {
        "label_name": spec.label_name,
        "holding_period_days": spec.holding_period_days,
        "bt_rebalance_freq": backtest_params.bt_rebalance_freq,
    }

    # 步骤 6：注册表绑定隔离输出根（禁止联接生产 models）
    registry = ModelRegistry(models_dir=str(root / "models" / "stock_selection"))

    # 步骤 7：store（臂③ panel 直读隔离根；臂①回放/臂② None）
    store = None
    if is_panel:
        from src.lazybull.v2.store.data_store import PanelDataStore

        store = PanelDataStore(root)

    return _StorageAndDomain(
        storage=storage,
        loader=loader,
        markets=markets,
        main_board_codes=main_board_codes,
        backtest_params=backtest_params,
        registry=registry,
        store=store,
        holding_period_evidence=holding_period_evidence,
    )


def _assemble_trainer_and_signal(
    spec: Any,
    inputs: _StorageAndDomain,
    source_mode: str,
    acceptance_purpose: Optional[str],
    sleeve_identity: tuple,
) -> tuple:
    """步骤 8~9：训练器（承接①身份注入；frame_cache 显式实例）+ 信号工厂。"""
    from src.lazybull.v2.train.frame_cache import TrainingFrameCache
    from src.lazybull.v2.train.trainer import V2Trainer

    trainer = V2Trainer(
        registry=inputs.registry,
        main_board_codes=inputs.main_board_codes,
        source_mode=source_mode,
        acceptance_purpose=acceptance_purpose,
        storage=inputs.storage,
        loader=inputs.loader,
        frame_cache=TrainingFrameCache(),
        runtime_derivers=None,  # 验收臂五族全关（spec recipe 生产口径）
        runtime_lookups=None,
        sleeve_identity=sleeve_identity,
    )
    signal_factory = build_signal_factory(spec, inputs.registry.models_dir)
    return trainer, signal_factory


def _assemble_host(
    arm: str,
    root: Path,
    runs_root: Path,
    spec: Any,
    inputs: _StorageAndDomain,
    source_mode: str,
    sleeve_identity: tuple,
    signal_factory: Callable[[int], Any],
    *,
    summary_name: Optional[str],
    export_topk_details: bool,
    write_batch: bool,
    code_state: Optional[Dict[str, object]],
) -> Any:
    """步骤 10：宿主配置（生产口径回测参数 + writer 接入 + sleeve_identity 并入 batch_meta）。"""
    from src.lazybull.v2.hosts.backtest.replay import run_oos_backtest
    from src.lazybull.v2.train.walk_forward.orchestrator import OrchestratorHostConfig

    if code_state is None:
        from src.lazybull.v2.store.store_identity import workspace_identity

        identity = workspace_identity(_REPO_ROOT)
        code_state = {"git_commit": identity["git_commit"], "git_dirty": identity["git_dirty"]}
    writer = None
    if write_batch:
        writer = build_runs_batch_writer(
            runs_root=Path(runs_root),
            config=build_runs_batch_config(spec, inputs.backtest_params, source_mode),
            code_state=code_state,
            arms=[arm],
            source=f"v2_p3_t5_{arm}",
        )
    summary_csv = None
    if summary_name:
        summary_csv = str(
            root / "walk_forward" / "raw" / f"walk_forward_summary_{summary_name}.csv"
        )
    return OrchestratorHostConfig(
        data_root=str(root),
        output_root=str(root),
        backtest=inputs.backtest_params,
        domain_markets=inputs.markets,
        wf_summary_csv=summary_csv,
        export_topk_details=export_topk_details,
        # batch_run_id/batch_period_label 不传（输出追踪参数，不影响数值路径——
        # spike 差异清单④同口径登记）
        backtest_runner=run_oos_backtest,
        signal_factory=signal_factory,
        runs_batch_writer=writer,
        domain=spec.domain,
        sleeve_identity=sleeve_identity,
    )


def assemble_arm_orchestrator(
    arm: str,
    root: Path,
    runs_root: Path,
    *,
    run_name: str,
    summary_name: Optional[str] = None,
    export_topk_details: bool = True,
    write_batch: bool = True,
    code_state: Optional[Dict[str, object]] = None,
) -> ArmAssembly:
    """§19.1-B 13 步组装（臂②/③训练与臂① skip 回放共用；差异只在 run() 调用参数）。

    - 臂② ``v2_cs_train_verify``：source_mode=cs_train_verify + g_code_parity + store=None；
    - 臂③ ``v2_panel``：source_mode=panel + store=PanelDataStore(隔离根) + runtime_lookups=None；
    - 臂①回放复用本组装（cs_train_verify；调用方传 ``export_topk_details=False``
      避免与旧 runner topk 明细 glob 双命中）；
    - ``write_batch=False`` ⇒ 不接 runs 批次写出器（续跑补训缺折场景：批次由
      随后的全折 skip 回放统一产出，防多批次目录）；
    - P2-5（复杂度门）：步骤 4~7 / 8~9 / 10 提取为 ``_assemble_storage_and_domain`` /
      ``_assemble_trainer_and_signal`` / ``_assemble_host``（行为逐字保持）。
    """
    from src.lazybull.v2.sleeves._registry import get_sleeve_spec
    from src.lazybull.v2.train.walk_forward.orchestrator import (
        _RETIRED_SWITCH_KEYS,
        V2WalkForwardOrchestrator,
        reject_retired_switches,
    )

    root = Path(root)
    is_panel = arm == ARM_V2_PANEL
    source_mode = "panel" if is_panel else "cs_train_verify"
    acceptance_purpose = None if is_panel else ACCEPTANCE_G_CODE_PARITY

    # 步骤 2：退役开关显式拒绝（本组装永不接线；键清单存在性即守卫）
    reject_retired_switches(**dict.fromkeys(_RETIRED_SWITCH_KEYS))

    # 步骤 3：spec → ModelConfig（训练配置零硬编码，承接④）；承接③启动门先行
    spec = get_sleeve_spec("value")
    gate_evidence = {"feature_declaration_gate": assert_feature_declaration_gate(spec)}
    config = spec.build_model_config()

    # 步骤 4~7：storage/loader + 域 + 注册表 + store（承接②回执随出）
    inputs = _assemble_storage_and_domain(root, spec, is_panel)
    gate_evidence["holding_period_assert"] = inputs.holding_period_evidence

    # 承接①：袖子身份同值双注入（trainer + host；编排器 run 期守卫）
    sleeve_identity = (spec.sleeve_id, spec.spec_version)

    # 步骤 8~10：训练器 + 信号工厂 + 宿主配置
    trainer, signal_factory = _assemble_trainer_and_signal(
        spec, inputs, source_mode, acceptance_purpose, sleeve_identity
    )
    host = _assemble_host(
        arm,
        root,
        runs_root,
        spec,
        inputs,
        source_mode,
        sleeve_identity,
        signal_factory,
        summary_name=summary_name,
        export_topk_details=export_topk_details,
        write_batch=write_batch,
        code_state=code_state,
    )

    # 步骤 11：编排器（registry 与 trainer 同一实例；frozen_through=None ⇒ 默认冻结上界）
    orchestrator = V2WalkForwardOrchestrator(
        host,
        registry=inputs.registry,
        source_mode=source_mode,
        acceptance_purpose=acceptance_purpose,
        storage=inputs.storage,
        loader=inputs.loader,
        runtime_lookups=None,  # panel 模式硬契约；cs_train_verify 五族全关 ⇒ None
        frozen_through=None,
    )
    gate_evidence["sleeve_identity"] = {
        "sleeve_id": spec.sleeve_id,
        "spec_version": spec.spec_version,
    }
    gate_evidence["deploy_train"] = {"enabled": False, "reason": DEPLOY_OFF_REASON}
    return ArmAssembly(
        spec=spec,
        config=config,
        host=host,
        orchestrator=orchestrator,
        trainer=trainer,
        store=inputs.store,
        registry=inputs.registry,
        backtest_params=inputs.backtest_params,
        source_mode=source_mode,
        gate_evidence=gate_evidence,
    )
