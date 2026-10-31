# -*- coding: utf-8 -*-
"""v2 P4 T2 纸面宿主装配工厂（create_paper_host）。

自 ``host.py`` 拆出的装配收口（行数硬上限拆分，§7.4「新增功能对应新增文件」）：
``host.py`` 持 PaperHost 单日原子驱动本体；本模块持「配置 → TradingConfig → 模型
版本收口 → 信号 → 数据面 → 引擎参数 → PaperHost」的装配函数与其私有 helper。

依赖注入接缝（``signal``/``loader``/``panel_store``/``data_storage``/``stock_basic``/
``trading_calendar``/``feature_columns``/``train_params``/``cost_model``）供测试与
影子臂显式注入；生产缺省走真实根（``data_root=None`` ⇒ 配置单例根）。注入桩信号时
**必须**显式传 ``feature_columns``（就绪门判定基准无模型通道）。``loader`` 注入实例
（含自定义 DataLoader 子类）原样透传宿主 ``data_loader``（F21 处置：不让宿主自
storage 重建而丢失子类语义）；``cost_model=None`` ⇒ 按现逻辑自配置构建。

engine 切换键（Q9）：本工厂只负责解析合法性 fail-fast（非法值拒绝）并登记解析结果
（``host.engine_mode``）；``legacy``/``v2`` 的路由决策归各入口（T5 薄入口切换接线）。
解析结果非 ``v2`` 时记 warning（影子/验收场景下 v2 宿主与现役旧链并行属合法用法）。

展示与信号加载同源（F7/D7/F19 收口）
====================================
``_resolve_model_versions`` 的解析产出挂在 ``host.resolved_model_versions``
（host.py 本轮不可改，实例属性为最小侵入挂点；直构造宿主无此属性 ⇒
:func:`format_host_model_info` 回退读 ``host.trading_config`` 现值）。
:func:`format_host_model_info` = **v2 宿主编排展示路径**：消费该已解析版本渲染与旧
``reporting.format_model_info``（冻结迁移件，自建 PaperStorage 读生产根、自带
registry/sidecar/candidates[-1] 回退链）等价的信息文本——**不再二次解析**
（不调 ``get_latest_version``、不走 candidates[-1] 兜底）：配置留空（latest 解析）
时展示版本恒 == 信号实际加载版本（同一解析产出）。T5 路由义务（bot 五符号切换到
本路径）登记在规划，本模块只提供实现。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
from loguru import logger

# 只读过渡依赖（T0 规划 §3.5 / V2R-02）：config 为全局单例，不复制；去除节点 = P4 展平。
from src.lazybull.common.config import get_data_root, get_stock_selection_models_root

# 只读过渡依赖（E4 登记在册）：旧数据面 Storage/DataLoader。
from src.lazybull.data import DataLoader, Storage

# 只读永久沿用（§3.5 登记表）：选股域与 universe 不迁移。
from src.lazybull.universe import BasicUniverse
from src.lazybull.v2.common.cost import CostModel
from src.lazybull.v2.common.model_registry import ModelRegistry
from src.lazybull.v2.common.protocols.core import PortfolioMerger
from src.lazybull.v2.common.trading_config import TradingConfig
from src.lazybull.v2.core.signal.ensemble_signal import EnsembleSignal
from src.lazybull.v2.core.signal.factory import create_signal
from src.lazybull.v2.core.signal.ml_signal import MLSignal
from src.lazybull.v2.hosts.paper.config_adapter import (
    ENGINE_V2,
    PaperHostError,
    load_paper_config,
    resolve_engine,
    resolve_model_version,
    to_engine_kwargs,
    to_trading_config,
)
from src.lazybull.v2.hosts.paper.daily_assembly import (
    DERIVED_FAMILY_SWITCHES,
    DerivedProducer,
)
from src.lazybull.v2.hosts.paper.host import PaperHost
from src.lazybull.v2.hosts.paper.state_store import PaperStateStore
from src.lazybull.v2.store.data_store import PanelDataStore

__all__ = ["create_paper_host", "format_host_model_info"]


def _resolve_store(store: Union[PaperStateStore, str, Path, None]) -> PaperStateStore:
    """store 入参三形态归一（实例 / 根路径 / None=生产默认根）。"""
    if isinstance(store, PaperStateStore):
        return store
    if store is None:
        return PaperStateStore()
    return PaperStateStore(root_path=str(store))


def _resolve_data_faces(
    data_root: Optional[str],
    data_storage: Optional[Storage],
    loader: Optional[DataLoader],
    verbose: bool,
) -> Tuple[Storage, DataLoader]:
    """旧数据面（Storage/DataLoader，E4 过渡依赖）解析：注入优先，缺省按根创建。"""
    if data_storage is None:
        if loader is not None:
            data_storage = loader.storage
        elif data_root is not None:
            data_storage = Storage(root_path=data_root)
        else:
            data_storage = Storage()
    if loader is None:
        loader = DataLoader(storage=data_storage, verbose=verbose)
    return data_storage, loader


def _resolve_model_versions(trading_config: TradingConfig, models_dir: str) -> Dict[str, Any]:
    """模型版本就地解析回写（F19 收口；model_version_b 非空同样转 int）。

    Returns:
        解析产出 payload（F7 展示同源）：``create_paper_host`` 挂在
        ``host.resolved_model_versions``，展示路径消费同一产出、不再二次解析。
        ``model_version_is_latest`` 标记配置留空走 registry latest 的形态
        （展示层「(最新)」标签同源依据）。
    """
    requested_a = trading_config.model_version
    trading_config.model_version = resolve_model_version(trading_config, models_dir)
    if trading_config.model_version_b is not None:
        trading_config.model_version_b = resolve_model_version(
            trading_config.model_version_b, models_dir
        )
    return {
        "model_version": trading_config.model_version,
        "model_version_b": trading_config.model_version_b,
        "model_version_is_latest": requested_a is None or str(requested_a).strip() == "",
        "ensemble_weight_a": float(trading_config.ensemble_weight_a),
        "models_dir": models_dir,
    }


def _merge_derived_family_switches(
    params_a: Mapping[str, Any], params_b: Mapping[str, Any]
) -> Dict[str, Any]:
    """train_params 合并（F8 处置）：以 A 为基底，派生族开关键取 A/B 并集。

    并集口径 = 任一臂启用即视为启用（键 = 就绪门第 5 条核对面
    ``daily_assembly.DERIVED_FAMILY_SWITCHES``，单一来源不复制键名）；
    非开关键仍以主模型 A 为准（既有行为冻结）。
    """
    merged = dict(params_a)
    for key in DERIVED_FAMILY_SWITCHES.values():
        enabled = params_a.get(key) or params_b.get(key)
        if enabled:
            merged[key] = enabled
    return merged


def _derive_signal_metadata_details(
    signal: Any,
) -> Tuple[
    List[str],
    Dict[str, Any],
    List[tuple[Sequence[str], Mapping[str, Any]]],
]:
    """信号元数据 + 各子模型 freshness 判定面（列集/参数不被跨臂混合）。"""
    if isinstance(signal, EnsembleSignal):
        columns_a, params_a, members_a = _derive_signal_metadata_details(signal.signal_a)
        columns_b, params_b, members_b = _derive_signal_metadata_details(signal.signal_b)
        return (
            list(dict.fromkeys([*columns_a, *columns_b])),
            _merge_derived_family_switches(params_a, params_b),
            [*members_a, *members_b],
        )
    if isinstance(signal, MLSignal):
        signal._load_model()
        train_params = dict((signal.metadata or {}).get("train_params") or {})
        columns = list(signal.feature_columns or [])
        return columns, train_params, [(columns, train_params)]
    raise PaperHostError(
        f"注入信号类型 {type(signal).__name__} 无模型元数据通道；"
        "测试/影子注入桩信号时须显式传 feature_columns=/train_params="
    )


def _derive_signal_metadata(signal: Any) -> Tuple[List[str], Dict[str, Any]]:
    """信号 → (feature_columns, train_params)（保留既有测试/工具接缝）。

    EnsembleSignal 取 A/B 列集并集（保序，A 先），train_params 以主模型 A 为基底、
    派生族开关键取 A/B 并集；freshness 的逐臂判定面由 details 形式额外返回。
    """
    columns, params, _ = _derive_signal_metadata_details(signal)
    return columns, params


def _extract_open_dates(frame: Optional[pd.DataFrame]) -> List[str]:
    """交易日历帧 → 开市日 YYYYMMDD 列表（复刻旧 ``_extract_open_trade_dates`` 语义）。"""
    if frame is None or frame.empty:
        return []
    if "cal_date" not in frame.columns or "is_open" not in frame.columns:
        return []
    cal_dates = frame["cal_date"]
    if pd.api.types.is_datetime64_any_dtype(cal_dates):
        cal_dates = cal_dates.dt.strftime("%Y%m%d")
    else:
        cal_dates = cal_dates.astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
    is_open = frame["is_open"].astype(str)
    return cal_dates[is_open == "1"].dropna().tolist()


def _load_full_calendar(loader: DataLoader) -> List[pd.Timestamp]:
    """完整交易日历（含未来日；复刻旧 ``_load_open_trade_dates_from_storage`` 取舍）。

    取舍规则复刻旧链：raw 延伸更远（或 clean 缺失）⇒ raw，否则 clean（或 raw 兜底）。
    """
    clean_dates = _extract_open_dates(loader.load_clean_trade_cal())
    raw_dates = _extract_open_dates(loader.load_trade_cal())
    if raw_dates and (not clean_dates or raw_dates[-1] > clean_dates[-1]):
        chosen = raw_dates
    else:
        chosen = clean_dates or raw_dates
    if not chosen:
        raise PaperHostError("无法加载交易日历（clean/raw trade_cal 均无开市日）")
    return sorted({pd.Timestamp(str(day)) for day in chosen})


def _load_stock_basic(loader: DataLoader) -> pd.DataFrame:
    """stock_basic 读取（空 ⇒ fail-fast，universe 装配前置）。"""
    stock_basic = loader.load_stock_basic()
    if stock_basic is None or stock_basic.empty:
        raise PaperHostError("无法加载 stock_basic（股票基本信息为空，universe 装配失败）")
    return stock_basic


def _create_universe(
    stock_basic: pd.DataFrame, trading_config: TradingConfig, verbose: bool
) -> BasicUniverse:
    """创建股票池（复刻旧 ``runner/signals.py::_create_universe`` 语义）。

    ``universe="mainboard"`` ⇒ 仅主板（``market == '主板'``）+ BasicUniverse
    （exclude_st/min_list_days 同参）；其它 ⇒ 全市场同参。
    """
    if str(trading_config.universe) == "mainboard":
        mainboard = stock_basic[stock_basic["market"] == "主板"].copy()
        logger.info(f"主板股票数: {len(mainboard)} / {len(stock_basic)}")
        return BasicUniverse(
            stock_basic=mainboard,
            exclude_st=trading_config.exclude_st,
            min_list_days=trading_config.min_list_days,
            verbose=verbose,
        )
    return BasicUniverse(
        stock_basic=stock_basic,
        exclude_st=trading_config.exclude_st,
        min_list_days=trading_config.min_list_days,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# F7：宿主编排展示路径（展示与信号加载同源；冻结件 reporting.format_model_info
# 的等价渲染，消费 ``host.resolved_model_versions`` 解析产出，不再二次解析）
# ---------------------------------------------------------------------------

#: 训练参数展示白名单（与冻结件 reporting._MODEL_INFO_KEY_PARAMS 同口径的本地承载——
#: 不 import reporting：其模块级依赖拖入旧链 runner/storage，且该件逐字冻结不可改）
_MODEL_INFO_KEY_PARAMS = (
    "algorithm",
    "task",
    "objective",
    "eval_metric",
    "label_transform",
    "n_estimators",
    "best_iteration",
    "early_stopping_rounds",
    "early_stopping_metric",
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "gamma",
    "reg_alpha",
    "reg_lambda",
    "min_child_weight",
    "tree_method",
    "device",
    "winsorize_p",
    "ensemble_seeds",
    "ensemble_weight_mode",
)
#: 单个训练参数值的展示截断长度（与冻结件同值）
_MODEL_INFO_PARAM_VALUE_MAX_CHARS = 80
#: 展示文本长度上限（钉钉单条消息防御；与冻结件 MODEL_INFO_MAX_CHARS 同值）
MODEL_INFO_TEXT_MAX_CHARS = 2000


def _find_model_meta(
    models: Sequence[Mapping[str, Any]], version: int
) -> Optional[Mapping[str, Any]]:
    """在模型元数据列表中按版本查找单条记录（与冻结件 ``_find_model_meta`` 同语义）。"""
    for model in models:
        if model.get("version") == version:
            return model
    return None


def _render_train_params_lines(train_params: Mapping[str, Any]) -> List[str]:
    """训练参数（关键白名单）展示行（与冻结件同口径：白名单 + 值截断 + 省略计数）。"""
    if not train_params:
        return []
    lines = ["  训练参数(关键):"]
    shown_count = 0
    for key in _MODEL_INFO_KEY_PARAMS:
        if key not in train_params:
            continue
        value_text = str(train_params[key])
        if len(value_text) > _MODEL_INFO_PARAM_VALUE_MAX_CHARS:
            value_text = value_text[: _MODEL_INFO_PARAM_VALUE_MAX_CHARS - 3] + "..."
        lines.append(f"    {key}: {value_text}")
        shown_count += 1
    omitted_count = len(train_params) - shown_count
    if omitted_count > 0:
        lines.append(f"    (其余 {omitted_count} 个参数省略)")
    return lines


def _render_split_metrics(split_data: Mapping[str, Any]) -> List[str]:
    """validation/test 汇总指标 → 展示部件（IC/RankIC/R2/RMSE，与冻结件同口径）。"""
    parts = []
    ic = split_data.get("ic") or split_data.get("rank_ic")
    rank_ic = split_data.get("rank_ic")
    r2 = split_data.get("r2")
    rmse = split_data.get("rmse")
    if ic is not None:
        parts.append(f"IC={ic:.4f}")
    if rank_ic is not None:
        parts.append(f"RankIC={rank_ic:.4f}")
    if r2 is not None:
        parts.append(f"R2={r2:.4f}")
    if rmse is not None:
        parts.append(f"RMSE={rmse:.4f}")
    return parts


def _render_daily_metrics(split_data: Mapping[str, Any]) -> List[str]:
    """validation_daily/test_daily 日级指标 → 展示部件（与冻结件同口径）。"""
    parts = []
    rankic_mean = split_data.get("daily_rankic_mean")
    rankic_ir = split_data.get("daily_rankic_ir")
    top30_return = split_data.get("top30_return_mean")
    if rankic_mean is not None:
        parts.append(f"DailyRankIC={rankic_mean:.4f}")
    if rankic_ir is not None:
        parts.append(f"IR={rankic_ir:.4f}")
    if top30_return is not None:
        parts.append(f"Top30Ret={top30_return:.4f}")
    return parts


def _render_performance_lines(performance: Mapping[str, Any]) -> List[str]:
    """性能指标区块展示行（与冻结件同口径：汇总两 split + 日级两 split）。"""
    if not performance:
        return []
    lines = ["  性能指标:"]
    for split in ("validation", "test"):
        split_data = performance.get(split)
        if isinstance(split_data, dict) and split_data:
            parts = _render_split_metrics(split_data)
            if parts:
                lines.append(f"    {split}: {', '.join(parts)}")
    for split in ("validation_daily", "test_daily"):
        split_data = performance.get(split)
        if isinstance(split_data, dict) and split_data:
            parts = _render_daily_metrics(split_data)
            if parts:
                lines.append(f"    {split}: {', '.join(parts)}")
    return lines


def _render_model_meta_lines(meta: Mapping[str, Any], *, is_latest: bool) -> List[str]:
    """单模型元数据 → 展示行（与冻结件同版式；``(最新)`` 标签由解析产出标记决定）。"""
    version_label = str(meta["version_str"]) + (" (最新)" if is_latest else "")
    lines = [
        f"当前模型: {version_label}",
        f"  模型类型: {meta.get('model_type', '未知')}",
        f"  训练区间: {meta.get('train_start_date', '?')} ~ {meta.get('train_end_date', '?')}",
        f"  特征数量: {meta.get('feature_count', '?')}",
        f"  训练样本: {meta.get('n_samples', '?')}",
        f"  标签列: {meta.get('label_column', '?')}",
        f"  创建时间: {meta.get('created_at', '?')}",
    ]
    lines.extend(_render_train_params_lines(meta.get("train_params") or {}))
    lines.extend(_render_performance_lines(meta.get("performance_metrics") or {}))
    return lines


def _render_model_not_found(version: int, available: Sequence[int]) -> str:
    """已解析版本在注册表/旁路均缺失的展示文本（超 10 个版本只列最近 10 个）。"""
    if len(available) > 10:
        available_text = f"可用版本共 {len(available)} 个，最近 10 个: {list(available)[-10:]}"
    else:
        available_text = f"可用版本: {list(available)}"
    return f"未找到版本 {version} 的模型。{available_text}"


def _resolution_from_config(trading_config: TradingConfig) -> Dict[str, Any]:
    """直构造宿主（非工厂装配，无 ``resolved_model_versions`` 挂点）的回退解析产出。"""
    return {
        "model_version": trading_config.model_version,
        "model_version_b": trading_config.model_version_b,
        "model_version_is_latest": False,
        "ensemble_weight_a": float(trading_config.ensemble_weight_a),
        "models_dir": None,
    }


def format_host_model_info(host: PaperHost, *, models_dir: Optional[str] = None) -> str:
    """v2 宿主编排展示路径（F7/D7/F19 收口）：渲染与旧 ``format_model_info`` 等价的文本。

    版本来源 = 宿主装配期 ``_resolve_model_versions`` 的产出
    （``host.resolved_model_versions``；直构造宿主回退 ``trading_config`` 现值）——
    **不再二次解析**（不调 registry latest、不走 candidates[-1] 兜底），配置留空
    （latest 解析）时展示版本恒 == 信号实际加载版本。元数据按已解析版本在
    ``models_dir``（缺省 = 解析产出登记的注册表目录）查找：registry 优先、
    sidecar 兜底（与冻结件同序）。

    Args:
        host: 纸面宿主实例。
        models_dir: 注册表目录显式覆盖（缺省 = 解析产出登记的目录，再无 ⇒ 生产默认根）。
    """
    resolution = getattr(host, "resolved_model_versions", None)
    if resolution is None:
        resolution = _resolution_from_config(host.trading_config)
    version = resolution.get("model_version")
    if version is None:
        return "宿主模型版本未解析（非 create_paper_host 装配或配置缺失），无法展示模型信息"
    version = int(version)
    resolved_dir = models_dir or resolution.get("models_dir") or get_stock_selection_models_root()
    registry = ModelRegistry(models_dir=resolved_dir)
    registry_models = registry.list_models()
    sidecar_models = registry.list_sidecar_models()
    meta = _find_model_meta(registry_models, version) or _find_model_meta(sidecar_models, version)
    if meta is None:
        available = sorted(
            {m["version"] for m in registry_models if m.get("version") is not None}
            | {m["version"] for m in sidecar_models if m.get("version") is not None}
        )
        return _render_model_not_found(version, available)
    lines = _render_model_meta_lines(
        meta, is_latest=bool(resolution.get("model_version_is_latest"))
    )
    version_b = resolution.get("model_version_b")
    if version_b is not None:
        weight_a = float(resolution.get("ensemble_weight_a", 0.5))
        lines.extend(
            [
                "",
                "集成模式: 双模型 Ensemble",
                f"  模型A: v{version} (权重 {weight_a})",
                f"  模型B: v{int(version_b)} (权重 {1 - weight_a})",
            ]
        )
    text = "\n".join(lines)
    if len(text) > MODEL_INFO_TEXT_MAX_CHARS:
        text = text[: MODEL_INFO_TEXT_MAX_CHARS - 30] + "\n...(内容过长已截断)"
    return text


def create_paper_host(
    store: Union[PaperStateStore, str, Path, None] = None,
    *,
    config_flat: Optional[Mapping[str, Any]] = None,
    data_root: Optional[str] = None,
    signal: Any = None,
    loader: Optional[DataLoader] = None,
    panel_store: Optional[PanelDataStore] = None,
    data_storage: Optional[Storage] = None,
    stock_basic: Optional[pd.DataFrame] = None,
    trading_calendar: Optional[List[pd.Timestamp]] = None,
    feature_columns: Optional[Sequence[str]] = None,
    train_params: Optional[Mapping[str, Any]] = None,
    cost_model: Optional[CostModel] = None,
    portfolio_merger: Optional[PortfolioMerger] = None,
    derived_families_provided: Sequence[str] = (),
    derived_producers: Optional[Mapping[str, DerivedProducer]] = None,
    verbose: bool = False,
    schedule_mode: str = "live",
    replay_calendar: Optional[List[pd.Timestamp]] = None,
) -> PaperHost:
    """装配纸面宿主（配置 → TradingConfig → 模型版本收口 → 信号 → 数据面 → 引擎参数）。

    装配顺序：store 归一 → 配置读取/展平 → engine 键解析（Q9 合法性 fail-fast +
    结果登记）→ TradingConfig → 模型版本收口（F19，版本回写 config）→ 信号创建
    （或注入）→ 模型列集/train_params（就绪门基准）→ 旧数据面 → stock_basic →
    universe（复刻旧 ``_create_universe``）→ 完整交易日历（含未来日）→
    panel_store → engine_kwargs → PaperHost。

    F21/F7 接缝与挂点见模块 docstring（cost_model 注入、data_loader 透传、
    ``host.resolved_model_versions`` 展示同源挂点）。
    """
    resolved_store = _resolve_store(store)
    flat = dict(config_flat) if config_flat is not None else load_paper_config(resolved_store)
    engine_mode = resolve_engine(flat)
    if engine_mode != ENGINE_V2:
        logger.warning(
            f"配置 engine={engine_mode}（缺省=legacy）：现役生产链路仍走旧链；"
            "v2 宿主供影子/验收场景（Q9 路由决策归入口层）"
        )
    trading_config = to_trading_config(flat)
    models_dir = get_stock_selection_models_root(data_root)
    model_resolution = _resolve_model_versions(trading_config, models_dir)
    if signal is None:
        signal = create_signal(trading_config, models_dir=models_dir, verbose=verbose)
    freshness_members = None
    if feature_columns is None:
        derived_columns, derived_params, freshness_members = _derive_signal_metadata_details(signal)
        feature_columns = derived_columns
        if train_params is None:
            train_params = derived_params
    data_storage, loader = _resolve_data_faces(data_root, data_storage, loader, verbose)
    if stock_basic is None:
        stock_basic = _load_stock_basic(loader)
    universe = _create_universe(stock_basic, trading_config, verbose)
    if trading_calendar is None:
        trading_calendar = _load_full_calendar(loader)
    if panel_store is None:
        panel_store = PanelDataStore(data_root or get_data_root())
    engine_kwargs = to_engine_kwargs(flat, trading_config)
    host = PaperHost(
        store=resolved_store,
        trading_config=trading_config,
        universe=universe,
        signal=signal,
        cost_model=cost_model if cost_model is not None else CostModel(),
        stock_basic=stock_basic,
        data_storage=data_storage,
        panel_store=panel_store,
        trading_calendar=trading_calendar,
        engine_kwargs=engine_kwargs,
        feature_columns=feature_columns,
        train_params=train_params,
        freshness_members=freshness_members,
        derived_families_provided=derived_families_provided,
        derived_producers=derived_producers,
        engine_mode=engine_mode,
        verbose=verbose,
        data_loader=loader,
        portfolio_merger=portfolio_merger,
        schedule_mode=schedule_mode,
        replay_calendar=replay_calendar,
    )
    host.resolved_model_versions = model_resolution
    logger.info(
        f"PaperHost 装配完成: engine={engine_mode}, model_version="
        f"{trading_config.model_version}, top_n={trading_config.top_n}, "
        f"根目录={resolved_store.root_path}"
    )
    return host
