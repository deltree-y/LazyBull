# -*- coding: utf-8 -*-
"""v2 P4 T2 纸面宿主配置适配（config.yaml 分组模板 → v2 内核参数）。

职责边界（行为冻结第一）
========================
旧 ``data/paper/config.yaml`` 分组模板与用户配置习惯不变（D7 配置载体）；本模块是
**唯一的配置解析收口**：读原始 YAML dict（``PaperStateStore.load_config`` 返回未规范化
dict）→ 展平分组 → 剔除退役键 → 映射 v2 ``TradingConfig`` / 引擎构造参数。旧纸面
``paper/storage/config.py`` 为冻结件，本模块对其只读引用（``CONFIG_SECTION_NAMES`` /
``is_retired_config_field`` / ``SECTION_DEFAULT_EXTRAS`` 三常量/函数，登记），
不复制其渲染/写盘逻辑。

退役键与退役家族
================
- 退役键剔除规则只读引用旧 ``is_retired_config_field``（``RETIRED_CONFIG_KEYS`` +
  ``RETIRED_CONFIG_PREFIXES`` 的唯一权威实现）；命中键在展平后剔除并记 info 日志。
- ``exposure.*`` 家族（Q5 退役）：展平后**透传不消费**——不进入 TradingConfig 映射、
  不进入引擎参数；配置含非空 ``exposure_policy`` 时记 warning（政策形态 = null）。
- ``signal_penalty.*`` → ``downside_penalty`` / ``downside_penalty_column`` 两键同名
  入 TradingConfig（默认 0.0 = 关闭；λ>0 由 v2 MLSignal 构造 fail-fast，本层不拦截）。

engine 切换键（Q9）各入口解析规则（成文）
=========================================
- 解析唯一入口 = :func:`resolve_engine`；``run`` / ``bot`` / ``adjust`` / ``config`` /
  只读展示各入口统一经本函数解析，禁止各自读键。
- 缺省（键缺失/空值）⇒ ``"legacy"``（现役旧链）；合法值闭集 ``{legacy, v2}``；
  非法值 ⇒ fail-fast（:class:`PaperHostError`），禁止静默回退任一侧。
- CLI 覆盖形态 = 入口层先改写配置再经本函数解析（不在本函数内做优先级合并）。
- 回退防呆（Q9）：``v2`` → ``legacy`` 回退只认 D11 三步程序（迁移工具双向校验后翻键）；
  过时旧状态不得接回当前账户——本函数只负责解析，状态防呆在宿主启动路径
  （权威快照 vs 旧链遗产投影识别，见 ``host.py``）。

模型选择收口（F19）
===================
:func:`resolve_model_version` 是 v2 纸面侧**唯一**模型版本解析点（信号加载与展示同源）：
``config.model_version`` 非空 ⇒ 转 int；空 ⇒ ``ModelRegistry.get_latest_version()``
（registry latest 旁路语义为准：latest_model_version.txt → 注册表尾 next_version-1 →
全表倒扫非 classifier）；解析失败 ⇒ PaperHostError。
差异登记：旧 ``reporting.format_model_info`` 展示路径在 registry latest 之外还有
sidecar 扫描 + ``candidates[-1]`` 兜底两层额外回退——执行侧**不收口**到该宽松形态，
以本函数为准（展示层残留兜底属旧链行为冻结范围，不改）。

capital_retention_ratio 漂移防御（F19 处置）
===========================================
旧链指令预算含 ``× (1 − capital_retention_ratio)`` 分支（旧 ``paper/runner`` 经
``_get_cost_setting`` 消费）；v2 侧（``to_trading_config`` / ``to_engine_kwargs``）
**无该键消费点**。现役配置 = 0.0（等价无操作）；一旦经覆盖形态设 >0，v2 指令股数
将系统性偏大且无检出。``to_engine_kwargs`` 对该键非 0/非 None ⇒ ``logger.warning``
明确告警「v2 侧无消费点，该配置不生效」——**不 fail-fast**（旧链仍消费该键，
影子期共用配置不能断）。

执行价格入冻结指纹（D7）
========================
- ``buy_price`` 必须为 ``"close"``：v2 引擎买入价索引恒定 = 不复权 close（无 open
  选项；旧链现役 config.yaml 亦为 close）；配置给 ``open`` ⇒ fail-fast（登记不支持）。
- ``sell_price`` ∈ ``{"open", "close"}`` ⇒ 映射引擎 ``sell_timing``。

completion_window_days 取值登记
===============================
取 **5**（与 v2 回测宿主 ``hosts/backtest/runtime.py::create_backtest_engine_from_config``
默认值一致——T4 门 3 双宿主比对前提）。旧纸面补位语义是「次数制（5 次）」而非
「日窗制」，引擎语义以注册表锚定（日窗），宿主取回测宿主同值，差异登记。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from loguru import logger

# 只读过渡依赖（T0 规划 §3.5 / V2R-02）：config 为全局单例，不复制；去除节点 = P4 展平。
from src.lazybull.common.config import get_stock_selection_models_root

# 冻结件只读引用（登记）：退役键判定与分区块名单的唯一权威在旧 paper/storage/config.py，
# 本模块不复制其常量内容；旧块名单变动时本模块行为随之变动（同源语义）。
from src.lazybull.paper.storage.config import (
    CONFIG_SECTION_NAMES,
    SECTION_DEFAULT_EXTRAS,
    is_retired_config_field,
)
from src.lazybull.v2.common.model_registry import ModelRegistry
from src.lazybull.v2.common.trading_config import TradingConfig
from src.lazybull.v2.hosts.paper.state_store import PaperStateStore

__all__ = [
    "PaperHostError",
    "PAPER_CONFIG_TO_V2",
    "ENGINE_SWITCH_KEY",
    "ENGINE_LEGACY",
    "ENGINE_V2",
    "VALID_ENGINES",
    "PAPER_COMPLETION_WINDOW_DAYS",
    "load_paper_config",
    "resolve_engine",
    "resolve_model_version",
    "to_trading_config",
    "to_engine_kwargs",
]


class PaperHostError(RuntimeError):
    """纸面宿主错误（配置非法 / 引擎切换键非法 / 模型版本解析失败 / 宿主守卫拒绝）。

    唯一定义点在本模块（避免 host ↔ adapter 循环依赖）；``host.py`` re-export。
    继承 RuntimeError 以对齐旧链 ``execute_trade_workflow`` 的拒绝形态
    （如日期回退守卫旧抛 RuntimeError）。
    """


#: engine 切换键名（Q9；config.yaml 仅允许新增此键，D7 禁区措辞）
ENGINE_SWITCH_KEY = "engine"
#: 引擎合法取值：legacy = 现役旧链（缺省），v2 = v2 宿主
ENGINE_LEGACY = "legacy"
ENGINE_V2 = "v2"
VALID_ENGINES = frozenset({ENGINE_LEGACY, ENGINE_V2})

#: 补齐窗口期取值（日窗制，与 v2 回测宿主同值；取值理由见模块 docstring 登记）
PAPER_COMPLETION_WINDOW_DAYS = 5

#: 旧纸面配置键 → v2 TradingConfig 字段映射表（消费白名单）。
#: 分组模板展平后两侧键名同名（旧模板本就以 TradingConfig 字段名为键），
#: 显式成表的意义 = **消费面白名单**：表外键（exposure.*/engine/extra 杂物）
#: 一律不进入 TradingConfig；TradingConfig 新增字段不会自动流入纸面消费面。
_PAPER_CONFIG_KEYS = (
    # model 区块
    "model_version",
    "model_version_b",
    "ensemble_weight_a",
    # portfolio 区块
    "top_n",
    "rebalance_freq",
    "stagger_tranches",
    "max_per_industry",
    "max_weight_per_stock",
    "enable_early_rebalance_on_empty",
    "exclude_st",
    "min_list_days",
    # stop_loss 区块
    "stop_loss_enabled",
    "stop_loss_drawdown_pct",
    "stop_loss_consecutive_limit_down",
    # position_management 区块
    "position_sizing",
    "kelly_vol_window",
    "kelly_max_leverage",
    # paper_trade 区块
    "buy_price",
    "sell_price",
    "initial_capital",
    "min_buy_value_ratio",
    "horizon",
    "universe",
    # signal_penalty 区块（λ=0 默认关；>0 由 MLSignal fail-fast）
    "downside_penalty",
    "downside_penalty_column",
)
PAPER_CONFIG_TO_V2: Dict[str, str] = {name: name for name in _PAPER_CONFIG_KEYS}

#: 展平时并入扁平 dict 的分区块名（旧模板分区块 + 兼容扩展块 extra；
#: 规则只读引用旧 CONFIG_SECTION_NAMES）
_FLATTEN_SECTION_NAMES = frozenset(CONFIG_SECTION_NAMES) | {"extra"}

#: 配置消费面之外的合法键（未消费键审计日志的白名单部分）；
#: ``capital_retention_ratio`` 有专属漂移告警（F19 处置，见 ``to_engine_kwargs``），
#: 不再列入通用未消费键 debug 审计
_KNOWN_UNCONSUMED_KEYS = frozenset(
    {ENGINE_SWITCH_KEY, "weight_method", "capital_retention_ratio"}
) | frozenset(SECTION_DEFAULT_EXTRAS)


def _flatten_grouped_config(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """分组 YAML dict → 扁平 dict（复刻旧 ``_flatten_grouped_config`` 规则）。

    规则：键命中已知分区块名（含 ``extra``）且值为 dict ⇒ 并入；其余键原样保留。
    未知块原样保留（旧规则同）。块名单只读引用旧 ``CONFIG_SECTION_NAMES``。
    """
    flat: Dict[str, Any] = {}
    for key, value in raw.items():
        if key in _FLATTEN_SECTION_NAMES and isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return flat


def load_paper_config(
    store_or_root: Union[PaperStateStore, str, Path, None] = None,
) -> Dict[str, Any]:
    """读取纸面配置并展平为扁平 dict（剔除退役键；exposure 家族透传不消费）。

    **只读形态**（F14 处置）：内部构造的 PaperStateStore 一律
    ``create_layout=False``——「读配置」不得在盘上创建七子目录；该形态的写
    路径在 ``acquire_write_lock`` 处 fail-fast（state_store 侧契约）。

    Args:
        store_or_root: PaperStateStore 实例 / 配置根路径（str/Path）/ None
            （None ⇒ PaperStateStore 默认根解析，生产口径；影子臂/测试显式传隔离根）。

    Returns:
        扁平配置 dict（未做 TradingConfig 规范化——规范化归 :func:`to_trading_config`）。

    Raises:
        PaperHostError: 配置文件缺失 / 内容非 dict 形态。
    """
    if isinstance(store_or_root, PaperStateStore):
        store = store_or_root
    elif store_or_root is None:
        store = PaperStateStore(create_layout=False)
    else:
        store = PaperStateStore(root_path=str(store_or_root), create_layout=False)
    raw = store.load_config()
    if raw is None:
        raise PaperHostError(
            f"未找到配置文件（根目录 {store.root_path}）；"
            "请先编辑 config.yaml 或运行 config 命令设置配置"
        )
    if not isinstance(raw, dict):
        raise PaperHostError(f"配置文件形态非法（顶层须为 dict）: {store.root_path}")
    flat = _flatten_grouped_config(raw)
    retired = sorted(key for key in flat if is_retired_config_field(key))
    for key in retired:
        flat.pop(key)
    if retired:
        logger.info(f"已剔除已下线功能的配置键（{len(retired)} 个）: {retired}")
    # exposure 家族（Q5 退役）：透传不消费；非空启用形态告警（政策形态 = null）
    if flat.get("exposure_policy") not in (None, ""):
        logger.warning(
            "配置含非空 exposure_policy：暴露政策族已退役（Q5），v2 宿主不消费该家族 "
            "（政策形态 = null，空操作相位仅保留顺序断言）"
        )
    return flat


def resolve_engine(flat: Mapping[str, Any]) -> str:
    """engine 切换键解析（Q9；各入口解析规则成文见模块 docstring）。

    缺省 ⇒ ``"legacy"``；合法值 ``{legacy, v2}``；非法 ⇒ PaperHostError fail-fast。
    """
    raw = flat.get(ENGINE_SWITCH_KEY)
    if raw is None or str(raw).strip() == "":
        return ENGINE_LEGACY
    value = str(raw).strip().lower()
    if value not in VALID_ENGINES:
        raise PaperHostError(
            f"engine 切换键非法: {raw!r}（合法值 {sorted(VALID_ENGINES)}，缺省 = "
            f"{ENGINE_LEGACY}）；禁止静默回退任一侧"
        )
    return value


def resolve_model_version(
    config_or_version: Union[TradingConfig, int, str, None],
    models_dir: Optional[str] = None,
) -> int:
    """模型版本解析唯一收口（F19；信号加载与展示同源，语义见模块 docstring）。

    Args:
        config_or_version: TradingConfig（读其 model_version）或原始版本值
            （int/str/None）；非空 ⇒ 转 int，空 ⇒ registry latest 旁路语义。
        models_dir: 模型注册表目录；None ⇒ ``get_stock_selection_models_root()``
            （只读过渡依赖，登记）。

    Raises:
        PaperHostError: 版本值非法 / 注册表无可用主模型 / 注册表读取失败。
    """
    raw = (
        config_or_version.model_version
        if isinstance(config_or_version, TradingConfig)
        else config_or_version
    )
    if raw is not None and str(raw).strip() != "":
        try:
            version = int(raw)
        except (TypeError, ValueError) as exc:
            raise PaperHostError(f"model_version 非法（须为正整数版本号）: {raw!r}") from exc
        if version < 1:
            raise PaperHostError(f"model_version 非法（须为正整数版本号）: {raw!r}")
        return version
    registry = ModelRegistry(models_dir=models_dir or get_stock_selection_models_root())
    try:
        latest = registry.get_latest_version()
    except (OSError, ValueError, KeyError) as exc:
        raise PaperHostError(
            f"注册表解析最新模型版本失败（{registry.models_dir}）: {type(exc).__name__}: {exc}"
        ) from exc
    if latest is None:
        raise PaperHostError(
            f"注册表无可用主模型版本（{registry.models_dir}）；"
            "请先训练注册模型，或在 config.yaml 显式指定 model_version"
        )
    logger.info(f"model_version 未指定，按注册表 latest 语义解析: v{latest}")
    return int(latest)


def to_trading_config(flat: Mapping[str, Any]) -> TradingConfig:
    """扁平纸面配置 → v2 TradingConfig（消费白名单映射 + D7 执行价格指纹校验）。

    - 只消费 ``PAPER_CONFIG_TO_V2`` 白名单键（表外键一律不进 TradingConfig）；
      ``weight_method`` 旧扁平别名 ⇒ ``position_sizing``（与旧 ``_normalize_config``
      及 TradingConfig.from_dict 同款别名规则）。
    - 执行价格指纹（D7）：``buy_price`` 必须 ``"close"``（v2 引擎买入价恒定不复权
      close，无 open 选项；配置 open ⇒ fail-fast 登记不支持）；``sell_price`` ∈
      ``{"open", "close"}``（映射引擎 sell_timing）。

    Raises:
        PaperHostError: 入参非 Mapping / 执行价格指纹非法 / TradingConfig 校验失败。
    """
    if not isinstance(flat, Mapping):
        raise PaperHostError(f"扁平配置形态非法（须为 Mapping）: {type(flat).__name__}")
    mapped: Dict[str, Any] = {
        v2_key: flat[paper_key]
        for paper_key, v2_key in PAPER_CONFIG_TO_V2.items()
        if paper_key in flat
    }
    if "position_sizing" not in mapped and flat.get("weight_method") is not None:
        mapped["position_sizing"] = flat["weight_method"]
    # 执行价格冻结指纹（D7）
    buy_price = str(mapped.get("buy_price", "close")).strip().lower()
    if buy_price != "close":
        raise PaperHostError(
            f"buy_price={buy_price!r} 不支持：v2 引擎买入价恒定 = 不复权 close（D7 执行价格"
            "冻结指纹，无 open 选项；旧链现役配置亦为 close）"
        )
    sell_price = str(mapped.get("sell_price", "open")).strip().lower()
    if sell_price not in ("open", "close"):
        raise PaperHostError(
            f"sell_price={sell_price!r} 非法（合法值 open/close；映射引擎 sell_timing，D7）"
        )
    try:
        return TradingConfig.from_dict(mapped)
    except (TypeError, ValueError) as exc:
        raise PaperHostError(f"纸面配置映射 TradingConfig 校验失败: {exc}") from exc


def _log_unconsumed_keys(flat: Mapping[str, Any]) -> None:
    """未消费键审计（debug 级）：展平后在消费白名单外的键登记（透传不生效）。"""
    extra = sorted(
        key for key in flat if key not in PAPER_CONFIG_TO_V2 and key not in _KNOWN_UNCONSUMED_KEYS
    )
    if extra:
        logger.debug(f"纸面配置含未消费键（透传不生效，不进入 TradingConfig/引擎参数）: {extra}")


def _warn_capital_retention_drift(flat: Mapping[str, Any]) -> None:
    """``capital_retention_ratio`` 配置漂移防御（F19 处置；语义见模块 docstring）。

    该键非 0/非 None ⇒ warning（非数值形态无法证明等价 0，同样告警）；
    0/缺省/None ⇒ 静默（现役配置等价形态）。**不 fail-fast**——旧链仍消费该键，
    影子期共用配置不能断。
    """
    raw = flat.get("capital_retention_ratio")
    if raw is None:
        return
    try:
        is_zero = float(raw) == 0.0
    except (TypeError, ValueError):
        is_zero = False
    if not is_zero:
        logger.warning(
            f"配置含 capital_retention_ratio={raw!r}：v2 侧（to_trading_config/"
            "to_engine_kwargs）无该键消费点，指令预算不会乘 (1 − 留存比)，"
            "该配置不生效（旧链仍消费；影子期共用配置不断链，仅告警不拒绝）"
        )


def to_engine_kwargs(flat: Mapping[str, Any], trading_config: TradingConfig) -> Dict[str, Any]:
    """纸面配置 → BacktestEngineML 构造参数映射（模板 = hosts/backtest 回测宿主）。

    只承载**配置来源**参数；非配置来源参数（universe/signal/cost_model/stock_basic/
    data_storage/features_by_date）由宿主装配注入，不在本映射。

    硬约束与模板值（登记）：
    - ``enable_trailing_early_rebalance=False``：live 复刻旧链「仅空仓提前调仓」语义
      （§17.7 硬项①；引擎默认 True 服务回测/回放，宿主侧恒 False）；
    - ``sell_timing=trading_config.sell_price``（D7 指纹）；
    - ``stop_loss_config=trading_config.create_stop_loss_config()``；
    - ``completion_window_days=5``（与回测宿主同值；旧纸面为次数制 5 次，差异登记
      见模块 docstring）；
    - 其余模板值（enable_pending_order=True / max_retry_count=5 / max_retry_days=10 /
      enable_position_completion=True）对齐回测宿主模板默认值。

    Raises:
        PaperHostError: ``rebalance_freq`` 为 None（live 锚定排期依赖具体正整数频率；
            旧链 ``int(config["rebalance_freq"])`` 对 None 同样失败，此处提前 fail-fast
            并带上下文）。
    """
    if trading_config.rebalance_freq is None:
        raise PaperHostError(
            "纸面宿主要求 rebalance_freq 为具体正整数（live 持久化锚定排期依赖）；"
            "config.yaml 缺省 20，请在 portfolio.rebalance_freq 显式配置"
        )
    _warn_capital_retention_drift(flat)  # F19 处置：v2 侧无消费点，非 0 配置告警
    _log_unconsumed_keys(flat)
    return {
        "initial_capital": float(trading_config.initial_capital),
        "rebalance_freq": int(trading_config.rebalance_freq),
        "stagger_tranches": int(trading_config.stagger_tranches),
        "stop_loss_config": trading_config.create_stop_loss_config(),
        "sell_timing": trading_config.sell_price,
        "enable_pending_order": True,
        "max_retry_count": 5,
        "max_retry_days": 10,
        "enable_position_completion": True,
        "completion_window_days": PAPER_COMPLETION_WINDOW_DAYS,
        "enable_early_rebalance_on_empty": bool(trading_config.enable_early_rebalance_on_empty),
        "position_sizing": trading_config.position_sizing,
        "kelly_vol_window": int(trading_config.kelly_vol_window),
        "kelly_max_leverage": float(trading_config.kelly_max_leverage),
        "min_buy_value_ratio": float(trading_config.min_buy_value_ratio),
        "max_weight_per_stock": trading_config.max_weight_per_stock,
        "max_per_industry": trading_config.max_per_industry,
        "enable_trailing_early_rebalance": False,
    }
