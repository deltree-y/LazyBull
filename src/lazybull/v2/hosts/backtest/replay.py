"""Walk-forward 单 split OOS 回测重放驱动（P2a-T7 迁移复制件，行为冻结）。

来源蓝本（B2）：旧 ``ml/walk_forward/backtest.py::run_oos_backtest`` 复制改指。
WF 编排（runner.py）本身不迁；本模块是 B0 重放 14 折的单折执行体。

改指清单：
- runtime 三函数 → ``src.lazybull.v2.hosts.backtest.runtime``（A1）；
- ``TradingConfig`` → ``src.lazybull.v2.common.trading_config``（T1 复制件）；
- ``get_data_root`` → 旧 ``src.lazybull.common.config``（只读过渡依赖，
  全局单例不复制，去除节点 P4 展平/切换评审）；
- ``DataLoader / Storage`` → 旧 ``src.lazybull.data``（只读过渡依赖，
  去除节点 P2b/P3 数据面切换，§3.5 登记表）；
- factors 四族运行时派生函数（holdertrade / repurchase / top10_floatholders /
  top_inst）→ 旧包**永久沿用**（§3.5 登记表）；
- ``BasicUniverse`` / ``domain_market_whitelist`` → 旧 ``universe``（永久沿用）。

D6 摘除（契约授权例外）：exposure 政策族已裁决退役——删除
``engine.set_exposure_table(...)`` 与 ``engine.set_exposure_policy(...)``
两次调用及文末 exposure 统计日志块；**签名保留 5 个 exposure 参数**，非默认值
显式 fail-fast（B0 重放只允许全默认值）。

D8 拆分（T7 实测触发）：旧 ``run_oos_backtest`` 在全仓复杂度基线在册
（func_lines / mccabe 21 两条），逐字复制即新增未豁免硬超限 ⇒ 按 D8 同例
提取式拆分为 4 个模块级 helper（fail-fast / 日线加载 / 单族派生 / 指标段），
语句、守卫条件、日志文本与顺序逐字保留，行为零变化。

P2b-T3 差异登记：新增缺省 ``None`` 的合并模式透传、构造后适配器装配和
逐折真实 ``merge()`` 调用计数采集。缺省调用不向引擎工厂新增 kwargs，
成交、净值与公开 metrics 行为不变。

P3-T3 特征源改造（§3.3 本期清偿 / §17.4）：
- 新增 ``feature_source`` / ``store`` / ``acceptance_purpose`` 三参数；
  **默认 ``feature_source="panel"``**——OOS 特征改指 store panel（整窗
  ``load_features`` + 按日切分），**显式关闭旧特征读取回退**（panel 缺数据
  ⇒ RuntimeError，旧「警告跳过」语义不再适用）；panel 模式传四族 lookup ⇒
  ValueError（物化直读，防 derive 双写，§3.3-③）；
- ``cs_train_verify`` = §3.1 例外验收源（旧逐日读取路径逐字保留，含空跳过回退），
  白名单守卫：调用方限 scripts/v2_p3/ + scripts/v2_p2a/ + scripts/v2_p2b/ + tests/
  （P2a/P2b 工具为 §7.1 回归边明文允许的 cs_train 验收源使用方）；
- 既有调用方（replay_b0 / replay_merger）已显式改传 ``cs_train_verify`` +
  ``regression_edge``（修改面入回滚清单，§17.9-12）；
- 字符串常量与适配层双源等价（hosts 依赖表不含 train，协议 §7——
  等价断言测试锁定，防漂移）；
- 行情读（``loader.load_clean_daily`` / 日历）保留旧 DataLoader——行情/日历类
  过渡依赖同 F14 改期 P4 口径（state.py / engine_utils.py / suspend_calendar.py
  三处的同族补充登记，§17.3 接线表 #4）。
"""

import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from loguru import logger

# 只读过渡依赖（去除节点 P4）：config 为全局单例，不复制（§3.5 / V2R-02）。
from src.lazybull.common.config import get_data_root

# 只读过渡依赖（去除节点 P2b/P3 → P3-T3 起仅限 cs_train_verify 验收例外路径,
# P4 评审清偿）：panel 默认路径不再使用旧特征读取。
from src.lazybull.data import DataLoader, Storage

# 运行时派生四族因子（永久沿用，§3.5 登记表）：factors 不迁移。
from src.lazybull.factors.holdertrade import derive_holdertrade_columns
from src.lazybull.factors.repurchase import derive_repurchase_columns
from src.lazybull.factors.top10_floatholders import derive_top10fh_columns
from src.lazybull.factors.top_inst import derive_top_inst_columns

# universe 永久沿用（§3.5 登记表）。
from src.lazybull.universe import BasicUniverse
from src.lazybull.universe.domains import domain_market_whitelist
from src.lazybull.v2.common.trading_config import TradingConfig
from src.lazybull.v2.common.types import FeatureQuery, ReplayContractError, TradeDate
from src.lazybull.v2.hosts.backtest.runtime import (
    configure_merger_mode,
    create_backtest_engine_from_config,
    create_or_reuse_signal,
    get_merger_call_count,
    infer_rebalance_freq_from_label,
    validate_merger_mode,
)

# ---------------------------------------------------------------------------
# P3-T3 特征源（§17.4）
# ---------------------------------------------------------------------------

#: OOS 特征源封闭集合：panel（默认，store panel 直读）/ cs_train_verify（验收例外）
FEATURE_SOURCE_PANEL = "panel"
FEATURE_SOURCE_CS_TRAIN_VERIFY = "cs_train_verify"

#: cs_train_verify 验收用途白名单（与 v2/train/data_adapter.ACCEPTANCE_PURPOSES 双源
#: 等价——hosts 依赖表不含 train（协议 §7），等价断言测试锁定防漂移）
REPLAY_ACCEPTANCE_PURPOSES = frozenset({"g_code_parity", "regression_edge"})

#: cs_train_verify 调用方白名单（仓库相对路径前缀；P2a/P2b 工具 = §7.1 回归边
#: 明文允许的验收源使用方）
_REPLAY_CS_TRAIN_CALLER_PREFIXES = (
    "scripts/v2_p3/",
    "scripts/v2_p2a/",
    "scripts/v2_p2b/",
    "tests/",
)
#: 引擎/宿主内部帧前缀（守卫跳过：orchestrator 经 backtest 薄组装调用属合法内部链）
_REPLAY_INTERNAL_PREFIXES = (
    "src/lazybull/v2/hosts/backtest/",
    "src/lazybull/v2/train/",
)

#: 仓库根（replay.py 位于 src/lazybull/v2/hosts/backtest/）
_REPO_ROOT = Path(__file__).resolve().parents[5]
_REPLAY_SELF_REL = "src/lazybull/v2/hosts/backtest/replay.py"


def _repo_relative_path(filename: str) -> Optional[str]:
    """绝对路径 → 仓库相对正斜杠路径;仓外/伪文件名(REPL/-c/`<stdin>`)⇒ None。"""
    try:
        return Path(filename).resolve().relative_to(_REPO_ROOT).as_posix()
    except (ValueError, OSError):
        return None


def _assert_replay_cs_train_reachable(acceptance_purpose: Optional[str]) -> None:
    """cs_train_verify 验收可达性守卫（用途白名单 + 调用方文件白名单，fail-closed）。

    P3-T3 评审处置（codex-R06）：调用帧统一规范为**仓库相对路径**后做前缀校验
    （仓外路径 ⇒ None ⇒ 拒绝）；仅精确跳过本模块自身与内部包前缀，删除
    「任意 replay.py / 含 inspect 帧」的泛化跳帧。
    """
    if acceptance_purpose not in REPLAY_ACCEPTANCE_PURPOSES:
        raise ReplayContractError(
            f"cs_train 验收源模式必须声明验收用途（{sorted(REPLAY_ACCEPTANCE_PURPOSES)}）: "
            f"{acceptance_purpose!r}"
        )
    frame = inspect.currentframe()
    if frame is None:
        raise ReplayContractError("cs_train 验收源模式调用方守卫：无法获取调用栈（fail-closed）")
    try:
        caller = frame.f_back  # 本函数 → run_oos_backtest 内联点
        while caller is not None:
            rel = _repo_relative_path(caller.f_code.co_filename)
            if rel is None:
                raise ReplayContractError(
                    "cs_train 验收源模式仅允许仓内验收入口调用"
                    f"（{_REPLAY_CS_TRAIN_CALLER_PREFIXES}），实际调用方不在仓库内: "
                    f"{caller.f_code.co_filename}"
                )
            if rel == _REPLAY_SELF_REL or any(
                rel.startswith(prefix) for prefix in _REPLAY_INTERNAL_PREFIXES
            ):
                caller = caller.f_back
                continue
            if any(rel.startswith(prefix) for prefix in _REPLAY_CS_TRAIN_CALLER_PREFIXES):
                return
            raise ReplayContractError(
                "cs_train 验收源模式仅允许验收入口调用"
                f"（{_REPLAY_CS_TRAIN_CALLER_PREFIXES}），实际调用方: {rel}"
            )
        raise ReplayContractError("cs_train 验收源模式调用方守卫：调用栈无外部帧（fail-closed）")
    finally:
        del frame


def _fail_fast_retired_exposure(
    exposure_table: Optional[Dict[str, float]],
    exposure_policy: Any,
    exposure_replenish: bool,
    exposure_trim_tolerance: Optional[float],
    exposure_budget_discount_replenish: bool,
) -> None:
    """D6 摘除 fail-fast：exposure 政策族已退役摘除（P2a D6），B0 重放只允许全默认值；
    非默认值立即报错，禁止静默丢弃（行为冻结的例外边界显式化）。"""
    if (
        exposure_table is not None
        or exposure_policy is not None
        or exposure_replenish
        or exposure_trim_tolerance is not None
        or exposure_budget_discount_replenish
    ):
        raise ValueError(
            "exposure 政策族已退役摘除（P2a D6），B0 重放只允许全默认值："
            f"exposure_table="
            f"{type(exposure_table).__name__ if exposure_table is not None else None}"
            f", exposure_policy="
            f"{type(exposure_policy).__name__ if exposure_policy is not None else None}"
            f", exposure_replenish={exposure_replenish}"
            f", exposure_trim_tolerance={exposure_trim_tolerance}"
            f", exposure_budget_discount_replenish={exposure_budget_discount_replenish}"
        )


def _load_replay_price_data(
    loader: DataLoader, bt_start: str, bt_end: str
) -> Optional[pd.DataFrame]:
    """日线加载与列裁剪（D8 提取自 run_oos_backtest；语句/日志逐字保留）。

    Returns:
        price_data；None = 旧侧早退（``return {}``）触发条件。
        注：列子集不改行数，``len(price_data)`` 即旧侧 ``len(daily_data)``。
    """
    daily_data = loader.load_clean_daily(bt_start, bt_end)
    if daily_data is None or len(daily_data) == 0:
        logger.warning(f"OOS回测: 无法加载 {bt_start}~{bt_end} 日线数据，跳过")
        return None

    desired_cols = [
        "ts_code",
        "trade_date",
        "close",
        "close_adj",
        "open",
        "open_adj",
        "is_suspended",
        "is_limit_up",
        "is_limit_down",
        "vol",
        "pct_chg",
        "is_st",
        "list_days",
        "tradable",
    ]
    existing_cols = [column for column in desired_cols if column in daily_data.columns]
    price_data = daily_data[existing_cols].copy()
    if "close" not in price_data.columns:
        logger.warning("OOS回测: 缺少 close 列，跳过")
        return None
    return price_data


def _derive_feature_family(
    features_by_date: Dict[str, pd.DataFrame],
    source: Any,
    derive_fn: Callable[[pd.DataFrame, Any], bool],
    log_label: str,
) -> None:
    """单族运行时派生逐日补齐（D8 参数化提取：旧四段的公共形态，日志文本逐字）。"""
    derived_days = 0
    for trade_date, features in features_by_date.items():
        if derive_fn(features, source):
            derived_days += 1
    logger.info(f"OOS 回测特征派生{log_label}列: {derived_days}/{len(features_by_date)} 日")


def _load_replay_features_cs_train(
    storage: Storage,
    trade_dates: list,
    bt_start: str,
    bt_end: str,
    holdertrade_lookup: Optional[Dict[str, pd.DataFrame]],
    repurchase_lookup: Optional[Dict[str, pd.DataFrame]],
    top10fh_panel: Optional[pd.DataFrame],
    top_inst_lookup: Optional[Dict[str, pd.DataFrame]],
) -> Optional[Dict[str, pd.DataFrame]]:
    """特征分区加载 + 四族运行时派生补齐（cs_train 验收例外路径，语句逐字保留）。

    Returns:
        features_by_date；None = 旧侧早退（``return {}``）触发条件（无特征数据）。
    """
    features_by_date = {}
    for trade_date in trade_dates:
        features = storage.load_cs_train_day(trade_date)
        if features is not None and len(features) > 0:
            features_by_date[trade_date] = features

    # 股东增减持因子为**运行时派生**（cs_train 不含本族列）：回测复用同一张查询表
    # 逐日补齐，否则 MLSignal 预测将因缺列而静默补 NaN（train/serve 偏差）。
    if holdertrade_lookup:
        _derive_feature_family(
            features_by_date, holdertrade_lookup, derive_holdertrade_columns, "股东增减持"
        )

    # 股票回购因子同为运行时派生（cs_train 不含本族列）：复用同一张查询表逐日补齐
    if repurchase_lookup:
        _derive_feature_family(
            features_by_date, repurchase_lookup, derive_repurchase_columns, "股票回购"
        )

    # 十大流通股东因子同为运行时派生（cs_train 不含本族列）：复用同一份报告期面板
    if top10fh_panel is not None and len(top10fh_panel) > 0:
        _derive_feature_family(
            features_by_date, top10fh_panel, derive_top10fh_columns, "十大流通股东"
        )

    # 龙虎榜机构席位因子同为运行时派生（cs_train 不含本族列）：复用同一张查询表逐日补齐
    if top_inst_lookup:
        _derive_feature_family(
            features_by_date, top_inst_lookup, derive_top_inst_columns, "龙虎榜机构席位"
        )

    if not features_by_date:
        logger.warning(f"OOS回测: 无特征数据 {bt_start}~{bt_end}，跳过")
        return None
    return features_by_date


def _split_panel_frame_by_date(panel_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """panel 整窗帧（index=(trade_date, ts_code)）按日切分为逐日平铺帧。

    输出形态镜像 cs_train 日分区（trade_date/ts_code 为列）；行序 = panel 排序
    索引序（与 cs_train 分区存储序的差异属 G-data 归因范围，§17.4-②）。
    """
    features_by_date: Dict[str, pd.DataFrame] = {}
    if panel_df is None or panel_df.empty:
        return features_by_date
    flat = panel_df.reset_index()
    flat["trade_date"] = flat["trade_date"].astype(str)
    for trade_date, day_df in flat.groupby("trade_date", sort=True):
        features_by_date[str(trade_date)] = day_df.reset_index(drop=True)
    return features_by_date


def _load_replay_features_panel(
    store: Any,
    trade_dates: list,
    bt_start: str,
    bt_end: str,
    holdertrade_lookup: Optional[Dict[str, pd.DataFrame]],
    repurchase_lookup: Optional[Dict[str, pd.DataFrame]],
    top10fh_panel: Optional[pd.DataFrame],
    top_inst_lookup: Optional[Dict[str, pd.DataFrame]],
    feature_columns: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """OOS 特征 panel 路径（P3-T3 默认）：整窗 ``load_features``（全登记列）+
    按日切分；四族物化直读（传 lookup ⇒ ValueError 防双写）；**显式关闭旧
    特征读取回退**——缺数据 ⇒ RuntimeError（旧「警告跳过 return {}」语义不再适用）。
    """
    retired_lookups = {
        "holdertrade_lookup": holdertrade_lookup,
        "repurchase_lookup": repurchase_lookup,
        "top10fh_panel": top10fh_panel,
        "top_inst_lookup": top_inst_lookup,
    }
    supplied = [name for name, value in retired_lookups.items() if value is not None]
    if supplied:
        raise ReplayContractError(
            f"panel 模式四族已物化直读，禁止传运行时派生 lookup（防双写/覆盖，§3.3-③）: "
            f"{supplied}"
        )
    # P3-T3 评审处置(reviewer1-P2-4):列集收窄——显式给定时按折模型实际特征列
    # 查询(内存/IO 收敛;MLSignal 只消费模型列,缺列由 check_feature_consistency
    # 硬报错兜住);None ⇒ 全登记列(回放工具兼容路径)
    if feature_columns:
        query_columns = list(dict.fromkeys(feature_columns))
    else:
        snapshot = store.manifest.snapshot()
        query_columns = sorted(snapshot["columns"])
    # P3-T3 复核二处置(reviewer1-P1-1 剩余 ≡ codex-F05):
    # ① 按协议转换为 TradeDate(store 的 available_from PIT 比较要求类型化,
    #    字符串直传会抛 TypeError);
    # ② 列集未登记 / PIT 越界等契约校验错误(ValueError/TypeError)包装为
    #    ReplayContractError 穿透「容错吞掉」路径(守卫契约,types.py:423)
    try:
        panel_frame = store.load_features(
            FeatureQuery(
                columns=query_columns,
                start_date=TradeDate.from_str(bt_start),
                end_date=TradeDate.from_str(bt_end),
            )
        )
    except (ValueError, TypeError) as exc:
        raise ReplayContractError(
            f"OOS回测(panel) 特征查询契约失败（列登记/PIT/类型守卫）: {exc}"
        ) from exc
    features_by_date = _split_panel_frame_by_date(panel_frame.df)
    if not features_by_date:
        raise ReplayContractError(
            f"OOS回测(panel): 无特征数据 {bt_start}~{bt_end}"
            "（旧特征读取回退已显式关闭；请检查 panel 分区覆盖或改用验收例外模式）"
        )
    days_without_partition = [day for day in trade_dates if day not in features_by_date]
    if days_without_partition:
        # P3-T3 评审处置(reviewer1-P2-4):部分缺日硬失败(与适配层覆盖校验同口径;
        # 旧「warning 后继续」会静默少日);**全空 ⇒ 上一分支(回退关闭语义)**
        raise ReplayContractError(
            f"OOS回测(panel): {len(days_without_partition)}/{len(trade_dates)} 个交易日缺 "
            f"core 分区（如 {days_without_partition[:5]}），请检查 panel 分区覆盖"
        )
    return features_by_date


def _select_replay_features(
    feature_source: str,
    store: Any,
    acceptance_purpose: Optional[str],
    storage: Storage,
    trade_dates: list,
    bt_start: str,
    bt_end: str,
    holdertrade_lookup: Optional[Dict[str, pd.DataFrame]],
    repurchase_lookup: Optional[Dict[str, pd.DataFrame]],
    top10fh_panel: Optional[pd.DataFrame],
    top_inst_lookup: Optional[Dict[str, pd.DataFrame]],
    feature_columns: Optional[List[str]] = None,
) -> Optional[Dict[str, pd.DataFrame]]:
    """特征源分派（P3-T3）：panel 默认 / cs_train_verify 验收例外（白名单守卫）。"""
    if feature_source == FEATURE_SOURCE_PANEL:
        if store is None:
            raise ReplayContractError("panel 特征源模式必须提供 PanelDataStore（store=）")
        return _load_replay_features_panel(
            store,
            trade_dates,
            bt_start,
            bt_end,
            holdertrade_lookup,
            repurchase_lookup,
            top10fh_panel,
            top_inst_lookup,
            feature_columns,
        )
    if feature_source == FEATURE_SOURCE_CS_TRAIN_VERIFY:
        _assert_replay_cs_train_reachable(acceptance_purpose)
        return _load_replay_features_cs_train(
            storage,
            trade_dates,
            bt_start,
            bt_end,
            holdertrade_lookup,
            repurchase_lookup,
            top10fh_panel,
            top_inst_lookup,
        )
    raise ReplayContractError(
        f"未知 OOS 特征源: {feature_source!r}"
        f"（合法值 {FEATURE_SOURCE_PANEL!r} / {FEATURE_SOURCE_CS_TRAIN_VERIFY!r}）"
    )


def _build_replay_trading_config(
    model_version: int,
    label_column: str,
    bt_top_n: int,
    bt_rebalance_freq: Optional[int],
    bt_exclude_st: bool,
    bt_min_list_days: int,
    bt_sell_timing: str,
    bt_max_weight_per_stock: Optional[float],
    bt_max_per_industry: Optional[int],
    bt_stop_loss_enabled: bool,
    bt_stop_loss_drawdown_pct: float,
    bt_stop_loss_consecutive_limit_down: int,
    position_sizing: str,
    kelly_vol_window: int,
    kelly_max_leverage: float,
    stagger_tranches: int,
    enable_early_rebalance_on_empty: bool,
    initial_capital: float,
) -> TradingConfig:
    """回测侧 TradingConfig 装配（D8 提取自 run_oos_backtest；参数映射逐字保留）。"""
    return TradingConfig(
        model_version=model_version,
        top_n=bt_top_n,
        rebalance_freq=(
            bt_rebalance_freq
            if bt_rebalance_freq is not None
            else infer_rebalance_freq_from_label(label_column)
        ),
        stagger_tranches=stagger_tranches,
        max_per_industry=bt_max_per_industry,
        max_weight_per_stock=bt_max_weight_per_stock,
        enable_early_rebalance_on_empty=enable_early_rebalance_on_empty,
        exclude_st=bt_exclude_st,
        min_list_days=bt_min_list_days,
        stop_loss_enabled=bt_stop_loss_enabled,
        stop_loss_drawdown_pct=bt_stop_loss_drawdown_pct,
        stop_loss_consecutive_limit_down=bt_stop_loss_consecutive_limit_down,
        position_sizing=position_sizing,
        kelly_vol_window=kelly_vol_window,
        kelly_max_leverage=kelly_max_leverage,
        initial_capital=initial_capital,
        sell_price=bt_sell_timing,
    )


def _assemble_replay_engine(
    effective_config: TradingConfig,
    stock_basic: pd.DataFrame,
    stock_domain: str,
    data_root: str,
    persistent_signal,
    features_by_date: Dict[str, pd.DataFrame],
    storage: Storage,
    initial_capital: float,
    bt_sell_timing: str,
    bt_exclude_st: bool,
    bt_min_list_days: int,
    merger_mode: Optional[str] = None,
):
    """universe / signal / 引擎装配（D8 提取自 run_oos_backtest；调用顺序逐字保留）。

    侧效应：``engine.record_holdings_snapshot = True``（快照是 runs 产物数据源）。
    P2b-T3：合并适配器只在引擎构造完成后按显式模式装配（D4/D6）。
    """
    universe = BasicUniverse(
        stock_basic=stock_basic,
        exclude_st=bt_exclude_st,
        min_list_days=bt_min_list_days,
        # 域市场白名单（universe/domains.py 单一来源；默认主板与历史行为逐位一致；
        # main_gem 放开创业板；市值上下限在 MLSignal 侧过滤）
        markets=domain_market_whitelist(stock_domain),
        verbose=False,
    )
    signal = create_or_reuse_signal(
        effective_config,
        data_root=data_root,
        persistent_signal=persistent_signal,
        verbose=False,
    )
    engine = create_backtest_engine_from_config(
        trading_config=effective_config,
        universe=universe,
        signal=signal,
        features_by_date=features_by_date,
        stock_basic=stock_basic,
        data_storage=storage,
        initial_capital=initial_capital,
        sell_timing=bt_sell_timing,
        verbose=False,
        completion_window_days=5,
        enable_pending_order=True,
    )
    configure_merger_mode(engine, merger_mode)
    # 政策旁路（terminal_loss P2-1）：开启逐日持仓快照（只读，不影响任何决策）
    engine.record_holdings_snapshot = True
    # D6 摘除：旧侧此处的 engine.set_exposure_table(...)（None = 不启用，B0 下 no-op
    # 守卫）与 engine.set_exposure_policy(...) 条件调用已删除；exposure 统计日志块
    # 同步删除（exposure 引擎属性不再存在）。
    return engine


def _compute_replay_metrics(
    nav_curve: pd.DataFrame,
    bt_start: str,
    bt_end: str,
    bt_top_n: int,
    split_num: Optional[int],
) -> Dict:
    """组合级绩效指标计算与结果日志（D8 提取自 run_oos_backtest 尾段，逐字保留）。"""
    total_return = nav_curve["return"].iloc[-1]
    nav_values = nav_curve["nav"].values
    cumulative_max = pd.Series(nav_values).cummax()
    max_drawdown = ((pd.Series(nav_values) - cumulative_max) / cumulative_max).min()
    trading_days = len(nav_curve)
    years = trading_days / 252
    annual_return = total_return / years if years > 0 else 0
    daily_returns = nav_curve["nav"].pct_change(fill_method=None).dropna()
    volatility = daily_returns.std() * (252**0.5)
    sharpe = (annual_return - 0.03) / volatility if volatility > 0 else 0
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    metrics = {
        "bt_total_return": round(total_return, 6),
        "bt_annual_return": round(annual_return, 6),
        "bt_max_drawdown": round(max_drawdown, 6),
        "bt_volatility": round(volatility, 6),
        "bt_sharpe": round(sharpe, 4),
        "bt_calmar": round(calmar, 4),
        "bt_trading_days": trading_days,
        "bt_start": bt_start,
        "bt_end": bt_end,
        "bt_top_n": bt_top_n,
    }
    split_tag = f"Split {split_num} | " if split_num is not None else ""
    logger.info(
        f"\n{'#' * 80}\n"
        f"{split_tag}{bt_start}-{bt_end}\n"
        f"OOS回测结果: 总收益={total_return*100:.2f}%, "
        f"年化={annual_return*100:.2f}%, "
        f"最大回撤={max_drawdown*100:.2f}%, "
        f"夏普={sharpe:.2f} \n"
        f"{'#' * 80}\n\n"
    )
    return metrics


def _run_replay_engine_and_collect(
    engine: Any,
    bt_start: str,
    bt_end: str,
    trade_dates: list,
    price_data: pd.DataFrame,
    bt_top_n: int,
    split_num: Optional[int],
) -> Dict:
    """引擎运行 + 指标与下划线键产物挂载(P3-T3 提取自 run_oos_backtest 尾段,逐字)。

    返回空 dict = 旧侧「净值曲线为空跳过」早退触发条件。
    """
    nav_curve = engine.run(
        start_date=pd.Timestamp(bt_start),
        end_date=pd.Timestamp(bt_end),
        trading_dates=[pd.Timestamp(date) for date in trade_dates],
        price_data=price_data,
    )
    if nav_curve is None or nav_curve.empty or "nav" not in nav_curve.columns:
        logger.warning("OOS回测: 净值曲线为空，跳过")
        return {}

    metrics = _compute_replay_metrics(nav_curve, bt_start, bt_end, bt_top_n, split_num)
    metrics["_nav_curve"] = nav_curve
    metrics["_trades"] = engine.get_trades()
    metrics["_execution_attribution"] = engine.get_execution_attribution()
    metrics["_holdings_snapshot"] = engine.get_holdings_snapshot()
    metrics["_merger_call_count"] = get_merger_call_count(engine)
    return metrics


def run_oos_backtest(
    model_version: int,
    bt_start: str,
    bt_end: str,
    storage: Storage,
    loader: DataLoader,
    trade_cal: pd.DataFrame,
    stock_basic: pd.DataFrame,
    label_column: str,
    bt_top_n: int = 30,
    bt_rebalance_freq: Optional[int] = None,
    data_root: Optional[str] = None,
    persistent_signal=None,
    bt_exclude_st: bool = True,
    bt_min_list_days: int = 365,
    bt_sell_timing: str = "open",
    bt_max_weight_per_stock: Optional[float] = None,
    bt_max_per_industry: Optional[int] = None,
    bt_stop_loss_enabled: bool = False,
    bt_stop_loss_drawdown_pct: float = 30.0,
    bt_stop_loss_consecutive_limit_down: int = 2,
    position_sizing: str = "equal",
    kelly_vol_window: int = 60,
    kelly_max_leverage: float = 0.25,
    stagger_tranches: int = 1,
    enable_early_rebalance_on_empty: bool = True,
    initial_capital: float = 1000000.0,
    split_num: Optional[int] = None,
    exposure_table: Optional[Dict[str, float]] = None,
    exposure_policy: Any = None,
    exposure_replenish: bool = False,
    exposure_trim_tolerance: Optional[float] = None,
    exposure_budget_discount_replenish: bool = False,
    stock_domain: str = "main",
    holdertrade_lookup: Optional[Dict[str, pd.DataFrame]] = None,
    repurchase_lookup: Optional[Dict[str, pd.DataFrame]] = None,
    top10fh_panel: Optional[pd.DataFrame] = None,
    top_inst_lookup: Optional[Dict[str, pd.DataFrame]] = None,
    merger_mode: Optional[str] = None,
    feature_source: str = FEATURE_SOURCE_PANEL,
    store: Any = None,
    acceptance_purpose: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
) -> Dict:
    """对单个 split 模型运行 OOS 回测并返回组合级绩效指标。

    Args:
        holdertrade_lookup: 股东增减持运行时查询表（与训练侧同一张）；提供时逐日
            补齐本族列，避免 MLSignal 因 cs_train 无本族列而静默补 NaN。
            **仅 cs_train_verify 模式可用**；panel 模式四族物化直读，传 ⇒ ValueError。
        repurchase_lookup: 股票回购运行时查询表（与训练侧同一张）；语义同上。
        top10fh_panel: 十大流通股东**报告期面板**（与训练侧同一份）；提供时逐日派生补齐本族列
            （本族逐日全市场稠密 ⇒ 用面板而非逐日字典）。
        merger_mode: P2b 迁移期合并模式；None=旧 P2a 调用，其余合法模式由
            runtime 封闭集合校验，未知值直接失败。
        feature_source: OOS 特征源（P3-T3）：``panel``（默认，store panel 整窗直读，
            缺数据 ⇒ RuntimeError）/ ``cs_train_verify``（§3.1 例外验收源，
            白名单守卫 + 必须声明 ``acceptance_purpose``）。
        store: panel 模式必填的 PanelDataStore（只读消费；读取回执随 store 产出）。
        acceptance_purpose: cs_train_verify 模式必填（回归边 = ``regression_edge``）。
        feature_columns: panel 模式列集收窄（P3-T3 P2-4）：折模型实际特征列;
            None ⇒ 全登记列（回放工具兼容路径）。
    """
    validate_merger_mode(merger_mode)
    _fail_fast_retired_exposure(
        exposure_table,
        exposure_policy,
        exposure_replenish,
        exposure_trim_tolerance,
        exposure_budget_discount_replenish,
    )
    data_root = data_root or get_data_root()
    logger.info(f"OOS 回测: {bt_start} ~ {bt_end}（模型 v{model_version}, Top{bt_top_n}）")

    price_data = _load_replay_price_data(loader, bt_start, bt_end)
    if price_data is None:
        return {}

    trade_dates = trade_cal[
        (trade_cal["cal_date"] >= bt_start)
        & (trade_cal["cal_date"] <= bt_end)
        & (trade_cal["is_open"] == 1)
    ]["cal_date"].tolist()

    features_by_date = _select_replay_features(
        feature_source,
        store,
        acceptance_purpose,
        storage,
        trade_dates,
        bt_start,
        bt_end,
        holdertrade_lookup,
        repurchase_lookup,
        top10fh_panel,
        top_inst_lookup,
        feature_columns,
    )
    if features_by_date is None:
        return {}

    logger.info(f"OOS回测数据: 日线={len(price_data)}条, 特征={len(features_by_date)}日")
    effective_config = _build_replay_trading_config(
        model_version,
        label_column,
        bt_top_n,
        bt_rebalance_freq,
        bt_exclude_st,
        bt_min_list_days,
        bt_sell_timing,
        bt_max_weight_per_stock,
        bt_max_per_industry,
        bt_stop_loss_enabled,
        bt_stop_loss_drawdown_pct,
        bt_stop_loss_consecutive_limit_down,
        position_sizing,
        kelly_vol_window,
        kelly_max_leverage,
        stagger_tranches,
        enable_early_rebalance_on_empty,
        initial_capital,
    )

    engine = _assemble_replay_engine(
        effective_config,
        stock_basic,
        stock_domain,
        data_root,
        persistent_signal,
        features_by_date,
        storage,
        initial_capital,
        bt_sell_timing,
        bt_exclude_st,
        bt_min_list_days,
        merger_mode=merger_mode,
    )

    return _run_replay_engine_and_collect(
        engine, bt_start, bt_end, trade_dates, price_data, bt_top_n, split_num
    )
