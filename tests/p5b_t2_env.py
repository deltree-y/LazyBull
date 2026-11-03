# -*- coding: utf-8 -*-
"""P5b T2 测试环境（tests/ 全量豁免复杂度；不是测试用例模块）。

只造合成历史输入：structural 段（>= 20260703）连续交易日、真实 parquet 的
``clean/daily`` 与 ``features/panel``（PanelDataStore）、模型三件套、扁平纸面配置、
确定性桩信号。**不读生产 data/paper、不联网、不补齐**。

多臂宿主以 ``PaperHost`` 直构造（与 P4 T4 paper_replay 同形），因此 T2 覆盖的是真实宿主
驱动语义（T+1、pending、无成交日估值），而不是桩宿主。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.lazybull.v2.core.signal.base import Signal as _Signal  # noqa: E402

#: 12 个连续交易日（2026-07-06 起）；窗口取前 10 日，后 2 日仅作"窗口外已知交易日"
CALENDAR = tuple(pd.bdate_range("2026-07-06", periods=12).strftime("%Y%m%d"))
#: campaign 虚拟窗口（structural 段内，连续 10 个交易日）
DAYS = CALENDAR[:10]
#: 模型训练截止（严格早于 D0 = 20260706）
TRAIN_END_DATE = "20260630"
#: 合成股票池
CODES = tuple(f"0000{index:02d}.SZ" for index in range(8))
#: 模型特征列（就绪门第 3/4 条判定基准）
FEATURES = ("ret_1", "ret_5", "bp")
#: 选股过滤列（panel 帧必须同带，否则就绪门 fail-closed）
_FILTER_COLUMNS = ("amount_ma20", "total_mv", "sw_l1_code")


class PlanSignal(_Signal):
    """确定性分数信号桩：按日期查表，缺省用 default，过滤到当日股票池。"""

    def __init__(self, top_n: int, plan: Mapping[str, Sequence[Tuple[str, float]]], default):
        super().__init__("p5b_t2_plan_stub")
        self.top_n = top_n
        self._plan = dict(plan)
        self._default = list(default)

    def generate(self, date, universe, data):
        return dict(self.generate_ranked(date, universe, data))

    def generate_ranked(self, date, universe, data):
        ranked = self._plan.get(pd.Timestamp(date).strftime("%Y%m%d"), self._default)
        universe_set = set(universe)
        return [(stock, score) for stock, score in ranked if stock in universe_set]


def plan_for(days: Sequence[str] = DAYS) -> Dict[str, List[Tuple[str, float]]]:
    """重仓日计划：第 1 日与第 6 日各换一批标的（rebalance_freq=5）。"""
    return {
        days[0]: [(CODES[0], 0.9), (CODES[1], 0.8)],
        days[5]: [(CODES[2], 0.95), (CODES[3], 0.85)],
    }


def base_config(version: int = 1, capital: float = 1_000_000.0) -> Dict[str, Any]:
    """最小可用扁平纸面配置（v2 引擎 + 显式冻结模型版本）。"""
    return {
        "model_version": version,
        "top_n": 2,
        "rebalance_freq": 5,
        "stagger_tranches": 1,
        "exclude_st": True,
        "min_list_days": 365,
        "position_sizing": "equal",
        "buy_price": "close",
        "sell_price": "open",
        "initial_capital": capital,
        "min_buy_value_ratio": 0.0,
        "horizon": 20,
        "universe": "mainboard",
        "engine": "v2",
    }


def write_models(data_root: Path, version: int = 1) -> Path:
    """写合成选股模型三件套（与 P4 门 1 身份同后缀与同载荷口径）。"""
    root = data_root / "models" / "stock_selection"
    root.mkdir(parents=True, exist_ok=True)
    (root / f"v{version}_model.joblib").write_bytes(f"joblib-v{version}".encode("utf-8"))
    (root / f"v{version}_features.json").write_text(json.dumps(list(FEATURES)), encoding="utf-8")
    (root / f"v{version}_metadata.json").write_text(
        json.dumps(
            {
                "model_type": "xgboost",
                "train_start_date": "20200101",
                "train_end_date": TRAIN_END_DATE,
                "version": version,
            }
        ),
        encoding="utf-8",
    )
    return root


def _feature_frame(
    day: str,
    *,
    missing: Sequence[str] = (),
    nan_columns: Sequence[str] = (),
    row_subset: Optional[int] = None,
) -> pd.DataFrame:
    count = len(CODES)
    frame = pd.DataFrame(
        {
            "trade_date": [day] * count,
            "ts_code": list(CODES),
            "ret_1": np.linspace(0.1, 0.17, count),
            "ret_5": np.linspace(0.2, 0.27, count),
            "bp": np.linspace(0.01, 0.08, count),
            "amount_ma20": np.full(count, 60000.0),
            "total_mv": np.full(count, 800000.0),
            "sw_l1_code": ["801780"] * count,
        }
    )
    for column in nan_columns:
        if column in frame.columns:
            frame[column] = np.nan
    for column in missing:
        if column in frame.columns:
            frame = frame.drop(columns=[column])
    if row_subset is not None:
        frame = frame.head(int(row_subset)).reset_index(drop=True)
    return frame


def _price_frame(
    day: str,
    *,
    limit_up: Sequence[str] = (),
    limit_down: Sequence[str] = (),
    untradable: Sequence[str] = (),
) -> pd.DataFrame:
    count = len(CODES)
    closes = [10.0 * (1 + 0.1 * index) for index in range(count)]
    blocked = set(untradable)
    return pd.DataFrame(
        {
            "trade_date": [day] * count,
            "ts_code": list(CODES),
            "close": closes,
            "close_adj": closes,
            "open": closes,
            "open_adj": closes,
            "is_suspended": [1 if code in blocked else 0 for code in CODES],
            "is_limit_up": [1 if code in set(limit_up) else 0 for code in CODES],
            "is_limit_down": [1 if code in set(limit_down) else 0 for code in CODES],
            "vol": np.full(count, 1000.0),
            "pct_chg": np.full(count, 0.0),
            "is_st": [0] * count,
            "list_days": [1000] * count,
            "tradable": [0 if code in blocked else 1 for code in CODES],
        }
    )


def write_market_data(
    data_root: Path,
    days: Sequence[str],
    *,
    limit_up: Optional[Mapping[str, Sequence[str]]] = None,
    limit_down: Optional[Mapping[str, Sequence[str]]] = None,
    untradable: Optional[Mapping[str, Sequence[str]]] = None,
) -> None:
    """写 ``clean/daily`` 日分区（真实 parquet：宿主行情窗口与身份门共用同一份文件）。"""
    daily_dir = data_root / "clean" / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    limits = dict(limit_up or {})
    bottoms = dict(limit_down or {})
    blocked = dict(untradable or {})
    for day in days:
        frame = _price_frame(
            day,
            limit_up=limits.get(day, ()),
            limit_down=bottoms.get(day, ()),
            untradable=blocked.get(day, ()),
        )
        frame.to_parquet(daily_dir / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet", index=False)


def write_panel(
    data_root: Path,
    days: Sequence[str],
    *,
    missing: Optional[Mapping[str, Sequence[str]]] = None,
    nan_columns: Optional[Mapping[str, Sequence[str]]] = None,
    row_subset: Optional[Mapping[str, int]] = None,
) -> None:
    """写 ``features/panel/<date>/core.parquet`` 并登记 manifest（行域由 store 派生）。

    ``missing``/``nan_columns``/``row_subset`` 可按日制造缺陷（用于构造就绪门 ``not_ready``
    场景：分区仍在、行域回执仍通过，但行数带/列缺失率不过）。
    """
    from src.lazybull.v2.common.types import TradeDate
    from src.lazybull.v2.store.data_store import PanelDataStore
    from src.lazybull.v2.store.panel_builder import bootstrap_manifest

    dropped = dict(missing or {})
    blanked = dict(nan_columns or {})
    subset = dict(row_subset or {})
    store = PanelDataStore(data_root)
    bootstrap_manifest(store)
    for day in days:
        store.append_features(
            TradeDate.from_str(day),
            "core",
            _feature_frame(
                day,
                missing=dropped.get(day, ()),
                nan_columns=blanked.get(day, ()),
                row_subset=subset.get(day),
            ),
        )


def stock_basic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": list(CODES),
            "name": [f"股票{index}" for index in range(len(CODES))],
            "market": ["主板"] * len(CODES),
            "list_date": ["20200101"] * len(CODES),
        }
    )


def build_env(
    tmp_path: Path,
    *,
    days: Sequence[str] = DAYS,
    calendar: Sequence[str] = CALENDAR,
    config: Optional[Mapping[str, Any]] = None,
    limit_up: Optional[Mapping[str, Sequence[str]]] = None,
    limit_down: Optional[Mapping[str, Sequence[str]]] = None,
    untradable: Optional[Mapping[str, Sequence[str]]] = None,
    missing_columns: Optional[Mapping[str, Sequence[str]]] = None,
    nan_columns: Optional[Mapping[str, Sequence[str]]] = None,
    row_subset: Optional[Mapping[str, int]] = None,
) -> SimpleNamespace:
    """构造隔离环境：数据根 + 模型根 + 试车场根 + 仓库根 + 日历（窗口 + 已知后续日）。"""
    data_root = tmp_path / "data"
    write_market_data(
        data_root, days, limit_up=limit_up, limit_down=limit_down, untradable=untradable
    )
    write_panel(
        data_root, days, missing=missing_columns, nan_columns=nan_columns, row_subset=row_subset
    )
    models_dir = write_models(data_root)
    return SimpleNamespace(
        data_root=data_root,
        models_dir=models_dir,
        base_root=tmp_path / "testbed",
        repo_root=ROOT,
        days=tuple(days),
        calendar=tuple(calendar),
        config=dict(config or base_config()),
        limit_up=dict(limit_up or {}),
        limit_down=dict(limit_down or {}),
        untradable=dict(untradable or {}),
        missing_columns=dict(missing_columns or {}),
        nan_columns=dict(nan_columns or {}),
        row_subset=dict(row_subset or {}),
    )


def prepare_campaign(
    env: SimpleNamespace,
    *,
    campaign_id: str = "t2_campaign",
    arms: Sequence[str] = ("aa_1", "aa_2"),
    roles: Optional[Mapping[str, str]] = None,
    configs: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> SimpleNamespace:
    """登记 campaign + 臂 + D0 fresh，返回 campaign 根与记录。"""
    from src.lazybull.v2.hosts.testbed import (
        initialize_d0_fresh,
        register_arm,
        register_campaign,
    )

    campaign = register_campaign(
        campaign_id=campaign_id,
        d0_date=env.days[0],
        virtual_days=env.days,
        calendar=env.calendar,
        data_root=str(env.data_root),
        base_root=str(env.base_root),
        repo_root=str(env.repo_root),
        allow_external_root=True,
    )
    root = Path(campaign["campaign_root"])
    records: Dict[str, Any] = {}
    for arm_id in arms:
        role = str((roles or {}).get(arm_id, "aa_clone"))
        flat = dict((configs or {}).get(arm_id, env.config))
        records[arm_id] = register_arm(
            campaign_root=root,
            arm_id=arm_id,
            arm_role=role,
            config_flat=flat,
            models_dir=str(env.models_dir),
            repo_root=str(env.repo_root),
            allow_external_root=True,
        )
    d0_record = initialize_d0_fresh(root, allow_external_root=True)
    return SimpleNamespace(campaign=campaign, root=root, arms=records, d0=d0_record)


def make_host_factory(
    env: SimpleNamespace,
    *,
    plan: Optional[Mapping[str, Sequence[Tuple[str, float]]]] = None,
):
    """``HostFactory``：每个虚拟日按臂的暂存根直构造真实 ``PaperHost``（固定 replay）。"""
    from src.lazybull.data import DataLoader, Storage
    from src.lazybull.universe import BasicUniverse
    from src.lazybull.v2.common.cost import CostModel
    from src.lazybull.v2.hosts.paper.config_adapter import to_engine_kwargs, to_trading_config
    from src.lazybull.v2.hosts.paper.host import PaperHost
    from src.lazybull.v2.hosts.paper.state_store import PaperStateStore
    from src.lazybull.v2.store.data_store import PanelDataStore
    from tests.test_v2_p4_t2_host import _coverage_train_params

    calendars = [pd.Timestamp(day) for day in env.calendar]
    window = [pd.Timestamp(day) for day in env.days]
    signal_plan = dict(plan or plan_for(env.days))
    default_plan = signal_plan[env.days[0]]
    stock_basic = stock_basic_frame()
    costs = CostModel(commission_rate=0.0003, min_commission=5.0, stamp_tax=0.001, slippage=0.0)

    def factory(spec, store_root: Path) -> PaperHost:
        flat = dict(spec.config_flat)
        config = to_trading_config(flat)
        universe = BasicUniverse(
            stock_basic,
            exclude_st=bool(config.exclude_st),
            min_list_days=int(config.min_list_days),
            markets=["主板"],
            verbose=False,
        )
        storage = Storage(root_path=str(env.data_root))
        return PaperHost(
            store=PaperStateStore(root_path=str(store_root)),
            trading_config=config,
            universe=universe,
            signal=PlanSignal(int(config.top_n), signal_plan, default_plan),
            cost_model=costs,
            stock_basic=stock_basic,
            data_storage=storage,
            panel_store=PanelDataStore(env.data_root),
            trading_calendar=calendars,
            engine_kwargs=to_engine_kwargs(flat, config),
            feature_columns=list(FEATURES),
            train_params=_coverage_train_params(list(FEATURES)),
            data_loader=DataLoader(storage=storage, verbose=False),
            schedule_mode="replay",
            replay_calendar=window,
            verbose=False,
        )

    return factory


def run_campaign(
    env: SimpleNamespace,
    prepared: SimpleNamespace,
    *,
    days: Optional[Sequence[str]] = None,
    fault_hook=None,
    write_report: bool = False,
    plan: Optional[Mapping[str, Sequence[Tuple[str, float]]]] = None,
) -> Dict[str, Any]:
    """以合成宿主工厂运行 campaign（T2 测试统一入口）。"""
    from src.lazybull.v2.hosts.testbed import run_virtual_campaign

    return run_virtual_campaign(
        prepared.root,
        days=days,
        host_factory=make_host_factory(env, plan=plan),
        fault_hook=fault_hook,
        write_report=write_report,
        allow_external_root=True,
    )


def arm_state(prepared: SimpleNamespace, arm_id: str) -> Any:
    """读取某臂权威状态（断言用；只读）。"""
    from src.lazybull.v2.hosts.paper.state_store import PaperStateStore

    store = PaperStateStore(root_path=str(prepared.root / "arms" / arm_id), create_layout=False)
    return store.load_engine_state()


def _load(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def shard(prepared: SimpleNamespace, arm_id: str, day: str) -> Dict[str, Any]:
    """读单臂单日分片（T2 落盘的权威逐日证据）。"""
    return _load(prepared.root / "ledger" / arm_id / f"{day}.json")


def journal(prepared: SimpleNamespace, day: str) -> Dict[str, Any]:
    """读当日族级事务 journal。"""
    return _load(prepared.root / "journal" / f"{day}.json")


def index(prepared: SimpleNamespace) -> Dict[str, Any]:
    """读读侧 index（已发布代唯一视图）。"""
    return _load(prepared.root / "_index.json")


def manifest(prepared: SimpleNamespace, arm_id: str) -> Dict[str, Any]:
    """读单臂分片清单。"""
    return _load(prepared.root / "ledger" / arm_id / "manifest.json")


def pending(prepared: SimpleNamespace, arm_id: str, queue: str = "buys") -> List[Dict[str, Any]]:
    """读 pending 队列投影（``queue`` ∈ {buys, sells}）。"""
    path = prepared.root / "arms" / arm_id / f"pending_{queue}" / f"pending_{queue}.json"
    return _load(path)
