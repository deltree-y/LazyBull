# -*- coding: utf-8 -*-
"""v2 P3-T2 TrainingFrameCache 测试（P3-T2 加载成本门，§16.5-①；评审处置 R-CODEX-01/02）。

不依赖真实 store 的用例注入假 ``frame_loader``（返回 SimpleNamespace 哨兵）+ spy 计数；
守卫回归与真实集成用例使用合成数据根（tmp_path，helper 与
``tests/test_v2_train_data_adapter.py`` 同源简化版——tests 目录非包无法 import，故自建）。

覆盖：
- 同键 3 次 get_or_load ⇒ frame_loader 恰调用 1 次且三次返回同一对象（is）；
- 异键各自加载：窗口 / 列集 / label_column / source_mode / frozen_through / universe /
  acceptance_purpose / 源身份（store.root 不同）逐维变化 ⇒ 调用次数递增；
- clear() 后重加载 ⇒ 再次调用 frame_loader；
- runtime_families 集合顺序无关（frozenset 不同构造顺序同键）；
- 命中守卫回归（R-CODEX-02）：同键非法请求（panel 携带 runtime_lookups / cs_train 启用族
  缺 lookup）命中路径同样抛 ValueError，与直接 load_training_frame 同错；
- 真实集成（R-CODEX-01）：真实 Storage/DataLoader + 默认 load_training_frame，同一 config
  连调 3 次 ⇒ 同一对象、计数 frame_loader 恰 1 次、与直接加载逐值一致（check_exact）。
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pandas as pd
import pytest

from src.lazybull.v2.common.types import TradeDate
from src.lazybull.v2.store.column_groups import (
    LABEL_TABLES,
    LABEL_VALUE_COLUMNS,
    PANEL_GROUPS,
)
from src.lazybull.v2.store.data_store import PanelDataStore
from src.lazybull.v2.train.data_adapter import (
    SOURCE_CS_TRAIN_VERIFY,
    AdapterConfig,
    load_training_frame,
)
from src.lazybull.v2.train.frame_cache import TrainingFrameCache


def _config(**overrides) -> AdapterConfig:
    """基准 panel 配置（AdapterConfig 构造不触发加载期校验，缓存键维度可自由取值）。"""
    params: Dict[str, Any] = {
        "start_date": "20240102",
        "end_date": "20240628",
        "feature_columns": ("ret_1", "turnover_rate"),
        "label_column": "neu_y_ret_20",
    }
    params.update(overrides)
    return AdapterConfig(**params)


def _spy_loader(calls: List[Tuple[AdapterConfig, Dict[str, Any]]]):
    """spy frame_loader：记录调用并逐次返回新的 SimpleNamespace 哨兵对象。"""

    def frame_loader(config: AdapterConfig, **kwargs):
        calls.append((config, kwargs))
        return SimpleNamespace(load_seq=len(calls))

    return frame_loader


#: panel 模式的合法 loader_kwargs（守卫要求 store 非 None；SimpleNamespace 仅有 root 属性即可）
def _panel_kwargs(root: str = "base") -> Dict[str, Any]:
    return {"store": SimpleNamespace(root=root)}


#: cs_train 验证源模式的合法 loader_kwargs（守卫要求 storage/loader/acceptance_purpose）
def _cs_train_kwargs(root: str = "cs") -> Dict[str, Any]:
    return {
        "storage": SimpleNamespace(root_path=root),
        "loader": object(),
        "acceptance_purpose": "g_code_parity",
    }


# ---------------------------------------------------------------------------
# 合成数据根 helper（与 tests/test_v2_train_data_adapter.py 同源的简化版）
# ---------------------------------------------------------------------------

_CODES = ["600000.SH", "000001.SZ", "000002.SZ"]
_FEATURE_COLUMNS = ("ret_1", "turnover_rate")
#: 标签分区协议列序（label_refresh._LABEL_COLUMNS；readiness 强校验）
_LABEL_COLUMNS = ["ts_code", "trade_date", "label_value", "neu_label_value", "maturity_status"]


def _calendar_days(n: int = 70, start: str = "2026-01-05") -> list:
    return pd.date_range(start, periods=n, freq="B").strftime("%Y%m%d").tolist()


def _write_calendar(root: Path, days: list) -> None:
    clean = root / "clean"
    clean.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"cal_date": days, "is_open": [1] * len(days)}).to_parquet(
        clean / "trade_cal.parquet", index=False
    )


def _write_daily_partitions(root: Path, days: list) -> None:
    daily = root / "clean" / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    for day in days:
        pd.DataFrame({"trade_date": [day], "ts_code": [_CODES[0]], "close": [1.0]}).to_parquet(
            daily / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet", index=False
        )


def _make_store(root: Path) -> PanelDataStore:
    store = PanelDataStore(root)
    store.manifest.register_columns(
        {col: {"group": "core", "source": "test"} for col in _FEATURE_COLUMNS}
    )
    store.manifest.save()
    return store


def _append_panel_day(store: PanelDataStore, day: str, value: float = 0.01) -> None:
    for group in PANEL_GROUPS:
        payload: dict = {"trade_date": [day] * len(_CODES), "ts_code": list(_CODES)}
        if group == "core":
            for col in _FEATURE_COLUMNS:
                payload[col] = [value + i * 0.001 for i in range(len(_CODES))]
        store.append_features(TradeDate.from_str(day), group, pd.DataFrame(payload))


def _append_label_day(store: PanelDataStore, day: str, maturity: str = "sealed") -> None:
    for offset, (name, _) in enumerate(LABEL_TABLES.items()):
        frame = pd.DataFrame(
            {
                "ts_code": list(_CODES),
                "trade_date": [day] * len(_CODES),
                "label_value": [0.01 * (offset + 1)] * len(_CODES),
                "neu_label_value": [0.02 * (offset + 1)] * len(_CODES),
                "maturity_status": [maturity] * len(_CODES),
            },
            columns=_LABEL_COLUMNS,
        )
        store.append_labels(name, frame)


def _build_panel_root(tmp_path: Path, days: list) -> PanelDataStore:
    """最小 panel 数据根：日历 + clean/daily 水位分区 + panel 八族 + 三表标签。"""
    calendar = _calendar_days()
    _write_calendar(tmp_path, calendar)
    _write_daily_partitions(tmp_path, calendar)
    store = _make_store(tmp_path)
    for day in days:
        _append_panel_day(store, day)
        _append_label_day(store, day)
    return store


def _panel_config(days: list, **overrides) -> AdapterConfig:
    params: Dict[str, Any] = {
        "start_date": days[0],
        "end_date": days[-1],
        "feature_columns": _FEATURE_COLUMNS,
    }
    params.update(overrides)
    return AdapterConfig(**params)


def _write_cs_train_day(root: Path, day: str) -> None:
    cs_train = root / "features" / "cs_train"
    cs_train.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, list] = {
        "trade_date": [day] * len(_CODES),
        "ts_code": list(_CODES),
        "ret_1": [0.01, 0.02, 0.03],
        "turnover_rate": [1.1, 1.2, 1.3],
    }
    for col in LABEL_VALUE_COLUMNS:
        payload[col] = [0.05, 0.04, -0.02]
    pd.DataFrame(payload).to_parquet(cs_train / f"{day}.parquet", index=False)


def _build_cs_train_root(tmp_path: Path, days: list) -> None:
    """最小 cs_train 数据根：日历 + clean/daily 水位分区 + cs_train 日分区。"""
    calendar = _calendar_days()
    _write_calendar(tmp_path, calendar)
    _write_daily_partitions(tmp_path, calendar[:30])
    for day in days:
        _write_cs_train_day(tmp_path, day)


def _cs_train_config(days: list, **overrides) -> AdapterConfig:
    params: Dict[str, Any] = {
        "start_date": days[0],
        "end_date": days[-1],
        "source_mode": SOURCE_CS_TRAIN_VERIFY,
    }
    params.update(overrides)
    return AdapterConfig(**params)


def _old_path_objects(root: Path):
    """旧 cs_train 链路对象（验收专用过渡依赖）。"""
    from src.lazybull.data.loader import DataLoader
    from src.lazybull.data.storage import Storage

    storage = Storage(root_path=str(root))
    return storage, DataLoader(storage)


# ---------------------------------------------------------------------------
# 纯缓存键用例（假 frame_loader + spy）
# ---------------------------------------------------------------------------


class TestTrainingFrameCache:
    def test_same_key_loads_once_and_returns_same_object(self):
        """同键 3 次 get_or_load ⇒ frame_loader 恰 1 次、三次返回同一对象（is）。"""
        cache = TrainingFrameCache()
        calls: List = []
        frame_loader = _spy_loader(calls)
        config = _config()
        kwargs = _panel_kwargs()

        first = cache.get_or_load(config, frame_loader=frame_loader, **kwargs)
        second = cache.get_or_load(config, frame_loader=frame_loader, **kwargs)
        third = cache.get_or_load(config, frame_loader=frame_loader, **kwargs)

        assert len(calls) == 1
        assert first is second is third
        assert len(cache) == 1

    def test_distinct_keys_load_separately(self):
        """异键各自加载：逐维变化缓存键各组成部分 ⇒ 调用次数递增。"""
        cache = TrainingFrameCache()
        calls: List = []
        frame_loader = _spy_loader(calls)
        base = _config()
        base_kwargs = _panel_kwargs()

        variants: List[Tuple[AdapterConfig, Dict[str, Any]]] = [
            (_config(start_date="20240103"), _panel_kwargs()),  # 窗口起点
            (_config(end_date="20240627"), _panel_kwargs()),  # 窗口终点
            (_config(feature_columns=("ret_1",)), _panel_kwargs()),  # 列集
            (_config(label_column="y_ret_20"), _panel_kwargs()),  # 标签列
            (
                # 数据源模式（cs_train 验证源禁传列集 ⇒ 列集置空）
                _config(source_mode=SOURCE_CS_TRAIN_VERIFY, feature_columns=()),
                _cs_train_kwargs(),
            ),
            (_config(frozen_through="20260601"), _panel_kwargs()),  # 冻结上界
            (_config(universe=("000001.SZ",)), _panel_kwargs()),  # 域（None vs 单代码）
            (base, {**_panel_kwargs(), "acceptance_purpose": "g_code_parity"}),  # 验收用途
            (base, _panel_kwargs(root="a")),  # 源身份 a
            (base, _panel_kwargs(root="b")),  # 源身份 b
        ]

        cache.get_or_load(base, frame_loader=frame_loader, **base_kwargs)
        assert len(calls) == 1
        for expected_calls, (config, kwargs) in enumerate(variants, start=2):
            cache.get_or_load(config, frame_loader=frame_loader, **kwargs)
            assert len(calls) == expected_calls, f"变体 {config} / {kwargs} 未触发新加载"

        # 全部键重复请求均命中，不再新增加载
        cache.get_or_load(base, frame_loader=frame_loader, **base_kwargs)
        for config, kwargs in variants:
            cache.get_or_load(config, frame_loader=frame_loader, **kwargs)
        assert len(calls) == 1 + len(variants)
        assert len(cache) == 1 + len(variants)

    def test_clear_triggers_reload(self):
        """clear() 后重加载 ⇒ 再次调用 frame_loader 且返回新对象。"""
        cache = TrainingFrameCache()
        calls: List = []
        frame_loader = _spy_loader(calls)
        config = _config()
        kwargs = _panel_kwargs()

        first = cache.get_or_load(config, frame_loader=frame_loader, **kwargs)
        assert len(cache) == 1
        cache.clear()
        assert len(cache) == 0
        second = cache.get_or_load(config, frame_loader=frame_loader, **kwargs)

        assert len(calls) == 2
        assert first is not second

    def test_runtime_families_order_insensitive(self):
        """runtime_families 集合顺序无关：frozenset 不同构造顺序 ⇒ 同键。"""
        cache = TrainingFrameCache()
        calls: List = []
        frame_loader = _spy_loader(calls)
        # panel 模式禁 derive，用 cs_train 验证源模式携带四族（守卫要求同源 lookup）
        families_ab = frozenset({"holdertrade", "repurchase"})
        families_ba = frozenset({"repurchase", "holdertrade"})
        lookups = {"holdertrade_lookup": {}, "repurchase_lookup": {}}
        config_ab = _config(
            source_mode=SOURCE_CS_TRAIN_VERIFY,
            feature_columns=(),
            runtime_families=families_ab,
            runtime_lookups=lookups,
        )
        config_ba = _config(
            source_mode=SOURCE_CS_TRAIN_VERIFY,
            feature_columns=(),
            runtime_families=families_ba,
            runtime_lookups=lookups,
        )
        kwargs = _cs_train_kwargs()

        first = cache.get_or_load(config_ab, frame_loader=frame_loader, **kwargs)
        second = cache.get_or_load(config_ba, frame_loader=frame_loader, **kwargs)

        assert len(calls) == 1
        assert first is second


# ---------------------------------------------------------------------------
# 命中守卫回归（R-CODEX-02）：同键非法请求命中也必须报错
# ---------------------------------------------------------------------------


class TestHitGuardRegression:
    def test_panel_same_key_runtime_lookups_rejected_on_hit(self, tmp_path):
        """panel 模式携带 runtime_lookups（禁 derive 双写）：缓存命中路径同样 ValueError。"""
        days = ["20260105"]
        store = _build_panel_root(tmp_path, days)
        config = _panel_config(days)
        cache = TrainingFrameCache()

        # 合法 panel 配置建立缓存（runtime_lookups 不进缓存键 ⇒ 下方非法配置同键）
        frame = cache.get_or_load(config, store=store)
        assert len(cache) == 1

        bad_config = _panel_config(days, runtime_lookups={"holdertrade_lookup": {}})
        with pytest.raises(ValueError, match="禁止再走 derive"):
            cache.get_or_load(bad_config, store=store)
        # 对照：同配置直接 load_training_frame 抛同样的错
        with pytest.raises(ValueError, match="禁止再走 derive"):
            load_training_frame(bad_config, store=store)

        # 缓存未被污染：合法同键仍命中原帧
        assert len(cache) == 1
        assert cache.get_or_load(config, store=store) is frame

    def test_cs_train_same_key_missing_lookup_rejected_on_hit(self, tmp_path):
        """cs_train 模式启用族缺 lookup（启用而未传必须报错）：命中路径同样 ValueError。"""
        days = ["20260105", "20260106", "20260107"]
        _build_cs_train_root(tmp_path, days)
        storage, loader = _old_path_objects(tmp_path)
        kwargs = {"storage": storage, "loader": loader, "acceptance_purpose": "g_code_parity"}
        cache = TrainingFrameCache()

        # 合法：启用 holdertrade 族并传同源 lookup（runtime_lookups 不进缓存键 ⇒ 同键）
        config = _cs_train_config(
            days,
            runtime_families=frozenset({"holdertrade"}),
            runtime_lookups={"holdertrade_lookup": {}},
        )
        frame = cache.get_or_load(config, **kwargs)
        assert len(cache) == 1

        # 同键非法：启用族但缺 lookup
        bad_config = _cs_train_config(days, runtime_families=frozenset({"holdertrade"}))
        with pytest.raises(ValueError, match="holdertrade_lookup"):
            cache.get_or_load(bad_config, **kwargs)
        with pytest.raises(ValueError, match="holdertrade_lookup"):
            load_training_frame(bad_config, **kwargs)

        assert len(cache) == 1
        assert cache.get_or_load(config, **kwargs) is frame


# ---------------------------------------------------------------------------
# 真实集成（R-CODEX-01）：真实 Storage/DataLoader + 默认 load_training_frame
# ---------------------------------------------------------------------------


class TestRealCsTrainIntegration:
    def test_real_loader_single_load_and_value_parity(self, tmp_path):
        """同一 config 连调 3 次：同一对象、计数 frame_loader 恰 1 次、与直接加载逐值一致。"""
        days = ["20260105", "20260106", "20260107"]
        _build_cs_train_root(tmp_path, days)
        storage, loader = _old_path_objects(tmp_path)
        config = _cs_train_config(days)
        kwargs = {"storage": storage, "loader": loader, "acceptance_purpose": "g_code_parity"}
        cache = TrainingFrameCache()
        calls: List = []

        def counting_frame_loader(cfg: AdapterConfig, **loader_kwargs):
            """包装真实 load_training_frame 并计数（观测真实加载次数）。"""
            calls.append(cfg)
            return load_training_frame(cfg, **loader_kwargs)

        first = cache.get_or_load(config, frame_loader=counting_frame_loader, **kwargs)
        second = cache.get_or_load(config, frame_loader=counting_frame_loader, **kwargs)
        third = cache.get_or_load(config, frame_loader=counting_frame_loader, **kwargs)

        assert first is second is third
        assert len(calls) == 1, "同一折 TrainingFrame 应只真实加载一次"

        # 缓存帧与直接 load_training_frame 读取逐值一致
        direct = load_training_frame(config, **kwargs)
        pd.testing.assert_frame_equal(first.df, direct.df, check_exact=True)
        assert first.trade_days == direct.trade_days == len(days)
        assert first.source_mode == direct.source_mode == SOURCE_CS_TRAIN_VERIFY
