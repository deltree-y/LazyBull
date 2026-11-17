"""固定续持研究的只读输入桥与原生连续回放，不进入生产。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.8）。
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.v2_p7.buffer_path_data import (
    SOURCE,
    TOPK_COLUMNS,
    _capture_identity,
    _reference_topk,
    validate_rankings,
)
from scripts.v2_p7.t25.params import build_params
from src.lazybull.data.loader import DataLoader
from src.lazybull.data.storage import Storage
from src.lazybull.universe import BasicUniverse
from src.lazybull.v2.common.cost import CostModel
from src.lazybull.v2.core.decision.portfolio_merger import SingleSleevePortfolioMerger

SPEC_SHA = "7f83ea0693587ff7a8e548a610fbb12a92d04086d480c91e25d667765fca930f"
OLD_SPEC_SHA = "208ff4dace85928672f381e6eb38f113d36d76c6402ec4804a957106208f9add"
OLD_DATA_BEFORE = "c8fc34b23bf1a87e52dbdf3a211c48575a333346a361d3a508353a4e81b4e57d"
OLD_DATA_AFTER = "8bcaae74f46212548f62f75b8878a4bd7d5613b14e60f72d8571a1f19cfbd796"


def sha256(path: Path) -> str:
    """按字节登记文件。"""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def plain(value: Any) -> Any:
    """保留浮点精度、顺序及缺失类型的JSON桥，不把NaN写成非法JSON。"""
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    if value is pd.NA or value is pd.NaT:
        return {"missing": str(value)}
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"证据对象不能转换：{type(value)}")


def canonical(value: Any) -> str:
    """完整精度规范序列，仅排序映射键，不排序行。"""
    return json.dumps(plain(value), ensure_ascii=False, sort_keys=True, allow_nan=False)


def write_json(path: Path, value: Any) -> None:
    """正式文件只创建，不覆盖。"""
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(plain(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read_historical_rankings(root: Path) -> tuple[dict, dict]:
    """继承旧完整排名为冻结数据；唯一定义过的注释漂移显式核验并登记。"""
    cache = root / "data/reports/p7_buffer_path_20261116/input_cache_r1"
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("completed") is not True or manifest.get("spec_sha256") != OLD_SPEC_SHA:
        raise ValueError("旧排名缓存未完成或规格不符")
    if set(manifest["outputs"]) != {"input_identity.json", "ranked.parquet", "validation.json"}:
        raise ValueError("旧排名缓存产物不完整")
    for name, digest in manifest["outputs"].items():
        if sha256(cache / name) != digest:
            raise ValueError(f"旧排名缓存被改写：{name}")
    old = json.loads((cache / "input_identity.json").read_text(encoding="utf-8"))
    params = build_params(root)
    reference, info, files = _reference_topk(root, params)
    current = _capture_identity(root, params, info, files, OLD_SPEC_SHA)
    name = "scripts/v2_p7/buffer_path_data.py"
    before = old["code"]["files"][name]
    after = current["code"]["files"][name]
    frozen_source = root / "data/reports/p7_buffer_path_20261116/run_r1/methods/buffer_path_data.py"
    if (before, after) != (OLD_DATA_BEFORE, OLD_DATA_AFTER):
        raise ValueError("旧排名继承的已登记注释差异身份不符")
    if sha256(frozen_source) != before:
        raise ValueError("旧运行的原始代码快照身份不符")
    ast_before = ast.dump(ast.parse(frozen_source.read_text(encoding="utf-8")))
    ast_after = ast.dump(ast.parse((root / name).read_text(encoding="utf-8")))
    if ast_before != ast_after:
        raise ValueError("旧数据适配器语义变化，不能继承冻结排名")
    normalized = copy.deepcopy(current)
    normalized["code"]["files"][name] = before
    if normalized != old:
        raise ValueError("除已登记注释外，旧排名输入、代码或日历发生变化")
    frame = pd.read_parquet(cache / "ranked.parquet", columns=TOPK_COLUMNS)
    ranked = {day: [] for day in params.signal_days}
    for day, group in frame.groupby("trade_date", sort=False):
        if str(day) not in ranked or group["rank"].tolist() != list(range(1, len(group) + 1)):
            raise ValueError(f"旧全排名日期/序号无效：{day}")
        ranked[str(day)] = list(zip(group.ts_code.astype(str), group.score.astype(float)))
    checks = validate_rankings(
        ranked, reference, params.signal_days, info["closure_days"], info["reference_empty_days"]
    )
    return ranked, {
        "calendar": list(params.extended_days),
        "signal_days": list(params.signal_days),
        "current_identity": current,
        "validation": checks,
        "inherited_cache": cache.relative_to(root).as_posix(),
        "cache_files": {p.name: sha256(p) for p in sorted(cache.iterdir()) if p.is_file()},
        "annotation_lineage": {"path": name, "before": before, "after": after, "ast_equal": True},
        "legacy_reference": SOURCE.as_posix(),
    }


def load_prices(
    root: Path, days: list[str], *, warmup_cutoff: str | None = None
) -> tuple[pd.DataFrame, dict[str, str]]:
    """完整读入登记日，不补齐，不删缺失股票；类别编码不改变数值或行序。"""
    frames, identities = [], {}
    for day in days:
        path = root / "data/clean/daily" / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet"
        identities[path.relative_to(root).as_posix()] = sha256(path)
        frame = pd.read_parquet(path)
        if frame.empty or not frame.trade_date.astype(str).eq(day).all():
            raise ValueError(f"日行情为空或日期不符：{day}")
        if frame.ts_code.duplicated().any():
            raise ValueError(f"日行情股票键重复：{day}")
        frames.append(frame)
    prices = pd.concat(frames, ignore_index=True)
    if warmup_cutoff is not None:
        # 必须是每股实际60行，不用60个日历槽代替停牌股的60行历史。
        before = prices.trade_date.astype(str) < warmup_cutoff
        history = prices.loc[before].groupby("ts_code", sort=False).tail(60)
        prices = pd.concat([history, prices.loc[~before]], ignore_index=True)
    prices["ts_code"] = pd.Categorical(prices.ts_code.astype(str))
    prices["trade_date"] = pd.Categorical(
        prices.trade_date.astype(str), categories=sorted(set(days)), ordered=True
    )
    return prices, identities


class FrozenRankSignal:
    """只传入已校验的完整合法排名；不再推理或改变并列顺序。"""

    top_n = 20

    def __init__(self, ranked: dict[str, list[tuple[str, float]]]) -> None:
        self.ranked = ranked

    def generate_ranked(self, date: pd.Timestamp, universe: Any, data: Any) -> list:
        """尾部退出日允许无排名；排名日已在输入适配器逐日登记。"""
        return list(self.ranked.get(date.strftime("%Y%m%d"), []))

    def generate(self, date: pd.Timestamp, universe: Any, data: Any) -> dict:
        """兼容Signal最小接口，正式内核使用generate_ranked。"""
        return dict(self.generate_ranked(date, universe, data)[: self.top_n])


class RecordNativeMixin:
    """在全部执行结束后取证，不额外调用估值、不修改账户。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.research_daily: list[dict] = []
        self.research_states: list[dict] = []
        super().__init__(*args, **kwargs)

    def run_step(
        self, date: pd.Timestamp, trading_dates_full: list, price_data: pd.DataFrame
    ) -> None:
        """保留原生顺序，旁路只读记录现金、股份、待执行义务。"""
        offset = len(self.trades)
        super().run_step(date, trading_dates_full, price_data)
        snapshot = self.portfolio_values[-1]
        day_trades = self.trades[offset:]
        corporate = getattr(self, "native_corporate", None)
        receivable = corporate.receivable_value() if corporate is not None else 0.0
        value = float(snapshot["portfolio_value"])
        row = {
            "trade_date": date.strftime("%Y%m%d"),
            "nav": value / self.initial_capital,
            "total_value": value,
            "cash": float(self.current_capital),
            "market_value": value - self.current_capital - receivable,
            "receivable": receivable,
            "fees": sum(float(x["cost"]) for x in day_trades),
            "buy_amount": sum(float(x["amount"]) for x in day_trades if x["action"] == "buy"),
            "sell_amount": sum(float(x["amount"]) for x in day_trades if x["action"] == "sell"),
            "positions": len(self.positions),
        }
        if row["cash"] < -1e-7 or any(p["shares"] < 0 for p in self.positions.values()):
            raise ValueError(f"现金/股份非负门失败：{date}")
        self.research_daily.append(row)
        self.research_states.append(
            copy.deepcopy(
                {
                    "trade_date": row["trade_date"],
                    "cash": self.current_capital,
                    "positions": self.positions,
                    "pending_signals": self.pending_signals,
                    "pending_condition_sells": self.pending_condition_sells,
                    "pending_stop_loss_sells": self.pending_stop_loss_sells,
                    "unfilled_slots": self.unfilled_slots,
                }
            )
        )


def engine_kwargs(root: Path, ranked: dict, capital: int, current_config: bool) -> dict:
    """显式冻结全部研究配置；current附表仅改变已存在的两项纸面配置。"""
    from scripts.v2_p7.t25.signal_layer import _signal_universe_config

    storage = Storage(root_path=str(root / "data"))
    loader = DataLoader(storage)
    stock_basic = loader.load_clean_stock_basic()
    config = _signal_universe_config()
    universe = BasicUniverse(stock_basic=stock_basic, verbose=False, **config)
    # 原生特征hook仅验证存在；此处的真实模型推理已在输入缓存逐日完成。
    witness = pd.DataFrame({"ranked_cache_verified": [True]})
    return {
        "portfolio_merger": SingleSleevePortfolioMerger(),
        "universe": universe,
        "signal": FrozenRankSignal(ranked),
        "features_by_date": {day: witness for day in ranked},
        "initial_capital": float(capital),
        "cost_model": CostModel(0.0001954, 5.0, 0.0005, 0.0005),
        "rebalance_freq": 20,
        "holding_period": 20,
        "stagger_tranches": 2,
        "stop_loss_config": None,
        "sell_timing": "open",
        "enable_pending_order": True,
        "max_retry_count": 5,
        "max_retry_days": 10,
        "enable_position_completion": True,
        "completion_window_days": 5,
        "verbose": False,
        "data_storage": storage,
        "max_weight_per_stock": 0.15,
        "max_per_industry": 5 if current_config else None,
        "stock_basic": stock_basic,
        "industry_reader": loader.load_shenwan_industry,
        "position_sizing": "kelly" if current_config else "half_kelly",
        "kelly_vol_window": 60,
        "kelly_max_leverage": 0.2,
        "min_buy_value_ratio": 0.2,
        "enable_early_rebalance_on_empty": True,
        "enable_trailing_early_rebalance": True,
    }


def initialize_driver(
    engine: Any, dates: list[pd.Timestamp], prices: pd.DataFrame, indexes: Any
) -> None:
    """按原生run的相同初始化驱动run_step；四个不可变价格索引只读共用。"""
    engine._trade_date_index = {date: i for i, date in enumerate(dates)}
    for key in (
        "trade_price_index",
        "pnl_price_index",
        "trade_price_open_index",
        "pnl_price_open_index",
    ):
        setattr(engine, key, getattr(indexes, key))
    engine.price_data_cache = prices
    engine._signal_dates = engine._get_rebalance_dates(dates)
    engine._run_total_days = len(dates)
    engine._cycle_anchor_idx = 0


def result_payload(engine: Any) -> dict:
    """原生全表和研究旁路分开，复权逐笔profit不用于经济汇总。"""
    result = {
        "daily": engine.research_daily,
        "states": engine.research_states,
        "trades": engine.get_trades().to_dict("records"),
        "attribution": engine.get_execution_attribution().to_dict("records"),
        "holdings_snapshot": engine.get_holdings_snapshot().to_dict("records"),
        "buffer_events": getattr(engine, "events", []),
        "corporate": (
            getattr(engine, "native_corporate", None).payload()
            if hasattr(engine, "native_corporate")
            else None
        ),
        "ending": {
            "cash": engine.current_capital,
            "positions": engine.positions,
            "pending_signals": engine.pending_signals,
            "pending_condition_sells": engine.pending_condition_sells,
            "unfilled_slots": engine.unfilled_slots,
        },
    }
    return result


def compare_exact(left: dict, right: dict) -> dict:
    """全字段精确比较，任何一行/列/顺序差异都阻断机械门。"""
    checks = {}
    for key in ("daily", "states", "trades", "attribution", "holdings_snapshot", "ending"):
        a, b = canonical(left[key]), canonical(right[key])
        checks[key] = {
            "equal": a == b,
            "left_sha": hashlib.sha256(a.encode("utf-8")).hexdigest(),
            "right_sha": hashlib.sha256(b.encode("utf-8")).hexdigest(),
            "left_rows": len(left[key]),
            "right_rows": len(right[key]),
        }
    return {"all_equal": all(row["equal"] for row in checks.values()), "tables": checks}
