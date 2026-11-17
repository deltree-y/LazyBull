"""经营月报预筛的 A 袖缓存继承与原生独立账户；不训练、不修改生产。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.8）。
历史两段缓存之间用冻结24021补齐每日推理；新缓存不称全新留出集。
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd
import pyarrow.parquet as pq

from scripts.v2_p7.buffer_native_corporate import NativeCorporateBook, NativeCorporateMixin
from scripts.v2_p7.buffer_native_inputs import HoldoutRankedProvider
from scripts.v2_p7.buffer_native_replay import (
    SPEC_SHA,
    RecordNativeMixin,
    engine_kwargs,
    initialize_driver,
    load_prices,
    read_historical_rankings,
    result_payload,
    sha256,
    write_json,
)
from scripts.v2_p7.buffer_native_validation import validate_account
from scripts.v2_p7.run_buffer_native import cash_anchor
from src.lazybull.v2.hosts.native_engine import NativeMergeBacktestEngineML

START = "20190401"
END = "20261029"
GAP_START = "20260106"
GAP_END = "20260202"
WARMUP_START = "20180101"
RECENT_CACHE = Path("data/reports/p7_buffer_native_20261116/run_r4/holdout_inputs")
Rankings = dict[str, list[tuple[str, float]]]


@dataclass(frozen=True)
class BaselineInputs:
    """完整交易时钟和现金锚点；排名缺席与合法空排名分别登记。"""

    ranked: Rankings
    calendar: tuple[str, ...]
    anchor: str
    price_days: tuple[str, ...]
    identity: dict[str, Any]


class OperatingBaselineEngine(RecordNativeMixin, NativeCorporateMixin, NativeMergeBacktestEngineML):
    """原生 A 决策，加现金/裸股/应收/锁股会计与只读旁路。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.locked_sell_deferrals: list[dict[str, Any]] = []
        self.locked_sell_intents: dict[str, dict[str, Any]] = {}
        super().__init__(*args, **kwargs)

    def _sell_stock_direct(
        self,
        date: pd.Timestamp,
        stock: str,
        sell_type: str = "holding_period",
        sell_reason: str | None = None,
        trigger_type: str | None = None,
        fraction: float = 1.0,
    ) -> None:
        """整仓退出遇锁股交原生永久卖单队列；解锁后保留原退出原因执行。

        原生条件队列调用后会移除记录，故不能只跳过成交。原生延迟卖单仅在真实
        成交后清理、不会超时；部分减仓继续走旧桥严格校验，不放大为整仓。
        """
        adapter = getattr(self, "native_corporate", None)
        held = self.positions.get(stock)
        prior = self.locked_sell_intents.get(stock)
        if prior is not None and fraction == 1.0:
            sell_type = prior["sell_type"]
            sell_reason = prior["sell_reason"]
            trigger_type = prior["trigger_type"]
        if adapter is not None and held and fraction == 1.0:
            locked = adapter.locked_shares(stock)
            if locked:
                available = adapter.sellable_shares(self, stock)
                if not self.enable_pending_order or self.pending_order_manager is None:
                    raise ValueError(f"{date} {stock} 锁股整仓退出需要原生延迟卖单队列")
                if prior is None:
                    self.locked_sell_intents[stock] = {
                        "first_attempt_date": date.strftime("%Y%m%d"),
                        "sell_type": sell_type,
                        "sell_reason": sell_reason,
                        "trigger_type": trigger_type,
                        "fraction": 1.0,
                    }
                self.locked_sell_deferrals.append(
                    {
                        "trade_date": date.strftime("%Y%m%d"),
                        "ts_code": stock,
                        "requested_shares": int(held["shares"]),
                        "sellable_shares": available,
                        "locked_shares": locked,
                        **self.locked_sell_intents[stock],
                        "policy": "defer_entire_order_until_all_shares_unlocked",
                        "pending_sell_already_exists": (stock, "sell")
                        in self.pending_order_manager.pending_orders,
                    }
                )
                self._defer_sell_order(date, stock, "送转锁股未上市，整仓卖单顺延", True)
                return
        offset = len(self.trades)
        super()._sell_stock_direct(
            date, stock, sell_type, sell_reason, trigger_type, fraction=fraction
        )
        if (
            prior is not None
            and fraction == 1.0
            and stock not in self.positions
            and any(
                row["stock"] == stock and row["action"] == "sell" and row["date"] == date
                for row in self.trades[offset:]
            )
        ):
            self.locked_sell_intents.pop(stock, None)


def baseline_result_payload(engine: OperatingBaselineEngine) -> dict[str, Any]:
    """补充锁股顺延旁路和未决卖单，不改写原生及公司行为既有取证。"""
    result = result_payload(engine)
    result["locked_sell_deferrals"] = list(engine.locked_sell_deferrals)
    result["locked_sell_pending_intents"] = {
        stock: dict(value) for stock, value in engine.locked_sell_intents.items()
    }
    result["pending_order_obligations"] = (
        [dict(vars(order)) for order in engine.pending_order_manager.get_all_orders()]
        if engine.pending_order_manager is not None
        else []
    )
    return result


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"A 输入元信息必须为对象：{path}")
    return value


def _days(values: Sequence[str], context: str) -> tuple[str, ...]:
    result = tuple(str(day) for day in values)
    if not result or result != tuple(sorted(set(result))):
        raise ValueError(f"{context}日期必须非空、唯一且递增")
    for day in result:
        if len(day) != 8 or not day.isdigit():
            raise ValueError(f"{context}日期格式非法：{day}")
        pd.to_datetime(day, format="%Y%m%d", errors="raise")
    return result


def _validate_ranked(ranked: Mapping[str, Sequence[tuple[str, float]]]) -> None:
    for day, rows in ranked.items():
        codes = [str(code) for code, _ in rows]
        scores = [float(score) for _, score in rows]
        if len(set(codes)) != len(codes) or any(not code for code in codes):
            raise ValueError(f"A 排名股票重复或为空：{day}")
        if any(not math.isfinite(score) for score in scores):
            raise ValueError(f"A 排名含非有限分数：{day}")
        if any(a < b for a, b in zip(scores, scores[1:])):
            raise ValueError(f"A 排名不按分数递减：{day}")


class GapRankedProvider(HoldoutRankedProvider):
    """冻结24021补接缝；复用旧官方panel/freshness方法，不改旧模块常量。

    装配与调用的窄复制来源为 buffer_native_inputs.HoldoutRankedProvider；
    仅模型版本与允许日期不同，排名、空日和选择语义保持一致。
    """

    model_version = 24021

    def __init__(self, root: Path, days: Sequence[str]) -> None:
        from scripts.v2_p3.t05b_fingerprint import _configure_signal_selection
        from scripts.v2_p7.t25.signal_layer import _signal_selection, _signal_universe_config
        from src.lazybull.data.loader import DataLoader
        from src.lazybull.data.storage import Storage
        from src.lazybull.universe import BasicUniverse
        from src.lazybull.v2.core.signal.ml_signal import MLSignal
        from src.lazybull.v2.hosts.paper import daily_assembly as assembly
        from src.lazybull.v2.store.data_store import PanelDataStore

        self.allowed_days = frozenset(_days(days, "24021补推理"))
        if min(self.allowed_days) != GAP_START or max(self.allowed_days) != GAP_END:
            raise ValueError("24021补推理必须完整覆盖冻结接缝")
        self.root = root
        self.loader = DataLoader(Storage(root_path=str(root / "data")))
        self.store = PanelDataStore(root / "data")
        self.assembly = assembly
        self.signal = MLSignal(
            top_n=20,
            model_version=self.model_version,
            models_dir=str(root / "data/models/stock_selection"),
            verbose=False,
        )
        self.signal._load_model()
        self.selection = dict(_signal_selection())
        self.universe_config = dict(_signal_universe_config())
        _configure_signal_selection(self.signal, self.selection)
        basic = self.loader.load_clean_stock_basic()
        if basic is None or basic.empty:
            raise ValueError("24021接缝缺 clean/stock_basic，禁止raw回退")
        self.universe = BasicUniverse(stock_basic=basic, verbose=False, **self.universe_config)
        registered = set(self.store.manifest.snapshot()["columns"])
        classification = assembly._classify_columns(self.signal.feature_columns, registered)
        gaps, uncovered = assembly._check_column_sources(classification, (), frozenset())
        if uncovered:
            raise ValueError(f"24021接缝模型列来源不完整：{gaps}")
        self.freshness = assembly._required_freshness_columns(
            self.signal.feature_columns, self.signal.metadata.get("train_params", {})
        )
        self.columns = assembly._store_query_columns(classification, registered, self.freshness)
        missing = set(assembly.SELECTION_FILTER_COLUMNS) - set(self.columns)
        if missing:
            raise ValueError(f"24021接缝缺选择列：{sorted(missing)}")
        self.day_records: list[dict[str, Any]] = []

    def __call__(self, day: str) -> list[tuple[str, float]]:
        if day not in self.allowed_days:
            raise ValueError(f"日期不在24021冻结接缝：{day}")
        frame = self._features(day)
        quote = self.loader.load_clean_daily_by_date(day, auto_ensure=False)
        if quote is None or quote.empty or not quote.trade_date.astype(str).eq(day).all():
            raise ValueError(f"24021接缝行情缺失或日期不符：{day}")
        codes = self.universe.get_stocks(pd.Timestamp(day), quote_data=quote)
        values = self.signal.generate_ranked(pd.Timestamp(day), codes, {"features": frame.copy()})
        result = [(str(code), float(score)) for code, score in values]
        _validate_ranked({day: result})
        record: dict[str, Any] = {
            "day": day,
            "model_version": self.model_version,
            "feature_rows": len(frame),
            "ranked_rows": len(result),
        }
        if not result:
            selected, _, _ = self.signal._apply_selection_filters(
                frame.loc[frame.ts_code.isin(codes)].copy()
            )
            record["empty_reason"] = (
                "universe_empty"
                if not codes
                else "selection_empty" if selected.empty else "kernel_refused"
            )
            record["all_nan_model_columns"] = [
                name
                for name in self.signal.feature_columns
                if name in selected and selected[name].isna().all()
            ]
        self.day_records.append(record)
        return result


def _gap_identity(root: Path, days: Sequence[str], provider: GapRankedProvider) -> dict:
    """冻结补推理实际模型、逐日列域与原件字节；不存在的freshness输入明确登记。"""
    fixed = [
        "data/clean/trade_cal.parquet",
        "data/clean/stock_basic.parquet",
        "data/raw/shenwan_industry.parquet",
        "data/features/manifest.json",
        "data/models/stock_selection/v24021_metadata.json",
        "data/models/stock_selection/v24021_features.json",
        "data/models/stock_selection/v24021_model.joblib",
        "configs/base.yaml",
        "scripts/v2_p7/operating_monthly_baseline.py",
        "scripts/v2_p7/buffer_native_inputs.py",
        "scripts/v2_p7/t25/signal_layer.py",
    ]
    paths = {root / name for name in fixed}
    paths.update((root / "src/lazybull").rglob("*.py"))
    paths.update((root / "scripts/v2_p3").glob("*.py"))
    paths.update((root / "scripts/v2_p7/t25").glob("*.py"))
    manifest = _json(root / "data/features/manifest.json")
    coverage, absent = [], []
    for day in days:
        clean = root / "data/clean/daily" / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet"
        panel = sorted((root / "data/features/panel" / day).glob("*.parquet"))
        if not clean.is_file() or len(panel) != 8:
            raise ValueError(f"24021接缝行情或八组panel缺失：{day}")
        paths.update([clean, *panel])
        domain = {
            manifest["partitions"][p.relative_to(root / "data/features").as_posix()]["row_domain"]
            for p in panel
        }
        if domain != {"label_filtered"}:
            raise ValueError(f"24021接缝行域发生变化：{day}/{domain}")
        columns = sorted({c for p in panel for c in pq.ParquetFile(p).schema_arrow.names})
        infer = root / "data/features/cs_infer" / f"{day}.parquet"
        if infer.is_file():
            paths.add(infer)
        else:
            absent.append(infer.relative_to(root).as_posix())
        coverage.append(
            {
                "day": day,
                "row_domain": "label_filtered",
                "physical_columns": columns,
                "cs_infer_exists": infer.is_file(),
            }
        )
    return {
        "schema": "p7_operating_monthly_gap_inputs/v1",
        "model_version": 24021,
        "days": list(days),
        "columns_consumed": list(provider.columns),
        "model_columns": list(provider.signal.feature_columns),
        "freshness": list(provider.freshness),
        "selection": provider.selection,
        "universe": provider.universe_config,
        "coverage": coverage,
        "input_files": {p.relative_to(root).as_posix(): sha256(p) for p in sorted(paths)},
        "absent_input_paths": absent,
        "retrained": False,
        "cs_train_fallback": False,
        "not_virgin_holdout": True,
    }


def build_gap_rankings(
    root: Path,
    output: Path,
    *,
    days: Sequence[str],
    spec_sha256: str,
    budget_guard: Callable[[], None] | None = None,
) -> tuple[Rankings, dict]:
    """新证据根内一次补推理；所有正式运行由调用者冻结规格后启动。"""
    started = time.perf_counter()
    if len(spec_sha256) != 64 or any(c not in "0123456789abcdef" for c in spec_sha256):
        raise ValueError("经营月报规格SHA必须为64位小写十六进制")
    output.mkdir(parents=True, exist_ok=False)
    provider = GapRankedProvider(root, days)
    before = _gap_identity(root, days, provider)
    write_json(output / "input_identity.json", before)
    mark = time.perf_counter()
    rankings = {}
    for day in days:
        if budget_guard is not None:
            budget_guard()
        rankings[day] = provider(day)
    if _gap_identity(root, days, provider) != before:
        raise ValueError("24021接缝推理期间输入身份漂移，拒绝发布")
    frame = pd.DataFrame(
        [
            (day, rank, code, score)
            for day, rows in rankings.items()
            for rank, (code, score) in enumerate(rows, 1)
        ],
        columns=["trade_date", "rank", "ts_code", "score"],
    )
    frame.to_parquet(output / "ranked.parquet", index=False)
    identity = {
        **before,
        "complete": True,
        "spec_sha256": spec_sha256,
        "ranked_sha256": sha256(output / "ranked.parquet"),
        "ranked_rows": len(frame),
        "day_records": provider.day_records,
        "seconds": time.perf_counter() - mark,
        "preparation_and_inference_seconds": time.perf_counter() - started,
    }
    write_json(output / "complete.json", identity)
    return rankings, identity


def read_gap_rankings(root: Path, cache: Path, *, spec_sha256: str) -> tuple[Rankings, dict]:
    """复用本轮新建接缝缓存；核规格、模型、源输入和逐日完整性。"""
    identity = _json(cache / "complete.json")
    if (
        identity.get("schema") != "p7_operating_monthly_gap_inputs/v1"
        or identity.get("complete") is not True
        or identity.get("model_version") != 24021
        or identity.get("spec_sha256") != spec_sha256
    ):
        raise ValueError("本轮24021接缝缓存未完成或身份不符")
    if sha256(cache / "ranked.parquet") != identity["ranked_sha256"]:
        raise ValueError("本轮24021接缝缓存排名被改写")
    for name, digest in identity["input_files"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"本轮24021接缝缓存输入发生变化：{name}")
    if any((root / name).exists() for name in identity["absent_input_paths"]):
        raise ValueError("本轮24021接缝缓存的缺席输入随后出现")
    days = _days(identity["days"], "本轮24021接缝")
    rows = identity["day_records"]
    if [row["day"] for row in rows] != list(days):
        raise ValueError("本轮24021接缝缓存逐日登记不完整")
    frame = pd.read_parquet(cache / "ranked.parquet")
    if len(frame) != identity["ranked_rows"]:
        raise ValueError("本轮24021接缝缓存总行数不符")
    ranked = {day: [] for day in days}
    for day, group in frame.groupby("trade_date", sort=False):
        day = str(day)
        if day not in ranked or group["rank"].tolist() != list(range(1, len(group) + 1)):
            raise ValueError(f"本轮24021接缝缓存日期或名次非法：{day}")
        ranked[day] = list(zip(group.ts_code.astype(str), group.score.astype(float)))
    if any(len(ranked[row["day"]]) != row["ranked_rows"] for row in rows):
        raise ValueError("本轮24021接缝缓存单日行数不符")
    _validate_ranked(ranked)
    return ranked, identity


def read_recent_rankings(root: Path, cache: Path = RECENT_CACHE) -> tuple[Rankings, dict]:
    """只继承前轮 24022 已冻结缓存；核实际输入字节，不触发重新推理。"""
    root = root.resolve()
    folder = root / cache
    manifest = _json(folder / "manifest.json")
    if (
        manifest.get("schema") != "p7_buffer_native_rankings/v1"
        or manifest.get("complete") is not True
        or manifest.get("model_version") != 24022
        or manifest.get("spec_sha256") != SPEC_SHA
    ):
        raise ValueError("24022 冻结缓存身份、完成状态或旧规格不符")
    if set(manifest["files"]) != {"audit.json", "ranked.parquet", "validation.json"}:
        raise ValueError("24022 缓存清单缺件或额外件")
    for name, digest in manifest["files"].items():
        if sha256(folder / name) != digest:
            raise ValueError(f"24022 缓存被改写：{name}")
    audit, validation = _json(folder / "audit.json"), _json(folder / "validation.json")
    days = _days(audit["days"], "24022缓存")
    if (
        audit.get("model_version") != 24022
        or validation.get("complete") is not True
        or validation.get("input_files_verified_after") is not True
        or validation.get("days") != len(days)
    ):
        raise ValueError("24022 缓存核验未闭合")
    for item in [*audit["input_files"], manifest["source_code"]]:
        if sha256(root / item["path"]) != item["sha256"]:
            raise ValueError(f"24022 缓存输入字节发生变化：{item['path']}")
    appeared = [name for name in audit["absent_input_paths"] if (root / name).exists()]
    if appeared:
        raise ValueError(f"24022 缓存缺席输入随后出现：{appeared}")
    frame = pd.read_parquet(
        folder / "ranked.parquet", columns=["trade_date", "rank", "ts_code", "score"]
    )
    if len(frame) != validation["ranked_rows"]:
        raise ValueError("24022 缓存排名总行数不符")
    records = validation["day_records"]
    if [str(row["day"]) for row in records] != list(days):
        raise ValueError("24022 缓存逐日登记不完整或乱序")
    ranked: Rankings = {day: [] for day in days}
    for day, group in frame.groupby("trade_date", sort=False):
        day = str(day)
        if day not in ranked or group["rank"].tolist() != list(range(1, len(group) + 1)):
            raise ValueError(f"24022 缓存名次/日期非法：{day}")
        ranked[day] = list(zip(group.ts_code.astype(str), group.score.astype(float)))
    for row in records:
        if int(row["ranked_rows"]) != len(ranked[str(row["day"])]):
            raise ValueError(f"24022 缓存日行数不符：{row['day']}")
    empty_days = {str(row["day"]) for row in validation["empty_days"]}
    if empty_days != {day for day, values in ranked.items() if not values}:
        raise ValueError("24022 缓存空排名日登记不符")
    _validate_ranked(ranked)
    return ranked, {
        "cache": cache.as_posix(),
        "cache_manifest_sha256": sha256(folder / "manifest.json"),
        "files": manifest["files"],
        "model_version": 24022,
        "days": list(days),
        "empty_days": sorted(empty_days),
        "input_files_verified_now": True,
        "not_virgin_holdout": True,
    }


def stitch_rankings(
    historical: Rankings,
    historical_identity: Mapping[str, Any],
    recent: Rankings,
    recent_identity: Mapping[str, Any],
    full_calendar: Sequence[str],
    *,
    start: str,
    end: str,
    gap: Rankings,
    gap_identity: Mapping[str, Any],
    warmup_start: str = WARMUP_START,
) -> BaselineInputs:
    """同一现金起点与连续时钟拼接三段；接缝必须完整补推理，不静默填空。"""
    full = _days(full_calendar, "A总日历")
    if start not in full or end not in full or start > end or full.index(start) == 0:
        raise ValueError("A研究窗端点或前一交易日现金锚点缺失")
    old_days = _days(historical_identity["signal_days"], "A历史信号")
    _days(historical_identity["calendar"], "A历史执行")
    new_days = _days(recent_identity["days"], "A近期信号")
    if set(historical) != set(old_days) or set(recent) != set(new_days):
        raise ValueError("A缓存排名日期与登记日期不一致")
    if set(historical) & set(recent) or max(old_days) >= min(new_days):
        raise ValueError("A两段信号日期重叠或次序错误")
    _validate_ranked(historical)
    _validate_ranked(recent)
    _validate_ranked(gap)
    days = tuple(day for day in full if start <= day <= end)
    expected_gap = [day for day in full if max(old_days) < day < min(new_days)]
    if (
        set(gap) != set(expected_gap)
        or list(gap_identity["days"]) != expected_gap
        or gap_identity.get("model_version") != 24021
    ):
        raise ValueError("A接缝排名必须按24021完整补齐且日期与登记一致")
    inherited = {**historical, **gap, **recent}
    missing = [day for day in days if day not in inherited]
    if missing:
        raise ValueError(f"A完整排名仍存在缺口：{missing}")
    ranked = {
        day: [(str(code), float(score)) for code, score in inherited[day]]
        for day in days
        if day in inherited
    }
    price_days = tuple(day for day in full if warmup_start <= day <= end)
    if not price_days or price_days[0] >= start:
        raise ValueError("A价格预热起点必须早于现金起步")
    anchor = full[full.index(start) - 1]
    return BaselineInputs(
        ranked=ranked,
        calendar=days,
        anchor=anchor,
        price_days=price_days,
        identity={
            "historical": dict(historical_identity),
            "recent": dict(recent_identity),
            "gap": dict(gap_identity),
            "start": start,
            "end": end,
            "anchor": anchor,
            "gap_policy": "fill_v24021",
            "gap_days": expected_gap,
            "not_virgin_holdout": True,
            "clock_reset_at_model_seam": False,
            "warmup_start": warmup_start,
            "warmup": "显式扩展至20180101；每股实际可得行取尾60，停牌不足沿原生降级并登记",
        },
    )


def load_baseline_inputs(
    root: Path,
    *,
    gap_output: Path,
    spec_sha256: str,
    start: str = START,
    end: str = END,
    budget_guard: Callable[[], None] | None = None,
) -> BaselineInputs:
    """核验两段已存缓存，只补24021接缝推理；不下载、不重训。"""
    root = root.resolve()
    historical, old_identity = read_historical_rankings(root)
    recent, new_identity = read_recent_rankings(root)
    calendar_path = root / "data/clean/trade_cal.parquet"
    table = pd.read_parquet(calendar_path, columns=["exchange", "is_open", "cal_date"])
    calendar = sorted(
        table.loc[table.exchange.eq("SSE") & table.is_open.astype(int).eq(1), "cal_date"]
        .astype(str)
        .unique()
        .tolist()
    )
    gap_days = [day for day in calendar if GAP_START <= day <= GAP_END]
    if gap_output.exists():
        gap, gap_identity = read_gap_rankings(root, gap_output, spec_sha256=spec_sha256)
    else:
        gap, gap_identity = build_gap_rankings(
            root, gap_output, days=gap_days, spec_sha256=spec_sha256, budget_guard=budget_guard
        )
    inputs = stitch_rankings(
        historical,
        old_identity,
        recent,
        new_identity,
        calendar,
        start=start,
        end=end,
        gap=gap,
        gap_identity=gap_identity,
    )
    inputs.identity["calendar_file"] = {
        "path": calendar_path.relative_to(root).as_posix(),
        "sha256": sha256(calendar_path),
    }
    return inputs


def load_baseline_prices(root: Path, inputs: BaselineInputs) -> tuple[pd.DataFrame, dict]:
    """独立扩展预热输入身份；仅计实际行不足，不改变既有原生方差降级。"""
    prices, files = load_prices(root, list(inputs.price_days), warmup_cutoff=inputs.calendar[0])
    history = prices.loc[prices.trade_date.astype(str).lt(inputs.calendar[0])]
    counts = history.groupby("ts_code", observed=True).size().to_dict()
    ranked_codes = sorted({code for rows in inputs.ranked.values() for code, _ in rows})
    insufficient = [
        {"ts_code": code, "pre_start_rows": int(counts.get(code, 0))}
        for code in ranked_codes
        if counts.get(code, 0) < 60
    ]
    first_codes = {code for code, _ in inputs.ranked[inputs.calendar[0]]}
    return prices, {
        "files": files,
        "warmup_start_requested": WARMUP_START,
        "warmup_first_trade_day": inputs.price_days[0],
        "cutoff": inputs.calendar[0],
        "actual_rows_per_stock_cap": 60,
        "ranked_union_codes": len(ranked_codes),
        "ranked_union_insufficient_60_at_start": insufficient,
        "ranked_union_insufficient_count": len(insufficient),
        "first_day_ranked_insufficient_count": sum(
            x["ts_code"] in first_codes for x in insufficient
        ),
        "limitation": "全窗排名并集含后来上市股票；起点不足计数不等于其实际买入日仍不足。",
        "native_variance_fallback_unchanged": True,
    }


def run_baseline_account(
    root: Path,
    *,
    inputs: BaselineInputs,
    prices: pd.DataFrame,
    indexes: Any,
    capital: int,
    current_config: bool = False,
    corporate_book: NativeCorporateBook,
    output: Path | None = None,
    budget_guard: Callable[[], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """每个资本档新建真实原生账户；不缩放旧NAV，不借款，不跨袖净额化。"""
    if isinstance(capital, bool) or int(capital) != capital or capital <= 0:
        raise ValueError("A账户初始资金必须为正整数元")
    if current_config is not False:
        raise ValueError("经营月报A基准固定half-Kelly且无行业上限，禁止切当前配置")
    if corporate_book is None:
        raise ValueError("A经济账户必须显式提供公司行为事件簿")
    days = _days(inputs.calendar, "A执行")
    if inputs.anchor >= days[0] or set(inputs.ranked) - set(days):
        raise ValueError("A锚点/排名日超出统一执行窗口")
    expected = tuple([inputs.anchor, *days])
    if tuple(corporate_book._days) != expected:
        raise ValueError("A公司行为事件簿与现金锚点/执行日历不一致")
    mark = time.perf_counter()
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
    kwargs = engine_kwargs(root, inputs.ranked, capital, current_config)
    engine = OperatingBaselineEngine(**kwargs, corporate_book=corporate_book)
    dates = [pd.Timestamp(day) for day in days]
    initialize_driver(engine, dates, prices, indexes)
    daily, state = cash_anchor(inputs.anchor, capital)
    engine.native_corporate.before_day(engine, pd.Timestamp(inputs.anchor))
    engine.native_corporate.after_day(engine, pd.Timestamp(inputs.anchor))
    engine.research_daily.append(daily)
    engine.research_states.append(state)
    try:
        for count, date in enumerate(dates, 1):
            if budget_guard is not None:
                budget_guard()
            engine.run_step(date, dates, prices)
            if progress is not None and (count % 100 == 0 or count == len(dates)):
                progress(count, len(dates))
        result = baseline_result_payload(engine)
        validation = validate_account(result, initial_capital=capital)
        if not validation["valid"]:
            raise ValueError(f"A账户守恒失败：{validation['violations'][:3]}")
    except Exception as error:
        if output is not None:
            write_json(output / "failed_checkpoint.json", baseline_result_payload(engine))
            write_json(
                output / "failure.json",
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "completed_days": len(engine.research_daily),
                    "economic_success_claimed": False,
                },
            )
        raise
    result["validation"] = validation
    result["baseline_identity"] = {
        "initial_capital": capital,
        "current_config": current_config,
        "position_sizing": "kelly" if current_config else "half_kelly",
        "max_per_industry": 5 if current_config else None,
        "initial_point": inputs.anchor,
        "start": days[0],
        "end": days[-1],
        "capital_scaled_from_other_nav": False,
        "cross_sleeve_borrowing": False,
        "cross_sleeve_netting": False,
        "locked_sell_policy": "defer_entire_order_until_all_shares_unlocked",
        "not_virgin_holdout": True,
        "seconds": time.perf_counter() - mark,
    }
    if output is not None:
        for key, value in result.items():
            write_json(output / f"{key}.json", value)
    return result


def combine_independent_daily(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    total_capital: int,
) -> list[dict[str, Any]]:
    """逐日直接合计两份独立资金账；输入须含同一个现金锚点，不合成线性NAV。"""
    if total_capital <= 0 or not left or len(left) != len(right):
        raise ValueError("独立资金账必须非空、等长且总资本为正")
    fields = (
        "total_value",
        "cash",
        "market_value",
        "receivable",
        "fees",
        "buy_amount",
        "sell_amount",
    )
    combined = []
    for a, b in zip(left, right):
        if a["trade_date"] != b["trade_date"]:
            raise ValueError("独立资金账日期不一致，禁止交集缩窗")
        row: dict[str, Any] = {"trade_date": str(a["trade_date"])}
        for key in fields:
            values = [float(a[key]), float(b[key])]
            if any(not math.isfinite(value) for value in values):
                raise ValueError(f"独立资金账含非有限值：{row['trade_date']}/{key}")
            row[key] = math.fsum(values)
        if (
            abs(
                row["total_value"]
                - math.fsum(row[k] for k in ("cash", "market_value", "receivable"))
            )
            > 1e-6
        ):
            raise ValueError(f"独立资金账资产恒等式失败：{row['trade_date']}")
        if row["cash"] < -1e-6 or row["receivable"] < -1e-6:
            raise ValueError(f"独立资金账现金/应收为负：{row['trade_date']}")
        row["nav"] = row["total_value"] / total_capital
        combined.append(row)
    days = [row["trade_date"] for row in combined]
    _days(days, "合并资金账")
    initial = combined[0]
    if (
        abs(initial["cash"] - total_capital) > 1e-6
        or abs(initial["total_value"] - total_capital) > 1e-6
        or any(
            abs(initial[key]) > 1e-6
            for key in ("market_value", "receivable", "fees", "buy_amount", "sell_amount")
        )
    ):
        raise ValueError("独立资金账首日必须为总本金的纯现金锚点")
    return combined
