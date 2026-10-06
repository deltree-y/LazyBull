# -*- coding: utf-8 -*-
"""v2 panel 构建器（实现 v2 FeatureBuilder 协议，docs/contracts/protocols.md §1.2）。

核心策略 = **捕获复用**（行为冻结）：旧数值路径（`features.pipeline.build_features_data`
→ `features.builder.FeatureBuilder.build_features_for_day`）原样跑，只经
`_CaptureStorage` 拦截 `save_cs_train_day` 的输出帧，随后：

1. **后置物化**（冻结 §6 拍板：运行时派生四族历史区间一律物化进 announcement 族）：
   - ti_* 6 列：`TopInstFactorHandler` + `factors.top_inst.load_top_inst_lookup`
     （与训练/OOS 评估/OOS 回测/纸面四侧同一数值实现，查询表一次性加载按日切片喂入）；
   - has_* 4 列：`factors.availability.derive_availability_markers` 同一实现；
2. **拆分**：`_split_groups` 按 column_groups 单一来源拆 8 族（标签列剥离）；
3. **落 store**：build_daily 全热区；backfill 按冻结 §3 冷热边界（20250630）——
   冷区经 `_ArchiveMonthBuffer` 月缓冲 flush 为 `append_archive_features`，
   热区逐日 `append_features`；labels 逐日经 labels_builder 落盘。

`bootstrap_manifest`：按 column_groups 批量登记 manifest 列（物化列 deprecated）
与 dependencies 依赖声明，回填/标定前各跑一次。
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Sequence

import pandas as pd
from loguru import logger

from src.lazybull.data.storage import Storage
from src.lazybull.factors.availability import derive_availability_markers
from src.lazybull.factors.top_inst import load_top_inst_lookup
from src.lazybull.features.builder import FeatureBuilder as LegacyFeatureBuilder
from src.lazybull.features.factor_handlers import TopInstFactorHandler
from src.lazybull.features.pipeline import build_features_data
from src.lazybull.v2.common.types import TradeDate
from src.lazybull.v2.store.column_groups import (
    ARCHIVE_HOT_BOUNDARY,
    LABEL_VALUE_COLUMNS,
    MATERIALIZED_COLUMNS,
    PANEL_GROUPS,
    columns_of_group,
    validate_mapping,
)
from src.lazybull.v2.store.labels_builder import append_labels_for_day

__all__ = ["V2PanelBuilder", "bootstrap_manifest"]

_KEY_COLUMNS = ("trade_date", "ts_code")
_ALL_GROUPS = tuple(PANEL_GROUPS)

#: 捕获大表封闭列集（_split_groups 闭合断言用）：全族列 ∪ 物化列 ∪ 6 标签列 ∪ 键列
#: （377+34+6+2=419；标签列名取 LABEL_VALUE_COLUMNS 单一来源，不重复硬编码）
_ALLOWED_CAPTURE_COLUMNS = frozenset(
    {c for group in PANEL_GROUPS for c in columns_of_group(group)}
    | set(MATERIALIZED_COLUMNS)
    | set(LABEL_VALUE_COLUMNS)
    | set(_KEY_COLUMNS)
)

#: 族级来源标识（manifest columns.source 一句话口径）
_GROUP_SOURCES = {
    "core": "clean/daily+daily_basic 直通与基础量价（features.builder 基线）",
    "fundamental": "factors 基本面家族（fina_indicator/forecast/express/一致预期/现金流/分红）",
    "moneyflow": "资金流向+两融+北向+龙虎榜（moneyflow/margin/moneyflow_hsgt/top_list）",
    "technical": "筹码+基金持仓+高级量价+个股特征（cyq_perf/fund_portfolio/builder）",
    "announcement": "公告类（股东户数/质押/解禁/大宗）",
    "risk": "factors.risk 注册风控因子族",
    "market_state": "features.market_state 市场状态广播列",
    "neutralized": "features.neutralization 行业（申万）+规模中性化",
}
_MATERIALIZED_SOURCE = (
    "运行时派生物化（holdertrade/repurchase/top10fh/top_inst/availability，冻结§6）"
)

#: manifest dependencies.raw_datasets 登记清单（回填构建全链路依赖）
_RAW_DEPENDENCIES = [
    "daily",
    "adj_factor",
    "daily_basic",
    "moneyflow",
    "stk_limit",
    "suspend",
    "stock_st",
    "margin_detail",
    "fina_indicator",
    "forecast",
    "express",
    "cashflow",
    "income",
    "report_rc",
    "cyq_perf",
    "fund_portfolio",
    "moneyflow_hsgt",
    "top_list",
    "dividend",
    "stk_holdertrade",
    "repurchase",
    "top10_floatholders",
    "top_inst",
    "pledge_stat",
    "share_float",
    "block_trade",
    "stk_holdernumber",
    "shenwan_industry",
    "trade_cal",
    "stock_basic",
]

#: manifest dependencies.factor_functions 登记（构建/物化数值实现标识）
_FACTOR_FUNCTIONS = {
    "panel_builder": "features.pipeline.build_features_data（捕获复用）",
    "top_inst": "factors.top_inst.build_top_inst_feature_frame",
    "availability": "factors.availability.derive_availability_markers",
    "labels": "features.labels（y_ret/neu_y_ret，T+1 收盘买 T+1+N 开盘卖）",
    "neutralization": "features.neutralization（行业+规模）",
}


class _CaptureStorage(Storage):
    """捕获型 Storage：拦截 `save_cs_train_day` 不落盘，转发给注入的 sink 回调。

    其余方法（is_feature_exists 等）保持原样——backfill 一律 force=True 绕过跳过逻辑，
    捕获路径上不触发任何读盘；has_label=False 与父类同语义（不保存也不转发）。
    """

    def __init__(self, root_path: str | Any, sink: Callable[[pd.DataFrame, str], None]) -> None:
        super().__init__(root_path=str(root_path))
        self._sink = sink

    def save_cs_train_day(
        self,
        df: pd.DataFrame,
        trade_date: str,
        format: str = "parquet",
        has_label: bool = True,
        subdir: str = "cs_train",
    ) -> None:
        if not has_label:
            logger.info(f"捕获 {trade_date}：无标签帧，按父类语义跳过（不转发 sink）")
            return
        self._sink(df, str(trade_date))


def _split_groups(df_day: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """单日大表 → 8 族帧（每族含 trade_date+ts_code 键列，固定列序）。

    announcement = 10 基础列 + 34 物化列；族内列允许个别日期整列缺失 ⇒ 补 NaN
    （物化列天然稀疏 + 族缺日整族 NaN 语义）；6 个标签列不属于任何 panel 族，
    拆分即剥离（labels 由 labels_builder 先行抽走）。
    列集闭合断言：大表非键列 ⊆ 全族列 ∪ MATERIALIZED ∪ 6 标签列（377+34+6+2=419
    列的封闭列集）——列新增/改名必须显式登记，禁止静默 reindex 丢列。
    """
    unknown = sorted(set(df_day.columns) - _ALLOWED_CAPTURE_COLUMNS)
    if unknown:
        raise RuntimeError(
            f"单日大表含未登记列 {unknown[:10]}（须登记进 column_groups 单一来源，禁止静默丢列）"
        )
    frames: dict[str, pd.DataFrame] = {}
    for group in PANEL_GROUPS:
        data_cols = [c for c in columns_of_group(group) if c not in _KEY_COLUMNS]
        frame = df_day.reindex(columns=[*_KEY_COLUMNS, *data_cols])
        frames[group] = frame
    return frames


def bootstrap_manifest(store: Any) -> None:
    """按 column_groups 单一来源批量登记 manifest 列 + dependencies（幂等，可重跑）。

    available_from=None（恒可用，单元 5 剖面报告回填精化）；物化列 status="deprecated"
    （已终结家族，物化是数据资产合规，不代表重开实验），其余 "active"。
    已登记列只校验 group/definition_version 一致（冲突 ⇒ ValueError），治理元数据
    （available_from/backfilled_at/status 等）保留不动；未登记列才 register_column。
    """
    validate_mapping()
    manifest = store.manifest
    planned: list[tuple[str, str, str, str]] = []  # (列, 族, source, status)
    for group, cols in PANEL_GROUPS.items():
        for col in cols:
            planned.append((col, group, _GROUP_SOURCES[group], "active"))
    for col in MATERIALIZED_COLUMNS:
        planned.append((col, "announcement", _MATERIALIZED_SOURCE, "deprecated"))
    for col, group, source, status in planned:
        if manifest.has_column(col):
            meta = manifest.column_meta(col)
            conflicts = [
                f"{key} 已登记 {meta.get(key)!r} ≠ bootstrap {expected!r}"
                for key, expected in (("group", group), ("definition_version", "v1"))
                if meta.get(key) != expected
            ]
            if conflicts:
                raise ValueError(f"列 {col} bootstrap 与既有登记冲突: {conflicts}")
            continue  # 已登记 ⇒ 治理元数据保留不动
        manifest.register_column(col, group, source=source, definition_version="v1", status=status)
    manifest.register_dependencies(
        raw_datasets=list(_RAW_DEPENDENCIES), factor_functions=dict(_FACTOR_FUNCTIONS)
    )
    manifest.save()
    logger.info(
        f"manifest bootstrap 完成: {sum(len(c) for c in PANEL_GROUPS.values())} 列 + "
        f"{len(MATERIALIZED_COLUMNS)} 物化列，raw 依赖 {len(_RAW_DEPENDENCIES)} 项"
    )


class _ArchiveMonthBuffer:
    """backfill 冷区月缓冲：同自然月的族帧攒批，月切换 / flush 时落 append_archive_features。"""

    def __init__(self, store: Any, groups: Sequence[str]) -> None:
        self._store = store
        self._groups = list(groups)
        self._month: Optional[str] = None
        self._buf: dict[str, list[pd.DataFrame]] = {}

    def add(self, date_str: str, frames: dict[str, pd.DataFrame]) -> None:
        """缓冲单日族帧；月切换自动 flush 上月。"""
        month = f"{date_str[:4]}-{date_str[4:6]}"
        if self._month is not None and month != self._month:
            self.flush()
        if self._month is None:
            self._month = month
            self._buf = {g: [] for g in self._groups}
        for group in self._groups:
            self._buf[group].append(frames[group])

    def flush(self) -> None:
        """冲刷当前月缓冲（无缓冲时 no-op）。"""
        if self._month is None:
            return
        for group in self._groups:
            parts = self._buf[group]
            if not parts:
                continue
            month_df = pd.concat(parts, ignore_index=True)
            self._store.append_archive_features(self._month, group, month_df)
            logger.info(f"冷区月分区 flush: {self._month}/{group}（{len(month_df)} 行）")
        self._month = None
        self._buf = {}


class V2PanelBuilder:
    """v2 FeatureBuilder 协议实现（捕获复用旧数值路径 + 后置物化 + 拆分落 store）。

    Args:
        loader: 旧 DataLoader（依赖注入，本类不建全局单例）
        min_list_days: 样本过滤最小上市天数（冻结 §1：365）
        verbose: 旧构建器详细日志开关
    """

    def __init__(self, loader: Any, min_list_days: int = 365, verbose: bool = False) -> None:
        self._loader = loader
        # 冻结 §1：min_list_days=365 / horizons={5,10,20} / require_label。
        # label_filter_mode 实测修正（2026-10-03 标定发现）：现产物 cs_train 全历史分区
        # 均含 y_ret_5/y_ret_10 NaN 行（y_ret_20 恒非空）⇒ 生产构建走的是
        # build_clean_features.py --horizon 20 单值分支（label_filter_mode="single"，
        # 仅按主 horizon=20 过滤），而非冻结 §1 文字推断的 "all"。等价复刻以现产物为准，
        # 冻结文档 §1 行须走重冻结程序修订（已登记完成报告）。
        self._builder = LegacyFeatureBuilder(
            min_list_days=min_list_days,
            horizon=20,
            horizons=[5, 10, 20],
            require_label=True,
            label_filter_mode="single",
            verbose=verbose,
        )

    @staticmethod
    def _build_flags() -> dict[str, bool]:
        """生产开关集（冻结 §1 现产物列集口径；ht/rp/tfh 开 = 物化拍板 §6）。

        ti_* 无管线开关（走后置物化）；has_* 同为后置。开关集经单日标定
        （temp/p1_flag_calib_20261003.py）验证：捕获列集 == cs_train 383 ∪ ht/rp/tfh 24。
        """
        return {
            "apply_industry_neutralization": True,
            "apply_size_neutralization": True,
            "enable_fundamental": True,
            "enable_alt": True,
            "enable_margin": True,
            "enable_cyq": True,
            "enable_fund": True,
            "enable_express": True,
            "enable_north": True,
            "enable_lhb": True,
            "enable_consensus": True,
            "enable_cashflow_quality": True,
            "enable_consensus_revision": True,
            "enable_dividend_policy": True,
            "enable_announcement_risk": True,
            "enable_holdertrade": True,
            "enable_repurchase": True,
            "enable_top10fh": True,
        }

    # ---------- 协议方法 ----------

    def build_daily(self, date: TradeDate, groups: Sequence[str], store: Any) -> None:
        """构建指定日特征分区（单日增量/影子通路，全热区；防冲突沿 store 语义）。

        捕获区间 = [D−7 个自然月， D]，sink 只落 D——与旧 cs_infer 单日通路
        （`features/ensure/entry.py`，FEATURE_DATA_HISTORY_MONTHS=7）的
        trading_dates_str 输入同口径。标定实证（temp/p1_flag_calib_20261003.py）：
        若按 [D, D] 单日捕获，北向 rolling 级联列（north_net_buy_sum5/ma5/ma20/z20）
        在长度 1 的交易日序列上全市场算错（4684/4684 行超容差），mkt_ma250 等
        长记忆特征亦退化；带预热段后与批量回填口径一致（长记忆 EMA 尾部随
        2012 起回填消失）。过滤口径与批量回填完全一致（全市场统一截面，
        禁止以当日局部截面重排，冻结 §1）。

        标签不可终 fail-fast（R3-03 防御性收口）：T+21 端点超数据水位/加载窗口的
        目标日直接 RuntimeError（不再走到缺日断言报误导性「缺日」）；当日特征日更
        依赖未来标签、样本域需契约裁决，历史影子通路不受影响。
        """
        date_str = str(date)
        use_groups = self._normalize_groups(groups)
        calendar = self._full_calendar()
        if not self._filter_label_starved([date_str], calendar, date_str):
            raise RuntimeError(
                f"{date_str} 标签不可终（T+21 端点超数据水位/加载窗口）：当日特征日更依赖"
                "未来标签，样本域需契约裁决（登记号 R3-03），历史影子通路不受影响"
            )
        ti_lookup = self._build_ti_lookup_for_day(calendar, date_str)
        capture_start = self._daily_capture_start(date_str)
        captured = self._run_capture(capture_start, date_str, keep_dates={date_str})
        for trade_date, df_day in captured:
            materialized = self._materialize_day(df_day, trade_date, ti_lookup)
            append_labels_for_day(store, materialized, calendar, data_end=self._data_horizon())
            frames = _split_groups(materialized)
            for group in use_groups:
                store.append_features(TradeDate.from_str(trade_date), group, frames[group])
        self._assert_days_captured(captured, [date_str])

    @staticmethod
    def _daily_capture_start(date_str: str) -> str:
        """单日捕获起点 = D − 7 个自然月（对齐旧单日通路预热窗口常量）。"""
        start = pd.to_datetime(date_str, format="%Y%m%d") - pd.DateOffset(months=7)
        return start.strftime("%Y%m%d")

    def backfill(
        self,
        start_date: TradeDate,
        end_date: TradeDate,
        groups: Sequence[str],
        store: Any,
        keep_dates: Optional[set[str]] = None,
    ) -> None:
        """历史回填（冷热分层：<=20250630 冷区月分区，其后热区日分区；可重入续传）。

        防冲突：store 两阶段提交（指纹一致放行/no-op、不一致报错），重跑即续传。
        keep_dates 非空时 sink 只落 keep 内的日（管线仍 force=True 全量构建——
        分块重叠日以不同预热窗重建时 EMA/长记忆列尾差 >1e-6，重复 append 必触发
        指纹冲突；sink 丢弃重叠日 ⇒ panel ≡ 参照，见单元 4 构造约束）。
        断言：keep_dates 内全部捕获（缺日显式报错，禁止静默缺口落盘）。
        """
        start_str, end_str = str(start_date), str(end_date)
        use_groups = self._normalize_groups(groups)
        calendar = self._full_calendar()
        range_dates = [d for d in calendar if start_str <= d <= end_str]
        if not range_dates:
            raise ValueError(f"回填区间无交易日: {start_str}~{end_str}")
        if keep_dates is not None:
            unknown = sorted(set(keep_dates) - set(range_dates))
            if unknown:
                raise ValueError(f"keep_dates 含区间外日期: {unknown[:5]}（{start_str}~{end_str}）")
            if not keep_dates:
                logger.info(
                    f"backfill {start_str}~{end_str}: keep_dates 为空（全部已写），跳过构建"
                )
                return
        ti_lookup = load_top_inst_lookup(self._loader, range_dates)
        buffer = _ArchiveMonthBuffer(store, use_groups)
        captured = self._run_capture(start_str, end_str, keep_dates=keep_dates)
        # manifest 逐分区落盘是全量回填 O(n²) 瓶颈 ⇒ 延迟到本块出口统一落盘
        with store.defer_manifest_save():
            for trade_date, df_day in captured:
                materialized = self._materialize_day(df_day, trade_date, ti_lookup)
                append_labels_for_day(store, materialized, calendar, data_end=self._data_horizon())
                frames = {g: f for g, f in _split_groups(materialized).items() if g in use_groups}
                if trade_date <= ARCHIVE_HOT_BOUNDARY:
                    buffer.add(trade_date, frames)
                else:
                    buffer.flush()
                    for group, frame in frames.items():
                        store.append_features(TradeDate.from_str(trade_date), group, frame)
            buffer.flush()
        expected = keep_dates if keep_dates is not None else range_dates
        expected = self._filter_label_starved(expected, calendar, end_str)
        self._assert_days_captured(captured, expected)

    # ---------- 捕获驱动与后置物化 ----------

    def _run_capture(
        self,
        start_str: str,
        end_str: str,
        keep_dates: Optional[set[str]] = None,
    ) -> list[tuple[str, pd.DataFrame]]:
        """跑旧 pipeline（force=True 绕过跳过），捕获逐日大表。

        keep_dates 非空时 sink 只保留指定日（预热日的帧不进内存累积，
        单日通路防 ~140 日大表占内存）。
        """
        captured: list[tuple[str, pd.DataFrame]] = []

        def _sink(df: pd.DataFrame, trade_date: str) -> None:
            if keep_dates is None or trade_date in keep_dates:
                captured.append((trade_date, df))

        capture = _CaptureStorage(
            root_path=self._loader.storage.root_path,
            sink=_sink,
        )
        build_features_data(
            capture,
            self._loader,
            self._builder,
            start_str,
            end_str,
            force=True,
            shenwan_industry=self._load_shenwan(),
            **self._build_flags(),
        )
        return captured

    def _materialize_day(
        self, df_day: pd.DataFrame, date_str: str, ti_lookup: Optional[dict]
    ) -> pd.DataFrame:
        """后置物化：ti_*（TopInstFactorHandler 同一实现）+ has_*（availability 同一实现）。"""
        out = df_day.copy()
        day_data = ti_lookup.get(date_str) if ti_lookup else None
        # day_data 缺失 ⇒ 传空表（语义 = 窗口内无机构事件 ⇒ 0 填充）；
        # 不能传 None（handler 视 None 为整族未启用而返回空字典）
        produced = TopInstFactorHandler().apply(
            out, day_data if day_data is not None else pd.DataFrame(), date_str, pd.DataFrame()
        )
        for col, series in produced.items():
            out[col] = series
        added = derive_availability_markers(out)
        if len(added) != 4:
            logger.warning(f"{date_str} has_* 标记只派生 {len(added)}/4 个: {added}")
        return out

    # ---------- 日历与 shenwan ----------

    def _full_calendar(self) -> list[str]:
        """全量交易日历（clean trade_cal，YYYYMMDD 升序；标签成熟度判定需要 T 后窗口）。"""
        from src.lazybull.common.date_utils import normalize_series_to_yyyymmdd

        trade_cal = self._loader.load_clean_trade_cal()
        if trade_cal is None:
            raise ValueError("缺少 clean 层 trade_cal 数据（panel 构建的日历依赖）")
        dates = normalize_series_to_yyyymmdd(trade_cal["cal_date"])
        open_dates = dates[trade_cal["is_open"] == 1]
        return sorted(open_dates.tolist())

    def _build_ti_lookup_for_day(self, calendar: list[str], date_str: str) -> dict:
        """单日 ti 查询表（窗口 [T-19,T] 需 20 个交易日，留 25 天余量切片）。"""
        if date_str not in calendar:
            raise ValueError(f"{date_str} 不在交易日历中")
        idx = calendar.index(date_str)
        lo = max(0, idx - 24)  # 切片上下界先行赋值（black/flake8 E203 兼容：界内不放表达式）
        hi = idx + 1
        window_dates = calendar[lo:hi]
        return load_top_inst_lookup(self._loader, window_dates)

    def _filter_label_starved(
        self, days: Iterable[str], calendar: Sequence[str], end_str: str
    ) -> list[str]:
        """剔除数据视界内标签不可终日（T+21 端点超出管线加载窗口或数据水位）。

        管线加载窗口 = end+1 个自然月（pipeline.py:384-385）；数据水位 = clean daily
        最大分区日。端点超界 ⇒ 标签全 NaN ⇒ require_label 语义下当日无产出
        （与 cs_train 区间末端语义一致）。冻结全区间末端 20260702 不触发
        （T+21=20260730 ≤ 20260731 水位）；冒烟/短窗口可能触发。
        """
        load_end = (pd.to_datetime(end_str, format="%Y%m%d") + pd.DateOffset(months=1)).strftime(
            "%Y%m%d"
        )
        horizon = min(load_end, self._data_horizon())
        cal_index = {d: i for i, d in enumerate(calendar)}
        buildable, starved = [], []
        for day in days:
            idx = cal_index.get(day)
            endpoint = calendar[idx + 21] if idx is not None and idx + 21 < len(calendar) else None
            if endpoint is not None and endpoint <= horizon:
                buildable.append(day)
            else:
                starved.append(day)
        if starved:
            logger.warning(
                f"标签不可终日 {len(starved)} 天（T+21 端点 > {horizon}），"
                f"按数据视界语义排除: {starved[:5]}"
            )
        return buildable

    def _data_horizon(self) -> str:
        """clean daily 数据水位（最大分区日，YYYYMMDD）。"""
        daily_dir = self._loader.storage.clean_path / "daily"
        stems = [p.stem.replace("-", "") for p in daily_dir.glob("*.parquet")]
        if not stems:
            raise ValueError(f"clean daily 分区目录为空: {daily_dir}")
        return max(stems)

    def _load_shenwan(self) -> Optional[pd.DataFrame]:
        """申万行业数据（行业中性化必需；生产开关恒开）。"""
        shenwan = self._loader.load_shenwan_industry()
        if shenwan is None:
            raise ValueError("启用行业中性化但缺少申万行业数据（shenwan_industry）")
        return shenwan

    # ---------- 校验辅助 ----------

    @staticmethod
    def _normalize_groups(groups: Optional[Sequence[str]]) -> tuple[str, ...]:
        """族参数归一（None/空 = 全 8 族；未知族显式报错）。"""
        if not groups:
            return _ALL_GROUPS
        unknown = [g for g in groups if g not in PANEL_GROUPS]
        if unknown:
            raise ValueError(f"未知 panel 族 {unknown}（合法值 {sorted(PANEL_GROUPS)}）")
        return tuple(dict.fromkeys(groups))

    @staticmethod
    def _assert_days_captured(
        captured: list[tuple[str, pd.DataFrame]], expected: Iterable[str]
    ) -> None:
        """缺日兜底：pipeline 逐日吞异常，回填/单日构建禁止静默缺口。"""
        seen = {td for td, _ in captured}
        missing = [d for d in expected if d not in seen]
        if missing:
            raise RuntimeError(
                f"panel 构建缺日 {len(missing)} 天（如 {missing[:5]}）；"
                "旧 pipeline 逐日失败已吞（见上方 ERROR 日志），排查后可安全重跑（两阶段提交可重入）"
            )
