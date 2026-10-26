# -*- coding: utf-8 -*-
# 复杂度豁免：P3 T5 验收专用工具（不入生产链路；CLAUDE §7.8 临时开发文件契约）。
# 拆分计划：Stock Connect 闭市识别已拆出 t05b_stock_connect.py；score_model_fingerprint
# 为验收脚本单入口，随工具退役/迁入正式链路前不再拆分，避免验收期行为漂移。
"""P3 T0.5b 确定性 spike / G-code 预测指纹工具（§7.1-0；序列化 §13.3.3-S1 冻结）。

指纹序列化（S1 冻结规格）：
- ``-0.0`` 归一为 ``0.0``；**拒绝 NaN 与 Inf**（出现即报错）；
- 每行 ``YYYYMMDD,TS_CODE,%.17g``；键 ``(trade_date, ts_code)`` 升序；
- UTF-8 + LF；sha256；逐折指纹 + 全批汇总指纹；**逐种子指纹仅作诊断列**（不升格为门）。

冻结矩阵（主门口径，§13.3.3-M1/B6）：每折 OOS 打分日 × 当日行集 × 训练实际特征列。
行集口径按 ``source_mode`` 分列（reviewer1-S7 处置：P3 T5 起本工具服务两模式——
cs_train 验证源 + panel 源（臂③ v2_panel / G-data 归因）；**两源行集口径不同，
禁止跨源比对指纹**，指纹只在同源同口径下可比）：
- ``cs_train_verify``（默认）：打分日 = 测试窗内 cs_train 实际分区日；行集 =
  cs_train 分区行（**隐含标签非空**，cs_train 分区行即含标签行）∩ 主板 ∩
  非 ST/停牌/涨停（过滤列 = 分区自带 is_st/is_suspended/is_limit_up）；
- ``panel``：打分日 = panel 分区日域（PanelDataStore 无公开分区日清单 API ⇒
  整窗 ``load_features`` 后按 ``trade_date`` 分组取域；**禁止 glob cs_train**）；
  行集 = panel 当日行 ∩ 主板 ∩ 非 ST/停牌/涨停（**无标签非空隐含条件**——
  panel 特征行不携带标签，标签在独立 labels 表；过滤列 = panel core 族登记列
  is_st/is_suspended/is_limit_up，随读取列契约一并查询，未登记 ⇒ store 契约报错）。
panel 读取/切日与 replay panel 路径同一实现同一口径：整窗
``PanelDataStore.load_features``（v2/store/data_store.py:569）+ 按日切分
``_split_panel_frame_by_date``（v2/hosts/backtest/replay.py:295-308，import 复用
不复制）；逐日行集过滤复用本模块 ``_filter_tradable_section`` 同一实现。
打分必须调用 ``v2/core/signal/ml_signal.py`` 同一 ``_build_model_input`` /
预测实现（共享函数、禁止复制）。域定义来源（写死）：
``universe/domains.py`` main 域市场白名单 + ``training_core._build_main_board_codes``；
可交易过滤语义与旧评估路径（split_training.py:215-220，``~col.astype(bool)``）一致。

预测行与信号 topk 产物（R2-T5A-06 + 复审 R2-T5A-R1-01 处置：阶段 E 归因与
§13.3.3 独立信号路径辅证的前置）：
- ``score_model_fingerprint(..., persist_predictions=True, predictions_path=...)`` ⇒
  集成主预测行（**非逐种子诊断行**）落盘 parquet：列 ``trade_date``（str，YYYYMMDD）/
  ``ts_code``（str）/ ``score``（float64）；键 ``(trade_date, ts_code)`` 升序、
  ``-0.0`` 归一 ``0.0``、写入前拒绝 NaN/Inf——与 S1 文本序列化共用
  ``normalize_prediction_rows`` / ``_normalize_score`` 同一实现（口径共用，禁止两套
  语义）；写出参数固定 ``engine="pyarrow", compression="snappy", index=False``
  （仓库 parquet 惯例的显式钉住，同输入 ⇒ 文件字节一致），tmp + ``os.replace``
  原子替换；
- ``signal_selection`` 提供时（键封闭集合 ``top_n`` / ``min_amount_ma20`` /
  ``min_total_mv`` / ``max_total_mv`` / ``exclude_financial``，缺键/未知键 ⇒
  ValueError；**必须同时 persist_predictions=True 并提供 signal_universe_config**）⇒
  逐打分日先以正式 ``BasicUniverse.get_stocks(date, quote_data=同日 clean/daily)``
  构造候选域（参数 exclude_st/min_list_days/markets 与回放同源；停牌/无行情过滤沿用
  BasicUniverse 默认行为），再对当日特征帧经**真实信号通路**同一实现过滤（候选域 +
  ``MLSignal._apply_selection_filters``，import 复用禁止复制；消费列 =
  ``amount_ma20`` / ``total_mv`` 恒在 + ``exclude_financial`` 时 ``sw_l1_code``，
  ml_signal.py:216/222/228-235）后 ``_build_model_input`` + ``_predict_scores``
  （``generate_ranked`` 同序），按 score 降序（同分 ts_code 升序，与
  辅证定秩口径一致）取 top-N，产信号路径
  选择行；落盘 ``signal_topk.parquet``（**路径写死 = ``predictions_path`` 同父目录**）：
  列 ``trade_date``（str，YYYYMMDD）/ ``rank``（int，自 1 起）/ ``ts_code``（str）/
  ``score``（float64）；``(trade_date, rank)`` 升序、``-0.0`` 归一、写前拒绝
  NaN/Inf、pyarrow/snappy/index=False 钉死——与预测行同一落盘纪律（任一产物含
  非法值 ⇒ 两产物均不落盘）；
- 结果 JSON 增键（仅 persist 开启时）：``predictions_path`` / ``predictions_sha256`` /
  ``predictions_rows``；（仅 signal_selection 提供时再加）``signal_topk_path`` /
  ``signal_topk_sha256`` / ``signal_topk_rows`` / ``signal_topk_days``（覆盖日清单）/
  ``signal_topk_empty_days``（合法零选股日显式登记）。路径登记统一走锚点机制
  （``_register_artifact_path``）：传 ``predictions_anchor``（臂 run 目录）⇒ 路径须
  落在锚点内（越界**写前**拒绝），登记相对锚点 POSIX 路径（如
  ``folds/split00/predictions.parquet``）；不传 ⇒ 登记传入路径原样 POSIX 形态；
- ``cs_train_verify`` / ``panel`` 两模式通用；panel 模式读取列契约随 signal_selection
  扩展过滤消费列（缺列 ⇒ store 契约明确报错）；cs_train 帧列全无需扩展。
  信号开启时要求 clean/stock_basic 及各日 clean/daily 完整，缺失明确拒绝，不回退
  未保护的 raw 元数据或静态主板集合；``signal_selection=None``（默认）不增加股票池/
  行情读取。主门矩阵与逐种子预测保持原口径，信号候选域只收窄信号辅证分支；
  ``persist_predictions=False`` 且 ``signal_selection=None``（默认）⇒ 不写文件、
  结果无新键，两模式行为与现状逐位一致；
- 消费方：``t5_attribution_sections.py`` b 节（键域/逐值归因证据粒度）、
  ``t5_compare_gates.py`` ``signal_topk_auxiliary``（§13.3.3 独立信号路径辅证）
  与 runner 续跑校验。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# 独立运行时仓库根不入 sys.path，按薄入口惯例显式引导
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Stock Connect 闭市识别独立模块（复杂度门拆分；t05b_fingerprint 保留同名 re-export）
from scripts.v2_p3.t05b_stock_connect import (  # noqa: E402, F401
    STOCK_CONNECT_CLOSED_REASON,
    resolve_stock_connect_closure_days,
)

# 行三元组：(YYYYMMDD 字符串, ts_code, 预测值)
PredictionRow = Tuple[str, str, float]

# 信号路径选择行四元组：(YYYYMMDD 字符串, rank 自 1 起, ts_code, 预测值)
SignalTopkRow = Tuple[str, int, str, float]

#: 信号 topk 产物文件名（写死；路径 = predictions_path 同父目录，R2-T5A-R1-01）
SIGNAL_TOPK_FILENAME = "signal_topk.parquet"

#: ``signal_selection`` 键封闭集合（缺键/未知键 ⇒ ValueError）
_SIGNAL_SELECTION_KEYS = (
    "top_n",
    "min_amount_ma20",
    "min_total_mv",
    "max_total_mv",
    "exclude_financial",
)

#: 信号候选池参数与回放 BasicUniverse 构造参数同源；停牌过滤沿用该类默认值。
_SIGNAL_UNIVERSE_KEYS = ("exclude_st", "min_list_days", "markets")

# 环境身份记录的线程/GPU 相关环境变量（缺失记为 None）
_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "CUDA_VISIBLE_DEVICES",
)

#: 打分源模式封闭集合（P3 T5：cs_train 验证源 + panel 源）
SOURCE_MODE_CS_TRAIN_VERIFY = "cs_train_verify"
SOURCE_MODE_PANEL = "panel"
SOURCE_MODES = (SOURCE_MODE_CS_TRAIN_VERIFY, SOURCE_MODE_PANEL)

#: 结果 JSON ``row_set_semantics`` 口径文本（两源分列；口径不同 ⇒ 禁止跨源比对指纹）
ROW_SET_SEMANTICS = {
    SOURCE_MODE_CS_TRAIN_VERIFY: ("cs_train 分区行（隐含标签非空）∩ 主板 ∩ 非 ST/停牌/涨停"),
    SOURCE_MODE_PANEL: "panel 当日行 ∩ 主板 ∩ 非 ST/停牌/涨停，无标签非空隐含条件",
}

#: panel 模式行集过滤列（panel core 族登记列，与 cs_train 分区同族同义；
#: 随读取列契约一并查询，未登记 ⇒ store 契约报错穿透）
_PANEL_FILTER_COLUMNS = ("is_st", "is_suspended", "is_limit_up")


def _validate_source_mode_args(
    source_mode: str,
    store: Optional[object],
    feature_columns: Optional[Sequence[str]],
) -> None:
    """跨源误用守卫（reviewer1-S7 处置）：模式非法 / 跨源参数组合即 ValueError。"""
    if source_mode not in SOURCE_MODES:
        raise ValueError(f"未知打分源模式: {source_mode!r}（合法值 {SOURCE_MODES}）")
    if source_mode == SOURCE_MODE_CS_TRAIN_VERIFY:
        if store is not None or feature_columns is not None:
            raise ValueError("cs_train_verify 模式禁止传 store/feature_columns（跨源误用守卫）")
        return
    if store is None:
        raise ValueError("panel 模式必须提供 PanelDataStore（store=）")
    if not feature_columns:
        raise ValueError("panel 模式必须提供非空 feature_columns（panel 读取列契约）")


def _validate_persist_args(persist_predictions: bool, predictions_path: Optional[Path]) -> None:
    """预测行持久化参数守卫（R2-T5A-06）：开启必须显式给路径、关闭禁止传路径。"""
    if persist_predictions and predictions_path is None:
        raise ValueError("persist_predictions=True 必须显式提供 predictions_path（预测行产物路径）")
    if not persist_predictions and predictions_path is not None:
        raise ValueError("persist_predictions=False 禁止传 predictions_path（误用守卫）")


def _validate_signal_selection_args(
    signal_selection: Optional[Mapping[str, object]], persist_predictions: bool
) -> None:
    """信号路径 top-k 参数守卫（R2-T5A-R1-01）：键封闭集合；开启必须 persist 开启。"""
    if signal_selection is None:
        return
    if not persist_predictions:
        raise ValueError(
            "signal_selection 开启必须 persist_predictions=True"
            "（信号产物路径写死为 predictions_path 同父目录的 signal_topk.parquet）"
        )
    missing = [key for key in _SIGNAL_SELECTION_KEYS if key not in signal_selection]
    if missing:
        raise ValueError(f"signal_selection 缺键 {missing}（封闭集合 {_SIGNAL_SELECTION_KEYS}）")
    unknown = sorted(set(signal_selection) - set(_SIGNAL_SELECTION_KEYS))
    if unknown:
        raise ValueError(f"signal_selection 未知键 {unknown}（封闭集合 {_SIGNAL_SELECTION_KEYS}）")


def _validate_signal_universe_args(
    signal_selection: Optional[Mapping[str, object]],
    signal_universe_config: Optional[Mapping[str, object]],
) -> None:
    """真实候选池必须显式配置；关闭信号辅证时不得引入额外数据读取。"""
    if signal_selection is None:
        if signal_universe_config is not None:
            raise ValueError("signal_selection 关闭时禁止传 signal_universe_config")
        return
    if signal_universe_config is None:
        raise ValueError("signal_selection 开启必须提供 signal_universe_config（真实候选股票池）")
    missing = sorted(set(_SIGNAL_UNIVERSE_KEYS) - set(signal_universe_config))
    unknown = sorted(set(signal_universe_config) - set(_SIGNAL_UNIVERSE_KEYS))
    if missing or unknown:
        raise ValueError(
            f"signal_universe_config 参数不完整或未知: 缺键 {missing} / 未知键 {unknown}"
        )


def _load_signal_candidate_codes(
    storage: object,
    score_days: Sequence[str],
    config: Mapping[str, object],
) -> Dict[str, Set[str]]:
    """复用回放 BasicUniverse 与同日 clean 行情，禁止用特征帧冒充行情或静态股票池。

    股票元数据只读消费集合已保护的 clean/stock_basic；行情只读，不触发 ensure。
    必需元数据/行情缺失直接拒绝，避免 BasicUniverse 缺列时跳过过滤。
    """
    from src.lazybull.data.loader import DataLoader
    from src.lazybull.universe import BasicUniverse

    loader = DataLoader(storage)
    stock_basic = loader.load_clean_stock_basic()
    required = {"ts_code", "name", "list_date", "market"}
    if stock_basic is None or stock_basic.empty:
        raise ValueError("信号候选股票池缺 clean/stock_basic（禁止回退未保护的 raw 元数据）")
    missing = sorted(required - set(stock_basic.columns))
    if missing:
        raise ValueError(f"信号候选股票池 stock_basic 缺必要列 {missing}")
    universe = BasicUniverse(
        stock_basic=stock_basic,
        exclude_st=bool(config["exclude_st"]),
        min_list_days=config["min_list_days"],
        markets=list(config["markets"]),
        verbose=False,
    )
    candidates: Dict[str, Set[str]] = {}
    for day in score_days:
        quote = loader.load_clean_daily_by_date(day, auto_ensure=False)
        if quote is None or quote.empty:
            raise ValueError(f"信号候选股票池 {day} 缺同日 clean/daily 行情")
        missing = sorted({"ts_code", "trade_date"} - set(quote.columns))
        if missing or not ({"is_suspended", "vol"} & set(quote.columns)):
            raise ValueError(f"信号候选股票池 {day} 行情缺必要列: {missing} / 停牌状态或成交量")
        quote = quote[quote["trade_date"].astype(str) == day]
        if quote.empty:
            raise ValueError(f"信号候选股票池 {day} clean/daily 不含同日行情")
        candidates[day] = set(universe.get_stocks(pd.Timestamp(day), quote_data=quote))
    return candidates


def _normalize_score(value: float, context: str) -> float:
    """预测分数归一（S1 口径唯一来源）：**拒绝 NaN 与 Inf**、``-0.0`` 归一 ``0.0``。"""
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"预测值含 NaN/Inf，按 S1 规格拒绝: {context}")
    if score == 0.0:
        score = 0.0  # -0.0 归一为 0.0（-0.0 == 0.0 为 True，赋值即归一）
    return score


def normalize_prediction_rows(rows: Iterable[PredictionRow]) -> List[PredictionRow]:
    """按 S1 冻结规格归一预测行（文本序列化与 parquet 产物共用同一实现，禁止两套语义）。

    键 ``(trade_date, ts_code)`` 升序；**拒绝 NaN 与 Inf**（出现即 ValueError）；
    ``-0.0`` 归一为 ``0.0``（``-0.0 == 0.0`` 为 True，赋值即归一）。
    """
    normalized: List[PredictionRow] = []
    for trade_date, ts_code, value in sorted(rows, key=lambda row: (row[0], row[1])):
        score = _normalize_score(value, f"{trade_date},{ts_code}")
        normalized.append((str(trade_date), str(ts_code), score))
    return normalized


def serialize_prediction_rows(rows: Iterable[PredictionRow]) -> str:
    """按 S1 冻结规格序列化预测行：键升序、``-0.0→0.0``、拒绝 NaN/Inf、LF 结尾。"""
    normalized = normalize_prediction_rows(rows)
    lines = [f"{day},{code},{score:.17g}" for day, code, score in normalized]
    return "\n".join(lines) + ("\n" if lines else "")


def sha256_text(text: str) -> str:
    """UTF-8 编码的 sha256 十六进制摘要。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint_prediction_rows(rows: Iterable[PredictionRow]) -> str:
    """预测行集合的指纹 = 序列化文本的 sha256。"""
    return sha256_text(serialize_prediction_rows(rows))


def persist_prediction_rows(rows: Iterable[PredictionRow], path: Path) -> Tuple[int, str]:
    """集成主预测行 parquet 产物落盘（R2-T5A-06 预测行持久化；两打分源通用）。

    口径与 S1 文本序列化共用 ``normalize_prediction_rows`` 同一实现（禁止两套语义）：
    键 ``(trade_date, ts_code)`` 升序、``-0.0`` 归一 ``0.0``、拒绝 NaN/Inf——校验
    先于任何文件/目录创建，拒绝即不落盘（亦无 tmp 残留）。列 = ``trade_date``
    （str，YYYYMMDD）/ ``ts_code``（str）/ ``score``（float64）；写出参数固定
    ``engine="pyarrow", compression="snappy", index=False``（仓库 parquet 惯例的
    显式钉住，同输入 ⇒ 文件字节一致）；tmp + ``os.replace`` 原子替换。

    Returns:
        (行数, 写入文件内容的 sha256 十六进制摘要)。
    """
    normalized = normalize_prediction_rows(rows)
    frame = pd.DataFrame(normalized, columns=["trade_date", "ts_code", "score"])
    frame = frame.astype({"trade_date": str, "ts_code": str, "score": np.float64})
    return _write_parquet_artifact(frame, path)


def normalize_signal_topk_rows(rows: Iterable[SignalTopkRow]) -> List[SignalTopkRow]:
    """信号 topk 行归一（分数口径与 S1 共用 ``_normalize_score`` 同一实现）。

    键 ``(trade_date, rank)`` 升序；**拒绝 NaN 与 Inf**；``-0.0`` 归一 ``0.0``。
    """
    normalized: List[SignalTopkRow] = []
    for trade_date, rank, ts_code, value in sorted(rows, key=lambda row: (row[0], row[1])):
        score = _normalize_score(value, f"{trade_date},rank{rank},{ts_code}")
        normalized.append((str(trade_date), int(rank), str(ts_code), score))
    return normalized


def persist_signal_topk_rows(rows: Iterable[SignalTopkRow], path: Path) -> Tuple[int, str]:
    """信号路径 top-k 产物落盘（R2-T5A-R1-01；与预测行同一落盘纪律）。

    列 = ``trade_date``（str，YYYYMMDD）/ ``rank``（int64，自 1 起）/ ``ts_code``
    （str）/ ``score``（float64）；``(trade_date, rank)`` 升序、``-0.0`` 归一、
    写前拒绝 NaN/Inf（共用 ``normalize_signal_topk_rows``，校验先于任何文件创建）；
    写出参数固定 pyarrow + snappy + index=False（同输入 ⇒ 文件字节一致）；
    tmp + ``os.replace`` 原子替换。零选股批次（全空）落 0 行 parquet 并随结果登记。

    Returns:
        (行数, 写入文件内容的 sha256 十六进制摘要)。
    """
    normalized = normalize_signal_topk_rows(rows)
    frame = pd.DataFrame(normalized, columns=["trade_date", "rank", "ts_code", "score"])
    frame = frame.astype({"trade_date": str, "rank": np.int64, "ts_code": str, "score": np.float64})
    return _write_parquet_artifact(frame, path)


def _write_parquet_artifact(frame: pd.DataFrame, path: Path) -> Tuple[int, str]:
    """parquet 产物原子落盘（唯一写出实现）：pyarrow + snappy + index=False 钉死。

    Returns:
        (行数, 写入文件内容的 sha256 十六进制摘要)。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp_path, engine="pyarrow", compression="snappy", index=False)
    digest = hashlib.sha256(tmp_path.read_bytes()).hexdigest()
    os.replace(tmp_path, path)
    return len(frame), digest


def environment_identity() -> Dict[str, object]:
    """环境身份（记录 + 约束，§5-3）：解释器/关键库版本 + 线程与 GPU 环境变量 + CPU 数。"""
    import pyarrow
    import xgboost

    return {
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "pyarrow": pyarrow.__version__,
        "xgboost": xgboost.__version__,
        "cpu_count": os.cpu_count(),
        "env_vars": {name: os.environ.get(name) for name in _ENV_VARS},
    }


def list_oos_score_days(features_root: Path, test_start: str, test_end: str) -> List[str]:
    """OOS 打分日 = 测试窗 [test_start, test_end] 内 cs_train 实际分区日（YYYYMMDD 升序）。"""
    cs_train_dir = Path(features_root) / "cs_train"
    if not cs_train_dir.is_dir():
        raise FileNotFoundError(f"cs_train 分区目录不存在: {cs_train_dir}")
    days = []
    for path in cs_train_dir.glob("*.parquet"):
        day = path.stem
        if len(day) == 8 and day.isdigit() and test_start <= day <= test_end:
            days.append(day)
    days.sort()
    if not days:
        raise ValueError(f"测试窗 [{test_start}, {test_end}] 内无 cs_train 分区: {cs_train_dir}")
    return days


def list_panel_score_days(store: object, test_start: str, test_end: str) -> List[str]:
    """panel 源打分日域 = 窗口内 panel 分区日（YYYYMMDD 升序；禁止 glob cs_train）。

    PanelDataStore 无公开分区日清单 API ⇒ 经 ``load_features`` 键列整窗读取后
    按 ``trade_date`` 取日域（core 键域空列帧，不落特征列数据）。
    """
    from src.lazybull.v2.common.types import FeatureQuery, TradeDate

    panel_frame = store.load_features(
        FeatureQuery(
            columns=["trade_date", "ts_code"],
            start_date=TradeDate.from_str(test_start),
            end_date=TradeDate.from_str(test_end),
        )
    )
    days = sorted({str(day) for day in panel_frame.df.index.get_level_values("trade_date")})
    if not days:
        raise ValueError(f"panel 在 [{test_start}, {test_end}] 内无分区日")
    return days


def resolve_panel_score_days(
    expected_days: Sequence[str], store: object, test_start: str, test_end: str
) -> List[str]:
    """预期打分日 ∩ panel 分区日域：缺失即失败并列出缺失日（N3 口径的 panel 对应件）。"""
    existing = set(list_panel_score_days(store, test_start, test_end))
    missing = [day for day in expected_days if day not in existing]
    if missing:
        suffix = "…" if len(missing) > 10 else ""
        raise RuntimeError(
            f"预期打分日缺 panel 分区（{len(missing)} 日）: {missing[:10]}{suffix}（N3）"
        )
    return list(expected_days)


def _load_panel_score_frames(
    store: object,
    feature_columns: Sequence[str],
    score_days: Sequence[str],
    extra_columns: Sequence[str] = (),
) -> Dict[str, pd.DataFrame]:
    """panel 整窗读取 + 按日切分（replay.py:295-308 同一实现 import 复用，禁止复制）。

    读取列契约 = ``feature_columns`` ∪ ``_PANEL_FILTER_COLUMNS`` ∪ ``extra_columns``
    （保序去重；过滤列随契约一并查询；``extra_columns`` = 信号选择过滤消费列，
    未登记 ⇒ store 契约报错穿透）；返回 {YYYYMMDD: 当日平铺帧}，其键集即 panel 分区日域。
    """
    from src.lazybull.v2.common.types import FeatureQuery, TradeDate
    from src.lazybull.v2.hosts.backtest.replay import _split_panel_frame_by_date

    if not score_days:
        raise ValueError("score_days 为空，禁止产出指纹（panel 模式整窗读取无锚）")
    query_columns = list(dict.fromkeys([*feature_columns, *_PANEL_FILTER_COLUMNS, *extra_columns]))
    panel_frame = store.load_features(
        FeatureQuery(
            columns=query_columns,
            start_date=TradeDate.from_str(min(score_days)),
            end_date=TradeDate.from_str(max(score_days)),
        )
    )
    return _split_panel_frame_by_date(panel_frame.df)


def load_main_board_codes(data_root: Path, stock_domain: str = "main") -> Set[str]:
    """域股票池（写死来源）：stock_basic × ``domain_market_whitelist`` 市场白名单。"""
    from src.lazybull.data.loader import DataLoader
    from src.lazybull.data.storage import Storage
    from src.lazybull.ml.walk_forward.training_core import _build_main_board_codes
    from src.lazybull.universe.domains import domain_market_whitelist

    storage = Storage(root_path=str(data_root))
    loader = DataLoader(storage)
    stock_basic = loader.load_clean_stock_basic()
    if stock_basic is None:
        stock_basic = loader.load_stock_basic()
    markets = tuple(domain_market_whitelist(stock_domain))
    codes = _build_main_board_codes(stock_basic, markets=markets)
    logger.info(f"域股票池（{stock_domain}，markets={markets}）: {len(codes)} 只")
    return set(codes)


def _filter_tradable_section(df: pd.DataFrame) -> pd.DataFrame:
    """非 ST/停牌/涨停过滤（语义同 split_training.py:215-220：``~col.astype(bool)``）。"""
    mask = pd.Series(True, index=df.index)
    for col in ("is_st", "is_suspended", "is_limit_up"):
        if col in df.columns:
            mask = mask & (~df[col].astype(bool))
    return df[mask]


def assert_feature_names_aligned(
    reference: Sequence[str], actual: Sequence[str], context: str
) -> None:
    """特征名逐列同名同序断言（§13.3.3-B6-①：不一致直接失败，**禁止取交集降级**）。

    G-code 两侧模型与冻结矩阵列集的比对该断言为硬门；任何顺序/集合差异即拒绝。
    """
    reference = list(reference)
    actual = list(actual)
    if reference != actual:
        ref_set, act_set = set(reference), set(actual)
        raise ValueError(
            f"特征名逐列同名同序断言失败（{context}）："
            f"参照 {len(reference)} 列 / 实际 {len(actual)} 列；"
            f"仅参照侧 {sorted(ref_set - act_set)[:5]}；仅实际侧 {sorted(act_set - ref_set)[:5]}；"
            f"首个乱序位置 = {_first_mismatch_index(reference, actual)}"
        )


def _first_mismatch_index(reference: Sequence[str], actual: Sequence[str]) -> int:
    for idx, (left, right) in enumerate(zip(reference, actual)):
        if left != right:
            return idx
    return min(len(reference), len(actual))


def resolve_sub_model_seed(sub_model: object) -> Optional[int]:
    """从子模型对象解析其实际 ``random_state``（R2-T05B-05：禁止按请求顺序贴标签）。

    依次尝试：sklearn 包装 ``get_params()['random_state']``、早停包装 ``base_model``
    同名参数、Booster 属性；均不可得返回 None（调用方以 index 占位并显式标注）。
    """
    candidates = [sub_model, getattr(sub_model, "__dict__", {}).get("base_model")]
    for candidate in candidates:
        if candidate is None:
            continue
        get_params = getattr(candidate, "get_params", None)
        if callable(get_params):
            try:
                seed = get_params().get("random_state")
            except (TypeError, AttributeError):
                seed = None
            if seed is not None:
                return int(seed)
        seed = getattr(candidate, "random_state", None)
        if seed is not None:
            return int(seed)
    return None


def _register_artifact_path(path: Path, anchor: Optional[Path]) -> str:
    """产物路径登记（锚点机制唯一实现，R2-T5A-06 锚点复核处置；预测行/信号 topk 共用）。

    传 ``anchor``（臂 run 目录）⇒ 路径须落在锚点内（**越界写前拒绝**），登记相对
    锚点的 POSIX 路径（如 ``folds/split00/predictions.parquet``；归因消费侧
    ``t5_attribution_io.load_fold_predictions`` 只接受相对臂目录口径）；不传锚点 ⇒
    登记传入路径原样的 POSIX 形态。
    """
    if anchor is None:
        return Path(path).as_posix()
    anchor_resolved = Path(anchor).resolve()
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(anchor_resolved):
        raise ValueError(
            f"产物路径越出锚点目录（拒绝登记绝对/越界路径）: {resolved} !< {anchor_resolved}"
        )
    return resolved.relative_to(anchor_resolved).as_posix()


def _maybe_persist_predictions(
    ensemble_rows: List[PredictionRow],
    persist_predictions: bool,
    predictions_path: Optional[Path],
    predictions_anchor: Optional[Path] = None,
) -> Optional[Dict[str, object]]:
    """预测行产物落盘 + 结果键组装（R2-T5A-06）；persist 关闭 ⇒ None（现状逐位不变）。

    ``predictions_path is None`` 分支不可达（``_validate_persist_args`` 入口守卫）。
    """
    if not persist_predictions or predictions_path is None:
        return None
    registered = _register_artifact_path(predictions_path, predictions_anchor)
    rows_written, digest = persist_prediction_rows(ensemble_rows, predictions_path)
    info: Dict[str, object] = {
        "predictions_path": registered,
        "predictions_sha256": digest,
        "predictions_rows": rows_written,
    }
    logger.info(f"预测行产物已落盘: {registered}（{rows_written} 行，sha256 {digest}）")
    return info


def _maybe_persist_signal_topk(
    signal_rows: Optional[List[SignalTopkRow]],
    signal_days: Optional[List[str]],
    signal_empty_days: Optional[List[str]],
    predictions_path: Optional[Path],
    predictions_anchor: Optional[Path],
) -> Optional[Dict[str, object]]:
    """信号 topk 产物落盘 + 结果键组装（R2-T5A-R1-01）；信号路径关闭 ⇒ None。

    产物路径写死 = ``predictions_path`` 同父目录的 ``signal_topk.parquet``；
    锚点登记复用 ``_register_artifact_path``（越界写前拒绝）。``signal_rows is None``
    ⇒ 关闭；``predictions_path is None`` 分支不可达（入口守卫已保证 persist 开启）。
    """
    if signal_rows is None or predictions_path is None:
        return None
    topk_path = Path(predictions_path).parent / SIGNAL_TOPK_FILENAME
    registered = _register_artifact_path(topk_path, predictions_anchor)
    rows_written, digest = persist_signal_topk_rows(signal_rows, topk_path)
    info: Dict[str, object] = {
        "signal_topk_path": registered,
        "signal_topk_sha256": digest,
        "signal_topk_rows": rows_written,
        "signal_topk_days": list(signal_days or []),
        "signal_topk_empty_days": list(signal_empty_days or []),
    }
    logger.info(
        f"信号 topk 产物已落盘: {registered}（{rows_written} 行，"
        f"覆盖 {len(info['signal_topk_days'])} 日，零选股 {len(info['signal_topk_empty_days'])} 日，"
        f"sha256 {digest}）"
    )
    return info


def _signal_filter_columns(signal_selection: Mapping[str, object]) -> List[str]:
    """真实 ``_apply_selection_filters`` 实际消费列（panel 读取列契约扩展来源）。

    来源行号（ml_signal.py）：``amount_ma20``（:216-217）与 ``total_mv``（:222-223）
    恒消费；``sw_l1_code``（:228-235）仅 ``exclude_financial=True`` 时消费。
    """
    columns = ["amount_ma20", "total_mv"]
    if bool(signal_selection["exclude_financial"]):
        columns.append("sw_l1_code")
    return columns


def _configure_signal_selection(signal: object, signal_selection: Mapping[str, object]) -> int:
    """信号选择参数装订到 signal 实例（真实 ``_apply_selection_filters`` 读实例属性：
    ml_signal.py:217/223/228）；返回 top_n（排序取前 N 由本工具侧兑现）。"""
    signal.min_amount_ma20 = float(signal_selection["min_amount_ma20"])
    signal.min_total_mv = float(signal_selection["min_total_mv"])
    signal.max_total_mv = float(signal_selection["max_total_mv"])
    signal.exclude_financial = bool(signal_selection["exclude_financial"])
    return int(signal_selection["top_n"])


def score_model_fingerprint(
    models_dir: Path,
    model_version: int,
    data_root: Path,
    score_days: Sequence[str],
    main_board_codes: Set[str],
    strict_days: bool = True,
    producer_identity: Optional[Dict[str, object]] = None,
    source_mode: str = SOURCE_MODE_CS_TRAIN_VERIFY,
    store: Optional[object] = None,
    feature_columns: Optional[Sequence[str]] = None,
    persist_predictions: bool = False,
    predictions_path: Optional[Path] = None,
    predictions_anchor: Optional[Path] = None,
    signal_selection: Optional[Mapping[str, object]] = None,
    signal_universe_config: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """对指定模型版本在冻结矩阵上打分并产出指纹（集成主指纹 + 逐种子诊断指纹）。

    打分调用 v2 ``MLSignal._build_model_input`` / ``_predict_scores`` 同一实现；
    集成聚合 = ``EnsembleModel.predict``（子模型列表顺序固定取均值）。

    覆盖完整性门（R2-T05B-03）：``strict_days=True`` 时任一预期打分日缺分区 /
    过滤后为空 / 被质量门禁拒绝 ⇒ 直接失败，禁止缩域后继续产出「通过」指纹；
    空预测集拒绝产出指纹。合法空日只能由调用方在 ``score_days`` 构造时显式剔除
    并登记（本工具不接受隐式缩域）。

    键域完整性（R2-T05B-03 复核余项）：逐日断言主预测与各子模型预测**长度等于
    冻结输入行数**（禁止 zip 静默截断），逐日 ts_code 键唯一。

    逐种子诊断（R2-T05B-05）：子模型按验证得分排序入集成（training_core.py:458），
    诊断标签一律取子模型**实际** ``random_state``，请求顺序与实际聚合顺序分列记录。

    行集口径注记（reviewer1-S7，P3 T5 起本函数服务两模式）：
    ``cs_train_verify``（默认）行集 = cs_train 分区行（**隐含标签非空**）∩ 主板 ∩
    非 ST/停牌/涨停；``panel`` 行集 = panel 当日行 ∩ 主板 ∩ 非 ST/停牌/涨停
    （**无标签非空隐含条件**；过滤列 = panel core 族登记列，随读取列契约查询）。
    **两源口径不同，禁止跨源比对指纹**；跨源误用守卫：cs_train_verify 传
    store/feature_columns、panel 缺 store 或 feature_columns ⇒ ValueError。

    预测行与信号 topk 产物（产物格式与消费方见模块 docstring「预测行与信号 topk
    产物」节）：``persist_predictions=True`` ⇒ 集成主预测行落盘 parquet，结果增键
    ``predictions_path`` / ``predictions_sha256`` / ``predictions_rows``；
    ``signal_selection`` 提供时（必须 persist 开启）⇒ 逐打分日经真实信号通路
    （``MLSignal._apply_selection_filters`` 同一实现 + ``_build_model_input`` +
    ``_predict_scores``）产 top-N 选择行，落盘 ``signal_topk.parquet``
    （``predictions_path`` 同父目录，文件名写死），结果增键 ``signal_topk_path`` /
    ``signal_topk_sha256`` / ``signal_topk_rows`` / ``signal_topk_days`` /
    ``signal_topk_empty_days``（合法零选股日显式登记）。默认全关 ⇒ 不写文件、
    结果无新键，两模式行为与现状逐位一致。

    Args:
        source_mode: 打分源（``cs_train_verify`` 默认 / ``panel``）。
        store: panel 模式必传的 PanelDataStore（只读消费）。
        feature_columns: panel 模式必传非空的读取列契约（模型特征列；过滤列
            is_st/is_suspended/is_limit_up 由本函数随契约一并查询，无需调用方列入；
            signal_selection 提供时再把信号过滤消费列保序并入，缺列 ⇒ store 契约报错）。
        persist_predictions: 预测行持久化开关（默认 False）。
        predictions_path: parquet 产物路径；persist 开启时必填、关闭时禁传
            （守卫于打分前生效）。结果登记值：传 ``predictions_anchor`` 时登记为
            相对锚点（臂 run 目录）的 POSIX 路径（消费侧契约口径）；不传则登记
            传入路径原样的 POSIX 形态。
        predictions_anchor: 臂 run 目录锚点（可选）；传入后 ``predictions_path``
            必须落在锚点内，结果登记相对路径。
        signal_selection: 信号路径 top-k 选择参数（可选；键封闭集合 ``top_n`` /
            ``min_amount_ma20`` / ``min_total_mv`` / ``max_total_mv`` /
            ``exclude_financial``，缺键/未知键 ⇒ ValueError；提供时必须
            ``persist_predictions=True``）。默认 None ⇒ 信号路径关闭。
        signal_universe_config: 开启信号辅证时必填，exclude_st/min_list_days/markets
            与真实回放同源；停牌与无行情过滤调用正式 BasicUniverse.get_stocks。
    """
    _validate_source_mode_args(source_mode, store, feature_columns)
    _validate_persist_args(persist_predictions, predictions_path)
    _validate_signal_selection_args(signal_selection, persist_predictions)
    _validate_signal_universe_args(signal_selection, signal_universe_config)

    from src.lazybull.data.storage import Storage
    from src.lazybull.v2.core.signal.ml_signal import MLSignal

    storage = Storage(root_path=str(data_root))
    signal = MLSignal(
        top_n=20,
        model_version=int(model_version),
        models_dir=str(models_dir),
        verbose=False,
    )
    signal._load_model()
    model_columns = list(signal.feature_columns or [])

    # B6-①：模型 metadata 特征列与 Booster 特征名逐列同名同序（可取得时）
    booster_feature_names = _extract_booster_feature_names(signal.model)
    if booster_feature_names is not None:
        assert_feature_names_aligned(
            model_columns, booster_feature_names, context=f"v{model_version} 模型特征名"
        )

    panel_frames: Optional[Dict[str, pd.DataFrame]] = None
    if source_mode == SOURCE_MODE_PANEL:
        extra_columns: Sequence[str] = (
            _signal_filter_columns(signal_selection) if signal_selection is not None else ()
        )
        panel_frames = _load_panel_score_frames(
            store, feature_columns, score_days, extra_columns=extra_columns
        )

    signal_top_n: Optional[int] = None
    signal_candidates: Optional[Dict[str, Set[str]]] = None
    if signal_selection is not None:
        assert signal_universe_config is not None  # 参数守卫已先行，供类型检查明确收窄。
        signal_top_n = _configure_signal_selection(signal, signal_selection)
        signal_candidates = _load_signal_candidate_codes(
            storage, score_days, signal_universe_config
        )

    # 窄口径合法闭市日（§19.18）：仅 Stock Connect 闭市；其余拒绝仍由严格日门拒绝。
    excluded_days = resolve_stock_connect_closure_days(
        Path(data_root), score_days, model_columns, panel_frames
    )
    effective_score_days = [day for day in score_days if day not in excluded_days]

    ensemble_rows, seed_rows, used_days, signal_rows, signal_days, signal_empty_days = (
        _score_all_days(
            signal,
            storage,
            panel_frames,
            effective_score_days,
            main_board_codes,
            model_columns,
            strict_days,
            signal_top_n,
            signal_candidates,
        )
    )

    if not ensemble_rows:
        raise ValueError("空预测集禁止产出指纹（R2-T05B-03）")

    if signal_rows is not None:
        # 两产物写前拒绝纪律（R2-T5A-R1-01）：信号行 NaN/Inf 先验——任一产物含非法值
        # ⇒ 两产物均不落盘（预测行 NaN/Inf 由 persist_prediction_rows 自身写前拒绝）
        normalize_signal_topk_rows(signal_rows)
    persistence_info = _maybe_persist_predictions(
        ensemble_rows, persist_predictions, predictions_path, predictions_anchor
    )
    signal_info = _maybe_persist_signal_topk(
        signal_rows, signal_days, signal_empty_days, predictions_path, predictions_anchor
    )
    result = _build_fingerprint_result(
        signal,
        model_version,
        used_days,
        ensemble_rows,
        seed_rows,
        model_columns,
        producer_identity,
        source_mode,
        excluded_days,
    )
    for info in (persistence_info, signal_info):
        if info is not None:
            result.update(info)
    return result


#: 单日打分返回：(集成行, 逐种子行, 信号 topk 行)；集成行 None ⇒ 当日被拒绝
_DayScore = Tuple[
    Optional[List[PredictionRow]],
    List[List[PredictionRow]],
    Optional[List[SignalTopkRow]],
]


def _signal_topk_for_day(
    signal: object, df: pd.DataFrame, day: str, top_n: int
) -> Optional[List[SignalTopkRow]]:
    """单日信号路径 top-N 选择行（真实 ``_apply_selection_filters`` 同一实现 import 复用）。

    口径 = 调用方先应用 BasicUniverse.get_stocks 的逐日候选域，再复用真实信号通路
    ``_load_prediction_frame`` 的成交额/市值/金融股选择过滤 + ``_build_model_input``
    + ``_predict_scores(ranked=True)``（``generate_ranked`` 同序）；排序 = score 降序、
    同分 ts_code 升序（与 ``t5_compare_gates._derive_signal_topk`` 定秩口径一致），
    rank 自 1 起。返回 None ⇒ 质量门禁拒绝（调用方按 strict 语义处置）；
    空列表 ⇒ 合法零选股日（登记 ``signal_topk_empty_days``）。
    """
    filtered, _, _ = signal._apply_selection_filters(df)
    if len(filtered) == 0:
        return []
    built = signal._build_model_input(pd.Timestamp(day), filtered)
    if built is None:
        return None
    features_df, X = built
    scores = signal._predict_scores(X, 0, 0, ranked=True)
    _assert_prediction_length(day, "信号路径主预测", scores, len(features_df))
    ranked = sorted(
        zip(features_df["ts_code"].tolist(), scores),
        key=lambda item: (-float(item[1]), str(item[0])),
    )[:top_n]
    return [
        (day, rank, str(code), float(score)) for rank, (code, score) in enumerate(ranked, start=1)
    ]


def _score_one_day(
    signal: object,
    storage: object,
    day: str,
    main_board_codes: Set[str],
    feature_columns: List[str],
    strict_days: bool,
    signal_top_n: Optional[int] = None,
    signal_universe_codes: Optional[Set[str]] = None,
) -> _DayScore:
    """单日冻结矩阵打分（cs_train 验证源）；strict 时任何拒绝即抛错。"""
    df = storage.load_cs_train_day(day)
    if df is None or len(df) == 0:
        _day_refused(day, "无 cs_train 分区数据", strict_days)
        return None, [], None
    return _score_day_frame(
        signal,
        df,
        day,
        main_board_codes,
        feature_columns,
        strict_days,
        signal_top_n,
        signal_universe_codes,
    )


def _score_one_day_panel(
    signal: object,
    panel_frames: Dict[str, pd.DataFrame],
    day: str,
    main_board_codes: Set[str],
    feature_columns: List[str],
    strict_days: bool,
    signal_top_n: Optional[int] = None,
    signal_universe_codes: Optional[Set[str]] = None,
) -> _DayScore:
    """单日冻结矩阵打分（panel 源）；打分日缺 panel 分区日 ⇒ strict 语义下失败。"""
    df = panel_frames.get(day)
    if df is None or len(df) == 0:
        _day_refused(day, "无 panel 分区数据", strict_days)
        return None, [], None
    return _score_day_frame(
        signal,
        df,
        day,
        main_board_codes,
        feature_columns,
        strict_days,
        signal_top_n,
        signal_universe_codes,
    )


def _score_day_frame(
    signal: object,
    df: pd.DataFrame,
    day: str,
    main_board_codes: Set[str],
    feature_columns: List[str],
    strict_days: bool,
    signal_top_n: Optional[int] = None,
    signal_universe_codes: Optional[Set[str]] = None,
) -> _DayScore:
    """单日行集过滤 + 模型输入构建 + 集成/逐种子预测（两源共享打分核，口径唯一）。

    信号路先按正式 BasicUniverse 的逐日候选域过滤，再经选择过滤/预测产 top-N。
    主门保持原主板与可交易矩阵；信号路不叠加主门的涨停过滤（T 日涨停可进入候选）。
    """
    signal_rows: Optional[List[SignalTopkRow]] = None
    if signal_top_n is not None:
        if signal_universe_codes is None:
            raise ValueError(f"{day} 信号辅证缺正式 BasicUniverse 候选域，禁止静态股票池回退")
        signal_frame = df[df["ts_code"].isin(signal_universe_codes)]
        signal_rows = _signal_topk_for_day(signal, signal_frame, day, signal_top_n)
        if signal_rows is None:
            _day_refused(day, "信号路径模型输入构建被质量门禁拒绝", strict_days)
            return None, [], None
    df = _filter_tradable_section(df[df["ts_code"].isin(main_board_codes)])
    if len(df) == 0:
        _day_refused(day, "过滤后截面为空", strict_days)
        return None, [], None
    built = signal._build_model_input(pd.Timestamp(day), df)
    if built is None:
        _day_refused(day, "模型输入构建被质量门禁拒绝", strict_days)
        return None, [], None
    features_df, X = built
    assert_feature_names_aligned(feature_columns, list(X.columns), context=f"{day} 冻结矩阵列集")

    ts_codes = [str(code) for code in features_df["ts_code"].tolist()]
    if len(set(ts_codes)) != len(ts_codes):
        raise RuntimeError(f"{day} 冻结矩阵 ts_code 键不唯一，键域完整性拒绝（R2-T05B-03）")

    predictions = signal._predict_scores(X, 0, 0, ranked=False)
    _assert_prediction_length(day, "主预测", predictions, len(ts_codes))
    ensemble_rows = [(day, code, float(score)) for code, score in zip(ts_codes, predictions)]

    seed_rows: List[List[PredictionRow]] = []
    for idx, sub_model in enumerate(getattr(signal.model, "models", None) or []):
        sub_predictions = sub_model.predict(X)
        _assert_prediction_length(day, f"子模型[{idx}]", sub_predictions, len(ts_codes))
        seed_rows.append(
            [(day, code, float(score)) for code, score in zip(ts_codes, sub_predictions)]
        )
    return ensemble_rows, seed_rows, signal_rows


def _accumulate_signal_day(
    day: str,
    day_signal: Optional[List[SignalTopkRow]],
    signal_rows: List[SignalTopkRow],
    signal_days: List[str],
    signal_empty_days: List[str],
) -> None:
    """逐日信号行累计：非空 ⇒ 覆盖日登记；空 ⇒ 合法零选股日显式登记。"""
    if day_signal:
        signal_rows.extend(day_signal)
        signal_days.append(day)
    else:
        signal_empty_days.append(day)


def _score_all_days(
    signal: object,
    storage: object,
    panel_frames: Optional[Dict[str, pd.DataFrame]],
    score_days: Sequence[str],
    main_board_codes: Set[str],
    feature_columns: List[str],
    strict_days: bool,
    signal_top_n: Optional[int],
    signal_candidates: Optional[Mapping[str, Set[str]]] = None,
) -> Tuple[
    List[PredictionRow],
    List[List[PredictionRow]],
    List[str],
    Optional[List[SignalTopkRow]],
    Optional[List[str]],
    Optional[List[str]],
]:
    """逐日打分循环（两源分发）：(集成行, 逐种子行, 覆盖日, 信号行/覆盖日/零选股日)。

    信号三元组在 ``signal_top_n is None``（信号路径关闭）时全为 None。
    """
    signal_on = signal_top_n is not None
    ensemble_rows: List[PredictionRow] = []
    seed_rows: List[List[PredictionRow]] = []
    used_days: List[str] = []
    signal_rows: List[SignalTopkRow] = []
    signal_days: List[str] = []
    signal_empty_days: List[str] = []
    for day in score_days:
        universe_codes = signal_candidates[day] if signal_candidates is not None else None
        if panel_frames is not None:
            day_ensemble, day_seeds, day_signal = _score_one_day_panel(
                signal,
                panel_frames,
                day,
                main_board_codes,
                feature_columns,
                strict_days,
                signal_top_n,
                universe_codes,
            )
        else:
            day_ensemble, day_seeds, day_signal = _score_one_day(
                signal,
                storage,
                day,
                main_board_codes,
                feature_columns,
                strict_days,
                signal_top_n,
                universe_codes,
            )
        if day_ensemble is None:
            continue
        ensemble_rows.extend(day_ensemble)
        while len(seed_rows) < len(day_seeds):
            seed_rows.append([])
        for idx, rows in enumerate(day_seeds):
            seed_rows[idx].extend(rows)
        used_days.append(day)
        if signal_on:
            _accumulate_signal_day(day, day_signal, signal_rows, signal_days, signal_empty_days)
    if not signal_on:
        return ensemble_rows, seed_rows, used_days, None, None, None
    return ensemble_rows, seed_rows, used_days, signal_rows, signal_days, signal_empty_days


def _day_refused(
    day: str, reason: str, strict_days: bool
) -> Tuple[Optional[List[PredictionRow]], List[List[PredictionRow]]]:
    if strict_days:
        raise RuntimeError(f"{day} {reason}，覆盖完整性门拒绝（R2-T05B-03）")
    logger.warning(f"{day} {reason}，跳过")
    return None, []


def _assert_prediction_length(day: str, label: str, predictions: object, expected: int) -> None:
    """预测长度 == 冻结输入行数（R2-T05B-03 复核余项：禁止 zip 截断放行短预测）。"""
    actual = len(predictions)  # type: ignore[arg-type]
    if actual != expected:
        raise RuntimeError(
            f"{day} {label}长度 {actual} != 冻结输入行数 {expected}，键域完整性拒绝（R2-T05B-03）"
        )


def _build_fingerprint_result(
    signal: object,
    model_version: int,
    used_days: List[str],
    ensemble_rows: List[PredictionRow],
    seed_rows: List[List[PredictionRow]],
    feature_columns: List[str],
    producer_identity: Optional[Dict[str, object]],
    source_mode: str,
    excluded_days: Optional[Mapping[str, str]] = None,
) -> Dict[str, object]:
    """汇总指纹结果（请求/实际种子顺序分列；生产者身份与行集口径随附）。"""
    metadata = signal.metadata or {}
    train_params = metadata.get("train_params", {}) if isinstance(metadata, dict) else {}
    requested_seeds = train_params.get("ensemble_seeds")
    actual_seed_order: List[object] = []
    for idx, sub_model in enumerate(getattr(signal.model, "models", None) or []):
        seed = resolve_sub_model_seed(sub_model)
        actual_seed_order.append(seed if seed is not None else f"index_{idx}")

    result: Dict[str, object] = {
        "model_version": int(model_version),
        "source_mode": source_mode,
        "row_set_semantics": ROW_SET_SEMANTICS[source_mode],
        "n_days": len(used_days),
        "n_rows": len(ensemble_rows),
        "score_days": used_days,
        "feature_columns_count": len(feature_columns),
        "requested_ensemble_seeds": requested_seeds,
        "actual_seed_order": actual_seed_order,
        "fingerprint": fingerprint_prediction_rows(ensemble_rows),
        "per_seed_diagnostic": [
            {
                "seed": actual_seed_order[idx] if idx < len(actual_seed_order) else f"index_{idx}",
                "n_rows": len(rows),
                "fingerprint": fingerprint_prediction_rows(rows),
            }
            for idx, rows in enumerate(seed_rows)
        ],
    }
    if producer_identity is not None:
        result["producer_identity"] = dict(producer_identity)
    if excluded_days:
        ordered = sorted(excluded_days)
        result["excluded_days"] = ordered
        result["excluded_day_reasons"] = {day: excluded_days[day] for day in ordered}
    logger.info(
        f"指纹打分完成: v{model_version}，{result['n_days']} 日 {result['n_rows']} 行，"
        f"指纹 {result['fingerprint']}，实际种子顺序 {actual_seed_order}"
    )
    return result


def _extract_booster_feature_names(model: object) -> Optional[List[str]]:
    """提取 XGBoost Booster 特征名（不可取得时返回 None，由调用方决定是否跳过断言）。"""
    candidates = [model]
    base = getattr(model, "__dict__", {}).get("base_model")
    if base is not None:
        candidates.append(base)
    for candidate in candidates:
        get_booster = getattr(candidate, "get_booster", None)
        if callable(get_booster):
            try:
                names = get_booster().feature_names
            except (TypeError, AttributeError):
                names = None
            if names:
                return list(names)
    return None


def _load_model_feature_columns(models_dir: Path, model_version: int) -> List[str]:
    """读取模型登记特征列（panel 模式 CLI 读取列契约来源：模型 metadata）。"""
    from src.lazybull.v2.core.signal.ml_signal import MLSignal

    probe = MLSignal(
        top_n=20,
        model_version=int(model_version),
        models_dir=str(models_dir),
        verbose=False,
    )
    probe._load_model()
    columns = list(probe.feature_columns or [])
    if not columns:
        raise ValueError(f"模型 v{model_version} 无登记特征列（feature_columns 为空）")
    return columns


def main() -> None:
    """standalone 打分入口（spike 与 G-code 共用；panel 源服务臂③ v2_panel / G-data 归因）。"""
    parser = argparse.ArgumentParser(description="P3 验收冻结矩阵预测指纹打分")
    parser.add_argument("--models-dir", required=True, help="模型注册表目录（隔离根内）")
    parser.add_argument("--model-version", type=int, required=True, help="模型版本号")
    parser.add_argument("--data-root", required=True, help="数据根（隔离运行根）")
    parser.add_argument("--test-start", required=True, help="测试窗起始 YYYYMMDD")
    parser.add_argument("--test-end", required=True, help="测试窗截止 YYYYMMDD")
    parser.add_argument("--stock-domain", default="main", help="选股域（默认 main）")
    parser.add_argument(
        "--source-mode",
        choices=list(SOURCE_MODES),
        default=SOURCE_MODE_CS_TRAIN_VERIFY,
        help="打分源（默认 cs_train_verify；panel = v2 panel 直读，读取列契约取模型登记特征列）",
    )
    parser.add_argument("--out", required=True, help="指纹结果 JSON 输出路径")
    parser.add_argument(
        "--persist-predictions",
        action="store_true",
        help="持久化集成主预测行 parquet（R2-T5A-06；必须同时显式给出 --predictions-path）",
    )
    parser.add_argument(
        "--predictions-path",
        default=None,
        help="预测行 parquet 产物路径（默认 None；--persist-predictions 开启时必填）",
    )
    args = parser.parse_args()
    if args.persist_predictions and args.predictions_path is None:
        parser.error("--persist-predictions 开启时必须显式给出 --predictions-path")

    data_root = Path(args.data_root)
    # 预期日来源独立（N3）：trade_cal 开市日 ∩ 测试窗，缺分区即失败
    from scripts.v2_p3.t05b_spike_runner import list_expected_oos_days, resolve_score_days

    expected_days = list_expected_oos_days(data_root, args.test_start, args.test_end)
    extra_kwargs: Dict[str, object] = {}
    if args.source_mode == SOURCE_MODE_PANEL:
        from src.lazybull.v2.store.data_store import PanelDataStore

        store = PanelDataStore(data_root)
        # panel 源打分日域 = panel 分区日域（禁止 glob cs_train）；读取列契约 = 模型登记特征列
        score_days = resolve_panel_score_days(expected_days, store, args.test_start, args.test_end)
        extra_kwargs = {
            "source_mode": SOURCE_MODE_PANEL,
            "store": store,
            "feature_columns": _load_model_feature_columns(
                Path(args.models_dir), args.model_version
            ),
        }
    else:
        score_days = resolve_score_days(expected_days, data_root / "features")
    main_board_codes = load_main_board_codes(data_root, stock_domain=args.stock_domain)
    predictions_path = Path(args.predictions_path) if args.predictions_path is not None else None
    result = score_model_fingerprint(
        models_dir=Path(args.models_dir),
        model_version=args.model_version,
        data_root=data_root,
        score_days=score_days,
        main_board_codes=main_board_codes,
        persist_predictions=args.persist_predictions,
        predictions_path=predictions_path,
        **extra_kwargs,
    )
    result["environment_identity"] = environment_identity()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"指纹结果已写出: {out_path}")


if __name__ == "__main__":
    main()
