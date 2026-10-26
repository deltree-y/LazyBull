# -*- coding: utf-8 -*-
"""数据态身份收集组(v2 侧,P3-T3;orchestrator 文件行硬上限拆分,§17.8)。

职责:非 store 输入运行前枚举采集(cs_train_verify 分区 / trade_cal /
stock_basic / clean/daily / raw/suspend 回测窗)+ 消费清单合并去重 +
数据态附加段(水位三对象 / manifest 指纹 / 代码态摘要)。
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
from dateutil.relativedelta import relativedelta
from loguru import logger

from src.lazybull.v2.store.store_identity import InputIdentityDriftError

from .data_state import collect_consumed_file_identities
from .utils import WalkForwardSplit


def dedup_identity_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """合并身份清单按显示路径去重;同路径不同 sha256 ⇒ 漂移(InputIdentityDriftError)。"""
    merged: Dict[str, Dict[str, Any]] = {}
    for record in records:
        path = str(record["path"])
        existing = merged.get(path)
        if existing is not None:
            if (existing["sha256"], existing["size"]) != (record["sha256"], record["size"]):
                raise InputIdentityDriftError(
                    f"消费清单合并冲突(读取间漂移): {path}(sha256 不一致)"
                )
            continue
        merged[path] = dict(record)
    return [merged[path] for path in sorted(merged)]


def _enum_base_inputs(root: Path) -> List[Path]:
    """日历与基础表单文件:clean 优先 raw 兜底(与适配层 resolve_trading_dates 同口径)。"""
    files: List[Path] = []
    for name in ("trade_cal.parquet", "stock_basic.parquet"):
        for tier in ("clean", "raw"):
            candidate = root / tier / name
            if candidate.is_file():
                files.append(candidate)
                break
    return files


def _split_days_union(
    splits: List[WalkForwardSplit],
    trade_cal: pd.DataFrame,
    oos_backtest_months: int,
) -> set:
    """逐折训练窗 + 测试窗 + 回测窗交易日并集。"""
    days: set = set()
    for split in splits:
        bt_end_dt = datetime.strptime(split.test_start, "%Y%m%d") + relativedelta(
            months=oos_backtest_months
        )
        bt_end = bt_end_dt.strftime("%Y%m%d")
        for day in trade_cal[trade_cal["is_open"] == 1]["cal_date"].astype(str):
            if split.train_start <= day <= split.test_end or split.test_start <= day <= bt_end:
                days.add(day)
    return days


def _enum_partition_files(root: Path, days: set, cs_train_verify: bool) -> List[Path]:
    """cs_train_verify 分区 + clean/daily + raw/suspend 按日域枚举存在的分区文件。"""
    files: List[Path] = []
    if cs_train_verify:
        cs_dir = root / "features" / "cs_train"
        for day in sorted(days):
            candidate = cs_dir / f"{day}.parquet"
            if candidate.is_file():
                files.append(candidate)
    daily_dir = root / "clean" / "daily"
    suspend_dir = root / "raw" / "suspend"
    for day in sorted(days):
        dashed = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        candidate = daily_dir / f"{dashed}.parquet"
        if candidate.is_file():
            files.append(candidate)
        # raw/suspend 逐日分区(P3-T3 评审处置 codex-R01:内核停牌日历实际读取,
        # 卖出路径 sell_execution → suspend_calendar;枚举回测窗日域并集)
        for name in (f"{dashed}.parquet", f"{day}.parquet"):
            candidate = suspend_dir / name
            if candidate.is_file():
                files.append(candidate)
                break
    return files


def acquire_non_store_identities(
    data_root: str,
    source_mode: str,
    splits: List[WalkForwardSplit],
    trade_cal: pd.DataFrame,
    oos_backtest_months: int,
    cs_train_verify_mode: str,
) -> List[Dict[str, Any]]:
    """运行前枚举采集非 store 输入身份(§5-2:非 store 输入现算)。

    覆盖:trade_cal / stock_basic 单文件 + cs_train_verify 分区(逐折训练窗 +
    测试窗日域并集)+ clean/daily 与 raw/suspend 回测窗分区(逐折 bt 窗日域并集);
    panel/labels 分区由 store receipt 读前采集 + 读后核验闭环(§13.3.5 P2),
    运行中会话级累积(``store.consumed_files``)。

    **口径登记(复核处置 reviewer1-P3-5)**:非 store 枚举是**保守超集**
    (按日域并集枚举存在分区;raw/suspend 等按需读取的文件可能未实际进入
    某次读取),运行后统一哈希复核;它不是「逐文件实际读取」证明——实际读取
    证明由 store receipt(读前采集绑定读后核验)承载,两者口径分列。
    """
    root = Path(data_root)
    extra_files = _enum_base_inputs(root)
    days = _split_days_union(splits, trade_cal, oos_backtest_months)
    extra_files += _enum_partition_files(root, days, source_mode == cs_train_verify_mode)
    records = collect_consumed_file_identities([], extra_files, base_dir=root)
    logger.info(f"数据态(增强): 非 store 输入运行前枚举 {len(records)} 文件")
    return records


def collect_identity_extra_sections(data_root: str, store: Any) -> Dict[str, Any]:
    """数据态附加段(P3-T3 评审处置 reviewer1-P3-6):水位三对象分列(§4.1)/
    manifest 内容指纹 / 代码态内容摘要(§13.3.8 source_digest,复用
    ``store_identity.workspace_identity`` 稳定集合摘要)。"""
    from src.lazybull.v2.store.store_identity import workspace_identity
    from src.lazybull.v2.train.data_adapter import resolve_input_watermark

    sections: Dict[str, Any] = {}
    # 复核处置(reviewer1 复核 P2-3):watermarks 先初始化,各段独立 try——
    # 水位解析失败不再连坐丢 manifest_fingerprint/label_sealed_through,
    # 日志归因分别准确
    watermarks: Dict[str, Any] = {}
    try:
        watermarks["input_watermark"] = resolve_input_watermark(data_root)
    except Exception as exc:
        logger.warning(f"数据态附加段:水位解析失败(不影响训练结果): {exc}")
    if store is not None:
        try:
            snapshot = store.manifest.snapshot()
            labels = snapshot.get("labels", {})
            watermarks["label_sealed_through"] = {
                name: (labels.get(name) or {}).get("sealed_through") for name in sorted(labels)
            }
        except Exception as exc:
            logger.warning(f"数据态附加段:标签水位摘要失败(不影响训练结果): {exc}")
        try:
            sections["manifest_fingerprint"] = store.manifest.content_fingerprint()
        except Exception as exc:
            logger.warning(f"数据态附加段:manifest 指纹失败(不影响训练结果): {exc}")
    if watermarks:
        sections["watermarks"] = watermarks
    try:
        identity = workspace_identity(Path(__file__).resolve().parents[5])  # 项目根
        sections["code_state"] = {
            "git_commit": identity.get("git_commit"),
            "git_dirty": identity.get("git_dirty"),
            "source_digest": identity.get("source_digest"),
        }
    except Exception as exc:
        # 键恒存在、值降级 None(与 collect_data_state 的 git 降级口径一致;
        # 全量回归高负载下 git 偶发超时不应导致整段缺失——复核处置轮定位)
        logger.warning(f"数据态附加段:代码态摘要失败(键保留、值降级 None): {exc}")
        sections["code_state"] = {
            "git_commit": None,
            "git_dirty": None,
            "source_digest": None,
        }
    return sections
