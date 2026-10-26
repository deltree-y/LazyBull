# -*- coding: utf-8 -*-
"""P3 T5 打分侧窄口径 Stock Connect 闭市日识别（验收专用，不入生产链路）。

从 ``t05b_fingerprint`` 纯提取（复杂度门：文件/函数硬上限），行为逐字不变：
仅当 raw ``moneyflow_hsgt`` 的 ``hgt/sgt/north_money`` 三列全空且源侧
``north_net_buy`` 全空时登记为合法闭市排除日；文件缺失/有值/非全空一律不排除，
当日仍走原质量门禁（任何其他拒绝仍 FAIL）。

读取门与台账（R2-T5AE-R2-01/R3-01，2026-10-26）：``raw_read_guard`` 激活期间，本模块
闭市判定的每次实际读取采用「**读取一次 → 校验该字节 → 解析同一缓冲区**」——raw
``moneyflow_hsgt`` 全三门、源侧 cs_train 成员门 + 内容身份（仅闭市候选日触发）；
跑前缺失候选被读取 / 快照外实读候选 / 读时内容与跑前登记身份不符 ⇒ **立即拒绝**；
台账哈希 = 实际解析字节；补评分与主臂评分共用同一约束。
"""

from __future__ import annotations

import hashlib
import io
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd
from loguru import logger

#: 北向闭市日登记原因（窄口径；审批见规划 §19.18，2026-10-24 用户裁决）
STOCK_CONNECT_CLOSED_REASON = "stock_connect_closed"

#: raw ``moneyflow_hsgt`` 闭市判定三列（同时全空 ⇒ Stock Connect 闭市）
_RAW_MONEYFLOW_HSGT_COLUMNS = ("hgt", "sgt", "north_money")


@dataclass(frozen=True)
class RawReadRecord:
    """闭市判定实读记录（绝对 POSIX 路径 + 读时点内容身份；读取台账条目）。"""

    path: str
    sha256: str


#: 激活态读取门（None = 未激活，闭市判定行为逐位不变）
_READ_GUARD_STATE: Optional[Dict[str, object]] = None
#: 本次激活期的读取台账（enter 清零；收尾核验消费）
_RAW_READ_LEDGER: List[RawReadRecord] = []


@contextmanager
def raw_read_guard(
    missing_paths: Iterable[Path],
    registered_identities: Mapping[str, str],
):
    """评分期间闭市判定读取门 + 读取台账（补评分与主臂评分共用，R2-T5AE-R2-01）。

    激活期间 ``_is_stock_connect_closure_raw`` / ``_feature_north_all_null``
    每次实际读取均为「读取一次 → 校验该字节 → 解析同一缓冲区」
    （R2-T5AE-R3-01：校验/使用分离不复存在）：

    - raw ``moneyflow_hsgt``（全三门）：① 路径 ∈ ``missing_paths``（跑前缺失
      候选——「创建→读取→移除」不再可能绕过：读取动作自身被 gate）⇒ 立即拒绝；
      ② 路径 ∉ ``registered_identities``（快照外实读候选，无跑前内容身份）⇒
      立即拒绝；③ 实际读取字节 sha256 ≠ 登记身份（含读时替换）⇒ 立即拒绝；
    - 源侧 cs_train 分区（成员门 ①② + ③内容身份）：③ 只在 raw 判为闭市候选
      后触发（候选日少量分区），不扩展到普通评分日。

    ``registered_identities`` 键 = 绝对 POSIX 路径、值 = 跑前快照 sha256；
    通过门后登记 ``RawReadRecord``。yield 返回台账列表（enter 清零，收尾复核）。
    未激活 = 现行为逐位不变。收尾闭合核验须与本读取门**共同**执行
    （不能仅靠最后一次目录重枚举——reviewer2 复审二 §2）。
    """
    global _READ_GUARD_STATE
    if _READ_GUARD_STATE is not None:
        raise RuntimeError("raw_read_guard 不可嵌套激活")
    _READ_GUARD_STATE = {
        "missing": {Path(path).resolve().as_posix() for path in missing_paths},
        "identities": {str(key): str(value) for key, value in registered_identities.items()},
    }
    _RAW_READ_LEDGER.clear()
    try:
        yield _RAW_READ_LEDGER
    finally:
        _READ_GUARD_STATE = None


def raw_read_ledger_snapshot() -> List[RawReadRecord]:
    """读取台账快照（拷贝；收尾核验/verdict 登记消费，防后续激活清零失引用）。"""
    return list(_RAW_READ_LEDGER)


def _guard_raw_bytes(path: Path, payload: bytes) -> None:
    """raw moneyflow 读取门（三条全检，对**实际读取字节**校验；仅激活态生效）。

    R2-T5AE-R3-01：校验对象 = 本次解析将消费的同一字节缓冲——「预哈希→按路径
    重读解析」的校验/使用分离不复存在，台账哈希 = 实际解析字节。
    """
    state = _READ_GUARD_STATE
    if state is None:
        return
    posix = Path(path).resolve().as_posix()
    missing = state["missing"]  # type: ignore[index]
    if posix in missing:
        raise RuntimeError(f"读取门拒绝：跑前缺失候选被实际读取（无跑前内容身份）: {posix}")
    identities = state["identities"]  # type: ignore[index]
    if posix not in identities:
        raise RuntimeError(f"读取门拒绝：快照外实读候选: {posix}")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != identities[posix]:
        raise RuntimeError(f"读取门拒绝：读取内容与跑前登记身份不符: {posix}")
    _RAW_READ_LEDGER.append(RawReadRecord(path=posix, sha256=digest))


def _guard_member_bytes(path: Path, payload: bytes) -> None:
    """源侧 cs_train 成员读取门（①缺失 ②快照外 ③内容身份；仅激活态生效）。

    R2-T5AE-R3-01：③补齐——临时变更再恢复无法由收尾哈希捕获，实际读取字节
    须与跑前登记身份相符；该读取只在 raw 判为闭市候选后触发（候选日少量
    分区），不扩展到普通评分日。通过后登记读取台账（= 实际解析字节）。
    """
    state = _READ_GUARD_STATE
    if state is None:
        return
    posix = Path(path).resolve().as_posix()
    missing = state["missing"]  # type: ignore[index]
    if posix in missing:
        raise RuntimeError(f"读取门拒绝：跑前缺失候选被实际读取（无跑前内容身份）: {posix}")
    identities = state["identities"]  # type: ignore[index]
    if posix not in identities:
        raise RuntimeError(f"读取门拒绝：快照外实读候选: {posix}")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != identities[posix]:
        raise RuntimeError(f"读取门拒绝：读取内容与跑前登记身份不符: {posix}")
    _RAW_READ_LEDGER.append(RawReadRecord(path=posix, sha256=digest))


def _raw_moneyflow_hsgt_path(data_root: Path, day: str) -> Path:
    """raw 资金流分区路径（``YYYY-MM-DD.parquet``）。"""
    return Path(data_root) / "raw" / "moneyflow_hsgt" / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet"


def _is_stock_connect_closure_raw(path: Path) -> bool:
    """raw ``moneyflow_hsgt`` 三列全空 ⇒ Stock Connect 闭市。

    文件缺失 / 空 / 缺列 / 读取失败一律 False（不排除；当日仍由质量门禁正常
    拒绝），防止把数据缺口误判为合法闭市。激活态（R2-T5AE-R3-01）：
    **读取一次 → 校验该字节 → 解析同一缓冲区**——校验/使用分离（预哈希后按
    路径重读）不复存在，台账哈希 = 实际解析字节；未激活 = 现行为逐位不变。
    """
    if not path.is_file():
        return False
    if _READ_GUARD_STATE is not None:
        try:
            payload = Path(path).read_bytes()
        except OSError:
            return False  # is_file 后被移除：按原「文件缺失不排除」语义
        _guard_raw_bytes(path, payload)
        try:
            raw = pd.read_parquet(io.BytesIO(payload), columns=list(_RAW_MONEYFLOW_HSGT_COLUMNS))
        except (OSError, ValueError, KeyError):
            return False
    else:
        try:
            raw = pd.read_parquet(path, columns=list(_RAW_MONEYFLOW_HSGT_COLUMNS))
        except (OSError, ValueError, KeyError):
            return False
    if raw.empty:
        return False
    return bool(raw[list(_RAW_MONEYFLOW_HSGT_COLUMNS)].isna().all().all())


def _feature_north_all_null(
    data_root: Path, day: str, panel_frames: Optional[Mapping[str, pd.DataFrame]]
) -> bool:
    """源侧 ``north_net_buy`` 全空校验（禁止只凭 raw 闭市就排除）。

    激活态（R2-T5AE-R3-01）：读取一次 → 成员门（①缺失 ②快照外）+ **③内容
    身份** → 解析同一缓冲区；该读取只在 raw 判为闭市候选后触发（候选日少量
    分区，不扩展普通评分日）。缺列/读取失败仍按原语义不排除；未激活 =
    现行为逐位不变。
    """
    if panel_frames is not None:
        frame = panel_frames.get(day)
        if frame is None or "north_net_buy" not in frame.columns or frame.empty:
            return False
        return bool(frame["north_net_buy"].isna().all())
    path = Path(data_root) / "features" / "cs_train" / f"{day}.parquet"
    if not path.is_file():
        return False
    if _READ_GUARD_STATE is not None:
        try:
            payload = Path(path).read_bytes()
        except OSError:
            return False
        _guard_member_bytes(path, payload)
        try:
            series = pd.read_parquet(io.BytesIO(payload), columns=["north_net_buy"])[
                "north_net_buy"
            ]
        except (OSError, ValueError, KeyError):
            return False
    else:
        try:
            series = pd.read_parquet(path, columns=["north_net_buy"])["north_net_buy"]
        except (OSError, ValueError, KeyError):
            return False
    if series.empty:
        return False
    return bool(series.isna().all())


def resolve_stock_connect_closure_days(
    data_root: Path,
    score_days: Sequence[str],
    required_columns: Sequence[str],
    panel_frames: Optional[Mapping[str, pd.DataFrame]] = None,
) -> Dict[str, str]:
    """窄口径合法闭市日：raw 北向三列全空 ∧ 源侧 ``north_net_buy`` 全空。

    审批口径（规划 §19.18，2026-10-24 用户裁决）：仅排除 Stock Connect 闭市日；
    模型不消费 ``north_net_buy`` / raw 缺失 / raw 有值 / 源侧该列非全空 ⇒ 不排除，
    当日仍走原质量门禁（任何其他拒绝仍 FAIL）。返回 ``{day: reason}``。
    """
    if "north_net_buy" not in set(required_columns):
        return {}
    excluded: Dict[str, str] = {}
    for day in score_days:
        if not _is_stock_connect_closure_raw(_raw_moneyflow_hsgt_path(data_root, day)):
            continue
        if not _feature_north_all_null(data_root, day, panel_frames):
            continue
        excluded[day] = STOCK_CONNECT_CLOSED_REASON
    if excluded:
        logger.info(
            f"北向闭市日窄口径排除: {len(excluded)} 日 {sorted(excluded)}"
            "（Stock Connect 闭市；其余拒绝仍 FAIL）"
        )
    return excluded
