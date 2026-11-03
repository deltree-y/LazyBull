# -*- coding: utf-8 -*-
"""P5b 试车场产物落盘与记录封存（T1 共用件）。

评审处置落点（R2-T1-03/04/05/09/20 + 复审 R2R-05/06）：

- 注册记录以 ``O_CREAT|O_EXCL`` **独占创建**（并发重复注册由内核拒绝，不靠先查后写）；
- 普通记录写临时文件 + ``os.replace`` 原子替换（临时名带 pid，失败清理临时文件）；
- 记录写入 ``record_sha256`` 封存（对除自身外的全字段 canonical 哈希），读取即复算；
- 建根失败回滚不留半初始化现场；回滚失败显式上报残留路径，禁止静默吞掉；
- 清理失败不得覆盖原始异常（清理异常只记 warning）。
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, NoReturn

from loguru import logger

from src.lazybull.v2.hosts.testbed.paths import TestbedError
from src.lazybull.v2.store.store_identity import canonical_json_sha256

__all__ = [
    "SEAL_FIELD",
    "raise_rollback_failure",
    "read_json",
    "seal_record",
    "verify_seal",
    "write_json_atomic",
    "write_json_exclusive",
]

#: 记录封存哈希字段名（不参与自身计算）
SEAL_FIELD = "record_sha256"


def _dump(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _safe_unlink(path: Path) -> None:
    """尽力删除残留文件；失败只告警，禁止覆盖调用方的原始异常。"""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning(f"试车场临时文件清理失败（需人工清理）: {path}（{exc}）")


def read_json(path: Path, *, what: str) -> Dict[str, Any]:
    """读取 JSON 对象；缺失/损坏/非对象一律抛带上下文的 ``TestbedError``。"""
    source = Path(path)
    if not source.is_file():
        raise TestbedError(f"{what}缺失: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TestbedError(f"{what}读取失败（损坏/非法 JSON）: {source}（{exc}）") from exc
    if not isinstance(payload, dict):
        raise TestbedError(f"{what}顶层必须是对象: {source}（实际 {type(payload).__name__}）")
    return payload


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """原子写 JSON（临时名带 pid；写失败清理临时文件，不留半写现场）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(_dump(payload), encoding="utf-8")
        os.replace(tmp, target)
    except OSError:
        _safe_unlink(tmp)
        raise


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """独占创建 JSON：目标已存在即 ``FileExistsError``（并发重复注册的唯一裁决点）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(target), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        handle = os.fdopen(descriptor, "w", encoding="utf-8")
    except Exception:
        # fdopen 失败时句柄未被接管，必须显式关闭后再清理，且不得覆盖原始异常
        try:
            os.close(descriptor)
        except OSError as close_error:
            logger.warning(f"试车场独占句柄关闭失败: {target}（{close_error}）")
        _safe_unlink(target)
        raise
    try:
        with handle:
            # 序列化失败（非 OSError，如循环引用）也必须清掉独占路径，否则重试永久冲突
            handle.write(_dump(payload))
    except Exception:
        _safe_unlink(target)
        raise


def seal_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """给记录加 ``record_sha256`` 封存（对除自身外全字段 canonical 哈希）。"""
    sealed = dict(record)
    sealed.pop(SEAL_FIELD, None)
    sealed[SEAL_FIELD] = canonical_json_sha256(sealed)
    return sealed


def verify_seal(record: Mapping[str, Any], *, what: str) -> None:
    """复算封存哈希；缺失或不一致即说明记录被改写/伪造。"""
    recorded = record.get(SEAL_FIELD)
    if not recorded:
        raise TestbedError(f"{what}缺封存哈希 {SEAL_FIELD}（禁止使用未封存记录）")
    payload = {key: value for key, value in record.items() if key != SEAL_FIELD}
    actual = canonical_json_sha256(payload)
    if actual != recorded:
        raise TestbedError(f"{what}封存哈希不一致（记录已被改写）: 记录={recorded} 实际={actual}")


def raise_rollback_failure(
    paths: Iterable[Path],
    original: BaseException,
    *,
    empty_dirs: Iterable[Path] = (),
) -> NoReturn:
    """回滚本次已建目录；失败则显式上报残留现场（禁止静默吞异常）。

    ``paths`` 只放**本次创建的叶子目录**（禁止放可能被并发方使用的父目录）；
    父目录回收走 ``empty_dirs``：只在删空后仍为空时才 ``rmdir``，非空即保持原样。
    """
    leftovers: list[str] = []
    for path in paths:
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            continue
        except OSError as exc:
            leftovers.append(f"{path}（{exc}）")
    for directory in empty_dirs:
        try:
            directory.rmdir()
        except FileNotFoundError:
            continue
        except OSError:
            # 非空（并发方已建内容）或占用：保持原样，不算回滚失败
            continue
    if leftovers:
        raise TestbedError(
            f"回滚失败，失败现场保留（需人工核对后清理）: {leftovers}；原始错误: {original}"
        ) from original
    raise original
