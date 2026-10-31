# -*- coding: utf-8 -*-
"""T0.5b spike：旧臂禁下载幂等性实证探针（P4 规划 §2.1-3 / D4 / R14，阶段专用临时开发文件）。

方法：把 TuShare 客户端的全部 `get_*` / `query*` / `download*` / `fetch*` 方法
替换为「**先计数、再抛 `_NetworkCallError`**」的禁网桩（调用事件独立留痕，
不依赖异常向外传播——ensure 链路存在 `except Exception` fail-soft 分支
（如 `data/ensure.py` stock_st/margin_detail），吞掉异常后仍可能返回成功，
仅以异常是否抛出判触网会产生假 PASS），然后对**已存在且最新**的数据日期跑
旧链路 ensure 三级，门禁 = **实测调用计数为零**且返回成功。

结论适用范围（评审 P2-1 收窄）：「数据已发布且已落盘的常态日」成立；
moneyflow/stk_limit/stock_st/margin_detail 未发布无占位日期的重试触网
属真缺口语义，**协调者无法在发布前消除**（登记为已知残留，见报告 §5）。

装配与生产 runner 一致（评审 R2-02 处置）：``DataCleaner(verbose=False)``、
``FeatureBuilder(horizon=20, require_label=False)``（生产落点
`paper/runner/__init__.py:116-118`）；生产数据根写方法（Storage.save_*/write_*）
全部包裹为「触发即抛 + 计数」守卫——「只读」不再仅依赖短路前提。

注意：本探针只对「数据已存在」的日期运行；对缺口日期运行会触发真实下载
（属 M-P4-1 追平语义，不在本探针范围）。

复杂度豁免依据：CLAUDE.md §7.5（阶段专用验收工具按临时开发文件处理）。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class _NetworkCallError(RuntimeError):
    """禁网桩触发：任何 TuShare 网络方法被调用即抛出（抛出前已计数留痕）。"""


class _WriteGuardError(RuntimeError):
    """写守卫触发：探针运行期任何生产根写方法被调用即抛出。"""


_NETWORK_PREFIXES = ("get_", "query", "download", "fetch")
_WRITE_PREFIXES = ("save_", "write_")


class CallRecorder:
    """调用事件记录器（桩与守卫共用）。"""

    def __init__(self) -> None:
        self.events: List[Dict] = []

    def record(self, kind: str, owner: str, name: str, kwargs: Dict) -> None:
        self.events.append(
            {
                "kind": kind,
                "owner": owner,
                "method": name,
                "date_args": sorted(
                    str(v) for k, v in kwargs.items() if "date" in k.lower()
                ),
            }
        )


def wrap_client_no_network(client, recorder: CallRecorder) -> List[str]:
    """把 client 的全部网络方法替换为「计数 + 抛错」禁网桩；返回被包裹方法清单。"""
    wrapped: List[str] = []
    for name in dir(client):
        if name.startswith(_NETWORK_PREFIXES) and callable(getattr(client, name, None)):

            def _blocked(*args, _name=name, **kwargs):
                recorder.record("network", type(client).__name__, _name, kwargs)
                raise _NetworkCallError(f"禁网探针：{type(client).__name__}.{_name} 被调用")

            setattr(client, name, _blocked)
            wrapped.append(name)
    return sorted(wrapped)


def wrap_storage_no_write(storage, recorder: CallRecorder) -> List[str]:
    """把 Storage 的全部写方法替换为「计数 + 抛错」守卫；返回被包裹方法清单。"""
    wrapped: List[str] = []
    for name in dir(storage):
        if name.startswith(_WRITE_PREFIXES) and callable(getattr(storage, name, None)):

            def _guarded(*args, _name=name, **kwargs):
                recorder.record("write", type(storage).__name__, _name, kwargs)
                raise _WriteGuardError(f"写守卫：Storage.{_name} 被调用")

            setattr(storage, name, _guarded)
            wrapped.append(name)
    return sorted(wrapped)


def _run_step(checks: List[Dict], recorder: CallRecorder, step: str, date: str, fn) -> None:
    """执行单步并登记实测调用计数（网络/写入分列）；异常即失败并留痕。"""
    before = len(recorder.events)
    try:
        result = fn()
        success = bool(result[0]) if isinstance(result, tuple) else bool(result)
        entry: Dict = {"step": step, "date": date, "success": success}
        if isinstance(result, tuple):
            entry["missing_factors"] = list(result[1])
            entry["error_detail"] = result[2]
    except (_NetworkCallError, _WriteGuardError) as e:
        entry = {"step": step, "date": date, "success": False, "error_detail": str(e)}
    new_events = recorder.events[before:]
    entry["network_calls"] = sum(1 for e in new_events if e["kind"] == "network")
    entry["write_calls"] = sum(1 for e in new_events if e["kind"] == "write")
    entry["events"] = new_events
    checks.append(entry)


def run_idempotency_probe(
    data_root: Path,
    existing_clean_date: str = "20260731",
    existing_cs_infer_date: str = "20260702",
) -> Dict:
    """对已存在日期跑 ensure 三级，门禁 = 实测零网络调用 + 零写入 + 返回成功。

    Args:
        data_root: 生产数据根（全部写方法已上守卫；ensure 对已存在数据为 no-op）。
        existing_clean_date: clean/daily 已存在的日期（最新水位日）。
        existing_cs_infer_date: cs_infer 已存在的日期（最新水位日）。
    """
    from src.lazybull.data.cleaner import DataCleaner
    from src.lazybull.data.ensure import ensure_basic_data, ensure_clean_data_for_date
    from src.lazybull.data.loader import DataLoader
    from src.lazybull.data.storage import Storage
    from src.lazybull.data.tushare_client import TushareClient
    from src.lazybull.features.builder import FeatureBuilder
    from src.lazybull.features.ensure.entry import ensure_features_for_date

    recorder = CallRecorder()
    storage = Storage(str(data_root))
    write_guarded = wrap_storage_no_write(storage, recorder)
    loader = DataLoader(storage)
    # 生产 runner 装配口径（paper/runner/__init__.py:116-118）
    cleaner = DataCleaner(verbose=False)
    builder = FeatureBuilder(horizon=20, require_label=False)
    client = TushareClient()
    net_wrapped = wrap_client_no_network(client, recorder)

    checks: List[Dict] = []
    _run_step(
        checks,
        recorder,
        "ensure_basic_data",
        existing_clean_date,
        lambda: ensure_basic_data(client, storage, existing_clean_date),
    )
    _run_step(
        checks,
        recorder,
        "ensure_clean_data_for_date",
        existing_clean_date,
        lambda: ensure_clean_data_for_date(storage, loader, cleaner, client, existing_clean_date),
    )
    _run_step(
        checks,
        recorder,
        "ensure_features_for_date(cs_infer 一级短路)",
        existing_cs_infer_date,
        lambda: ensure_features_for_date(
            storage, loader, builder, cleaner, client, existing_cs_infer_date
        ),
    )

    passed = all(
        c["success"] and c["network_calls"] == 0 and c["write_calls"] == 0 for c in checks
    )
    return {
        "gate": "v2_p4_t05b_no_download_idempotency",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(data_root),
        "assembly": "DataCleaner(verbose=False) / FeatureBuilder(horizon=20, require_label=False)"
        "（生产 runner 口径）+ Storage 写方法守卫",
        "conclusion_scope": "数据已发布且已落盘的常态日；未发布无占位数据集的触网重试属真缺口残留",
        "wrapped_client_methods": net_wrapped,
        "guarded_write_methods": write_guarded,
        "checks": checks,
        "verdict": "PASS" if passed else "FAIL",
    }


def main(argv=None) -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    result = run_idempotency_probe(_REPO_ROOT / "data")
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "idempotency_probe.json").write_text(text, encoding="utf-8")
    print(f"禁下载幂等性探针 verdict = {result['verdict']}")
    for c in result["checks"]:
        print(
            f"  {c['step']}({c['date']}): success={c['success']}, "
            f"network_calls={c['network_calls']}, write_calls={c['write_calls']}"
        )
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
