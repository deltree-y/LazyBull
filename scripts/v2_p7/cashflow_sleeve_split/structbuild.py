# -*- coding: utf-8 -*-
"""结构构建：行业 PIT 注入、标签反事实与分块成本实测（审点一 + r0_003 复核整改）。

职责
====
1. **T 日行业注入**（r0_003 复核第 1 项）：生产通路默认行业快照
   （``data/raw/shenwan_industry.parquet``）是**当前**申万分类（无日期维度），
   用在历史日会把未来行业改写写入特征分组与 ``neu_y_ret_*`` 标签分组
   （实例：600783.SH 在 2019-04-01 属 SW2014 非银金融 490000/801790.SI，
   快照写 SW2021 机械设备 801890.SI）。``industry_pit_snapshot`` 从
   ``sw_industry_pit`` 区间命中构造与生产快照同 schema 的当日帧，
   经 ``_run_capture(..., shenwan_industry=...)`` 显式注入：资格、
   行业中性化与需重建的标签共用同一输入。
2. **标签反事实**：三遍结构构建（原值 / 值改写 / 缺失位置重排），比较
   股票键、特征列逐值与 F/R/B 来源掩码。``reshaped`` 模式让已知股票的
   未来标签发生「有→无」「无→有」（r0_003 复核：仅改值不能证明特征
   不依赖标签**缺失位置**）。
3. **键控制重算**（r0_003 复核第 3 项）：从已落盘的结构逐日文件直接读键，
   逐日相加断言，禁止用旧摘要字段冒充逐日数。
4. **分块成本实测**：墙钟、峰值 RSS（采样）、写盘计时/字节，供预算报价。

两条纪律
========
- 结构构建**只读**生产输入，输出只在内存/研究缓存，不调用 ``build_daily``；
- 反事实比较用精确相等（浮点逐值、列序、行键、来源掩码），不一致即失败。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from scripts.v2_p7.cashflow_sleeve_split import spec
from scripts.v2_p7.common import REPO_ROOT


class BuildError(RuntimeError):
    """结构构建/反事实失败（含标签被读取、输入缺件）。"""


#: 单日捕获的预热窗（与生产单日通路常量一致：D − 7 个自然月）
CAPTURE_WARMUP_MONTHS = 7


def capture_start(day: str, months: int = CAPTURE_WARMUP_MONTHS) -> str:
    """捕获起点 = D 往前 ``months`` 个自然月（生产单日通路同口径）。"""
    import pandas as pd

    start = pd.to_datetime(str(day), format="%Y%m%d") - pd.DateOffset(months=months)
    return start.strftime("%Y%m%d")


def window_start(day: str, days: int) -> str:
    """反事实用的短窗起点（成本控制；键/特征比较不依赖预热长度）。"""
    import pandas as pd

    start = pd.to_datetime(str(day), format="%Y%m%d") - pd.DateOffset(days=int(days))
    return start.strftime("%Y%m%d")


def _rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss) / 1e6
    except Exception:  # noqa: BLE001 - 缺 psutil 只降级为 0
        return 0.0


class _RssSampler:
    """后台线程采样进程 RSS 峰值（r0_003 复核：前后读数不能充当峰值）。"""

    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval = float(interval_seconds)
        self.peak_mb = 0.0
        self.samples = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        def _loop() -> None:
            while not self._stop.wait(self.interval):
                self.peak_mb = max(self.peak_mb, _rss_mb())
                self.samples += 1

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        return {
            "peak_rss_sampled_mb": round(self.peak_mb, 1),
            "samples": self.samples,
            "interval_seconds": self.interval,
            "method": "background_thread_sampling",
        }


@contextlib.contextmanager
def _labels_masked(mode: str = "perturbed") -> Iterator[None]:
    """标签反事实：``perturbed`` 改写标签值，``reshaped`` 重排缺失位置。

    ``perturbed``：全部 ``y_ret_*`` / ``neu_y_ret_*`` 列值改为 ``−3×+0.5``；
    缺失位置不变 → 只证明特征不依赖标签**值**。
    ``reshaped``：让标签发生「有→无」「无→有」（确定性重排）→ 证明特征
    与评分行键不依赖标签**缺失位置**（r0_003 复核要求）。
    """
    import src.lazybull.features.labels as labels_mod
    import src.lazybull.v2.store.labels_builder as labels_builder

    original_labels = labels_mod.compute_forward_returns
    original_append = labels_builder.append_labels_for_day

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        return _reshape_label_frame(original_labels(*args, **kwargs))

    def _wrapped_append(*args: Any, **kwargs: Any) -> Any:
        original_append(*args, **kwargs)
        frame = kwargs.get("df_day")
        if frame is None and len(args) >= 2:
            frame = args[1]
        if frame is not None:
            _reshape_label_frame(frame)
        return None

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise BuildError("结构通路要求标签可用：removed 模式下构建失败")

    try:
        if mode == "perturbed":
            labels_mod.compute_forward_returns = _wrap_perturb(  # type: ignore[assignment]
                original_labels
            )
            labels_builder.append_labels_for_day = _wrap_perturb(  # type: ignore[assignment]
                original_append
            )
        elif mode == "reshaped":
            labels_mod.compute_forward_returns = _wrapped  # type: ignore[assignment]
            labels_builder.append_labels_for_day = _wrapped_append  # type: ignore[assignment]
        elif mode == "removed":
            labels_mod.compute_forward_returns = _boom  # type: ignore[assignment]
            labels_builder.append_labels_for_day = _boom  # type: ignore[assignment]
        else:
            raise BuildError(f"未知标签反事实模式：{mode}")
        yield
    finally:
        labels_mod.compute_forward_returns = original_labels  # type: ignore[assignment]
        labels_builder.append_labels_for_day = original_append  # type: ignore[assignment]


def _wrap_perturb(original: Any) -> Any:
    """值改写包装：标签值 → ``−3×+0.5``（缺失位置不变）。"""

    def _perturb(frame: Any) -> Any:
        import pandas as pd

        if isinstance(frame, pd.DataFrame):
            for column in frame.columns:
                if str(column).startswith(("y_ret_", "neu_y_ret_")):
                    frame[column] = frame[column] * -3.0 + 0.5
        return frame

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        return _perturb(original(*args, **kwargs))

    return _wrapped


def _reshape_label_frame(frame: Any) -> Any:
    """确定性重排标签缺失位置：有→无（前 1/3 有限值置 NaN）、无→有（后 1/3 缺失填 −0.5）。

    按 ``ts_code`` 升序定位，保证可复现；返回重排统计（``attrs["label_reshape_changed"]``）。
    """
    import numpy as np
    import pandas as pd

    if not isinstance(frame, pd.DataFrame) or len(frame) == 0:
        return frame
    label_cols = [
        str(column) for column in frame.columns if str(column).startswith(("y_ret_", "neu_y_ret_"))
    ]
    if not label_cols:
        return frame
    out = (
        frame.sort_values("ts_code").reset_index(drop=True)
        if "ts_code" in frame.columns
        else frame.reset_index(drop=True)
    )
    k = max(len(out) // 3, 1)
    changed = {"to_nan": 0, "filled": 0, "k": k, "label_columns": label_cols}
    for column in label_cols:
        values = pd.to_numeric(out[column], errors="coerce")
        finite_idx = values.dropna().index[:k]
        missing_idx = values[values.isna()].index[-k:]
        values.loc[finite_idx] = np.nan
        values.loc[missing_idx] = -0.5
        out[column] = values
        changed["to_nan"] += len(finite_idx)
        changed["filled"] += len(missing_idx)
    out.attrs["label_reshape_changed"] = changed
    return out


def _builder(root: Path) -> Any:
    import sys

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.lazybull.data.loader import DataLoader
    from src.lazybull.data.storage import Storage
    from src.lazybull.v2.store.panel_builder import V2PanelBuilder

    return V2PanelBuilder(DataLoader(Storage(str(root / "data"))))


def industry_pit_snapshot(day: str, root: Path | None = None) -> Dict[str, Any]:
    """构造 ``day`` 当日的行业 PIT 快照帧（与生产快照同 schema，可显式注入）。

    映射：``l1_code``/``l2_code``/``ind_code`` 旧 6 位码 →
    ``sw_industry_code_table``（src=sw_version, level=L1/L2/L3）反查
    ``index_code``；名称直接取 PIT 的 ``l1_name``/``l2_name``/``l3_name``。
    反查失败的行保留 NaN 并登记（资格过滤按 ``industry_known`` 剔除）。
    """
    import pandas as pd

    base = Path(root) if root is not None else REPO_ROOT
    pit_path = base / spec.INDUSTRY_PIT_INJECTION["pit_relative"]
    table_path = base / spec.INDUSTRY_PIT_INJECTION["code_table_relative"]
    for path in (pit_path, table_path):
        if not path.exists():
            raise BuildError(f"行业 PIT 注入缺件：{path}")
    pit = pd.read_parquet(pit_path)
    table = pd.read_parquet(table_path)
    day = str(day)
    has_in = pit["in_date"].astype(str).str.fullmatch(r"\d{8}")
    mask = (
        has_in
        & (pit["in_date"].astype(str) <= day)
        & (pit["out_date"].fillna("99999999").astype(str) > day)
    )
    hit = pit[mask].copy()
    hit = hit.sort_values(["ts_code", "in_date", "code_alias"]).drop_duplicates(
        "ts_code", keep="last"
    )

    lut = table.set_index(["src", "level", "industry_code"])["index_code"]

    def _lookup(source: Any, level: str, code: Any) -> Any:
        if pd.isna(source) or pd.isna(code):
            return None
        try:
            value = lut.loc[(str(source), level, str(code))]
            return None if pd.isna(value) else str(value)
        except KeyError:
            return None

    frame = pd.DataFrame(
        {
            "ts_code": hit["ts_code"].astype(str).values,
            "sw_l1_code": [_lookup(s, "L1", c) for s, c in zip(hit["sw_version"], hit["l1_code"])],
            "sw_l1": hit["l1_name"].values,
            "sw_l2_code": [_lookup(s, "L2", c) for s, c in zip(hit["sw_version"], hit["l2_code"])],
            "sw_l2": hit["l2_name"].values,
            "sw_l3_code": [_lookup(s, "L3", c) for s, c in zip(hit["sw_version"], hit["ind_code"])],
            "sw_l3": hit["l3_name"].values,
            "in_date": hit["in_date"].astype(str).values,
        }
    )
    identity = _identity_probe(frame, day)
    stats = {
        "day": day,
        "pit_rows": int(len(pit)),
        "interval_hits": int(len(hit)),
        "codes": int(frame["ts_code"].nunique()),
        "l1_unresolved": int(frame["sw_l1_code"].isna().sum()),
        "l2_unresolved": int(frame["sw_l2_code"].isna().sum()),
        "l3_unresolved": int(frame["sw_l3_code"].isna().sum()),
        "l1_name_missing": int(frame["sw_l1"].isna().sum()),
        "schema": list(frame.columns),
        "identity_probe": identity,
    }
    return {"frame": frame, "stats": stats}


def _identity_probe(frame: Any, day: str) -> Dict[str, Any]:
    """注册身份探针：600783.SH 在 2019-04-01 应命中 490000（非银金融）。"""
    probe = dict(spec.INDUSTRY_PIT_INJECTION.get("identity_probe") or {})
    expected_code = str(probe.get("ts_code") or "")
    if not expected_code or str(day) != str(probe.get("day")):
        return {"checked": False, "reason": "非注册验证日"}
    row = frame[frame["ts_code"].astype(str) == expected_code]
    if len(row) == 0:
        return {
            "checked": True,
            "ts_code": expected_code,
            "match": False,
            "reason": "当日 PIT 命中为空（资格过滤将剔除）",
        }
    actual = row.iloc[0]
    expected_name = str(probe.get("expected_l1_name") or "")
    expected_index = str(probe.get("expected_l1_index") or "")
    return {
        "checked": True,
        "ts_code": expected_code,
        "expected_l1_code": str(probe.get("expected_l1_code") or ""),
        "expected_l1_name": expected_name,
        "actual_l1_code": str(actual.get("sw_l1_code") or ""),
        "actual_l1_name": str(actual.get("sw_l1") or ""),
        "actual_l3_code": str(actual.get("sw_l3_code") or ""),
        "match": bool(
            str(actual.get("sw_l1") or "") == expected_name
            and str(actual.get("sw_l1_code") or "") == expected_index
        ),
    }


def write_frames(frames: Sequence[Any], out_dir: Path | str) -> Dict[str, Any]:
    """逐日写 Parquet 并计时/计字节（写盘成本单列，不混入捕获墙钟）。"""
    started = time.perf_counter()
    bytes_written = 0
    written: List[str] = []
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    for date, frame in frames:
        path = target / f"{date}.parquet"
        frame.to_parquet(path)
        bytes_written += path.stat().st_size
        written.append(str(date))
    return {
        "files": len(written),
        "dates": written,
        "bytes": int(bytes_written),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "path": str(target),
    }


def capture(
    root: Path | None = None,
    start: str = "",
    end: str = "",
    keep_dates: Sequence[str] | None = None,
    label_mode: str = "normal",
    industry_frame: Any = None,
    industry_source: str = "production_snapshot_default",
    write_dir: Path | str | None = None,
    sample_rss: bool = False,
) -> Dict[str, Any]:
    """跑一次结构捕获（不写生产），返回帧与成本。

    ``industry_frame`` 非 None 时经 ``_run_capture(shenwan_industry=...)``
    显式注入当日 PIT 行业快照（资格、中性化与标签分组共用该输入）；
    ``write_dir`` 非 None 时逐日落盘研究缓存并单列写盘计时/字节；
    ``sample_rss`` 时后台线程采样峰值 RSS。
    """
    base = Path(root) if root is not None else REPO_ROOT
    builder = _builder(base)
    keep = set(keep_dates) if keep_dates else None
    sampler = _RssSampler()
    if sample_rss:
        sampler.start()
    started = time.perf_counter()
    try:
        if label_mode == "normal":
            frames = builder._run_capture(
                start,
                end,
                keep_dates=keep,
                structural=True,
                shenwan_industry=industry_frame,
            )
        else:
            with _labels_masked(label_mode):
                frames = builder._run_capture(
                    start,
                    end,
                    keep_dates=keep,
                    structural=True,
                    shenwan_industry=industry_frame,
                )
    finally:
        rss_sampled = sampler.stop() if sample_rss else None
    elapsed = time.perf_counter() - started
    if not frames:
        raise BuildError(f"结构捕获无输出：{start}~{end}")
    rows = sum(len(frame) for _, frame in frames)
    columns = sorted(frames[0][1].columns.tolist())
    estimate = sum(float(frame.memory_usage(deep=True).sum()) for _, frame in frames)
    write = write_frames(frames, write_dir) if write_dir is not None else None
    return {
        "days": len(frames),
        "dates": [str(date) for date, _ in frames],
        "rows": int(rows),
        "columns_n": len(columns),
        "columns": columns,
        "frames": frames,
        "elapsed_seconds": round(elapsed, 3),
        "peak_rss_mb": round(max(_rss_mb(), sampler.peak_mb), 1),
        "rss_sampled": rss_sampled,
        "output_bytes_estimate": int(estimate),
        "write": write,
        "industry_source": industry_source,
        "label_mode": label_mode,
    }


def _label_columns(frame: Any) -> List[str]:
    """捕获帧里的标签列（``y_ret_*`` / ``neu_y_ret_*`` / 冻结禁用列）。"""
    forbidden = set(spec.FORBIDDEN_MATRIX_COLUMNS)
    return [
        str(column)
        for column in frame.columns
        if str(column) in forbidden or str(column).startswith(("y_ret_", "neu_y_ret_"))
    ]


def _signature(result: Mapping[str, Any]) -> Dict[str, Any]:
    """逐帧键/列/值/来源掩码的稳定签名（**排除标签列**）。

    捕获帧本身带 ``y_ret_*``（它们归属 labels 列组，由 ``_materialize_day`` 分走），
    因此标签反事实必须在排除标签列后比较：否则签名差异只反映「标签列被改写」，
    无法回答「标签是否进入了特征」。
    """
    frames = result["frames"]
    digest = hashlib.sha256()
    keys: List[str] = []
    label_columns: List[str] = []
    feature_columns = 0
    for date, frame in frames:
        labels = _label_columns(frame)
        label_columns.extend(labels)
        features = frame.drop(columns=labels)
        feature_columns = len(features.columns)
        digest.update(str(date).encode("utf-8"))
        digest.update("|".join(sorted(map(str, features.columns))).encode("utf-8"))
        ordered = features.sort_values("ts_code") if "ts_code" in features.columns else features
        digest.update(ordered.to_csv(index=False, float_format="%.10g").encode("utf-8"))
        if "ts_code" in ordered.columns:
            keys.extend(ordered["ts_code"].astype(str).tolist())
    return {
        "digest": digest.hexdigest(),
        "rows": int(result["rows"]),
        "columns_n": int(feature_columns),
        "codes_n": len(set(keys)),
        "dates": list(result["dates"]),
        "label_columns_in_frame": sorted(set(label_columns)),
        "label_columns_compared": False,
    }


def finite_source_mask(result: Mapping[str, Any], columns: Sequence[str]) -> Dict[str, Any]:
    """F/R/B 关注列的**有限值**来源掩码签名（逐键；inf/NaN 均不算有值）。

    r0_003 复核：``.notna()`` 会把单个无穷值计入来源，掩码必须用有限值判定。
    """
    import numpy as np
    import pandas as pd

    frames = result["frames"]
    digest = hashlib.sha256()
    present: List[str] = []
    per_frame: Dict[str, Dict[str, int]] = {}
    for date, frame in frames:
        usable = [column for column in columns if column in frame.columns]
        if not usable:
            raise BuildError(f"来源掩码列在捕获帧缺失：{date}（{columns[:5]}…）")
        present = usable
        ordered = frame[["ts_code", *usable]].sort_values("ts_code")
        numeric = ordered[usable].apply(lambda series: pd.to_numeric(series, errors="coerce"))
        mask = np.isfinite(numeric.to_numpy(dtype="float64", na_value=np.nan))
        digest.update(str(date).encode("utf-8"))
        digest.update("|".join(usable).encode("utf-8"))
        digest.update(mask.tobytes())
        digest.update(ordered["ts_code"].astype(str).str.cat(sep=",").encode("utf-8"))
        per_frame[str(date)] = {
            "rows": int(len(ordered)),
            "finite_rows": int(mask.any(axis=1).sum()),
        }
    return {
        "digest": digest.hexdigest(),
        "columns": present,
        "columns_n": len(present),
        "per_frame": per_frame,
        "rule": "有限值掩码（np.isfinite）；NaN/inf/非数值均不算来源",
    }


def _known_label_targets(result: Mapping[str, Any], column: str = "y_ret_20") -> Dict[str, Any]:
    """登记「已知股票」：标签有限值的第一只（有→无）与缺失的第一只（无→有）。"""
    import pandas as pd

    for _, frame in result["frames"]:
        if column not in frame.columns or "ts_code" not in frame.columns:
            continue
        ordered = frame[["ts_code", column]].sort_values("ts_code")
        values = pd.to_numeric(ordered[column], errors="coerce")
        finite = ordered.loc[values.notna(), "ts_code"].astype(str).tolist()
        missing = ordered.loc[values.isna(), "ts_code"].astype(str).tolist()
        return {
            "column": column,
            "has_to_none_first": finite[0] if finite else None,
            "has_to_none_candidates": finite[:3],
            "none_to_has_first": missing[0] if missing else None,
            "none_to_has_candidates": missing[:3],
        }
    return {"column": column, "reason": "捕获帧无标签列"}


def counterfactual(
    root: Path | None = None,
    days: Sequence[str] = (),
    capture_days: int = 90,
    routes: Mapping[int, Mapping[str, Any]] | None = None,
    version: int | None = None,
    industry_frame: Any = None,
    industry_source: str = "production_snapshot_default",
    write_dir: Path | str | None = None,
    check_removal: bool = False,
    prebuilt_normal: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """三遍结构构建：原值 vs 值改写 vs 缺失位置重排，特征侧必须逐值一致。

    比较三层：① 股票键与特征列逐值（签名）；② F/R/B 关注列**有限值**
    来源掩码（逐键）；③ 标签列不参与比较（单独落盘语义）。

    ``prebuilt_normal``：调用方已用**同参数**跑过的 normal 遍结果（如单日验证
    先行捕获），复用以免重复构建；结果体即 ``capture()`` 的返回值。
    """
    base = Path(root) if root is not None else REPO_ROOT
    if not days:
        raise BuildError("反事实需要至少一个日期")
    days = sorted(str(day) for day in days)
    start = window_start(days[0], capture_days)
    common = dict(industry_frame=industry_frame, industry_source=industry_source)
    if prebuilt_normal is not None:
        normal = dict(prebuilt_normal)
        if list(normal.get("dates") or []) != days:
            raise BuildError(f"prebuilt_normal 日期不符：{normal.get('dates')} != {days}")
    else:
        normal = capture(base, start, days[-1], keep_dates=days, label_mode="normal", **common)
    perturbed = capture(base, start, days[-1], keep_dates=days, label_mode="perturbed", **common)
    reshaped = capture(base, start, days[-1], keep_dates=days, label_mode="reshaped", **common)
    left = _signature(normal)
    value_only = _signature(perturbed)
    position = _signature(reshaped)
    mask_columns = _route_columns(routes, version)
    mask_normal = finite_source_mask(normal, mask_columns)
    mask_value = finite_source_mask(perturbed, mask_columns)
    mask_position = finite_source_mask(reshaped, mask_columns)

    removal: Dict[str, Any] = {
        "checked": bool(check_removal),
        "status": "not_run",
        "reason": "结构通路会调用标签函数；删除标签不是受支持状态（见 findings）",
    }
    if check_removal:
        try:
            capture(base, start, days[-1], keep_dates=days, label_mode="removed", **common)
            removal = {"checked": True, "status": "supported"}
        except BuildError as error:
            removal = {"checked": True, "status": "not_supported", "reason": str(error)}

    reshape_targets = _known_label_targets(normal)
    identical = left == value_only == position
    if mask_columns:
        mask_section: Dict[str, Any] = {
            "columns_source": "路由列（F+R+B 保序去重 ∩ 捕获帧列）",
            "route_version": version,
            "normal": mask_normal,
            "value_perturbed": mask_value,
            "position_reshaped": mask_position,
            "identical": mask_normal == mask_value == mask_position,
        }
    else:
        mask_section = {"status": "not_run", "reason": "未提供当折路由"}
    return {
        "status": "ok" if identical else "leak",
        "days": days,
        "capture_window": [start, days[-1]],
        "capture_days_nominal": int(capture_days),
        "modes": {
            "normal": "原值",
            "perturbed": "值改写（y_ret_*/neu_y_ret_* → −3×+0.5，缺失位置不变）",
            "reshaped": "缺失位置重排（有→无：前 1/3 有限值置 NaN；无→有：后 1/3 缺失填 −0.5）",
        },
        "signature_normal": left,
        "signature_label_value_perturbed": value_only,
        "signature_label_position_reshaped": position,
        "identical_value_only": left == value_only,
        "identical_position": left == position,
        "identical": identical,
        "source_mask": mask_section,
        "known_label_targets": reshape_targets,
        "label_removal": removal,
        "elapsed_seconds": {
            "normal": normal["elapsed_seconds"],
            "label_value_perturbed": perturbed["elapsed_seconds"],
            "label_position_reshaped": reshaped["elapsed_seconds"],
        },
        "peak_rss_mb": max(
            normal["peak_rss_mb"], perturbed["peak_rss_mb"], reshaped["peak_rss_mb"]
        ),
        "rows": normal["rows"],
        "columns_n": normal["columns_n"],
        "industry_source": industry_source,
        "write": normal.get("write"),
        "note": (
            "三遍都走 _run_capture(structural=True)；签名排除标签列（labels 列组单独落盘），"
            "只比较股票键、特征列、数值与有限值来源掩码，回答「标签值/缺失位置是否进入特征」"
        ),
    }


def _route_columns(
    routes: Mapping[int, Mapping[str, Any]] | None, version: int | None
) -> List[str]:
    """从当折路由取 F+R+B 列（保序去重）；无路由/版本时返回空列表。"""
    if routes is None or version is None:
        return []
    route = routes.get(int(version))
    if route is None:
        raise BuildError(f"来源掩码缺路由：v{version}")
    merged: List[str] = []
    for arm in ("F", "R", "B"):
        for column in route.get(arm) or []:
            if column not in merged:
                merged.append(column)
    # 路由列可能含捕获帧没有的列（如运行时派生族）；掩码函数只收帧内存在的列
    return merged


def measurement_content_digest(payload: Mapping[str, Any]) -> str:
    """测量附件的自校验摘要（去掉 ``content_digest`` 后的规范化 JSON SHA）。"""
    body = {key: value for key, value in payload.items() if key != "content_digest"}
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def verify_measurement(payload: Mapping[str, Any]) -> bool:
    """读入测量附件时重算自校验摘要（r0_003 复核：父 run 消费前校验）。"""
    recorded = payload.get("content_digest")
    if not recorded:
        return False
    return str(recorded) == measurement_content_digest(payload)


def key_control_from_files(
    day_dir: Path | str,
    root: Path | None = None,
    day_list: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """从已落盘的结构逐日文件直接读键，与旧训练域逐日对照（r0_003 复核第 3 项）。

    键摘要口径（与复核复算一致）：按当日唯一代码升序、每行一个代码、保留
    末尾换行计算 SHA256；逐日 ``only_structural`` 相加必须等于全块总数。
    """
    import pandas as pd

    base = Path(root) if root is not None else REPO_ROOT
    directory = Path(day_dir)
    if not directory.is_dir():
        raise BuildError(f"结构逐日文件缺目录：{directory}")
    files = sorted(directory.glob("*.parquet"))
    if day_list is not None:
        wanted = {str(day) for day in day_list}
        files = [path for path in files if path.stem in wanted]
    if not files:
        raise BuildError(f"结构逐日文件为空：{directory}")

    cs_train_dir = base / "data/features/cs_train"
    rows: List[Dict[str, Any]] = []
    total_only_structural = 0
    total_only_cs_train = 0
    for path in files:
        day = path.stem
        codes = sorted(pd.read_parquet(path, columns=["ts_code"])["ts_code"].astype(str).unique())
        cs_path = cs_train_dir / f"{day}.parquet"
        if not cs_path.exists():
            raise BuildError(f"旧训练域缺当日分区：{cs_path}")
        cs_codes = sorted(
            pd.read_parquet(cs_path, columns=["ts_code"])["ts_code"].astype(str).unique()
        )
        structural_set, cs_set = set(codes), set(cs_codes)
        only_structural = sorted(structural_set - cs_set)
        only_cs_train = sorted(cs_set - structural_set)
        total_only_structural += len(only_structural)
        total_only_cs_train += len(only_cs_train)
        rows.append(
            {
                "date": day,
                "structural_n": len(codes),
                "cs_train_n": len(cs_codes),
                "intersection_n": len(structural_set & cs_set),
                "only_structural_n": len(only_structural),
                "only_cs_train_n": len(only_cs_train),
                "structural_parquet_bytes": int(path.stat().st_size),
                "structural_keys_sha256": _keys_sha(codes),
                "cs_train_keys_sha256": _keys_sha(cs_codes),
            }
        )
    return {
        "days_sample": rows,
        "totals": {
            "only_structural": total_only_structural,
            "only_cs_train": total_only_cs_train,
            "days": len(rows),
        },
        "keys_sha_rule": "按当日唯一代码升序、每行一个代码、保留末尾换行的 SHA256",
    }


def _keys_sha(codes: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(codes) + "\n").encode("utf-8")).hexdigest()


def trading_day_split(root: Path | None, start: str, end: str) -> Dict[str, Any]:
    """捕获窗交易日拆分（预热 / 输出），修正 r0_003 的预热天数（145/57）。"""
    from scripts.v2_p7.cashflow_sleeve_split.fold_clock import load_trade_days

    base = Path(root) if root is not None else REPO_ROOT
    days = load_trade_days(base, start, end)
    return {
        "window": [start, end],
        "trading_days": len(days),
        "first": days[0] if days else "",
        "last": days[-1] if days else "",
    }


def rebuild_block_measurement(
    measured_path: Path | str,
    day_dir: Path | str,
    root: Path | None = None,
) -> Dict[str, Any]:
    """用已落盘 57 日文件修正 r0_003 的分块测量附件（逐日键数 + 预热天数）。

    修正项（r0_003 复核第 3 项）：
    - 逐日键数直接来自产物；``only_structural`` 相加必须等于全块总数；
    - 捕获窗交易日拆分按交易日历重算（145 预热 + 57 输出，原附件写 148）；
    - 墙钟/字节保留原测量并注明「写盘在停表后发生」；RSS 前后读数不能充当峰值。
    """
    base = Path(root) if root is not None else REPO_ROOT
    path = Path(measured_path)
    if not path.exists():
        raise BuildError(f"原测量附件缺件：{path}")
    previous = json.loads(path.read_text(encoding="utf-8"))
    key_control = key_control_from_files(day_dir, base, day_list=previous.get("kept_dates"))
    block = previous.get("block") or []
    if len(block) != 2:
        raise BuildError(f"原测量附件缺 block：{block}")
    # r0_003 复核第 3 项：预热天数按**捕获窗**（含预热）重算 = 202 交易日（145 预热 +
    # 57 输出），不得用输出块区间（57 日）当捕获窗
    capture_window = previous.get("capture_window") or block
    window = trading_day_split(base, capture_window[0], capture_window[1])
    computed_days = int(previous.get("computed_days") or 0)
    kept = int(previous.get("kept_written_days") or 0)
    warmup = window["trading_days"] - kept
    wall = float(previous.get("wall_seconds") or 0.0)
    corrected: Dict[str, Any] = {
        "schema": "p7_cashflow_split_structural_block/v3",
        "batch": previous.get("batch"),
        "block": block,
        "capture_window": previous.get("capture_window"),
        "computed_days": computed_days,
        "kept_written_days": kept,
        "warmup_trading_days": warmup,
        "window_trading_days": window["trading_days"],
        "window_split_check": {
            "expected": f"{warmup} 预热 + {kept} 输出 = {window['trading_days']} 交易日",
            "consistent": warmup + kept == window["trading_days"],
            "source": "clean/trade_cal（fold_clock.load_trade_days）",
        },
        "wall_seconds": wall,
        "wall_note": "计到捕获结束；写 Parquet 在停表后发生（写盘成本单列）",
        "seconds_per_computed_day": round(wall / max(window["trading_days"], 1), 4),
        "seconds_per_kept_day": round(wall / max(kept, 1), 4),
        "rss_before_mb": previous.get("rss_before_mb"),
        "rss_after_mb": previous.get("rss_after_mb"),
        "rss_is_true_peak": False,
        "rss_note": "捕获前后读数取大，非采样峰值；单日验证（20190401）另测采样峰值",
        "disk_bytes": previous.get("disk_bytes"),
        "rows": previous.get("rows"),
        "columns_n": previous.get("columns_n"),
        "key_control": {
            "days_sample": key_control["days_sample"],
            "totals": key_control["totals"],
            # 兼容键：与旧 v2 附件字段同名，保证本函数可重入（v3 → v3 重算）
            "total_only_structural": key_control["totals"]["only_structural"],
            "keys_sha_rule": key_control["keys_sha_rule"],
            "source": "temp 结构逐日文件逐日直读（key_control_from_files）",
            "previous_summary_only_structural": (
                (previous.get("key_control") or {}).get("total_only_structural")
                or ((previous.get("key_control") or {}).get("totals") or {}).get("only_structural")
            ),
            "sum_matches_total": key_control["totals"]["only_structural"]
            == int(
                (
                    (previous.get("key_control") or {}).get("total_only_structural")
                    or ((previous.get("key_control") or {}).get("totals") or {}).get(
                        "only_structural"
                    )
                    or -1
                )
            ),
        },
        "previous_provenance": previous.get("provenance"),
        "corrected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corrected_by": "structbuild.rebuild_block_measurement（随 run 封存）",
    }
    corrected["content_digest"] = measurement_content_digest(corrected)
    return corrected


def measure_block(
    root: Path | None = None,
    start: str = "",
    end: str = "",
    keep_dates: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """固定分块成本实测（含预热），供整窗外推。"""
    base = Path(root) if root is not None else REPO_ROOT
    result = capture(base, start, end, keep_dates=keep_dates, label_mode="perturbed")
    days = int(result["days"])
    per_day = result["elapsed_seconds"] / max(days, 1)
    return {
        "status": "measured",
        "block": [start, end],
        "days": days,
        "keep_dates": list(keep_dates or []),
        "rows": result["rows"],
        "columns_n": result["columns_n"],
        "elapsed_seconds": result["elapsed_seconds"],
        "seconds_per_day": round(per_day, 4),
        "peak_rss_mb": result["peak_rss_mb"],
        "output_bytes_estimate": result["output_bytes_estimate"],
        "serial_full_window_extrapolation": {
            "days_needed": int(spec.WINDOW["feature_warmup_trading_days"])
            + int(spec.WINDOW["return_intervals"]),
            "basis": "本分块实测 seconds_per_day × （预热 120 + 账户窗 1838）",
        },
    }


__all__ = [
    "BuildError",
    "CAPTURE_WARMUP_MONTHS",
    "_RssSampler",
    "capture",
    "capture_start",
    "counterfactual",
    "finite_source_mask",
    "industry_pit_snapshot",
    "key_control_from_files",
    "measure_block",
    "measurement_content_digest",
    "rebuild_block_measurement",
    "trading_day_split",
    "verify_measurement",
    "window_start",
    "write_frames",
]
