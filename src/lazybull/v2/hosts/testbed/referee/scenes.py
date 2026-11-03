# -*- coding: utf-8 -*-
"""P5b T4 收益裁判的训练/holdout 场景（设计 v4 §5.3 四类 + §5.5 冻结参数）。

四类场景（固定 seed/实现编号/参数，禁止按 holdout 结果回调）::

    ma20            d_t = sum_{j=0..H-1} eps_{t+j}（前向窗口；与 eps_{t-j} 同分布），
                    eps ~ N(0, (sd/sqrt(H))^2) ⇒ 持有期重叠
    ar1_06          AR(1) rho=0.6，平稳标准差 sd（强正相关，依赖长度远小于 H）
    regime_hetero   确定性两段异方差（块长 100，倍数 0.5/1.5）
    real_seed_block B0/B0' 真实换种子差的 circular block 重采样（block_length=H）

"超 H 依赖"与"fresh 前段样本不足的冷启动"两类**未单列场景**（T0.5b 冻结的场景集合不含它们，
后补场景会改 ``c_k`` 的 max-over-scene 口径）：前者由 ``real_seed_block`` 块内保留的原始依赖
间接承担，后者的短样本行为由 :func:`statistics.score_paths` 的"每 look ≥6 有效 block"
fail-closed 与 10% 无效率门承担。该缺口在三处登记：manifest ``deviation_registry``（必登记项）、
``limitations``（场景集边界条）、报告 markdown §5"已知差异登记"表。

block 重采样硬口径（§5.5）：``circular``（索引按池长取模环回）、**按源序列自身均值去中心**、
``block_length = H``、**不按较短长度截断**、训练/验证池禁止交叉采样。

真实源：同一对 ``source_run_id``（B0 = base、B0' = noiseB0），训练/验证以**互斥源折**
（0..9 / 10..13）隔离；两组源折均已观察，因此这是「已观察源条件下的新 Monte Carlo」，
不是独立真实源对验证（报告必须如实声明）。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.lazybull.v2.hosts.testbed.paths import TestbedError

__all__ = [
    "DgpParams",
    "SCENE_NAMES",
    "SCENE_REQUIRED",
    "SceneError",
    "SceneRegistry",
    "SourcePair",
    "block_date_blocks",
    "build_scene_registry",
    "circular_block_resample",
    "gen_ar1",
    "gen_ma",
    "gen_real_block",
    "gen_regime",
    "load_source_pair",
    "paired_log_diff",
    "pool_for_folds",
    "scene_parameters",
]

#: 四类依赖场景（固定集合，禁止增删——删除困难场景属违反设计 §5.3）
SCENE_NAMES: Tuple[str, ...] = ("ma20", "ar1_06", "regime_hetero", "real_seed_block")
#: 注册场景（= 全部四类；manifest 必须与之一致）
SCENE_REQUIRED: Tuple[str, ...] = SCENE_NAMES
#: AR(1) 预热长度（与 T0.5b harness 同口径；不是截断，是丢弃未收敛前段）
AR_WARMUP_DAYS = 100


class SceneError(TestbedError):
    """场景构造非法（参数不符冻结值、池过短、源哈希不符）。"""

    __test__ = False


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class DgpParams:
    """冻结 DGP 参数（取值唯一来源 = 校准 manifest）。"""

    sd_daily: float = 0.0037
    path_days: int = 500
    ar_rho: float = 0.6
    regime_block_length: int = 100
    regime_multipliers: Tuple[float, float] = (0.5, 1.5)
    block_length: int = 20

    def __post_init__(self) -> None:
        if self.path_days <= 0 or self.block_length <= 0:
            raise SceneError(
                f"DGP 路径长度/块长必须为正: path_days={self.path_days} block_length={self.block_length}"
            )
        if len(self.regime_multipliers) != 2:
            raise SceneError(f"regime 倍数必须是两个: {self.regime_multipliers!r}")

    @classmethod
    def from_manifest(cls, doc: Mapping[str, Any], *, h: int) -> "DgpParams":
        """由 manifest ``calibration.dgp`` + ``block_resample`` 构造并核验一致性。"""
        dgp = dict((doc.get("calibration") or {}).get("dgp") or {})
        block = dict((doc.get("calibration") or {}).get("block_resample") or {})
        params = cls(
            sd_daily=float(dgp["sd_daily"]),
            path_days=int(dgp["path_days"]),
            ar_rho=float(dgp["ar_rho"]),
            regime_block_length=int(dgp["regime_block_length"]),
            regime_multipliers=tuple(float(x) for x in dgp["regime_multipliers"]),
            block_length=int(block.get("block_length", dgp.get("block_length"))),
        )
        dgp_block = int(dgp.get("block_length", h))
        if dgp_block != params.block_length:
            raise SceneError(
                f"dgp.block_length={dgp_block} 与 "
                f"block_resample.block_length={params.block_length} 不一致"
            )
        if params.block_length != int(h):
            raise SceneError(
                f"block_length={params.block_length} 必须等于 H={h}（设计 §5.5 冻结口径）"
            )
        # T0.5b manifest 写作 "circular block"，T4 冻结件写作 "circular"：按前缀归一后校验
        method = str(block.get("method") or "").strip().lower()
        if not method.startswith("circular"):
            raise SceneError(f"block 重采样方法必须是 circular，实际 {block.get('method')!r}")
        if block.get("cross_pool_sampling_forbidden") is not True:
            raise SceneError("block_resample.cross_pool_sampling_forbidden 必须为 true")
        return params

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sd_daily": float(self.sd_daily),
            "path_days": int(self.path_days),
            "ar_rho": float(self.ar_rho),
            "regime_block_length": int(self.regime_block_length),
            "regime_multipliers": [float(x) for x in self.regime_multipliers],
            "block_length": int(self.block_length),
        }


def gen_ma(rng: np.random.Generator, n_paths: int, params: DgpParams) -> np.ndarray:
    """MA(H-1)：``d_t = sum_{j=0..H-1} eps_{t+j}``（**前向窗口和**），``eps ~ N(0, sd^2/H)``。

    前向与后向记法（``eps_{t-j}``）对 iid 高斯创新**分布相同**：自协方差
    ``gamma(k) = (H-|k|) * sd^2/H`` 关于 lag 对称，故有限维分布一致；此处用前向仅为向量化实现。
    """
    h = params.block_length
    sigma = params.sd_daily / math.sqrt(h)
    eps = rng.normal(0.0, sigma, size=(int(n_paths), params.path_days + h - 1))
    cumulative = np.pad(np.cumsum(eps, axis=1), ((0, 0), (1, 0)))
    return cumulative[:, h:] - cumulative[:, :-h]


def gen_ar1(rng: np.random.Generator, n_paths: int, params: DgpParams) -> np.ndarray:
    """AR(1) ``rho=0.6``：平稳标准差 = ``sd_daily``；丢弃前 :data:`AR_WARMUP_DAYS` 日。"""
    rho = params.ar_rho
    sigma = params.sd_daily * math.sqrt(1.0 - rho * rho)
    eps = rng.normal(0.0, sigma, size=(int(n_paths), params.path_days + AR_WARMUP_DAYS))
    series = np.empty_like(eps)
    series[:, 0] = eps[:, 0]
    for t in range(1, series.shape[1]):
        series[:, t] = rho * series[:, t - 1] + eps[:, t]
    return series[:, AR_WARMUP_DAYS:]


def gen_regime(rng: np.random.Generator, n_paths: int, params: DgpParams) -> np.ndarray:
    """确定性两段异方差：按 ``regime_block_length`` 交替乘 0.5 / 1.5。"""
    base = rng.normal(0.0, params.sd_daily, size=(int(n_paths), params.path_days))
    block_index = np.arange(params.path_days) // int(params.regime_block_length)
    low, high = params.regime_multipliers
    multiplier = np.where(block_index % 2 == 0, low, high)
    return base * multiplier[None, :]


def circular_block_resample(
    rng: np.random.Generator,
    n_paths: int,
    pool: np.ndarray,
    *,
    days: int,
    block_length: int,
) -> np.ndarray:
    """circular block 重采样（环回取模；**不按较短长度截断**）。

    Args:
        rng: 本单元命名子流的生成器。
        n_paths: 路径数。
        pool: 已去中心的源池（1 维；长度必须 >= block_length）。
        days: 每条路径的天数。
        block_length: 块长（必须 = H）。

    Returns:
        ``(n_paths, days)``。

    Raises:
        SceneError: 池长度 < block_length（无法形成完整 block，禁止截断）。
    """
    values = np.asarray(pool, dtype=float).ravel()
    length = int(block_length)
    if values.size < length:
        raise SceneError(f"源池长度 {values.size} < block_length {length}（禁止按较短长度截断）")
    blocks = int(math.ceil(int(days) / length))
    starts = rng.integers(0, values.size, size=(int(n_paths), blocks))
    index = (starts[:, :, None] + np.arange(length)[None, None, :]) % values.size
    return values[index].reshape(int(n_paths), blocks * length)[:, : int(days)]


def gen_real_block(
    rng: np.random.Generator, n_paths: int, params: DgpParams, pool: np.ndarray
) -> np.ndarray:
    """真实换种子差的 block 重采样路径。"""
    return circular_block_resample(
        rng,
        n_paths,
        pool,
        days=params.path_days,
        block_length=params.block_length,
    )


@dataclass(frozen=True)
class SceneRegistry:
    """场景名 → 生成函数（``(rng, n_paths) -> (n, days)``）的注册表。

    ``pool_role`` 记录该注册表服务的数据池角色（``train`` / ``holdout``），
    训练阶段只能拿 ``train`` 注册表，防止读取 holdout（结构性强制的第一道）。
    """

    pool_role: str
    functions: Mapping[str, Callable[[np.random.Generator, int], np.ndarray]]
    params: DgpParams

    def __post_init__(self) -> None:
        if self.pool_role not in {"train", "holdout"}:
            raise SceneError(f"pool_role 必须是 train/holdout，实际 {self.pool_role!r}")
        missing = [name for name in SCENE_REQUIRED if name not in self.functions]
        if missing:
            raise SceneError(f"场景注册表缺必需场景（禁止删除困难场景）: {missing}")
        extra = sorted(set(self.functions) - set(SCENE_REQUIRED))
        if extra:
            raise SceneError(f"场景注册表含未注册场景: {extra}")

    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self.functions))

    def build(self, name: str, rng: np.random.Generator, n_paths: int) -> np.ndarray:
        """生成一个单元的路径矩阵。"""
        if name not in self.functions:
            raise SceneError(f"未注册场景: {name!r}（允许 {sorted(self.functions)}）")
        return np.asarray(self.functions[name](rng, int(n_paths)), dtype=float)


def build_scene_registry(
    pool: Optional[np.ndarray], *, params: DgpParams, pool_role: str
) -> SceneRegistry:
    """构造场景注册表；``real_seed_block`` 用本角色的去中心池（禁止跨池采样）。"""
    if pool is None:
        raise SceneError(f"场景注册表需要 {pool_role} 池（禁止用空池/跨池代替）")
    centred = np.asarray(pool, dtype=float).ravel()
    functions: Dict[str, Callable[[np.random.Generator, int], np.ndarray]] = {
        "ma20": lambda rng, n: gen_ma(rng, n, params),
        "ar1_06": lambda rng, n: gen_ar1(rng, n, params),
        "regime_hetero": lambda rng, n: gen_regime(rng, n, params),
        "real_seed_block": lambda rng, n: gen_real_block(rng, n, params, centred),
    }
    return SceneRegistry(pool_role=str(pool_role), functions=functions, params=params)


def scene_parameters(params: DgpParams) -> Dict[str, Any]:
    """§5.5 需冻结的场景参数表（写进 manifest）。"""
    return {
        "ma20": {
            "formula": "d_t = sum_{j=0..H-1} eps_{t+j}（前向窗口；与后向 eps_{t-j} 同分布）",
            "innovations": "standard normal scaled to sd_daily/sqrt(H)",
            "sd_daily": float(params.sd_daily),
            "H_equals_block_length": int(params.block_length),
        },
        "ar1_06": {
            "formula": "x_t = rho*x_{t-1} + eps_t",
            "rho": float(params.ar_rho),
            "innovation_sd": float(params.sd_daily * math.sqrt(1.0 - params.ar_rho**2)),
            "stationary_sd": float(params.sd_daily),
            "warmup_days": int(AR_WARMUP_DAYS),
        },
        "regime_hetero": {
            "formula": "sd_t = sd_daily * multiplier[floor(t/block) % 2]",
            "block": int(params.regime_block_length),
            "multipliers": [float(x) for x in params.regime_multipliers],
        },
        "real_seed_block": {
            "formula": "circular block resample of centred real seed-swap difference",
            "block_length": int(params.block_length),
            "centre": "source pool own mean",
            "wrap": "index modulo pool length",
            "truncation_forbidden": True,
            "cross_pool_sampling_forbidden": True,
        },
    }


# ---------------------------------------------------------------------------
# 真实源对（B0 / B0'）与配对收益差
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourcePair:
    """B0/B0' 配对差（``d_t = ln(1+r_noise) - ln(1+r_base)``，按 (split_index, date) 对齐）。"""

    frame: pd.DataFrame
    base_path: str
    noise_path: str
    base_sha256: str
    noise_sha256: str

    @property
    def rows(self) -> int:
        return int(len(self.frame))

    def folds(self) -> List[int]:
        return sorted(int(x) for x in self.frame["split_index"].unique())

    def identity(self) -> Dict[str, Any]:
        """源对身份（manifest/report 引用）。"""
        return {
            "base_chain_nav": self.base_path,
            "noise_chain_nav": self.noise_path,
            "base_sha256": self.base_sha256,
            "noise_sha256": self.noise_sha256,
            "rows": self.rows,
            "folds": self.folds(),
            "date_key": ["split_index", "date"],
            "pair_order": "a=noise_run_id(B0'), b=base_run_id(B0); d=ln(1+r_a)-ln(1+r_b)",
        }


def load_source_pair(
    base_path: Path,
    noise_path: Path,
    *,
    expected_base_sha256: Optional[str] = None,
    expected_noise_sha256: Optional[str] = None,
) -> SourcePair:
    """读取 B0/B0' 链式净值并按冻结口径构造配对日收益差。

    口径（与 T0.5b PSD harness 完全一致，故可复现 r3 数值）：
    每个文件按 ``["split_index","date"]`` 排序、在 ``split_index`` 内做相邻收益、两文件按
    ``(split_index,date)`` inner merge 后取 ``d = log1p(r_noise) - log1p(r_base)``。

    Raises:
        SceneError: 文件缺失或哈希与冻结值不符（禁止换源）。
    """
    base = Path(base_path)
    noise = Path(noise_path)
    for path, expected, what in (
        (base, expected_base_sha256, "base"),
        (noise, expected_noise_sha256, "noise"),
    ):
        if not path.is_file():
            raise SceneError(f"源对文件缺失（{what}）: {path}")
        if expected is not None:
            actual = _sha256(path)
            if actual != str(expected):
                raise SceneError(
                    f"源对文件哈希与冻结值不符（{what}，禁止换源）: {path} 实际={actual} 期望={expected}"
                )
    frames = []
    for path in (base, noise):
        frame = pd.read_csv(path).sort_values(["split_index", "date"]).copy()
        # NAV 必须有限且为正：非法值不得被后续 dropna() 静默压成更短的有效序列（R2-P5B-05）
        nav = pd.to_numeric(frame["nav"], errors="coerce")
        if not np.isfinite(nav.to_numpy(dtype=float)).all() or (nav.to_numpy(dtype=float) <= 0).any():
            raise SceneError(f"源对 NAV 含非有限或非正值（禁止静默丢弃坏源样本）: {path}")
        frame["nav"] = nav
        previous = frame.groupby("split_index")["nav"].shift(1)
        frame["r"] = frame["nav"] / previous - 1.0
        frames.append(frame)
    # 配对键必须两侧唯一且完全一致：inner join 会把缺日静默压成「双日 vs 单日」收益差，
    # 从而在两条相同路径之间制造并不存在的 d（R2-P5B-05）。
    keys = []
    for frame, label in ((frames[0], "base"), (frames[1], "noise")):
        pair = list(zip(frame["split_index"].tolist(), frame["date"].tolist()))
        if len(pair) != len(set(pair)):
            raise SceneError(f"源对 {label} 存在重复配对键（split_index,date）")
        keys.append(set(pair))
    missing = keys[0] ^ keys[1]
    if missing:
        raise SceneError(
            f"源对配对键集不一致（仅一侧存在 {len(missing)} 个键），禁止用 inner join 静默压缩"
        )
    merged = frames[0].merge(frames[1], on=["split_index", "date"], how="inner").dropna()
    if merged.empty:
        raise SceneError(f"源对无可对齐的配对收益: base={base} noise={noise}")
    difference = np.log1p(merged["r_y"].to_numpy()) - np.log1p(merged["r_x"].to_numpy())
    frame = pd.DataFrame(
        {
            "split_index": merged["split_index"].to_numpy(),
            "date": merged["date"].to_numpy(),
            "d": difference,
        }
    )
    if not np.isfinite(frame["d"].to_numpy()).all():
        raise SceneError("配对收益差含非有限值（源文件异常）")
    return SourcePair(
        frame=frame,
        base_path=base.as_posix(),
        noise_path=noise.as_posix(),
        base_sha256=_sha256(base),
        noise_sha256=_sha256(noise),
    )


def paired_log_diff(nav_a: Iterable[float], nav_b: Iterable[float]) -> np.ndarray:
    """``d_t = ln(V_a,t/V_a,t-1) - ln(V_b,t/V_b,t-1)``（禁止用逐日独立方差）。"""
    first = np.asarray(list(nav_a), dtype=float)
    second = np.asarray(list(nav_b), dtype=float)
    if first.shape != second.shape:
        raise SceneError(f"两臂净值序列长度不一致: {first.shape} vs {second.shape}")
    if first.size < 2:
        raise SceneError(f"配对收益至少需要两个估值点（D0 只作估值日），实际 {first.size}")
    if not (np.isfinite(first).all() and np.isfinite(second).all()):
        raise SceneError("净值序列含非有限值")
    if (first <= 0).any() or (second <= 0).any():
        raise SceneError("净值序列含非正值（无法取对数）")
    diff = np.log(first[1:] / first[:-1]) - np.log(second[1:] / second[:-1])
    return diff


def pool_for_folds(pair: SourcePair, folds: Sequence[int]) -> Dict[str, Any]:
    """取指定源折的配对差并按**池自身均值**去中心（训练/验证各自去中心，禁止跨池）。"""
    wanted = [int(fold) for fold in folds]
    subset = pair.frame[pair.frame["split_index"].isin(wanted)]
    values = subset["d"].to_numpy(dtype=float)
    if values.size == 0:
        raise SceneError(f"源折 {wanted} 无配对收益（源文件或折集合异常）")
    mean = float(values.mean())
    return {
        "folds": wanted,
        "rows": int(values.size),
        "mean": mean,
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "centred": values - mean,
    }


def block_date_blocks(pair: SourcePair, folds: Sequence[int]) -> Dict[str, Any]:
    """源折 → (split_index, date) 复合键的范围证据（日期块互斥/覆盖的登记材料）。

    注意：``date`` 为 **fold-local 序号**（0..125），不是自然日；源文件没有自然日期列。
    """
    wanted = [int(fold) for fold in folds]
    subset = pair.frame[pair.frame["split_index"].isin(wanted)]
    blocks = []
    for fold in sorted(wanted):
        chunk = subset[subset["split_index"] == fold]
        if chunk.empty:
            continue
        blocks.append(
            {
                "split_index": fold,
                "date_min": int(chunk["date"].min()),
                "date_max": int(chunk["date"].max()),
                "rows": int(len(chunk)),
            }
        )
    return {
        "key": ["split_index", "date"],
        "date_semantics": "fold_local_ordinal",
        "blocks": blocks,
    }
