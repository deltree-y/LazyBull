# -*- coding: utf-8 -*-
# 复杂度豁免：一次性 P5b PSD 校准脚本（PSD r3 已闭合，不再维护；不入生产链路）。
# 拆分计划：若将来复用，先抽 _build_corpus/_emit_reports 两个子过程再降 run 的行数与分支。
"""Auditable P5b PSD income-referee calibration harness.

Historical simulation only. No production account, runs, ledger, or SLA writes.
The output is calibration evidence, not forward evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "docs/plans/v2/p5b_calibration_manifest_psd_20261101.json"
FORBIDDEN_OUTPUT_PARTS = ("data/paper", "data/runs", "data/ledger", "data/paper_shadow")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rooted(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return max(0.0, center - half), min(1.0, center + half)


def psd_variance(block_means: np.ndarray) -> float:
    """Bartlett lag-1 variance with common J-1 denominator.

    ``block_means`` is a 1-D array of J centered block means.
    Returns a non-negative float; J<2 is invalid.
    """
    x = np.asarray(block_means, dtype=float)
    j = x.size
    if j < 2 or not np.all(np.isfinite(x)):
        return float("nan")
    c = x - x.mean()
    value = (c[0] ** 2 + c[-1] ** 2 + np.sum((c[1:] + c[:-1]) ** 2)) / (2.0 * (j - 1))
    return float(max(0.0, value))


def t_scores(paths: np.ndarray, h: int, look_blocks: tuple[int, ...], delta: float) -> np.ndarray:
    """PSD boundary-centered T for each path/look; invalid entries are NaN."""
    out = np.full((paths.shape[0], len(look_blocks)), np.nan)
    for k, j in enumerate(look_blocks):
        b = paths[:, : j * h].reshape(paths.shape[0], j, h).mean(axis=2)
        for i in range(paths.shape[0]):
            v = psd_variance(b[i])
            if np.isfinite(v) and v > 0:
                out[i, k] = (b[i].mean() + delta) / math.sqrt(v / j)
    return out


def load_pair(base: Path, noise: Path) -> pd.DataFrame:
    frames = []
    for path in (base, noise):
        df = pd.read_csv(path).sort_values(["split_index", "date"]).copy()
        df["r"] = df.groupby("split_index")["nav"].pct_change()
        frames.append(df)
    merged = frames[0].merge(frames[1], on=["split_index", "date"], how="inner").dropna()
    d = np.log1p(merged["r_y"].to_numpy()) - np.log1p(merged["r_x"].to_numpy())
    return pd.DataFrame({
        "split_index": merged["split_index"].to_numpy(),
        "date": merged["date"].to_numpy(),
        "d": d,
    })


def gen_ma(rng: np.random.Generator, n: int, sd_daily: float, h: int) -> np.ndarray:
    sigma = sd_daily / math.sqrt(h)
    eps = rng.normal(0, sigma, size=(n, 500 + h - 1))
    cs = np.pad(np.cumsum(eps, axis=1), ((0, 0), (1, 0)))
    return cs[:, h:] - cs[:, :-h]


def gen_ar(rng: np.random.Generator, n: int, sd_daily: float, rho: float = 0.6) -> np.ndarray:
    sigma = sd_daily * math.sqrt(1 - rho * rho)
    eps = rng.normal(0, sigma, size=(n, 600))
    x = np.empty_like(eps)
    x[:, 0] = eps[:, 0]
    for t in range(1, x.shape[1]):
        x[:, t] = rho * x[:, t - 1] + eps[:, t]
    return x[:, 100:]


def gen_regime(rng: np.random.Generator, n: int, sd_daily: float) -> np.ndarray:
    base = rng.normal(0, sd_daily, size=(n, 500))
    block = np.arange(500) // 100
    return base * np.where(block % 2 == 0, 0.5, 1.5)[None, :]


def gen_real(rng: np.random.Generator, n: int, pool: np.ndarray, h: int) -> np.ndarray:
    out = np.empty((n, 500))
    for i in range(n):
        starts = rng.integers(0, len(pool), size=math.ceil(500 / h))
        blocks = [np.take(pool, np.arange(s, s + h), mode="wrap") for s in starts]
        out[i] = np.concatenate(blocks)[:500]
    return out


def build_scenes(pool: np.ndarray, h: int, sd_daily: float) -> dict[str, Any]:
    return {
        "ma20": lambda rng, n: gen_ma(rng, n, sd_daily, h),
        "ar1_06": lambda rng, n: gen_ar(rng, n, sd_daily),
        "regime_hetero": lambda rng, n: gen_regime(rng, n, sd_daily),
        "real_seed_block": lambda rng, n: gen_real(rng, n, pool, h),
    }


def _effect_rate(z: np.ndarray, thresholds: np.ndarray) -> tuple[int, int]:
    reject = np.isfinite(z) & (z < -thresholds[None, :])
    return int(reject.any(axis=1).sum()), int(z.shape[0])


def run(manifest_path: Path, out_path: Path) -> Mapping[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_resolved = out_path.resolve()
    parts = "/".join(out_resolved.parts).lower()
    if any(part in parts for part in FORBIDDEN_OUTPUT_PARTS):
        raise ValueError(f"refusing production output path: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prereg = rooted(manifest["prereg"]["path"])
    if sha256(prereg) != manifest["prereg"]["sha256"]:
        raise RuntimeError("prereg hash mismatch")

    source = manifest["calibration"]["source_pair"]
    base = rooted(source["base_chain_nav"])
    noise = rooted(source["noise_chain_nav"])
    for path, key in ((base, "base_sha256"), (noise, "noise_sha256")):
        if sha256(path) != source[key]:
            raise RuntimeError(f"source hash mismatch: {path}")

    sample = manifest["sample"]
    h = int(sample["H"])
    look_blocks = tuple(int(x) for x in sample["look_blocks"])
    delta = float(manifest["statistic"]["delta_log_daily"])
    alpha = float(manifest["threshold"]["alpha_train"])
    alpha_look = float(manifest["threshold"]["alpha_per_look"])
    max_invalid = float(manifest["threshold"]["invalid_rate_per_look_max"])
    counts = manifest["calibration"]["counts"]
    n_train = int(counts["train_per_unit"])
    n_valid = int(counts["validation_per_unit"])
    n_effect = int(counts["effect_reps_per_scene_per_level"])
    seed = int(counts["seed"])
    sd_daily = 0.0037

    pair = load_pair(base, noise)
    train = pair[pair["split_index"].isin(source["train_folds"])].copy()
    valid = pair[pair["split_index"].isin(source["validation_source_folds"])].copy()
    train_pool = train["d"].to_numpy()
    valid_pool = valid["d"].to_numpy()
    train_pool = train_pool - train_pool.mean()
    valid_pool = valid_pool - valid_pool.mean()

    rng = np.random.default_rng(seed)
    train_stats: dict[tuple[str, str], np.ndarray] = {}
    for scene, gen in build_scenes(train_pool, h, sd_daily).items():
        for null, mu in (("mu=0", 0.0), ("mu=-delta", -delta)):
            train_stats[(scene, null)] = t_scores(gen(rng, n_train) + mu, h, look_blocks, delta)

    invalid_units: list[dict[str, Any]] = []
    threshold_units: dict[str, list[float | None]] = {}
    thresholds: list[float] = []
    for k in range(len(look_blocks)):
        values = []
        for key, z in train_stats.items():
            v = z[:, k]
            v = v[np.isfinite(v)]
            invalid = 1.0 - len(v) / len(z)
            if invalid > max_invalid or len(v) < 100:
                invalid_units.append({"unit": list(key), "look": k,
                                      "invalid_rate": invalid, "valid_count": len(v)})
                values.append(None)
            else:
                values.append(float(-np.quantile(v, alpha_look)))
        valid_values = [x for x in values if x is not None]
        if not valid_values:
            raise RuntimeError(f"no valid threshold unit at look {k}")
        thresholds.append(max(valid_values))

    validation: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for scene, gen in build_scenes(valid_pool, h, sd_daily).items():
        for null, mu in (("mu=0", 0.0), ("mu=-delta", -delta)):
            z = t_scores(gen(rng, n_valid) + mu, h, look_blocks, delta)
            successes, total = _effect_rate(z, np.asarray(thresholds))
            rate = successes / total
            lo, hi = wilson_interval(successes, total)
            invalid = (1.0 - np.isfinite(z).mean(axis=0)).tolist()
            key = f"{scene}|{null}"
            validation[key] = {
                "successes": successes, "total": total,
                "rate": rate, "wilson_low": lo, "wilson_high": hi,
                "invalid_rate_by_look": invalid,
            }
            if hi > 0.06:
                failures.append({"unit": key, "wilson_high": hi})

    effects: dict[str, Any] = {}
    for level in (0.5, 1.0, 1.5, 3.0, 5.0):
        effects[str(level)] = {}
        for scene, gen in build_scenes(valid_pool, h, sd_daily).items():
            z = t_scores(gen(rng, n_effect) - level * delta, h, look_blocks, delta)
            successes, total = _effect_rate(z, np.asarray(thresholds))
            lo, hi = wilson_interval(successes, total)
            effects[str(level)][scene] = {
                "successes": successes, "total": total,
                "rate": successes / total, "wilson_low": lo, "wilson_high": hi,
            }

    passed = not invalid_units and not failures
    mde: dict[str, float | None] = {}
    for scene in build_scenes(valid_pool, h, sd_daily):
        hit = [float(level) for level in (0.5, 1.0, 1.5, 3.0, 5.0)
               if effects[str(level)][scene]["rate"] >= 0.8]
        mde[scene] = min(hit) if hit else None

    result = {
        "run_id": manifest["run_id"],
        "status": "conditional_monte_carlo_not_forward_evidence",
        "verdict": "PASS" if passed else "FAIL",
        "manifest_sha256": sha256(manifest_path),
        "prereg_sha256": sha256(prereg),
        "script_sha256": sha256(Path(__file__)),
        "source_sha256": {"base": sha256(base), "noise": sha256(noise)},
        "sample": {"H": h, "look_blocks": list(look_blocks), "K": len(look_blocks)},
        "alpha_train": alpha,
        "alpha_per_look": alpha_look,
        "delta": delta,
        "thresholds": thresholds,
        "invalid_units": invalid_units,
        "validation": validation,
        "effects": effects,
        "mde_delta_by_scene": mde,
        "failures": failures,
        "old_fail_retained": bool(manifest["evidence_identity"]["old_fail_retained"]),
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out", default=str(ROOT / "temp/p5b_psd_cal_20261101_r1/audit_result.json"))
    args = parser.parse_args(argv)
    result = run(Path(args.manifest), Path(args.out))
    print(json.dumps({"verdict": result["verdict"], "out": args.out}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
