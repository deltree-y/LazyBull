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
import sys
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


STAGE_CODES = {"train": 1, "validation": 2, "effect": 3}
SCENE_CODES = {"ma20": 1, "ar1_06": 2, "regime_hetero": 3, "real_seed_block": 4}
NULL_CODES = {"mu=0": 0, "mu=-delta": 1}


def unit_rng(seed: int, stage: str, scene: str, null: str = "", level: float = 0.0,
             prefix: str = "r2") -> tuple[np.random.Generator, str]:
    """Stable per-unit SeedSequence; independent of call order."""
    code = [int(seed), STAGE_CODES[stage], SCENE_CODES[scene], NULL_CODES.get(null, 0),
            int(round(level * 1000))]
    rid = f"{prefix}:{stage}:{scene}:{null or 'effect'}:{level:g}"
    return np.random.default_rng(np.random.SeedSequence(code)), rid


def config_consistency_errors(manifest: Mapping[str, Any]) -> list[str]:
    """Return manifest consistency errors; empty means executable."""
    errors: list[str] = []
    sample = manifest["sample"]
    h = int(sample["H"])
    look_blocks = [int(x) for x in sample["look_blocks"]]
    look_days = [int(x) for x in sample["look_days"]]
    k = int(sample["K"])
    if k != len(look_blocks) or look_days != [h * j for j in look_blocks]:
        errors.append("K/look_days/look_blocks inconsistent")
    dgp = manifest["calibration"]["dgp"]
    block = manifest["calibration"]["block_resample"]
    if not (float(block["block_length"]) == float(dgp["block_length"]) == float(h)):
        errors.append("block_resample.block_length != dgp.block_length != H")
    alpha = float(manifest["threshold"]["alpha_train"])
    alpha_look = float(manifest["threshold"]["alpha_per_look"])
    if not math.isclose(alpha_look, alpha / k, rel_tol=0.0, abs_tol=1e-12):
        errors.append("alpha_per_look != alpha_train/K")
    source = manifest["calibration"]["source_pair"]
    if set(source["train_folds"]) & set(source["validation_source_folds"]):
        errors.append("train/validation source folds overlap")
    return errors


def evaluation_failures(validation: Mapping[str, Any], max_invalid: float,
                        max_wilson: float) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for unit, item in validation.items():
        for look, rate in enumerate(item["invalid_rate_by_look"]):
            if float(rate) > max_invalid:
                failures.append({"unit": unit, "stage": "validation", "look": look,
                                 "invalid_rate": float(rate), "reason": "invalid_rate"})
        if float(item["wilson_high"]) > max_wilson:
            failures.append({"unit": unit, "stage": "validation",
                             "wilson_high": float(item["wilson_high"]),
                             "reason": "wilson"})
    return failures


def register_mde(passed: bool, effects: Mapping[str, Any], grid: list[float],
                 scenes: list[str]) -> dict[str, float | None] | None:
    if not passed:
        return None
    out: dict[str, float | None] = {}
    for scene in scenes:
        hit = [float(x) for x in grid if effects[str(x)][scene]["rate"] >= 0.8]
        out[scene] = min(hit) if hit else None
    return out


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
    return float(value)


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
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {out_path}")
    consistency_errors = config_consistency_errors(manifest)
    if consistency_errors:
        raise ValueError("manifest inconsistent: " + "; ".join(consistency_errors))
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
    look_days = [int(x) for x in sample["look_days"]]
    k_declared = int(sample["K"])
    delta = float(manifest["statistic"]["delta_log_daily"])
    alpha = float(manifest["threshold"]["alpha_train"])
    alpha_look = float(manifest["threshold"]["alpha_per_look"])
    max_invalid = float(manifest["threshold"]["invalid_rate_per_look_max"])
    max_wilson = float(manifest["threshold"]["holdout_wilson_upper_max"])
    counts = manifest["calibration"]["counts"]
    n_train = int(counts["train_per_unit"])
    n_valid = int(counts["validation_per_unit"])
    n_effect = int(counts["effect_reps_per_scene_per_level"])
    seed = int(counts["seed"])
    prefix = str(counts["realization_prefix"])
    dgp = manifest["calibration"]["dgp"]
    effect_grid = [float(x) for x in manifest["calibration"]["effect_grid_delta"]]

    if k_declared != len(look_blocks) or look_days != [h * j for j in look_blocks]:
        raise ValueError("manifest K/look_days/look_blocks inconsistent")
    if float(dgp["block_length"]) != h or float(dgp["path_days"]) != 500:
        raise ValueError("unsupported DGP block/path length")
    if float(dgp["sd_daily"]) != 0.0037 or float(dgp["ar_rho"]) != 0.6:
        raise ValueError("unsupported DGP sd/ar_rho")
    if int(dgp["regime_block_length"]) != 100 or list(dgp["regime_multipliers"]) != [0.5, 1.5]:
        raise ValueError("unsupported DGP regime parameters")
    required_scenes = {"ma20", "ar1_06", "regime_hetero", "real_seed_block"}
    if set(manifest["calibration"]["scenes"]) != required_scenes:
        raise ValueError("unsupported scene set")
    if set(manifest["calibration"]["nulls"]) != {"mu=0", "mu=-delta"}:
        raise ValueError("unsupported null set")
    if set(source["train_folds"]) & set(source["validation_source_folds"]):
        raise ValueError("train/validation source folds overlap")

    pair = load_pair(base, noise)
    train = pair[pair["split_index"].isin(source["train_folds"])].copy()
    valid = pair[pair["split_index"].isin(source["validation_source_folds"])].copy()
    train_pool = train["d"].to_numpy()
    valid_pool = valid["d"].to_numpy()
    train_pool = train_pool - train_pool.mean()
    valid_pool = valid_pool - valid_pool.mean()

    scene_names = sorted(required_scenes)
    scene_fns = build_scenes(train_pool, h, float(dgp["sd_daily"]))
    valid_scene_fns = build_scenes(valid_pool, h, float(dgp["sd_daily"]))
    train_stats: dict[tuple[str, str], np.ndarray] = {}
    realization_ids: dict[str, str] = {}
    for scene in scene_names:
        for null, mu in (("mu=0", 0.0), ("mu=-delta", -delta)):
            rng, rid = unit_rng(seed, "train", scene, null, prefix=prefix)
            train_stats[(scene, null)] = t_scores(
                scene_fns[scene](rng, n_train) + mu, h, look_blocks, delta)
            realization_ids[f"train|{scene}|{null}"] = rid

    invalid_units: list[dict[str, Any]] = []
    threshold_units: dict[str, list[dict[str, Any] | None]] = {}
    thresholds: list[float] = []
    threshold_sources: list[str] = []
    for k in range(len(look_blocks)):
        values = []
        unit_details = []
        for key, z in train_stats.items():
            v = z[:, k]
            valid_values_in_unit = v[np.isfinite(v)]
            invalid = 1.0 - len(valid_values_in_unit) / len(z)
            detail: dict[str, Any] = {
                "valid_count": int(len(valid_values_in_unit)),
                "total": int(len(z)),
                "invalid_rate": float(invalid),
                "quantile": None,
            }
            if invalid > max_invalid or len(valid_values_in_unit) < 100:
                invalid_units.append({"stage": "train", "unit": list(key), "look": k,
                                      "invalid_rate": invalid,
                                      "valid_count": len(valid_values_in_unit)})
                values.append(None)
            else:
                q = float(-np.quantile(valid_values_in_unit, alpha_look))
                detail["quantile"] = q
                values.append(q)
            unit_details.append(detail)
        valid_pairs = [(key, vals) for key, vals in zip(train_stats, values) if vals is not None]
        if not valid_pairs:
            raise RuntimeError(f"no valid threshold unit at look {k}")
        threshold_sources.append(str(valid_pairs[int(np.argmax([x[1] for x in valid_pairs]))][0]))
        thresholds.append(max(x[1] for x in valid_pairs))
        threshold_units[str(k)] = unit_details

    validation: dict[str, Any] = {}
    for scene in scene_names:
        for null, mu in (("mu=0", 0.0), ("mu=-delta", -delta)):
            rng, rid = unit_rng(seed, "validation", scene, null, prefix=prefix)
            z = t_scores(valid_scene_fns[scene](rng, n_valid) + mu, h, look_blocks, delta)
            successes, total = _effect_rate(z, np.asarray(thresholds))
            rate = successes / total
            lo, hi = wilson_interval(successes, total)
            key = f"{scene}|{null}"
            validation[key] = {
                "successes": successes, "total": total,
                "rate": rate, "wilson_low": lo, "wilson_high": hi,
                "invalid_rate_by_look": (1.0 - np.isfinite(z).mean(axis=0)).tolist(),
                "realization_id": rid,
            }

    failures = evaluation_failures(validation, max_invalid, max_wilson)
    failures.extend(invalid_units)

    effects: dict[str, Any] = {}
    for level in effect_grid:
        effects[str(level)] = {}
        for scene in scene_names:
            rng, rid = unit_rng(seed, "effect", scene, level=level, prefix=prefix)
            z = t_scores(valid_scene_fns[scene](rng, n_effect) - level * delta,
                         h, look_blocks, delta)
            successes, total = _effect_rate(z, np.asarray(thresholds))
            lo, hi = wilson_interval(successes, total)
            effects[str(level)][scene] = {
                "successes": successes, "total": total,
                "rate": successes / total, "wilson_low": lo, "wilson_high": hi,
                "realization_id": rid,
            }

    passed = not failures
    mde = register_mde(passed, effects, effect_grid, scene_names)

    result = {
        "run_id": manifest["run_id"],
        "status": "conditional_monte_carlo_not_forward_evidence",
        "verdict": "PASS" if passed else "FAIL",
        "manifest_sha256": sha256(manifest_path),
        "prereg_sha256": sha256(prereg),
        "script_sha256": sha256(Path(__file__)),
        "source_sha256": {"base": sha256(base), "noise": sha256(noise)},
        "sample": {"H": h, "look_blocks": list(look_blocks),
                   "look_days": look_days, "K": k_declared},
        "config": {"effect_grid_delta": effect_grid,
                   "max_wilson_upper": max_wilson,
                   "max_invalid_rate_per_look": max_invalid,
                   "dgp": dgp,
                   "realization_prefix": prefix},
        "environment": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__},
        "alpha_train": alpha,
        "alpha_per_look": alpha_look,
        "delta": delta,
        "thresholds": thresholds,
        "threshold_units": threshold_units,
        "threshold_sources": threshold_sources,
        "invalid_units": invalid_units,
        "validation": validation,
        "effects": effects,
        "mde_registered": passed,
        "mde_delta_by_scene": mde,
        "failures": failures,
        "realization_ids": realization_ids,
        "old_fail_retained": bool(manifest["evidence_identity"]["old_fail_retained"]),
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    digest = sha256(out_path)
    out_path.with_suffix(out_path.suffix + ".sha256").write_text(digest + "\n", encoding="utf-8")
    result["result_sha256"] = digest
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out", default=str(ROOT / "temp/p5b_psd_cal_20261101_r2/audit_result.json"))
    args = parser.parse_args(argv)
    result = run(Path(args.manifest), Path(args.out))
    print(json.dumps({"verdict": result["verdict"], "out": args.out}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
