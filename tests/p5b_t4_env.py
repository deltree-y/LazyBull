# -*- coding: utf-8 -*-
"""P5b T4 测试环境（tests/ 全量豁免复杂度；不是测试用例模块）。

提供两类**完全隔离**的合成现场（不读真实生产数据、不写真实试车场根）：

1. ``synthetic_source``：在 tmp 里造一个「假仓库」——两份合成链式净值 CSV（B0/B0' 形态）、
   预登记与 PSD manifest 占位文件，供 ``load_source_pair`` / ``verify_source_files`` 使用；
2. ``synthetic_campaign``：直接按 journal→分片→index 的真实写入口构造已发布配对账本，
   供 ``paired_returns_from_ledger`` 使用（不驱动宿主、不建 PaperHost）。

另含：小规模 manifest 文档工厂、完整流水线（冻结→训练→holdout）驱动、树快照与生产根快照。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.lazybull.v2.hosts.testbed.journal import (  # noqa: E402
    build_index,
    load_all_attempts,
    save_day_journal,
    write_index,
    write_shard,
)
from src.lazybull.v2.hosts.testbed.layout import layout_for  # noqa: E402
from src.lazybull.v2.hosts.testbed.ledger_common import (  # noqa: E402
    LedgerGap,
    record_invalidations,
)
from src.lazybull.v2.hosts.testbed.referee import (  # noqa: E402
    DgpParams,
    build_manifest_document,
    calibrator_identity,
    freeze_calibration_manifest,
    load_source_pair,
    run_holdout,
    source_pair_for_manifest,
    train_calibration,
)

#: 裁判子包目录（源码级守卫与校准器身份用）
REFEREE_DIR = ROOT / "src/lazybull/v2/hosts/testbed/referee"
#: 裁判子包模块名
REFEREE_MODULES: Tuple[str, ...] = (
    "statistics.py",
    "sampling.py",
    "scenes.py",
    "training.py",
    "holdout.py",
    "power.py",
    "ledger_source.py",
    "manifest.py",
    "artifacts.py",
    "report.py",
    "calibration.py",
    "__init__.py",
)
#: 合成源折（训练折 0/1，holdout 折 2/3）
TRAIN_FOLDS: Tuple[int, ...] = (0, 1)
HOLDOUT_FOLDS: Tuple[int, ...] = (2, 3)
#: 每折行数
FOLD_ROWS = 60
#: 小规模实现数（协议测试用；远低于 §5.3 下限，因此必须 require_scale_floor=False）
SMALL_COUNTS: Mapping[str, Any] = {
    "train_per_unit": 120,
    "holdout_per_unit": 200,
    "effect_per_scene_per_level": 60,
    "seed": 20261103,
    "realization_prefix": "p5b_t4_test",
}
#: §5.3 下限实现数（脚本入口测试用：必须能过 ``require_scale_floor``；向量化后约 2~4 秒）
FLOOR_COUNTS: Mapping[str, Any] = {
    "train_per_unit": 2000,
    "holdout_per_unit": 10000,
    "effect_per_scene_per_level": 2000,
    "seed": 20261103,
    "realization_prefix": "p5b_t4_test",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_state(root: Path) -> Dict[str, Any]:
    """目录递归快照（相对路径 → {size, sha256}）。"""
    base = Path(root)
    if not base.exists():
        return {}
    state: Dict[str, Any] = {}
    for path in sorted(base.rglob("*")):
        if path.is_file():
            state[path.relative_to(base).as_posix()] = _sha256(path)
    return state


def metadata_state(root: Path) -> Dict[str, Any]:
    """目录元数据快照（相对路径 → (size, mtime_ns)）。"""
    base = Path(root)
    if not base.exists():
        return {}
    state: Dict[str, Any] = {}
    for path in sorted(base.rglob("*")):
        if path.is_file():
            stat = path.stat()
            state[path.relative_to(base).as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return state


def production_snapshot() -> Dict[str, Any]:
    """生产根快照：``data/paper``、``data/runs`` 递归哈希 + ``data/models`` 元数据。"""
    return {
        "paper": tree_state(ROOT / "data" / "paper"),
        "runs": tree_state(ROOT / "data" / "runs"),
        "models": metadata_state(ROOT / "data" / "models"),
    }


def production_unchanged(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    return dict(before) == dict(after)


# ---------------------------------------------------------------------------
# 合成源对（假仓库）
# ---------------------------------------------------------------------------


def _nav_frame(fold: int, seed: int, drift: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, 0.004, size=FOLD_ROWS)
    nav = np.cumprod(1.0 + steps)
    return pd.DataFrame(
        {"date": np.arange(FOLD_ROWS), "nav": nav, "split_index": np.full(FOLD_ROWS, fold)}
    )


def synthetic_source(tmp_path: Path) -> SimpleNamespace:
    """在 tmp 构造假仓库（源对 CSV + 预登记 + PSD manifest 占位）。

    Returns:
        ``SimpleNamespace(repo_root, base_path, noise_path, prereg, psd_manifest, source)``。
    """
    repo = Path(tmp_path) / "fake_repo"
    base_dir = repo / "data/walk_forward/batches/factor_ab_base/raw"
    noise_dir = repo / "data/walk_forward/batches/factor_ab_noise/raw"
    base_dir.mkdir(parents=True, exist_ok=True)
    noise_dir.mkdir(parents=True, exist_ok=True)
    base_path = base_dir / "chain_nav_base.csv"
    noise_path = noise_dir / "chain_nav_noise.csv"
    pd.concat([_nav_frame(fold, 900 + fold, 0.0004) for fold in range(4)]).to_csv(
        base_path, index=False
    )
    pd.concat([_nav_frame(fold, 700 + fold, 0.0002) for fold in range(4)]).to_csv(
        noise_path, index=False
    )
    docs = repo / "docs/plans/v2"
    docs.mkdir(parents=True, exist_ok=True)
    prereg = docs / "prereg_fake.md"
    prereg.write_text("# fake prereg\n", encoding="utf-8")
    psd = docs / "psd_manifest_fake.json"
    psd.write_text(json.dumps({"status": "frozen_fake"}, ensure_ascii=False), encoding="utf-8")
    pair = load_source_pair(base_path, noise_path)
    return SimpleNamespace(
        repo_root=repo,
        base_path=base_path,
        noise_path=noise_path,
        prereg={"path": prereg.relative_to(repo).as_posix(), "sha256": _sha256(prereg)},
        psd_manifest={"path": psd.relative_to(repo).as_posix(), "sha256": _sha256(psd)},
        source=pair,
    )


def write_psd_manifest(
    env: SimpleNamespace, *, seed: int = 20261103, prefix: str = "p5b_t4_test"
) -> Tuple[Path, str]:
    """在假仓库里写一份 T0.5b 形态的 PSD manifest（供脚本 ``--psd-manifest`` 覆盖）。"""
    payload = {
        "schema_version": 1,
        "status": "frozen_for_new_mc_validation_not_evidence",
        "run_id": "p5b_psd_fake",
        "sample": {
            "n_max": 503,
            "H": 20,
            "look_blocks": [6, 12, 18, 25],
            "look_days": [120, 240, 360, 500],
            "K": 4,
        },
        "statistic": {"delta_log_daily": 0.0001936117626},
        "prereg": {"path": env.prereg["path"], "sha256": env.prereg["sha256"]},
        "calibration": {
            "source_pair": {
                "base_chain_nav": env.base_path.relative_to(env.repo_root).as_posix(),
                "noise_chain_nav": env.noise_path.relative_to(env.repo_root).as_posix(),
                "base_sha256": _sha256(env.base_path),
                "noise_sha256": _sha256(env.noise_path),
                "train_folds": list(TRAIN_FOLDS),
                "validation_source_folds": list(HOLDOUT_FOLDS),
            },
            "counts": {"seed": int(seed), "realization_prefix": str(prefix)},
            "dgp": {
                "sd_daily": 0.0037,
                "path_days": 500,
                "ar_rho": 0.6,
                "regime_block_length": 100,
                "regime_multipliers": [0.5, 1.5],
                "block_length": 20,
            },
            "block_resample": {
                "method": "circular",
                "block_length": 20,
                "center": "each pool own mean",
                "cross_pool_sampling_forbidden": True,
            },
        },
    }
    target = env.repo_root / "docs/plans/v2/psd_manifest_fake_full.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target, _sha256(target)


def dgp_params(**overrides: Any) -> DgpParams:
    params = {
        "sd_daily": 0.0037,
        "path_days": 500,
        "ar_rho": 0.6,
        "regime_block_length": 100,
        "regime_multipliers": (0.5, 1.5),
        "block_length": 20,
    }
    params.update(overrides)
    return DgpParams(**params)


def probe_document(env: SimpleNamespace, *, h: int = 20) -> Dict[str, Any]:
    """构造 ``source_pair_for_manifest`` 可接受的最小文档（用于取得 PreparedSource）。"""
    return {
        "source_pair": {
            "base_chain_nav": env.source.base_path,
            "noise_chain_nav": env.source.noise_path,
            "base_sha256": _sha256(env.base_path),
            "noise_sha256": _sha256(env.noise_path),
            "train_folds": list(TRAIN_FOLDS),
            "holdout_folds": list(HOLDOUT_FOLDS),
        },
        "files": {"prereg": env.prereg, "psd_manifest": env.psd_manifest},
        "income_referee": {"sample": {"H": h, "n_max": 503}},
        "calibration": {
            "dgp": {
                "sd_daily": 0.0037,
                "path_days": 500,
                "ar_rho": 0.6,
                "regime_block_length": 100,
                "regime_multipliers": [0.5, 1.5],
                "block_length": 20,
            },
            "block_resample": {
                "method": "circular",
                "block_length": 20,
                "center": "each pool own mean",
                "cross_pool_sampling_forbidden": True,
            },
        },
    }


def fast_manifest(
    tmp_path: Path,
    *,
    counts: Optional[Mapping[str, Any]] = None,
    h: int = 20,
    run_id: str = "p5b_t4_test_referee",
) -> SimpleNamespace:
    """构造小规模 manifest 文档（协议测试用，低于 §5.3 下限）。"""
    env = synthetic_source(tmp_path)
    probe = probe_document(env, h=h)
    source = source_pair_for_manifest(probe, repo_root=env.repo_root)
    merged = dict(SMALL_COUNTS)
    merged.update(dict(counts or {}))
    doc = build_manifest_document(
        run_id=run_id,
        source=source,
        prereg=env.prereg,
        psd_manifest=env.psd_manifest,
        counts=merged,
        calibrator=calibrator_identity(),
        repo_root=env.repo_root,
        h=h,
    )
    return SimpleNamespace(env=env, doc=doc, counts=merged, repo_root=env.repo_root)


def _apply_manifest_source_paths(doc: Dict[str, Any], repo_root: Path) -> Dict[str, Any]:
    """把 probe 用的绝对源路径改回仓库相对路径（manifest 只登记相对路径）。"""
    fixed = json.loads(json.dumps(doc))
    for key in ("base_chain_nav", "noise_chain_nav"):
        value = Path(fixed["source_pair"][key])
        if value.is_absolute():
            fixed["source_pair"][key] = value.relative_to(repo_root).as_posix()
    return fixed


def prepare_campaign(
    tmp_path: Path,
    *,
    campaign_id: str = "p5b_t4_campaign",
    counts: Optional[Mapping[str, Any]] = None,
) -> SimpleNamespace:
    """建 campaign 根并冻结 manifest（``apply=True``）。"""
    built = fast_manifest(tmp_path, counts=counts)
    doc = _apply_manifest_source_paths(built.doc, built.repo_root)
    root = Path(tmp_path) / "campaigns" / campaign_id
    root.mkdir(parents=True, exist_ok=True)
    frozen = freeze_calibration_manifest(root, doc, apply=True, allow_external_root=True)
    return SimpleNamespace(root=root, doc=doc, counts=built.counts, env=built.env, frozen=frozen)


def train(prepared: SimpleNamespace, **kwargs: Any) -> Dict[str, Any]:
    return train_calibration(
        prepared.root,
        repo_root=prepared.env.repo_root,
        apply=kwargs.pop("apply", True),
        allow_external_root=True,
        require_scale_floor=kwargs.pop("require_scale_floor", False),
        **kwargs,
    )


def holdout(prepared: SimpleNamespace, **kwargs: Any) -> Dict[str, Any]:
    return run_holdout(
        prepared.root,
        repo_root=prepared.env.repo_root,
        apply=kwargs.pop("apply", True),
        allow_external_root=True,
        require_scale_floor=kwargs.pop("require_scale_floor", False),
        **kwargs,
    )


def full_pipeline(
    tmp_path: Path,
    *,
    campaign_id: str = "p5b_t4_campaign",
    counts: Optional[Mapping[str, Any]] = None,
) -> SimpleNamespace:
    """冻结 → 训练 → holdout 全链路（小规模）。"""
    prepared = prepare_campaign(tmp_path, campaign_id=campaign_id, counts=counts)
    trained = train(prepared)
    held = holdout(prepared)
    return SimpleNamespace(prepared=prepared, train=trained, holdout=held)


# ---------------------------------------------------------------------------
# 合成配对账本（journal → 分片 → index 的真实写入口）
# ---------------------------------------------------------------------------


def synthetic_campaign(
    tmp_path: Path,
    *,
    nav: Mapping[str, Sequence[float]],
    dates: Sequence[str],
    campaign_id: str = "t4_ledger_campaign",
    omit_valuation: Optional[str] = None,
) -> Path:
    """按 journal/分片/index 真实写入口构造已发布账本（不驱动宿主）。

    Args:
        tmp_path: 临时根。
        nav: ``{arm_id: [D0..Dn 净值]}``（长度必须 = ``len(dates)``）。
        dates: 虚拟日序列（升序；第一日承担 D0 角色由调用方决定，这里逐日发布）。
        campaign_id: campaign 标识（目录名即 campaign_id）。
        omit_valuation: 该臂的分片省略 ``valuation`` 字段（构造拒绝用例）。
    """
    root = Path(tmp_path) / "campaigns" / campaign_id
    layout = layout_for(root)
    root.mkdir(parents=True, exist_ok=True)
    attempts: List[Dict[str, Any]] = []
    for index, date in enumerate(dates):
        generation = index + 1
        arms: Dict[str, Any] = {}
        for arm_id, series in nav.items():
            evidence: Dict[str, Any] = {
                "virtual_date": date,
                "driven_date": date,
                "driven": True,
                "run_kind": "t1",
                "t0_status": "driven",
                "complete_generation": True,
                "day_digest": f"digest-{arm_id}-{date}",
            }
            if arm_id != omit_valuation:
                evidence["valuation"] = {
                    "row_count": 1,
                    "nav": float(series[index]),
                    "date": date,
                }
            arms[arm_id] = {
                "arm_role": "aa_clone",
                "evidence": evidence,
                "staged": {"content_sha": f"content-{arm_id}-{date}"},
            }
        attempt = {
            "attempt_id": f"a-{date}",
            "family_txn_id": f"txn-{date}",
            "virtual_date": date,
            "target_generation": generation,
            "parent_generation": generation - 1,
            "published_generation": generation,
            "published_txn_id": f"txn-{date}",
            "status": "COMMITTED",
            "promoted_count": len(arms),
            "arms": arms,
        }
        doc = {
            "schema": "p5b_testbed_family_journal/v1",
            "campaign_id": campaign_id,
            "virtual_date": date,
            "attempt_seq": 1,
            "attempts": [attempt],
            "active_attempt_id": attempt["attempt_id"],
        }
        save_day_journal(layout, doc)
        for arm_id in arms:
            write_shard(layout, campaign_id, attempt, arm_id)
        attempts.append(attempt)
    write_index(layout, build_index(campaign_id, attempts))
    return root


def publish_extra_day(root: Path, *, campaign_id: str, date: str, nav: Mapping[str, float]) -> None:
    """追加一个只含指定臂的已发布日（构造「某臂缺该日」的缺口用例）。"""
    layout = layout_for(root)
    attempts = load_all_attempts(layout)
    generation = max(int(item["target_generation"]) for item in attempts) + 1
    arms = {
        arm_id: {
            "arm_role": "aa_clone",
            "evidence": {
                "virtual_date": date,
                "driven_date": date,
                "driven": True,
                "complete_generation": True,
                "day_digest": f"digest-{arm_id}-{date}",
                "valuation": {"row_count": 1, "nav": float(value), "date": date},
            },
            "staged": {"content_sha": f"content-{arm_id}-{date}"},
        }
        for arm_id, value in nav.items()
    }
    attempt = {
        "attempt_id": f"a-{date}",
        "family_txn_id": f"txn-{date}",
        "virtual_date": date,
        "target_generation": generation,
        "parent_generation": generation - 1,
        "published_generation": generation,
        "published_txn_id": f"txn-{date}",
        "status": "COMMITTED",
        "promoted_count": len(arms),
        "arms": arms,
    }
    doc = {
        "schema": "p5b_testbed_family_journal/v1",
        "campaign_id": campaign_id,
        "virtual_date": date,
        "attempt_seq": 1,
        "attempts": [attempt],
        "active_attempt_id": attempt["attempt_id"],
    }
    save_day_journal(layout, doc)
    for arm_id in arms:
        write_shard(layout, campaign_id, attempt, arm_id)
    write_index(layout, build_index(campaign_id, attempts + [attempt]))


def register_invalidated(root: Path, *, arm_id: str, date: str, campaign_id: str) -> None:
    """登记一条 INVALIDATED 终态（fail-closed 用例）。"""
    layout = layout_for(root)
    record_invalidations(
        layout,
        campaign_id,
        [
            LedgerGap(
                arm_id=arm_id,
                virtual_date=date,
                kind="bad_shard",
                detail="测试注入：已发布分片内容身份不符",
                repairable=False,
                generation=1,
            )
        ],
    )


def ledger_dates(count: int) -> List[str]:
    """连续虚拟日（P5b 形态 ``YYYYMMDD``）。"""
    return list(pd.bdate_range("2026-07-06", periods=int(count)).strftime("%Y%m%d"))


def linear_nav(start: float, daily_log: float, count: int) -> List[float]:
    """按恒定 log 收益构造净值序列（便于逐值核对 ``d_t``）。"""
    return [float(start * np.exp(daily_log * index)) for index in range(int(count))]


def iter_source_lines() -> Iterable[Tuple[str, str]]:
    """裁判子包逐行源码（源码级守卫用）。"""
    for name in REFEREE_MODULES:
        path = REFEREE_DIR / name
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            yield f"{name}:{number}", line
