# -*- coding: utf-8 -*-
"""P5b T5 测试环境（tests/ 全量豁免复杂度；不是测试用例模块）。

提供三类**完全隔离**的合成现场（不读真实生产数据、不写真实试车场根）：

1. ``synthetic_repo``：在 tmp 里造「假仓库」——两份合成链式净值 CSV（B0/B0′ 形态，含可控回撤）、
   预登记与 PSD manifest 占位文件；
2. ``parent_campaign``：用**收益裁判库层**（未改动的 T4 实现）在同一假仓库上冻结并训练一份
   小规模父收益仪器（manifest + 临界值），供 T5 的上游链接核验；
3. ``t5_campaign`` / ``full_t5``：按 T5 真实入口冻结回撤 manifest → 训练 c^W/c^I →（可选）跑
   holdout + 功效，产物落 tmp campaign 目录。

另含：轨迹/生产根快照、T4 冻结件字节清单、方向 fixture 路径构造。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.lazybull.v2.hosts.testbed.referee import (  # noqa: E402
    build_manifest_document,
    calibrator_identity as income_calibrator_identity,
    freeze_calibration_manifest,
    load_source_pair as income_load_source_pair,
    source_pair_for_manifest as income_source_pair_for_manifest,
    train_calibration,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown import (  # noqa: E402
    DEFAULT_CHUNK,
    build_drawdown_manifest,
    calibrator_identity,
    freeze_drawdown_manifest,
    inspect_parent,
    lineage_section,
    load_source_window,
    prepare_source,
    run_drawdown_holdout,
    source_from_pair_document,
    train_drawdown_calibration,
)

#: 回撤子包目录（源码级守卫与身份改写用例）
DRAWDOWN_DIR = ROOT / "src/lazybull/v2/hosts/testbed/referee/drawdown"
#: 回撤子包模块（含未纳入身份的 report/__init__，供源码守卫遍历）
DRAWDOWN_MODULES: Tuple[str, ...] = (
    "mdd.py",
    "crash_window.py",
    "injection.py",
    "sampling.py",
    "realizations.py",
    "thresholds.py",
    "holdout.py",
    "power.py",
    "artifacts.py",
    "manifest.py",
    "sections.py",
    "report.py",
    "calibration.py",
    "__init__.py",
)
#: 合成源折数与每折行数（6×100 ⇒ 594 个收益，足够 500 窗口）
FOLDS = 6
ROWS = 100
#: 训练/web 源折
TRAIN_FOLDS: Tuple[int, ...] = (0, 1, 2, 3)
HOLDOUT_FOLDS: Tuple[int, ...] = (4, 5)
#: 小规模实现数（协议测试用；远低于下限，故必须 require_scale_floor=False）
SMALL_COUNTS: Mapping[str, Any] = {
    "train_per_unit": 120,
    "holdout_per_unit": 200,
    "effect_per_scene_per_level": 60,
    "seed": 20261103,
    "realization_prefix": "p5b_t5_test",
}
#: 父收益仪器的小规模实现数
PARENT_COUNTS: Mapping[str, Any] = {
    "train_per_unit": 200,
    "holdout_per_unit": 200,
    "effect_per_scene_per_level": 100,
    "seed": 20261103,
    "realization_prefix": "p5b_t4_parent",
}
#: 父收益仪器的下限实现数（脚本入口要求父仪器自身也过 §5.3 下限）
PARENT_FLOOR_COUNTS: Mapping[str, Any] = {
    "train_per_unit": 2000,
    "holdout_per_unit": 10000,
    "effect_per_scene_per_level": 2000,
    "seed": 20261103,
    "realization_prefix": "p5b_t4_parent_floor",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_state(root: Path) -> Dict[str, str]:
    """目录递归快照（相对路径 → sha256）。"""
    base = Path(root)
    if not base.exists():
        return {}
    return {
        path.relative_to(base).as_posix(): _sha256(path)
        for path in sorted(base.rglob("*"))
        if path.is_file()
    }


def production_snapshot() -> Dict[str, Any]:
    """生产根快照：``data/paper``、``data/runs`` 递归哈希 + ``data/models`` 元数据。"""
    models = ROOT / "data" / "models"
    metadata = {}
    if models.exists():
        for path in sorted(models.rglob("*")):
            if path.is_file():
                stat = path.stat()
                metadata[path.relative_to(models).as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return {
        "paper": tree_state(ROOT / "data" / "paper"),
        "runs": tree_state(ROOT / "data" / "runs"),
        "models": metadata,
    }


def t4_campaign_state() -> Dict[str, str]:
    """T4 campaign 目录递归字节快照（只含该目录）。"""
    return tree_state(ROOT / "data" / "paper_shadow" / "p5b_testbed" / "p5b_t4_referee_20261102")


def t4_repo_files_state() -> Dict[str, str]:
    """T4 仓库冻结件字节快照（只含两份文件）。"""
    files = {
        "manifest": ROOT / "docs/plans/v2/p5b_t4_calibration_manifest_20261102.json",
        "psd_r3": ROOT / "docs/plans/v2/p5b_calibration_manifest_psd_20261101_r3.json",
        "prereg": ROOT / "docs/plans/v2/p5b_income_referee_psd_prereg_draft_20261101.md",
    }
    return {name: _sha256(path) for name, path in files.items() if path.is_file()}


# ---------------------------------------------------------------------------
# 合成源（含可控回撤）
# ---------------------------------------------------------------------------


def _nav_frame(seed: int, drift: float, crash: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, 0.004, size=FOLDS * ROWS)
    if crash:
        steps[-60:-20] -= 0.005
        steps[-20:] += 0.002
    nav = np.cumprod(1.0 + steps)
    split_index = np.repeat(np.arange(FOLDS), ROWS)
    date = np.tile(np.arange(ROWS), FOLDS)
    return pd.DataFrame({"date": date, "nav": nav, "split_index": split_index})


def synthetic_repo(tmp_path: Path) -> SimpleNamespace:
    """构造假仓库（源对 CSV + 预登记占位 + PSD manifest 占位）。"""
    repo = Path(tmp_path) / "fake_repo"
    base_dir = repo / "data/walk_forward/batches/factor_ab_base/raw"
    noise_dir = repo / "data/walk_forward/batches/factor_ab_noise/raw"
    base_dir.mkdir(parents=True, exist_ok=True)
    noise_dir.mkdir(parents=True, exist_ok=True)
    base_path = base_dir / "chain_nav_base.csv"
    noise_path = noise_dir / "chain_nav_noise.csv"
    _nav_frame(11, 0.0006, True).to_csv(base_path, index=False)
    _nav_frame(12, 0.0003, True).to_csv(noise_path, index=False)
    docs = repo / "docs/plans/v2"
    docs.mkdir(parents=True, exist_ok=True)
    prereg = docs / "prereg_fake.md"
    prereg.write_text("# fake prereg\n", encoding="utf-8")
    psd = docs / "psd_manifest_fake.json"
    psd.write_text(json.dumps({"status": "frozen_fake"}), encoding="utf-8")
    return SimpleNamespace(
        repo_root=repo,
        base_path=base_path,
        noise_path=noise_path,
        prereg={"path": prereg.relative_to(repo).as_posix(), "sha256": _sha256(prereg)},
        psd_manifest={"path": psd.relative_to(repo).as_posix(), "sha256": _sha256(psd)},
    )


def _income_probe(env: SimpleNamespace) -> Dict[str, Any]:
    """构造收益裁判的 probe 文档（只为拿到 PreparedSource）。"""
    return {
        "source_pair": {
            "base_chain_nav": env.base_path.relative_to(env.repo_root).as_posix(),
            "noise_chain_nav": env.noise_path.relative_to(env.repo_root).as_posix(),
            "base_sha256": _sha256(env.base_path),
            "noise_sha256": _sha256(env.noise_path),
            "train_folds": list(TRAIN_FOLDS),
            "holdout_folds": list(HOLDOUT_FOLDS),
        },
        "files": {"prereg": env.prereg, "psd_manifest": env.psd_manifest},
        "income_referee": {"sample": {"H": 20, "n_max": 503}},
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
                "cross_pool_sampling_forbidden": True,
            },
        },
    }


def parent_campaign(
    tmp_path: Path,
    *,
    campaign_id: str = "t5_parent_income",
    counts: Optional[Mapping[str, Any]] = None,
) -> SimpleNamespace:
    """在假仓库里冻结并训练一份父收益仪器（manifest + 临界值）。"""
    env = synthetic_repo(tmp_path)
    probe = _income_probe(env)
    source = income_source_pair_for_manifest(probe, repo_root=env.repo_root)
    merged = dict(PARENT_COUNTS)
    merged.update(dict(counts or {}))
    doc = build_manifest_document(
        run_id="p5b_t4_parent_test",
        source=source,
        prereg=env.prereg,
        psd_manifest=env.psd_manifest,
        counts=merged,
        calibrator=income_calibrator_identity(),
        repo_root=env.repo_root,
    )
    # campaign 放在假仓库内：manifest 只登记仓库相对路径 ⇒ 同 seed 重跑逐字节可比
    root = env.repo_root / "campaigns" / campaign_id
    root.mkdir(parents=True, exist_ok=True)
    frozen = freeze_calibration_manifest(root, doc, apply=True, allow_external_root=True)
    trained = train_calibration(
        root,
        repo_root=env.repo_root,
        apply=True,
        allow_external_root=True,
        require_scale_floor=False,
        chunk=DEFAULT_CHUNK,
    )
    return SimpleNamespace(
        env=env,
        root=root,
        manifest_path=root / "reports/calibration_manifest.json",
        thresholds_path=root / "reports/calibration_thresholds.json",
        manifest_sha256=frozen["sha256"],
        frozen=frozen,
        trained=trained,
    )


def source_window(env: SimpleNamespace) -> Any:
    """假仓库 B0 的冻结窗口（最后 500 个收益样本）。"""
    return load_source_window(env.base_path, window=500)


def t5_document(
    parent: SimpleNamespace,
    *,
    run_id: str = "p5b_t5_test_drawdown",
    counts: Optional[Mapping[str, Any]] = None,
    seed: Optional[int] = None,
    prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """按父仪器 + 假仓库构造 T5 manifest 文档（未冻结）。"""
    info = inspect_parent(
        parent.manifest_path,
        parent.thresholds_path,
        require_scale_floor=False,
    )
    source = source_from_pair_document(
        info["parent_source_pair"],
        dgp_doc=info["parent_calibration"],
        repo_root=parent.env.repo_root,
        h=20,
        window_samples=500,
    )
    merged = dict(SMALL_COUNTS)
    merged.update(dict(counts or {}))
    if seed is not None:
        merged["seed"] = int(seed)
    if prefix is not None:
        merged["realization_prefix"] = str(prefix)
    return build_drawdown_manifest(
        run_id=run_id,
        source=source,
        windows=source.windows,
        counts=merged,
        upstream=info,
        calibrator=calibrator_identity(),
        repo_root=parent.env.repo_root,
        lineage=lineage_section(),
    )


def t5_campaign(
    parent: SimpleNamespace,
    *,
    campaign_id: str = "t5_drawdown",
    counts: Optional[Mapping[str, Any]] = None,
    seed: Optional[int] = None,
    prefix: Optional[str] = None,
    train: bool = True,
) -> SimpleNamespace:
    """建 campaign 根 → 冻结 T5 manifest →（可选）训练临界值。"""
    doc = t5_document(parent, counts=counts, seed=seed, prefix=prefix)
    root = Path(parent.env.repo_root) / "campaigns" / campaign_id
    root.mkdir(parents=True, exist_ok=True)
    frozen = freeze_drawdown_manifest(root, doc, apply=True, allow_external_root=True)
    trained = None
    if train:
        trained = train_drawdown_calibration(
            root,
            repo_root=parent.env.repo_root,
            apply=True,
            allow_external_root=True,
            require_scale_floor=False,
            chunk=DEFAULT_CHUNK,
        )
    return SimpleNamespace(
        parent=parent,
        root=root,
        doc=doc,
        frozen=frozen,
        trained=trained,
    )


def run_holdout(
    campaign: SimpleNamespace, *, apply: bool = True, chunk: Optional[int] = DEFAULT_CHUNK
) -> Dict[str, Any]:
    """跑 T5 holdout + 功效（小规模）。"""
    return run_drawdown_holdout(
        campaign.root,
        repo_root=campaign.parent.env.repo_root,
        apply=apply,
        allow_external_root=True,
        chunk=chunk,
        require_scale_floor=False,
    )


def full_t5(
    tmp_path: Path,
    *,
    campaign_id: str = "t5_drawdown",
    counts: Optional[Mapping[str, Any]] = None,
    seed: Optional[int] = None,
) -> SimpleNamespace:
    """父仪器 → T5 冻结 → 训练 → holdout 全链路（小规模）。"""
    parent = parent_campaign(tmp_path)
    campaign = t5_campaign(parent, campaign_id=campaign_id, counts=counts, seed=seed)
    outcome = run_holdout(campaign)
    return SimpleNamespace(parent=parent, campaign=campaign, holdout=outcome)


def prepared(campaign: SimpleNamespace) -> Any:
    """按已冻结 manifest 取 PreparedSource。"""
    return prepare_source(
        campaign.doc, repo_root=campaign.parent.env.repo_root, require_scale_floor=False
    )


# ---------------------------------------------------------------------------
# 确定性 fixture 路径（方向自检）
# ---------------------------------------------------------------------------


def flat_path(drawdown_pp: float, decline_days: int = 100, days: int = 500) -> np.ndarray:
    """构造 MDD 恰为 ``drawdown_pp`` 的路径（前 ``decline_days`` 日等速下跌后走平）。

    所有 look（≥``decline_days``）上的 MDD 都相同，便于逐 look 核对 ``D_k``。
    """
    total = float(np.log(1.0 + float(drawdown_pp) / 100.0))
    path = np.zeros(int(days), dtype=float)
    path[: int(decline_days)] = total / float(decline_days)
    return path


def iter_source_lines() -> Any:
    """回撤子包逐行源码（源码级守卫用）。"""
    for name in DRAWDOWN_MODULES:
        path = DRAWDOWN_DIR / name
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            yield f"{name}:{number}", line


def copy_modules(target: Path, *, names: Sequence[str]) -> Path:
    """把回撤子包模块复制到目标目录（身份改写用例）。"""
    import shutil

    directory = Path(target)
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copyfile(DRAWDOWN_DIR / name, directory / name)
    return directory


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def module_list() -> List[str]:
    """身份覆盖的模块清单（用于断言 `files.calibrator.modules` 完整性）。"""
    from src.lazybull.v2.hosts.testbed.referee.drawdown.manifest import CALIBRATOR_MODULES

    return list(CALIBRATOR_MODULES)
