# -*- coding: utf-8 -*-
"""代码态身份、输入清单与旧 A 锚核验（协议 §2.3、§4、§9）。

身份口径
========
- **代码态**：git commit + 是否脏 + 脏改动内容哈希（逐文件 SHA256，忽略被 gitignore
  的数据目录）。只作登记，不作为放行条件。
- **输入身份**：小文件/关键文件做**整字节 SHA256**；百万级分区集（cs_train 24.7 GB）
  做 ``name_size_v1``（逐分区 ``名称\\t字节数`` 的有序摘要）+ 探针日/首末分区的
  **抽样整字节 SHA**。抽样是显式声明的降级，``full_sha_status`` 写明延后到 run-a
  的理由与预估成本，禁止把抽样摘要当整集身份。
- **旧 A 锚**：manifest 整字节 SHA 必须等于协议 §4 冻结值；每件实际消费文件都登记
  SHA 并复核结构（1,839 净值点、锚日全现金、A80+现金恒等式），最后用唯一指标实现
  ``calculate_chain_metrics`` 复算已发表参照读数并登记差值。
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from scripts.v2_p7.cashflow_sleeve_split import spec
from scripts.v2_p7.common import REPO_ROOT, sha256_file

#: 抽样整字节 SHA 的分区数上限（首/末/探针日；其余只进 name_size 摘要）
MAX_SAMPLED_PARTITIONS = 24


class IdentityError(RuntimeError):
    """身份不符、缺件或结构不自洽（fail-closed）。"""


def _run_git(args: Sequence[str], root: Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise IdentityError(f"git {' '.join(args)} 失败：{completed.stderr.strip()}")
    # 只去尾部换行：`status --porcelain` 行首的空格是状态位，strip() 会把首行吃掉一列
    return completed.stdout.rstrip("\r\n")


def git_identity(root: Path | None = None) -> Dict[str, Any]:
    """代码提交与脏改动内容哈希（失败只降级为 unknown，不阻断）。"""
    base = Path(root) if root is not None else REPO_ROOT
    try:
        commit = _run_git(["rev-parse", "HEAD"], base)
        branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], base)
        porcelain = _run_git(["status", "--porcelain"], base)
    except (IdentityError, OSError) as error:
        return {"status": "unknown", "reason": str(error)}
    changed = [line[3:].strip() for line in porcelain.splitlines() if line.strip()]
    inventory = _code_inventory(base, changed)
    digest = ""
    if inventory:
        import hashlib

        lines = [f"{item['path']}:{item['sha256']}" for item in inventory]
        digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    return {
        "status": "ok",
        "commit": commit,
        "branch": branch,
        "dirty": bool(changed),
        "changed_count": len(changed),
        "hashed_count": len(inventory),
        "dirty_content_sha256": digest,
        "changed_paths_sample": changed[:40],
        "inventory": inventory,
        "inventory_note": (
            "未跟踪目录递归列举到文件层（复审第 6 项）；" "每项含 path/size_bytes/sha256"
        ),
    }


#: 代码可还原清单只纳入这些后缀（源码/测试/配置/文档）
_CODE_SUFFIXES = frozenset(
    {".py", ".md", ".toml", ".yaml", ".yml", ".json", ".cfg", ".ini", ".ps1", ".txt"}
)


def _code_inventory(base: Path, changed: Sequence[str]) -> List[Dict[str, Any]]:
    """把 git status 的变动路径递归展开成逐文件 SHA 清单。"""
    inventory: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for relative in changed[:2000]:
        path = base / relative
        if path.is_dir():
            files = sorted(p for p in path.rglob("*") if p.is_file())
        elif path.is_file():
            files = [path]
        else:
            files = []
        for item in files:
            relative_path = item.relative_to(base).as_posix()
            if relative_path in seen:
                continue
            if item.suffix.lower() not in _CODE_SUFFIXES:
                continue
            seen.add(relative_path)
            inventory.append(
                {
                    "path": relative_path,
                    "size_bytes": item.stat().st_size,
                    "sha256": sha256_file(item),
                }
            )
    return inventory


def code_restore_payload(root: Path | None = None) -> Dict[str, Any]:
    """可还原代码清单（逐文件 SHA + tracked 补丁文本的 SHA）。"""
    base = Path(root) if root is not None else REPO_ROOT
    identity = git_identity(base)
    patch_sha = None
    patch_bytes = 0
    try:
        patch = _run_git(["diff", "--binary", "--no-color"], base)
        patch_bytes = len(patch.encode("utf-8"))
        import hashlib

        patch_sha = hashlib.sha256(patch.encode("utf-8")).hexdigest()
    except (IdentityError, OSError):
        patch_sha = None
    return {
        "schema": "p7_cashflow_split_code_restore/v1",
        "commit": identity.get("commit"),
        "branch": identity.get("branch"),
        "dirty_content_sha256": identity.get("dirty_content_sha256"),
        "files": identity.get("inventory") or [],
        "files_n": len(identity.get("inventory") or []),
        "tracked_patch_sha256": patch_sha,
        "tracked_patch_bytes": patch_bytes,
        "restore_recipe": (
            "git checkout <commit>；按 files[] 逐文件校验 sha256；"
            "tracked 改动由 tracked_patch（git diff）复原，未跟踪文件由 files[] 清单复原"
        ),
    }


# ── 分区摘要 ───────────────────────────────────────────────────────────


def _partition_files(base: Path) -> List[Path]:
    if base.is_dir():
        return sorted(p for p in base.rglob("*.parquet") if p.is_file())
    if base.is_file():
        return [base]
    return []


def partition_digest(base: Path, sampled: Iterable[Path] = ()) -> Dict[str, Any]:
    """分区集身份：``name_size_v1`` + 抽样整字节 SHA（显式降级口径）。

    ``partition_units`` = 顶层分区单元数（日/月/季目录或单文件分区），
    ``files`` = 实际 parquet 文件数（列组存储下二者不同，禁止混读）。
    """
    files = _partition_files(base)
    units = _partition_units(base)
    if not files:
        return {
            "exists": base.exists(),
            "partitions": 0,
            "partition_units": 0,
            "files": 0,
            "digest_kind": "name_size_v1",
            "digest": "",
            "sampled": {},
            "full_sha_status": "empty",
        }
    import hashlib

    lines = []
    total = 0
    for path in files:
        size = path.stat().st_size
        total += size
        lines.append(f"{path.relative_to(base).as_posix() if base.is_dir() else path.name}\t{size}")
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    sampled_paths = [Path(p) for p in sampled if Path(p).exists()]
    sampled_sha = {
        path.name: sha256_file(path) for path in sorted(set(sampled_paths))[:MAX_SAMPLED_PARTITIONS]
    }
    return {
        "exists": True,
        "partitions": len(units),
        "partition_units": len(units),
        "files": len(files),
        "first": units[0] if units else files[0].name,
        "last": units[-1] if units else files[-1].name,
        "total_bytes": total,
        "digest_kind": "name_size_v1",
        "digest": digest,
        "sampled": sampled_sha,
        "full_sha_status": "sampled_only",
    }


def _partition_units(base: Path) -> List[str]:
    """顶层分区单元名（目录名或单文件分区名），升序。"""
    if not base.is_dir():
        return [base.name] if base.exists() else []
    units = [
        path.name
        for path in base.iterdir()
        if path.is_dir() or (path.is_file() and path.suffix == ".parquet")
    ]
    return sorted(units)


def daily_partition(base: Path, day: str) -> Path:
    """``YYYYMMDD`` → 分区路径（兼容 ``YYYY-MM-DD.parquet`` 与 ``YYYYMMDD.parquet``）。"""
    dashed = f"{day[:4]}-{day[4:6]}-{day[6:]}"
    for name in (f"{dashed}.parquet", f"{day}.parquet"):
        candidate = base / name
        if candidate.exists():
            return candidate
    return base / f"{dashed}.parquet"


_PARTITION_KINDS = frozenset(
    {"partition_daily", "partition_month", "partition_quarter", "partition_year"}
)


def _source_record(
    item: Mapping[str, Any],
    base: Path,
    probe_dates: Sequence[str],
    a_baseline: Mapping[str, Any] | None,
    gaps: List[str],
) -> Dict[str, Any]:
    """单个源的身份记录（按 ``kind`` 分派；嵌套已拆平以守复杂度上限）。"""
    relative = str(item["relative"])
    path = base / relative
    kind = str(item["kind"])
    record: Dict[str, Any] = {
        "name": item["name"],
        "relative": relative,
        "kind": kind,
        "role": item.get("role"),
        "exists": path.exists(),
    }
    if not path.exists():
        gaps.append(f"{item['name']} 缺件：{relative}")
        record["status"] = "missing"
        return record
    if kind == "file":
        record.update(
            {
                "size_bytes": path.stat().st_size,
                "digest_kind": "file_sha256",
                "digest": sha256_file(path),
                "status": "ok",
            }
        )
        return record
    if kind in _PARTITION_KINDS:
        sampled = [daily_partition(path, day) for day in probe_dates]
        record.update(partition_digest(path, sampled))
        record["status"] = "ok" if record["partitions"] else "empty"
        if not record["partitions"]:
            gaps.append(f"{item['name']} 无分区：{relative}")
        return record
    if kind == "model_artifacts":
        record.update(_model_artifacts(path, gaps))
        return record
    if kind != "file_tree":
        record["status"] = "unsupported_kind"
        return record
    if item["name"] == "a_baseline" and a_baseline and a_baseline.get("consumed_files"):
        # 已由 verify_a_baseline 逐件核验过：直接复用，避免二次哈希整棵树
        record.update(
            {
                "files": dict(a_baseline["consumed_files"]),
                "count": len(a_baseline["consumed_files"]),
                "digest_kind": "consumed_files_only",
                "digest": a_baseline.get("manifest_sha256"),
                "status": "ok",
                "note": "只登记实际消费文件；整树身份见 a_baseline 段",
            }
        )
        return record
    record.update(_tree_record(path, gaps))
    return record


def build_input_manifest(
    root: Path | None = None,
    probe_dates: Sequence[str] = (),
    a_baseline: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """构建登记用输入清单（不校验冻结值；门在 :func:`verify_a_baseline`）。"""
    base = Path(root) if root is not None else REPO_ROOT
    sources: List[Dict[str, Any]] = []
    gaps: List[str] = []
    for item in spec.SOURCES:
        sources.append(_source_record(item, base, probe_dates, a_baseline, gaps))
    baseline = dict(a_baseline) if a_baseline else None
    return {
        "schema": "p7_cashflow_split_input_manifest/v1",
        "candidate_id": spec.CANDIDATE_ID,
        "generated_at": _now(),
        "protocol_sha256": spec.protocol_sha256(base),
        "spec_sha256": spec.spec_sha256(),
        "sources": sources,
        "a_baseline": baseline,
        "gaps": gaps,
    }


def _now() -> str:
    from scripts.v2_p7.cashflow_sleeve_split.artifacts import utc_now

    return utc_now()


def _model_artifacts(root: Path, gaps: List[str]) -> Dict[str, Any]:
    """逐版本登记模型三件。

    ``features.json`` / ``metadata.json`` 做整字节 SHA（列路由与配置取证的唯一依据）；
    ``model.joblib`` 只有体积登记，SHA 延后到 run-a（本阶段不训练、不评分、不消费
    模型字节；15 件合计约 1.5~4 GB，此阶段无用途）。
    """
    versions = list(spec.FOLD_VERSIONS) + [spec.DEPLOY_VERSION]
    rows: Dict[str, Any] = {}
    for version in versions:
        entry: Dict[str, Any] = {}
        for suffix in ("_features.json", "_metadata.json", "_model.joblib"):
            path = root / f"v{version}{suffix}"
            if not path.exists():
                entry[suffix] = None
                gaps.append(f"模型件缺件：{path}")
                continue
            if suffix == "_model.joblib":
                entry[suffix] = {
                    "size_bytes": path.stat().st_size,
                    "sha256": None,
                    "sha256_status": "deferred_to_run_a",
                    "reason": "prepare 阶段不消费模型字节",
                }
                continue
            entry[suffix] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        rows[str(version)] = entry
    return {"versions": rows, "count": len(rows), "status": "ok"}


def _tree_record(root: Path, gaps: List[str]) -> Dict[str, Any]:
    """目录树整字节 SHA（只用于中小型产物树：旧 A 账户）。"""
    files = sorted(p for p in root.rglob("*") if p.is_file())
    rows = {
        p.relative_to(root).as_posix(): {"size_bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in files
    }
    return {"files": rows, "count": len(rows), "status": "ok" if rows else "empty"}


# ── 旧 A 锚 ───────────────────────────────────────────────────────────


def verify_a_baseline(root: Path | None = None) -> Dict[str, Any]:
    """核验冻结旧 A 锚：manifest 字节、逐件消费文件、账户日历与已发表读数。"""
    base = Path(root) if root is not None else REPO_ROOT
    anchor = base / spec.A_BASELINE["relative"]
    manifest_path = anchor / "manifest.json"
    if not manifest_path.exists():
        raise IdentityError(f"旧 A 锚缺 manifest：{manifest_path}")
    manifest_sha = sha256_file(manifest_path)
    expected = str(spec.A_BASELINE["expected_manifest_sha256"])
    if manifest_sha != expected:
        raise IdentityError(
            f"旧 A 锚 manifest 字节不符（协议 §4 冻结值）：{manifest_sha} != {expected}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen = _verify_frozen_files(anchor, manifest.get("files") or {})

    calendar = _window_calendar(base)
    files: Dict[str, Any] = {}
    checks: Dict[str, Any] = {}
    reference: Dict[str, Any] = {}
    for capital, harness in spec.A_BASELINE["harness"].items():
        a100 = anchor / harness["a100_daily"]
        a80 = anchor / harness["a80_daily"]
        a80_cash = anchor / harness["a80_cash_daily"]
        for path in (a100, a80, a80_cash):
            if not path.exists():
                raise IdentityError(f"旧 A 锚消费文件缺件：{path}")
            files[path.relative_to(anchor).as_posix()] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        block = _verify_account_block(a100, a80, a80_cash, float(capital), calendar)
        checks[capital] = block
        reference[capital] = block["recomputed"]
        for detail in spec.A_BASELINE["a80_account_detail"]:
            path = anchor / capital / detail
            if not path.exists():
                raise IdentityError(f"旧 A 锚账户明细缺件（不得静默略过）：{path}")
            files[f"{capital}/{detail}"] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

    mismatches = _reference_deltas(reference)
    return {
        "relative": spec.A_BASELINE["relative"],
        "manifest_sha256": manifest_sha,
        "manifest_expected": expected,
        "manifest_keys": sorted(manifest.keys()) if isinstance(manifest, dict) else [],
        "frozen_files": frozen,
        "consumed_files": files,
        "account_checks": checks,
        "recomputed_reference": reference,
        "published_reference": spec.A_BASELINE["published_reference"],
        "reference_delta": mismatches,
        "status": "pass",
    }


def _verify_frozen_files(anchor: Path, frozen: Mapping[str, Any]) -> Dict[str, Any]:
    """按冻结清单逐件验 SHA（复审第 6 项：缺件/改一字节都必须失败）。"""
    if not frozen:
        raise IdentityError(f"旧 A 锚 manifest 未登记 files 清单：{anchor}")
    missing: List[str] = []
    mismatched: List[Dict[str, str]] = []
    verified = 0
    for relative, expected in frozen.items():
        expected_sha = expected if isinstance(expected, str) else expected.get("sha256")
        path = anchor / str(relative)
        if not path.exists():
            missing.append(str(relative))
            continue
        actual = sha256_file(path)
        if expected_sha and actual != expected_sha:
            mismatched.append(
                {"relative": str(relative), "expected": str(expected_sha), "actual": actual}
            )
            continue
        verified += 1
    if missing or mismatched:
        raise IdentityError(
            f"旧 A 锚冻结清单未通过：缺件 {missing[:5]}（{len(missing)} 件）；"
            f"SHA 不符 {mismatched[:3]}（{len(mismatched)} 件）"
        )
    return {
        "manifest_files_n": len(frozen),
        "verified_n": verified,
        "missing": missing,
        "mismatched": mismatched,
        "status": "ok",
    }


def _window_calendar(root: Path) -> List[str]:
    """账户日历：锚日 + 窗口内全部开市日（协议 §4）。"""
    import pandas as pd

    path = root / "data/clean/trade_cal.parquet"
    if not path.exists():
        raise IdentityError(f"交易日历缺件：{path}")
    cal = pd.read_parquet(path, columns=["cal_date", "is_open"])
    opened = cal.loc[cal["is_open"] == 1, "cal_date"].astype(str)
    window = sorted(day for day in opened if spec.WINDOW["start"] <= day <= spec.WINDOW["end"])
    return [spec.WINDOW["cash_anchor"], *window]


def _verify_account_block(
    a100_path: Path, a80_path: Path, a80_cash_path: Path, capital: float, calendar: Sequence[str]
) -> Dict[str, Any]:
    """单个本金档的会计恒等式与日历核验。"""
    import pandas as pd

    a100 = pd.read_parquet(a100_path)
    a80 = pd.DataFrame(json.loads(a80_path.read_text(encoding="utf-8")))
    a80_cash = pd.read_parquet(a80_cash_path)
    for name, frame in (("A100", a100), ("A80", a80), ("A80_CASH", a80_cash)):
        if "total_value" not in frame.columns:
            raise IdentityError(f"{a100_path.parent.name} {name} 缺 total_value 列")
    dates = [str(d) for d in a100["trade_date"].tolist()]
    if dates != list(calendar):
        raise IdentityError(
            f"旧 A 账户日历与交易日历不一致（本金 {capital:.0f}）："
            f"len={len(dates)} vs {len(calendar)}，首末={dates[0]}~{dates[-1]}"
        )
    if len(dates) != int(spec.WINDOW["nav_points"]):
        raise IdentityError(f"净值点数不符（协议 §4 = 1839）：{len(dates)}")
    anchor_row = a100.iloc[0]
    if (
        abs(float(anchor_row["nav"]) - 1.0) > 1e-12
        or abs(float(anchor_row["total_value"]) - capital) > 1e-6
    ):
        raise IdentityError(f"锚日必须 100% 现金且 nav=1：{anchor_row.to_dict()}")
    expected_cash = capital * float(spec.ACCOUNTS["legs"]["B20"])
    residual = (
        a80_cash["total_value"].astype(float).to_numpy()
        - a80["total_value"].astype(float).to_numpy()
        - expected_cash
    )
    max_residual = float(abs(residual).max())
    if max_residual > 1e-6:
        raise IdentityError(f"A80+0.2K 现金恒等式破缺：max|residual|={max_residual}")

    metrics = _chain_metrics(a100[["trade_date", "nav"]].copy())
    return {
        "capital": capital,
        "rows": len(dates),
        "first": dates[0],
        "last": dates[-1],
        "a80_cash_identity_max_abs_residual": max_residual,
        "cash_leg_yuan": expected_cash,
        "recomputed": metrics,
        "ending_total_value": float(a100["total_value"].iloc[-1]),
    }


def _chain_metrics(daily: Any) -> Dict[str, Any]:
    """调用唯一指标实现（协议 §7.1；禁止本包自建年化公式）。"""
    from src.lazybull.v2.common.chain_metrics import calculate_chain_metrics

    frame = daily.copy()
    frame["nav"] = frame["nav"].astype(float)
    raw = calculate_chain_metrics(
        frame,
        annual_trading_days=int(spec.CHAIN_METRICS["trading_days_per_year"]),
        annual_risk_free_rate=float(spec.CHAIN_METRICS["risk_free_rate"]),
    )
    mdd = raw.get("max_drawdown")
    return {
        "cagr": raw.get("cagr"),
        "max_drawdown": mdd,
        "mdd_loss": (-float(mdd)) if mdd is not None else None,
        "sharpe": raw.get("sharpe"),
        "total_return": raw.get("total_return"),
        "trading_days": raw.get("trading_days"),
    }


def _reference_deltas(reference: Mapping[str, Any]) -> Dict[str, Any]:
    """复算读数与协议 §4 已发表参照的差值（只查错，不设通过门）。"""
    out: Dict[str, Any] = {}
    for capital, published in spec.A_BASELINE["published_reference"].items():
        actual = reference.get(capital) or {}
        cagr = actual.get("cagr")
        mdd_loss = actual.get("mdd_loss")
        out[capital] = {
            "cagr_delta": (None if cagr is None else float(cagr) - float(published["cagr"])),
            "mdd_loss_delta": (
                None if mdd_loss is None else float(mdd_loss) - float(published["mdd_loss"])
            ),
            "status": (
                "match_within_1e-4"
                if (
                    cagr is not None
                    and mdd_loss is not None
                    and abs(float(cagr) - float(published["cagr"])) <= 1e-4
                    and abs(float(mdd_loss) - float(published["mdd_loss"])) <= 1e-4
                )
                else "mismatch_register"
            ),
        }
    return out


def measure_hash_throughput(paths: Sequence[Path]) -> Dict[str, Any]:
    """实测整字节 SHA 吞吐（用于给 run-a 的全集哈希报价）。"""
    started = time.perf_counter()
    total = 0
    count = 0
    for path in paths:
        if not Path(path).is_file():
            continue
        total += Path(path).stat().st_size
        sha256_file(Path(path))
        count += 1
    elapsed = max(time.perf_counter() - started, 1e-9)
    return {
        "files": count,
        "bytes": total,
        "seconds": round(elapsed, 3),
        "mb_per_second": round(total / 1e6 / elapsed, 1),
    }


def verify_registered_manifest(
    registered: Mapping[str, Any], current: Mapping[str, Any]
) -> Dict[str, Any]:
    """把当前输入身份与已登记清单逐源比对（run-a 起每次复用必做）。"""
    index = {str(item["name"]): item for item in current.get("sources") or []}
    mismatched: List[str] = []
    missing: List[str] = []
    for item in registered.get("sources") or []:
        name = str(item["name"])
        now = index.get(name)
        if now is None:
            missing.append(name)
            continue
        if item.get("digest") and now.get("digest") != item.get("digest"):
            mismatched.append(name)
    return {
        "checked": len(registered.get("sources") or []),
        "missing": missing,
        "mismatched": mismatched,
        "status": "match" if not missing and not mismatched else "drift",
    }


def sources_by_name(manifest: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(item["name"]): dict(item) for item in manifest.get("sources") or []}


__all__ = [
    "IdentityError",
    "build_input_manifest",
    "code_restore_payload",
    "daily_partition",
    "git_identity",
    "measure_hash_throughput",
    "partition_digest",
    "sources_by_name",
    "verify_a_baseline",
    "verify_registered_manifest",
]
