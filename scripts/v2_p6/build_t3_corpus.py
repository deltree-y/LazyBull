# -*- coding: utf-8 -*-
"""P6 T3 合成语料构建（标准/压力两档实验面；确定性、不依赖真实生产数据）。

语料形态（每档）：

```text
<out>/<profile>/
├── runs/base/      # 规范化 runs 批次（契约全量：batch_meta/summary/chain_nav/folds）
├── runs/arm/
├── legacy/base/    # 旧 WF 三件（chain_nav_*.csv / walk_forward_summary_*.csv / data_state_*.json）
├── legacy/arm/
└── corpus_manifest.json
```

规模取 ``contracts.CORPUS_PROFILES[profile]["experiment"]``（链/逐日/成交/归因/TopK 行数）；
TopK 明细**不用于私算正日占比**，只为验证“必须被忽略”的降级路径。双跑一致性由
``content_sha256`` 与逐文件 SHA 保证。
"""

# 复杂度豁免：阶段专用语料脚本（scripts/v2_p6，不入生产链路，见 CLAUDE.md §7.5）

from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.reporting import contracts  # noqa: E402
from tests.p6_t3_env import DEFAULT_STRATEGY, write_legacy_run, write_runs_batch  # noqa: E402

__all__ = ["CorpusError", "build", "main"]

CORPUS_SCHEMA = "p6_t3_corpus/v1"


class CorpusError(RuntimeError):
    """语料构建参数不合法（未知 profile / 输出目录非空）。"""


def _digest_file(path: Path) -> str:
    """文件字节 SHA-256（语料身份，不含 mtime）。"""
    return sha256(path.read_bytes()).hexdigest()


def _write_arm_pair(
    profile_root: Path,
    counts: dict[str, int],
    *,
    legacy: bool,
    arm_id: str,
) -> None:
    """写一档的两条臂（runs 或 legacy）。"""
    suffix = "legacy" if legacy else "runs"
    target = profile_root / suffix / arm_id
    if legacy:
        write_legacy_run(
            target,
            run_id=f"wf_20260101_000000_{arm_id}",
            batch_run_id=f"t3_{arm_id}",
            batch_period_label=arm_id,
            folds=int(counts["folds"]),
            rows_per_fold=max(2, int(counts["chain_rows_per_arm"]) // int(counts["folds"])),
            topk=True,
        )
        return
    write_runs_batch(
        target,
        batch_id=f"t3_{arm_id}",
        # 期间身份：两臂代表**不同期间**（真实跨期正例）；同窗口重跑不得冒充两期（R3-04）
        config={**DEFAULT_STRATEGY, "batch_period_label": f"P_{arm_id}"},
        folds=int(counts["folds"]),
        total_chain_rows=int(counts["chain_rows_per_arm"]),
        trades_rows=int(counts["trade_rows_per_arm"]),
        attribution_rows=int(counts["attribution_rows_per_arm"]),
        topk_rows=int(counts["topk_detail_rows_per_arm"]) or None,
        with_topk=bool(counts["topk_detail_rows_per_arm"]),
        # 对照臂在中段注入更深回撤：使 ΔMaxDD 非零，成品侧方向复核具备判别力（A-13）
        nav_dip=0.03 if arm_id == "arm" else 0.0,
    )


def _manifest(profile_root: Path, profile: str) -> dict[str, Any]:
    """语料清单（逐文件 SHA + 内容哈希；双跑逐字节可复现）。"""
    files = []
    for path in sorted(profile_root.rglob("*")):
        if path.is_file() and path.name != "corpus_manifest.json":
            files.append(
                {
                    "path": path.relative_to(profile_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _digest_file(path),
                }
            )
    content = sha256(
        "".join(f"{item['path']}:{item['sha256']}\n" for item in files).encode("utf-8")
    ).hexdigest()
    return {"schema": CORPUS_SCHEMA, "profile": profile, "files": files, "content_sha256": content}


def build(out: Path | str, *, profile: str = "all") -> dict[str, Any]:
    """构建语料并返回总清单（``profiles.*.content_sha256`` 用于双跑一致）。"""
    target = Path(out)
    profiles = ["standard", "stress"] if profile == "all" else [profile]
    if any(name not in contracts.CORPUS_PROFILES for name in profiles):
        raise CorpusError(f"未知语料 profile：{profile!r}")
    if target.exists() and any(target.iterdir()):
        raise CorpusError(f"输出目录非空（拒绝覆盖既有语料）：{target}")
    target.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"schema": CORPUS_SCHEMA, "profiles": {}}
    for name in profiles:
        counts = contracts.CORPUS_PROFILES[name]["experiment"]
        profile_root = target / name
        profile_root.mkdir(parents=True, exist_ok=True)
        for arm_id, legacy in (("base", False), ("arm", False), ("base", True), ("arm", True)):
            _write_arm_pair(profile_root, counts, legacy=legacy, arm_id=arm_id)
        profile_manifest = _manifest(profile_root, name)
        (profile_root / "corpus_manifest.json").write_text(
            json.dumps(profile_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest["profiles"][name] = {
            "content_sha256": profile_manifest["content_sha256"],
            "file_count": len(profile_manifest["files"]),
        }
    (target / "corpus_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P6 T3 合成语料构建（标准/压力）")
    parser.add_argument("--out", required=True)
    parser.add_argument("--profile", choices=["all", "standard", "stress"], default="all")
    args = parser.parse_args(argv)
    try:
        manifest = build(args.out, profile=args.profile)
    except CorpusError as exc:
        print(f"[P6-T3] 语料未构建：{exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
