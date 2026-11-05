# -*- coding: utf-8 -*-
"""P6 T0.5 固定合成参考生成器（只验证生成计时协议，不冒充 P6 报告实现）。

设计 §7.2：生成时间 = 从原始稳定语料到 projection+HTML/CSV+manifest 的完整进程单调墙钟。
本脚本使用标准语料的驾驶舱/实验片段，走 Jinja2 自包含 HTML + CSV + manifest 全链；输出显式
标记 ``reference_only=true / not_a_p6_report=true``，不得作为 P6 验收产物引用。

所有输入取自显式 ``--corpus``，``generated_at`` 由调用方给定；输出目录必须不存在。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
if __name__ == "__main__":  # 验收进程禁写 __pycache__（设计 §2.5）；pytest 导入时不改全局
    sys.dont_write_bytecode = True
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.reporting import deps, validation  # noqa: E402
from src.lazybull.v2.store.store_identity import canonical_json_sha256  # noqa: E402

__all__ = ["ReferenceError", "generate", "main"]

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
TEMPLATE_NAME = "t05_reference.html.j2"
SCHEMA = "p6_t05_reference/v1"
MANIFEST_SCHEMA = "p6_t05_reference_manifest/v1"
MAX_SVG_POINTS = 120


class ReferenceError(RuntimeError):
    """参考生成器输入/输出不合法。"""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferenceError(f"读取 JSON 失败：{path}（{exc}）") from exc


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [dict(row) for row in reader]
            return list(reader.fieldnames or []), rows
    except OSError as exc:
        raise ReferenceError(f"读取 CSV 失败：{path}（{exc}）") from exc


def _polyline(nav_rows: list[dict[str, str]]) -> tuple[str, int]:
    values = [float(row["total_value"]) for row in nav_rows]
    if not values:
        return "", 0
    low, high = min(values), max(values)
    span = (high - low) or 1.0
    count = min(MAX_SVG_POINTS, len(values))
    if count == 1:
        indexes = [0]
    else:
        indexes = sorted({round(index * (len(values) - 1) / (count - 1)) for index in range(count)})
    points = []
    for position, index in enumerate(indexes):
        x = 0.0 if len(indexes) == 1 else 720.0 * position / (len(indexes) - 1)
        y = 170.0 - 160.0 * (values[index] - low) / span
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points), len(indexes)


def _quality_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, int]] = {}
    for row in rows:
        status = row.get("status", "")
        if status not in {"error", "warning"}:
            continue
        key = (row.get("layer", ""), row.get("dataset", ""))
        bucket = grouped.setdefault(key, {"abnormal": 0, "error": 0, "warning": 0})
        bucket["abnormal"] += 1
        bucket[status] += 1
    return [
        {"layer": layer, "dataset": dataset, **counts}
        for (layer, dataset), counts in sorted(grouped.items())
    ]


def _input_identity(corpus_root: Path, paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": validation.relative_posix(path, corpus_root),
            "sha256": validation.sha256_file(path),
        }
        for path in paths
    ]


def generate(corpus: Path | str, out: Path | str, *, generated_at: str) -> dict[str, Any]:
    """生成参考报告到新目录，返回 manifest。"""
    if len(generated_at) != 16 or generated_at[8] != "T" or not generated_at.endswith("Z"):
        raise ReferenceError(f"generated_at 必须为 UTC YYYYMMDDTHHMMSSZ：{generated_at!r}")
    if not generated_at[:8].isdigit() or not generated_at[9:15].isdigit():
        raise ReferenceError(f"generated_at 必须为 UTC YYYYMMDDTHHMMSSZ：{generated_at!r}")

    corpus_root = Path(corpus)
    profile_root = corpus_root / "standard"
    if not profile_root.is_dir():
        raise ReferenceError(f"语料缺少 standard/ 目录：{corpus_root}")

    target = Path(out)
    validation.ensure_new_dir(target)

    dashboard = profile_root / "dashboard"
    experiment_summary = profile_root / "experiment" / "runs" / "arm_a" / "summary.csv"
    positions = _read_json(dashboard / "positions.json")
    instructions_header, instructions = _read_csv(dashboard / "instructions.csv")
    nav_header, nav_rows = _read_csv(dashboard / "nav.csv")
    quality_header, quality_rows = _read_csv(dashboard / "quality_metrics.csv")
    summary_header, summary_rows = _read_csv(experiment_summary)
    manifest_path = corpus_root / "corpus_manifest.json"
    if not manifest_path.is_file():
        # 协议异常必须 fail-closed：静默空标识会进入 projection 与 HTML 且难以察觉（评审 R1-5）
        raise ReferenceError(f"语料清单缺失：{manifest_path}")
    corpus_manifest = _read_json(manifest_path)

    polyline, sampled = _polyline(nav_rows)
    quality = _quality_summary(quality_rows)
    abnormal_total = sum(item["abnormal"] for item in quality)
    cards = [
        {"label": "持仓数", "value": str(len(positions))},
        {"label": "当日指令数", "value": str(len(instructions))},
        {"label": "NAV 行数", "value": str(len(nav_rows))},
        {"label": "质量异常指标行", "value": str(abnormal_total)},
        {"label": "实验折数", "value": str(len(summary_rows))},
    ]

    projection: dict[str, Any] = {
        "schema": SCHEMA,
        "reference_only": True,
        "not_a_p6_report": True,
        "generated_at": generated_at,
        "corpus_content_sha256": corpus_manifest.get("content_sha256", ""),
        "counts": {
            "positions": len(positions),
            "instructions": len(instructions),
            "nav_rows": len(nav_rows),
            "quality_abnormal_rows": abnormal_total,
            "experiment_folds": len(summary_rows),
        },
        "cards": cards,
        "positions": positions,
        "instructions": [
            {
                key: row.get(key, "")
                for key in ("ts_code", "action", "shares", "price", "amount", "reason")
            }
            for row in instructions
        ],
        "nav": {
            "header": nav_header,
            "rows": nav_rows,
            "sampled_points": sampled,
        },
        "quality": quality,
        "quality_header": quality_header,
        "instructions_header": instructions_header,
    }

    template_path = TEMPLATE_DIR / TEMPLATE_NAME
    template_sha = validation.sha256_file(template_path)
    environment = deps.build_environment(template_dir=TEMPLATE_DIR)
    html = environment.get_template(TEMPLATE_NAME).render(
        title="LazyBull P6 参考报告（T0.5 计时协议）",
        generated_at=generated_at,
        corpus_content_sha256=projection["corpus_content_sha256"],
        cards=cards,
        positions=projection["positions"],
        instructions=projection["instructions"],
        nav_count=len(nav_rows),
        nav_polyline=polyline,
        nav_polyline_points=sampled,
        quality=quality,
    )

    html_path = target / "report.html"
    html_path.write_bytes(html.encode("utf-8"))
    projection_path = target / "projection.json"
    validation.write_json_new(projection_path, projection)

    csv_path = target / "arm_metrics_reference.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(summary_header)
        for row in summary_rows:
            writer.writerow([row.get(column, "") for column in summary_header])

    inputs = _input_identity(
        corpus_root,
        [
            dashboard / "positions.json",
            dashboard / "instructions.csv",
            dashboard / "nav.csv",
            dashboard / "quality_metrics.csv",
            experiment_summary,
            manifest_path,
        ],
    )
    outputs = {
        name: {"bytes": path.stat().st_size, "sha256": validation.sha256_file(path)}
        for name, path in (
            ("report.html", html_path),
            ("projection.json", projection_path),
            ("arm_metrics_reference.csv", csv_path),
        )
    }
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "reference_only": True,
        "not_a_p6_report": True,
        "generated_at": generated_at,
        "corpus_content_sha256": projection["corpus_content_sha256"],
        "template_sha256": template_sha,
        "jinja2_version": getattr(deps.load_jinja2(), "__version__", ""),
        "inputs": inputs,
        "outputs": outputs,
    }
    manifest["record_sha256"] = canonical_json_sha256(manifest)
    validation.write_json_new(target / "report_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 P6 T0.5 固定合成参考报告（非正式 P6 报告）")
    parser.add_argument("--corpus", required=True, help="语料根（须含 standard/）")
    parser.add_argument("--out", required=True, help="新的输出目录（已存在即失败）")
    parser.add_argument("--generated-at", required=True, help="UTC YYYYMMDDTHHMMSSZ（显式输入）")
    args = parser.parse_args(argv)
    try:
        manifest = generate(args.corpus, args.out, generated_at=args.generated_at)
    except (ReferenceError, validation.ValidationError, deps.ReportingDependencyError) as exc:
        print(f"[P6-T0.5] 参考生成失败：{exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {"record_sha256": manifest["record_sha256"], "out": str(args.out)}, ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
