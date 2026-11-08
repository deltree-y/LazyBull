# -*- coding: utf-8 -*-
"""P6 T4 合成夹具：趋势语料、生成请求与合成索引行（**不引用真实生产数据**）。

覆盖点：

- ``build_corpus``：``scripts/v2_p6/build_t4_corpus.build`` 的 tiny 档（runs/旧 WF/纸面），
  满足 runs 契约全部硬校验；``upto`` 控制已物化 run 数（增量正例）；
- ``research_request`` / ``production_request``：显式请求（临时 reports 根，不写生产根）；
- ``synthetic_rows``：直接构造 N run × 5 指标的索引/趋势行（2,000 行截断与抽样用例，
  不读盘、秒级）；
- ``native_t5_bundle``：调用上游 P5b T5 原生管线产出 sealed bundle（calibration 正向依据，
  自造 JSON 只能作负例）。
"""

from __future__ import annotations

from datetime import date, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.v2_p6 import build_t4_corpus
from src.lazybull.v2.reporting import generate, trend_contract

__all__ = [
    "FIXED_AT",
    "build_corpus",
    "native_t5_bundle",
    "production_request",
    "read_csv_text",
    "research_request",
    "synthetic_rows",
]

#: 固定 generated_at（同输入字节一致；新墙钟即新 run）
FIXED_AT = "20261106T090000Z"


def build_corpus(root: Path, *, upto: int | None = None) -> dict[str, Any]:
    """构建 tiny 档语料（``upto`` 控制 runs 物化数，用于增量用例）。"""
    return build_t4_corpus.build(root, "tiny", upto=upto)


def research_request(
    reports_root: Path,
    corpus: Path,
    *,
    attempt: str,
    sources: Sequence[str] | None = None,
    previous: Path | None = None,
    generated_at: str = FIXED_AT,
    publish: bool = False,
) -> generate.TrendRequest:
    """research 面请求（默认收 runs + legacy 两个来源根）。"""
    material = tuple(sources) if sources is not None else (
        f"runs:{corpus / 'runs'}",
        f"legacy_wf:{corpus / 'legacy'}",
    )
    return generate.TrendRequest(
        reports_root=reports_root,
        evidence_class="research",
        generated_at=generated_at,
        attempt_id=attempt,
        input_evidence_classes=("research",),
        sources=material,
        previous=previous,
        publish_run=publish,
    )


def production_request(
    reports_root: Path,
    corpus: Path,
    *,
    attempt: str,
    previous: Path | None = None,
    generated_at: str = FIXED_AT,
    publish: bool = False,
) -> generate.TrendRequest:
    """production 面请求（只收纸面根）。"""
    return generate.TrendRequest(
        reports_root=reports_root,
        evidence_class="production",
        generated_at=generated_at,
        attempt_id=attempt,
        input_evidence_classes=("production",),
        paper_root=corpus / "paper",
        previous=previous,
        publish_run=publish,
    )


def synthetic_rows(count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """构造 ``count`` 个 run 的索引行与 5×count 条指标行（键集守恒、值有限）。"""
    index_rows: list[dict[str, Any]] = []
    trend_rows: list[dict[str, Any]] = []
    for number in range(count):
        run_key = sha256(f"run-{number}".encode("utf-8")).hexdigest()
        day = (date(2024, 1, 1) + timedelta(days=number)).strftime("%Y%m%d")
        index_rows.append(
            {
                "run_key": run_key,
                "run_id": f"run_{number:05d}",
                "batch_id": f"run_{number:05d}",
                "campaign_id": None,
                "evidence_class": "research",
                "host_mode": "backtest",
                "implementation": "runs",
                "window_start": day,
                "window_end": day,
                "date_semantics": "natural_day",
                "data_state_id": f"ds_{number % 7}",
                "code_state_id": "corpus",
                "source_ref": f"src0:run_{number:05d}",
                "source_identity_sha256": run_key,
                "source_manifest_sha256": run_key,
                "source_sha256": run_key,
                "index_mode": trend_contract.INDEX_MODE_FULL,
                "report_status": "available",
                "degradation_reason_code": None,
            }
        )
        for metric in trend_contract.METRICS:
            unit = dict(metric["unit_by_class"])["research"]
            trend_rows.append(
                {
                    "run_key": run_key,
                    "metric_name": metric["name"],
                    "value": 0.01 * (number % 11) if unit else None,
                    "unit": unit,
                    "window_start": day,
                    "window_end": day,
                    "date_semantics": "natural_day",
                    "source_ref": f"src0:run_{number:05d}/chain_nav.parquet",
                    "source_sha256": run_key,
                    "report_status": "available" if unit else "not_available",
                    "degradation_reason_code": None if unit else "no_source",
                    "source_field": f"corpus:{metric['name']}",
                }
            )
    return index_rows, trend_rows


def native_t5_bundle(base: Path, campaign_id: str = "p6_t4_native_t5") -> Path:
    """上游原生 P5b T5 管线产出 sealed bundle（返回 ``reports`` 目录）。"""
    from tests import p5b_t5_env

    p5b_t5_env.full_t5(base, campaign_id=campaign_id)
    matches = sorted(Path(base).glob(f"**/campaigns/{campaign_id}/reports"))
    if len(matches) != 1:
        raise AssertionError(f"合成 T5 bundle 定位失败：{matches}")
    return matches[0]


def read_csv_text(result: Mapping[str, Any], name: str) -> str:
    """取生成结果中的 CSV 文本（UTF-8）。"""
    return bytes(result["files"][name]).decode("utf-8")
