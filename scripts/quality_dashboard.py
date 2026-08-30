"""生成全历史数据质量静态报告。"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lazybull.common.config import Config  # noqa: E402
from src.lazybull.data.storage import Storage  # noqa: E402
from src.lazybull.quality.report import compare_snapshots, write_html_report  # noqa: E402
from src.lazybull.quality.scanner import evaluate_quality, save_snapshot, scan_quality  # noqa: E402


def main() -> int:
    """执行质量扫描并返回适合任务调度消费的退出码。"""
    parser = argparse.ArgumentParser(description="生成 LazyBull 数据质量静态报告")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    try:
        config = Config()
        quality_config = config.get("quality", {})
        storage = Storage(args.data_root)
        output_dir = Path(
            args.output_dir or quality_config.get("output_dir", "data/reports/quality")
        )
        metrics = evaluate_quality(
            scan_quality(storage, quality_config, args.start_date, args.end_date), quality_config
        )
        snapshot_path = output_dir / "latest_metrics.parquet"
        previous = pd.read_parquet(snapshot_path) if snapshot_path.exists() else metrics.iloc[0:0]
        changes = compare_snapshots(metrics, previous)
        save_snapshot(metrics, snapshot_path)
        write_html_report(
            metrics,
            changes,
            output_dir / "quality_dashboard.html",
            max_detail_rows=int(quality_config.get("html_max_detail_rows", 100)),
        )
        error_count = int((metrics["status"] == "error").sum())
        logger.info(f"数据质量报告已生成: {output_dir}，错误数: {error_count}")
        return 1 if error_count else 0
    except Exception as exc:
        logger.exception(f"生成数据质量报告失败: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
