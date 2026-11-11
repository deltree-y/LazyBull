# -*- coding: utf-8 -*-
"""P7 T0.5 预检工具的计数口径与规范化写出测试（不依赖真实生产数据）。"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "v2_p7" / "precheck_t05_inputs.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("p7_t05_precheck", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def precheck():
    return _load_module()


def test_guard_shorter_than_horizon_rejected(precheck):
    """guard 必须不小于持有期，否则相邻块会共享成熟窗口。"""
    with pytest.raises(ValueError):
        precheck._block_table(np.asarray([0], dtype=np.int64), 30, 10, guard_days=5)


def test_single_block_effective_when_matured(precheck):
    """块内入场且到期日在截止内 ⇒ 计 1 个有效块。"""
    table = precheck._block_table(np.asarray([0], dtype=np.int64), 30, 10)
    assert table["blocks_total"] == 1
    assert table["effective_blocks"] == 1
    assert table["events_right_censored"] == 0
    assert table["gate_pass"] is False  # 1 < 60


def test_guard_entry_not_assigned(precheck):
    """落在 guard 的入场日不归任何块，因此不产生有效块。"""
    table = precheck._block_table(np.asarray([25], dtype=np.int64), 40, 10)
    assert table["blocks_total"] == 1
    assert table["effective_blocks"] == 0
    assert table["events_in_guard_or_outside"] == 1


def test_right_censored_not_counted(precheck):
    """到期日超出截止 ⇒ 右删失单列，不记有效块、不补 0 收益。"""
    table = precheck._block_table(np.asarray([18], dtype=np.int64), 25, 10)
    assert table["effective_blocks"] == 0
    assert table["events_right_censored"] == 1


def test_blocks_separated_by_guard(precheck):
    """块与块之间固定间隔 20+10=30 个交易日。"""
    table = precheck._block_table(np.asarray([], dtype=np.int64), 95, 10)
    starts = [row["start_position"] for row in table["rows"]]
    assert starts == [0, 30, 60]
    assert table["blocks_per_year"] == round(250 / 30, 4)


def test_write_json_is_lf_and_hash_matches_disk(precheck, tmp_path):
    """规范化 JSON 必须二进制写 LF，且返回 SHA 与磁盘字节一致。"""
    target = tmp_path / "nested" / "sample.json"
    digest = precheck._write_json(target, {"b": 1, "a": "中文"})
    data = target.read_bytes()
    assert b"\r\n" not in data
    assert data.endswith(b"\n")
    assert digest == hashlib.sha256(data).hexdigest()
    assert json.loads(data.decode("utf-8")) == {"a": "中文", "b": 1}
