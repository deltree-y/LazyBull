"""读数前的预登记/源包身份失败关闭。"""

import hashlib
import json

import pytest

from scripts.v2_p7 import run_operating_monthly as runner


def test_mutated_prereg_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    spec = tmp_path / "rule.md"
    spec.write_text("冻结规则", encoding="utf-8")
    prereg = tmp_path / "prereg"
    prereg.mkdir()
    (prereg / "prereg.json").write_text(
        json.dumps(
            {
                "spec_path": "rule.md",
                "spec_sha256": runner.sha256(spec),
            }
        ),
        encoding="utf-8",
    )
    runner.validate_prereg(tmp_path)
    spec.write_text("改阈值", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA"):
        runner.validate_prereg(tmp_path)


def test_records_mutation_and_failed_validation_rejected(tmp_path):
    source = tmp_path / "input_announcements"
    source.mkdir()
    (source / "records.json").write_bytes(b"[]")
    (source / "source_manifest.json").write_text(
        json.dumps(
            {
                "records_sha256": hashlib.sha256(b"[]").hexdigest(),
                "all_catalogs_complete": True,
            }
        ),
        encoding="utf-8",
    )
    (source / "validation.json").write_text('{"all_passed": true}', encoding="utf-8")
    assert runner.load_frozen_announcements(tmp_path)[0] == []
    (source / "validation.json").write_text('{"all_passed": false}', encoding="utf-8")
    with pytest.raises(ValueError, match="自验"):
        runner.load_frozen_announcements(tmp_path)
    (source / "validation.json").write_text('{"all_passed": true}', encoding="utf-8")
    (source / "records.json").write_bytes(b"[1]")
    with pytest.raises(ValueError, match="SHA"):
        runner.load_frozen_announcements(tmp_path)
