"""P2b-T3：生产模型实际路径与冻结参照身份的合成测试。"""

import hashlib
import json
from pathlib import Path

import pytest

from scripts.v2_p2b.check_frozen_inputs import (
    check_model_paths_in_manifest,
    digest_b0_raw_reference,
    digest_p2a_runs_reference,
    resolve_b0_raw_reference,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _model_case(tmp_path, *, model_file="v24008_model.joblib", fallback=False):
    data_root = tmp_path / "data"
    models = data_root / "models" / "stock_selection"
    models.mkdir(parents=True)
    metadata = {
        "version": 24008,
        "version_str": "v24008",
        "model_type": "xgboost",
        "model_file": model_file,
        "features_file": "v24008_features.json",
        "train_params": {"objective": "mse"},
        "train_start_date": "20120101",
        "train_end_date": "20180101",
        "feature_count": 0,
    }
    (models / "v24008_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    features = models / "v24008_features.json"
    features.write_text("[]", encoding="utf-8")
    if not fallback:
        (models / model_file).write_bytes(b"model")
    else:
        (models / "v24008_model.json").write_bytes(b"native-fallback")
    manifest = {
        "files": {
            "models/stock_selection/v24008_features.json": {
                "sha256": _sha(features),
                "size": features.stat().st_size,
            }
        }
    }
    selected_model = models / model_file
    if selected_model.exists():
        manifest["files"][f"models/stock_selection/{model_file}"] = {
            "sha256": _sha(selected_model),
            "size": selected_model.stat().st_size,
        }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return data_root, manifest_path


def test_joblib_actual_paths_in_manifest_pass(tmp_path):
    data_root, manifest = _model_case(tmp_path)
    report = check_model_paths_in_manifest(data_root, manifest, versions=(24008,))
    assert report["ok"], report["errors"]
    assert report["selected"]["24008"] == {
        "model_path": "models/stock_selection/v24008_model.joblib",
        "features_path": "models/stock_selection/v24008_features.json",
    }


def test_actual_json_and_fallback_outside_manifest_rejected(tmp_path):
    data_root, manifest = _model_case(tmp_path / "json", model_file="v24008_model.json")
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["files"].pop("models/stock_selection/v24008_model.json")
    manifest.write_text(json.dumps(document), encoding="utf-8")
    report = check_model_paths_in_manifest(data_root, manifest, versions=(24008,))
    assert not report["ok"]
    assert any("v24008_model.json" in error and "清单外" in error for error in report["errors"])

    data_root, manifest = _model_case(tmp_path / "fallback", fallback=True)
    report = check_model_paths_in_manifest(data_root, manifest, versions=(24008,))
    assert not report["ok"]
    assert any("native" not in error and "v24008_model.json" in error for error in report["errors"])


def test_selected_model_content_hash_mismatch_rejected(tmp_path):
    data_root, manifest_path = _model_case(tmp_path)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = "models/stock_selection/v24008_model.joblib"
    document["files"][key]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    report = check_model_paths_in_manifest(data_root, manifest_path, versions=(24008,))
    assert not report["ok"]
    assert any("内容哈希不符" in error for error in report["errors"])


def _raw_case(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    (raw / "walk_forward_summary_0101_0001.csv").write_text("split_index\n0\n", encoding="utf-8")
    (raw / "chain_nav_wf_syn.csv").write_text("date,nav,split_index\n0,1.0,0\n", encoding="utf-8")
    (raw / "walk_forward_trades_syn_split00.csv").write_text("date\n20240102\n", encoding="utf-8")
    (raw / "walk_forward_execution_attribution_syn_split00.csv").write_text(
        "status\nfilled\n", encoding="utf-8"
    )
    return raw


def test_raw_reference_missing_and_ambiguous_candidates_rejected(tmp_path):
    raw = _raw_case(tmp_path)
    refs = resolve_b0_raw_reference(raw, splits=(0,))
    assert refs.summary.name == "walk_forward_summary_0101_0001.csv"
    first = digest_b0_raw_reference(raw, refs)
    (raw / "walk_forward_summary_0101_0001.csv").write_text("split_index\n1\n", encoding="utf-8")
    assert digest_b0_raw_reference(raw, refs)["digest"] != first["digest"]

    (raw / "chain_nav_wf_other.csv").write_text("date,nav,split_index\n0,1.0,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="命中数=2"):
        resolve_b0_raw_reference(raw, splits=(0,))
    (raw / "chain_nav_wf_other.csv").unlink()
    (raw / "walk_forward_execution_attribution_syn_split00.csv").unlink()
    with pytest.raises(ValueError, match="命中数=0"):
        resolve_b0_raw_reference(raw, splits=(0,))


def _p2a_case(tmp_path):
    batch = tmp_path / "p2a_reference"
    batch.mkdir()
    (batch / "batch_meta.json").write_text(json.dumps({"batch_id": batch.name}), encoding="utf-8")
    (batch / "summary.parquet").write_bytes(b"summary")
    (batch / "chain_nav.parquet").write_bytes(b"chain")
    for split in (0, 1):
        fold = batch / "folds" / f"split{split:02d}"
        fold.mkdir(parents=True)
        for name in ("trades", "attribution", "holdings_snapshot", "daily"):
            (fold / f"{name}.parquet").write_bytes(f"{split}-{name}".encode())
        (fold / "_meta.json").write_text("{}", encoding="utf-8")
    return batch


def test_p2a_reference_fixed_set_and_content_digest(tmp_path):
    batch = _p2a_case(tmp_path)
    first = digest_p2a_runs_reference(batch, splits=(0, 1))
    assert len(first["files"]) == 13
    target = batch / "folds" / "split00" / "daily.parquet"
    target.write_bytes(b"changed")
    assert digest_p2a_runs_reference(batch, splits=(0, 1))["digest"] != first["digest"]
    target.unlink()
    with pytest.raises(FileNotFoundError, match="固定文件"):
        digest_p2a_runs_reference(batch, splits=(0, 1))
