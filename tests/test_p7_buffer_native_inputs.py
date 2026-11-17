"""原生续持复验输入的合成边界测试，不依赖真实数据与模型。"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.v2_p7 import buffer_native_inputs as subject


def _save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def synthetic_root(tmp_path, monkeypatch):
    monkeypatch.setattr(subject, "START", "20260702")
    monkeypatch.setattr(subject, "END", "20260703")
    for name in ["data/clean", "data/models/stock_selection", "data/paper", "configs"]:
        (tmp_path / name).mkdir(parents=True)
    pd.DataFrame(
        {"exchange": ["SSE", "SSE"], "is_open": [1, 1], "cal_date": ["20260702", "20260703"]}
    ).to_parquet(tmp_path / "data/clean/trade_cal.parquet")
    pd.DataFrame({"ts_code": ["000001.SZ"]}).to_parquet(tmp_path / "data/clean/stock_basic.parquet")
    for name in ["data/paper/config.yaml", "configs/base.yaml"]:
        (tmp_path / name).write_text("model_version: 24022", encoding="utf-8")
    models = tmp_path / "data/models/stock_selection"
    _save_json(models / "model_registry.json", {"models": []})
    for version in [24021, 24022]:
        model_file, features_file = f"v{version}_model.joblib", f"v{version}_features.json"
        (models / model_file).write_bytes(b"synthetic model")
        (models / features_file).write_text('["factor"]', encoding="utf-8")
        _save_json(
            models / f"v{version}_metadata.json",
            {
                "train_start_date": "20200106",
                "train_end_date": "20260105",
                "created_at": "2026-09-07",
                "label_column": "y_ret_20",
                "model_file": model_file,
                "features_file": features_file,
            },
        )
    pd.DataFrame(
        {
            "model_version": [24022],
            "val_es_end_date": ["20250828"],
            "val_calib_end_date": ["20251203"],
        }
    ).to_csv(models / "ml_train_runs.csv", index=False)
    partitions = {}
    groups = [
        "core",
        "fundamental",
        "technical",
        "risk",
        "announcement",
        "market_state",
        "moneyflow",
        "neutralized",
    ]
    for day, domain in [("20260702", "label_filtered"), ("20260703", "structural")]:
        quote_path = tmp_path / "data/clean/daily" / f"2026-07-{day[-2:]}.parquet"
        quote_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"trade_date": [day], "ts_code": ["000001.SZ"]}).to_parquet(quote_path)
        for group in groups:
            path = tmp_path / "data/features/panel" / day / f"{group}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"trade_date": [day], "ts_code": ["000001.SZ"]}).to_parquet(path)
            partitions[f"panel/{day}/{group}.parquet"] = {"row_domain": domain}
    _save_json(
        tmp_path / "data/features/manifest.json",
        {
            "partitions": partitions,
            "row_domain_segments": {
                "label_filtered_through": "20260702",
                "structural_from": "20260703",
            },
        },
    )
    path = tmp_path / subject.PREREG
    path.parent.mkdir(parents=True)
    path.write_text("synthetic specification", encoding="utf-8")
    code = tmp_path / "scripts/v2_p7/buffer_native_inputs.py"
    code.parent.mkdir(parents=True)
    code.write_text("# synthetic code", encoding="utf-8")
    return tmp_path


def test_audit_records_two_row_domains_and_actual_sha(synthetic_root):
    audit = subject.audit_holdout(synthetic_root)
    assert audit["day_count"] == 2
    assert [row["row_domain"] for row in audit["coverage"]] == ["label_filtered", "structural"]
    assert audit["training_dates"][0]["val_calib_end_date"] == 20251203
    assert audit["industry"]["historical_pit_proven"] is False
    for item in audit["input_files"]:
        assert (
            item["sha256"]
            == hashlib.sha256((synthetic_root / item["path"]).read_bytes()).hexdigest()
        )


def test_missing_panel_never_uses_cs_train(synthetic_root):
    (synthetic_root / "data/features/panel/20260703/core.parquet").unlink()
    fallback = synthetic_root / "data/features/cs_train/20260703.parquet"
    fallback.parent.mkdir(parents=True)
    pd.DataFrame({"trade_date": ["20260703"]}).to_parquet(fallback)
    with pytest.raises(ValueError, match="禁止 cs_train 回退"):
        subject.audit_holdout(synthetic_root)


def test_changed_row_domain_is_rejected(synthetic_root):
    path = synthetic_root / "data/features/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["partitions"]["panel/20260703/core.parquet"]["row_domain"] = "label_filtered"
    _save_json(path, manifest)
    with pytest.raises(ValueError, match="行域"):
        subject.audit_holdout(synthetic_root)


class SyntheticProvider:
    def __init__(self, root):
        self.root = root
        self.day_records = []
        self.selection = {"top_n": 20}
        self.universe_config = {"markets": ["主板"]}

    def __call__(self, day):
        rows = [("000002.SZ", 1.0), ("000001.SZ", 1.0)] if day == "20260702" else []
        self.day_records.append({"day": day, "ranked_rows": len(rows), "empty_reason": "test_gate"})
        return rows


def test_build_preserves_kernel_ties_and_empty_days(synthetic_root, monkeypatch):
    monkeypatch.setattr(subject, "HoldoutRankedProvider", SyntheticProvider)
    output = synthetic_root / "output"
    result = subject.build_holdout_rankings(
        synthetic_root, output, spec_sha256=subject._sha(synthetic_root / subject.PREREG)
    )
    assert list(result) == ["20260702", "20260703"]
    assert result["20260702"][0][0] == "000002.SZ"
    assert result["20260703"] == []
    validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    assert validation["empty_days"][0]["day"] == "20260703"
    assert validation["cs_train_fallback"] is False
    assert json.loads((output / "manifest.json").read_text())["complete"] is True
    with pytest.raises(FileExistsError):
        subject.build_holdout_rankings(
            synthetic_root, output, spec_sha256=subject._sha(synthetic_root / subject.PREREG)
        )


def test_drift_refuses_complete_manifest(synthetic_root, monkeypatch):
    class DriftingProvider(SyntheticProvider):
        def __call__(self, day):
            rows = super().__call__(day)
            (self.root / "configs/base.yaml").write_text("changed", encoding="utf-8")
            return rows

    monkeypatch.setattr(subject, "HoldoutRankedProvider", DriftingProvider)
    output = synthetic_root / "output"
    with pytest.raises(ValueError, match="漂移"):
        subject.build_holdout_rankings(
            synthetic_root, output, spec_sha256=subject._sha(synthetic_root / subject.PREREG)
        )
    assert (output / "audit.json").exists()
    assert not (output / "manifest.json").exists()


def test_wrong_spec_rejected_before_output(synthetic_root):
    output = synthetic_root / "output"
    with pytest.raises(ValueError, match="预登记 SHA"):
        subject.build_holdout_rankings(synthetic_root, output, spec_sha256="0" * 64)
    assert not output.exists()


def test_new_optional_input_during_run_is_rejected(synthetic_root, monkeypatch):
    class AppearingWalProvider(SyntheticProvider):
        def __call__(self, day):
            rows = super().__call__(day)
            (self.root / "data/features/manifest_wal.jsonl").write_text("{}", encoding="utf-8")
            return rows

    monkeypatch.setattr(subject, "HoldoutRankedProvider", AppearingWalProvider)
    output = synthetic_root / "output"
    with pytest.raises(ValueError, match="推理期间出现"):
        subject.build_holdout_rankings(
            synthetic_root, output, spec_sha256=subject._sha(synthetic_root / subject.PREREG)
        )
    assert not (output / "manifest.json").exists()


def test_features_require_requested_day_without_fallback():
    from src.lazybull.v2.hosts.paper import daily_assembly

    provider = subject.HoldoutRankedProvider.__new__(subject.HoldoutRankedProvider)
    provider.columns = ["factor"]
    provider.assembly = daily_assembly
    provider.store = SimpleNamespace(load_features=lambda query: SimpleNamespace(df=pd.DataFrame()))
    with pytest.raises(ValueError, match="禁止 cs_train"):
        provider._features("20260703")


def test_provider_loads_lazy_model_before_official_column_classification(monkeypatch, tmp_path):
    from src.lazybull import universe
    from src.lazybull.data import loader, storage
    from src.lazybull.v2.core.signal import ml_signal
    from src.lazybull.v2.store import data_store

    class LazySignal:
        def __init__(self, **kwargs):
            self.feature_columns = None
            self.metadata = None

        def _load_model(self):
            self.feature_columns = ["factor"]
            self.metadata = {"train_params": {}}

    monkeypatch.setattr(ml_signal, "MLSignal", LazySignal)
    monkeypatch.setattr(storage, "Storage", lambda **kwargs: None)
    monkeypatch.setattr(
        loader,
        "DataLoader",
        lambda ignored: SimpleNamespace(
            load_clean_stock_basic=lambda: pd.DataFrame({"ts_code": ["000001.SZ"]})
        ),
    )
    monkeypatch.setattr(universe, "BasicUniverse", lambda **kwargs: object())
    monkeypatch.setattr(
        data_store,
        "PanelDataStore",
        lambda root: SimpleNamespace(
            manifest=SimpleNamespace(
                snapshot=lambda: {
                    "columns": {
                        name: {} for name in ["factor", "amount_ma20", "total_mv", "sw_l1_code"]
                    }
                }
            )
        ),
    )
    provider = subject.HoldoutRankedProvider(tmp_path)
    assert provider.signal.feature_columns == ["factor"]
    assert set(provider.columns) == {"factor", "amount_ma20", "total_mv", "sw_l1_code"}
