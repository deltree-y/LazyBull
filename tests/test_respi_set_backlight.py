import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "set_backlight",
        PROJECT_ROOT / "scripts" / "respi" / "set_backlight.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_percent_to_sysfs_value_scales_by_max_brightness():
    module = _load_module()

    raw_value = module._percent_to_sysfs_value(25, 255)

    assert raw_value == 63


def test_read_sysfs_backlight_reads_percent_and_raw_value(tmp_path):
    module = _load_module()
    brightness_path = tmp_path / "brightness"
    max_path = tmp_path / "max_brightness"
    brightness_path.write_text("50\n", encoding="utf-8")
    max_path.write_text("200\n", encoding="utf-8")

    result = module._read_sysfs_backlight(brightness_path=brightness_path, max_path=max_path)

    assert result["method"] == "sysfs"
    assert result["raw_value"] == 50
    assert result["max_brightness"] == 200
    assert result["percent"] == 25


def test_set_backlight_prefers_sysfs_when_auto_mode_and_paths_exist(tmp_path):
    module = _load_module()
    brightness_path = tmp_path / "brightness"
    max_path = tmp_path / "max_brightness"
    brightness_path.write_text("0\n", encoding="utf-8")
    max_path.write_text("255\n", encoding="utf-8")

    result = module.set_backlight(
        40,
        method="auto",
        brightness_path=brightness_path,
        max_path=max_path,
    )

    assert result["method"] == "sysfs"
    assert result["raw_value"] == 102
    assert brightness_path.read_text(encoding="utf-8").strip() == "102"


def test_set_backlight_falls_back_to_pwm_when_sysfs_unavailable(monkeypatch, tmp_path):
    module = _load_module()
    marker = {"method": "pwm", "percent": 15, "pin": 18, "frequency": 1000}

    monkeypatch.setattr(module, "_set_pwm_backlight", lambda *args, **kwargs: marker)

    result = module.set_backlight(
        15,
        method="auto",
        brightness_path=tmp_path / "missing_brightness",
        max_path=tmp_path / "missing_max",
    )

    assert result is marker


def test_main_reads_current_sysfs_backlight(capsys, tmp_path):
    module = _load_module()
    brightness_path = tmp_path / "brightness"
    max_path = tmp_path / "max_brightness"
    brightness_path.write_text("32\n", encoding="utf-8")
    max_path.write_text("128\n", encoding="utf-8")

    exit_code = module.main(
        [
            "--read",
            "--sysfs-brightness-path",
            str(brightness_path),
            "--sysfs-max-path",
            str(max_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "当前背光: 25%" in captured.out