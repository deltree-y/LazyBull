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


def test_build_preview_framebuffer_bytes_has_expected_size_and_color_variation():
    module = _load_module()

    payload = module._build_preview_framebuffer_bytes(width=16, height=12)

    assert len(payload) == 16 * 12 * 2
    colors = {payload[index:index + 2] for index in range(0, len(payload), 2)}
    assert len(colors) >= 6


def test_discover_sysfs_backlights_returns_available_devices(tmp_path):
    module = _load_module()
    backlight_root = tmp_path / "backlight"
    device_dir = backlight_root / "rpi_backlight"
    device_dir.mkdir(parents=True)
    (device_dir / "brightness").write_text("32\n", encoding="utf-8")
    (device_dir / "max_brightness").write_text("255\n", encoding="utf-8")

    devices = module._discover_sysfs_backlights(backlight_root)

    assert len(devices) == 1
    assert devices[0]["name"] == "rpi_backlight"


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


def test_set_backlight_discovers_non_default_sysfs_device_in_auto_mode(tmp_path):
    module = _load_module()
    backlight_root = tmp_path / "backlight"
    device_dir = backlight_root / "display0"
    device_dir.mkdir(parents=True)
    (device_dir / "brightness").write_text("0\n", encoding="utf-8")
    (device_dir / "max_brightness").write_text("100\n", encoding="utf-8")

    result = module.set_backlight(
        35,
        method="auto",
        brightness_path=tmp_path / "missing_brightness",
        max_path=tmp_path / "missing_max",
        backlight_root=backlight_root,
    )

    assert result["method"] == "sysfs"
    assert result["backlight_name"] == "display0"
    assert (device_dir / "brightness").read_text(encoding="utf-8").strip() == "35"


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


def test_set_pwm_backlight_prefers_lgpio_backend(monkeypatch):
    module = _load_module()
    marker = {"method": "pwm", "backend": "lgpio", "percent": 10, "pin": 18, "frequency": 1000}

    monkeypatch.setattr(module, "_set_lgpio_backlight", lambda *args, **kwargs: marker)
    monkeypatch.setattr(
        module,
        "_set_rpi_gpio_backlight",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("不应回退到 RPi.GPIO")),
    )

    result = module._set_pwm_backlight(10)

    assert result is marker


def test_set_pwm_backlight_falls_back_to_rpi_gpio_when_lgpio_fails(monkeypatch):
    module = _load_module()
    marker = {
        "method": "pwm",
        "backend": "rpi-gpio",
        "percent": 10,
        "pin": 18,
        "frequency": 1000,
    }

    monkeypatch.setattr(
        module,
        "_set_lgpio_backlight",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("lgpio failed")),
    )
    monkeypatch.setattr(module, "_set_rpi_gpio_backlight", lambda *args, **kwargs: marker)

    result = module._set_pwm_backlight(10)

    assert result is marker


def test_update_pwm_backlight_state_updates_lgpio_backend():
    module = _load_module()
    calls = []

    class DummyLgpio:
        def tx_pwm(self, handle, pin, frequency, duty_cycle):
            calls.append((handle, pin, frequency, duty_cycle))

    state = {
        "backend": "lgpio",
        "handle": 3,
        "pin": 18,
        "frequency": 1000,
        "lgpio_module": DummyLgpio(),
        "percent": 10,
    }

    updated = module.update_pwm_backlight_state(state, 25)

    assert updated["percent"] == 25
    assert calls == [(3, 18, 1000, 25.0)]


def test_cleanup_backlight_state_runs_cleanup_callback():
    module = _load_module()
    called = []

    module.cleanup_backlight_state({"cleanup": lambda: called.append(True)})

    assert called == [True]


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


def test_main_reads_discovered_sysfs_backlight_when_default_path_missing(capsys, tmp_path):
    module = _load_module()
    backlight_root = tmp_path / "backlight"
    device_dir = backlight_root / "display0"
    device_dir.mkdir(parents=True)
    (device_dir / "brightness").write_text("40\n", encoding="utf-8")
    (device_dir / "max_brightness").write_text("200\n", encoding="utf-8")

    exit_code = module.main(["--read", "--backlight-root", str(backlight_root)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "当前背光: 20%" in captured.out
    assert "背光节点: display0" in captured.out


def test_main_writes_preview_by_default_when_setting_brightness(monkeypatch, capsys):
    module = _load_module()
    preview_calls = []

    monkeypatch.setattr(
        module,
        "set_backlight",
        lambda *args, **kwargs: {"method": "sysfs", "percent": 10, "raw_value": 25, "max_brightness": 255},
    )
    monkeypatch.setattr(
        module,
        "_write_preview_pattern",
        lambda **kwargs: preview_calls.append(kwargs) or {"fb_path": "/dev/fb1", "width": 480, "height": 320},
    )

    exit_code = module.main(["10"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert len(preview_calls) == 1
    assert "已写入亮度测试画面" in captured.out


def test_main_skips_preview_when_no_preview_is_set(monkeypatch):
    module = _load_module()

    monkeypatch.setattr(
        module,
        "set_backlight",
        lambda *args, **kwargs: {"method": "sysfs", "percent": 10, "raw_value": 25, "max_brightness": 255},
    )
    monkeypatch.setattr(
        module,
        "_write_preview_pattern",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("不应写入测试画面")),
    )

    exit_code = module.main(["10", "--no-preview"])

    assert exit_code == 0