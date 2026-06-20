"""Device resolution — name matching and the mic fallback to the system
default when a configured device isn't currently plugged in."""

import pytest

from earpiece.audio import capture
from earpiece.audio.capture import DeviceError, resolve_device

_DEVICES = [
    {"name": "MacBook Pro Microphone", "max_input_channels": 1, "max_output_channels": 0},
    {"name": "MacBook Pro Speakers", "max_input_channels": 0, "max_output_channels": 2},
    {"name": "BlackHole 2ch", "max_input_channels": 2, "max_output_channels": 2},
]


@pytest.fixture(autouse=True)
def fake_devices(monkeypatch):
    monkeypatch.setattr(capture.sd, "query_devices", lambda *a: _DEVICES)


def test_name_substring_resolves_to_index():
    assert resolve_device("blackhole", kind="input") == 2


def test_missing_device_falls_back_to_default_when_allowed():
    # AT2020 isn't plugged in — with fallback we use the system default (None)
    assert resolve_device("AT2020", kind="input", fallback=True) is None


def test_missing_device_still_errors_without_fallback():
    with pytest.raises(DeviceError):
        resolve_device("AT2020", kind="input")


def test_present_device_is_used_even_with_fallback():
    # fallback must not override a device that *is* available
    assert resolve_device("MacBook Pro Microphone", kind="input", fallback=True) == 0
