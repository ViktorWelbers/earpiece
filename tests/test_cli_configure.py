"""The configure wizard writes a config file that from_env then accepts."""

import pytest
from typer.testing import CliRunner

from earpiece.cli import app
from earpiece.config import Settings

runner = CliRunner()


@pytest.fixture()
def isolated_config(monkeypatch, tmp_path):
    cfg = tmp_path / "config.toml"
    monkeypatch.setenv("EARPIECE_CONFIG", str(cfg))
    for var in (
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "LLM_RESPONDER_MODEL",
        "LLM_WATCHER_MODEL",
        "LLM_VERIFY_TLS",
        "EARPIECE_STT",
        "STT_BASE_URL",
        "STT_MODEL",
        "DEEPGRAM_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    return cfg


def test_configure_whisper_writes_usable_config(isolated_config):
    # model discovery hits a closed local port and falls back to manual entry
    answers = "\n".join(
        [
            "http://localhost:1/v1",  # LLM base URL (unreachable — discovery skipped)
            "local",  # API key
            "n",  # verify TLS
            "my-responder",  # responder model (no discovered default)
            "",  # watcher model (defaults to responder)
            "whisper",  # STT engine
            "",  # whisper endpoint (default)
            "",  # whisper model (default)
        ]
    )
    result = runner.invoke(app, ["configure"], input=answers + "\n")
    assert result.exit_code == 0, result.output
    assert isolated_config.is_file()

    s = Settings.from_env("mission")
    assert s.responder.base_url == "http://localhost:1/v1"
    assert s.responder.verify_tls is False
    assert s.responder.model == "my-responder"
    assert s.watcher.model == "my-responder"
    assert s.stt_engine == "whisper"
    assert s.stt_base_url == "http://localhost:8001/v1"
    assert s.stt_model == "Systran/faster-whisper-small"


def test_configure_deepgram_asks_for_key(isolated_config):
    answers = "\n".join(
        [
            "http://localhost:1/v1",  # unreachable — keeps the test offline
            "sk-test",
            "y",
            "gpt-4.1",
            "gpt-4.1-mini",
            "deepgram",
            "dg-key",
        ]
    )
    result = runner.invoke(app, ["configure"], input=answers + "\n")
    assert result.exit_code == 0, result.output

    s = Settings.from_env("mission")
    assert s.stt_engine == "deepgram"
    assert s.deepgram_api_key == "dg-key"
    assert s.watcher.model == "gpt-4.1-mini"
