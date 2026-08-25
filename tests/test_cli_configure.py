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
        "AGENT_CMD",
        "AGENT_CWD",
        "AGENT_AUTO_TOOLS",
        "EARPIECE_STT",
        "STT_BASE_URL",
        "STT_MODEL",
        "DEEPGRAM_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    return cfg


def test_configure_whisper_writes_usable_config(isolated_config):
    answers = "\n".join(
        [
            "my-agent --acp",  # ACP harness command
            "whisper",  # STT engine
            "",  # whisper endpoint (default)
            "",  # whisper model (default)
        ]
    )
    result = runner.invoke(app, ["configure"], input=answers + "\n")
    assert result.exit_code == 0, result.output
    assert isolated_config.is_file()

    s = Settings.from_env("mission")
    assert s.agent_cmd == "my-agent --acp"
    assert s.stt_engine == "whisper"
    assert s.stt_base_url == "http://localhost:8001/v1"
    assert s.stt_model == "Systran/faster-whisper-small"


def test_configure_deepgram_asks_for_key(isolated_config):
    answers = "\n".join(
        [
            "",  # agent command (default)
            "deepgram",  # STT engine
            "dg-key",  # deepgram key
        ]
    )
    result = runner.invoke(app, ["configure"], input=answers + "\n")
    assert result.exit_code == 0, result.output

    s = Settings.from_env("mission")
    assert s.agent_cmd == "opencode acp"  # wizard default
    assert s.stt_engine == "deepgram"
    assert s.deepgram_api_key == "dg-key"


def test_configure_show_prints_values_and_sources(isolated_config, monkeypatch):
    isolated_config.write_text(
        'SOME_BASE_URL = "http://my-service:8000/v1"\nSOME_API_KEY = "sk-verysecretkey123"\n'
    )
    monkeypatch.setenv("SOME_BASE_URL", "http://from-env/v1")

    result = runner.invoke(app, ["configure", "show"])
    assert result.exit_code == 0, result.output
    assert "http://from-env/v1" in result.output  # env wins
    assert "overrides file" in result.output
    assert "sk-verysecretkey123" not in result.output  # masked
    assert "sk-v…23" in result.output


def test_configure_show_without_file_hints_at_wizard(isolated_config):
    result = runner.invoke(app, ["configure", "show"])
    assert result.exit_code == 0, result.output
    assert "no config file" in result.output
