import pytest

from earpiece.config import ConfigError, Settings


@pytest.fixture()
def env(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_RESPONDER_MODEL", "model-big")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg")
    for var in (
        "LLM_BASE_URL",
        "LLM_WATCHER_MODEL",
        "LLM_JSON_SCHEMA",
        "LLM_VERIFY_TLS",
        "STT_BASE_URL",
        "STT_MODEL",
        "STT_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_minimal_env(env):
    s = Settings.from_env("mission")
    assert s.responder.model == "model-big"
    assert s.watcher.model == "model-big"  # falls back
    assert s.responder.base_url == "https://api.openai.com/v1"
    assert s.responder.supports_json_schema is True


def test_watcher_overrides(env):
    env.setenv("LLM_WATCHER_MODEL", "model-small")
    env.setenv("LLM_WATCHER_BASE_URL", "http://localhost:11434/v1")
    s = Settings.from_env("mission")
    assert s.watcher.model == "model-small"
    assert s.watcher.base_url == "http://localhost:11434/v1"
    assert s.responder.base_url == "https://api.openai.com/v1"


def test_json_schema_flag(env):
    env.setenv("LLM_JSON_SCHEMA", "false")
    s = Settings.from_env("mission")
    assert s.responder.supports_json_schema is False


def test_verify_tls_flag(env):
    s = Settings.from_env("mission")
    assert s.responder.verify_tls is True
    env.setenv("LLM_VERIFY_TLS", "false")
    s = Settings.from_env("mission")
    assert s.responder.verify_tls is False
    assert s.watcher.verify_tls is False


def test_missing_api_key_raises(env):
    env.delenv("LLM_API_KEY")
    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        Settings.from_env("mission")


def test_deepgram_required_only_for_deepgram(env):
    env.delenv("DEEPGRAM_API_KEY")
    with pytest.raises(ConfigError, match="DEEPGRAM_API_KEY"):
        Settings.from_env("mission", stt_engine="deepgram")
    env.setenv("STT_BASE_URL", "http://localhost:8001/v1")
    env.setenv("STT_MODEL", "openai/whisper-large-v3")
    s = Settings.from_env("mission", stt_engine="whisper")
    assert s.stt_engine == "whisper"


def test_whisper_requires_endpoint_and_model(env):
    with pytest.raises(ConfigError, match="STT_BASE_URL"):
        Settings.from_env("mission", stt_engine="whisper")
    env.setenv("STT_BASE_URL", "http://localhost:8001/v1")
    env.setenv("STT_MODEL", "openai/whisper-large-v3")
    env.setenv("STT_API_KEY", "secret")
    s = Settings.from_env("mission", stt_engine="whisper")
    assert s.stt_base_url == "http://localhost:8001/v1"
    assert s.stt_model == "openai/whisper-large-v3"
    assert s.stt_api_key == "secret"


def test_stt_api_key_defaults_to_local(env):
    env.setenv("STT_BASE_URL", "http://localhost:8001/v1")
    env.setenv("STT_MODEL", "whisper-1")
    s = Settings.from_env("mission", stt_engine="whisper")
    assert s.stt_api_key == "local"


def test_flags_win_over_defaults(env):
    s = Settings.from_env("mission", eager=True, tts_engine="say", mic_device="2")
    assert s.eager is True
    assert s.tts_engine == "say"
    assert s.mic_device == "2"
