import pytest

from earpiece.config import ConfigError, Settings


@pytest.fixture()
def env(monkeypatch, tmp_path):
    # isolate from any real ~/.config/earpiece/config.toml
    monkeypatch.setenv("EARPIECE_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setenv("AGENT_CMD", "fake-agent --acp")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_RESPONDER_MODEL", "model-big")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg")
    for var in (
        "AGENT_CWD",
        "AGENT_AUTO_TOOLS",
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


def write_config(env, tmp_path, body: str):
    cfg = tmp_path / "config.toml"
    cfg.write_text(body)
    env.setenv("EARPIECE_CONFIG", str(cfg))
    return cfg


def test_config_file_fills_in_missing_env(env, tmp_path):
    write_config(env, tmp_path, 'LLM_BASE_URL = "http://my-llm:8000/v1"\nLLM_VERIFY_TLS = false\n')
    s = Settings.from_env("mission")
    assert s.responder.base_url == "http://my-llm:8000/v1"
    assert s.responder.verify_tls is False


def test_env_wins_over_config_file(env, tmp_path):
    write_config(env, tmp_path, 'LLM_BASE_URL = "http://from-file/v1"\n')
    env.setenv("LLM_BASE_URL", "http://from-env/v1")
    s = Settings.from_env("mission")
    assert s.responder.base_url == "http://from-env/v1"


def test_config_file_can_select_stt_engine(env, tmp_path):
    env.delenv("DEEPGRAM_API_KEY")
    write_config(
        env,
        tmp_path,
        'EARPIECE_STT = "whisper"\n'
        'STT_BASE_URL = "http://localhost:8001/v1"\n'
        'STT_MODEL = "Systran/faster-whisper-small"\n',
    )
    s = Settings.from_env("mission")  # no --stt flag: file decides
    assert s.stt_engine == "whisper"
    assert s.stt_base_url == "http://localhost:8001/v1"


def test_devices_from_config_file_with_flag_override(env, tmp_path):
    write_config(env, tmp_path, 'EARPIECE_OUTPUT_DEVICE = "FreeBuds"\nEARPIECE_MIC_DEVICE = "6"\n')
    s = Settings.from_env("mission")
    assert s.output_device == "FreeBuds"
    assert s.mic_device == "6"
    assert s.system_device is None  # unset stays auto-detect
    s = Settings.from_env("mission", output_device="9")  # CLI flag wins
    assert s.output_device == "9"


def test_invalid_config_file_raises_config_error(env, tmp_path):
    write_config(env, tmp_path, "not valid toml ===\n")
    with pytest.raises(ConfigError, match="invalid TOML"):
        Settings.from_env("mission")


def test_save_config_roundtrip(env, tmp_path):
    from earpiece.config import save_config

    env.setenv("EARPIECE_CONFIG", str(tmp_path / "config.toml"))
    env.delenv("LLM_API_KEY")
    env.delenv("LLM_RESPONDER_MODEL")
    save_config(
        {"LLM_API_KEY": "k2", "LLM_RESPONDER_MODEL": 'has "quotes"', "LLM_VERIFY_TLS": False}
    )
    s = Settings.from_env("mission")
    assert s.responder.api_key == "k2"
    assert s.responder.model == 'has "quotes"'
    assert s.responder.verify_tls is False


def test_agent_cmd_required(env):
    env.delenv("AGENT_CMD")
    with pytest.raises(ConfigError, match="AGENT_CMD"):
        Settings.from_env("mission")


def test_mcp_servers_tables_are_parsed(env, tmp_path):
    write_config(
        env,
        tmp_path,
        "[mcp_servers.jira]\n"
        'command = "uvx"\n'
        'args = ["mcp-atlassian"]\n'
        "[mcp_servers.jira.env]\n"
        'JIRA_URL = "https://jira.example"\n'
        "[mcp_servers.search]\n"
        'url = "https://mcp.example/sse"\n'
        'transport = "sse"\n',
    )
    s = Settings.from_env("mission")
    assert s.mcp_servers["jira"]["command"] == "uvx"
    assert s.mcp_servers["jira"]["env"]["JIRA_URL"] == "https://jira.example"
    assert s.mcp_servers["search"]["url"] == "https://mcp.example/sse"


def test_save_config_roundtrips_mcp_servers(env, tmp_path):
    from earpiece.config import save_config

    env.setenv("EARPIECE_CONFIG", str(tmp_path / "config.toml"))
    save_config(
        {"AGENT_CMD": "npx pi-acp"},
        mcp_servers={
            "jira": {
                "command": "uvx",
                "args": ["mcp-atlassian"],
                "env": {"JIRA_TOKEN": "t"},
            }
        },
    )
    s = Settings.from_env("mission")
    assert s.mcp_servers["jira"]["args"] == ["mcp-atlassian"]
    assert s.mcp_servers["jira"]["env"] == {"JIRA_TOKEN": "t"}


def test_mcp_servers_map_to_acp_shape(env, tmp_path):
    from earpiece.brain.acp import acp_mcp_servers

    servers = acp_mcp_servers(
        {
            "jira": {"command": "uvx", "args": ["mcp-atlassian"], "env": {"A": "1"}},
            "api": {"url": "https://mcp.example", "headers": {"Authorization": "Bearer x"}},
        }
    )
    assert servers[0] == {
        "name": "jira",
        "command": "uvx",
        "args": ["mcp-atlassian"],
        "env": [{"name": "A", "value": "1"}],
    }
    assert servers[1]["type"] == "http"
    assert servers[1]["headers"] == [{"name": "Authorization", "value": "Bearer x"}]
