# earpiece

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green) ![Platform: macOS](https://img.shields.io/badge/platform-macOS-lightgrey)

A real-time "voice of god" assistant for your Mac. Start it with a mission — *"help me in
this sales discussion"*, *"help me with this tech trivia"* — and it listens to your
**microphone** and your **system audio** (the other side of the call, a video, a meeting),
keeps a live speaker-tagged transcript, and streams back guidance as **live text** and
optionally as **voice in your earpiece**.

The point is that it behaves like a real-time participant, not a batch chatbot: a fast
watcher model decides *when* to speak, an **agent** produces *what to say — and can act*:
answers come from a coding-agent harness ([pi](https://pi.dev), Claude Code, Gemini CLI, …)
speaking the [Agent Client Protocol](https://agentclientprotocol.com), so the assistant can
create a JIRA ticket from the conversation, run a web search, or use any MCP tool — with a
y/n confirmation gate before anything that mutates state. If the conversation moves on
mid-answer, the in-flight answer is cancelled mid-sentence and replaced.

- **Bring your own agent** — any ACP harness works (`npx pi-acp`, `npx claude-agent-acp`,
  `gemini --experimental-acp`); the harness brings its model endpoint and tools
- **Tools with a leash** — read-only tools run instantly; writes (tickets, files, shell)
  pause for your `y`/`n`; MCP servers from your config are forwarded to the harness
- **Pluggable STT** — [Deepgram](https://deepgram.com) streaming, or any OpenAI-compatible
  `/audio/transcriptions` endpoint (local whisper via Docker included)
- **Speaker attribution without diarization** — mic and system audio are separate capture
  channels, so utterances are tagged `ME` / `THEM` for free
- **Natural interruption** — stale answers are cancelled at the next word, queued speech is
  flushed, and the agent turn is cancelled over the protocol
- **Barge-in** — the moment you speak, voice output pauses; it never talks over you
- **Shared-mode mic capture** — Zoom/Meet/games keep full use of the same microphone
- **Local-friendly** — whisper in Docker, watcher on your own vLLM/Ollama, and a harness
  like pi configured against a local endpoint

## Quick start

```sh
brew install blackhole-2ch          # one-time: system-audio loopback (see Setup below)

git clone https://github.com/ViktorWelbers/earpiece && cd earpiece
uv tool install .                   # installs the `earpiece` command on your PATH

earpiece configure                  # interactive setup, stored in ~/.config/earpiece/
earpiece run "help me with tech trivia" --eager
```

Talk, or play something with speech — the left pane fills with the transcript, the right
pane streams timestamped answers.

## How it works

```
mic ────────────► STT ("ME")  ──┐
                                ├─► rolling transcript (speaker-tagged, cache-friendly)
system audio ──► STT ("THEM") ──┘                 │ on each finalized utterance
   (BlackHole)                                    ▼
                                  Watcher (fast model): stay silent?
                                  respond? interrupt the current answer?
                                                  │
                                                  ▼
                                  Agent harness (ACP subprocess: pi / claude / …)
                                  owns the model + tools; MCP servers forwarded
                                       │ text deltas        │ tool calls
                                       ▼                    ▼
                              live answer timeline    ⏸ y/n confirmation gate
                              + sentence-chunked TTS  (reads auto, writes confirm)
```

*When to speak* and *what to say* are different problems: the **watcher** is a fast, cheap
OpenAI-compatible model returning a structured decision per utterance (kept as a direct
LLM call — an agent loop would blow the latency budget). The **responder** is an external
agent harness spawned as a subprocess and driven over the
[Agent Client Protocol](https://agentclientprotocol.com): each turn forwards the new
transcript lines, the harness streams the answer back and may call tools along the way.
Tool calls show up in the answers timeline (`⚙ create_ticket(...)`); anything that isn't
read-only pauses with a status-bar banner until you press `y` or `n` (30 s ⇒ denied).
Every new utterance is evaluated against the partial answer in flight — if it's stale, the
turn is cancelled over the protocol, unspoken TTS is dropped, the history gets an
`[interrupted]` marker, and a fresh turn starts. `--eager` skips the watcher entirely and
answers every `THEM` utterance (lowest latency, ideal for trivia).

The transcript is kept append-only with a frozen system prompt, so provider-side prefix
caching works; long sessions are compacted by summarizing the oldest half. Full design doc:
[AGENTS.md](AGENTS.md).

## Setup

### 1. Install

Requires macOS, Python 3.12+, and [uv](https://docs.astral.sh/uv/).

```sh
uv tool install .              # system-wide `earpiece` command (isolated venv)
# or: uv tool install -e .     # editable — code changes in this checkout apply immediately
# or: pipx install .           # same idea via pipx
# or: uv sync                  # dev environment only; run via `uv run earpiece`
```

### 2. System-audio loopback (one-time)

Capturing "the other side" needs a loopback device:

```sh
brew install blackhole-2ch
```

Then in **Audio MIDI Setup** (`/Applications/Utilities`):

1. `+` → **Create Multi-Output Device**
2. Check **your real output** (speakers / earpiece) **and** **BlackHole 2ch**
3. Set the system output device to this Multi-Output Device

You hear everything normally; earpiece records the BlackHole side. Without this, only the
mic channel works.

> **Routing rule for `--voice`:** the TTS output device (`--output-device`, e.g. your
> earbuds) should **not** be part of the Multi-Output Device — otherwise the people you're
> talking to hear the assistant. (A software guard mutes the `THEM` capture channel while
> TTS plays — plus a 1s tail — so the assistant won't transcribe itself either way.)

### 3. Configure

```sh
earpiece configure
```

An interactive wizard: the agent harness command, the watcher LLM endpoint + key (models
are auto-discovered from `/v1/models` when reachable), TLS verification, and the STT
engine. Answers are stored in `~/.config/earpiece/config.toml`. Running `earpiece run`
without any configuration offers the wizard automatically.

The file uses the same names as the environment variables, and **env vars override the
file**, so one-off overrides stay easy:

```toml
# ~/.config/earpiece/config.toml
AGENT_CMD = "npx pi-acp"               # the ACP harness that answers (and acts)
LLM_BASE_URL = "https://my-vllm-host/v1"
LLM_API_KEY = "local"
LLM_RESPONDER_MODEL = "my-model"       # watcher + transcript summarizer
LLM_VERIFY_TLS = false                 # self-signed certs
EARPIECE_STT = "whisper"
STT_BASE_URL = "http://localhost:8001/v1"
STT_MODEL = "Systran/faster-whisper-small"

# optional: extra MCP tools, forwarded to the harness at session start
[mcp_servers.jira]
command = "uvx"
args = ["mcp-atlassian"]
[mcp_servers.jira.env]
JIRA_URL = "https://yourcompany.atlassian.net"
JIRA_PERSONAL_TOKEN = "..."
```

The harness must be set up on its own once (`pi` needs a provider/model configured,
`claude-agent-acp` needs Claude Code credentials, …) — earpiece just spawns `AGENT_CMD`
and speaks ACP to it. The harness's model is configured *in the harness*; the `LLM_*`
keys only power the watcher.

Optional keys: `AGENT_CWD` to pin the harness's working directory,
`AGENT_AUTO_TOOLS = "jira_search*,lookup_*"` (comma-separated globs) to let specific
non-read-only tools run without confirmation, `LLM_WATCHER_BASE_URL` /
`LLM_WATCHER_API_KEY` / `LLM_WATCHER_MODEL` to put the watcher on a different
provider/model, `LLM_JSON_SCHEMA = false` for providers without structured-output
support, `DEEPGRAM_API_KEY` for `--stt deepgram`, `STT_API_KEY` for hosted whisper
endpoints, and `EARPIECE_MIC_DEVICE` / `EARPIECE_SYSTEM_DEVICE` / `EARPIECE_OUTPUT_DEVICE`
to pin audio devices (index or name substring; the matching CLI flags override).
`EARPIECE_CONFIG=/path/to/file.toml` relocates the config file.
`earpiece configure show` prints the effective configuration and where each value
comes from.

## Fully local mode

No cloud STT, no cloud LLM: whisper runs in a local Docker container
([speaches](https://speaches.ai), CPU image, OpenAI-compatible `/v1/audio/transcriptions`
on `localhost:8001`), the watcher points at your own vLLM/Ollama server, and the agent
harness is one that supports custom OpenAI-compatible providers (e.g.
[pi](https://github.com/earendil-works/pi) with a custom provider entry for your local
endpoint). Note: tool calling requires the local server to parse tool calls — for vLLM
that means starting it with `--enable-auto-tool-choice --tool-call-parser <parser>`.

```sh
# needs Docker Desktop / OrbStack running; reads ~/.config/earpiece/config.toml
scripts/run-local.sh "help me with tech trivia" --eager --voice say

# env vars override the config file for one-off runs:
STT_MODEL=Systran/faster-distil-whisper-large-v3 \
scripts/run-local.sh "help me in this sales discussion"
```

The script is just convenience around `docker compose -f local/docker-compose.yml up -d`
plus a one-time whisper-model install — once the container is running, a plain
`earpiece run "..."` works too.

Notes:

- Docker on macOS is CPU-only; `Systran/faster-whisper-small` (the default) keeps latency
  realtime-ish. Bigger models = better accuracy, slower finalization.
- The whisper backend does utterance endpointing locally (energy VAD) and posts each
  finished segment as a small WAV — slightly higher latency than Deepgram's streaming
  endpoint, fine for everyday use.
- vLLM supports structured outputs (`LLM_JSON_SCHEMA = true`, the default); set it to
  `false` if your build rejects `response_format: json_schema`.
- Stop the whisper container with `docker compose -f local/docker-compose.yml down`.

## Usage

```sh
# list audio devices (find your mic / loopback / earpiece indices)
earpiece devices

# text-only, watcher decides when to speak
earpiece run "help me in this sales discussion"

# trivia mode: answer every utterance from the other side, speak into the earpiece
earpiece run "help me with this tech trivia" --eager --voice say

# solo testing without system audio: answer your own mic too
earpiece run "quiz me on networking" --eager --eager-source both

# explicit devices (two-mic setup: mic 6 feeds earpiece, you reply on another mic)
earpiece run "..." --mic-device 6 --system-device BlackHole --output-device 1
```

**Hotkeys while running:** `space` force an answer now · `y` / `n` approve or deny a
pending tool call · `m` mute mic · `v` toggle voice · `q` quit.

**The status bar** shows channel health (mic / sys / stt / voice), the last watcher
decision and its reason, prompt-cache usage — and any error, so failures are never silent
(details land in `earpiece.log`). When the agent wants to run a non-read-only tool, the
status bar switches to a yellow `confirm: <tool> — press y to run, n to deny` banner and
the answers pane shows the pending action (`⏸`), then its outcome (`⚙` running, `✓` done,
`✗` denied/failed).

### Microphone sharing

Capture is **shared mode** — other apps (Zoom, Meet, games) keep full use of the same mic
while earpiece listens. For hard separation, use two mics via `--mic-device`.

## Development

```sh
uv sync
uv run pytest          # unit tests (no audio hardware, no network)
uv run ruff check .    # lint
uv run earpiece run "..." --debug-dump-wav   # dump captured audio to debug_audio/*.wav
```

Logs go to `earpiece.log` (`-v` for debug). Architecture and design rationale live in
[AGENTS.md](AGENTS.md). STT and TTS engines are small registries
(`src/earpiece/stt/`, `src/earpiece/output/tts/`) — adding an engine is one module with a
`@register` decorator.

## A note on consent

Recording calls may require the other party's consent depending on your jurisdiction.
This tool is for assisting **your own** conversations — know your local rules.
