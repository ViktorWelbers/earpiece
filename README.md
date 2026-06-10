# earpiece

A real-time "voice of god" assistant for your computer. Start it with a mission —
*"help me in this sales discussion"*, *"help me with this tech trivia"* — and it listens to
your **microphone** and your **system audio** (the other side of the call / a video), keeps a
live speaker-tagged transcript, and streams back guidance as **live text** and optionally as
**voice in your earpiece**. If the conversation moves on mid-answer, the answer is cancelled
mid-sentence and replaced — it behaves like a real-time participant, not a batch chatbot.

See [PLAN.md](PLAN.md) for the full architecture.

## Requirements

- macOS (Apple Silicon tested), Python 3.12+, [uv](https://docs.astral.sh/uv/)
- STT: a [Deepgram](https://deepgram.com) API key (`--stt deepgram`), **or** any
  OpenAI-compatible `/audio/transcriptions` endpoint (`--stt whisper`) — e.g. vLLM serving a
  Whisper model, [speaches](https://speaches.ai), or OpenAI's `whisper-1`
- LLM: any **OpenAI-compatible** chat-completions endpoint (OpenAI, Anthropic compat,
  OpenRouter, Groq, local Ollama/vLLM, ...)
- [BlackHole 2ch](https://github.com/ExistentialAudio/BlackHole) for system-audio capture

## Setup

### 1. Install

For development (run via `uv run earpiece ...`):

```sh
uv sync
```

Or install system-wide as an `earpiece` command (isolated venv, on your PATH):

```sh
uv tool install --editable .   # tracks this checkout — code changes apply immediately
# or: uv tool install .        # pinned snapshot; rerun with --reinstall to update
# or: pipx install .           # same idea via pipx
```

`uv build` produces a normal wheel in `dist/` if you want to `pip install` it elsewhere.

### 2. System-audio loopback (one-time)

```sh
brew install blackhole-2ch
```

Then in **Audio MIDI Setup** (`/Applications/Utilities`):

1. `+` → **Create Multi-Output Device**
2. Check **your real output** (speakers / earpiece) **and** **BlackHole 2ch**
3. Set the system output device to this Multi-Output Device

You hear everything normally; earpiece records the BlackHole side.

> **Important routing rule:** if you use `--voice`, the TTS output device (`--output-device`,
> e.g. your earbuds) must **not** be part of the Multi-Output Device — otherwise the assistant
> hears and transcribes itself. (There's a second software guard, but get the routing right.)

### 3. Configure

```sh
export LLM_API_KEY=sk-...
export LLM_BASE_URL=https://api.openai.com/v1     # any OpenAI-compatible endpoint
export LLM_RESPONDER_MODEL=...                    # the smart model (answers)
export LLM_WATCHER_MODEL=...                      # a fast cheap model (turn-taking decisions)
export DEEPGRAM_API_KEY=...
```

Optional: `LLM_WATCHER_BASE_URL` / `LLM_WATCHER_API_KEY` to put the watcher on a different
provider, `LLM_JSON_SCHEMA=false` for providers without structured-output support.

For `--stt whisper` instead of Deepgram:

```sh
export STT_BASE_URL=http://localhost:8001/v1      # any /audio/transcriptions endpoint
export STT_MODEL=openai/whisper-large-v3-turbo
export STT_API_KEY=local                          # optional; local servers ignore it
```

### Local mode: whisper in Docker + your vLLM server

`scripts/run-local.sh` is the one-command path for running without cloud STT: whisper runs in
a local Docker container ([speaches](https://speaches.ai), CPU image, OpenAI-compatible
`/v1/audio/transcriptions` on `localhost:8001`), the LLM slots point at an existing vLLM
server (`LLM_BASE_URL`, model auto-discovered from `/v1/models`), and voice output stays
local via `--voice say`:

```sh
# needs Docker Desktop / OrbStack running
LLM_BASE_URL=https://my-vllm-host/v1 \
scripts/run-local.sh "help me with tech trivia" --eager --voice say

# more overrides via env:
LLM_BASE_URL=https://my-vllm-host/v1 \
LLM_VERIFY_TLS=false \                          # self-signed internal cert
STT_MODEL=Systran/faster-distil-whisper-large-v3 \
scripts/run-local.sh "help me in this sales discussion"
```

Details:

- The first transcription downloads the whisper model into the `hf-hub-cache` Docker volume —
  expect a delay once, then it's cached.
- Docker on macOS is CPU-only; `Systran/faster-whisper-small` (default) keeps latency
  realtime-ish. Bigger models = better accuracy, slower finalization.
- The whisper backend does utterance endpointing locally (energy VAD) and posts each finished
  segment as a small WAV — slightly higher latency than Deepgram's streaming endpoint, fine
  for testing.
- vLLM supports structured outputs (`LLM_JSON_SCHEMA=true`, the default); set it to `false`
  if your build rejects `response_format: json_schema`.
- Stop the whisper container with `docker compose -f local/docker-compose.yml down`.

## Usage

```sh
# list audio devices (find your mic / loopback / earpiece indices)
uv run earpiece devices

# text-only, watcher decides when to speak
uv run earpiece run "help me in this sales discussion"

# trivia mode: answer every utterance from the other side, speak into the earpiece
uv run earpiece run "help me with this tech trivia" --eager --voice say

# explicit devices (two-mic setup: mic 6 feeds earpiece, you reply on another mic)
uv run earpiece run "..." --mic-device 6 --system-device BlackHole --output-device 1
```

Hotkeys while running: `space` = force an answer now · `m` = mute mic · `v` = toggle voice ·
`q` = quit.

### Microphone sharing

Capture is **shared mode** — other apps (Zoom, Meet, games) keep full use of the same mic
while earpiece listens. If you prefer hard separation, use two mics via `--mic-device`.

## Development

```sh
uv run pytest          # unit tests (no audio hardware, no network)
uv run ruff check .    # lint
uv run earpiece run "..." --debug-dump-wav   # dump captured audio to debug_audio/*.wav
```

Logs go to `earpiece.log` (`-v` for debug).

## A note on consent

Recording calls may require the other party's consent depending on your jurisdiction.
This tool is for assisting **your own** conversations — know your local rules.
