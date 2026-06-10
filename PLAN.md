# Earpiece — Real-Time "Voice of God" Assistant

> Implementation plan. Status: **approved design, pre-implementation.**
> Target platform: macOS (Apple Silicon), Python 3.12+, managed with `uv`.

---

## 1. What this is

A desktop program that acts as a real-time copilot for live situations. The operator starts it
with a **mission** ("help me in this sales discussion", "help me with tech trivia"), and it:

1. Continuously captures **microphone** (the operator) and **system audio** (the other side of a
   call, a video, a game), and — in Phase 2 — **periodic screen captures**.
2. Transcribes audio in real time into a rolling, speaker-tagged transcript.
3. Decides, per utterance, *whether* it should speak up (stay silent / respond / interrupt its
   own in-flight answer).
4. Streams the answer to the operator as **live text** and optionally as **voice in the
   earpiece** — and cancels mid-sentence if the conversation moves on, so the assistance feels
   like a real-time participant, not a batch Q&A bot.

### Key architectural constraint

Standard LLM chat APIs (OpenAI-compatible chat completions) have **no realtime audio input**.
The realtime feel must be assembled from parts:

```
streaming STT  →  rolling text transcript (+ image frames in Phase 2)
              →  streaming LLM responses
              →  streaming TTS
```

…with **cancellation propagated through every stage**. Interruption is the product; everything
else is plumbing.

### Decisions already made

| Topic | Decision |
|---|---|
| Stack | Python, asyncio pipeline, `uv`-managed project |
| LLM | **No hard provider dependency.** OpenAI-compatible chat-completions client (`openai` SDK + configurable `base_url`). Works with OpenAI, Anthropic compat endpoint, OpenRouter, Groq, Ollama, vLLM. |
| Output | Live text (terminal first, overlay later) **and** optional voice (TTS), both interruptible |
| STT | Pluggable engine interface. Deepgram cloud streaming, plus `whisper` via any OpenAI-compatible `/audio/transcriptions` endpoint (vLLM, speaches, OpenAI) — nothing baked into the binary |
| Phasing | Phase 1 = audio-only core loop. Phase 2a = fully local mode (whisper API + local LLM endpoint). Phase 2b = screen capture, ElevenLabs, overlay window |
| Mic access | **Must not block the mic for other apps** (shared-mode capture), and every device is selectable so a two-mic setup works |

---

## 2. Architecture overview

```
 ┌─ mic (sounddevice, shared mode) ──► STT stream A  ── "ME" ───┐
 ┌─ system audio (BlackHole tap) ────► STT stream B  ── "THEM" ─┤
 │                                                              ▼
 │                                            TranscriptStore (rolling, speaker-
 │                                            tagged, timestamped utterances)
 │                                                              │ utterance-final events
 │                                                              ▼
 │                                     Watcher (fast/cheap model, structured output)
 │                                     per utterance: stay_silent | respond
 │                                                  | interrupt_and_respond
 │                                                              │
 │                                                              ▼
 │                                     Responder (smart model, streaming chat
 │                                     completion over the transcript conversation)
 │                                            │ text deltas
 │                              ┌─────────────┴──────────────┐
 │                              ▼                            ▼
 │                     Console view (rich.Live)      TTS engine (pluggable)
 │                                                            │
 └────────────── playback routed to earpiece device ◄─────────┘
                 (excluded from capture path — no feedback loop)
```

### 2.1 Why two model slots (Watcher + Responder)

*When to speak* and *what to say* are different problems:

- **Watcher** — a fast, cheap model called once per finalized utterance with the recent
  transcript (and the partial in-flight answer, if any). Returns a strict-JSON decision:
  `{action, reason, urgency}`. Target latency ~300–500 ms.
- **Responder** — the smart model. Streams the actual guidance over the full (windowed)
  conversation. Target time-to-first-token ~500–800 ms.

Both are just configured model slots on the same OpenAI-compatible client; they may point at
different providers (e.g. Groq for the watcher, OpenAI for the responder).

An `--eager` flag bypasses the Watcher entirely (respond on every "THEM" utterance end) for
trivia-style usage where the answer is always wanted.

### 2.2 Speaker attribution for free

Mic and system audio are captured as **separate channels**, each with its own STT stream.
"ME" vs "THEM" tagging therefore requires **no diarization** — the channel *is* the speaker.

### 2.3 Interruption semantics (the core requirement)

1. A new final utterance arrives while the Responder is streaming.
2. The Watcher — whose prompt includes the partial answer streamed so far — returns
   `interrupt_and_respond` if the new input makes the in-flight answer stale.
3. The Orchestrator then, in order:
   a. cancels the asyncio task wrapping the streaming completion (closes the HTTP stream),
   b. cancels TTS playback at the next word boundary and flushes its queue,
   c. records the partial answer in history with an explicit
      `[interrupted — conversation moved on]` marker,
   d. starts a fresh Responder request over the updated transcript.
4. **Barge-in:** voice-activity on the mic channel always pauses TTS playback instantly —
   the assistant never talks over the operator. Resume vs discard is the Watcher's call.

### 2.4 Latency budget (utterance end → first words in ear)

| Stage | Budget |
|---|---|
| STT finalization (endpointing) | ~300 ms |
| Watcher decision | ~300–500 ms |
| Responder TTFT | ~500–800 ms |
| TTS first audio chunk | ~200 ms |
| **Total** | **~1.3–1.8 s** (`--eager` drops the Watcher hop) |

---

## 3. Project layout

Managed with **uv**: `uv init`, `uv add`, committed `uv.lock`. Dev commands are `uv run earpiece …`
and `uv run pytest`. Optional features are uv extras (currently `--extra elevenlabs`).

```
earpiece/
  pyproject.toml                 # uv-managed; console script `earpiece`
  uv.lock
  README.md                      # device setup (BlackHole), usage, troubleshooting
  PLAN.md                        # this file
  earpiece/
    __init__.py
    cli.py                       # arg parsing, device validation, hotkeys, wiring
    config.py                    # Settings: env + flags → typed config object
    llm.py                       # LLMClient: openai SDK against any OpenAI-compatible base_url
    events.py                    # frozen dataclasses passed between stages
    orchestrator.py              # asyncio task graph, queues, cancellation authority
    audio/
      __init__.py
      capture.py                 # MicCapture, SystemAudioCapture (sounddevice InputStream)
      playback.py                # PlaybackQueue: device-routed, cancellable, barge-in aware
      vad.py                     # lightweight energy/webrtc VAD for barge-in detection
    stt/
      __init__.py
      base.py                    # STTEngine protocol + registry (lazy engine loading)
      deepgram.py                # streaming websocket, interim+final, endpointing
      whisper_api.py             # local VAD endpointing + OpenAI-compatible transcriptions API
    brain/
      __init__.py
      transcript.py              # TranscriptStore + chat-messages builder
      watcher.py                 # decision calls (structured output w/ JSON fallback)
      responder.py               # streaming answers, interruption bookkeeping
      prompts.py                 # frozen system prompts (cache-safe), mission templating
    output/
      __init__.py
      console.py                 # rich.Live three-pane view
      tts/
        __init__.py
        base.py                  # TTSEngine protocol + registry
        macos_say.py             # zero-setup local default (`say` subprocess)
        elevenlabs.py            # Phase 2: streaming websocket TTS
    screen/                      # Phase 2
      __init__.py
      capture.py                 # periodic screenshot, downscale, phash dedupe
  tests/
    fakes.py                     # FakeSTT, FakeLLM, FakeTTS — fully scripted, no I/O
    test_transcript.py
    test_watcher.py
    test_interruption.py
    test_config.py
    test_whisper_api.py
```

Dependencies (Phase 1): `openai`, `sounddevice`, `numpy`, `websockets`, `rich`, `pydantic`,
`typer` (CLI). Dev: `pytest`, `pytest-asyncio`, `ruff`. Extras: `elevenlabs`.

---

## 4. Module specifications

### 4.1 `events.py` — the pipeline vocabulary

Frozen dataclasses; every queue between stages carries exactly one of these:

```python
Speaker = Literal["ME", "THEM"]

@dataclass(frozen=True)
class AudioChunk:            # capture → STT
    source: Speaker
    pcm: bytes               # 16 kHz mono s16le
    ts: float                # monotonic capture time

@dataclass(frozen=True)
class TranscriptEvent:       # STT → transcript store
    source: Speaker
    text: str
    is_final: bool           # interim results update the UI only; finals drive the brain
    started_at: float
    ended_at: float

class Action(StrEnum):
    STAY_SILENT = "stay_silent"
    RESPOND = "respond"
    INTERRUPT_AND_RESPOND = "interrupt_and_respond"

class Decision(BaseModel):   # Watcher output (pydantic — parsed from JSON)
    action: Action
    reason: str              # one sentence, for the status bar / logs
    urgency: Literal["low", "normal", "high"]

@dataclass(frozen=True)
class AnswerDelta:           # responder → console + TTS
    answer_id: str
    text: str

@dataclass(frozen=True)
class AnswerEnd:
    answer_id: str
    interrupted: bool
```

### 4.2 `config.py`

A single `Settings` object resolved from env vars + CLI flags (flags win):

| Setting | Env | Default |
|---|---|---|
| LLM base URL | `LLM_BASE_URL` | `https://api.openai.com/v1` |
| LLM API key | `LLM_API_KEY` | — (required) |
| Responder model | `LLM_RESPONDER_MODEL` | — (required) |
| Watcher model | `LLM_WATCHER_MODEL` | falls back to responder model |
| Watcher base URL/key | `LLM_WATCHER_BASE_URL` / `LLM_WATCHER_API_KEY` | falls back to main |
| Supports json_schema | `LLM_JSON_SCHEMA` | `true` (set `false` for providers without it) |
| Deepgram key | `DEEPGRAM_API_KEY` | required when `--stt deepgram` |
| Mic device | `--mic-device` | system default input |
| System-audio device | `--system-device` | auto-detect "BlackHole" |
| TTS output device | `--output-device` | system default output |
| STT engine | `--stt` | `deepgram` |
| TTS engine | `--voice` | off; `say` / `elevenlabs` |
| Eager mode | `--eager` | off |

`earpiece devices` subcommand prints `sounddevice.query_devices()` in a table with indices and
marks the auto-detected candidates.

### 4.3 `llm.py` — provider-agnostic LLM client

Thin wrapper around `openai.AsyncOpenAI(base_url=…, api_key=…)`:

- `stream_chat(messages, model) -> AsyncIterator[str]` — yields content deltas; raises
  `asyncio.CancelledError` cleanly through (the stream's `close()` runs in a `finally`).
- `structured(messages, model, schema: type[BaseModel]) -> BaseModel` — uses
  `response_format={"type": "json_schema", …}` when `LLM_JSON_SCHEMA=true`; otherwise appends a
  "reply ONLY with JSON matching …" instruction and parses with pydantic, with **one retry** on
  validation failure. Capability is a config flag, not runtime sniffing.
- Two pre-bound handles exposed: `llm.responder` and `llm.watcher` (model + endpoint resolved
  from Settings).

**Cache-friendliness rule (provider-independent):** conversation prefix must be byte-stable —
frozen system prompt, **no timestamps/UUIDs in the system prompt**, append-only history. That way
OpenAI automatic prefix caching / Anthropic prompt caching / vLLM prefix cache all work without
provider-specific code. Clock time lives only inside message *content* lines.

### 4.4 `audio/capture.py`

- `MicCapture(device)` and `SystemAudioCapture(device)` — both wrap a
  `sounddevice.InputStream` (16 kHz, mono, int16, blocksize ~320 samples = 20 ms) whose callback
  pushes `AudioChunk`s onto an `asyncio.Queue` via `loop.call_soon_threadsafe`.
- **Shared-mode requirement:** never request exclusive/hog mode. PortAudio on macOS opens
  Core Audio devices shared by default — other apps (Zoom, Meet, games) keep working. This is a
  documented invariant, enforced by code review + the smoke checklist (§8).
- Startup validation: requested devices exist, have input channels, and the system-audio device
  looks like a loopback (warn if it's a physical mic — that usually means BlackHole isn't set up).

### 4.5 `audio/playback.py` + `audio/vad.py`

- `PlaybackQueue(device)` — plays PCM chunks from TTS on the **configured output device**
  (the earpiece), *not* the multi-output device that feeds capture. Exposes:
  - `enqueue(pcm)`, `cancel()` (drops queue, stops at the current buffer ≈ word boundary),
  - `gate(active: bool)` used by barge-in.
- `vad.py` — cheap energy-threshold VAD over mic chunks (upgradeable to `webrtcvad`).
  Mic speech ⇒ `PlaybackQueue.gate(True)` immediately (pause TTS, never talk over the operator).
- **Feedback-loop guards** (assistant must never transcribe itself):
  1. TTS routes to a device excluded from the capture path, and
  2. belt-and-suspenders: system-audio STT events are dropped while TTS is actively playing.

### 4.6 `stt/` — pluggable engines

```python
class STTEngine(Protocol):
    async def stream(self, audio: AsyncIterator[AudioChunk]) -> AsyncIterator[TranscriptEvent]: ...
```

- One engine instance **per channel** (two Deepgram websockets: ME and THEM).
- `deepgram.py`: raw websocket via `websockets` (no heavy SDK) —
  `wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=16000&interim_results=true&endpointing=300&smart_format=true`.
  Maps interim → `is_final=False`, `speech_final` → `is_final=True`. Reconnects with backoff;
  surfaces connection state to the status bar.
- `whisper_api.py` (`--stt whisper`): whisper over any **OpenAI-compatible
  `/audio/transcriptions` endpoint** — nothing baked into the binary. Local energy-VAD
  endpointing (with ~200 ms pre-roll so word onsets aren't clipped) buffers each utterance;
  finalized segments are posted as small WAVs to `STT_BASE_URL`. Long utterances get interim
  transcriptions every ~2.5 s. Works against vLLM serving a Whisper model, speaches /
  faster-whisper-server, LocalAI, or OpenAI's hosted `whisper-1` — fully local operation is
  just `STT_BASE_URL=http://localhost:…`. Config: `STT_BASE_URL`, `STT_MODEL`,
  `STT_API_KEY` (default `local`; local servers ignore it).
- Engine registry keyed by name with lazy module loading, so new engines are a single file +
  one mapping line, and optional dependencies fail with an actionable install hint.

### 4.7 `brain/transcript.py`

- `TranscriptStore.add(event)` — merges interim/final events into utterances
  `(speaker, text, t_start, t_end)`; interim events only update the live console.
- `as_messages()` — builds the chat history for LLM calls:
  - one frozen system message (persona + mission + output-style rules),
  - transcript deltas as user messages, each line `"[hh:mm:ss] THEM: …"`,
  - prior assistant answers as assistant messages
    (interrupted ones suffixed `"[interrupted — conversation moved on]"`).
  - **Append-only.** New utterances since the last call become one new user message; history is
    never rewritten (keeps provider-side prefix caches valid).
- **Sliding window:** beyond ~60 K tokens (estimated `len/4`), summarize the oldest half via the
  watcher model into a single `"[earlier conversation summary] …"` user message. This is the only
  history rewrite, and it intentionally resets the cache prefix once per long session.

### 4.8 `brain/watcher.py`

Called once per **final** utterance (skipped entirely in `--eager` mode). Prompt contains:

- a short role description ("you decide if a discreet real-time assistant should speak"),
- the mission,
- the last ~15 transcript lines,
- the in-flight partial answer, if any (so it can choose `interrupt_and_respond`),
- decision guidance: silence is the default; respond when the operator clearly benefits
  (a question was asked, a factual claim needs checking, an objection needs handling);
  interrupt only when the topic genuinely moved.

Returns a `Decision`. On any LLM/parse error: fail safe to `STAY_SILENT` and log.
ME-utterances also flow through (the operator may address the assistant directly:
"what was that thing called…"), but with guidance that ME usually doesn't need an answer.

### 4.9 `brain/responder.py`

- `respond(history) -> answer_id` spawns the streaming task; deltas fan out to console + TTS
  (sentence-buffered for TTS: split on `.?!` ≥ ~40 chars so speech doesn't stutter).
- Owns answer bookkeeping: on natural end append the full answer as an assistant message; on
  cancellation append what was streamed + the interrupted marker.
- Style is enforced by the system prompt (§4.11): 1–3 sentences for spoken cues, bullets allowed
  in text-only mode; never preamble ("Here's what you could say" is banned — output *is* the cue).

### 4.10 `orchestrator.py`

The only component that owns tasks and cancellation:

```
queues:  mic_audio, sys_audio  →  stt tasks  →  transcript_events
tasks:   mic_capture, sys_capture, stt_me, stt_them, brain_loop, console, playback
state:   current_answer: {id, task, text_so_far} | None
```

`brain_loop` (sequential, the decision point):

```
for each final TranscriptEvent:
    transcript.add(event)
    if eager and event.source == THEM:          decision = RESPOND-or-INTERRUPT (auto)
    else:                                       decision = await watcher.decide(...)
    match decision.action:
        STAY_SILENT            → continue
        RESPOND                → if no answer in flight: start responder task
        INTERRUPT_AND_RESPOND  → cancel current answer task → flush TTS → mark history
                                 → start responder task
```

Watcher calls are serialized (one at a time); utterances arriving during a watcher call are
coalesced into the next call. Push-to-ask hotkey injects a synthetic `RESPOND` decision.
Shutdown: cancel all tasks, close streams, drain queues — `q` exits cleanly in <1 s.

### 4.11 `brain/prompts.py`

- `SYSTEM_TEMPLATE` (frozen at session start, mission interpolated once):
  persona ("discreet real-time copilot whispering in the operator's ear"), the mission verbatim,
  transcript format explanation (ME/THEM), output style rules (short, directly speakable,
  no meta-commentary, no preamble), language: respond in the conversation's language.
- `WATCHER_TEMPLATE`: as in §4.8, with the JSON contract appended only in fallback mode.
- Templates contain **no dynamic values** other than the mission (cache safety).

### 4.12 `output/console.py`

`rich.Live` layout, ~10 fps:

```
┌─ transcript ────────────────────────────┬─ answer ───────────────────────┐
│ [14:02:03] THEM: …                      │ (streaming answer text)        │
│ [14:02:11] ME: …                        │                                │
│ ▌interim line in dim style              │                                │
├─────────────────────────────────────────┴────────────────────────────────┤
│ status: ● mic ● sys ● stt ● llm | mode: normal | last decision: respond  │
│         (“why”: watcher reason) | tokens: prompt 12.3k (cached 11.8k)    │
└──────────────────────────────────────────────────────────────────────────┘
```

Hotkeys (raw tty reader task): `space` push-to-ask, `m` mute mic, `v` toggle voice, `q` quit.

### 4.13 `output/tts/`

```python
class TTSEngine(Protocol):
    async def synthesize(self, text: str) -> AsyncIterator[bytes]   # PCM chunks
```

- `macos_say.py` — default, zero setup: `say -o` to AIFF in a temp file per sentence chunk →
  decode → `PlaybackQueue`. Latency is fine for short cues; fully local.
- `elevenlabs.py` (Phase 2) — streaming websocket, sub-300 ms first chunk.
- Cancellation contract: orchestrator calls `PlaybackQueue.cancel()`; any in-flight synthesis
  task is cancelled alongside the answer task.

### 4.14 `screen/capture.py` (Phase 2)

- Every ~3 s: `screencapture -x -t jpg` (or CoreGraphics via pyobjc later) → downscale longest
  edge to ≤1568 px → perceptual hash; skip if unchanged from last sent frame.
- Latest frame only (not a backlog) is attached to Responder requests as a standard
  OpenAI-compatible `image_url` content part (base64 data URI). Per-provider capability flag —
  local models without vision simply never get frames.
- The Watcher gains a `look_at_screen` hint field so "look at this" in the transcript can force
  a fresh capture before responding.

---

## 5. System-audio capture on macOS (setup)

**BlackHole 2ch** (free virtual device) + a **Multi-Output Device**:

1. `brew install blackhole-2ch`
2. Audio MIDI Setup → create Multi-Output Device = {real output (speakers/earpiece), BlackHole 2ch}
3. Set system output to the Multi-Output Device — the user hears everything normally, and the
   app records the BlackHole side via `sounddevice`.

README documents this with screenshots; `cli.py` validates devices at startup and prints
actionable errors ("BlackHole not found — run: brew install blackhole-2ch …").

**Routing invariants:**

- TTS output device (earpiece) is **not** part of the Multi-Output Device → the assistant never
  hears itself.
- Mic capture is shared-mode → other apps keep full mic access. Two-mic setups are supported via
  `--mic-device` (e.g. mic A feeds earpiece, mic B is used to reply in a trivia call).

---

## 6. Sequence walkthroughs

### 6.1 Happy path ("tech trivia" with `--eager`)

```
THEM (video): "…what year was the transistor invented?"
  → Deepgram speech_final (≈300ms) → TranscriptStore
  → eager: skip watcher → responder.stream()
  → console starts printing ≈700ms later: "1947 — Bell Labs (Bardeen, Brattain, Shockley)."
  → first TTS words in ear ≈1.2s after the question ended
```

### 6.2 Interruption ("sales discussion")

```
THEM: "What does the integration timeline look like?"
  → watcher: respond → responder streams "Typical rollout is 4–6 weeks…"
THEM (2s later): "—actually, hold on, the bigger issue for us is pricing."
  → watcher sees partial answer + new utterance → interrupt_and_respond
  → orchestrator: cancel stream task → TTS stops at word boundary → history gets
    "Typical rollout is 4–6 weeks… [interrupted — conversation moved on]"
  → new stream: "On pricing: lead with the platform fee structure…"
  → total gap from their pivot to new first words: ≈1.5s
```

### 6.3 Barge-in

```
TTS speaking a cue → operator starts talking (mic VAD fires)
  → PlaybackQueue.gate(True) within one 20ms block — TTS pauses instantly
  → operator finishes → watcher decides: resume remaining cue, or discard (default: discard
    if >5s elapsed or topic moved)
```

---

## 7. Milestones (implementation order)

Each milestone is independently runnable/testable.

| # | Milestone | Proves |
|---|---|---|
| M1 | Project scaffold: uv init, pyproject, `events.py`, `config.py`, `devices` subcommand | `uv run earpiece devices` lists hardware |
| M2 | Capture: mic + system audio → wav dump debug flag | both channels record; mic stays shared |
| M3 | STT: Deepgram engine, dual websocket, live transcript in console | real-time ME/THEM transcript |
| M4 | Brain (text-only): transcript store, prompts, responder streaming into console; `--eager` | mission-driven streamed answers |
| M5 | Watcher + interruption: decisions, cancellation path, history markers | natural turn-taking, §6.2 works |
| M6 | TTS: `say` engine, PlaybackQueue, barge-in VAD, feedback guards | voice in earpiece, interruptible |
| M7 | Polish: hotkeys, status bar (incl. cached-token readout), reconnects, sliding window | usable end-to-end v1 |
| P2a | Fully local mode: `whisper_api` STT against any `/audio/transcriptions` endpoint (vLLM/speaches) — **done** | no-cloud testing loop |
| P2b | Phase 2: screen capture, elevenlabs, overlay window | full multimodal |

---

## 8. Verification

1. **Unit (no audio hardware, no API):** `tests/fakes.py` provides scripted `FakeSTT` /
   `FakeLLM` / `FakeTTS`. Assertions:
   - transcript tagging + interim/final merging (`test_transcript.py`),
   - watcher JSON parsing incl. fallback + fail-safe-to-silent (`test_watcher.py`),
   - `interrupt_and_respond` cancels the stream task, flushes TTS, writes the interrupted
     marker, and starts exactly one new answer (`test_interruption.py`),
   - config precedence env < flag (`test_config.py`).
2. **Live smoke test:** play a YouTube interview (system audio) while speaking into the mic;
   `uv run earpiece "help me with tech trivia"`; confirm dual-channel ME/THEM transcript,
   streamed answers, and cached-prefix usage in the status bar (when the provider reports it).
3. **Interruption test:** ask a question, change topic mid-answer — answer stops within ~1 s and
   the replacement addresses the new topic.
4. **Feedback test:** with `--voice say`, TTS output must never appear in the transcript.
5. **Shared-mic test:** while running, record in QuickTime (same mic) — both apps receive audio.
   Then select a second mic via `--mic-device` and confirm only that device feeds the transcript.

---

## 9. Risks / open questions

- **`say`-based TTS latency** may exceed budget for long cues → sentence-chunking mitigates;
  ElevenLabs streaming is the upgrade path.
- **Deepgram endpointing tuning** (300 ms) trades snappiness vs mid-sentence splits — make it a
  setting.
- **BlackHole setup friction** is the worst onboarding step; macOS 14.2+ Core Audio process taps
  could remove it later (native loopback) — out of scope for v1, noted as a future replacement
  for `SystemAudioCapture`.
- **Watcher cost/latency on chatty calls**: one cheap call per utterance; coalescing + eager
  mode are the levers.
- **Ethics/consent**: recording calls may require consent depending on jurisdiction — README
  gets a prominent note; this tool is for the operator's own conversations.
