#!/usr/bin/env bash
# Run earpiece with local whisper STT (Docker) + your own vLLM/OpenAI-compatible LLM server.
#
#   LLM_BASE_URL=https://my-vllm-host/v1 scripts/run-local.sh "help me with tech trivia" --eager
#
# Environment:
#   LLM_BASE_URL          OpenAI-compatible /chat/completions endpoint (required)
#   LLM_API_KEY           default "local" (vLLM accepts any non-empty key unless --api-key set)
#   LLM_RESPONDER_MODEL   default: auto-discovered from $LLM_BASE_URL/models
#   LLM_WATCHER_MODEL     default: same as responder
#   LLM_VERIFY_TLS        set "false" for self-signed internal certs
#   STT_MODEL             default Systran/faster-whisper-small (CPU-friendly);
#                         try Systran/faster-distil-whisper-large-v3 for accuracy
set -euo pipefail
cd "$(dirname "$0")/.."

# Pull defaults from the earpiece config file (`earpiece configure`); env vars win.
config="${EARPIECE_CONFIG:-$HOME/.config/earpiece/config.toml}"
if [ -f "$config" ]; then
  eval "$(python3 - "$config" <<'PY'
import os, shlex, sys, tomllib
with open(sys.argv[1], "rb") as f:
    for key, value in tomllib.load(f).items():
        if key not in os.environ:
            if isinstance(value, bool):
                value = "true" if value else "false"
            print(f"export {key}={shlex.quote(str(value))}")
PY
)"
fi

: "${LLM_BASE_URL:?set LLM_BASE_URL (or run: earpiece configure), e.g. https://my-vllm-host/v1}"
: "${LLM_API_KEY:=local}"
: "${STT_BASE_URL:=http://localhost:8001/v1}"
: "${STT_MODEL:=Systran/faster-whisper-small}"
export LLM_BASE_URL LLM_API_KEY STT_BASE_URL STT_MODEL

curl_flags=(-fsS)
[ "${LLM_VERIFY_TLS:-true}" = "false" ] && curl_flags+=(-k)

# Auto-discover the served model when not pinned (vLLM lists it at /v1/models).
if [ -z "${LLM_RESPONDER_MODEL:-}" ]; then
  if ! LLM_RESPONDER_MODEL="$(curl "${curl_flags[@]}" "$LLM_BASE_URL/models" \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"; then
    echo "error: could not list models at $LLM_BASE_URL/models —" \
         "check connectivity/VPN, or set LLM_RESPONDER_MODEL explicitly" >&2
    exit 1
  fi
  export LLM_RESPONDER_MODEL
fi

echo "LLM:     $LLM_BASE_URL  (model: $LLM_RESPONDER_MODEL)"
echo "Whisper: $STT_BASE_URL  (model: $STT_MODEL, local Docker)"

# Start the local whisper server and wait until it answers.
docker compose -f local/docker-compose.yml up --detach
health_url="${STT_BASE_URL%/v1}/health"
printf "waiting for whisper server"
for _ in $(seq 1 60); do
  if curl -fsS "$health_url" >/dev/null 2>&1; then
    echo " ready"
    break
  fi
  printf "."
  sleep 1
done
if ! curl -fsS "$health_url" >/dev/null 2>&1; then
  echo " timed out — check: docker compose -f local/docker-compose.yml logs" >&2
  exit 1
fi

# speaches does NOT auto-download models — install the STT model once.
installed="$(curl -fsS "$STT_BASE_URL/models" \
  | python3 -c 'import json,sys; print("\n".join(m["id"] for m in json.load(sys.stdin)["data"]))')"
if ! printf '%s\n' "$installed" | grep -qxF "$STT_MODEL"; then
  echo "downloading $STT_MODEL into the whisper container (one-time)…"
  curl -fsS -X POST "$STT_BASE_URL/models/$STT_MODEL"
  echo
fi

exec uv run earpiece run "$@" --stt whisper
