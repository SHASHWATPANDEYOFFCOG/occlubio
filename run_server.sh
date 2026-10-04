#!/usr/bin/env bash
# Launch the occlubio surveillance console (FastAPI + SQLite + FAISS) on macOS / Linux.
# Mirrors run_server.ps1. Examples:
#   ./run_server.sh
#   ./run_server.sh --port 8080 --reload
#   ./run_server.sh --authority-code my-secret --no-browser
#   ./run_server.sh --bind 0.0.0.0 --cert lan.pem --key lan-key.pem   # HTTPS on the LAN (phones/iPad)
set -euo pipefail

PORT=8001
BIND=127.0.0.1
AUTHORITY_CODE=""
DB_URL=""
CERT=""
KEY=""
RELOAD=0
NO_BROWSER=0
SKIP_DEP_CHECK=0

usage() { sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)            PORT="$2"; shift 2 ;;
        --bind)            BIND="$2"; shift 2 ;;
        --authority-code)  AUTHORITY_CODE="$2"; shift 2 ;;
        --db)              DB_URL="$2"; shift 2 ;;
        --cert)            CERT="$2"; shift 2 ;;
        --key)             KEY="$2"; shift 2 ;;
        --reload)          RELOAD=1; shift ;;
        --no-browser)      NO_BROWSER=1; shift ;;
        --skip-dep-check)  SKIP_DEP_CHECK=1; shift ;;
        -h|--help)         usage 0 ;;
        *) echo "unknown option: $1" >&2; usage 2 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
red()    { printf '\033[31m%s\033[0m\n' "$*"; }
green()  { printf '\033[32m%s\033[0m\n' "$*"; }

# --- venv -------------------------------------------------------------------
if [[ ! -x "$PY" ]]; then
    yellow "[setup] no .venv found - creating one"
    SYS_PY=""
    # insightface / onnxruntime / faiss wheels cover 3.10-3.12; prefer those over a newer default python3.
    for cand in python3.12 python3.11 python3.10 python3; do
        if command -v "$cand" >/dev/null 2>&1; then SYS_PY="$(command -v "$cand")"; break; fi
    done
    [[ -n "$SYS_PY" ]] || { red "python3 not found. Install Python 3.10-3.12 (e.g. brew install python@3.12)."; exit 1; }
    VER="$("$SYS_PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    case "$VER" in
        3.10|3.11|3.12) ;;
        *) yellow "[setup] WARNING: python $VER - the CV wheels only cover 3.10-3.12" ;;
    esac
    "$SYS_PY" -m venv "$ROOT/.venv"
    "$PY" -m pip install --upgrade pip
    yellow "[setup] installing occlubio[infer,api] (first run pulls ~300MB of models later)"
    "$PY" -m pip install -e ".[infer,api]"
fi

# --- dependencies -----------------------------------------------------------
if [[ $SKIP_DEP_CHECK -eq 0 ]]; then
    if ! "$PY" -c "import fastapi, uvicorn, sqlalchemy, insightface, onnxruntime, faiss, occlubio" 2>/dev/null; then
        yellow "[setup] missing dependencies - installing occlubio[infer,api]"
        "$PY" -m pip install -e ".[infer,api]"
    fi
fi

# --- port -------------------------------------------------------------------
if "$PY" -c "import socket,sys; s=socket.socket(); sys.exit(0 if s.connect_ex(('127.0.0.1', $PORT)) == 0 else 1)"; then
    OWNER=""
    if command -v lsof >/dev/null 2>&1; then
        OWNER="$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | awk 'NR==2 {print $1" (pid "$2")"}')"
    fi
    red "port $PORT is already served${OWNER:+ by $OWNER}. Use --port <n>, or stop that process."
    exit 1
fi

# --- environment ------------------------------------------------------------
[[ -n "$AUTHORITY_CODE" ]] && export OCCLUBIO_AUTHORITY_CODE="$AUTHORITY_CODE"
[[ -n "$DB_URL" ]] && export OCCLUBIO_DB="$DB_URL"

if [[ "$BIND" != "127.0.0.1" && "$BIND" != "localhost" ]]; then
    red "[warn] binding $BIND exposes a biometric system beyond this machine."
    red "[warn] See the responsible-use section of OCCLUSION_ROBUST_FR_ARCHITECTURE.md."
fi

SCHEME=http
TLS_ARGS=()
if [[ -n "$CERT" || -n "$KEY" ]]; then
    [[ -f "$CERT" && -f "$KEY" ]] || { red "--cert and --key must both point to existing files"; exit 1; }
    SCHEME=https
    TLS_ARGS=(--ssl-certfile "$CERT" --ssl-keyfile "$KEY")
fi

HOST_FOR_URL="$BIND"
[[ "$BIND" == "0.0.0.0" ]] && HOST_FOR_URL=127.0.0.1
URL="$SCHEME://$HOST_FOR_URL:$PORT/"

# --- browser ----------------------------------------------------------------
POLLER=""
if [[ $NO_BROWSER -eq 0 ]]; then
    if [[ "$(uname -s)" == "Darwin" ]]; then OPENER=open; else OPENER=xdg-open; fi
    if command -v "$OPENER" >/dev/null 2>&1; then
        (
            for _ in $(seq 1 90); do
                if curl -fsk -o /dev/null --max-time 3 "${URL}login"; then "$OPENER" "$URL" >/dev/null 2>&1; exit 0; fi
                sleep 1
            done
        ) &
        POLLER=$!
    fi
fi
cleanup() {
    [[ -n "$POLLER" ]] && kill "$POLLER" 2>/dev/null || true
    printf '\n\033[90m[occlubio] server stopped.\033[0m\n'
}
trap cleanup EXIT

# --- serve ------------------------------------------------------------------
echo
green "  occlubio console  ->  $URL"
green "  API docs          ->  ${URL}docs"
printf '\033[90m  Ctrl+C to stop\033[0m\n\n'

ARGS=(-m uvicorn occlubio.api.app:app --host "$BIND" --port "$PORT")
[[ $RELOAD -eq 1 ]] && ARGS+=(--reload)
"$PY" "${ARGS[@]}" ${TLS_ARGS[@]+"${TLS_ARGS[@]}"}
