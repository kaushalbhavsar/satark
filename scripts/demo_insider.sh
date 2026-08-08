#!/usr/bin/env bash
# Scripted SATARK working demo for terminal recording.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.local/bin:${PATH}"
export FORCE_COLOR=1
export TERM="${TERM:-xterm-256color}"

pause() {
  sleep "${1:-1.2}"
}

type_line() {
  local text="$1"
  local i
  for ((i = 0; i < ${#text}; i++)); do
    printf '%s' "${text:i:1}"
    sleep 0.018
  done
  printf '\n'
  sleep 0.25
}

run() {
  type_line "\$ $*"
  # shellcheck disable=SC2086
  eval "$@"
  pause 1.4
}

clear 2>/dev/null || true
echo
echo "SATARK — working demo"
echo "Insider-threat analysis against sample USB/file telemetry"
echo
pause 1.0

run "uv run satark version"
run "uv run satark list-plugins"
run "uv run satark analyze -p insider -d examples/data/sample_insider.csv --threshold 0.5"

echo
echo "Demo complete: elevated USB and file-activity findings with explainable scores."
pause 2.0
