#!/usr/bin/env bash
# mempalace-mine.sh — Run mempalace mine on Hermes sessions directory.
#
# Called by the cron schedule (hermes cron, not system cron).
# Configuration is read from ~/.hermes/config.yaml under memory.mempalace.mine:
#
#   memory:
#     provider: mempalace
#     mempalace:
#       mine:
#         sessions_dir: ~/.hermes/sessions      # default
#         wing: hermes-sessions                  # default
#         extract: exchange                      # default (exchange|general)
#         limit: 0                               # 0 = process all (default)
#
# Environment variable overrides:
#   MEMPALACE_MINE_DIR     — override sessions directory
#   MEMPALACE_MINE_WING    — override wing name
#   MEMPALACE_MINE_LIMIT   — override file limit
#   MEMPALACE_MINE_EXTRACT — override extraction strategy
#
# Output: JSON line to stdout when new files are processed.
#         Empty output when nothing new — cron stays silent.

set -euo pipefail

# Resolve mempalace binary
BINARY="${MEMALACE_BINARY:-}"
if [ -z "$BINARY" ]; then
    # Try from hermes config
    BINARY=$(python3 -c "
import yaml, os, pathlib
try:
    p = pathlib.Path(os.environ.get('HERMES_HOME', os.path.expanduser('~/.hermes'))) / 'config.yaml'
    if p.exists():
        with open(p) as f:
            cfg = yaml.safe_load(f) or {}
        mp = cfg.get('memory', {}).get('mempalace', {})
        print(mp.get('binary', ''))
except: pass
" 2>/dev/null || echo "")
fi

# Final fallback: PATH + common paths
if [ -z "$BINARY" ] || [ ! -x "$BINARY" ]; then
    BINARY=$(command -v mempalace 2>/dev/null || echo "")
fi
if [ -z "$BINARY" ]; then
    for candidate in \
        "$HOME/.local/bin/mempalace" \
        "/usr/local/bin/mempalace" \
        "/usr/bin/mempalace"; do
        if [ -x "$candidate" ]; then
            BINARY="$candidate"
            break
        fi
    done
fi

if [ -z "$BINARY" ]; then
    echo '{"error":"mempalace binary not found"}' >&2
    exit 1
fi

# Resolve sessions dir
SESSIONS_DIR="${MEMPALACE_MINE_DIR:-}"
if [ -z "$SESSIONS_DIR" ]; then
    SESSIONS_DIR=$(python3 -c "
import yaml, os, pathlib
try:
    p = pathlib.Path(os.environ.get('HERMES_HOME', os.path.expanduser('~/.hermes'))) / 'config.yaml'
    if p.exists():
        with open(p) as f:
            cfg = yaml.safe_load(f) or {}
        mp = cfg.get('memory', {}).get('mempalace', {}).get('mine', {})
        d = mp.get('sessions_dir', '')
        if d: print(os.path.expanduser(d))
except: pass
" 2>/dev/null || echo "")
fi
SESSIONS_DIR="${SESSIONS_DIR:-$HERMES_HOME/sessions}"
SESSIONS_DIR="${SESSIONS_DIR:-$HOME/.hermes/sessions}"

if [ ! -d "$SESSIONS_DIR" ]; then
    echo "{\"error\":\"sessions dir not found: $SESSIONS_DIR\"}" >&2
    exit 1
fi

# Resolve other params from config
WING="${MEMPALACE_MINE_WING:-}"
if [ -z "$WING" ]; then
    WING=$(python3 -c "
import yaml, os, pathlib
try:
    p = pathlib.Path(os.environ.get('HERMES_HOME', os.path.expanduser('~/.hermes'))) / 'config.yaml'
    if p.exists():
        with open(p) as f:
            cfg = yaml.safe_load(f) or {}
        print(cfg.get('memory', {}).get('mempalace', {}).get('mine', {}).get('wing', 'hermes-sessions'))
except: pass
" 2>/dev/null || echo "hermes-sessions")
fi

EXTRACT="${MEMPALACE_MINE_EXTRACT:-}"
if [ -z "$EXTRACT" ]; then
    EXTRACT=$(python3 -c "
import yaml, os, pathlib
try:
    p = pathlib.Path(os.environ.get('HERMES_HOME', os.path.expanduser('~/.hermes'))) / 'config.yaml'
    if p.exists():
        with open(p) as f:
            cfg = yaml.safe_load(f) or {}
        print(cfg.get('memory', {}).get('mempalace', {}).get('mine', {}).get('extract', 'exchange'))
except: pass
" 2>/dev/null || echo "exchange")
fi

LIMIT="${MEMPALACE_MINE_LIMIT:-}"
if [ -z "$LIMIT" ]; then
    LIMIT=$(python3 -c "
import yaml, os, pathlib
try:
    p = pathlib.Path(os.environ.get('HERMES_HOME', os.path.expanduser('~/.hermes'))) / 'config.yaml'
    if p.exists():
        with open(p) as f:
            cfg = yaml.safe_load(f) or {}
        l = cfg.get('memory', {}).get('mempalace', {}).get('mine', {}).get('limit', 0)
        print(l)
except: pass
" 2>/dev/null || echo "0")
fi

# --- Run ---
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
BEFORE_COUNT=$("$BINARY" status 2>/dev/null | grep -oP 'embeddings.*?\K\d+' || echo "0")

OUTPUT=$("$BINARY" mine \
    --mode convos \
    --wing "$WING" \
    --extract "$EXTRACT" \
    --limit "$LIMIT" \
    "$SESSIONS_DIR" 2>&1)

EXIT_CODE=$?

if [ -z "$OUTPUT" ] || echo "$OUTPUT" | grep -q "Files processed: 0"; then
    # Nothing new — silent exit (cron stays quiet)
    exit 0
fi

# Parse how many new files were processed
FILES_PROCESSED=$(echo "$OUTPUT" | grep -oP 'Files processed:\s*\K\d+' || echo "0")
FILES_SKIPPED=$(echo "$OUTPUT" | grep -oP 'Files skipped.*?:\s*\K\d+' || echo "0")
DRAWERS=$(echo "$OUTPUT" | grep -oP 'Drawers filed:\s*\K\d+' || echo "0")

AFTER_COUNT=$("$BINARY" status 2>/dev/null | grep -oP 'embeddings.*?\K\d+' || echo "0")

# Output JSON summary (only when there's something to report)
cat <<JSON
{
  "timestamp": "$TIMESTAMP",
  "files_processed": $FILES_PROCESSED,
  "files_skipped": $FILES_SKIPPED,
  "drawers_filed": $DRAWERS,
  "embeddings_before": $BEFORE_COUNT,
  "embeddings_after": $AFTER_COUNT,
  "exit_code": $EXIT_CODE,
  "dir": "$SESSIONS_DIR",
  "wing": "$WING",
  "extract": "$EXTRACT"
}
JSON
