#!/usr/bin/env bash
# install.sh — Install mempalace-hermes plugin and optional cron mining.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/ipawanktiwari/mempalace-hermes/main/install.sh | bash
#
#   # Or clone and run:
#   git clone https://github.com/ipawanktiwari/mempalace-hermes.git
#   cd mempalace-hermes && bash install.sh
#
# Options (environment variables):
#   HERMES_HOME            — Hermes home directory (default: ~/.hermes)
#   MEMPALACE_BINARY       — Path to mempalace executable
#   MEMPALACE_SCHEDULE     — Set up mine schedule (e.g. "every 6h", "0 9 * * *")
#                           Leave empty to skip cron setup
#   MEMPALACE_NO_CRON      — Set to 1 to skip cron setup
#   MEMPALACE_SCRIPTS_DIR  — Where to install cron scripts (default: $HERMES_HOME/scripts)

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo ""
echo "  ⚡ mempalace-hermes installer"
echo "  ─────────────────────────────"
echo ""

# Resolve paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
PLUGINS_DIR="$HERMES_HOME/plugins/mempalace"
SCRIPTS_DIR="${MEMPALACE_SCRIPTS_DIR:-$HERMES_HOME/scripts}"

echo "  Hermes home:    $HERMES_HOME"
echo "  Plugin dir:     $PLUGINS_DIR"
echo "  Scripts dir:    $SCRIPTS_DIR"
echo ""

# ---- 1. Install plugin files ----

mkdir -p "$PLUGINS_DIR"

cp "$SCRIPT_DIR/plugin/__init__.py" "$PLUGINS_DIR/__init__.py"
cp "$SCRIPT_DIR/plugin/plugin.yaml" "$PLUGINS_DIR/plugin.yaml"
cp "$SCRIPT_DIR/plugin/cli.py" "$PLUGINS_DIR/cli.py"

echo -e "  ${GREEN}✓${NC} Plugin installed to $PLUGINS_DIR"

# ---- 2. Install cron script ----

mkdir -p "$SCRIPTS_DIR"
cp "$SCRIPT_DIR/cron/mempalace-mine.sh" "$SCRIPTS_DIR/mempalace-mine.sh"
chmod +x "$SCRIPTS_DIR/mempalace-mine.sh"

echo -e "  ${GREEN}✓${NC} Cron script installed to $SCRIPTS_DIR/mempalace-mine.sh"

# ---- 3. Activate provider ----

hermes config set memory.provider mempalace 2>/dev/null || {
    echo -e "  ${YELLOW}⚠${NC} Could not auto-activate provider. Run: hermes config set memory.provider mempalace"
}
echo -e "  ${GREEN}✓${NC} Provider activated"

# ---- 4. Optional: Set up cron schedule ----

if [ "${MEMPALACE_NO_CRON:-0}" = "1" ]; then
    echo ""
    echo "  Skipping cron setup (MEMPALACE_NO_CRON=1)."
    echo "  Set up later: hermes mempalace schedule --every 6h"
elif [ -n "${MEMPALACE_SCHEDULE:-}" ]; then
    echo ""
    echo "  Setting up mine schedule: $MEMPALACE_SCHEDULE"

    # Configure
    hermes config set memory.mempalace.mine.wing hermes-sessions 2>/dev/null || true
    hermes config set memory.mempalace.mine.extract exchange 2>/dev/null || true

    # Create cron job
    hermes cron create "$MEMPALACE_SCHEDULE" \
        --name mempalace-mine \
        --script mempalace-mine.sh \
        --no-agent \
        --deliver local 2>/dev/null || {
        echo -e "  ${YELLOW}⚠${NC} Cron job may already exist. Run: hermes cron list"
    }
    echo -e "  ${GREEN}✓${NC} Mine schedule set to: $MEMPALACE_SCHEDULE"
fi

# ---- Done ----

echo ""
echo -e "  ${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  ${GREEN}  mempalace-hermes installed!${NC}"
echo -e "  ${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Next steps:"
echo "    1. hermes mempalace status        — verify installation"
echo "    2. hermes mempalace mine          — run initial mining"
echo "    3. hermes mempalace schedule      — set up recurring mining"
echo "    4. hermes memory status           — confirm provider is active"
echo ""
echo "  Docs: https://github.com/ipawanktiwari/mempalace-hermes"
echo ""
