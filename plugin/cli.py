"""CLI commands for MemPalace memory provider management.

Provides:
  ``hermes mempalace status``   — Show connection and store stats
  ``hermes mempalace search``   — Search past conversations
  ``hermes mempalace mine``     — Run mempalace mine on sessions (dry-run supported)
  ``hermes mempalace schedule`` — Set up recurring mining via Hermes cron
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Binary discovery (inlined to avoid cross-namespace import issues for
# user-installed plugins loaded outside the bundled namespace)
# ---------------------------------------------------------------------------

def _resolve_binary() -> str:
    """Find the mempalace executable.

    Resolution order:
      1. ``MEMALACE_BINARY`` environment variable
      2. ``memory.mempalace.binary`` from config.yaml
      3. ``mempalace`` on ``$PATH`` via ``shutil.which``
      4. Common installation paths
    """
    env_binary = os.environ.get("MEMALACE_BINARY", "").strip()
    if env_binary and Path(env_binary).is_file():
        return env_binary

    try:
        from hermes_cli.config import load_config
        config = load_config()
        mem_config = config.get("memory", {}) if isinstance(config, dict) else {}
        mp_config = mem_config.get("mempalace", {}) if isinstance(mem_config, dict) else {}
        config_binary = mp_config.get("binary", "")
        if config_binary and Path(config_binary).is_file():
            return config_binary
    except Exception:
        pass

    path_binary = shutil.which("mempalace")
    if path_binary:
        return path_binary

    common = [
        os.path.expanduser("~/.local/bin/mempalace"),
        "/usr/local/bin/mempalace",
        "/usr/bin/mempalace",
    ]
    for candidate in common:
        if Path(candidate).is_file():
            return candidate

    return ""


def _resolve_sessions_dir() -> str:
    """Resolve the sessions directory from config or default."""
    try:
        from hermes_cli.config import load_config
        from hermes_constants import get_hermes_home
        config = load_config()
        mem_config = config.get("memory", {}) if isinstance(config, dict) else {}
        mp_config = mem_config.get("mempalace", {}) if isinstance(mem_config, dict) else {}
        mine_config = mp_config.get("mine", {}) if isinstance(mp_config, dict) else {}
        configured = mine_config.get("sessions_dir", "")
        if configured:
            return os.path.expanduser(configured)
        return str(get_hermes_home() / "sessions")
    except Exception:
        return os.path.expanduser("~/.hermes/sessions")


def _read_mine_config() -> dict:
    """Read mine configuration from config.yaml."""
    try:
        from hermes_cli.config import load_config
        config = load_config()
        mem_config = config.get("memory", {}) if isinstance(config, dict) else {}
        mp_config = mem_config.get("mempalace", {}) if isinstance(mem_config, dict) else {}
        return mp_config.get("mine", {}) if isinstance(mp_config, dict) else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> None:
    """Show mempalace connection and store status."""
    binary = _resolve_binary()

    print("\nMemPalace memory provider\n" + "─" * 40)
    if not binary:
        print("  Binary:  NOT FOUND ✗")
        print("\n  Install mempalace: pip install mempalace")
        print("  Then run: hermes mempalace status\n")
        return

    print(f"  Binary:  {binary}")

    try:
        result = subprocess.run(
            [binary, "--help"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print("  Runtime: available ✓")
        else:
            print(f"  Runtime: exit code {result.returncode} ✗")
            print(f"    {result.stderr.strip()[:200]}")
            print()
            return
    except FileNotFoundError:
        print("  Runtime: binary not found ✗\n")
        return
    except subprocess.TimeoutExpired:
        print("  Runtime: timed out ✗\n")
        return

    chroma_dir = Path.home() / ".mempalace" / "palace"
    if chroma_dir.exists():
        chroma_sqlite = chroma_dir / "chroma.sqlite3"
        if chroma_sqlite.exists():
            size_kb = chroma_sqlite.stat().st_size / 1024
            print(f"  Store:   {chroma_sqlite} ({size_kb:.0f} KB)")
            try:
                import sqlite3
                conn = sqlite3.connect(str(chroma_sqlite))
                count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
                conn.close()
                print(f"  Vectors: {count:,}")
            except Exception:
                pass
        else:
            print(f"  Store:   {chroma_dir} (no chroma.sqlite3)")
    else:
        print(f"  Store:   NOT INITIALIZED ✗")
        print(f"\n  Initialize with: hermes mempalace mine\n")

    # Show mine schedule config
    mine_config = _read_mine_config()
    if mine_config.get("schedule"):
        print(f"  Mine schedule: {mine_config['schedule']} (active)")
    elif mine_config:
        print("  Mine schedule: not set (manual only)")
    else:
        print("  Mine config:  defaults (run 'hermes mempalace schedule' to automate)")

    print()


def cmd_search(args: argparse.Namespace) -> None:
    """Search MemPalace for past conversation context."""
    binary = _resolve_binary()
    if not binary:
        print("\n  MemPalace binary not found.\n")
        return

    query = args.query
    if not query:
        print("\n  Usage: hermes mempalace search <query>\n")
        return

    cmd = [binary, "search", query, "--results", str(args.results)]
    if args.wing:
        cmd.extend(["--wing", args.wing])
    if args.room:
        cmd.extend(["--room", args.room])

    print()
    try:
        subprocess.run(cmd, timeout=30)
        print()
    except subprocess.TimeoutExpired:
        print("\n  Search timed out (30s). Try with fewer results.\n")
    except FileNotFoundError:
        print("\n  mempalace binary not found.\n")


def cmd_mine(args: argparse.Namespace) -> None:
    """Run mempalace mine on sessions directory."""
    binary = _resolve_binary()
    if not binary:
        print("\n  MemPalace binary not found.\n")
        return

    sessions_dir = _resolve_sessions_dir()
    mine_config = _read_mine_config()

    wing = args.wing or mine_config.get("wing", "hermes-sessions")
    extract = args.extract or mine_config.get("extract", "exchange")
    limit = args.limit if args.limit is not None else mine_config.get("limit", 0)

    if not os.path.isdir(sessions_dir):
        print(f"\n  Sessions directory not found: {sessions_dir}")
        print("  Configure with: hermes config set memory.mempalace.mine.sessions_dir <path>\n")
        return

    cmd = [
        binary, "mine",
        "--mode", "convos",
        "--wing", wing,
        "--extract", extract,
    ]
    if limit > 0:
        cmd.extend(["--limit", str(limit)])
    if args.dry_run:
        cmd.append("--dry-run")
    cmd.append(sessions_dir)

    print(f"\n  Mempalace mine")
    print("  " + "─" * 38)
    print(f"  Dir:     {sessions_dir}")
    print(f"  Wing:    {wing}")
    print(f"  Extract: {extract}")
    if limit > 0:
        print(f"  Limit:   {limit} files")
    if args.dry_run:
        print(f"  Mode:    DRY RUN (no files will be stored)")
    print()

    try:
        subprocess.run(cmd, timeout=1800)
    except subprocess.TimeoutExpired:
        print("\n  Mine timed out. Run with --limit <N> for smaller batches.\n")
    except FileNotFoundError:
        print("\n  mempalace binary not found.\n")


def cmd_schedule(args: argparse.Namespace) -> None:
    """Set up or remove recurring mining via Hermes cron.

    The cron job runs the mempalace-mine.sh script from this plugin's
    cron/ directory. Empty output = silent (no notification spam).
    """
    from hermes_cli.config import load_config, save_config

    config = load_config()
    if not isinstance(config.get("memory"), dict):
        config["memory"] = {}
    mem_config = config["memory"]
    if "mempalace" not in mem_config or not isinstance(mem_config["mempalace"], dict):
        mem_config["mempalace"] = {}
    mp_config = mem_config["mempalace"]
    if "mine" not in mp_config or not isinstance(mp_config["mine"], dict):
        mp_config["mine"] = {}
    mine_config = mp_config["mine"]

    # -- Disable --
    if args.disable:
        mine_config.pop("schedule", None)
        config["memory"]["mempalace"]["mine"] = mine_config
        save_config(config)

        # Remove existing cron job if any
        try:
            subprocess.run(
                ["hermes", "cron", "remove", "mempalace-mine"],
                capture_output=True, timeout=10,
            )
            print("\n  ✓ Schedule removed. Weekly mining disabled.\n")
        except Exception:
            print("\n  ✓ Schedule config cleared (cron job may need manual removal: hermes cron remove mempalace-mine)\n")
        return

    # -- Status/show --
    current_schedule = mine_config.get("schedule", "")
    if args.status:
        if current_schedule:
            print(f"\n  Mine schedule: {current_schedule}")
            print(f"  Wing:          {mine_config.get('wing', 'hermes-sessions')}")
            print(f"  Extract:       {mine_config.get('extract', 'exchange')}")
            print(f"  Sessions dir:  {mine_config.get('sessions_dir', _resolve_sessions_dir())}")
            print()
        else:
            print("\n  No mine schedule set.")
            print("  Set one with: hermes mempalace schedule --every 6h\n")
        return

    # -- Set up --
    schedule = args.every or "6h"
    wing = args.wing or mine_config.get("wing", "hermes-sessions")
    extract = args.extract or mine_config.get("extract", "exchange")

    mine_config["schedule"] = schedule
    mine_config["wing"] = wing
    mine_config["extract"] = extract
    config["memory"]["mempalace"]["mine"] = mine_config
    save_config(config)

    print(f"\n  Mine schedule configured:")
    print(f"    Every:        {schedule}")
    print(f"    Wing:         {wing}")
    print(f"    Extract:      {extract}")
    print(f"    Sessions dir: {_resolve_sessions_dir()}")
    print()

    # Create Hermes cron job
    try:
        result = subprocess.run(
            [
                "hermes", "cron", "create",
                schedule,
                "--name", "mempalace-mine",
                "--script", "mempalace-mine.sh",
                "--no-agent",              # script IS the job, no LLM needed
                "--deliver", "local",      # JSON output to file, not chat
            ],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            print("  ✓ Cron job 'mempalace-mine' created.")
            print("    View:  hermes cron list")
            print("    Pause: hermes cron pause mempalace-mine")
            print("    Remove: hermes mempalace schedule --disable")
        else:
            stderr = result.stderr.strip()
            if stderr:
                # Job might already exist — try update
                subprocess.run(
                    ["hermes", "cron", "resume", "mempalace-mine"],
                    capture_output=True, timeout=10,
                )
                print(f"  ⚠ Cron job note: {stderr[:200]}")
    except Exception as e:
        print(f"  ⚠ Could not create cron job: {e}")
        print("    Manual setup:")
        print(f"    hermes cron create {schedule} --name mempalace-mine \\")
        print("      --script mempalace-mine.sh --no-agent --deliver local")

    print()


def mempalace_command(args: argparse.Namespace) -> None:
    """Route mempalace subcommands."""
    sub = getattr(args, "mempalace_command", None)
    if sub == "status":
        cmd_status(args)
    elif sub == "search":
        cmd_search(args)
    elif sub == "mine":
        cmd_mine(args)
    elif sub == "schedule":
        cmd_schedule(args)
    else:
        cmd_status(args)


def register_cli(subparser: argparse.ArgumentParser) -> None:
    """Build the ``hermes mempalace`` argparse subcommand tree.

    Called by the plugin CLI registration system during argparse setup.
    """
    subs = subparser.add_subparsers(dest="mempalace_command")

    subs.add_parser("status", help="Show MemPalace connection and store status")

    # --- search ---
    search_parser = subs.add_parser("search", help="Search past conversations via MemPalace")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--results", type=int, default=5, help="Number of results (default: 5)")
    search_parser.add_argument("--wing", help="Limit to one project/wing")
    search_parser.add_argument(
        "--room",
        choices=("technical", "decisions", "problems", "architecture", "general"),
        help="Limit to one room type",
    )

    # --- mine ---
    mine_parser = subs.add_parser("mine", help="Run mempalace mine on sessions directory")
    mine_parser.add_argument("--wing", help="Wing name (default: hermes-sessions)")
    mine_parser.add_argument(
        "--extract",
        choices=("exchange", "general"),
        help="Extraction strategy (default: exchange)",
    )
    mine_parser.add_argument("--limit", type=int, help="Max files to process (0 = all)")
    mine_parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be filed without filing",
    )

    # --- schedule ---
    schedule_parser = subs.add_parser(
        "schedule",
        help="Set up recurring mining via Hermes cron (or view/disable)",
    )
    schedule_parser.add_argument(
        "--every", metavar="SCHEDULE",
        help="Schedule expression: '30m', 'every 6h', '0 9 * * *'",
    )
    schedule_parser.add_argument("--wing", help="Wing name for mined sessions")
    schedule_parser.add_argument(
        "--extract",
        choices=("exchange", "general"),
        help="Extraction strategy",
    )
    schedule_parser.add_argument(
        "--status", action="store_true",
        help="Show current schedule without making changes",
    )
    schedule_parser.add_argument(
        "--disable", action="store_true",
        help="Remove the mine schedule and cron job",
    )
