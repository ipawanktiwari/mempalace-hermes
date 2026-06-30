# MemPalace for Hermes

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Hermes](https://img.shields.io/badge/Hermes-Agent-6C5CE7)](https://github.com/NousResearch/hermes-agent)

**Semantic long-term memory for [Hermes Agent](https://github.com/NousResearch/hermes-agent) powered by MemPalace (ChromaDB vector search).**

Give your Hermes agent true long-term recall — search every past conversation you've ever had, semantically, without any external services or API keys. All local. All private.

---

## Why This Exists

Hermes has built-in memory (~2,200 chars injected per turn) and optional external providers like Honcho, Mem0, and Holographic. But none of them let you search the *full text of past conversations* for semantically relevant context.

MemPalace indexes your entire session history into a local ChromaDB vector store. This plugin bridges that store into Hermes so the agent can recall discussions, decisions, and debug sessions from weeks or months ago — automatically.

---

## Features

| | |
|---|---|
| **Auto-recall every turn** | `prefetch()` searches MemPalace for context relevant to the current query before every response |
| **On-demand search** | Agent can call `mempalace_search` with optional room/wing filters |
| **Scheduled mining** | `hermes mempalace schedule --every "0 2 * * *"` keeps the index fresh |
| **No external services** | Local ChromaDB + ONNX embeddings. Zero API keys. Works offline. |
| **Configurable** | Everything via `config.yaml`. Binary path, results count, score threshold, mining schedule, extraction strategy. |
| **Silent cron** | No notification spam — mining script outputs JSON only when new files are indexed |

---

## Quick Start

```bash
# 1. Install MemPalace
pip install mempalace

# 2. Initial mining (index your existing sessions)
mempalace mine --mode convos ~/.hermes/sessions

# 3. Install this plugin
curl -fsSL https://raw.githubusercontent.com/ipawanktiwari/mempalace-hermes/main/install.sh | bash

# 4. Activate the provider
hermes config set memory.provider mempalace

# 5. Verify
hermes memory status

# 6. Schedule daily mining (optional)
hermes mempalace schedule --every "0 2 * * *"
```

**Restart your Hermes session** after activation (`/restart` in gateway, or exit and relaunch CLI).

---

## Installation

### Option 1: One-liner

```bash
curl -fsSL https://raw.githubusercontent.com/ipawanktiwari/mempalace-hermes/main/install.sh | bash
```

Environment variables you can set:
- `MEMPALACE_SCHEDULE="every 6h"` — auto-set up mining cron
- `MEMPALACE_NO_CRON=1` — skip cron setup
- `HERMES_HOME=/path/to/hermes` — custom Hermes home

### Option 2: Manual

```bash
git clone https://github.com/ipawanktiwari/mempalace-hermes.git
cd mempalace-hermes

# Copy plugin files
cp -r plugin/* ~/.hermes/plugins/mempalace/

# Copy cron script
mkdir -p ~/.hermes/scripts
cp cron/mempalace-mine.sh ~/.hermes/scripts/
chmod +x ~/.hermes/scripts/mempalace-mine.sh

# Activate
hermes config set memory.provider mempalace
```

---

## Configuration

All settings live under `memory.mempalace` in `~/.hermes/config.yaml`:

```yaml
memory:
  provider: mempalace

  mempalace:
    # Path to mempalace binary (optional — auto-discovered via PATH)
    binary: ~/.local/bin/mempalace

    # Default search parameters
    results: 5           # Results per search
    min_score: 0.3       # Minimum similarity threshold (0.0–1.0)
    timeout: 30          # Search timeout in seconds

    # Mining configuration
    mine:
      sessions_dir: ~/.hermes/sessions   # Directory to mine from
      wing: hermes-sessions              # MemPalace wing name
      extract: exchange                  # Extraction strategy: exchange | general
      limit: 0                           # Max files per run (0 = all)
      schedule: "0 2 * * *"             # Cron schedule expression
```

Environment variable override (highest priority):
- `MEMALACE_BINARY` — path to mempalace executable

---

## Commands

```bash
# Provider status + store stats (vector count, DB size)
hermes mempalace status

# Search past conversations from the terminal
hermes mempalace search "how did we fix the n8n pipeline"
hermes mempalace search "react native expo" --wing project-x --room technical --results 10

# Run mining on sessions directory
hermes mempalace mine                    # Full run
hermes mempalace mine --dry-run          # Preview without storing
hermes mempalace mine --limit 100        # Process first 100 new files
hermes mempalace mine --extract general  # Use general extraction strategy

# Manage recurring mining
hermes mempalace schedule --status       # View current schedule
hermes mempalace schedule --every "6h"   # Mine every 6 hours
hermes mempalace schedule --every "0 9 * * *"  # Daily at 9am
hermes mempalace schedule --disable      # Remove schedule and cron job
```

---

## Agent Tools

The agent receives one tool:

### `mempalace_search`

Search past conversation sessions for relevant context. Semantic search — finds conceptually related content, not just keyword matches.

```
Parameters:
  query    (required) What to search for in past sessions
  wing     (optional) Limit to one project/wing
  room     (optional) technical | decisions | problems | architecture | general
  results  (optional) Number of results (default: 5)
```

The agent also receives automatic context via `prefetch()` — before every turn, MemPalace searches for relevant past conversations and injects them as recalled memory.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Hermes Agent                             │
│                                                              │
│  ┌──────────────────┐   prefetch()         ┌──────────────┐ │
│  │  MemoryManager   │◄────────────────────│  MemPalace   │ │
│  │  (built-in +     │                     │  Provider    │ │
│  │   external)      │──► handle_tool()───►│              │ │
│  └──────────────────┘                     └──────┬───────┘ │
│                                                 │         │
│                                          subprocess        │
│                                                 │         │
│                                          ┌──────▼───────┐ │
│                                          │  mempalace   │ │
│                                          │  search/mine │ │
│                                          └──────┬───────┘ │
│                                                 │         │
│                                          ┌──────▼───────┐ │
│                                          │  ChromaDB    │ │
│                                          │  Vector Store│ │
│                                          └──────────────┘ │
│                                                              │
│  ┌──────────────────┐   cron schedule     ┌──────────────┐ │
│  │  Hermes Cron     │─────────────────────│ mine.sh      │ │
│  │  (no-agent mode) │                     │ (script)     │ │
│  └──────────────────┘                     └──────┬───────┘ │
│                                                 │         │
│                                          mempalace mine   │
│                                                 │         │
│                                          ┌──────▼───────┐ │
│                                          │  Sessions    │ │
│                                          │  Directory   │ │
│                                          └──────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## Comparison with Other Memory Options

| Feature | Built-in Memory | Holographic | Honcho / Mem0 | **MemPalace** |
|---|---|---|---|---|
| Auto-injected every turn | ✓ | ✓ | ✓ | ✓ |
| Cross-session recall | ✗ | ✗* | ✓ | ✓ |
| Semantic search | ✗ | ✗ | ✓ | ✓ |
| Full conversation text | ✗ | ✗ | ✗ | **✓** |
| External service | ✗ | ✗ | Required | **✗** |
| Works offline | ✓ | ✓ | ✗ | **✓** |
| Zero API keys | ✓ | ✓ | ✗ | **✓** |

*Holographic stores extracted facts, not raw conversation text.

**Best used together.** Built-in memory handles compact always-on context. Holographic handles structured facts. MemPalace handles deep semantic recall over your complete conversation history.

---

## Prerequisites

- [Hermes Agent](https://github.com/NousResearch/hermes-agent) (any recent version)
- [MemPalace](https://github.com/NousResearch/hermes-agent) installed and initialized
- Python 3.10+ (already included with Hermes)

---

## FAQ

**Does this replace Hermes built-in memory?**
No. It runs *alongside* it. Built-in memory injects short, curated notes (~2,200 chars) every turn. MemPalace searches the full text of past sessions for deeper recall when needed.

**How is this different from Honcho or Mem0?**
Honcho and Mem0 require external services/API keys. MemPalace is entirely local — your data stays on your machine.

**How often should I run mining?**
Daily is good. Every 6 hours if you're very active. The cron script is smart — silent when nothing new to index.

**Does mining use a lot of CPU?**
~1-2 minutes per new session file on modest hardware. The cron script runs at low priority and won't disrupt your system.

**What if mempalace isn't in my PATH?**
Set `MEMALACE_BINARY=/path/to/mempalace` in your environment, or configure `memory.mempalace.binary` in `config.yaml`.

---

## Contributing

Issues and PRs welcome. The plugin follows the [Hermes Memory Provider plugin convention](https://github.com/NousResearch/hermes-agent) — any `MemoryProvider` implementation in `$HERMES_HOME/plugins/<name>/` is auto-discovered.

---

## License

MIT © [Pawan Kumar Tiwari](https://github.com/ipawanktiwari)
