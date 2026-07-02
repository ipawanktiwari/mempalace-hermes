# MemPalace for Hermes

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Hermes](https://img.shields.io/badge/Hermes-Agent-6C5CE7)](https://github.com/NousResearch/hermes-agent)
[![Release](https://img.shields.io/github/v/release/ipawanktiwari/mempalace-hermes)](https://github.com/ipawanktiwari/mempalace-hermes/releases)

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

### Token Efficiency (Built-in)

These optimizations run automatically — no config needed:

| | |
|---|---|
| **Keyword extraction** | Strips 200+ filler words from queries before searching. 50-70% shorter search strings = stronger semantic signal from fewer tokens. |
| **Room-targeted search** | Searches high-signal rooms first, then sorts by room priority client-side. Single call, less I/O. |
| **Freshness boosting** | Recent sessions get score multipliers: ≤7 days ×1.15, ≤30 days ×1.08. ChromaDB has no recency concept — this compensates. |
| **Adaptive threshold** | Auto-tunes injection sensitivity per session. Loosens when too many queries miss, tightens when too many hit. Self-optimizing. |
| **Confidence metadata** | Every injection includes `high/medium/low` confidence, score stats, and room distribution. Agent knows what to trust at a glance. |
| **Keyword snippets** | Extracts 1-2 most query-relevant sentences per result (keyword density heuristic). Instant context without reading full content. |
| **Query expansion** | Enriches short follow-ups ("yes go ahead") with keywords from the last 3 messages. No more dead searches on filler queries. |
| **Content quality filter** | Scores results by human-signal density — drops stack traces, JSON dumps, and terminal output before injection. Less noise, fewer wasted tokens. |
| **Cross-session state** | Persists adaptive threshold history to disk. New sessions start already calibrated instead of blind for the first 10 turns. |

**Real-world impact:** A 20-turn session burns ~5,000 tokens on memory vs ~20,000 with naive always-inject — **75% fewer tokens wasted** while the adaptive threshold ensures strong matches always surface and weak queries stay silent.

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

## Performance: Token Savings

Token burn matters — it's money. Here's what MemPalace for Hermes costs vs saves:

### The 8-Step Prefetch Pipeline

Every turn, before the agent responds:

1. **Expand** — enriches short follow-ups with recent conversation keywords
2. **Target search** — hits high-signal rooms first (`decisions` → `problems` → `architecture` → `general`), `technical` as last resort
3. **Boost freshness** — recent sessions (≤7d: ×1.15, ≤30d: ×1.08) get priority
4. **Adaptive threshold** — auto-loosens (0.50) or tightens (0.60) based on last 10 turns
5. **Sort** — by room priority then score
6. **Snippet** — extract 1-2 most relevant sentences per result
7. **Format with meta** — confidence level, scores, room distribution in header
8. **Budget** — cap at configurable char limit

### Token Cost Comparison

| Scenario | Without MemPalace | Naive always-inject | MemPalace + Smart Pipeline |
|---|---|---|---|
| **Weak query** (irrelevant match) | 0 tokens | ~1,000 tokens | **0 tokens** (skipped by threshold) |
| **Moderate query** (partial match) | 0 tokens | ~800 tokens | **~450 tokens** (injected with snippets) |
| **Strong query** (clear match) | 0 tokens | ~700 tokens | **~600 tokens** (high-quality, confidence-tagged) |
| **Short follow-up** ("yes go ahead") | 0 tokens | ~800 tokens (dead search) | **~500 tokens** (context-expanded search) |
| **20-turn session** (mixed queries) | 0 tokens | ~20,000 tokens | **~5,000 tokens** |

### Adaptive Threshold Logic

```python
# Base thresholds
single ≥ 0.55 → inject       multi ≥ 0.45 + avg ≥ 0.40 → inject

# Auto-loosen: 7+ of last 10 queries skipped (surface more)
single ≥ 0.50 → inject       multi ≥ 0.40 + avg ≥ 0.35 → inject

# Auto-tighten: 9+ of last 10 queries injected (save tokens)
single ≥ 0.60 → inject       multi ≥ 0.50 + avg ≥ 0.45 → inject
```

**Bottom line:** Dynamic threshold alone saves ~15,000 tokens per session. Keyword extraction + room targeting improve recall quality. Freshness + query expansion + snippets make the injected context actually readable. Combined: higher-quality recall at 75% lower token cost, with zero config required.

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

- [Hermes Agent](https://github.com/NousResearch/hermes-agent) (v0.3.0+)
- [MemPalace](https://github.com/ipawanktiwari/mempalace) installed and initialized
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

## Development

### Branching

```
main     ← tagged releases (v1.0.0, v1.1.0, ...)
  └── develop  ← active feature/fix work, merged into main for releases
```

**Workflow:**
1. All new features and fixes go to `develop`
2. When ready for release: `git checkout main && git merge develop && git tag vX.Y.Z && git push --tags origin main`
3. Continue on `develop` for next cycle

### Release History

| Version | Date | Highlights |
|---|---|---|
| **v1.1.0** | 2026-07-02 | Single-call search (5x faster), content quality filter, cross-session state persistence, 76 tests |
| **v1.0.3** | 2026-07-01 | CI pipeline, 64 tests, CHANGELOG, CONTRIBUTING, is_available() cache, install.sh fix |
| **v1.0.2** | 2026-07-01 | plugin.yaml version fix, housekeeping |
| **v1.0.1** | 2026-07-01 | Batched logging, log level fixes, README badges |
| **v1.0.0** | 2026-06-30 | Initial release — 10 features, 8-step prefetch pipeline, 75% token savings |

---

## Contributing

Issues and PRs welcome. Target `develop` for all changes. The plugin follows the [Hermes Memory Provider plugin convention](https://github.com/NousResearch/hermes-agent) — any `MemoryProvider` implementation in `$HERMES_HOME/plugins/<name>/` is auto-discovered.

---

## License

MIT © [Pawan Kumar Tiwari](https://github.com/ipawanktiwari)
