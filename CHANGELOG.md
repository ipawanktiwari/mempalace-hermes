# Changelog

All notable changes to mempalace-hermes are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.2] — 2026-07-01

### Fixed
- `plugin.yaml` version bumped from `0.1.0` to `1.0.0` to match the release tag
- Install script now shows correct docs URL (`ipawanktiwari` instead of `<user>`)
- Hermes min version documented in `plugin.yaml` and README

### Added
- `CHANGELOG.md` — project changelog
- `CONTRIBUTING.md` — contribution guide
- Test suite (pytest) covering search parser, content cleaner, keyword extraction, adaptive threshold, freshness boosting
- CI pipeline (GitHub Actions) — lint + test on push/PR to develop and main

## [1.0.1] — 2026-07-01

### Fixed
- Batched logging — 1 INFO line per 10 prefetch turns instead of per-turn noise
- Prefetch log level promotion from DEBUG to INFO for better operational visibility
- README badges and branching strategy docs

## [1.0.0] — 2026-06-30

### Added
- Initial MemPalace memory provider for Hermes Agent
- Auto-recall every turn via `prefetch()` — searches MemPalace for context relevant to the current query
- On-demand `mempalace_search` tool with optional room/wing filters
- 8-step prefetch pipeline: expand → targeted search → freshness boost → adaptive threshold → sort → snippet → format → budget
- Token efficiency: 75% fewer tokens vs naive always-inject
- Keyword extraction — strips 200+ filler words from queries before searching
- Room-targeted search — high-signal rooms first (decisions, problems, architecture, general)
- Freshness boosting — recent sessions get score multipliers (≤7d ×1.15, ≤30d ×1.08)
- Adaptive threshold — auto-loosens/tightens injection sensitivity based on last 10 turns
- Confidence metadata — every injection includes high/medium/low confidence, scores, room distribution
- Keyword snippets — extracts 1-2 most query-relevant sentences per result
- Query expansion — enriches short follow-ups with keywords from recent context
- CLI commands: `hermes mempalace status`, `search`, `mine`, `schedule`
- Silent cron — mining script outputs JSON only when new files are indexed
- Install script with optional cron setup
- Scheduled mining via Hermes cron (no-agent mode)
- Config schema for `hermes memory setup` wizard
