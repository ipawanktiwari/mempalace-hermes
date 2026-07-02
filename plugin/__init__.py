"""MemPalace memory provider — semantic search over past conversation sessions.

Uses the standalone mempalace CLI (ChromaDB-backed vector search) to
recall relevant context from past conversations. Works as a read-only
semantic retrieval layer — write operations happen through ``mempalace mine``,
not through this provider.

Installation
  Drop this directory into ``$HERMES_HOME/plugins/mempalace/``.
  Then: ``hermes config set memory.provider mempalace``
  Or: ``hermes memory setup`` and select from the picker.

Config in ``config.yaml`` under ``memory.mempalace`` (all optional)::

  memory:
    provider: mempalace
    mempalace:
      binary: /path/to/mempalace
      results: 5
      min_score: 0.3
      timeout: 30
      deduplicate: true    # collapse duplicate session sources
      max_prefetch_chars: 4000  # max chars injected per turn

Environment variable:
  ``MEMALACE_BINARY`` — path to mempalace executable (overrides config + PATH)

Tuning for token efficiency:
  The provider extracts keywords from verbose queries before searching,
  prioritises high-signal rooms (decisions, problems), and skips injection
  entirely when no strong matches exist — saving ~600-1000 tokens per
  turn vs. injecting low-quality context.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schema — exposed to the agent as mempalace_search
# ---------------------------------------------------------------------------

MEMPALACE_SEARCH_SCHEMA = {
    "name": "mempalace_search",
    "description": (
        "Search past conversation sessions for relevant context. "
        "Use when you need to recall decisions, discussions, or facts "
        "from earlier sessions that aren't in active memory. "
        "Semantic search — finds conceptually related content, not just keyword matches."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in past sessions",
            },
            "wing": {
                "type": "string",
                "description": "Limit to one project/wing (optional)",
            },
            "room": {
                "type": "string",
                "description": (
                    "Limit to one room type: technical, decisions, "
                    "problems, architecture, general (optional)"
                ),
            },
            "results": {
                "type": "integer",
                "description": "Number of results (default: 5)",
            },
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# Binary discovery — env var > config > PATH > common paths
# ---------------------------------------------------------------------------

def _discover_binary(config_binary: str | None = None) -> str:
    """Find the mempalace executable.

    Resolution order:
      1. ``MEMALACE_BINARY`` environment variable
      2. ``memory.mempalace.binary`` from config.yaml
      3. ``mempalace`` on ``$PATH`` (via ``shutil.which``)
      4. Common installation paths

    Returns the resolved path, or empty string if not found.
    """
    # 1. Environment variable (highest priority)
    env_binary = os.environ.get("MEMALACE_BINARY", "").strip()
    if env_binary and Path(env_binary).is_file():
        return env_binary

    # 2. Config value
    if config_binary and config_binary.strip():
        p = Path(config_binary.strip())
        if p.is_file():
            return str(p)

    # 3. $PATH lookup
    path_binary = shutil.which("mempalace")
    if path_binary:
        return path_binary

    # 4. Common installation paths
    common = [
        os.path.expanduser("~/.local/bin/mempalace"),
        "/usr/local/bin/mempalace",
        "/usr/bin/mempalace",
    ]
    for candidate in common:
        if Path(candidate).is_file():
            return candidate

    return ""


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Read provider config from ``config.yaml`` -> ``memory.mempalace``."""
    try:
        from hermes_cli.config import load_config
        config = load_config()
        mem_config = config.get("memory", {})
        return (
            mem_config.get("mempalace", {})
            if isinstance(mem_config, dict)
            else {}
        )
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# MemPalaceProvider
# ---------------------------------------------------------------------------

class MemPalaceProvider(MemoryProvider):
    """Memory provider wrapping the mempalace CLI for semantic recall."""

    def __init__(self, config: dict | None = None):
        self._config = config or _load_config()
        self._binary = _discover_binary(self._config.get("binary", ""))
        self._default_results = int(self._config.get("results", 5))
        self._min_score = float(self._config.get("min_score", 0.3))
        self._timeout = int(self._config.get("timeout", 30))
        self._deduplicate = self._config.get("deduplicate", True) not in (False, "false", "False")
        self._max_prefetch_chars = int(self._config.get("max_prefetch_chars", 4000))
        self._wing = self._config.get("wing", "sessions")
        self._session_id: str = ""
        # Cache binary availability — checked once, cached for the session
        self._available: bool | None = None
        # Adaptive threshold state — tracks last 10 injection decisions
        self._injection_history: List[bool] = []  # True=injected, False=skipped
        self._inject_count: int = 0  # total injections this session
        # Query context — recent messages for query expansion
        self._recent_queries: List[str] = []  # last 3 user messages (keywords only)
        # Batched summary state — avoid per-turn log spam
        self._summary_skips: int = 0
        self._summary_injects: int = 0
        self._summary_chars: int = 0
        # State file for cross-session persistence
        self._state_dir: Path = Path.home() / ".hermes" / "mempalace"
        self._state_file: Path = self._state_dir / "provider_state.json"

    # -- MemoryProvider ABC ------------------------------------------------

    @property
    def name(self) -> str:
        return "mempalace"

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        if not self._binary:
            logger.debug("MemPalace binary not found")
            self._available = False
            return False
        try:
            result = subprocess.run(
                [self._binary, "--help"],
                capture_output=True,
                timeout=10,
            )
            self._available = result.returncode == 0
            return self._available
        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError) as e:
            logger.debug("MemPalace availability check failed: %s", e)
            self._available = False
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._load_state()  # restore cross-session threshold state
        logger.info("MemPalace provider initialized, binary=%s, session=%s",
                     self._binary, session_id)

    def system_prompt_block(self) -> str:
        if not self._binary:
            return ""
        return (
            "# MemPalace Memory\n"
            "Active. Semantic search over past conversation sessions available.\n"
            "Use mempalace_search to recall decisions, discussions, or context "
            "from earlier sessions."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not query or not query.strip() or not self._binary:
            return ""

        try:
            # 1. Expand short follow-ups with recent context keywords
            search_query = self._expand_query(query)

            # 2. Room-targeted search: high-signal rooms first, fall back to technical
            results = self._targeted_search(search_query)

            # 3. Freshness boosting — recent sessions get score multiplier
            self._boost_freshness(results)

            # 4. Adaptive threshold — skip injection if matches are weak (auto-tunes)
            injected = self._should_inject(results)
            self._injection_history.append(injected)
            if len(self._injection_history) > 10:
                self._injection_history.pop(0)

            if not injected:
                self._summary_skips += 1
                self._emit_batched_summary()
                return ""

            self._inject_count += 1
            self._summary_injects += 1

            # 5. Sort by room priority (decisions > problems > architecture > general > technical)
            room_order = {r: i for i, r in enumerate(self._PRIORITY_ROOMS)}
            results.sort(key=lambda r: (room_order.get(r.get("room", ""), 99), -(r.get("score", 0))))

            # 6. Smart snippet — 1-2 sentence summary per result for quick scanning
            sources = self._extract_keyword_snippets(results, search_query, max_sentences=2, max_chars=200)

            # 7. Format with confidence metadata + snippets
            prefetch_text = self._format_prefetch_with_meta(search_query, results, sources)

            # 8. Enforce char budget
            if len(prefetch_text) > self._max_prefetch_chars:
                prefetch_text = prefetch_text[:self._max_prefetch_chars] + "\n\n... (budget limit)"

            self._summary_chars += len(prefetch_text)
            self._emit_batched_summary()

            logger.debug(
                "MemPalace prefetch: query='%s' top=%.3f results=%d chars=%d total_injects=%d",
                search_query[:80],
                results[0]["score"] if results else 0,
                len(results),
                len(prefetch_text),
                self._inject_count,
            )
            return prefetch_text
        except Exception as e:
            logger.debug("MemPalace prefetch failed: %s", e)
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        pass  # search is fast enough inline

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        # Capture recent user keywords for query expansion
        if user_content and len(user_content) > 5:
            keywords = self._extract_keywords(user_content)
            if keywords:
                self._recent_queries.append(keywords)
                if len(self._recent_queries) > 3:
                    self._recent_queries.pop(0)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [MEMPALACE_SEARCH_SCHEMA]

    def handle_tool_call(
        self, tool_name: str, args: Dict[str, Any], **kwargs
    ) -> str:
        if tool_name != "mempalace_search":
            return tool_error(f"Unknown tool: {tool_name}")

        try:
            query = args.get("query", "")
            if not query:
                return tool_error("query is required")

            results = self._search(
                query,
                wing=args.get("wing"),
                room=args.get("room"),
                limit=int(args.get("results", self._default_results)),
            )

            if not results:
                return json.dumps({
                    "results": [],
                    "count": 0,
                    "message": "No matching past conversations found.",
                })

            return json.dumps({
                "results": results,
                "count": len(results),
            })
        except Exception as e:
            return tool_error(f"MemPalace search failed: {e}")

    def shutdown(self) -> None:
        self._save_state()  # persist threshold state for next session
        self._available = None

    # -- Config schema for hermes memory setup wizard ----------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "binary",
                "description": "Path to mempalace executable",
                "default": self._binary or shutil.which("mempalace") or "",
                "env_var": "MEMALACE_BINARY",
            },
            {
                "key": "wing",
                "description": "MemPalace wing to search (default: sessions)",
                "default": "sessions",
            },
            {
                "key": "results",
                "description": "Default number of results per search",
                "default": "5",
            },
            {
                "key": "min_score",
                "description": "Minimum similarity score (0.0–1.0)",
                "default": "0.3",
            },
            {
                "key": "timeout",
                "description": "Search timeout in seconds",
                "default": "30",
            },
            {
                "key": "deduplicate",
                "description": "Collapse duplicate session sources, keeping highest score",
                "default": "true",
                "choices": ["true", "false"],
            },
            {
                "key": "max_prefetch_chars",
                "description": "Max chars injected per turn via prefetch",
                "default": "4000",
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write non-secret config to config.yaml under memory.mempalace."""
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            existing = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8-sig") as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("memory", {})
            existing["memory"]["mempalace"] = values
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(existing, f, default_flow_style=False)
            # Re-resolve all config values
            self._config = values
            self._binary = _discover_binary(values.get("binary", ""))
            self._default_results = int(values.get("results", 5))
            self._min_score = float(values.get("min_score", 0.3))
            self._timeout = int(values.get("timeout", 30))
            self._deduplicate = values.get("deduplicate", True) not in (False, "false", "False")
            self._max_prefetch_chars = int(values.get("max_prefetch_chars", 4000))
            self._wing = values.get("wing", "sessions")
        except Exception as e:
            logger.warning("Failed to save mempalace config: %s", e)

    # ------------------------------------------------------------------
    # Content cleaning — strip JSON/tool-call noise from exchange-mode results
    # ------------------------------------------------------------------

    @classmethod
    def _clean_content(cls, content: str) -> str:
        """Strip JSON tool-call noise from exchange-mode content.

        Exchange mode preserves raw JSON tool invocations alongside
        narrative text. We keep only lines that look like human content
        and discard JSON blobs, escaped strings, and structural markers.
        """
        if not content:
            return ""

        import re
        lines = content.splitlines()
        kept: list[str] = []

        # Patterns that indicate a line is structural noise
        _noise_line = re.compile(
            r'^\s*'
            r'(?:\{\s*"?|\}\s*"?|"\s*[a-z_]+\s*"?:\s*|'
            r'\[?\s*\{\s*"?|"\s*\]|'  # JSON brackets
            r'\\\s*"|\\\\|[{}[\]",]{8,}|'  # heavy escaping / raw JSON
            r'\"[a-z_]+\"\s*:\s*[\[{]|'  # key-value start
            r'\s*"[a-z_]+\":|'  # JSON key
            r'\s*\}{1,3}\s*$)'
        )

        # Lines that are entirely non-human (pure symbols, braces, quotes)
        _json_only = re.compile(r'^[\s"{}[\]\\,:]+$')

        for line in lines:
            stripped = line.strip()

            # Skip empty
            if not stripped:
                continue

            # Skip pure punctuation/JSON lines
            if _json_only.match(stripped):
                continue

            # Skip structural noise
            if _noise_line.match(stripped):
                continue

            # Skip lines that are mostly raw JSON (high ratio of json chars)
            json_chars = sum(1 for c in stripped if c in '{}[]":\\')
            if len(stripped) > 5 and json_chars / len(stripped) > 0.35:
                continue

            kept.append(line)

        result = "\n".join(kept).strip()

        # Collapse 3+ blank lines
        result = re.sub(r'\n{3,}', '\n\n', result)

        return result

    # ------------------------------------------------------------------
    # Result processing (dedup, clean, budget)
    # ------------------------------------------------------------------

    def _process_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean content, deduplicate by session, filter low-signal noise."""
        if not results:
            return []

        # Clean each result's content
        for r in results:
            r["content"] = self._clean_content(r.get("content", ""))
        results = [r for r in results if r.get("content", "").strip()]

        # Quality filter: drop results below minimum signal threshold
        results = [r for r in results if self._content_quality_score(r.get("content", "")) >= 0.15]

        # Deduplicate by session source (keep highest score)
        if self._deduplicate:
            seen: Dict[str, Dict[str, Any]] = {}
            for r in results:
                source = r.get("source", "")
                # Normalize: strip date suffixes from session filenames
                base = source.rsplit("_", 3)[0] if "_" in source else source
                if base not in seen or r.get("score", 0) > seen[base].get("score", 0):
                    seen[base] = r
            results = sorted(seen.values(), key=lambda r: r.get("score", 0), reverse=True)

        return results

    # ------------------------------------------------------------------
    # Content quality scoring: filter low-signal results before injection
    # ------------------------------------------------------------------

    # Patterns that indicate low-human-signal content
    _LOW_SIGNAL_PATTERNS = [
        re.compile(r, re.IGNORECASE) for r in [
            r'traceback\s*\(most recent call',  # Python traceback header
            r'^\s*File\s+["\']',                 # Python stack frame (single/double quotes)
            r'^\s+\.\.\.\s+\d+\s+more',          # Java omitted frames
            r'^\s*at\s+\w+\.\w+\([\w.]+:\d+\)',  # Java stack trace
            r'^\s*raise\s+\w+',                  # Python raise statement
            r'^\s*\w+Error[:(]',                 # Error classes: ValueError: ..., KeyError(...
            r'exit\s*code:\s*\d+',               # process exit codes
            r'ERROR:?\s',                        # error markers (capped ratio)
            r'^\s*\{.*\}\s*$',                   # pure JSON line
            r'^std(out|err):',                   # terminal output markers
            r'Executed\s+\d+\s+tasks?',          # batch job summary
            r'successfully\s+completed',         # success markers
            r'^\s*\d+\s+(items?|files?|results?)',  # count-only lines
        ]
    ]

    @classmethod
    def _content_quality_score(cls, content: str) -> float:
        r"""Score content by human-signal density (0.0 – 1.0).

        Heuristic that penalizes:
        - Stack traces, error dumps, tool output
        - High ratio of JSON/symbol chars
        - Lines that are single-word or pure numbers
        - Content dominated by boilerplate markers

        Rewards:
        - Longer sentences (12+ chars)
        - Mixed case, natural punctuation
        - Keyword-level diversity

        Returns 0.0 for empty content, ~0.8+ for human narrative,
        ~0.05-0.20 for tool-output dumps.
        """
        if not content or not content.strip():
            return 0.0

        lines = content.splitlines()
        total_lines = len(lines)
        if total_lines == 0:
            return 0.0

        signal_lines = 0
        noise_lines = 0

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Skip lines matching low-signal patterns
            is_noise = False
            for pat in cls._LOW_SIGNAL_PATTERNS:
                if pat.search(stripped):
                    is_noise = True
                    break
            if is_noise:
                noise_lines += 1
                continue

            # Score individual line quality
            words = len(stripped.split())
            has_mixed_case = stripped != stripped.lower() and stripped != stripped.upper()
            has_punct = any(c in stripped for c in '.!?,;:')
            json_ratio = sum(1 for c in stripped if c in '{}[]":\\') / max(len(stripped), 1)

            # Good signal: medium-length line with mixed case and punctuation
            if words >= 5 and has_mixed_case and json_ratio < 0.15:
                signal_lines += 1
            elif words >= 8 and json_ratio < 0.1:
                signal_lines += 1
            elif json_ratio > 0.3:
                noise_lines += 1
            # Short mixed-case lines with punctuation are weak but not noise
            elif words >= 3 and has_mixed_case and json_ratio < 0.1:
                signal_lines += 0.5  # type: ignore[operator]

        scored_lines = signal_lines + noise_lines
        if scored_lines == 0:
            # Edge case: content is very short but not pure noise.
            # Give it a pass at low confidence rather than dropping it entirely.
            if total_lines <= 3 and len(content) < 100:
                return 0.2
            return 0.0

        return round(signal_lines / scored_lines, 3)

    # ------------------------------------------------------------------
    # Query refinement: keyword extraction for better semantic search
    # ------------------------------------------------------------------

    # Words to strip from queries — these dilute semantic signal
    _FILLER_WORDS = frozenset({
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
        'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our',
        'their', 'this', 'that', 'these', 'those', 'what', 'which',
        'who', 'whom', 'how', 'when', 'where', 'why', 'do', 'does',
        'did', 'can', 'could', 'will', 'would', 'shall', 'should',
        'may', 'might', 'must', 'have', 'has', 'had', 'not', 'no',
        'nor', 'so', 'if', 'then', 'else', 'than', 'too', 'very',
        'just', 'about', 'also', 'only', 'even', 'still', 'already',
        'really', 'actually', 'basically', 'please', 'thanks',
        'go', 'ahead', 'let', 'know', 'want', 'need', 'get', 'make',
        'right', 'sure', 'think', 'say', 'tell', 'use', 'help',
        'check', 'see', 'ok', 'okay', 'yes', 'yeah', 'well', 'like',
        'and', 'but', 'or', 'for', 'with', 'from', 'into', 'onto',
        'to', 'on', 'in', 'at', 'by', 'of', 'up', 'down', 'out',
        'because', 'without', 'something', 'anything', 'nothing',
        'any', 'some', 'each', 'every', 'all', 'both', 'few', 'more',
        'most', 'other', 'such', 'people', 'person', 'thing', 'things',
        'way', 'ways', 'kind', 'kinds', 'much', 'many', 'one', 'two',
        'remember', 'discussed', 'discuss', 'discussion', 'earlier',
        'before', 'after', 'now', 'later', 'often', 'always', 'never',
    })

    @classmethod
    def _extract_keywords(cls, query: str, max_words: int = 12) -> str:
        """Extract signal-bearing keywords from a verbose query.

        Strips fillers, question words, and conversational fluff.
        Returns a space-separated string of up to ``max_words``
        content-bearing terms — much better semantic search input
        than a raw 200-char chat message.
        """
        if not query or len(query) < 20:
            return query

        # Tokenize: lowercase, strip punctuation
        tokens = re.findall(r'[a-zA-Z0-9_-]+', query.lower())
        kept = [t for t in tokens if t not in cls._FILLER_WORDS and len(t) > 1]

        # If stripping gutted the query completely, return original
        if len(kept) < 1:
            return query

        # Deduplicate while preserving order
        seen: set = set()
        unique = []
        for t in kept:
            if t not in seen:
                seen.add(t)
                unique.append(t)

        return ' '.join(unique[:max_words])

    # ------------------------------------------------------------------
    # Query expansion: enrich short/ambiguous queries with recent context
    # ------------------------------------------------------------------

    def _expand_query(self, query: str) -> str:
        """Build an expanded search query from current + recent context.

        When a user says "fix it" or "yes go ahead", the current query has
        almost no semantic signal. This merges keywords from the last 1-3
        messages to reconstruct the topic.

        Returns the expanded query string.
        """
        current_keywords = self._extract_keywords(query)

        # Expansion trigger: raw query < 40 chars = likely follow-up
        # (Avoid relying on keyword count — filler-heavy queries break that)
        if len(query) >= 40 or not self._recent_queries:
            return current_keywords

        # Collect unique keywords from recent context (newest first)
        all_recent: List[str] = []
        seen: set = set()
        curr_tokens = set(current_keywords.lower().split())
        for rq in reversed(self._recent_queries):
            for token in rq.split():
                if token not in seen and token not in curr_tokens:
                    seen.add(token)
                    all_recent.append(token)

        if not all_recent:
            return current_keywords

        # Cap: 8 keywords from context
        context_part = ' '.join(all_recent[:8])

        # Only append current keywords if they're not pure filler
        if current_keywords.strip().lower() != query.strip().lower():
            expanded = f"{context_part} {current_keywords}"
        else:
            expanded = context_part

        logger.debug("MemPalace query expanded: '%s' → '%s'", query, expanded)
        return expanded

    # ------------------------------------------------------------------
    # Room priority: high-signal rooms first
    # ------------------------------------------------------------------

    # Rooms ordered by signal quality. Technical room (exchange mode)
    # is noisy raw JSON; decisions/problems are structured extracts.
    _PRIORITY_ROOMS = ['decisions', 'problems', 'architecture', 'general', 'technical']

    # ------------------------------------------------------------------
    # Room-targeted search: high-signal rooms first
    # ------------------------------------------------------------------

    def _targeted_search(self, query: str) -> List[Dict[str, Any]]:
        """Single broad search with client-side room-priority sorting.

        Previously made 4-5 subprocess calls per turn (one per room + fallback).
        Now: one broad search → deduplicate → sort by room priority → return top N.

        Returns processed + deduplicated results, capped at ``_default_results``.
        """
        # Single broad search — fetch 3x to have buffer for dedup + filtering
        broad = self._search(query, wing=self._wing, limit=self._default_results * 3)
        all_results = self._process_results(broad)

        # Sort by room priority then score (descending)
        room_order = {r: i for i, r in enumerate(self._PRIORITY_ROOMS)}
        all_results.sort(
            key=lambda r: (
                room_order.get(r.get('room', ''), 99),
                -(r.get('score', 0)),
            )
        )

        return all_results[:self._default_results]

    # ------------------------------------------------------------------
    # Freshness boosting: recent sessions get score multiplier
    # ------------------------------------------------------------------

    # Session filenames embed dates: session_20260428_... or session_cron_...
    _SESSION_DATE_RE = re.compile(r'session(?:_cron)?_(\d{8})_')

    @classmethod
    def _boost_freshness(cls, results: List[Dict[str, Any]]) -> None:
        """Apply recency multiplier to scores in-place.

        Sessions from the last 7 days get ×1.15, last 30 days get ×1.08.
        ChromaDB has no concept of time — this compensates.
        """
        now = datetime.now(timezone.utc)
        for r in results:
            source = r.get('source', '')
            m = cls._SESSION_DATE_RE.search(source)
            if not m:
                continue
            try:
                dt = datetime.strptime(m.group(1), '%Y%m%d').replace(tzinfo=timezone.utc)
                age_days = (now - dt).days
                score = r.get('score', 0)
                if age_days <= 7:
                    r['score'] = round(score * 1.15, 4)
                    r['_freshness'] = 'week'
                elif age_days <= 30:
                    r['score'] = round(score * 1.08, 4)
                    r['_freshness'] = 'month'
            except (ValueError, TypeError):
                pass

    # ------------------------------------------------------------------
    # Batched logging: summary every 10 turns instead of per-turn spam
    # ------------------------------------------------------------------

    def _emit_batched_summary(self) -> None:
        """Emit a single INFO log every 10 prefetch calls with aggregate stats.

        Per-turn activity stays at DEBUG. Only the summary hits INFO —
        one line per ~10 turns instead of one per turn.
        """
        total = self._summary_skips + self._summary_injects
        if total >= 10:
            logger.info(
                "MemPalace summary: %d turns (%d injected, %d skipped) — %d chars, %d total injects this session",
                total, self._summary_injects, self._summary_skips,
                self._summary_chars, self._inject_count,
            )
            self._summary_skips = 0
            self._summary_injects = 0
            self._summary_chars = 0

    # ------------------------------------------------------------------
    # Cross-session state persistence: save/load adaptive threshold
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        """Persist adaptive threshold state to disk for next session.

        Without this, every new Hermes session starts with a blank
        injection history and the threshold takes ~10 turns to calibrate.
        Saving the last 10 decisions means the next session starts
        already calibrated.
        """
        if not self._injection_history:
            return
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            state = {
                "injection_history": self._injection_history,
                "inject_count": self._inject_count,
                "last_session": self._session_id,
            }
            with open(self._state_file, "w", encoding="utf-8") as f:
                json.dump(state, f)
            logger.debug("MemPalace state saved to %s", self._state_file)
        except (OSError, TypeError) as e:
            logger.debug("MemPalace state save failed: %s", e)

    def _load_state(self) -> None:
        """Restore adaptive threshold state from a previous session."""
        try:
            if not self._state_file.exists():
                return
            with open(self._state_file, encoding="utf-8") as f:
                state = json.load(f)
            saved_history = state.get("injection_history", [])
            if isinstance(saved_history, list) and len(saved_history) <= 10:
                self._injection_history = [bool(v) for v in saved_history]
                self._inject_count = int(state.get("inject_count", 0))
                logger.debug(
                    "MemPalace state loaded: %d history entries, %d total injects",
                    len(self._injection_history),
                    self._inject_count,
                )
        except (json.JSONDecodeError, OSError, ValueError) as e:
            logger.debug("MemPalace state load failed: %s", e)
            self._injection_history = []
            self._inject_count = 0

    # ------------------------------------------------------------------
    # Adaptive threshold: auto-tunes injection sensitivity
    # ------------------------------------------------------------------

    def _should_inject(self, results: List[Dict[str, Any]]) -> bool:
        """Decide whether prefetch results are strong enough to inject.

        Auto-tunes thresholds based on recent history:
        - If most recent queries were skipped → loosen (surface more)
        - If most recent queries were injected → tighten (save tokens)
        - Otherwise use base thresholds

        Returns True only when results are likely to be actually useful.
        Saves ~1000 tokens per turn when nothing matches.
        """
        if not results:
            return False

        scores = [r.get('score', 0) for r in results]
        top = scores[0]
        avg = sum(scores) / len(scores) if scores else 0

        # Determine adaptive bias from recent history
        recent = self._injection_history[-10:] if self._injection_history else []
        skipped_count = sum(1 for v in recent if not v)
        injected_count = sum(1 for v in recent if v)

        # Base thresholds
        single_threshold = 0.55
        multi_top = 0.45
        multi_avg = 0.40

        if len(recent) >= 5:
            if skipped_count >= 7:  # mostly missing → loosen
                single_threshold = 0.50
                multi_top = 0.40
                multi_avg = 0.35
                logger.debug("MemPalace threshold: LOOSENED (skipped=%d/10)", skipped_count)
            elif injected_count >= 9:  # mostly injecting → tighten
                single_threshold = 0.60
                multi_top = 0.50
                multi_avg = 0.45
                logger.debug("MemPalace threshold: TIGHTENED (injected=%d/10)", injected_count)

        # Strong single match
        if top >= single_threshold:
            return True

        # Multiple moderate matches
        if top >= multi_top and len(results) >= 2 and avg >= multi_avg:
            return True

        # Otherwise: skip — not worth the token burn
        logger.debug(
            "MemPalace prefetch skipped: top=%.3f avg=%.3f results=%d threshold=%.2f",
            top, avg, len(results), single_threshold,
        )
        return False

    # ------------------------------------------------------------------
    # Core provider methods
    # ------------------------------------------------------------------

    def _search(
        self,
        query: str,
        wing: Optional[str] = None,
        room: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        if not self._binary:
            logger.debug("MemPalace binary not set; cannot search")
            return []

        cmd = [self._binary, "search", query, "--results", str(max(limit, 1))]
        if wing:
            cmd.extend(["--wing", wing])
        if room:
            cmd.extend(["--room", room])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired:
            logger.warning("MemPalace search timed out (%ds): %.80s",
                           self._timeout, query)
            return []
        except FileNotFoundError:
            logger.warning("MemPalace binary vanished: %s", self._binary)
            return []

        if result.returncode != 0:
            logger.debug("MemPalace search exit=%d: %.200s",
                         result.returncode, result.stderr)
            return []

        return self._parse_search_output(result.stdout, limit)

    @staticmethod
    def _parse_search_output(output: str, limit: int) -> List[Dict[str, Any]]:
        """Parse mempalace human-readable search output.

        Expected format::

            ============================================================
              Results for: "query"
            ============================================================

              [N] wing / room
                  Source: filename.json
                  Match:  0.XXX

                  <content excerpt>

              ────────────────────────────────────────────────────────
        """
        results: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None
        content_lines: List[str] = []

        for line in output.splitlines():
            stripped = line.strip()

            # Start of a result block: ``[N] wing / room``
            if stripped and stripped.startswith("[") and "]" in stripped[:8]:
                if current and content_lines:
                    current["content"] = "\n".join(content_lines).strip()
                    results.append(current)
                    content_lines = []

                idx_end = stripped.index("]")
                try:
                    idx = int(stripped[1:idx_end])
                except ValueError:
                    idx = len(results) + 1
                current = {"index": idx}

                rest = stripped[idx_end + 1:].strip()
                if " / " in rest:
                    parts = rest.split(" / ", 1)
                    current["wing"] = parts[0].strip()
                    current["room"] = parts[1].strip() if len(parts) > 1 else ""
                continue

            if current is None:
                continue

            if stripped.startswith("Source:"):
                current["source"] = stripped[len("Source:"):].strip()
                continue

            if stripped.startswith("Match:"):
                try:
                    current["score"] = float(stripped[len("Match:"):].strip())
                except ValueError:
                    current["score"] = 0.0
                continue

            # Skip horizontal rules and empty separator lines
            if stripped.startswith("──") or stripped.startswith("==="):
                continue

            # Accumulate content
            if stripped or content_lines:
                content_lines.append(line)

        # Final result
        if current and content_lines:
            current["content"] = "\n".join(content_lines).strip()
            results.append(current)

        return results[:limit]

    @staticmethod
    def _format_prefetch_with_meta(
        query: str,
        results: List[Dict[str, Any]],
        sources: Dict[str, str],
    ) -> str:
        """Format results with confidence metadata + keyword snippets.

        Produces a compact format that tells the agent:
        - How confident the match is (scores, source count)
        - What the most relevant sentence is (snippet)
        - The full content below for deeper context
        """
        if not results:
            return ""

        scores = [r.get("score", 0) for r in results]
        top_score = scores[0]
        avg_score = round(sum(scores) / len(scores), 3) if scores else 0
        room_dist = {}
        for r in results:
            room = r.get("room", "?")
            room_dist[room] = room_dist.get(room, 0) + 1

        confidence = "high" if top_score >= 0.58 else "medium" if top_score >= 0.48 else "low"

        lines = [
            "## MemPalace Recall",
            f"> query: \"{query}\" | confidence: {confidence} ({len(results)} sources, top={top_score:.2f}, avg={avg_score:.2f})",
            f"> rooms: {', '.join(f'{room}({n})' for room, n in sorted(room_dist.items()))}",
            "",
        ]

        for r in results:
            score = r.get("score", 0)
            source = r.get("source", "?")
            room = r.get("room", "")
            wing = r.get("wing", "")
            content = r.get("content", "")

            # Get the snippet for this source
            snippet = sources.get(source, "")

            scope = f"{wing}/{room}" if wing and room else wing or room or ""
            header = f"[{score:.2f}] {scope}"
            freshness = r.get("_freshness", "")
            if freshness:
                header += f" ({freshness})"

            lines.append(f"### {header} | {source}")
            if snippet:
                lines.append(f"> {snippet}")
                lines.append("")

            # Content (trimmed per result)
            max_content = 1500
            trimmed = content[:max_content] if len(content) > max_content else content
            if len(content) > max_content:
                trimmed += "\n... (truncated)"
            lines.append(trimmed)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Smart snippet: extract most keyword-relevant sentences
    # ------------------------------------------------------------------

    @classmethod
    def _extract_keyword_snippets(
        cls,
        results: List[Dict[str, Any]],
        query: str,
        max_sentences: int = 2,
        max_chars: int = 200,
    ) -> Dict[str, str]:
        """Extract 1-2 most query-relevant sentences from each result.

        Uses keyword-density heuristic — sentences that share the most
        unique terms with the query get selected. No model needed.

        Returns a dict mapping source → snippet string.
        """
        # Tokenize query keywords
        q_tokens = set(re.findall(r'[a-zA-Z0-9]{3,}', query.lower()))

        snippets: Dict[str, str] = {}
        if not q_tokens:
            return snippets

        for r in results:
            content = r.get("content", "")
            source = r.get("source", "")
            if not content or not source:
                continue

            # Split into sentences (rough: ., ?, !, newlines)
            sentences = re.split(r'(?<=[.!?])\s+|\n+', content)
            if not sentences:
                continue

            # Score each sentence by keyword overlap
            scored: List[tuple] = []
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 15 or len(sent) > 300:
                    continue
                s_tokens = set(re.findall(r'[a-zA-Z0-9]{3,}', sent.lower()))
                overlap = len(q_tokens & s_tokens)
                if overlap > 0:
                    scored.append((overlap, sent))

            if scored:
                scored.sort(key=lambda x: -x[0])
                best = [s[1] for s in scored[:max_sentences]]
                snippet = " ".join(best)
                if len(snippet) > max_chars:
                    snippet = snippet[:max_chars] + "..."
                snippets[source] = snippet

        return snippets


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the MemPalace memory provider with the plugin system."""
    provider = MemPalaceProvider()
    ctx.register_memory_provider(provider)
