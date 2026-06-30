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
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
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
    """Read provider config from ``config.yaml`` → ``memory.mempalace``."""
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
        self._session_id: str = ""

    # -- MemoryProvider ABC ------------------------------------------------

    @property
    def name(self) -> str:
        return "mempalace"

    def is_available(self) -> bool:
        if not self._binary:
            logger.debug("MemPalace binary not found")
            return False
        try:
            result = subprocess.run(
                [self._binary, "--help"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError) as e:
            logger.debug("MemPalace availability check failed: %s", e)
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
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
            results = self._search(query, limit=self._default_results)
            results = self._process_results(results)
            if not results:
                return ""
            prefetch_text = self._format_prefetch(results)
            # Respect char budget
            if len(prefetch_text) > self._max_prefetch_chars:
                prefetch_text = prefetch_text[:self._max_prefetch_chars] + "\n\n... (budget limit)"
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
        pass  # read-only — writes happen through mempalace mine

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
        pass

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
        """Clean content, deduplicate by session, and filter noise."""
        if not results:
            return []

        # Clean each result's content
        for r in results:
            r["content"] = self._clean_content(r.get("content", ""))
            # Drop results that became empty after cleaning
        results = [r for r in results if r.get("content", "").strip()]

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

    @staticmethod
    def _format_prefetch(results: List[Dict[str, Any]]) -> str:
        if not results:
            return ""

        lines = ["## MemPalace Recall"]
        for r in results:
            score = r.get("score", 0)
            source = r.get("source", "unknown")
            room = r.get("room", "")
            wing = r.get("wing", "")
            content = r.get("content", "")

            scope = f"{wing}/{room}" if wing and room else wing or room or ""
            header = f"[{score:.2f}] {scope}" if scope else f"[{score:.2f}]"
            if source:
                header += f" ({source})"

            lines.append(f"\n### {header}")

            max_len = 2000
            trimmed = content[:max_len] if len(content) > max_len else content
            if len(content) > max_len:
                trimmed += "\n... (truncated)"
            lines.append(trimmed)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
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
    def _format_prefetch(results: List[Dict[str, Any]]) -> str:
        if not results:
            return ""

        lines = ["## MemPalace Recall"]
        for r in results:
            score = r.get("score", 0)
            source = r.get("source", "unknown")
            room = r.get("room", "")
            wing = r.get("wing", "")
            content = r.get("content", "")

            scope = f"{wing}/{room}" if wing and room else wing or room or ""
            header = f"[{score:.2f}] {scope}" if scope else f"[{score:.2f}]"
            if source:
                header += f" ({source})"

            lines.append(f"\n### {header}")

            max_len = 2000
            trimmed = content[:max_len] if len(content) > max_len else content
            if len(content) > max_len:
                trimmed += "\n... (truncated)"
            lines.append(trimmed)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the MemPalace memory provider with the plugin system."""
    provider = MemPalaceProvider()
    ctx.register_memory_provider(provider)
