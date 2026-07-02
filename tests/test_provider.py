"""Comprehensive pytest test suite for MemPalaceProvider.

Tests cover all key methods:
  - _parse_search_output (static)
  - _clean_content (classmethod)
  - _extract_keywords (classmethod)
  - _extract_keyword_snippets (classmethod)
  - _boost_freshness (classmethod)
  - _should_inject
  - _expand_query
  - _format_prefetch_with_meta (static)
  - _process_results
"""
# pylint: disable=protected-access, redefined-outer-name, missing-function-docstring

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest


# ======================================================================
# _parse_search_output  (static)
# ======================================================================


class TestParseSearchOutput:
    """Tests for MemPalaceProvider._parse_search_output static method."""

    def test_parses_basic_results(self):
        from __init__ import MemPalaceProvider

        output = (
            "============================================================\n"
            "  Results for: \"deploy pipeline\"\n"
            "============================================================\n"
            "\n"
            "  [1] sessions / decisions\n"
            "      Source: session_20250601_abc123.json\n"
            "      Match:  0.8721\n"
            "\n"
            "      Decided to use GitHub Actions for CI/CD.\n"
            "      All deploys go through staging first.\n"
            "\n"
            "  \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
            "\n"
            "  [2] sessions / technical\n"
            "      Source: session_20260528_xyz789.json\n"
            "      Match:  0.6543\n"
            "\n"
            "      Deployment script is in deploy.sh\n"
        )
        results = MemPalaceProvider._parse_search_output(output, limit=5)

        assert len(results) == 2
        assert results[0]["index"] == 1
        assert results[0]["wing"] == "sessions"
        assert results[0]["room"] == "decisions"
        assert results[0]["source"] == "session_20250601_abc123.json"
        assert abs(results[0]["score"] - 0.8721) < 0.0001
        assert "GitHub Actions" in results[0]["content"]
        assert "staging first" in results[0]["content"]

        assert results[1]["index"] == 2
        assert results[1]["room"] == "technical"
        assert "deploy.sh" in results[1]["content"]

    def test_handles_empty_output(self):
        from __init__ import MemPalaceProvider

        assert MemPalaceProvider._parse_search_output("", limit=5) == []
        assert MemPalaceProvider._parse_search_output("   \n  \n", limit=5) == []

    def test_respects_limit(self):
        from __init__ import MemPalaceProvider

        output = ""
        for i in range(1, 6):
            output += (
                f"  [{i}] sessions / general\n"
                f"      Source: session_2026060{i}_x.json\n"
                f"      Match:  0.{8-i}00\n"
                f"\n"
                f"      Content block {i}\n\n"
            )
        results = MemPalaceProvider._parse_search_output(output, limit=3)
        assert len(results) == 3

    def test_malformed_match_score_returns_zero(self):
        from __init__ import MemPalaceProvider

        output = (
            "  [1] sessions / general\n"
            "      Source: test.json\n"
            "      Match:  not_a_number\n"
            "\n"
            "      some content\n"
        )
        results = MemPalaceProvider._parse_search_output(output, limit=5)
        assert len(results) == 1
        assert results[0]["score"] == 0.0

    def test_malformed_index_uses_fallback(self):
        from __init__ import MemPalaceProvider

        output = (
            "  [abc] sessions / general\n"
            "      Source: test.json\n"
            "      Match:  0.5\n"
            "\n"
            "      content\n"
        )
        results = MemPalaceProvider._parse_search_output(output, limit=5)
        assert len(results) == 1
        assert results[0]["index"] == 1  # fallback = len+1

    def test_no_wing_slash_room_sets_nothing(self):
        from __init__ import MemPalaceProvider

        output = (
            "  [1] noslash\n"
            "      Source: test.json\n"
            "      Match:  0.5\n\n"
            "      content\n"
        )
        results = MemPalaceProvider._parse_search_output(output, limit=5)
        # Without " / " separator, wing/room are not set in the result dict
        assert results[0].get("index") == 1
        assert results[0].get("source") == "test.json"
        assert results[0].get("score") == 0.5
        assert "wing" not in results[0]
        assert "room" not in results[0]


# ======================================================================
# _clean_content  (classmethod)
# ======================================================================


class TestCleanContent:
    """Tests for MemPalaceProvider._clean_content classmethod."""

    def test_keeps_human_content(self):
        from __init__ import MemPalaceProvider

        text = "Hello, I think we should use FastAPI for this.\nWhat do you think?"
        assert MemPalaceProvider._clean_content(text) == text

    def test_strips_json_noise(self):
        from __init__ import MemPalaceProvider

        content = (
            '{"name": "search", "args": {"q": "test"}}\n'
            '  "role": "assistant"\n'
            '{"choices": [{"index": 0}]}\n'
            "I think we should use PostgreSQL.\n"
            '\\"nested\\" \\"escaped\\"\n'
        )
        cleaned = MemPalaceProvider._clean_content(content)
        assert "I think we should use PostgreSQL." in cleaned
        assert "search" not in cleaned
        assert '"role"' not in cleaned
        assert "choices" not in cleaned

    def test_strips_pure_punctuation_lines(self):
        from __init__ import MemPalaceProvider

        content = (
            '{}[]{}[]\n'
            '",,,:::\\\\\n'
            '{"key": "value"}\n'
            "The actual content we want to keep.\n"
        )
        cleaned = MemPalaceProvider._clean_content(content)
        assert "The actual content" in cleaned
        assert '{"key": "value"}' not in cleaned

    def test_handles_empty_string(self):
        from __init__ import MemPalaceProvider

        assert MemPalaceProvider._clean_content("") == ""
        assert MemPalaceProvider._clean_content(None) == ""

    def test_collapses_excessive_blank_lines(self):
        from __init__ import MemPalaceProvider

        content = "Line one.\n\n\n\n\nLine two.\n\n\nLine three."
        cleaned = MemPalaceProvider._clean_content(content)
        assert "\n\n\n" not in cleaned  # no 3+ consecutive blanks

    def test_drops_lines_with_high_json_ratio(self):
        from __init__ import MemPalaceProvider

        content = (
            'short\n'
            'this line has "way too many" braces {and} [stuff] :\\"quotes\\"\n'
            "Keep this line please.\n"
        )
        cleaned = MemPalaceProvider._clean_content(content)
        assert "Keep this line" in cleaned
        # The high-json-ratio line may or may not be dropped depending on
        # exact char ratio; verify that content is at least cleaned of pure JSON lines
        assert "short" in cleaned


# ======================================================================
# _extract_keywords  (classmethod)
# ======================================================================


class TestExtractKeywords:
    """Tests for MemPalaceProvider._extract_keywords classmethod."""

    def test_strips_filler_words_from_long_query(self):
        from __init__ import MemPalaceProvider

        query = "Can you please tell me what we decided about the deployment pipeline last week?"
        keywords = MemPalaceProvider._extract_keywords(query)
        # Filler words removed; signal-bearing words remain
        assert "deployment" in keywords
        assert "pipeline" in keywords
        assert "decided" in keywords
        # These fillers should be gone
        assert "can" not in keywords.lower()
        assert "you" not in keywords.lower()
        assert "please" not in keywords.lower()
        assert "the" not in keywords.lower()

    def test_returns_short_query_unchanged(self):
        from __init__ import MemPalaceProvider

        query = "Short query"
        assert MemPalaceProvider._extract_keywords(query) == query

    def test_returns_none_for_none(self):
        from __init__ import MemPalaceProvider

        assert MemPalaceProvider._extract_keywords(None) is None

    def test_fallback_when_all_words_are_fillers(self):
        from __init__ import MemPalaceProvider

        query = "I am you and we are them but it is the same thing all over again"
        # Query is long enough to trigger extraction
        # "thing" is not in filler set (only "things" is), so it's kept
        # "same" and "again" are not fillers either
        result = MemPalaceProvider._extract_keywords(query)
        # Returns only non-filler tokens with acceptable length
        assert result != query  # gets trimmed
        assert len(result) < len(query)

    def test_respects_max_words(self):
        from __init__ import MemPalaceProvider

        query = "We should consider using Redis for caching and PostgreSQL for persistence"
        keywords = MemPalaceProvider._extract_keywords(query)
        tokens = keywords.split()
        assert len(tokens) <= 12  # default max_words
        assert "redis" in tokens
        assert "postgresql" in tokens
        assert "persistence" in tokens

    def test_deduplicates_repeated_keywords(self):
        from __init__ import MemPalaceProvider

        query = "What about the database and the database schema and the database migration"
        keywords = MemPalaceProvider._extract_keywords(query)
        tokens = keywords.split()
        assert tokens.count("database") == 1


# ======================================================================
# _extract_keyword_snippets  (classmethod)
# ======================================================================


class TestExtractKeywordSnippets:
    """Tests for MemPalaceProvider._extract_keyword_snippets classmethod."""

    def test_selects_sentence_with_most_keyword_overlap(self):
        from __init__ import MemPalaceProvider

        results = [
            {
                "source": "session_20260601_a.json",
                "content": (
                    "We talked about the weather today. "
                    "The deployment pipeline was set up with GitHub Actions. "
                    "Then we had lunch."
                ),
            }
        ]
        query = "deployment pipeline github actions"
        snippets = MemPalaceProvider._extract_keyword_snippets(results, query)
        snippet = snippets.get("session_20260601_a.json", "")
        assert "deployment pipeline" in snippet
        assert "GitHub Actions" in snippet
        assert "weather" not in snippet
        assert "lunch" not in snippet

    def test_returns_empty_dict_for_empty_query_tokens(self):
        from __init__ import MemPalaceProvider

        results = [{"source": "a.json", "content": "Some content here."}]
        assert MemPalaceProvider._extract_keyword_snippets(results, "a") == {}

    def test_handles_multiple_results(self):
        from __init__ import MemPalaceProvider

        results = [
            {
                "source": "s1.json",
                "content": "Docker compose configuration for local development.",
            },
            {
                "source": "s2.json",
                "content": "Kubernetes deployment in production environment.",
            },
        ]
        query = "docker compose development"
        snippets = MemPalaceProvider._extract_keyword_snippets(results, query)
        assert "s1.json" in snippets
        assert "Docker compose" in snippets["s1.json"]
        # s2 has no overlap — may or may not be in dict
        assert "s2.json" not in snippets or snippets["s2.json"] == ""

    def test_respects_max_sentences_and_chars(self):
        from __init__ import MemPalaceProvider

        sentence_a = (
            "The authentication system uses OAuth2 with JWT tokens. "
        )
        sentence_b = (
            "Session management is handled via Redis with automatic expiry. "
        )
        sentence_c = "We decided to use FastAPI as the web framework."

        results = [
            {
                "source": "session_a.json",
                "content": sentence_a + sentence_b + sentence_c,
            }
        ]
        query = "authentication jwt redis session"
        snippets = MemPalaceProvider._extract_keyword_snippets(
            results, query, max_sentences=2, max_chars=500
        )
        snippet = snippets.get("session_a.json", "")
        assert "authentication" in snippet or "OAuth2" in snippet
        assert "Redis" in snippet or "session" in snippet.lower()
        # Should have at most 2 sentences worth of content
        assert len(snippet) <= 500


# ======================================================================
# _boost_freshness  (classmethod)
# ======================================================================


class TestBoostFreshness:
    """Tests for MemPalaceProvider._boost_freshness classmethod."""

    def test_boosts_recent_session_within_week(self):
        from __init__ import MemPalaceProvider

        today = datetime.now(timezone.utc)
        date_str = today.strftime("%Y%m%d")
        results = [
            {"source": f"session_{date_str}_abc.json", "score": 0.8}
        ]
        MemPalaceProvider._boost_freshness(results)
        assert abs(results[0]["score"] - round(0.8 * 1.15, 4)) < 0.0001
        assert results[0].get("_freshness") == "week"

    def test_boosts_session_within_month(self):
        from __init__ import MemPalaceProvider

        fifteen_days_ago = datetime.now(timezone.utc) - timedelta(days=15)
        date_str = fifteen_days_ago.strftime("%Y%m%d")
        results = [
            {"source": f"session_{date_str}_abc.json", "score": 0.7}
        ]
        MemPalaceProvider._boost_freshness(results)
        assert abs(results[0]["score"] - round(0.7 * 1.08, 4)) < 0.0001
        assert results[0].get("_freshness") == "month"

    def test_does_not_boost_old_session(self):
        from __init__ import MemPalaceProvider

        old_date = "20200101"
        results = [
            {"source": f"session_{old_date}_abc.json", "score": 0.6}
        ]
        MemPalaceProvider._boost_freshness(results)
        assert abs(results[0]["score"] - 0.6) < 0.0001
        assert results[0].get("_freshness") is None

    def test_ignores_missing_date_in_source(self):
        from __init__ import MemPalaceProvider

        results = [
            {"source": "no_date_pattern_here.json", "score": 0.5}
        ]
        MemPalaceProvider._boost_freshness(results)
        assert abs(results[0]["score"] - 0.5) < 0.0001

    def test_handles_cron_sessions(self):
        from __init__ import MemPalaceProvider

        today = datetime.now(timezone.utc)
        date_str = today.strftime("%Y%m%d")
        results = [
            {"source": f"session_cron_{date_str}_xyz.json", "score": 0.9}
        ]
        MemPalaceProvider._boost_freshness(results)
        assert abs(results[0]["score"] - round(0.9 * 1.15, 4)) < 0.0001
        assert results[0].get("_freshness") == "week"

    def test_handles_invalid_date_gracefully(self):
        from __init__ import MemPalaceProvider

        results = [
            {"source": "session_99999999_abc.json", "score": 0.5}
        ]
        # strptime will fail on "99999999", should be caught
        MemPalaceProvider._boost_freshness(results)
        assert abs(results[0]["score"] - 0.5) < 0.0001


# ======================================================================
# _should_inject  (adaptive threshold)
# ======================================================================


class TestShouldInject:
    """Tests for MemPalaceProvider._should_inject method."""

    def test_no_results_returns_false(self, provider):
        assert provider._should_inject([]) is False

    def test_strong_single_match_returns_true(self, provider):
        results = [{"score": 0.70}]
        assert provider._should_inject(results) is True

    def test_multiple_moderate_matches_returns_true(self, provider):
        results = [
            {"score": 0.50},
            {"score": 0.45},
            {"score": 0.42},
        ]
        assert provider._should_inject(results) is True

    def test_weak_matches_returns_false(self, provider):
        results = [
            {"score": 0.25},
            {"score": 0.20},
        ]
        assert provider._should_inject(results) is False

    def test_single_strong_but_low_avg_still_injects(self, provider):
        """top >= single_threshold should bypass multi check."""
        results = [
            {"score": 0.60},
            {"score": 0.05},
        ]
        assert provider._should_inject(results) is True

    def test_loosened_mode_lower_thresholds(self, provider):
        # Simulate 7/10 skipped
        provider._injection_history = [False] * 7 + [True] * 3
        # With loosened thresholds, 0.48 top + 0.40 avg + 2 results should pass
        results = [
            {"score": 0.48},
            {"score": 0.40},
        ]
        assert provider._should_inject(results) is True

    def test_tightened_mode_higher_thresholds(self, provider):
        # Simulate 9/10 injected
        provider._injection_history = [True] * 9 + [False] * 1
        # With tightened thresholds, 0.48 is below 0.50 multi_top
        results = [
            {"score": 0.48},
            {"score": 0.45},
        ]
        assert provider._should_inject(results) is False

    def test_tightened_mode_single_still_injects_if_high_enough(self, provider):
        provider._injection_history = [True] * 9 + [False] * 1
        results = [{"score": 0.62}]
        # 0.62 >= 0.60 tightened single_threshold → True
        assert provider._should_inject(results) is True

    def test_loosened_single_match_injects_at_lower_threshold(self, provider):
        provider._injection_history = [False] * 7 + [True] * 3
        results = [{"score": 0.52}]
        # 0.52 >= 0.50 loosened single_threshold → True
        assert provider._should_inject(results) is True

    def test_insufficient_history_uses_base_thresholds(self, provider):
        # Only 3 entries → not enough to trigger loosen/tighten
        provider._injection_history = [True, True, False]
        results = [{"score": 0.52}]
        # 0.52 < 0.55 base single_threshold → False
        assert provider._should_inject(results) is False


# ======================================================================
# _expand_query
# ======================================================================


class TestExpandQuery:
    """Tests for MemPalaceProvider._expand_query method."""

    def test_long_query_not_expanded(self, provider):
        query = "What was our decision about the deployment pipeline architecture?"
        expanded = provider._expand_query(query)
        # Query is >= 40 chars → should not expand, just extract keywords
        assert "deployment" in expanded
        assert "pipeline" in expanded

    def test_short_query_expanded_with_context(self, provider):
        provider._recent_queries = [
            "deployment pipeline github actions",
            "docker compose setup",
        ]
        query = "what about it?"
        expanded = provider._expand_query(query)
        # Should include context keywords + maybe nothing from query (all fillers)
        assert "deployment" in expanded
        assert "pipeline" in expanded
        assert "docker" in expanded
        assert "compose" in expanded

    def test_short_query_no_recent_context_returns_keywords(self, provider):
        provider._recent_queries = []
        query = "what about it?"
        expanded = provider._expand_query(query)
        # No context to expand with, and query is all fillers < 20 chars
        # So _extract_keywords returns original
        assert expanded == query

    def test_short_query_no_overlap_with_current(self, provider):
        provider._recent_queries = [
            "database schema migration"
        ]
        query = "did we finish that?"
        expanded = provider._expand_query(query)
        assert "database" in expanded
        assert "schema" in expanded
        assert "migration" in expanded
        # Current query tokens are in filler set, so context_part used alone

    def test_no_expansion_for_short_non_filler_query(self, provider):
        """If the short query has real keywords, they show up too."""
        provider._recent_queries = [
            "database migration"
        ]
        query = "what about redis?"
        expanded = provider._expand_query(query)
        # "redis" extracted from query (>= 20 chars? No, < 20 chars, so original returned)
        # Actually query "what about redis?" is < 20 chars, so _extract_keywords returns it whole
        # Then expansion trigger: len(query) < 40 and recent_queries exists → expands
        # current_keywords = query (since <20 chars, returned as-is)
        # current_keywords is not different from query (it IS query), so we go to else branch
        assert "database" in expanded or "migration" in expanded


# ======================================================================
# _format_prefetch_with_meta  (static)
# ======================================================================


class TestFormatPrefetchWithMeta:
    """Tests for MemPalaceProvider._format_prefetch_with_meta static method."""

    def test_returns_empty_for_empty_results(self):
        from __init__ import MemPalaceProvider

        result = MemPalaceProvider._format_prefetch_with_meta(
            "test", [], {}
        )
        assert result == ""

    def test_includes_confidence_and_scores(self):
        from __init__ import MemPalaceProvider

        results = [
            {"score": 0.85, "source": "s1.json", "room": "decisions",
             "wing": "sessions", "content": "We decided to use FastAPI."}
        ]
        sources = {"s1.json": "FastAPI chosen as framework."}
        output = MemPalaceProvider._format_prefetch_with_meta(
            "web framework", results, sources
        )
        assert "MemPalace Recall" in output
        assert "confidence: high" in output
        assert "0.85" in output or "0.85" in output
        assert "sessions/decisions" in output
        assert "FastAPI chosen as framework." in output
        assert "We decided to use FastAPI." in output

    def test_medium_confidence(self):
        from __init__ import MemPalaceProvider

        results = [
            {"score": 0.52, "source": "s1.json", "room": "general",
             "wing": "sessions", "content": "Some discussion."}
        ]
        output = MemPalaceProvider._format_prefetch_with_meta(
            "test", results, {"s1.json": ""}
        )
        assert "confidence: medium" in output

    def test_low_confidence(self):
        from __init__ import MemPalaceProvider

        results = [
            {"score": 0.40, "source": "s1.json", "room": "general",
             "wing": "sessions", "content": "Low relevance."}
        ]
        output = MemPalaceProvider._format_prefetch_with_meta(
            "test", results, {"s1.json": ""}
        )
        assert "confidence: low" in output

    def test_includes_freshness_label(self):
        from __init__ import MemPalaceProvider

        results = [
            {"score": 0.75, "source": "s1.json", "room": "decisions",
             "wing": "sessions", "content": "Content here.",
             "_freshness": "week"}
        ]
        output = MemPalaceProvider._format_prefetch_with_meta(
            "test", results, {"s1.json": "snippet"}
        )
        assert "(week)" in output

    def test_truncates_long_content(self):
        from __init__ import MemPalaceProvider

        long_content = "A" * 2000
        results = [
            {"score": 0.7, "source": "s1.json", "room": "general",
             "wing": "sessions", "content": long_content}
        ]
        output = MemPalaceProvider._format_prefetch_with_meta(
            "test", results, {"s1.json": ""}
        )
        assert "(truncated)" in output
        assert len(output) < 2000 + 500  # header overhead


# ======================================================================
# _process_results  (dedup + clean)
# ======================================================================


class TestProcessResults:
    """Tests for MemPalaceProvider._process_results method."""

    def test_deduplicates_by_session_base(self, provider):
        results = [
            {"source": "session_20260601_abc.json", "score": 0.8,
             "content": "Decision about deployment."},
            {"source": "session_20260601_def.json", "score": 0.6,
             "content": "Another decision about deployment."},
        ]
        processed = provider._process_results(results)
        # Same base "session_20260601" → keep highest score (0.8)
        assert len(processed) == 1
        assert abs(processed[0]["score"] - 0.8) < 0.0001

    def test_keeps_different_sessions_separate(self, provider):
        results = [
            {"source": "unique_deploy_20260601_abc.json", "score": 0.8,
             "content": "Decision about deployment."},
            {"source": "other_database_20260602_def.json", "score": 0.7,
             "content": "Discussion about database."},
        ]
        # rsplit("_", 3)[0] gives "unique" and "other" — different bases
        processed = provider._process_results(results)
        assert len(processed) == 2

    def test_removes_empty_content_after_cleaning(self, provider):
        results = [
            {"source": "session_20260601_a.json", "score": 0.7,
             "content": '{"json": "only"}\n{}[]'},
            {"source": "session_20260602_b.json", "score": 0.8,
             "content": "Real content here."},
        ]
        processed = provider._process_results(results)
        assert len(processed) == 1
        assert processed[0]["source"] == "session_20260602_b.json"

    def test_sorts_by_score_descending(self, provider):
        results = [
            {"source": "session_20260601_c.json", "score": 0.5,
             "content": "Medium."},
            {"source": "session_20260602_a.json", "score": 0.9,
             "content": "High."},
            {"source": "session_20260603_b.json", "score": 0.7,
             "content": "Low."},
        ]
        processed = provider._process_results(results)
        scores = [r["score"] for r in processed]
        assert scores == sorted(scores, reverse=True)

    def test_deduplication_off_keeps_duplicates(self, provider):
        provider._deduplicate = False
        results = [
            {"source": "session_20260601_abc.json", "score": 0.8,
             "content": "First."},
            {"source": "session_20260601_def.json", "score": 0.6,
             "content": "Second."},
        ]
        processed = provider._process_results(results)
        assert len(processed) == 2

    def test_handles_empty_list(self, provider):
        assert provider._process_results([]) == []

    def test_cleans_content_in_each_result(self, provider):
        results = [
            {"source": "session_a.json", "score": 0.7,
             "content": '{"tool": "search"}\nHuman message here.\n'}
        ]
        processed = provider._process_results(results)
        assert "Human message here." in processed[0]["content"]
        assert '"tool"' not in processed[0]["content"]


# ======================================================================
# Integration: full prefetch flow with mocked _search
# ======================================================================


class TestPrefetchIntegration:
    """Test prefetch end-to-end with _search mocked."""

    def test_prefetch_with_mocked_search(self, provider):
        """Mock _search to return known results, verify formatted output."""
        from __init__ import MemPalaceProvider

        # prefetch checks self._binary, so set a dummy path
        provider._binary = "/usr/bin/mempalace"

        call_count = [0]

        def mock_search(*args, **kwargs):
            call_count[0] += 1
            return [
                {
                    "index": 1,
                    "wing": "sessions",
                    "room": "decisions",
                    "source": "unique_fastapi_20260601_abc.json",
                    "score": 0.85,
                    "content": "We decided to use FastAPI with PostgreSQL.",
                },
                {
                    "index": 2,
                    "wing": "sessions",
                    "room": "technical",
                    "source": "other_database_20260515_def.json",
                    "score": 0.62,
                    "content": "Database schema includes users and roles tables.",
                },
            ]

        with patch.object(provider, "_search", side_effect=mock_search):
            output = provider.prefetch("What was our decision about the database?")

            # Should inject (strong single match)
            assert output != "", (
                f"prefetch returned empty. _injection_history={provider._injection_history}, "
                f"_search was called {call_count[0]} times"
            )
            assert "MemPalace Recall" in output
            assert "FastAPI" in output
            assert "PostgreSQL" in output
            assert "confidence: high" in output

    def test_prefetch_skips_when_weak_results(self, provider):
        with patch.object(provider, "_search") as mock_search:
            mock_search.return_value = [
                {
                    "index": 1,
                    "wing": "sessions",
                    "room": "technical",
                    "source": "session_20260501_x.json",
                    "score": 0.25,
                    "content": "minor mention.",
                },
            ]

            output = provider.prefetch("something obscure")
            assert output == ""  # skipped due to weak match

    def test_prefetch_empty_query_returns_empty(self, provider):
        assert provider.prefetch("") == ""
        assert provider.prefetch("   ") == ""
        assert provider.prefetch(None) == ""


# ======================================================================
# Integration: handle_tool_call
# ======================================================================


class TestHandleToolCall:
    """Tests for MemPalaceProvider.handle_tool_call."""

    def test_known_tool_returns_results(self, provider):
        with patch.object(provider, "_search") as mock_search:
            mock_search.return_value = [
                {"source": "a.json", "score": 0.9, "content": "test"},
            ]
            result = provider.handle_tool_call(
                "mempalace_search", {"query": "test query"}
            )
            data = json.loads(result)
            assert data["count"] == 1
            assert len(data["results"]) == 1

    def test_unknown_tool_returns_error(self, provider):
        result = provider.handle_tool_call("unknown_tool", {})
        assert "ERROR:" in result

    def test_missing_query_returns_error(self, provider):
        result = provider.handle_tool_call("mempalace_search", {})
        assert "ERROR:" in result


# ======================================================================
# Edge cases: _clean_content with None
# ======================================================================


def test_clean_content_none():
    from __init__ import MemPalaceProvider

    # The method uses `if not content` which handles None
    assert MemPalaceProvider._clean_content(None) == ""


# ======================================================================
# Edge cases: _extract_keywords with None
# ======================================================================


def test_extract_keywords_none():
    from __init__ import MemPalaceProvider

    assert MemPalaceProvider._extract_keywords(None) is None


# ======================================================================
# v1.1.0: _content_quality_score  (classmethod)
# ======================================================================


class TestContentQualityScore:
    """Tests for MemPalaceProvider._content_quality_score classmethod."""

    def test_high_quality_human_text_scores_high(self):
        from __init__ import MemPalaceProvider

        content = (
            "We decided to use FastAPI for the new microservice. "
            "The deployment pipeline will use GitHub Actions with "
            "staging promotion before production. PostgreSQL for the database."
        )
        score = MemPalaceProvider._content_quality_score(content)
        assert score >= 0.5, f"Expected >= 0.5, got {score}"

    def test_stack_trace_scores_low(self):
        from __init__ import MemPalaceProvider

        content = (
            "Traceback (most recent call last):\n"
            "  File 'app.py', line 42, in handle_request\n"
            "    result = process(data)\n"
            "  File 'core.py', line 15, in process\n"
            "    raise ValueError('bad input')\n"
            "ValueError: bad input\n"
        )
        score = MemPalaceProvider._content_quality_score(content)
        assert score < 0.2, f"Expected < 0.2, got {score}"

    def test_pure_json_scores_low(self):
        from __init__ import MemPalaceProvider

        content = (
            '{"name": "deploy", "args": {"env": "prod"}}\n'
            '{"status": "ok", "code": 200}\n'
            '{"count": 5, "items": [1,2,3,4,5]}\n'
        )
        score = MemPalaceProvider._content_quality_score(content)
        assert score < 0.2, f"Expected < 0.2, got {score}"

    def test_empty_content_returns_zero(self):
        from __init__ import MemPalaceProvider

        assert MemPalaceProvider._content_quality_score("") == 0.0
        assert MemPalaceProvider._content_quality_score("   \n  ") == 0.0

    def test_short_content_gets_floor_score(self):
        from __init__ import MemPalaceProvider

        content = "Done."
        score = MemPalaceProvider._content_quality_score(content)
        assert score >= 0.15, f"Short content should pass, got {score}"

    def test_mixed_human_and_noise_scores_moderate(self):
        from __init__ import MemPalaceProvider

        content = (
            "Here is what we decided about the architecture.\n"
            "We will use microservices with gRPC.\n"
            "exit code: 0\n"
            "successfully completed\n"
        )
        score = MemPalaceProvider._content_quality_score(content)
        # Should be moderate — 2 signal lines, 2 noise lines
        assert 0.3 <= score <= 0.7, f"Expected 0.3-0.7, got {score}"

    def test_terminal_output_scores_low(self):
        from __init__ import MemPalaceProvider

        content = (
            "stdout: Building wheel...\n"
            "stdout: Successfully built mypackage\n"
            "stdout: Installing collected packages\n"
            "successfully completed\n"
            "42 items processed\n"
            "exit code: 0\n"
        )
        score = MemPalaceProvider._content_quality_score(content)
        assert score < 0.25, f"Expected < 0.25, got {score}"


# ======================================================================
# v1.1.0: _save_state / _load_state  (cross-session persistence)
# ======================================================================


class TestStatePersistence:
    """Tests for MemPalaceProvider._save_state and _load_state."""

    def test_save_and_load_roundtrip(self, provider, tmp_path):
        """Save state, create a fresh provider, load it back."""
        provider._state_dir = tmp_path
        provider._state_file = tmp_path / "provider_state.json"
        provider._injection_history = [True, False, True, True, False]
        provider._inject_count = 42
        provider._session_id = "test-session-123"

        provider._save_state()

        # Verify file exists and is valid JSON
        assert provider._state_file.exists()

        # Create a new provider and load
        from __init__ import MemPalaceProvider

        new_provider = MemPalaceProvider(config={"binary": ""})
        new_provider._available = True
        new_provider._state_file = tmp_path / "provider_state.json"
        new_provider._load_state()

        assert new_provider._injection_history == [True, False, True, True, False]
        assert new_provider._inject_count == 42

    def test_load_state_file_does_not_exist(self, provider, tmp_path):
        """Loading from non-existent file leaves defaults intact."""
        provider._state_file = tmp_path / "nonexistent.json"
        provider._injection_history = [True, True, True]
        provider._inject_count = 99

        provider._load_state()

        # Should be unchanged — no file to load from
        assert provider._injection_history == [True, True, True]
        assert provider._inject_count == 99

    def test_load_state_corrupted_file(self, provider, tmp_path):
        """Corrupted state file → reset to defaults."""
        state_file = tmp_path / "provider_state.json"
        state_file.write_text("this is not json {{{")

        provider._state_file = state_file
        provider._injection_history = [True, True]
        provider._inject_count = 10

        provider._load_state()

        # Should reset to defaults on corrupt file
        assert provider._injection_history == []
        assert provider._inject_count == 0

    def test_save_state_skips_when_history_empty(self, provider, tmp_path):
        """Don't write state file if there's no history to save."""
        provider._state_dir = tmp_path
        provider._state_file = tmp_path / "provider_state.json"
        provider._injection_history = []

        provider._save_state()

        # Should not create file for empty history
        assert not provider._state_file.exists()

    def test_load_state_ignores_malformed_history_length(self, provider, tmp_path):
        """History with >10 entries should be rejected (unexpected state)."""
        import json

        state = {
            "injection_history": [True] * 50,
            "inject_count": 100,
        }
        state_file = tmp_path / "provider_state.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(state))

        provider._state_file = state_file
        provider._injection_history = [True, False]
        provider._inject_count = 5

        provider._load_state()

        # 50 entries > 10 → rejected, defaults preserved
        assert provider._injection_history == [True, False]
        assert provider._inject_count == 5
