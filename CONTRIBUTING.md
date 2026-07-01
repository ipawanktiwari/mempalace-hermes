# Contributing to mempalace-hermes

Thanks for contributing! This document outlines how to set up a development
environment, run tests, and submit changes.

## Development Setup

```bash
# 1. Clone the repo
git clone https://github.com/ipawanktiwari/mempalace-hermes.git
cd mempalace-hermes

# 2. Create a virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dev dependencies
pip install pytest pytest-mock

# 4. Install mempalace (required for integration tests)
pip install mempalace
```

## Running Tests

```bash
pytest                          # all tests
pytest tests/ -v                # with verbose output
pytest tests/test_parser.py     # single test file
```

Tests live in `tests/` and use pytest. Unit tests don't require a running
Hermes instance or mempalace binary — they test the internal parsing and
logic functions directly.

## Code Style

- Python: PEP 8. Ruff linter runs in CI.
- Lines: max 100 characters.
- Imports: standard lib first, then third-party, then local.
- Type hints: required for all function signatures.

## Branching

```
main     ← tagged releases
  └── develop  ← active feature/fix work
```

1. Create a feature/fix branch from `develop`
2. Make your changes
3. Add/update tests as needed
4. Run `pytest` — all tests must pass
5. Push and open a PR targeting `develop`
6. Ensure CI passes (lint + test)

## Release Process

Maintainers only:

```bash
git checkout main && git merge develop
git tag vX.Y.Z && git push --tags origin main
```

Then create a GitHub Release with changelog notes.
