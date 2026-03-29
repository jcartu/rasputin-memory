# Contributing to RASPUTIN Memory

## Dev Environment Setup

```bash
git clone https://github.com/jcartu/rasputin-memory.git
cd rasputin-memory
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Code Style

- **Linter:** [Ruff](https://docs.astral.sh/ruff/) — run `ruff check .` before committing
- **Naming:** `snake_case` for Python files and functions
- **Line length:** 120 characters max

## Running Tests

```bash
pytest                  # smoke tests (no Docker needed)
docker-compose up -d    # for integration tests
pytest tests/           # full suite
```

## Pull Request Process

1. Fork the repo and create a feature branch (`git checkout -b feat/my-feature`)
2. Make your changes — keep commits atomic and well-described
3. Ensure `ruff check .` and `pytest` pass
4. Open a PR against `main` with a clear description of what and why
5. CI must pass before merge

## Reporting Issues

- Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) for bugs
- Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md) for ideas
- Include logs, Python version, and OS when reporting bugs

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how the 4-stage pipeline works internally. Start there before modifying core retrieval logic.
