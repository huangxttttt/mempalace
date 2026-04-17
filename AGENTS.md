# Repository Guidelines

## Project Structure & Module Organization

`mempalace/` contains the core Python package and CLI entry points such as `cli.py`, `miner.py`, `searcher.py`, and `mcp_server.py`. Put new runtime code here and keep module names focused and in `snake_case`. `tests/` holds the main pytest suite, while `tests/benchmarks/` contains benchmark-oriented tests and helpers. Use `benchmarks/` for reproducible benchmark runners and docs, `examples/` for integration samples, `hooks/` for shell hooks, and `assets/` for static media.

## Build, Test, and Development Commands

Use `uv` for local development.

- `uv sync --dev`: install runtime and contributor dependencies.
- `uv run pytest`: run the default test suite; benchmark, slow, and stress tests are excluded by default.
- `uv run pytest -m benchmark`: run benchmark-marked tests when needed.
- `uv run ruff check .`: lint the repository.
- `uv run ruff format .`: apply formatting.
- `uv run mempalace --help`: verify the CLI entry point locally.

If you prefer pip, `pip install -e ".[dev]"` also works, but `uv` is the documented path in this repo.

## Coding Style & Naming Conventions

Target Python 3.9+ and keep lines within the configured 100-character limit. Ruff is the formatter and linter; pre-commit runs `ruff --fix` and `ruff-format`. Use `snake_case` for functions, variables, modules, and tests; use `PascalCase` for classes. Add docstrings to modules and public functions, and prefer type hints where they clarify interfaces. Keep dependencies minimal and avoid adding new ones without discussion.

## Testing Guidelines

Write tests with `pytest` under `tests/` using `test_*.py` filenames and descriptive `test_*` function names. Mirror the target module when practical, for example `mempalace/searcher.py` with `tests/test_searcher.py`. Tests should run offline and without API keys. Coverage is tracked for `mempalace/`, with the current floor set to 30%.

## Commit & Pull Request Guidelines

Recent history follows Conventional Commit prefixes such as `feat:`, `fix:`, `chore:`, and `docs:`. Keep subjects short and imperative, for example `fix: handle empty transcript files`. For pull requests, include a clear summary, note any user-visible behavior changes, link related issues, and mention the commands you ran (`uv run pytest`, `uv run ruff check .`). Add screenshots only when changing docs or web UI behavior.
