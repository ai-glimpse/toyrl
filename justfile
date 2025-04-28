lint:
    uv run ruff check
    uv run ruff format
    uv run mypy .

test:
    uv run pytest --doctest-modules -v --cov=toyrl toyrl tests
