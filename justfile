lint:
    uv run ruff check
    uv run ruff format
    uv run mypy .

test:
    uv run pytest --doctest-modules -v --cov=toyrl --cov-fail-under 90 --cov-report=term --cov-report=xml --cov-report=html toyrl tests
    open htmlcov/index.html
