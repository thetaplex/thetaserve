set windows-shell := ["pwsh", "-NoLogo", "-NoProfile", "-Command"]

default:
    @just --list

# Install the project and dev tools from the lockfile
setup:
    uv sync --locked

# Run the test suite
test *args:
    uv run pytest {{args}}

# Lint, check formatting and type-check
lint:
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy src tests

# Format and apply safe lint fixes
fmt:
    uv run ruff format .
    uv run ruff check --fix .

# Everything CI runs
check: lint test

# Fetch the dry-run sample data again (needs the Runpod key and network access)
samples:
    uv run python -m thetaserve.samples

# Run the CLI, e.g. `just run list models`
run *args:
    uv run thetaserve {{args}}
