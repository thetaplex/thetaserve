# Contributing

Thanks for helping. By taking part you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Before you start

Read [docs/DESIGN.md](docs/DESIGN.md); it is the source of truth for scope and decisions. For anything larger than a bug fix, open an issue first so we can agree on the approach.

## Development

Prerequisites: [uv](https://docs.astral.sh/uv/getting-started/installation/) and [just](https://just.systems/man/en/packages.html).

```sh
just setup    # install from the lockfile
just check    # lint, format check, type check, tests (what CI runs)
just fmt      # format and apply safe lint fixes
just run list models
```

Add dependencies with `uv add` (or `uv add --dev`) and commit the updated `uv.lock`.

## Rules

- Tests never call real provider APIs; mock HTTP with respx and AWS with moto.
- Never print or log a secret. Persist secrets only to keyring or the mode-600 secrets file.
- No hard-coded POSIX paths; use platformdirs. Code must work on Linux, macOS and Windows.
- Every number in a profile comes from a recorded run. Unmeasured values stay marked `TODO`.

## Pull requests

- Keep each PR to one change, with tests.
- Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.
- Add a line under `Unreleased` in [CHANGELOG.md](CHANGELOG.md) for user-visible changes.
- `just check` must pass.
