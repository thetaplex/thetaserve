# Contributing

Thanks for helping. By taking part you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Before you start

Read [docs/DESIGN.md](docs/DESIGN.md); it is the source of truth for scope and decisions. For anything larger than a bug fix, open an issue first so we can agree on the approach.

## Development

Prerequisites: [uv](https://docs.astral.sh/uv/getting-started/installation/) and [just](https://just.systems/man/en/packages.html).

```sh
git clone https://github.com/thetaplex/thetaserve.git
cd thetaserve
just setup    # install from the lockfile
just check    # lint, format check, type check, tests (what CI runs)
just fmt      # format and apply safe lint fixes
just run list models
just samples  # refresh the dry-run sample data; needs the Runpod key
```

Add dependencies with `uv add` (or `uv add --dev`) and commit the updated `uv.lock`.

## Rules

- Tests never call real provider APIs; mock HTTP with respx and AWS with moto.
- Never print or log a secret. Persist secrets only to keyring or the mode-600 secrets file.
- No hard-coded POSIX paths; use platformdirs. Code must work on Linux, macOS and Windows.
- Every hand-written number (overhead, family flags) comes from a recorded run. Unmeasured values stay marked `TODO`.

## Pull requests

- Keep each PR to one change, with tests.
- Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.
- Add a line under `Unreleased` in [CHANGELOG.md](CHANGELOG.md) for user-visible changes.
- `just check` must pass.

## Releases

1. In a PR, run `uv version X.Y.Z`. In CHANGELOG.md, move the `Unreleased` entries under
   `## [X.Y.Z] - YYYY-MM-DD` and update the compare links at the end. Merge the PR.
2. On the merge commit, push the tag: `git tag -a vX.Y.Z -m vX.Y.Z && git push origin vX.Y.Z`.

The tag starts `release.yml`. It stops if the tag is not the package version or CHANGELOG.md has
no section for the version. Then it publishes to PyPI and creates the GitHub release, with that
section as the notes. A version on PyPI cannot be uploaded again.
