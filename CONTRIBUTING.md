# Contributing to OAAX DEEPX Acceleration

Thank you for your interest in contributing! Before you start, please read the
[OAAX general contributing guide](https://github.com/OAAX-standard/OAAX/blob/main/CONTRIBUTING.md)
for project-wide conventions (bug reports, commit messages and DCO sign-off).

## How to contribute

1. **Open an issue** for a bug or feature, before writing code for anything beyond a small fix.
2. **Fork the repository** and create a branch from `main`.
3. **Make your changes**, keeping each PR focused on one thing.
4. **Run the tests** locally (see below) to confirm nothing is broken.
5. **Sign off every commit** (`git commit -s`) and open a pull request.

## Development environment

The conversion toolchain (Docker) and the runtime library (C++) build independently with the scripts in their
folders; see the [toolchain guide](conversion-toolchain/README.md) and the [runtime guide](runtime-library/README.md).
DX-RT is proprietary DEEPX software and must be installed separately where the runtime guide asks for it; the toolchain build downloads DX-COM from `sdk.deepx.ai`, so it needs internet access.

## Running the tests

- No hardware needed (these run in CI; `pip install pytest` first): `pytest tests/test_models.py` and the mock DX-RT runtime tests described in the
  [runtime guide](runtime-library/README.md#unit-tests).
- Conversion and inference tests need Docker and, for inference, DEEPX hardware: see the [test suite guide](tests/README.md).

## Linting

```bash
pip install pre-commit
pre-commit run --all-files   # ruff, clang-format, shellcheck, hadolint: the same checks CI runs
```

The self-hosted build jobs are skipped on a PR from a fork, so its CI does not build or upload anything; a maintainer's run covers it. Releases are uploaded to the bucket only when a maintainer pushes a `vX.Y.Z` tag, never on a merge to `main`.
