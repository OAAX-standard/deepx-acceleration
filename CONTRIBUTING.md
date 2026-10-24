# Contributing to OAAX DEEPX Acceleration

Thank you for your interest in contributing! Before you start, please read the
[OAAX general contributing guide](https://github.com/OAAX-standard/OAAX/blob/main/CONTRIBUTING.md)
for project-wide conventions (bug reports, commit messages and DCO sign-off, code style).

## How to contribute

1. **Open an issue** to discuss the bug or feature before writing code.
2. **Fork the repository** and create a branch from `main`.
3. **Make your changes**, keeping each PR focused on one thing.
4. **Run the tests** locally (see below) to confirm nothing is broken.
5. **Sign off every commit** (`git commit -s`) and open a pull request.

## Development environment

The conversion toolchain (Docker) and the runtime library (C++) build independently with the scripts in their
folders; see the [toolchain guide](conversion-toolchain/README.md) and the [runtime guide](runtime-library/README.md).
DX-RT and DX-COM are proprietary DEEPX software and must be installed separately where those guides ask for them.

## Running the tests

- No hardware needed (these run in CI): `pytest tests/test_models.py` and the mock DX-RT runtime tests described in the
  [runtime guide](runtime-library/README.md#unit-tests).
- Conversion and inference tests need Docker and, for inference, DEEPX hardware: see the [test suite guide](tests/README.md).

## Linting

```bash
pip install pre-commit
pre-commit run --all-files   # ruff, clang-format, shellcheck, hadolint: the same checks CI runs
```
