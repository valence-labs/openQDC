# Contribute

The below documents the development lifecycle of OpenQDC.

## Setup a dev environment

```bash
mamba env create -n openqdc -f env.yml
mamba activate datamol
pip install -e .
```

## Pre commit installation

```bash
pre-commit install
pre-commit run --all-files
```

## Continuous Integration

OpenQDC uses Github Actions to:

- **Build and test** `openQDC`.
  - Multiple combinations of OS and Python versions are tested.
- **Check** the code:
  - Formatting with `black`.
  - Static type check with `mypy`.
  - Modules import formatting with `isort`.
  - Pre-commit hooks.
- **Documentation**:
  - Google docstring format.
  - build and deploy the documentation on `main` and for every new git tag.


## Run tests

```bash
pytest
```

## Build the documentation

You can build and serve the documentation locally with:

```bash
# Build and serve the doc
mike serve
```

or with

```bash
mkdocs serve
```

### Multi-versionning

The doc is built for eash push on `main` and every git tags using [mike](https://github.com/jimporter/mike). Everything is automated using Github Actions. Please refer to the official mike's documentation for the details.
