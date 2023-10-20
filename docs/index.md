# openQDC

Open Quantum Data Commons

## Setup Datasets

Use the scripts in `setup/` to download the datasets. For more information, see the [README](setup/README.md) in the `setup/` directory.

# Install the library in dev mode
```bash
# Install the deps
mamba env create -n qdc -f env.yml

# Activate the environment
mamba activate  qdc

# Install the qdc library in dev mode
pip install -e .

```

## Development lifecycle

### Tests

You can run tests locally with:

```bash
pytest .
```
