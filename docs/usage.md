# Usage

## How to use

OpenQDC has been designed to be used with a single import:

```python
import openqdc as qdc
dataset = qdc.QM9()
```

All `openQDC` functions are available under `qdc`.
Or if you want to directly import a specific dataset:

```python
from openqdc as Spice
# Spice dataset with distance unit in angstrom instead of bohr
dataset = Spice(distance_unit="ang")
```

## Lazy loading

OpenQDC uses lazy loading to dynamically expose all its API without imposing a long import time during `import openqdc as qdc`. In case of trouble you can always disable lazy loading by setting the environment variable `OPENQDC_DISABLE_LAZY_LOADING` to `1`.
