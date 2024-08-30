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
dataset = Spice(distance_unit="ang",
                array_format = "jax"
)
dataset[0] # dict of jax array
```

Or if you prefer handling `ase.Atoms` objects:

```python
dataset.get_ase_atoms(0)
```

## Iterators

OpenQDC provides a simple way to get the data as iterators:

```python
for data in dataset.as_iter(atoms=True):
    print(data) # Atoms object
    break
```

or if you want to just iterate over the data:

```python
for data in dataset:
    print(data) # dict of arrays
    break
```

## Lazy loading

OpenQDC uses lazy loading to dynamically expose all its API without imposing a long import time during `import openqdc as qdc`. In case of trouble you can always disable lazy loading by setting the environment variable `OPENQDC_DISABLE_LAZY_LOADING` to `1`.
