# Overview

OpenQDC is a python library to work with quantum datasets. It's a package aimed at providing a simple and efficient way to download, load and utilize various datasets and provide a way to standardize the data for easy use in machine learning models.

- 🐍 Simple pythonic API
- 🕹️  ML-Ready: all you manipulate are `torch.Tensor`,`jax.Array` or `numpy.Array`objects.
- ⚛️ Quantum Ready: The quantum methods are checked and standardized to provide addictional values.
- ✅ Standardized: The datasets are written in standard and performant formats with annotated metadata like units and labels.
- 🧠 Performance matters: read and write multiple formats (memmap, zarr, xyz, etc).
- 📈 Data: have access to 1.5+ billion datapoints

Visit our website at https://openqdc.io .

## Installation

Use mamba:

```bash
conda install -c conda-forge openqdc
```

_**Tips:** You can replace `conda` by `mamba`._

_**Note:** We highly recommend using a [Conda Python distribution](https://github.com/conda-forge/miniforge) to install OpenQDC. The package is also pip installable if you need it: `pip install openqdc`._

## Quick API Tour

```python
from openqdc as Spice

# Load the original dataset
dataset = Spice()

# Load the dataset with a different units
dataset = Spice(
    energy_unit = "kcal/mol",
    distance_unit = "ang",
    energy_type = "formation",
    array_format = "torch"
)

# Access the data
data = dataset[0]

# Get relevant statistics
dataset.get_statistics()

# Get dataset metadata
dataset.average_n_atoms
dataset.chemical_species
dataset.charges

# Compute physical descriptors
dataset.calculate_descriptors(
    descriptor_name = "soap"
)
```

## How to cite

Please cite OpenQDC if you use it in your research: [![Pending Publication](Pending Publication)](Pending Publication).

## Compatibilities

OpenQDC is compatible with Python >= 3.8 and is tested on Linux, MacOS and Windows.
