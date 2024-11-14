<div align="center">
    <img src="docs/assets/logo-title.png" width="100%">
</div>

<p align="center">
    <b>OpenQDC - Open Quantum Data Commons </b> <br />
</p>
<p align="center">
  <a href="https://docs.openqdc.io/" target="_blank">
      Docs
  </a> |
  <a href="https://openqdc.io/" target="_blank">
      Homepage
  </a>
</p>

---

[![license](https://licensebuttons.net/l/by-nc/4.0/80x15.png)](https://github.com/valence-labs/openQDC/blob/main/LICENSE)

The OpenQDC Python library makes it easy to work with all of the quantum datasets in the hub.
It’s a package that aims to provide a simple and efficient way to download, load, and utilize various datasets. You can download datasets with just one line of code. 

- 🐍 A simple pythonic API: The simplicity of the Python interface ensures ease of use, making it perfect for quick prototyping.
- 🕹️ ML-Ready: all you manipulate are torch.Tensor, jax.Array or numpy.Array objects.
- ⚛️ Quantum ready: The quantum methods used by the datasets are checked and standardized to provide additional values, useful normalization, and different statistics.
- ✅ Standardized: The datasets are written in standard and performant formats with annotated metadata like units and labels.
- 🧠 Performance matters: read and write multiple formats (memmap, zarr, xyz, etc).
- 📈 Data: have access to 1.5+ billion data points
- 👩‍🔧 Open source & extensible: OpenQDC and all its files and datasets are open source, and you can add your own dataset and share with the community in just a few minutes


### Installing OpenQDC 

Use mamba:

```bash
mamba install -c conda-forge openqdc
```
Tips: You can replace mamba by conda.

Note: We highly recommend using a Conda Python distribution to install OpenQDC. The package is also pip installable: 
```bash
pip install openqdc
```

### Installing OpenQDC as development version

```bash
git clone https://github.com/valence-labs/openQDC.git
cd openQDC
# use mamba/conda
mamba env create -n openqdc -f env.yml
pip install -e .
```

### Tests

You can run tests locally with:

```bash
pytest
```

### Documentation

You can build the documentation locally with:

```bash
mkdocs serve
```

# Downloading Datasets

A command line interface is available to download datasets or see which dataset is available, for more information please run openqdc --help.

```bash
# Display the available datasets
openqdc datasets

# Display the help message for the download command
openqdc download --help

# Download the Spice and QMugs dataset
openqdc download Spice QMugs
```

# CI Status

The CI runs tests and performs code quality checks for the following combinations:

- The three major platforms: Windows, OSX and Linux.
- The four latest Python versions.

|                                         | `main`                                                                                                                                                                              |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Lib build & Testing                     | [![test](https://github.com/valence-labs/openQDC/actions/workflows/test.yml/badge.svg)](https://github.com/valence-labs/openQDC/actions/workflows/test.yml)                         |
| Code Sanity (linting and type analysis) | [![code-check](https://github.com/valence-labs/openQDC/actions/workflows/code-check.yml/badge.svg)](https://github.com/valence-labs/openQDC//actions/workflows/code-check.yml)      |
| Documentation Build                     | [![doc](https://github.com/valence-labs/openQDC/actions/workflows/doc.yml/badge.svg)](https://github.com/valence-labs/openQDC/actions/workflows/doc.yml)                            |
| Pre-Commit                              | [![pre-commit](https://github.com/valence-labs/openQDC/actions/workflows/pre-commit-ci.yml/badge.svg)](https://github.com/valence-labs/openQDC/actions/workflows/pre-commit-ci.yml) |

# How to cite

All data presented in the OpenQDC are already published in scientific journals, full reference to the respective paper is attached to each dataset class. When citing data obtained from OpenQDC, you should cite both the original paper(s) the data come from and our paper on OpenQDC itself. The reference is:

ADD REF HERE LATER
