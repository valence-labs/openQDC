# openQDC

Open Quantum Data Commons

[![license](https://licensebuttons.net/l/by-nc/4.0/80x15.png)](https://github.com/valence-labs/openQDC/blob/main/LICENSE)

### Installing openQDC
```bash
git clone git@github.com:OpenDrugDiscovery/openQDC.git
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

# Overview of Datasets

<!-- Create a table with the following columns
1. Name of Dataset (with reference of paper) [Dataset Name](paper link)
2. Number of Molecules
3. Number of Conformers
4. Average Conformer to Molecule Ratio (in 2 lines)
5. Labels
6. QM Level of Theory
 -->

We provide support for the following publicly available QM Potential Energy Datasets.

# Potential Energy

| Dataset | # Molecules | # Conformers | Average Conformers per Molecule | Force Labels | Atom Types | QM Level of Theory | Off-Equilibrium Conformations|
| --- | --- | --- | --- | --- | --- | --- | --- |
| [ANI](https://pubs.rsc.org/en/content/articlelanding/2017/SC/C6SC05720A) |  57,462 | 20,000,000 | 348 | No | 4 | ωB97x:6-31G(d) | Yes |
| [GEOM](https://www.nature.com/articles/s41597-022-01288-4) |  450,000 | 37,000,000 | 82 | No | 18 | GFN2-xTB | No |
| [Molecule3D](https://arxiv.org/abs/2110.01717) |  3,899,647 | 3,899,647 | 1 | No | 5 | B3LYP/6-31G* | No |
| [NablaDFT](https://pubs.rsc.org/en/content/articlelanding/2022/CP/D2CP03966D) |  1,000,000 | 5,000,000 | 5 | No | 6 | ωB97X-D/def2-SVP | |
| [OrbNet Denali](https://arxiv.org/abs/2107.00299) | 212,905 | 2,300,000 | 11 | No | 16 | GFN1-xTB | Yes |
| [PCQM_PM6](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.0c00740) | | | 1| No| | PM6 | No
| [PCQM_B3LYP](https://arxiv.org/abs/2305.18454) | 85,938,443|85,938,443 | 1| No| | B3LYP/6-31G* | No
| [QMugs](https://www.nature.com/articles/s41597-022-01390-7) |  665,000 | 2,000,000 | 3 | No | 10 | GFN2-xTB, ωB97X-D/def2-SVP | No |
| [QM7X](https://www.nature.com/articles/s41597-021-00812-2) |  6,950 | 4,195,237 | 603 | Yes | 7 | PBE0+MBD | Yes |
| [SN2RXN](https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181) | 39 | 452709 | 11,600 | Yes | 6 | DSD-BLYP-D3(BJ)/def2-TZVP | |
| [SolvatedPeptides](https://doi.org/10.1021/acs.jctc.9b00181) |   | 2,731,180 |  | Yes |  | revPBE-D3(BJ)/def2-TZVP |  |
| [Spice](https://arxiv.org/abs/2209.10702) |  19,238 | 1,132,808 | 59 | Yes | 15 | ωB97M-D3(BJ)/def2-TZVPPD | Yes |
| [tmQM](https://pubs.acs.org/doi/10.1021/acs.jcim.0c01041) |  86,665 | 86,665| 1| No | | TPSSh-D3BJ/def2-SVP | |
| [Transition1X](https://www.nature.com/articles/s41597-022-01870-w) |   | 9,654,813| | Yes | | ωB97x/6–31 G(d) | Yes |
| [WaterClusters](https://doi.org/10.1063/1.5128378) | 1  | 4,464,740| | No | 2 | TTM2.1-F | Yes|


# Interaction energy

We also provide support for the following publicly available QM Noncovalent Interaction Energy Datasets.

| Dataset |
| --- |
| [DES370K](https://www.nature.com/articles/s41597-021-00833-x) |
| [DES5M](https://www.nature.com/articles/s41597-021-00833-x)   |
| [Metcalf](https://pubs.aip.org/aip/jcp/article/152/7/074103/1059677/Approaches-for-machine-learning-intermolecular) |
| [DESS66](https://www.nature.com/articles/s41597-021-00833-x) |
| [DESS66x8](https://www.nature.com/articles/s41597-021-00833-x) |
| [Splinter](https://www.nature.com/articles/s41597-023-02443-1) |
| [X40](https://pubs.acs.org/doi/10.1021/ct300647k) |
| [L7](https://pubs.acs.org/doi/10.1021/ct400036b)  |

# CI Status

The CI runs tests and performs code quality checks for the following combinations:

- The three major platforms: Windows, OSX and Linux.
- The four latest Python versions.

|                                         | `main`                                                                                                                                                                    |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Lib build & Testing                     | [![test](https://github.com/valence-labs/openQDC/actions/workflows/test.yml/badge.svg)](https://github.com/valence-labs/openQDC/actions/workflows/test.yml)                   |
| Code Sanity (linting and type analysis) | [![code-check](https://github.com/valence-labs/openQDC/actions/workflows/code-check.yml/badge.svg)](https://github.com/valence-labs/openQDC//actions/workflows/code-check.yml) |
| Documentation Build                     | [![doc](https://github.com/valence-labs/openQDC/actions/workflows/doc.yml/badge.svg)](https://github.com/valence-labs/openQDC/actions/workflows/doc.yml)      |
| Pre-Commit                              | [![pre-commit](https://github.com/valence-labs/openQDC/actions/workflows/pre-commit-ci.yml/badge.svg)](https://github.com/valence-labs/openQDC/actions/workflows/pre-commit-ci.yml)    |


# How to cite
All data presented in the OpenQDC are already published in scientific journals, full reference to the respective paper is attached to each dataset class. When citing data obtained from OpenQDC, you should cite both the original paper(s) the data come from and our paper on OpenQDC itself. The reference is:

ADD REF HERE LATER
