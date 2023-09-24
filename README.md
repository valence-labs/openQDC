# openQDC

Open Quantum Data Commons

## Setup Datasets

Use the scripts in `setup/` to download the datasets. For more information, see the [README](setup/README.md) in the `setup/` directory.

# Install the library in dev mode
pip install -e .
```

## Development lifecycle

### Tests

You can run tests locally with:

```bash
pytest
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

We provide support for the following publicly available QM Datasets.

| Dataset | # Molecules | # Conformers | Average Conformers per Molecule | Labels | QM Level of Theory |
| --- | --- | --- | --- | --- | --- |
| [GEOM](https://www.nature.com/articles/s41597-022-01288-4) | 450,000 | 37,000,000 | 82 | energy | GFN2-xTB |
| [Molecule3D](https://arxiv.org/abs/2110.01717) | 3,899,647 | 3,899,647 | 1 | energy | B3LYP/6-31G* |
| [NablaDFT](https://pubs.rsc.org/en/content/articlelanding/2022/CP/D2CP03966D) | 1,000,000 | 5,000,000 | 5 | energy | ωB97X-D/def2-SVP |
| [QMugs](https://www.nature.com/articles/s41597-022-01390-7) | 665,000 | 2,000,000 | 3 | energy | GFN2-xTB, ωB97X-D/def2-SVP |
| [Spice](https://arxiv.org/abs/2209.10702) | 19,238 | 1,132,808 | 59 | energy, forces | ωB97M-D3(BJ)/def2-TZVPPD |
| [ANI](https://pubs.rsc.org/en/content/articlelanding/2017/SC/C6SC05720A) | 57,462 | 348 | 20,000,000 | energy | ωB97x:6-31G(d) |
| [tmQM](https://pubs.acs.org/doi/10.1021/acs.jcim.0c01041) | 86,665 | | | energy | TPSSh-D3BJ/def2-SVP |
| [DES370K](https://www.nature.com/articles/s41597-021-00833-x) | 3,700 | 370,000 | 100 | energy | CCSD(T) |
| [DES5M](https://www.nature.com/articles/s41597-021-00833-x) | 3,700 | 5,000,000 | 1351 | energy | SNS-MP2 |
