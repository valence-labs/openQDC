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

| Dataset | # Molecules | # Conformers | Average Conformers per Molecule | Force Labels | Atom Types | QM Level of Theory | Off-Equilibrium Conformations|
| --- | --- | --- | --- | --- | --- | --- | --- |
| [GEOM](https://www.nature.com/articles/s41597-022-01288-4) |  450,000 | 37,000,000 | 82 | No | 18 | GFN2-xTB | No |
| [Molecule3D](https://arxiv.org/abs/2110.01717) |  3,899,647 | 3,899,647 | 1 | No | 5 | B3LYP/6-31G* | No |
| [NablaDFT](https://pubs.rsc.org/en/content/articlelanding/2022/CP/D2CP03966D) |  1,000,000 | 5,000,000 | 5 | No | 6 | ωB97X-D/def2-SVP | |
| [QMugs](https://www.nature.com/articles/s41597-022-01390-7) |  665,000 | 2,000,000 | 3 | No | 10 | GFN2-xTB, ωB97X-D/def2-SVP | No |
| [Spice](https://arxiv.org/abs/2209.10702) |  19,238 | 1,132,808 | 59 | Yes | 15 | ωB97M-D3(BJ)/def2-TZVPPD | Yes |
| [ANI](https://pubs.rsc.org/en/content/articlelanding/2017/SC/C6SC05720A) |  57,462 | 20,000,000 | 348 | No | 4 | ωB97x:6-31G(d) | Yes |
| [tmQM](https://pubs.acs.org/doi/10.1021/acs.jcim.0c01041) |  86,665 | |  | No | | TPSSh-D3BJ/def2-SVP | |
| [DES370K](https://www.nature.com/articles/s41597-021-00833-x) | 3,700 | 370,000 | 100 | No | 20 | CCSD(T) | Yes |
| [DES5M](https://www.nature.com/articles/s41597-021-00833-x) | 3,700 | 5,000,000 | 1351 | No | 20 | SNS-MP2 | Yes |
| [OrbNet Denali](https://arxiv.org/abs/2107.00299) | 212,905 | 2,300,000 | 11 | No | 16 | GFN1-xTB | Yes |
| [SN2RXN](https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181) | 39 | 452709 | 11,600 | Yes | 6 | DSD-BLYP-D3(BJ)/def2-TZVP | |
| [QM7X](https://www.nature.com/articles/s41597-021-00812-2) |  6,950 | 4,195,237 | 603 | Yes | 7 | PBE0+MBD | Yes |
