import os
from os.path import join as p_join

from numpy import array, float32

from openqdc.datasets.base import BaseDataset, read_qc_archive_h5
from openqdc.utils.constants import NOT_DEFINED
from openqdc.utils.io import get_local_cache


class ANI1(BaseDataset):
    """
    The ANI-1 dataset is a collection of 22 x 10^6 structural conformations from 57,000 distinct small
    organic molecules with energy labels calculated using DFT. The molecules
    contain 4 distinct atoms, C, N, O and H.

    Usage
    ```python
    from openqdc.datasets import ANI1
    dataset = ANI1()
    ```

    References:
    - ANI-1: https://www.nature.com/articles/sdata2017193
    - Github: https://github.com/aiqm/ANI1x_datasets
    """

    __name__ = "ani1"

    __energy_methods__ = [
        "wb97x/6-31g(d)",
    ]

    energy_target_names = [
        "ωB97x:6-31G(d) Energy",
    ]
    __energy_unit__ = "hartree"
    __distance_unit__ = "bohr"
    __forces_unit__ = "hartree/bohr"
    __average_nb_atoms__ = 15.91676229984414

    @property
    def root(self):
        return p_join(get_local_cache(), "ani")

    @property
    def preprocess_path(self):
        path = p_join(self.root, "preprocessed", self.__name__)
        os.makedirs(path, exist_ok=True)
        return path

    def read_raw_entries(self):
        raw_path = p_join(self.root, f"{self.__name__}.h5")
        samples = read_qc_archive_h5(raw_path, self.__name__, self.energy_target_names, self.force_target_names)
        return samples

    @property
    def _stats(self):
        return {
            "formation": {
                "energy": {
                    "mean": self.convert_energy(array([-2.37376472])),
                    "std": self.convert_energy(array([0.50266975])),
                },
                "forces": NOT_DEFINED,
            },
            "total": {
                "energy": {
                    "mean": self.convert_energy(array([-333.67322], dtype=float32)),
                    "std": self.convert_energy(array([61.21667], dtype=float32)),
                },
                "forces": NOT_DEFINED,
            },
        }


class ANI1CCX(ANI1):
    """
    ANI1-CCX is a dataset of 500k conformers subsampled from the 5.5M conformers of ANI-1X dataset. The selected
    conformations are then labelled using a high accuracy CCSD(T)*/CBS method.

    Usage
    ```python
    from openqdc.datasets import ANI1CCX
    dataset = ANI1CCX()
    ```

    References:
    - ANI-1ccx: https://doi.org/10.1038/s41467-019-10827-4
    - Github: https://github.com/aiqm/ANI1x_datasets
    """

    __name__ = "ani1ccx"
    __energy_unit__ = "hartree"
    __distance_unit__ = "ang"
    __forces_unit__ = "hartree/ang"
    __average_nb_atoms__ = 15.274685315870588

    __energy_methods__ = [
        "ccsd(t)/cbs",
        "ccsd(t)/cc-pvdz",
        "ccsd(t)/cc-pvtz",
        "tccsd(t)/cc-pvdz",
    ]

    energy_target_names = [
        "CCSD(T)*:CBS Total Energy",
        "NPNO-CCSD(T):cc-pVDZ Correlation Energy",
        "NPNO-CCSD(T):cc-pVTZ Correlation Energy",
        "TPNO-CCSD(T):cc-pVDZ Correlation Energy",
    ]

    __force_methods__ = []
    force_target_names = []

    @property
    def _stats(self):
        return {
            "formation": {
                "energy": {
                    "mean": self.convert_energy(array([-3.23959548, 500.30384627, 500.40706776, 500.76740432])),
                    "std": self.convert_energy(array([1.03021261, 132.52707152, 132.56092469, 132.65261362])),
                },
                "forces": NOT_DEFINED,
            },
            "total": {
                "energy": {
                    "mean": self.convert_energy(array([-374.40665, -1.2378153, -1.505962, -1.2396905], dtype=float32)),
                    "std": self.convert_energy(array([101.63995, 0.32444745, 0.39500558, 0.3250212], dtype=float32)),
                },
                "forces": NOT_DEFINED,
            },
        }


class ANI1X(ANI1):
    """
    The ANI-1X dataset consists of ANI-1 molecules + some molecules added using active learning which leads to
    a total of 5,496,771 conformers with 63,865 unique molecules.

    Usage
    ```python
    from openqdc.datasets import ANI1X
    dataset = ANI1X()
    ```

    References:
    - ANI-1x: https://doi.org/10.1063/1.5023802
    - Github: https://github.com/aiqm/ANI1x_datasets
    """

    __name__ = "ani1x"
    __energy_unit__ = "hartree"
    __distance_unit__ = "ang"
    __forces_unit__ = "hartree/ang"

    __energy_methods__ = [
        "hf/cc-pvdz",
        "hf/cc-pvqz",
        "hf/cc-pvtz",
        "mp2/cc-pvdz",
        "mp2/cc-pvqz",
        "mp2/cc-pvtz",
        "wb97x/6-31g(d)",
        "wb97x/cc-pvtz",
    ]

    energy_target_names = [
        "HF:cc-pVDZ Total Energy",
        "HF:cc-pVQZ Total Energy",
        "HF:cc-pVTZ Total Energy",
        "MP2:cc-pVDZ Correlation Energy",
        "MP2:cc-pVQZ Correlation Energy",
        "MP2:cc-pVTZ Correlation Energy",
        "wB97x:6-31G(d) Total Energy",
        "wB97x:def2-TZVPP Total Energy",
    ]

    force_target_names = [
        "wB97x:6-31G(d) Atomic Forces",
        "wB97x:def2-TZVPP Atomic Forces",
    ]

    __force_methods__ = [
        "wb97x/6-31g(d)",
        "wb97x/cc-pvtz",
    ]

    __average_nb_atoms__ = 15.274685315870588

    @property
    def _stats(self):
        return {
            "formation": {
                "energy": {
                    "mean": self.convert_energy(
                        array(
                            [
                                -2.87910686,
                                -2.91460298,
                                -2.91182519,
                                500.00748497,
                                500.27885605,
                                500.11130961,
                                -3.66090173,
                                -4.40643278,
                            ]
                        )
                    ),
                    "std": self.convert_energy(
                        array(
                            [
                                0.92849657,
                                0.93421854,
                                0.93411345,
                                132.44580372,
                                132.52326771,
                                132.47987395,
                                1.60180792,
                                1.75414812,
                            ]
                        )
                    ),
                },
                "forces": {
                    "mean": self.convert_forces(array([-6.139757e-06])),
                    "std": self.convert_forces(array([0.07401004])),
                    "components": {
                        "mean": self.convert_forces(
                            array(
                                [
                                    [6.6829815e-13, 3.5682501e-07],
                                    [-5.1223647e-13, -1.8487021e-06],
                                    [8.1159564e-13, -3.6849189e-05],
                                ],
                                dtype=float32,
                            )
                        ),
                        "std": self.convert_forces(
                            array(
                                [[0.0759203, 0.06799112], [0.07694941, 0.06652647], [0.06229663, 0.05442103]],
                                dtype=float32,
                            )
                        ),
                        "rms": self.convert_forces(
                            array(
                                [[0.0759203, 0.06799113], [0.07694941, 0.06652647], [0.06229663, 0.05442095]],
                                dtype=float32,
                            )
                        ),
                    },
                },
            },
            "total": {
                "energy": {
                    "mean": self.convert_energy(
                        array(
                            [
                                -372.68945,
                                -372.74274,
                                -372.7326,
                                -1.1540408,
                                -1.5152899,
                                -1.4195863,
                                -392.72458,
                                -391.208,
                            ],
                            dtype=float32,
                        )
                    ),
                    "std": self.convert_energy(
                        array(
                            [
                                101.166664,
                                101.19915,
                                101.191895,
                                0.30445468,
                                0.39988872,
                                0.37456134,
                                136.79112,
                                137.48692,
                            ],
                            dtype=float32,
                        )
                    ),
                },
                "forces": {
                    "mean": self.convert_forces(array([-6.139757e-06])),
                    "std": self.convert_forces(array([0.07401004])),
                    "components": {
                        "mean": self.convert_forces(
                            array(
                                [
                                    [6.6829815e-13, 3.5682501e-07],
                                    [-5.1223647e-13, -1.8487021e-06],
                                    [8.1159564e-13, -3.6849189e-05],
                                ],
                                dtype=float32,
                            )
                        ),
                        "std": self.convert_forces(
                            array(
                                [[0.0759203, 0.06799112], [0.07694941, 0.06652647], [0.06229663, 0.05442103]],
                                dtype=float32,
                            )
                        ),
                        "rms": self.convert_forces(
                            array(
                                [[0.0759203, 0.06799113], [0.07694941, 0.06652647], [0.06229663, 0.05442095]],
                                dtype=float32,
                            )
                        ),
                    },
                },
            },
        }

    def convert_forces(self, x):
        return super().convert_forces(x) * 0.529177249  # correct the Dataset error
