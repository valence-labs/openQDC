from os.path import join as p_join

from numpy import array, float32, nan

from openqdc.datasets.base import BaseDataset, read_qc_archive_h5


class COMP6(BaseDataset):
    """
    COMP6 is a benchmark suite consisting of broad regions of bio-chemical and organic space
    developed for testing the ANI-1x potential. It is curated from 6 benchmark sets:
    S66x8, ANI Molecular Dynamics, GDB7to9, GDB10to13, DrugBank, and Tripeptides.

    Usage
    ```python
    from openqdc.datasets import COMP6
    dataset = COMP6()
    ```

    References:
    - https://aip.scitation.org/doi/abs/10.1063/1.5023802
    - Github: https://github.com/isayev/COMP6
    """

    __name__ = "comp6"

    # watchout that forces are stored as -grad(E)
    __energy_unit__ = "kcal/mol"
    __distance_unit__ = "bohr"  # bohr
    __forces_unit__ = "kcal/mol/bohr"

    __energy_methods__ = [
        "wb97x/6-31g*",
        "b3lyp-d3mbj/def2-tzvp",
        "b3lyp/def2-tzvp",
        "hf/def2-tzvp",
        "pbe-d3bj/def2-tzvp",
        "pbe/def2-tzvp",
        "svwn/def2-tzvp",
        "wb97m-d3bj/def2-tzvp",
        "wb97m/def2-tzvp",
    ]

    energy_target_names = [
        "Energy",
        "B3LYP-D3M(BJ):def2-tzvp",
        "B3LYP:def2-tzvp",
        "HF:def2-tzvp",
        "PBE-D3M(BJ):def2-tzvp",
        "PBE:def2-tzvp",
        "SVWN:def2-tzvp",
        "WB97M-D3(BJ):def2-tzvp",
        "WB97M:def2-tzvp",
    ]

    __force_methods__ = [
        "wb97x/6-31g*",
    ]

    force_target_names = [
        "Gradient",
    ]

    __average_nb_atoms__ = 25.74051563378753

    @property
    def _stats(self):
        return {
            "formation": {
                "energy": {
                    "mean": self.convert_energy(
                        array(
                            [
                                -2579.52016333,
                                -2543.74519203,
                                -354694.46157991,
                                -2506.4300631,
                                -2616.89224817,
                                -3157.54118509,
                                -354031.62984212,
                                nan,
                                nan,
                            ]
                        )
                    ),
                    "std": self.convert_energy(
                        array(
                            [
                                1811.03171965,
                                1471.95818836,
                                201545.89189168,
                                1385.50993753,
                                1456.09915473,
                                1728.51133182,
                                208097.95666257,
                                nan,
                                nan,
                            ]
                        )
                    ),
                },
                "forces": {
                    "mean": self.convert_forces(6.6065984e-13),
                    "std": self.convert_forces(0.056459695),
                    "components": {
                        "mean": self.convert_forces(
                            array([[-4.1767219e-13], [1.0024132e-12], [-9.4386771e-13]], dtype=float32)
                        ),
                        "std": self.convert_forces(array([[0.05781676], [0.05793402], [0.05330585]], dtype=float32)),
                        "rms": self.convert_forces(array([[0.05781676], [0.05793402], [0.05330585]], dtype=float32)),
                    },
                },
            },
            "total": {
                "energy": {
                    "mean": self.convert_energy(
                        array(
                            [
                                -360972.16,
                                -354729.66,
                                -354699.38,
                                -349555.7,
                                -351555.97,
                                -351530.44,
                                -354027.8,
                                nan,
                                nan,
                            ],
                            dtype=float32,
                        ),
                    ),
                    "std": self.convert_energy(
                        array(
                            [254766.0, 201559.77, 201537.8, 188725.47, 191028.78, 191016.1, 208089.4, nan, nan],
                            dtype=float32,
                        ),
                    ),
                },
                "forces": {
                    "mean": self.convert_forces(array([6.6065984e-13])),
                    "std": self.convert_forces(array([0.056459695])),
                    "components": {
                        "mean": self.convert_forces(
                            array([[-4.1767219e-13], [1.0024132e-12], [-9.4386771e-13]], dtype=float32)
                        ),
                        "std": self.convert_forces(array([[0.05781676], [0.05793402], [0.05330585]], dtype=float32)),
                        "rms": self.convert_forces(array([[0.05781676], [0.05793402], [0.05330585]], dtype=float32)),
                    },
                },
            },
        }

    def read_raw_entries(self):
        samples = []
        for subset in ["ani_md", "drugbank", "gdb7_9", "gdb10_13", "s66x8", "tripeptides"]:
            raw_path = p_join(self.root, f"{subset}.h5")
            samples += read_qc_archive_h5(raw_path, subset, self.energy_target_names, self.force_target_names)

        return samples
