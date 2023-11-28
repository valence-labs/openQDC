from os.path import join as p_join

from numpy import array, float32

from openqdc.datasets.base import BaseDataset, read_qc_archive_h5


class GDML(BaseDataset):
    """
    Gradient Domain Machine Learning (GDML) is a dataset consisting of samples from ab initio
    molecular dynamics (AIMD) trajectories. The dataset consists of,
    - Benzene: 627000 samples
    - Uracil: 133000 samples
    - Naptalene: 326000 samples
    - Aspirin: 211000 samples
    - Salicylic Acid: 320000 samples
    - Malonaldehyde: 993000 samples
    - Ethanol: 555000 samples
    - Toluene: 100000 samples

    Usage
    ```python
    from openqdc.datasets import GDML
    dataset = GDML()
    ```

    References:
    - https://www.science.org/doi/10.1126/sciadv.1603015
    - http://www.sgdml.org/#datasets
    """

    __name__ = "gdml"

    __energy_methods__ = [
        "ccsd/cc-pvdz",
        "ccsd(t)/cc-pvdz",
        # "pbe/mbd",  # MD22
        # "pbe+mbd/tight", #MD22
        "pbe/vdw-ts",  # MD17
    ]

    energy_target_names = [
        "CCSD Energy",
        "CCSD(T) Energy",
        "PBE-TS Energy",
    ]

    __force_methods__ = [
        "ccsd/cc-pvdz",
        "ccsd(t)/cc-pvdz",
        # "pbe/mbd",  # MD22
        # "pbe+mbd/tight", #MD22
        "pbe/vdw-ts",  # MD17
    ]

    force_target_names = [
        "CCSD Gradient",
        "CCSD(T) Gradient",
        "PBE-TS Gradient",
    ]

    __energy_unit__ = "kcal/mol"
    __distance_unit__ = "bohr"
    __forces_unit__ = "kcal/mol/bohr"
    __average_nb_atoms__ = 13.00299550

    def read_raw_entries(self):
        raw_path = p_join(self.root, "gdml.h5")
        samples = read_qc_archive_h5(raw_path, "gdml", self.energy_target_names, self.force_target_names)

        return samples

    @property
    def _stats(self):
        return {
            "formation": {
                "energy": {
                    "mean": self.convert_energy(array([-2466.00011563, -1213.94691714, -1916.02068252])),
                    "std": self.convert_energy(array([6.65779492, 310.70204248, 729.2143015])),
                },
                "forces": {
                    "mean": self.convert_forces(array(-1.42346325e-05)),
                    "std": self.convert_forces(array(27.009315)),
                    "components": {
                        "mean": self.convert_forces(
                            array(
                                [
                                    [-8.3862792e-09, -1.9758134e-07, -7.7199416e-05],
                                    [-2.7550591e-09, -1.9665436e-08, 5.3315878e-05],
                                    [-7.5688439e-10, 5.6149121e-09, -1.8894127e-05],
                                ],
                                dtype=float32,
                            )
                        ),
                        "std": self.convert_forces(
                            array(
                                [
                                    [31.060509, 29.168474, 27.547812],
                                    [31.365385, 26.67319, 26.068623],
                                    [31.024155, 27.272366, 22.33925],
                                ],
                                dtype=float32,
                            )
                        ),
                        "rms": self.convert_forces(
                            array(
                                [
                                    [31.060509, 29.168474, 27.547802],
                                    [31.365385, 26.67319, 26.068628],
                                    [31.024155, 27.272366, 22.33925],
                                ],
                                dtype=float32,
                            )
                        ),
                    },
                },
            },
            "total": {
                "energy": {
                    "mean": self.convert_energy(array([-405688.28, -141134.3, -194075.56], dtype=float32)),
                    "std": self.convert_energy(array([7.2360396e00, 3.0755928e04, 8.4138445e04], dtype=float32)),
                },
                "forces": {
                    "mean": self.convert_forces(array(-1.42346325e-05)),
                    "std": self.convert_forces(array(27.009315)),
                    "components": {
                        "mean": self.convert_forces(
                            array(
                                [
                                    [-8.3862792e-09, -1.9758134e-07, -7.7199416e-05],
                                    [-2.7550591e-09, -1.9665436e-08, 5.3315878e-05],
                                    [-7.5688439e-10, 5.6149121e-09, -1.8894127e-05],
                                ],
                                dtype=float32,
                            )
                        ),
                        "std": self.convert_forces(
                            array(
                                [
                                    [31.060509, 29.168474, 27.547812],
                                    [31.365385, 26.67319, 26.068623],
                                    [31.024155, 27.272366, 22.33925],
                                ],
                                dtype=float32,
                            )
                        ),
                        "rms": self.convert_forces(
                            array(
                                [
                                    [31.060509, 29.168474, 27.547802],
                                    [31.365385, 26.67319, 26.068628],
                                    [31.024155, 27.272366, 22.33925],
                                ],
                                dtype=float32,
                            )
                        ),
                    },
                },
            },
        }
