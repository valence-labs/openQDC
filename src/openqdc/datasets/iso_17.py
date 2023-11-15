from os.path import join as p_join

from numpy import array, float32

from openqdc.datasets.base import BaseDataset, read_qc_archive_h5


class ISO17(BaseDataset):
    """
    ISO17 dataset consists of the largest set of isomers from the QM9 dataset that consists of a fixed
    composition of atoms (C7O2H10) arranged in different chemically valid structures. It consists of consist
    of 129 molecules each containing 5,000 conformational geometries, energies and forces with a resolution
    of 1 femtosecond in the molecular dynamics trajectories. The simulations were carried out using the
    Perdew-Burke-Ernzerhof (PBE) functional and the Tkatchenko-Scheffler (TS) van der Waals correction method.

    Usage:
    ```python
    from openqdc.datasets import ISO17
    dataset = ISO17()
    ```

    References:
    - https://paperswithcode.com/dataset/iso17
    """

    __name__ = "iso_17"

    __energy_methods__ = [
        "pbe/vdw-ts",
    ]

    energy_target_names = [
        "PBE-TS Energy",
    ]

    __force_methods__ = [
        "pbe/vdw-ts",
    ]

    force_target_names = [
        "PBE-TS Gradient",
    ]

    __energy_unit__ = "ev"
    __distance_unit__ = "bohr"  # bohr
    __forces_unit__ = "ev/bohr"
    __average_nb_atoms__ = 19.0

    @property
    def _stats(self):
        return {
            "formation": {
                "energy": {
                    "mean": self.convert_energy(array([-103.58336533])),
                    "std": self.convert_energy(array([0.79709836])),
                },
                "forces": {
                    "mean": self.convert_forces(array([-1.2548699e-11])),
                    "std": self.convert_forces(array([1.1287293])),
                    "components": {
                        "mean": self.convert_forces(
                            array([[-2.7712117e-11], [-1.8989450e-12], [3.9721233e-11]], dtype=float32)
                        ),
                        "std": self.convert_forces(array([[1.1013116], [1.1273879], [1.1195794]], dtype=float32)),
                        "rms": self.convert_forces(array([[1.1013116], [1.1273879], [1.1195794]], dtype=float32)),
                    },
                },
            },
            "total": {
                "energy": {
                    "mean": self.convert_energy(array([-11503.619]), dtype=float32),
                    "std": self.convert_energy(array([0.79709935]), dtype=float32),
                },
                "forces": {
                    "mean": self.convert_forces(array([-1.2548699e-11])),
                    "std": self.convert_forces(array([1.1287293])),
                    "components": {
                        "mean": self.convert_forces(
                            array([[-2.7712117e-11], [-1.8989450e-12], [3.9721233e-11]], dtype=float32)
                        ),
                        "std": self.convert_forces(array([[1.1013116], [1.1273879], [1.1195794]], dtype=float32)),
                        "rms": self.convert_forces(array([[1.1013116], [1.1273879], [1.1195794]], dtype=float32)),
                    },
                },
            },
        }

    def read_raw_entries(self):
        raw_path = p_join(self.root, "iso_17.h5")
        samples = read_qc_archive_h5(raw_path, "iso_17", self.energy_target_names, self.force_target_names)

        return samples
