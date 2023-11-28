from os.path import join as p_join

from numpy import array, float32

from openqdc.datasets.base import BaseDataset, read_qc_archive_h5


class SolvatedPeptides(BaseDataset):
    __name__ = "solvated_peptides"

    __energy_methods__ = [
        "revpbe-d3(bj)/def2-tzvp",
    ]

    energy_target_names = [
        "revPBE-D3(BJ):def2-TZVP Atomization Energy",
    ]

    __force_methods__ = [
        "revpbe-d3(bj)/def2-tzvp",
    ]

    force_target_names = [
        "revPBE-D3(BJ):def2-TZVP Gradient",
    ]

    # TO CHECK
    __energy_unit__ = "hartree"
    __distance_unit__ = "bohr"
    __forces_unit__ = "hartree/bohr"
    __average_nb_atoms__ = 21.380975029465297

    def __smiles_converter__(self, x):
        """util function to convert string to smiles: useful if the smiles is
        encoded in a different format than its display format
        """
        return "_".join(x.decode("ascii").split("_")[:-1])

    def read_raw_entries(self):
        raw_path = p_join(self.root, "solvated_peptides.h5")
        samples = read_qc_archive_h5(raw_path, "solvated_peptides", self.energy_target_names, self.force_target_names)

        return samples

    # TODO : Check the values in this

    @property
    def _stats(self):
        return {
            "formation": {
                "energy": {
                    "mean": self.convert_energy(array([423.49523618])),
                    "std": self.convert_energy(array([309.76172829])),
                },
                "forces": {
                    "mean": self.convert_forces(array([-3.792959e-12])),
                    "std": self.convert_forces(array([1.4568169])),
                    "components": {
                        "mean": self.convert_forces(
                            array([[-4.1655182e-12], [-6.9530774e-12], [2.5650127e-12]], dtype=float32)
                        ),
                        "std": self.convert_forces(array([[1.3502095], [1.3478843], [1.3509929]], dtype=float32)),
                        "rms": self.convert_forces(array([[1.3502095], [1.3478843], [1.3509929]], dtype=float32)),
                    },
                },
            },
            "total": {
                "energy": {
                    "mean": self.convert_energy(array([-79.619286], dtype=float32)),
                    "std": self.convert_energy(array([40.01196], dtype=float32)),
                },
                "forces": {
                    "mean": self.convert_forces(array([-3.792959e-12])),
                    "std": self.convert_forces(array([1.4568169])),
                    "components": {
                        "mean": self.convert_forces(
                            array([[-4.1655182e-12], [-6.9530774e-12], [2.5650127e-12]], dtype=float32)
                        ),
                        "std": self.convert_forces(array([[1.3502095], [1.3478843], [1.3509929]], dtype=float32)),
                        "rms": self.convert_forces(array([[1.3502095], [1.3478843], [1.3509929]], dtype=float32)),
                    },
                },
            },
        }
