from os.path import join as p_join

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

    def __smiles_converter__(self, x):
        """util function to convert string to smiles: useful if the smiles is
        encoded in a different format than its display format
        """
        return "_".join(x.decode("ascii").split("_")[:-1])

    def read_raw_entries(self):
        raw_path = p_join(self.root, "solvated_peptides.h5")
        samples = read_qc_archive_h5(raw_path, "solvated_peptides", self.energy_target_names, self.force_target_names)

        return samples
