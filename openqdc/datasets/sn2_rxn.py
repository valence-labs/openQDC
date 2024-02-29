from os.path import join as p_join

from openqdc.datasets.base import BaseDataset, read_qc_archive_h5


class SN2RXN(BaseDataset):
    __name__ = "sn2_rxn"

    __energy_methods__ = [
        "dsd-blyp-d3(bj)/def2-tzvp",
    ]
    __energy_unit__ = "ev"
    __distance_unit__ = "bohr"
    __forces_unit__ = "ev/bohr"

    energy_target_names = [
        "DSD-BLYP-D3(BJ):def2-TZVP Atomization Energy",
    ]

    __force_methods__ = [
        "dsd-blyp-d3(bj)/def2-tzvp",
    ]

    force_target_names = [
        "DSD-BLYP-D3(BJ):def2-TZVP Gradient",
    ]

    def __smiles_converter__(self, x):
        """util function to convert string to smiles: useful if the smiles is
        encoded in a different format than its display format
        """
        return "-".join(x.decode("ascii").split("_")[:-1])

    def read_raw_entries(self):
        raw_path = p_join(self.root, "sn2_rxn.h5")

        # raw_path = p_join(self.root, "sn2_reactions.npz")
        # data = np.load(raw_path)

        # # as example for accessing individual entries, print the data for entry idx=0
        # idx = 0
        # print("Data for entry " + str(idx)+":")
        # print("Number of atoms")
        # print(data["N"][idx])
        # print("Energy [eV]")
        # print(data["E"][idx])
        # print("Total charge")
        # print(data["Q"][idx])
        # print("Dipole moment vector (with respect to [0.0 0.0 0.0]) [eA]")
        # print(data["D"][idx,:])
        # print("Nuclear charges")
        # print(data["Z"][idx,:data["N"][idx]])
        # print("Cartesian coordinates [A]")
        # print(data["R"][idx,:data["N"][idx],:])
        # print("Forces [eV/A]")
        # print(data["F"][idx,:data["N"][idx],:])

        # exit()

        samples = read_qc_archive_h5(raw_path, "sn2_rxn", self.energy_target_names, self.force_target_names)

        return samples
