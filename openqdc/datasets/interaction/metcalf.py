import os
from typing import Dict, List

import numpy as np

from openqdc.datasets.interaction import BaseInteractionDataset
from openqdc.utils.molecule import atom_table


class Metcalf(BaseInteractionDataset):
    __name__ = "metcalf"
    __energy_unit__ = "hartree"
    __distance_unit__ = "ang"
    __forces_unit__ = None
    __energy_methods__ = ["SAPT0/jun-cc-pVDZ"]
    energy_target_names = [
        "total energy",
        "electrostatic energy",
        "exchange energy",
        "induction energy",
        "dispersion energy",
    ]

    def read_raw_entries(self) -> List[Dict]:
        """
        SMILES strings are inferred from the
        .xyz files using the RDKit xyz2mol function.
        More details here:

        https://github.com/jensengroup/xyz2mol

        """
        data = []
        for dirname in os.listdir(self.root):
            xyz_dir = os.path.join(self.root, dirname)
            if not os.path.isdir(xyz_dir):
                continue
            subset = np.array([dirname.split("-")[0].lower()])  # training, validation, or test
            for filename in os.listdir(xyz_dir):
                if not filename.endswith(".xyz"):
                    continue
                lines = list(map(lambda x: x.strip(), open(os.path.join(xyz_dir, filename), "r").readlines()))
                line_two = lines[1].split(",")
                energies = np.array([line_two[1:6]], dtype=np.float32)
                num_atoms = np.array([int(lines[0])])

                elem_xyz = np.array([x.split() for x in lines[2:]])
                elements = elem_xyz[:, 0]
                xyz = elem_xyz[:, 1:].astype(np.float32)
                atomic_nums = np.expand_dims(np.array([atom_table.GetAtomicNumber(x) for x in elements]), axis=1)
                charges = np.expand_dims(np.array([0] * num_atoms[0]), axis=1)

                atomic_inputs = np.concatenate((atomic_nums, charges, xyz), axis=-1, dtype=np.float32)

                item = dict(
                    n_atoms=num_atoms,
                    subset=subset,
                    energies=energies,
                    positions=xyz,
                    atomic_inputs=atomic_inputs,
                    name=np.array([""]),
                )
                data.append(item)
        return data
