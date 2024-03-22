import os
from typing import Dict, List

import numpy as np
import yaml
from loguru import logger

from openqdc.datasets.interaction.base import BaseInteractionDataset
from openqdc.datasets.interaction.L7 import get_loader
from openqdc.utils.molecule import atom_table


class X40(BaseInteractionDataset):
    """
    X40 interaction dataset of 40 dimer pairs as
    introduced in the following paper:

    Benchmark Calculations of Noncovalent Interactions of Halogenated Molecules
    Jan Řezáč, Kevin E. Riley, and Pavel Hobza
    Journal of Chemical Theory and Computation 2012 8 (11), 4285-4292
    DOI: 10.1021/ct300647k

    Dataset retrieved and processed from:
    http://cuby4.molecular.cz/dataset_x40.html
    """

    __name__ = "X40"
    __energy_unit__ = "hartree"
    __distance_unit__ = "ang"
    __forces_unit__ = "hartree/ang"
    __energy_methods__ = [
        "CCSD(T)/CBS",
        "MP2/CBS",
        "dCCSD(T)/haDZ",
        "dCCSD(T)/haTZ",
        "MP2.5/CBS(aDZ)",
    ]

    energy_target_names = []

    def read_raw_entries(self) -> List[Dict]:
        yaml_fpath = os.path.join(self.root, "x40.yaml")
        logger.info(f"Reading X40 interaction data from {self.root}")
        yaml_file = open(yaml_fpath, "r")
        data = []
        data_dict = yaml.load(yaml_file, Loader=get_loader())
        charge0 = int(data_dict["description"].global_setup["molecule_a"]["charge"])
        charge1 = int(data_dict["description"].global_setup["molecule_b"]["charge"])

        for idx, item in enumerate(data_dict["items"]):
            energies = []
            name = np.array([item.shortname])
            energies.append(float(item.reference_value))
            xyz_file = open(os.path.join(self.root, f"{item.shortname}.xyz"), "r")
            lines = list(map(lambda x: x.strip().split(), xyz_file.readlines()))
            setup = lines.pop(1)
            n_atoms = np.array([int(lines[0][0])], dtype=np.int32)
            n_atoms_first = setup[0].split("-")[1]
            n_atoms_first = np.array([int(n_atoms_first)], dtype=np.int32)
            subset = np.array([item.group])
            energies += [float(val[idx]) for val in list(data_dict["alternative_reference"].values())]
            energies = np.array([energies], dtype=np.float32)
            pos = np.array(lines[1:])[:, 1:].astype(np.float32)
            elems = np.array(lines[1:])[:, 0]
            atomic_nums = np.expand_dims(np.array([atom_table.GetAtomicNumber(x) for x in elems]), axis=1)
            natoms0 = n_atoms_first[0]
            natoms1 = n_atoms[0] - natoms0
            charges = np.expand_dims(np.array([charge0] * natoms0 + [charge1] * natoms1), axis=1)
            atomic_inputs = np.concatenate((atomic_nums, charges, pos), axis=-1, dtype=np.float32)

            item = dict(
                energies=energies,
                subset=subset,
                n_atoms=n_atoms,
                n_atoms_first=n_atoms_first,
                atomic_inputs=atomic_inputs,
                name=name,
            )
            data.append(item)
        return data