import os
import numpy as np
import pandas as pd

from typing import Dict, List

from tqdm import tqdm
from rdkit import Chem
from ruamel.yaml import YAML
from loguru import logger
from openqdc.datasets.interaction import BaseInteractionDataset
from openqdc.utils.molecule import atom_table, molecule_groups


class L7(BaseInteractionDataset):
    __name__ = "L7"
    __energy_unit__ = "hartree"
    __distance_unit__ = "ang"
    __forces_unit__ = "hartree/ang"
    __energy_methods__ = [
        "CSD(T) | QCISD(T)",
        "DLPNO-CCSD(T)",
        "MP2/CBS",
        "MP2C/CBS",
        "fixed",
        "DLPNO-CCSD(T0)",
        "LNO-CCSD(T)",
        "FN-DMC",
    ]

    energy_target_names = []

    def read_raw_entries(self) -> List[Dict]:
        yaml_fpath = os.path.join(self.root, "l7.yaml")
        logger.info(f"Reading L7 interaction data from {self.root}")
        yaml_file = open(yaml_fpath, "r")
        yaml = YAML()
        data = []
        data_dict = yaml.load(yaml_file)
        charge0 = int(data_dict["description"]["global_setup"]["molecule_a"]["charge"])
        charge1 = int(data_dict["description"]["global_setup"]["molecule_b"]["charge"])

        for idx, item in enumerate(data_dict["items"]):
            energies = []
            name = np.array([item["shortname"]])
            fname = item["geometry"].split(":")[1]
            energies.append(item["reference_value"])
            xyz_file = open(os.path.join(self.root, f"{fname}.xyz"), "r")
            lines = list(map(lambda x: x.strip().split(), xyz_file.readlines()))
            lines.pop(1) 
            n_atoms = np.array([int(lines[0][0])], dtype=np.int32)
            n_atoms_first = np.array([int(item["setup"]["molecule_a"]["selection"].split("-")[1])], dtype=np.int32)
            subset = np.array([item["group"]])
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
