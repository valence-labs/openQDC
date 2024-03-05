import os
import numpy as np
import pandas as pd

from typing import Dict, List

from tqdm import tqdm
from loguru import logger
from openqdc.datasets.interaction import BaseInteractionDataset
from openqdc.utils.molecule import atom_table


class DES370K(BaseInteractionDataset):
    __name__ = "des370k_interaction"
    __energy_unit__ = "hartree"
    __distance_unit__ = "ang"
    __forces_unit__ = "hartree/ang"
    __energy_methods__ = [
        "mp2/cc-pvdz",
        "mp2/cc-pvqz",
        "mp2/cc-pvtz",
        "mp2/cbs",
        "ccsd(t)/cc-pvdz",
        "ccsd(t)/cbs",  # cbs
        "ccsd(t)/nn",  # nn
        "sapt0/aug-cc-pwcvxz",
        "sapt0/aug-cc-pwcvxz_es",
        "sapt0/aug-cc-pwcvxz_ex",
        "sapt0/aug-cc-pwcvxz_exs2",
        "sapt0/aug-cc-pwcvxz_ind",
        "sapt0/aug-cc-pwcvxz_exind",
        "sapt0/aug-cc-pwcvxz_disp",
        "sapt0/aug-cc-pwcvxz_exdisp_os",
        "sapt0/aug-cc-pwcvxz_exdisp_ss",
        "sapt0/aug-cc-pwcvxz_delta_HF",
    ]

    energy_target_names = [
        "cc_MP2_all",
        "qz_MP2_all",
        "tz_MP2_all",
        "cbs_MP2_all",
        "cc_CCSD(T)_all",
        "cbs_CCSD(T)_all",
        "nn_CCSD(T)_all",
        "sapt_all",
        "sapt_es",
        "sapt_ex",
        "sapt_exs2",
        "sapt_ind",
        "sapt_exind",
        "sapt_disp",
        "sapt_exdisp_os",
        "sapt_exdisp_ss",
        "sapt_delta_HF",
    ]

    def read_raw_entries(self) -> List[Dict]:
        self.filepath = os.path.join(self.root, "DES370K.csv")
        logger.info(f"Reading DES370K interaction data from {self.filepath}")
        df = pd.read_csv(self.filepath)
        data = []
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            smiles0, smiles1 = row["smiles0"], row["smiles1"]
            charge0, charge1 = row["charge0"], row["charge1"]
            natoms0, natoms1 = row["natoms0"], row["natoms1"]
            pos = np.array(list(map(float, row["xyz"].split()))).reshape(-1, 3)
            pos0 = pos[:natoms0]
            pos1 = pos[natoms0:]
            
            elements = row["elements"].split()
            elements0 = np.array(elements[:natoms0])
            elements1 = np.array(elements[natoms0:])

            atomic_nums = np.expand_dims(np.array([atom_table.GetAtomicNumber(x) for x in elements]), axis=1)
            atomic_nums0 = np.array(atomic_nums[:natoms0])
            atomic_nums1 = np.array(atomic_nums[natoms0:])

            charges = np.expand_dims(np.array([charge0] * natoms0 + [charge1] * natoms1), axis=1)

            atomic_inputs = np.concatenate((atomic_nums, charges, pos), axis=-1, dtype=np.float32)
            atomic_inputs0 = atomic_inputs[:natoms0, :]
            atomic_inputs1 = atomic_inputs[natoms0:, :]

            energies = np.array(row[self.energy_target_names].values).astype(np.float32)[None, :]

            name = np.array([smiles0 + "." + smiles1])

            item = dict(
                energies=energies,
                subset=np.array(["DES370K"]),
                n_atoms=np.array([natoms0 + natoms1], dtype=np.int32),
                n_atoms_first=np.array([natoms0], dtype=np.int32),
                atomic_inputs=atomic_inputs,
                name=name,
            )
            data.append(item)
        return data
