import os
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from openqdc.datasets.interaction import DES370K
from openqdc.utils.molecule import atom_table, molecule_groups


class DES5M(DES370K):
    __name__ = "des5m_interaction"
    __energy_methods__ = [
        "mp2/cc-pvqz",
        "mp2/cc-pvtz",
        "mp2/cbs",
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
        "qz_MP2_all",
        "tz_MP2_all",
        "cbs_MP2_all",
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
        self.filepath = os.path.join(self.root, "DES5M.csv")
        logger.info(f"Reading DES5M interaction data from {self.filepath}")
        df = pd.read_csv(self.filepath)
        data = []
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            smiles0, smiles1 = row["smiles0"], row["smiles1"]
            charge0, charge1 = row["charge0"], row["charge1"]
            natoms0, natoms1 = row["natoms0"], row["natoms1"]
            pos = np.array(list(map(float, row["xyz"].split()))).reshape(-1, 3)

            elements = row["elements"].split()

            atomic_nums = np.expand_dims(np.array([atom_table.GetAtomicNumber(x) for x in elements]), axis=1)

            charges = np.expand_dims(np.array([charge0] * natoms0 + [charge1] * natoms1), axis=1)

            atomic_inputs = np.concatenate((atomic_nums, charges, pos), axis=-1, dtype=np.float32)

            energies = np.array(row[self.energy_target_names].values).astype(np.float32)[None, :]

            name = np.array([smiles0 + "." + smiles1])

            subsets = []
            # for smiles in [canon_smiles0, canon_smiles1]:
            for smiles in [smiles0, smiles1]:
                found = False
                for functional_group, smiles_set in molecule_groups.items():
                    if smiles in smiles_set:
                        subsets.append(functional_group)
                        found = True
                if not found:
                    logger.info(f"molecule group lookup failed for {smiles}")

            item = dict(
                energies=energies,
                subset=np.array([subsets]),
                n_atoms=np.array([natoms0 + natoms1], dtype=np.int32),
                n_atoms_first=np.array([natoms0], dtype=np.int32),
                atomic_inputs=atomic_inputs,
                name=name,
            )
            data.append(item)
        return data
