import os
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from openqdc.datasets.interaction import BaseInteractionDataset
from openqdc.utils.molecule import atom_table


class DESS66(BaseInteractionDataset):
    """
    DE Shaw Research interaction energy
    estimates of all 66 conformers from
    the original S66 dataset as described
    in the paper:

    Quantum chemical benchmark databases of gold-standard dimer interaction energies.
    Donchev, A.G., Taube, A.G., Decolvenaere, E. et al.
    Sci Data 8, 55 (2021).
    https://doi.org/10.1038/s41597-021-00833-x

    Data was downloaded from Zenodo:
    https://zenodo.org/records/5676284
    """

    __name__ = "des_s66"
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
        self.filepath = os.path.join(self.root, "DESS66.csv")
        logger.info(f"Reading DESS66 interaction data from {self.filepath}")
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

            subset = row["system_name"]

            item = dict(
                energies=energies,
                subset=np.array([subset]),
                n_atoms=np.array([natoms0 + natoms1], dtype=np.int32),
                n_atoms_first=np.array([natoms0], dtype=np.int32),
                atomic_inputs=atomic_inputs,
                name=name,
            )
            data.append(item)
        return data
