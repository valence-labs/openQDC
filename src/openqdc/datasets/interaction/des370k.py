import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Dict, List

from loguru import logger
from openqdc.datasets.interaction import BaseInteractionDataset

class Dimer:
    def __init__(
        self,
        smiles_0: str,
        smiles_1: str,
        charge_0: int,
        charge_1: int,
        n_atoms_0: int,
        n_atoms_1: int,
        pos: np.array,
        sapt_energies: List[float],
    ) -> None:
        self.smiles_0 = smiles_0
        self.smiles_1 = smiles_1
        self.charge_1 = charge_0
        self.charge_1 = charge_1
        self.n_atoms_0 = n_atoms_0
        self.n_atoms_1 = n_atoms_1
        self.pos = pos
        self.sapt_energies = sapt_energies
        (
            self.sapt_es,
            self.sapt_ex,
            self.sapt_exs2,
            self.sapt_ind,
            self.sapt_exind,
            self.sapt_disp,
            self.sapt_exdisp_os,
            self.sapt_exdisp_ss,
            self.sapt_delta_HF,
            self.sapt_all
        ) = tuple(sapt_energies)

    def __str__(self) -> str:
        return f"Dimer(smiles_0='{self.smiles_0}', smiles_1='{self.smiles_1}')"

    def __repr__(self) -> str:
        return str(self)


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
        "sapt0/aug-cc-pwcvxz",
        "sapt0/aug-cc-pwcvxz",
        "sapt0/aug-cc-pwcvxz",
        "sapt0/aug-cc-pwcvxz",
        "sapt0/aug-cc-pwcvxz",
        "sapt0/aug-cc-pwcvxz",
        "sapt0/aug-cc-pwcvxz",
        "sapt0/aug-cc-pwcvxz",
        "sapt0/aug-cc-pwcvxz",
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
        logger.info(f"Reading data from {self.filepath}")
        df = pd.read_csv(self.filepath)
        data = []
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            smiles0, smiles1 = row["smiles0"], row["smiles1"]
            natoms0, natoms1 = row["natoms0"], row["natoms1"]
            pos = np.array(list(map(float, row["xyz"].split()))).reshape(-1, 3)
            pos0 = pos[:natoms0]
            pos1 = pos[natoms0:]
            # sapt_components = {col: row[col] for col in df.columns if "sapt" in col}
            item = dict(
                mol0=dict(
                    smiles=smiles0,
                    atomic_inputs=pos0,
                    n_atoms=natoms0,
                ),
                mol1=dict(
                    smiles=smiles1,
                    atomic_inputs=pos1,
                    n_atoms=natoms1,
                ),
                targets=row[self.energy_target_names].values,
            )
            data.append(item)
        return data
