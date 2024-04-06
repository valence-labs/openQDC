import os
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from openqdc.datasets.interaction.base import BaseInteractionDataset
from openqdc.methods import InteractionMethod, InterEnergyType
from openqdc.utils.constants import ATOM_TABLE
from openqdc.utils.io import get_local_cache
from openqdc.utils.molecule import molecule_groups


class DES370K(BaseInteractionDataset):
    """
    DE Shaw Research interaction energy of over 370K
    small molecule dimers as described in the paper:

    Quantum chemical benchmark databases of gold-standard dimer interaction energies.
    Donchev, A.G., Taube, A.G., Decolvenaere, E. et al.
    Sci Data 8, 55 (2021).
    https://doi.org/10.1038/s41597-021-00833-x
    """

    __name__ = "des370k_interaction"
    __energy_unit__ = "kcal/mol"
    __distance_unit__ = "ang"
    __forces_unit__ = "kcal/mol/ang"
    __energy_methods__ = [
        InteractionMethod.MP2_CC_PVDZ,
        InteractionMethod.MP2_CC_PVQZ,
        InteractionMethod.MP2_CC_PVTZ,
        InteractionMethod.MP2_CBS,
        InteractionMethod.CCSD_T_CC_PVDZ,
        InteractionMethod.CCSD_T_CBS,
        InteractionMethod.CCSD_T_NN,
        InteractionMethod.SAPT0_AUG_CC_PWCVXZ,
        InteractionMethod.SAPT0_AUG_CC_PWCVXZ,
        InteractionMethod.SAPT0_AUG_CC_PWCVXZ,
        InteractionMethod.SAPT0_AUG_CC_PWCVXZ,
        InteractionMethod.SAPT0_AUG_CC_PWCVXZ,
        InteractionMethod.SAPT0_AUG_CC_PWCVXZ,
        InteractionMethod.SAPT0_AUG_CC_PWCVXZ,
        InteractionMethod.SAPT0_AUG_CC_PWCVXZ,
        InteractionMethod.SAPT0_AUG_CC_PWCVXZ,
        InteractionMethod.SAPT0_AUG_CC_PWCVXZ,
    ]

    __energy_type__ = [
        InterEnergyType.TOTAL,
        InterEnergyType.TOTAL,
        InterEnergyType.TOTAL,
        InterEnergyType.TOTAL,
        InterEnergyType.TOTAL,
        InterEnergyType.TOTAL,
        InterEnergyType.TOTAL,
        InterEnergyType.TOTAL,
        InterEnergyType.ES,
        InterEnergyType.EX,
        InterEnergyType.EX_S2,
        InterEnergyType.IND,
        InterEnergyType.EX_IND,
        InterEnergyType.DISP,
        InterEnergyType.EX_DISP_OS,
        InterEnergyType.EX_DISP_SS,
        InterEnergyType.DELTA_HF,
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

    _filename = "DES370K.csv"
    _name = "des370k_interaction"

    @classmethod
    def _root(cls):
        return os.path.join(get_local_cache(), cls._name)

    @classmethod
    def _read_raw_entries(cls) -> List[Dict]:
        filepath = os.path.join(cls._root(), cls._filename)
        logger.info(f"Reading {cls._name} interaction data from {filepath}")
        df = pd.read_csv(filepath)
        data = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            smiles0, smiles1 = row["smiles0"], row["smiles1"]
            charge0, charge1 = row["charge0"], row["charge1"]
            natoms0, natoms1 = row["natoms0"], row["natoms1"]
            pos = np.array(list(map(float, row["xyz"].split()))).reshape(-1, 3)

            elements = row["elements"].split()
            atomic_nums = np.expand_dims(np.array([ATOM_TABLE.GetAtomicNumber(x) for x in elements]), axis=1)
            charges = np.expand_dims(np.array([charge0] * natoms0 + [charge1] * natoms1), axis=1)
            atomic_inputs = np.concatenate((atomic_nums, charges, pos), axis=-1, dtype=np.float32)
            energies = np.array(row[cls.energy_target_names].values).astype(np.float32)[None, :]
            name = np.array([smiles0 + "." + smiles1])

            subsets = []
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

    def read_raw_entries(self) -> List[Dict]:
        return DES370K._read_raw_entries()
