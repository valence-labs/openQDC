import os
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from openqdc.datasets.interaction.base import BaseInteractionDataset
from openqdc.methods import InteractionMethod, InterEnergyType
from openqdc.utils.constants import ATOM_TABLE
from openqdc.utils.molecule import molecule_groups


def parse_des_df(row, energy_target_names):
    smiles0, smiles1 = row["smiles0"], row["smiles1"]
    charge0, charge1 = row["charge0"], row["charge1"]
    natoms0, natoms1 = row["natoms0"], row["natoms1"]
    pos = np.array(list(map(float, row["xyz"].split()))).reshape(-1, 3)
    elements = row["elements"].split()
    atomic_nums = np.expand_dims(np.array([ATOM_TABLE.GetAtomicNumber(x) for x in elements]), axis=1)
    charges = np.expand_dims(np.array([charge0] * natoms0 + [charge1] * natoms1), axis=1)
    atomic_inputs = np.concatenate((atomic_nums, charges, pos), axis=-1, dtype=np.float32)
    energies = np.array(row[energy_target_names].values).astype(np.float32)[None, :]
    name = np.array([smiles0 + "." + smiles1])
    return {
        "energies": energies,
        "n_atoms": np.array([natoms0 + natoms1], dtype=np.int32),
        "name": name,
        "atomic_inputs": atomic_inputs,
        "charges": charges,
        "atomic_nums": atomic_nums,
        "elements": elements,
        "natoms0": natoms0,
        "natoms1": natoms1,
        "smiles0": smiles0,
        "smiles1": smiles1,
        "charge0": charge0,
        "charge1": charge1,
    }


def create_subset(smiles0, smiles1):
    subsets = []
    for smiles in [smiles0, smiles1]:
        found = False
        for functional_group, smiles_set in molecule_groups.items():
            if smiles in smiles_set:
                subsets.append(functional_group)
                found = True
        if not found:
            logger.info(f"molecule group lookup failed for {smiles}")
    return subsets


def convert_to_record(item):
    return dict(
        energies=item["energies"],
        subset=np.array([item["subset"]]),
        n_atoms=np.array([item["natoms0"] + item["natoms1"]], dtype=np.int32),
        n_atoms_first=np.array([item["natoms0"]], dtype=np.int32),
        atomic_inputs=item["atomic_inputs"],
        name=item["name"],
    )


class IDES(ABC):
    @abstractmethod
    def _create_subsets(self, **kwargs):
        raise NotImplementedError


class DES370K(BaseInteractionDataset, IDES):
    """
    DE Shaw Research interaction energy of over 370K
    small molecule dimers as described in the paper:

    Quantum chemical benchmark databases of gold-standard dimer interaction energies.
    Donchev, A.G., Taube, A.G., Decolvenaere, E. et al.
    Sci Data 8, 55 (2021).
    https://doi.org/10.1038/s41597-021-00833-x
    """

    __name__ = "des370k_interaction"
    __filename__ = "DES370K.csv"
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

    @property
    def csv_path(self):
        return os.path.join(self.root, self.__filename__)

    def _create_subsets(self, **kwargs):
        return create_subset(kwargs["smiles0"], kwargs["smiles1"])

    def read_raw_entries(self) -> List[Dict]:
        filepath = self.csv_path
        logger.info(f"Reading {self.__name__} interaction data from {filepath}")
        df = pd.read_csv(filepath)
        data = []
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            item = parse_des_df(row, self.energy_target_names)
            item["subset"] = self._create_subsets(**item)
            item = convert_to_record(item)
            data.append(item)
        return data


class DES5M(DES370K):
    """
    DE Shaw Research interaction energy calculations for
    over 5M small molecule dimers as described in the paper:

    Quantum chemical benchmark databases of gold-standard dimer interaction energies.
    Donchev, A.G., Taube, A.G., Decolvenaere, E. et al.
    Sci Data 8, 55 (2021).
    https://doi.org/10.1038/s41597-021-00833-x
    """

    __name__ = "des5m_interaction"
    __filename__ = "DES5M.csv"


class DESS66(DES370K):
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
    __filename__ = "DESS66.csv"

    # def read_raw_entries(self) -> List[Dict]:
    #    filepath = self.csv_path
    #    logger.info(f"Reading DESS66 interaction data from {filepath}")
    #    df = pd.read_csv(filepath)
    #    data = []
    #    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    #        item = parse_des_df(row)
    #        item["subset"] = row["system_name"]
    #        data.append(convert_to_record(item))
    #    return data


class DESS66x8(DESS66):
    """
    DE Shaw Research interaction energy
    estimates of all 528 conformers from
    the original S66x8 dataset as described
    in the paper:

    Quantum chemical benchmark databases of gold-standard dimer interaction energies.
    Donchev, A.G., Taube, A.G., Decolvenaere, E. et al.
    Sci Data 8, 55 (2021).
    https://doi.org/10.1038/s41597-021-00833-x

    Data was downloaded from Zenodo:

    https://zenodo.org/records/5676284
    """

    __name__ = "des_s66x8"
    __filename__ = "DESS66x8.csv"
