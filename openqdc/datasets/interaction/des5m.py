from typing import Dict, List
from openqdc.methods import InteractionMethod, InterEnergyType
from openqdc.datasets.interaction.des370k import DES370K


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
    __energy_methods__ = [
        InteractionMethod.MP2_CC_PVQZ,
        InteractionMethod.MP2_CC_PVTZ,
        InteractionMethod.MP2_CBS,
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

    _filename = "DES5M.csv"
    _name = "des5m_interaction"

    def read_raw_entries(self) -> List[Dict]:
        return DES5M._read_raw_entries()
