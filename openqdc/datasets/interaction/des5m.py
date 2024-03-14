from typing import Dict, List

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

    _filename = "DES5M.csv"
    _name = "des5m_interaction"

    def read_raw_entries(self) -> List[Dict]:
        return DES5M._read_raw_entries()
