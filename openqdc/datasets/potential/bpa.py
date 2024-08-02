from openqdc import BaseDataset
from openqdc.methods import PotentialMethod


class BPA(BaseDataset):
    """
    _summary_


    Usage:
    ```python
    from openqdc.datasets import BPA
    dataset = BPA()
    ```


    References:
        https://pubs.acs.org/doi/10.1021/acs.jctc.1c00647
    """

    __name__ = "BPA"
    __energy_unit__ = "ev"
    __forces_unit__ = "eV/ang"
    __distance_unit__ = "ang"
    __energy_methods__ = ([PotentialMethod.WB97X_6_31G_D],)
    __links__ = {"BPA.zip": "https://pubs.acs.org/doi/suppl/10.1021/acs.jctc.1c00647/suppl_file/ct1c00647_si_002.zip"}
