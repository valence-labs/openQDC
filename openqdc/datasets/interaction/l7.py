import numpy as np

from openqdc.methods import InteractionMethod

from ._utils import YamlDataset


class L7(YamlDataset):
    """
    The L7 interaction energy dataset consists of 7 dispersion stabilized non-covalent complexes with
    energies labelled using semi-empirical and quantum mechanical methods. The intial geometries are
    taken from crystal X-ray data and optimized with a DFT method specific to the complex.

    Usage:
    ```python
    from openqdc.datasets import L7
    dataset = L7()
    ```

    Reference:
        https://pubs.acs.org/doi/10.1021/ct400036b
    """

    __name__ = "l7"
    __energy_methods__ = [
        InteractionMethod.QCISDT_CBS,  # "QCISD(T)/CBS",
        InteractionMethod.DLPNO_CCSDT,  # "DLPNO-CCSD(T)",
        InteractionMethod.MP2_CBS,  # "MP2/CBS",
        InteractionMethod.MP2C_CBS,  # "MP2C/CBS",
        InteractionMethod.FIXED,  # "fixed", TODO: we should remove this level of theory because unless we have a pro
        InteractionMethod.DLPNO_CCSDT0,  # "DLPNO-CCSD(T0)",
        InteractionMethod.LNO_CCSDT,  # "LNO-CCSD(T)",
        InteractionMethod.FN_DMC,  # "FN-DMC",
    ]
    __links__ = {
        "l7.yaml": "http://cuby4.molecular.cz/download_datasets/l7.yaml",
        "geometries.tar.gz": "http://cuby4.molecular.cz/download_geometries/L7.tar",
    }

    def _process_name(self, item):
        return item.geometry.split(":")[1]

    def get_n_atoms_ptr(self, item, root, filename):
        return np.array([int(item.setup["molecule_a"]["selection"].split("-")[1])], dtype=np.int32)
