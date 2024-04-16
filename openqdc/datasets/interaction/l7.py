import numpy as np

from openqdc.methods import InteractionMethod

from ._utils import YamlDataset


class L7(YamlDataset):
    """
    The L7 interaction energy dataset as described in:

    Accuracy of Quantum Chemical Methods for Large Noncovalent Complexes
    Robert Sedlak, Tomasz Janowski, Michal Pitoňák, Jan Řezáč, Peter Pulay, and Pavel Hobza
    Journal of Chemical Theory and Computation 2013 9 (8), 3364-3374
    DOI: 10.1021/ct400036b

    Data was downloaded and extracted from:
    http://cuby4.molecular.cz/dataset_l7.html
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

    def _process_name(self, item):
        return item.geometry.split(":")[1]

    def get_n_atoms_ptr(self, item, root, filename):
        return np.array([int(item.setup["molecule_a"]["selection"].split("-")[1])], dtype=np.int32)
