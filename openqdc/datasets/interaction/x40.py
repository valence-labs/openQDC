from os.path import join as p_join

import numpy as np

from openqdc.datasets.interaction._utils import YamlDataset
from openqdc.methods import InteractionMethod


class X40(YamlDataset):
    """
    X40 interaction dataset of 40 dimer pairs as
    introduced in the following paper:

    Benchmark Calculations of Noncovalent Interactions of Halogenated Molecules
    Jan Řezáč, Kevin E. Riley, and Pavel Hobza
    Journal of Chemical Theory and Computation 2012 8 (11), 4285-4292
    DOI: 10.1021/ct300647k

    Dataset retrieved and processed from:
    http://cuby4.molecular.cz/dataset_x40.html
    """

    __name__ = "x40"
    __energy_methods__ = [
        InteractionMethod.CCSD_T_CBS,  # "CCSD(T)/CBS",
        InteractionMethod.MP2_CBS,  # "MP2/CBS",
        InteractionMethod.DCCSDT_HA_DZ,  # "dCCSD(T)/haDZ",
        InteractionMethod.DCCSDT_HA_TZ,  # "dCCSD(T)/haTZ",
        InteractionMethod.MP2_5_CBS_ADZ,  # "MP2.5/CBS(aDZ)",
    ]

    def _process_name(self, item):
        return item.shortname

    def get_n_atoms_ptr(self, item, root, filename):
        xyz_path = p_join(root, f"{filename}.xyz")
        with open(xyz_path, "r") as xyz_file:  # avoid not closing the file
            lines = list(map(lambda x: x.strip().split(), xyz_file.readlines()))
            setup = lines.pop(1)
            n_atoms_first = setup[0].split("-")[1]
            n_atoms_ptr = np.array([int(n_atoms_first)], dtype=np.int32)
            return n_atoms_ptr
