from os.path import join as p_join

import numpy as np

from openqdc.datasets.interaction._utils import YamlDataset
from openqdc.methods import InteractionMethod


class X40(YamlDataset):
    """
    X40 interaction dataset of 40 noncovalent complexes of organic halides, halohydrides, and halogen molecules
    where the halogens participate in various interaction types such as electrostatic interactions, london
    dispersion, hydrogen bonds, halogen bonding, halogen-pi interactions and stacking of halogenated aromatic
    molecules. For each complex 10 geometries are generated resulting in 400 geometries in the dataset. The geometries
    are optimized using the MP2 level of theory with cc-pVTZ basis set whereas the interaction energies are
    computed with CCSD(T)/CBS level of theory.

    Usage:
    ```python
    from openqdc.datasets import X40
    dataset = X40()
    ```

    Reference:
        https://pubs.acs.org/doi/10.1021/ct300647k
    """

    __name__ = "x40"
    __energy_methods__ = [
        InteractionMethod.CCSD_T_CBS,  # "CCSD(T)/CBS",
        InteractionMethod.MP2_CBS,  # "MP2/CBS",
        InteractionMethod.DCCSDT_HA_DZ,  # "dCCSD(T)/haDZ",
        InteractionMethod.DCCSDT_HA_TZ,  # "dCCSD(T)/haTZ",
        InteractionMethod.MP2_5_CBS_ADZ,  # "MP2.5/CBS(aDZ)",
    ]
    __links__ = {
        "x40.yaml": "http://cuby4.molecular.cz/download_datasets/x40.yaml",
        "geometries.tar.gz": "http://cuby4.molecular.cz/download_geometries/X40.tar",
    }

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
