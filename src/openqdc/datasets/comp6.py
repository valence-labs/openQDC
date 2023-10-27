from os.path import join as p_join

import numpy as np

from openqdc.datasets.base import BaseDataset, read_qc_archive_h5
from openqdc.utils.constants import MAX_ATOMIC_NUMBER


class COMP6(BaseDataset):
    """
    COMP6 is a benchmark suite consisting of broad regions of bio-chemical and organic space
    developed for testing the ANI-1x potential. It is curated from 6 benchmark sets:
    S66x8, ANI Molecular Dynamics, GDB7to9, GDB10to13, DrugBank, and Tripeptides.

    Usage
    ```python
    from openqdc.datasets import COMP6
    dataset = COMP6()
    ```

    References:
    - https://aip.scitation.org/doi/abs/10.1063/1.5023802
    - Github: https://github.com/isayev/COMP6
    """

    __name__ = "comp6"

    # Energy in hartree, all zeros by default
    atomic_energies = np.zeros((MAX_ATOMIC_NUMBER,), dtype=np.float32)
    # watchout that forces are stored as -grad(E)
    __energy_unit__ = "kcal/mol"
    __distance_unit__ = "bohr"  # bohr
    __forces_unit__ = "kcal/mol/bohr"

    __energy_methods__ = [
        "wb97x/6-31g*",
        "b3lyp-d3mbj/def2-tzvp",
        "b3lyp/def2-tzvp",
        "hf/def2-tzvp",
        "pbe-d3bj/def2-tzvp",
        "pbe/def2-tzvp",
        "svwn/def2-tzvp",
        "wb97m-d3bj/def2-tzvp",
        "wb97m/def2-tzvp",
    ]

    energy_target_names = [
        "Energy",
        "B3LYP-D3M(BJ):def2-tzvp",
        "B3LYP:def2-tzvp",
        "HF:def2-tzvp",
        "PBE-D3M(BJ):def2-tzvp",
        "PBE:def2-tzvp",
        "SVWN:def2-tzvp",
        "WB97M-D3(BJ):def2-tzvp",
        "WB97M:def2-tzvp",
    ]

    __force_methods__ = [
        "wb97x/6-31g*",
    ]

    force_target_names = [
        "Gradient",
    ]

    def read_raw_entries(self):
        samples = []
        for subset in ["ani_md", "drugbank", "gdb7_9", "gdb10_13", "s66x8", "tripeptides"]:
            raw_path = p_join(self.root, f"{subset}.h5")
            samples += read_qc_archive_h5(raw_path, subset, self.energy_target_names, self.force_target_names)

        return samples


if __name__ == "__main__":
    for data_class in [COMP6]:
        data = data_class()
        n = len(data)

        for i in np.random.choice(n, 3, replace=False):
            x = data[i]
            print(x.name, x.subset, end=" ")
            for k in x:
                if x[k] is not None:
                    print(k, x[k].shape, end=" ")

            print()
