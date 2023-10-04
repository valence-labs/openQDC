import os
from os.path import join as p_join

import numpy as np

from openqdc.datasets.base import BaseDataset, read_qc_archive_h5
from openqdc.utils.constants import MAX_ATOMIC_NUMBER
from openqdc.utils.io import get_local_cache


class ANI1(BaseDataset):
    """
    The ANI-1 dataset is a collection of 22 x 10^6 structural conformations from 57,000 distinct small
    organic molecules with energy labels calculated using DFT. The molecules
    contain 4 distinct atoms, C, N, O and H.

    Usage
    ```python
    from openqdc.datasets import ANI1
    dataset = ANI1()
    ```

    References:
    - ANI-1: https://www.nature.com/articles/sdata2017193
    - Github: https://github.com/aiqm/ANI1x_datasets
    """

    __name__ = "ani1"

    # Energy in hartree, all zeros by default
    atomic_energies = np.zeros((MAX_ATOMIC_NUMBER,), dtype=np.float32)

    __energy_methods__ = [
        "wb97x_6-31g(d)",
    ]

    energy_target_names = [
        "ωB97x:6-31G(d) Energy",
    ]

    def __init__(self) -> None:
        super().__init__()

    @property
    def root(self):
        return p_join(get_local_cache(), "ani")

    @property
    def preprocess_path(self):
        path = p_join(self.root, "preprocessed", self.__name__)
        os.makedirs(path, exist_ok=True)
        return path

    def read_raw_entries(self):
        raw_path = p_join(self.root, f"{self.__name__}.h5")
        samples = read_qc_archive_h5(raw_path, self.__name__, self.energy_target_names, self.force_target_names)
        return samples


class ANI1CCX(ANI1):
    """


    Usage
    ```python
    from openqdc.datasets import ANI1CCX
    dataset = ANI1CCX()
    ```

    References:
    - ANI-1ccx: https://doi.org/10.1038/s41467-019-10827-4
    - Github: https://github.com/aiqm/ANI1x_datasets
    """

    __name__ = "ani1ccx"

    # Energy in hartree, all zeros by default
    atomic_energies = np.zeros((MAX_ATOMIC_NUMBER,), dtype=np.float32)

    __energy_methods__ = [
        "ccsd(t)_cbs",
        "npno_ccsd(t)_dz",
        "npno_ccsd(t)_tz",
        "tpno_ccsd(t)_dz",
    ]

    energy_target_names = [
        "CCSD(T)*:CBS Total Energy",
        "NPNO-CCSD(T):cc-pVDZ Correlation Energy",
        "NPNO-CCSD(T):cc-pVTZ Correlation Energy",
        "TPNO-CCSD(T):cc-pVDZ Correlation Energy",
    ]

    __force_methods__ = []
    force_target_names = []

    def __init__(self) -> None:
        super().__init__()


class ANI1X(ANI1):
    """
    The ANI-1X dataset consists of ANI-1 molecules + some molecules added using active learning which leads to
    a total of 5,496,771 conformers with 63,865 unique molecules.

    Usage
    ```python
    from openqdc.datasets import ANI1X
    dataset = ANI1X()
    ```

    References:
    - ANI-1x: https://doi.org/10.1063/1.5023802
    - Github: https://github.com/aiqm/ANI1x_datasets
    """

    __name__ = "ani1x"

    # Energy in hartree, all zeros by default
    atomic_energies = np.zeros((MAX_ATOMIC_NUMBER,), dtype=np.float32)

    __energy_methods__ = [
        "hf_dz",
        "hf_qz",
        "hf_tz",
        "mp2_dz",
        "mp2_qz",
        "mp2_tz",
        "wb97x_6-31g(d)",
        "wb97x_tz",
    ]

    energy_target_names = [
        "HF:cc-pVDZ Total Energy",
        "HF:cc-pVQZ Total Energy",
        "HF:cc-pVTZ Total Energy",
        "MP2:cc-pVDZ Correlation Energy",
        "MP2:cc-pVQZ Correlation Energy",
        "MP2:cc-pVTZ Correlation Energy",
        "wB97x:6-31G(d) Total Energy",
        "wB97x:def2-TZVPP Total Energy",
    ]

    force_target_names = [
        "wB97x:6-31G(d) Atomic Forces",
        "wB97x:def2-TZVPP Atomic Forces",
    ]

    __force_methods__ = [
        "wb97x_6-31g(d)",
        "wb97x_tz",
    ]

    def __init__(self) -> None:
        super().__init__()


if __name__ == "__main__":
    for data_class in [
        ANI1,
        # ANI1CCX,
        # ANI1X
    ]:
        data = data_class()
        n = len(data)

        for i in np.random.choice(n, 3, replace=False):
            x = data[i]
            print(x.name, x.subset, end=" ")
            for k in x:
                if x[k] is not None:
                    print(k, x[k].shape, end=" ")

            print()
        exit()
