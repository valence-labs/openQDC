from os.path import join as p_join

import numpy as np

from openqdc.datasets.base import BaseDataset, read_qc_archive_h5
from openqdc.utils.constants import MAX_ATOMIC_NUMBER


class GDML(BaseDataset):
    """
    Gradient Domain Machine Learning (GDML) is a dataset consisting of samples from ab initio
    molecular dynamics (AIMD) trajectories. The dataset consists of,
    - Benzene: 627000 samples
    - Uracil: 133000 samples
    - Naptalene: 326000 samples
    - Aspirin: 211000 samples
    - Salicylic Acid: 320000 samples
    - Malonaldehyde: 993000 samples
    - Ethanol: 555000 samples
    - Toluene: 100000 samples

    Usage
    ```python
    from openqdc.datasets import GDML
    dataset = GDML()
    ```

    References:
    - https://www.science.org/doi/10.1126/sciadv.1603015
    - http://www.sgdml.org/#datasets
    """

    __name__ = "gdml"

    # Energy in hartree, all zeros by default
    atomic_energies = np.zeros((MAX_ATOMIC_NUMBER,), dtype=np.float32)

    __energy_methods__ = [
        "ccsd",
        "ccsd(t)",
        "pbe-ts",
    ]

    energy_target_names = [
        "CCSD Energy",
        "CCSD(T) Energy",
        "PBE-TS Energy",
    ]

    __force_methods__ = [
        "ccsd",
        "ccsd(t)",
        "pbe-ts",
    ]

    force_target_names = [
        "CCSD Gradient",
        "CCSD(T) Gradient",
        "PBE-TS Gradient",
    ]

    def __init__(self) -> None:
        super().__init__()

    def read_raw_entries(self):
        raw_path = p_join(self.root, "gdml.h5")
        samples = read_qc_archive_h5(raw_path, "gdml", self.energy_target_names, self.force_target_names)

        return samples


if __name__ == "__main__":
    for data_class in [GDML]:
        data = data_class()
        n = len(data)

        for i in np.random.choice(n, 3, replace=False):
            x = data[i]
            print(x.name, x.subset, end=" ")
            for k in x:
                if x[k] is not None:
                    print(k, x[k].shape, end=" ")

            print()
