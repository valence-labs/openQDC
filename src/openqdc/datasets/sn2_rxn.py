from os.path import join as p_join

import numpy as np

from openqdc.datasets.base import BaseDataset, read_qc_archive_h5
from openqdc.utils.constants import MAX_ATOMIC_NUMBER


class SN2RXN(BaseDataset):
    __name__ = "iso_17"

    # Energy in hartree, all zeros by default
    atomic_energies = np.zeros((MAX_ATOMIC_NUMBER,), dtype=np.float32)

    __energy_methods__ = [
        "pbe-ts",
    ]

    energy_target_names = [
        "PBE-TS Energy",
    ]

    __force_methods__ = [
        "pbe-ts",
    ]

    force_target_names = [
        "PBE-TS Gradient",
    ]

    def __init__(self) -> None:
        super().__init__()

    def read_raw_entries(self):
        raw_path = p_join(self.root, "iso_17.h5")
        samples = read_qc_archive_h5(raw_path, "iso_17", self.energy_target_names, self.force_target_names)

        return samples


if __name__ == "__main__":
    for data_class in [SN2RXN]:
        data = data_class()
        n = len(data)

        for i in np.random.choice(n, 3, replace=False):
            x = data[i]
            print(x.name, x.subset, end=" ")
            for k in x:
                if x[k] is not None:
                    print(k, x[k].shape, end=" ")

            print()