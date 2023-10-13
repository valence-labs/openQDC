from os.path import join as p_join

import numpy as np

from openqdc.datasets.base import BaseDataset, read_qc_archive_h5
from openqdc.utils.constants import MAX_ATOMIC_NUMBER


class ISO17(BaseDataset):
    __name__ = "iso_17"

    # Energy in hartree, all zeros by default
    atomic_energies = np.zeros((MAX_ATOMIC_NUMBER,), dtype=np.float32)

    __energy_methods__ = [
        "pbe+vdw-ts",
    ]

    energy_target_names = [
        "PBE-TS Energy",
    ]

    __force_methods__ = [
        "pbe+vdw-ts",
    ]

    force_target_names = [
        "PBE-TS Gradient",
    ]
    
    __energy_unit__   = "ev"
    __distance_unit__ = "ang"
    __forces_unit__   = "ev/ang"
    
    def __init__(self, energy_unit = None, distance_unit = None) -> None:
        super().__init__(energy_unit=energy_unit, distance_unit=distance_unit)

    def read_raw_entries(self):
        raw_path = p_join(self.root, "iso_17.h5")
        samples = read_qc_archive_h5(raw_path, "iso_17", self.energy_target_names, self.force_target_names)

        return samples


if __name__ == "__main__":
    for data_class in [ISO17]:
        data = data_class()
        n = len(data)

        for i in np.random.choice(n, 3, replace=False):
            x = data[i]
            print(x.name, x.subset, end=" ")
            for k in x:
                if x[k] is not None:
                    print(k, x[k].shape, end=" ")

            print()
