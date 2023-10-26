from os.path import join as p_join

import numpy as np

from openqdc.datasets.base import BaseDataset, read_qc_archive_h5
from openqdc.utils.constants import MAX_ATOMIC_NUMBER


class SN2RXN(BaseDataset):
    __name__ = "sn2_rxn"

    # Energy in hartree, all zeros by default
    atomic_energies = np.zeros((MAX_ATOMIC_NUMBER,), dtype=np.float32)

    __energy_methods__ = [
        "dsd-blyp-d3(bj)/def2-tzvp",
    ]
    __energy_unit__ = "ev"
    __distance_unit__ = "ang"
    __forces_unit__ = "ev/ang"

    energy_target_names = [
        "DSD-BLYP-D3(BJ):def2-TZVP Atomization Energy",
    ]

    __force_methods__ = [
        "dsd-blyp-d3(bj)/def2-tzvp",
    ]

    force_target_names = [
        "DSD-BLYP-D3(BJ):def2-TZVP Gradient",
    ]

    def read_raw_entries(self):
        raw_path = p_join(self.root, "sn2_rxn.h5")
        samples = read_qc_archive_h5(raw_path, "sn2_rxn", self.energy_target_names, self.force_target_names)

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
