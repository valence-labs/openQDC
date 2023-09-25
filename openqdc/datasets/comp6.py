from os.path import join as p_join

import numpy as np

from openqdc.datasets.base import BaseDataset, read_qc_archive_h5
from openqdc.utils.constants import MAX_ATOMIC_NUMBER


class COMP6(BaseDataset):
    __name__ = "comp6"

    # Energy in hartree, all zeros by default
    atomic_energies = np.zeros((MAX_ATOMIC_NUMBER,), dtype=np.float32)

    __energy_methods__ = [
        "wb97x_6-31g*",
        "b3lyp-d3m(bj)_tz",
        "b3lyp_tz",
        "hf_tz",
        "pbe-d3(bj)_dz",
        "pbe_tz",
        "svwm_tz",
        "wb97m-d3(bj)_tz",
        "wb97m_tz",
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
        "wb97x_6-31g*",
    ]

    force_target_names = [
        "Gradient",
    ]

    def __init__(self) -> None:
        super().__init__()

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
