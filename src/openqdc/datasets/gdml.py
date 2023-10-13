from os.path import join as p_join

import numpy as np

from openqdc.datasets.base import BaseDataset, read_qc_archive_h5
from openqdc.utils.constants import MAX_ATOMIC_NUMBER


class GDML(BaseDataset):
    __name__ = "gdml"

    # Energy in hartree, all zeros by default
    atomic_energies = np.zeros((MAX_ATOMIC_NUMBER,), dtype=np.float32)

    __energy_methods__ = [
        "ccsd/cc-pvdz",
        "ccsd(t)/cc-pvdz",
        #"pbe+mbd/light", #MD22
        #"pbe+mbd/tight", #MD22
        "pbe+vdw-ts", #MD17
    ]

    energy_target_names = [
        "CCSD Energy",
        "CCSD(T) Energy",
        "PBE-TS Energy",
    ]

    __force_methods__ = [
        "ccsd/cc-pvdz",
        "ccsd(t)/cc-pvdz",
        #"pbe+mbd/light", #MD22
        #"pbe+mbd/tight", #MD22
        "pbe+vdw-ts", #MD17
    ]

    force_target_names = [
        "CCSD Gradient",
        "CCSD(T) Gradient",
        "PBE-TS Gradient",
    ]

    __energy_unit__   = "kcal/mol"
    __distance_unit__ = "ang"
    __forces_unit__   = "kcal/mol/ang"

    def __init__(self, energy_unit = None, distance_unit = None) -> None:
        super().__init__(energy_unit=energy_unit, distance_unit=distance_unit)

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
