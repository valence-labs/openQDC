import os
import numpy as np
from os.path import join as p_join
from openqdc.utils.constants import MAX_ATOMIC_NUMBER
from openqdc.datasets.base import BaseDataset, read_qc_archive_h5


class GDML(BaseDataset):
    __name__ = 'gdml'

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
        raw_path = p_join(self.root, f'gdml.h5')
        samples = read_qc_archive_h5(raw_path, "gdml", self.energy_target_names, 
                                      self.force_target_names)

        return samples



if __name__ == '__main__':
    for data_class in [GDML]:
        data = data_class()
        n = len(data)

        for i in np.random.choice(n, 3, replace=False):
            x = data[i]
            print(x.name, x.subset, end=' ')
            for k in x:
                if x[k] is not None:
                    print(k, x[k].shape, end=' ')
                
            print()
