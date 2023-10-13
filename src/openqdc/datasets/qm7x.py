from os.path import join as p_join

import numpy as np
from tqdm import tqdm

from openqdc.datasets.base import BaseDataset
from openqdc.utils.constants import MAX_ATOMIC_NUMBER
from openqdc.utils.io import load_hdf5_file


def read_mol(mol_h5, mol_name, energy_target_names, force_target_names):
    m = mol_h5
    cids = list(mol_h5.keys())

    zs = [m[c]["atNUM"] for c in cids]
    xyz = np.concatenate([m[c]["atXYZ"] for c in cids], axis=0)
    n_atoms = np.array([len(z) for z in zs], dtype=np.int32)
    n, zs = len(n_atoms), np.concatenate(zs, axis=0)
    a_inputs = np.concatenate([np.stack([zs, np.zeros_like(zs)], axis=-1), xyz], axis=-1)

    forces = np.concatenate([np.stack([m[c][f_tag] for f_tag in force_target_names], axis=-1) for c in cids], axis=0)
    energies = np.stack([np.array([m[c][e_tag][0] for e_tag in energy_target_names]) for c in cids], axis=0)

    res = dict(
        name=np.array([mol_name] * n),
        subset=np.array(["qm7x"] * n),
        energies=energies.astype(np.float32),
        atomic_inputs=a_inputs.astype(np.float32),
        forces=forces.astype(np.float32),
        n_atoms=n_atoms,
    )

    return res


class QM7X(BaseDataset):
    __name__ = "qm7x"

    # Energy in hartree, all zeros by default
    atomic_energies = np.zeros((MAX_ATOMIC_NUMBER,), dtype=np.float32)

    __energy_methods__ = ["pbe+vdw-ts", "mbd"]

    energy_target_names = ["ePBE0", "eMBD"]

    __force_methods__ = ["pbe+vdw-ts", "mbd"]

    force_target_names = ["pbe0FOR", "vdwFOR"]

    __energy_unit__ = "ev"
    __distance_unit__ = "ang"
    __forces_unit__ = "ev/ang"

    def read_raw_entries(self):
        samples = []
        for i in range(1, 9):
            raw_path = p_join(self.root, f"{i}000")
            data = load_hdf5_file(raw_path)
            samples += [
                read_mol(data[k], k, self.energy_target_names, self.force_target_names) for k in tqdm(data.keys())
            ]

        return samples


if __name__ == "__main__":
    for data_class in [QM7X]:
        data = data_class()
        n = len(data)

        for i in np.random.choice(n, 3, replace=False):
            x = data[i]
            print(x.name, x.subset, end=" ")
            for k in x:
                if x[k] is not None:
                    print(k, x[k].shape, end=" ")
