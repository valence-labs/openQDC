from os.path import join as p_join

import numpy as np
from tqdm import tqdm

from openqdc.datasets.base import BaseDataset
from openqdc.methods import PotentialMethod
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
        energies=energies.astype(np.float64),
        atomic_inputs=a_inputs.astype(np.float32),
        forces=forces.astype(np.float32),
        n_atoms=n_atoms,
    )

    return res


class QM7X(BaseDataset):
    """
    QM7X is a collection of almost 4.2 million conformers from 6,950 unique molecules. It contains DFT
    energy and force labels at the PBE0+MBD level of theory. It consists of structures for molecules with
    up to seven heavy (C, N, O, S, Cl) atoms from the GDB13 database. For each molecule, (meta-)stable
    equilibrium structures including constitutional/structural isomers and stereoisomers are
    searched using density-functional tight binding (DFTB). Then, for each (meta-)stable structure, 100
    off-equilibrium structures are obtained and labeled with PBE0+MBD.

    Usage:
    ```python
    from openqdc.datasets import QM7X
    dataset = QM7X()
    ```

    References:
    - https://arxiv.org/abs/2006.15139
    - https://zenodo.org/records/4288677
    """

    __name__ = "qm7x"

    __energy_methods__ = [PotentialMethod.PBE0_DEF2_TZVP, PotentialMethod.DFT3B]  # "pbe0/def2-tzvp", "dft3b"]

    energy_target_names = ["ePBE0", "eMBD"]

    __force_mask__ = [True, True]

    force_target_names = ["pbe0FOR", "vdwFOR"]

    __energy_unit__ = "ev"
    __distance_unit__ = "ang"
    __forces_unit__ = "ev/ang"
    __links__ = {f"{i}000.xz": f"https://zenodo.org/record/4288677/files/{i}000.xz" for i in range(1, 9)}

    def read_raw_entries(self):
        samples = []
        for i in range(1, 9):
            raw_path = p_join(self.root, f"{i}000")
            data = load_hdf5_file(raw_path)
            samples += [
                read_mol(data[k], k, self.energy_target_names, self.force_target_names) for k in tqdm(data.keys())
            ]

        return samples


class QM7X_V2(QM7X):
    __name__ = "qm7x_v2"
    __energy_methods__ = QM7X.__energy_methods__ + [PotentialMethod.PM6]
    __force_mask__ = QM7X.__force_mask__ + [False]
    energy_target_names = QM7X.energy_target_names + ["PM6"]
    force_target_names = QM7X.force_target_names
