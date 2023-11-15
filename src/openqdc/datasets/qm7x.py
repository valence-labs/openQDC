from os.path import join as p_join

import numpy as np
from numpy import array, float32
from tqdm import tqdm

from openqdc.datasets.base import BaseDataset
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

    __energy_methods__ = ["pbe0/mbd", "dft3b"]

    energy_target_names = ["ePBE0", "eMBD"]

    __force_methods__ = ["pbe0/mbd", "dft3b"]

    force_target_names = ["pbe0FOR", "vdwFOR"]

    __energy_unit__ = "ev"
    __distance_unit__ = "ang"
    __forces_unit__ = "ev/ang"
    __average_nb_atoms__ = 16.84668721109399

    @property
    def _stats(self):
        return {
            "formation": {
                "energy": {
                    "mean": self.convert_energy(array([-82.57984067, 372.52167714])),
                    "std": self.convert_energy(array([9.85675539, 39.76633713])),
                },
                "forces": {
                    "mean": self.convert_forces(array([-1.1617619e-07])),
                    "std": self.convert_forces(array([1.1451852])),
                    "components": {
                        "mean": self.convert_forces(
                            array(
                                [
                                    [-7.1192130e-07, -6.0926320e-11],
                                    [-4.3502279e-08, -3.7376963e-11],
                                    [5.8300976e-08, 2.9215352e-11],
                                ],
                                dtype=float32,
                            )
                        ),
                        "std": self.convert_forces(
                            array(
                                [[1.4721272, 0.00549965], [1.4861498, 0.00508684], [1.4812028, 0.00496012]],
                                dtype=float32,
                            )
                        ),
                        "rms": self.convert_forces(
                            array(
                                [[1.4721272, 0.00549965], [1.4861498, 0.00508684], [1.4812028, 0.00496012]],
                                dtype=float32,
                            )
                        ),
                    },
                },
            },
            "total": {
                "energy": {
                    "mean": self.convert_energy(array([-8.6828701e03, -2.7446982e-01], dtype=float32)),
                    "std": self.convert_energy(array([1.4362784e03, 5.8798514e-02], dtype=float32)),
                },
                "forces": {
                    "mean": self.convert_forces(array([-1.1617619e-07])),
                    "std": self.convert_forces(array([1.1451852])),
                    "components": {
                        "mean": self.convert_forces(
                            array(
                                [
                                    [-7.1192130e-07, -6.0926320e-11],
                                    [-4.3502279e-08, -3.7376963e-11],
                                    [5.8300976e-08, 2.9215352e-11],
                                ],
                                dtype=float32,
                            )
                        ),
                        "std": self.convert_forces(
                            array(
                                [[1.4721272, 0.00549965], [1.4861498, 0.00508684], [1.4812028, 0.00496012]],
                                dtype=float32,
                            )
                        ),
                        "rms": self.convert_forces(
                            array(
                                [[1.4721272, 0.00549965], [1.4861498, 0.00508684], [1.4812028, 0.00496012]],
                                dtype=float32,
                            )
                        ),
                    },
                },
            },
        }

    def read_raw_entries(self):
        samples = []
        for i in range(1, 9):
            raw_path = p_join(self.root, f"{i}000")
            data = load_hdf5_file(raw_path)
            samples += [
                read_mol(data[k], k, self.energy_target_names, self.force_target_names) for k in tqdm(data.keys())
            ]

        return samples
