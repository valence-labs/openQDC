import os
from os.path import join as p_join
from typing import Optional

import numpy as np
from ase.io.extxyz import write_extxyz
from sklearn.utils import Bunch

from openqdc.datasets.base import BaseDataset
from openqdc.utils.constants import MAX_CHARGE
from openqdc.utils.io import to_atoms


class BaseInteractionDataset(BaseDataset):
    __energy_type__ = []

    @property
    def pkl_data_types(self):
        return {
            "name": str,
            "subset": str,
            "n_atoms": np.int32,
            "n_atoms_ptr": np.int32,
        }

    @property
    def data_types(self):
        return {
            "atomic_inputs": np.float32,
            "position_idx_range": np.int32,
            "energies": np.float32,
            "forces": np.float32,
        }

    def __getitem__(self, idx: int):
        shift = MAX_CHARGE
        p_start, p_end = self.data["position_idx_range"][idx]
        input = self.data["atomic_inputs"][p_start:p_end]
        z, c, positions, energies = (
            self._convert_array(np.array(input[:, 0], dtype=np.int32)),
            self._convert_array(np.array(input[:, 1], dtype=np.int32)),
            self._convert_array(np.array(input[:, -3:], dtype=np.float32)),
            self._convert_array(np.array(self.data["energies"][idx], dtype=np.float32)),
        )
        name = self.__smiles_converter__(self.data["name"][idx])
        subset = self.data["subset"][idx]
        n_atoms_ptr = self.data["n_atoms_ptr"][idx]

        forces = None
        if "forces" in self.data:
            forces = self._convert_array(np.array(self.data["forces"][p_start:p_end], dtype=np.float32))

        e0 = self._convert_array(np.array(self.__isolated_atom_energies__[..., z, c + shift].T, dtype=np.float32))

        bunch = Bunch(
            positions=positions,
            atomic_numbers=z,
            charges=c,
            e0=e0,
            energies=energies,
            name=name,
            subset=subset,
            forces=forces,
            n_atoms_ptr=n_atoms_ptr,
        )

        if self.transform is not None:
            bunch = self.transform(bunch)

        return bunch

    def get_ase_atoms(self, idx: int):
        entry = self[idx]
        at = to_atoms(entry["positions"], entry["atomic_numbers"])
        at.info["n_atoms"] = entry["n_atoms_ptr"]
        return at

    def save_xyz(self, idx: int, path: Optional[str] = None):
        """
        Save the entry at index idx as an extxyz file.
        """
        if path is None:
            path = os.getcwd()
        at = self.get_ase_atoms(idx)
        n_atoms = at.info.pop("n_atoms")
        write_extxyz(p_join(path, f"mol_{idx}.xyz"), at, plain=True, comment=str(n_atoms))
