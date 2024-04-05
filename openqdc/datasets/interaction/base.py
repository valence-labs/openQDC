import os
import pickle as pkl
from os.path import join as p_join
from typing import Dict, List, Optional

import numpy as np
from ase.io.extxyz import write_extxyz
from loguru import logger
from sklearn.utils import Bunch

from openqdc.datasets.base import BaseDataset
from openqdc.utils.constants import MAX_CHARGE, NB_ATOMIC_FEATURES
from openqdc.utils.io import pull_locally, push_remote, to_atoms


class BaseInteractionDataset(BaseDataset):
    __energy_type__ = []

    def collate_list(self, list_entries: List[Dict]):
        # concatenate entries
        res = {
            key: np.concatenate([r[key] for r in list_entries if r is not None], axis=0)
            for key in list_entries[0]
            if not isinstance(list_entries[0][key], dict)
        }

        csum = np.cumsum(res.get("n_atoms"))
        x = np.zeros((csum.shape[0], 2), dtype=np.int32)
        x[1:, 0], x[:, 1] = csum[:-1], csum
        res["position_idx_range"] = x

        return res

    @property
    def data_shapes(self):
        return {
            "atomic_inputs": (-1, NB_ATOMIC_FEATURES),
            "position_idx_range": (-1, 2),
            "energies": (-1, len(self.__energy_methods__)),
            "forces": (-1, 3, len(self.force_target_names)),
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
        n_atoms_first = self.data["n_atoms_first"][idx]

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
            n_atoms_first=n_atoms_first,
        )

        if self.transform is not None:
            bunch = self.transform(bunch)

        return bunch

    def save_preprocess(self, data_dict):
        # save memmaps
        logger.info("Preprocessing data and saving it to cache.")
        for key in self.data_keys:
            local_path = p_join(self.preprocess_path, f"{key}.mmap")
            out = np.memmap(local_path, mode="w+", dtype=data_dict[key].dtype, shape=data_dict[key].shape)
            out[:] = data_dict.pop(key)[:]
            out.flush()
            push_remote(local_path, overwrite=True)

        # save all other keys in props.pkl
        local_path = p_join(self.preprocess_path, "props.pkl")
        for key in data_dict:
            if key not in self.data_keys:
                x = data_dict[key]
                x[x == None] = -1  # noqa
                data_dict[key] = np.unique(x, return_inverse=True)

        with open(local_path, "wb") as f:
            pkl.dump(data_dict, f)
        push_remote(local_path, overwrite=True)

    def read_preprocess(self, overwrite_local_cache=False):
        logger.info("Reading preprocessed data.")
        logger.info(
            f"Dataset {self.__name__} with the following units:\n\
                     Energy: {self.energy_unit},\n\
                     Distance: {self.distance_unit},\n\
                     Forces: {self.force_unit if self.__force_methods__ else 'None'}"
        )
        self.data = {}
        for key in self.data_keys:
            filename = p_join(self.preprocess_path, f"{key}.mmap")
            pull_locally(filename, overwrite=overwrite_local_cache)
            self.data[key] = np.memmap(filename, mode="r", dtype=self.data_types[key]).reshape(self.data_shapes[key])

        filename = p_join(self.preprocess_path, "props.pkl")
        pull_locally(filename, overwrite=overwrite_local_cache)
        with open(filename, "rb") as f:
            tmp = pkl.load(f)
            for key in set(tmp.keys()) - set(self.data_keys):
                x = tmp.pop(key)
                if len(x) == 2:
                    self.data[key] = x[0][x[1]]
                else:
                    self.data[key] = x

        for key in self.data:
            logger.info(f"Loaded {key} with shape {self.data[key].shape}, dtype {self.data[key].dtype}")

    def get_ase_atoms(self, idx: int):
        entry = self[idx]
        at = to_atoms(entry["positions"], entry["atomic_numbers"])
        at.info["n_atoms"] = entry["n_atoms_first"]
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
