import os
from os.path import join as p_join
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.utils import Bunch
from tqdm import tqdm

from openqdc.utils.constants import NB_ATOMIC_FEATURES
from openqdc.utils.io import (
    copy_exists,
    get_local_cache,
    load_hdf5_file,
    pull_locally,
    push_remote,
)
from openqdc.utils.molecule import atom_table
from openqdc.utils.units import get_conversion


def extract_entry(
    df: pd.DataFrame,
    i: int,
    subset: str,
    energy_target_names: List[str],
    force_target_names: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    x = np.array([atom_table.GetAtomicNumber(s) for s in df["symbols"][i]])
    xs = np.stack((x, np.zeros_like(x)), axis=-1)
    positions = df["geometry"][i].reshape((-1, 3))
    energies = np.array([df[k][i] for k in energy_target_names])

    res = dict(
        name=np.array([df["name"][i]]),
        subset=np.array([subset]),
        energies=energies.reshape((1, -1)).astype(np.float32),
        atomic_inputs=np.concatenate((xs, positions), axis=-1, dtype=np.float32),
        n_atoms=np.array([x.shape[0]], dtype=np.int32),
    )
    if force_target_names is not None and len(force_target_names) > 0:
        forces = np.zeros((positions.shape[0], 3, len(force_target_names)), dtype=np.float32)
        forces += np.nan
        for j, k in enumerate(force_target_names):
            if len(df[k][i]) != 0:
                forces[:, :, j] = df[k][i].reshape((-1, 3))
        res["forces"] = forces

    return res


def read_qc_archive_h5(
    raw_path: str, subset: str, energy_target_names: List[str], force_target_names: List[str]
) -> List[Dict[str, np.ndarray]]:
    data = load_hdf5_file(raw_path)
    data_t = {k2: data[k1][k2][:] for k1 in data.keys() for k2 in data[k1].keys()}
    n = len(data_t["molecule_id"])

    samples = [extract_entry(data_t, i, subset, energy_target_names, force_target_names) for i in tqdm(range(n))]
    return samples


class BaseDataset(torch.utils.data.Dataset):
    __energy_methods__ = []
    __force_methods__ = []
    energy_target_names = []
    force_target_names = []
    # convert force gradient -1

    __energy_unit__ = "hartree"
    __distance_unit__ = "bohr"
    __forces_unit__ = "hartree/bohr"
    __fn_energy__ = lambda x: x
    __fn_distance__ = lambda x: x
    __fn_forces__ = lambda x: x

    def __init__(self, energy_unit=None, distance_unit=None) -> None:
        self.data = None
        self._set_units(energy_unit, distance_unit)
        if not self.is_preprocessed():
            entries = self.read_raw_entries()
            res = self.collate_list(entries)
            self.save_preprocess(res)
        self.read_preprocess()

    @property
    def energy_unit(self):
        return self.__energy_unit__

    @property
    def distance_unit(self):
        return self.__distance_unit__

    @property
    def force_unit(self):
        return self.__forces_unit__

    @property
    def root(self):
        return p_join(get_local_cache(), self.__name__)

    @property
    def preprocess_path(self):
        path = p_join(self.root, "preprocessed")
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def data_keys(self):
        keys = list(self.data_types.keys())
        if len(self.__force_methods__) == 0:
            keys.remove("forces")
        return keys

    @property
    def data_types(self):
        return {
            "atomic_inputs": np.float32,
            "position_idx_range": np.int32,
            "energies": np.float32,
            "forces": np.float32,
        }

    @property
    def data_shapes(self):
        return {
            "atomic_inputs": (-1, NB_ATOMIC_FEATURES),
            "position_idx_range": (-1, 2),
            "energies": (-1, len(self.energy_target_names)),
            "forces": (-1, 3, len(self.force_target_names)),
        }

    def _set_units(self, en, ds):
        old_en, old_ds = self.energy_unit, self.distance_unit
        if en is not None:
            self.set_energy_unit(en)
        if ds is not None:
            self.set_distance_unit(ds)
        if self.__force_methods__:
            self.__forces_unit__ = self.energy_unit + "/" + self.distance_unit
            self.__class__.__fn_forces__ = get_conversion(old_en + "/" + old_ds, self.__forces_unit__)

    def convert_energy(self, x):
        return self.__class__.__fn_energy__(x)

    def convert_distance(self, x):
        return self.__class__.__fn_distance__(x)

    def convert_forces(self, x):
        return self.__class__.__fn_forces__(x)

    def set_energy_unit(self, value):
        old_unit = self.energy_unit
        self.__energy_unit__ = value
        self.__class__.__fn_energy__ = get_conversion(old_unit, value)

    def set_distance_unit(self, value):
        old_unit = self.distance_unit
        self.__distance_unit__ = value
        self.__class__.__fn_distance__ = get_conversion(old_unit, value)

    def read_raw_entries(self):
        raise NotImplementedError

    def collate_list(self, list_entries):
        # concatenate entries
        res = {key: np.concatenate([r[key] for r in list_entries if r is not None], axis=0) for key in list_entries[0]}

        csum = np.cumsum(res.pop("n_atoms"))
        x = np.zeros((csum.shape[0], 2), dtype=np.int32)
        x[1:, 0], x[:, 1] = csum[:-1], csum
        res["position_idx_range"] = x
        return res

    def save_preprocess(self, data_dict):
        # save memmaps
        logger.info("Preprocessing data and saving it to cache.")
        logger.info(
            f"Dataset {self.__name__} data with the following units:\n"
            f"Energy: {self.energy_unit}, Distance: {self.distance_unit}, "
            f"Forces: {self.force_unit if self.__force_methods__ else 'None'}"
        )
        for key in self.data_keys:
            local_path = p_join(self.preprocess_path, f"{key}.mmap")
            out = np.memmap(local_path, mode="w+", dtype=data_dict[key].dtype, shape=data_dict[key].shape)
            out[:] = data_dict.pop(key)[:]
            out.flush()
            push_remote(local_path)

        # save smiles and subset
        for key in ["name", "subset"]:
            local_path = p_join(self.preprocess_path, f"{key}.npz")
            uniques, inv_indices = np.unique(data_dict[key], return_inverse=True)
            with open(local_path, "wb") as f:
                np.savez_compressed(f, uniques=uniques, inv_indices=inv_indices)
            push_remote(local_path)

    def read_preprocess(self):
        logger.info("Reading preprocessed data")
        logger.info(
            f"{self.__name__} data with the following units:\
                     Energy: {self.energy_unit},\
                     Distance: {self.distance_unit},\
                     Forces: {self.force_unit}"
        )
        self.data = {}
        for key in self.data_keys:
            filename = p_join(self.preprocess_path, f"{key}.mmap")
            pull_locally(filename)
            self.data[key] = np.memmap(
                filename,
                mode="r",
                dtype=self.data_types[key],
            ).reshape(self.data_shapes[key])

        for key in self.data:
            print(f"Loaded {key} with shape {self.data[key].shape}, dtype {self.data[key].dtype}")

        for key in ["name", "subset"]:
            filename = p_join(self.preprocess_path, f"{key}.npz")
            pull_locally(filename)
            self.data[key] = dict()
            with open(filename, "rb") as f:
                tmp = np.load(f)
                for k in tmp:
                    self.data[key][k] = tmp[k]
                    print(f"Loaded {key}_{k} with shape {self.data[key][k].shape}, dtype {self.data[key][k].dtype}")

    def is_preprocessed(self):
        predicats = [copy_exists(p_join(self.preprocess_path, f"{key}.mmap")) for key in self.data_keys]
        predicats += [copy_exists(p_join(self.preprocess_path, f"{x}.npz")) for x in ["name", "subset"]]
        return all(predicats)

    def __len__(self):
        return self.data["energies"].shape[0]

    def __getitem__(self, idx: int):
        p_start, p_end = self.data["position_idx_range"][idx]
        input = self.data["atomic_inputs"][p_start:p_end]
        z, c, positions, energies = (
            np.array(input[:, 0], dtype=np.int32),
            np.array(input[:, 1], dtype=np.int32),
            self.convert_distance(np.array(input[:, -3:], dtype=np.float32)),
            self.convert_energy(np.array(self.data["energies"][idx], dtype=np.float32)),
        )
        name = self.data["name"]["uniques"][self.data["name"]["inv_indices"][idx]]
        subset = self.data["subset"]["uniques"][self.data["subset"]["inv_indices"][idx]]

        if "forces" in self.data:
            forces = self.convert_forces(np.array(self.data["forces"][p_start:p_end], dtype=np.float32))
        else:
            forces = None

        return Bunch(
            positions=positions,
            atomic_numbers=z,
            charges=c,
            e0=self.convert_energy(self.atomic_energies[z]),
            energies=energies,
            name=name,
            subset=subset,
            forces=forces,
        )
