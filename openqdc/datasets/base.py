import os
from os.path import join as p_join

import numpy as np
import torch
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


def extract_entry(df, i, subset, energy_target_names, force_target_names=None):
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


def read_qc_archive_h5(raw_path, subset, energy_target_names, force_target_names):
    data = load_hdf5_file(raw_path)
    data_t = {k2: data[k1][k2][:] for k1 in data.keys() for k2 in data[k1].keys()}
    n = len(data_t["molecule_id"])
    # print(f"Reading {n} entries from {raw_path}")
    # for k in data_t:
    #     print(f"Loaded {k} with shape {data_t[k].shape}, dtype {data_t[k].dtype}")
    #     if "Energy" in k:
    #         print(np.isnan(data_t[k]).mean(), f"{data_t[k][0]}")

    # print('\n'*3)
    # exit()

    samples = [extract_entry(data_t, i, subset, energy_target_names, force_target_names) for i in tqdm(range(n))]
    return samples


class BaseDataset(torch.utils.data.Dataset):
    __energy_methods__ = []
    __force_methods__ = []
    energy_target_names = []
    force_target_names = []

    energy_unit = "hartree"

    def __init__(self) -> None:
        self.data = None
        if not self.is_preprocessed():
            entries = self.read_raw_entries()
            res = self.collate_list(entries)
            self.save_preprocess(res)
        self.read_preprocess()

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
            # with open(filename, "rb") as f:
            self.data[key] = np.load(open(filename, "rb"))
            for k in self.data[key]:
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
        z, c, positions = input[:, 0], input[:, 1], input[:, -3:]
        z, c = z.astype(np.int32), c.astype(np.int32)
        energies = self.data["energies"][idx]
        name = self.data["name"]["uniques"][self.data["name"]["inv_indices"][idx]]
        subset = self.data["subset"]["uniques"][self.data["subset"]["inv_indices"][idx]]

        if "forces" in self.data:
            forces = self.data["forces"][p_start:p_end]
        else:
            forces = None

        return Bunch(
            positions=positions,
            atomic_numbers=z,
            charges=c,
            e0=self.atomic_energies[z],
            energies=energies,
            name=name,
            subset=subset,
            forces=forces,
        )
