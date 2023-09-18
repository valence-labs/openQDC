import os
import torch
import numpy as np
import pickle as pkl
from os.path import join as p_join
from sklearn.utils import Bunch
from openqdc.utils.paths import get_local_cache
from openqdc.utils.constants import BOHR2ANG, MAX_ATOMIC_NUMBER


class BaseDataset(torch.utils.data.Dataset):
    __qm_methods__ = []

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
        path = p_join(self.root, 'preprocessed')
        os.makedirs(path, exist_ok=True)
        return path
    
    @property
    def data_types(self):
        return {
            "atom_data_and_positions": np.float32, 
            "position_idx_range": np.int32, 
            "energies": np.float32,
            "forces": np.float32
        }
    
    @property
    def data_shapes(self):
        return {
            "atom_data_and_positions": (-1, 5), 
            "position_idx_range": (-1, 2), 
            "energies": (-1, len(self.energy_target_names)),
            "forces": (-1, 3, len(self.force_target_names))
        }
    
    def read_raw_entries(self):
        raise NotImplementedError
    
    def collate_list(self, list_entries):
        # concatenate entries
        res = {key: np.concatenate([r[key] for r in list_entries if r is not None], axis=0) 
               for key in list_entries[0]}

        csum = np.cumsum(res.pop("n_atoms"))
        x = np.zeros((csum.shape[0], 2), dtype=np.int32)
        x[1:, 0], x[:, 1] = csum[:-1], csum
        res["position_idx_range"] = x
        return res

    def save_preprocess(self, data_dict):
        # save memmaps
        for key in self.data_types:
            if key not in data_dict:
                continue
            out = np.memmap(p_join(self.preprocess_path, f"{key}.mmap"), 
                            mode="w+", 
                            dtype=data_dict[key].dtype, 
                            shape=data_dict[key].shape)
            out[:] = data_dict.pop(key)[:]
            out.flush()

        # save smiles and subset
        for key in ["smiles", "subset"]:
            uniques, inv_indices = np.unique(data_dict[key], return_inverse=True)
            with open(p_join(self.preprocess_path, f"{key}.npz"), "wb") as f:
                np.savez_compressed(f, uniques=uniques, inv_indices=inv_indices)
    
    def read_preprocess(self):        
        self.data = {}
        for key in self.data_types:
            filename = p_join(self.preprocess_path, f"{key}.mmap")
            if not os.path.exists(filename):
                continue
            self.data[key] = np.memmap(
                filename, mode='r', 
                dtype=self.data_types[key],
            ).reshape(self.data_shapes[key])
            
        for key in self.data:
            print(f'Loaded {key} with shape {self.data[key].shape}, dtype {self.data[key].dtype}')

        for key in ["smiles", "subset"]:
            filename = p_join(self.preprocess_path, f"{key}.npz")
            # with open(filename, "rb") as f:
            self.data[key] = np.load(open(filename, "rb"))
            for k in self.data[key]:
                print(f'Loaded {key}_{k} with shape {self.data[key][k].shape}, dtype {self.data[key][k].dtype}')

    def is_preprocessed(self):
        filenames = [p_join(self.preprocess_path, f"{key}.mmap") 
                     for key in self.data_types]
        filenames += [p_join(self.preprocess_path, f"{x}.npz") 
                      for x in ["smiles", "subset"]]
        return all([os.path.exists(f) for f in filenames])

    def __len__(self):
        return self.data['energies'].shape[0]

    def __getitem__(self, idx: int):
        p_start, p_end = self.data["position_idx_range"][idx]
        input = self.data["atom_data_and_positions"][p_start:p_end]
        z, positions = input[:, 0].astype(np.int32), input[:, 1:]
        energies = self.data["energies"][idx]
        e0 = self.atomic_energies[z]
        smiles = self.data["smiles"]["uniques"][self.data["smiles"]["inv_indices"][idx]]
        subset = self.data["smiles"]["uniques"][self.data["subset"]["inv_indices"][idx]]

        if "forces" in self.data:
            forces = self.data["forces"][p_start:p_end]
        else:
            forces = None

        return Bunch(
            positions=positions,
            atomic_numbers=z,
            e0=e0,
            energies=energies,
            smiles=smiles,
            subset=subset,
            forces=forces
        )
