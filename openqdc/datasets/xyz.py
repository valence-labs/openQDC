from openqdc.datasets.base import BaseDataset
from abc import ABC, abstractmethod
from tqdm import tqdm
from ase.atoms import Atoms
import numpy as np 
import datamol as dm 
from typing import List
def try_retrieve(obj, callable, default=None):
    try:
        return callable(obj)
    except Exception:
        return default

class FromFileDataset(ABC):
    @classmethod
    def __init__(self,
                 path : str,
                 energy_unit: str,
                 distance_unit: str,
                 level_of_theory: str):
        """
        Create a dataset from a xyz file.
        
        Parameters
        ----------
        file_path : str
            The path to the file.
        """
        raise NotImplementedError
    
    def __str__(self):
        return str(self.__class__.__name__).lower()
    
    def __repr__(self):
        return str(self)
    
    def collate_list(self, list_entries):
        # concatenate entries
        res = {key: np.concatenate([r[key] for r in list_entries if r is not None], axis=0) for key in list_entries[0]}

        csum = np.cumsum(res.get("n_atoms"))
        x = np.zeros((csum.shape[0], 2), dtype=np.int32)
        x[1:, 0], x[:, 1] = csum[:-1], csum
        res["position_idx_range"] = x

        return res
    
    def _convert_to_record(self, obj : Atoms):
        name = obj.info.get("name", None)
        subset = obj.info.get("subset", str(self))
        positions = obj.positions
        energies = try_retrieve(obj, lambda x: x.get_potential_energy(), np.nan)
        forces = try_retrieve(obj, lambda x: x.get_forces(), None)
        fall_back_charges = np.zeros(len(positions)) if name else dm.to_mol(name, remove_hs=False, ordered=True)
        charges = try_retrieve(obj, lambda x: x.get_initial_charges(), fall_back_charges)
        return dict(
            name=np.array([name]) if name else np.array([str(obj.symbols)]),
            subset=np.array([subset]),
            energies=np.array([energies], dtype=np.float32),
            forces=forces.reshape(-1, 3, 1).astype(np.float32) if forces is not None else None,
            atomic_inputs=np.concatenate((charges[:,None], positions), axis=-1, dtype=np.float32),
            n_atoms=np.array([len(positions)], dtype=np.int32),
        )
    
class XYZDataset(FromFileDataset):
    
    def __init__(self,
                 path : List[str],
                 energy_unit: str,
                 distance_unit: str,
                 level_of_theory: str):
        """
        Create a dataset from a xyz file.
        
        Parameters
        ----------
        file_path : str
            The path to the file.
        """
        self.path = path
        entries= self.read_raw_entries()
        self.data= self.collate_list(entries)
        
        
    def read_raw_entries(self):
        import numpy as np
        from ase.io import iread
        entries_list = []
        for entry in iread(self.path, format="extxyz"):
            entries_list.append(self._convert_to_record(entry))
        return entries_list
            
        
        