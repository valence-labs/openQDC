from typing import Dict, List, Optional

import numpy as np
from sklearn.utils import Bunch

from openqdc.datasets.base import BaseDataset
from openqdc.utils.atomization_energies import IsolatedAtomEnergyFactory
from openqdc.utils.constants import NB_ATOMIC_FEATURES


class BaseInteractionDataset(BaseDataset):
    def __init__(
        self,
        energy_unit: Optional[str] = None,
        distance_unit: Optional[str] = None,
        overwrite_local_cache: bool = False,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__(
            energy_unit=energy_unit,
            distance_unit=distance_unit,
            overwrite_local_cache=overwrite_local_cache,
            cache_dir=cache_dir,
        )

    def collate_list(self, list_entries: List[Dict]):
        # concatenate entries
        print(list_entries[0])
        res = {
            key: np.concatenate([r[key] for r in list_entries if r is not None], axis=0)
            for key in list_entries[0]
            if not isinstance(list_entries[0][key], dict)
        }

        csum = np.cumsum(res.get("n_atoms"))
        print(csum)
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
            "n_atoms_first": (-1,),
        }

    @property
    def data_types(self):
        return {
            "atomic_inputs": np.float32,
            "position_idx_range": np.int32,
            "energies": np.float32,
            "forces": np.float32,
            "n_atoms_first": np.int32,
        }

    def __getitem__(self, idx: int):
        shift = IsolatedAtomEnergyFactory.max_charge
        p_start, p_end = self.data["position_idx_range"][idx]
        input = self.data["atomic_inputs"][p_start:p_end]
        z, c, positions, energies = (
            np.array(input[:, 0], dtype=np.int32),
            np.array(input[:, 1], dtype=np.int32),
            np.array(input[:, -3:], dtype=np.float32),
            np.array(self.data["energies"][idx], dtype=np.float32),
        )
        name = self.__smiles_converter__(self.data["name"][idx])
        subset = self.data["subset"][idx]
        n_atoms_first = self.data["n_atoms_first"][idx]

        if "forces" in self.data:
            forces = np.array(self.data["forces"][p_start:p_end], dtype=np.float32)
        else:
            forces = None
        return Bunch(
            positions=positions,
            atomic_numbers=z,
            charges=c,
            e0=self.__isolated_atom_energies__[..., z, c + shift].T,
            energies=energies,
            name=name,
            subset=subset,
            forces=forces,
            n_atoms_first=n_atoms_first,
        )
