from abc import ABC, abstractmethod
from typing import List, Optional

import datamol as dm
import numpy as np
from ase.atoms import Atoms

from openqdc.datasets.base import BaseDataset
from openqdc.methods import PotentialMethod, QmMethod


def try_retrieve(obj, callable, default=None):
    try:
        return callable(obj)
    except Exception:
        return default


class FromFileDataset(BaseDataset, ABC):
    def __init__(
        self,
        path: List[str],
        *,
        dataset_name: Optional[str] = None,
        energy_unit: Optional[str] = "hartree",
        distance_unit: Optional[str] = "ang",
        level_of_theory: Optional[QmMethod] = None,
        regressor_kwargs={
            "solver_type": "linear",
            "sub_sample": None,
            "stride": 1,
        },
    ):
        """
        Create a dataset from a xyz file.

        Parameters
        ----------
        path : List[str]
            The path to the file or a list of paths.
        """
        self.path = [path] if isinstance(path, str) else path
        self.__name__ = self.__class__.__name__ if dataset_name is None else dataset_name
        self.__energy_unit__ = energy_unit
        self.__distance_unit__ = distance_unit
        self.__energy_methods__ = [PotentialMethod.NONE if not level_of_theory else level_of_theory]
        self.regressor_kwargs = regressor_kwargs
        self._read_and_preprocess()
        self._post_init(True, energy_unit, distance_unit)

    def __str__(self):
        return self.__name__.lower()

    def __repr__(self):
        return str(self)

    @abstractmethod
    def read_as_atoms(self, path: str) -> List[Atoms]:
        """
        Method that reads a path and return a list of Atoms objects.
        """
        raise NotImplementedError

    def collate_list(self, list_entries):
        res = {key: np.concatenate([r[key] for r in list_entries if r is not None], axis=0) for key in list_entries[0]}
        csum = np.cumsum(res.get("n_atoms"))
        x = np.zeros((csum.shape[0], 2), dtype=np.int32)
        x[1:, 0], x[:, 1] = csum[:-1], csum
        res["position_idx_range"] = x

        return res

    def read_raw_entries(self):
        entries_list = []
        for path in self.path:
            for entry in self.read_as_atoms(path):
                entries_list.append(self._convert_to_record(entry))
        return entries_list

    def _read_and_preprocess(self):
        entries_list = self.read_raw_entries()
        self.data = self.collate_list(entries_list)

    def _convert_to_record(self, obj: Atoms):
        name = obj.info.get("name", None)
        subset = obj.info.get("subset", str(self))
        positions = obj.positions
        energies = try_retrieve(obj, lambda x: x.get_potential_energy(), np.nan)
        forces = try_retrieve(obj, lambda x: x.get_forces(), None)
        if forces is not None:
            self.__force_mask__ = [True]
        fall_back_charges = np.zeros(len(positions)) if not name else dm.to_mol(name, remove_hs=False, ordered=True)
        charges = try_retrieve(obj, lambda x: x.get_initial_charges(), fall_back_charges)
        return dict(
            name=np.array([name]) if name else np.array([str(obj.symbols)]),
            subset=np.array([subset]),
            energies=np.array([[energies]], dtype=np.float32),
            forces=forces.reshape(-1, 3, 1).astype(np.float32) if forces is not None else None,
            atomic_inputs=np.concatenate(
                (obj.numbers[:, None], charges[:, None], positions), axis=-1, dtype=np.float32
            ),
            n_atoms=np.array([len(positions)], dtype=np.int32),
        )


class XYZDataset(FromFileDataset):
    def read_as_atoms(self, path):
        from ase.io import iread

        return iread(path, format="extxyz")
