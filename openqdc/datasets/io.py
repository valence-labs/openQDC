from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

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
    """Abstract class for datasets that read from a common format file like xzy, netcdf, gro, hdf5, etc."""

    def __init__(
        self,
        path: List[str],
        *,
        dataset_name: Optional[str] = None,
        energy_type: Optional[str] = "regression",
        energy_unit: Optional[str] = "hartree",
        distance_unit: Optional[str] = "ang",
        array_format: Optional[str] = "numpy",
        level_of_theory: Optional[QmMethod] = None,
        transform: Optional[Callable] = None,
        skip_statistics: bool = False,
        regressor_kwargs={
            "solver_type": "linear",
            "sub_sample": None,
            "stride": 1,
        },
    ):
        """
        Create a dataset from a list of files.

        Parameters
        ----------
        path : List[str]
            The path to the file or a list of paths.
        dataset_name : Optional[str], optional
            The name of the dataset, by default None.
        energy_type : Optional[str], optional
            The type of isolated atom energy by default "regression".
            Supported types: ["formation", "regression", "null", None]
        energy_unit
            Energy unit of the dataset. Default is "hartree".
        distance_unit
            Distance unit of the dataset. Default is "ang".
        level_of_theory: Optional[QmMethod, str]
            The level of theory of the dataset.
            Used if energy_type is "formation" to fetch the correct isolated atom energies.
        transform, optional
            transformation to apply to the __getitem__ calls
        regressor_kwargs
            Dictionary of keyword arguments to pass to the regressor.
            Default: {"solver_type": "linear", "sub_sample": None, "stride": 1}
            solver_type can be one of ["linear", "ridge"]
        """
        self.path = [path] if isinstance(path, str) else path
        self.__name__ = self.__class__.__name__ if dataset_name is None else dataset_name
        self.recompute_statistics = True
        self.refit_e0s = True
        self.energy_type = energy_type
        self.skip_statistics = skip_statistics
        self.__energy_unit__ = energy_unit
        self._original_unit = self.energy_unit
        self.__distance_unit__ = distance_unit
        self.__energy_methods__ = [PotentialMethod.NONE if not level_of_theory else level_of_theory]
        self.energy_target_names = ["xyz"]
        self.regressor_kwargs = regressor_kwargs
        self.transform = transform
        self._read_and_preprocess()
        if "forces" in self.data:
            self.__force_mask__ = [True]
            self.__class__.__force_methods__ = [level_of_theory]
            self.force_target_names = ["xyz"]
        self.set_array_format(array_format)
        self._post_init(True, energy_unit, distance_unit)

    @abstractmethod
    def read_as_atoms(self, path: str) -> List[Atoms]:
        """
        Method that reads a file and return a list of Atoms objects.
        path : str
            The path to the file.
        """
        raise NotImplementedError

    def read_raw_entries(self) -> List[Dict]:
        """
        Process the files and return a list of data objects.
        """
        entries_list = []
        for path in self.path:
            for entry in self.read_as_atoms(path):
                entries_list.append(self._convert_to_record(entry))
        return entries_list

    def _read_and_preprocess(self):
        entries_list = self.read_raw_entries()
        self.data = self.collate_list(entries_list)

    def _convert_to_record(self, obj: Atoms):
        """
        Convert an Atoms object to a record for the openQDC dataset processing.
        obj : Atoms
            The ase.Atoms object to convert
        """
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

    def __str__(self):
        return self.__name__.lower()

    def __repr__(self):
        return str(self)


class XYZDataset(FromFileDataset):
    """
    Baseclass to read datasets from xyz and extxyz files.
    """

    def read_as_atoms(self, path):
        from ase.io import iread

        return iread(path, format="extxyz")
