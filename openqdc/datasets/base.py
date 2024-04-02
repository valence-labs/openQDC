"""The BaseDataset defining shared functionality between all datasets."""

import os
import pickle as pkl
from copy import deepcopy
from itertools import compress
from os.path import join as p_join
from typing import Dict, List, Optional, Union

import numpy as np
from ase.io.extxyz import write_extxyz
from loguru import logger
from sklearn.utils import Bunch

from openqdc.datasets.energies import AtomEnergies
from openqdc.datasets.properties import DatasetPropertyMixIn
from openqdc.datasets.statistics import (
    ForcesCalculatorStats,
    FormationEnergyStats,
    PerAtomFormationEnergyStats,
    StatisticManager,
    TotalEnergyStats,
)
from openqdc.utils.atomization_energies import IsolatedAtomEnergyFactory
from openqdc.utils.constants import NB_ATOMIC_FEATURES
from openqdc.utils.descriptors import get_descriptor
from openqdc.utils.exceptions import (
    DatasetNotAvailableError,
    StatisticsNotAvailableError,
)
from openqdc.utils.io import (
    copy_exists,
    dict_to_atoms,
    get_local_cache,
    pull_locally,
    push_remote,
    set_cache_dir,
)
from openqdc.utils.package_utils import requires_package
from openqdc.utils.regressor import Regressor  # noqa
from openqdc.utils.units import get_conversion


class BaseDataset(DatasetPropertyMixIn):
    """
    Base class for datasets in the openQDC package.
    """

    energy_target_names = []
    force_target_names = []
    __energy_methods__ = []
    __force_mask__ = []
    __isolated_atom_energies__ = []

    __energy_unit__ = "hartree"
    __distance_unit__ = "ang"
    __forces_unit__ = "hartree/ang"
    __fn_energy__ = lambda x: x
    __fn_distance__ = lambda x: x
    __fn_forces__ = lambda x: x
    __average_nb_atoms__ = None

    def __init__(
        self,
        energy_unit: Optional[str] = None,
        distance_unit: Optional[str] = None,
        energy_type: str = "formation",
        overwrite_local_cache: bool = False,
        cache_dir: Optional[str] = None,
        recompute_statistics: bool = False,
        regressor_kwargs={
            "solver_type": "linear",
            "sub_sample": None,
            "stride": 1,
        },
    ) -> None:
        """

        Parameters
        ----------
        energy_unit
            Energy unit to convert dataset to. Supported units: ["kcal/mol", "kj/mol", "hartree", "ev"]
        distance_unit
            Distance unit to convert dataset to. Supported units: ["ang", "nm", "bohr"]
        energy_type
            Type of isolated atom energy to use for the dataset. Default: "formation"
            Supported types: ["formation", "regression"]
        overwrite_local_cache
            Whether to overwrite the locally cached dataset.
        cache_dir
            Cache directory location. Defaults to "~/.cache/openqdc"
        recompute_statistics
            Whether to recompute the statistics of the dataset.
        regressor_kwargs
            Dictionary of keyword arguments to pass to the regressor.
            Default: {"solver_type": "linear", "sub_sample": None, "stride": 1}
            solver_type can be one of ["linear", "ridge"]
        """
        set_cache_dir(cache_dir)
        self.data = None
        self.recompute_statistics = recompute_statistics
        self.regressor_kwargs = regressor_kwargs
        self.energy_type = energy_type
        self.refit_e0s = recompute_statistics or overwrite_local_cache
        if not self.is_preprocessed():
            raise DatasetNotAvailableError(self.__name__)
        else:
            self.read_preprocess(overwrite_local_cache=overwrite_local_cache)
        self._post_init(overwrite_local_cache, energy_unit, distance_unit)

    def _post_init(
        self,
        overwrite_local_cache: bool = False,
        energy_unit: Optional[str] = None,
        distance_unit: Optional[str] = None,
    ) -> None:
        self._set_units(None, None)
        self._set_isolated_atom_energies()
        self._precompute_statistics(overwrite_local_cache=overwrite_local_cache)
        self._set_units(energy_unit, distance_unit)
        self._convert_data()
        self._set_isolated_atom_energies()

    def _precompute_statistics(self, overwrite_local_cache: bool = False):
        # if self.recompute_statistics or overwrite_local_cache:
        self.statistics = StatisticManager(
            self,
            self.recompute_statistics or overwrite_local_cache,
            ForcesCalculatorStats,
            TotalEnergyStats,
            FormationEnergyStats,
            PerAtomFormationEnergyStats,
        )
        self.statistics.run_calculators()

    @classmethod
    def no_init(cls):
        """
        Class method to avoid the __init__ method to be called when the class is instanciated.
        Useful for debugging purposes or preprocessing data.
        """
        return cls.__new__(cls)

    def _convert_data(self):
        logger.info(
            f"Converting {self.__name__} data to the following units:\n\
                     Energy: {self.energy_unit},\n\
                     Distance: {self.distance_unit},\n\
                     Forces: {self.force_unit if self.__force_methods__ else 'None'}"
        )
        for key in self.data_keys:
            self.data[key] = self._convert_on_loading(self.data[key], key)

    @property
    def __force_methods__(self):
        """
        For backward compatibility. To be removed in the future.
        """
        return self.force_methods

    @property
    def energy_methods(self):
        return self.__energy_methods__

    @property
    def force_methods(self):
        return list(compress(self.energy_methods, self.force_mask))

    @property
    def force_mask(self):
        if len(self.__class__.__force_mask__) == 0:
            self.__class__.__force_mask__ = [False] * len(self.energy_methods)
        return self.__class__.__force_mask__

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
        en = en if en is not None else old_en
        ds = ds if ds is not None else old_ds
        self.set_energy_unit(en)
        self.set_distance_unit(ds)
        if self.__force_methods__:
            self.__forces_unit__ = self.energy_unit + "/" + self.distance_unit
            self.__class__.__fn_forces__ = get_conversion(old_en + "/" + old_ds, self.__forces_unit__)

    def _set_isolated_atom_energies(self):
        if self.energy_methods is None:
            logger.error("No energy methods defined for this dataset.")
        f = get_conversion("hartree", self.__energy_unit__)
        self.__isolated_atom_energies__ = f(self.e0s_dispatcher.e0s_matrix)

    @property
    def e0s_dispatcher(self):
        if not hasattr(self, "_e0s_dispatcher"):
            self._e0s_dispatcher = AtomEnergies(self, **self.regressor_kwargs)
        return self._e0s_dispatcher

    def convert_energy(self, x):
        return self.__class__.__fn_energy__(x)

    def convert_distance(self, x):
        return self.__class__.__fn_distance__(x)

    def convert_forces(self, x):
        return self.__class__.__fn_forces__(x)

    def set_energy_unit(self, value: str):
        """
        Set a new energy unit for the dataset.
        """
        old_unit = self.energy_unit
        self.__energy_unit__ = value
        self.__class__.__fn_energy__ = get_conversion(old_unit, value)

    def set_distance_unit(self, value: str):
        """
        Set a new distance unit for the dataset.
        """
        old_unit = self.distance_unit
        self.__distance_unit__ = value
        self.__class__.__fn_distance__ = get_conversion(old_unit, value)

    def as_iter(self, atoms: bool = False):
        """
        Return the dataset as an iterator.

        Parameters
        ----------
        atoms : bool, optional
            Whether to return the items as ASE atoms object, by default False
        """
        func = self.get_ase_atoms if atoms else self.__getitem__
        for i in range(len(self)):
            yield func(i)

    def read_raw_entries(self):
        raise NotImplementedError

    def collate_list(self, list_entries):
        # concatenate entries
        res = {key: np.concatenate([r[key] for r in list_entries if r is not None], axis=0) for key in list_entries[0]}

        csum = np.cumsum(res.get("n_atoms"))
        x = np.zeros((csum.shape[0], 2), dtype=np.int32)
        x[1:, 0], x[:, 1] = csum[:-1], csum
        res["position_idx_range"] = x

        return res

    def save_preprocess(self, data_dict):
        # save memmaps
        logger.info("Preprocessing data and saving it to cache.")
        for key in self.data_keys:
            local_path = p_join(self.preprocess_path, f"{key}.mmap")
            out = np.memmap(local_path, mode="w+", dtype=data_dict[key].dtype, shape=data_dict[key].shape)
            out[:] = data_dict.pop(key)[:]
            out.flush()
            push_remote(local_path, overwrite=True)

        # save smiles and subset
        local_path = p_join(self.preprocess_path, "props.pkl")
        for key in ["name", "subset"]:
            data_dict[key] = np.unique(data_dict[key], return_inverse=True)

        with open(local_path, "wb") as f:
            pkl.dump(data_dict, f)
        push_remote(local_path, overwrite=True)

    def _convert_on_loading(self, x, key):
        if key == "energies":
            return self.convert_energy(x)
        elif key == "forces":
            return self.convert_forces(x)
        elif key == "atomic_inputs":
            x = np.array(x, dtype=np.float32)
            x[:, -3:] = self.convert_distance(x[:, -3:])
            return x
        else:
            return x

    def read_preprocess(self, overwrite_local_cache=False):
        logger.info("Reading preprocessed data.")
        logger.info(
            f"Dataset {self.__name__} with the following units:\n\
                     Energy: {self.energy_unit},\n\
                     Distance: {self.distance_unit},\n\
                     Forces: {self.force_unit if self.force_methods else 'None'}"
        )
        self.data = {}
        for key in self.data_keys:
            filename = p_join(self.preprocess_path, f"{key}.mmap")
            pull_locally(filename, overwrite=overwrite_local_cache)
            self.data[key] = np.memmap(filename, mode="r", dtype=self.data_types[key]).reshape(*self.data_shapes[key])

        filename = p_join(self.preprocess_path, "props.pkl")
        pull_locally(filename, overwrite=overwrite_local_cache)
        with open(filename, "rb") as f:
            tmp = pkl.load(f)
            for key in ["name", "subset", "n_atoms"]:
                x = tmp.pop(key)
                if len(x) == 2:
                    self.data[key] = x[0][x[1]]
                else:
                    self.data[key] = x

        for key in self.data:
            logger.info(f"Loaded {key} with shape {self.data[key].shape}, dtype {self.data[key].dtype}")

    def is_preprocessed(self):
        predicats = [copy_exists(p_join(self.preprocess_path, f"{key}.mmap")) for key in self.data_keys]
        predicats += [copy_exists(p_join(self.preprocess_path, "props.pkl"))]
        return all(predicats)

    def is_cached(self):
        """
        Check if the dataset is cached locally.
        """
        predicats = [os.path.exists(p_join(self.preprocess_path, f"{key}.mmap")) for key in self.data_keys]
        predicats += [os.path.exists(p_join(self.preprocess_path, "props.pkl"))]
        return all(predicats)

    def preprocess(self, overwrite=False):
        if overwrite or not self.is_preprocessed():
            entries = self.read_raw_entries()
            res = self.collate_list(entries)
            self.save_preprocess(res)

    def save_xyz(self, idx: int, path: Optional[str] = None, ext=True):
        """
        Save the entry at index idx as an extxyz file.
        """
        if path is None:
            path = os.getcwd()
        at = self.get_ase_atoms(idx, ext=ext)
        write_extxyz(p_join(path, f"mol_{idx}.xyz"), at)

    def get_ase_atoms(self, idx: int, ext=True):
        """
        Get the ASE atoms object for the entry at index idx.

        Parameters
        ----------
        idx : int
            Index of the entry.
        ext : bool, optional
            Whether to include additional informations
        """
        entry = self[idx]
        at = dict_to_atoms(entry, ext=ext)
        return at

    def subsample(self, n_samples: Optional[Union[List[int], int]] = None):
        if n_samples is None:
            idxs = list(range(len(self)))
        elif isinstance(n_samples, int):
            idxs = np.random.choice(len(self), size=n_samples, replace=False)
        else:  # list, set, np.ndarray
            idxs = n_samples
        return idxs

    @requires_package("datamol")
    def calculate_descriptors(
        self,
        model: str = "soap",
        chemical_species: Optional[List[str]] = None,
        n_samples: Optional[Union[List[int], int]] = None,
        progress: bool = True,
        **descriptor_kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Compute the descriptors for the dataset.

        Parameters
        ----------
        model : str
            Name of the descriptor to use. Supported descriptors are ["soap"]
        chemical_species : Optional[List[str]], optional
            List of chemical species to use for the descriptor computation, by default None.
            If None, the chemical species of the dataset are used.
        n_samples : Optional[Union[List[int],int]], optional
            Number of samples to use for the computation, by default None. If None, all the dataset is used.
            If a list of integers is provided, the descriptors are computed for each of the specified idx of samples.
        progress : bool, optional
            Whether to show a progress bar, by default True.
        **descriptor_kwargs : dict
            Keyword arguments to pass to the descriptor instantiation of the model.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing the following keys:
                - values : np.ndarray of shape (N, M) containing the SOAP descriptors for the dataset
                - idxs : np.ndarray of shape (N,) containing the indices of the samples used

        """
        import datamol as dm

        idxs = self.subsample(n_samples)
        model = get_descriptor(model.lower())(
            species=self.chemical_species if chemical_species is None else chemical_species, **descriptor_kwargs
        )

        def wrapper(idx):
            entry = self.get_ase_atoms(idx, ext=False)
            return model.calculate(entry)

        descr_values = dm.parallelized(wrapper, idxs, progress=progress, scheduler="threads", n_jobs=-1)
        return {"values": np.vstack(descr_values), "idxs": idxs}

    def get_statistics(self, return_none: bool = True):
        """
        Get the statistics of the dataset.
        normalization : str, optional
            Type of energy, by default "formation", must be one of ["formation", "total",
            "residual_regression", "per_atom_formation", "per_atom_residual_regression"]
        return_none : bool, optional
            Whether to return None if the statistics for the forces are not available, by default True
            Otherwise, the statistics for the forces are set to 0.0
        """
        stats = deepcopy(self.statistics.get_results())
        if len(stats) == 0:
            raise StatisticsNotAvailableError(self.__name__)
        selected_stats = stats
        if not return_none:
            selected_stats.update(
                {
                    "ForceStatistics": {
                        "mean": np.array([0.0]),
                        "std": np.array([0.0]),
                        "components": {
                            "mean": np.array([[0.0], [0.0], [0.0]]),
                            "std": np.array([[0.0], [0.0], [0.0]]),
                            "rms": np.array([[0.0], [0.0], [0.0]]),
                        },
                    }
                }
            )
        # cycle trough dict to convert units
        for key in selected_stats:
            if key == "forces":
                for key2 in selected_stats[key]:
                    if key2 != "components":
                        selected_stats[key][key2] = self.convert_forces(selected_stats[key][key2])
                    else:
                        for key2 in selected_stats[key]["components"]:
                            selected_stats[key]["components"][key2] = self.convert_forces(
                                selected_stats[key]["components"][key2]
                            )
            else:
                for key2 in selected_stats[key]:
                    selected_stats[key][key2] = self.convert_energy(selected_stats[key][key2])
        return selected_stats

    def __str__(self):
        return f"{self.__name__}"

    def __repr__(self):
        return f"{self.__name__}"

    def __len__(self):
        return self.data["energies"].shape[0]

    def __smiles_converter__(self, x):
        """util function to convert string to smiles: useful if the smiles is
        encoded in a different format than its display format
        """
        return x

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
        e0s = self.__isolated_atom_energies__[..., z, c + shift].T
        formation_energies = energies - e0s.sum(axis=0)
        if "forces" in self.data:
            forces = np.array(self.data["forces"][p_start:p_end], dtype=np.float32)
        else:
            forces = None
        return Bunch(
            positions=positions,
            atomic_numbers=z,
            charges=c,
            e0=e0s,
            energies=energies,
            formation_energies=formation_energies,
            per_atom_formation_energies=formation_energies / len(z),
            name=name,
            subset=subset,
            forces=forces,
        )
