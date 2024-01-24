import os
import pickle as pkl
from copy import deepcopy
from os.path import join as p_join
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from ase.io.extxyz import write_extxyz
from loguru import logger
from sklearn.utils import Bunch
from tqdm import tqdm

from openqdc.utils.atomization_energies import (
    IsolatedAtomEnergyFactory,
    chemical_symbols,
)
from openqdc.utils.constants import (
    NB_ATOMIC_FEATURES,
    NOT_DEFINED,
    POSSIBLE_NORMALIZATION,
)
from openqdc.utils.exceptions import (
    DatasetNotAvailableError,
    NormalizationNotAvailableError,
    StatisticsNotAvailableError,
)
from openqdc.utils.io import (
    copy_exists,
    dict_to_atoms,
    get_local_cache,
    load_hdf5_file,
    load_pkl,
    pull_locally,
    push_remote,
    set_cache_dir,
)
from openqdc.utils.molecule import atom_table, z_to_formula
from openqdc.utils.package_utils import requires_package
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
        subset=np.array([subset if subset is not None else z_to_formula(x)]),
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
    __isolated_atom_energies__ = []

    __energy_unit__ = "hartree"
    __distance_unit__ = "ang"
    __forces_unit__ = "hartree/ang"
    __fn_energy__ = lambda x: x
    __fn_distance__ = lambda x: x
    __fn_forces__ = lambda x: x
    __average_nb_atoms__ = None
    __stats__ = {}

    def __init__(
        self,
        energy_unit: Optional[str] = None,
        distance_unit: Optional[str] = None,
        overwrite_local_cache: bool = False,
        cache_dir: Optional[str] = None,
    ) -> None:
        set_cache_dir(cache_dir)
        self.data = None
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

    def _convert_data(self):
        logger.info(
            f"Converting {self.__name__} data to the following units:\n\
                     Energy: {self.energy_unit},\n\
                     Distance: {self.distance_unit},\n\
                     Forces: {self.force_unit if self.__force_methods__ else 'None'}"
        )
        for key in self.data_keys:
            self.data[key] = self._convert_on_loading(self.data[key], key)

    def _precompute_statistics(self, overwrite_local_cache: bool = False):
        local_path = p_join(self.preprocess_path, "stats.pkl")
        if self.is_preprocessed_statistics() and not overwrite_local_cache:
            stats = load_pkl(local_path)
            logger.info("Loaded precomputed statistics")
        else:
            logger.info("Precomputing relevant statistics")
            (
                inter_E_mean,
                inter_E_std,
                formation_E_mean,
                formation_E_std,
                total_E_mean,
                total_E_std,
            ) = self._precompute_E()
            forces_dict = self._precompute_F()
            stats = {
                "formation": {"energy": {"mean": formation_E_mean, "std": formation_E_std}, "forces": forces_dict},
                "inter": {"energy": {"mean": inter_E_mean, "std": inter_E_std}, "forces": forces_dict},
                "total": {"energy": {"mean": total_E_mean, "std": total_E_std}, "forces": forces_dict},
            }
            with open(local_path, "wb") as f:
                pkl.dump(stats, f)
        self._compute_average_nb_atoms()
        self.__stats__ = stats

    def _compute_average_nb_atoms(self):
        self.__average_nb_atoms__ = np.mean(self.data["n_atoms"])

    def _precompute_E(self):
        splits_idx = self.data["position_idx_range"][:, 1]
        s = np.array(self.data["atomic_inputs"][:, :2], dtype=int)
        s[:, 1] += IsolatedAtomEnergyFactory.max_charge
        matrixs = [matrix[s[:, 0], s[:, 1]] for matrix in self.__isolated_atom_energies__]
        converted_energy_data = self.data["energies"]
        # calculation per molecule formation energy statistics
        E = []
        for i, matrix in enumerate(matrixs):
            c = np.cumsum(np.append([0], matrix))[splits_idx]
            c[1:] = c[1:] - c[:-1]
            E.append(converted_energy_data[:, i] - c)
        E = np.array(E).T
        inter_E_mean = np.nanmean(E / self.data["n_atoms"][:, None], axis=0)
        inter_E_std = np.nanstd(E / self.data["n_atoms"][:, None], axis=0)
        formation_E_mean = np.nanmean(E, axis=0)
        formation_E_std = np.nanstd(E, axis=0)
        total_E_mean = np.nanmean(converted_energy_data, axis=0)
        total_E_std = np.nanstd(converted_energy_data, axis=0)

        return (
            np.atleast_2d(inter_E_mean),
            np.atleast_2d(inter_E_std),
            np.atleast_2d(formation_E_mean),
            np.atleast_2d(formation_E_std),
            np.atleast_2d(total_E_mean),
            np.atleast_2d(total_E_std),
        )

    def _precompute_F(self):
        if len(self.__force_methods__) == 0:
            return NOT_DEFINED
        converted_force_data = self.convert_forces(self.data["forces"])
        force_mean = np.nanmean(converted_force_data, axis=0)
        force_std = np.nanstd(converted_force_data, axis=0)
        force_rms = np.sqrt(np.nanmean(converted_force_data**2, axis=0))
        return {
            "mean": np.atleast_2d(force_mean.mean(axis=0)),
            "std": np.atleast_2d(force_std.mean(axis=0)),
            "components": {"rms": force_rms, "std": force_std, "mean": force_mean},
        }

    @property
    def numbers(self):
        if hasattr(self, "_numbers"):
            return self._numbers
        self._numbers = pd.unique(self.data["atomic_inputs"][..., 0]).astype(np.int32)
        return self._numbers

    @property
    def chemical_species(self):
        return np.array(chemical_symbols)[self.numbers]

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

    @property
    def atoms_per_molecules(self):
        try:
            if hasattr(self, "_n_atoms"):
                return self._n_atoms
            self._n_atoms = self.data["n_atoms"]
            return self._n_atoms
        except:  # noqa
            return None

    def _set_units(self, en, ds):
        old_en, old_ds = self.energy_unit, self.distance_unit
        en = en if en is not None else old_en
        ds = ds if ds is not None else old_ds

        # if en is None:
        self.set_energy_unit(en)
        # if ds is not None:
        self.set_distance_unit(ds)
        if self.__force_methods__:
            self.__forces_unit__ = self.energy_unit + "/" + self.distance_unit
            self.__class__.__fn_forces__ = get_conversion(old_en + "/" + old_ds, self.__forces_unit__)

    def _set_isolated_atom_energies(self):
        if self.__energy_methods__ is None:
            logger.error("No energy methods defined for this dataset.")
        f = get_conversion("hartree", self.__energy_unit__)
        self.__isolated_atom_energies__ = f(
            np.array([IsolatedAtomEnergyFactory.get_matrix(en_method) for en_method in self.__energy_methods__])
        )

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
        logger.info("Reading preprocessed data")
        logger.info(
            f"{self.__name__} data with the following units:\n\
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

    def is_preprocessed_statistics(self):
        return bool(copy_exists(p_join(self.preprocess_path, "stats.pkl")))

    def preprocess(self, overwrite=False):
        if overwrite or not self.is_preprocessed():
            entries = self.read_raw_entries()
            res = self.collate_list(entries)
            self.save_preprocess(res)

    def save_xyz(self, idx: int, path: Optional[str] = None, name=None):
        """
        Save the entry at index idx as an extxyz file.
        """
        if path is None:
            path = os.getcwd()
        at = self.get_ase_atoms(idx, ext=True)
        if name is not None:
            name = at.info["name"]
        write_extxyz(p_join(path, f"{name}.xyz"), at)

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
        # _ = entry.pop("forces")
        at = dict_to_atoms(entry, ext=ext)
        return at

    @requires_package("dscribe")
    @requires_package("datamol")
    def soap_descriptors(
        self,
        n_samples: Optional[Union[List[int], int]] = None,
        return_idxs: bool = True,
        progress: bool = True,
        **soap_kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Compute the SOAP descriptors for the dataset.

        Parameters
        ----------
        n_samples : Optional[Union[List[int],int]], optional
            Number of samples to use for the computation, by default None. If None, all the dataset is used.
            If a list of integers is provided, the descriptors are computed for each of the specified idx of samples.
        return_idxs : bool, optional
            Whether to return the indices of the samples used, by default True.
        progress : bool, optional
            Whether to show a progress bar, by default True.
        **soap_kwargs : dict
            Keyword arguments to pass to the SOAP descriptor.
            By defaut, the following values are used:
                - r_cut : 5.0
                - n_max : 8
                - l_max : 6
                - average : "inner"
                - periodic : False
                - compression : {"mode" : "mu1nu1"}

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing the following keys:
                - soap : np.ndarray of shape (N, M) containing the SOAP descriptors for the dataset
                - soap_kwargs : dict containing the keyword arguments used for the SOAP descriptor
                - idxs : np.ndarray of shape (N,) containing the indices of the samples used

        """
        import datamol as dm
        from dscribe.descriptors import SOAP

        if n_samples is None:
            idxs = list(range(len(self)))
        elif isinstance(n_samples, int):
            idxs = np.random.choice(len(self), size=n_samples, replace=False)
        else:  # list, set, np.ndarray
            idxs = n_samples
        datum = {}
        r_cut = soap_kwargs.pop("r_cut", 5.0)
        n_max = soap_kwargs.pop("n_max", 8)
        l_max = soap_kwargs.pop("l_max", 6)
        average = soap_kwargs.pop("average", "inner")
        periodic = soap_kwargs.pop("periodic", False)
        compression = soap_kwargs.pop("compression", {"mode": "mu1nu1"})
        soap = SOAP(
            species=self.chemical_species,
            periodic=periodic,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            average=average,
            compression=compression,
        )
        datum["soap_kwargs"] = {
            "r_cut": r_cut,
            "n_max": n_max,
            "l_max": l_max,
            "average": average,
            "compression": compression,
            "species": self.chemical_species,
            "periodic": periodic,
            **soap_kwargs,
        }

        def wrapper(idx):
            entry = self.get_ase_atoms(idx, ext=False)
            return soap.create(entry, centers=entry.positions)

        descr = dm.parallelized(wrapper, idxs, progress=progress, scheduler="threads", n_jobs=-1)
        datum["soap"] = np.vstack(descr)
        if return_idxs:
            datum["idxs"] = idxs
        return datum

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
        )

    def __str__(self):
        return f"{self.__name__}"

    def __repr__(self):
        return f"{self.__name__}"

    @property
    def _stats(self):
        return self.__stats__

    @property
    def average_n_atoms(self):
        """
        Average number of atoms in a molecule in the dataset.
        """
        if self.__average_nb_atoms__ is None:
            raise StatisticsNotAvailableError(self.__name__)
        return self.__average_nb_atoms__

    def get_statistics(self, normalization: str = "formation", return_none: bool = True):
        """
        Get the statistics of the dataset.
        normalization : str, optional
            Type of energy, by default "formation", must be one of ["formation", "total", "inter"]
        return_none : bool, optional
            Whether to return None if the statistics for the forces are not available, by default True
            Otherwise, the statistics for the forces are set to 0.0
        """
        stats = deepcopy(self._stats)
        if len(stats) == 0:
            raise StatisticsNotAvailableError(self.__name__)
        if normalization not in POSSIBLE_NORMALIZATION:
            raise NormalizationNotAvailableError(normalization)
        selected_stats = stats[normalization]
        if len(self.__force_methods__) == 0 and not return_none:
            selected_stats.update(
                {
                    "forces": {
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
