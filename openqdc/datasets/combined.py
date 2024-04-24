"""The BaseDataset defining shared functionality between all datasets."""

import os
import pickle as pkl
from functools import partial
from itertools import compress
from os.path import join as p_join
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from ase.io.extxyz import write_extxyz
from loguru import logger
from sklearn.utils import Bunch
from tqdm import tqdm

from openqdc.datasets import AVAILABLE_DATASETS, COMMON_MAP_POTENTIALS
from openqdc.datasets.energies import AtomEnergies
from openqdc.datasets.properties import DatasetPropertyMixIn
from openqdc.datasets.statistics import (
    ForcesCalculatorStats,
    FormationEnergyStats,
    PerAtomFormationEnergyStats,
    StatisticManager,
    TotalEnergyStats,
)
from openqdc.utils.constants import MAX_CHARGE, NB_ATOMIC_FEATURES
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
from openqdc.utils.package_utils import has_package, requires_package
from openqdc.utils.regressor import Regressor  # noqa
from openqdc.utils.units import get_conversion

from .base import BaseDataset

if has_package("torch"):
    import torch

if has_package("jax"):
    import jax.numpy as jnp


@requires_package("torch")
def to_torch(x: np.ndarray):
    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(x)


@requires_package("jax")
def to_jax(x: np.ndarray):
    if isinstance(x, jnp.ndarray):
        return x
    return jnp.array(x)


_CONVERT_DICT = {"torch": to_torch, "jax": to_jax, "numpy": lambda x: x}


class CombinedBaseDataset(BaseDataset):
    """
    Base class for datasets in the openQDC package.
    """

    __name__ = "custom"
    __energy_methods__ = []
    __force_mask__ = []
    __isolated_atom_energies__ = []
    _fn_energy = lambda x: x
    _fn_distance = lambda x: x
    _fn_forces = lambda x: x

    __energy_unit__ = "kcal/mol"
    __distance_unit__ = "ang"
    __forces_unit__ = "kcal/mol/ang"
    __average_nb_atoms__ = None

    def __init__(
        self,
        level_of_theory: str,
        energy_unit: Optional[str] = None,
        distance_unit: Optional[str] = None,
        array_format: str = "numpy",
        energy_type: str = "formation",
        overwrite_local_cache: bool = False,
        cache_dir: Optional[str] = None,
        recompute_statistics: bool = False,
        transform: Optional[Callable] = None,
        regressor_kwargs={
            "solver_type": "linear",
            "sub_sample": None,
            "stride": 1,
        },
    ) -> None:
        # assert level_of_theory in COMMON_MAP_POTENTIALS, f"Level of theory {level_of_theory} not available for multiple datasets"
        set_cache_dir(cache_dir)
        # self._init_lambda_fn()
        self._level_of_theory_map = level_of_theory
        self.data = None
        self.recompute_statistics = recompute_statistics
        self.regressor_kwargs = regressor_kwargs
        self.transform = transform
        self.energy_type = energy_type
        self.refit_e0s = recompute_statistics or overwrite_local_cache
        self.initialize_multi_dataset()
        self.set_array_format(array_format)
        self._post_init(overwrite_local_cache, energy_unit, distance_unit)

    def initialize_multi_dataset(self):
        self.datasets = [
            AVAILABLE_DATASETS[dataset](
                energy_unit=energy_unit,
                distance_unit=distance_unit,
                array_format=array_format,
                energy_type=energy_type,
                overwrite_local_cache=overwrite_local_cache,
                cache_dir=cache_dir,
                recompute_statistics=recompute_statistics,
                transform=transform,
                regressor_kwargs=regressor_kwargs,
            )
            for dataset in COMMON_MAP_POTENTIALS[self._level_of_theory_map]
        ]

        self.num_datasets = len(self.datasets)
        # Store the number of graphs in each dataset
        self.sizes = [len(d) for d in self.datasets]
        # Stores the cumulative sum of the number of graphs in each dataset
        self.cum_sizes = [0] + list(np.cumsum(self.sizes))
        # Store which index corresponds to which dataset
        self.which_dataset = []
        for i, d in enumerate(self.datasets):
            self.which_dataset += [i] * len(d)

        self._setup_energy_attrs()
        self._setup_force_attrs()
        # we need to shift and recollate the data
        self._shift_data()

    def _setup_energy_attrs(self):
        """Creates energy_methods and energy_method_to_idx attributes.

        - energy_methods: List of energy methods used in the dataset.
        - energy_method_to_idx: Dict mapping energy methods to indices.
        """
        self.energy_methods = [ds.energy_methods for ds in self.datasets]
        self.energy_method_to_idx = {em: i for i, em in enumerate(list(dict.fromkeys(self.energy_methods)))}

    def _setup_force_attrs(self):
        """Creates force_methods and force_method_to_idx attributes.

        - force_methods: List of force methods used in the dataset.
        - force_method_to_idx: Dict mapping force methods to indices.
        """
        self.force_methods = [ds.force_method for ds in self.datasets]
        self.force_method_to_idx = {fm: i if fm else -1 for i, fm in enumerate(list(dict.fromkeys(self.force_methods)))}

    def len(self):
        if not hasattr(self, "_length"):
            self._length = sum([len(dataset) for dataset in self.datasets])
        return self._length

    def _shift_data(self):
        self.data = {}
        for key in self.data.keys():
            if key not in ["position_idx_range"]:
                pass
            else:
                shift_idx = np.cumsum([0] + [len(d) for d in self.datasets])
                for i in range(1, len(shift_idx)):
                    self.data[key][i] += shift_idx[i - 1]
            self.data[key] = np.vstack([self.datasets[i].data[key] for i in range(self.num_datasets)])


# with combined dataset I want to override the single stat and get method
# name should be a list of the combined datasets names
