import numpy as np
import pandas as pd

from openqdc.utils.constants import ATOM_SYMBOLS
from openqdc.utils.exceptions import StatisticsNotAvailableError


class DatasetPropertyMixIn:
    """
    Mixin class for BaseDataset class to add
    properties that are common to all datasets.
    """

    @property
    def atoms_per_molecules(self):
        try:
            if hasattr(self, "_n_atoms"):
                return self._n_atoms
            self._n_atoms = self.data["n_atoms"]
            return self._n_atoms
        except:  # noqa
            return None

    @property
    def _stats(self):
        return self.__stats__

    def _compute_average_nb_atoms(self):
        self.__average_nb_atoms__ = np.mean(self.data["n_atoms"])

    @property
    def average_n_atoms(self):
        """
        Average number of atoms in a molecule in the dataset.
        """
        if self.__average_nb_atoms__ is None:
            raise StatisticsNotAvailableError(self.__name__)
        return self.__average_nb_atoms__

    @property
    def numbers(self):
        if hasattr(self, "_numbers"):
            return self._numbers
        self._numbers = pd.unique(self.data["atomic_inputs"][..., 0]).astype(np.int32)
        return self._numbers

    @property
    def charges(self):
        if hasattr(self, "_charges"):
            return self._charges
        self._charges = np.unique(self.data["atomic_inputs"][..., :2], axis=0).astype(np.int32)
        return self._charges

    @property
    def min_max_charges(self):
        if hasattr(self, "_min_max_charges"):
            return self._min_max_charges
        self._min_max_charges = np.min(self.charges[:, 1]), np.max(self.charges[:, 1])
        return self._min_max_charges

    @property
    def chemical_species(self):
        return np.array(ATOM_SYMBOLS)[self.numbers]