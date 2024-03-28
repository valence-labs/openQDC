from abc import ABC, abstractmethod
from dataclasses import dataclass
from os.path import join as p_join

import numpy as np
from loguru import logger

from openqdc.utils.atomization_energies import (
    IsolatedAtomEnergyFactory,
    chemical_symbols,
)
from openqdc.utils.io import load_pkl, save_pkl
from openqdc.utils.regressor import Regressor

POSSIBLE_ENERGIES = ["formation", "regression", "null"]
MAX_CHARGE_NUMBER = 21


def dispatch_factory(data, **kwargs):
    if data.energy_type == "formation":
        return PhysicalEnergy(data, **kwargs)
    elif data.energy_type == "regression":
        try:
            return RegressionEnergy(data, **kwargs)
        except np.linalg.LinAlgError:
            logger.warning("Error! Using physical energies instead.")
            return PhysicalEnergy(data, **kwargs)
    elif data.energy_type == "null":
        return NullEnergy(data, **kwargs)


class AtomEnergies:
    def __init__(self, data, **kwargs):
        self.atom_energies = data.energy_type
        self.factory = dispatch_factory(data, **kwargs)

    @property
    def e0s_matrix(self):
        return self.factory.e0_matrix


class IsolatedEnInterface(ABC):
    _e0_matrixs = []

    def __init__(self, data, **kwargs):
        self.kwargs = kwargs
        self.data = data
        self._post_init()

    @abstractmethod
    def _post_init(self):
        pass

    def __len__(self):
        return len(self.data.energy_methods)

    @property
    def e0_matrix(self):
        return np.array(self._e0_matrixs)

    def __str__(self) -> str:
        return self.__class__.__name__.lower()


class NullEnergy(IsolatedEnInterface):

    def _post_init(self):
        self._e0_matrixs = [np.zeros((max(chemical_symbols) + 1, MAX_CHARGE_NUMBER)) for _ in range(len(self))]


class PhysicalEnergy(IsolatedEnInterface):

    def _post_init(self):
        self._e0_matrixs = [IsolatedAtomEnergyFactory.get_matrix(en_method) for en_method in self.data.energy_methods]


class RegressionEnergy(IsolatedEnInterface):

    def _post_init(self):
        if not self.attempt_load():
            self.regressor = Regressor.from_openqdc_dataset(self.data, **self.kwargs)
            E0s, cov = self._compute_regression_e0s()
            self._set_lin_atom_species_dict(E0s, cov)
        self._set_linear_e0s()

    def _compute_regression_e0s(self):
        try:

            E0s, cov = self.regressor.solve()
        except np.linalg.LinAlgError:
            logger.warning(f"Failed to compute E0s using {self.regressor.solver_type} regression.")
            raise np.linalg.LinAlgError
        return E0s, cov

    def _set_lin_atom_species_dict(self, E0s, covs):
        atomic_energies_dict = {}
        for i, z in enumerate(self.regressor.numbers):
            atomic_energies_dict[z] = E0s[i]
        self._e0s_dict = atomic_energies_dict
        self.save_e0s()

    def _set_linear_e0s(self):
        new_e0s = [np.zeros((max(self.data.numbers) + 1, MAX_CHARGE_NUMBER)) for _ in range(len(self))]
        for z, e0 in self._e0s_dict.items():
            for i in range(len(self)):
                new_e0s[i][z, :] = e0[i]
        self._e0_matrixs = new_e0s

    def save_e0s(self) -> None:
        save_pkl(self._e0s_dict, self.preprocess_path)

    def attempt_load(self) -> bool:
        try:
            self._e0s_dict = load_pkl(self.preprocess_path)
            logger.info(f"Found energy file for {str(self)}.")
            return True
        except FileNotFoundError:
            logger.warning(f"Energy file for {str(self)} not found.")
            return False

    @property
    def preprocess_path(self):
        path = p_join(self.data.root, "preprocessed", str(self) + ".pkl")
        return path
 