from abc import ABC, abstractmethod
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


def dispatch_factory(data, **kwargs) -> "IsolatedEnergyInterface":
    """
    Factory function that select the correct
    energy class for the fetching/calculation
    of isolated atom energies.

    Parameters
    ----------
    data : openqdc.datasets.Dataset
        Dataset object that contains the information
        about the isolated atom energies. Info will be passed
        by references
    kwargs : dict
        Additional arguments that will be passed to the
        selected energy class. Mostly used for regression
        to pass the regressor_kwargs.
    """
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
    """
    Manager class for interface with the isolated atom energies classes
    and providing the generals function to retrieve the data
    """

    def __init__(self, data, **kwargs) -> None:
        """
        Parameters
        ----------
        data : openqdc.datasets.Dataset
            Dataset object that contains the information
            about the isolated atom energies. Info will be passed
            by references
        kwargs : dict
            Additional arguments that will be passed to the
            selected energy class. Mostly used for regression
            to pass the regressor_kwargs.
        """

        self.atom_energies = data.energy_type
        self.factory = dispatch_factory(data, **kwargs)

    @property
    def e0s_matrix(self) -> np.ndarray:
        """
        Returns the isolated atom energies matrixes
        """
        return self.factory.e0_matrix


class IsolatedEnergyInterface(ABC):
    """
    Abstract class that defines the interface for the
    different implementation of an isolated atom energy value
    """

    _e0_matrixs = []

    def __init__(self, data, **kwargs):
        """
        Parameters
        ----------
        data : openqdc.datasets.Dataset
            Dataset object that contains the information
            about the isolated atom energies. Info will be passed
            by references
        kwargs : dict
            Additional arguments that will be passed to the
            selected energy class. Mostly used for regression
            to pass the regressor_kwargs.
        """

        self.kwargs = kwargs
        self.data = data
        self._post_init()

    @property
    def refit(self) -> bool:
        return self.data.refit_e0s

    @abstractmethod
    def _post_init(self):
        """
        Main method to fetch/compute/recomputed the isolated atom energies.
        Need to be implemented in all child classes.
        """
        pass

    def __len__(self):
        return len(self.data.energy_methods)

    @property
    def e0_matrix(self) -> np.ndarray:
        """
        Return the isolated atom energies matrixes
        """
        return np.array(self._e0_matrixs)

    def __str__(self) -> str:
        return self.__class__.__name__.lower()


class NullEnergy(IsolatedEnergyInterface):
    """
    Class that returns a null (zeros) matrix for the isolated atom energies in case
    of no energies are available.
    """

    def _post_init(self):
        self._e0_matrixs = [np.zeros((max(chemical_symbols) + 1, MAX_CHARGE_NUMBER)) for _ in range(len(self))]


class PhysicalEnergy(IsolatedEnergyInterface):
    """
    Class that returns a physical (SE,DFT,etc) isolated atom energies.
    """

    def _post_init(self):
        self._e0_matrixs = [IsolatedAtomEnergyFactory.get_matrix(en_method) for en_method in self.data.energy_methods]


class RegressionEnergy(IsolatedEnergyInterface):
    """
    Class that compute and returns the regressed isolated atom energies.
    """

    def _post_init(self):
        if not self.attempt_load() or self.refit:
            self.regressor = Regressor.from_openqdc_dataset(self.data, **self.kwargs)
            E0s, cov = self._compute_regression_e0s()
            self._set_lin_atom_species_dict(E0s, cov)
        self._set_linear_e0s()

    def _compute_regression_e0s(self):
        """
        Try to compute the regressed isolated atom energies.
        raise an error if the regression fails.
        return the regressed isolated atom energies and the uncertainty values.
        """
        try:
            E0s, cov = self.regressor.solve()
        except np.linalg.LinAlgError:
            logger.warning(f"Failed to compute E0s using {self.regressor.solver_type} regression.")
            raise np.linalg.LinAlgError
        return E0s, cov

    def _set_lin_atom_species_dict(self, E0s, covs) -> None:
        """
        Set the regressed isolated atom energies in a dictionary format
        and Save the values in a pickle file to easy loading.
        """
        atomic_energies_dict = {}
        for i, z in enumerate(self.regressor.numbers):
            atomic_energies_dict[z] = E0s[i]
        self._e0s_dict = atomic_energies_dict
        self.save_e0s()

    def _set_linear_e0s(self) -> None:
        """
        Transform the e0s dictionary into the correct e0s
        matrix format
        """
        new_e0s = [np.zeros((max(self.data.numbers) + 1, MAX_CHARGE_NUMBER)) for _ in range(len(self))]
        for z, e0 in self._e0s_dict.items():
            for i in range(len(self)):
                new_e0s[i][z, :] = e0[i]
        self._e0_matrixs = new_e0s

    def save_e0s(self) -> None:
        """
        Save the regressed isolated atom energies in a pickle file.
        """
        save_pkl(self._e0s_dict, self.preprocess_path)

    def attempt_load(self) -> bool:
        """
        Try to load the regressed isolated atom energies from the
        object pickle file and return the success of the operation.
        """
        try:
            self._e0s_dict = load_pkl(self.preprocess_path)
            logger.info(f"Found energy file for {str(self)}.")
            return True
        except FileNotFoundError:
            logger.warning(f"Energy file for {str(self)} not found.")
            return False

    @property
    def preprocess_path(self):
        """
        Return the path to the object pickle file.
        """
        path = p_join(self.data.root, "preprocessed", str(self) + ".pkl")
        return path
