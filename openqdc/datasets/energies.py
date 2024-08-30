from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from os.path import join as p_join
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from loguru import logger

from openqdc.methods.enums import PotentialMethod
from openqdc.utils.constants import ATOM_SYMBOLS, ATOMIC_NUMBERS, MAX_CHARGE_NUMBER
from openqdc.utils.io import load_pkl, save_pkl
from openqdc.utils.regressor import Regressor

POSSIBLE_ENERGIES = ["formation", "regression", "null"]


def dispatch_factory(data: Any, **kwargs: Dict) -> "IsolatedEnergyInterface":
    """
    Factory function that select the correct
    energy class for the fetching/calculation
    of isolated atom energies.

    Parameters:
        data : openqdc.datasets.Dataset
            Dataset object that contains the information
            about the isolated atom energies. Info will be passed
            by references
        kwargs : dict
            Additional arguments that will be passed to the
            selected energy class. Mostly used for regression
            to pass the regressor_kwargs.

    Returns:
        Initialized IsolatedEnergyInterface-like object
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


@dataclass(frozen=False, eq=True)
class AtomSpecies:
    """
    Structure that defines a tuple of chemical specie and charge
    and provide hash and automatic conversion from atom number to
    checmical symbol
    """

    symbol: Union[str, int]
    charge: int = 0

    def __post_init__(self):
        if not isinstance(self.symbol, str):
            self.symbol = ATOM_SYMBOLS[self.symbol]
        self.number = ATOMIC_NUMBERS[self.symbol]

    def __hash__(self):
        return hash((self.symbol, self.charge))

    def __eq__(self, other):
        if not isinstance(other, AtomSpecies):
            symbol, charge = other[0], other[1]
            other = AtomSpecies(symbol=symbol, charge=charge)
        return (self.number, self.charge) == (other.number, other.charge)


@dataclass
class AtomEnergy:
    """
    Datastructure to store isolated atom energies
    and the std deviation associated to the value.
    By default the std will be 1 if no value was calculated
    or not available (formation energy case)
    """

    mean: np.array
    std: np.array = field(default_factory=lambda: np.array([1], dtype=np.float32))

    def __post_init__(self):
        if not isinstance(self.mean, np.ndarray):
            self.mean = np.array([self.mean], dtype=np.float32)

    def append(self, other: "AtomEnergy"):
        """
        Append the mean and std of another atom energy
        """
        self.mean = np.append(self.mean, other.mean)
        self.std = np.append(self.std, other.std)


class AtomEnergies:
    """
    Manager class for interface with the isolated atom energies classes
    and providing the generals function to retrieve the data
    """

    def __init__(self, data, **kwargs) -> None:
        self.atom_energies = data.energy_type
        self.factory = dispatch_factory(data, **kwargs)

    @property
    def e0s_matrix(self) -> np.ndarray:
        """
        Return the isolated atom energies dictionary

        Returns:
            Matrix Array with the isolated atom energies
        """
        return self.factory.e0_matrix

    @property
    def e0s_dict(self) -> Dict[AtomSpecies, AtomEnergy]:
        """
        Return the isolated atom energies dictionary

        Returns:
            Dictionary with the isolated atom energies
        """
        return self.factory.e0_dict

    def __str__(self):
        return f"Atoms: { list(set(map(lambda x : x.symbol, self.e0s_dict.keys())))}"

    def __repr__(self):
        return str(self)

    def __getitem__(self, item: AtomSpecies) -> AtomEnergy:
        """
        Retrieve a key from the isolated atom dictionary.
        Item can be written as tuple(Symbol, charge),
        tuple(Chemical number, charge). If no charge is passed,
        it will be automatically set to 0.

        Examples:
            AtomEnergies[6], AtomEnergies[6,1], \n
            AtomEnergies["C",1], AtomEnergies[(6,1)], \n
            AtomEnergies[("C,1)]

        Parameters:
            item:
                AtomSpecies object or tuple with the atom symbol and charge

        Returns:
            AtomEnergy object with the isolated atom energy
        """
        try:
            atom, charge = item[0], item[1]
        except TypeError:
            atom = item
            charge = 0
        except IndexError:
            atom = item[0]
            charge = 0
        if not isinstance(atom, str):
            atom = ATOM_SYMBOLS[atom]
        return self.e0s_dict[(atom, charge)]


class IsolatedEnergyInterface(ABC):
    """
    Abstract class that defines the interface for the
    different implementation of an isolated atom energy value
    """

    def __init__(self, data, **kwargs):
        """
        Parameters:
            data : openqdc.datasets.Dataset
                Dataset object that contains the information
                about the isolated atom energies. Info will be passed
                by references
            kwargs : dict
                Additional arguments that will be passed to the
                selected energy class. Mostly used for regression
                to pass the regressor_kwargs.
        """
        self._e0_matrixs = []
        self._e0_dict = None
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

        Returns:
            Matrix Array with the isolated atom energies
        """
        return np.array(self._e0_matrixs)

    @property
    def e0_dict(self) -> Dict:
        """
        Return the isolated atom energies dict

        Returns:
            Dictionary with the isolated atom energies
        """

        return self._e0s_dict

    def __str__(self) -> str:
        return self.__class__.__name__.lower()


class PhysicalEnergy(IsolatedEnergyInterface):
    """
    Class that returns a physical (SE,DFT,etc) isolated atom energies.
    """

    def _assembly_e0_dict(self):
        datum = {}
        for method in self.data.__energy_methods__:
            for key, values in method.atom_energies_dict.items():
                atm = AtomSpecies(*key)
                ens = AtomEnergy(values)
                if atm not in datum:
                    datum[atm] = ens
                else:
                    datum[atm].append(ens)
        self._e0s_dict = datum

    def _post_init(self):
        self._e0_matrixs = [energy_method.atom_energies_matrix for energy_method in self.data.__energy_methods__]
        self._assembly_e0_dict()


class NullEnergy(IsolatedEnergyInterface):
    """
    Class that returns a null (zeros) matrix for the isolated atom energies in case
    of no energies are available.
    """

    def _assembly_e0_dict(self):
        datum = {}
        for _ in self.data.__energy_methods__:
            for key, values in PotentialMethod.NONE.atom_energies_dict.items():
                atm = AtomSpecies(*key)
                ens = AtomEnergy(values)
                if atm not in datum:
                    datum[atm] = ens
                else:
                    datum[atm].append(ens)
        self._e0s_dict = datum

    def _post_init(self):
        self._e0_matrixs = [PotentialMethod.NONE.atom_energies_matrix for _ in range(len(self.data.energy_methods))]
        self._assembly_e0_dict()


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

    def _compute_regression_e0s(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Try to compute the regressed isolated atom energies.
        raise an error if the regression fails.
        return the regressed isolated atom energies and the uncertainty values.

        Returns:
            Tuple with the regressed isolated atom energies and the uncertainty values of the regression
            if available.
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
            for charge in range(-10, 11):
                atomic_energies_dict[AtomSpecies(z, charge)] = AtomEnergy(E0s[i], 1 if covs is None else covs[i])
            # atomic_energies_dict[z] = E0s[i]
        self._e0s_dict = atomic_energies_dict
        self.save_e0s()

    def _set_linear_e0s(self) -> None:
        """
        Transform the e0s dictionary into the correct e0s
        matrix format.
        """
        new_e0s = [np.zeros((max(self.data.numbers) + 1, MAX_CHARGE_NUMBER)) for _ in range(len(self))]
        for z, e0 in self._e0s_dict.items():
            for i in range(len(self)):
                # new_e0s[i][z, :] = e0[i]
                new_e0s[i][z.number, z.charge] = e0.mean[i]
            # for atom_sp, values in
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
