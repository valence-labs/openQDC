from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict, dataclass
from os.path import join as p_join
from typing import Any, Callable, Dict, Optional

import numpy as np
from loguru import logger

from openqdc.utils.io import get_local_cache, load_pkl, save_pkl


class StatisticsResults:
    """
    Parent class to statistics results
    to provide general methods.
    """

    def to_dict(self) -> Dict:
        """
        Convert the class to a dictionary

        Returns:
            Dictionary representation of the class
        """
        return asdict(self)

    def transform(self, func: Callable):
        """
        Apply a function to all the attributes of the class

        Parameters:
            func:
                Function to apply to the attributes
        """
        for k, v in self.to_dict().items():
            if v is not None:
                setattr(self, k, func(v))


@dataclass
class EnergyStatistics(StatisticsResults):
    """
    Dataclass for energy related statistics
    """

    mean: Optional[np.ndarray]
    std: Optional[np.ndarray]


@dataclass
class ForceStatistics(StatisticsResults):
    """
    Dataclass for force statistics
    """

    mean: Optional[np.ndarray]
    std: Optional[np.ndarray]
    component_mean: Optional[np.ndarray]
    component_std: Optional[np.ndarray]
    component_rms: Optional[np.ndarray]


class StatisticManager:
    """
    Manager class that automatically handle the shared state between
    the statistic calculators
    """

    def __init__(self, dataset: Any, recompute: bool = False, *statistic_calculators: "AbstractStatsCalculator"):
        """
        Parameters:
            dataset : openqdc.datasets.base.BaseDataset
                The dataset object to compute the statistics
            recompute:
                Flag to recompute the statistics
            *statistic_calculators:
                List of statistic calculators to run
        """
        self._state = {}
        self._results = {}
        self._statistic_calculators = [
            statistic_calculators.from_openqdc_dataset(dataset, recompute)
            for statistic_calculators in statistic_calculators
        ]

    @property
    def state(self) -> Dict:
        """
        Return the dictionary state of the manager

        Returns:
            State of the StatisticManager
        """
        return self._state

    def reset_state(self):
        """
        Reset the state dictionary
        """
        self._state = {}

    def reset_results(self):
        """
        Reset the results dictionary
        """
        self._results = {}

    def get_state(self, key: Optional[str] = None) -> Optional[Any]:
        """
        Return the value of the key in the state dictionary

        Parameters:
            key: str, default = None
        Returns:
            the value of the key in the state dictionary
            or the whole state dictionary if key is None
        """
        if key is None:
            return self._state
        return self._state.get(key, None)

    def has_state(self, key: str) -> bool:
        """
        Check is state has key

        Parameters:
            key:
                Key to check in the state dictionary

        Returns:
            True if the key is in the state dictionary
        """
        return key in self._state

    def get_results(self, as_dict: bool = False):
        """
        Aggregate results from all the calculators

        Parameters:
            as_dict:
                Flag to return the results as a dictionary
        """
        results = deepcopy(self._results)
        if as_dict:
            return {k: v.as_dict() for k, v in results.items()}
        return {k: v for k, v in self._results.items()}

    def run_calculators(self):
        """
        Run the saved calculators and save the results in the manager
        """
        logger.info("Processing dataset statistics")
        for calculator in self._statistic_calculators:
            calculator.run(self.state)
            self._results[calculator.__class__.__name__] = calculator.result


class AbstractStatsCalculator(ABC):
    """
    Abstract class that defines the interface for all
    the calculators object and the methods to
    compute the statistics.
    """

    # State Dependencies of the calculator to skip part of the calculation
    state_dependency = []
    name = None

    def __init__(
        self,
        name: str,
        energy_type: Optional[str] = None,
        force_recompute: bool = False,
        energies: Optional[np.ndarray] = None,
        n_atoms: Optional[np.ndarray] = None,
        atom_species: Optional[np.ndarray] = None,
        position_idx_range: Optional[np.ndarray] = None,
        e0_matrix: Optional[np.ndarray] = None,
        atom_charges: Optional[np.ndarray] = None,
        forces: Optional[np.ndarray] = None,
    ):
        """
        Parameters:
            name :
                Name of the dataset for saving and loading.
            energy_type :
                Type of the energy for the computation of the statistics. Used for loading and saving.
            force_recompute :
                Flag to force the recomputation of the statistics
            energies : n
                Energies of the dataset
            n_atoms :
                Number of atoms in the dataset
            atom_species :
                Atomic species of the dataset
            position_idx_range : n
                Position index range of the dataset
            e0_matrix :
                Isolated atom energies matrix of the dataset
            atom_charges :
                Atomic charges of the dataset
            forces :
                Forces of the dataset
        """
        self.name = name
        self.energy_type = energy_type
        self.force_recompute = force_recompute
        self.energies = energies
        self.forces = forces
        self.position_idx_range = position_idx_range
        self.e0_matrix = e0_matrix
        self.n_atoms = n_atoms
        self.atom_species_charges_tuple = (atom_species, atom_charges)
        self._root = p_join(get_local_cache(), self.name)
        if atom_species is not None and atom_charges is not None:
            # by value not reference
            self.atom_species_charges_tuple = np.concatenate((atom_species[:, None], atom_charges[:, None]), axis=-1)

    @property
    def has_forces(self) -> bool:
        return self.forces is not None

    @property
    def preprocess_path(self):
        path = p_join(self.root, "statistics", self.name + f"_{str(self)}" + ".pkl")
        return path

    @property
    def root(self):
        """
        Path to the dataset folder
        """
        return self._root

    @classmethod
    def from_openqdc_dataset(cls, dataset, recompute: bool = False):
        """
        Create a calculator object from a dataset object.
        """
        obj = cls(
            name=dataset.__name__,
            force_recompute=recompute,
            energy_type=dataset.energy_type,
            energies=dataset.data["energies"],
            forces=dataset.data["forces"] if "forces" in dataset.data else None,
            n_atoms=dataset.data["n_atoms"],
            position_idx_range=dataset.data["position_idx_range"],
            atom_species=dataset.data["atomic_inputs"][:, 0].ravel(),
            atom_charges=dataset.data["atomic_inputs"][:, 1].ravel(),
            e0_matrix=dataset.__isolated_atom_energies__,
        )
        obj._root = dataset.root  # set to the dataset root in case of multiple datasets
        return obj

    @abstractmethod
    def compute(self) -> StatisticsResults:
        """
        Abstract method to compute the statistics.
        Must return a StatisticsResults object and be implemented
        in all the childs
        """
        raise NotImplementedError

    def save_statistics(self) -> None:
        """
        Save statistics file to the dataset folder as a pkl file
        """
        save_pkl(self.result, self.preprocess_path)

    def attempt_load(self) -> bool:
        """
        Load precomputed statistics file and return the success of the operation
        """
        try:
            self.result = load_pkl(self.preprocess_path)
            logger.info(f"Statistics for {str(self)} loaded successfully")
            return True
        except FileNotFoundError:
            logger.warning(f"Statistics for {str(self)} not found. Computing...")
            return False

    def _setup_deps(self, state: Dict) -> None:
        """
        Check if the dependencies of calculators are satisfied
        from the state object and set the attributes of the calculator
        to skip part of the calculation
        """
        self.state = state
        self.deps_satisfied = all([dep in state for dep in self.state_dependency])
        if self.deps_satisfied:
            for dep in self.state_dependency:
                setattr(self, dep, state[dep])

    def write_state(self, update: Dict) -> None:
        """
        Write/update the state dictionary with the update dictionary

        update:
            dictionary containing the update to the state
        """
        self.state.update(update)

    def run(self, state: Dict) -> None:
        """
        Main method to run the calculator.
        Setup the dependencies from the state dictionary
        Check if the statistics are already computed and load them or
        recompute them
        Save the statistics in the correct folder

        state:
            dictionary containing the state of the calculator
        """
        self._setup_deps(state)
        if self.force_recompute or not self.attempt_load():
            self.result = self.compute()
            self.save_statistics()

    def __str__(self) -> str:
        return self.__class__.__name__.lower()


class ForcesCalculatorStats(AbstractStatsCalculator):
    """
    Forces statistics calculator class
    """

    def compute(self) -> ForceStatistics:
        if not self.has_forces:
            return ForceStatistics(mean=None, std=None, component_mean=None, component_std=None, component_rms=None)
        converted_force_data = self.forces
        num_methods = converted_force_data.shape[2]
        mean = np.nanmean(converted_force_data.reshape(-1, num_methods), axis=0)
        std = np.nanstd(converted_force_data.reshape(-1, num_methods), axis=0)
        component_mean = np.nanmean(converted_force_data, axis=0)
        component_std = np.nanstd(converted_force_data, axis=0)
        component_rms = np.sqrt(np.nanmean(converted_force_data**2, axis=0))
        return ForceStatistics(
            mean=np.atleast_2d(mean),
            std=np.atleast_2d(std),
            component_mean=np.atleast_2d(component_mean),
            component_std=np.atleast_2d(component_std),
            component_rms=np.atleast_2d(component_rms),
        )


class TotalEnergyStats(AbstractStatsCalculator):
    """
    Total Energy statistics calculator class
    """

    def compute(self) -> EnergyStatistics:
        converted_energy_data = self.energies
        total_E_mean = np.nanmean(converted_energy_data, axis=0)
        total_E_std = np.nanstd(converted_energy_data, axis=0)
        return EnergyStatistics(mean=np.atleast_2d(total_E_mean), std=np.atleast_2d(total_E_std))


class FormationEnergyInterface(AbstractStatsCalculator, ABC):
    """
    Formation Energy interface calculator class.
    Define the use of the dependency formation_energy in the
    compute method
    """

    state_dependency = ["formation_energy"]

    def compute(self) -> EnergyStatistics:
        # if the state has not the dependency satisfied
        if not self.deps_satisfied:
            # run the main computation
            from openqdc.utils.constants import MAX_CHARGE

            splits_idx = self.position_idx_range[:, 1]
            s = np.array(self.atom_species_charges_tuple, dtype=int)
            s[:, 1] += MAX_CHARGE
            matrixs = [matrix[s[:, 0], s[:, 1]] for matrix in self.e0_matrix]
            converted_energy_data = self.energies
            E = []
            for i, matrix in enumerate(matrixs):
                c = np.cumsum(np.append([0], matrix))[splits_idx]
                c[1:] = c[1:] - c[:-1]
                E.append(converted_energy_data[:, i] - c)
        else:
            # if the dependency is satisfied get the dependency
            E = getattr(self, self.state_dependency[0])
        self.write_state({self.state_dependency[0]: E})
        E = np.array(E).T
        return self._compute(E)

    @abstractmethod
    def _compute(self, energy) -> EnergyStatistics:
        raise NotImplementedError

    def __str__(self) -> str:
        # override the __str__ method to add the energy type to the name
        # to differentiate between formation and regression type
        return f"{self.__class__.__name__.lower()}_{self.energy_type.lower()}"


class FormationEnergyStats(FormationEnergyInterface):
    """
    Formation Energy  calculator class.
    """

    def _compute(self, energy) -> EnergyStatistics:
        formation_E_mean = np.nanmean(energy, axis=0)
        formation_E_std = np.nanstd(energy, axis=0)
        return EnergyStatistics(mean=np.atleast_2d(formation_E_mean), std=np.atleast_2d(formation_E_std))


class PerAtomFormationEnergyStats(FormationEnergyInterface):
    """
    Per atom Formation Energy  calculator class.
    """

    def _compute(self, energy) -> EnergyStatistics:
        inter_E_mean = np.nanmean((energy / self.n_atoms[:, None]), axis=0)
        inter_E_std = np.nanstd((energy / self.n_atoms[:, None]), axis=0)
        return EnergyStatistics(mean=np.atleast_2d(inter_E_mean), std=np.atleast_2d(inter_E_std))
