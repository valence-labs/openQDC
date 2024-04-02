from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from os.path import join as p_join
from typing import Optional

import numpy as np
from loguru import logger

from openqdc.utils.io import get_local_cache, load_pkl, save_pkl


class StatisticsResults:
    pass

    def to_dict(self):
        return asdict(self)

    def convert(self, func):
        for k, v in self.to_dict().items():
            if isinstance(v, dict):
                self.convert(func)
            else:
                setattr(self, k, func(v))


@dataclass
class EnergyStatistics(StatisticsResults):
    mean: Optional[np.ndarray]
    std: Optional[np.ndarray]


@dataclass
class ForceComponentsStatistics(StatisticsResults):
    mean: Optional[np.ndarray]
    std: Optional[np.ndarray]
    rms: Optional[np.ndarray]


@dataclass
class ForceStatistics(StatisticsResults):
    mean: Optional[np.ndarray]
    std: Optional[np.ndarray]
    components: ForceComponentsStatistics


class StatisticManager:
    """
    The Context defines the interface of interest to clients. It also maintains
    a reference to an instance of a State subclass, which represents the current
    state of the Context.
    """

    _state = {}
    _results = {}

    def __init__(self, dataset, recompute: bool = False, *statistic_calculators):
        self._statistic_calculators = [
            statistic_calculators.from_openqdc_dataset(dataset, recompute)
            for statistic_calculators in statistic_calculators
        ]

    @property
    def state(self):
        return self._state

    def get_state(self, key):
        if key is None:
            return self._state
        return self._state.get(key, None)

    def has_state(self, key):
        return key in self._state

    def get_results(self, as_dict=False):
        results = self._results
        if as_dict:
            results = {k: v.to_dict() for k, v in results.items()}
        return results

    def run_calculators(self):
        for calculator in self._statistic_calculators:
            calculator.run(self.state)
            self._results[calculator.__class__.__name__] = calculator.result


class AbstractStatsCalculator(ABC):
    deps = []
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
        self.name = name
        self.energy_type = energy_type
        self.force_recompute = force_recompute
        self.energies = energies
        self.forces = forces
        self.position_idx_range = position_idx_range
        self.e0_matrix = e0_matrix
        self.n_atoms = n_atoms
        self.atom_species_charges_tuple = (atom_species, atom_charges)
        if atom_species is not None and atom_charges is not None:
            # by value not reference
            self.atom_species_charges_tuple = np.concatenate((atom_species[:, None], atom_charges[:, None]), axis=-1)

    @property
    def has_forces(self) -> bool:
        return self.forces is not None

    @property
    def preprocess_path(self):
        path = p_join(self.root, "preprocessed", str(self) + ".pkl")
        return path

    @property
    def root(self):
        return p_join(get_local_cache(), self.name)

    @classmethod
    def from_openqdc_dataset(cls, dataset, recompute: bool = False):
        return cls(
            name=dataset.__name__,
            force_recompute=recompute,
            energy_type=dataset.energy_type,
            energies=dataset.data["energies"],
            forces=dataset.data["forces"],
            n_atoms=dataset.data["n_atoms"],
            position_idx_range=dataset.data["position_idx_range"],
            atom_species=dataset.data["atomic_inputs"][:, 0].ravel(),
            atom_charges=dataset.data["atomic_inputs"][:, 1].ravel(),
            e0_matrix=dataset.__isolated_atom_energies__,
        )

    @abstractmethod
    def compute(self) -> StatisticsResults:
        raise NotImplementedError

    def save_statistics(self) -> None:
        save_pkl(self.result.to_dict(), self.preprocess_path)

    def attempt_load(self) -> bool:
        try:
            self.result = load_pkl(self.preprocess_path)
            return True
        except FileNotFoundError:
            logger.warning(f"Statistics for {str(self)} not found. Computing...")
            return False

    def _setup_deps(self, state) -> None:
        self.state = state
        self.deps_satisfied = all([dep in state for dep in self.deps])
        if self.deps_satisfied:
            for dep in self.deps:
                setattr(self, dep, state[dep])

    def write_state(self, update) -> None:
        self.state.update(update)

    def run(self, state) -> None:
        self._setup_deps(state)
        if self.force_recompute or not self.attempt_load():
            self.result = self.compute()
            self.save_statistics()

    def __str__(self) -> str:
        return self.__class__.__name__.lower()


class ForcesCalculatorStats(AbstractStatsCalculator):
    def compute(self) -> ForceStatistics:
        if not self.has_forces:
            return ForceStatistics(
                mean=None, std=None, components=ForceComponentsStatistics(rms=None, std=None, mean=None)
            )
        converted_force_data = self.forces
        force_mean = np.nanmean(converted_force_data, axis=0)
        force_std = np.nanstd(converted_force_data, axis=0)
        force_rms = np.sqrt(np.nanmean(converted_force_data**2, axis=0))
        return ForceStatistics(
            mean=force_mean,
            std=force_std,
            components=ForceComponentsStatistics(rms=force_rms, std=force_std, mean=force_mean),
        )


class TotalEnergyStats(AbstractStatsCalculator):
    def compute(self):
        converted_energy_data = self.energies
        total_E_mean = np.nanmean(converted_energy_data, axis=0)
        total_E_std = np.nanstd(converted_energy_data, axis=0)
        return EnergyStatistics(mean=total_E_mean, std=total_E_std)


class FormationEnergyInterface(AbstractStatsCalculator, ABC):
    deps = ["formation_energy"]

    def compute(self) -> EnergyStatistics:
        if not self.deps_satisfied:
            from openqdc.utils.atomization_energies import IsolatedAtomEnergyFactory

            splits_idx = self.position_idx_range[:, 1]
            s = np.array(self.atom_species_charges_tuple, dtype=int)
            s[:, 1] += IsolatedAtomEnergyFactory.max_charge
            matrixs = [matrix[s[:, 0], s[:, 1]] for matrix in self.e0_matrix]
            converted_energy_data = self.energies
            E = []
            for i, matrix in enumerate(matrixs):
                c = np.cumsum(np.append([0], matrix))[splits_idx]
                c[1:] = c[1:] - c[:-1]
                E.append(converted_energy_data[:, i] - c)
        else:
            E = getattr(self, self.deps[0])
        self.write_state({self.deps[0]: E})
        E = np.array(E).T
        return self._compute(E)

    @abstractmethod
    def _compute(self, energy) -> EnergyStatistics:
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.__class__.__name__.lower()}_{self.energy_type.lower()}"


class FormationEnergyStats(FormationEnergyInterface):
    def _compute(self, energy) -> EnergyStatistics:
        formation_E_mean = np.nanmean(energy, axis=0)
        formation_E_std = np.nanstd(energy, axis=0)
        return EnergyStatistics(mean=formation_E_mean, std=formation_E_std)


class PerAtomFormationEnergyStats(FormationEnergyInterface):
    def _compute(self, energy) -> EnergyStatistics:
        inter_E_mean = np.nanmean((energy / self.n_atoms[:, None]), axis=0)
        inter_E_std = np.nanstd((energy / self.n_atoms[:, None]), axis=0)
        return EnergyStatistics(mean=inter_E_mean, std=inter_E_std)
