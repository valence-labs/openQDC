import pickle as pkl
from os.path import join as p_join

import numpy as np
import pandas as pd
from loguru import logger

from openqdc.utils.atomization_energies import (
    IsolatedAtomEnergyFactory,
    chemical_symbols,
)
from openqdc.utils.constants import NOT_DEFINED
from openqdc.utils.exceptions import StatisticsNotAvailableError
from openqdc.utils.io import load_pkl
from openqdc.utils.regressor import Regressor


class StatisticsMixIn:
    def _set_lin_atom_species_dict(self, E0s, covs, zs):
        atomic_energies_dict = {}
        for i, z in enumerate(zs):
            atomic_energies_dict[z] = E0s[i]
        self.linear_e0s = atomic_energies_dict

    def _compute_linear_e0s(self):
        try:
            regressor = Regressor.from_openqdc_dataset(self, **self.regressor_kwargs)
            E0s, cov = regressor.solve()
        except np.linalg.LinAlgError:
            logger.warning(f"Failed to compute E0s using {regressor.solver_type} regression.")
            raise np.linalg.LinAlgError
        self._set_lin_atom_species_dict(E0s, cov, regressor.numbers)

    def _precompute_statistics(self, overwrite_local_cache: bool = False):
        local_path = p_join(self.preprocess_path, "stats.pkl")
        if self.is_preprocessed_statistics() and not (overwrite_local_cache or self.recompute_statistics):
            stats = load_pkl(local_path)
            try:
                self.linear_e0s = stats.get("linear_regression_values")
                self._set_linear_e0s()
            except Exception:
                logger.warning(f"Failed to load linear regression values for {self.__name__} dataset.")
            logger.info("Loaded precomputed statistics")
        else:
            logger.info("Precomputing relevant statistics")
            stats = self._precompute_E()
            forces_dict = self._precompute_F()
            for key in stats:
                if key != "linear_regression_values":
                    stats[key].update({"forces": forces_dict})
            with open(local_path, "wb") as f:
                pkl.dump(stats, f)
        self._compute_average_nb_atoms()
        self.__stats__ = stats

    def _compute_average_nb_atoms(self):
        self.__average_nb_atoms__ = np.mean(self.data["n_atoms"])

    def _set_linear_e0s(self):
        new_e0s = [np.zeros((max(self.numbers) + 1, 21)) for _ in range(len(self.__energy_methods__))]
        for z, e0 in self.linear_e0s.items():
            for i in range(len(self.__energy_methods__)):
                new_e0s[i][z, :] = e0[i]
        self.new_e0s = np.array(new_e0s)

    def _precompute_E(self):
        splits_idx = self.data["position_idx_range"][:, 1]
        s = np.array(self.data["atomic_inputs"][:, :2], dtype=int)
        s[:, 1] += IsolatedAtomEnergyFactory.max_charge
        matrixs = [matrix[s[:, 0], s[:, 1]] for matrix in self.__isolated_atom_energies__]
        REGRESSOR_SUCCESS = False
        try:
            self._compute_linear_e0s()
            self._set_linear_e0s()
            linear_matrixs = [matrix[s[:, 0], s[:, 1]] for matrix in self.new_e0s]
            REGRESSOR_SUCCESS = True
        except np.linalg.LinAlgError:
            logger.warning(f"Failed to compute linear regression values for {self.__name__} dataset.")
        converted_energy_data = self.data["energies"]
        # calculation per molecule formation energy statistics
        E, E_lin = [], []
        for i, matrix in enumerate(matrixs):
            c = np.cumsum(np.append([0], matrix))[splits_idx]
            c[1:] = c[1:] - c[:-1]
            E.append(converted_energy_data[:, i] - c)
            if REGRESSOR_SUCCESS:
                c = np.cumsum(np.append([0], linear_matrixs[i]))[splits_idx]
                c[1:] = c[1:] - c[:-1]
                E_lin.append(converted_energy_data[:, i] - c)
        E = np.array(E).T
        inter_E_mean = np.nanmean(E / self.data["n_atoms"][:, None], axis=0)
        inter_E_std = np.nanstd(E / self.data["n_atoms"][:, None], axis=0)
        formation_E_mean = np.nanmean(E, axis=0)
        formation_E_std = np.nanstd(E, axis=0)
        total_E_mean = np.nanmean(converted_energy_data, axis=0)
        total_E_std = np.nanstd(converted_energy_data, axis=0)
        statistics_dict = {
            "formation": {"energy": {"mean": np.atleast_2d(formation_E_mean), "std": np.atleast_2d(formation_E_std)}},
            "inter": {"energy": {"mean": np.atleast_2d(inter_E_mean), "std": np.atleast_2d(inter_E_std)}},
            "total": {"energy": {"mean": np.atleast_2d(total_E_mean), "std": np.atleast_2d(total_E_std)}},
        }
        if REGRESSOR_SUCCESS:
            E_lin = np.array(E_lin).T
            linear_E_mean = np.nanmean(E_lin, axis=0)
            linear_E_std = np.nanstd(E_lin, axis=0)
            linear_inter_E_mean = np.nanmean(E_lin / self.data["n_atoms"][:, None], axis=0)
            linear_inter_E_std = np.nanmean(E_lin / self.data["n_atoms"][:, None], axis=0)
            statistics_dict.update(
                {
                    "regression": {
                        "energy": {"mean": np.atleast_2d(linear_E_mean), "std": np.atleast_2d(linear_E_std)}
                    },
                    "regression_inter": {
                        "energy": {"mean": np.atleast_2d(linear_inter_E_mean), "std": np.atleast_2d(linear_inter_E_std)}
                    },
                    "linear_regression_values": self.linear_e0s,
                }
            )
        return statistics_dict

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


class DatasetPropertyMixIn(StatisticsMixIn):
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
        return np.array(chemical_symbols)[self.numbers]


from abc import ABC, abstractmethod
from dataclasses import dataclass 
from typing import Optional

class StatisticsResults:
    pass 

@dataclass
class EnergyStatistics(StatisticsResults):
    mean: Optional[np.ndarray]
    std: Optional[np.ndarray]

@dataclass
class ForceComponentsStatistics:
    mean: Optional[np.ndarray]
    std: Optional[np.ndarray]
    rms: Optional[np.ndarray]
    
@dataclass
class ForceStatistics(StatisticsResults):
    mean: Optional[np.ndarray]
    std: Optional[np.ndarray]
    components: ForceComponentsStatistics 
    
class CalculatorHandler(ABC):
    @abstractmethod
    def set_next_calculator(self, calculator: CalculatorHandler) -> CalculatorHandler:
        pass
    
    @abstractmethod
    def run_calculator(self):
        pass
    
class AbstractStatsCalculator(CalculatorHandler):
    deps = []
    provided_deps = []
    avoid_calculations = []
    _next_calculator: Optional[CalculatorHandler] = None
    
    def __init__(self, energies : Optional[np.ndarray] = None,
                 n_atoms : Optional[np.ndarray] = None,
                 atom_species : Optional[np.ndarray] = None,
                 position_idx_range : Optional[np.ndarray] = None,
                 e0_matrix:  Optional[np.ndarray] = None,
                 atom_charges: Optional[np.ndarray] = None,
                 forces: Optional[np.ndarray] = None):
        self.energies=energies
        self.forces=forces
        self.position_idx_range=position_idx_range
        self.e0_matrix=e0_matrix
        self.n_atoms=n_atoms
        self.atom_species_charges_tuple = (atom_species, atom_charges)
        if atom_species is not None and atom_charges is not None:
            self.atom_species_charges_tuple = np.concatenate((atom_species[:,None], atom_charges[:,None]), axis=-1)
    
    @property
    def has_forces(self):
        return self.forces is not None
    
    def set_next_calculator(self, calculator: AbstractStatsCalculator) -> AbstractStatsCalculator:
        self._next_calculator = calculator
        
        if set(self.provided_deps) & set(self._next_calculator.deps):
            [setattr(self._next_calculator, attr, getattr(self, attr) ) for attr in set(self.provided_deps) & set(self._next_calculator.deps)]
        return calculator 

    def run_calculator(self):
        self.compute()
        if self._next_handler:
            return self._next_calculator.compute()
        return None
    
    @classmethod
    def from_openqdc_dataset(cls, dataset):
        return cls(energies=dataset.data["energies"],
                   forces=dataset.data["forces"],
                   n_atoms=dataset.data["n_atoms"],
                   position_idx_range=dataset.data["position_idx_range"],
                   atom_species=dataset.data["atomic_inputs"][:,0].ravel(),
                   atom_charges=dataset.data["atomic_inputs"][:,1].ravel(),
                   e0_matrix=dataset.__isolated_atom_energies__)
    
    
    @abstractmethod
    def compute(self)->StatisticsResults:
        pass
    
class StatisticsOrderHandler(Handler):
    strategies = []
    def __init__(self, *strategies)
        pass 
        
    def set_strategy(self):
        pass 
    
    
class ForcesCalculator(AbstractStatsCalculator):
    deps = []
    provided_deps = []
    avoid_calculations = []
    
    def compute_forces_statistics(self)->ForceStatistics:
        if not self.has_forces:
            return ForceStatistics(mean=None,
                               std=None,
                               components=ForceComponentsStatistics(rms=None,
                                                                    std=None,
                                                                    mean=None)
        converted_force_data = self.forces
        force_mean = np.nanmean(converted_force_data, axis=0)
        force_std = np.nanstd(converted_force_data, axis=0)
        force_rms = np.sqrt(np.nanmean(converted_force_data**2, axis=0))
        return ForceStatistics(mean=force_mean,
                               std=force_std,
                               components=ForceComponentsStatistics(rms=force_rms,
                                                                    std=force_std,
                                                                    mean=force_mean)
        )
       
class TotalEnergyStats(AbstractStatsCalculator):
    deps = []
    provided_deps = []
    avoid_calculations = []
    
    def compute_energy_statistics(self):
        converted_energy_data = self.data["energies"]
        total_E_mean = np.nanmean(converted_energy_data, axis=0)
        total_E_std = np.nanstd(converted_energy_data, axis=0)
        return EnergyStatistics(mean=total_E_mean, std=total_E_std)
         
class FormationEnergyInterface(AbstractStatsCalculator, ABC):
    deps = ["formation_energy"]
    provided_deps = ["formation_energy"]
    avoid_calculations = []
    
    def compute(self)->EnergyStatistics:
        if not hasattr(self, "formation_energy"):
            from openqdc.utils.atomization_energies import IsolatedAtomEnergyFactory
            splits_idx = self.position_idx_range[:, 1]
            s = np.array(self.atom_species_charges_tuple, dtype=int)
            s[:, 1] += IsolatedAtomEnergyFactory.max_charge
            matrixs = [matrix[s[:, 0], s[:, 1]] for matrix in self.e0_matrix]
            converted_energy_data = self.energy
            # calculation per molecule formation energy statistics
            E = []
            for i, matrix in enumerate(matrixs):
                c = np.cumsum(np.append([0], matrix))[splits_idx]
                c[1:] = c[1:] - c[:-1]
                E.append(converted_energy_data[:, i] - c)
        else:
            E = self.formation_energy
        E = np.array(E).T
        return self._compute(E)
   
    @abstractmethod
    def _compute(self, energy)->EnergyStatistics:
        raise NotImplementedError
    
class FormationStats(FormationEnergyInterface):
    deps = ["formation_energy"]
    provided_deps = ["formation_energy"]
    avoid_calculations = []
    
    def _compute(self, energy)->EnergyStatistics:
        formation_E_mean = np.nanmean(energy, axis=0)
        formation_E_std = np.nanstd(energy, axis=0)
        return EnergyStatistics(mean=formation_E_mean, std=formation_E_std)
        
class PerAtomFormationEnergyStats(FormationEnergyInterface):
    deps = ["formation_energy"]
    provided_deps = ["formation_energy"]
    avoid_calculations = []
    
    def _compute(self, energy)->EnergyStatistics:
        inter_E_mean = np.nanmean(energy / self.n_atoms][:, None]), axis=0)
        inter_E_std = np.nanstd(energy / self,n_atoms][:, None]), axis=0)
        return EnergyStatistics(mean=inter_E_mean, std=inter_E_std)
        
    
class RegressionStats(AbstractStatsCalculator):
        
    
    def _compute_linear_e0s(self):
        try:
            regressor = Regressor.from_openqdc_dataset(self, **self.regressor_kwargs)
            E0s, cov = regressor.solve()
        except np.linalg.LinAlgError:
            logger.warning(f"Failed to compute E0s using {regressor.solver_type} regression.")
            raise np.linalg.LinAlgError
        self._set_lin_atom_species_dict(E0s, cov, regressor.numbers)
        
    def _set_lin_atom_species_dict(self, E0s, covs, zs):
        atomic_energies_dict = {}
        for i, z in enumerate(zs):
            atomic_energies_dict[z] = E0s[i]
        self.linear_e0s = atomic_energies_dict
        
    def _set_linear_e0s(self):
        new_e0s = [np.zeros((max(self.numbers) + 1, 21)) for _ in range(len(self.__energy_methods__))]
        for z, e0 in self.linear_e0s.items():
            for i in range(len(self.__energy_methods__)):
                new_e0s[i][z, :] = e0[i]
        self.new_e0s = np.array(new_e0s)
    
    