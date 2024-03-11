"""Isolated Atom Energies regression utilities."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

SubSampleFrac = Union[float, int]


def non_nan_idxs(array):
    """
    Return non nan indices of an array.
    """
    return np.where(~np.isnan(array))[0]


class Solver(ABC):
    """Abstract class for regression solvers."""

    _regr_str: str

    @staticmethod
    @abstractmethod
    def solve(X, Y):
        pass

    def __call__(self, X, Y):
        return self.solve(X, Y)

    def __str__(self):
        return self._regr_str

    def __repr__(self):
        return str(self)


class Regressor:
    """Regressor class for preparing and solving regression problem for isolated atom energies."""

    solver: Solver

    def __init__(
        self,
        energies: np.ndarray,
        atomic_numbers: np.ndarray,
        position_idx_range: np.ndarray,
        solver_type: str = "linear",
        stride: int = 1,
        subsample: Optional[SubSampleFrac] = None,
        remove_nan: bool = True,
        *args,
        **kwargs,
    ):
        """
        Parameters
        ----------
        energies
            numpy array of energies in the shape (n_samples, n_energy_methods)
        atomic_numbers
            numpy array of atomic numbers in the shape (n_atoms,)
        position_idx_range
            array of shape (n_samples, 2) containing the start and end indices of the atoms in the dataset
        stride
            Stride to use for the regression.
        subsample
            Sumsample the dataset. If a float, it is interpreted as a fraction of the dataset to use.
            If >1 it is interpreted as the number of samples to use.
        remove_nan
            Sanitize the dataset by removing energies samples with NaN values.
        """
        self.subsample = subsample
        self.stride = stride
        self.solver_type = solver_type.lower()
        self.energies = energies
        self.atomic_numbers = atomic_numbers
        self.numbers = pd.unique(atomic_numbers)
        self.position_idx_range = position_idx_range
        self.remove_nan = remove_nan
        self.hparams = {
            "subsample": subsample,
            "stride": stride,
            "solver_type": solver_type,
        }
        self._post_init()

    @classmethod
    def from_openqdc_dataset(cls, dataset, *args, **kwargs):
        energies = dataset.data["energies"]
        position_idx_range = dataset.data["position_idx_range"]
        atomic_numbers = dataset.data["atomic_inputs"][:, 0].astype("int32")
        return cls(energies, atomic_numbers, position_idx_range, *args, **kwargs)

    def _post_init(self):
        if self.subsample is not None:
            self._downsample()
        self._prepare_inputs()
        self.solver = self._get_solver()

    def update_hparams(self, hparams):
        self.hparams.update(hparams)

    def _downsample(self):
        if self.subsample < 1:
            idxs = np.arange(self.energies.shape[0])
            np.random.shuffle(idxs)
            idxs = idxs[: int(self.energies.shape[0] * self.subsample)]
            self.energies = self.energies[:: int(1 / self.subsample)]
            self.position_idx_range = self.position_idx_range[:: int(1 / self.subsample)]
        else:
            idxs = np.random.randint(0, self.energies.shape[0], int(self.subsample))
            self.energies = self.energies[idxs]
            self.position_idx_range = self.position_idx_range[idxs]
        self.update_hparams({"idxs": idxs})

    def _get_solver(self):
        if self.solver_type == "linear":
            return LinearSolver()
        elif self.solver_type == "ridge":
            return RidgeSolver()
        logger.warning(f"Unknown solver type {self.solver_type}, defaulting to linear regression.")
        return LinearSolver()

    def _prepare_inputs(self) -> Tuple[np.ndarray, np.ndarray]:
        logger.info("Preparing inputs for regression.")
        len_train = self.energies.shape[0]
        len_zs = len(self.numbers)
        A = np.zeros((len_train, len_zs))[:: self.stride]
        B = self.energies[:: self.stride]
        for i, ij in enumerate(self.position_idx_range[:: self.stride]):
            tmp = self.atomic_numbers[ij[0] : ij[1]]
            for j, z in enumerate(self.numbers):
                A[i, j] = np.count_nonzero(tmp == z)
        self.X = A
        self.y = B

    def solve(self):
        logger.info(f"Solving regression with {self.solver}.")
        E0_list, cov_list = [], []
        for energy_idx in range(self.y.shape[1]):
            if self.remove_nan:
                idxs = non_nan_idxs(self.y[:, energy_idx])
                X, y = self.X[idxs], self.y[idxs, energy_idx]
            else:
                X, y = self.X, self.y[:, energy_idx]
            E0s, cov = self.solver(X, y)
            E0_list.append(E0s)
            cov_list.append(cov)
        return np.vstack(E0_list).T, np.vstack(cov_list).T

    def __call__(self):
        return self.solve()


def atom_standardization(X, y):
    X_norm = X.sum()
    X = X / X_norm
    y = y / X_norm
    y_mean = y.sum() / X.sum()
    return X, y, y_mean


class LinearSolver(Solver):
    _regr_str = "LinearRegression"

    @staticmethod
    def solve(X, y):
        X, y, y_mean = atom_standardization(X, y)
        E0s = np.linalg.lstsq(X, y, rcond=None)[0]
        return E0s, None


class RidgeSolver(Solver):
    _regr_str = "RidgeRegression"

    @staticmethod
    def solve(X, y):
        X, y, y_mean = atom_standardization(X, y)
        A = X.T @ X
        dy = y - (np.sum(X, axis=1, keepdims=True) * y_mean).reshape(y.shape)
        Xy = X.T @ dy
        mean = np.linalg.solve(A, Xy)
        sigma2 = np.var(X @ mean - dy)
        Ainv = np.linalg.inv(A)
        cov = np.sqrt(sigma2 * np.einsum("ij,kj,kl,li->i", Ainv, X, X, Ainv))
        mean = mean + y_mean.reshape([-1])
        return mean, cov