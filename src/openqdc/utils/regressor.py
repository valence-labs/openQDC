from typing import Optional, Tuple, Union
import numpy as np
from abc import ABC, abstractmethod
from loguru import logger
import pandas as pd

SubSampleFrac = Union[float, int]


class Solver(ABC):
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
    solver: Solver

    def __init__(
        self,
        energies: np.ndarray,
        atomic_numbers: np.ndarray,
        position_idx_range: np.ndarray,
        solver_type: str = "linear",
        stride: int = 1,
        subsample: Optional[SubSampleFrac] = None,
        *args,
        **kwargs,
    ):
        self.subsample = subsample
        self.stride = stride
        self.solver_type = solver_type
        self.energies = energies
        self.atomic_numbers = atomic_numbers
        self.numbers = pd.unique(atomic_numbers)
        self.position_idx_range = position_idx_range
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
        logger.warning(f"Unknown solver type {self.solver_type}, using linear regression.")
        return LinearSolver()

    def _prepare_inputs(self) -> Tuple[np.ndarray, np.ndarray]:
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
        return self.solver(self.X, self.y)

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
        X, y, y_mean=atom_standardization(X,y)
        E0s = np.linalg.lstsq(X, y, rcond=None)[0]        
        #Ainv=X.T @ X
        #residuals=np.var(y - np.dot(X, E0s))
        #np.sqrt(residuals * np.einsum("ij,kj,kl,li->i", Ainv, X, X, Ainv))
        return E0s, None 

class RidgeSolver(Solver):
    _regr_str = "RidgeRegression"

    @staticmethod
    def solve(X, y):
        X, y, y_mean=atom_standardization(X,y)
        A = X.T @ X
        dy = y - (np.sum(X, axis=1, keepdims=True) * y_mean).reshape(y.shape)
        Xy = X.T @ dy
        mean = np.linalg.solve(A, Xy)
        sigma2 = np.var(X @ mean - dy)
        Ainv = np.linalg.inv(A)
        cov = np.sqrt(sigma2 * np.einsum("ij,kj,kl,li->i", Ainv, X, X, Ainv))
        mean = mean + y_mean.reshape([-1])
        return mean, cov
