from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np
from ase.atoms import Atoms
from numpy import ndarray

from openqdc.utils.io import to_atoms
from openqdc.utils.package_utils import requires_package


class Descriptor(ABC):
    _model: Any

    def __init__(self, *, species: List[str], **kwargs):
        self.chemical_species = species
        self._model = self.instantiate_model(**kwargs)

    @property
    def model(self):
        return self._model

    @abstractmethod
    def instantiate_model(self, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def calculate(self, atoms: Atoms) -> ndarray:
        raise NotImplementedError

    def from_xyz(self, positions: np.ndarray, atomic_numbers: np.ndarray):
        atoms = to_atoms(positions, atomic_numbers)
        return self.calculate(atoms)

    def __str__(self):
        return str(self.__class__.__name__).lower()

    def __repr__(self):
        return str(self)


class SOAP(Descriptor):
    @requires_package("dscribe")
    def instantiate_model(self, **kwargs):
        from dscribe.descriptors import SOAP as SOAPModel

        r_cut = kwargs.pop("r_cut", 5.0)
        n_max = kwargs.pop("n_max", 8)
        l_max = kwargs.pop("l_max", 6)
        average = kwargs.pop("average", "inner")
        periodic = kwargs.pop("periodic", False)
        compression = kwargs.pop("compression", {"mode": "mu1nu1"})

        return SOAPModel(
            species=self.chemical_species,
            periodic=periodic,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            average=average,
            compression=compression,
        )

    def calculate(self, atoms: Atoms) -> ndarray:
        return self.model.create(atoms, centers=atoms.positions)


class MBTR(SOAP):
    @requires_package("dscribe")
    def instantiate_model(self, **kwargs):
        from dscribe.descriptors import MBTR as MBTRModel

        r_cut = kwargs.pop("r_cut", 5.0)
        geometry = kwargs.pop("geometry", {"function": "inverse_distance"})
        grid = kwargs.pop("grid", {"min": 0, "max": 1, "n": 100, "sigma": 0.1})
        weighting = kwargs.pop("weighting", {"function": "exp", "scale": 0.5, "threshold": 1e-3})
        normalization = kwargs.pop("normalization", "l2")
        periodic = kwargs.pop("periodic", False)

        return MBTRModel(
            species=self.chemical_species,
            periodic=periodic,
            r_cut=r_cut,
            geometry=geometry,
            grid=grid,
            weighting=weighting,
            normalization=normalization,
        )


class ACSF(SOAP):
    @requires_package("dscribe")
    def instantiate_model(self, **kwargs):
        from dscribe.descriptors import ACSF as ACSFModel

        r_cut = kwargs.pop("r_cut", 5.0)
        g2_params = kwargs.pop("g2_params", [[1, 1], [1, 2], [1, 3]])
        g3_params = kwargs.pop("g3_params", [[1], [1], [1], [2]])
        g4_params = kwargs.pop("g4_params", [[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]])
        g5_params = kwargs.pop("g5_params", [[1, 2, -1], [1, 1, 1], [-1, 1, 1], [1, 2, 1]])
        periodic = kwargs.pop("periodic", False)

        return ACSFModel(
            species=self.chemical_species,
            periodic=periodic,
            r_cut=r_cut,
            g2_params=g2_params,
            g3_params=g3_params,
            g4_params=g4_params,
            g5_params=g5_params,
        )


AVAILABLE_DESCRIPTORS = {
    str_name.lower(): cls
    for str_name, cls in globals().items()
    if isinstance(cls, type) and issubclass(cls, Descriptor) and str_name != "Descriptor"
}


def get_descriptor(name: str) -> Descriptor:
    try:
        return AVAILABLE_DESCRIPTORS[name.lower()]
    except KeyError:
        raise ValueError(f"Descriptor {name} not found. Available descriptors are {list(AVAILABLE_DESCRIPTORS.keys())}")
