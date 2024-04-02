from abc import ABC, abstractmethod
from typing import Any, List

import datamol as dm
import numpy as np
from ase.atoms import Atoms
from numpy import ndarray

from openqdc.utils.io import to_atoms
from openqdc.utils.package_utils import requires_package


class Descriptor(ABC):
    """
    Base class for all descriptors.
    Descriptors are used to transform 3D atomic structures into feature vectors.
    """

    _model: Any

    def __init__(self, *, species: List[str], **kwargs) -> None:
        """
        Parameters
        ----------
        species : List[str]
            List of chemical species for the descriptor embedding.
        kwargs : dict
            Additional keyword arguments to be passed to the descriptor model.
        """
        self.chemical_species = species
        self._model = self.instantiate_model(**kwargs)

    @property
    def model(self) -> Any:
        """Simple property that returns the model."""
        return self._model

    @abstractmethod
    def instantiate_model(self, **kwargs) -> Any:
        """
        Instantiate the descriptor model with the provided kwargs parameters
        and return it. The model will be stored in the _model attribute.
        If a package is required to instantiate the model, it should be checked
        using the requires_package decorator or in the method itself.

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments to be passed to the descriptor model.
        """
        raise NotImplementedError

    @abstractmethod
    def calculate(self, atoms: Atoms, **kwargs) -> ndarray:
        """
        Calculate the descriptor for a single given Atoms object.

        Parameters
        ----------
        atoms : Atoms
            Ase Atoms object to calculate the descriptor for.

        Returns
        -------
        ndarray
            ndarray containing the descriptor values
        """
        raise NotImplementedError

    def fit_transform(self, atoms: List[Atoms], **kwargs) -> List[ndarray]:
        """Parallelized version of the calculate method.
        Parameters
        ----------
        atoms : List[Atoms]
            List of Ase Atoms object to calculate the descriptor for.
        kwargs : dict
            Additional keyword arguments to be passed to the datamol parallelized model.

        Returns
        -------
        List[ndarray]
            List of ndarray containing the descriptor values
        """

        descr_values = dm.parallelized(self.calculate, atoms, scheduler="threads", **kwargs)
        return descr_values

    def from_xyz(self, positions: np.ndarray, atomic_numbers: np.ndarray) -> ndarray:
        """
        Calculate the descriptor from positions and atomic numbers of a single structure.

        Parameters
        ----------
        positions : np.ndarray (n_atoms, 3)
            Positions of the chemical structure.
        atomic_numbers : np.ndarray (n_atoms,)
            Atomic numbers of the chemical structure.

        Returns
        -------
        ndarray
            ndarray containing the descriptor values
        """
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

    def calculate(self, atoms: Atoms, **kwargs) -> ndarray:
        kwargs = kwargs or {}
        if "centers" not in kwargs:
            # add a center to every atom
            kwargs["centers"] = list(range(len(atoms.positions)))
        return self.model.create(atoms, **kwargs)


class ACSF(SOAP):
    @requires_package("dscribe")
    def instantiate_model(self, **kwargs):
        from dscribe.descriptors import ACSF as ACSFModel

        r_cut = kwargs.pop("r_cut", 5.0)
        g2_params = kwargs.pop("g2_params", [[1, 1], [1, 2], [1, 3]])
        g3_params = kwargs.pop("g3_params", [1, 1, 2, -1])
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


class MBTR(SOAP):
    @requires_package("dscribe")
    def instantiate_model(self, **kwargs):
        from dscribe.descriptors import MBTR as MBTRModel

        geometry = kwargs.pop("geometry", {"function": "inverse_distance"})
        grid = kwargs.pop("grid", {"min": 0, "max": 1, "n": 100, "sigma": 0.1})
        weighting = kwargs.pop("weighting", {"function": "exp", "r_cut": 5, "threshold": 1e-3})
        normalization = kwargs.pop("normalization", "l2")
        normalize_gaussians = kwargs.pop("normalize_gaussians", True)
        periodic = kwargs.pop("periodic", False)

        return MBTRModel(
            species=self.chemical_species,
            periodic=periodic,
            geometry=geometry,
            grid=grid,
            weighting=weighting,
            normalize_gaussians=normalize_gaussians,
            normalization=normalization,
        )

    def calculate(self, atoms: Atoms, **kwargs) -> ndarray:
        return self.model.create(atoms, **kwargs)


# Dynamic mapping of available descriptors
AVAILABLE_DESCRIPTORS = {
    str_name.lower(): cls
    for str_name, cls in globals().items()
    if isinstance(cls, type) and issubclass(cls, Descriptor) and str_name != "Descriptor"  # Exclude the base class
}


def get_descriptor(name: str) -> Descriptor:
    """
    Utility function that returns a descriptor class from its name.
    """
    try:
        return AVAILABLE_DESCRIPTORS[name.lower()]
    except KeyError:
        raise ValueError(f"Descriptor {name} not found. Available descriptors are {list(AVAILABLE_DESCRIPTORS.keys())}")
