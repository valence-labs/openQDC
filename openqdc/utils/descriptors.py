from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from ase.atoms import Atoms
from numpy import ndarray

from openqdc.utils.package_utils import requires_package


class Descriptor(ABC):

    @abstractmethod
    def calculate(self, atoms: Atoms) -> ndarray:
        raise NotImplementedError


@requires_package("dscribe")
@requires_package("datamol")
class SmoothOverlapOfAtomicPositions(Descriptor):
    import datamol as dm
    from dscribe.descriptors import SOAP

    def __init__(self, species, periodic, r_cut, n_max, l_max, average, compression):
        self.soap = SOAP(
            species=chemical_species,
            periodic=periodic,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            average=average,
            compression=compression,
        )

    def calculate(self, atoms: Atoms) -> ndarray:
        return self.soap.create(entry, centers=entry.positions)

    def __str__(self):
        return "SOAP"

    def __repr__(self):
        return "SOAP"
