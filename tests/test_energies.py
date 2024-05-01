import numpy as np
import pytest

from openqdc.datasets.energies import AtomEnergies, AtomEnergy
from openqdc.methods import PotentialMethod


class Container:
    __name__ = "container"
    __energy_methods__ = [PotentialMethod.WB97M_D3BJ_DEF2_TZVPPD]
    energy_methods = [str(PotentialMethod.WB97M_D3BJ_DEF2_TZVPPD)]
    refit_e0s = True

    def __init__(self, energy_type="formation"):
        self.energy_type = energy_type


@pytest.fixture
def physical_energies():
    dummy = Container()
    return AtomEnergies(dummy)


def test_atom_energies_object(physical_energies):
    assert isinstance(physical_energies, AtomEnergies)


def test_indexing(physical_energies):
    assert isinstance(physical_energies[6], AtomEnergy)
    assert isinstance(physical_energies[(6, 1)], AtomEnergy)
    assert isinstance(physical_energies[6, 1], AtomEnergy)
    assert isinstance(physical_energies[("C", 1)], AtomEnergy)
    assert isinstance(physical_energies["C", 1], AtomEnergy)
    assert physical_energies[("C", 1)] == physical_energies[(6, 1)]
    assert not physical_energies[("Cl", -2)] == physical_energies[(6, 1)]
    with pytest.raises(KeyError):
        physical_energies[("Cl", -6)]


def test_matrix(physical_energies):
    matrix = physical_energies.e0s_matrix
    assert len(matrix) == 1
    assert isinstance(matrix, np.ndarray)
    assert np.any(matrix)
