"""Path hack to make tests work."""

from openqdc.datasets.potential.dummy import Dummy  # noqa: E402
from openqdc.utils.atomization_energies import (
    ISOLATED_ATOM_ENERGIES,
    IsolatedAtomEnergyFactory,
)


def test_dummy():
    ds = Dummy()
    assert len(ds) > 10
    assert ds[100]


def test_is_at_factory():
    res = IsolatedAtomEnergyFactory.get("mp2/cc-pvdz")
    assert len(res) == len(ISOLATED_ATOM_ENERGIES["mp2"]["cc-pvdz"])
    res = IsolatedAtomEnergyFactory.get("PM6")
    assert len(res) == len(ISOLATED_ATOM_ENERGIES["pm6"])
    assert isinstance(res[("H", 0)], float)
