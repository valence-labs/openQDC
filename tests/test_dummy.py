"""Path hack to make tests work."""

import pytest

from openqdc.datasets.interaction.dummy import DummyInteraction  # noqa: E402
from openqdc.datasets.potential.dummy import Dummy  # noqa: E402


@pytest.fixture
def dummy():
    return Dummy()


@pytest.fixture
def dummy_interaction():
    return DummyInteraction()


@pytest.mark.parametrize("cls", ["dummy", "dummy_interaction"])
def test_basic(cls, request):
    # init
    ds = request.getfixturevalue(cls)

    # len
    assert len(ds) == 9999

    # __getitem__
    assert ds[0]


@pytest.mark.parametrize("cls", ["dummy", "dummy_interaction"])
@pytest.mark.parametrize(
    "normalization",
    [
        "formation",
        "total",
        # "residual_regression",
        # "per_atom_formation",
        # "per_atom_residual_regression"
    ],
)
def test_stats(cls, normalization, request):
    ds = request.getfixturevalue(cls)

    stats = ds.get_statistics(normalization=normalization)
    assert stats is not None


# def test_is_at_factory():
#     res = IsolatedAtomEnergyFactory.get("mp2/cc-pvdz")
#     assert len(res) == len(ISOLATED_ATOM_ENERGIES["mp2"]["cc-pvdz"])
#     res = IsolatedAtomEnergyFactory.get("PM6")
#     assert len(res) == len(ISOLATED_ATOM_ENERGIES["pm6"])
#     assert isinstance(res[("H", 0)], float)
