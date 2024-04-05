import pytest

from openqdc import Dummy
from openqdc.utils.descriptors import ACSF, MBTR, SOAP, Descriptor


@pytest.fixture
def dummy():
    return Dummy()


@pytest.mark.parametrize("model", [SOAP, ACSF, MBTR])
def test_init(model):
    model = model(species=["H"])
    assert isinstance(model, Descriptor)


@pytest.mark.parametrize("model", [SOAP, ACSF, MBTR])
def test_descriptor(model, dummy):
    model = model(species=dummy.chemical_species)
    results = model.fit_transform([dummy.get_ase_atoms(i) for i in range(4)])
    assert len(results) == 4


@pytest.mark.parametrize("model", [SOAP, ACSF, MBTR])
def test_from_positions(model):
    model = model(species=["H"])
    _ = model.from_xyz([[0, 0, 0], [1, 1, 1]], [1, 1])


@pytest.mark.parametrize(
    "model,override", [(SOAP, {"r_cut": 3.0}), (ACSF, {"r_cut": 3.0}), (MBTR, {"normalize_gaussians": False})]
)
def test_overwrite(model, override, dummy):
    model = model(species=dummy.chemical_species, **override)
