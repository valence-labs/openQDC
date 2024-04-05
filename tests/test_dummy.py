"""Path hack to make tests work."""

import os

import numpy as np
import pytest

from openqdc.datasets.potential.dummy import Dummy  # noqa: E402
from openqdc.utils.io import get_local_cache
from openqdc.utils.package_utils import has_package

# start by removing any cached data
cache_dir = get_local_cache()
os.system(f"rm -rf {cache_dir}/dummy")


if has_package("torch"):
    import torch

if has_package("jax"):
    import jax

format_to_type = {
    "numpy": np.ndarray,
    "torch": torch.Tensor if has_package("torch") else None,
    "jax": jax.numpy.ndarray if has_package("jax") else None,
}


@pytest.fixture
def ds():
    return Dummy()


def test_dummy(ds):
    assert len(ds) > 10
    assert ds[100]


# def test_is_at_factory():
#     res = IsolatedAtomEnergyFactory.get("mp2/cc-pvdz")
#     assert len(res) == len(ISOLATED_ATOM_ENERGIES["mp2"]["cc-pvdz"])
#     res = IsolatedAtomEnergyFactory.get("PM6")
#     assert len(res) == len(ISOLATED_ATOM_ENERGIES["pm6"])
#     assert isinstance(res[("H", 0)], float)


@pytest.mark.parametrize("format", ["numpy", "torch", "jax"])
def test_array_format(format):
    if not has_package(format):
        pytest.skip(f"{format} is not installed, skipping test")

    ds = Dummy(array_format=format)

    keys = [
        "positions",
        "atomic_numbers",
        "charges",
        "energies",
        "forces",
        "e0",
        "formation_energies",
        "per_atom_formation_energies",
    ]

    data = ds[0]
    for key in keys:
        assert isinstance(data[key], format_to_type[format])


def test_transform():
    def custom_fn(bunch):
        # create new name
        bunch.new_key = bunch.name + bunch.subset
        return bunch

    ds = Dummy(transform=custom_fn)

    data = ds[0]

    assert "new_key" in data
    assert data["new_key"] == data["name"] + data["subset"]


def test_get_statistics(ds):
    stats = ds.get_statistics()

    keys = ["ForcesCalculatorStats", "FormationEnergyStats", "PerAtomFormationEnergyStats", "TotalEnergyStats"]
    assert all(k in stats for k in keys)


def test_energy_statistics_shapes(ds):
    stats = ds.get_statistics()

    num_methods = len(ds.energy_methods)

    formation_energy_stats = stats["FormationEnergyStats"]
    assert formation_energy_stats["mean"].shape == (1, num_methods)
    assert formation_energy_stats["std"].shape == (1, num_methods)

    per_atom_formation_energy_stats = stats["PerAtomFormationEnergyStats"]
    assert per_atom_formation_energy_stats["mean"].shape == (1, num_methods)
    assert per_atom_formation_energy_stats["std"].shape == (1, num_methods)

    total_energy_stats = stats["TotalEnergyStats"]
    assert total_energy_stats["mean"].shape == (1, num_methods)
    assert total_energy_stats["std"].shape == (1, num_methods)


def test_force_statistics_shapes(ds):
    stats = ds.get_statistics()
    num_force_methods = len(ds.force_methods)

    forces_stats = stats["ForcesCalculatorStats"]
    keys = ["mean", "std", "component_mean", "component_std", "component_rms"]
    assert all(k in forces_stats for k in keys)

    assert forces_stats["mean"].shape == (1, num_force_methods)
    assert forces_stats["std"].shape == (1, num_force_methods)
    assert forces_stats["component_mean"].shape == (3, num_force_methods)
    assert forces_stats["component_std"].shape == (3, num_force_methods)
    assert forces_stats["component_rms"].shape == (3, num_force_methods)
