"""Path hack to make tests work."""

import os

import numpy as np
import pytest

from openqdc.datasets.interaction.dummy import DummyInteraction  # noqa: E402
from openqdc.datasets.potential.dummy import Dummy  # noqa: E402
from openqdc.utils.io import get_local_cache
from openqdc.utils.package_utils import has_package


# start by removing any cached data
@pytest.fixture(autouse=True)
def clean_before_run():
    # start by removing any cached data
    cache_dir = get_local_cache()
    os.system(f"rm -rf {cache_dir}/dummy")
    os.system(f"rm -rf {cache_dir}/dummy_interaction")
    yield


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
def dummy():
    return Dummy()


@pytest.fixture
def dummy_interaction():
    return DummyInteraction()


@pytest.mark.parametrize("ds", ["dummy", "dummy_interaction"])
def test_dummy(ds, request):
    ds = request.getfixturevalue(ds)
    assert ds is not None
    assert len(ds) == 9999
    assert ds[100]


@pytest.mark.parametrize("interaction_ds", [False, True])
@pytest.mark.parametrize("format", ["numpy", "torch", "jax"])
def test_dummy_array_format(interaction_ds, format):
    if not has_package(format):
        pytest.skip(f"{format} is not installed, skipping test")

    ds = DummyInteraction(array_format=format) if interaction_ds else Dummy(array_format=format)

    keys = [
        "positions",
        "atomic_numbers",
        "charges",
        "energies",
        "forces",
        "e0",
    ]
    if not interaction_ds:
        # additional keys returned from the potential dataset
        keys.extend(["formation_energies", "per_atom_formation_energies"])

    data = ds[0]
    for key in keys:
        if data[key] is None:
            continue
        assert isinstance(data[key], format_to_type[format])


@pytest.mark.parametrize("interaction_ds", [False, True])
def test_transform(interaction_ds):
    def custom_fn(bunch):
        # create new name
        bunch.new_key = bunch.name + bunch.subset
        return bunch

    ds = DummyInteraction(transform=custom_fn) if interaction_ds else Dummy(transform=custom_fn)

    data = ds[0]

    assert "new_key" in data
    assert data["new_key"] == data["name"] + data["subset"]


@pytest.mark.parametrize("ds", ["dummy", "dummy_interaction"])
def test_get_statistics(ds, request):
    ds = request.getfixturevalue(ds)
    stats = ds.get_statistics()

    keys = ["ForcesCalculatorStats", "FormationEnergyStats", "PerAtomFormationEnergyStats", "TotalEnergyStats"]
    assert all(k in stats for k in keys)


@pytest.mark.parametrize("ds", ["dummy", "dummy_interaction"])
def test_energy_statistics_shapes(ds, request):
    ds = request.getfixturevalue(ds)
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


@pytest.mark.parametrize("ds", ["dummy", "dummy_interaction"])
def test_force_statistics_shapes(ds, request):
    ds = request.getfixturevalue(ds)
    stats = ds.get_statistics()
    num_force_methods = len(ds.force_methods)

    forces_stats = stats["ForcesCalculatorStats"]
    keys = ["mean", "std", "component_mean", "component_std", "component_rms"]
    assert all(k in forces_stats for k in keys)

    if len(ds.force_methods) > 0:
        assert forces_stats["mean"].shape == (1, num_force_methods)
        assert forces_stats["std"].shape == (1, num_force_methods)
        assert forces_stats["component_mean"].shape == (3, num_force_methods)
        assert forces_stats["component_std"].shape == (3, num_force_methods)
        assert forces_stats["component_rms"].shape == (3, num_force_methods)


@pytest.mark.parametrize("interaction_ds", [False, True])
@pytest.mark.parametrize("format", ["numpy", "torch", "jax"])
def test_stats_array_format(interaction_ds, format):
    if not has_package(format):
        pytest.skip(f"{format} is not installed, skipping test")

    ds = DummyInteraction(array_format=format) if interaction_ds else Dummy(array_format=format)
    stats = ds.get_statistics()

    for key in stats.keys():
        for k, v in stats[key].items():
            if v is None:
                continue
            assert isinstance(v, format_to_type[format])
