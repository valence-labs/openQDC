import os
from io import StringIO

import numpy as np
import pytest

from openqdc.datasets.io import XYZDataset
from openqdc.methods.enums import PotentialMethod
from openqdc.utils.io import get_local_cache
from openqdc.utils.package_utils import has_package

if has_package("torch"):
    import torch

if has_package("jax"):
    import jax

format_to_type = {
    "numpy": np.ndarray,
    "torch": torch.Tensor if has_package("torch") else None,
    "jax": jax.numpy.ndarray if has_package("jax") else None,
}


@pytest.fixture(autouse=True)
def clean_before_run():
    # start by removing any cached data
    cache_dir = get_local_cache()
    os.system(f"rm -rf {cache_dir}/XYZDataset")
    yield


@pytest.fixture
def xyz_filelike():
    xyz_str = """3
Properties=species:S:1:pos:R:3:initial_charges:R:1:forces:R:3 energy=-10.0 pbc="F F F"
O      0.88581973       0.54890931      -3.39794898       0.00000000       0.01145078      -0.01124914       0.03187728
H      1.09592915      15.43154144       8.50078392       0.00000000      -0.02147313      -0.01223383       0.01807558
H     -1.68552792      14.76088047      11.56200695      -1.00000000       0.03393034      -0.02250720      -0.04456452
2
Properties=species:S:1:pos:R:3:initial_charges:R:1:forces:R:3 energy=-20.0 pbc="F F F"
C      1.34234893       4.15623617      -3.27245665       0.00000000       0.00179922      -0.03140596       0.01925333
C      0.11595206       5.01309919      -0.78672481       0.00000000       0.05754307       0.05001242      -0.02333626
    """
    return StringIO(xyz_str)


def test_xyz_dataset(xyz_filelike):
    ds = XYZDataset(path=[xyz_filelike], level_of_theory=PotentialMethod.B3LYP_6_31G_D)
    assert len(ds) == 2
    assert len(ds.numbers) == 3
    assert ds[1].energies == -20.0
    assert set(ds.chemical_species) == {"H", "O", "C"}


@pytest.mark.parametrize("format", ["numpy", "torch", "jax"])
def test_array_format(xyz_filelike, format):
    if not has_package(format):
        pytest.skip(f"{format} is not installed, skipping test")

    ds = XYZDataset(path=[xyz_filelike], array_format=format)

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
        assert isinstance(getattr(data, key), format_to_type[format])
