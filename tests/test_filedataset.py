from io import StringIO

import pytest

from openqdc.datasets.io import XYZDataset
from openqdc.methods.enums import PotentialMethod


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
