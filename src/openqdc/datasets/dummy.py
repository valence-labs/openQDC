import numpy as np  # noqa
from sklearn.utils import Bunch

from openqdc.datasets.base import BaseDataset


class Dummy(BaseDataset):
    """
    Dummy dataset
    """

    __name__ = "dummy"
    __energy_methods__ = ["I_solved_the_schrodinger_equation_by_hand"]
    __force_methods__ = ["I_made_up_random_forces"]
    __energy_unit__ = "kcal/mol"
    __distance_unit__ = "ang"
    __forces_unit__ = "kcal/mol/ang"

    energy_target_names = ["energy"]

    force_target_names = ["forces"]

    def __init__(self, energy_unit=None, distance_unit=None, cache_dir=None) -> None:
        try:
            super().__init__(energy_unit=energy_unit, distance_unit=distance_unit, cache_dir=cache_dir)
        except:  # noqa
            pass

    def read_raw_entries(self):
        pass

    def __len__(self):
        return 999999999

    def __getitem__(self, idx: int):
        size = np.random.randint(1, 250)
        z = np.random.randint(1, 100, size)
        return Bunch(
            positions=np.random.rand(size, 3) * 10,
            atomic_numbers=z,
            charges=np.random.randint(-1, 2, size),
            e0=np.zeros(size),
            energies=np.random.rand(1) * 100,
            name="dummy_{}".format(idx),
            subset="dummy",
            forces=np.random.rand(size, 3) * 100,
        )
