import numpy as np  # noqa
from sklearn.utils import Bunch

from openqdc.datasets.base import BaseDataset
from openqdc.utils.atomization_energies import IsolatedAtomEnergyFactory
from numpy import array


class Dummy(BaseDataset):
    """
    Dummy dataset
    """

    __name__ = "dummy"
    __energy_methods__ = ["I_solved_the_schrodinger_equation_by_hand", "PM6"]
    __force_methods__ = ["I_made_up_random_forces", "writing_1_to_every_coordinate"]
    __energy_unit__ = "kcal/mol"
    __distance_unit__ = "ang"
    __forces_unit__ = "kcal/mol/ang"

    energy_target_names = [f"energy{i}" for i in range(len(__energy_methods__))]

    force_target_names = [f"forces{i}" for i in range(len(__force_methods__))]
    __isolated_atom_energies__ = []
    __average_n_atoms__ = 20

    @property
    def _stats(self):
        return {
            "formation": {
                "energy": {
                    "mean": array([-12.94348027, -9.83037297]),
                    "std": array([4.39971409, 3.3574188]),
                },
                "forces": {
                    "mean": array([0]),
                    "std": array([0]),
                    "components": {
                        "mean": array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
                        "std": array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
                        "rms": array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
                    },
                },
            },
            "total": {
                "energy": {
                    "mean": array([-89.44242, -1740.5336]),
                    "std": array([29.599571, 791.48663]),
                },
                "forces": {
                    "mean": array([[0.0], [0.0], [0.0]]),
                    "std": array([[0.0], [0.0], [0.0]]),
                    "rms": array([[0.0], [0.0], [0.0]]),
                },
            },
        }

    def __init__(self, energy_unit=None, distance_unit=None, cache_dir=None) -> None:
        try:
            super().__init__(energy_unit=energy_unit, distance_unit=distance_unit, cache_dir=cache_dir)

        except:  # noqa
            pass
        self._set_isolated_atom_energies()

    def is_preprocessed(self):
        return True

    def read_raw_entries(self):
        pass

    def __len__(self):
        return 9999

    def __getitem__(self, idx: int):
        shift = IsolatedAtomEnergyFactory.max_charge
        size = np.random.randint(1, 100)
        z = np.random.randint(1, 100, size)
        c = np.random.randint(-1, 2, size)
        return Bunch(
            positions=np.random.rand(size, 3) * 10,
            atomic_numbers=z,
            charges=c,
            e0=self.__isolated_atom_energies__[..., z, c + shift].T,
            energies=np.random.randn(len(self.__energy_methods__)),
            name="dummy_{}".format(idx),
            subset="dummy",
            forces=(np.random.randn(size, 3, len(self.__force_methods__)) * 100),
        )
