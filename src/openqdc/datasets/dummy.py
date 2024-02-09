import numpy as np

from openqdc.datasets.base import BaseDataset
from openqdc.utils.constants import NOT_DEFINED


class Dummy(BaseDataset):
    """
    Dummy dataset for testing.
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
    __average_n_atoms__ = None

    @property
    def _stats(self):
        return {
            "formation": {
                "energy": {
                    "mean": np.array([[-12.94348027, -9.83037297]]),
                    "std": np.array([[4.39971409, 3.3574188]]),
                },
                "forces": NOT_DEFINED,
            },
            "total": {
                "energy": {
                    "mean": np.array([[-89.44242, -1740.5336]]),
                    "std": np.array([[29.599571, 791.48663]]),
                },
                "forces": NOT_DEFINED,
            },
        }

    def __init__(
        self,
        energy_unit=None,
        distance_unit=None,
        cache_dir=None,
    ) -> None:
        try:
            super().__init__(energy_unit=energy_unit, distance_unit=distance_unit, cache_dir=cache_dir)

        except:  # noqa
            pass
        self._set_isolated_atom_energies()
        self.setup_dummy()
        
    def setup_dummy(self):
        n_atoms = np.array([np.random.randint(1, 100) for _ in range(len(self))])
        position_idx_range = np.concatenate([[0], np.cumsum(n_atoms)]).repeat(2)[1:-1].reshape(-1, 2)
        atomic_inputs = np.concatenate([np.concatenate([
            # z, c, x, y, z
            np.random.randint(1, 100, size=(size, 1)),
            np.random.randint(-1, 2, size=(size, 1)),
            np.random.randn(size, 3)
        ], axis=1) for size in n_atoms], axis=0) # (sum(n_atoms), 5)
        name=[f'dummy_{i}' for i in range(len(self))]
        subset=["dummy" for i in range(len(self))]
        energies = np.random.rand(len(self), len(self.__energy_methods__))
        forces = np.concatenate([
            np.random.randn(size, 3, len(self.__force_methods__)) * 100
            for size in n_atoms
        ])
        self.data = dict(
            n_atoms=n_atoms,
            position_idx_range=position_idx_range,
            name=name,
            atomic_inputs=atomic_inputs,
            subset=subset,
            energies=energies,
            forces=forces,
        )
        self.__average_nb_atoms__ = self.data["n_atoms"].mean()

    def is_preprocessed(self):
        return True

    def read_raw_entries(self):
        pass

    def __len__(self):
        return 9999
