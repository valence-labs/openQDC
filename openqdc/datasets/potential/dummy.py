import numpy as np

from openqdc.datasets.base import BaseDataset
from openqdc.methods import PotentialMethod


class Dummy(BaseDataset):
    """
    Dummy dataset for testing.
    """

    __name__ = "dummy"
    __energy_methods__ = [PotentialMethod.SVWN_DEF2_TZVP, PotentialMethod.PM6, PotentialMethod.GFN2_XTB]
    __force_mask__ = [False, True, True]
    __energy_unit__ = "kcal/mol"
    __distance_unit__ = "ang"
    __forces_unit__ = "kcal/mol/ang"

    energy_target_names = [f"energy{i}" for i in range(len(__energy_methods__))]

    force_target_names = [f"forces{i}" for i in range(len(__force_mask__))]
    __isolated_atom_energies__ = []
    __average_n_atoms__ = None

    def _post_init(self, overwrite_local_cache, energy_unit, distance_unit) -> None:
        self.setup_dummy()
        return super()._post_init(overwrite_local_cache, energy_unit, distance_unit)

    def setup_dummy(self):
        n_atoms = np.array([np.random.randint(1, 100) for _ in range(len(self))])
        position_idx_range = np.concatenate([[0], np.cumsum(n_atoms)]).repeat(2)[1:-1].reshape(-1, 2)
        atomic_inputs = np.concatenate(
            [
                np.concatenate(
                    [
                        # z, c, x, y, z
                        np.random.randint(1, 100, size=(size, 1)),
                        np.random.randint(-1, 2, size=(size, 1)),
                        np.random.randn(size, 3),
                    ],
                    axis=1,
                )
                for size in n_atoms
            ],
            axis=0,
        )  # (sum(n_atoms), 5)
        name = [f"dummy_{i}" for i in range(len(self))]
        subset = ["dummy" for i in range(len(self))]
        energies = np.random.rand(len(self), len(self.energy_methods))
        forces = np.concatenate([np.random.randn(size, 3, len(self.force_methods)) * 100 for size in n_atoms])
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

    def read_preprocess(self, overwrite_local_cache=False):
        return

    def is_preprocessed(self):
        return True

    def read_raw_entries(self):
        pass

    def __len__(self):
        return 9999
