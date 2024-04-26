import pickle as pkl
from os.path import join as p_join

import numpy as np
from loguru import logger

from openqdc.datasets.base import BaseDataset
from openqdc.methods import PotentialMethod


class Dummy(BaseDataset):
    """
    Dummy dataset for testing.
    """

    __name__ = "dummy"
    __energy_methods__ = [PotentialMethod.GFN2_XTB, PotentialMethod.WB97X_D_DEF2_SVP, PotentialMethod.PM6]
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
        n_atoms = np.array([np.random.randint(2, 100) for _ in range(len(self))])
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


class PredefinedDataset(BaseDataset):
    __name__ = "predefineddataset"
    __energy_methods__ = [PotentialMethod.WB97M_D3BJ_DEF2_TZVPPD]  # "wb97m-d3bj/def2-tzvppd"]
    __force_mask__ = [True]
    __energy_unit__ = "hartree"
    __distance_unit__ = "bohr"
    __forces_unit__ = "hartree/bohr"
    force_target_names = __energy_methods__
    energy_target_names = __energy_methods__

    @property
    def preprocess_path(self, overwrite_local_cache=False):
        from os.path import join as p_join

        from openqdc import get_project_root

        return p_join(get_project_root(), "tests", "files", self.__name__, "preprocessed")

    def is_preprocessed(self):
        return True

    def read_raw_entries(self):
        pass

    def read_preprocess(self, overwrite_local_cache=False):
        logger.info("Reading preprocessed data.")
        logger.info(
            f"Dataset {self.__name__} with the following units:\n\
                     Energy: {self.energy_unit},\n\
                     Distance: {self.distance_unit},\n\
                     Forces: {self.force_unit if self.force_methods else 'None'}"
        )
        self.data = {}
        for key in self.data_keys:
            print(key, self.data_shapes[key], self.data_types[key])
            filename = p_join(self.preprocess_path, f"{key}.mmap")
            self.data[key] = np.memmap(filename, mode="r", dtype=self.data_types[key]).reshape(*self.data_shapes[key])

        filename = p_join(self.preprocess_path, "props.pkl")
        with open(filename, "rb") as f:
            tmp = pkl.load(f)
            for key in ["name", "subset", "n_atoms"]:
                x = tmp.pop(key)
                if len(x) == 2:
                    self.data[key] = x[0][x[1]]
                else:
                    self.data[key] = x

        for key in self.data:
            logger.info(f"Loaded {key} with shape {self.data[key].shape}, dtype {self.data[key].dtype}")
