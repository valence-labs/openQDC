import os
from os.path import join as p_join

import numpy as np

from openqdc.datasets.base import BaseDataset
from openqdc.methods import PotentialMethod
from openqdc.utils import load_hdf5_file, read_qc_archive_h5
from openqdc.utils.io import get_local_cache


def read_ani2_h5(raw_path):
    h5f = load_hdf5_file(raw_path)
    samples = []
    for _, props in h5f.items():
        samples.append(extract_ani2_entries(props))
    return samples


def extract_ani2_entries(properties):
    coordinates = properties["coordinates"]
    species = properties["species"]
    forces = properties["forces"]
    energies = properties["energies"]
    n_atoms = coordinates.shape[1]
    n_entries = coordinates.shape[0]
    flattened_coordinates = coordinates[:].reshape((-1, 3))
    xs = np.stack((species[:].flatten(), np.zeros(flattened_coordinates.shape[0])), axis=-1)
    res = dict(
        name=np.array(["ANI2"] * n_entries),
        subset=np.array([str(n_atoms)] * n_entries),
        energies=energies[:].reshape((-1, 1)).astype(np.float64),
        atomic_inputs=np.concatenate((xs, flattened_coordinates), axis=-1, dtype=np.float32),
        n_atoms=np.array([n_atoms] * n_entries, dtype=np.int32),
        forces=forces[:].reshape(-1, 3, 1).astype(np.float32),
    )
    return res


class QM9(BaseDataset):
    """
    The ANI-1 dataset is a collection of 22 x 10^6 structural conformations from 57,000 distinct small
    organic molecules with energy labels calculated using DFT. The molecules
    contain 4 distinct atoms, C, N, O and H.

    Usage
    ```python
    from openqdc.datasets import ANI1
    dataset = ANI1()
    ```

    References:
    - ANI-1: https://www.nature.com/articles/sdata2017193
    - Github: https://github.com/aiqm/ANI1x_datasets
    """

    __name__ = "qm9"

    __energy_methods__ = [
        PotentialMethod.WB97X_6_31G_D,  # "wb97x/6-31g(d)"
    ]

    energy_target_names = [
        "ωB97x:6-31G(d) Energy",
    ]

    __energy_unit__ = "hartree"
    __distance_unit__ = "bohr"
    __forces_unit__ = "hartree/bohr"
    __links__ = {"ani1.hdf5.gz": "https://zenodo.org/record/3585840/files/214.hdf5.gz"}

    @property
    def root(self):
        return p_join(get_local_cache(), "qmx")

    @property
    def preprocess_path(self):
        path = p_join(self.root, "preprocessed", self.__name__)
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def config(self):
        assert len(self.__links__) > 0, "No links provided for fetching"
        return dict(dataset_name="qmx", links=self.__links__)

    def __smiles_converter__(self, x):
        """util function to convert string to smiles: useful if the smiles is
        encoded in a different format than its display format
        """
        return "-".join(x.decode("ascii").split("-")[:-1])

    @property
    def preprocess_path(self):
        path = p_join(self.root, "preprocessed", self.__name__)
        os.makedirs(path, exist_ok=True)
        return path

    def read_raw_entries(self):
        raw_path = p_join(self.root, f"{self.__name__}.h5.gz")
        samples = read_qc_archive_h5(raw_path, self.__name__, self.energy_target_names, self.force_target_names)
        return samples