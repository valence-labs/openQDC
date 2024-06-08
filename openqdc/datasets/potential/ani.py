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


class ANI1(BaseDataset):
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

    __name__ = "ani1"

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
        return p_join(get_local_cache(), "ani")

    @property
    def config(self):
        assert len(self.__links__) > 0, "No links provided for fetching"
        return dict(dataset_name="ani", links=self.__links__)

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


class ANI1CCX(ANI1):
    """
    ANI1-CCX is a dataset of 500k conformers subsampled from the 5.5M conformers of ANI-1X dataset. The selected
    conformations are then labelled using a high accuracy CCSD(T)*/CBS method.

    Usage
    ```python
    from openqdc.datasets import ANI1CCX
    dataset = ANI1CCX()
    ```

    References:
    - ANI-1ccx: https://doi.org/10.1038/s41467-019-10827-4
    - Github: https://github.com/aiqm/ANI1x_datasets
    """

    __name__ = "ani1ccx"
    __energy_unit__ = "hartree"
    __distance_unit__ = "ang"
    __forces_unit__ = "hartree/ang"

    __energy_methods__ = [
        PotentialMethod.NONE,  # "ccsd(t)/cbs",
        PotentialMethod.NONE,  # "ccsd(t)/cc-pvdz",
        PotentialMethod.NONE,  # "ccsd(t)/cc-pvtz",
        PotentialMethod.NONE,  # "tccsd(t)/cc-pvdz",
    ]

    energy_target_names = [
        "CCSD(T)*:CBS Total Energy",
        "NPNO-CCSD(T):cc-pVDZ Correlation Energy",
        "NPNO-CCSD(T):cc-pVTZ Correlation Energy",
        "TPNO-CCSD(T):cc-pVDZ Correlation Energy",
    ]
    force_target_names = []
    __links__ = {"ani1x.hdf5.gz": "https://zenodo.org/record/4081694/files/292.hdf5.gz"}

    def __smiles_converter__(self, x):
        """util function to convert string to smiles: useful if the smiles is
        encoded in a different format than its display format
        """
        return x


class ANI1X(ANI1):
    """
    The ANI-1X dataset consists of ANI-1 molecules + some molecules added using active learning, which leads to
    a total of 5,496,771 conformers with 63,865 unique molecules.

    Usage
    ```python
    from openqdc.datasets import ANI1X
    dataset = ANI1X()
    ```

    References:
    - ANI-1x: https://doi.org/10.1063/1.5023802
    - Github: https://github.com/aiqm/ANI1x_datasets
    """

    __name__ = "ani1x"
    __energy_unit__ = "hartree"
    __distance_unit__ = "ang"
    __forces_unit__ = "hartree/ang"

    __energy_methods__ = [
        "hf/cc-pvdz",
        "hf/cc-pvqz",
        "hf/cc-pvtz",
        "mp2/cc-pvdz",
        "mp2/cc-pvqz",
        "mp2/cc-pvtz",
        "wb97x/6-31g(d)",
        "wb97x/cc-pvtz",
    ]

    energy_target_names = [
        "HF:cc-pVDZ Total Energy",
        "HF:cc-pVQZ Total Energy",
        "HF:cc-pVTZ Total Energy",
        "MP2:cc-pVDZ Correlation Energy",
        "MP2:cc-pVQZ Correlation Energy",
        "MP2:cc-pVTZ Correlation Energy",
        "wB97x:6-31G(d) Total Energy",
        "wB97x:def2-TZVPP Total Energy",
    ]

    force_target_names = [
        "wB97x:6-31G(d) Atomic Forces",
        "wB97x:def2-TZVPP Atomic Forces",
    ]

    __force_mask__ = [False, False, False, False, False, False, True, True]
    __links__ = {"ani1ccx.hdf5.gz": "https://zenodo.org/record/4081692/files/293.hdf5.gz"}

    def convert_forces(self, x):
        return super().convert_forces(x) * 0.529177249  # correct the Dataset error

    def __smiles_converter__(self, x):
        """util function to convert string to smiles: useful if the smiles is
        encoded in a different format than its display format
        """
        return x


class ANI1CCX_V2(ANI1CCX):
    __name__ = "ani1ccx_v2"

    __energy_methods__ = ANI1CCX.__energy_methods__ + [PotentialMethod.PM6, PotentialMethod.GFN2_XTB]
    energy_target_names = ANI1CCX.energy_target_names + ["PM6", "GFN2"]


class ANI2(ANI1):
    """ """

    __name__ = "ani2"
    __energy_unit__ = "hartree"
    __distance_unit__ = "ang"
    __forces_unit__ = "hartree/ang"

    __energy_methods__ = [
        # PotentialMethod.NONE,  # "b973c/def2mtzvp",
        PotentialMethod.WB97X_6_31G_D,  # "wb97x/631gd", # PAPER DATASET
        # PotentialMethod.NONE,  # "wb97md3bj/def2tzvpp",
        # PotentialMethod.NONE,  # "wb97mv/def2tzvpp",
        # PotentialMethod.NONE,  # "wb97x/def2tzvpp",
    ]

    energy_target_names = [
        # "b973c/def2mtzvp",
        "wb97x/631gd",
        # "wb97md3bj/def2tzvpp",
        # "wb97mv/def2tzvpp",
        # "wb97x/def2tzvpp",
    ]

    force_target_names = ["wb97x/631gd"]  # "b973c/def2mtzvp",

    __force_mask__ = [True]
    __links__ = {  # "ANI-2x-B973c-def2mTZVP.tar.gz": "https://zenodo.org/records/10108942/files/ANI-2x-B973c-def2mTZVP.tar.gz?download=1",  # noqa
        # "ANI-2x-wB97MD3BJ-def2TZVPP.tar.gz": "https://zenodo.org/records/10108942/files/ANI-2x-wB97MD3BJ-def2TZVPP.tar.gz?download=1", # noqa
        # "ANI-2x-wB97MV-def2TZVPP.tar.gz": "https://zenodo.org/records/10108942/files/ANI-2x-wB97MV-def2TZVPP.tar.gz?download=1", # noqa
        "ANI-2x-wB97X-631Gd.tar.gz": "https://zenodo.org/records/10108942/files/ANI-2x-wB97X-631Gd.tar.gz?download=1",  # noqa
        # "ANI-2x-wB97X-def2TZVPP.tar.gz": "https://zenodo.org/records/10108942/files/ANI-2x-wB97X-def2TZVPP.tar.gz?download=1", # noqa
    }

    def __smiles_converter__(self, x):
        """util function to convert string to smiles: useful if the smiles is
        encoded in a different format than its display format
        """
        return x

    def read_raw_entries(self):
        samples = []
        for lvl_theory in self.__links__.keys():
            raw_path = p_join(self.root, "final_h5", f"{lvl_theory.split('.')[0]}.h5")
            samples.extend(read_ani2_h5(raw_path))
        return samples
