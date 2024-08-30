import os
from os.path import join as p_join

import numpy as np

from openqdc.datasets.base import BaseDataset
from openqdc.methods import PotentialMethod


def shape_atom_inputs(coords, atom_species):
    xs = np.stack((atom_species, np.zeros_like(atom_species)), axis=-1)
    return np.concatenate((xs, coords), axis=-1, dtype=np.float32)


def read_npz_entry(raw_path):
    samples = np.load(raw_path, allow_pickle=True)
    # get name of file without extension
    subset = os.path.basename(raw_path).split(".")[0]

    # atoms
    # coordinates
    coordinates = np.concatenate(samples["coordinates"])
    atom_species = np.concatenate(samples["atoms"]).ravel()
    names = list(map(lambda x: x.split("_")[0], samples["compounds"]))
    n_comps = len(names)

    # graphs
    # inchi
    # Etot
    # Eatomization
    res = dict(
        name=np.array(list(map(lambda x: x.split("_")[0], samples["compounds"]))),
        subset=np.array([subset] * n_comps),
        energies=samples["Etot"][:, None].astype(np.float64),
        atomic_inputs=shape_atom_inputs(coordinates, atom_species),
        n_atoms=np.array(list(map(lambda x: len(x), samples["coordinates"])), dtype=np.int32),
    )
    return res


# graphs is smiles
class VQM24(BaseDataset):
    """
    Vector-QM24 (VQM24) dataset consists of small organic and inorganic molecules with quantum mechanical
    properties calculated at wB97x-D3//cc-pVDZ level of theory. This leads to 258,242 unique constitutional
    isomers and 577,705 conformers of varying stoichiometries. Geometries are generated using GFN2-xTB, and
    relaxed with DFT method wB97x-D3/cc-pVDZ. The energy values are calculated with DFT method wB97x-D3/cc-pVDZ.

    Usage:
    ```python
    from openqdc.datasets import VQM24
    dataset = VQM24()
    ```

    Reference:
        https://arxiv.org/abs/2405.05961
    """

    __name__ = "vqm24"

    __energy_methods__ = [
        PotentialMethod.WB97X_D3_CC_PVDZ,  # "wB97x-D3/cc-pVDZ."
    ]

    energy_target_names = [
        "wB97x-D3/cc-pVDZ",
    ]
    # ωB97X-D3/cc-pVDZ
    __energy_unit__ = "hartree"
    __distance_unit__ = "ang"
    __forces_unit__ = "hartree/ang"
    __links__ = {
        f"{name}.npz": f"https://zenodo.org/records/11164951/files/{name}.npz?download=1"
        for name in ["DFT_all", "DFT_saddles", "DFT_uniques", "DMC"]
    }

    def read_raw_entries(self):
        samples = []
        for name in self.__links__:
            raw_path = p_join(self.root, f"{name}")
            samples.append(read_npz_entry(raw_path))
        return samples
