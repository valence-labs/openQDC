from os.path import join as p_join

import numpy as np
from openqdc.datasets.base import BaseDataset
from openqdc.utils.constants import NB_ATOMIC_FEATURES
from openqdc.utils.io import load_hdf5_file
from tqdm import tqdm


def read_record(r, group):
    res = []
    ikeys = ["", "/product", "/reactant", "/transition_state"]
    for rxn in r.keys():
        x = r[rxn]["atomic_numbers"][:]
        positions = np.concatenate([r[f"{rxn}{k}/positions"][:] for k in ikeys], axis=0)
        energies = np.concatenate([r[f"{rxn}{k}/wB97x_6-31G(d).energy"][:] for k in ikeys], axis=0)
        forces = np.concatenate([r[f"{rxn}{k}/wB97x_6-31G(d).forces"][:] for k in ikeys], axis=0)

        n_confs = positions.shape[0]
        atomic_numbers = x.reshape(1, -1).repeat(n_confs, axis=0)[..., None]
        atomic_charges = np.zeros_like(atomic_numbers)
        atomic_inputs = np.concatenate((atomic_numbers, atomic_charges, positions), axis=-1, dtype=np.float32)

        res.append(
            dict(
                name=np.array([rxn] * n_confs),
                subset=np.array([group] * n_confs),
                energies=energies.astype(np.float32).reshape(-1, 1),
                forces=forces.astype(np.float32).reshape(-1, 3),
                atomic_inputs=atomic_inputs.astype(np.float32).reshape(-1, NB_ATOMIC_FEATURES),
                n_atoms=np.array([atomic_numbers.shape[1]] * n_confs, dtype=np.int32),
            )
        )

    return res


class Transition1X(BaseDataset):
    """
    The Transition1x dataset contains structures from 10k organic reaction pathways of various types.
    It contains DFT energy and force labels for 9.6 mio. conformers calculated at the
    wB97x/6-31-G(d) level of theory.

    Usage:
    ```python
    from openqdc.datasets import Transition1X
    dataset = Transition1X()
    ```

    References:
    - https://www.nature.com/articles/s41597-022-01870-w
    - https://gitlab.com/matschreiner/Transition1x
    """

    __name__ = "transition1x"

    __energy_methods__ = [
        "wb97x/6-31G(d)",
    ]

    energy_target_names = [
        "wB97x_6-31G(d).energy",
    ]

    __force_methods__ = [
        "wb97x/6-31G(d)",
    ]

    force_target_names = [
        "wB97x_6-31G(d).forces",
    ]

    __energy_unit__ = "hartree"
    __distance_unit__ = "ang"
    __forces_unit__ = "hartree/ang"

    def read_raw_entries(self):
        raw_path = p_join(self.root, "Transition1x.h5")
        f = load_hdf5_file(raw_path)["data"]

        res = sum([read_record(f[g], group=g) for g in tqdm(f)], [])  # don't use parallelized here
        return res
