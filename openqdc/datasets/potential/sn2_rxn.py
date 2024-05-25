from os.path import join as p_join

import numpy as np
from tqdm import tqdm

from openqdc.datasets.base import BaseDataset
from openqdc.methods import PotentialMethod


def extract_npz_entry(data):
    n_entries = data["N"].shape[0]

    nuclear_charges_full = data["Z"]
    n_atoms_full = data["N"]
    coors_full = data["R"]
    forces_full = data["F"]
    energies_full = data["E"]

    entries = []
    for idx in tqdm(range(n_entries)):
        n_atoms = n_atoms_full[idx]
        energies = energies_full[idx]
        nuclear_charges = nuclear_charges_full[idx, :n_atoms][:, None]
        coords = coors_full[idx, :n_atoms, :].reshape(-1, 3)
        forces = forces_full[idx, :n_atoms, :]
        res = dict(
            name=np.array(["SN2RXN"]),
            subset=np.array(["SN2RXN"]),
            energies=energies.reshape(-1, 1).astype(np.float64),
            forces=forces.reshape(-1, 3, 1).astype(np.float32),
            atomic_inputs=np.concatenate(
                (nuclear_charges, np.zeros_like(nuclear_charges), coords), axis=-1, dtype=np.float32
            ),
            n_atoms=np.array([n_atoms], dtype=np.int32),
        )
        entries.append(res)
    return entries


class SN2RXN(BaseDataset):
    """
    This dataset probes chemical reactions of methyl halides with halide anions, i.e.
    X- + CH3Y -> CH3X +  Y-, and contains structures for all possible combinations of
    X,Y = F, Cl, Br, I. It contains energy and forces for 452709 conformations calculated
    at the DSD-BLYP-D3(BJ)/def2-TZVP level of theory.

    Usage:
    ```python
    from openqdc.datasets import SN2RXN
    dataset = SN2RXN()
    ```

    References:
    - https://doi.org/10.1021/acs.jctc.9b00181
    - https://zenodo.org/records/2605341
    """

    __name__ = "sn2_rxn"

    __energy_methods__ = [
        PotentialMethod.DSD_BLYP_D3_BJ_DEF2_TZVP
        # "dsd-blyp-d3(bj)/def2-tzvp",
    ]
    __energy_unit__ = "ev"
    __distance_unit__ = "ang"
    __forces_unit__ = "ev/ang"
    __links__ = {"sn2_rxn.npz": "https://zenodo.org/records/2605341/files/sn2_reactions.npz"}

    energy_target_names = [
        # TODO: We need to revalidate this to make sure that is not atomization energies.
        "DSD-BLYP-D3(BJ):def2-TZVP Atomization Energy",
    ]

    __force_mask__ = [True]

    force_target_names = [
        "DSD-BLYP-D3(BJ):def2-TZVP Gradient",
    ]

    def read_raw_entries(self):
        raw_path = p_join(self.root, "sn2_rxn.npz")
        data = np.load(raw_path)
        samples = extract_npz_entry(data)

        return samples
