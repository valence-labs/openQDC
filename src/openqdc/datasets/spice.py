from os.path import join as p_join

import datamol as dm
import numpy as np
from tqdm import tqdm

from openqdc.datasets.base import BaseDataset
from openqdc.utils import load_hdf5_file
from openqdc.utils.constants import BOHR2ANG, MAX_ATOMIC_NUMBER
from openqdc.utils.molecule import get_atomic_number_and_charge


def read_record(r):
    smiles = r["smiles"].asstr()[0]
    subset = r["subset"][0].decode("utf-8")
    n_confs = r["conformations"].shape[0]
    x = get_atomic_number_and_charge(dm.to_mol(smiles, add_hs=True))
    positions = r["conformations"][:] * BOHR2ANG

    res = dict(
        name=np.array([smiles] * n_confs),
        subset=np.array([Spice.subset_mapping[subset]] * n_confs),
        energies=r[Spice.energy_target_names[0]][:][:, None].astype(np.float32),
        forces=r[Spice.force_target_names[0]][:].reshape(-1, 3, 1) / BOHR2ANG * (-1.0),  # forces -ve of energy gradient
        atomic_inputs=np.concatenate(
            (x[None, ...].repeat(n_confs, axis=0), positions), axis=-1, dtype=np.float32
        ).reshape(-1, 5),
        n_atoms=np.array([x.shape[0]] * n_confs, dtype=np.int32),
    )

    return res


class Spice(BaseDataset):
    """
    Spice Dataset consists of 1.1 million conformations for a diverse set of 19k unique molecules consisting of
    small molecules, dimers, dipeptides, and solvated amino acids. It consists of both forces and energies calculated
    at {\omega}B97M-D3(BJ)/def2-TZVPPD level of theory.

    Usage:
    ```python
    from openqdc.datasets import Spice
    dataset = Spice()
    ```

    References:
    - https://arxiv.org/abs/2209.10702
    - https://github.com/openmm/spice-dataset
    """

    __name__ = "spice"
    __energy_methods__ = ["wb97x_tz"]
    __force_methods__ = ["wb97x_tz"]

    energy_target_names = ["dft_total_energy"]

    force_target_names = ["dft_total_gradient"]

    # Energy in hartree, all zeros by default
    atomic_energies = np.zeros((MAX_ATOMIC_NUMBER,), dtype=np.float32)
    tmp = {
        35: -2574.2451510945853,
        6: -37.91424135791358,
        20: -676.9528465198214,
        17: -460.3350243496703,
        9: -99.91298732343974,
        1: -0.5027370838721259,
        53: -297.8813829975981,
        19: -599.8025677513111,
        3: -7.285254714046546,
        12: -199.2688420040449,
        7: -54.62327513368922,
        11: -162.11366478783253,
        8: -75.17101657391741,
        15: -341.3059197024934,
        16: -398.2405387031612,
    }
    for key in tmp:
        atomic_energies[key] = tmp[key]

    subset_mapping = {
        "SPICE Solvated Amino Acids Single Points Dataset v1.1": "Solvated Amino Acids",
        "SPICE Dipeptides Single Points Dataset v1.2": "Dipeptides",
        "SPICE DES Monomers Single Points Dataset v1.1": "DES370K Monomers",
        "SPICE DES370K Single Points Dataset v1.0": "DES370K Dimers",
        "SPICE DES370K Single Points Dataset Supplement v1.0": "DES370K Dimers",
        "SPICE PubChem Set 1 Single Points Dataset v1.2": "PubChem",
        "SPICE PubChem Set 2 Single Points Dataset v1.2": "PubChem",
        "SPICE PubChem Set 3 Single Points Dataset v1.2": "PubChem",
        "SPICE PubChem Set 4 Single Points Dataset v1.2": "PubChem",
        "SPICE PubChem Set 5 Single Points Dataset v1.2": "PubChem",
        "SPICE PubChem Set 6 Single Points Dataset v1.2": "PubChem",
        "SPICE Ion Pairs Single Points Dataset v1.1": "Ion Pairs",
    }

    def __init__(self) -> None:
        super().__init__()

    def read_raw_entries(self):
        raw_path = p_join(self.root, "SPICE-1.1.4.hdf5")

        data = load_hdf5_file(raw_path)
        tmp = [read_record(data[mol_name]) for mol_name in tqdm(data)]  # don't use parallelized here

        return tmp


if __name__ == "__main__":
    data = Spice()
    n = len(data)

    for i in np.random.choice(n, 10, replace=False):
        x = data[i]
        print(x.smiles, x.subset, end=" ")
        for k in x:
            if k != "smiles" and k != "subset":
                print(k, x[k].shape if x[k] is not None else None, end=" ")

        print()
