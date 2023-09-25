from os.path import join as p_join

import datamol as dm
import numpy as np

from openqdc.datasets.base import BaseDataset
from openqdc.utils import load_json, load_pkl
from openqdc.utils.constants import MAX_ATOMIC_NUMBER
from openqdc.utils.molecule import get_atomic_number_and_charge


def read_mol(mol_id, mol_dict, base_path, partition):
    """Read molecule from pickle file and return dict with conformers and energies

    Parameters
    ----------
    mol_id: str
        Unique identifier for the molecule
    mol_dict: dict
        Dictionary containing the pickle_path and smiles of the molecule
    base_path: str
        Path to the folder containing the pickle files

    Returns
    -------
    res: dict
        Dictionary containing the following keys:
            - atomic_inputs: flatten np.ndarray of shape (M, 4) containing the atomic numbers and positions
            - smiles: np.ndarray of shape (N,) containing the smiles of the molecule
            - energies: np.ndarray of shape (N,1) containing the energies of the conformers
            - n_atoms: np.ndarray of shape (N,) containing the number of atoms in each conformer
    """

    try:
        d = load_pkl(p_join(base_path, mol_dict["pickle_path"]), False)
        confs = d["conformers"]
        x = get_atomic_number_and_charge(confs[0]["rd_mol"])
        positions = np.array([cf["rd_mol"].GetConformer().GetPositions() for cf in confs])
        n_confs = positions.shape[0]

        res = dict(
            atomic_inputs=np.concatenate(
                (x[None, ...].repeat(n_confs, axis=0), positions), axis=-1, dtype=np.float32
            ).reshape(-1, 5),
            name=np.array([d["smiles"] for _ in confs]),
            energies=np.array([cf["totalenergy"] for cf in confs], dtype=np.float32)[:, None],
            n_atoms=np.array([positions.shape[1]] * n_confs, dtype=np.int32),
            subset=np.array([partition] * n_confs),
        )

    except Exception as e:
        print(f"Skipping: {mol_id} due to {e}")
        res = None

    return res


class GEOM(BaseDataset):
    __name__ = "geom"
    __energy_methods__ = ["gfn2_xtb"]

    energy_target_names = ["gfn2_xtb.energy"]
    force_target_names = []

    # Energy in hartree, all zeros by default
    atomic_energies = np.zeros((MAX_ATOMIC_NUMBER,), dtype=np.float32)

    partitions = ["qm9", "drugs"]

    def __init__(self) -> None:
        super().__init__()

    def _read_raw_(self, partition):
        raw_path = p_join(self.root, "rdkit_folder")

        mols = load_json(p_join(raw_path, f"summary_{partition}.json"))
        mols = list(mols.items())

        fn = lambda x: read_mol(x[0], x[1], raw_path, partition)  # noqa E731
        samples = dm.parallelized(fn, mols, n_jobs=1, progress=True)  # don't use more than 1 job
        return samples

    def read_raw_entries(self):
        samples = sum([self._read_raw_(partition) for partition in self.partitions], [])
        return samples


if __name__ == "__main__":
    for data_class in [GEOM]:
        data = data_class()
        n = len(data)

        for i in np.random.choice(n, 3, replace=False):
            x = data[i]
            print(x.name, x.subset, end=" ")
            for k in x:
                if x[k] is not None:
                    print(k, x[k].shape, end=" ")

            print()
