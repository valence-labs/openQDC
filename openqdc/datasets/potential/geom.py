from os.path import join as p_join
from typing import Dict

import datamol as dm
import numpy as np

from openqdc.datasets.base import BaseDataset
from openqdc.methods import PotentialMethod
from openqdc.utils import load_json, load_pkl
from openqdc.utils.molecule import get_atomic_number_and_charge


def read_mol(mol_id: str, mol_dict, base_path: str, partition: str) -> Dict[str, np.ndarray]:
    """Read molecule from pickle file and return dict with conformers and energies

    Parameters
    ----------
    mol_id: str
        Unique identifier for the molecule
    mol_dict: dict
        Dictionary containing the pickle_path and smiles of the molecule
    base_path: str
        Path to the folder containing the pickle files
    partition: str
        Name of the dataset partition, one of ['qm9', 'drugs']

    Returns
    -------
    res: dict
        Dictionary containing the following keys:
        - atomic_inputs: flatten np.ndarray of shape (M, 5) containing the atomic numbers, charges and positions
        - smiles: np.ndarray of shape (N,) containing the smiles of the molecule
        - energies: np.ndarray of shape (N,1) containing the energies of the conformers
        - n_atoms: np.ndarray of shape (N,) containing the number of atoms in each conformer
        - subset: np.ndarray of shape (N,) containing the name of the dataset partition
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
            energies=np.array([cf["totalenergy"] for cf in confs], dtype=np.float64)[:, None],
            n_atoms=np.array([positions.shape[1]] * n_confs, dtype=np.int32),
            subset=np.array([partition] * n_confs),
        )

    except Exception as e:
        print(f"Skipping: {mol_id} due to {e}")
        res = None

    return res


class GEOM(BaseDataset):
    """
    Geometric Ensemble Of Molecules (GEOM) dataset contains 37 million conformers for 133,000 molecules
    from QM9, and 317,000 molecules with experimental data related to biophysics, physiology, and physical chemistry.
    For each molecule, the initial structure is generated with RDKit, optimized with the GFN2-xTB energy method and
    the lowest energy conformer is fed to the CREST software. CREST software uses metadynamics for exploring the
    conformational space for each molecule. Energies in the dataset are computed using semi-empirical method GFN2-xTB.

    Usage:
    ```python
    from openqdc.datasets import GEOM
    dataset = GEOM()
    ```

    References:
        https://www.nature.com/articles/s41597-022-01288-4\n
        https://github.com/learningmatter-mit/geom\n
        CREST Software: https://pubs.rsc.org/en/content/articlelanding/2020/cp/c9cp06869d
    """

    __name__ = "geom"
    __energy_methods__ = [PotentialMethod.GFN2_XTB]

    __energy_unit__ = "hartree"
    __distance_unit__ = "ang"
    __forces_unit__ = "hartree/ang"

    energy_target_names = ["gfn2_xtb.energy"]
    force_target_names = []

    partitions = ["qm9", "drugs"]
    __links__ = {"rdkit_folder.tar.gz": "https://dataverse.harvard.edu/api/access/datafile/4327252"}

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
