import os
import torch
import pickle as pkl
import numpy as np
from tqdm import tqdm
import datamol as dm
from sklearn.utils import Bunch
from os.path import join as p_join
from openqdc.utils import load_pkl, load_json
from openqdc.utils.molecule import get_atom_data
from openqdc.utils.paths import get_local_cache
from openqdc.utils.constants import MAX_ATOMIC_NUMBER
from openqdc.datasets.base import BaseDataset


def read_mol(mol_id, mol_dict, base_path, partition):
    """ Read molecule from pickle file and return dict with conformers and energies

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
            - atom_data_and_positions: flatten np.ndarray of shape (M, 4) containing the atomic numbers and positions
            - smiles: np.ndarray of shape (N,) containing the smiles of the molecule
            - energies: np.ndarray of shape (N,1) containing the energies of the conformers
            - n_atoms: np.ndarray of shape (N,) containing the number of atoms in each conformer
    """

    try:
        d = load_pkl(p_join(base_path, mol_dict['pickle_path']), False)
        confs = d['conformers']
        x = get_atom_data(confs[0]['rd_mol'])
        positions = np.array([cf['rd_mol'].GetConformer().GetPositions() for cf in confs])
        n_confs = positions.shape[0]

        res = dict(
            atom_data_and_positions = np.concatenate((
                x[None, ...].repeat(n_confs, axis=0), 
                positions), axis=-1, dtype=np.float32).reshape(-1, 5),
            smiles = np.array([d['smiles'] for _ in confs]),
            energies = np.array([cf['totalenergy'] for cf in confs], dtype=np.float32)[:, None],
            n_atoms = np.array([positions.shape[1]] * n_confs, dtype=np.int32),
            subset = np.array([partition] * n_confs),
        )

    except Exception as e:
        print (f'Skipping: {mol_id} due to {e}')
        res = None

    return res


class NablaDFT(BaseDataset):
    __name__ = 'nabladft'
    __qm_methods__ = ["wb97x_svp"]

    energy_target_names = ["wb97x_svp.energy"]
    force_target_names = []

    # Energy in hartree, all zeros by default
    atomic_energies = np.zeros((MAX_ATOMIC_NUMBER,), dtype=np.float32)

    partitions = ['qm9', 'drugs']

    def __init__(self) -> None:
        super().__init__()

    def read_raw_entries(self):
        raw_path = p_join(self.root, 'nabladft')

        mols = load_json(p_join(raw_path, f'summary_{partition}.json'))
        mols = list(mols.items())

        fn = lambda x: read_mol(x[0], x[1], raw_path, partition)
        samples = dm.parallelized(fn, mols, n_jobs=1, progress=True) # don't use more than 1 job
        return samples
    

if __name__ == '__main__':
    from openqdc.utils.paths import get_local_cache
    from nablaDFT.dataset import HamiltonianDatabase

    f_path = p_join(get_local_cache(), "nabladft", "train_2k_energy.db")
    f_path = p_join(get_local_cache(), "nabladft", "dataset_train_2k.db")
    print(f_path)
    train = HamiltonianDatabase(f_path)
    Z, R, E, F, H, S, C = train[0]
    print(Z.shape, R.shape, E.shape, F.shape, H.shape, S.shape, C.shape)


    #
    # data = NablaDFT()
    # n = len(data)

    # for i in np.random.choice(n, 10, replace=False):
    #     x = data[i]
    #     print(x.smiles, x.subset, end=' ')
    #     for k in x:
    #         if k != 'smiles' and k != 'subset':
    #             print(k, x[k].shape if x[k] is not None else None, end=' ')
            
    #     print()
