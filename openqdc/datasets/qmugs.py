import os
import pickle as pkl
import numpy as np
import pandas as pd
import os.path as osp
import datamol as dm
from tqdm import tqdm
from glob import glob
from sklearn.utils import Bunch
from rdkit import Chem
from os.path import join as p_join
from openqdc.utils import load_hdf5_file
from openqdc.utils.molecule import get_atom_data
from openqdc.utils.paths import get_local_cache
from openqdc.utils.constants import BOHR2ANG, MAX_ATOMIC_NUMBER
from openqdc.datasets.base import BaseDataset


def read_mol(mol_dir):
    filenames = glob(p_join(mol_dir, "*.sdf"))
    mols = [dm.read_sdf(f)[0] for f in filenames]
    n_confs = len(mols)

    if len(mols) == 0:
        return None

    smiles = dm.to_smiles(mols[0], explicit_hs=False)
    subset = dm.to_smiles(dm.to_scaffold_murcko(mols[0], make_generic=True), explicit_hs=False)
    x = get_atom_data(mols[0])[None, ...].repeat(n_confs, axis=0)
    positions= np.array([mol.GetConformer().GetPositions() for mol in mols])
    props = [mol.GetPropsAsDict() for mol in mols]
    targets = np.array([[p[el]for el in QMugs.energy_target_names] for p in props])
    
    res = dict(
        smiles= np.array([smiles]*n_confs),
        subset= np.array([subset]*n_confs),     
        energies= targets.astype(np.float32),
        atom_data_and_positions = np.concatenate((x, positions), 
                                    axis=-1, dtype=np.float32).reshape(-1, 5),
        n_atoms = np.array([x.shape[1]]*n_confs, dtype=np.int32),
    )

    # for key in res:
    #     print(key, res[key].shape, res[key].dtype)
    # exit()

    return res


class QMugs(BaseDataset):

    __name__ = 'qmugs'
    __qm_methods__ = ["b3lyp/6-31g*"]

    energy_target_names = ["GFN2:TOTAL_ENERGY", "DFT:TOTAL_ENERGY",]
    # target_names = [
    #     "GFN2:TOTAL_ENERGY",
    #     "GFN2:ATOMIC_ENERGY",
    #     "GFN2:FORMATION_ENERGY",
    #     "DFT:TOTAL_ENERGY",
    #     "DFT:ATOMIC_ENERGY",
    #     "DFT:FORMATION_ENERGY",
    # ]

    force_target_names = []

    # Energy in hartree, all zeros by default
    atomic_energies = np.zeros((MAX_ATOMIC_NUMBER,), dtype=np.float32)


    def __init__(self) -> None:
        super().__init__()

    def read_raw_entries(self):
        raw_path = p_join(self.root, 'structures')
        mol_dirs = [p_join(raw_path, d) for d in os.listdir(raw_path)]

        tmp = dm.parallelized(read_mol, mol_dirs, n_jobs=-1, 
                              progress=True, scheduler="threads")
        return tmp
    

if __name__ == '__main__':
    data = QMugs()
    n = len(data)

    for i in np.random.choice(n, 10, replace=False):
        x = data[i]
        print(x.smiles, x.subset, end=' ')
        for k in x:
            if k != 'smiles' and k != 'subset':
                print(k, x[k].shape if x[k] is not None else None, end=' ')
            
        print()