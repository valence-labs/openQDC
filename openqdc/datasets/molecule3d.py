import os
import torch
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


def get_props(df, sdf, idx):
    id = sdf.GetItemText(idx).split(" ")[1]
    return df.loc[[id]].to_dict(orient="records")[0]


def read_mol(mol, props):
    smiles = dm.to_smiles(mol, explicit_hs=False)
    subset = dm.to_smiles(dm.to_scaffold_murcko(mol, make_generic=True), explicit_hs=False)
    x = get_atom_data(mol)
    positions= mol.GetConformer().GetPositions() * BOHR2ANG
    
    res = dict(
        smiles= np.array([smiles]),
        subset= np.array([subset]),     
        energies= np.array([props["scf energy"]]).astype(np.float32)[:, None],
        atom_data_and_positions = np.concatenate((x, positions), axis=-1, dtype=np.float32),
        n_atoms = np.array([x.shape[0]], dtype=np.int32),
    )

    # for key in res:
    #     print(key, res[key].shape, res[key].dtype)
    # exit()

    return res


class Molecule3D(BaseDataset):
    __name__ = 'molecule3d'
    __qm_methods__ = ["b3lyp/6-31g*"]

    energy_target_names = ["b3lyp/6-31g*.energy"]
    force_target_names = []

    # Energy in hartree, all zeros by default
    atomic_energies = np.zeros((MAX_ATOMIC_NUMBER,), dtype=np.float32)

    def __init__(self) -> None:
        super().__init__()

    def read_raw_entries(self):
        raw = p_join(self.root, 'data', 'raw')
        sdf_paths = glob(p_join(raw, '*.sdf'))
        properties_path = p_join(raw, 'properties.csv')

        properties = pd.read_csv(properties_path, dtype={"cid": str})
        properties.drop_duplicates(subset="cid", inplace=True, keep="first")
        properties.set_index("cid", inplace=True)
        n = len(sdf_paths)
        
        tmp = []
        for i, path in enumerate(sdf_paths):
            suppl = Chem.SDMolSupplier(path, removeHs=False, sanitize=True)
            n = len(suppl)
            
            tmp += [
                read_mol(suppl[j], get_props(properties, suppl, j))
                for j in tqdm(range(n), desc=f"{i+1}/{n}")
            ]

        return tmp


if __name__ == '__main__':
    data = Molecule3D()
    n = len(data)

    for i in np.random.choice(n, 10, replace=False):
        x = data[i]
        print(x.smiles, x.subset, end=' ')
        for k in x:
            if k != 'smiles' and k != 'subset':
                print(k, x[k].shape if x[k] is not None else None, end=' ')
            
        print()