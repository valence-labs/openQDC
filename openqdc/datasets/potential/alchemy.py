import os
from os.path import join as p_join

import numpy as np

from openqdc.datasets.base import BaseDataset
from openqdc.methods import PotentialMethod
from openqdc.utils import load_hdf5_file, read_qc_archive_h5
from openqdc.utils.io import get_local_cache
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import pandas as pd 
from glob import glob 
from openqdc.utils.molecule import get_atomic_number_and_charge
import datamol as dm 
from tqdm import tqdm 
#['gdb_idx', 'atom number', 'zpve\n(Ha, zero point vibrational energy)',
#'Cv\n(cal/molK, heat capacity at 298.15 K)', 'gap\n(Ha, LUMO-HOMO)',
# 'G\n(Ha, Free energy at 298.15 K)', 'HOMO\n(Ha, energy of HOMO)',
# 'U\n(Ha, internal energy at 298.15 K)', 'alpha\n(a_0^3, Isotropic polarizability)',
# 'U0\n(Ha, internal energy at 0 K)', 'H\n(Ha, enthalpy at 298.15 K)',
# 'LUMO\n(Ha, energy of LUMO)', 'mu\n(D, dipole moment)',
# 'R2\n(a_0^2, electronic spatial extent)']

def read_mol(file, energy):
    try:
        mol = dm.read_sdf(file, remove_hs=False)[0]
        positions = mol.GetConformer().GetPositions()
        x = get_atomic_number_and_charge(mol)
        n_atoms=positions.shape[0]
        res = dict(
            atomic_inputs=np.concatenate(
                (x, positions), axis=-1, dtype=np.float32
            ).reshape(-1, 5),
            name=np.array([dm.to_smiles(mol)]),
            energies=np.array([energy], dtype=np.float64)[:, None],
            n_atoms=np.array([n_atoms], dtype=np.int32),
            subset=np.array([f"atoms_{n_atoms}"]),
        )

    except Exception as e:
        print(f"Skipping due to {e}")
        res = None

    return res
    
# e B3LYP/6-31G(2df,p) model with the density fitting
# approximation for electron repulsion integrals. The auxiliary basis cc-pVDZ-jkf

class Alchemy(BaseDataset):


    __name__ = "alchemy"

    __energy_methods__ = [
        PotentialMethod.WB97X_6_31G_D,  # "wb97x/6-31g(d)"
    ]

    energy_target_names = [
        "ωB97x:6-31G(d) Energy",
    ]

    __energy_unit__ = "hartree"
    __distance_unit__ = "ang"
    __forces_unit__ = "hartree/ang"
    __links__ = {f"alchemy.zip": "https://alchemy.tencent.com/data/alchemy-v20191129.zip"}

    def read_raw_entries(self):
        dir_path = p_join(self.root, f"Alchemy-v20191129")
        full_csv = pd.read_csv(p_join(dir_path, "final_version.csv"))
        energies =  full_csv['U0\n(Ha, internal energy at 0 K)'].tolist()
        atom_folder = full_csv['atom number']
        gdb_idx = full_csv['gdb_idx']
        idxs = full_csv.index.tolist()
        samples=[]
        for i in tqdm(idxs):
            sdf_file = p_join(dir_path, f"atom_{atom_folder[i]}", f"{gdb_idx[i]}.sdf")
            energy = energies[i]
            samples.append(read_mol(sdf_file, energy))
        return samples