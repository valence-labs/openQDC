from os.path import join as p_join

import datamol as dm
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit.Chem import MolFromMolBlock
from openqdc.datasets.base import BaseDataset
from openqdc.utils import load_json, load_pkl
from openqdc.utils.constants import MAX_ATOMIC_NUMBER
from openqdc.utils.molecule import get_atomic_number_and_charge


def read_mol(mol_path, smiles, subset, targets):
    try:
        with open(mol_path, "r") as f:
            mol_block = f.read()
            mol = dm.read_molblock(mol_block, remove_hs=False, fail_if_invalid=True)

        x = get_atomic_number_and_charge(mol)
        positions = mol.GetConformer().GetPositions()

        res = dict(
            name=np.array([smiles]),
            subset=np.array([subset]),
            energies=np.array(targets).astype(np.float32)[None, :],
            atomic_inputs=np.concatenate((x, positions), axis=-1, dtype=np.float32),
            n_atoms=np.array([x.shape[0]], dtype=np.int32),
        )
    except Exception as e:
        print(f"Skipping: {mol_path} due to {e}")
        res = None

    return res


class DESS(BaseDataset):
    __name__ = "dess"
    __energy_methods__ = [
        'mp2_cc',
        'mp2_qz', 
        'mp2_tz', 
        'mp2_cbs', 
        'ccsd(t)_cc', 
        'ccsd(t)_cbs', 
        'ccsd(t)_nn',
        'sapt',
    ]

    energy_target_names = [
        'cc_MP2_all',
        'qz_MP2_all', 
        'tz_MP2_all', 
        'cbs_MP2_all', 
        'cc_CCSD(T)_all', 
        'cbs_CCSD(T)_all', 
        'nn_CCSD(T)_all',
        'sapt_all',
    ]
    # ['qz_MP2_all', 'tz_MP2_all', 'cbs_MP2_all', 'sapt_all', 'nn_CCSD(T)_all']

    # Energy in hartree, all zeros by default
    atomic_energies = np.zeros((MAX_ATOMIC_NUMBER,), dtype=np.float32)

    partitions = ["DES370K", "DES5M"]

    def __init__(self) -> None:
        super().__init__()

    def _read_raw_(self, part):
        df = pd.read_csv(p_join(self.root, f"{part}.csv"))
        for col in self.energy_target_names:
            if col not in df.columns:
                df[col] = np.nan
        smiles = (df['smiles0'] + '.' + df['smiles1']).tolist()
        subsets = (f"{part}_" + df["group_orig"]).tolist()
        targets = df[self.energy_target_names].values
        paths = p_join(self.root, "geometries/") + df["system_id"].astype(str) + f"/{part}_" + df["geom_id"].astype(str) + ".mol"
        
        inputs = [dict(smiles=smiles[i], subset=subsets[i], targets=targets[i], mol_path=paths[i]) 
                  for i in tqdm(range(len(smiles)))]
        f = lambda xs: [read_mol(**x) for x in xs]
        samples = dm.parallelized_with_batches(f, inputs, n_jobs=-1, progress=True, 
                                               batch_size=1024, scheduler= "threads")
        return samples

    def read_raw_entries(self):
        samples = sum([self._read_raw_(partition) for partition in self.partitions], [])
        return samples


if __name__ == "__main__":
    for data_class in [DESS]:
        data = data_class()
        n = len(data)

        for i in np.random.choice(n, 3, replace=False):
            x = data[i]
            print(x.name, x.subset, end=" ")
            for k in x:
                if x[k] is not None:
                    print(k, x[k].shape, end=" ")
