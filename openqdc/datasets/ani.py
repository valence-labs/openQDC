import numpy as np
from tqdm import tqdm
import datamol as dm
from os.path import join as p_join
from openqdc.utils import load_hdf5_file
from openqdc.utils.molecule import get_atom_data
from openqdc.utils.constants import BOHR2ANG, MAX_ATOMIC_NUMBER
from openqdc.datasets.base import BaseDataset


def read_record(r, r_name):
    n_confs = r["coordinates"].shape[0]
    x = r["atomic_numbers"][()]
    xs = np.stack((x, np.zeros_like(x)), axis=-1)
    positions= r["coordinates"][()] * BOHR2ANG
    energies= np.stack([r[k] for k in Ani1.energy_target_names], axis=-1)
    forces= np.stack([r[k] for k in Ani1.force_target_names], axis=-1)
    
    res = dict(
        smiles= np.array([r_name]*n_confs),
        subset= np.array([r_name]*n_confs),     
        energies= energies.astype(np.float32),
        forces= forces.reshape(-1, *forces.shape[-2:]).astype(np.float32),
        atom_data_and_positions = np.concatenate((
            xs[None, ...].repeat(n_confs, axis=0), 
            positions), axis=-1, dtype=np.float32).reshape(-1, 5),
        n_atoms = np.array([x.shape[0]]*n_confs, dtype=np.int32),
    )

    return res


class Ani1(BaseDataset):
    __name__ = 'ani'

    # Energy in hartree, all zeros by default
    atomic_energies = np.zeros((MAX_ATOMIC_NUMBER,), dtype=np.float32)

    __qm_methods__ = [
        "ccsd(t)_cbs",
        "hf_dz",
        "hf_qz",
        "hf_tz",
        "mp2_dz",
        "mp2_qz",
        "mp2_tz",
        "npno_ccsd(t)_dz",
        "npno_ccsd(t)_tz",
        "tpno_ccsd(t)_dz",
        "wb97x_dz",
        "wb97x_tz",
    ]

    energy_target_names = [
        "ccsd(t)_cbs.energy",
        "hf_dz.energy",
        "hf_qz.energy",
        "hf_tz.energy",
        "mp2_dz.corr_energy",
        "mp2_qz.corr_energy",
        "mp2_tz.corr_energy",
        "npno_ccsd(t)_dz.corr_energy",
        "npno_ccsd(t)_tz.corr_energy",
        "tpno_ccsd(t)_dz.corr_energy",
        "wb97x_dz.energy",
        "wb97x_tz.energy",
    ]

    force_target_names = [
        "wb97x_dz.forces",
        "wb97x_tz.forces"    
    ]

    def __init__(self) -> None:
        super().__init__()

    def read_raw_entries(self):
        raw_path = p_join(self.root, 'ani1.h5')
        data = load_hdf5_file(raw_path)

        fn = lambda x: read_record(x[0], x[1])
        tmp = [(data[mol_name], mol_name) for mol_name in data.keys()]
        samples = dm.parallelized(fn, tmp, n_jobs=1, progress=True) # don't use more than 1 job
        return samples


if __name__ == '__main__':
    data = Ani1()
    n = len(data)

    for i in np.random.choice(n, 100, replace=False):
        x = data[i]
        for k in x:
            print(x.smiles, x.subset, end=' ')
            print(k, x[k].shape, end=' ')
            
        print()