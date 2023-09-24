
import os
import numpy as np
import datamol as dm
from tqdm import tqdm
from os.path import join as p_join
from openqdc.utils.constants import MAX_ATOMIC_NUMBER
from openqdc.datasets.base import BaseDataset
from nablaDFT.dataset import HamiltonianDatabase


def to_mol(entry):
    Z, R, E, F = entry[:4]
    C = np.zeros_like(Z)

    res = dict(
        atomic_inputs = np.concatenate((Z[:, None], C[:, None],  R), axis=-1).astype(np.float32),
        name = np.array(['']),
        energies = E[:, None].astype(np.float32),
        forces = F[:, :, None].astype(np.float32),
        n_atoms = np.array([Z.shape[0]], dtype=np.int32),
        subset = np.array(['nabla']),
    )

    return res


def read_chunk_from_db(raw_path, start_idx, stop_idx, step_size=1000):
    print(f'Loading from {start_idx} to {stop_idx}')
    db = HamiltonianDatabase(raw_path)
    idxs = list(np.arange(start_idx, stop_idx))
    n, s = len(idxs), step_size

    samples = [to_mol(entry) for i in tqdm(range(0,  n, s)) for entry in db[idxs[i:i + s]]]
    return samples
    

class NablaDFT(BaseDataset):
    __name__ = 'nabladft'
    __energy_methods__ = ["wb97x-d_svp"]

    energy_target_names = ["ωB97X-D/def2-SVP"]

    # Energy in hartree, all zeros by default
    atomic_energies = np.zeros((MAX_ATOMIC_NUMBER,), dtype=np.float32)

    def __init__(self) -> None:
        super().__init__()


    def read_raw_entries(self):
        raw_path = p_join(self.root, 'dataset_full.db')
        train = HamiltonianDatabase(raw_path)
        n, c = len(train), 20
        step_size = int(np.ceil(n / os.cpu_count()))

        fn = lambda i: read_chunk_from_db(raw_path, i*step_size, min((i + 1) * step_size, n))
        samples = dm.parallelized(fn, list(range(c)), n_jobs=c, progress=False, scheduler="threads") # don't use more than 1 job
            
        return sum(samples, [])
    

if __name__ == '__main__':
    for data_class in [NablaDFT]:
        data = data_class()
        n = len(data)

        for i in np.random.choice(n, 3, replace=False):
            x = data[i]
            print(x.name, x.subset, end=' ')
            for k in x:
                if x[k] is not None:
                    print(k, x[k].shape, end=' ')
                
            print()

