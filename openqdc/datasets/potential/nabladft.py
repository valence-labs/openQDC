import os
from os.path import join as p_join
from typing import Dict

import datamol as dm
import numpy as np
import pandas as pd

from openqdc.datasets.base import BaseDataset
from openqdc.methods import PotentialMethod
from openqdc.utils.molecule import z_to_formula
from openqdc.utils.package_utils import requires_package


def to_mol(entry, metadata) -> Dict[str, np.ndarray]:
    Z, R, E, F = entry[:4]
    C = np.zeros_like(Z)
    E[0] = metadata["DFT TOTAL ENERGY"]

    res = dict(
        atomic_inputs=np.concatenate((Z[:, None], C[:, None], R), axis=-1).astype(np.float32),
        name=np.array([metadata["SMILES"]]),
        energies=E[:, None].astype(np.float64),
        forces=F[:, :, None].astype(np.float32),
        n_atoms=np.array([Z.shape[0]], dtype=np.int32),
        subset=np.array([z_to_formula(Z)]),
    )

    return res


@requires_package("nablaDFT")
def read_chunk_from_db(raw_path, start_idx, stop_idx, labels, step_size=1000):
    from nablaDFT.dataset import HamiltonianDatabase

    print(f"Loading from {start_idx} to {stop_idx}")
    db = HamiltonianDatabase(raw_path)
    idxs = list(np.arange(start_idx, stop_idx))
    n, s = len(idxs), step_size

    cursor = db._get_connection().cursor()
    data_idxs = cursor.execute("""SELECT * FROM dataset_ids WHERE id IN (""" + str(idxs)[1:-1] + ")").fetchall()
    c_idxs = [tuple(x[1:]) for x in data_idxs]

    samples = [
        to_mol(entry, labels[c_idxs[i + j]]) for i in range(0, n, s) for j, entry in enumerate(db[idxs[i : i + s]])
    ]
    return samples


class NablaDFT(BaseDataset):
    """
    NablaDFT is a dataset constructed from a subset of the
    [Molecular Sets (MOSES) dataset](https://github.com/molecularsets/moses) consisting of 1 million molecules
    with 5,340,152 unique conformations. Conformations for each molecule are generated in 2 steps. First, a set of
    conformations are generated using RDKit. Second, using Butina Clustering Method on conformations, clusters that
    cover 95% of the conformations are selected and the centroids of those clusters are selected as the final set.
    This results in 1-62 conformations per molecule. For generating quantum properties, Kohn-Sham method at
    wB97X-D/def2-XVP levels are used to generate the energy.

    Usage:
    ```python
    from openqdc.datasets import NablaDFT
    dataset = NablaDFT()
    ```

    References:
        https://pubs.rsc.org/en/content/articlelanding/2022/CP/D2CP03966D\n
        https://github.com/AIRI-Institute/nablaDFT
    """

    __name__ = "nabladft"
    __energy_methods__ = [
        PotentialMethod.WB97X_D_DEF2_SVP,
    ]  # "wb97x-d/def2-svp"

    energy_target_names = ["wb97x-d/def2-svp"]
    __energy_unit__ = "hartree"
    __distance_unit__ = "bohr"
    __forces_unit__ = "hartree/bohr"
    __links__ = {"nabladft.db": "https://n-usr-31b1j.s3pd12.sbercloud.ru/b-usr-31b1j-qz9/data/moses_db/dataset_full.db"}

    @property
    def data_types(self):
        return {
            "atomic_inputs": np.float32,
            "position_idx_range": np.int32,
            "energies": np.float32,
            "forces": np.float32,
        }

    @requires_package("nablaDFT")
    def read_raw_entries(self):
        from nablaDFT.dataset import HamiltonianDatabase

        label_path = p_join(self.root, "summary.csv")
        df = pd.read_csv(label_path, usecols=["MOSES id", "CONFORMER id", "SMILES", "DFT TOTAL ENERGY"])
        labels = df.set_index(keys=["MOSES id", "CONFORMER id"]).to_dict("index")

        raw_path = p_join(self.root, "dataset_full.db")
        train = HamiltonianDatabase(raw_path)
        n, c = len(train), 20
        step_size = int(np.ceil(n / os.cpu_count()))

        fn = lambda i: read_chunk_from_db(raw_path, i * step_size, min((i + 1) * step_size, n), labels=labels)
        samples = dm.parallelized(
            fn, list(range(c)), n_jobs=c, progress=False, scheduler="threads"
        )  # don't use more than 1 job

        return sum(samples, [])
