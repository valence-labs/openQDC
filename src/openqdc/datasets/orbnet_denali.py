from os.path import join as p_join
from typing import Dict, List

import datamol as dm
import numpy as np
import pandas as pd

from openqdc.datasets.base import BaseDataset
from openqdc.utils.constants import MAX_ATOMIC_NUMBER
from openqdc.utils.molecule import atom_table


def read_archive(mol_id, conf_dict, base_path, energy_target_names: List[str]) -> Dict[str, np.ndarray]:
    res = []
    for conf_id, conf_label in conf_dict.items():
        try:
            cf_name = p_join(base_path, "xyz_files", mol_id, f"{conf_id}.xyz")
            d = np.loadtxt(cf_name, skiprows=2, dtype="str")
            z, positions = d[:, 0], d[:, 1:].astype(np.float32)
            z = np.array([atom_table.GetAtomicNumber(s) for s in z])
            xs = np.stack((z, np.zeros_like(z)), axis=-1)

            conf = dict(
                atomic_inputs=np.concatenate((xs, positions), axis=-1, dtype=np.float32),
                name=np.array([mol_id]),
                energies=np.array([conf_label[k] for k in energy_target_names], dtype=np.float32)[None, :],
                n_atoms=np.array([positions.shape[0]], dtype=np.int32),
                subset=np.array([conf_label["subset"]]),
            )
            res.append(conf)
        except Exception as e:
            print(f"Skipping: {mol_id} {conf_id} due to {e}")

    return res


class OrbnetDenali(BaseDataset):
    """
    Orbnet Denali is a collection of 2.3 million conformers from 212,905 unique molecules. It performs
    DFT (ωB97X-D3/def2-TZVP) calculations on molecules and geometries consisting of organic molecules
    and chemistries, with protonation and tautomeric states, non-covalent interactions, common salts,
    and counterions, spanning the most common elements in bio and organic chemistry.

    Usage:
    ```python
    from openqdc.datasets import OrbnetDenali
    dataset = OrbnetDenali()
    ```

    References:
    - https://arxiv.org/pdf/2107.00299.pdf
    - https://figshare.com/articles/dataset/OrbNet_Denali_Training_Data/14883867
    """

    __name__ = "orbnet_denali"
    __energy_methods__ = ["wb97x-d3_tz", "gfn1_xtb"]

    energy_target_names = ["dft_energy", "xtb1_energy"]

    # Energy in hartree, all zeros by default
    atomic_energies = np.zeros((MAX_ATOMIC_NUMBER,), dtype=np.float32)

    def __init__(self) -> None:
        super().__init__()

    def read_raw_entries(self):
        label_path = p_join(self.root, "denali_labels.csv")
        df = pd.read_csv(label_path, usecols=["sample_id", "mol_id", "subset", "dft_energy", "xtb1_energy"])
        labels = {
            mol_id: group.drop(["mol_id"], axis=1).drop_duplicates("sample_id").set_index("sample_id").to_dict("index")
            for mol_id, group in df.groupby("mol_id")
        }

        # print(df.head())
        # tmp = df.to_dict('index')
        # for i, k in enumerate(tmp):
        #     print(k, tmp[k])
        #     if i > 10:
        #         break
        # exit()
        fn = lambda x: read_archive(x[0], x[1], self.root, self.energy_target_names)
        res = dm.parallelized(fn, list(labels.items()), scheduler="threads", n_jobs=-1, progress=True)
        samples = sum(res, [])
        return samples


if __name__ == "__main__":
    for data_class in [OrbnetDenali]:
        data = data_class()
        n = len(data)

        for i in np.random.choice(n, 3, replace=False):
            x = data[i]
            print(x.name, x.subset, end=" ")
            for k in x:
                if x[k] is not None:
                    print(k, x[k].shape, end=" ")

            print()
