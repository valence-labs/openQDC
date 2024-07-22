from os.path import join as p_join
from typing import Dict, List

import datamol as dm
import numpy as np
import pandas as pd

from openqdc.datasets.base import BaseDataset
from openqdc.methods import PotentialMethod
from openqdc.utils.constants import ATOM_TABLE


def read_archive(mol_id, conf_dict, base_path, energy_target_names: List[str]) -> Dict[str, np.ndarray]:
    res = []
    for conf_id, conf_label in conf_dict.items():
        try:
            cf_name = p_join(base_path, "xyz_files", mol_id, f"{conf_id}.xyz")
            d = np.loadtxt(cf_name, skiprows=2, dtype="str")
            z, positions = d[:, 0], d[:, 1:].astype(np.float32)
            z = np.array([ATOM_TABLE.GetAtomicNumber(s) for s in z])
            xs = np.stack((z, np.zeros_like(z)), axis=-1)

            conf = dict(
                atomic_inputs=np.concatenate((xs, positions), axis=-1, dtype=np.float32),
                name=np.array([mol_id]),
                energies=np.array([conf_label[k] for k in energy_target_names], dtype=np.float64)[None, :],
                n_atoms=np.array([positions.shape[0]], dtype=np.int32),
                subset=np.array([conf_label["subset"]]),
            )
            res.append(conf)
        except Exception as e:
            print(f"Skipping: {mol_id} {conf_id} due to {e}")

    return res


class OrbnetDenali(BaseDataset):
    """
    Orbnet Denali is a collection of 2.3 million conformers from 212,905 unique molecules. Molecules include a range
    of organic molecules with protonation and tautomeric states, non-covalent interactions, common salts, and
    counterions, spanning the most common elements in bio and organic chemistry. Geometries are generated in 2 steps.
    First, four energy-minimized conformations are generated for each molecule using the ENTOS BREEZE conformer
    generator. Second, using the four energy-minimized conformers, non-equilibrium geometries are generated using
    normal mode sampling at 300K or ab initio molecular dynamics (AIMD) for 200fs at 500K; using GFN1-xTB level of
    theory. Energies are calculated using DFT method wB97X-D3/def2-TZVP and semi-empirical method GFN1-xTB level of
    theory.

    Usage:
    ```python
    from openqdc.datasets import OrbnetDenali
    dataset = OrbnetDenali()
    ```

    References:
        https://arxiv.org/abs/2107.00299\n
        https://figshare.com/articles/dataset/OrbNet_Denali_Training_Data/14883867
    """

    __name__ = "orbnet_denali"
    __energy_methods__ = [
        PotentialMethod.WB97X_D3_DEF2_TZVP,
        PotentialMethod.GFN1_XTB,
    ]  # ["wb97x-d3/def2-tzvp", "gfn1_xtb"]
    energy_target_names = ["dft_energy", "xtb1_energy"]
    __energy_unit__ = "hartree"
    __distance_unit__ = "ang"
    __forces_unit__ = "hartree/ang"
    __links__ = {
        "orbnet_denali.tar.gz": "https://figshare.com/ndownloader/files/28672287",
        "orbnet_denali_targets.tar.gz": "https://figshare.com/ndownloader/files/28672248",
    }

    def read_raw_entries(self):
        label_path = p_join(self.root, "denali_labels.csv")
        df = pd.read_csv(label_path, usecols=["sample_id", "mol_id", "subset", "dft_energy", "xtb1_energy"])
        labels = {
            mol_id: group.drop(["mol_id"], axis=1).drop_duplicates("sample_id").set_index("sample_id").to_dict("index")
            for mol_id, group in df.groupby("mol_id")
        }

        fn = lambda x: read_archive(x[0], x[1], self.root, self.energy_target_names)
        res = dm.parallelized(fn, list(labels.items()), scheduler="threads", n_jobs=-1, progress=True)
        samples = sum(res, [])
        return samples
