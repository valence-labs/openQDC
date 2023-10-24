from io import StringIO
from os.path import join as p_join

import numpy as np
import pandas as pd
from tqdm import tqdm

from openqdc.datasets.base import BaseDataset
from openqdc.utils.constants import MAX_ATOMIC_NUMBER
from openqdc.utils.molecule import atom_table


def content_to_xyz(content, e_map):
    try:
        tmp = content.split("\n")[1].split(" | ")
        code = tmp[0].split(" ")[-1]
        name = tmp[3].split(" ")[-1]
    except Exception:
        print(content)
        return None

    s = StringIO(content)
    d = np.loadtxt(s, skiprows=2, dtype="str")
    z, positions = d[:, 0], d[:, 1:].astype(np.float32)
    z = np.array([atom_table.GetAtomicNumber(s) for s in z])
    xs = np.stack((z, np.zeros_like(z)), axis=-1)
    e = e_map[code]

    conf = dict(
        atomic_inputs=np.concatenate((xs, positions), axis=-1, dtype=np.float32),
        name=np.array([name]),
        energies=np.array([e], dtype=np.float32)[:, None],
        n_atoms=np.array([positions.shape[0]], dtype=np.int32),
        subset=np.array(["tmqm"]),
    )

    return conf


def read_xyz(fname, e_map):
    with open(fname, "r") as f:
        contents = f.read().split("\n\n")

    print("toto", len(contents))
    res = [content_to_xyz(content, e_map) for content in tqdm(contents)]
    return res


class TMQM(BaseDataset):
    __name__ = "tmqm"

    # Energy in hartree, all zeros by default
    atomic_energies = np.zeros((MAX_ATOMIC_NUMBER,), dtype=np.float32)

    __energy_methods__ = ["tpssh/def2-tzvp"]

    energy_target_names = ["TPSSh/def2TZVP level"]

    __energy_unit__ = "hartree"
    __distance_unit__ = "ang"
    __forces_unit__ = "hartree/ang"

    def __init__(self) -> None:
        super().__init__()

    def read_raw_entries(self):
        df = pd.read_csv(p_join(self.root, "tmQM_y.csv"), sep=";", usecols=["CSD_code", "Electronic_E"])
        e_map = dict(zip(df["CSD_code"], df["Electronic_E"]))
        raw_fnames = ["tmQM_X1.xyz", "tmQM_X2.xyz", "Benchmark2_TPSSh_Opt.xyz"]
        samples = []
        for fname in raw_fnames:
            data = read_xyz(p_join(self.root, fname), e_map)
            samples += data

        return samples


if __name__ == "__main__":
    for data_class in [TMQM]:
        data = data_class()
        n = len(data)

        for i in np.random.choice(n, 3, replace=False):
            x = data[i]
            print(x.name, x.subset, end=" ")
            for k in x:
                if x[k] is not None:
                    print(k, x[k].shape, end=" ")
