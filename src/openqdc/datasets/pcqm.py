import json
import tarfile
from glob import glob
from os.path import join as p_join

import datamol as dm
import numpy as np
import pandas as pd

from openqdc.datasets.base import BaseDataset
from openqdc.utils.constants import MAX_ATOMIC_NUMBER


def flatten_dict(d, sep: str = "."):
    return pd.json_normalize(d, sep=sep).to_dict(orient="records")[0]


def read_content(f):
    try:
        r = flatten_dict(json.load(f))
        x = np.concatenate(
            (
                r["atoms.elements.number"][:, None],
                r["atoms.core electrons"][:, None],
                r["atoms.coords.3d"].reshape(-1, 3),
            ),
            axis=-1,
        ).astype(np.float32)

        res = dict(
            name=np.array([r["smiles"]]),
            subset=np.array([r["formula"]]),
            energies=np.array(["properties.energy.total"]).astype(np.float32)[None, :],
            atomic_inputs=x,
            n_atoms=np.array([x.shape[0]], dtype=np.int32),
        )
    except Exception:
        res = None

    return res


def read_archive(path):
    with tarfile.open(path) as tar:
        res = [read_content(tar.extractfile(member)) for member in tar.getmembers()]
    # print(len(res))
    return res


class PubchemQC(BaseDataset):
    __name__ = "pubchemqc"
    __energy_methods__ = [
        "b3lyp/6-31g*",
        "pm6",
    ]

    __energy_unit__ = "ev"
    __distance_unit__ = "ang"
    __forces_unit__ = "ev/ang"

    energy_target_names = [
        "b3lyp",
        "pm6",
    ]

    # Energy in hartree, all zeros by default
    atomic_energies = np.zeros((MAX_ATOMIC_NUMBER,), dtype=np.float32)

    partitions = ["b3lyp", "pm6"]

    def __init__(self, energy_unit=None, distance_unit=None) -> None:
        super().__init__(energy_unit=energy_unit, distance_unit=distance_unit)

    def _read_raw_(self, part):
        arxiv_paths = glob(p_join(self.root, f"{part}", "*.tar.gz"))
        print(len(arxiv_paths))
        samples = dm.parallelized(read_archive, arxiv_paths, n_jobs=-1, progress=True, scheduler="threads")
        res = sum(samples, [])
        print(len(res))
        exit()
        return res

    def read_raw_entries(self):
        samples = sum([self._read_raw_(partition) for partition in self.partitions], [])
        return samples


if __name__ == "__main__":
    for data_class in [PubchemQC]:
        data = data_class()
        n = len(data)

        for i in np.random.choice(n, 3, replace=False):
            x = data[i]
            print(x.name, x.subset, end=" ")
            for k in x:
                if x[k] is not None:
                    print(k, x[k].shape, end=" ")
