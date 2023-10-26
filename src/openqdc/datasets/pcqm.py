import json
import os
import pickle as pkl
import tarfile
from glob import glob
from os.path import join as p_join

import datamol as dm
import numpy as np
import pandas as pd
from loguru import logger

from openqdc.datasets.base import BaseDataset
from openqdc.utils.io import get_local_cache


def flatten_dict(d, sep: str = "."):
    return pd.json_normalize(d, sep=sep).to_dict(orient="records")[0]


def read_content(f, prefix="pubchem.PM6."):
    try:
        content = f.read()
        r = flatten_dict(json.loads(content))

        x = np.concatenate(
            (
                np.array(r[f"{prefix}atoms.elements.number"])[:, None],
                np.array(r[f"{prefix}atoms.core electrons"])[:, None],  # not sure if this is correct
                np.array(r[f"{prefix}atoms.coords.3d"]).reshape(-1, 3),
            ),
            axis=-1,
        ).astype(np.float32)

        res = dict(
            name=np.array([r[f"{prefix}openbabel.Canonical SMILES"]]),
            subset=np.array([r["pubchem.molecular formula"]]),
            energies=np.array([r[f"{prefix}properties.energy.total"]]).astype(np.float32)[None, :],
            atomic_inputs=x,
            n_atoms=np.array([x.shape[0]], dtype=np.int32),
        )
    except Exception:
        logger.warning(f"Failed to parse {content}")
        res = None
    return res


def read_archive(path):
    res = []
    partition = path.split("/")[-2]
    prefix = "pubchem.PM6." if partition == "pm6" else "pubchem.B3LYP@PM6."
    with tarfile.open(path, mode="r:xz") as tar:
        members = [tar.extractfile(x) for x in tar.getmembers() if x is not None and x.name.endswith(".json")]
        res = [read_content(x, prefix=prefix) for x in members if x is not None]

    return res


def read_preprocessed_archive(path):
    res = []
    if os.path.exists(path):
        with open(path, "rb") as f:
            res = pkl.load(f)
    return res


class PCQM_PM6(BaseDataset):
    __name__ = "pubchemqc_pm6"
    __energy_methods__ = ["pm6"]

    energy_target_names = ["pm6"]

    __force_methods__ = []
    force_target_names = []

    def __init__(self, energy_unit=None, distance_unit=None) -> None:
        super().__init__(energy_unit=energy_unit, distance_unit=distance_unit)

    @property
    def root(self):
        return p_join(get_local_cache(), "pubchemqc")

    def collate_list(self, list_entries, partial=False):
        # default partial=False is necessary for compatibility with the base class
        if partial:
            predicat = list_entries is not None and len(list_entries) > 0
            list_entries = [x for x in list_entries if x is not None]
            return super().collate_list(list_entries) if predicat else None
        else:
            n = 0
            for i in range(len(list_entries)):
                list_entries[i]["position_idx_range"] += n
                n += list_entries[i]["position_idx_range"].max()
            res = {key: np.concatenate([r[key] for r in list_entries], axis=0) for key in list_entries[0]}
            return res

    def read_raw_entries(self):
        arxiv_paths = glob(p_join(self.root, f"{self.__energy_methods__[0]}", "*.pkl"))
        f = lambda x: self.collate_list(read_preprocessed_archive(x), partial=True)
        samples = dm.parallelized(f, arxiv_paths, n_jobs=1, progress=True)
        samples = [x for x in samples if x is not None]
        return samples


class PCQM_B3LYP(PCQM_PM6):
    __name__ = "pubchemqc_b3lyp"
    __energy_methods__ = ["b3lyp"]

    energy_target_names = ["b3lyp"]

    def __init__(self, energy_unit=None, distance_unit=None) -> None:
        super().__init__(energy_unit=energy_unit, distance_unit=distance_unit)
