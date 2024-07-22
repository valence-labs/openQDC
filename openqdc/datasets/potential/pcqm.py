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
from openqdc.methods import PotentialMethod
from openqdc.utils.io import get_local_cache, push_remote


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
    """
    PubChemQC PM6 (PCQM_PM6) is an exhaustive dataset containing 221 million organic molecules with optimized
    molecular geometries and electronic properties. To generate the dataset, only molecules with weights less
    than 1000g/mol are considered from the PubChem ftp site. The initial structure is generated using OpenBabel
    and then is optimized using geometry optimization with the semi-empirical method PM6. The energies are also
    computed using the PM6 method.

    Usage:
    ```python
    from openqdc.datasets import PCQM_PM6
    dataset = PCQM_PM6()
    ```

    References:
        https://pubs.acs.org/doi/abs/10.1021/acs.jcim.0c00740
    """

    __name__ = "pubchemqc_pm6"
    __energy_methods__ = [PotentialMethod.PM6]

    energy_target_names = ["pm6"]

    __force_methods__ = []
    force_target_names = []

    @property
    def root(self):
        return p_join(get_local_cache(), "pubchemqc")

    @property
    def preprocess_path(self):
        path = p_join(self.root, "preprocessed", self.__name__)
        os.makedirs(path, exist_ok=True)
        return path

    def collate_list(self, list_entries):
        predicat = list_entries is not None and len(list_entries) > 0
        list_entries = [x for x in list_entries if x is not None]
        if predicat:
            res = super().collate_list(list_entries)
        else:
            res = None
        return res

    @property
    def data_types(self):
        return {
            "atomic_inputs": np.float32,
            "position_idx_range": np.int32,
            "energies": np.float32,
            "forces": np.float32,
        }

    def read_raw_entries(self):
        arxiv_paths = glob(p_join(self.root, f"{self.__energy_methods__[0]}", "*.pkl"))
        f = lambda x: self.collate_list(read_preprocessed_archive(x))
        samples = dm.parallelized(f, arxiv_paths, n_jobs=1, progress=True)
        samples = [x for x in samples if x is not None]
        return samples

    def preprocess(self, overwrite=False):
        if overwrite or not self.is_preprocessed():
            logger.info("Preprocessing data and saving it to cache.")
            logger.info(
                f"Dataset {self.__name__} data with the following units:\n"
                f"Energy: {self.energy_unit}, Distance: {self.distance_unit}, "
                f"Forces: {self.force_unit if self.__force_methods__ else 'None'}"
            )
            entries = self.read_raw_entries()
            self.collate_and_save_list(entries)

    def collate_and_save_list(self, list_entries):
        n_molecules, n_atoms = 0, 0
        for i in range(len(list_entries)):
            list_entries[i]["position_idx_range"] += n_atoms
            n_atoms += list_entries[i]["position_idx_range"].max()
            n_molecules += list_entries[i]["position_idx_range"].shape[0]

        for key in self.data_keys:
            first = list_entries[0][key]
            shape = (n_molecules, *first.shape[1:])
            local_path = p_join(self.preprocess_path, f"{key}.mmap")
            out = np.memmap(local_path, mode="w+", dtype=first.dtype, shape=shape)

            start = 0
            for i in range(len(list_entries)):
                x = list_entries[i].pop(key)
                n = x.shape[0]
                out[start : start + n] = x
                out.flush()
            push_remote(local_path, overwrite=True)

        # save smiles and subset
        tmp, n = dict(name=[]), len(list_entries)
        local_path = p_join(self.preprocess_path, "props.pkl")
        names = [list_entries[i].pop("name") for i in range(n)]
        f = lambda xs: [dm.to_inchikey(x) for x in xs]
        res = dm.parallelized(f, names, n_jobs=-1, progress=False)
        for x in res:
            tmp["name"] += x
        for key in ["subset", "n_atoms"]:
            tmp[key] = []
            for i in range(n):
                tmp[key] += list(list_entries[i].pop(key))
        with open(local_path, "wb") as f:
            pkl.dump(tmp, f)
        push_remote(local_path, overwrite=True)


class PCQM_B3LYP(PCQM_PM6):
    """
    PubChemQC B3LYP/6-31G* (PCQM_B3LYP) comprises of 85 million molecules ranging from essential compounds to
    biomolecules. The geometries for the molecule are optimized using PM6. Using the optimized geometry,
    the electronic structure and properties are calculated using B3LIP/6-31G* method.

    Usage:
    ```python
    from openqdc.datasets import PCQM_B3LYP
    dataset = PCQM_B3LYP()
    ```

    References:
        https://arxiv.org/abs/2305.18454
    """

    __name__ = "pubchemqc_b3lyp"
    __energy_methods__ = ["b3lyp/6-31g*"]
    energy_target_names = ["b3lyp"]
