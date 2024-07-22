from io import StringIO
from os.path import join as p_join

import numpy as np
import pandas as pd
from tqdm import tqdm

from openqdc.datasets.base import BaseDataset
from openqdc.methods import PotentialMethod
from openqdc.utils.constants import ATOM_TABLE


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
    z = np.array([ATOM_TABLE.GetAtomicNumber(s) for s in z])
    xs = np.stack((z, np.zeros_like(z)), axis=-1)
    e = e_map[code]

    conf = dict(
        atomic_inputs=np.concatenate((xs, positions), axis=-1, dtype=np.float32),
        name=np.array([name]),
        energies=np.array([e], dtype=np.float64)[:, None],
        n_atoms=np.array([positions.shape[0]], dtype=np.int32),
        subset=np.array(["tmqm"]),
    )

    return conf


def read_xyz(fname, e_map):
    with open(fname, "r") as f:
        contents = f.read().split("\n\n")

    res = [content_to_xyz(content, e_map) for content in tqdm(contents)]
    return res


class TMQM(BaseDataset):
    """
    tmQM dataset contains the geometries of a large transition metal-organic compound space with a large variety of
    organic ligands and 30 transition metals. It contains energy labels for 86,665 mononuclear complexes calculated
    at the TPSSh-D3BJ/def2-SV DFT level of theory. Structures are first extracted from Cambridge Structure Database
    and then optimized in gas phase with the extended tight-binding GFN2-xTB method.

    Usage:
    ```python
    from openqdc.datasets import TMQM
    dataset = TMQM()
    ```

    References:
        https://pubs.acs.org/doi/10.1021/acs.jcim.0c01041\n
        https://github.com/bbskjelstad/tmqm
    """

    __name__ = "tmqm"

    __energy_methods__ = [PotentialMethod.TPSSH_DEF2_TZVP]  # "tpssh/def2-tzvp"]

    energy_target_names = ["TPSSh/def2TZVP level"]

    __energy_unit__ = "hartree"
    __distance_unit__ = "ang"
    __forces_unit__ = "hartree/ang"
    __links__ = {
        x: f"https://raw.githubusercontent.com/bbskjelstad/tmqm/master/data/{x}"
        for x in ["tmQM_X1.xyz.gz", "tmQM_X2.xyz.gz", "tmQM_y.csv", "Benchmark2_TPSSh_Opt.xyz"]
    }

    def read_raw_entries(self):
        df = pd.read_csv(p_join(self.root, "tmQM_y.csv"), sep=";", usecols=["CSD_code", "Electronic_E"])
        e_map = dict(zip(df["CSD_code"], df["Electronic_E"]))
        raw_fnames = ["tmQM_X1.xyz", "tmQM_X2.xyz", "Benchmark2_TPSSh_Opt.xyz"]
        samples = []
        for fname in raw_fnames:
            data = read_xyz(p_join(self.root, fname), e_map)
            samples += data

        return samples
