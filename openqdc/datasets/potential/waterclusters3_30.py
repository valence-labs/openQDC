import zipfile
from io import StringIO
from os.path import join as p_join

import numpy as np
from tqdm import tqdm

from openqdc.datasets.base import BaseDataset
from openqdc.methods import PotentialMethod
from openqdc.utils.constants import ATOM_TABLE, MAX_ATOMIC_NUMBER


def content_to_xyz(content, n_waters):
    content = content.strip()

    try:
        tmp = content.splitlines()
        s = StringIO(content)
        d = np.loadtxt(s, skiprows=2, dtype="str")
        z, positions = d[:, 0], d[:, 1:].astype(np.float32)
        z = np.array([ATOM_TABLE.GetAtomicNumber(s) for s in z])
        xs = np.stack((z, np.zeros_like(z)), axis=-1)
        e = float(tmp[1].strip().split(" ")[-1])
    except Exception:
        print("Error in reading xyz file")
        print(n_waters, content)
        return None

    conf = dict(
        atomic_inputs=np.concatenate((xs, positions), axis=-1, dtype=np.float32),
        name=np.array([f"water_{n_waters}"]),
        energies=np.array([e], dtype=np.float64)[:, None],
        n_atoms=np.array([positions.shape[0]], dtype=np.int32),
        subset=np.array([f"water_{n_waters}"]),
    )

    return conf


def read_xyz(fname, n_waters):
    s = 3 * n_waters + 2
    with open(fname, "r") as f:
        lines = f.readlines()
        contents = ["".join(lines[i : i + s]) for i in range(0, len(lines), s)]

    res = [content_to_xyz(content, n_waters) for content in tqdm(contents)]
    return res


class WaterClusters(BaseDataset):
    """
    The WaterClusters dataset contains putative minima and low energy networks for water
    clusters of sizes n = 3 - 30. The cluster structures are derived and labeled with
    the TTM2.1-F ab-initio based interaction potential for water.
    It contains approximately 4.5 mil. structures.
    Sampling was done with the Monte Carlo Temperature Basin Paving (MCTBP) method.

    Chemical Species:
        ["H", "O"]

    Usage:
    ```python
    from openqdc.datasets import WaterClusters
    dataset = WaterClusters()
    ```

    References:
        https://doi.org/10.1063/1.5128378\n
        https://sites.uw.edu/wdbase/database-of-water-clusters/\n
    """

    __name__ = "waterclusters3_30"

    # Energy in hartree, all zeros by default
    atomic_energies = np.zeros((MAX_ATOMIC_NUMBER,), dtype=np.float32)
    __energy_unit__ = "kcal/mol"
    __distance_unit__ = "ang"
    __forces_unit__ = "kcal/mol/ang"

    __energy_methods__ = [PotentialMethod.TTM2_1_F]  # "ttm2.1-f"
    energy_target_names = ["TTM2.1-F Potential"]
    __links__ = {"W3-W30_all_geoms_TTM2.1-F.zip": "https://drive.google.com/uc?id=18Y7OiZXSCTsHrQ83GCc4fyE_abbL6E_n"}

    def read_raw_entries(self):
        samples = []
        parent_folder = p_join(self.root, "W3-W30_all_geoms_TTM2.1-F/")
        for i in range(3, 31):
            name = f"W{i}_geoms_all"
            zip_path = p_join(parent_folder, f"{name}.zip")
            xyz_path = p_join(parent_folder, f"{name}.xyz")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(parent_folder)

            data = read_xyz(xyz_path, i)
            samples += data

        return samples
