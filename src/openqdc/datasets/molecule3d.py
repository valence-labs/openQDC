from glob import glob
from os.path import join as p_join

import datamol as dm
import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from openqdc.datasets.base import BaseDataset
from openqdc.utils.constants import MAX_ATOMIC_NUMBER
from openqdc.utils.molecule import get_atomic_number_and_charge


def read_mol(mol, energy):
    smiles = dm.to_smiles(mol, explicit_hs=False)
    # subset = dm.to_smiles(dm.to_scaffold_murcko(mol, make_generic=True), explicit_hs=False)
    x = get_atomic_number_and_charge(mol)
    positions = mol.GetConformer().GetPositions()

    res = dict(
        name=np.array([smiles]),
        subset=np.array(["molecule3d"]),
        energies=np.array([energy]).astype(np.float32)[:, None],
        atomic_inputs=np.concatenate((x, positions), axis=-1, dtype=np.float32),
        n_atoms=np.array([x.shape[0]], dtype=np.int32),
    )

    return res


def _read_sdf(sdf_path, properties_path):
    properties = pd.read_csv(properties_path, dtype={"cid": str})
    properties.drop_duplicates(subset="cid", inplace=True, keep="first")
    xys = properties[["cid", "scf energy"]]
    properties = dict(zip(xys.cid.values, xys["scf energy"].values))

    get_e = lambda mol: properties[mol.GetProp("_Name").split(" ")[1]]
    fn = lambda x: read_mol(x, get_e(x))

    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
    tmp = [fn(suppl[j]) for j in tqdm(range(len(suppl)))]

    return tmp


class Molecule3D(BaseDataset):
    __name__ = "molecule3d"
    __energy_methods__ = ["b3lyp/6-31g*"]
    # UNITS MOST LIKELY WRONG, MUST CHECK THEM MANUALLY
    __energy_unit__ = "hartree"
    __distance_unit__ = "ang"
    __forces_unit__ = "hartree/ang"

    energy_target_names = ["b3lyp/6-31g*.energy"]

    # Energy in hartree, all zeros by default
    atomic_energies = np.zeros((MAX_ATOMIC_NUMBER,), dtype=np.float32)

    def __init__(self, energy_unit=None, distance_unit=None) -> None:
        super().__init__(energy_unit=energy_unit, distance_unit=distance_unit)

    def read_raw_entries(self):
        raw = p_join(self.root, "data", "raw")
        sdf_paths = glob(p_join(raw, "*.sdf"))
        properties_path = p_join(raw, "properties.csv")

        fn = lambda x: _read_sdf(x, properties_path)
        res = dm.parallelized(fn, sdf_paths, n_jobs=1)  # don't use more than 1 job
        samples = sum(res, [])
        return samples


if __name__ == "__main__":
    for data_class in [Molecule3D]:
        data = data_class()
        n = len(data)

        for i in np.random.choice(n, 3, replace=False):
            x = data[i]
            print(x.name, x.subset, end=" ")
            for k in x:
                if x[k] is not None:
                    print(k, x[k].shape, end=" ")

            print()
