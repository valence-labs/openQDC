import os
from glob import glob
from os.path import join as p_join

import datamol as dm
import numpy as np

from openqdc.datasets.base import BaseDataset
from openqdc.utils.constants import MAX_ATOMIC_NUMBER
from openqdc.utils.molecule import get_atomic_number_and_charge


def read_mol(mol_dir):
    filenames = glob(p_join(mol_dir, "*.sdf"))
    mols = [dm.read_sdf(f, remove_hs=False)[0] for f in filenames]
    n_confs = len(mols)

    if len(mols) == 0:
        return None

    smiles = dm.to_smiles(mols[0], explicit_hs=False)
    x = get_atomic_number_and_charge(mols[0])[None, ...].repeat(n_confs, axis=0)
    positions = np.array([mol.GetConformer().GetPositions() for mol in mols])
    props = [mol.GetPropsAsDict() for mol in mols]
    targets = np.array([[p[el] for el in QMugs.energy_target_names] for p in props])

    res = dict(
        name=np.array([smiles] * n_confs),
        subset=np.array(["qmugs"] * n_confs),
        energies=targets.astype(np.float32),
        atomic_inputs=np.concatenate((x, positions), axis=-1, dtype=np.float32).reshape(-1, 5),
        n_atoms=np.array([x.shape[1]] * n_confs, dtype=np.int32),
    )

    return res


class QMugs(BaseDataset):
    __name__ = "qmugs"
    __energy_methods__ = ["gfn2_xtb", "b3lyp/6-31g*"]
    __energy_unit__ = "hartree"
    __distance_unit__ = "ang"
    __forces_unit__ = "hartree/ang"

    energy_target_names = [
        "GFN2:TOTAL_ENERGY",
        "DFT:TOTAL_ENERGY",
    ]

    # Energy in hartree, all zeros by default
    atomic_energies = np.zeros((MAX_ATOMIC_NUMBER,), dtype=np.float32)

    def __init__(self, energy_unit=None, distance_unit=None) -> None:
        super().__init__(energy_unit=energy_unit, distance_unit=distance_unit)

    def read_raw_entries(self):
        raw_path = p_join(self.root, "structures")
        mol_dirs = [p_join(raw_path, d) for d in os.listdir(raw_path)]

        samples = dm.parallelized(read_mol, mol_dirs, n_jobs=-1, progress=True, scheduler="threads")
        return samples


if __name__ == "__main__":
    for data_class in [QMugs]:
        data = data_class()
        n = len(data)

        for i in np.random.choice(n, 3, replace=False):
            x = data[i]
            print(x.name, x.subset, end=" ")
            for k in x:
                if x[k] is not None:
                    print(k, x[k].shape, end=" ")

            print()
