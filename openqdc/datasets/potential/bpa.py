from typing import Any, Dict, List

import numpy as np
from ase.atoms import Atoms

from openqdc import BaseDataset
from openqdc.methods import PotentialMethod


def read_bpa_record(subset: str, atoms: Atoms) -> Dict[str, Any]:
    return dict(
        name=np.array([str(atoms.symbols)]),
        subset=subset,
        energies=np.array([atoms.get_potential_energy()], dtype=np.float64),
        forces=atoms.get_forces().reshape(-1, 3, 1).astype(np.float32),
        atomic_inputs=np.column_stack((atoms.numbers, atoms.get_initial_charges(), atoms.positions)).astype(np.float32),
        n_atoms=np.array([len(atoms)], dtype=np.int32),
        split=np.array([subset.item().split("_")[0]]),
    )


class BPA(BaseDataset):
    """
    BPA (or 3BPA) dataset is a dataset consisting of a flexible druglike
    molecule 3-(benzyloxy)pyridin-2-amine. This dataset features
    complex dihedral potential energy surface with many local minima,
    which can be challenging to approximate using classical or ML force fields.
    The configuration were sampled from short (0.5 ps) MD simulations using the ANI-1x force field to
    perturb the toward lower potential energies. Furthermore, long 25 ps MD simulation were performed at
    three different temperatures (300, 600, and 1200 K) using the Langevin thermostat and a 1 fs time step.
    The final configurations were re-evaluated using ORCA at the DFT level of
    theory using the ωB97X exchange correlation functional and the 6-31G(d) basis set.

    Usage:
    ```python
    from openqdc.datasets import BPA
    dataset = BPA()
    ```


    References:
        https://pubs.acs.org/doi/10.1021/acs.jctc.1c00647
    """

    __name__ = "BPA"
    __energy_unit__ = "ev"
    __forces_unit__ = "ev/ang"
    __distance_unit__ = "ang"
    __force_mask__ = [True]
    __energy_methods__ = [PotentialMethod.WB97X_6_31G_D]
    __links__ = {"BPA.zip": "https://figshare.com/ndownloader/files/31325990"}

    def read_raw_entries(self) -> List[Dict]:
        import os.path as osp
        from glob import glob

        from ase.io import iread

        files = glob(osp.join(self.root, "dataset_3BPA", "*.xyz"))
        files = [f for f in files if "iso_atoms.xyz" not in f]
        all_records = []

        for file in files:
            subset = np.array([osp.basename(file).split(".")[0]])

            for atoms in iread(file, format="extxyz"):
                all_records.append(read_bpa_record(subset, atoms))

        return all_records

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data.__setattr__("split", self._convert_array(self.data["split"][idx]))
        return data
