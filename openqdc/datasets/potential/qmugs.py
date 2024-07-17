import os
from glob import glob
from os.path import join as p_join

import datamol as dm
import numpy as np

from openqdc.datasets.base import BaseDataset
from openqdc.methods import PotentialMethod
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
        energies=targets.astype(np.float64),
        atomic_inputs=np.concatenate((x, positions), axis=-1, dtype=np.float32).reshape(-1, 5),
        n_atoms=np.array([x.shape[1]] * n_confs, dtype=np.int32),
    )

    return res


class QMugs(BaseDataset):
    """
    The QMugs dataset contains 2 million conformers for 665k biologically and pharmacologically relevant molecules
    extracted from the ChEMBL database. Three geometries per molecule are generated and optimized using the GFN2-xTB
    method. Using the optimized geometry, the atomic and molecular properties are calculated using both, semi-empirical
    method (GFN2-xTB) and DFT method (ωB97X-D/def2-SVP).

    Usage:
    ```python
    from openqdc.datasets import QMugs
    dataset = QMugs()
    ```

    References:
        https://arxiv.org/abs/2107.00367\n
        https://www.nature.com/articles/s41597-022-01390-7#ethics\n
        https://www.research-collection.ethz.ch/handle/20.500.11850/482129
    """

    __name__ = "qmugs"
    __energy_methods__ = [PotentialMethod.GFN2_XTB, PotentialMethod.WB97X_D_DEF2_SVP]  # "gfn2_xtb", "wb97x-d/def2-svp"
    __energy_unit__ = "hartree"
    __distance_unit__ = "ang"
    __forces_unit__ = "hartree/ang"
    __links__ = {
        "summary.csv": "https://libdrive.ethz.ch/index.php/s/X5vOBNSITAG5vzM/download?path=%2F&files=summary.csv",
        "structures.tar.gz": "https://libdrive.ethz.ch/index.php/s/X5vOBNSITAG5vzM/download?path=%2F&files=structures.tar.gz",  # noqa
    }

    energy_target_names = [
        "GFN2:TOTAL_ENERGY",
        "DFT:TOTAL_ENERGY",
    ]

    def read_raw_entries(self):
        raw_path = p_join(self.root, "structures")
        mol_dirs = [p_join(raw_path, d) for d in os.listdir(raw_path)]

        samples = dm.parallelized(read_mol, mol_dirs, n_jobs=-1, progress=True, scheduler="threads")
        return samples


class QMugs_V2(QMugs):
    """
    QMugs_V2 is an extension of the QMugs dataset containing PM6 labels for each of the 4.2M geometries.

    Usage:
    ```python
    from openqdc.datasets import QMugs_V2
    dataset = QMugs_V2()
    ```
    """

    __name__ = "qmugs_v2"
    __energy_methods__ = QMugs.__energy_methods__ + [PotentialMethod.PM6]
    energy_target_names = QMugs.energy_target_names + ["PM6"]
    __force_mask__ = QMugs.__force_mask__ + [False]
