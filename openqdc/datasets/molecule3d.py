from glob import glob
from os.path import join as p_join
from typing import Dict, List

import datamol as dm
import numpy as np
import pandas as pd
from openqdc.datasets.base import BaseDataset
from openqdc.utils.molecule import get_atomic_number_and_charge
from rdkit import Chem
from tqdm import tqdm


def read_mol(mol: Chem.rdchem.Mol, energy: float) -> Dict[str, np.ndarray]:
    """Read molecule (Chem.rdchem.Mol) and energy (float) and return dict with conformers and energies

    Parameters
    ----------
    mol: Chem.rdchem.Mol
        RDKit molecule
    energy: float
        Energy of the molecule

    Returns
    -------
    res: dict
        Dictionary containing the following keys:
        - name: np.ndarray of shape (N,) containing the smiles of the molecule
        - atomic_inputs: flatten np.ndarray of shape (M, 5) containing the atomic numbers, charges and positions
        - energies: np.ndarray of shape (1,) containing the energy of the conformer
        - n_atoms: np.ndarray of shape (1) containing the number of atoms in the conformer
        - subset: np.ndarray of shape (1) containing "molecule3d"
    """
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


def _read_sdf(sdf_path: str, properties_path: str) -> List[Dict[str, np.ndarray]]:
    """Reads the sdf path and properties file."""
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
    """
    Molecule3D dataset consists of 3,899,647 molecules with ground state geometries and energies
    calculated at the B3LYP/6-31G* level of theory. The molecules are extracted from the
    PubChem database and cleaned by removing invalid molecule files.

    Usage:
    ```python
    from openqdc.datasets import Molecule3D
    dataset = Molecule3D()
    ```

    References:
    - https://arxiv.org/abs/2110.01717
    - https://github.com/divelab/MoleculeX
    """

    __name__ = "molecule3d"
    __energy_methods__ = ["b3lyp/6-31g*"]
    # UNITS MOST LIKELY WRONG, MUST CHECK THEM MANUALLY
    __energy_unit__ = "ev"  # CALCULATED
    __distance_unit__ = "ang"
    __forces_unit__ = "ev/ang"

    energy_target_names = ["b3lyp/6-31g*.energy"]

    def read_raw_entries(self):
        raw = p_join(self.root, "data", "raw")
        sdf_paths = glob(p_join(raw, "*.sdf"))
        properties_path = p_join(raw, "properties.csv")

        fn = lambda x: _read_sdf(x, properties_path)
        res = dm.parallelized(fn, sdf_paths, n_jobs=1)  # don't use more than 1 job
        samples = sum(res, [])
        return samples
