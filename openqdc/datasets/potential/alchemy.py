from os.path import join as p_join

import datamol as dm
import numpy as np
import pandas as pd
from tqdm import tqdm

from openqdc.datasets.base import BaseDataset
from openqdc.methods import PotentialMethod
from openqdc.utils.molecule import get_atomic_number_and_charge

# ['gdb_idx', 'atom number', 'zpve\n(Ha, zero point vibrational energy)',
#'Cv\n(cal/molK, heat capacity at 298.15 K)', 'gap\n(Ha, LUMO-HOMO)',
# 'G\n(Ha, Free energy at 298.15 K)', 'HOMO\n(Ha, energy of HOMO)',
# 'U\n(Ha, internal energy at 298.15 K)', 'alpha\n(a_0^3, Isotropic polarizability)',
# 'U0\n(Ha, internal energy at 0 K)', 'H\n(Ha, enthalpy at 298.15 K)',
# 'LUMO\n(Ha, energy of LUMO)', 'mu\n(D, dipole moment)',
# 'R2\n(a_0^2, electronic spatial extent)']


def read_mol(file, energy):
    try:
        mol = dm.read_sdf(file, remove_hs=False)[0]
        positions = mol.GetConformer().GetPositions()
        x = get_atomic_number_and_charge(mol)
        n_atoms = positions.shape[0]
        res = dict(
            atomic_inputs=np.concatenate((x, positions), axis=-1, dtype=np.float32).reshape(-1, 5),
            name=np.array([dm.to_smiles(mol)]),
            energies=np.array([energy], dtype=np.float64)[:, None],
            n_atoms=np.array([n_atoms], dtype=np.int32),
            subset=np.array([f"atoms_{n_atoms}"]),
        )

    except Exception as e:
        print(f"Skipping due to {e}")
        res = None

    return res


# e B3LYP/6-31G(2df,p) model with the density fitting
# approximation for electron repulsion integrals. The auxiliary basis cc-pVDZ-jkf


class Alchemy(BaseDataset):
    """
    Alchemy comprises of 119,487 organic molecules with up to 14 heavy atoms, sampled from the GDB MedChem database.
    Molecular properties are calculated using PySCF's implementation of the DFT Kohn-Sham method at the B3LYP level
    with the basis set 6-31G(2df,p). The equilibrium geometry is optimized in three passes. First, OpenBabel is used
    to parse SMILES string and build the Cartesian coordinates with MMFF94 force field optimization. Second, HF/STO3G
    is used to generate the preliminary geometry. Third, for the final pass of geometry relaxation, the
    B3LYP/6-31G(2df,p) model with the density fittting approximation for electron repulsion integrals is used. The
    auxillary basis cc-pVDZ-jkfit is employed in density fitting to build the Coulomb matrix and the HF exchange
    matrix.

    Usage:
    ```python
    from openqdc.datasets import Alchemy
    dataset = Alchemy()
    ```

    Reference:
        https://arxiv.org/abs/1906.09427
        https://alchemy.tencent.com/
    """

    __name__ = "alchemy"

    __energy_methods__ = [
        PotentialMethod.WB97X_6_31G_D,  # "wb97x/6-31g(d)"
    ]

    energy_target_names = [
        "ωB97x:6-31G(d) Energy",
    ]

    __energy_unit__ = "hartree"
    __distance_unit__ = "ang"
    __forces_unit__ = "hartree/ang"
    __links__ = {"alchemy.zip": "https://alchemy.tencent.com/data/alchemy-v20191129.zip"}

    def read_raw_entries(self):
        dir_path = p_join(self.root, "Alchemy-v20191129")
        full_csv = pd.read_csv(p_join(dir_path, "final_version.csv"))
        energies = full_csv["U0\n(Ha, internal energy at 0 K)"].tolist()
        atom_folder = full_csv["atom number"]
        gdb_idx = full_csv["gdb_idx"]
        idxs = full_csv.index.tolist()
        samples = []
        for i in tqdm(idxs):
            sdf_file = p_join(dir_path, f"atom_{atom_folder[i]}", f"{gdb_idx[i]}.sdf")
            energy = energies[i]
            samples.append(read_mol(sdf_file, energy))
        return samples
