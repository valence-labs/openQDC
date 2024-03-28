from os.path import join as p_join

import datamol as dm
import numpy as np
from tqdm import tqdm
from openqdc.methods import QmMethod
from openqdc.datasets.base import BaseDataset
from openqdc.utils import load_hdf5_file
from openqdc.utils.molecule import get_atomic_number_and_charge


def read_record(r, obj):
    """
    Read record from hdf5 file.
        r : hdf5 record
        obj : Spice class object used to grab subset and names
    """
    smiles = r["smiles"].asstr()[0]
    subset = r["subset"][0].decode("utf-8")
    n_confs = r["conformations"].shape[0]
    x = get_atomic_number_and_charge(dm.to_mol(smiles, remove_hs=False, ordered=True))
    positions = r["conformations"][:]

    res = dict(
        name=np.array([smiles] * n_confs),
        subset=np.array([obj.subset_mapping[subset]] * n_confs),
        energies=r[obj.energy_target_names[0]][:][:, None].astype(np.float32),
        forces=r[obj.force_target_names[0]][:].reshape(
            -1, 3, 1
        ),  # forces -ve of energy gradient but the -1.0 is done in the convert_forces method
        atomic_inputs=np.concatenate(
            (x[None, ...].repeat(n_confs, axis=0), positions), axis=-1, dtype=np.float32
        ).reshape(-1, 5),
        n_atoms=np.array([x.shape[0]] * n_confs, dtype=np.int32),
    )

    return res


class Spice(BaseDataset):
    """
    The Spice dataset consists of 1.1 million conformations for a diverse set of 19k unique molecules consisting of
    small molecules, dimers, dipeptides, and solvated amino acids. It consists of both forces and energies calculated
    at the {\omega}B97M-D3(BJ)/def2-TZVPPD level of theory.

    Usage:
    ```python
    from openqdc.datasets import Spice
    dataset = Spice()
    ```

    References:
    - https://arxiv.org/abs/2209.10702
    - https://github.com/openmm/spice-dataset
    """

    __name__ = "spice"
    __energy_methods__ = [QmMethod.WB97M_D3BJ_DEF2_TZVPPD] # "wb97m-d3bj/def2-tzvppd"]
    __force_mask__ = [True]
    __energy_unit__ = "hartree"
    __distance_unit__ = "bohr"
    __forces_unit__ = "hartree/bohr"

    energy_target_names = ["dft_total_energy"]

    force_target_names = ["dft_total_gradient"]

    subset_mapping = {
        "SPICE Solvated Amino Acids Single Points Dataset v1.1": "Solvated Amino Acids",
        "SPICE Dipeptides Single Points Dataset v1.2": "Dipeptides",
        "SPICE DES Monomers Single Points Dataset v1.1": "DES370K Monomers",
        "SPICE DES370K Single Points Dataset v1.0": "DES370K Dimers",
        "SPICE DES370K Single Points Dataset Supplement v1.0": "DES370K Dimers",
        "SPICE PubChem Set 1 Single Points Dataset v1.2": "PubChem",
        "SPICE PubChem Set 2 Single Points Dataset v1.2": "PubChem",
        "SPICE PubChem Set 3 Single Points Dataset v1.2": "PubChem",
        "SPICE PubChem Set 4 Single Points Dataset v1.2": "PubChem",
        "SPICE PubChem Set 5 Single Points Dataset v1.2": "PubChem",
        "SPICE PubChem Set 6 Single Points Dataset v1.2": "PubChem",
        "SPICE Ion Pairs Single Points Dataset v1.1": "Ion Pairs",
    }

    def convert_forces(self, x):
        return (-1.0) * super().convert_forces(x)

    def read_raw_entries(self):
        raw_path = p_join(self.root, "SPICE-1.1.4.hdf5")

        data = load_hdf5_file(raw_path)
        tmp = [read_record(data[mol_name], self) for mol_name in tqdm(data)]  # don't use parallelized here

        return tmp


class SpiceV2(Spice):
    """
    SpiceV2 dataset augmented with amino acids complexes, water boxes,
    pubchem solvated molecules.
    It consists of both forces and energies calculated
    at the {\omega}B97M-D3(BJ)/def2-TZVPPD level of theory.

    Usage:
    ```python
    from openqdc.datasets import SpiceV2
    dataset = SpiceV2()
    ```

    References:
    - https://github.com/openmm/spice-dataset/releases/tag/2.0.0
    - https://github.com/openmm/spice-dataset
    """

    __name__ = "spicev2"

    subset_mapping = {
        "SPICE Dipeptides Single Points Dataset v1.3": "Dipeptides",
        "SPICE Solvated Amino Acids Single Points Dataset v1.1": "Solvated Amino Acids",
        "SPICE Water Clusters v1.0": "Water Clusters",
        "SPICE Solvated PubChem Set 1 v1.0": "Solvated PubChem",
        "SPICE Amino Acid Ligand v1.0": "Amino Acid Ligand",
        "SPICE PubChem Set 1 Single Points Dataset v1.3": "PubChem",
        "SPICE PubChem Set 2 Single Points Dataset v1.3": "PubChem",
        "SPICE PubChem Set 3 Single Points Dataset v1.3": "PubChem",
        "SPICE PubChem Set 4 Single Points Dataset v1.3": "PubChem",
        "SPICE PubChem Set 5 Single Points Dataset v1.3": "PubChem",
        "SPICE PubChem Set 6 Single Points Dataset v1.3": "PubChem",
        "SPICE PubChem Set 7 Single Points Dataset v1.0": "PubChem",
        "SPICE PubChem Set 8 Single Points Dataset v1.0": "PubChem",
        "SPICE PubChem Set 9 Single Points Dataset v1.0": "PubChem",
        "SPICE PubChem Set 10 Single Points Dataset v1.0": "PubChem",
        "SPICE DES Monomers Single Points Dataset v1.1": "DES370K Monomers",
        "SPICE DES370K Single Points Dataset v1.0": "DES370K Dimers",
        "SPICE DES370K Single Points Dataset Supplement v1.1": "DES370K Dimers",
        "SPICE PubChem Boron Silicon v1.0": "PubChem Boron Silicon",
        "SPICE Ion Pairs Single Points Dataset v1.2": "Ion Pairs",
    }

    def read_raw_entries(self):
        raw_path = p_join(self.root, "spice-2.0.0.hdf5")

        data = load_hdf5_file(raw_path)
        # Entry 40132 without positions, skip it
        # don't use parallelized here
        tmp = [read_record(data[mol_name], self) for i, mol_name in enumerate(tqdm(data)) if i != 40132]

        return tmp
