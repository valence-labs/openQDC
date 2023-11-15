from os.path import join as p_join

import datamol as dm
import numpy as np
from numpy import array
from tqdm import tqdm

from openqdc.datasets.base import BaseDataset
from openqdc.utils import load_hdf5_file
from openqdc.utils.molecule import get_atomic_number_and_charge


def read_record(r):
    smiles = r["smiles"].asstr()[0]
    subset = r["subset"][0].decode("utf-8")
    n_confs = r["conformations"].shape[0]
    x = get_atomic_number_and_charge(dm.to_mol(smiles, add_hs=True))
    positions = r["conformations"][:]

    res = dict(
        name=np.array([smiles] * n_confs),
        subset=np.array([Spice.subset_mapping[subset]] * n_confs),
        energies=r[Spice.energy_target_names[0]][:][:, None].astype(np.float32),
        forces=r[Spice.force_target_names[0]][:].reshape(
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
    Spice Dataset consists of 1.1 million conformations for a diverse set of 19k unique molecules consisting of
    small molecules, dimers, dipeptides, and solvated amino acids. It consists of both forces and energies calculated
    at {\omega}B97M-D3(BJ)/def2-TZVPPD level of theory.

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
    __energy_methods__ = ["wb97m-d3bj/def2-tzvppd"]
    __force_methods__ = ["wb97m-d3bj/def2-tzvppd"]
    __energy_unit__ = "hartree"
    __distance_unit__ = "bohr"
    __forces_unit__ = "hartree/bohr"
    __average_nb_atoms__ = 29.88387509402179

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

    @property
    def _stats(self):
        return {
            "formation": {
                "energy": {
                    "mean": self.convert_energy(array([-5.67757058])),
                    "std": self.convert_energy(array([2.33714861])),
                },
                "forces": {
                    "mean": self.convert_forces(array([-1.0387013e-08])),
                    "std": self.convert_forces(array([0.021063408])),
                    "components": {
                        "mean": self.convert_forces(array([[5.7479990e-09], [-4.8940532e-08], [1.2032132e-08]])),
                        "std": self.convert_forces(array([[0.02017307], [0.02016141], [0.02014796]])),
                        "rms": self.convert_forces(array([[0.02017307], [0.02016142], [0.02014796]])),
                    },
                },
            },
            "total": {
                "energy": {
                    "mean": self.convert_energy(array([-1244.6562])),
                    "std": self.convert_energy(array([1219.4248])),
                },
                "forces": {
                    "mean": self.convert_forces(array([-1.0387013e-08])),
                    "std": self.convert_forces(array([0.021063408])),
                    "components": {
                        "mean": self.convert_forces(array([[5.7479990e-09], [-4.8940532e-08], [1.2032132e-08]])),
                        "std": self.convert_forces(array([[0.02017307], [0.02016141], [0.02014796]])),
                        "rms": self.convert_forces(array([[0.02017307], [0.02016142], [0.02014796]])),
                    },
                },
            },
        }

    def convert_forces(self, x):
        return (-1.0) * super().convert_forces(x)

    def read_raw_entries(self):
        raw_path = p_join(self.root, "SPICE-1.1.4.hdf5")

        data = load_hdf5_file(raw_path)
        tmp = [read_record(data[mol_name]) for mol_name in tqdm(data)]  # don't use parallelized here

        return tmp
