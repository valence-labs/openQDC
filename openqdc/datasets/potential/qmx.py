import os
from abc import ABC
from os.path import join as p_join

import datamol as dm
import numpy as np
import pandas as pd

from openqdc.datasets.base import BaseDataset
from openqdc.methods import PotentialMethod
from openqdc.utils import read_qc_archive_h5
from openqdc.utils.io import get_local_cache
from openqdc.utils.molecule import get_atomic_number_and_charge


def extract_ani2_entries(properties):
    coordinates = properties["coordinates"]
    species = properties["species"]
    forces = properties["forces"]
    energies = properties["energies"]
    n_atoms = coordinates.shape[1]
    n_entries = coordinates.shape[0]
    flattened_coordinates = coordinates[:].reshape((-1, 3))
    xs = np.stack((species[:].flatten(), np.zeros(flattened_coordinates.shape[0])), axis=-1)
    res = dict(
        name=np.array(["ANI2"] * n_entries),
        subset=np.array([str(n_atoms)] * n_entries),
        energies=energies[:].reshape((-1, 1)).astype(np.float64),
        atomic_inputs=np.concatenate((xs, flattened_coordinates), axis=-1, dtype=np.float32),
        n_atoms=np.array([n_atoms] * n_entries, dtype=np.int32),
        forces=forces[:].reshape(-1, 3, 1).astype(np.float32),
    )
    return res


class QMX(ABC, BaseDataset):
    """
    QMX dataset base abstract class
    """

    __name__ = "qm9"

    __energy_methods__ = [
        PotentialMethod.WB97X_6_31G_D,  # "wb97x/6-31g(d)"
    ]

    energy_target_names = [
        "ωB97x:6-31G(d) Energy",
    ]

    __energy_unit__ = "hartree"
    __distance_unit__ = "bohr"
    __forces_unit__ = "hartree/bohr"
    __links__ = {}

    @property
    def root(self):
        return p_join(get_local_cache(), "qmx")

    @property
    def preprocess_path(self):
        path = p_join(self.root, "preprocessed", self.__name__)
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def config(self):
        assert len(self.__links__) > 0, "No links provided for fetching"
        return dict(dataset_name="qmx", links=self.__links__)

    def read_raw_entries(self):
        raw_path = p_join(self.root, f"{self.__name__}.h5.gz")
        samples = read_qc_archive_h5(raw_path, self.__name__, self.energy_target_names, None)
        return samples


# ['smiles', 'E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2', 'E1-PBE0', 'E2-PBE0', 'f1-PBE0', 'f2-PBE0',
# 'E1-PBE0.1', 'E2-PBE0.1', 'f1-PBE0.1', 'f2-PBE0.1', 'E1-CAM', 'E2-CAM', 'f1-CAM', 'f2-CAM']
class QM7(QMX):
    __links__ = {"qm7.hdf5.gz": "https://zenodo.org/record/3588337/files/150.hdf5.gz?download=1"}
    __name__ = "qm7"

    energy_target_names = [
        "B2PLYP-D3(BJ):aug-cc-pvdz",
        "B2PLYP-D3(BJ):aug-cc-pvtz",
        "B2PLYP-D3(BJ):def2-svp",
        "B2PLYP-D3(BJ):def2-tzvp",
        "B2PLYP-D3(BJ):sto-3g",
        "B2PLYP-D3:aug-cc-pvdz",
        "B2PLYP-D3:aug-cc-pvtz",
        "B2PLYP-D3:def2-svp",
        "B2PLYP-D3:def2-tzvp",
        "B2PLYP-D3:sto-3g",
        "B2PLYP-D3M(BJ):aug-cc-pvdz",
        "B2PLYP-D3M(BJ):aug-cc-pvtz",
        "B2PLYP-D3M(BJ):def2-svp",
        "B2PLYP-D3M(BJ):def2-tzvp",
        "B2PLYP-D3M(BJ):sto-3g",
        "B2PLYP-D3M:aug-cc-pvdz",
        "B2PLYP-D3M:aug-cc-pvtz",
        "B2PLYP-D3M:def2-svp",
        "B2PLYP-D3M:def2-tzvp",
        "B2PLYP-D3M:sto-3g",
        "B2PLYP:aug-cc-pvdz",
        "B2PLYP:aug-cc-pvtz",
        "B2PLYP:def2-svp",
        "B2PLYP:def2-tzvp",
        "B2PLYP:sto-3g",
        "B3LYP-D3(BJ):aug-cc-pvdz",
        "B3LYP-D3(BJ):aug-cc-pvtz",
        "B3LYP-D3(BJ):def2-svp",
        "B3LYP-D3(BJ):def2-tzvp",
        "B3LYP-D3(BJ):sto-3g",
        "B3LYP-D3:aug-cc-pvdz",
        "B3LYP-D3:aug-cc-pvtz",
        "B3LYP-D3:def2-svp",
        "B3LYP-D3:def2-tzvp",
        "B3LYP-D3:sto-3g",
        "B3LYP-D3M(BJ):aug-cc-pvdz",
        "B3LYP-D3M(BJ):aug-cc-pvtz",
        "B3LYP-D3M(BJ):def2-svp",
        "B3LYP-D3M(BJ):def2-tzvp",
        "B3LYP-D3M(BJ):sto-3g",
        "B3LYP-D3M:aug-cc-pvdz",
        "B3LYP-D3M:aug-cc-pvtz",
        "B3LYP-D3M:def2-svp",
        "B3LYP-D3M:def2-tzvp",
        "B3LYP-D3M:sto-3g",
        "B3LYP:aug-cc-pvdz",
        "B3LYP:aug-cc-pvtz",
        "B3LYP:def2-svp",
        "B3LYP:def2-tzvp",
        "B3LYP:sto-3g",
        "HF:aug-cc-pvdz",
        "HF:aug-cc-pvtz",
        "HF:def2-svp",
        "HF:def2-tzvp",
        "HF:sto-3g",
        "MP2:aug-cc-pvdz",
        "MP2:aug-cc-pvtz",
        "MP2:def2-svp",
        "MP2:def2-tzvp",
        "MP2:sto-3g",
        "PBE0:aug-cc-pvdz",
        "PBE0:aug-cc-pvtz",
        "PBE0:def2-svp",
        "PBE0:def2-tzvp",
        "PBE0:sto-3g",
        "PBE:aug-cc-pvdz",
        "PBE:aug-cc-pvtz",
        "PBE:def2-svp",
        "PBE:def2-tzvp",
        "PBE:sto-3g",
        "WB97M-V:aug-cc-pvdz",
        "WB97M-V:aug-cc-pvtz",
        "WB97M-V:def2-svp",
        "WB97M-V:def2-tzvp",
        "WB97M-V:sto-3g",
        "WB97X-D:aug-cc-pvdz",
        "WB97X-D:aug-cc-pvtz",
        "WB97X-D:def2-svp",
        "WB97X-D:def2-tzvp",
        "WB97X-D:sto-3g",
    ]

    __energy_methods__ = [PotentialMethod.NONE for _ in range(len(energy_target_names))]  # "wb97x/6-31g(d)"


class QM7b(QMX):
    __links__ = {"qm7b.hdf5.gz": "https://zenodo.org/record/3588335/files/200.hdf5.gz?download=1"}
    __name__ = "qm7b"
    energy_target_names = [
        "CCSD(T0):cc-pVDZ",
        "HF:cc-pVDZ",
        "HF:cc-pVTZ",
        "MP2:cc-pVTZ",
        "B2PLYP-D3:aug-cc-pvdz",
        "B2PLYP-D3:aug-cc-pvtz",
        "B2PLYP-D3:def2-svp",
        "B2PLYP-D3:def2-tzvp",
        "B2PLYP-D3:sto-3g",
        "B2PLYP-D3M(BJ):aug-cc-pvdz",
        "B2PLYP-D3M(BJ):aug-cc-pvtz",
        "B2PLYP-D3M(BJ):def2-svp",
        "B2PLYP-D3M(BJ):def2-tzvp",
        "B2PLYP-D3M(BJ):sto-3g",
        "B2PLYP-D3M:aug-cc-pvdz",
        "B2PLYP-D3M:aug-cc-pvtz",
        "B2PLYP-D3M:def2-svp",
        "B2PLYP-D3M:def2-tzvp",
        "B2PLYP-D3M:sto-3g",
        "B2PLYP:aug-cc-pvdz",
        "B2PLYP:aug-cc-pvtz",
        "B2PLYP:def2-svp",
        "B2PLYP:def2-tzvp",
        "B2PLYP:sto-3g",
        "B3LYP-D3(BJ):aug-cc-pvdz",
        "B3LYP-D3(BJ):aug-cc-pvtz",
        "B3LYP-D3(BJ):def2-svp",
        "B3LYP-D3(BJ):def2-tzvp",
        "B3LYP-D3(BJ):sto-3g",
        "B3LYP-D3:aug-cc-pvdz",
        "B3LYP-D3:aug-cc-pvtz",
        "B3LYP-D3:def2-svp",
        "B3LYP-D3:def2-tzvp",
        "B3LYP-D3:sto-3g",
        "B3LYP-D3M(BJ):aug-cc-pvdz",
        "B3LYP-D3M(BJ):aug-cc-pvtz",
        "B3LYP-D3M(BJ):def2-svp",
        "B3LYP-D3M(BJ):def2-tzvp",
        "B3LYP-D3M(BJ):sto-3g",
        "B3LYP-D3M:aug-cc-pvdz",
        "B3LYP-D3M:aug-cc-pvtz",
        "B3LYP-D3M:def2-svp",
        "B3LYP-D3M:def2-tzvp",
        "B3LYP-D3M:sto-3g",
        "B3LYP:aug-cc-pvdz",
        "B3LYP:aug-cc-pvtz",
        "B3LYP:def2-svp",
        "B3LYP:def2-tzvp",
        "B3LYP:sto-3g",
        "HF:aug-cc-pvdz",
        "HF:aug-cc-pvtz",
        "HF:cc-pvtz",
        "HF:def2-svp",
        "HF:def2-tzvp",
        "HF:sto-3g",
        "PBE0:aug-cc-pvdz",
        "PBE0:aug-cc-pvtz",
        "PBE0:def2-svp",
        "PBE0:def2-tzvp",
        "PBE0:sto-3g",
        "PBE:aug-cc-pvdz",
        "PBE:aug-cc-pvtz",
        "PBE:def2-svp",
        "PBE:def2-tzvp",
        "PBE:sto-3g",
        "SVWN:sto-3g",
        "WB97M-V:aug-cc-pvdz",
        "WB97M-V:aug-cc-pvtz",
        "WB97M-V:def2-svp",
        "WB97M-V:def2-tzvp",
        "WB97M-V:sto-3g",
        "WB97X-D:aug-cc-pvdz",
        "WB97X-D:aug-cc-pvtz",
        "WB97X-D:def2-svp",
        "WB97X-D:def2-tzvp",
        "WB97X-D:sto-3g",
    ]
    __energy_methods__ = [PotentialMethod.NONE for _ in range(len(energy_target_names))]  # "wb97x/6-31g(d)"]


class QM8(QMX):
    """QM8 is the dataset used in a study on modeling quantum
    mechanical calculations of electronic spectra and excited
    state energy (ka increase of energy from the ground states) of small molecules. Multiple methods, including
    time-dependent density functional theories (TDDFT) and
    second-order approximate coupled-cluster (CC2)
    - Column 1: Molecule ID (gdb9 index) mapping to the .sdf file
    - Columns 2-5: RI-CC2/def2TZVP
    - Columns 6-9: LR-TDPBE0/def2SVP
    - Columns 10-13: LR-TDPBE0/def2TZVP
    - Columns 14-17: LR-TDCAM-B3LYP/def2TZVP

    """

    __name__ = "qm8"

    __energy_methods__ = [
        PotentialMethod.NONE,  # "wb97x/6-31g(d)"
        PotentialMethod.NONE,
        PotentialMethod.NONE,
        PotentialMethod.NONE,
        PotentialMethod.NONE,
        PotentialMethod.NONE,
        PotentialMethod.NONE,
        PotentialMethod.NONE,
    ]

    __links__ = {
        "qm8.csv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm8.csv",
        "qm8.tar.gz": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb8.tar.gz",
    }

    def read_raw_entries(self):
        df = pd.read_csv(p_join(self.root, "qm8.csv"))
        mols = dm.read_sdf(p_join(self.root, "qm8.sdf"), sanitize=False, remove_hs=False)
        samples = []
        for idx_row, mol in zip(df.iterrows(), mols):
            _, row = idx_row
            positions = mol.GetConformer().GetPositions()
            x = get_atomic_number_and_charge(mol)
            n_atoms = positions.shape[0]
            samples.append(
                dict(
                    atomic_inputs=np.concatenate((x, positions), axis=-1, dtype=np.float32).reshape(-1, 5),
                    name=np.array([row["smiles"]]),
                    energies=np.array(
                        [
                            row[
                                ["E1-CC2", "E2-CC2", "E1-PBE0", "E2-PBE0", "E1-PBE0.1", "E2-PBE0.1", "E1-CAM", "E2-CAM"]
                            ].tolist()
                        ],
                        dtype=np.float64,
                    ).reshape(1, -1),
                    n_atoms=np.array([n_atoms], dtype=np.int32),
                    subset=np.array([f"{self.__name__}"]),
                )
            )
        return samples


class QM9(QMX):
    __links__ = {"qm9.hdf5.gz": "https://zenodo.org/record/3588339/files/155.hdf5.gz?download=1"}
    __name__ = "qm9"
    energy_target_names = [
        "Internal energy at 0 K",
        "B3LYP:def2-svp",
        "HF:cc-pvtz",
        "HF:sto-3g",
        "PBE:sto-3g",
        "SVWN:sto-3g",
        "WB97X-D:aug-cc-pvtz",
        "WB97X-D:def2-svp",
        "WB97X-D:def2-tzvp",
    ]

    __energy_methods__ = [
        PotentialMethod.NONE,  # "wb97x/6-31g(d)"
        PotentialMethod.NONE,
        PotentialMethod.NONE,
        PotentialMethod.NONE,
        PotentialMethod.NONE,
        PotentialMethod.NONE,
        PotentialMethod.NONE,
        PotentialMethod.NONE,
        PotentialMethod.NONE,
    ]
