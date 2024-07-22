from os.path import join as p_join

from openqdc.datasets.base import BaseDataset
from openqdc.methods import PotentialMethod
from openqdc.utils import read_qc_archive_h5


class GDML(BaseDataset):
    """
    Gradient Domain Machine Learning (GDML) is a dataset consisting of samples from ab initio
    molecular dynamics (AIMD) trajectories at a resolution of 0.5fs. The dataset consists of, Benzene
    (627000 conformations), Uracil (133000 conformations), Naptalene (326000 conformations), Aspirin
    (211000 conformations) Salicylic Acid (320000 conformations), Malonaldehyde (993000 conformations),
    Ethanol (555000 conformations) and Toluene (100000 conformations). Energy and force labels for
    each conformation are computed using the PBE + vdW-TS electronic structure method.
    molecular dynamics (AIMD) trajectories.

    The dataset consists of the following trajectories:
        Benzene: 627000 samples\n
        Uracil: 133000 samples\n
        Naptalene: 326000 samples\n
        Aspirin: 211000 samples\n
        Salicylic Acid: 320000 samples\n
        Malonaldehyde: 993000 samples\n
        Ethanol: 555000 samples\n
        Toluene: 100000 samples\n

    Usage:
    ```python
    from openqdc.datasets import GDML
    dataset = GDML()
    ```

    References:
        https://www.science.org/doi/10.1126/sciadv.1603015
        http://www.sgdml.org/#datasets
    """

    __name__ = "gdml"

    __energy_methods__ = [
        PotentialMethod.CCSD_CC_PVDZ,  # "ccsd/cc-pvdz",
        PotentialMethod.CCSD_T_CC_PVDZ,  # "ccsd(t)/cc-pvdz",
        # TODO: verify if basis set vdw-ts == def2-tzvp and
        # it is the same in ISO17 and revmd17
        PotentialMethod.PBE_DEF2_TZVP,  # "pbe/def2-tzvp",  # MD17
    ]

    energy_target_names = [
        "CCSD Energy",
        "CCSD(T) Energy",
        "PBE-TS Energy",
    ]

    __force_mask__ = [True, True, True]

    force_target_names = [
        "CCSD Gradient",
        "CCSD(T) Gradient",
        "PBE-TS Gradient",
    ]

    __energy_unit__ = "kcal/mol"
    __distance_unit__ = "ang"
    __forces_unit__ = "kcal/mol/ang"
    __links__ = {
        "gdb7_9.hdf5.gz": "https://zenodo.org/record/3588361/files/208.hdf5.gz",
        "gdb10_13.hdf5.gz": "https://zenodo.org/record/3588364/files/209.hdf5.gz",
        "drugbank.hdf5.gz": "https://zenodo.org/record/3588361/files/207.hdf5.gz",
        "tripeptides.hdf5.gz": "https://zenodo.org/record/3588368/files/211.hdf5.gz",
        "ani_md.hdf5.gz": "https://zenodo.org/record/3588341/files/205.hdf5.gz",
        "s66x8.hdf5.gz": "https://zenodo.org/record/3588367/files/210.hdf5.gz",
    }

    def read_raw_entries(self):
        raw_path = p_join(self.root, "gdml.h5.gz")
        samples = read_qc_archive_h5(raw_path, "gdml", self.energy_target_names, self.force_target_names)

        return samples
