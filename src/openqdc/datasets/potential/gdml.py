from os.path import join as p_join

from openqdc.datasets.potential.base import BaseDataset, read_qc_archive_h5


class GDML(BaseDataset):
    """
    Gradient Domain Machine Learning (GDML) is a dataset consisting of samples from ab initio
    molecular dynamics (AIMD) trajectories. The dataset consists of,
    - Benzene: 627000 samples
    - Uracil: 133000 samples
    - Naptalene: 326000 samples
    - Aspirin: 211000 samples
    - Salicylic Acid: 320000 samples
    - Malonaldehyde: 993000 samples
    - Ethanol: 555000 samples
    - Toluene: 100000 samples

    Usage
    ```python
    from openqdc.datasets import GDML
    dataset = GDML()
    ```

    References:
    - https://www.science.org/doi/10.1126/sciadv.1603015
    - http://www.sgdml.org/#datasets
    """

    __name__ = "gdml"

    __energy_methods__ = [
        "ccsd/cc-pvdz",
        "ccsd(t)/cc-pvdz",
        # "pbe/mbd",  # MD22
        # "pbe+mbd/tight", #MD22
        "pbe/vdw-ts",  # MD17
    ]

    energy_target_names = [
        "CCSD Energy",
        "CCSD(T) Energy",
        "PBE-TS Energy",
    ]

    __force_methods__ = [
        "ccsd/cc-pvdz",
        "ccsd(t)/cc-pvdz",
        # "pbe/mbd",  # MD22
        # "pbe+mbd/tight", #MD22
        "pbe/vdw-ts",  # MD17
    ]

    force_target_names = [
        "CCSD Gradient",
        "CCSD(T) Gradient",
        "PBE-TS Gradient",
    ]

    __energy_unit__ = "kcal/mol"
    __distance_unit__ = "bohr"
    __forces_unit__ = "kcal/mol/bohr"

    def read_raw_entries(self):
        raw_path = p_join(self.root, "gdml.h5")
        samples = read_qc_archive_h5(raw_path, "gdml", self.energy_target_names, self.force_target_names)

        return samples
