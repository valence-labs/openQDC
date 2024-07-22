from os.path import join as p_join

from openqdc.datasets.base import BaseDataset
from openqdc.methods import PotentialMethod
from openqdc.utils import read_qc_archive_h5


class SolvatedPeptides(BaseDataset):
    """
    The solvated protein fragments dataset probes many-body intermolecular interactions between "protein fragments"
    and water molecules. Geometries are first optimized with the semi-empirical method PM7 and then MD simulations are
    run at 1000K with a time-step of 0.1fs using Atomic Simulations Environment (ASE). Structures are saved every 10
    steps, where energies, forces and dipole moments are calculated at revPBE-D3(BJ)/def2-TZVP level of theory.

    Usage:
    ```python
    from openqdc.datasets import SolvatedPeptides
    dataset = SolvatedPeptides()
    ```

    References:
        https://doi.org/10.1021/acs.jctc.9b00181\n
        https://zenodo.org/records/2605372
    """

    __name__ = "solvated_peptides"

    __energy_methods__ = [
        PotentialMethod.REVPBE_D3_BJ_DEF2_TZVP
        # "revpbe-d3(bj)/def2-tzvp",
    ]

    energy_target_names = [
        "revPBE-D3(BJ):def2-TZVP Atomization Energy",
    ]

    __force_mask__ = [True]

    force_target_names = [
        "revPBE-D3(BJ):def2-TZVP Gradient",
    ]

    # TO CHECK
    __energy_unit__ = "ev"
    __distance_unit__ = "ang"
    __forces_unit__ = "ev/ang"
    __links__ = {"solvated_peptides.hdf5.gz": "https://zenodo.org/record/3585804/files/213.hdf5.gz"}

    def __smiles_converter__(self, x):
        """util function to convert string to smiles: useful if the smiles is
        encoded in a different format than its display format
        """
        return "_".join(x.decode("ascii").split("_")[:-1])

    def read_raw_entries(self):
        raw_path = p_join(self.root, "solvated_peptides.h5.gz")
        samples = read_qc_archive_h5(raw_path, "solvated_peptides", self.energy_target_names, self.force_target_names)

        return samples
