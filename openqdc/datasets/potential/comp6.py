from os.path import join as p_join

from openqdc.datasets.base import BaseDataset
from openqdc.methods import PotentialMethod
from openqdc.utils import read_qc_archive_h5


class COMP6(BaseDataset):
    """
    COMP6 is a benchmark suite consisting of broad regions of bio-chemical and organic space developed for testing the
    ANI-1x potential. It is curated from 6 benchmark sets: S66x8, ANI-MD, GDB7to9, GDB10to13, DrugBank, and
    Tripeptides. Energies and forces for all non-equilibrium molecular conformations are calculated using
    the wB97x density functional with the 6-31G(d) basis set. The dataset also includes Hirshfield charges and
    molecular dipoles.

    Details of the benchmark sets are as follows:
        S66x8: Consists of 66 dimeric systems involving hydrogen bonding, pi-pi stacking, London interactions and
    mixed influence interactions.\n
        ANI Molecular Dynamics (ANI-MD): Forces from the ANI-1x potential are used for running 1ns vacuum molecular
    dynamics with a 0.25fs time step at 300K using the Langevin thermostat of 14 well-known drug molecules and 2 small
    proteins. A random subsample of 128 frames from each 1ns trajectory is selected, and reference DFT single point
    calculations are performed to calculate energies and forces.\n
        GDB7to9: Consists of 1500 molecules where 500 per 7, 8 and 9 heavy atoms subsampled from the GDB-11 dataset.
    The intial structure are randomly embedded into 3D space using RDKit and are optimized with tight convergence
    criteria. Normal modes/force constants are computer using the reference DFT model. Finally, Diverse normal
    mode sampling (DNMS) is carried out to generate non-equilibrium conformations.\n
        GDB10to13: Consists of 3000 molecules where 500 molecules per 10 and 11 heavy atoms are subsampled from GDB-11
    and 1000 molecules per 12 and 13 heavy atom are subsampled from GDB-13. Non-equilibrium conformations are
    generated via DNMS.\n
        Tripeptide: Consists of 248 random tripeptides. Structures are optimized similar to GDB7to9.\n
        DrugBank: Consists of 837 molecules subsampled from the original DrugBank database of real drug molecules.
    Structures are optimized similar to GDB7to9.

    Usage:
    ```python
    from openqdc.datasets import COMP6
    dataset = COMP6()
    ```

    References:
        https://aip.scitation.org/doi/abs/10.1063/1.5023802\n
        https://github.com/isayev/COMP6\n
        S66x8: https://pubs.rsc.org/en/content/articlehtml/2016/cp/c6cp00688d\n
        GDB-11: https://pubmed.ncbi.nlm.nih.gov/15674983/\n
        GDB-13: https://pubmed.ncbi.nlm.nih.gov/19505099/\n
        DrugBank: https://pubs.acs.org/doi/10.1021/ja902302h
    """

    __name__ = "comp6"

    # watchout that forces are stored as -grad(E)
    __energy_unit__ = "kcal/mol"
    __distance_unit__ = "ang"  # angstorm
    __forces_unit__ = "kcal/mol/ang"

    __energy_methods__ = [
        PotentialMethod.WB97X_6_31G_D,  # "wb97x/6-31g*",
        PotentialMethod.B3LYP_D3_BJ_DEF2_TZVP,  # "b3lyp-d3(bj)/def2-tzvp",
        PotentialMethod.B3LYP_DEF2_TZVP,  # "b3lyp/def2-tzvp",
        PotentialMethod.HF_DEF2_TZVP,  # "hf/def2-tzvp",
        PotentialMethod.PBE_D3_BJ_DEF2_TZVP,  # "pbe-d3(bj)/def2-tzvp",
        PotentialMethod.PBE_DEF2_TZVP,  # "pbe/def2-tzvp",
        PotentialMethod.SVWN_DEF2_TZVP,  # "svwn/def2-tzvp",
    ]

    energy_target_names = [
        "Energy",
        "B3LYP-D3M(BJ):def2-tzvp",
        "B3LYP:def2-tzvp",
        "HF:def2-tzvp",
        "PBE-D3M(BJ):def2-tzvp",
        "PBE:def2-tzvp",
        "SVWN:def2-tzvp",
    ]
    __force_mask__ = [True, False, False, False, False, False, False]

    force_target_names = [
        "Gradient",
    ]

    def __smiles_converter__(self, x):
        """util function to convert string to smiles: useful if the smiles is
        encoded in a different format than its display format
        """
        return "-".join(x.decode("ascii").split("_")[:-1])

    def read_raw_entries(self):
        samples = []
        for subset in ["ani_md", "drugbank", "gdb7_9", "gdb10_13", "s66x8", "tripeptides"]:
            raw_path = p_join(self.root, f"{subset}.h5.gz")
            samples += read_qc_archive_h5(raw_path, subset, self.energy_target_names, self.force_target_names)

        return samples
