from os.path import join as p_join
from openqdc.methods import PotentialMethod
from openqdc.datasets.base import BaseDataset, read_qc_archive_h5


class SN2RXN(BaseDataset):
    """
    This dataset probes chemical reactions of methyl halides with halide anions, i.e.
    X- + CH3Y -> CH3X +  Y-, and contains structures for all possible combinations of
    X,Y = F, Cl, Br, I. It contains energy and forces for 452709 conformations calculated
    at the DSD-BLYP-D3(BJ)/def2-TZVP level of theory.

    Usage:
    ```python
    from openqdc.datasets import SN2RXN
    dataset = SN2RXN()
    ```

    References:
    - https://doi.org/10.1021/acs.jctc.9b00181
    - https://zenodo.org/records/2605341
    """

    __name__ = "sn2_rxn"

    __energy_methods__ = [
        PotentialMethod.DSD_BLYP_D3_BJ_DEF2_TZVP
        # "dsd-blyp-d3(bj)/def2-tzvp",
    ]
    __energy_unit__ = "ev"
    __distance_unit__ = "bohr"
    __forces_unit__ = "ev/bohr"

    energy_target_names = [
        "DSD-BLYP-D3(BJ):def2-TZVP Atomization Energy", #TODO: We need to revalidate this to make sure that is not atomization energies.
    ]

    __force_mask__ = [True]

    force_target_names = [
        "DSD-BLYP-D3(BJ):def2-TZVP Gradient",
    ]

    def __smiles_converter__(self, x):
        """util function to convert string to smiles: useful if the smiles is
        encoded in a different format than its display format
        """
        return "-".join(x.decode("ascii").split("_")[:-1])

    def read_raw_entries(self):
        raw_path = p_join(self.root, "sn2_rxn.h5")
        samples = read_qc_archive_h5(raw_path, "sn2_rxn", self.energy_target_names, self.force_target_names)

        return samples
