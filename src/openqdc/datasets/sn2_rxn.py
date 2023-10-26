from os.path import join as p_join

from openqdc.datasets.base import BaseDataset, read_qc_archive_h5


class SN2RXN(BaseDataset):
    __name__ = "sn2_rxn"

    __energy_methods__ = [
        "dsd-blyp-d3(bj)_tz",
    ]
    __energy_unit__ = "ev"
    __distance_unit__ = "ang"
    __forces_unit__ = "ev/ang"

    energy_target_names = [
        "DSD-BLYP-D3(BJ):def2-TZVP Atomization Energy",
    ]

    __force_methods__ = [
        "dsd-blyp-d3(bj)_tz",
    ]

    force_target_names = [
        "DSD-BLYP-D3(BJ):def2-TZVP Gradient",
    ]

    def __init__(self, energy_unit=None, distance_unit=None) -> None:
        super().__init__(energy_unit=energy_unit, distance_unit=distance_unit)

    def read_raw_entries(self):
        raw_path = p_join(self.root, "sn2_rxn.h5")
        samples = read_qc_archive_h5(raw_path, "sn2_rxn", self.energy_target_names, self.force_target_names)

        return samples
