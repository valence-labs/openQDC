from os.path import join as p_join

from openqdc.datasets.base import BaseDataset, read_qc_archive_h5


class COMP6(BaseDataset):
    """
    COMP6 is a benchmark suite consisting of broad regions of bio-chemical and organic space
    developed for testing the ANI-1x potential. It is curated from 6 benchmark sets:
    S66x8, ANI Molecular Dynamics, GDB7to9, GDB10to13, DrugBank, and Tripeptides.

    Usage
    ```python
    from openqdc.datasets import COMP6
    dataset = COMP6()
    ```

    References:
    - https://aip.scitation.org/doi/abs/10.1063/1.5023802
    - Github: https://github.com/isayev/COMP6
    """

    __name__ = "comp6"

    # watchout that forces are stored as -grad(E)
    __energy_unit__ = "hartree"
    __distance_unit__ = "ang"
    __forces_unit__ = "hartree/ang"

    __energy_methods__ = [
        "wb97x/6-31g*",
        "b3lyp-d3m(bj)_tz",
        "b3lyp_tz",
        "hf_tz",
        "pbe-d3(bj)_dz",
        "pbe_tz",
        "svwm_tz",
        "wb97m-d3(bj)_tz",
        "wb97m_tz",
    ]

    energy_target_names = [
        "Energy",
        "B3LYP-D3M(BJ):def2-tzvp",
        "B3LYP:def2-tzvp",
        "HF:def2-tzvp",
        "PBE-D3M(BJ):def2-tzvp",
        "PBE:def2-tzvp",
        "SVWN:def2-tzvp",
        "WB97M-D3(BJ):def2-tzvp",
        "WB97M:def2-tzvp",
    ]

    __force_methods__ = [
        "wb97x/6-31g*",
    ]

    force_target_names = [
        "Gradient",
    ]

    def __init__(self, energy_unit=None, distance_unit=None) -> None:
        super().__init__(energy_unit=energy_unit, distance_unit=distance_unit)

    def read_raw_entries(self):
        samples = []
        for subset in ["ani_md", "drugbank", "gdb7_9", "gdb10_13", "s66x8", "tripeptides"]:
            raw_path = p_join(self.root, f"{subset}.h5")
            samples += read_qc_archive_h5(raw_path, subset, self.energy_target_names, self.force_target_names)

        return samples
