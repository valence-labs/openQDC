from collections import defaultdict
from os.path import join as p_join

import numpy as np

from openqdc.datasets.base import BaseDataset
from openqdc.methods import PotentialMethod
from openqdc.utils.package_utils import requires_package

_default_basis_sets = {
    "BEGDB_H2O": "aug-cc-pVQZ",
    "WATER27": "aug-cc-pVQZ",
    "H2O_alkali_clusters": "def2-QZVPPD",
    "H2O_halide_clusters": "def2-QZVPPD",
}


@requires_package("monty")
@requires_package("pymatgen")
def read_geometries(fname, dataset):
    from monty.serialization import loadfn

    geometries = {k: v.to_ase_atoms() for k, v in loadfn(fname)[dataset].items()}
    return geometries


@requires_package("monty")
def read_energies(fname, dataset):
    from monty.serialization import loadfn

    # fname
    _energies = loadfn(fname)[dataset]
    metadata_restrictions = {"basis_set": _default_basis_sets.get(dataset)}

    functionals_to_return = []
    for dfa, at_dfa_d in _energies.items():
        functionals_to_return += [f"{dfa}" if dfa == at_dfa else f"{dfa}@{at_dfa}" for at_dfa in at_dfa_d]

    energies = defaultdict(dict)
    for f in functionals_to_return:
        if "-FLOSIC" in f and "@" not in f:
            func = f.split("-FLOSIC")[0]
            at_f = "-FLOSIC"
        else:
            func = f.split("@")[0]
            at_f = f.split("@")[-1]

        if func not in _energies:
            print(f"No functional {func} included in dataset" f"- available options:\n{', '.join(_energies.keys())}")
        elif at_f not in _energies[func]:
            print(
                f"No @functional {at_f} included in {func} dataset"
                f"- available options:\n{', '.join(_energies[func].keys())}"
            )
        else:
            if isinstance(_energies[func][at_f], list):
                for entry in _energies[func][at_f]:
                    if all(entry["metadata"].get(k) == v for k, v in metadata_restrictions.items()):
                        energies[f] = entry
                        break
            else:
                energies[f] = _energies[func][at_f]
    return dict(energies)


def extract_desc(atom):
    # atom_dict=atom.__dict__
    # arrays -> numbers, positions
    # charge, spin_multiplicity
    pos = atom.get_positions()
    z = atom.get_atomic_numbers()
    charges = atom.get_initial_charges()
    formula = atom.get_chemical_formula()
    return pos, z, charges, formula


def format_geometry_and_entries(geometries, energies, subset):
    entries_list = []
    for entry, atoms in geometries.items():
        pos, z, charges, formula = extract_desc(atoms)
        energies_list = []
        for level_of_theory, entry_en_dict in energies.items():
            en = entry_en_dict.get(entry, np.nan)
            energies_list.append(en)
        energy_array = np.array(energies_list)
        if subset in ["WATER27", "H2O_alkali_clusters", "H2O_halide_clusters"]:
            # only the first 9 energies are available
            energy_array.resize(19)
            energy_array[energy_array == 0] = np.nan
        res = dict(
            atomic_inputs=np.concatenate(
                (np.hstack((z[:, None], charges[:, None])), pos), axis=-1, dtype=np.float32
            ).reshape(-1, 5),
            name=np.array([formula]),
            energies=np.array(energy_array, dtype=np.float64).reshape(1, -1),
            n_atoms=np.array([pos.shape[0]], dtype=np.int32),
            subset=np.array([subset]),
        )
        entries_list.append(res)
    return entries_list


class SCANWaterClusters(BaseDataset):
    """
    The SCAN Water Clusters dataset contains conformations of
    neutral water clusters containing up to 20 monomers, charged water clusters,
    and alkali- and halide-water clusters. This dataset consists of our data sets of water clusters:
    the benchmark energy and geometry database (BEGDB) neutral water cluster subset; the WATER2723 set of 14
    neutral, 5 protonated, 7 deprotonated, and one auto-ionized water cluster; and two sets of
    ion-water clusters M...(H2O)n, where M = Li+, Na+, K+, F−, Cl−, or Br−.
    Water clusters were obtained from  10 nanosecond gas-phase molecular dynamics
    simulations using AMBER 9 and optimized to obtain
    lowest energy isomers were determined using MP2/aug-cc-pVDZ//MP2/6-31G* Gibbs free energies.


    Chemical Species:
        [H, O, Li, Na, K, F, Cl, Br]

    Usage:
    ```python
    from openqdc.datasets import SCANWaterClusters
    dataset = SCANWaterClusters()
    ```

    References:
        https://chemrxiv.org/engage/chemrxiv/article-details/662aaff021291e5d1db7d8ec\n
        https://github.com/esoteric-ephemera/water_cluster_density_errors
    """

    __name__ = "scanwaterclusters"

    __energy_unit__ = "hartree"
    __distance_unit__ = "ang"
    __forces_unit__ = "hartree/ang"
    energy_target_names = [
        "HF",
        "HF-r2SCAN-DC4",
        "SCAN",
        "SCAN@HF",
        "SCAN@r2SCAN50",
        "r2SCAN",
        "r2SCAN@HF",
        "r2SCAN@r2SCAN50",
        "r2SCAN50",
        "r2SCAN100",
        "r2SCAN10",
        "r2SCAN20",
        "r2SCAN25",
        "r2SCAN30",
        "r2SCAN40",
        "r2SCAN60",
        "r2SCAN70",
        "r2SCAN80",
        "r2SCAN90",
    ]
    __energy_methods__ = [PotentialMethod.NONE for _ in range(len(energy_target_names))]
    force_target_names = []
    # 27            # 9 level
    subsets = ["BEGDB_H2O", "WATER27", "H2O_alkali_clusters", "H2O_halide_clusters"]
    __links__ = {
        "geometries.json.gz": "https://github.com/esoteric-ephemera/water_cluster_density_errors/blob/main/data_files/geometries.json.gz?raw=True",  # noqa
        "total_energies.json.gz": "https://github.com/esoteric-ephemera/water_cluster_density_errors/blob/main/data_files/total_energies.json.gz?raw=True",  # noqa
    }

    def read_raw_entries(self):
        entries = []  # noqa
        for i, subset in enumerate(self.subsets):
            geometries = read_geometries(p_join(self.root, "geometries.json.gz"), subset)
            energies = read_energies(p_join(self.root, "total_energies.json.gz"), subset)
            datum = {}
            for k in energies:
                _ = energies[k].pop("metadata")
                datum[k] = energies[k]["total_energies"]
            entries.extend(format_geometry_and_entries(geometries, datum, subset))
        return entries
