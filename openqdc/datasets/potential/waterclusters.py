import zipfile
from io import StringIO
from os.path import join as p_join

import numpy as np
from tqdm import tqdm
from collections import defaultdict
from openqdc.datasets.base import BaseDataset
from openqdc.methods import PotentialMethod
from openqdc.utils.constants import ATOM_TABLE, MAX_ATOMIC_NUMBER
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
        functionals_to_return += [
            f"{dfa}" if dfa == at_dfa else f"{dfa}@{at_dfa}"
            for at_dfa in at_dfa_d
        ]
    
    energies = defaultdict(dict)
    for f in functionals_to_return:
        if "-FLOSIC" in f and "@" not in f:
            func = f.split("-FLOSIC")[0]
            at_f = "-FLOSIC"
        else:
            func = f.split("@")[0]
            at_f = f.split("@")[-1]

        if func not in _energies:
            print(f"No functional {func} included in dataset - available options:\n{', '.join(_energies.keys())}")
        elif at_f not in _energies[func]:
            print(f"No @functional {at_f} included in {func} dataset - available options:\n{', '.join(_energies[func].keys())}")
        else:
            if isinstance(_energies[func][at_f],list):
                for entry in _energies[func][at_f]:
                    if all(
                        entry["metadata"].get(k) == v for k,  v in metadata_restrictions.items()
                    ):
                        energies[f] = entry
                        break
                if f not in energies:
                    print(f"No matching metadata {json.dumps(metadata_restrictions)} for method {f}")
            else:
                energies[f] = _energies[func][at_f]
    return dict(energies)

def format_geometry_and_entries(geometry, energies, subset):
    pass


class SCANWaterClusters(BaseDataset):
    """https://chemrxiv.org/engage/chemrxiv/article-details/662aaff021291e5d1db7d8ec"""

    __name__ = "scanwaterclusters"
    __energy_methods__ = [PotentialMethod.GFN2_XTB]

    __energy_unit__ = "hartree"
    __distance_unit__ = "ang"
    __forces_unit__ = "hartree/ang"

    energy_target_names = ['HF', 'HF-r2SCAN-DC4', 'SCAN', 'SCAN@HF', 'SCAN@r2SCAN50', 'r2SCAN', 'r2SCAN@HF', 'r2SCAN@r2SCAN50', 'r2SCAN50', 'r2SCAN100', 'r2SCAN10', 'r2SCAN20', 'r2SCAN25', 'r2SCAN30', 'r2SCAN40', 'r2SCAN60', 'r2SCAN70', 'r2SCAN80', 'r2SCAN90']
    force_target_names = []

    subsets = ["BEGDB_H2O","WATER27","H2O_alkali_clusters","H2O_halide_clusters"]
    __links__ = {
        "geometries.json.gz" : "https://github.com/esoteric-ephemera/water_cluster_density_errors/blob/main/data_files/geometries.json.gz?raw=True",
        "total_energies.json.gz" : "https://github.com/esoteric-ephemera/water_cluster_density_errors/blob/main/data_files/total_energies.json.gz?raw=True"
    }
    
    def read_raw_entries(self):
        entries=[]
        for i, subset in enumerate(self.subsets):
            
            geometries = read_geometries(p_join(self.root, "geometries.json.gz" ), subset)
            energies = read_energies(p_join(self.root, "total_energies.json.gz" ), subset)
            datum ={}
            for k in energies:
                 _ = energies[k].pop("metadata")  
                 datum[k] = energies[k]["total_energies"]
                
            return pd.concat([pd.DataFrame({"positions" : geometries}),datum], axis=1)
        
        
