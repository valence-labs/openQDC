import re
from functools import partial
from os.path import join as p_join

import datamol as dm
import numpy as np

from openqdc.datasets.base import BaseDataset
from openqdc.methods import PotentialMethod
from openqdc.utils.constants import ATOMIC_NUMBERS
from openqdc.utils.molecule import get_atomic_number_and_charge


def parse_mace_xyz(xyzpath):
    energy_re = re.compile(r"energy=(\S+)")
    smiles_re = re.compile(r"smiles=(\S+)")
    subset_re = re.compile(r"config_type=(\S+)")
    with open(xyzpath, "r") as f:
        n_atoms = None
        counter = 0
        positions = []
        numbers = []
        forces = []
        energy = None
        for line in f:
            if n_atoms is None:
                n_atoms = int(line)
                positions = []
                numbers = []
                forces = []
                energy = None
                counter = 1
                continue
            if counter == 1:
                props = line
                energy = float(energy_re.search(props).group(1))
                subset = subset_re.search(props).group(1)
                try:
                    smiles = smiles_re.search(props).group(1)
                except AttributeError:  # water and qmugs subsets do not have smiles
                    smiles = ""
                counter = 2
                continue
            el, x, y, z, fx, fy, fz, _, _, _ = line.split()
            numbers.append(ATOMIC_NUMBERS[el])
            positions.append([float(x), float(y), float(z)])
            forces.append([float(fx), float(fy), float(fz)])
            smiles = smiles.replace('"', "")
            subset = subset.replace('"', "")
            counter += 1
            if counter == n_atoms + 2:
                n_atoms = None
                yield energy, numbers, positions, forces, smiles, subset


def build_data_object(data, split):
    energy, numbers, positions, forces, smiles, subset = data
    if smiles == "":
        x = np.concatenate((np.array(numbers)[:, None], np.zeros((len(numbers), 1))), axis=-1)
    else:
        x = get_atomic_number_and_charge(dm.to_mol(smiles, remove_hs=False, ordered=True))
    res = dict(
        name=np.array([smiles]),
        subset=np.array([subset]),
        energies=np.array([[energy]], dtype=np.float64),
        forces=np.array(forces, dtype=np.float32).reshape(
            -1, 3, 1
        ),  # forces -ve of energy gradient but the -1.0 is done in the convert_forces method
        atomic_inputs=np.concatenate((x, np.array(positions)), axis=-1, dtype=np.float32).reshape(-1, 5),
        n_atoms=np.array([x.shape[0]], dtype=np.int32),
        split=np.array([split]),
    )
    return res


class MACEOFF(BaseDataset):
    __name__ = "maceoff"

    __energy_methods__ = [PotentialMethod.WB97M_D3BJ_DEF2_TZVPPD]
    __force_mask__ = [True]
    __energy_unit__ = "ev"
    __distance_unit__ = "ang"
    __forces_unit__ = "ev/ang"

    energy_target_names = ["dft_total_energy"]
    force_target_names = ["dft_total_gradient"]

    __links__ = {
        "train_large_neut_no_bad_clean.tar.gz": "https://api.repository.cam.ac.uk/server/api/core/bitstreams/b185b5ab-91cf-489a-9302-63bfac42824a/content",  # noqa: E501
        "test_large_neut_all.tar.gz": "https://api.repository.cam.ac.uk/server/api/core/bitstreams/cb8351dd-f09c-413f-921c-67a702a7f0c5/content",  # noqa: E501
    }

    def read_raw_entries(self):
        entries = []
        for filename in self.__links__:
            filename = filename.split(".")[0]
            xyzpath = p_join(self.root, f"{filename}.xyz")
            split = filename.split("_")[0]
            structure_iterator = parse_mace_xyz(xyzpath)
            func = partial(build_data_object, split=split)
            entries.extend(dm.utils.parallelized(func, structure_iterator))
        return entries

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data.__setattr__("split", self._convert_array(self.data["split"][idx]))
        return data
