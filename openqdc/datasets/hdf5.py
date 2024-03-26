from abc import ABC, abstractmethod
from typing import List, Optional

import datamol as dm
import numpy as np
from ase.atoms import Atoms
from .xyz import FromFileDataset
from openqdc.utils.io import load_hdf5_file, print_h5_tree 



class HDF5Dataset(FromFileDataset):
    def read_as_atoms(self, path):
        data = load_hdf5_file(raw_path)
        data_t = {k2: data[k1][k2][:] for k1 in data.keys() for k2 in data[k1].keys()}

        return atoms 
