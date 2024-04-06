import os
from dataclasses import dataclass
from functools import partial
from os.path import join as p_join
from typing import Dict, List, Optional

import numpy as np
import yaml
from loguru import logger

from openqdc.datasets.interaction.base import BaseInteractionDataset
from openqdc.methods import InteractionMethod, InterEnergyType
from openqdc.utils.constants import ATOM_TABLE


@dataclass
class DataSet:
    description: Dict
    items: List[Dict]
    alternative_reference: Dict


@dataclass
class DataItemYAMLObj:
    name: str
    shortname: str
    geometry: str
    reference_value: float
    setup: Dict
    group: str
    tags: str


@dataclass
class DataSetDescription:
    name: Dict
    references: str
    text: str
    groups_by: str
    groups: List[str]
    global_setup: Dict
    method_energy: str
    method_geometry: Optional[str] = None


def get_loader():
    """Add constructors to PyYAML loader."""

    def constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode, cls):
        return cls(**loader.construct_mapping(node))

    loader = yaml.SafeLoader

    loader.add_constructor("!ruby/object:ProtocolDataset::DataSet", partial(constructor, cls=DataSet))
    loader.add_constructor("!ruby/object:ProtocolDataset::DataSetItem", partial(constructor, cls=DataItemYAMLObj))
    loader.add_constructor(
        "!ruby/object:ProtocolDataset::DataSetDescription", partial(constructor, cls=DataSetDescription)
    )
    return loader


def read_xyz_file(xyz_path):
    with open(xyz_path, "r") as xyz_file:  # avoid not closing the file
        lines = list(map(lambda x: x.strip().split(), xyz_file.readlines()))
        lines.pop(1)
        n_atoms = np.array([int(lines[0][0])], dtype=np.int32)
        pos = np.array(lines[1:])[:, 1:].astype(np.float32)
        elems = np.array(lines[1:])[:, 0]
        atomic_nums = np.expand_dims(np.array([ATOM_TABLE.GetAtomicNumber(x) for x in elems]), axis=1)
    return n_atoms, pos, atomic_nums


def convert_to_record(item):
    return dict(
        energies=item["energies"],
        subset=np.array([item["subset"]]),
        n_atoms=np.array([item["natoms0"] + item["natoms1"]], dtype=np.int32),
        n_atoms_first=np.array([item["natoms0"]], dtype=np.int32),
        atomic_inputs=item["atomic_inputs"],
        name=item["name"],
    )


def build_item(item, charge0, charge1, idx, data_dict, root, filename):
    datum = {
        "energies": [],
    }
    datum["name"] = np.array([item.shortname])
    datum["energies"].append(item.reference_value)
    datum["subset"] = np.array([item.group])
    datum["energies"] += [float(val[idx]) for val in list(data_dict.alternative_reference.values())]
    datum["energies"] = np.array([datum["energies"]], dtype=np.float32)
    n_atoms, pos, atomic_nums = read_xyz_file(p_join(root, f"{filename}.xyz"))
    datum["n_atoms"] = n_atoms
    datum["pos"] = pos
    datum["atomic_nums"] = atomic_nums
    datum["n_atoms_first"] = np.array([int(item.setup["molecule_a"]["selection"].split("-")[1])], dtype=np.int32)
    datum["natoms0"] = datum["n_atoms_first"][0]
    datum["natoms1"] = datum["n_atoms"][0] - datum["natoms0"]
    datum["charges"] = np.expand_dims(np.array([charge0] * datum["natoms0"] + [charge1] * datum["natoms1"]), axis=1)
    datum["atomic_inputs"] = np.concatenate(
        (datum["atomic_nums"], datum["charges"], datum["pos"]), axis=-1, dtype=np.float32
    )
    return datum


class L7(BaseInteractionDataset):
    """
    The L7 interaction energy dataset as described in:

    Accuracy of Quantum Chemical Methods for Large Noncovalent Complexes
    Robert Sedlak, Tomasz Janowski, Michal Pitoňák, Jan Řezáč, Peter Pulay, and Pavel Hobza
    Journal of Chemical Theory and Computation 2013 9 (8), 3364-3374
    DOI: 10.1021/ct400036b

    Data was downloaded and extracted from:
    http://cuby4.molecular.cz/dataset_l7.html
    """

    __name__ = "l7"
    __energy_unit__ = "kcal/mol"
    __distance_unit__ = "ang"
    __forces_unit__ = "kcal/mol/ang"
    __energy_methods__ = [
        InteractionMethod.QCISDT_CBS,  # "QCISD(T)/CBS",
        InteractionMethod.DLPNO_CCSDT,  # "DLPNO-CCSD(T)",
        InteractionMethod.MP2_CBS,  # "MP2/CBS",
        InteractionMethod.MP2C_CBS,  # "MP2C/CBS",
        InteractionMethod.FIXED,  # "fixed", TODO: we should remove this level of theory because unless we have a pro
        InteractionMethod.DLPNO_CCSDT0,  # "DLPNO-CCSD(T0)",
        InteractionMethod.LNO_CCSDT,  # "LNO-CCSD(T)",
        InteractionMethod.FN_DMC,  # "FN-DMC",
    ]

    __energy_type__ = [InterEnergyType.TOTAL] * 8

    energy_target_names = []

    @property
    def yaml_path(self):
        return os.path.join(self.root, self.__name__ + ".yaml")

    def read_raw_entries(self) -> List[Dict]:
        yaml_fpath = self.yaml_path
        logger.info(f"Reading {self.__name__} interaction data from {self.root}")
        with open(yaml_fpath, "r") as yaml_file:
            data_dict = yaml.load(yaml_file, Loader=get_loader())
        data = []
        charge0 = int(data_dict.description.global_setup["molecule_a"]["charge"])
        charge1 = int(data_dict.description.global_setup["molecule_b"]["charge"])

        for idx, item in enumerate(data_dict.items):
            tmp_item = build_item(item, charge0, charge1, idx, data_dict, self.root, self._process_name(item))
            item = convert_to_record(tmp_item)
            data.append(item)
        return data

    def _process_name(self, item):
        return item.geometry.split(":")[1]
