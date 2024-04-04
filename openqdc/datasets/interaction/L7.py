import os
from typing import Dict, List

import numpy as np
import yaml
from loguru import logger

from openqdc.datasets.interaction.base import BaseInteractionDataset
from openqdc.methods import InteractionMethod, InterEnergyType
from openqdc.utils.constants import ATOM_TABLE


class DataItemYAMLObj:
    def __init__(self, name, shortname, geometry, reference_value, setup, group, tags):
        self.name = name
        self.shortname = shortname
        self.geometry = geometry
        self.reference_value = reference_value
        self.setup = setup
        self.group = group
        self.tags = tags


class DataSetYAMLObj:
    def __init__(self, name, references, text, method_energy, groups_by, groups, global_setup, method_geometry=None):
        self.name = name
        self.references = references
        self.text = text
        self.method_energy = method_energy
        self.method_geometry = method_geometry
        self.groups_by = groups_by
        self.groups = groups
        self.global_setup = global_setup


def data_item_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode):
    return DataItemYAMLObj(**loader.construct_mapping(node))


def dataset_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode):
    return DataSetYAMLObj(**loader.construct_mapping(node))


def get_loader():
    """Add constructors to PyYAML loader."""
    loader = yaml.SafeLoader
    loader.add_constructor("!ruby/object:ProtocolDataset::DataSetItem", data_item_constructor)
    loader.add_constructor("!ruby/object:ProtocolDataset::DataSetDescription", dataset_constructor)
    return loader


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

    __name__ = "L7"
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

    def read_raw_entries(self) -> List[Dict]:
        yaml_fpath = os.path.join(self.root, "l7.yaml")
        logger.info(f"Reading L7 interaction data from {self.root}")
        yaml_file = open(yaml_fpath, "r")
        data = []
        data_dict = yaml.load(yaml_file, Loader=get_loader())
        charge0 = int(data_dict["description"].global_setup["molecule_a"]["charge"])
        charge1 = int(data_dict["description"].global_setup["molecule_b"]["charge"])

        for idx, item in enumerate(data_dict["items"]):
            energies = []
            name = np.array([item.shortname])
            fname = item.geometry.split(":")[1]
            energies.append(item.reference_value)
            xyz_file = open(os.path.join(self.root, f"{fname}.xyz"), "r")
            lines = list(map(lambda x: x.strip().split(), xyz_file.readlines()))
            lines.pop(1)
            n_atoms = np.array([int(lines[0][0])], dtype=np.int32)
            n_atoms_first = np.array([int(item.setup["molecule_a"]["selection"].split("-")[1])], dtype=np.int32)
            subset = np.array([item.group])
            energies += [float(val[idx]) for val in list(data_dict["alternative_reference"].values())]
            energies = np.array([energies], dtype=np.float32)
            pos = np.array(lines[1:])[:, 1:].astype(np.float32)
            elems = np.array(lines[1:])[:, 0]
            atomic_nums = np.expand_dims(np.array([ATOM_TABLE.GetAtomicNumber(x) for x in elems]), axis=1)
            natoms0 = n_atoms_first[0]
            natoms1 = n_atoms[0] - natoms0
            charges = np.expand_dims(np.array([charge0] * natoms0 + [charge1] * natoms1), axis=1)
            atomic_inputs = np.concatenate((atomic_nums, charges, pos), axis=-1, dtype=np.float32)

            item = dict(
                energies=energies,
                subset=subset,
                n_atoms=n_atoms,
                n_atoms_first=n_atoms_first,
                atomic_inputs=atomic_inputs,
                name=name,
            )
            data.append(item)
        return data
