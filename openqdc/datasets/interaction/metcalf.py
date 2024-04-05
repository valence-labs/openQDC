import os
from typing import Dict, List

import numpy as np
from loguru import logger

from openqdc.datasets.interaction.base import BaseInteractionDataset
from openqdc.methods import InteractionMethod, InterEnergyType
from openqdc.utils.constants import ATOM_TABLE


def extract_raw_tar_gz(folder):
    # go over all files
    logger.info(f"Extracting all tar.gz files in {folder}")
    expected_tar_files = {
        "train": [
            "TRAINING-2073-ssi-neutral.tar.gz",
            "TRAINING-2610-donors-perturbed.tar.gz",
            "TRAINING-4795-acceptors-perturbed.tar.gz",
        ],
        "val": ["VALIDATION-125-donors.tar.gz", "VALIDATION-254-acceptors.tar.gz"],
        "test": [
            "TEST-Acc--3-methylbutan-2-one_Don--NMe-acetamide-PLDB.tar.gz",
            "TEST-Acc--Cyclohexanone_Don--NMe-acetamide-PLDB.tar.gz",
            "TEST-Acc--Isoquinolone_NMe-acetamide.tar.gz",
            "TEST-Acc--NMe-acetamide_Don--Aniline-CSD.tar.gz",
            "TEST-Acc--NMe-acetamide_Don--Aniline-PLDB.tar.gz",
            "TEST-Acc--NMe-acetamide_Don--N-isopropylacetamide-PLDB.tar.gz",
            "TEST-Acc--NMe-acetamide_Don--N-phenylbenzamide-PLDB.tar.gz",
            "TEST-Acc--NMe-acetamide_Don--Naphthalene-1H-PLDB.tar.gz",
            "TEST-Acc--NMe-acetamide_Don--Uracil-PLDB.tar.gz",
            "TEST-Acc--Tetrahydro-2H-pyran-2-one_NMe-acetamide-PLDB.tar.gz",
            "TEST-NMe-acetamide_Don--Benzimidazole-PLDB.tar.gz",
        ],
    }

    # create a folder with the same name as the tar.gz file
    for subset in expected_tar_files:
        for tar_file in expected_tar_files[subset]:
            logger.info(f"Extracting {tar_file}")
            tar_file_path = os.path.join(folder, tar_file)

            # check if tar file exists
            if not os.path.exists(tar_file_path):
                raise FileNotFoundError(f"File {tar_file_path} not found")

            # skip if extracted folder exists
            if os.path.exists(os.path.join(folder, tar_file.replace(".tar.gz", ""))):
                logger.info(f"Skipping {tar_file}")
                continue

            tar_folder_path = tar_file_path.replace(".tar.gz", "")
            os.mkdir(tar_folder_path)
            os.system(f"tar -xzf {tar_file_path} -C {tar_folder_path}")


class Metcalf(BaseInteractionDataset):
    """
    Hydrogen-bonded dimers of NMA with 126 molecules as described in:

    Approaches for machine learning intermolecular interaction energies and
    application to energy components from symmetry adapted perturbation theory.
    Derek P. Metcalf, Alexios Koutsoukas, Steven A. Spronk, Brian L. Claus,
    Deborah A. Loughney, Stephen R. Johnson, Daniel L. Cheney, C. David Sherrill;
    J. Chem. Phys. 21 February 2020; 152 (7): 074103.
    https://doi.org/10.1063/1.5142636

    Further details:
    "Hydrogen-bonded dimers involving N-methylacetamide (NMA) and 126 molecules
    (46 donors and 80 acceptors; Figs. 2 and 3) were used. Optimized geometries
    for the 126 individual monomers were obtained and paired with NMA in broad
    arrays of spatial configurations to generate thousands of complexes for training.
    """

    __name__ = "metcalf"
    __energy_unit__ = "kcal/mol"
    __distance_unit__ = "ang"
    __forces_unit__ = "kcal/mol/ang"
    __energy_methods__ = [
        InteractionMethod.SAPT0_JUN_CC_PVDZ,
        InteractionMethod.SAPT0_JUN_CC_PVDZ,
        InteractionMethod.SAPT0_JUN_CC_PVDZ,
        InteractionMethod.SAPT0_JUN_CC_PVDZ,
        InteractionMethod.SAPT0_JUN_CC_PVDZ,
    ]
    __energy_type__ = [
        InterEnergyType.TOTAL,
        InterEnergyType.ES,
        InterEnergyType.EX,
        InterEnergyType.IND,
        InterEnergyType.DISP,
    ]
    energy_target_names = [
        "total energy",
        "electrostatic energy",
        "exchange energy",
        "induction energy",
        "dispersion energy",
    ]

    def read_raw_entries(self) -> List[Dict]:
        # extract in folders
        extract_raw_tar_gz(self.root)
        data = []
        for dirname in os.listdir(self.root):
            xyz_dir = os.path.join(self.root, dirname)
            if not os.path.isdir(xyz_dir):
                continue
            subset = np.array([dirname.split("-")[0].lower()])  # training, validation, or test
            for filename in os.listdir(xyz_dir):
                if not filename.endswith(".xyz"):
                    continue
                lines = list(map(lambda x: x.strip(), open(os.path.join(xyz_dir, filename), "r").readlines()))
                line_two = lines[1].split(",")
                energies = np.array([line_two[1:6]], dtype=np.float32)
                num_atoms = np.array([int(lines[0])])

                elem_xyz = np.array([x.split() for x in lines[2:]])
                elements = elem_xyz[:, 0]
                xyz = elem_xyz[:, 1:].astype(np.float32)
                atomic_nums = np.expand_dims(np.array([ATOM_TABLE.GetAtomicNumber(x) for x in elements]), axis=1)
                charges = np.expand_dims(np.array([0] * num_atoms[0]), axis=1)

                atomic_inputs = np.concatenate((atomic_nums, charges, xyz), axis=-1, dtype=np.float32)

                item = dict(
                    n_atoms=num_atoms,
                    subset=subset,
                    energies=energies,
                    positions=xyz,
                    atomic_inputs=atomic_inputs,
                    name=np.array([""]),
                    n_atoms_first=np.array([-1]),
                )
                data.append(item)
        return data
