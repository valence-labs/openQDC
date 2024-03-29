import os
from typing import Dict, List

import numpy as np
from loguru import logger
from tqdm import tqdm

from openqdc.datasets.interaction.base import BaseInteractionDataset
from openqdc.utils.molecule import atom_table


class Splinter(BaseInteractionDataset):
    """
    A dataset of over 1.7 million protein-ligand
    interactions as described in the paper:

    A quantum chemical interaction energy dataset for accurately modeling protein-ligand interactions.
    Spronk, S.A., Glick, Z.L., Metcalf, D.P. et al.
    Sci Data 10, 619 (2023).
    https://doi.org/10.1038/s41597-023-02443-1
    """

    __name__ = "splinter"
    __energy_methods__ = [
        "sapt0/jun-cc-pV(D+d)Z_unscaled",
        "sapt0/jun-cc-pV(D+d)Z_es_unscaled",
        "sapt0/jun-cc-pV(D+d)Z_ex_unscaled",
        "sapt0/jun-cc-pV(D+d)Z_ind_unscaled",
        "sapt0/jun-cc-pV(D+d)Z_disp_unscaled",
        "sapt0/jun-cc-pV(D+d)Z_scaled",
        "sapt0/jun-cc-pV(D+d)Z_es_scaled",
        "sapt0/jun-cc-pV(D+d)Z_ex_scaled",
        "sapt0/jun-cc-pV(D+d)Z_ind_scaled",
        "sapt0/jun-cc-pV(D+d)Z_disp_scaled",
        "sapt0/aug-cc-pV(D+d)Z_unscaled",
        "sapt0/aug-cc-pV(D+d)Z_es_unscaled",
        "sapt0/aug-cc-pV(D+d)Z_ex_unscaled",
        "sapt0/aug-cc-pV(D+d)Z_ind_unscaled",
        "sapt0/aug-cc-pV(D+d)Z_disp_unscaled",
        "sapt0/aug-cc-pV(D+d)Z_scaled",
        "sapt0/aug-cc-pV(D+d)Z_es_scaled",
        "sapt0/aug-cc-pV(D+d)Z_ex_scaled",
        "sapt0/aug-cc-pV(D+d)Z_ind_scaled",
        "sapt0/aug-cc-pV(D+d)Z_disp_scaled",
    ]

    energy_target_names = []

    def read_raw_entries(self) -> List[Dict]:
        logger.info(f"Reading Splinter interaction data from {self.root}")
        data = []
        i = 0
        with tqdm(total=1680022) as progress_bar:
            for root, dirs, files in os.walk(self.root):  # total is currently an approximation
                for filename in files:
                    if not filename.endswith(".xyz"):
                        continue
                    i += 1
                    filepath = os.path.join(root, filename)
                    filein = open(filepath, "r")
                    lines = list(map(lambda x: x.strip(), filein.readlines()))
                    n_atoms = np.array([int(lines[0])], dtype=np.int32)
                    metadata = lines[1].split(",")
                    try:
                        (
                            protein_monomer_name,
                            protein_interaction_site_type,
                            ligand_monomer_name,
                            ligand_interaction_site_type,
                            index,
                            r,
                            theta_P,
                            tau_P,
                            theta_L,
                            tau_L,
                            tau_PL,
                        ) = metadata[0].split("_")
                        index, r, theta_P, tau_P, theta_L, tau_L, tau_PL = list(
                            map(float, [index, r, theta_P, tau_P, theta_L, tau_L, tau_PL])
                        )
                    except ValueError:
                        (
                            protein_monomer_name,
                            protein_interaction_site_type,
                            ligand_monomer_name,
                            ligand_interaction_site_type,
                            index,
                            _,
                        ) = metadata[0].split("_")
                        r, theta_P, tau_P, theta_L, tau_L, tau_PL = [None] * 6
                    energies = np.array([list(map(float, metadata[4:-1]))]).astype(np.float32)
                    n_atoms_first = np.array([int(metadata[-1])], dtype=np.int32)
                    total_charge, charge0, charge1 = list(map(int, metadata[1:4]))
                    lines = list(map(lambda x: x.split(), lines[2:]))
                    pos = np.array(lines)[:, 1:].astype(np.float32)
                    elems = np.array(lines)[:, 0]
                    atomic_nums = np.expand_dims(np.array([atom_table.GetAtomicNumber(x) for x in elems]), axis=1)
                    natoms0 = n_atoms_first[0]
                    natoms1 = n_atoms[0] - natoms0
                    charges = np.expand_dims(np.array([charge0] * natoms0 + [charge1] * natoms1), axis=1)
                    atomic_inputs = np.concatenate((atomic_nums, charges, pos), axis=-1, dtype=np.float32)
                    subset = np.array([root.split("/")[-1]])

                    item = dict(
                        energies=energies,
                        subset=subset,
                        n_atoms=n_atoms,
                        n_atoms_first=n_atoms_first,
                        atomic_inputs=atomic_inputs,
                        protein_monomer_name=np.array([protein_monomer_name]),
                        protein_interaction_site_type=np.array([protein_interaction_site_type]),
                        ligand_monomer_name=np.array([ligand_monomer_name]),
                        ligand_interaction_site_type=np.array([ligand_interaction_site_type]),
                        index=np.array([index], dtype=np.float32),
                        r=np.array([r], dtype=np.float32),
                        theta_P=np.array([theta_P], dtype=np.float32),
                        tau_P=np.array([tau_P], dtype=np.float32),
                        theta_L=np.array([theta_L], dtype=np.float32),
                        tau_L=np.array([tau_L], dtype=np.float32),
                        tau_PL=np.array([tau_PL], dtype=np.float32),
                        name=np.array([protein_monomer_name + "." + ligand_monomer_name]),
                    )
                    data.append(item)
                    progress_bar.update(1)
        logger.info(f"Processed {i} files in total")
        return data
