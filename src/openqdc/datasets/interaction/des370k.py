import torch
import pandas as pd

from typing import Dict, List
from collections import defaultdict, Counter
from torch.utils.data import Dataset

class Dimer:
    def __init__(
        self,
        smiles_0: str,
        smiles_1: str,
        charge_0: int,
        charge_1: int,
        n_atoms_0: int,
        n_atoms_1: int,
        pos: torch.Tensor,
        sapt_energies: List[float],
    ) -> None:
        self.smiles_0 = smiles_0
        self.smiles_1 = smiles_1
        self.charge_1 = charge_0
        self.charge_1 = charge_1
        self.n_atoms_0 = n_atoms_0
        self.n_atoms_1 = n_atoms_1
        self.pos = pos
        self.sapt_energies = sapt_energies
        (
            self.sapt_es,
            self.sapt_ex,
            self.sapt_exs2,
            self.sapt_ind,
            self.sapt_exind,
            self.sapt_disp,
            self.sapt_exdisp_os,
            self.sapt_exdisp_ss,
            self.sapt_delta_HF,
            self.sapt_all
        ) = tuple(sapt_energies)

    def __str__(self) -> str:
        return f"Dimer(smiles_0='{self.smiles_0}', smiles_1='{self.smiles_1}')"

    def __repr__(self) -> str:
        return str(self)


class DES370K(Dataset):
    def __init__(self, filepath="data/des370k.csv") -> None:
        self.df = pd.read_csv(filepath)
        self._atom_types = defaultdict(int)
        self.smiles = set()
        self.data = []
        self._preprocess()
    
    def _preprocess(self) -> None:
        for idx, row in self.df.iterrows():
            smiles0, smiles1 = row["smiles0"], row["smiles1"]
            charge0, charge1 = row["charge0"], row["charge1"]
            natoms0, natoms1 = row["natoms0"], row["natoms1"]
            pos = torch.tensor(list(map(float, row["xyz"].split()))).view(-1, 3)
            sapt_energies = [row[col] for col in self.df.columns if "sapt" in col]
            dimer = Dimer(
                smiles0, smiles1,
                charge0, charge1,
                natoms0, natoms1,
                pos, sapt_energies
            )
            self.data.append(dimer)

            # keep track of unique smiles strings
            self.smiles.add(smiles0)
            self.smiles.add(smiles1)

            # get atom types
            elems = row["elements"].split()
            counts = Counter(set(elems))
            for key in counts:
                self._atom_types[key] += counts[key]

        # convert defaultdict to regular dict
        self._atom_types = dict(self._atom_types)

    def __str__(self) -> str:
        return f"DES370K(n_atoms={self.num_atoms},\
               n_molecules={self.num_molecules},\
               atom_types={self.species})"

    def __repr__(self) -> str:
        return str(self)

    @property
    def atom_types(self) -> Dict[str, int]:
        """
        Returns a dictionary of 
        (element, count) pairs.
        """
        return self._atom_types

    @property
    def num_dimers(self) -> int:
        """
        Returns the number of 
        dimers in the dataset.
        """
        return len(self.data)

    @property
    def num_unique_molecules(self) -> int:
        """
        Returns the number of unique
        molecules in the dataset.
        """
        return len(self.smiles)

    @property
    def num_atoms(self) -> int:
        """
        Returns the total number of atoms in 
        the dataset.
        """
        if not hasattr(self, "_num_atoms"):
            self._num_atoms = sum(self.atom_types.values())
        return self._num_atoms 

    @property
    def species(self) -> List[str]:
        """
        Returns a list of the unique atom
        species contained in the dataset.
        """
        if not hasattr(self, "_species"):
            self._species = list(self.atom_types.keys())
        return self._species

    def atom_count(self, element: str) -> int:
        """
        Returns the count of a given
        element in the dataset.
        """
        return self.atom_types[element]
