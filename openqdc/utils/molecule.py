import numpy as np
from rdkit import Chem

atom_table = Chem.GetPeriodicTable()

def get_atomic_number(mol: Chem.Mol):
    """Returns atomic numbers for rdkit molecule"""
    return np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])


def get_atomic_charge(mol: Chem.Mol):
    """Returns atom charges for rdkit molecule"""
    return np.array([atom.GetFormalCharge() for atom in mol.GetAtoms()])


def get_atomic_numuber_and_charge(mol: Chem.Mol):
    """Returns atoms number and charge for rdkit molecule"""
    return np.array([[atom.GetAtomicNum(), atom.GetFormalCharge()] 
                     for atom in mol.GetAtoms()])

