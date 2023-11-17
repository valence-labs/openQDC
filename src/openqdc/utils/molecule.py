from typing import Any

import numpy as np
from numpy import ndarray
from rdkit import Chem

atom_table = Chem.GetPeriodicTable()


def get_atomic_number(mol: Chem.Mol):
    """Returns atomic numbers for rdkit molecule"""
    return np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])


def get_atomic_charge(mol: Chem.Mol):
    """Returns atom charges for rdkit molecule"""
    return np.array([atom.GetFormalCharge() for atom in mol.GetAtoms()])


def get_atomic_number_and_charge(mol: Chem.Mol):
    """Returns atoms number and charge for rdkit molecule"""
    return np.array([[atom.GetAtomicNum(), atom.GetFormalCharge()] for atom in mol.GetAtoms()])


def rmsd(P: ndarray, Q: ndarray, **kwargs) -> float:
    """
    Calculate Root-mean-square deviation from two sets of vectors V and W.

    Parameters
    ----------
    V : array
        (N,D) matrix, where N is points and D is dimension.
    W : array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    rmsd : float
        Root-mean-square deviation between the two vectors
    """
    diff = P - Q
    return np.sqrt((diff * diff).sum() / P.shape[0])


def kabsch_rmsd(
    P: ndarray,
    Q: ndarray,
    translate: bool = False,
    **kwargs: Any,
) -> float:
    """
    Rotate matrix P unto Q using Kabsch algorithm and calculate the RMSD.

    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    translate : bool
        Use centroids to translate vector P and Q unto each other.

    Returns
    -------
    rmsd : float
        root-mean squared deviation
    """

    if translate:
        Q = Q - Q.mean(axis=0)
        P = P - P.mean(axis=0)

    # Computation of the covariance matrix
    C = np.dot(np.transpose(P), Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = np.dot(V, W)

    # Rotate P
    P_prime = np.dot(P, U)
    return rmsd(P_prime, Q)
