"""Computations on molecular graphs."""

from typing import Any

import numpy as np
from numpy import ndarray
from rdkit import Chem

from openqdc.utils.constants import ATOM_SYMBOLS

# molecule group classification for DES datasets
molecule_groups = {
    "acids": set(["CCC(=O)O", "CC(=O)O", "OC=O", "OC(=O)CC(=O)O"]),
    "alcohols": set(["CCCO", "CCC(O)C", "CCO", "CC(O)C", "CO", "OC1CCCC1", "OC1CCCCC1", "OCCCCO", "OCCCO", "OCCO"]),
    "alkanes": set(
        [
            "C1CCCC1",
            "C1CCCCC1",
            "C",
            "CC1CCCC1",
            "CC1CCCCC1",
            "CC",
            "CCC",
            "CC(C)C",
            "CCCC",
            "CC(C)(C)C",
            "CCC(C)C",
            "CCCCC",
            "CCC(C)(C)C",
            "CCCCCC",
        ]
    ),
    "alkenes": set(
        [
            "C=C",
            "CC=C",
            "CC=CC",
            "CC(=C)C",
            "CCC=C",
            "CC=C(C)C",
            "CCC=CC",
            "CCC(=C)C",
            "CC(=C(C)C)C",
            "CCC=C(C)C",
            "CCC(=CC)C",
            "CCC(=C(C)C)C",
        ]
    ),
    "amides": set(
        [
            "CCCNC=O",
            "CCC(=O)N",
            "CCC(=O)NC",
            "CCC(=O)N(C)C",
            "CCC(=O)N(CC)C",
            "CCNC=O",
            "CCNC(=O)C",
            "CCN(C=O)CC",
            "CCN(C(=O)C)C",
            "CCNC(=O)CC",
            "CCN(C(=O)C)CC",
            "CC(=O)N",
            "CC(=O)N(C)C",
            "CNC=O",
            "CNC(=O)C",
            "CN(C=O)CC",
            "CNC(=O)CC(=O)N",
            "CNC(=O)CC(=O)NC",
            "CNC(=O)CNC=O",
            "CNC(=O)CNC(=O)C",
            "CNC(=O)C(NC(=O)C)C",
            "NC=O",
            "NC(=O)CC(=O)N",
            "O=CN(C)C",
            "O=CNCCC(=O)N",
            "O=CNCCC(=O)NC",
            "O=CNCCNC=O",
            "O=CNCC(=O)N",
        ]
    ),
    "amines": set(
        [
            "C1CCCN1",
            "C1CCCNC1",
            "CCCN",
            "CCCNC",
            "CCCN(C)C",
            "CCN",
            "CCN(C)C",
            "CCNCC",
            "CCN(CC)C",
            "CN",
            "CNC",
            "CN(C)C",
            "CNCC",
            "CNCCCN",
            "CNCCCNC",
            "CNCCN",
            "CNCCNC",
            "N",
            "NCCCN",
            "NCCN",
        ]
    ),
    "ammoniums": set(["CC[NH3+]", "C[N+](C)(C)C", "C[NH2+]C", "C[NH3+]", "C[NH+](C)C", "[NH4+]"]),
    "benzene": set(["c1ccccc1", "Cc1ccccc1", "CCc1ccccc1"]),
    "carboxylates": set(["[O-]C=O", "[O-]C(=O)C", "[O-]C(=O)CC"]),
    "esters": set(
        [
            "CCCOC=O",
            "CCC(=O)OC",
            "CCOC(=O)CC",
            "CCOC(=O)C",
            "CCOC=O",
            "COC(=O)C",
            "COC=O",
            "O=COCCCOC=O",
            "O=COCCOC=O",
            "O=COCOC=O",
        ]
    ),
    "ethers": set(
        [
            "C1CCCO1",
            "C1CCCOC1",
            "C1CCOCO1",
            "C1OCCO1",
            "CCCOC",
            "CCCOCOC",
            "CCOCC",
            "COCCCOC",
            "COCC",
            "COCCOC",
            "COC",
            "COCOCC",
            "COCOC",
            "O1CCOCC1",
            "O1COCOC1",
        ]
    ),
    "guanidiums": set(["CCNC(=[NH2+])N", "CNC(=[NH2+])N", "NC(=[NH2+])N"]),
    "imidazolium": set(["c1[nH]cc[nH+]1", "Cc1c[nH]c[nH+]1", "CCc1c[nH]c[nH+]1"]),
    "ketones": set(["CCC(=O)CC", "CCC(=O)C", "CCC=O", "CC(=O)C", "CC=O", "C=O"]),
    "monoatomics": set(
        [
            "[Ar]",
            "[Br-]",
            "[Ca+2]",
            "[Cl-]",
            "[F-]",
            "[He]",
            "[I-]",
            "[K+]",
            "[Kr]",
            "[Li+]",
            "[Mg+2]",
            "[Na+]",
            "[Ne]",
            "[Xe]",
        ]
    ),
    "other": set(
        [
            "Brc1ccc(cc1)Br",
            "Brc1ccccc1",
            "BrC(Br)Br",
            "BrCBr",
            "BrCCBr",
            "CBr",
            "CC(Br)Br",
            "CCBr",
            "CCCC#CC",
            "CCCC(Cl)(Cl)Cl",
            "CCCC(Cl)Cl",
            "CCCCCl",
            "CCC#CC",
            "CCCC#C",
            "CCCC(F)(F)F",
            "CCCC(F)F",
            "CCCCF",
            "CCC(Cl)(Cl)Cl",
            "CCC(Cl)Cl",
            "CCCCl",
            "CCCC#N",
            "CC#CC",
            "CCC#C",
            "CCC(F)(F)F",
            "CCC(F)F",
            "CCCF",
            "CC(Cl)(Cl)Cl",
            "CC(Cl)Cl",
            "CCCl",
            "CCC#N",
            "CC#C",
            "CC(F)(F)F",
            "CC(F)F",
            "CCF",
            "CC(I)I",
            "CCI",
            "CCl",
            "CC#N",
            "CCOP(=O)(OC)OC",
            "CCOP(=O)(OC)[O-]",
            "CCOP(=O)(OC)O",
            "C#C",
            "CF",
            "CI",
            "Clc1ccc(cc1)Cl",
            "Clc1cccc(c1)Cl",
            "Clc1ccccc1Cl",
            "Clc1ccccc1",
            "Clc1cc(Cl)c(c(c1Cl)Cl)Cl",
            "Clc1cc(Cl)cc(c1)Cl",
            "Clc1c(Cl)c(Cl)c(c(c1Cl)Cl)Cl",
            "ClC(C(Cl)(Cl)Cl)(Cl)Cl",
            "ClC(C(Cl)(Cl)Cl)Cl",
            "ClCC(Cl)(Cl)Cl",
            "ClCC(Cl)Cl",
            "ClCCCl",
            "ClC(Cl)Cl",
            "ClCCl",
            "CNCCCOC=O",
            "CNCCCOC",
            "CNCCC(=O)NC",
            "CNCCC(=O)N",
            "CNCCC(=O)O",
            "CNCCCO",
            "CNCCCSC",
            "CNCCCS",
            "CNCCNC=O",
            "CNCCOC=O",
            "CNCCOC",
            "CNCC(=O)NC",
            "CNCC(=O)N",
            "CNCC(=O)O",
            "CNCCO",
            "CNCCSC",
            "CNCCS",
            "CNC(=O)CCN",
            "CNC(=O)CC(=O)O",
            "CNC(=O)CCO",
            "CNC(=O)CCS",
            "CNC(=O)CN",
            "CNC(=O)COC=O",
            "CNC(=O)CO",
            "CNCOC=O",
            "CNCOC",
            "CNC(=O)CS",
            "CNCSC",
            "C#N",
            "COCCCN",
            "COCCCOC=O",
            "COCCC(=O)NC",
            "COCCC(=O)N",
            "COCCC(=O)O",
            "COCCCO",
            "COCCCSC",
            "COCCCS",
            "COCCNC=O",
            "COCCN",
            "COCCOC=O",
            "COCC(=O)NC",
            "COCC(=O)N",
            "COCC(=O)O",
            "COCCO",
            "COCCSC",
            "COCCS",
            "COCNC=O",
            "COCN",
            "COCOC=O",
            "COCO",
            "COCSC",
            "COCS",
            "COP(=O)(OC)OC",
            "COP(=O)(OC)[O-]",
            "COP(=O)(OC)O",
            "COP(=O)(O)O",
            "COP(=O)(OP(=O)(O)O)[O-]",
            "CSCCCNC=O",
            "CSCCCN",
            "CSCCCOC=O",
            "CSCCC(=O)N",
            "CSCCC(=O)O",
            "CSCCCO",
            "CSCCN",
            "CSCCOC=O",
            "CSCC(=O)NC",
            "CSCC(=O)N",
            "CSCC(=O)O",
            "CSCCO",
            "CSCNC=O",
            "CSCN",
            "CSCOC=O",
            "CSCO",
            "Fc1ccc(cc1)F",
            "Fc1cccc(c1)F",
            "Fc1ccccc1F",
            "Fc1ccccc1",
            "Fc1cc(F)c(c(c1F)F)F",
            "Fc1cc(F)cc(c1)F",
            "Fc1c(F)c(F)c(c(c1F)F)F",
            "FC(C(F)(F)F)(F)F",
            "FC(C(F)(F)F)F",
            "FCC(F)(F)F",
            "FCC(F)F",
            "FCCF",
            "FC(F)F",
            "FCF",
            "ICCI",
            "ICI",
            "NCCCOC=O",
            "NCCC(=O)N",
            "NCCC(=O)O",
            "NCCCO",
            "NCCCS",
            "NCCNC=O",
            "NCCOC=O",
            "NCC(=O)N",
            "NCC(=O)O",
            "NCCO",
            "NCCS",
            "NC(=O)CC(=O)O",
            "NC(=O)CCO",
            "NC(=O)CCS",
            "NC(=O)CO",
            "NCOC=O",
            "NC(=O)CS",
            "OCCCNC=O",
            "OCCCOC=O",
            "OCCC(=O)O",
            "OCCCS",
            "OCCNC=O",
            "OCCOC=O",
            "OCC(=O)O",
            "OCCS",
            "O=CNCCC(=O)O",
            "O=CNCCOC=O",
            "O=CNCC(=O)O",
            "O=CNCOC=O",
            "O=COCCC(=O)NC",
            "O=COCCC(=O)N",
            "O=COCCC(=O)O",
            "O=COCC(=O)N",
            "O=COCC(=O)O",
            "OC(=O)CCS",
            "OCOC=O",
            "OC(=O)CS",
            "OP(=O)(O)O",
            "[O-]P(=O)(OP(=O)(OC)O)O",
            "SCCCOC=O",
            "SCCNC=O",
            "SCCOC=O",
            "SCOC=O",
            "[H][H]",
        ]
    ),
    "phenol": set(["Cc1ccc(cc1)O", "CCc1ccc(cc1)O", "Oc1ccccc1"]),
    "pyridine": set(["c1cccnc1", "c1ccncn1", "n1ccncc1"]),
    "pyrrole": set(
        [
            "c1ccc2c(c1)[nH]cc2",
            "c1ccc[nH]1",
            "c1ncc[nH]1",
            "Cc1cnc[nH]1",
            "Cc1c[nH]c2c1cccc2",
            "Cc1c[nH]cn1",
            "CCc1cnc[nH]1",
            "CCc1c[nH]c2c1cccc2",
            "CCc1c[nH]cn1",
        ]
    ),
    "sulfides": set(
        [
            "C1CCCS1",
            "C1CCCSC1",
            "C1CCSCS1",
            "C1CCSSC1",
            "C1CSSC1",
            "C1SCCS1",
            "CCCSCSC",
            "CCCSC",
            "CCCSSC",
            "CCSCC",
            "CCSSCC",
            "CCSSC",
            "CSCCCSC",
            "CSCCSC",
            "CSCC",
            "CSCSCC",
            "CSCSC",
            "CSC",
            "CSSC",
            "S1CCSCC1",
            "S1CSCSC1",
        ]
    ),
    "thiols": set(["CCCSS", "CCCS", "CCSS", "CCS", "CSCCCS", "CSCCS", "CSCS", "CSS", "CS", "SCCCS", "SCCS", "SS", "S"]),
    "water": set(["O"]),
    "flourane": set(["F"]),
    "hydrogen chloride": set(["Cl"]),
}


def z_to_formula(z):
    u, c = np.unique(z, return_counts=True)
    idxs = np.argsort(u)
    u, c = u[idxs], c[idxs]

    return "".join([f"{ATOM_SYMBOLS[u[i]]}{c[i] if c[i] > 1 else ''}" for i in range(len(u))])


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
