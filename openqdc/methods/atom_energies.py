import os
import ast
import numpy as np
from loguru import logger
from typing import Dict, Tuple
from openqdc.utils.constants import MAX_ATOMIC_NUMBER, MAX_CHARGE, ATOMIC_NUMBERS


EF_KEY = Tuple[str, int]


with open(os.path.join(os.path.dirname(__file__), "atom_energies.txt")) as fd:
    atom_energy_collection = ast.literal_eval(fd.read())
    atom_energy_collection = {k.lower():v for k, v in atom_energy_collection.items()}


def to_e_matrix(atom_energies: dict) -> np.ndarray:
    """
    Get the matrix of isolated atom energies for a dict of non-null values calculates

    Parameters
    ----------
    atom_energies: dict
        Dict of energies computed for a given QM method. 
        Keys are pairs of (atom, charge) and values are energy values

    Returns
    -------
    np.ndarray of shape (MAX_ATOMIC_NUMBER, 2 * MAX_CHARGE + 1)
        Matrix containing the isolated atom energies for each atom and charge written in the form:

                        |   | -2 | -1 | 0 | +1 | +2 | <- charges
                        |---|----|----|---|----|----|
                        | 0 |    |    |   |    |    |
                        | 1 |    |    |   |    |    |
                        | 2 |    |    |   |    |    |
    """

    matrix = np.zeros((MAX_ATOMIC_NUMBER, MAX_CHARGE * 2 + 1))
    if len(atom_energies) > 0:
        for key in atom_energies.keys():
            try:
                matrix[ATOMIC_NUMBERS[key[0]], key[1] + MAX_CHARGE] = atom_energies[key]
            except KeyError:
                logger.error(f"Isolated atom energies not found for {key}")
    return matrix