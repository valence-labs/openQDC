import ast
import pkgutil
from typing import Dict, Tuple

import numpy as np
from loguru import logger

from openqdc.utils.constants import (
    ATOMIC_NUMBERS,
    MAX_ATOMIC_NUMBER,
    MAX_CHARGE,
    MAX_CHARGE_NUMBER,
)

EF_KEY = Tuple[str, int]

atom_energy_collection = ast.literal_eval(pkgutil.get_data(__name__, "atom_energies.txt").decode("utf-8"))
atom_energy_collection = {k.lower(): v for k, v in atom_energy_collection.items()}


def to_e_matrix(atom_energies: Dict) -> np.ndarray:
    """
    Get the matrix of isolated atom energies for a dict of non-null values calculates

    Parameters:
        atom_energies: Dict of energies computed for a given QM method.
            Keys are pairs of (atom, charge) and values are energy values

    Returns: np.ndarray of shape (MAX_ATOMIC_NUMBER, 2 * MAX_CHARGE + 1)
        Matrix containing the isolated atom energies for each atom and charge written in the form:

                        |   | -2 | -1 | 0 | +1 | +2 | <- charges
                        |---|----|----|---|----|----|
                        | 0 |    |    |   |    |    |
                        | 1 |    |    |   |    |    |
                        | 2 |    |    |   |    |    |
    """

    matrix = np.zeros((MAX_ATOMIC_NUMBER, MAX_CHARGE_NUMBER))
    if len(atom_energies) > 0:
        for key in atom_energies.keys():
            try:
                matrix[ATOMIC_NUMBERS[key[0]], key[1] + MAX_CHARGE] = atom_energies[key]
            except KeyError:
                logger.error(f"Isolated atom energies not found for {key}")
    return matrix
