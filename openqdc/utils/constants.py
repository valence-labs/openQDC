import numpy as np
from rdkit import Chem
from typing import Final, List

MAX_CHARGE: Final[int] = 6

NB_ATOMIC_FEATURES: Final[int] = 5

MAX_ATOMIC_NUMBER: Final[int] = 119

HAR2EV: Final[float] = 27.211386246

BOHR2ANG: Final[float] = 0.52917721092

POSSIBLE_NORMALIZATION: Final[List[str]] = [
    "formation",
    "total",
    "per_atom_formation",
    "residual_regression",
    "per_atom_residual_regression",
]

NOT_DEFINED = {
    "mean": None,
    "std": None,
    "components": {
        "mean": None,
        "std": None,
        "rms": None,
    },
}

ATOM_TABLE = Chem.GetPeriodicTable()
ATOM_SYMBOLS = np.array(["X"] + [ATOM_TABLE.GetElementSymbol(z) for z in range(1, 118)])
ATOMIC_NUMBERS = {symbol: Z for Z, symbol in enumerate(ATOM_SYMBOLS)}

