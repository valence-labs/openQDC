from typing import Final, List

NB_ATOMIC_FEATURES: Final[int] = 5

MAX_ATOMIC_NUMBER: Final[int] = 119

HAR2EV: Final[float] = 27.211386246

BOHR2ANG: Final[float] = 0.52917721092

POSSIBLE_NORMALIZATION: Final[List[str]] = ["formation", "total", "inter"]

NOT_DEFINED = {
    "mean": None,
    "std": None,
    "components": {
        "mean": None,
        "std": None,
        "rms": None,
    },
}
