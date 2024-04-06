from .base import BaseInteractionDataset  # noqa
from .des import DES5M, DES370K, DESS66, DESS66x8
from .l7x40 import L7, X40
from .metcalf import Metcalf
from .splinter import Splinter

AVAILABLE_INTERACTION_DATASETS = {
    "des5m": DES5M,
    "des370k": DES370K,
    "dess66": DESS66,
    "dess66x8": DESS66x8,
    "l7": L7,
    "metcalf": Metcalf,
    "splinter": Splinter,
    "x40": X40,
}
