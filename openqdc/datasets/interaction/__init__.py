from .base import BaseInteractionDataset
from .des import DES5M, DES370K, DESS66, DESS66x8
from .l7 import L7
from .metcalf import Metcalf
from .splinter import Splinter
from .x40 import X40

AVAILABLE_INTERACTION_DATASETS = {
    "DES5M": DES5M,
    "DES370K": DES370K,
    "DESS66": DESS66,
    "DESS66x8": DESS66x8,
    "L7": L7,
    "Metcalf": Metcalf,
    "Splinter": Splinter,
    "X40": X40,
}
