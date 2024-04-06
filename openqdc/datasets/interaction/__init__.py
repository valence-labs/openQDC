from .base import BaseInteractionDataset  # noqa
from .des import DES5M, DES370K, DESS66, DESS66x8
from .L7 import L7
from .metcalf import Metcalf
from .splinter import Splinter
from .X40 import X40

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
