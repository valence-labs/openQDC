from .base import BaseInteractionDataset
from .des5m import DES5M
from .des370k import DES370K
from .dess66 import DESS66
from .dess66x8 import DESS66x8
from .L7 import L7
from .metcalf import Metcalf
from .splinter import Splinter
from .X40 import X40

AVAILABLE_INTERACTION_DATASETS = {
    "base": BaseInteractionDataset,
    "des5m": DES5M,
    "des370k": DES370K,
    "dess66": DESS66,
    "dess66x8": DESS66x8,
    "l7": L7,
    "metcalf": Metcalf,
    "splinter": Splinter,
    "x40": X40,
}
