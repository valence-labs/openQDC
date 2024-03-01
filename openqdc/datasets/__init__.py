from .ani import ANI1, ANI1CCX, ANI1X  # noqa
from .base import BaseDataset  # noqa
from .comp6 import COMP6  # noqa
from .des import DES  # noqa
from .dummy import Dummy  # noqa
from .gdml import GDML  # noqa
from .geom import GEOM  # noqa
from .iso_17 import ISO17  # noqa
from .molecule3d import Molecule3D  # noqa
from .nabladft import NablaDFT  # noqa
from .orbnet_denali import OrbnetDenali  # noqa
from .pcqm import PCQM_B3LYP, PCQM_PM6  # noqa
from .qm7x import QM7X  # noqa
from .qmugs import QMugs  # noqa
from .sn2_rxn import SN2RXN  # noqa
from .solvated_peptides import SolvatedPeptides  # noqa
from .spice import Spice  # noqa
from .tmqm import TMQM  # noqa
from .transition1x import Transition1X  # noqa
from .waterclusters3_30 import WaterClusters  # noqa

AVAILABLE_DATASETS = {
    "ANI1": ANI1,
    "ANI1CCX": ANI1CCX,
    "ANI1X": ANI1X,
    "COMP6": COMP6,
    "DES": DES,
    "GDML": GDML,
    "GEOM": GEOM,
    "ISO17": ISO17,
    "Molecule3D": Molecule3D,
    "NablaDFT": NablaDFT,
    "OrbnetDenali": OrbnetDenali,
    "PCQM_B3LYP": PCQM_B3LYP,
    "PCQM_PM6": PCQM_PM6,
    "QM7X": QM7X,
    "QMugs": QMugs,
    "SN2RXN": SN2RXN,
    "SolvatedPeptides": SolvatedPeptides,
    "Spice": Spice,
    "TMQM": TMQM,
    "Transition1X": Transition1X,
    "WaterClusters": WaterClusters,
}
