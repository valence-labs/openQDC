from .ani import ANI1, ANI1CCX, ANI1CCX_V2, ANI1X, ANI2X
from .comp6 import COMP6
from .dummy import Dummy
from .gdml import GDML
from .geom import GEOM
from .iso_17 import ISO17
from .md22 import MD22
from .molecule3d import Molecule3D
from .multixcqm9 import MultixcQM9, MultixcQM9_V2
from .nabladft import NablaDFT
from .orbnet_denali import OrbnetDenali
from .pcqm import PCQM_B3LYP, PCQM_PM6
from .qm7x import QM7X, QM7X_V2
from .qmugs import QMugs, QMugs_V2
from .revmd17 import RevMD17
from .sn2_rxn import SN2RXN
from .solvated_peptides import SolvatedPeptides
from .spice import Spice, SpiceV2, SpiceVL2
from .tmqm import TMQM
from .transition1x import Transition1X
from .waterclusters3_30 import WaterClusters

AVAILABLE_POTENTIAL_DATASETS = {
    "ANI1": ANI1,
    "ANI1CCX": ANI1CCX,
    "ANI1CCX_V2": ANI1CCX_V2,
    "ANI1X": ANI1X,
    "ANI2X": ANI2X,
    "COMP6": COMP6,
    "GDML": GDML,
    "GEOM": GEOM,
    "ISO17": ISO17,
    "Molecule3D": Molecule3D,
    "NablaDFT": NablaDFT,
    "OrbnetDenali": OrbnetDenali,
    "PCQM_B3LYP": PCQM_B3LYP,
    "PCQM_PM6": PCQM_PM6,
    "QM7X": QM7X,
    "QM7X_V2": QM7X_V2,
    "QMugs": QMugs,
    "QMugs_V2": QMugs_V2,
    "SN2RXN": SN2RXN,
    "SolvatedPeptides": SolvatedPeptides,
    "Spice": Spice,
    "SpiceV2": SpiceV2,
    "SpiceVL2": SpiceVL2,
    "TMQM": TMQM,
    "Transition1X": Transition1X,
    "WaterClusters": WaterClusters,
    "MultixcQM9": MultixcQM9,
    "MultixcQM9_V2": MultixcQM9_V2,
    "RevMD17": RevMD17,
    "MD22": MD22,
}
