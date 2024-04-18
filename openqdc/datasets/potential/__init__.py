from .ani import ANI1, ANI1CCX, ANI1CCX_V2, ANI1X
from .comp6 import COMP6
from .dummy import Dummy
from .gdml import GDML
from .geom import GEOM
from .iso_17 import ISO17
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
    "ani1": ANI1,
    "ani1ccx": ANI1CCX,
    "ani1ccxv2": ANI1CCX_V2,
    "ani1x": ANI1X,
    "comp6": COMP6,
    "gdml": GDML,
    "geom": GEOM,
    "iso17": ISO17,
    "molecule3d": Molecule3D,
    "nabladft": NablaDFT,
    "orbnetdenali": OrbnetDenali,
    "pcqmb3lyp": PCQM_B3LYP,
    "pcqmpm6": PCQM_PM6,
    "qm7x": QM7X,
    "qm7xv2": QM7X_V2,
    "qmugs": QMugs,
    "qmugsv2": QMugs_V2,
    "sn2rxn": SN2RXN,
    "solvatedpeptides": SolvatedPeptides,
    "spice": Spice,
    "spicev2": SpiceV2,
    "spicevl2": SpiceVL2,
    "tmqm": TMQM,
    "transition1x": Transition1X,
    "watercluster": WaterClusters,
    "multixcqm9": MultixcQM9,
    "multixcqm9v2": MultixcQM9_V2,
    "revmd17": RevMD17,
}
