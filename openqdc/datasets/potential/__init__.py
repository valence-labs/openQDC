from .ani import ANI1, ANI1CCX, ANI1X  # noqa
from .comp6 import COMP6  # noqa
from .dummy import Dummy  # noqa
from .gdml import GDML  # noqa
from .geom import GEOM  # noqa
from .iso_17 import ISO17  # noqa
from .molecule3d import Molecule3D  # noqa
from .multixcqm9 import MultixcQM9  # noqa
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
    "ani1": ANI1,
    "ani1ccx": ANI1CCX,
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
    "qmugs": QMugs,
    "sn2rxn": SN2RXN,
    "solvatedpeptides": SolvatedPeptides,
    "spice": Spice,
    "tmqm": TMQM,
    "transition1x": Transition1X,
    "watercluster": WaterClusters,
    "multixcqm9": MultixcQM9,
}
