from .base import BaseDataset  # noqa
from .potential.ani import ANI1, ANI1CCX, ANI1X  # noqa
from .potential.comp6 import COMP6  # noqa
from .potential.dummy import Dummy  # noqa
from .potential.gdml import GDML  # noqa
from .potential.geom import GEOM  # noqa
from .potential.iso_17 import ISO17  # noqa
from .potential.molecule3d import Molecule3D  # noqa
from .potential.nabladft import NablaDFT  # noqa
from .potential.orbnet_denali import OrbnetDenali  # noqa
from .potential.pcqm import PCQM_B3LYP, PCQM_PM6  # noqa
from .potential.qm7x import QM7X  # noqa
from .potential.qmugs import QMugs  # noqa
from .potential.sn2_rxn import SN2RXN  # noqa
from .potential.solvated_peptides import SolvatedPeptides  # noqa
from .potential.spice import Spice  # noqa
from .potential.tmqm import TMQM  # noqa
from .potential.transition1x import Transition1X  # noqa
from .potential.waterclusters3_30 import WaterClusters  # noqa

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
}
