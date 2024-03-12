from .base import BaseDataset  # noqa
from .interaction import AVAILABLE_INTERACTION_DATASETS  # noqa
from .interaction import DES  # noqa
from .potential import AVAILABLE_POTENTIAL_DATASETS  # noqa
from .potential.ani import ANI1, ANI1CCX, ANI1X  # noqa
from .potential.comp6 import COMP6  # noqa
from .potential.dummy import Dummy  # noqa
from .potential.gdml import GDML  # noqa
from .potential.geom import GEOM  # noqa
from .potential.iso_17 import ISO17  # noqa
from .potential.molecule3d import Molecule3D  # noqa
from .potential.multixcqm9 import MultixcQM9  # noqa
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

AVAILABLE_DATASETS = {**AVAILABLE_POTENTIAL_DATASETS, **AVAILABLE_INTERACTION_DATASETS}
