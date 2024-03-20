import importlib
import os
from typing import TYPE_CHECKING  # noqa F401

# The below lazy import logic is coming from openff-toolkit:
# https://github.com/openforcefield/openff-toolkit/blob/b52879569a0344878c40248ceb3bd0f90348076a/openff/toolkit/__init__.py#L44

# Dictionary of objects to lazily import; maps the object's name to its module path

_lazy_imports_obj = {
    "__version__": "openqdc._version",
    "BaseDataset": "openqdc.datasets.base",
    "ANI1": "openqdc.datasets.potential.ani",
    "ANI1CCX": "openqdc.datasets.potential.ani",
    "ANI1X": "openqdc.datasets.potential.ani",
    "Spice": "openqdc.datasets.potential.spice",
    "SpiceV2": "openqdc.datasets.potential.spice",
    "GEOM": "openqdc.datasets.potential.geom",
    "QMugs": "openqdc.datasets.potential.qmugs",
    "ISO17": "openqdc.datasets.potential.iso_17",
    "COMP6": "openqdc.datasets.potential.comp6",
    "GDML": "openqdc.datasets.potential.gdml",
    "Molecule3D": "openqdc.datasets.potential.molecule3d",
    "OrbnetDenali": "openqdc.datasets.potential.orbnet_denali",
    "SN2RXN": "openqdc.datasets.potential.sn2_rxn",
    "QM7X": "openqdc.datasets.potential.qm7x",
    "NablaDFT": "openqdc.datasets.potential.nabladft",
    "SolvatedPeptides": "openqdc.datasets.potential.solvated_peptides",
    "WaterClusters": "openqdc.datasets.potential.waterclusters3_30",
    "TMQM": "openqdc.datasets.potential.tmqm",
    "Dummy": "openqdc.datasets.potential.dummy",
    "PCQM_B3LYP": "openqdc.datasets.potential.pcqm",
    "PCQM_PM6": "openqdc.datasets.potential.pcqm",
    "RevMD17": "openqdc.datasets.potential.revmd17",
    "Transition1X": "openqdc.datasets.potential.transition1x",
    "MultixcQM9": "openqdc.datasets.potential.multixcqm9",
    "AVAILABLE_DATASETS": "openqdc.datasets",
    "AVAILABLE_POTENTIAL_DATASETS": "openqdc.datasets.potential",
    "AVAILABLE_INTERACTION_DATASETS": "openqdc.datasets.interaction",
}

_lazy_imports_mod = {"datasets": "openqdc.datasets", "utils": "openqdc.utils"}


def __getattr__(name):
    """Lazily import objects from _lazy_imports_obj or _lazy_imports_mod

    Note that this method is only called by Python if the name cannot be found
    in the current module."""
    obj_mod = _lazy_imports_obj.get(name)
    if obj_mod is not None:
        mod = importlib.import_module(obj_mod)
        return mod.__dict__[name]

    lazy_mod = _lazy_imports_mod.get(name)
    if lazy_mod is not None:
        return importlib.import_module(lazy_mod)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Add _lazy_imports_obj and _lazy_imports_mod to dir(<module>)"""
    keys = (*globals().keys(), *_lazy_imports_obj.keys(), *_lazy_imports_mod.keys())
    return sorted(keys)


if TYPE_CHECKING or os.environ.get("OPENQDC_DISABLE_LAZY_LOADING", "0") == "1":
    # These types are imported lazily at runtime, but we need to tell type
    # checkers what they are.
    from ._version import __version__  # noqa
    from .datasets import AVAILABLE_DATASETS  # noqa
    from .datasets.base import BaseDataset  # noqa
    from .datasets.potential.ani import ANI1, ANI1CCX, ANI1X  # noqa
    from .datasets.potential.comp6 import COMP6  # noqa
    from .datasets.potential.dummy import Dummy  # noqa
    from .datasets.potential.gdml import GDML  # noqa
    from .datasets.potential.geom import GEOM  # noqa
    from .datasets.potential.iso_17 import ISO17  # noqa
    from .datasets.potential.molecule3d import Molecule3D  # noqa
    from .datasets.potential.multixcqm9 import MultixcQM9  # noqa
    from .datasets.potential.nabladft import NablaDFT  # noqa
    from .datasets.potential.orbnet_denali import OrbnetDenali  # noqa
    from .datasets.potential.pcqm import PCQM_B3LYP, PCQM_PM6  # noqa
    from .datasets.potential.qm7x import QM7X  # noqa
    from .datasets.potential.qmugs import QMugs  # noqa
    from .datasets.potential.revmd17 import RevMD17  # noqa
    from .datasets.potential.sn2_rxn import SN2RXN  # noqa
    from .datasets.potential.solvated_peptides import SolvatedPeptides  # noqa
    from .datasets.potential.spice import Spice, SpiceV2  # noqa
    from .datasets.potential.tmqm import TMQM  # noqa
    from .datasets.potential.transition1x import Transition1X  # noqa
    from .datasets.potential.waterclusters3_30 import WaterClusters  # noqa
