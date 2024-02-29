import importlib
import os
from typing import TYPE_CHECKING  # noqa F401

# The below lazy import logic is coming from openff-toolkit:
# https://github.com/openforcefield/openff-toolkit/blob/b52879569a0344878c40248ceb3bd0f90348076a/openff/toolkit/__init__.py#L44

# Dictionary of objects to lazily import; maps the object's name to its module path

_lazy_imports_obj = {
    "__version__": "openqdc._version",
    "ANI1": "openqdc.datasets.ani",
    "ANI1CCX": "openqdc.datasets.ani",
    "ANI1X": "openqdc.datasets.ani",
    "Spice": "openqdc.datasets.spice",
    "GEOM": "openqdc.datasets.geom",
    "QMugs": "openqdc.datasets.qmugs",
    "ISO17": "openqdc.datasets.iso_17",
    "COMP6": "openqdc.datasets.comp6",
    "GDML": "openqdc.datasets.gdml",
    "Molecule3D": "openqdc.datasets.molecule3d",
    "OrbnetDenali": "openqdc.datasets.orbnet_denali",
    "SN2RXN": "openqdc.datasets.sn2_rxn",
    "QM7X": "openqdc.datasets.qm7x",
    "DESS": "openqdc.datasets.dess",
    "NablaDFT": "openqdc.datasets.nabladft",
    "SolvatedPeptides": "openqdc.datasets.solvated_peptides",
    "WaterClusters": "openqdc.datasets.waterclusters3_30",
    "TMQM": "openqdc.datasets.tmqm",
    "Dummy": "openqdc.datasets.dummy",
    "PCQM_B3LYP": "openqdc.datasets.pcqm",
    "PCQM_PM6": "openqdc.datasets.pcqm",
    "Transition1X": "openqdc.datasets.transition1x",
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
    from .datasets import *
    from ._version import __version__
    from .utils import * 
