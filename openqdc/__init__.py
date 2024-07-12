import importlib
import os
from typing import TYPE_CHECKING

# The below lazy import logic is coming from openff-toolkit:
# https://github.com/openforcefield/openff-toolkit/blob/b52879569a0344878c40248ceb3bd0f90348076a/openff/toolkit/__init__.py#L44


# Dictionary of objects to lazily import; maps the object's name to its module path
def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


_lazy_imports_obj = {
    "__version__": "openqdc._version",
    "BaseDataset": "openqdc.datasets.base",
    # POTENTIAL
    "Alchemy": "openqdc.datasets.potential.alchemy",
    "ANI1": "openqdc.datasets.potential.ani",
    "ANI1CCX": "openqdc.datasets.potential.ani",
    "ANI1CCX_V2": "openqdc.datasets.potential.ani",
    "ANI1X": "openqdc.datasets.potential.ani",
    "ANI2X": "openqdc.datasets.potential.ani",
    "Spice": "openqdc.datasets.potential.spice",
    "SpiceV2": "openqdc.datasets.potential.spice",
    "SpiceVL2": "openqdc.datasets.potential.spice",
    "GEOM": "openqdc.datasets.potential.geom",
    "QMugs": "openqdc.datasets.potential.qmugs",
    "QMugs_V2": "openqdc.datasets.potential.qmugs",
    "ISO17": "openqdc.datasets.potential.iso_17",
    "COMP6": "openqdc.datasets.potential.comp6",
    "GDML": "openqdc.datasets.potential.gdml",
    "Molecule3D": "openqdc.datasets.potential.molecule3d",
    "OrbnetDenali": "openqdc.datasets.potential.orbnet_denali",
    "SN2RXN": "openqdc.datasets.potential.sn2_rxn",
    "QM7X": "openqdc.datasets.potential.qm7x",
    "QM7X_V2": "openqdc.datasets.potential.qm7x",
    "QM1B": "openqdc.datasets.potential.qm1b",
    "QM1B_SMALL": "openqdc.datasets.potential.qm1b",
    "NablaDFT": "openqdc.datasets.potential.nabladft",
    "SolvatedPeptides": "openqdc.datasets.potential.solvated_peptides",
    "WaterClusters": "openqdc.datasets.potential.waterclusters3_30",
    "SCANWaterClusters": "openqdc.datasets.potential.waterclusters",
    "TMQM": "openqdc.datasets.potential.tmqm",
    "PCQM_B3LYP": "openqdc.datasets.potential.pcqm",
    "PCQM_PM6": "openqdc.datasets.potential.pcqm",
    "RevMD17": "openqdc.datasets.potential.revmd17",
    "MD22": "openqdc.datasets.potential.md22",
    "Transition1X": "openqdc.datasets.potential.transition1x",
    "MultixcQM9": "openqdc.datasets.potential.multixcqm9",
    "MultixcQM9_V2": "openqdc.datasets.potential.multixcqm9",
    "QM7": "openqdc.datasets.potential.qmx",
    "QM7b": "openqdc.datasets.potential.qmx",
    "QM8": "openqdc.datasets.potential.qmx",
    "QM9": "openqdc.datasets.potential.qmx",
    "ProteinFragments": "openqdc.datasets.potential.proteinfragments",
    "MDDataset": "openqdc.datasets.potential.proteinfragments",
    "VQM24": "openqdc.datasets.potential.vqm24",
    # INTERACTION
    "DES5M": "openqdc.datasets.interaction.des",
    "DES370K": "openqdc.datasets.interaction.des",
    "DESS66": "openqdc.datasets.interaction.des",
    "DESS66x8": "openqdc.datasets.interaction.des",
    "L7": "openqdc.datasets.interaction.l7",
    "X40": "openqdc.datasets.interaction.x40",
    "Metcalf": "openqdc.datasets.interaction.metcalf",
    "Splinter": "openqdc.datasets.interaction.splinter",
    # DEBUG
    "Dummy": "openqdc.datasets.potential.dummy",
    "PredefinedDataset": "openqdc.datasets.potential.dummy",
    # ALL
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
    from ._version import __version__
    from .datasets import AVAILABLE_DATASETS
    from .datasets.base import BaseDataset

    # INTERACTION
    from .datasets.interaction.des import DES5M, DES370K, DESS66, DESS66x8
    from .datasets.interaction.l7 import L7
    from .datasets.interaction.metcalf import Metcalf
    from .datasets.interaction.splinter import Splinter
    from .datasets.interaction.x40 import X40

    # POTENTIAL
    from .datasets.potential.alchemy import Alchemy
    from .datasets.potential.ani import ANI1, ANI1CCX, ANI1CCX_V2, ANI1X, ANI2X
    from .datasets.potential.comp6 import COMP6
    from .datasets.potential.dummy import Dummy, PredefinedDataset
    from .datasets.potential.gdml import GDML
    from .datasets.potential.geom import GEOM
    from .datasets.potential.iso_17 import ISO17
    from .datasets.potential.md22 import MD22
    from .datasets.potential.molecule3d import Molecule3D
    from .datasets.potential.multixcqm9 import MultixcQM9, MultixcQM9_V2
    from .datasets.potential.nabladft import NablaDFT
    from .datasets.potential.orbnet_denali import OrbnetDenali
    from .datasets.potential.pcqm import PCQM_B3LYP, PCQM_PM6
    from .datasets.potential.proteinfragments import MDDataset, ProteinFragments
    from .datasets.potential.qm1b import QM1B, QM1B_SMALL
    from .datasets.potential.qm7x import QM7X, QM7X_V2
    from .datasets.potential.qmugs import QMugs, QMugs_V2
    from .datasets.potential.qmx import QM7, QM8, QM9, QM7b
    from .datasets.potential.revmd17 import RevMD17
    from .datasets.potential.sn2_rxn import SN2RXN
    from .datasets.potential.solvated_peptides import SolvatedPeptides
    from .datasets.potential.spice import Spice, SpiceV2, SpiceVL2
    from .datasets.potential.tmqm import TMQM
    from .datasets.potential.transition1x import Transition1X
    from .datasets.potential.vqm24 import VQM24
    from .datasets.potential.waterclusters import SCANWaterClusters
    from .datasets.potential.waterclusters3_30 import WaterClusters
