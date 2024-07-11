"""
Units conversion utilities module.

Available Energy units:
    ["kcal/mol", "kj/mol", "hartree", "ev" "mev", "ryd]

Available Distance units:
    ["ang", "nm", "bohr"]

Available Force units:
    Combinations between Energy and Distance units
"""

from enum import Enum, unique
from typing import Callable

from openqdc.utils.exceptions import ConversionAlreadyDefined, ConversionNotDefinedError

CONVERSION_REGISTRY = {}


# Redefined to avoid circular imports
class StrEnum(str, Enum):
    def __str__(self):
        return self.value.lower()


# Parent class for all conversion enums
class ConversionEnum(Enum):
    pass


@unique
class EnergyTypeConversion(ConversionEnum, StrEnum):
    """
    Define the possible energy units for conversion
    """

    KCAL_MOL = "kcal/mol"
    KJ_MOL = "kj/mol"
    HARTREE = "hartree"
    EV = "ev"
    MEV = "mev"
    RYD = "ryd"

    def to(self, energy: "EnergyTypeConversion") -> Callable[[float], float]:
        """
        Get the conversion function to convert the energy to the desired units.

        Parameters:
            energy: energy unit to convert to

        Returns:
            Callable to convert the distance to the desired units
        """
        return get_conversion(str(self), str(energy))


@unique
class DistanceTypeConversion(ConversionEnum, StrEnum):
    """
    Define the possible distance units for conversion
    """

    ANG = "ang"
    NM = "nm"
    BOHR = "bohr"

    def to(self, distance: "DistanceTypeConversion", fraction: bool = False) -> Callable[[float], float]:
        """
        Get the conversion function to convert the distance to the desired units.

        Parameters:
            distance: distance unit to convert to
            fraction: whether it is distance^1 or distance^-1

        Returns:
            callable to convert the distance to the desired units
        """
        return get_conversion(str(self), str(distance)) if not fraction else get_conversion(str(distance), str(self))


@unique
class ForceTypeConversion(ConversionEnum):
    """
    Define the possible foce units for conversion
    """

    #     Name      = EnergyTypeConversion,         , DistanceTypeConversion
    HARTREE_BOHR = EnergyTypeConversion.HARTREE, DistanceTypeConversion.BOHR
    HARTREE_ANG = EnergyTypeConversion.HARTREE, DistanceTypeConversion.ANG
    HARTREE_NM = EnergyTypeConversion.HARTREE, DistanceTypeConversion.NM
    EV_BOHR = EnergyTypeConversion.EV, DistanceTypeConversion.BOHR
    EV_ANG = EnergyTypeConversion.EV, DistanceTypeConversion.ANG
    EV_NM = EnergyTypeConversion.EV, DistanceTypeConversion.NM
    KCAL_MOL_BOHR = EnergyTypeConversion.KCAL_MOL, DistanceTypeConversion.BOHR
    KCAL_MOL_ANG = EnergyTypeConversion.KCAL_MOL, DistanceTypeConversion.ANG
    KCAL_MOL_NM = EnergyTypeConversion.KCAL_MOL, DistanceTypeConversion.NM
    KJ_MOL_BOHR = EnergyTypeConversion.KJ_MOL, DistanceTypeConversion.BOHR
    KJ_MOL_ANG = EnergyTypeConversion.KJ_MOL, DistanceTypeConversion.ANG
    KJ_MOL_NM = EnergyTypeConversion.KJ_MOL, DistanceTypeConversion.NM
    MEV_BOHR = EnergyTypeConversion.MEV, DistanceTypeConversion.BOHR
    MEV_ANG = EnergyTypeConversion.MEV, DistanceTypeConversion.ANG
    MEV_NM = EnergyTypeConversion.MEV, DistanceTypeConversion.NM
    RYD_BOHR = EnergyTypeConversion.RYD, DistanceTypeConversion.BOHR
    RYD_ANG = EnergyTypeConversion.RYD, DistanceTypeConversion.ANG
    RYD_NM = EnergyTypeConversion.RYD, DistanceTypeConversion.NM

    def __init__(self, energy: EnergyTypeConversion, distance: DistanceTypeConversion):
        self.energy = energy
        self.distance = distance

    def __str__(self):
        return f"{self.energy}/{self.distance}"

    def to(self, energy: EnergyTypeConversion, distance: DistanceTypeConversion) -> Callable[[float], float]:
        """
        Get the conversion function to convert the force to the desired units.

        Parameters:
            energy: energy unit to convert to
            distance: distance unit to convert to

        Returns:
            callable to convert the distance to the desired units
        """
        return lambda x: self.distance.to(distance, fraction=True)(self.energy.to(energy)(x))


class Conversion:
    """
    Conversion from one unit system to another defined by a name and a callable
    """

    def __init__(self, in_unit: str, out_unit: str, func: Callable[[float], float]):
        """

        Parameters:
            in_unit: String defining the units of the current values
            out_unit: String defining the target units
            func: The callable to compute the conversion
        """
        name = "convert_" + in_unit.lower().strip() + "_to_" + out_unit.lower().strip()

        if name in CONVERSION_REGISTRY:
            raise ConversionAlreadyDefined(in_unit, out_unit)
        CONVERSION_REGISTRY[name] = self

        self.name = name
        self.fn = func

    def __call__(self, x):
        return self.fn(x)


def get_conversion(in_unit: str, out_unit: str) -> Callable[[float], float]:
    """
    Utility function to get the conversion function between two units.

    Parameters:
        in_unit : The input unit
        out_unit : The output unit

    Returns:
        The conversion function
    """
    name = "convert_" + in_unit.lower().strip() + "_to_" + out_unit.lower().strip()
    if in_unit.lower().strip() == out_unit.lower().strip():
        return lambda x: x
    if name not in CONVERSION_REGISTRY:
        raise ConversionNotDefinedError(in_unit, out_unit)
    return CONVERSION_REGISTRY[name]


# Conversion definitions

# ev conversion
Conversion("ev", "kcal/mol", lambda x: x * 23.0605)
Conversion("ev", "hartree", lambda x: x * 0.0367493)
Conversion("ev", "kj/mol", lambda x: x * 96.4853)
Conversion("ev", "mev", lambda x: x * 1000.0)
Conversion("mev", "ev", lambda x: x * 0.0001)
Conversion("ev", "ryd", lambda x: x * 0.07349864)

# kcal/mol conversion
Conversion("kcal/mol", "ev", lambda x: x * 0.0433641)
Conversion("kcal/mol", "hartree", lambda x: x * 0.00159362)
Conversion("kcal/mol", "kj/mol", lambda x: x * 4.184)
Conversion("kcal/mol", "mev", lambda x: get_conversion("ev", "mev")(get_conversion("kcal/mol", "ev")(x)))
Conversion("kcal/mol", "ryd", lambda x: x * 0.00318720)

# hartree conversion
Conversion("hartree", "ev", lambda x: x * 27.211386246)
Conversion("hartree", "kcal/mol", lambda x: x * 627.509)
Conversion("hartree", "kj/mol", lambda x: x * 2625.5)
Conversion("hartree", "mev", lambda x: get_conversion("ev", "mev")(get_conversion("hartree", "ev")(x)))
Conversion("hartree", "ryd", lambda x: x * 2.0)

# kj/mol conversion
Conversion("kj/mol", "ev", lambda x: x * 0.0103643)
Conversion("kj/mol", "kcal/mol", lambda x: x * 0.239006)
Conversion("kj/mol", "hartree", lambda x: x * 0.000380879)
Conversion("kj/mol", "mev", lambda x: get_conversion("ev", "mev")(get_conversion("kj/mol", "ev")(x)))
Conversion("kj/mol", "ryd", lambda x: x * 0.000301318)

# Rydberg conversion
Conversion("ryd", "ev", lambda x: x * 13.60569301)
Conversion("ryd", "kcal/mol", lambda x: x * 313.7545)
Conversion("ryd", "hartree", lambda x: x * 0.5)
Conversion("ryd", "kj/mol", lambda x: x * 1312.75)
Conversion("ryd", "mev", lambda x: get_conversion("ev", "mev")(get_conversion("ryd", "ev")(x)))

# distance conversions
Conversion("bohr", "ang", lambda x: x * 0.52917721092)
Conversion("ang", "bohr", lambda x: x / 0.52917721092)
Conversion("ang", "nm", lambda x: x * 0.1)
Conversion("nm", "ang", lambda x: x * 10.0)
Conversion("nm", "bohr", lambda x: x * 18.8973)
Conversion("bohr", "nm", lambda x: x / 18.8973)
