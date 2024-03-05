"""
Unit conversion utils.

Energy units:
    ["kcal/mol", "kj/mol", "hartree", "ev"]

Distance units:
    ["ang", "nm", "bohr"]
"""

from typing import Callable

from openqdc.utils.exceptions import ConversionAlreadyDefined, ConversionNotDefinedError

CONVERSION_REGISTRY = {}


class Conversion:
    """
    Conversion from one unit system to another.

    Attributes
    ----------
    name
        A human-readable name for the conversion
    fn:
        The callable to compute the conversion
    """

    def __init__(self, in_unit: str, out_unit: str, func: Callable[[float], float]):
        """

        Parameters
        ----------
        in_unit
            String defining the units of the current values
        out_unit
            String defining the target units
        func
            The callable to compute the conversion
        """
        name = "convert_" + in_unit.lower().strip() + "_to_" + out_unit.lower().strip()

        if name in CONVERSION_REGISTRY:
            raise ConversionAlreadyDefined(in_unit, out_unit)
        CONVERSION_REGISTRY[name] = self

        self.name = name
        self.fn = func

    def __call__(self, x):
        """Convert measure"""
        return self.fn(x)


def get_conversion(in_unit: str, out_unit: str):
    name = "convert_" + in_unit.lower().strip() + "_to_" + out_unit.lower().strip()
    if in_unit.lower().strip() == out_unit.lower().strip():
        return lambda x: x
    if name not in CONVERSION_REGISTRY:
        raise ConversionNotDefinedError(in_unit, out_unit)
    return CONVERSION_REGISTRY[name]


# ev conversion
Conversion("ev", "kcal/mol", lambda x: x * 23.0605)
Conversion("ev", "hartree", lambda x: x * 0.0367493)
Conversion("ev", "kj/mol", lambda x: x * 96.4853)
Conversion("mev", "ev", lambda x: x * 1000.0)
Conversion("ev", "mev", lambda x: x * 0.0001)

# kcal/mol conversion
Conversion("kcal/mol", "ev", lambda x: x * 0.0433641)
Conversion("kcal/mol", "hartree", lambda x: x * 0.00159362)
Conversion("kcal/mol", "kj/mol", lambda x: x * 4.184)

# hartree conversion
Conversion("hartree", "ev", lambda x: x * 27.211386246)
Conversion("hartree", "kcal/mol", lambda x: x * 627.509)
Conversion("hartree", "kj/mol", lambda x: x * 2625.5)

# kj/mol conversion
Conversion("kj/mol", "ev", lambda x: x * 0.0103643)
Conversion("kj/mol", "kcal/mol", lambda x: x * 0.239006)
Conversion("kj/mol", "hartree", lambda x: x * 0.000380879)

# distance conversions
Conversion("bohr", "ang", lambda x: x * 0.52917721092)
Conversion("ang", "bohr", lambda x: x / 0.52917721092)
Conversion("ang", "nm", lambda x: x * 0.1)
Conversion("nm", "ang", lambda x: x * 10.0)
Conversion("nm", "bohr", lambda x: x * 18.8973)
Conversion("bohr", "nm", lambda x: x / 18.8973)

# common forces conversion
Conversion("hartree/bohr", "ev/ang", lambda x: get_conversion("ang", "bohr")(get_conversion("hartree", "ev")(x)))
Conversion("hartree/bohr", "ev/bohr", lambda x: get_conversion("hartree", "ev")(x))
Conversion("hartree/bohr", "kcal/mol/bohr", lambda x: get_conversion("hartree", "kcal/mol")(x))
Conversion(
    "hartree/bohr", "kcal/mol/ang", lambda x: get_conversion("ang", "bohr")(get_conversion("hartree", "kcal/mol")(x))
)
Conversion("hartree/ang", "kcal/mol/ang", lambda x: get_conversion("hartree", "kcal/mol")(x))
Conversion("hartree/ang", "hartree/bohr", lambda x: get_conversion("bohr", "ang")(x))
Conversion("hartree/bohr", "hartree/ang", lambda x: get_conversion("ang", "bohr")(x))
Conversion("kcal/mol/bohr", "hartree/bohr", lambda x: get_conversion("kcal/mol", "hartree")(x))
Conversion("ev/ang", "hartree/ang", lambda x: get_conversion("ev", "hartree")(x))
Conversion("ev/bohr", "hartree/bohr", lambda x: get_conversion("ev", "hartree")(x))
Conversion("ev/bohr", "ev/ang", lambda x: get_conversion("ang", "bohr")(x))
Conversion("ev/bohr", "kcal/mol/ang", lambda x: get_conversion("ang", "bohr")(get_conversion("ev", "kcal/mol")(x)))
Conversion("kcal/mol/bohr", "kcal/mol/ang", lambda x: get_conversion("ang", "bohr")(x))
Conversion("ev/ang", "kcal/mol/ang", lambda x: get_conversion("ev", "kcal/mol")(x))