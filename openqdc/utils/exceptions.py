from typing import Final

from openqdc.utils.constants import POSSIBLE_NORMALIZATION

PROPERTY_NOT_AVAILABLE_ERROR: Final[
    str
] = """This property for this dataset not available.
Please open an issue on Github for the team to look into it."""


class OpenQDCException(Exception):
    """Base exception for custom exceptions raised by the openQDC"""

    def __init__(self, msg: str):
        super().__init__(msg)
        self.msg = msg

    def __str__(self):
        return self.msg


class DatasetNotAvailableError(OpenQDCException):
    """Raised when a dataset is not available"""

    msg = "Dataset {dataset_name} is not available. Please open an issue on Github for the team to look into it."

    def __init__(self, dataset_name):
        super().__init__(self.msg.format(dataset_name=dataset_name))


class StatisticsNotAvailableError(DatasetNotAvailableError):
    """Raised when statistics are not available"""

    msg = (
        "Statistics for dataset {dataset_name} are not available."
        + "Please open an issue on Github for the team to look into it."
    )


class NormalizationNotAvailableError(OpenQDCException):
    """Raised when normalization is not available"""

    def __init__(self, normalization):
        msg = f"Normalization={normalization} is not valid. Must be one of {POSSIBLE_NORMALIZATION}"
        super().__init__(msg)


class ConversionNotDefinedError(OpenQDCException, ValueError):
    """Raised when a conversion is not defined"""

    _error_message = """
    Conversion from {in_unit} to {out_unit} is not defined in the conversion registry.
    To add a new conversion, use the following syntax or open an issue on Github for the team to look into it:

    Conversion("{in_unit}", "{out_unit}", lambda x: x * conversion_factor)
    """

    def __init__(self, in_unit, out_unit):
        super().__init__(self._error_message.format(in_unit=in_unit, out_unit=out_unit))


class ConversionAlreadyDefined(ConversionNotDefinedError):
    """Raised when a conversion is already defined"""

    _error_message = """
    Conversion from {in_unit} to {out_unit} is alread defined in the conversion registry.
    To reuse the same metric, use get_conversion({in_unit}, {out_unit}).
    """
