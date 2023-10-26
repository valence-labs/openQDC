import importlib
from functools import wraps
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class MissingOptionalDependencyError(BaseException):
    """
    An exception raised when an optional dependency is required
    but cannot be found.

    Attributes
    ----------
    library_name
        The name of the missing library.
    """

    def __init__(self, library_name: str):
        """

        Parameters
        ----------
        library_name
            The name of the missing library.
        license_issue
            Whether the library was importable but was unusable due
            to a missing license.
        """

        message = f"The required {library_name} module could not be imported."

        super(MissingOptionalDependencyError, self).__init__(message)

        self.library_name = library_name


def has_package(package_name: str) -> bool:
    """
    Helper function to generically check if a Python package is installed.
    Intended to be used to check for optional dependencies.

    Parameters
    ----------
    package_name : str
        The name of the Python package to check the availability of

    Returns
    -------
    package_available : bool
        Boolean indicator if the package is available or not

    Examples
    --------
    >>> has_numpy = has_package('numpy')
    >>> has_numpy
    True
    >>> has_foo = has_package('other_non_installed_package')
    >>> has_foo
    False
    """
    try:
        importlib.import_module(package_name)
    except ModuleNotFoundError:
        return False
    return True


def requires_package(package_name: str) -> Callable[..., Any]:
    """
    Helper function to denote that a funciton requires some optional
    dependency. A function decorated with this decorator will raise
    `MissingOptionalDependencyError` if the package is not found by
    `importlib.import_module()`.

    Parameters
    ----------
    package_name : str
        The name of the module to be imported.

    Raises
    ------
    MissingOptionalDependencyError

    """

    def inner_decorator(function: F) -> F:
        @wraps(function)
        def wrapper(*args, **kwargs):
            import importlib

            try:
                importlib.import_module(package_name)
            except ImportError:
                raise MissingOptionalDependencyError(library_name=package_name)
            except Exception as e:
                raise e

            return function(*args, **kwargs)

        return wrapper

    return inner_decorator


def get_dir():
    r"""
    Get the Torch Hub cache directory used for storing downloaded models & weights.

    If :func:`~torch.hub.set_dir` is not called, default path is ``$TORCH_HOME/hub`` where
    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesystem layout, with a default value ``~/.cache`` if the environment
    variable is not set.
    """

    if _hub_dir is not None:
        return _hub_dir
    # return os.path.join(_get_torch_home(), 'hub')


def set_dir(d):
    r"""
    Optionally set the Torch Hub directory used to save downloaded models & weights.

    Args:
        d (str): path to a local folder to save downloaded models & weights.
    """
    global _hub_dir
    # _hub_dir = os.path.expanduser(d)