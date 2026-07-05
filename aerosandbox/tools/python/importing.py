import importlib.util
import sys


def lazy_import(name):
    """
    Lazily import a module.

    Parameters
    ----------
    name
        The package name.

    Returns
    -------
    module
        The module itself, to be loaded on first use.

    Examples
    --------
    >>> asb = lazy_import("aerosandbox")  # Runs instantly
    >>> asb.Airplane()  # When this is called, ASB will be loaded.
    """
    spec = importlib.util.find_spec(name)
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module
