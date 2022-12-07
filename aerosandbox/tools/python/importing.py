import importlib
import sys


def lazy_import(name):
    """
    Allows you to lazily import a module.

    Usage:
    >>> asb = lazy_import("aerosandbox")  # Runs instantly
    >>> asb.Airplane()  # When this is called, ASB will be loaded.

    Args:
        name: The package name

    Returns: The module itself, to be loaded on first use.

    """
    spec = importlib.util.find_spec(name)
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module
