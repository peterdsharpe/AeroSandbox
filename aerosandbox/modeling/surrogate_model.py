from aerosandbox.common import AeroSandboxObject
from abc import abstractmethod


class SurrogateModel(AeroSandboxObject):
    """
    A SurrogateModel is effectively a callable; it only has the __call__ method, and all subclasses must explicitly
    overwrite this. The only reason it is not a callable is that you want to be able to save it to disk (via
    pickling) while also having the capability to save associated data (for example, constants associated with a
    particular model).
    """
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, *args):
        pass
