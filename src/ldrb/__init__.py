from importlib.metadata import metadata

from . import calculus
from . import ldrb

# from . import save
from . import utils
from .ldrb import dolfinx_ldrb
from .ldrb import project_gradients
from .ldrb import scalar_laplacians
from .utils import space_from_string

meta = metadata("fenicsx-ldrb")
__version__ = meta["Version"]
__author__ = meta["Author"]
__license__ = meta["License"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]


__all__ = [
    "ldrb",
    "dolfinx_ldrb",
    "scalar_laplacians",
    "project_gradients",
    "utils",
    "space_from_string",
    "calculus",
]
