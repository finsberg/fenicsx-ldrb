from importlib.metadata import metadata

from . import calculus, io, ldrb, utils
from .ldrb import dolfinx_ldrb, project_gradients, scalar_laplacians
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
    "io",
]
