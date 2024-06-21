import logging
import shutil
from pathlib import Path
from typing import NamedTuple, Sequence

from mpi4py import MPI

import adios4dolfinx
import dolfinx
import numpy as np

from . import utils

logger = logging.getLogger(__name__)


class LDRBOutput(NamedTuple):
    fiber: dolfinx.fem.Function
    sheet: dolfinx.fem.Function
    sheet_normal: dolfinx.fem.Function
    lv: dolfinx.fem.Function | None = None
    rv: dolfinx.fem.Function | None = None
    epi: dolfinx.fem.Function | None = None
    lv_rv: dolfinx.fem.Function | None = None
    apex: dolfinx.fem.Function | None = None
    lv_scalar: dolfinx.fem.Function | None = None
    rv_scalar: dolfinx.fem.Function | None = None
    epi_scalar: dolfinx.fem.Function | None = None
    lv_rv_scalar: dolfinx.fem.Function | None = None
    apex_scalar: dolfinx.fem.Function | None = None
    lv_gradient: dolfinx.fem.Function | None = None
    rv_gradient: dolfinx.fem.Function | None = None
    epi_gradient: dolfinx.fem.Function | None = None
    lv_rv_gradient: dolfinx.fem.Function | None = None
    apex_gradient: dolfinx.fem.Function | None = None
    markers_scalar: dolfinx.fem.Function | None = None


class FiberSheetSystem(NamedTuple):
    fiber: np.ndarray
    sheet: np.ndarray
    sheet_normal: np.ndarray


def save(
    comm: MPI.Comm,
    filename: Path,
    functions: Sequence[dolfinx.fem.Function],
    overwrite: bool = False,
) -> None:
    attributes = {}

    if filename.exists():
        if overwrite:
            shutil.rmtree(filename)
        else:
            logger.info(f"File {filename} already exists, skipping")
            return

    for i, f in enumerate(functions):
        name = f.name
        if i == 0:
            adios4dolfinx.write_mesh(mesh=f.function_space.mesh, filename=filename)
        adios4dolfinx.write_function(u=f, filename=filename)
        attributes[name] = utils.element2array(f.ufl_element().basix_element)

    adios4dolfinx.write_attributes(
        comm=comm,
        filename=filename,
        name="function_space",
        attributes=attributes,
    )


def load(
    comm: MPI.Comm,
    filename: Path,
) -> list[dolfinx.fem.Function]:
    if not Path(filename).exists():
        raise FileNotFoundError(f"File {filename} does not exist")

    mesh = adios4dolfinx.read_mesh(comm=comm, filename=filename)

    function_space = adios4dolfinx.read_attributes(
        comm=comm, filename=filename, name="function_space"
    )
    # Assume same function space for all functions
    functions = []
    for key, value in function_space.items():
        element = utils.array2element(value)
        V = dolfinx.fem.functionspace(mesh, element)
        f = dolfinx.fem.Function(V, name=key)
        adios4dolfinx.read_function(u=f, filename=filename, name=key)
        functions.append(f)

    return functions
