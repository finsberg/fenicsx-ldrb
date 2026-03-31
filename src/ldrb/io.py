import logging
import shutil
from pathlib import Path
from typing import NamedTuple, Sequence

from mpi4py import MPI

import dolfinx
import io4dolfinx
import numpy as np

from . import utils

logger = logging.getLogger(__name__)


class LDRBOutput(NamedTuple):
    f0: dolfinx.fem.Function
    s0: dolfinx.fem.Function
    n0: dolfinx.fem.Function
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
    f0: np.ndarray
    s0: np.ndarray
    n0: np.ndarray


def save(
    comm: MPI.Comm,
    filename: Path,
    functions: Sequence[dolfinx.fem.Function],
    overwrite: bool = False,
) -> None:
    attributes = {}

    if filename.exists():
        if overwrite:
            comm.barrier()
            shutil.rmtree(filename, ignore_errors=True)
            comm.barrier()
        else:
            logger.info(f"File {filename} already exists, skipping")
            return

    for i, f in enumerate(functions):
        name = f.name
        if i == 0:
            io4dolfinx.write_mesh(mesh=f.function_space.mesh, filename=filename)
        io4dolfinx.write_function_on_input_mesh(u=f, filename=filename)
        attributes[name] = utils.element2array(f.ufl_element())

    io4dolfinx.write_attributes(
        comm=comm,
        filename=filename,
        name="function_space",
        attributes=attributes,
    )


def load(
    comm: MPI.Comm,
    filename: Path,
    mesh: dolfinx.mesh.Mesh | None = None,
    function_space: dict[str, np.ndarray] | None = None,
) -> dict[str, dolfinx.fem.Function]:
    if not Path(filename).exists():
        raise FileNotFoundError(f"File {filename} does not exist")

    if mesh is None:
        mesh = io4dolfinx.read_mesh(comm=comm, filename=filename)

    function_space = io4dolfinx.read_attributes(comm=comm, filename=filename, name="function_space")
    assert function_space is not None

    # Assume same function space for all functions
    functions = {}
    for key, value in function_space.items():
        element = utils.array2element(value)
        V = dolfinx.fem.functionspace(mesh, element)
        f = dolfinx.fem.Function(V, name=key)
        io4dolfinx.read_function(u=f, filename=filename, name=key)
        functions[key] = f

    return functions
