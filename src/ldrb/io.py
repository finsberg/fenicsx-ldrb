import logging
import shutil
from pathlib import Path
from typing import NamedTuple, Sequence

from mpi4py import MPI

import adios2
import adios4dolfinx
import dolfinx
import numpy as np
from packaging.version import Version

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
            adios4dolfinx.write_mesh(mesh=f.function_space.mesh, filename=filename)
        adios4dolfinx.write_function_on_input_mesh(u=f, filename=filename)
        attributes[name] = utils.element2array(f.ufl_element())

    adios4dolfinx.write_attributes(
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
        mesh = adios4dolfinx.read_mesh(comm=comm, filename=filename)

    if Version(np.__version__) >= Version("2.11") and Version(adios2.__version__) < Version(
        "2.10.2"
    ):
        # Broken on new numpy and old adios2
        function_space = adios4dolfinx.read_attributes(
            comm=comm, filename=filename, name="function_space"
        )
    else:
        if not function_space:
            raise ValueError(
                "function_space must be provided if numpy version is lower "
                "than 1.21.0 and adios2 version is lower than 2.10."
            )
    assert function_space is not None

    # Assume same function space for all functions
    functions = {}
    for key, value in function_space.items():
        element = utils.array2element(value)
        V = dolfinx.fem.functionspace(mesh, element)
        f = dolfinx.fem.Function(V, name=key)
        adios4dolfinx.read_function(u=f, filename=filename, name=key)
        functions[key] = f

    return functions
