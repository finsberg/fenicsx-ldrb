from enum import Enum
from typing import Iterable, Sequence

from mpi4py import MPI

import basix
import dolfinx
import numpy as np

QUADRATURE_FAMILY = 100


def default_markers() -> dict[str, list[int]]:
    """
    Default markers for the mesh boundaries
    """
    return dict(base=[10], rv=[20], lv=[30], epi=[40])


def parse_element(space_string: str, mesh: dolfinx.mesh.Mesh, dim: int) -> basix.ufl._ElementBase:
    """
    Parse a string representation of a basix element family
    """
    family_str, degree_str = space_string.split("_")
    kwargs = {"degree": int(degree_str), "cell": mesh.basix_cell()}
    if dim > 1:
        if family_str in ["Quadrature", "Q", "Quad"]:
            kwargs["value_shape"] = (dim,)
        else:
            kwargs["shape"] = (dim,)

    if family_str in ["Lagrange", "P", "CG"]:
        el = basix.ufl.element(family=basix.ElementFamily.P, discontinuous=False, **kwargs)
    elif family_str in ["Discontinuous Lagrange", "DG", "dP"]:
        el = basix.ufl.element(family=basix.ElementFamily.P, discontinuous=True, **kwargs)

    elif family_str in ["Quadrature", "Q", "Quad"]:
        el = basix.ufl.quadrature_element(scheme="default", **kwargs)
    else:
        families = list(basix.ElementFamily.__members__.keys())
        msg = f"Unknown element family: {family_str}, available families: {families}"
        raise ValueError(msg)
    return el


def space_from_string(
    space_string: str, mesh: dolfinx.mesh.Mesh, dim: int
) -> dolfinx.fem.FunctionSpace:
    """
    Constructed a finite elements space from a string
    representation of the space

    Arguments
    ---------
    space_string : str
        A string on the form {family}_{degree} which
        determines the space. Example 'Lagrange_1'.
    mesh : df.Mesh
        The mesh
    dim : int
        1 for scalar space, 3 for vector space.
    """
    el = parse_element(space_string, mesh, dim)
    return dolfinx.fem.functionspace(mesh, el)


def geometry_degree(mesh: dolfinx.mesh.Mesh) -> int:
    """The degree of the coordinate element of the mesh"""
    domain = mesh.ufl_domain()
    assert domain is not None
    return int(domain.ufl_coordinate_element().degree)


def laplace_space(mesh: dolfinx.mesh.Mesh) -> dolfinx.fem.FunctionSpace:
    """The space in which the Laplace problems are solved: Lagrange elements of the
    same degree as the geometry, so that the solutions are as accurate as the geometry.
    """
    return dolfinx.fem.functionspace(mesh, ("Lagrange", geometry_degree(mesh)))


def closest_dofs(V: dolfinx.fem.FunctionSpace, point: Sequence[float] | np.ndarray) -> np.ndarray:
    """The local (owned and ghost) dofs at the dof coordinate closest to `point`."""
    comm = V.mesh.comm
    coords = V.tabulate_dof_coordinates()
    distance = np.linalg.norm(coords - np.asarray(point), axis=1)
    local = (float(distance.min()), comm.rank) if len(distance) > 0 else (np.inf, comm.rank)
    _, owner = comm.allreduce(local, op=MPI.MINLOC)
    closest = coords[distance.argmin()] if comm.rank == owner else None
    closest = comm.bcast(closest, root=owner)
    return np.flatnonzero(np.all(coords == closest, axis=1)).astype(np.int32)


def element2array(el: basix.ufl._BlockedElement) -> np.ndarray:
    try:
        el = el.basix_element
        family = int(el.family)
        cell_type = int(el.cell_type)
        degree = int(el.degree)
        discontinuous = int(el.discontinuous)

    except NotImplementedError:
        assert el.family_name == "quadrature"

        family = QUADRATURE_FAMILY
        cell_type = int(el.cell_type)
        degree = int(el.degree)
        discontinuous = int(el.discontinuous)

    return np.array(
        [family, cell_type, degree, discontinuous],
        dtype=np.uint8,
    )


def number2Enum(num: int, enum: Iterable) -> Enum:
    for e in enum:
        if int(e) == num:
            return e
    raise ValueError(f"Invalid value {num} for enum {enum}")


def array2element(arr: np.ndarray) -> basix.finite_element.FiniteElement:
    cell_type = number2Enum(arr[1], basix.CellType)
    degree = int(arr[2])
    discontinuous = bool(arr[3])
    if arr[0] == QUADRATURE_FAMILY:
        return basix.ufl.quadrature_element(
            scheme="default",
            cell=cell_type,
            degree=degree,
            value_shape=(3,),
        )
    else:
        family = number2Enum(arr[0], basix.ElementFamily)

        # TODO: Shape is hardcoded to (3,) for now, but this should also be stored
        return basix.ufl.element(
            family=family,
            cell=cell_type,
            degree=degree,
            discontinuous=discontinuous,
            shape=(3,),
        )
