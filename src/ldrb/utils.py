from enum import Enum
from typing import Iterable

import basix
import dolfinx
import numpy as np


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
    kwargs = {"degree": int(degree_str), "cell": mesh.ufl_cell().cellname()}
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
) -> dolfinx.fem.functionspace:
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


def element2array(el: basix.finite_element.FiniteElement) -> np.ndarray:
    return np.array(
        [int(el.family), int(el.cell_type), int(el.degree), int(el.discontinuous)],
        dtype=np.uint8,
    )


def number2Enum(num: int, enum: Iterable) -> Enum:
    for e in enum:
        if int(e) == num:
            return e
    raise ValueError(f"Invalid value {num} for enum {enum}")


def array2element(arr: np.ndarray) -> basix.finite_element.FiniteElement:
    family = number2Enum(arr[0], basix.ElementFamily)
    cell_type = number2Enum(arr[1], basix.CellType)
    degree = int(arr[2])
    discontinuous = bool(arr[3])
    # TODO: Shape is hardcoded to (3,) for now, but this should also be stored
    return basix.ufl.element(
        family=family,
        cell=cell_type,
        degree=degree,
        discontinuous=discontinuous,
        shape=(3,),
    )
