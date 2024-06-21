from __future__ import annotations

import logging

# from dolfin.mesh.meshfunction import MeshFunction
from mpi4py import MPI

import dolfinx

# import dolfin as df
import numpy as np
import ufl
from dolfinx.fem.petsc import LinearProblem

from . import io, utils

logger = logging.getLogger(__name__)


def standard_dofs(n: int) -> np.ndarray:
    """
    Get the standard list of dofs for a given length
    """

    x_dofs = np.arange(0, 3 * n, 3)
    y_dofs = np.arange(1, 3 * n, 3)
    z_dofs = np.arange(2, 3 * n, 3)
    scalar_dofs = np.arange(0, n)
    return np.stack([x_dofs, y_dofs, z_dofs, scalar_dofs], -1)


def compute_fiber_sheet_system(
    lv_scalar: np.ndarray,
    lv_gradient: np.ndarray,
    epi_scalar: np.ndarray,
    epi_gradient: np.ndarray,
    apex_gradient: np.ndarray,
    dofs: np.ndarray | None = None,
    rv_scalar: np.ndarray | None = None,
    rv_gradient: np.ndarray | None = None,
    lv_rv_scalar: np.ndarray | None = None,
    marker_scalar: np.ndarray | None = None,
    alpha_endo_lv: float = 40,
    alpha_epi_lv: float = -50,
    alpha_endo_rv: float | None = None,
    alpha_epi_rv: float | None = None,
    alpha_endo_sept: float | None = None,
    alpha_epi_sept: float | None = None,
    beta_endo_lv: float = -65,
    beta_epi_lv: float = 25,
    beta_endo_rv: float | None = None,
    beta_epi_rv: float | None = None,
    beta_endo_sept: float | None = None,
    beta_epi_sept: float | None = None,
) -> io.FiberSheetSystem:
    """
    Compute the fiber-sheets system on all degrees of freedom.
    """
    if dofs is None:
        dofs = standard_dofs(len(lv_scalar))
    if rv_scalar is None:
        rv_scalar = np.zeros_like(lv_scalar)
    if lv_rv_scalar is None:
        lv_rv_scalar = np.zeros_like(lv_scalar)
    if rv_gradient is None:
        rv_gradient = np.zeros_like(lv_gradient)

    alpha_endo_rv = alpha_endo_rv or alpha_endo_lv
    alpha_epi_rv = alpha_epi_rv or alpha_epi_lv
    alpha_endo_sept = alpha_endo_sept or alpha_endo_lv
    alpha_epi_sept = alpha_epi_sept or alpha_epi_lv

    beta_endo_rv = beta_endo_rv or beta_endo_lv
    beta_epi_rv = beta_epi_rv or beta_epi_lv
    beta_endo_sept = beta_endo_sept or beta_endo_lv
    beta_epi_sept = beta_epi_sept or beta_epi_lv

    logger.info("Compute fiber-sheet system")
    logger.info("Angles: ")
    logger.info(
        (
            "alpha: "
            f"\n endo_lv: {alpha_endo_lv}"
            f"\n epi_lv: {alpha_epi_lv}"
            f"\n endo_septum: {alpha_endo_sept}"
            f"\n epi_septum: {alpha_epi_sept}"
            f"\n endo_rv: {alpha_endo_rv}"
            f"\n epi_rv: {alpha_epi_rv}"
            ""
        )
    )
    logger.info(
        (
            "beta: "
            f"\n endo_lv: {beta_endo_lv}"
            f"\n epi_lv: {beta_epi_lv}"
            f"\n endo_septum: {beta_endo_sept}"
            f"\n epi_septum: {beta_epi_sept}"
            f"\n endo_rv: {beta_endo_rv}"
            f"\n epi_rv: {beta_epi_rv}"
            ""
        )
    )

    f0 = np.zeros_like(lv_gradient)
    s0 = np.zeros_like(lv_gradient)
    n0 = np.zeros_like(lv_gradient)
    if marker_scalar is None:
        marker_scalar = np.zeros_like(lv_scalar)

    from .calculus import (
        compute_fiber_sheet_system as _compute_fiber_sheet_system,
    )

    _compute_fiber_sheet_system(
        f0,
        s0,
        n0,
        dofs[:, 0],
        dofs[:, 1],
        dofs[:, 2],
        dofs[:, 3],
        lv_scalar,
        rv_scalar,
        epi_scalar,
        lv_rv_scalar,
        lv_gradient,
        rv_gradient,
        epi_gradient,
        apex_gradient,
        marker_scalar,
        alpha_endo_lv,
        alpha_epi_lv,
        alpha_endo_rv,
        alpha_epi_rv,
        alpha_endo_sept,
        alpha_epi_sept,
        beta_endo_lv,
        beta_epi_lv,
        beta_endo_rv,
        beta_epi_rv,
        beta_endo_sept,
        beta_epi_sept,
    )

    return io.FiberSheetSystem(fiber=f0, sheet=s0, sheet_normal=n0)


def dofs_from_function_space(mesh: dolfinx.mesh.Mesh, fiber_space: str) -> np.ndarray:
    """
    Get the dofs from a function spaces define in the
    fiber_space string.
    """
    Vv = utils.space_from_string(fiber_space, mesh, dim=3)
    V = utils.space_from_string(fiber_space, mesh, dim=1)
    # Get dofs
    # FIXME: Need to make this more robust and working in parallel
    dim = Vv.mesh.geometry.dim
    bs = Vv.dofmap.index_map_bs
    start, end = Vv.dofmap.index_map.local_range
    x_dofs = np.arange(0, bs * (end - start), dim)
    y_dofs = np.arange(1, bs * (end - start), dim)
    z_dofs = np.arange(2, bs * (end - start), dim)

    start, end = V.dofmap.index_map.local_range
    scalar_dofs = np.arange(0, end - start)
    # scalar_dofs = [
    #     dof
    #     for dof in range(end - start)
    #     if V.dofmap.index_map.local_to_global_index(dof)
    #       not in V.dofmap().local_to_global_unowned()
    # ]
    # breakpoint()

    return np.stack([x_dofs, y_dofs, z_dofs, scalar_dofs], -1)


def transform_markers(markers: dict[str, list[int]]) -> dict[str, list[int]]:
    """Convert markers generated with gmsh to format for ldrb"""
    if "ENDO_LV" in markers:
        return dict(
            lv=[markers["ENDO_LV"][0]],
            rv=[markers["ENDO_RV"][0]],
            epi=[markers["EPI"][0]],
            base=[markers["BASE"][0]],
        )
    elif "ENDO" in markers:
        return dict(
            lv=[markers["ENDO"][0]],
            epi=[markers["EPI"][0]],
            base=[markers["BASE"][0]],
        )
    else:
        return markers


def dolfinx_ldrb(
    mesh: dolfinx.mesh.Mesh,
    fiber_space: str = "CG_1",
    ffun: dolfinx.mesh.MeshTags | None = None,
    markers: dict[str, list[int]] | dict[str, int] | None = None,
    alpha_endo_lv: float = 40,
    alpha_epi_lv: float = -50,
    alpha_endo_rv: float | None = None,
    alpha_epi_rv: float | None = None,
    alpha_endo_sept: float | None = None,
    alpha_epi_sept: float | None = None,
    beta_endo_lv: float = -65,
    beta_epi_lv: float = 25,
    beta_endo_rv: float | None = None,
    beta_epi_rv: float | None = None,
    beta_endo_sept: float | None = None,
    beta_epi_sept: float | None = None,
) -> io.LDRBOutput:
    r"""
    Create fiber, cross fibers and sheet directions

    Arguments
    ---------
    mesh : dolfinx.mesh.Mesh
        The mesh
    fiber_space : str
        A string on the form {family}_{degree} which
        determines for what space the fibers should be calculated for.
        If not provided, then a first order Lagrange space will be used,
        i.e Lagrange_1.
    ffun : dolfinx.mesh.MeshTags
        A facet function containing markers for the boundaries.
    markers : dict[str, int | list[int]] (optional)
        A dictionary with the markers for the
        different boundaries defined in the facet function
        or within the mesh itself.
        The following markers must be provided:
        'base', 'lv', 'epi, 'rv' (optional).
        If the markers are not provided the following default
        vales will be used: base = 10, rv = 20, lv = 30, epi = 40
    save_markers: bool
        If true save markings of the geometry. This is nice if you
        want to see that the LV, RV and Septum are marked correctly.
    angles : kwargs
        Keyword arguments with the fiber and sheet angles.
        It is possible to set different angles on the LV,
        RV and septum, however it either the RV or septum
        angles are not provided, then the angles on the LV
        will be used. The default values are taken from the
        original paper, namely

        .. math::

            \alpha_{\text{endo}} &= 40 \\
            \alpha_{\text{epi}} &= -50 \\
            \beta_{\text{endo}} &= -65 \\
            \beta_{\text{epi}} &= 25

        The following keyword arguments are possible:

        alpha_endo_lv : scalar
            Fiber angle at the LV endocardium.
        alpha_epi_lv : scalar
            Fiber angle at the LV epicardium.
        beta_endo_lv : scalar
            Sheet angle at the LV endocardium.
        beta_epi_lv : scalar
            Sheet angle at the LV epicardium.
        alpha_endo_rv : scalar
            Fiber angle at the RV endocardium.
        alpha_epi_rv : scalar
            Fiber angle at the RV epicardium.
        beta_endo_rv : scalar
            Sheet angle at the RV endocardium.
        beta_epi_rv : scalar
            Sheet angle at the RV epicardium.
        alpha_endo_sept : scalar
            Fiber angle at the septum endocardium.
        alpha_epi_sept : scalar
            Fiber angle at the septum epicardium.
        beta_endo_sept : scalar
            Sheet angle at the septum endocardium.
        beta_epi_sept : scalar
            Sheet angle at the septum epicardium.

    """
    # Solve the Laplace-Dirichlet problem
    processed_markers = transform_markers(process_markers(markers))

    logger.info("Calculating scalar fields")
    scalar_solutions = scalar_laplacians(
        mesh=mesh,
        markers=processed_markers,
        ffun=ffun,
    )

    # Create gradients
    logger.info("\nCalculating gradients")
    data, output = project_gradients(
        mesh=mesh,
        fiber_space=fiber_space,
        scalar_solutions=scalar_solutions,
    )

    dofs = dofs_from_function_space(mesh, fiber_space)
    marker_scalar = np.zeros_like(data["lv_scalar"])
    system = compute_fiber_sheet_system(
        dofs=dofs,
        marker_scalar=marker_scalar,
        alpha_endo_lv=alpha_endo_lv,
        alpha_epi_lv=alpha_epi_lv,
        alpha_endo_rv=alpha_endo_rv,
        alpha_epi_rv=alpha_epi_rv,
        alpha_endo_sept=alpha_endo_sept,
        alpha_epi_sept=alpha_epi_sept,
        beta_endo_lv=beta_endo_lv,
        beta_epi_lv=beta_epi_lv,
        beta_endo_rv=beta_endo_rv,
        beta_epi_rv=beta_epi_rv,
        beta_endo_sept=beta_endo_sept,
        beta_epi_sept=beta_epi_sept,
        **data,
    )  # type:ignore

    V = utils.space_from_string(fiber_space, mesh, dim=1)
    markers_fun = dolfinx.fem.Function(V)
    markers_fun.x.array[:] = marker_scalar

    Vv = utils.space_from_string(fiber_space, mesh, dim=3)
    f0 = array_to_function(Vv, system.fiber, "fiber")
    s0 = array_to_function(Vv, system.sheet, "sheet")
    n0 = array_to_function(Vv, system.sheet_normal, "sheet_normal")
    return io.LDRBOutput(
        fiber=f0,
        sheet=s0,
        sheet_normal=n0,
        markers_scalar=markers_fun,
        **output,
    )


def array_to_function(
    V: dolfinx.fem.FunctionSpace, array: np.ndarray, name
) -> dolfinx.fem.Function:
    f = dolfinx.fem.Function(V)
    f.x.array[:] = array
    f.name = name
    return f


def apex_to_base(
    mesh: dolfinx.mesh.Mesh,
    base_marker: list[int],
    ffun: dolfinx.mesh.MeshTags,
) -> dolfinx.fem.Function:
    """
    Find the apex coordinate and compute the laplace
    equation to find the apex to base solution

    Arguments
    ---------
    mesh : dolfin.Mesh
        The mesh
    base_marker : int
        The marker value for the basal facets
    ffun : dolfin.MeshFunctionSizet (optional)
        A facet function containing markers for the boundaries.
        If not provided, the markers stored within the mesh will
        be used.
    """
    # Find apex by solving a laplacian with base solution = 0
    # Create Base variational problem

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = v * dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0)) * ufl.dx

    base_facets = np.hstack([ffun.find(marker) for marker in base_marker])
    base_dofs = dolfinx.fem.locate_dofs_topological(V, 2, base_facets)
    one = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0))
    base_bc = dolfinx.fem.dirichletbc(one, base_dofs, V)

    bcs = [base_bc]

    problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "apex.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(uh)

    # Maybe we can use
    # uh.x.scatter_forward()
    # and uh.x.array.argmax() to find the apex_points ?

    local_max_val = uh.x.array.max()
    local_apex_coord = V.tabulate_dof_coordinates()[uh.x.array.argmax()]

    global_max, apex_coord = mesh.comm.allreduce(
        sendobj=(local_max_val, local_apex_coord),
        op=MPI.MAXLOC,
    )

    logger.info("  Apex coord: ({0:.2f}, {1:.2f}, {2:.2f})".format(*apex_coord))

    # Update rhs
    L = v * dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0)) * ufl.dx

    apex_points = dolfinx.mesh.locate_entities_boundary(
        mesh,
        0,
        lambda x: np.isclose(x[0], apex_coord[0])
        & np.isclose(x[1], apex_coord[1])
        & np.isclose(x[2], apex_coord[2]),
    )

    zero = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    apex_bc = dolfinx.fem.dirichletbc(zero, apex_points, V)

    # Solve the poisson equation
    bcs = [apex_bc, base_bc]
    L = v * zero * ufl.dx
    problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    apex = problem.solve()

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "apex_base.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(apex)

    return apex


def project_gradients(
    mesh: dolfinx.mesh.Mesh,
    scalar_solutions: dict[str, dolfinx.fem.Function],
    fiber_space: str = "P_1",
) -> tuple[dict[str, np.ndarray], dict[str, dolfinx.fem.Function]]:
    """
    Calculate the gradients using projections

    Arguments
    ---------
    mesh : dolfin.Mesh
        The mesh
    fiber_space : str
        A string on the form {family}_{degree} which
        determines for what space the fibers should be calculated for.
    scalar_solutions: dict
        A dictionary with the scalar solutions that you
        want to compute the gradients of.
    """
    Vv = utils.space_from_string(fiber_space, mesh, dim=3)
    V = utils.space_from_string(fiber_space, mesh, dim=1)

    output = {}
    data = {}
    for case, scalar_solution in scalar_solutions.items():
        output[case] = scalar_solution
        if case != "lv_rv":
            grad_expr = dolfinx.fem.Expression(
                ufl.grad(scalar_solution), Vv.element.interpolation_points()
            )
            f = dolfinx.fem.Function(Vv)
            f.interpolate(grad_expr)

            # Add gradient data
            data[case + "_gradient"] = f.x.array
            output[case + "_gradient"] = f

        # Add scalar data
        if case != "apex":
            f = dolfinx.fem.Function(V)
            expr = dolfinx.fem.Expression(scalar_solution, V.element.interpolation_points())
            f.interpolate(expr)
            data[case + "_scalar"] = f.x.array
            output[case + "_scalar"] = f

    # Return data
    return data, output


def scalar_laplacians(
    mesh: dolfinx.mesh.Mesh,
    markers: dict[str, list[int]],
    ffun: dolfinx.mesh.MeshTags,
) -> dict[str, dolfinx.fem.Function]:
    """
    Calculate the laplacians

    Arguments
    ---------
    mesh : dolfin.Mesh
       A dolfin mesh
    markers : dict (optional)
        A dictionary with the markers for the
        different boundaries defined in the facet function
        or within the mesh itself.
        The following markers must be provided:
        'base', 'lv', 'epi, 'rv' (optional).
        If the markers are not provided the following default
        vales will be used: base = 10, rv = 20, lv = 30, epi = 40.

    """

    if not isinstance(mesh, dolfinx.mesh.Mesh):
        raise TypeError("Expected a dolfin.Mesh as the mesh argument.")

    # Boundary markers, solutions and cases
    cases, boundaries = find_cases_and_boundaries(markers)
    markers_str = "\n".join(["{}: {}".format(k, v) for k, v in markers.items()])
    logger.info(
        ("Compute scalar laplacian solutions with the markers: \n" "{}").format(
            markers_str,
        ),
    )

    # check_boundaries_are_marked(
    #     mesh=mesh,
    #     ffun=ffun,
    #     markers=markers,
    #     boundaries=boundaries,
    # )

    # Compte the apex to base solutions
    num_vertices = mesh.topology.index_map(0).size_global
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_global

    logger.info("  Num vertices: {0}".format(num_vertices))
    logger.info("  Num cells: {0}".format(num_cells))

    solutions: dict[str, dolfinx.fem.Function] = {}
    apex = apex_to_base(mesh, markers["base"], ffun)
    V = apex.function_space
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx

    zero = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    one = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0))
    L = v * zero * ufl.dx
    solutions = dict((what, dolfinx.fem.Function(V)) for what in cases)

    solutions["apex"] = apex

    for case in cases:
        endo_markers = markers[case]
        endo_facets = np.hstack([ffun.find(marker) for marker in endo_markers])
        endo_dofs = dolfinx.fem.locate_dofs_topological(V, 2, endo_facets)
        endo_bc = dolfinx.fem.dirichletbc(one, endo_dofs, V)

        epi_markers = []
        for what in cases:
            if what != case:
                epi_markers.extend(markers[what])
        epi_facets = np.hstack([ffun.find(marker) for marker in epi_markers])
        epi_dofs = dolfinx.fem.locate_dofs_topological(V, 2, epi_facets)
        epi_bc = dolfinx.fem.dirichletbc(zero, epi_dofs, V)

        bcs = [endo_bc, epi_bc]

        problem = LinearProblem(
            a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        uh = problem.solve()
        solutions[case] = uh

        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{case}.xdmf", "w") as file:
            file.write_mesh(mesh)
            file.write_function(uh)

    if "rv" in cases:
        endo_markers = markers["lv"]
        endo_facets = np.hstack([ffun.find(marker) for marker in endo_markers])
        endo_dofs = dolfinx.fem.locate_dofs_topological(V, 2, endo_facets)
        endo_bc = dolfinx.fem.dirichletbc(one, endo_dofs, V)

        epi_markers = markers["rv"]
        epi_facets = np.hstack([ffun.find(marker) for marker in epi_markers])
        epi_dofs = dolfinx.fem.locate_dofs_topological(V, 2, epi_facets)
        epi_bc = dolfinx.fem.dirichletbc(zero, epi_dofs, V)

        bcs = [endo_bc, epi_bc]

        problem = LinearProblem(
            a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        uh = problem.solve()
        solutions["lv_rv"] = uh

        # with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "lv_rv.xdmf", "w") as file:
        #     file.write_mesh(mesh)
        #     file.write_function(uh)

    return solutions


def process_markers(
    markers: dict[str, list[int]] | dict[str, int] | None,
) -> dict[str, list[int]]:
    if markers is None:
        return utils.default_markers()

    markers_to_lists: dict[str, list[int]] = {}
    for name, values in markers.items():
        if not isinstance(values, list):
            markers_to_lists[name] = [values]
        else:
            assert isinstance(values, list)
            markers_to_lists[name] = values

    return markers_to_lists


def find_cases_and_boundaries(
    markers: dict[str, list[int]],
) -> tuple[list[str], list[str]]:
    potential_cases = {"rv", "lv", "epi"}
    potential_boundaries = potential_cases | {"base", "mv", "av"}

    cases = []
    boundaries = []

    for marker in markers:
        msg = f"Unknown marker {marker}. Expected one of {potential_boundaries}"
        if marker not in potential_boundaries:
            logging.warning(msg)
        if marker in potential_boundaries:
            boundaries.append(marker)
        if marker in potential_cases:
            cases.append(marker)

    return cases, boundaries


# def check_boundaries_are_marked(
#     mesh: dolfinx.mesh.Mesh,
#     ffun: dolfinx.mesh.MeshTags,
#     markers: Dict[str, List[int]],
#     boundaries: List[str],
# ) -> None:
#     # Check that all boundary faces are marked
#     breakpoint()
#     num_boundary_facets = df.BoundaryMesh(mesh, "exterior").num_cells()

#     if num_boundary_facets != sum(
#         [np.sum(ffun.array() == idx) for marker in markers.values() for idx in marker],
#     ):
#         raise RuntimeError(
#             (
#                 "Not all boundary faces are marked correctly. Make sure all "
#                 "boundary facets are marked as: {}"
#                 ""
#             ).format(", ".join(["{} = {}".format(k, v) for k, v in markers.items()])),
#         )
