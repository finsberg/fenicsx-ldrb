r"""Rule-based fibers for biventricular geometries with outflow tracts.

This is the outflow tract extended rule-based method (OT-RBM) from

    Doste R, Soto-Iglesias D, Bernardino G, et al. A rule-based method to
    model myocardial fiber orientation in cardiac biventricular geometries
    with outflow tracts. Int J Numer Meth Biomed Engng. 2019;35:e3185.
    https://doi.org/10.1002/cnm.3185

Unlike the method of Bayer et al. in :func:`ldrb.dolfinx_ldrb`, the
longitudinal direction is defined separately in each ventricle, as a blend of
the direction from the apex to the inflow valve (mitral or tricuspid) and the
direction from the apex to the outflow valve (aortic or pulmonary). The blend
also interpolates the fiber angles between the values in the ventricle and in
the outflow tract.

The angles follow the convention of :func:`ldrb.dolfinx_ldrb`: the
longitudinal direction points from the apex towards the valves, the
transmural direction points from the endocardium to the epicardium in both
ventricles, and :math:`\alpha` is the angle from the circumferential
direction towards the longitudinal direction. In this convention the values
in the paper are, with the opposite sign of the paper's LV values,

.. math::

    \alpha_{\text{endo}} = 60, \alpha_{\text{epi}} = -60 \quad \text{(LV)}, \qquad
    \alpha_{\text{endo}} = 90, \alpha_{\text{epi}} = -25 \quad \text{(RV)}, \qquad
    \alpha_{\text{endo}} = 90, \alpha_{\text{epi}} = 0 \quad \text{(outflow tracts)}.

The transverse angle :math:`\beta` rotates the fiber about the longitudinal
direction, as in the paper, and its values here also have the opposite sign
of the paper's. The septal interpolation of the paper (Section 2.3.2) is not
implemented.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from mpi4py import MPI

import dolfinx
import numba
import numpy as np
import ufl
from dolfinx.fem.petsc import LinearProblem
from packaging.version import Version

from . import io, utils
from .calculus import axis, normalize
from .ldrb import (
    array_to_function,
    default_petsc_options,
    dofs_from_function_space,
    process_markers,
    project_gradients,
)

_dolfinx_version = Version(dolfinx.__version__)

logger = logging.getLogger(__name__)


def _facets(ffun: dolfinx.mesh.MeshTags, markers: list[int]) -> np.ndarray:
    return np.hstack([ffun.find(marker) for marker in markers]).astype(np.int32)


def _facet_points(mesh: dolfinx.mesh.Mesh, facets: np.ndarray) -> np.ndarray:
    """The coordinates of the geometry nodes of `facets` (with duplicates)."""
    fdim = mesh.topology.dim - 1
    nodes = dolfinx.mesh.entities_to_geometry(mesh, fdim, facets)
    return mesh.geometry.x[nodes.reshape(-1)]


def find_apex(
    mesh: dolfinx.mesh.Mesh,
    ffun: dolfinx.mesh.MeshTags,
    endo_markers: list[int],
    valve_markers: list[int],
) -> np.ndarray:
    """Find the apex as the point on the endocardium farthest from the
    centre of the valve.

    Arguments
    ---------
    mesh : dolfinx.mesh.Mesh
        The mesh
    ffun : dolfinx.mesh.MeshTags
        A facet function containing markers for the boundaries.
    endo_markers : list[int]
        The markers of the endocardium of the ventricle
    valve_markers : list[int]
        The markers of the inflow valve of the ventricle (mitral or tricuspid)

    Returns
    -------
    np.ndarray
        The coordinates of the apex
    """
    comm = mesh.comm
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    valve_points = _facet_points(mesh, _facets(ffun, valve_markers))
    total = comm.allreduce(valve_points.sum(axis=0), op=MPI.SUM)
    count = comm.allreduce(len(valve_points), op=MPI.SUM)
    if count == 0:
        raise ValueError(f"No facets marked with {valve_markers}")
    valve_centre = total / count

    endo_points = _facet_points(mesh, _facets(ffun, endo_markers))
    if len(endo_points) > 0:
        distance = np.linalg.norm(endo_points - valve_centre, axis=1)
        local = (float(distance.max()), comm.rank)
    else:
        local = (-np.inf, comm.rank)
    max_distance, owner = comm.allreduce(local, op=MPI.MAXLOC)
    if max_distance == -np.inf:
        raise ValueError(f"No facets marked with {endo_markers}")
    apex = endo_points[distance.argmax()] if comm.rank == owner else None
    return np.asarray(comm.bcast(apex, root=owner), dtype=np.float64)


def _solve_laplace(
    V: dolfinx.fem.FunctionSpace,
    conditions: list[tuple[np.ndarray, float]],
    petsc_options: dict[str, Any],
    name: str,
) -> dolfinx.fem.Function:
    """Solve the Laplace equation with the Dirichlet conditions given as a list
    of (dofs, value). The rest of the boundary has zero Neumann conditions."""
    mesh = V.mesh
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    zero = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    L = v * zero * ufl.dx

    bcs = [
        dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(value), dofs, V)
        for dofs, value in conditions
    ]

    kwargs = {}
    if _dolfinx_version >= Version("0.10"):
        kwargs["petsc_options_prefix"] = f"ldrb_doste_{name}"

    problem = LinearProblem(  # type: ignore[call-overload]
        a,
        L,
        bcs=bcs,
        petsc_options=petsc_options,
        **kwargs,  # type: ignore[arg-type]
    )
    uh = problem.solve()
    assert isinstance(uh, dolfinx.fem.Function)
    uh.name = name
    return uh


def scalar_laplacians(
    mesh: dolfinx.mesh.Mesh,
    ffun: dolfinx.mesh.MeshTags,
    markers: dict[str, list[int]],
    apex_lv: Sequence[float] | np.ndarray,
    apex_rv: Sequence[float] | np.ndarray | None = None,
    petsc_options: dict[str, Any] | None = None,
) -> dict[str, dolfinx.fem.Function]:
    """Solve the Laplace problems of the method, with Lagrange elements of the
    same degree as the geometry.

    Returns the transmural solution ``phi`` (-2 on the LV endocardium, 1 on
    the RV endocardium and 0 on the epicardium), and for each ventricle (``lv``
    and, if there is an RV, ``rv``) the solutions ``psi_{ventricle}_basal``
    and ``psi_{ventricle}_ot`` (0 at the apex and 1 at the inflow and outflow
    valve respectively) and the interpolation function ``w_{ventricle}`` (1 at
    the apex and the inflow valve, 0 at the outflow valve). For biventricular
    geometries it also returns ``lv_rv`` (1 on the LV endocardium and 0 on
    the RV endocardium), which decides the ventricle where ``phi`` is zero,
    i.e. on the epicardium.
    """
    if petsc_options is None:
        petsc_options = default_petsc_options()

    V = utils.laplace_space(mesh)
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)

    def dofs(name: str) -> np.ndarray:
        return dolfinx.fem.locate_dofs_topological(V, fdim, _facets(ffun, markers[name]))

    transmural = [(dofs("lv"), -2.0), (dofs("epi"), 0.0)]
    ventricles = {"lv": ("mv", "av", apex_lv)}
    if "rv" in markers:
        assert apex_rv is not None
        transmural.append((dofs("rv"), 1.0))
        ventricles["rv"] = ("tv", "pv", apex_rv)

    solutions = {"phi": _solve_laplace(V, transmural, petsc_options, "phi")}
    if "rv" in markers:
        solutions["lv_rv"] = _solve_laplace(
            V, [(dofs("lv"), 1.0), (dofs("rv"), 0.0)], petsc_options, "lv_rv"
        )
    for ventricle, (inflow, outflow, apex) in ventricles.items():
        apex_dofs = utils.closest_dofs(V, apex)
        inflow_dofs = dofs(inflow)
        outflow_dofs = dofs(outflow)
        solutions[f"psi_{ventricle}_basal"] = _solve_laplace(
            V,
            [(apex_dofs, 0.0), (inflow_dofs, 1.0)],
            petsc_options,
            f"psi_{ventricle}_basal",
        )
        solutions[f"psi_{ventricle}_ot"] = _solve_laplace(
            V,
            [(apex_dofs, 0.0), (outflow_dofs, 1.0)],
            petsc_options,
            f"psi_{ventricle}_ot",
        )
        solutions[f"w_{ventricle}"] = _solve_laplace(
            V,
            [(apex_dofs, 1.0), (inflow_dofs, 1.0), (outflow_dofs, 0.0)],
            petsc_options,
            f"w_{ventricle}",
        )
    return solutions


@numba.njit
def _rotate(u: np.ndarray, k: np.ndarray, angle: float) -> np.ndarray:
    """Rotate `u` counterclockwise about the unit vector `k` by `angle` (degrees)."""
    c = np.cos(np.radians(angle))
    s = np.sin(np.radians(angle))
    return c * u + s * np.cross(k, u) + (1 - c) * k.dot(u) * k


@numba.njit
def compute_fiber_sheet_system(
    f0,
    s0,
    n0,
    marker_scalar,
    xdofs,
    ydofs,
    zdofs,
    sdofs,
    phi_scalar,
    phi_gradient,
    lv_rv_scalar,
    w_lv_scalar,
    psi_lv_basal_gradient,
    psi_lv_ot_gradient,
    w_rv_scalar,
    psi_rv_basal_gradient,
    psi_rv_ot_gradient,
    angles,
    has_rv,
):
    """Compute the fiber, sheet and sheet-normal directions at each dof.

    `angles` has shape (2, 2, 4): ventricle (LV, RV), region (ventricle,
    outflow tract) and angle (alpha endo, alpha epi, beta endo, beta epi).
    """
    grad_phi = np.zeros(3)
    grad_basal = np.zeros(3)
    grad_ot = np.zeros(3)

    for i in range(len(xdofs)):
        phi = phi_scalar[sdofs[i]]
        grad_phi[0] = phi_gradient[xdofs[i]]
        grad_phi[1] = phi_gradient[ydofs[i]]
        grad_phi[2] = phi_gradient[zdofs[i]]

        if has_rv and abs(phi) < 1e-10:
            # On the epicardium, where phi is zero in both ventricles
            in_lv = lv_rv_scalar[sdofs[i]] >= 0.5
        else:
            in_lv = phi < 0 or not has_rv

        if in_lv:
            # LV, where phi goes from -2 on the endocardium to 0
            ventricle = 0
            depth = 1 + phi / 2
            w = w_lv_scalar[sdofs[i]]
            grad_basal[0] = psi_lv_basal_gradient[xdofs[i]]
            grad_basal[1] = psi_lv_basal_gradient[ydofs[i]]
            grad_basal[2] = psi_lv_basal_gradient[zdofs[i]]
            grad_ot[0] = psi_lv_ot_gradient[xdofs[i]]
            grad_ot[1] = psi_lv_ot_gradient[ydofs[i]]
            grad_ot[2] = psi_lv_ot_gradient[zdofs[i]]
            transmural = grad_phi.copy()
        else:
            # RV, where phi goes from 1 on the endocardium to 0
            ventricle = 1
            depth = 1 - phi
            w = w_rv_scalar[sdofs[i]]
            grad_basal[0] = psi_rv_basal_gradient[xdofs[i]]
            grad_basal[1] = psi_rv_basal_gradient[ydofs[i]]
            grad_basal[2] = psi_rv_basal_gradient[zdofs[i]]
            grad_ot[0] = psi_rv_ot_gradient[xdofs[i]]
            grad_ot[1] = psi_rv_ot_gradient[ydofs[i]]
            grad_ot[2] = psi_rv_ot_gradient[zdofs[i]]
            transmural = -grad_phi

        depth = min(max(depth, 0.0), 1.0)
        w = min(max(w, 0.0), 1.0)
        marker_scalar[sdofs[i]] = ventricle + 1

        # Angles at the endocardium and epicardium, interpolated between the
        # ventricle (w = 1) and the outflow tract (w = 0), and then transmurally
        a = w * angles[ventricle, 0] + (1 - w) * angles[ventricle, 1]
        alpha = a[0] * (1 - depth) + a[1] * depth
        beta = a[2] * (1 - depth) + a[3] * depth

        longitudinal = w * grad_basal + (1 - w) * grad_ot
        # Columns: circumferential, longitudinal and transmural directions
        Q = axis(longitudinal, transmural)
        e_c = Q[:, 0].copy()
        e_l = Q[:, 1].copy()
        e_t = Q[:, 2].copy()

        fiber = np.cos(np.radians(alpha)) * e_c + np.sin(np.radians(alpha)) * e_l
        fiber = normalize(_rotate(fiber, e_l, beta))
        sheet = normalize(e_t - e_t.dot(fiber) * fiber)
        normal = np.cross(fiber, sheet)

        f0[xdofs[i]] = fiber[0]
        f0[ydofs[i]] = fiber[1]
        f0[zdofs[i]] = fiber[2]

        s0[xdofs[i]] = sheet[0]
        s0[ydofs[i]] = sheet[1]
        s0[zdofs[i]] = sheet[2]

        n0[xdofs[i]] = normal[0]
        n0[ydofs[i]] = normal[1]
        n0[zdofs[i]] = normal[2]


def dolfinx_doste(
    mesh: dolfinx.mesh.Mesh,
    ffun: dolfinx.mesh.MeshTags,
    markers: dict[str, list[int]] | dict[str, tuple[int, ...]] | dict[str, int],
    fiber_space: str = "P_1",
    apex_lv: Sequence[float] | None = None,
    apex_rv: Sequence[float] | None = None,
    alpha_endo_lv: float = 60,
    alpha_epi_lv: float = -60,
    beta_endo_lv: float = 20,
    beta_epi_lv: float = -20,
    alpha_endo_rv: float = 90,
    alpha_epi_rv: float = -25,
    beta_endo_rv: float = 0,
    beta_epi_rv: float = -20,
    alpha_endo_ot: float = 90,
    alpha_epi_ot: float = 0,
    beta_endo_ot: float = 0,
    beta_epi_ot: float = 0,
    petsc_options: dict[str, Any] | None = None,
) -> io.DosteOutput:
    r"""
    Create fiber, sheet and sheet-normal directions with the rule-based
    method for geometries with outflow tracts of Doste et al. (2019).

    See :mod:`ldrb.doste` for the method and the convention of the angles.

    Arguments
    ---------
    mesh : dolfinx.mesh.Mesh
        The mesh
    ffun : dolfinx.mesh.MeshTags
        A facet function containing markers for the boundaries.
    markers : dict[str, int | list[int]]
        The markers of the boundaries in `ffun`: 'lv' (LV endocardium),
        'epi' (epicardium), 'mv' (mitral valve) and 'av' (aortic valve), and
        for biventricular geometries also 'rv' (RV endocardium), 'tv'
        (tricuspid valve) and 'pv' (pulmonary valve). Other boundaries get
        zero Neumann conditions.
    fiber_space : str
        A string on the form {family}_{degree} which determines the space of
        the fibers, by default "P_1".
    apex_lv : Sequence[float] (optional)
        The coordinates of the LV endocardial apex. The closest dof is used.
        If not provided, the point on the LV endocardium farthest from the
        centre of the mitral valve is used, see :func:`find_apex`.
    apex_rv : Sequence[float] (optional)
        The coordinates of the RV endocardial apex. If not provided, the point
        on the RV endocardium farthest from the centre of the tricuspid valve
        is used.
    alpha_endo_lv, alpha_epi_lv, beta_endo_lv, beta_epi_lv : float
        Fiber and transverse angles at the LV endocardium and epicardium.
    alpha_endo_rv, alpha_epi_rv, beta_endo_rv, beta_epi_rv : float
        Fiber and transverse angles at the RV endocardium and epicardium.
    alpha_endo_ot, alpha_epi_ot, beta_endo_ot, beta_epi_ot : float
        Fiber and transverse angles at the endocardium and epicardium of the
        outflow tracts.
    petsc_options : dict (optional)
        A dictionary with petsc options to pass to the linear solvers.
        If not provided, the default options will be used, which are
        ksp_type = preonly and pc_type = lu.
    """
    processed_markers = process_markers(markers)
    required = ["lv", "epi", "mv", "av"]
    has_rv = "rv" in processed_markers
    if has_rv:
        required += ["rv", "tv", "pv"]
    missing = [name for name in required if name not in processed_markers]
    if missing:
        raise ValueError(f"Missing markers {missing}, got {list(processed_markers)}")

    if apex_lv is None:
        lv_apex = find_apex(mesh, ffun, processed_markers["lv"], processed_markers["mv"])
    else:
        lv_apex = np.asarray(apex_lv, dtype=np.float64)
    logger.info(f"LV apex: {lv_apex}")
    rv_apex = None
    if has_rv:
        if apex_rv is None:
            rv_apex = find_apex(mesh, ffun, processed_markers["rv"], processed_markers["tv"])
        else:
            rv_apex = np.asarray(apex_rv, dtype=np.float64)
        logger.info(f"RV apex: {rv_apex}")

    logger.info("Calculating scalar fields")
    solutions = scalar_laplacians(
        mesh=mesh,
        ffun=ffun,
        markers=processed_markers,
        apex_lv=lv_apex,
        apex_rv=rv_apex,
        petsc_options=petsc_options,
    )

    logger.info("Calculating gradients")
    data, _ = project_gradients(mesh=mesh, scalar_solutions=solutions, fiber_space=fiber_space)
    if not has_rv:
        for name in ["lv_rv_scalar", "w_rv_scalar", "psi_rv_basal_gradient", "psi_rv_ot_gradient"]:
            reference = "w_lv_scalar" if name.endswith("scalar") else "psi_lv_basal_gradient"
            data[name] = np.zeros_like(data[reference])

    angles = np.array(
        [
            [
                [alpha_endo_lv, alpha_epi_lv, beta_endo_lv, beta_epi_lv],
                [alpha_endo_ot, alpha_epi_ot, beta_endo_ot, beta_epi_ot],
            ],
            [
                [alpha_endo_rv, alpha_epi_rv, beta_endo_rv, beta_epi_rv],
                [alpha_endo_ot, alpha_epi_ot, beta_endo_ot, beta_epi_ot],
            ],
        ],
        dtype=np.float64,
    )

    dofs = dofs_from_function_space(mesh, fiber_space)
    f0 = np.zeros_like(data["phi_gradient"])
    s0 = np.zeros_like(f0)
    n0 = np.zeros_like(f0)
    marker_scalar = np.zeros_like(data["phi_scalar"])
    compute_fiber_sheet_system(
        f0,
        s0,
        n0,
        marker_scalar,
        dofs[:, 0],
        dofs[:, 1],
        dofs[:, 2],
        dofs[:, 3],
        data["phi_scalar"],
        data["phi_gradient"],
        data["lv_rv_scalar"],
        data["w_lv_scalar"],
        data["psi_lv_basal_gradient"],
        data["psi_lv_ot_gradient"],
        data["w_rv_scalar"],
        data["psi_rv_basal_gradient"],
        data["psi_rv_ot_gradient"],
        angles,
        has_rv,
    )

    Vv = utils.space_from_string(fiber_space, mesh, dim=3)
    V = utils.space_from_string(fiber_space, mesh, dim=1)
    functions = {
        "f0": array_to_function(Vv, f0, "f0"),
        "s0": array_to_function(Vv, s0, "s0"),
        "n0": array_to_function(Vv, n0, "n0"),
        "markers_scalar": array_to_function(V, marker_scalar, "markers_scalar"),
    }
    # Only the owned dofs are computed above
    for f in functions.values():
        f.x.scatter_forward()

    return io.DosteOutput(**functions, **solutions)
