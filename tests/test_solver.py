from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

import cardiac_geometries
import ldrb

LU = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
STAGES = ("ldrb: Laplace solves", "ldrb: gradients", "ldrb: frames")


@pytest.fixture(scope="module")
def biv_geometry(tmpdir_factory):
    comm = MPI.COMM_WORLD
    outdir = comm.bcast(tmpdir_factory.mktemp("biv_solver"), root=0)
    return cardiac_geometries.mesh.biv_ellipsoid(outdir=outdir)


def timer_count(name):
    try:
        return dolfinx.common.timing(name)[0]
    except RuntimeError:  # never started
        return 0


def owned_vectors(f):
    n = f.function_space.dofmap.index_map.size_local
    return f.x.array[: 3 * n].reshape(-1, 3)


def max_angle_deg(f, g, comm):
    u, v = owned_vectors(f), owned_vectors(g)
    cos = np.abs(np.sum(u * v, axis=1)) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    local = float(np.degrees(np.arccos(np.clip(cos, 0, 1))).max(initial=0.0))
    return comm.allreduce(local, op=MPI.MAX)


def test_default_solver_is_iterative_and_checks_convergence():
    options = ldrb.default_petsc_options()
    assert options["ksp_type"] == "cg"
    assert options["pc_type"] == "gamg"
    assert options["ksp_error_if_not_converged"] is True


def test_default_fibers_match_lu(biv_geometry):
    geo = biv_geometry
    common = dict(mesh=geo.mesh, ffun=geo.ffun, markers=geo.markers, fiber_space="P_1")
    default = ldrb.dolfinx_ldrb(**common)
    lu = ldrb.dolfinx_ldrb(**common, petsc_options=LU)
    assert max_angle_deg(default.f0, lu.f0, geo.mesh.comm) < 1e-3


def test_stage_timers_are_recorded(biv_geometry):
    geo = biv_geometry
    before = [timer_count(name) for name in STAGES]
    ldrb.dolfinx_ldrb(mesh=geo.mesh, ffun=geo.ffun, markers=geo.markers, fiber_space="P_1")
    after = [timer_count(name) for name in STAGES]
    assert [b - a for a, b in zip(before, after)] == [1, 1, 1]
