import math

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import scifem
import ufl

import cardiac_geometries
import ldrb

tol = 1e-12


@pytest.fixture(scope="session", params=["marker_is_int", "marker_is_list"])
def lv_geometry_markers(request, tmpdir_factory):
    comm = MPI.COMM_WORLD
    outdir = comm.bcast(tmpdir_factory.mktemp("lv"), root=0)
    geo = cardiac_geometries.mesh.lv_ellipsoid(outdir=outdir, comm=comm)
    if request.param == "marker_is_int":
        markers = {
            "base": geo.markers["BASE"][0],
            "epi": geo.markers["EPI"][0],
            "lv": geo.markers["ENDO"][0],
        }
    else:
        markers = {
            "base": [geo.markers["BASE"][0]],
            "epi": [geo.markers["EPI"][0]],
            "lv": [geo.markers["ENDO"][0]],
        }

    geo.markers.update(markers)
    return geo


@pytest.fixture(scope="session")
def lv_geometry(tmpdir_factory):
    comm = MPI.COMM_WORLD
    outdir = comm.bcast(tmpdir_factory.mktemp("lv"), root=0)
    geo = cardiac_geometries.mesh.lv_ellipsoid(outdir=outdir, comm=comm)
    geo.markers.update(
        {
            "base": [geo.markers["BASE"][0]],
            "epi": [geo.markers["EPI"][0]],
            "lv": [geo.markers["ENDO"][0]],
        }
    )

    return geo


@pytest.fixture(scope="session")
def biv_geometry(tmpdir_factory):
    comm = MPI.COMM_WORLD
    outdir = comm.bcast(tmpdir_factory.mktemp("biv"), root=0)
    geo = cardiac_geometries.mesh.biv_ellipsoid(outdir=outdir)
    return geo


def norm(v):
    return np.linalg.norm(v)


def test_scalar_laplacians():
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_cube(comm, nx=2, ny=2, nz=2)
    endo_facets = dolfinx.mesh.locate_entities_boundary(mesh, 2, lambda x: np.isclose(x[0], 0.0))
    epi_facets = dolfinx.mesh.locate_entities_boundary(mesh, 2, lambda x: np.isclose(x[0], 1.0))
    base_facets = dolfinx.mesh.locate_entities_boundary(mesh, 2, lambda x: np.isclose(x[1], 0.0))
    endo_marker = 10
    epi_marker = 20
    base_marker = 30
    tags = dolfinx.mesh.meshtags(
        mesh,
        2,
        np.hstack([endo_facets, epi_facets, base_facets]),
        np.hstack(
            [
                np.full_like(endo_facets, endo_marker),
                np.full_like(epi_facets, epi_marker),
                np.full_like(base_facets, base_marker),
            ],
            dtype=np.int32,
        ),
    )

    solutions = ldrb.ldrb.scalar_laplacians(
        mesh=mesh,
        ffun=tags,
        markers=dict(lv=[endo_marker], epi=[epi_marker], base=[base_marker]),
    )
    assert len(solutions) == 3
    # Check apex to base - apex will be located at (1, 1, 1)
    assert np.isclose(scifem.evaluate_function(solutions["apex"], np.array([[1.0, 1.0, 1.0]])), 0.0)
    assert np.isclose(scifem.evaluate_function(solutions["apex"], np.array([[0.0, 0.0, 0.0]])), 1.0)

    # Check LV
    assert np.isclose(scifem.evaluate_function(solutions["lv"], np.array([[0.0, 0.0, 0.0]])), 1.0)
    assert np.isclose(scifem.evaluate_function(solutions["lv"], np.array([[0.5, 0.5, 0.5]])), 0.5)
    assert np.isclose(scifem.evaluate_function(solutions["lv"], np.array([[0.5, 0.1, 0.1]])), 0.5)
    assert np.isclose(scifem.evaluate_function(solutions["lv"], np.array([[1.0, 1.0, 1.0]])), 0.0)

    # Check EPI (which should be opposite of LV)
    assert np.isclose(scifem.evaluate_function(solutions["epi"], np.array([[0.0, 0.0, 0.0]])), 0.0)
    assert np.isclose(scifem.evaluate_function(solutions["epi"], np.array([[0.5, 0.5, 0.5]])), 0.5)
    assert np.isclose(scifem.evaluate_function(solutions["epi"], np.array([[0.5, 0.1, 0.1]])), 0.5)
    assert np.isclose(scifem.evaluate_function(solutions["epi"], np.array([[1.0, 1.0, 1.0]])), 1.0)


def test_apex_to_base(lv_geometry):
    apex = ldrb.ldrb.apex_to_base(lv_geometry.mesh, lv_geometry.markers["base"], lv_geometry.ffun)
    assert np.isclose(lv_geometry.mesh.comm.allreduce(apex.x.array.max(), op=MPI.MAX), 1.0)
    assert np.isclose(lv_geometry.mesh.comm.allreduce(apex.x.array.min(), op=MPI.MIN), 0.0)

    # Base is located at x=5.0
    x_base = np.array([[5.0, 8.0, 0.0]])
    # Value at base should be 1.0
    assert np.isclose(scifem.evaluate_function(apex, x_base), 1.0)

    # Apex is located at x=-20.0
    x_apex = np.array([[-20.0, 0.0, 0.0]])
    # Value at apex should be 0.0
    assert np.isclose(scifem.evaluate_function(apex, x_apex), 0.0)


@pytest.mark.parametrize("fiber_space", ["P_1", "P_2", "dP_0", "dP_1", "Q_1", "Q_2"])
def test_lv_regression(lv_geometry, fiber_space):
    ldrb.dolfinx_ldrb(
        mesh=lv_geometry.mesh,
        ffun=lv_geometry.ffun,
        markers=lv_geometry.markers,
        fiber_space=fiber_space,
    )


@pytest.mark.parametrize(
    "fiber_space",
    [
        "P_1",
        "P_2",
        "dP_0",
        "dP_1",
        "Q_1",
        "Q_2",
    ],
)
def test_biv_regression(biv_geometry, fiber_space):
    ldrb.dolfinx_ldrb(
        mesh=biv_geometry.mesh,
        ffun=biv_geometry.ffun,
        markers=biv_geometry.markers,
        fiber_space=fiber_space,
    )


def degree(f: dolfinx.fem.Function) -> int:
    return f.function_space.element.basix_element.degree


@pytest.fixture(scope="session", params=[2, 3])
def lv_geometry_higher_order(request, tmpdir_factory):
    comm = MPI.COMM_WORLD
    outdir = comm.bcast(tmpdir_factory.mktemp(f"lv_order{request.param}"), root=0)
    geo = cardiac_geometries.mesh.lv_ellipsoid(
        outdir=outdir, comm=comm, psize_ref=6.0, order=request.param
    )
    geo.markers.update(
        {
            "base": [geo.markers["BASE"][0]],
            "epi": [geo.markers["EPI"][0]],
            "lv": [geo.markers["ENDO"][0]],
        }
    )
    return geo


def test_apex_to_base_higher_order(lv_geometry_higher_order):
    geo = lv_geometry_higher_order
    apex = ldrb.ldrb.apex_to_base(geo.mesh, geo.markers["base"], geo.ffun)
    assert degree(apex) == geo.mesh.ufl_domain().ufl_coordinate_element().degree
    assert np.isclose(geo.mesh.comm.allreduce(apex.x.array.max(), op=MPI.MAX), 1.0)
    assert np.isclose(geo.mesh.comm.allreduce(apex.x.array.min(), op=MPI.MIN), 0.0)
    assert np.isclose(scifem.evaluate_function(apex, np.array([[5.0, 8.0, 0.0]])), 1.0)
    assert np.isclose(scifem.evaluate_function(apex, np.array([[-20.0, 0.0, 0.0]])), 0.0)


def test_scalar_laplacians_are_as_accurate_as_the_geometry(tmp_path):
    # In a spherical shell cut through the centre the transmural solution is
    # u = (1 / r - 1 / R_epi) / (1 / R_endo - 1 / R_epi)
    comm = MPI.COMM_WORLD
    errors = []
    for order in [1, 2, 3]:
        geo = cardiac_geometries.mesh.lv_ellipsoid(
            outdir=comm.bcast(tmp_path / f"order{order}", root=0),
            comm=comm,
            r_short_endo=7.0,
            r_long_endo=7.0,
            r_short_epi=10.0,
            r_long_epi=10.0,
            mu_base_endo=-math.pi / 2,
            mu_base_epi=-math.pi / 2,
            order=order,
        )
        m = geo.markers
        solutions = ldrb.ldrb.scalar_laplacians(
            mesh=geo.mesh,
            markers={"lv": [m["ENDO"][0]], "epi": [m["EPI"][0]], "base": [m["BASE"][0]]},
            ffun=geo.ffun,
        )
        assert all(degree(f) == order for f in solutions.values())
        x = ufl.SpatialCoordinate(geo.mesh)
        exact = (1 / ufl.sqrt(ufl.dot(x, x)) - 1 / 10) / (1 / 7 - 1 / 10)
        error = dolfinx.fem.assemble_scalar(
            dolfinx.fem.form((solutions["lv"] - exact) ** 2 * ufl.dx)
        )
        errors.append(math.sqrt(comm.allreduce(error)))
    assert errors[1] < errors[0] / 5
    assert errors[2] < errors[1] / 5


@pytest.mark.parametrize("fiber_space", ["P_1", "P_2"])
def test_biv_higher_order(tmpdir_factory, fiber_space):
    comm = MPI.COMM_WORLD
    outdir = comm.bcast(tmpdir_factory.mktemp("biv_order2"), root=0)
    geo = cardiac_geometries.mesh.biv_ellipsoid(outdir=outdir, comm=comm, order=2)
    system = ldrb.dolfinx_ldrb(
        mesh=geo.mesh, ffun=geo.ffun, markers=geo.markers, fiber_space=fiber_space
    )
    assert degree(system.lv) == 2
    n = system.f0.function_space.dofmap.index_map.size_local
    f0 = system.f0.x.array[: 3 * n].reshape(-1, 3)
    s0 = system.s0.x.array[: 3 * n].reshape(-1, 3)
    assert np.allclose(np.linalg.norm(f0, axis=1), 1)
    assert np.allclose(np.linalg.norm(s0, axis=1), 1)
    assert np.allclose(np.sum(f0 * s0, axis=1), 0)
