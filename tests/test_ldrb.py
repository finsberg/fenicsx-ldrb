from mpi4py import MPI

import dolfinx
import ldrb
import numpy as np
import pytest
from dolfinx.geometry import bb_tree, compute_colliding_cells, compute_collisions_points

tol = 1e-12


def norm(v):
    return np.linalg.norm(v)


def evaluate(u, x, tree=None):
    mesh = u.function_space.mesh
    if tree is None:
        tree = bb_tree(mesh, mesh.geometry.dim)
    cell_candidates = compute_collisions_points(tree, x)
    cell = compute_colliding_cells(mesh, cell_candidates, x).array
    return u.eval(x, cell[0])


def test_scalar_laplacians():
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, nx=2, ny=2, nz=2)
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
    assert np.isclose(evaluate(solutions["apex"], np.array([1.0, 1.0, 1.0])), 0.0)
    assert np.isclose(evaluate(solutions["apex"], np.array([0.0, 0.0, 0.0])), 1.0)

    # Check LV
    assert np.isclose(evaluate(solutions["lv"], np.array([0.0, 0.0, 0.0])), 1.0)
    assert np.isclose(evaluate(solutions["lv"], np.array([0.5, 0.5, 0.5])), 0.5)
    assert np.isclose(evaluate(solutions["lv"], np.array([0.5, 0.1, 0.1])), 0.5)
    assert np.isclose(evaluate(solutions["lv"], np.array([1.0, 1.0, 1.0])), 0.0)

    # Check EPI (which should be opposite of LV)
    assert np.isclose(evaluate(solutions["epi"], np.array([0.0, 0.0, 0.0])), 0.0)
    assert np.isclose(evaluate(solutions["epi"], np.array([0.5, 0.5, 0.5])), 0.5)
    assert np.isclose(evaluate(solutions["epi"], np.array([0.5, 0.1, 0.1])), 0.5)
    assert np.isclose(evaluate(solutions["epi"], np.array([1.0, 1.0, 1.0])), 1.0)


def test_apex_to_base(lv_geometry):
    apex = ldrb.ldrb.apex_to_base(lv_geometry.mesh, lv_geometry.markers["base"], lv_geometry.ffun)
    assert np.isclose(apex.x.array.max(), 1.0)
    assert np.isclose(apex.x.array.min(), 0.0)

    # Base is located at x=5.0
    x_base = np.array([5.0, 8.0, 0.0])
    # Value at base should be 1.0
    assert np.isclose(evaluate(apex, x_base), 1.0)

    # Apex is located at x=-20.0
    x_apex = np.array([-20.0, 0.0, 0.0])
    # Value at apex should be 0.0
    assert np.isclose(evaluate(apex, x_apex), 0.0)


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
