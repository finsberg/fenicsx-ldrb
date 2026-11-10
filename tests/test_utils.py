from mpi4py import MPI

import dolfinx
import numpy as np

from ldrb import utils


def irregular_cube() -> dolfinx.mesh.Mesh:
    """A unit cube with irrational, jittered coordinates, so that the dof coordinates of a node
    tabulated on different processes (owned and ghost copies) may differ in the last bits."""
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 6, 6, 6)
    x = mesh.geometry.x
    x[:, :3] = np.sin(1.3 * x[:, :3] + 0.1) * np.array([np.pi, np.e, np.sqrt(2)])
    return mesh


def test_closest_dofs_returns_owned_and_ghost_copies():
    # A Dirichlet condition at a point must mark the node on every process that has it,
    # owned or ghost; otherwise the ghost process assembles into the constrained row.
    mesh = irregular_cube()
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    im = V.dofmap.index_map
    coords = V.tabulate_dof_coordinates()
    local_globals = im.local_to_global(np.arange(im.size_local + im.num_ghosts, dtype=np.int32))
    owned_coords = np.vstack(mesh.comm.allgather(coords[: im.size_local]))
    owned_globals = np.hstack(mesh.comm.allgather(local_globals[: im.size_local]))
    # Shared nodes: the ghosts of every process
    ghost_globals = np.unique(np.hstack(mesh.comm.allgather(local_globals[im.size_local :])))
    failures = []
    for g in ghost_globals[:: max(1, len(ghost_globals) // 25)]:
        point = owned_coords[owned_globals == g][0]
        dofs = utils.closest_dofs(V, point)
        expected = np.flatnonzero(local_globals == g)
        if not np.array_equal(np.sort(dofs), expected):
            failures.append((int(g), dofs.tolist(), expected.tolist()))
    # Collect the failures of every process before asserting, so no process waits in a collective
    failures = [f for rank in mesh.comm.allgather(failures) for f in rank]
    assert not failures, failures[:5]


def test_closest_dofs_point_between_nodes():
    mesh = irregular_cube()
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    coords = V.tabulate_dof_coordinates()
    point = np.array([0.5, 0.5, 0.5])
    dofs = utils.closest_dofs(V, point)
    # Exactly one node in the whole mesh is chosen, and it is the closest one
    im = V.dofmap.index_map
    globals_ = set(im.local_to_global(dofs.astype(np.int32)).tolist()) if len(dofs) else set()
    chosen = set().union(*mesh.comm.allgather(globals_))
    assert len(chosen) == 1
    best = mesh.comm.allreduce(float(np.linalg.norm(coords - point, axis=1).min()), op=MPI.MIN)
    ok = not len(dofs) or np.linalg.norm(coords[dofs[0]] - point) <= best + 1e-12
    assert all(mesh.comm.allgather(ok))
