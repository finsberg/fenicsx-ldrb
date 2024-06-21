from mpi4py import MPI

import dolfinx
import ldrb
import numpy as np
import pytest


@pytest.mark.parametrize("space1", ["P_1", "P_2", "dP_0", "dP_1"])
@pytest.mark.parametrize("space2", ["P_1", "P_2", "dP_0", "dP_1"])
def test_save_load(tmp_path, space1, space2):
    # FIXME: Make it work for Quadrature spaces
    mesh = dolfinx.mesh.create_unit_cube(comm=MPI.COMM_WORLD, nx=3, ny=3, nz=3)
    U = ldrb.utils.space_from_string(space1, mesh=mesh, dim=3)
    u = dolfinx.fem.Function(U, name="u")
    u.interpolate(lambda x: x)

    V = ldrb.utils.space_from_string(space2, mesh=mesh, dim=3)
    v = dolfinx.fem.Function(V, name="v")
    v.interpolate(lambda x: -x)
    functions = [u, v]
    filename = tmp_path / "test_save_load.bp"
    ldrb.io.save(comm=mesh.comm, filename=filename, functions=functions)
    loaded_functions = ldrb.io.load(comm=mesh.comm, filename=filename)
    assert len(loaded_functions) == 2

    assert loaded_functions[0].name == "u"
    assert loaded_functions[1].name == "v"

    assert np.allclose(loaded_functions[0].x.array, u.x.array)
    assert np.allclose(loaded_functions[1].x.array, v.x.array)
