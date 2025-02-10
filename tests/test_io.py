from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

import ldrb


@pytest.mark.parametrize("space1", ["P_1", "P_2", "dP_0", "dP_1", "Q_2"])
@pytest.mark.parametrize("space2", ["P_1", "P_2", "dP_0", "dP_1", "Q_2"])
def test_save_load(tmp_path, space1, space2):
    mesh = dolfinx.mesh.create_unit_cube(comm=MPI.COMM_WORLD, nx=3, ny=3, nz=3)
    U = ldrb.utils.space_from_string(space1, mesh=mesh, dim=3)
    u = dolfinx.fem.Function(U, name="u")
    u.interpolate(lambda x: x)

    V = ldrb.utils.space_from_string(space2, mesh=mesh, dim=3)
    v = dolfinx.fem.Function(V, name="v")
    v.interpolate(lambda x: -x)

    functions = [u, v]
    filename = mesh.comm.bcast(tmp_path / "test_save_load.bp", root=0)
    function_space = {
        "u": ldrb.utils.element2array(u.ufl_element()),
        "v": ldrb.utils.element2array(v.ufl_element()),
    }
    ldrb.io.save(
        comm=mesh.comm,
        filename=filename,
        functions=functions,
    )
    loaded_functions = ldrb.io.load(
        comm=mesh.comm, filename=filename, mesh=mesh, function_space=function_space
    )
    assert len(loaded_functions) == 2

    assert loaded_functions["u"].name == "u"
    assert loaded_functions["v"].name == "v"

    assert np.allclose(loaded_functions["u"].x.array, u.x.array)
    assert np.allclose(loaded_functions["v"].x.array, v.x.array)
