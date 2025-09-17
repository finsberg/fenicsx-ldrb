from pathlib import Path

from mpi4py import MPI

import numpy as np

import cardiac_geometries
import ldrb


def test_aha_biv(tmp_path: Path):
    comm = MPI.COMM_WORLD

    mode = -1
    std = 0
    geodir = comm.bcast(tmp_path / "out", root=0)

    char_length = 10.0
    geo = cardiac_geometries.mesh.ukb(
        outdir=geodir,
        comm=comm,
        mode=mode,
        std=std,
        case="ED",
        create_fibers=False,
        char_length_max=char_length,
        char_length_min=char_length,
        clipped=True,
    )
    aha = ldrb.aha.gernerate_aha_biv(
        mesh=geo.mesh,
        ffun=geo.ffun,
        markers=cardiac_geometries.mesh.transform_markers(geo.markers, clipped=True),
        function_space="DG_0",
        base_max=0.748,
        mid_base=0.71,
        apex_mid=0.65,
    )

    assert set(np.hstack(comm.allgather(aha.x.array))) == {0, 1, 2, 3, 4, 5, 6}
