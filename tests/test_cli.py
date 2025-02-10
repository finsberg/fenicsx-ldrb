from mpi4py import MPI

import cardiac_geometries
import ldrb
import ldrb.cli


def test_cli(tmp_path):
    comm = MPI.COMM_WORLD
    geodir = comm.bcast(tmp_path / "lv", root=0)

    cardiac_geometries.mesh.biv_ellipsoid(outdir=geodir, comm=comm)
    outdir = comm.bcast(tmp_path / "out", root=0)
    args = [
        str(geodir / "mesh.xdmf"),
        "--markers-file",
        str(geodir / "markers.json"),
        "-o",
        str(outdir),
    ]
    ldrb.cli.main(args)
    assert (outdir / "microstructure.bp").exists()
