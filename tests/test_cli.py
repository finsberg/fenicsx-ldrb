from mpi4py import MPI

import gmsh
import pytest

import cardiac_geometries
import ldrb
import ldrb.cli


@pytest.mark.skipif(gmsh.__version__ == "4.14.0", reason="GMSH 4.14.0 has a bug with fuse")
def test_cli_biv(tmp_path):
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


def test_cli_ukb_full(tmp_path):
    comm = MPI.COMM_WORLD
    geodir = comm.bcast(tmp_path / "lv", root=0)

    cardiac_geometries.mesh.ukb(outdir=geodir, comm=comm)
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


def test_cli_ukb_clipped(tmp_path):
    comm = MPI.COMM_WORLD
    geodir = comm.bcast(tmp_path / "lv", root=0)

    cardiac_geometries.mesh.ukb(outdir=geodir, comm=comm, clipped=True)
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
