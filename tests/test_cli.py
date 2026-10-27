import inspect

from mpi4py import MPI

import pytest

import cardiac_geometries
import ldrb
import ldrb.cli


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

    cardiac_geometries.mesh.ukb(outdir=geodir, comm=comm, create_fibers=False)
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

    cardiac_geometries.mesh.ukb(outdir=geodir, comm=comm, clipped=True, create_fibers=False)
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


@pytest.mark.parametrize("order", [1, 2, 3])
def test_cli_biv_bp(tmp_path, order):
    if "order" not in inspect.signature(cardiac_geometries.mesh.biv_ellipsoid).parameters:
        if order > 1:
            pytest.skip("cardiac-geometriesx does not support meshes of higher order")
        kwargs = {}
    else:
        kwargs = {"order": order}
    comm = MPI.COMM_WORLD
    geodir = comm.bcast(tmp_path / "lv", root=0)

    cardiac_geometries.mesh.biv_ellipsoid(outdir=geodir, comm=comm, **kwargs)
    outdir = comm.bcast(tmp_path / "out", root=0)
    args = [
        str(geodir / "geometry.bp"),
        "--markers-file",
        str(geodir / "markers.json"),
        "-o",
        str(outdir),
    ]
    ldrb.cli.main(args)
    f0 = ldrb.io.load(comm, outdir / "microstructure.bp")["f0"]
    assert f0.function_space.mesh.ufl_domain().ufl_coordinate_element().degree == order


def test_cli_unsupported_mesh_file(tmp_path):
    with pytest.raises(ValueError, match="Unsupported mesh file"):
        ldrb.cli.main([str(tmp_path / "mesh.msh")])
