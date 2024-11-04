import cardiac_geometries
import ldrb
import ldrb.cli


def test_cli(tmp_path):
    geodir = tmp_path / "lv"
    cardiac_geometries.mesh.biv_ellipsoid(outdir=geodir)
    outdir = tmp_path / "out"
    args = [
        str(geodir / "mesh.xdmf"),
        "--markers-file",
        str(geodir / "markers.json"),
        "-o",
        str(outdir),
    ]
    ldrb.cli.main(args)
    assert (outdir / "microstructure.bp").exists()
