import pytest
import cardiac_geometries


@pytest.fixture(scope="session", params=["marker_is_int", "marker_is_list"])
def lv_geometry_markers(request, tmpdir_factory):
    outdir = tmpdir_factory.mktemp("lv")
    geo = cardiac_geometries.mesh.lv_ellipsoid(outdir=outdir)
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
    outdir = tmpdir_factory.mktemp("lv")
    geo = cardiac_geometries.mesh.lv_ellipsoid(outdir=outdir)
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
    outdir = tmpdir_factory.mktemp("biv")
    geo = cardiac_geometries.mesh.biv_ellipsoid(outdir=outdir)
    markers = {
        "base": geo.markers["BASE"][0],
        "epi": geo.markers["EPI"][0],
        "lv": geo.markers["ENDO_LV"][0],
        "rv": geo.markers["ENDO_RV"][0],
    }
    geo.markers.update(markers)
    return geo
