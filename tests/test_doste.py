from mpi4py import MPI

import numpy as np
import pytest

import cardiac_geometries
import ldrb

try:
    import ukb.cli
except ImportError:
    HAS_UKB = False
else:
    HAS_UKB = True


def ukb_geometry(tmpdir_factory, geometry: str, order: int = 1):
    comm = MPI.COMM_WORLD
    outdir = comm.bcast(tmpdir_factory.mktemp(geometry), root=0)
    if comm.rank == 0:
        surf_args = ["--refine", "1"] if order > 1 else []
        ukb.cli.main(["surf", str(outdir), "--geometry", geometry, *surf_args])
        ukb.cli.main(
            [
                "mesh",
                str(outdir),
                "--char_length_max",
                "8",
                "--char_length_min",
                "8",
                "--order",
                str(order),
            ]
        )
    comm.barrier()
    geo = cardiac_geometries.utils.gmsh2dolfin(comm=comm, msh_file=outdir / "ED.msh")
    if "LV_ENDO" not in geo.markers:
        pytest.skip("Requires ukb-atlas>=2.0")
    return geo


def biv_markers(geo):
    m = {name: value[0] for name, value in geo.markers.items()}
    return {
        "lv": m["LV_ENDO"],
        "rv": m["RV_ENDO"],
        "epi": m["EPI"],
        "mv": m["MV"],
        "av": m["AV"],
        "tv": m["TV"],
        "pv": m["PV"],
    }


@pytest.fixture(scope="session")
def biv_geometry(tmpdir_factory):
    geo = ukb_geometry(tmpdir_factory, "biv")
    return geo, biv_markers(geo)


@pytest.fixture(scope="session")
def biv_geometry_order2(tmpdir_factory):
    geo = ukb_geometry(tmpdir_factory, "biv", order=2)
    return geo, biv_markers(geo)


@pytest.fixture(scope="session")
def lv_geometry(tmpdir_factory):
    geo = ukb_geometry(tmpdir_factory, "lv")
    m = {name: value[0] for name, value in geo.markers.items()}
    markers = {
        "lv": [m["LV_ENDO"]],
        "epi": [m["EPI"], m["RV_SEPTUM"]],
        "mv": [m["MV"]],
        "av": [m["AV"]],
    }
    return geo, markers


def owned_vectors(f):
    n = f.function_space.dofmap.index_map.size_local
    return f.x.array[: 3 * n].reshape(-1, 3)


def angle(u, v):
    cos = np.abs(np.sum(u * v, axis=1)) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    return np.degrees(np.arccos(np.clip(cos, 0, 1)))


def single_dof_fiber(phi, w, grad_phi, grad_basal, grad_ot, angles, lv_rv=1.0, has_rv=True):
    f0, s0, n0 = np.zeros(3), np.zeros(3), np.zeros(3)
    marker = np.zeros(1)
    dof = np.array([0])
    zeros = np.zeros(3)
    if phi <= 0:
        lv_grads, rv_grads, w_lv, w_rv = (grad_basal, grad_ot), (zeros, zeros), w, 0.0
    else:
        lv_grads, rv_grads, w_lv, w_rv = (zeros, zeros), (grad_basal, grad_ot), 0.0, w
    ldrb.doste.compute_fiber_sheet_system(
        f0,
        s0,
        n0,
        marker,
        dof,
        dof + 1,
        dof + 2,
        dof,
        np.array([phi]),
        np.asarray(grad_phi, dtype=float),
        np.array([lv_rv]),
        np.array([w_lv]),
        np.asarray(lv_grads[0], dtype=float),
        np.asarray(lv_grads[1], dtype=float),
        np.array([w_rv]),
        np.asarray(rv_grads[0], dtype=float),
        np.asarray(rv_grads[1], dtype=float),
        angles,
        has_rv,
    )
    return f0, s0, n0, marker[0]


def angle_array(lv=(0, 0, 0, 0), rv=(0, 0, 0, 0), ot=(0, 0, 0, 0)):
    return np.array([[lv, ot], [rv, ot]], dtype=np.float64)


def test_fiber_angle_convention_at_the_lv_endocardium():
    # Longitudinal direction z, transmural direction x, so circumferential z x x = y
    alpha = 60.0
    f0, s0, n0, marker = single_dof_fiber(
        phi=-2.0,
        w=1.0,
        grad_phi=[1, 0, 0],
        grad_basal=[0, 0, 1],
        grad_ot=[0, 1, 0],
        angles=angle_array(lv=(alpha, -alpha, 0, 0)),
    )
    a = np.radians(alpha)
    assert np.allclose(f0, [0, np.cos(a), np.sin(a)])
    assert np.allclose(s0, [1, 0, 0])
    assert np.allclose(n0, np.cross(f0, s0))
    assert marker == 1


def test_outflow_tract_uses_the_outflow_direction_and_angles():
    # w = 0: longitudinal direction from the outflow gradient, outflow tract angles
    f0, *_ = single_dof_fiber(
        phi=-1.0,
        w=0.0,
        grad_phi=[1, 0, 0],
        grad_basal=[0, 0, 1],
        grad_ot=[0, 1, 0],
        angles=angle_array(lv=(60, -60, 0, 0), ot=(90, 90, 0, 0)),
    )
    assert np.allclose(np.abs(f0), [0, 1, 0])


def test_rv_transmural_direction_points_to_the_epicardium():
    # In the RV phi increases towards the endocardium, so the transmural direction is -grad(phi)
    f0, s0, _, marker = single_dof_fiber(
        phi=1.0,
        w=1.0,
        grad_phi=[1, 0, 0],
        grad_basal=[0, 0, 1],
        grad_ot=[0, 1, 0],
        angles=angle_array(rv=(0, 0, 0, 0)),
    )
    assert np.allclose(s0, [-1, 0, 0])
    # Circumferential direction z x (-x) = -y
    assert np.allclose(f0, [0, -1, 0])
    assert marker == 2


def test_epicardium_is_assigned_with_lv_rv():
    _, _, _, marker = single_dof_fiber(
        phi=0.0,
        w=1.0,
        grad_phi=[1, 0, 0],
        grad_basal=[0, 0, 1],
        grad_ot=[0, 1, 0],
        angles=angle_array(),
        lv_rv=0.2,
    )
    assert marker == 2


def test_transverse_angle_rotates_the_fiber_about_the_longitudinal_direction():
    beta = 30.0
    f0, *_ = single_dof_fiber(
        phi=-2.0,
        w=1.0,
        grad_phi=[1, 0, 0],
        grad_basal=[0, 0, 1],
        grad_ot=[0, 1, 0],
        angles=angle_array(lv=(0, 0, beta, beta)),
    )
    b = np.radians(beta)
    # The circumferential direction y rotated counterclockwise about z
    assert np.allclose(f0, [-np.sin(b), np.cos(b), 0])


def test_missing_markers():
    comm = MPI.COMM_WORLD
    import dolfinx

    mesh = dolfinx.mesh.create_unit_cube(comm, 2, 2, 2)
    ffun = dolfinx.mesh.meshtags(mesh, 2, np.empty(0, np.int32), np.empty(0, np.int32))
    with pytest.raises(ValueError, match="Missing markers"):
        ldrb.dolfinx_doste(mesh, ffun, {"lv": 1, "epi": 2, "mv": 3, "av": 4, "rv": 5})


@pytest.mark.skipif(not HAS_UKB, reason="ukb-atlas is not installed")
@pytest.mark.parametrize("fiber_space", ["P_1", "P_2", "Q_2"])
def test_biv_fibers_are_orthonormal(biv_geometry, fiber_space):
    geo, markers = biv_geometry
    system = ldrb.dolfinx_doste(geo.mesh, geo.ffun, markers, fiber_space=fiber_space)
    f0, s0, n0 = (owned_vectors(f) for f in (system.f0, system.s0, system.n0))
    assert np.allclose(np.linalg.norm(f0, axis=1), 1)
    assert np.allclose(np.linalg.norm(s0, axis=1), 1)
    assert np.allclose(np.sum(f0 * s0, axis=1), 0)
    assert np.allclose(n0, np.cross(f0, s0))

    marker = system.markers_scalar.x.array
    num_lv = geo.mesh.comm.allreduce(int(np.sum(marker == 1)))
    num_rv = geo.mesh.comm.allreduce(int(np.sum(marker == 2)))
    assert num_lv > 0 and num_rv > 0


@pytest.mark.skipif(not HAS_UKB, reason="ukb-atlas is not installed")
def test_biv_agrees_with_bayer_away_from_the_outflow_tracts(biv_geometry):
    # With the same angles everywhere, the methods only differ in the
    # longitudinal direction, which is close to apex-to-base away from the
    # outflow tracts
    geo, markers = biv_geometry
    comm = geo.mesh.comm
    doste = ldrb.dolfinx_doste(
        geo.mesh,
        geo.ffun,
        markers,
        alpha_endo_rv=60,
        alpha_epi_rv=-60,
        beta_endo_lv=0,
        beta_epi_lv=0,
        beta_endo_rv=0,
        beta_epi_rv=0,
    )
    bayer = ldrb.dolfinx_ldrb(
        geo.mesh,
        ffun=geo.ffun,
        markers={
            "lv": markers["lv"],
            "rv": markers["rv"],
            "epi": markers["epi"],
            "base": [markers[valve] for valve in ("mv", "av", "tv", "pv")],
        },
        alpha_endo_lv=60,
        alpha_epi_lv=-60,
        beta_endo_lv=0,
        beta_epi_lv=0,
    )
    diff = angle(owned_vectors(doste.f0), owned_vectors(bayer.f0))
    n = len(diff)
    marker = doste.markers_scalar.x.array[:n]
    w_lv = doste.w_lv.x.array[:n]
    w_rv = doste.w_rv.x.array[:n]

    lv = np.concatenate(comm.allgather(diff[(marker == 1) & (w_lv > 0.9)]))
    rv = np.concatenate(comm.allgather(diff[(marker == 2) & (w_rv > 0.9)]))
    assert np.median(lv) < 15
    assert np.median(rv) < 15


@pytest.mark.skipif(not HAS_UKB, reason="ukb-atlas is not installed")
def test_biv_given_apex_matches_default(biv_geometry):
    geo, markers = biv_geometry
    apex_lv = ldrb.doste.find_apex(geo.mesh, geo.ffun, [markers["lv"]], [markers["mv"]])
    apex_rv = ldrb.doste.find_apex(geo.mesh, geo.ffun, [markers["rv"]], [markers["tv"]])
    default = ldrb.dolfinx_doste(geo.mesh, geo.ffun, markers)
    given = ldrb.dolfinx_doste(geo.mesh, geo.ffun, markers, apex_lv=apex_lv, apex_rv=apex_rv)
    assert np.allclose(default.f0.x.array, given.f0.x.array)


@pytest.mark.skipif(not HAS_UKB, reason="ukb-atlas is not installed")
def test_lv_only(lv_geometry):
    geo, markers = lv_geometry
    system = ldrb.dolfinx_doste(geo.mesh, geo.ffun, markers)
    assert system.w_rv is None
    assert system.lv_rv is None
    f0 = owned_vectors(system.f0)
    assert np.allclose(np.linalg.norm(f0, axis=1), 1)
    assert np.all(system.markers_scalar.x.array == 1)


@pytest.mark.skipif(not HAS_UKB, reason="ukb-atlas is not installed")
@pytest.mark.parametrize("fiber_space", ["P_1", "P_2"])
def test_biv_higher_order(biv_geometry_order2, fiber_space):
    geo, markers = biv_geometry_order2
    system = ldrb.dolfinx_doste(geo.mesh, geo.ffun, markers, fiber_space=fiber_space)
    assert system.phi.function_space.element.basix_element.degree == 2
    f0, s0, n0 = (owned_vectors(f) for f in (system.f0, system.s0, system.n0))
    assert np.allclose(np.linalg.norm(f0, axis=1), 1)
    assert np.allclose(np.linalg.norm(s0, axis=1), 1)
    assert np.allclose(np.sum(f0 * s0, axis=1), 0)
    assert np.allclose(n0, np.cross(f0, s0))
