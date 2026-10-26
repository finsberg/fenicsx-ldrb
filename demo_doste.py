# # Fibers with outflow tracts
# This demo compares the rule-based method of [Bayer et al.
# (2012)](https://doi.org/10.1007/s10439-012-0593-5) (`ldrb.dolfinx_ldrb`) with the method of [Doste
# et al. (2019)](https://doi.org/10.1002/cnm.3185) (`ldrb.dolfinx_doste`) on a biventricular
# geometry from the UK Biobank atlas, which includes the outflow tracts and all four valves.
#
# The method of Bayer et al. uses a single direction from the apex to the base. On a geometry with
# outflow tracts, the base has to be all four valves, so the longitudinal direction points to
# whichever valve is closest. The method of Doste et al. instead defines the longitudinal direction
# in each ventricle as a blend of the direction from the apex to the inflow valve (mitral or
# tricuspid) and the direction from the apex to the outflow valve (aortic or pulmonary), and uses
# separate fiber angles in the outflow tracts.
#
# First we import the necessary packages

import dolfinx
import numpy as np
import pyvista

import cardiac_geometries
import ldrb

# We use [`cardiac-geometries`](https://github.com/ComputationalPhysiology/cardiac-geometriesx) to
# create the mean shape of the UK Biobank atlas at end diastole using
# [`ukb-atlas`](https://github.com/ComputationalPhysiology/ukb-atlas), and save it to a folder
# called `ukb`

geo = cardiac_geometries.mesh.ukb(
    outdir="ukb",
    case="ED",
    char_length_max=5.0,
    char_length_min=5.0,
    create_fibers=False,
)

# The markers are the endocardium of each ventricle, the epicardium and the rims of the four valves

print(geo.markers)

# Let us plot the facet tags

vtk_mesh = dolfinx.plot.vtk_mesh(geo.mesh, geo.mesh.topology.dim)
vtk_bmesh = dolfinx.plot.vtk_mesh(geo.mesh, geo.ffun.dim, geo.ffun.indices)
bgrid = pyvista.UnstructuredGrid(*vtk_bmesh)
bgrid.cell_data["Facet tags"] = geo.ffun.values
bgrid.set_active_scalars("Facet tags")
plotter = pyvista.Plotter(window_size=[800, 800])
plotter.add_mesh(bgrid, show_edges=True)
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    plotter.screenshot("ukb_facet_tags.png")

# The method of Bayer et al. takes the valves together as the base, while the method of Doste et al.
# needs each valve separately

m = {name: value[0] for name, value in geo.markers.items()}
bayer_markers = {
    "lv": m["LV_ENDO"],
    "rv": m["RV_ENDO"],
    "epi": m["EPI"],
    "base": [m["MV"], m["AV"], m["TV"], m["PV"]],
}
doste_markers = {
    "lv": m["LV_ENDO"],
    "rv": m["RV_ENDO"],
    "epi": m["EPI"],
    "mv": m["MV"],
    "av": m["AV"],
    "tv": m["TV"],
    "pv": m["PV"],
}

# Now we compute the fibers with both methods. For the method of Bayer et al. we use fiber angles of
# 60/-60 degrees on the endo- and epicardium, which are also the LV angles of Doste et al.

bayer = ldrb.dolfinx_ldrb(
    mesh=geo.mesh,
    ffun=geo.ffun,
    markers=bayer_markers,
    alpha_endo_lv=60,
    alpha_epi_lv=-60,
    beta_endo_lv=0,
    beta_epi_lv=0,
    fiber_space="P_1",
)

# For the method of Doste et al. we use the default angles, which are the ones from the paper,
# written in the convention of `ldrb.dolfinx_ldrb`:
#
# | | $\alpha$ endo | $\alpha$ epi | $\beta$ endo | $\beta$ epi |
# |---|---|---|---|---|
# | LV | 60 | -60 | 20 | -20 |
# | RV | 90 | -25 | 0 | -20 |
# | Outflow tracts | 90 | 0 | 0 | 0 |
#
# In the RV the endocardial fibers are longitudinal, and in the outflow tracts the fibers go from
# longitudinal on the endocardium to circumferential on the epicardium. The apex of each ventricle
# is the point on its endocardium farthest from the inflow valve, unless it is given with `apex_lv`
# and `apex_rv`.

doste = ldrb.dolfinx_doste(
    mesh=geo.mesh,
    ffun=geo.ffun,
    markers=doste_markers,
    fiber_space="P_1",
)

# The longitudinal direction and the angles are blended between the ventricle and the outflow tract
# with the function $w$, which is 1 at the apex and the inflow valve, and 0 at the outflow valve.
# Let us plot $w$ in each ventricle, using `markers_scalar` (1 in the LV and 2 in the RV) to pick
# the ventricle

V = doste.w_lv.function_space
w = dolfinx.fem.Function(V)
in_lv = doste.markers_scalar.x.array == 1
assert isinstance(doste.w_lv, dolfinx.fem.Function)
assert isinstance(doste.w_rv, dolfinx.fem.Function)
w.x.array[:] = np.where(in_lv, doste.w_lv.x.array, doste.w_rv.x.array)

grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(V))
grid.point_data["w"] = w.x.array
plotter = pyvista.Plotter(window_size=[800, 800])
plotter.add_mesh(grid, scalars="w", cmap="viridis")
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    plotter.screenshot("ukb_w.png")

# Next we plot the fibers from both methods side by side. Since fibers have no direction, we draw
# each fiber as a line segment centred at its point. The surface is opaque, so we only see the
# fibers on the epicardium


def fiber_glyphs(f0, factor=4.0):
    topology, cell_types, x = dolfinx.plot.vtk_mesh(f0.function_space)
    fiber_grid = pyvista.UnstructuredGrid(topology, cell_types, x)
    fiber_grid["f0"] = f0.x.array.reshape((x.shape[0], 3))
    line = pyvista.Line(pointa=(-0.5, 0.0, 0.0), pointb=(0.5, 0.0, 0.0))
    return fiber_grid.glyph(orient="f0", scale=False, factor=factor, geom=line)


mesh_grid = pyvista.UnstructuredGrid(*vtk_mesh)
plotter = pyvista.Plotter(shape=(1, 2), window_size=[1200, 600])
for column, (title, system) in enumerate([("Bayer et al.", bayer), ("Doste et al.", doste)]):
    plotter.subplot(0, column)
    plotter.add_text(title, font_size=12)
    plotter.add_mesh(mesh_grid, color="white")
    plotter.add_mesh(fiber_glyphs(system.f0), color="firebrick", line_width=3)  # type: ignore[attr-defined]
plotter.link_views()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    plotter.screenshot("ukb_fibers.png")

# To see where the methods differ, we compute the angle between the fibers at each point. Fibers
# have no direction, so the angle is between 0 and 90 degrees


def fiber_angle(f, g):
    f = f.x.array.reshape(-1, 3)
    g = g.x.array.reshape(-1, 3)
    cos = np.abs(np.sum(f * g, axis=1))
    return np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))


difference = fiber_angle(bayer.f0, doste.f0)
grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(doste.markers_scalar.function_space))
grid.point_data["Angle difference"] = difference
plotter = pyvista.Plotter(window_size=[800, 800])
plotter.add_mesh(grid, scalars="Angle difference", cmap="magma", clim=[0, 90])
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    plotter.screenshot("ukb_angle_difference.png")

# The fibers differ the most in the RV and in the outflow tracts, where Doste et al. use other
# angles than in the LV. In the LV away from the outflow tract the angles are the same, apart from
# the transverse angle $\beta$, which we set to zero for Bayer et al. Let us summarise the
# differences by region, using $w > 0.9$ for the ventricles away from the outflow tracts and
# $w < 0.2$ for the outflow tracts

phi = doste.phi.x.array
regions = {
    "LV, away from the outflow tract": in_lv & (w.x.array > 0.9),
    "LV outflow tract": in_lv & (w.x.array < 0.2),
    "RV endocardium, away from the outflow tract": ~in_lv & np.isclose(phi, 1) & (w.x.array > 0.9),
    "RV epicardium, away from the outflow tract": ~in_lv & np.isclose(phi, 0) & (w.x.array > 0.9),
    "RV outflow tract": ~in_lv & (w.x.array < 0.2),
}
for name, region in regions.items():
    print(f"{name}: median angle difference {np.median(difference[region]):.0f} degrees")

# Finally, we compute the fibers of Doste et al. with the same angles as Bayer et al. everywhere,
# so that the methods only differ in the longitudinal direction

doste_same_angles = ldrb.dolfinx_doste(
    mesh=geo.mesh,
    ffun=geo.ffun,
    markers=doste_markers,
    fiber_space="P_1",
    alpha_endo_rv=60,
    alpha_epi_rv=-60,
    alpha_endo_ot=60,
    alpha_epi_ot=-60,
    beta_endo_lv=0,
    beta_epi_lv=0,
    beta_endo_rv=0,
    beta_epi_rv=0,
)
difference_same_angles = fiber_angle(bayer.f0, doste_same_angles.f0)
for name, region in regions.items():
    median = np.median(difference_same_angles[region])
    print(f"{name}: median angle difference {median:.0f} degrees")

# With the same angles, the fibers are close everywhere. Since the base of Bayer et al. includes
# all four valves, its longitudinal direction already points towards the nearest valve, also in the
# outflow tracts. On this geometry, the main difference between the methods is therefore that Doste
# et al. let the fiber angles vary between the LV, the RV and the outflow tracts, following
# histology.
