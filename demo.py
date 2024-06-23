# # Demo
# This script demonstrates how to generate fiber for an idealized biventricular geometry.
# First we import the necessary packages

import ldrb
import numpy as np
import pyvista
import dolfinx
from dolfinx.plot import vtk_mesh
import cardiac_geometries

# We will use [`cardiac-geometries`](https://github.com/ComputationalPhysiology/cardiac-geometriesx) for generate an idealized BiV geometry and save the geometry to a folder called `biv`

geo = cardiac_geometries.mesh.biv_ellipsoid(outdir="biv")

# Next we will use `pyvista` to plot the mesh

pyvista.start_xvfb()
topology, cell_types, geometry = vtk_mesh(geo.mesh, geo.mesh.topology.dim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("biv_mesh.png")

# It is important that the mesh is marked correctly. For a BiV geometry we need the surfaces for the endocardium (LV and RV), base and the epicardium to be marked wih a specific tag. In `cardiac_geometries`, the markers are saved in a dictionary

print(geo.markers)

# Here the keys are `ENDO_LV`, `ENDO_RV`, `EPI` and `BASE` and the values are a list with two elements, the first being the marker and the second being the dimension for the marker (which is two in these cases). The ldrb algorithm accepts this format, but the default format is a dictionary with keys being `lv`, `rv`, `epi` and `base`, and the values being the values of the markers, either as an integer or as a list of integer.
#
# ```{note}
# For single ventricular mesh, you don't need to provide the `rv` key, and instead of `ENDO_LV` you can use `ENDO`
# ```
#
# Next lets plot the facet tags with pyvista.

bmesh, _, _, _ = dolfinx.mesh.create_submesh(geo.mesh, 2, geo.ffun.indices)
btopology, bcell_types, bgeometry = vtk_mesh(bmesh, bmesh.topology.dim)
bgrid = pyvista.UnstructuredGrid(btopology, bcell_types, bgeometry)
bgrid.cell_data["Facet tags"] = geo.ffun.values
bgrid.set_active_scalars("Facet tags")
p = pyvista.Plotter(window_size=[800, 800])
p.add_mesh(bgrid, show_edges=True)
if not pyvista.OFF_SCREEN:
    p.show()
else:
    figure = p.screenshot("subdomains_unstructured.png")

# Now let us generate fibers with 60/-60 fibers angles on the endo- and epicardium

# +

system = ldrb.dolfinx_ldrb(
    mesh=geo.mesh,
    ffun=geo.ffun,
    markers=geo.markers,
    alpha_endo_lv=60,
    alpha_epi_lv=-60,
    beta_endo_lv=0,
    beta_epi_lv=0,
    fiber_space="P_1",
)
# -

# And let us plot the fibers with pyvista

f0 = system.f0
pyvista.start_xvfb()
topology, cell_types, geometry = vtk_mesh(f0.function_space)
values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
values[:, : len(f0)] = f0.x.array.real.reshape((geometry.shape[0], len(f0)))
function_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
function_grid["u"] = values
glyphs = function_grid.glyph(orient="u", factor=0.2)
geo.mesh.topology.create_connectivity(geo.mesh.topology.dim, geo.mesh.topology.dim)
grid = pyvista.UnstructuredGrid(*vtk_mesh(geo.mesh, geo.mesh.topology.dim))
plotter = pyvista.Plotter()
plotter.add_mesh(grid, style="wireframe", color="r")
plotter.add_mesh(glyphs)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    fig_as_array = plotter.screenshot("fiber.png")
