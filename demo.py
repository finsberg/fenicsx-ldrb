# # Demo
# This script demonstrates how to generate a simple plot of the fiber field on a mesh.

import ldrb
import numpy as np
import pyvista
from dolfinx.plot import vtk_mesh

import cardiac_geometries

# Generate an idealized BiV geometry
geo = cardiac_geometries.mesh.biv_ellipsoid(outdir="biv")
# Generate fibers with 60/-60 fibers angles on the endo- and epicardium
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

f0 = system.f0
pyvista.start_xvfb()
topology, cell_types, geometry = vtk_mesh(f0.function_space)
values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
values[:, : len(f0)] = f0.x.array.real.reshape((geometry.shape[0], len(f0)))

# Create a point cloud of glyphs
function_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
function_grid["u"] = values
glyphs = function_grid.glyph(orient="u", factor=0.2)

# Create a pyvista-grid for the mesh
geo.mesh.topology.create_connectivity(geo.mesh.topology.dim, geo.mesh.topology.dim)
grid = pyvista.UnstructuredGrid(*vtk_mesh(geo.mesh, geo.mesh.topology.dim))

# Create plotter
plotter = pyvista.Plotter()
plotter.add_mesh(grid, style="wireframe", color="k")
plotter.add_mesh(glyphs)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    fig_as_array = plotter.screenshot("glyphs.png")
