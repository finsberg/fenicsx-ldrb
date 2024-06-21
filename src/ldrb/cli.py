import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from mpi4py import MPI

import dolfinx

from . import io, ldrb


@dataclass
class Geometry:
    mesh: dolfinx.mesh.Mesh
    ffun: dolfinx.mesh.meshtags
    markers: dict[str, list[int]]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ldrb is a simple tool to manage bookmarks in the terminal"
    )

    parser.add_argument("mesh_file", type=Path, help="Path to mesh file")
    parser.add_argument("-fo", "--full-output", action="store_true", help="Output all fields")
    parser.add_argument("-o", "--output", type=Path, default="output", help="Output file")
    parser.add_argument("-mn", "--mesh-name", type=str, default="Mesh", help="Name of the mesh")
    parser.add_argument(
        "-mf", "--markers-file", type=Path, default=None, help="Path to markers file"
    )
    parser.add_argument(
        "-ffn", "--facet-tags-name", type=str, default="Facet tags", help="Name of the facet tags"
    )
    parser.add_argument("-fs", "--fiber-space", type=str, default="P_1", help="Fiber space to use")
    parser.add_argument("-aenlv", "--alpha-endo-lv", type=float, default=60.0, help="Alpha endo LV")
    parser.add_argument("-aeplv", "--alpha-epi-lv", type=float, default=-60.0, help="Alpha epi LV")
    parser.add_argument("-aenrv", "--alpha-endo-rv", type=float, default=60.0, help="Alpha endo RV")
    parser.add_argument("-aepirv", "--alpha-epi-rv", type=float, default=-60.0, help="Alpha epi RV")
    parser.add_argument(
        "-aensept", "--alpha-endo-sept", type=float, default=60.0, help="Alpha endo sept"
    )
    parser.add_argument(
        "-aepisept", "--alpha-epi-sept", type=float, default=-60.0, help="Alpha epi sept"
    )
    parser.add_argument("-benlv", "--beta-endo-lv", type=float, default=0.0, help="Beta endo LV")
    parser.add_argument("-beplv", "--beta-epi-lv", type=float, default=0.0, help="Beta epi LV")
    parser.add_argument("-benrv", "--beta-endo-rv", type=float, default=0.0, help="Beta endo RV")
    parser.add_argument("-bepirv", "--beta-epi-rv", type=float, default=0.0, help="Beta epi RV")
    parser.add_argument(
        "-bensept", "--beta-endo-sept", type=float, default=0.0, help="Beta endo sept"
    )
    parser.add_argument(
        "-bepisept", "--beta-epi-sept", type=float, default=0.0, help="Beta epi sept"
    )
    parser.add_argument("-lv", "--lv-marker", type=int, default=10, help="Left ventricle marker")
    parser.add_argument("-rv", "--rv-marker", type=int, default=20, help="RV ventricle marker")
    parser.add_argument("-epi", "--epi-marker", type=int, default=30, help="Epicardium marker")
    parser.add_argument("-base", "--base-marker", type=int, default=40, help="Base marker")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-ow", "--overwrite", action="store_true", help="Overwrite output file")

    # parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments to pass to the command")
    return parser


def parse_geometry(
    mesh_file: Path,
    mesh_name: str = "Mesh",
    facet_tags_name="Facet tags",
    markers_file: Path | None = None,
    lv_marker: int = 10,
    rv_marker: int = 20,
    epi_marker: int = 30,
    base_marker: int = 40,
    **kwargs,
) -> Geometry:
    if mesh_file.suffix == ".xdmf":
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, str(mesh_file), "r") as xdmf:
            mesh = xdmf.read_mesh(name=mesh_name)

            # Read facet tags
            mesh.topology.create_connectivity(2, 3)
            facet_tags = xdmf.read_meshtags(mesh, name=facet_tags_name)

    if markers_file is not None:
        markers = json.loads(markers_file.read_text())

    else:
        markers = dict(lv=[lv_marker], rv=[rv_marker], epi=[epi_marker], base=[base_marker])

    return Geometry(mesh=mesh, ffun=facet_tags, markers=markers)


def main(argv: Sequence[str] | None = None) -> int:
    parser = get_parser()
    args = vars(parser.parse_args(argv))
    loglevel = logging.DEBUG if args["verbose"] else logging.INFO
    logging.basicConfig(level=loglevel)

    geo = parse_geometry(**args)
    results = ldrb.dolfinx_ldrb(
        mesh=geo.mesh,
        ffun=geo.ffun,
        markers=geo.markers,
        fiber_space=args["fiber_space"],
        alpha_endo_lv=args["alpha_endo_lv"],
        alpha_epi_lv=args["alpha_epi_lv"],
        alpha_endo_rv=args["alpha_endo_rv"],
        alpha_epi_rv=args["alpha_epi_rv"],
        alpha_endo_sept=args["alpha_endo_sept"],
        alpha_epi_sept=args["alpha_epi_sept"],
        beta_endo_lv=args["beta_endo_lv"],
        beta_epi_lv=args["beta_epi_lv"],
        beta_endo_rv=args["beta_endo_rv"],
        beta_epi_rv=args["beta_epi_rv"],
        beta_endo_sept=args["beta_endo_sept"],
        beta_epi_sept=args["beta_epi_sept"],
    )

    outdir = args["output"]
    outdir.mkdir(exist_ok=True, parents=True)

    io.save(
        comm=geo.mesh.comm,
        filename=outdir / "microstructure.bp",
        functions=[results.fiber, results.sheet, results.sheet_normal],
        overwrite=args["overwrite"],
    )

    if args["full_output"]:
        for k, v in results._asdict().items():
            if v is None:
                continue
            if args["fiber_space"].startswith("Q"):
                # Cannot visualize Quadrature spaces yet
                continue

            print(k, v)
            with dolfinx.io.VTXWriter(
                geo.mesh.comm, outdir / f"{k}-viz.bp", [v], engine="BP4"
            ) as vtx:
                vtx.write(0.0)

    return 0
