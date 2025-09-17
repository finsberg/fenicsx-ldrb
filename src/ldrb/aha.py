import logging

import dolfinx
import numpy as np

from .ldrb import dolfinx_ldrb

logger = logging.getLogger(__name__)


def gernerate_aha_biv(
    mesh: dolfinx.mesh.Mesh,
    ffun: dolfinx.mesh.MeshTags | None = None,
    markers: (dict[str, list[int]] | dict[str, tuple[int, ...]] | dict[str, int] | None) = None,
    base_max: float = 0.75,
    mid_base: float = 0.7,
    apex_mid: float = 0.65,
    function_space: str = "DG_0",
) -> dolfinx.fem.Function:
    """Generate an AHA regions for a biventricle mesh.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The mesh to generate the AHA function on.
    ffun : dolfinx.mesh.MeshTags | None, optional
        Facet function, by default None
    markers : dict[str, list[int]]  |  dict[str, tuple[int, ...]]  |  dict[str, int]  |  None
        Markers for the mesh, by default None
    base_max : float, optional
        Maximum value for the base region, by default 0.75
    mid_base : float, optional
        Maximum value for the mid region, by default 0.7
    apex_mid : float, optional
        Maximum value for the apex region, by default 0.65
    function_space : str, optional
        Function space for the AHA function, by default "DG_0"

    Returns
    -------
    dolfinx.fem.Function
        The AHA biventricle function.


    Notes
    -----
    This function first run the LDRB algorithm to get the apex to base
    function and the markers_scalar function which contains the
    segmentation of the mesh into different anatomical regions (RV, LV and Septum).
    We then use the apex to base function to define the basal, mid and apical regions
    by using the base_max, mid_base and apex_mid parameters. Note that the
    apex to base function is normalized to the range [0, 1], where 0 is the apex
    and 1 is the base. So even though it seems intuitive that you should choose the
    base, mid and apical region to be about 1/3 each it depends on the shape of the
    mesh, and these values should be adjusted accordingly. The default values
    are based on the mean shape of the ukb atlas.

    Also note that with this choice of parameters the basal region does not go
    all the way to the base, but stops a bit earlier. This is because the markers_scalar
    function is a bit inprecise at the base (especially that parts of the septum is
    labeled as LV). If you want the basal region to go all the way to the base you
    should increase the base_max parameter to be closer to 1.0.

    """
    system = dolfinx_ldrb(
        mesh=mesh,
        ffun=ffun,
        markers=markers,
        alpha_endo_lv=60,
        alpha_epi_lv=-60,
        alpha_endo_rv=90,
        alpha_epi_rv=-25,
        beta_endo_lv=-20,
        beta_epi_lv=20,
        beta_endo_rv=0,
        beta_epi_rv=20,
        fiber_space=function_space,
    )
    assert system.apex is not None
    assert system.markers_scalar is not None
    V = system.markers_scalar.function_space

    apex = dolfinx.fem.Function(V, name="apex_to_base")
    apex.interpolate(system.apex)
    aha = dolfinx.fem.Function(V, name="aha")
    apex_max = apex.x.array.max()
    apex_min = apex.x.array.min()
    max_value = base_max * (apex_max + apex_min)
    min_value = mid_base * (apex_max + apex_min)
    inds = np.where(np.logical_and(apex.x.array > min_value, apex.x.array < max_value))[0]

    logger.info("Basal region: %d inds between %f and %f", len(inds), min_value, max_value)
    aha.x.array[inds] = system.markers_scalar.x.array[inds]

    inds = np.where(np.logical_and(aha.x.array < 2.0, aha.x.array > 1.0))[0]
    aha.x.array[inds] = 3.0

    max_value = mid_base * (apex_max + apex_min)
    min_value = apex_mid * (apex_max + apex_min)
    inds = np.where(np.logical_and(apex.x.array > min_value, apex.x.array < max_value))[0]
    logger.info("Mid region: %d inds between %f and %f", len(inds), min_value, max_value)
    aha.x.array[inds] = system.markers_scalar.x.array[inds] + 3
    inds = np.where(np.logical_and(aha.x.array < 5.0, aha.x.array > 4.0))[0]
    aha.x.array[inds] = 6.0
    return aha
