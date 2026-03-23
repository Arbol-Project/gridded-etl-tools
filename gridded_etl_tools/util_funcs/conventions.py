"""GeoZarr convention attribute builders for Zarr stores.

Implements the proj: and spatial: Zarr conventions as purely additive
metadata on the root group. No store layout changes.

References:
    - proj:    https://github.com/zarr-conventions/geo-proj
    - spatial: https://github.com/zarr-conventions/spatial
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import xarray as xr
from pyproj import CRS
from pyproj.exceptions import CRSError

logger = logging.getLogger(__name__)

# Convention registration metadata (UUID + schema URLs)
_PROJ_CONVENTION = {
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/geo-proj/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/geo-proj/blob/v1/README.md",
    "uuid": "f17cb550-5864-4468-aeb7-f3180cfb622f",
    "name": "proj:",
    "description": "Coordinate reference system information for geospatial data",
}

_SPATIAL_CONVENTION = {
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/spatial/blob/v1/README.md",
    "uuid": "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4",
    "name": "spatial:",
    "description": "Spatial coordinate information",
}


def _crs_to_proj_attrs(crs: CRS) -> dict[str, str | dict[str, Any]]:
    """Extract proj: attributes from a pyproj CRS object.

    Returns a dict with any combination of ``proj:code``, ``proj:wkt2``,
    and ``proj:projjson``. At least one is required by the geo-proj spec.
    """
    result: dict = {}

    authority = crs.to_authority()
    if authority:
        result["proj:code"] = f"{authority[0]}:{authority[1]}"

    wkt2 = crs.to_wkt("WKT2_2019")
    if wkt2:
        result["proj:wkt2"] = wkt2

    projjson = crs.to_json_dict()
    if projjson:
        # Serialize as JSON string to ensure clean round-trip through Zarr attrs,
        # which may not preserve nested dicts depending on Zarr version/codec.
        result["proj:projjson"] = json.dumps(projjson)

    return result


def _parse_crs(crs_input: str, factory: Callable[[str], CRS], label: str) -> dict[str, str | dict[str, Any]]:
    """Parse a CRS input using the given pyproj factory and return proj: attrs.

    Parameters
    ----------
    crs_input : str
        The CRS input string (authority:code or WKT).
    factory : callable
        pyproj CRS factory method (e.g. ``CRS.from_user_input``, ``CRS.from_wkt``).
    label : str
        Human-readable label for log messages (e.g. "code", "WKT").

    Returns
    -------
    dict
        Dict of ``proj:*`` keys. Empty if the CRS cannot be parsed.
    """
    try:
        crs = factory(crs_input)
    except (CRSError, ValueError):
        logger.warning("Could not parse CRS %s '%s' with pyproj, skipping proj: convention", label, crs_input)
        return {}

    return _crs_to_proj_attrs(crs)


def build_proj_attrs(crs_code: str) -> dict:
    """Build proj: convention attributes from an authority:code CRS string.

    Parameters
    ----------
    crs_code : str
        An authority:code CRS identifier parseable by pyproj, e.g. ``"EPSG:4326"``.

    Returns
    -------
    dict
        Dict of ``proj:*`` keys. Empty if the CRS cannot be parsed.
    """
    return _parse_crs(crs_code, CRS.from_user_input, "code")


def build_proj_attrs_from_wkt(wkt: str) -> dict:
    """Build proj: convention attributes from a WKT string.

    Fallback for datasets that store CRS as a WKT string (e.g. in
    ``dataset.attrs["crs"]["crs_wkt"]``) but don't have a standard
    EPSG code. The resulting attrs may lack ``proj:code`` but will
    include ``proj:wkt2`` and ``proj:projjson``.

    Parameters
    ----------
    wkt : str
        A WKT CRS string parseable by pyproj.

    Returns
    -------
    dict
        Dict of ``proj:*`` keys. Empty if the WKT cannot be parsed.
    """
    return _parse_crs(wkt, CRS.from_wkt, "WKT")


def _is_regular_grid(coords: npt.NDArray[np.floating], tolerance: float = 0.01) -> bool:
    """Check whether coordinate spacing is uniform within a relative tolerance.

    The default 1% tolerance accommodates floating-point representation noise
    in geographic coordinates (e.g. 0.25° steps stored as float64 can accumulate
    ~1e-14 rounding errors) while still rejecting genuinely irregular grids like
    Gaussian grids where spacing varies by 10%+ across latitudes.

    Parameters
    ----------
    coords : np.ndarray
        1-D array of coordinate values.
    tolerance : float
        Maximum allowed relative deviation from median spacing (default 1%).

    Returns
    -------
    bool
        True if spacing is uniform within tolerance.
    """
    if len(coords) < 2:
        return False
    diffs = np.diff(coords)
    median = float(np.median(diffs))
    if median == 0:
        return False
    max_dev = float(np.max(np.abs(diffs - median)))
    return max_dev <= abs(median) * tolerance


def _compute_affine_transform(y_coords: np.ndarray, x_coords: np.ndarray, res_y: float, res_x: float) -> list[float]:
    """Compute a 6-element affine transform from coordinate arrays.

    The transform follows the spatial: convention format::

        [scale_x, 0, origin_x, 0, scale_y, origin_y]
    """
    return [res_x, 0.0, float(x_coords[0]), 0.0, res_y, float(y_coords[0])]


def _compute_bbox(y_coords: np.ndarray, x_coords: np.ndarray, res_y: float, res_x: float) -> list[float]:
    """Compute bounding box [xmin, ymin, xmax, ymax] for pixel-registered data.

    Extends by half a pixel beyond the outermost coordinate centres.
    """
    half_x = abs(res_x) / 2
    half_y = abs(res_y) / 2

    x_min = float(min(x_coords[0], x_coords[-1])) - half_x
    x_max = float(max(x_coords[0], x_coords[-1])) + half_x
    y_min = float(min(y_coords[0], y_coords[-1])) - half_y
    y_max = float(max(y_coords[0], y_coords[-1])) + half_y

    return [x_min, y_min, x_max, y_max]


def build_spatial_attrs(dataset: xr.Dataset, spatial_dims: list[str]) -> dict:
    """Build spatial: convention attributes from a dataset's spatial coordinates.

    If the grid has irregular spacing (>1% deviation from median), the affine
    transform is skipped and only ``spatial:dimensions`` and ``spatial:bbox``
    are populated.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with spatial coordinate arrays.
    spatial_dims : list[str]
        Names of the spatial dimensions, ordered [y_dim, x_dim].

    Returns
    -------
    dict
        Dict of ``spatial:*`` keys. Empty if spatial dims are not found or
        dataset is not a proper xarray Dataset.
    """
    if len(spatial_dims) != 2:
        logger.warning("Expected exactly 2 spatial dims, got %d, skipping spatial: convention", len(spatial_dims))
        return {}

    y_dim, x_dim = spatial_dims
    if y_dim not in dataset.coords or x_dim not in dataset.coords:
        logger.warning("Spatial dims %s not found in dataset coords, skipping spatial: convention", spatial_dims)
        return {}

    y_coords = dataset[y_dim].values
    x_coords = dataset[x_dim].values

    if len(y_coords) < 2 or len(x_coords) < 2:
        return {}

    attrs: dict = {}
    attrs["spatial:dimensions"] = [y_dim, x_dim]

    y_regular = _is_regular_grid(y_coords)
    x_regular = _is_regular_grid(x_coords)

    if y_regular and x_regular:
        # Compute resolution once, reuse in transform and bbox
        res_y = float(np.median(np.diff(y_coords)))
        res_x = float(np.median(np.diff(x_coords)))

        attrs["spatial:transform_type"] = "affine"
        attrs["spatial:transform"] = _compute_affine_transform(y_coords, x_coords, res_y, res_x)
        attrs["spatial:shape"] = [len(y_coords), len(x_coords)]
        attrs["spatial:bbox"] = _compute_bbox(y_coords, x_coords, res_y, res_x)
        attrs["spatial:registration"] = "pixel"
    else:
        logger.info(
            "Irregular grid spacing detected (y_regular=%s, x_regular=%s), "
            "skipping affine transform. Only spatial:dimensions and spatial:bbox will be set.",
            y_regular,
            x_regular,
        )
        # Still provide bbox from coordinate extremes (no half-pixel extension for irregular grids)
        attrs["spatial:bbox"] = [
            float(np.min(x_coords)),
            float(np.min(y_coords)),
            float(np.max(x_coords)),
            float(np.max(y_coords)),
        ]

    return attrs


def build_convention_attrs(
    crs_code: str | None,
    dataset: xr.Dataset,
    spatial_dims: list[str],
    crs_wkt: str | None = None,
) -> dict:
    """Build GeoZarr convention attributes for the root group.

    Orchestrates ``build_proj_attrs`` and ``build_spatial_attrs`` and assembles
    the ``zarr_conventions`` registration array.

    Parameters
    ----------
    crs_code : str or None
        Authority:code CRS identifier (e.g. ``"EPSG:4326"``). If None or not
        parseable by pyproj, falls back to *crs_wkt*.
    dataset : xr.Dataset
        The dataset to extract spatial coordinate information from.
    spatial_dims : list[str]
        Names of the spatial dimensions, ordered [y_dim, x_dim].
    crs_wkt : str or None
        Fallback WKT CRS string, e.g. from ``dataset.attrs["crs"]["crs_wkt"]``.
        Used when *crs_code* is absent or not parseable by pyproj. Useful for
        projected datasets (e.g. Lambert Conformal Conic) that store CRS as WKT
        but don't have a standard EPSG code.

    Returns
    -------
    dict
        Flat dict of convention attributes to set on the Zarr root group.
    """
    conventions: list[dict] = []
    attrs: dict = {}

    # --- proj: convention ---
    # Try authority:code first, fall back to WKT string
    proj_attrs: dict = {}
    if crs_code is not None:
        proj_attrs = build_proj_attrs(crs_code)
    if not proj_attrs and crs_wkt is not None:
        proj_attrs = build_proj_attrs_from_wkt(crs_wkt)

    if proj_attrs:
        conventions.append({**_PROJ_CONVENTION})
        attrs.update(proj_attrs)

    # --- spatial: convention ---
    spatial_attrs = build_spatial_attrs(dataset, spatial_dims)
    if spatial_attrs:
        conventions.append({**_SPATIAL_CONVENTION})
        attrs.update(spatial_attrs)

    if conventions:
        attrs["zarr_conventions"] = conventions

    return attrs
