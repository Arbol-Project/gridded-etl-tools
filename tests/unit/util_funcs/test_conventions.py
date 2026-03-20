"""Tests for gridded_etl_tools.util_funcs.conventions."""

import numpy as np
import pytest
import xarray as xr

from gridded_etl_tools.util_funcs.conventions import (
    build_convention_attrs,
    build_proj_attrs,
    build_proj_attrs_from_wkt,
    build_spatial_attrs,
    _is_regular_grid,
    _compute_affine_transform,
    _compute_bbox,
)


# --- Fixtures ---


@pytest.fixture
def regular_latlon_dataset():
    """A simple regular lat/lon dataset."""
    lat = np.arange(-90, 90.25, 0.25)
    lon = np.arange(-180, 180, 0.25)
    data = np.random.rand(len(lat), len(lon)).astype(np.float32)
    return xr.Dataset(
        {"temperature": (["latitude", "longitude"], data)},
        coords={"latitude": lat, "longitude": lon},
    )


@pytest.fixture
def irregular_lat_dataset():
    """A dataset with irregular latitude spacing (like Gaussian grid)."""
    lat = np.array([-89.14, -85.0, -70.5, -45.0, -20.3, 0.0, 20.3, 45.0, 70.5, 85.0, 89.14])
    lon = np.arange(0, 360, 1.0)
    data = np.random.rand(len(lat), len(lon)).astype(np.float32)
    return xr.Dataset(
        {"temperature": (["latitude", "longitude"], data)},
        coords={"latitude": lat, "longitude": lon},
    )


@pytest.fixture
def projected_dataset():
    """A dataset with projected x/y coordinates."""
    y = np.arange(0, 100000, 3000.0)
    x = np.arange(0, 200000, 3000.0)
    data = np.random.rand(len(y), len(x)).astype(np.float32)
    return xr.Dataset(
        {"temperature": (["y_projection", "x_projection"], data)},
        coords={"y_projection": y, "x_projection": x},
    )


# --- build_proj_attrs ---


class TestBuildProjAttrs:
    def test_epsg_4326(self):
        attrs = build_proj_attrs("EPSG:4326")
        assert attrs["proj:code"] == "EPSG:4326"
        assert "proj:wkt2" in attrs
        assert "proj:projjson" in attrs
        assert "WGS 84" in attrs["proj:wkt2"]
        assert attrs["proj:projjson"]["name"] == "WGS 84"

    def test_epsg_4269_nad83(self):
        attrs = build_proj_attrs("EPSG:4269")
        assert attrs["proj:code"] == "EPSG:4269"
        assert "NAD83" in attrs["proj:wkt2"]

    def test_projected_crs(self):
        attrs = build_proj_attrs("EPSG:32637")
        assert attrs["proj:code"] == "EPSG:32637"
        assert "proj:wkt2" in attrs
        assert "proj:projjson" in attrs

    def test_non_epsg_string_returns_empty(self):
        attrs = build_proj_attrs("Reduced Gaussian Grid")
        assert attrs == {}

    def test_none_like_string_returns_empty(self):
        attrs = build_proj_attrs("not_a_crs")
        assert attrs == {}

    def test_lambert_conformal_conic_string_returns_empty(self):
        attrs = build_proj_attrs("Lambert Conformal Conic")
        assert attrs == {}


# --- build_proj_attrs_from_wkt ---

# Realistic LCC WKT from HRRR-like dataset (custom sphere, no EPSG)
LCC_WKT = (
    'PROJCRS["undefined",BASEGEOGCRS["undefined",DATUM["undefined",'
    'ELLIPSOID["undefined",6371229,0,LENGTHUNIT["metre",1,ID["EPSG",9001]]]],'
    'PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8901]]],'
    'CONVERSION["unknown",METHOD["Lambert Conic Conformal (2SP)",ID["EPSG",9802]],'
    'PARAMETER["Latitude of 1st standard parallel",38.5,'
    'ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8823]],'
    'PARAMETER["Latitude of 2nd standard parallel",38.5,'
    'ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8824]],'
    'PARAMETER["Latitude of false origin",38.5,'
    'ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8821]],'
    'PARAMETER["Longitude of false origin",262.5,'
    'ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8822]],'
    'PARAMETER["Easting at false origin",0,LENGTHUNIT["metre",1],ID["EPSG",8826]],'
    'PARAMETER["Northing at false origin",0,LENGTHUNIT["metre",1],ID["EPSG",8827]]],'
    'CS[Cartesian,2],AXIS["(E)",east,ORDER[1],LENGTHUNIT["metre",1,ID["EPSG",9001]]],'
    'AXIS["(N)",north,ORDER[2],LENGTHUNIT["metre",1,ID["EPSG",9001]]]]'
)


class TestBuildProjAttrsFromWkt:
    def test_lcc_wkt_produces_wkt2_and_projjson(self):
        attrs = build_proj_attrs_from_wkt(LCC_WKT)
        # Custom sphere has no EPSG code
        assert "proj:code" not in attrs
        # But WKT2 and PROJJSON should be populated
        assert "proj:wkt2" in attrs
        assert "proj:projjson" in attrs
        assert "Lambert" in attrs["proj:wkt2"]

    def test_epsg_wkt_includes_code(self):
        from pyproj import CRS

        wkt = CRS.from_epsg(4326).to_wkt()
        attrs = build_proj_attrs_from_wkt(wkt)
        assert attrs["proj:code"] == "EPSG:4326"
        assert "proj:wkt2" in attrs
        assert "proj:projjson" in attrs

    def test_invalid_wkt_returns_empty(self):
        attrs = build_proj_attrs_from_wkt("not a valid WKT string")
        assert attrs == {}


# --- _is_regular_grid ---


class TestIsRegularGrid:
    def test_regular_spacing(self):
        coords = np.arange(0, 100, 0.25)
        assert _is_regular_grid(coords) is True

    def test_irregular_spacing(self):
        coords = np.array([0, 1, 3, 6, 10, 15])
        assert _is_regular_grid(coords) is False

    def test_single_point(self):
        assert _is_regular_grid(np.array([42.0])) is False

    def test_two_points(self):
        assert _is_regular_grid(np.array([0.0, 1.0])) is True

    def test_tiny_floating_point_jitter(self):
        coords = np.arange(0, 10, 0.1)
        assert _is_regular_grid(coords) is True


# --- _compute_affine_transform ---


class TestComputeAffineTransform:
    def test_basic_latlon(self):
        y = np.array([90.0, 89.75, 89.5])
        x = np.array([-180.0, -179.75, -179.5])
        transform = _compute_affine_transform(y, x, res_y=-0.25, res_x=0.25)
        assert len(transform) == 6
        assert transform[0] == pytest.approx(0.25)  # scale_x
        assert transform[1] == 0.0  # shear
        assert transform[2] == pytest.approx(-180.0)  # origin_x
        assert transform[3] == 0.0  # shear
        assert transform[4] == pytest.approx(-0.25)  # scale_y (descending lat)
        assert transform[5] == pytest.approx(90.0)  # origin_y


# --- _compute_bbox ---


class TestComputeBbox:
    def test_global_bbox(self):
        y = np.arange(-90, 90.25, 0.25)
        x = np.arange(-180, 180, 0.25)
        bbox = _compute_bbox(y, x, res_y=0.25, res_x=0.25)
        # Half pixel = 0.125
        assert bbox[0] == pytest.approx(-180.125)  # xmin
        assert bbox[1] == pytest.approx(-90.125)  # ymin
        assert bbox[2] == pytest.approx(179.875)  # xmax
        assert bbox[3] == pytest.approx(90.125)  # ymax


# --- build_spatial_attrs ---


class TestBuildSpatialAttrs:
    def test_regular_latlon(self, regular_latlon_dataset):
        attrs = build_spatial_attrs(regular_latlon_dataset, ["latitude", "longitude"])
        assert attrs["spatial:dimensions"] == ["latitude", "longitude"]
        assert attrs["spatial:transform_type"] == "affine"
        assert len(attrs["spatial:transform"]) == 6
        assert attrs["spatial:shape"] == [721, 1440]
        assert len(attrs["spatial:bbox"]) == 4
        assert attrs["spatial:registration"] == "pixel"

    def test_irregular_grid_skips_transform(self, irregular_lat_dataset):
        attrs = build_spatial_attrs(irregular_lat_dataset, ["latitude", "longitude"])
        assert attrs["spatial:dimensions"] == ["latitude", "longitude"]
        assert "spatial:transform" not in attrs
        assert "spatial:transform_type" not in attrs
        assert "spatial:registration" not in attrs
        # bbox should still be present
        assert "spatial:bbox" in attrs

    def test_projected_coords(self, projected_dataset):
        attrs = build_spatial_attrs(projected_dataset, ["y_projection", "x_projection"])
        assert attrs["spatial:dimensions"] == ["y_projection", "x_projection"]
        assert attrs["spatial:transform_type"] == "affine"

    def test_missing_dims_returns_empty(self, regular_latlon_dataset):
        attrs = build_spatial_attrs(regular_latlon_dataset, ["y", "x"])
        assert attrs == {}

    def test_wrong_dim_count_returns_empty(self, regular_latlon_dataset):
        attrs = build_spatial_attrs(regular_latlon_dataset, ["latitude"])
        assert attrs == {}


# --- build_convention_attrs (orchestrator) ---


class TestBuildConventionAttrs:
    def test_full_epsg4326(self, regular_latlon_dataset):
        attrs = build_convention_attrs("EPSG:4326", regular_latlon_dataset, ["latitude", "longitude"])
        # Should have both conventions registered
        assert "zarr_conventions" in attrs
        convention_names = [c["name"] for c in attrs["zarr_conventions"]]
        assert "proj:" in convention_names
        assert "spatial:" in convention_names
        # proj: attrs
        assert attrs["proj:code"] == "EPSG:4326"
        assert "proj:wkt2" in attrs
        assert "proj:projjson" in attrs
        # spatial: attrs
        assert "spatial:dimensions" in attrs
        assert "spatial:transform" in attrs

    def test_none_crs_skips_proj(self, regular_latlon_dataset):
        attrs = build_convention_attrs(None, regular_latlon_dataset, ["latitude", "longitude"])
        assert "proj:code" not in attrs
        assert "proj:wkt2" not in attrs
        # spatial: should still be present
        convention_names = [c["name"] for c in attrs["zarr_conventions"]]
        assert "spatial:" in convention_names
        assert "proj:" not in convention_names

    def test_non_epsg_crs_skips_proj(self, regular_latlon_dataset):
        attrs = build_convention_attrs("Reduced Gaussian Grid", regular_latlon_dataset, ["latitude", "longitude"])
        assert "proj:code" not in attrs
        convention_names = [c["name"] for c in attrs["zarr_conventions"]]
        assert "proj:" not in convention_names
        # spatial: should still be present
        assert "spatial:" in convention_names

    def test_irregular_grid_partial_spatial(self, irregular_lat_dataset):
        attrs = build_convention_attrs("EPSG:4326", irregular_lat_dataset, ["latitude", "longitude"])
        assert "proj:code" in attrs
        assert "spatial:dimensions" in attrs
        assert "spatial:transform" not in attrs
        assert "spatial:bbox" in attrs

    def test_missing_dims_no_spatial(self, regular_latlon_dataset):
        attrs = build_convention_attrs("EPSG:4326", regular_latlon_dataset, ["y", "x"])
        # Only proj: should be present
        convention_names = [c["name"] for c in attrs["zarr_conventions"]]
        assert "proj:" in convention_names
        assert "spatial:" not in convention_names

    def test_no_crs_no_dims_returns_empty(self, regular_latlon_dataset):
        attrs = build_convention_attrs(None, regular_latlon_dataset, ["y", "x"])
        assert attrs == {}

    def test_wkt_fallback_when_code_unparseable(self, projected_dataset):
        """Non-EPSG crs_code falls through to WKT fallback (e.g. HRRR/RTMA)."""
        attrs = build_convention_attrs(
            "Lambert Conformal Conic",
            projected_dataset,
            ["y_projection", "x_projection"],
            crs_wkt=LCC_WKT,
        )
        # proj: convention populated via WKT fallback
        convention_names = [c["name"] for c in attrs["zarr_conventions"]]
        assert "proj:" in convention_names
        # No EPSG code for custom sphere, but WKT2 and PROJJSON present
        assert "proj:code" not in attrs
        assert "proj:wkt2" in attrs
        assert "proj:projjson" in attrs
        # spatial: also present
        assert "spatial:" in convention_names

    def test_wkt_fallback_when_code_is_none(self, projected_dataset):
        """None crs_code falls through to WKT fallback."""
        attrs = build_convention_attrs(
            None,
            projected_dataset,
            ["y_projection", "x_projection"],
            crs_wkt=LCC_WKT,
        )
        convention_names = [c["name"] for c in attrs["zarr_conventions"]]
        assert "proj:" in convention_names
        assert "proj:wkt2" in attrs

    def test_code_takes_precedence_over_wkt(self, regular_latlon_dataset):
        """When crs_code is valid, WKT fallback is not used."""
        from pyproj import CRS

        other_wkt = CRS.from_epsg(4269).to_wkt()  # NAD83
        attrs = build_convention_attrs(
            "EPSG:4326",
            regular_latlon_dataset,
            ["latitude", "longitude"],
            crs_wkt=other_wkt,
        )
        # Should use EPSG:4326, not NAD83 from WKT
        assert attrs["proj:code"] == "EPSG:4326"
        assert "WGS 84" in attrs["proj:wkt2"]
