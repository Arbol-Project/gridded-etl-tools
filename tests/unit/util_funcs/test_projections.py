import pathlib

from pyproj import Transformer, CRS
import pytest
import xarray as xr

from gridded_etl_tools.util_funcs.projections import assign_crs_to_dataset, drop_coord_encoding

# specific to RTMA projection
EARTH_RADIUS = 6371200.0
GRID_MAPPING_NAME = "lambert_conformal_conic"

# downloaded from s3://noaa-rtma-pds/rtma2p5.20250206/rtma2p5.2025020609.pcp.184.grb2
# data type chosen for precip's small file size. timestamp chosen arbitrarily
SOURCE_FILE_PATH = pathlib.Path(__file__).parents[1] / "inputs" / "rtma_pcp.grib"


@pytest.fixture
def rtma_ds():
    ds = xr.open_dataset(SOURCE_FILE_PATH)

    # metpy expects only 1 time coord
    return ds.drop_vars("valid_time")


def test_assign_lambert_crs_to_grib(rtma_ds):
    data_var = list(rtma_ds.data_vars.keys())[0]
    cf_dict = {
        "semi_major_axis": EARTH_RADIUS,
        "semi_minor_axis": EARTH_RADIUS,
        "grid_mapping_name": "lambert_conformal_conic",
        "standard_parallel": [
            rtma_ds[data_var].attrs["GRIB_Latin1InDegrees"],
            rtma_ds[data_var].attrs["GRIB_Latin2InDegrees"],
        ],
        "latitude_of_projection_origin": rtma_ds[data_var].attrs["GRIB_LaDInDegrees"],
        "longitude_of_central_meridian": rtma_ds[data_var].attrs["GRIB_LoVInDegrees"],
    }
    projected_ds = assign_crs_to_dataset(rtma_ds, cf_dict)

    dims_mapping = {"x": "x_projection", "y": "y_projection"}
    assert set(projected_ds.dims.keys()) == set(dims_mapping.values())

    for old_dim, new_dim in dims_mapping.items():
        assert len(rtma_ds[old_dim]) == len(projected_ds[new_dim])

    # test nested crs attributes
    assert "crs" in projected_ds.attrs
    assert "crs_wkt" in projected_ds.attrs["crs"]

    # test we can project back to correct lat/lon
    wkt_string = projected_ds.attrs["crs"]["crs_wkt"]
    src_crs = CRS.from_wkt(wkt_string)
    dst_crs = CRS.from_epsg(4326)  # WGS84
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    # chosen arbitrarily
    x_index, y_index = 47, 524

    selected_point = projected_ds.isel(x_projection=x_index, y_projection=y_index)
    x_value = selected_point["x_projection"].values.item()
    y_value = selected_point["y_projection"].values.item()

    expected_lon = selected_point["longitude"].item()
    # standardize to [-180, 180), which pyproj and gridded_etl_tools use
    expected_lon = ((expected_lon + 180) % 360) - 180
    expected_lat = selected_point["latitude"].item()

    assert transformer.transform(x_value, y_value) == pytest.approx((expected_lon, expected_lat))


def test_drop_coord_encoding(rtma_ds):
    encodings_that_should_be_dropped = ["chunks", "preferred_chunks", "_FillValue", "missing_value", "filters"]
    coords_to_drop = ["latitude", "longitude"]
    drop_coord_encoding(rtma_ds, coords_to_drop)
    for coord in coords_to_drop:
        for encoding in encodings_that_should_be_dropped:
            assert rtma_ds[coord].encoding.get(encoding) is None
