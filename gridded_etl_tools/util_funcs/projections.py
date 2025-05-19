from typing import Any

import xarray as xr


def assign_crs_to_dataset(dataset: xr.Dataset, cf_dict: dict[str, Any]) -> xr.Dataset:
    """
    Assigns projected x/y dimensions and crs attributes to a dataset with existing lon/lat coords.


    Parameters
    ----------
    dataset : xr.Dataset
        Input dataset
    cf_dict : dict[str, Any]
        dict with keys and values formatted according to
        https://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html
        e.g., for a specific lambert comformal dataset:

        {
            "semi_major_axis": 6371200.0,
            "semi_minor_axis": 6371200.0,
            "grid_mapping_name": "lambert_conformal_conic",
            "standard_parallel": [25.0, 25.0],
            "latitude_of_projection_origin": 25.0,
            "longitude_of_central_meridian": 265.0,
        }

    Returns
    -------
    xr.Dataset
        Dataset with projected x/y dimensions and crs attrs added
    """
    dataset_with_crs = dataset.metpy.assign_crs(cf_dict)
    dataset_with_x_y = dataset_with_crs.metpy.assign_y_x()
    metpy_mapping = dataset_with_x_y.metpy_crs.values.item()
    crs_attrs = metpy_mapping.to_dict()
    crs_attrs["crs_proj4"] = metpy_mapping.to_pyproj().to_proj4()

    dataset_without_metpy = dataset_with_x_y.drop_vars("metpy_crs")
    dataset_without_metpy.attrs["crs"] = crs_attrs

    return dataset_without_metpy.rename({"x": "x_projection", "y": "y_projection"})


def drop_coord_encoding(dataset: xr.Dataset, coords: list[str]):
    """
    Remove various encodings from the coords in a dataset. This happens automatically
    for dimensions due to metadata.Metadata.remove_unwanted_fields, but this function
    must be applied in postprocess to coords that aren't dims

    Parameters
    ----------
    dataset : xr.Dataset
        Input dataset
    coords : list[str]
        Coordinates to have encoding keys dropped from
    """

    for coord in coords:
        dataset[coord].encoding.pop("chunks", None)
        dataset[coord].encoding.pop("preferred_chunks", None)
        dataset[coord].encoding.pop("_FillValue", None)
        dataset[coord].encoding.pop("missing_value", None)
        dataset[coord].encoding.pop("filters", None)
