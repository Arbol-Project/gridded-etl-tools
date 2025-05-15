import xarray as xr


def assign_lambert_crs_to_grib(dataset: xr.Dataset, earth_radius: float, data_var: str) -> xr.Dataset:
    """
    Assigns projected x/y dimensions and crs attributes to a dataset using existing lon/lat coords.
    Assumes the dataset uses a lambert conformal conic projection and has relevant attributes
    from the source GRIB.


    Args:
        dataset (xr.Dataset): Input dataset
        earth_radius (float): Radius of earth used in projection (depends on dataset)
        data_var (str): xarray data varaible namet hat contains the projection attributes

    Returns:
        xr.Dataset: Dataset with projected x/y dimensions and crs attrs added
    """
    dataset_with_crs = dataset.metpy.assign_crs(
        {
            "semi_major_axis": earth_radius,
            "semi_minor_axis": earth_radius,
            "grid_mapping_name": "lambert_conformal_conic",
            "standard_parallel": [
                dataset[data_var].attrs["GRIB_Latin1InDegrees"],
                dataset[data_var].attrs["GRIB_Latin2InDegrees"],
            ],
            "latitude_of_projection_origin": dataset[data_var].attrs["GRIB_LaDInDegrees"],
            "longitude_of_central_meridian": dataset[data_var].attrs["GRIB_LoVInDegrees"],
        }
    )
    dataset_with_x_y = dataset_with_crs.metpy.assign_y_x()

    metpy_mapping = dataset.metpy_crs.values.item()
    crs_attrs = metpy_mapping.to_dict()
    crs_attrs["crs_proj4"] = metpy_mapping.to_pyproj().to_proj4()

    dataset_without_metpy = dataset_with_x_y.drop_vars("metpy_crs")
    dataset_without_metpy.attrs["crs"] = crs_attrs

    return dataset_without_metpy.rename({"x": "x_projection", "y": "y_projection"})


def drop_coord_encoding(dataset: xr.Dataset, coords: list[str]) -> xr.Dataset:
    """
    Remove various encodings from the coords in a dataset. This happens automatically
    for dimensions due to metadata.Metadata.remove_unwanted_fields, but this function
    must be applied in postprocess to coords that aren't dims

    Args:
        dataset (xr.Dataset): Input dataset
        coords (list[str]): Coordinates to have encoding keys dropped from

    Returns:
        xr.Dataset: Dataset without messy encodings on the specified coords
    """

    for coord in coords:
        dataset[coord].encoding.pop("chunks", None)
        dataset[coord].encoding.pop("preferred_chunks", None)
        dataset[coord].encoding.pop("_FillValue", None)
        dataset[coord].encoding.pop("missing_value", None)
        dataset[coord].encoding.pop("filters", None)

    return dataset
