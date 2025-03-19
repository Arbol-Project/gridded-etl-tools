import pathlib
import shutil
import cftime
import numpy as np

from gridded_etl_tools.dataset_manager import DatasetManager

#
# Functions common to more than one test that can be imported with:
#
#     from common import *
#
# Or from within a subdirectory:
#
#     from ..common import *
#

# This patch should be applied to gridded_etl_tools.convenience.Convenience.output_path
mock_output_root = pathlib.Path("_climate_test")


def run_etl(
    manager_class: DatasetManager,
    input_path: pathlib.Path,
    store: str = "local",
    s3_bucket_name: str = "zarr-dev",
    allow_overwrite: bool | None = None,
    **kwargs,
):
    """
    Construct a DatasetManager, forwarding the input path, store,
    and optionally the overwrite flag, run a transform and parse, and return
    the constructed DatasetManager.
    Useful for running a standard transform and parse on a dataset and checking its values afterward.

    Parameters
    ----------
    manager_class
        A DatasetManager implementation for your chosen dataset
    input_path
        The path from which to source raw data to build your dataset.
        Should pertain to initial, insert, or append data.
    store
        The manager store to use. 'Local' in most implementations
    s3_bucket_name
        The bucket to pull/push from. Defaults to 'zarr-dev'
    allow_overwrite
        Optionally assign the allow_overwrite flag of the dataset manager.
        If this is left as None, the dataset manager's default value will be used.

    Returns
    -------
    manager
        Instance of the given manager class after running transform and parse
    """
    # Get the manager being requested by class_name
    manager = get_manager(
        manager_class,
        input_path=input_path,
        store=store,
        s3_bucket_name=s3_bucket_name,
        allow_overwrite=allow_overwrite,
        **kwargs,
    )
    # Parse
    manager.transform_data_on_disk()
    publish_dataset = manager.transform_dataset_in_memory()
    manager.parse(publish_dataset)
    manager.publish_metadata()
    return manager


def get_manager(
    manager_class: DatasetManager,
    input_path: str = None,
    store: str = "local",
    s3_bucket_name: str = "zarr-dev",
    time_chunk: int = 50,
    allow_overwrite: bool | None = None,
    **kwargs,
):
    """
    Prepare a manager for testing

    Parameters
    ----------
    manager_class
        A DatasetManager implementation for your chosen dataset
    input_path
        The path from which to source raw data to build your dataset.
        Should pertain to initial, insert, or append data.
        Defaults to None, in case you just want a manager for unit testing.
    store
        The manager store to use. 'Local' in most implementations
    s3_bucket_name
        The bucket to pull/push from. Defaults to 'zarr-dev'
    time_chunk
        Size of the Zarr and Dask time chunks to use instead of the dataset manager's default values.
        Defaults to a low value for the small files used in testing.
        Note that to use the dataset manager's default time chunk values,
        it will need to be constructed outside of this function.
    allow_overwrite
        Optionally assign the allow_overwrite flag of the dataset manager.
        If this is left as None, the dataset manager's default value will be used.

    Returns
    -------
    manager
        A DatasetManager corresponding to your chosen manager_class
    """
    # Get the manager being requested by class_name. Only pass allow_overwrite
    # if it is set to something other than None so that the DM's default
    # value can be used otherwise.
    if allow_overwrite is not None:
        manager = manager_class(
            custom_input_path=input_path,
            s3_bucket_name=s3_bucket_name,  # This will be ignored by stores other than S3
            store=store,
            allow_overwrite=allow_overwrite,
            **kwargs,
        )
    else:
        manager = manager_class(
            custom_input_path=input_path,
            s3_bucket_name=s3_bucket_name,  # This will be ignored by stores other than S3
            store=store,
            **kwargs,
        )
    if repr(manager.store) == "Local":
        manager.store.folder = "tests"

    # Override the default (usually very large) time chunk with the given value.
    # Intended to enable testing chunking with a smaller set of times.
    manager.requested_dask_chunks["time"] = time_chunk
    manager.requested_zarr_chunks["time"] = time_chunk
    return manager


# Delete mocked output folder (for local Zarrs)
def remove_mock_output():
    if mock_output_root.exists():
        shutil.rmtree(mock_output_root)
        print("Cleaned up mocked output root", mock_output_root)


def remove_zarr_json():
    """
    Remove the generated Zarr JSON
    """
    for path in pathlib.Path(".").glob("*_zarr.json"):
        path.unlink(missing_ok=True)
        print(f"Cleaned up {path}")


def remove_dask_worker_dir():
    """
    Remove the Dask worker space directory
    """
    dask_worker_space_path = pathlib.Path("dask-worker-space")
    if dask_worker_space_path.exists():
        shutil.rmtree(dask_worker_space_path)
        print(f"Cleaned up {dask_worker_space_path}")


def remove_performance_report():
    """
    Remove the performance report
    """
    for path in pathlib.Path(".").glob("performance_report_*.html"):
        path.unlink(missing_ok=True)
        print(f"Cleaned up {path}")


def clean_up_input_paths(*args):
    """
    Clean up hourly files and original copies in paths in `args`, which is a list of pathlib.Path objects
    """
    for path in args:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            print(f"Cleaned up {path}")
        originals_path = pathlib.Path(f"{path}_originals")
        if originals_path.exists():
            shutil.rmtree(originals_path, ignore_errors=True)
            print(f"Cleaned up {originals_path}")


# Change the key used by IPNS publish to clearly mark the dataset as a test in your key list
# This will allow other tests to reference the test dataset and prevent mixups with production data


original_key = DatasetManager.key


@property
def patched_output_root(self):
    return mock_output_root


def patched_key(self):
    return f"{self.dataset_name}-{self.time_resolution}_test_initial"


original_zarr_json_path = DatasetManager.zarr_json_path


def patched_zarr_json_path(self):
    return pathlib.Path(".") / f"{self.dataset_name}_zarr.json"


original_root_stac_catalog = DatasetManager.default_root_stac_catalog


def patched_root_stac_catalog(self):
    return {
        "id": f"{self.organization}_data_catalog_test",
        "type": "Catalog",
        "title": f"{self.organization} Data Catalog - test",
        "stac_version": "1.0.0",
        "description": f"This catalog contains all the data uploaded by \
            {self.organization} that has been issued STAC-compliant metadata. \
            The catalogs and collections describe single providers. Each may contain one or multiple datasets. \
            Each individual dataset has been documented as STAC Items.",
    }


@property
def patched_update_cadence_bounds(self):
    return [np.timedelta64(3, "D"), np.timedelta64(4, "D")]


original_raw_file_to_dataset = DatasetManager.raw_file_to_dataset


def original_ds_normal(self, *args, **kwargs):
    return original_raw_file_to_dataset(self, *args, **kwargs)


def original_ds_bad_data(self, *args, **kwargs):
    orig_ds = original_raw_file_to_dataset(self, *args, **kwargs)
    orig_ds[self.data_var][:] = 1234567
    return orig_ds


def original_ds_no_time_dim(self, *args, **kwargs):
    orig_ds = original_raw_file_to_dataset(self, *args, **kwargs)
    return orig_ds.squeeze()


def original_ds_no_time_at_all(self, *args, **kwargs):
    orig_ds = original_raw_file_to_dataset(self, *args, **kwargs)
    return orig_ds.squeeze().drop_vars("time")


def original_ds_no_time_dim_in_data_var(self, *args, **kwargs):
    orig_ds = original_raw_file_to_dataset(self, *args, **kwargs)
    orig_ds[self.data_var] = orig_ds[self.data_var].squeeze()
    return orig_ds


def original_ds_bad_time(self, *args, **kwargs):
    orig_ds = original_raw_file_to_dataset(self, *args, **kwargs)
    orig_ds = orig_ds.assign_coords({"time": np.atleast_1d(np.datetime64("1850-01-01"))})
    return orig_ds


def original_ds_single_time(self, *args, **kwargs):
    orig_ds = original_raw_file_to_dataset(self, *args, **kwargs)
    orig_ds = orig_ds.sel({"time": orig_ds.time.values[-1]}).expand_dims("time")
    return orig_ds


def original_ds_esoteric_time(self, *args, **kwargs):
    orig_ds = original_raw_file_to_dataset(self, *args, **kwargs)
    time_val = orig_ds.time.values[-1]
    cftime_val = cftime.DatetimeJulian(time_val)
    orig_ds = orig_ds.sel({"time": cftime_val}).expand_dims("time")
    return orig_ds


def original_ds_random(self, *args, **kwargs):
    orig_ds = original_raw_file_to_dataset(self, *args, **kwargs)
    orig_ds = orig_ds.sel({"time": orig_ds.time.values[-1]}).expand_dims("time")
    orig_ds.precip.values = np.random.rand(*np.shape(orig_ds.precip.values))
    return orig_ds


def original_ds_null(self, *args, **kwargs):
    orig_ds = original_raw_file_to_dataset(self, *args, **kwargs)
    orig_ds = orig_ds.sel({"time": orig_ds.time.values[-1]}).expand_dims("time")
    orig_ds["precip"].values = np.full_like(orig_ds["precip"], np.nan)
    return orig_ds


original_input_files = DatasetManager.input_files


def nc4_input_files(self):
    nc4s = [str(fil) for fil in list(original_input_files(self)) if fil.suffix == ".nc4"]
    return nc4s


# NOTE disabled due to regression in fsspec capabilities
# def json_input_files(self):
#     jsons = [
#         "s3://arbol-testing/gridded/chirps/json/" + fil.name
#         for fil in list(original_input_files(self))
#         if fil.suffix == ".json"
#     ]
#     return jsons
