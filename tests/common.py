import pathlib
import shutil
import numpy as np
import typing

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


def initial(
    manager_class: DatasetManager,
    input_path: pathlib.Path,
    store: str = "local",
    allow_overwrite: typing.Optional[bool] = None,
    **kwargs
):
    """
    Construct a DatasetManager, forwarding the input path, store, and optionally the overwrite flag, run a transform and parse, and return
    the constructed DatasetManager. Useful for running a standard transform and parse on a dataset and checking its values afterward.

    Parameters
    ----------
    manager_class
        A DatasetManager implementation for your chosen dataset
    input_path
        The path from which to source raw data to build your dataset. Should pertain to initial, insert, or append data.
    store
        The manager store to use. 'Local' in most implementations
    allow_overwrite
        Optionally assign the allow_overwrite flag of the dataset manager. If this is left as None, the dataset manager's default value will be used.
    
    Returns
    -------
    manager
        Instance of the given manager class after running transform and parse
    """
    # Get the manager being requested by class_name
    manager = get_manager(manager_class, input_path=input_path, store=store, allow_overwrite=allow_overwrite, **kwargs)
    # Parse
    manager.transform()
    manager.parse()
    manager.publish_metadata()
    return manager


def get_manager(
    manager_class: DatasetManager,
    input_path: str,
    store: str = "local",
    time_chunk: int = 50,
    allow_overwrite: typing.Optional[bool] = None,
    **kwargs,
):
    """
    Prepare a manager for testing

    Parameters
    ----------
    manager_class
        A DatasetManager implementation for your chosen dataset
    input_path
        The path from which to source raw data to build your dataset. Should pertain to initial, insert, or append data.
    store
        The manager store to use. 'Local' in most implementations
    time_chunk
        Size of the Zarr and Dask time chunks to use instead of the dataset manager's default values. Defaults to a low value for the small files
        used in testing. Note that to use the dataset manager's default time chunk values, it will need to be constructed outside of this function.
    allow_overwrite
        Optionally assign the allow_overwrite flag of the dataset manager. If this is left as None, the dataset manager's default value will be used.

    Returns
    -------
    manager
        A DatasetManager corresponding to your chosen manager_class
    """
    # Get the manager being requested by class_name. Only pass allow_overwrite if it is set to something other than None so that the DM's default
    # value can be used otherwise.
    if allow_overwrite is not None:
        manager = manager_class(
            custom_input_path=input_path,
            s3_bucket_name="zarr-dev",  # This will be ignored by stores other than S3
            store=store,
            allow_overwrite=allow_overwrite,
            **kwargs
        )
    else:
        manager = manager_class(
            custom_input_path=input_path,
            s3_bucket_name="zarr-dev",  # This will be ignored by stores other than S3
            store=store,
            **kwargs
        )

    # Override the default (usually very large) time chunk with the given value. Intended to enable testing chunking with a smaller set of times.
    manager.requested_dask_chunks["time"] = time_chunk
    manager.requested_zarr_chunks["time"] = time_chunk
    return manager


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


# Save the original IPNS publish function, so it can be mocked to force offline to True when the patched
# IPNS publish is applied.


original_ipns_publish = DatasetManager.ipns_publish


def offline_ipns_publish(self, key, cid, offline=False):
    """
    A mock version of `DatasetManager.ipns_publish` which forces offline mode so tests can run faster.
    """
    return original_ipns_publish(self, key, cid, offline=True)


def empty_ipns_publish(self, key, cid, offline=False):
    """
    A mock version of `DatasetManager.ipns_publish` which forces offline mode so tests can run faster.
    """
    return self.info("Skipping IPNS publish to preserve initial test dataset")


# Change the json_key used by IPNS publish to clearly mark the dataset as a test in your key list
# This will allow other tests to reference the test dataset and prevent mixups with production data


original_json_key = DatasetManager.json_key


def patched_json_key(self):
    return f"{self.dataset_name}-{self.time_resolution}_test_initial"


original_zarr_json_path = DatasetManager.zarr_json_path


def patched_zarr_json_path(self):
    return pathlib.Path(".") / f"{self.dataset_name}_zarr.json"


original_root_stac_catalog = DatasetManager.default_root_stac_catalog


def patched_root_stac_catalog(self):
    return {
        "id": f"{self.host_organization()}_data_catalog_test",
        "type": "Catalog",
        "title": f"{self.host_organization()} Data Catalog - test",
        "stac_version": "1.0.0",
        "description": f"This catalog contains all the data uploaded by \
            {self.host_organization()} that has been issued STAC-compliant metadata. \
            The catalogs and collections describe single providers. Each may contain one or multiple datasets. \
            Each individual dataset has been documented as STAC Items.",
    }


@property
def patched_update_cadence_bounds(self):
    return [np.timedelta64(3, "D"), np.timedelta64(4, "D")]

original_get_original_ds = DatasetManager.get_original_ds

def original_ds_normal(self):
    return self.get_original_ds()

def original_ds_bad_data(self):
    orig_ds = self.get_original_ds()    
    orig_ds[self.data_var()][:] = 1234567
    return orig_ds

def original_ds_no_time(self):
    orig_ds = self.get_original_ds()
    return orig_ds.drop("time")

def original_ds_bad_time(self):
    orig_ds = self.get_original_ds()
    orig_ds = orig_ds.assign_coords({"time" : np.datetime64("1850-1-1")})
    return orig_ds

def nc4_input_files(self):
    return [fil for fil in self.input_files() if fil.endswith() == '.nc4']

def json_input_files(self):
    return [fil for fil in self.input_files() if fil.endswith() == '.json']