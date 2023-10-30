import pathlib
import shutil
import numpy as np

from gridded_etl_tools.dataset_manager import DatasetManager
from .conftest import manager_class
#
# Functions common to more than one test that can be imported with:
#
#     from common import *
#
# Or from within a subdirectory:
#
#     from ..common import *
#

def get_manager(
    manager_class: DatasetManager,
    input_path: str = "",
    store: str = "local",
    time_chunk: int = 50,
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
        Optional paramter to modify the size of time_chunks. Necessary for some ETLs. Defaults to 50.

    Returns
    -------
    manager
        A DatasetManager corresponding to your chosen manager_class
    """
    # Get the manager being requested by class_name
    manager = manager_class(
        custom_input_path=input_path,
        s3_bucket_name="zarr-dev",  # This will be ignored by stores other than S3
        store=store
    )
    # Overriding the default (usually very large) time chunk to enable testing chunking with a smaller set of times
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
    return f"{self.name()}-{self.temporal_resolution()}_test_initial"


original_zarr_json_path = DatasetManager.zarr_json_path


def patched_zarr_json_path(self):
    return pathlib.Path(".") / f"{self.name()}_zarr.json"


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
            Each individual dataset has been documented as STAC Items."
        }

def patched_irregular_update_cadence(self):
    return [np.timedelta64(3,"D"), np.timedelta64(4,"D")]