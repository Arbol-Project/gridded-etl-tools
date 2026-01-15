
Running an ETL
==============

As described in the initial repo [README](../README.md), running an ETL can be done by either running the manager script as a standalone script or invoking DatasetManager's `run_etl` function within a notebook.

Users must specify a storage medium (a.k.a. "store") for the generated Zarr using the **store** flag/kwarg. Acceptable stores are "s3" for Amazon S3 and "local" for a local install under `<root_directory>/climate/your/etl/dataset.zarr`. If using S3, the environment variables `AWS_ACCESS_KEY_ID`and `AWS_SECRET_ACCESS_KEY` must be set in the `~/.aws/credentials` file or manually before the ETL is initialized.

## Basic process

#### Notebook / interactive environment

An ETL can be run within a notebook or interactive environment (e.g. ipython) as part of a larger workflow. In the following example, it is assumed the manager script lives in a directory underneath the notebook called `managers`.

    # other imports
    from .managers.manager_script import MyNewETLDataset

    etl = MyNewETLDataset(store='s3') # exports to S3
    etl.extract() # download the entire dataset, or just updates to the existing dataset
    ds = etl.transform()  # transform raw downloaded data into a single "virtual" Zarr of new or updated data
    etl.parse(ds)  # push this single Zarr it to the storage medium of choice
    
    # move onwards to retrieval (or whatever else your script demands)

An alternative configuration triggering a parse to Amazon's S3 cloud file store would first specify access and secret keys as environment variables, then specify the s3 store and corresponding s3 bucket name

```python
    # other imports
    import os
    from .managers.manager_script import MyNewETLDataset

    os.environ['AWS_ACCESS_KEY_ID'] = <ACCESS_KEY_STR>
    os.environ['AWS_SECRET_ACCESS_KEY'] = <SECRET_KEY_STR>

    etl = MyNewETLDataset(store='s3', s3_bucket_name='my_bucket_name') # exports to s3
    etl.extract()  # download the entire dataset, or just updates to the existing dataset
    ds = etl.transform()  # transform raw downloaded data into a single "virtual" Zarr of new or updated data
    etl.parse(ds)  # push this single Zarr it to the storage medium of choice
    
    # move onwards to retrieval (or whatever else your script demands)
```

Finally, to export to your local file system run the following

```python
    from .managers.manager_script import MyNewETLDataset

    etl = MyNewETLDataset(store='local') # exports to local file system
    etl.extract() # download the entire dataset, or just updates to the existing dataset
    ds = etl.transform()  # transform raw downloaded data into a single "virtual" Zarr of new or updated data
    etl.parse(ds)  # push this single Zarr it to the storage medium of choice
```

#### Passing parameters

Note that if desired, kwargs can be passed to the ETL manager, `extract` or `parse`.

For example, if you want to allow your ETL to rebuild a dataset from scratch, overwriting any existing data in the process (a potentially dangerous operation!) you would invoke it like so

```python
    from examples.managers.chirps import CHIRPSPrelim05
    etl = CHIRPSPrelim05(store='s3', rebuild_requested=True, allow_overwrite=True)
```

Or alternately, if you wished to write locally but to a custom location you would invoke it like so

```python
    from examples.managers.chirps import CHIRPSPrelim05
    etl = CHIRPSPrelim05(store='local', custom_output_path='~/path/to/desired/store/location/chirps_prelim_05.zarr')
```

In another example, if you only wanted to download (not parse) data for the first half of 2022 you would pass kwargs to the `extract` function

```python
    from examples.managers.chirps import CHIRPSPrelim05
    etl = CHIRPSPrelim05(store='s3')
    etl.extract(date_range=[datetime.datetime(2022, 1, 1, 0), datetime.datetime(2022, 6, 31, 0)], only_update_input=True)
```

Consult the docstring for `dataset_manager.__init__` and `extract` to see acceptable parameters you can pass.

#### Looping a parse

In most cases, the ETL is expected to run each of its phases once within the scope of the program. However, it may be useful to run one or more of the phases in a loop when, for example, resources like disk space, RAM, or CPU are low. This library is able to reset a `DatasetManager` to re-run an ETL, but currently the object must be deleted and re-initialized before re-running.

In the following example, the source data for 1981 to 2025 is downloaded and parsed four years at time. After every run, the `DatasetManager` is deleted, the source data folder is removed completely (to conserve space), and the `DatasetManager` is re-initialized for another ETL. The first run will create a new Zarr on S3, and subsequent runs will append the new data to the existing Zarr.

```python
if __name__ == "__main__":

    batch_start_date = datetime.datetime(1981, 1, 1)
    while batch_start_date <= datetime.datetime(2025, 1, 1):

        dm = chirps.CHIRPS3FinalRnl(
                store="s3",
                s3_bucket_name="my-bucket",
                output_zarr3=False,
                use_local_zarr_jsons=True,
                skip_pre_parse_nan_check=True,
                align_update_chunks=True,
            )
        dm.log_to_file()
        dm.extract(date_range=(batch_start_date, datetime.datetime(batch_start_date.year + 3, 12, 31)))
        dm.transform_data_on_disk()
        dataset = dm.transform_dataset_in_memory()
        dataset = dataset.chunk(dm.requested_dask_chunks)
        dm.parse(dataset)
        dm.publish_metadata()
        del dm

        shutil.rmtree("my-datasets/")
        batch_start_date = datetime.datetime(batch_start_date.year + 4, 1, 1)
```

## Retrieving a dataset

The retrieval method for a dataset depends on the store employed. Below we provide an example of retrieval for each store.

#### S3

If we had instead exported to s3, we would follow a different retrieval pattern

```python
    import xarray, s3fs

    s3.ls("my_bucket_name", refresh=True)
    ['my_bucket_name/chirps_final_25-daily.zarr']

    mapper = s3fs.S3Map(root='s3://my_bucket_name/chirps_final_25-daily.zarr', s3=s3)
    ds = xarray.open_zarr(mapper)
    ds
    <xarray.Dataset>
    Dimensions:    (latitude: 120, longitude: 300, time: 5736)
    Coordinates:
      * latitude   (latitude) float32 20.12 20.38 20.62 20.88 ... 49.38 49.62 49.88
      * longitude  (longitude) float32 -129.9 -129.6 -129.4 ... -55.62 -55.38 -55.12
      * time       (time) datetime64[ns] 2007-01-01 2007-01-02 ... 2022-09-14
    Data variables:
      precip     (time, latitude, longitude) float32 dask.array<chunksize=(1769, 24, 24), meta=np.ndarray>
    ...
```

#### Local

If we had instead exported to the local file system, we would retrieve normally like so

```python
    import xarray

    ds = xarray.open_zarr("<root>/climate/chirps/final/25/chirps_final_25.zarr")
    ds
    <xarray.Dataset>
    Dimensions:    (latitude: 120, longitude: 300, time: 5736)
    Coordinates:
      * latitude   (latitude) float32 20.12 20.38 20.62 20.88 ... 49.38 49.62 49.88
      * longitude  (longitude) float32 -129.9 -129.6 -129.4 ... -55.62 -55.38 -55.12
      * time       (time) datetime64[ns] 2007-01-01 2007-01-02 ... 2022-09-14
    Data variables:
      precip     (time, latitude, longitude) float32 dask.array<chunksize=(1769, 24, 24), meta=np.ndarray>
```
