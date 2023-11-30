
Running an ETL
==============

As described in the initial repo [README](../README.md), running an ETL can be done by either running the manager script as a standalone script or invoking DatasetManager's `run_etl` function within a notebook.

Users must specify a storage medium (a.k.a. "store") for the generated Zarr using the **store** flag/kwarg. Acceptable stores are "ipld" for IPFS, "s3" for Amazon S3, and "local" for a local install under `<root_directory>/climate/your/etl/dataset.zarr`. If using S3, the environment variables `AWS_ACCESS_KEY_ID`and `AWS_SECRET_ACCESS_KEY` must be set in the `~/.aws/credentials` file or manually before the ETL is initialized.

## Basic process

#### Notebook / interactive environment

An ETL can be run within a notebook or interactive environment (e.g. ipython) as part of a larger workflow. In the following example, it is assumed the manager script lives in a directory underneath the notebook called `managers`.

    # other imports
    from .managers.manager_script import MyNewETLDataset

    etl = MyNewETLDataset(store='ipld') # exports to IPFS
    etl.extract() # download the entire dataset, or just updates to the existing dataset
    etl.transform()  # transform raw downloaded data into a single "virtual" Zarr of new or updated data
    etl.parse()  # push this single Zarr it to the storage medium of choice
    
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
    etl.transform()  # transform raw downloaded data into a single "virtual" Zarr of new or updated data
    etl.parse()  # push this single Zarr it to the storage medium of choice
    
    # move onwards to retrieval (or whatever else your script demands)
```

Finally, to export to your local file system run the following

```python
    from .managers.manager_script import MyNewETLDataset

    etl = MyNewETLDataset(store='local') # exports to IPFS
    etl.extract() # download the entire dataset, or just updates to the existing dataset
    etl.transform()  # transform raw downloaded data into a single "virtual" Zarr of new or updated data
    etl.parse()  # push this single Zarr it to the storage medium of choice
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
    etl = CHIRPSPrelim05(store='ipld')
    etl.extract(date_range=[datetime.datetime(2022, 1, 1, 0), datetime.datetime(2022, 6, 31, 0)], only_update_input=True)
```

Consult the docstring for `dataset_manager.__init__` and `extract` to see acceptable parameters you can pass.


## Retrieving a dataset

The retrieval method for a dataset depends on the store employed. Below we provide an example of retrieval for each store.

#### IPFS 

If `store` is set to "ipld", a key corresponding to the dataset's name will be added to the IPFS node's key list of IPNS "names".

    $ ipfs key list -l | grep chirps_final_25
    [IPNS HASH]    chirps_final_25-daily

Follow that IPNS hash to get to the standalone metadata for the generated CPC U.S. precipitation Zarr

    $ ipfs dag get $(ipfs name resolve $(ipfs key list -l | grep chirps_final_25 | cut -d ' ' -f1)) | python -m json.tool
    {
      "assets": {
        "zmetadata": {
          "description": "Consolidated metadata file for chirps_final_25 Zarr store, readable as a Zarr dataset by Xarray",
    ...

In that metadata, there is a link to the Zarr in `assets->zmetadata->href`

    "href": {
      "/": "bafyreibnfezcttd5zcjlp74lohe3ieqmtc73cybvfhh6ugzusxc2yi6ky4"
    }
    
The Zarr containing the climate data can be opened with `xarray` and `ipldstore`, which were installed during the virtual environment setup

```python
    import xarray, ipldstore
    mapper = ipldstore.get_ipfs_mapper()
    mapper.set_root("bafyreibnfezcttd5zcjlp74lohe3ieqmtc73cybvfhh6ugzusxc2yi6ky4")
    xarray.open_zarr(mapper, consolidated=False)
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
For further help invoking scripts, read [the ETL running manual](docs/running_an_etl.md). To understand the various optional invocation flags, consult the docstring for the `run_etl` function in the [dataset_manager script](dataset_manager.py#296)

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
