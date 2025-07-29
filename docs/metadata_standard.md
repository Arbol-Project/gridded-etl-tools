Climate Data Format
===================

Metadata management within managers is described in the [ETL Builders README](./etl_developers_manual.md.md#metadata). This dataset metadata is published in two ways:

* In the `attrs` or `encoding` dictionaries of each Zarr, for reference when using datasets
* Standalone STAC Items on S3, as the primary metadata file

Note that the present iteration of this library _only_ publishes STAC metadata for datasets stored on S3.

Below follow guides for specifying and using both types of metadata.

In-Zarr metadata
----------------

Each climate data Zarr stored on S3 contains essential metadata fields relevant to analysis under its Attributes (`.attrs`) and Encoding (`.encoding`) fields. Example fields include units of measurement, encoding and compression standards, chunk sizes, missing values, and data types.

Retrieving in-Zarr metadata is as simple as opening the dataset in Xarray and calling the `.attrs` or `.encoding` methods on it. 

    $ ipython
    > import xarray
    > ds = xr.open_dataset("s3://your/path_here")
    > ds.attrs
    {'coordinate reference system': 'EPSG:4326',
    'name': 'cpc_temp_max',
    'tags': ['Temperature'],
    'title': 'Climate Prediction Center (CPC)',
    'Source': 'ftp://ftp.cpc.ncep.noaa.gov/precip/wd52ws/global_temp/',
    ...
    'provider url': 'https://www.cpc.ncep.noaa.gov/',
    'dataset_title': 'CPC GLOBAL TEMP' }

    > ds.encoding
    {'_FillValue' : -9999}

Note that the overall zarr and its individual dimensions and data variables each have separate attributes and encoding dictionaries. Attributes and encoding specific to a dimension or variable will be assigned to the relevant dictionary for that dimension or variable.

    > ds.tmin.attrs
    {'units': 'degC',
    'dataset': 'CPC Global Temperature',
    'statistic': 'Maximum',
    ...
    'cell_methods': 'time: mean'}

    > ds.tmax.encoding
    {'chunks': (50, 40, 40),
    'preferred_chunks': {'time': 50, 'latitude': 40, 'longitude': 40},
    'compressor': None,
    'filters': None,
    '_FillValue': -9.96921e+36,
    'dtype': dtype('float32')}


Standalone metadata
-------------------

Accompanying each dataset's in-zarr metadata are a series of standalone metadata files in JSON format fully compatible with the [SpatioTemporal Asset Catalog (STAC) standard](https://stacspec.org/en). The main file (the "item" file in STAC parlance) is designed to be compatible with the [STAC Item metadata standard](https://github.com/radiantearth/stac-spec/tree/master/item-spec) and contains a more expansive set of metadata fields better suited to understanding the dataset's provenance and purpose and displaying it in a catalog of datasets.

Example standalone metadata files can be found via the Arbol API using the code outlined in the [Retrieval](#retrieval) section.

The dataset's item metadata also links to the previous iterations of the dataset and its corresponding item metadata if it has been updated. By jumping through these links in successive metadata JSONs a user can "rewind" the dataset or its metadata to a desired historical date. 

Datasets belonging to the same provider are linked by a separate metadata file compatible with the [STAC Collection metadata standard](https://github.com/radiantearth/stac-spec/tree/master/catalog-spec) describing this provider. All datasets are linked by a Root Catalog compatible with [STAC's Catalog metadata standard](https://github.com/radiantearth/stac-spec/tree/master/collection-spec).

A full accounting of STAC's tags and standards is beyond the scope of this README; we encourage interested readers to consult STAC's [technical documentation of the specification](https://github.com/radiantearth/stac-spec).


### STAC Metadata concepts

1. Each STAC Item, Collection, or Catalog is published at a URI string that can be used to reliably pull the latest version of a dataset's metadata. This is stored in the dataset's STAC Item under `["analytics"]["href"]["/"]`. 
2. STAC Items are regenerated with every update; to preserve the history of a dataset, the previous Item's location is saved at `["links"]["previous"]["metadata href"]` alongside the path of the corresponding dataset at `["links"]["previous"]["href"]`.
3. Arbol follows the STAC specification's preferred top-to-bottom hierarchy of a Root Catalog --> Collections --> Items. 
4. STAC Collections are instantiated for each manager and contain the properties common to that manager's datasets. Links to the individual datasets are identified as `"rel" = "item"` under `["links"]`.
6. The Arbol Root STAC Catalog contains an overall description of Arbol and links to each STAC Collection.

The Pangeo community has been further developing a `datacube` extension for STAC that better describes N-dimensional datasets (see [discussion here](https://discourse.pangeo.io/t/stac-and-earth-systems-datasets/1472)). This extension is powerful but requires additional work to implement, maintain, and read. Given that all of our N-dimensional datasets in practice only have a time dimension we elected not to implement this extension, for now at least.


### Full spec

A standalone metadata file will always be in JSON, but the keys present can vary between sets depending on the climate category and the original publisher of the data

The metadata file contains information necessary for accessing and parsing the files in the set. For example, a metadata file for PRISM temperature data looks like so:

```json
    {
    "assets":
        {
            "zmetadata": {
                            "description": "Consolidated metadata file for "
                                            "prism-tmin Zarr store, readable as a "
                                            "Zarr dataset by Xarray",
                            "href": {"/": "bafyreih6reo6t6gyluzp5tjtve2gn6tktgnvwjuggey3diyspppq4p2jci"},
                            "roles": ["metadata", "zarr-consolidated-metadata"],
                            "title": "prism-tmin",
                            "type": "application/json"
                            }
        },
    "bbox": [-125, 24.083333333126994, -66.499999999532, 49.916666666667],
    "collection": "PRISM",
    "geometry": 
        {
        "type": "Polygon",  
        "coordinates": [[ [-66.499999999532,  24.083333333126994], 
                    [-66.499999999532, 49.916666666667], 
                    [-125.0, 49.916666666667], [-125.0, 24.083333333126994], 
                    [-66.499999999532, 24.083333333126994] ]]
                },
    "id": "prism-tmin",
    "links": 
        [
        {"href": "k2k4r8p5u22nbu463uxar89syw9cdhctqc83jw8zdacmdcnz90udr1gv",
                "rel": "parent",
                "title": "Parameter-elevation Regressions on Independent Slopes "
                        "Model (PRISM)",
                "type": "application/geo+json"},
            {"href": "bafyreiepvo6gvgxiydymjztldb35iuj7zpiiplybpilepce5laetthiija",
                "metadata href": {"/": "bafyreibfllvchxdysyzta26pa3s2lmfrk3wfbihxgn7e7bqh7omldqmknq"},
                "rel": "prev",
                "title": "prism-tmin",
                "type": "application/geo+json"},
            {"href": "k2k4r8p0cex0iwyj2i81u21ruy36ynqowxv5m3abw8akgioc246lbchg",
                "rel": "self",
                "title": "prism-tmin metadata",
                "type": "application/geo+json"}
        ],
    "properties": 
        {
        "CF convention": "CF-1.5",
                    "Fill value": -9999,
                    "Zarr chunk size": {"latitude": 27,
                                        "longitude": 27,
                                        "time": 625},
                    "array_size": {"latitude": 621,
                                "longitude": 1405,
                                "time": 15392},
                    "climate variable": "temperature min",
                    "created": "2022-10-26T07:30:34Z",
                    "coordinate reference system" : "EPSG:4269",
                    "dataset description": "PRISM climate datasets incorporate a "
                                        "variety of modeling techniques and are "
                                        "available at multiple spatial/temporal "
                                        "resolutions, covering the period from "
                                        "1895 to the present. The grid cells "
                                        "analyzed are the standard PRISM 4km\n"
                                        "Rather than calculating days based on "
                                        "the 24-hour period ending at midnight "
                                        "local time, PRISM defines a 'day' as "
                                        "the 24 hours ending at 12:00 Greenwich "
                                        "Mean Time (GMT, or 7:00am Eastern "
                                        "Standard Time). This means that PRISM "
                                        "data for May 26, for example, actually "
                                        "refers to the 24 hours ending at "
                                        "7:00am EST on May 26.More information "
                                        "at "
                                        "https://prism.oregonstate.edu/documents/PRISM_datasets.pdf",
                    "date range": ["2022110500", "2023022100"],
                    "dtype": "float32",
                    "end_datetime": "2023-02-21T00:00:00Z",
                    "finalization date": "2022-07-31T00:00:00",
                    "input history JSON documentation": "The JSON retrievable at "
                                                        "'input history cid' "
                                                        "stores the modification "
                                                        "times of the original "
                                                        "files used for input when "
                                                        "they were last downloaded "
                                                        "from the PRISM source "
                                                        "directory. It contains a "
                                                        "dict where the keys are "
                                                        "the day the file "
                                                        "represents and the values "
                                                        "are the last modification "
                                                        "time on the PRISM server. "
                                                        "If the file was marked "
                                                        "stable on the PRISM "
                                                        "server, the value is the "
                                                        "string 'stable' instead "
                                                        "of the modification time. "
                                                        "This can be used to "
                                                        "determine whether a file "
                                                        "most likely has new data "
                                                        "and should be downloaded.",
                    "input history cid": "bafyreid3r5rwibrufaztqcl5u7zfrynk3krettxnja74du5jb3xb2qfg2y",
                    "spatial precision": 0.01,
                    "spatial resolution": 0.04166667,
                    "start_datetime": "1981-01-01T00:00:00Z",
                    "tags": ["Temperature"],
                    "temporal resolution": "daily",
                    "unit of measurement": "degC",
                    "update cadence": "daily",
                    "updated": "2023-02-23T07:54:03Z"
        },
    "stac_version": "1.0.0",
    "type": "Feature"
    }
```
