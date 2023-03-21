
Creating new dataset sources
----------------------------

New climate sources can be added by creating a new manager class for that source within a python script dedicated to it -- e.g. [CPC.py](etls/managers/cpc.py). The recommended practice is to create child classes for each climate variable (minimum temperature, water salinity, etc.) extracted from that source -- as in the [CHIRPS](etls/managers/CHIRPS.py) manager. 

See the [ETL Developer's Manual](etls/README.md) for further instructions. 

A [sources.py](sources.py) file prepares each dataset manager for usage as an argument to the [generate_zarr.py](generate_zarr.py) script when it is run. You must import the newly created manager class in the [sources.py](sources.py) file before it can be provided as an argument to `generate_zarr.py`. 

Thus to run `python generate_zarr.py your_new_dataset_subclass` you must first insert the following in the header of `sources.py`:

    from etls.managers.your_new_dataset import your_new_dataset_subclass

If you create an additional subclass for that dataset, import it alongside like so

    from etls.managers.your_new_dataset import your_new_dataset_subclass, your_even_newer_dataset_subclass