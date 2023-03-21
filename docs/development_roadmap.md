Development roadmap
==================

The initial version of this repository is a minimum viable product for others to use; hence we released it as version 0.1.0. We have several short- and medium-term goals to further develop its capabilities and make it easier to use for ourselves and others. Each of these would constitute a minor and perhaps a major release.

We reserve the right to revise these timelines and/or their anticipated impacts (minor/major release) as Arbol's priorities shift and/or the scope of the problem grows.

### PyPI package

Users currently must built and operate ETLS by importing the directory in question. Our intention is for the library to function as a standalone PyPI package that can be imported and used like `pandas`, `xarray`, or other standard data manipulation frameworks.

This is the highest priority development goal and we anticipate its release within the next 3 months. These changes will represent a minor release as they may require reworking imports, but shouldn't otherwise change code behavior.

### Exports to S3

The repository was initially built to successfully export gridded climate data to IPFS in its "load" step and meets that condition. However, we recognize users may want to export some or all datasets to traditional cloud storage destinations. We are currently working on modifications to the `zarr_methods.py` script that will allow for exports to Amazon's S3 object storage.

This is a high priority and we anticipate its release within the next 6 months. These changes will represent a major release as they will break existing ETLs.

### Reconfigure workflow for instantiating ETLs

As currently configured the library assumes ETL scripts are stored within the repository and instantiated manually via the `generate_zarr.py` script. This works well enough but means the library cannot really function as an independent set of gridded climate data ETL building utilities. We would like to separate out the logic for triggering ETLs so it can be orchestrated separately from the ETL building methods, which in turn become simply a set of utilities to import to any given climate data ETL.

This is a high priority and we anticipate its release within the next 6 months. These changes will represent a major release as they will break existing ETLs.

### Work with forecast data

While Zarrs permit users to work across many dimensions, the library as constructed actually only handles three -- latitude, longitude, and time. We are keen to allow the use of a fourth dimension, forecast date, to allow handling of climate forecast data in Zarrs. Additional dimensions are also desirable but less urgently needed so we will consider them after seeing the level of effort it takes to work with forecasts.

This is a medium priority objective and we anticipate its release within the next 9-12 months. These changes will represent a minor release as they won't break existing ETLs, at least as currently scoped.

### Refactor docs as ReadTheDocs

The documentation for this repository is not the easiest to navigate due to being presented as a collection of markdown files. We would like to refactor the docs as a standalone ReadTheDocs instance where they can be more easily navigated -- not to mention better annotated by Sphinx's powerful markdown tools.

This is a medium priority objective and we anticipate its release within the next 9-12 months. Because the readthedocs would stand apart from the repo it won't require a release of any sort, except to point users to the newly activate documentation.