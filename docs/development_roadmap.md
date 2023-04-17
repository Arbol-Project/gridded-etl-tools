Development roadmap
==================

The initial version of this repository is a minimum viable product for others to use; hence we released it as version 0.1.0. We have several short- and medium-term goals to further develop its capabilities and make it easier to use for ourselves and others. Each of these would constitute a minor and perhaps a major release.

We reserve the right to revise these timelines and/or their anticipated impacts (minor/major release) as Arbol's priorities shift and/or the scope of the problem grows.

### PyPI package

Users currently must built and operate ETLS by importing the directory in question. Our intention is for the library to function as a standalone PyPI package that can be imported and used like `pandas`, `xarray`, or other standard data manipulation frameworks.

Unfortunately, the current iteration of this library relies on custom patches to the `ipldstore` and `kerchunk` libraries hosted on github. PyPI does not allow custom installs from github for its packages so we need to have our changes merged into the open source libraries before we can publish the library. We will prioritize this as time allows but it's not fully under our control.

### Allow publication of STAC metadata for non-IPFS stores

As currently configured the library only publishes the more expansive standalone STAC metadata files for datasets stored on IPFS. This is less than ideal as datasets on s3 or the local file system have incomplete metadata relative to IPFS-hosted datasets.

This is a high priority and we anticipate its release within the next 3 months. These changes will represent a minor release as they will significantly reconfigure metadata operations for non-IPFS datasets.

### Work with forecast data

While Zarrs permit users to work across many dimensions, the library as constructed actually only handles three -- latitude, longitude, and time. We are keen to allow the use of a fourth dimension, forecast date, to allow handling of climate forecast data in Zarrs. Additional dimensions are also desirable but less urgently needed so we will consider them after seeing the level of effort it takes to work with forecasts.

This is a medium priority objective and we anticipate its release within the next 9-12 months. These changes will represent a minor release as they won't break existing ETLs, at least as currently scoped.

### Refactor docs as ReadTheDocs

The documentation for this repository is not the easiest to navigate due to being presented as a collection of markdown files. We would like to refactor the docs as a standalone ReadTheDocs instance where they can be more easily navigated -- not to mention better annotated by Sphinx's powerful markdown tools.

This is a medium priority objective and we anticipate its release within the next 9-12 months. Because the readthedocs would stand apart from the repo it won't require a release of any sort, except to point users to the newly active documentation.