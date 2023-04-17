Understanding this repository
-----------------------------

This repository is written to build and run ETL workflows to place (climate) data on IPFS, from whence it can be shared over dClimate's Marketplace or used for private purposes. The repository provides a Dataset Manager abstract base class with which to implement ETLs, a host of supporting utilities which the DatasetManager leverages, and example ETLs previously implemented by the Arbol Data Engineering team which can serve as examples and inspiration. The exact breakdown of the repository is as such

* **doc** contains additional documentation
* **etls** contains the *dataset_manager.py* two sub-directories
    - **managers** houses ETL managers
    - **utils** houses utilities, broken down by theme and purpose, which feed into the *dataset_manager.py*
    - Additionally, this directory contains a detailed README walking users through the ETL building workflow 
* **logs** contain logs from ETLs, both failed and successful
* **ops** contains additional setup scripts
* **tests** contain unit tests for different ETLs and their key functions

ETL scripts will create two additional folders housing data when they are run
* **datasets**, which houses raw data downloaded from the internet
* **climate**, which houses completed Zarrs saved locally
