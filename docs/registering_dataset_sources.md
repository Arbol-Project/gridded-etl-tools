
Creating new dataset sources
----------------------------

New climate sources can be added by creating a new manager class for that source within a python script dedicated to it -- e.g. [CPC.py](etls/managers/cpc.py). The recommended practice is to create child classes for each climate variable (minimum temperature, water salinity, etc.) extracted from that source -- as can be seen in the [CHIRPS example ETL](examples/managers/CHIRPS.py).