
What's not covered in these docs
--------------------------------

Several technologies instrumental to using this library are beyond the scope of its documentation. 

First and foremost, a working knowledge of Python 3 and setting up Python environments on UNIX systems is required to successfully use this library. 

Secondly, the Zarr format for N-Dimensional data is the foundational piece for manipulating and saving data and readers unfamiliar with its nuances (or those of N-D data in general) should first read [its introductory docs](https://zarr.readthedocs.io/en/stable/getting_started.html). 

Finally, while these scripts can be run locally as one-offs for small time periods for small datasets, many (most?) users will be interested in regular updates, large time scales, and/or better performance. These users will need to orchestrate the execution of this library's scripts at regular intervals on remote servers. Orchestration is a data engineering specialization unto itself and well beyond the scope of this documentation. We advise that interested readers consider [Airflow](https://airflow.apache.org/), [Prefect](https://www.prefect.io/), or [Ansible](https://www.ansible.com/) as orchestation technologies.
