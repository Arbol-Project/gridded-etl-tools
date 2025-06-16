Python virtual environments
===========================

This project is developed and tested against Python 3.10.9. It is strongly recommended to set up a virtual environment that uses that version of Python so the version can be constant and also because there are a lot of requirements that will be installed by PIP that may be unnecessary for a system wide Python environment. 

However, it is not strictly necessary to set up a virtual environment, and it may not always make sense to do so, like when running the project on an ephemeral server. 

Note that this repository has been tested on Ubuntu Linux environments from 20.0+ and Mac OS X 12.0+ environments prior to the M1 Mac release. We have encountered some issues with M1 Macs and can't guarantee this code will work with it out of the box.

Setting up an environment on a local Mac OS X machine
--------------------------------------------------

Users parsing modest datasets or looking to test ETLs quickly may want to run them from a local Mac OS X machine. Follow the steps here to instantiate a Python 3.10.9 virtual environment using the [pyenv](https://realpython.com/intro-to-pyenv/) utility, install all the packages in `requirements.txt`, and run ETLs.

## Climate data binaries

To use this package you must install system wide binary tools used for data transformations

    brew install cdo eccodes netcdf gdal jpeg

## Pyenv setup

First install pyenv with `homebrew`

```
brew update
brew install pyenv
```

Insert the following into the end of your `~/.bashrc` or `~/.zshrc` files (whichever corresponds to your preferred shell scripting environment) and `source` that file.

```bash
alias brew='env PATH="${PATH//$(pyenv root)\/shims:/}" brew'
if command -v pyenv 1>/dev/null 2>&1; then
  eval "$(pyenv init --path)"
  eval "$(pyenv init -)"
  eval "$(pyenv virtualenv-init -) "
fi
```

## Environment setup

Install Python 3.10.9, create a new virtual environment with it, and install the package requirements

```bash
pyenv install 3.10.9
pyenv virtualenv 3.10.9 zarr_climate_etls
pyenv activate zarr_climate_etls
pip install --upgrade pip
pip install -r requirements.txt
```

Activate the environment whenever you're ready to run or test an ETL and you're good to run `python run_etl.py <ETL_NAME> <ETL FLAGS>`

Setting up an environment on a Linux OS
---------------------------------------

Power users will want to run this library from remote servers running Linux. This is a bit more involved as various binaries and Python 3.12.10 must be manually installed _in the correct order_ before a functioning virtual environment can be instantiated. 

Follow the steps here to set up a virtual Python environment and install external dependencies on a Linux OS. 

## Set up a Virtual Environment with existing Python binaries

If you already have Python 3.12.10 installed or want to try with your system's Python version, you can use `venv` to create the environment.

    python -m venv --python=path/to/python/binary venv/

## Set up a Virtual Environment with Python binaries built from source

If you want to use the specific version of Python this project is tested against or any other version of Python different from the one you have installed, download and compile Python in a separate directory. This example uses Python 3.12.10.

    cd
    sudo apt update && sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
        libsqlite3-dev wget curl llvm libncurses-dev xz-utils tk-dev cmake libssl-dev libffi-dev libbz2-dev liblzma-dev \
        libreadline-dev libsqlite3-dev libgdbm-compat-dev libnsl-dev
    wget https://www.python.org/ftp/python/3.12.10/Python-3.12.10.tgz
    tar -xf Python-3.12.10.tgz
    cd Python-3.12.10
    ./configure --enable-loadable-sqlite-extensions
    make -j3
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    ./python get-pip.py
    cd ..
    git clone https://github.com/Arbol-Project/gridded-etl-tools
    cd gridded-etl-tools
    ../Python-3.12.10/python -m venv venv/
    
## Use the virtual environment

    source venv/bin/activate

## Install required Python packages to the virtual environment

First install system wide dependencies (sometimes PIP doesn't handle this automatically)

    sudo apt install libjpeg-dev
    
Then install all the PIP packages, along with optional testing and developer packages

    pip install -e .[testing,dev]
    
## Install climate data binaries (necessary for parsing PRISM and ERA5)

Note that these packages will be installed to the system wide environment.

    sudo apt install gdal-bin netcdf-bin cdo libeccodes0

## Install CDS API .rc file (necessary for ERA5)

To request ERA5 data, it is necessary to have credentials for their API installed. Follow the instructions at https://cds.climate.copernicus.eu/api-how-to to install the credentials in your home directory.

## Regenerate the dependencies list when committing code containing new or updated libraries

    pip freeze > requirements.txt
