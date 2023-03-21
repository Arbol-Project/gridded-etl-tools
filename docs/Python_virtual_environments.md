Python virtual environments
===========================

This project is developed and tested against Python 3.10.9. It is strongly recommended to set up a virtual environment that uses that version of Python so the version can be constant and also because there are a lot of requirements that will be installed by PIP that may be unnecessary for a system wide Python environment. 

However, it is not strictly necessary to set up a virtual environment, and it may not always make sense to do so, like when running the project on an ephemeral server. 

Note that this repository has been tested on Ubuntu Linux environments from 20.0+ and Mac OS X 12.0+ environments prior to the M1 Mac release. We have encountered some issues with M1 Macs and can't guarantee this code will work with it out of the box.

Setting up an environment on a local Mac OS X machine
--------------------------------------------------

Users parsing modest datasets or looking to test ETLs quickly may want to run them from a local Mac OS X machine. Follow the steps here to instantiate a Python 3.10.9 virtual environment using the [pyenv](https://realpython.com/intro-to-pyenv/) utility, install all the packages in `requirements.txt`, and run ETLs.

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

Activate the environment whenever you're ready to run or test an ETL and you're good to run `python generate_zarr.py <ETL_NAME> <ETL FLAGS>`

Setting up an environment on a Linux OS
---------------------------------------

Power users will want to run this library from remote servers running Linux. This is a bit more involved as various binaries and Python 3.10.9 must be manually installed _in the correct order_ before a functioning virtual environment can be instantiated. 

Follow the steps here to set up a virtual Python environment and install external dependencies on a Linux OS. 

## Set up a Virtual Environment with existing Python binaries

If you already have Python 3.10.9 installed or want to try with your system's Python version, you can use `virtualenv` to create the environment.

    pip install virtualenv
    virtualenv --python=<path_to_python_binary> .

## Set up a Virtual Environment with Python binaries built from source

If you want to use the specific version of Python this project is tested against or any other version of Python different from the one you have installed, download and compile Python in a separate directory. This example uses Python 3.10.9 and assumes `zarr-climate-etl` is located in the home directory.

    cd
    sudo apt-get update && apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \ 
        libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev cmake libssl-dev libffi-dev \
        libbz2-dev liblzma-dev libreadline-dev
    wget https://www.python.org/ftp/python/3.10.9/Python-3.10.9.tgz
    tar -xf Python-3.10.9.tgz
    cd Python-3.10.9
    ./configure
    make -j8
    cd ..
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    Python-3.10.9/python get-pip.py
    Python-3.10.9/python -m pip install virtualenv
    cd zarr-climate-etl
    ../Python-3.10.9/python -m virtualenv .
    
## Activate the virtual environment

    source bin/activate

## Install required Python packages to the virtual environment

First install system wide dependencies (sometimes PIP doesn't handle this automatically)

    sudo apt install libjpeg-dev
    
Then install all the PIP packages

    pip install -r requirements.txt
    
If PIP is failing to compile any modules because Python development headers are missing, it may be necessary to install the Python development headers to the system wide environment and link to those headers from the root of the repo.

    # If the [PYTHON]/Include directory doesn't have all the necessary headers for building the
    # pip packages, try installing the latest dev package from the distro repository and linking
    # it into the virtual environment
    apt install -y python3-dev
    ln -s /usr/include/python[VERSION] include 
    pip install -r requirements.txt

## Install climate data binaries (necessary for parsing some datasets)

Note that these packages will be installed to the system wide environment.

    sudo apt install gdal-bin netcdf-bin cdo libeccodes0

## Regenerate the dependencies list when committing code containing new or updated libraries

    pip freeze > requirements.txt
