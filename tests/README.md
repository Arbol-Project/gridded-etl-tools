zarr-climate-etl testing
========================

Changes to the zarr-climate-etl codebase should be tested against the programs in this directory before being deployed to production. In fact, all PRs to the main branch must pass these tests because it is enforced by a [Github action](https://github.com/features/actions) [tied to this repo]( ../.github/workflows/merge-protection.yml) (note: this hasn't been implemented on the zarr-main branch yet). The tests can also be useful for quickly running work-in-progress code against sample data.

For testing against data not present in these tests, it is recommended to use [generate_zarr.py](../generate_zarr.py). If it is something being tested frequently, consider [adding a new test](#adding-a-test) to this directory along with relevant data.

Requirements
------------

Same as the [requirements of the repo](../README.md#requirements)

Running
-------

Run all tests and report either success or failure when fun from this directory

    > pytest
    
To run a specific collection of tests, specify the containing directory in the command

    > pytest chirps/
    
To run only test functions that contain a certain word, use the `-k` flag

    # Run test functions containing "initial", such as ERA5 and PRISM test_initial
    > pytest -k "initial"
    
The `-s` flag will enable stdout

    > pytest -s
    
See [pytest's documentation](https://docs.pytest.org/) for more options.

Adding a test
-------------

If a part of the codebase, such as a feature, code block, or edge case, can be tested with a script that runs in a short amount of time (under a minute or so) using a relatively small amount of data (around a few MB or less), consider adding a new script to the tests. In order to add a new test, there are some conventions to follow.

When `pytest` runs, it checks every directory recursively for files matching `test_*.py` and runs every function with a name matching `test_*` in the file. The simplest way to add a test would be to create a file such as `test_mytest.py` in the root directory with a function such as `test_mytest`:

    # test_mytest.py
    
    function test_mytest():
        assert True == False
        
When `pytest` is run from the root, it will automatically run `test_mytest` and fail because of the failed assertion.

### Subdirectories

Tests within a subdirectory will also be automatically discovered by `pytest`. This can be useful for organizing tests which need or want to use their own `conftest.py` or have data that will be checked into the repo along with the test code. Subdirectories need to have an `__init__.py` file.

### conftest.py

Pytest will look for a file named `conftest.py` in every directory and [provide all its fixtures](https://docs.pytest.org/en/7.1.x/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files) to the tests running in that directory. Each directory can have its own fixtures. For example, `chirps/conftest.py` can define a fixture called `root` which will return a path to the corresponding data for each dataset's tests.

The global `conftest.py` (in this directory) defines fixtures available to all tests. Please only add fixtures that will be useful to the entire suite of tests.

### common.py

Utility functions that are not fixtures and are useful to more than one test file or directory are defined in this file. For example, the function `remove_zarr_json` is a clean-up function that applies to any set that generates Zarr data.

### Options

Pytest allows for specifying options on the command line (and in an INI file) that will be forwarded to tests using the special function `pytest_addoption`. Unlike fixtures, all options are *global* and can only be defined once, meaning they *cannot* be overwritten per-directory. Therefore, options should be chosen carefully as every test and future test would have those options passed to it. Currently, we don't use any options, but that doesn't mean they can't be used. The best practice when adding an option would be to prefix it with the test name in order to indicate it is an option that only applies to that test, for example `--era5-time-chunk` instead of just `--time-chunk`.

### mocker

The [pytest-mock](https://pypi.org/project/pytest-mock/) module provides a built-in named `mocker` that can be added to test functions as an automatically supplied argument. It is not very well documented. See `offline_ipns_publish` in [common.py](common.py) and `mocker` in [chirps/test_chirps.py](chirps/test_chirps.py) for an example of how to override a function while still being able to call the original function and modify the arguments to it.

### ordering

The [pytest-order](https://pypi.org/project/pytest-order/) module provides a `@pytest.mark.order(<number>)` decorator that controls the order of decorated tests. We add this decorator to `test_initial` to ensure that test consistently runs first, overriding pytest's default behavior to run tests in a random order. `test_initial` stores a dataset in IPFS that the insert/append/insert_and_append tests all work with and as such is a necessary pre-condition for them.

Tests
-----

* [CHIRPS](chirps/test_chirps.py)
