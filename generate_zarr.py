#!/usr/bin/env python

### [generate_zarr.py]
###
### This utility uses the manager classes in this repo to build and store sets of climate data in the format used
### by Arbol. It can retrieve original published data by checking remote locations for updates, parse it into
### Arbol's format, and add it to IPFS.
###
### The three phases of extract, transform, and load are referred to as ETL in data processing. See
### https://en.wikipedia.org/wiki/Extract,_transform,_load for more information.
###

import argparse, datetime, logging
from sources import get_set_manager_from_name, SOURCES

def generate(dataset_class, date_range=None, latitude_range=None, longitude_range=None, rebuild=False, only_parse=False,
             only_update_input=False, only_prepare_input=False, only_metadata=False, local_output=False, custom_output_path=None,
             custom_head_metadata=None, custom_latest_hash=None, *args, **kwargs):
    """
    Perform all the ETL steps requested by the combination of flags passed. By default, this will run a full ETL on `dataset_class`, meaning
    it will update input, parse input, and store the parsed output on IPFS. See the flags at the bottom of this file or run `generate.py -h` to
    read more about the available flags.
    """
    # Initialize a manager for the given class. For example, if class is CHIRPSPrelim05, the manager will be CHIRPSPrelim05([args]). This will create
    # INFO and DEBUG logs in the current working directory.
    dataset_manager = dataset_class(
        custom_output_path=custom_output_path, custom_metadata_head_path=custom_head_metadata, custom_latest_hash=custom_latest_hash,
        rebuild=rebuild)
    dataset_manager.log_to_file()
    dataset_manager.log_to_file(level=logging.DEBUG)

    trigger_parse = False

    if only_parse:
        dataset_manager.info("only parse flag present, skipping update of local input and using locally available data")
    elif only_metadata:
        dataset_manager.info("only metadata flag present, skipping update of local input and parse to update metadata using the existing Zarr on IPFS")
    else:
        dataset_manager.info("updating local input")
        # update_local_input will return True if parse should be triggered
        trigger_parse = dataset_manager.update_local_input(
            rebuild=rebuild, date_range=date_range)
        if only_update_input:
            # we're finished if only update input was set
            dataset_manager.info("ending here because only update local input flag is set")
            return
    # parse if force flag is set or if parse was triggered by local input update return value 
    if only_parse or trigger_parse:
        dataset_manager.info(f"parsing {dataset_manager}")
        # parse will return `True` if new data was parsed
        if only_prepare_input:
            dataset_manager.info(f"Only prepare input requested, just preparing source files for parsing and creating corresponding Zarr JSON file")
            dataset_manager.prepare_input_files()
            dataset_manager.create_zarr_json()
        else:
            if dataset_manager.parse(rebuild=rebuild, date_range=date_range, latitude_range=latitude_range, longitude_range=longitude_range,
                    only_parse=only_parse, local_output=local_output):
                dataset_manager.info(f"Data for {dataset_manager} successfully parsed")
            else:
                dataset_manager.info("no new data parsed, ending here")
    elif only_metadata:
        dataset_manager.info(f"preparing metadata for {dataset_manager}")
        dataset_manager.only_update_metadata()
        dataset_manager.info(f"Metadata for {dataset_manager} successfully updated")
    else:
        dataset_manager.info("no new data detected and parse not set to force, ending here")

def parse_command_line():
    """
    When this file is called as a script, this function will run automatically, reading input arguments and flags from the
    command line
    """
    # these keys are the string representation of each source defined in sources.py
    valid_source_keys = [s.name() for s in SOURCES]
    # use argparse to parse submitted CLI options
    parser = argparse.ArgumentParser(
        description="""
        This utility can be used to retrieve climate data from various sources and store it on IPFS. Depending on the source,
        the data may be converted to Zarr format in the process.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("source", help=f"any of {valid_source_keys}")
    parser.add_argument("--rebuild", action="store_true", help="rebuild from beginning of history and generate a new CID independent of any existing data")
    parser.add_argument("--date-range", nargs=2, metavar="YYYY-MM-DD", type=datetime.datetime.fromisoformat,
        help="if supported by any of the specified sets, you can specify a range of dates to parse instead of the entire set") 
    parser.add_argument("--latitude-range", nargs=2, metavar=("MIN", "MAX"), type=float,
        help="if supported by any specified source, you can pass a latitude range to parse instead of the entire set")
    parser.add_argument("--longitude-range", nargs=2, metavar=("MIN", "MAX"), type=float,
        help="if supported by any specified source, you can pass a longitude range to parse instead of the entire set")
    parser.add_argument("--only-parse", action="store_true", help="only run a parse, using locally availabe data")
    parser.add_argument("--only-metadata", action="store_true", help="only update metadata, using data available on IPFS")
    parser.add_argument("--only-update-input", action="store_true", help="only run the update local input function")
    parser.add_argument("--only-prepare-input", action="store_true",
                        help="""
                        Instead of running the full parse, just run the dataset manager's prepare_input_files and create_zarr_json methods.
                        This will also run the update input function unless --only-parse has been specified as well.
                        """)
    parser.add_argument("--local-output", action="store_true", help="write output Zarr to disk instead of IPFS")
    parser.add_argument("--custom-output-path", help="override the class's automatic output path generation")
    parser.add_argument("--custom-head-metadata", help="override the class's automatic head lookup")
    parser.add_argument("--custom-latest-hash", help="override the class's automatic latest hash lookup")
    arguments = parser.parse_args()
    if arguments.source not in valid_source_keys:
        raise ValueError(f"{arguments.source} not a valid source key")
    # this replaces the passed string for each source with a set manager instance
    source_class = get_set_manager_from_name(arguments.source)
    return source_class, arguments

if __name__ == "__main__":
    # Get command line args and flags
    source_class, command_line_arguments = parse_command_line()
    # Pass the command line args to `generate`, excluding the original source argument
    function_arguments = vars(command_line_arguments)
    function_arguments.pop("source")
    generate(source_class, **function_arguments)
