import pathlib
import re
import pandas as pd
import numpy as np


def parse_filenames(files: list[pathlib.Path], file_name_patterns: dict[str, re.Pattern]) -> pd.DataFrame:
    """
    Parse a list of file paths to extract forecast metadata from filenames.

    This function expects filenames to follow a convention where
    the time, step, ensemble, etc. are encoded in the filename. It extracts these fields
    using regex patterns and returns a DataFrame with columns for the file path,
    forecast time, forecast step, ensemble type, etc.

    Parameters
    ----------
    files : list[pathlib.Path]
        List of file paths whose names encode forecast metadata.
    file_name_patterns : dict[str, re.Pattern]
        Dictionary of regex patterns for extracting the desired attributes from the file names

    Returns
    -------
    pd.DataFrame
        DataFrame with columns for each of the desired attributes:
            - 'file': pathlib.Path object for the file
            - 'time': str, forecast time extracted from the filename (e.g., "2025-07-01")
            - 'step': str, forecast step extracted from the filename (e.g., "6" for 6 hours)
            - 'ensemble': str, ensemble model number extracted from the filename (e.g., "71")
            - 'hindcast_reference_time': str, hindcast reference time extracted from the filename (e.g., "2025-07-01")

    Notes
    -----
    The function uses regex patterns to extract:
    - time: ISO format date (e.g., "2025-07-01")
    - step: Forecast step in format step-[0-9]{1,3} (e.g., "6")
    - ensemble: Ensemble number in format ensemble-[0-9]{1,3} (e.g., "71")

    Files that don't match the expected patterns are skipped.

    # NOTE this ignores the 'cf' / 'pf' distinction in some ECMWF files, since the ensemble number
    effectively encodes this distinction (i.e. cf = 0, 0 < pf <= 100)
    """
    files_and_corresponding_coords = []
    for file in files:
        # Skip incompatible files, i.e. grib .idx files
        if file.suffix not in [".grib2", ".nc4"]:
            continue

        # Try to extract data for all patterns for this file; fail if any pattern is not matched
        file_data = {"file": file}
        for key, pattern in file_name_patterns.items():
            if callable(pattern):
                # Handle callable functions, assuming these functions operate on the file name
                # (e.g., calc_fro_as_int to calculate the forecast reference offset for hindcasts)
                file_data[key] = pattern(file.name)
            else:
                # Handle regex patterns
                match = re.search(pattern, file.name)
                if not match:
                    raise ValueError(f"File {file.name} does not match pattern {pattern} for key {key}")
                file_data[key] = match.group(1)

        # Append the extracted data for the file to the files_and_corresponding_coords list
        files_and_corresponding_coords.append(file_data)

    # Turn the list of per-file extracted data into a DataFrame
    df = pd.DataFrame(files_and_corresponding_coords)
    if df.empty:
        raise ValueError(f"No files matched the provided patterns: {file_name_patterns}")
    return df


def nest_files(
    files_df: pd.DataFrame,
) -> list:
    """
    Organize files into a nested list structure by arbitrary dimensions.

    Creates an N-dimensional nested list structure where each file can be accessed
    via nested_list[dim0_idx][dim1_idx]...[dimN_idx].

    This function takes a DataFrame of files and their associated metadata
    (as produced by `parse_filenames`) and nests the files into an N-dimensional list
    structure. The dimensions are determined by the columns in files_df (excluding 'file').
    Each element is the corresponding file (pathlib.Path) or None if a file for that
    combination is missing.

    Parameters
    ----------
    files_df : pd.DataFrame
        DataFrame with file info and columns matching dim_columns.

    Returns
    -------
    nested : list
        N-dimensional nested list of files or None (if missing).

    Examples
    --------
    1. 3D Example for time, step, ensemble dimensions:

    >>> import pandas as pd
    >>> import pathlib
    >>> files_df = pd.DataFrame([
    ...     {"file": pathlib.Path("f1.grib2"), "time": "2025-01-01", "step": "0", "ensemble": "0"},
    ...     {"file": pathlib.Path("f2.grib2"), "time": "2025-01-01", "step": "0", "ensemble": "1"},
    ...     {"file": pathlib.Path("f3.grib2"), "time": "2025-01-01", "step": "6", "ensemble": "0"},
    ...     {"file": pathlib.Path("f4.grib2"), "time": "2025-01-01", "step": "6", "ensemble": "1"},
    ...     {"file": pathlib.Path("f5.grib2"), "time": "2025-01-02", "step": "0", "ensemble": "0"},
    ...     {"file": pathlib.Path("f6.grib2"), "time": "2025-01-02", "step": "0", "ensemble": "1"},
    ...     {"file": pathlib.Path("f7.grib2"), "time": "2025-01-02", "step": "6", "ensemble": "0"},
    ...     {"file": pathlib.Path("f8.grib2"), "time": "2025-01-02", "step": "6", "ensemble": "1"},
    ... ])
    >>> nested = nest_files(files_df)
    >>> nested[0][1][1]  # [time][step][ensemble]
    PosixPath('f4.grib2')
    >>> nested[1][0][1]
    PosixPath('f6.grib2')
    >>> len(nested), len(nested[0]), len(nested[0][0])
    (2, 2, 2)  # 2 times, 2 steps, 2 ensembles

    Raises
    ------
    ValueError
        If the input DataFrame is empty.
    FileNotFoundError
        If any expected file (combination of all unique values of dimentional columns)
        is missing from the DataFrame.
    ZeroDivisionError
        If any dimension is empty.
    """
    # Empty dataframes are not allowed
    if files_df.empty:
        raise ValueError("Input DataFrame is empty, cannot nest files")
    # All dimension columns (all except 'file'). The order specifies nesting order.
    dim_columns = [col for col in files_df.columns if col != "file"]

    # Sorted unique values for each dimension
    dim_values = [sorted(files_df[dim].unique()) for dim in dim_columns]

    # File lookup table:
    # Keys are tuples of dimension values (in the order of dim_columns), values are file paths.
    # Example (for dim_columns=["time", "step", "ensemble"]):
    # {
    #     ("2025-01-01", "0", "0"): PosixPath("f1.grib2"),
    #     ("2025-01-01", "0", "1"): PosixPath("f2.grib2"),
    #     ...
    # }
    file_lookup = {tuple(row[dim] for dim in dim_columns): row["file"] for _, row in files_df.iterrows()}

    # Calculate total shape
    shape = [len(values) for values in dim_values]
    total_size = int(np.prod(shape))

    # Fill flat list with files
    flat_list = []
    for i in range(total_size):
        # Convert flat index to multi-dimensional indices
        indices = []
        remaining = i
        for dim_size in reversed(shape):
            indices.append(remaining % dim_size)
            remaining //= dim_size
        indices.reverse()

        # Map the flat index (i) to multi-dimensional indices along each dimension,
        # then retrieve the unique value for each dimension at these indices.
        # This makes a key like ('2025-01-01', '0', '1') for [time, step, ensemble] at
        # indices [0, 0, 1], corresponding to one unique file.
        # For example, if dim_columns is ['time', 'step', 'ensemble']:
        #   indices = [0, 0, 1]  -> dim_values[0][0] = '2025-01-01', dim_values[1][0] = '0', dim_values[2][1] = '1'
        #   key = ('2025-01-01', '0', '1')
        #   file_lookup[key] gives the file at that coordinate.
        key = tuple(dim_values[dim_idx][idx] for dim_idx, idx in enumerate(indices))
        key_file = file_lookup.get(key, None)
        if key_file is None:
            raise FileNotFoundError(
                f"No file found for key: {key}. " "Nested file structure must be complete for all dimensions, exiting."
            )
        flat_list.append(key_file)

    # Reshape flat list into nested structure
    return reshape(flat_list, shape)


def reshape(data: list, shape: list[int]) -> list:
    """
    Recursively reshape a flat list into a nested list with specified dimensions.

    This function takes a flat list `data` and reshapes it into a nested list structure,
    where the nesting levels and sizes are determined by the `shape` parameter.

    Parameters
    ----------
    data : list
        The flat list of items to be reshaped.
    shape : list of int
        The desired shape of the output nested list. Each integer specifies the size of the corresponding dimension.

    Returns
    -------
    nested_list : list
        A nested list with the structure defined by `shape`, containing the elements of `data`
        arranged in row-major order. If `shape` has one dimension, returns the flat list as is.

    Examples
    --------
    >>> data = [1, 2, 3, 4, 5, 6]
    >>> shape = [2, 3]
    >>> reshape(data, shape)
    [[1, 2, 3], [4, 5, 6]]

    >>> data = list(range(8))
    >>> shape = [2, 2, 2]
    >>> reshape(data, shape)
    [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]

    Raises
    ------
    ValueError
        If the total number of elements in `data` does not match the product of dimensions in `shape`.
    """
    # Error out if data length does not match the expected shape
    # We expect only full lists and this would cause data loss
    expected_size = int(np.prod(shape))
    if len(data) != expected_size:
        raise ValueError(
            f"Cannot reshape data of length {len(data)} into shape {shape}. "
            f"Expected {expected_size} elements, got {len(data)}."
        )

    if len(shape) == 1:
        return data

    size = shape[0]
    step = len(data) // size
    return [reshape(data[i * step : (i + 1) * step], shape[1:]) for i in range(size)]
