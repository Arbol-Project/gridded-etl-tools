import pathlib
import re
import pandas as pd
import pytest
from gridded_etl_tools.util_funcs.data_intake import parse_filenames, nest_files

file_name_patterns = {
            "time": re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2})"),  # ISO format datetime with hours
            "step": re.compile(r"(F\d{3})"),  # Forecast step F followed by 3 digits
            "ensemble": re.compile(r"([cp]f)\.grib2$"),  # Control or perturbed forecast at end
        }

def test_parse_filenames_basic():
    """
    Test parse_filenames with a standard set of filenames.
    """
    files = [
        pathlib.Path("arbol_ensemble_2m_temp_2025-07-01T06_F006_cf.grib2"),
        pathlib.Path("arbol_ensemble_2m_temp_2025-07-01T06_F006_pf.grib2"),
        pathlib.Path("arbol_ensemble_2m_temp_2025-07-01T12_F012_cf.grib2"),
    ]
    df = parse_filenames(files, file_name_patterns)
    assert list(df.columns) == ["file", "time", "step", "ensemble"]
    assert df.shape[0] == 3
    assert set(df["ensemble"]) == {"cf", "pf"}
    assert set(df["time"]) == {"2025-07-01T06", "2025-07-01T12"}
    assert set(df["step"]) == {"F006", "F012"}
    # Check that file column is pathlib.Path
    assert all(isinstance(f, pathlib.Path) for f in df["file"])


def test_parse_filenames_regex_patterns():
    """
    Test that regex patterns correctly extract metadata from various valid filename formats.
    """
    # Test various valid filename formats
    files = [
        # Standard format
        pathlib.Path("arbol_ensemble_2m_temp_2025-07-01T06_F006_cf.grib2"),
        # Different prefix
        pathlib.Path("my_custom_prefix_2025-07-01T12_F012_pf.grib2"),
        # Minimal prefix
        pathlib.Path("2025-07-01T18_F018_cf.grib2"),
        # Complex prefix with numbers and special chars
        pathlib.Path("arbol_v1.2_ensemble_2m_temp_2025-07-01T00_F000_pf.grib2"),
        # Different time formats (all should work)
        pathlib.Path("arbol_2025-01-01T00_F000_cf.grib2"),
        pathlib.Path("arbol_2025-12-31T23_F999_pf.grib2"),
    ]

    df = parse_filenames(files, file_name_patterns)
    assert len(df) == 6

    # Verify all expected times are present
    expected_times = {
        "2025-07-01T06",
        "2025-07-01T12",
        "2025-07-01T18",
        "2025-07-01T00",
        "2025-01-01T00",
        "2025-12-31T23",
    }
    assert set(df["time"]) == expected_times

    # Verify all expected steps are present
    expected_steps = {"F006", "F012", "F018", "F000", "F999"}
    assert set(df["step"]) == expected_steps

    # Verify ensemble types
    assert set(df["ensemble"]) == {"cf", "pf"}


def test_parse_filenames_edge_cases():
    """
    Test parse_filenames with edge cases: missing fields, wrong extension, flexible naming.
    """
    # Flexible naming - should work regardless of prefix structure
    files = [
        pathlib.Path("prefix_with_extra_underscores_2025-07-01T06_F006_cf.grib2"),
        pathlib.Path("different_prefix_2025-07-01T12_F012_pf.grib2"),
        pathlib.Path("2025-07-01T18_F018_cf.grib2"),  # Minimal prefix
    ]
    df = parse_filenames(files, file_name_patterns)
    assert len(df) == 3
    assert set(df["time"]) == {"2025-07-01T06", "2025-07-01T12", "2025-07-01T18"}
    assert set(df["step"]) == {"F006", "F012", "F018"}
    assert set(df["ensemble"]) == {"cf", "pf"}

    # Wrong extension (should skip file, returning empty DataFrame)
    files = [
        pathlib.Path("arbol_ensemble_2m_temp_2025-07-01T06_F006_cf.txt"),
    ]
    with pytest.raises(ValueError, match="No files matched the provided patterns"):
        parse_filenames(files, file_name_patterns)

    # Missing required patterns (should skip file)
    files = [
        pathlib.Path("arbol_ensemble_2m_temp_short.grib2"),  # Missing time/step/ensemble
        pathlib.Path("arbol_ensemble_2m_temp_2025-07-01T06_cf.grib2"),  # Missing step
        pathlib.Path("arbol_ensemble_2m_temp_2025-07-01T06_F006.grib2"),  # Missing ensemble
    ]
    for file, missing_dim in zip(files, ["time", "step", "ensemble"]):
        with pytest.raises(ValueError, match=rf"File {file} does not match pattern .* for key {missing_dim}"):
            parse_filenames([file], file_name_patterns)


def test_parse_filenames_empty():
    """
    Test parse_filenames with an empty list.
    """
    with pytest.raises(ValueError, match="No files matched the provided patterns"):
        parse_filenames([], file_name_patterns)


def test_nest_files_basic():
    """
    Test nest_files with a simple DataFrame from parse_filenames.
    """
    files = [
        pathlib.Path("arbol_ensemble_2m_temp_2025-07-01T06_F006_cf.grib2"),
        pathlib.Path("arbol_ensemble_2m_temp_2025-07-01T06_F006_pf.grib2"),
        pathlib.Path("arbol_ensemble_2m_temp_2025-07-01T06_F012_cf.grib2"),
        pathlib.Path("arbol_ensemble_2m_temp_2025-07-01T06_F012_pf.grib2"),
        pathlib.Path("arbol_ensemble_2m_temp_2025-07-01T12_F006_cf.grib2"),
        pathlib.Path("arbol_ensemble_2m_temp_2025-07-01T12_F006_pf.grib2"),
        pathlib.Path("arbol_ensemble_2m_temp_2025-07-01T12_F012_cf.grib2"),
        pathlib.Path("arbol_ensemble_2m_temp_2025-07-01T12_F012_pf.grib2"),
    ]
    df = parse_filenames(files, file_name_patterns)
    nested = nest_files(df)
    # Should be 2 times, 2 steps, 2 ensembles
    assert len(nested) == 2  # times
    assert all(len(tg) == 2 for tg in nested)  # steps
    assert all(len(sg) == 2 for tg in nested for sg in tg)  # ensembles
    # Check that all files are present in the nested structure
    flat = [f for tg in nested for sg in tg for f in sg if f is not None]
    assert set(flat) == set(files)


def test_nest_files_missing():
    """
    Test nest_files with missing combinations (should fill with None).
    """
    files = [
        pathlib.Path("arbol_ensemble_2m_temp_2025-07-01T06_F006_cf.grib2"),
        pathlib.Path("arbol_ensemble_2m_temp_2025-07-01T12_F012_pf.grib2"),
    ]
    df = parse_filenames(files, file_name_patterns)
    with pytest.raises(
        FileNotFoundError,
        match=re.escape(
            "No file found for key: ('2025-07-01T06', 'F006', 'pf'). Nested file structure must be complete for all dimensions, exiting."
        ),
    ):
        nest_files(df)


def test_nest_files_empty():
    """
    Test nest_files with an empty DataFrame.
    """
    df = pd.DataFrame(columns=["file", "time", "step", "ensemble"])
    with pytest.raises(ValueError, match="Input DataFrame is empty, cannot nest files"):
        nest_files(df)

