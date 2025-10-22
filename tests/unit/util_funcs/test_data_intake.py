import pathlib
import re
import pandas as pd
import pytest
from gridded_etl_tools.util_funcs.data_intake import parse_filenames, nest_files

FILE_NAME_PATTERNS_V1 = {
    "time": re.compile(r"forecast_reference_time-(\d{4}-\d{2}-\d{2})_step"),
    "step": re.compile(r"step-([0-9]+)_ensemble"),
    "number": re.compile(r"ensemble-([0-9]+)_"),
}

FILE_NAME_PATTERNS_V2 = {
    "time": re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2})"),  # ISO format datetime with hours
    "step": re.compile(r"(F\d{3})"),  # Forecast step F followed by 3 digits
    "ensemble": re.compile(r"([cp]f)\.grib2$"),  # Control or perturbed forecast at end
}


def test_parse_filenames(tmp_path):
    """Test parse_filenames with a simple filename pattern against a realistic set of filenames"""
    filenames = []
    for day in range(1, 6):  # 5 days
        for step_num in range(0, 3):  # 3 steps
            step_hours = step_num * 6
            for ensemble_num in range(0, 4):  # 4 ensembles
                filenames.append(
                    tmp_path
                    / (
                        f"tp_forecast_reference_time-2025-01-{day:02d}_"
                        f"step-{step_hours}_ensemble-{ensemble_num}_stuff_here.grib2"
                    )
                )
    df = parse_filenames(filenames, FILE_NAME_PATTERNS_V1)
    assert len(df) == 60  # 5 days * 3 steps * 4 ensembles = 60
    assert list(df.columns) == ["file", "time", "step", "number"]
    assert set(df["time"]) == set(f"2025-01-{day:02d}" for day in range(1, 6))
    assert set(df["step"]) == set(f"{step_num * 6}" for step_num in range(0, 3))
    assert set(df["number"]) == set(f"{ensemble_num}" for ensemble_num in range(0, 4))
    # Check that file column is pathlib.Path
    assert all(isinstance(f, pathlib.Path) for f in df["file"])


def test_parse_filenames_with_callable():
    """Test parse_filenames with callable pattern (like hindcast calc_fro_as_int)."""
    files = [
        pathlib.Path("test_hindcast_reference_time-2020-01-01_forecast_reference_offset-2019-01-01_step-24_.grib2")
    ]

    def mock_calc(filename):
        return -5  # 2015 - 2020 = -5

    patterns = {"time": mock_calc, "step": re.compile(r"step-([0-9]+)_")}
    df = parse_filenames(files, patterns)

    assert len(df) == 1
    assert df.iloc[0]["time"] == -5


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

    df = parse_filenames(files, FILE_NAME_PATTERNS_V2)
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
    df = parse_filenames(files, FILE_NAME_PATTERNS_V2)
    assert len(df) == 3
    assert set(df["time"]) == {"2025-07-01T06", "2025-07-01T12", "2025-07-01T18"}
    assert set(df["step"]) == {"F006", "F012", "F018"}
    assert set(df["ensemble"]) == {"cf", "pf"}

    # Wrong extension (should skip file, returning empty DataFrame)
    files = [
        pathlib.Path("arbol_ensemble_2m_temp_2025-07-01T06_F006_cf.txt"),
    ]
    with pytest.raises(ValueError, match="No files matched the provided patterns"):
        parse_filenames(files, FILE_NAME_PATTERNS_V2)

    # Missing required patterns (should skip file)
    files = [
        pathlib.Path("arbol_ensemble_2m_temp_short.grib2"),  # Missing time/step/ensemble
        pathlib.Path("arbol_ensemble_2m_temp_2025-07-01T06_cf.grib2"),  # Missing step
        pathlib.Path("arbol_ensemble_2m_temp_2025-07-01T06_F006.grib2"),  # Missing ensemble
    ]
    for file, missing_dim in zip(files, ["time", "step", "ensemble"]):
        with pytest.raises(ValueError, match=rf"File {file} does not match pattern .* for key {missing_dim}"):
            parse_filenames([file], FILE_NAME_PATTERNS_V2)


def test_parse_filenames_empty():
    """
    Test parse_filenames with an empty list.
    """
    with pytest.raises(ValueError, match="No files matched the provided patterns"):
        parse_filenames([], FILE_NAME_PATTERNS_V2)


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
    df = parse_filenames(files, FILE_NAME_PATTERNS_V2)
    nested = nest_files(df)
    # Should be 2 times, 2 steps, 2 ensembles
    assert len(nested) == 2  # times
    assert all(len(tg) == 2 for tg in nested)  # steps
    assert all(len(sg) == 2 for tg in nested for sg in tg)  # ensembles
    # Check that all files are present in the nested structure
    flat = [f for tg in nested for sg in tg for f in sg if f is not None]
    assert set(flat) == set(files)


def test_nest_files_3d():
    """
    Test nest_files with a simple DataFrame from parse_filenames.
    All files for every (time, step, ensemble) combination are provided.
    """
    files_df = pd.DataFrame(
        [
            {"file": pathlib.Path("f1.grib2"), "time": "2025-01-01", "step": "0", "ensemble": "0"},
            {"file": pathlib.Path("f2.grib2"), "time": "2025-01-01", "step": "0", "ensemble": "1"},
            {"file": pathlib.Path("f3.grib2"), "time": "2025-01-01", "step": "6", "ensemble": "0"},
            {"file": pathlib.Path("f4.grib2"), "time": "2025-01-01", "step": "6", "ensemble": "1"},
            {"file": pathlib.Path("f5.grib2"), "time": "2025-01-02", "step": "0", "ensemble": "0"},
            {"file": pathlib.Path("f6.grib2"), "time": "2025-01-02", "step": "0", "ensemble": "1"},
            {"file": pathlib.Path("f7.grib2"), "time": "2025-01-02", "step": "6", "ensemble": "0"},
            {"file": pathlib.Path("f8.grib2"), "time": "2025-01-02", "step": "6", "ensemble": "1"},
        ]
    )

    result = nest_files(files_df)

    # Check structure
    assert isinstance(result, list)
    assert len(result) == 2  # 2 unique times
    assert all(isinstance(x, list) for x in result)
    assert all(len(time_level) == 2 for time_level in result)  # 2 steps per time
    assert all(len(step_level) == 2 for time_level in result for step_level in time_level)  # 2 ensembles per step

    # Check specific file placements
    assert result[0][0][0].name == "f1.grib2"  # 2025-01-01, step 0, ensemble 0
    assert result[0][0][1].name == "f2.grib2"  # 2025-01-01, step 0, ensemble 1
    assert result[0][1][0].name == "f3.grib2"  # 2025-01-01, step 6, ensemble 0
    assert result[0][1][1].name == "f4.grib2"  # 2025-01-01, step 6, ensemble 1
    assert result[1][0][0].name == "f5.grib2"  # 2025-01-02, step 0, ensemble 0
    assert result[1][0][1].name == "f6.grib2"  # 2025-01-02, step 0, ensemble 1
    assert result[1][1][0].name == "f7.grib2"  # 2025-01-02, step 6, ensemble 0
    assert result[1][1][1].name == "f8.grib2"  # 2025-01-02, step 6, ensemble 1


def test_nest_files_5d():
    """
    Test nest_files scales to working with a more complex DataFrame from parse_filenames.
    """
    # Create simple 2x2x2x2x2 = 32 files with predictable ordering
    # Use values that sort in the order we create them
    dims = [["a", "b"], ["c", "d"], ["e", "f"], ["g", "h"], ["i", "j"]]
    files_df = pd.DataFrame(
        [
            {
                "file": pathlib.Path(f"f{i}.grib2"),
                "d1": dims[0][i // 16],
                "d2": dims[1][(i // 8) % 2],
                "d3": dims[2][(i // 4) % 2],
                "d4": dims[3][(i // 2) % 2],
                "d5": dims[4][i % 2],
            }
            for i in range(32)
        ]
    )

    result = nest_files(files_df)

    # Check structure: 2x2x2x2x2
    assert len(result) == 2
    assert len(result[0][0][0][0]) == 2

    # Check that we can navigate the structure
    assert result[0][0][0][0][0].name == "f0.grib2"  # a,c,e,g,i
    assert result[0][0][0][0][1].name == "f1.grib2"  # a,c,e,g,j
    assert result[1][1][1][1][1].name == "f31.grib2"  # b,d,f,h,j


def test_nest_files_xarray_compatibility():
    """
    Explicitly test the compatibility of nest_files for producing lists of lists compatible with
    running xarray.open_mfdataset with combine='nested'
    """
    # Create complete 2x2 dataset
    files_df = pd.DataFrame(
        [
            {"file": pathlib.Path("test1.grib2"), "time": "2025-01-01", "step": "0"},
            {"file": pathlib.Path("test2.grib2"), "time": "2025-01-01", "step": "6"},
            {"file": pathlib.Path("test3.grib2"), "time": "2025-01-02", "step": "0"},
            {"file": pathlib.Path("test4.grib2"), "time": "2025-01-02", "step": "6"},
        ]
    )

    result = nest_files(files_df)

    # Should be compatible with xarray.open_mfdataset structure
    assert isinstance(result, list)
    assert isinstance(result[0], list)
    assert isinstance(result[0][0], pathlib.Path)

    # Check structure: 2x2
    assert len(result) == 2
    assert len(result[0]) == 2

    # All files should be present (no None values)
    all_files = [f for time_group in result for f in time_group]
    assert len(all_files) == 4
    assert all(isinstance(f, pathlib.Path) for f in all_files)


def test_nest_files_missing():
    """
    Test nest_files with missing combinations (should raise FileNotFoundError).
    """
    files = [
        pathlib.Path("arbol_ensemble_2m_temp_2025-07-01T06_F006_cf.grib2"),
        pathlib.Path("arbol_ensemble_2m_temp_2025-07-01T12_F012_pf.grib2"),
    ]
    df = parse_filenames(files, FILE_NAME_PATTERNS_V2)
    with pytest.raises(
        FileNotFoundError,
        match=re.escape(
            "No file found for key: ('2025-07-01T06', 'F006', 'pf'). "
            "Nested file structure must be complete for all dimensions, exiting."
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
