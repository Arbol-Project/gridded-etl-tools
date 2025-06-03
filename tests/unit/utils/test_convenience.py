import datetime
import json
import pathlib
import ftplib
from unittest.mock import Mock

import numpy as np
import pytest

from gridded_etl_tools.utils import store


class DummyFtpClient:
    def __init__(self):
        self.contexts = 0
        self.login = Mock()
        self.cwd = Mock()
        self.commands = []
        self.files = {
            "one.txt": {},
            "two.dat": {"timestamp": "20100512024200", "contents": b"Hi Mom!"},
            "three.dat": {"timestamp": "20201225000000", "contents": b"Hello Dad!"},
        }

    def __call__(self, host):
        self.host = host
        return self

    def __enter__(self):
        self.contexts += 1
        return self

    def __exit__(self, *exc_args):
        self.contexts -= 1

    def close(self):
        return Exception

    def nlst(self, *args):
        return ["one.txt", "two.dat", "three.dat"]

    def retrbinary(self, command, write):
        self.commands.append(command)
        assert command.startswith("RETR ")
        filename = command[5:]
        write(self.files[filename]["contents"])

    def pwd(self):
        if self.contexts == 0:
            raise ftplib.error_perm
        return "over here"


class TestConvenience:
    @staticmethod
    def test_root_directory(mocker, manager_class):
        cwd = mocker.patch("pathlib.Path.cwd")
        cwd.return_value = "/right/here/mom"
        dm = manager_class()
        assert dm.root_directory() == "/right/here/mom"
        cwd.assert_called_once_with()

    @staticmethod
    def test_root_directory_already_accessed(mocker, manager_class):
        cwd = mocker.patch("pathlib.Path.cwd")
        cwd.return_value = "/not/here/mom"
        dm = manager_class()
        dm._root_directory = "/right/here/mom"
        assert dm.root_directory() == "/right/here/mom"
        cwd.assert_not_called()

    @staticmethod
    def test_root_directory_refresh(mocker, manager_class):
        cwd = mocker.patch("pathlib.Path.cwd")
        cwd.return_value = "/right/here/mom"
        dm = manager_class()
        dm._root_directory = "/not/here/mom"
        assert dm.root_directory(refresh=True) == "/right/here/mom"
        cwd.assert_called_once_with()

    @staticmethod
    def test_local_input_root(manager_class):
        dm = manager_class()
        dm._root_directory = pathlib.Path("/the/root")
        assert dm.local_input_root == pathlib.Path("/the/root/datasets")

    @staticmethod
    def test_output_root(manager_class):
        dm = manager_class()
        dm._root_directory = pathlib.Path("/the/root")
        assert dm.output_root == pathlib.Path("/the/root/climate")

    @staticmethod
    def test_zarr_json_path(manager_class):
        dm = manager_class()
        dm._root_directory = pathlib.Path("/the/root")
        assert dm.zarr_json_path() == pathlib.Path("/the/root/datasets/merged_zarr_jsons/DummyManager_zarr.json")

    @staticmethod
    def test_key(manager_class):
        dm = manager_class()
        assert dm.key() == "DummyManager-daily"

    @staticmethod
    def test_key_append_date(mocker, manager_class):
        patched_now = datetime.datetime(2010, 5, 12, 2, 42)
        patched_dt = mocker.patch("datetime.datetime")
        patched_dt.now.return_value = patched_now
        dm = manager_class()
        assert dm.key(append_date=True) == "DummyManager-daily-20100512"

    @staticmethod
    def test_relative_path(manager_class):
        dm = manager_class()
        assert dm.relative_path() == pathlib.Path(".")  # Umm, ok

    @staticmethod
    def test_local_input_path(mocker, manager_class):
        mocker.patch("pathlib.Path.mkdir")
        dm = manager_class()
        local_input_path = dm.local_input_path()
        assert local_input_path == dm.local_input_root
        local_input_path.mkdir.assert_called_once_with(parents=True, mode=0o755, exist_ok=True)

    @staticmethod
    def test_local_input_path_customized(mocker, manager_class):
        mocker.patch("pathlib.Path.mkdir")
        dm = manager_class()
        dm.custom_input_path = "/custom/input/path"
        assert dm.local_input_path() == pathlib.Path("/custom/input/path")

    @staticmethod
    def test_input_files(mocker, manager_class):
        dm = manager_class()
        dm.custom_input_path = "/"
        root = dm.local_input_path()
        filenames = [".hidden", "2dogs", "20ducks", "3cats", "hamburgers", "notthis.idx", "aardvarks", "notafile"]
        entries = [root / name for name in filenames]
        iterdir = mocker.patch("pathlib.Path.iterdir")
        iterdir.return_value = entries
        mocker.patch("pathlib.Path.is_file", lambda self: self.name != "notafile")

        expected = ["/2dogs", "/3cats", "/20ducks", "/aardvarks", "/hamburgers"]
        assert list(dm.input_files()) == [pathlib.Path(path) for path in expected]

    @staticmethod
    def test_get_folder_path_from_date(manager_class):
        dm = manager_class()
        dm._root_directory = pathlib.Path("/theroot")
        date = datetime.datetime(2010, 5, 12, 2, 42)
        assert dm.get_folder_path_from_date(date) == pathlib.Path("/theroot/climate/20100512")

    @staticmethod
    def test_get_folder_path_from_date_omit_root(manager_class):
        dm = manager_class()
        dm._root_directory = pathlib.Path("/theroot")
        date = datetime.datetime(2010, 5, 12, 2, 42)
        assert dm.get_folder_path_from_date(date, omit_root=True) == pathlib.Path("20100512")

    @staticmethod
    def test_get_folder_path_from_date_hourly(manager_class):
        dm = manager_class()
        dm._root_directory = pathlib.Path("/theroot")
        dm.time_resolution = dm.SPAN_HOURLY
        date = datetime.datetime(2010, 5, 12, 2, 42)
        assert dm.get_folder_path_from_date(date) == pathlib.Path("/theroot/climate/2010051202")

    @staticmethod
    def test_output_path(manager_class):
        dm = manager_class()
        dm._root_directory = pathlib.Path("/theroot")
        assert dm.output_path() == pathlib.Path("/theroot/climate")

    @staticmethod
    def test_output_path_omit_root(manager_class):
        dm = manager_class()
        dm._root_directory = pathlib.Path("/theroot")
        assert dm.output_path(omit_root=True) == pathlib.Path(".")

    @staticmethod
    def test_get_metadata_date_range(manager_class):
        dm = manager_class()
        dataset = Mock(attrs={"date range": ("2000010100", "2020123123")})
        dm.store = Mock(has_existing=True, dataset=Mock(return_value=dataset), spec=store.Local)
        assert dm.get_metadata_date_range() == {
            "start": datetime.datetime(2000, 1, 1, 0, 0),
            "end": datetime.datetime(2020, 12, 31, 23, 0),
        }

    @staticmethod
    def test_get_metadata_date_range_date_range_missing(manager_class):
        dm = manager_class()
        dataset = Mock(attrs={"foo": "bar"})
        dm.store = Mock(has_existing=True, dataset=Mock(return_value=dataset), spec=store.Local)
        with pytest.raises(ValueError):
            assert dm.get_metadata_date_range()

    @staticmethod
    def test_get_metadata_date_range_no_dataset(manager_class):
        dm = manager_class()
        dm.store = Mock(has_existing=False, spec=store.Local)
        with pytest.raises(ValueError):
            assert dm.get_metadata_date_range()

    @staticmethod
    def test_get_metadata_date_range_ipld(manager_class):
        dm = manager_class()
        stac_metadata = {"properties": {"date range": ("2000010100", "2020123123")}}
        dm.store = Mock(spec=store.IPLD)
        dm.load_stac_metadata = Mock(return_value=stac_metadata)
        assert dm.get_metadata_date_range() == {
            "start": datetime.datetime(2000, 1, 1, 0, 0),
            "end": datetime.datetime(2020, 12, 31, 23, 0),
        }

    @staticmethod
    def test_convert_date_range(manager_class):
        dm = manager_class()
        with pytest.deprecated_call():
            range = dm.convert_date_range(["2000/1/1", "2020/12/31"])
        assert range == (
            datetime.datetime(2000, 1, 1, 0, 0),
            datetime.datetime(2020, 12, 31, 0, 0),
        )

    @staticmethod
    def test_convert_date_range_iso(manager_class):
        dm = manager_class()
        with pytest.deprecated_call():
            range = dm.convert_date_range(["2000-01-01", "2020-12-31"])
        assert range == (
            datetime.datetime(2000, 1, 1, 0, 0),
            datetime.datetime(2020, 12, 31, 0, 0),
        )

    @staticmethod
    def test_iso_to_datetime(manager_class):
        dm = manager_class()
        with pytest.deprecated_call():
            assert dm.iso_to_datetime("2000-01-01") == datetime.datetime(2000, 1, 1, 0, 0)

    @staticmethod
    def test_numpydate_to_py(manager_class):
        dm = manager_class()
        assert dm.numpydate_to_py(np.datetime64("2000-01-01"), tz="UTC") == datetime.datetime(
            2000, 1, 1, 0, 0, tzinfo=datetime.timezone.utc
        )

    @staticmethod
    def test_today(mocker, manager_class):
        last_day_of_2020 = datetime.date(2020, 12, 31)
        date_module = mocker.patch("datetime.date")
        date_module.today.return_value = last_day_of_2020

        dm = manager_class()
        assert dm.today() == "2020-12-31"

    @staticmethod
    def test_get_date_range_from_dataset(manager_class, fake_original_dataset):
        dm = manager_class(set_key_dims=False)
        assert dm.get_date_range_from_dataset(fake_original_dataset) == (
            datetime.datetime(2021, 9, 16, 0, 0),
            datetime.datetime(2022, 1, 31, 0, 0),
        )

    @staticmethod
    def test_get_date_range_from_dataset_already_set_time_dim(manager_class, fake_original_dataset):
        dm = manager_class()
        dm.set_key_dims()
        assert dm.get_date_range_from_dataset(fake_original_dataset) == (
            datetime.datetime(2021, 9, 16, 0, 0),
            datetime.datetime(2022, 1, 31, 0, 0),
        )

    @staticmethod
    def test_get_date_range_from_dataset_single_time(manager_class, single_time_instant_dataset):
        dm = manager_class()
        assert dm.get_date_range_from_dataset(single_time_instant_dataset) == (
            datetime.datetime(2021, 9, 16, 0, 0),
            datetime.datetime(2021, 9, 16, 0, 0),
        )

    @staticmethod
    def test_get_date_range_from_file(mocker, manager_class, fake_original_dataset):
        xr = mocker.patch("gridded_etl_tools.utils.convenience.xr")
        xr.open_dataset.return_value = fake_original_dataset

        dm = manager_class()
        kwargs = {"engine": "Rolls Royce"}
        assert dm.get_date_range_from_file("some/arbitrary/path", {"foo": "bar"}, **kwargs) == (
            datetime.datetime(2021, 9, 16, 0, 0),
            datetime.datetime(2022, 1, 31, 0, 0),
        )

        xr.open_dataset.assert_called_once_with("some/arbitrary/path", backend_kwargs={"foo": "bar"}, **kwargs)

    @staticmethod
    def test_date_range_to_string(manager_class):
        date_range = (
            datetime.datetime(2021, 9, 16, 0, 0),
            datetime.datetime(2022, 1, 31, 0, 0),
        )
        dm = manager_class()
        assert dm.date_range_to_string(date_range) == ("2021091600", "2022013100")

    @staticmethod
    def test_strings_to_date_range(manager_class):
        string_dates = ("2021091600", "2022013100")
        dm = manager_class()
        assert dm.strings_to_date_range(string_dates) == (
            datetime.datetime(2021, 9, 16, 0, 0),
            datetime.datetime(2022, 1, 31, 0, 0),
        )

    @staticmethod
    def test_get_newest_file_date_range(mocker, manager_class, fake_original_dataset):
        xr = mocker.patch("gridded_etl_tools.utils.convenience.xr")
        xr.open_dataset.return_value = fake_original_dataset

        dm = manager_class()
        dm.input_files = Mock(return_value=("notthisone", "thisone"))
        assert dm.get_newest_file_date_range(engine="Rolls Royce") == (
            datetime.datetime(2021, 9, 16, 0, 0),
            datetime.datetime(2022, 1, 31, 0, 0),
        )

        xr.open_dataset.assert_called_once_with("thisone", backend_kwargs=None, engine="Rolls Royce")

    @staticmethod
    def test_next_date(manager_class, fake_original_dataset):
        dm = manager_class()
        dm.store.dataset = Mock()
        dm.store.dataset.return_value = fake_original_dataset
        dm.set_key_dims = Mock()

        dm.set_key_dims.assert_not_called()
        assert dm.next_date == datetime.datetime(2022, 2, 1)

    @staticmethod
    def test_next_date_irregular_cadence(mocker, manager_class, fake_original_dataset):
        mocker.patch(
            "gridded_etl_tools.dataset_manager.DatasetManager.update_cadence_bounds", return_value=["doesnt", "matter"]
        )
        dm = manager_class()
        dm.store.dataset = Mock()
        dm.store.dataset.return_value = fake_original_dataset

        with pytest.raises(ValueError):
            dm.next_date

    @staticmethod
    def test_next_date_no_key_dims(manager_class, fake_original_dataset):
        dm = manager_class()
        dm.store.dataset = Mock()
        dm.store.dataset.return_value = fake_original_dataset
        del dm.time_dim
        dm.set_key_dims = Mock(side_effect=dm.set_key_dims)

        assert dm.next_date == datetime.datetime(2022, 2, 1)
        dm.set_key_dims.assert_called_once_with()

    @staticmethod
    def test_get_next_date_as_date_range(manager_class, fake_original_dataset):
        dm = manager_class()
        dm.store.dataset = Mock()
        dm.store.dataset.return_value = fake_original_dataset
        dm.set_key_dims = Mock(side_effect=dm.set_key_dims)

        next_date_date_range = dm.get_next_date_as_date_range()
        assert next_date_date_range == (datetime.datetime(2022, 2, 1), datetime.datetime(2022, 2, 1))
        dm.set_key_dims.assert_not_called()

    @staticmethod
    def test_get_next_date_as_date_range_irregular_cadence(mocker, manager_class, fake_original_dataset):
        mocker.patch(
            "gridded_etl_tools.dataset_manager.DatasetManager.update_cadence_bounds", return_value=["doesnt", "matter"]
        )
        dm = manager_class()
        dm.store.dataset = Mock()
        dm.store.dataset.return_value = fake_original_dataset

        with pytest.raises(ValueError):
            dm.get_next_date_as_date_range()

    @staticmethod
    def test_bbox_coords(manager_class, fake_original_dataset):
        dm = manager_class()
        assert dm.bbox_coords(fake_original_dataset) == (100.0, 10.0, 130.0, 40.0)

    @staticmethod
    def test_json_to_bytes(manager_class):
        dm = manager_class()
        encoded = dm.json_to_bytes({"foo": "bar"})
        encoded.seek(0)  # rewind
        decoded = json.load(encoded)
        assert decoded == {"foo": "bar"}

    @staticmethod
    def test_check_if_new_data(manager_class):
        dm = manager_class()
        dm.get_newest_file_date_range = Mock(return_value=["foo", datetime.datetime(1066, 10, 14, 0, 0, 0)])
        assert dm.check_if_new_data(datetime.datetime(1066, 10, 13, 0, 0, 0)) is True
        assert dm.check_if_new_data(datetime.datetime(1066, 10, 15, 0, 0, 0)) is False

    @staticmethod
    def test_check_if_new_data_last_existing_date_unavailable(manager_class):
        dm = manager_class()
        dm.get_newest_file_date_range = Mock(return_value=["foo"])
        assert dm.check_if_new_data(datetime.datetime(1066, 10, 13, 0, 0, 0)) is False

    @staticmethod
    def test_standardize_longitudes(manager_class, manager_y_x_class, fake_original_dataset):
        # Test basic conversion from 0-360 to -180-180
        dataset = fake_original_dataset.assign_coords(longitude=[165, 175, 185, 195])
        dataset = manager_class.standardize_longitudes(dataset)
        assert np.array_equal(dataset["longitude"], np.array([-175, -165, 165, 175]))

        # Test multi-dimensional longitude coordinates
        multi_dim_lon = np.array([[165, 175], [185, 195]])
        dataset = fake_original_dataset.assign_coords(longitude=(["x", "y"], multi_dim_lon))
        dm = manager_y_x_class()
        dataset = dm.standardize_longitudes(dataset)
        expected = np.array([[-175, -165], [165, 175]])
        assert np.array_equal(dataset["longitude"], expected)

        # Test edge cases around 180/-180 boundary
        dataset = fake_original_dataset.assign_coords(longitude=[175, 180, 185, 190])
        dataset = manager_class.standardize_longitudes(dataset)
        assert np.array_equal(dataset["longitude"], np.array([-180, -175, -170, 175]))

        # Test sorting after conversion
        dataset = fake_original_dataset.assign_coords(longitude=[190, 170, 180, 160])
        dataset = manager_class.standardize_longitudes(dataset)
        assert np.array_equal(dataset["longitude"], np.array([-180, -170, 160, 170]))

    @staticmethod
    def test_get_random_coords(manager_class, fake_original_dataset):
        dm = manager_class()
        coords = dm.get_random_coords(fake_original_dataset)
        assert coords["time"] in fake_original_dataset["time"]
        assert coords["longitude"] in fake_original_dataset["longitude"]
        assert coords["latitude"] in fake_original_dataset["latitude"]

    @staticmethod
    def test_extreme_values_by_unit(manager_class):
        dm = manager_class()
        assert dm.extreme_values_by_unit == manager_class.EXTREME_VALUES_BY_UNIT
