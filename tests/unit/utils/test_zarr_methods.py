import copy
import json
import os
import pathlib

from unittest import mock

import numpy
import pytest

from gridded_etl_tools.utils import store


@pytest.fixture
def input_files(tmp_path, mocker):
    folder = tmp_path / "datasets" / "input_files"
    folder.mkdir(parents=True)

    files = {
        folder / "one.nc": {"one": "fun"},
        folder / "two.nc4": {"two": "shoe"},
        folder / "three.grib": {"three": "free"},
        folder / "four.grib1": {"four": "soar"},
        folder / "five.grib2": {"five": "alive"},
        folder / "six.grb1": {"six": "styx"},
        folder / "seven.grb2": {"seven": "eleven"},
    }

    for file in files:
        with open(file, "w") as f:
            json.dump(files[file], f)

    def kerchunkify(self, path):
        return json.load(open(path))

    mocker.patch("gridded_etl_tools.utils.zarr_methods.Transform.kerchunkify", kerchunkify)
    mocker.patch("gridded_etl_tools.utils.zarr_methods.Transform.input_files", mock.Mock(return_value=files))

    return files


class TestTransform:
    @staticmethod
    def test_create_zarr_json(manager_class, tmp_path, mocker, input_files):
        mzz = mocker.patch("gridded_etl_tools.utils.zarr_methods.MultiZarrToZarr")
        md = manager_class()
        md._root_directory = tmp_path

        md.create_zarr_json()

        mzz.assert_called_once_with(
            path=list(map(str, input_files.keys())),
            indicts=list(input_files.values()),
            remote_protocol="handshake",
            remote_options={"anon": True},
            identical_dims=["x", "y"],
            concat_dims=["z", "zz"],
            preprocess=md.preprocess_kerchunk,
        )

        outfile = tmp_path / "datasets" / "merged_zarr_jsons" / "DummyManager_zarr.json"
        mzz.return_value.translate.assert_called_once_with(filename=outfile)

    @staticmethod
    def test_create_zarr_json_w_file_filter(manager_class, tmp_path, mocker, input_files):
        mzz = mocker.patch("gridded_etl_tools.utils.zarr_methods.MultiZarrToZarr")
        md = manager_class()
        md._root_directory = tmp_path

        md.create_zarr_json(file_filter="three")

        mzz.assert_called_once_with(
            path=[str(list(input_files.keys())[2])],
            indicts=[list(input_files.values())[2]],
            remote_protocol="handshake",
            remote_options={"anon": True},
            identical_dims=["x", "y"],
            concat_dims=["z", "zz"],
            preprocess=md.preprocess_kerchunk,
        )

        outfile = tmp_path / "datasets" / "merged_zarr_jsons" / "DummyManager_zarr.json"
        mzz.return_value.translate.assert_called_once_with(filename=outfile)

    @staticmethod
    def test_create_zarr_json_w_zarr_jsons_attribute(manager_class, tmp_path, mocker, input_files):
        mzz = mocker.patch("gridded_etl_tools.utils.zarr_methods.MultiZarrToZarr")
        md = manager_class()
        md._root_directory = tmp_path
        md.zarr_jsons = ["how's", "my", "driving?"]

        md.create_zarr_json()

        mzz.assert_called_once_with(
            path=["how's", "my", "driving?"],
            remote_protocol="handshake",
            remote_options={"anon": True},
            identical_dims=["x", "y"],
            concat_dims=["z", "zz"],
            preprocess=md.preprocess_kerchunk,
        )

        outfile = tmp_path / "datasets" / "merged_zarr_jsons" / "DummyManager_zarr.json"
        mzz.return_value.translate.assert_called_once_with(filename=outfile)

    @staticmethod
    def test_create_zarr_json_w_outfile_path(manager_class, tmp_path, mocker, input_files):
        mzz = mocker.patch("gridded_etl_tools.utils.zarr_methods.MultiZarrToZarr")
        md = manager_class()
        md._root_directory = tmp_path

        md.create_zarr_json(outfile_path="put/it/here")

        mzz.assert_called_once_with(
            path=list(map(str, input_files.keys())),
            indicts=list(input_files.values()),
            remote_protocol="handshake",
            remote_options={"anon": True},
            identical_dims=["x", "y"],
            concat_dims=["z", "zz"],
            preprocess=md.preprocess_kerchunk,
        )

        mzz.return_value.translate.assert_called_once_with(filename="put/it/here")

    @staticmethod
    def test_create_zarr_json_already_done(manager_class, tmp_path, mocker):
        folder = tmp_path / "datasets" / "merged_zarr_jsons"
        folder.mkdir(parents=True)
        outfile = folder / "DummyManager_zarr.json"
        open(outfile, "w").write("already done")

        mzz = mocker.patch("gridded_etl_tools.utils.zarr_methods.MultiZarrToZarr")
        md = manager_class()
        md._root_directory = tmp_path

        md.create_zarr_json(force_overwrite=False)
        mzz.assert_not_called()

    @staticmethod
    def test_create_zarr_json_force_overwrite(manager_class, tmp_path, mocker, input_files):
        folder = tmp_path / "datasets" / "merged_zarr_jsons"
        folder.mkdir(parents=True)
        outfile = folder / "DummyManager_zarr.json"
        open(outfile, "w").write("already done")

        mzz = mocker.patch("gridded_etl_tools.utils.zarr_methods.MultiZarrToZarr")
        md = manager_class()
        md._root_directory = tmp_path

        md.create_zarr_json()

        mzz.assert_called_once_with(
            path=list(map(str, input_files.keys())),
            indicts=list(input_files.values()),
            remote_protocol="handshake",
            remote_options={"anon": True},
            identical_dims=["x", "y"],
            concat_dims=["z", "zz"],
            preprocess=md.preprocess_kerchunk,
        )

        outfile = tmp_path / "datasets" / "merged_zarr_jsons" / "DummyManager_zarr.json"
        mzz.return_value.translate.assert_called_once_with(filename=outfile)

    @staticmethod
    def test_kerchunkify_local(manager_class):
        md = manager_class()
        md.local_kerchunk = mock.Mock()
        md.remote_kerchunk = mock.Mock()
        md.zarr_json_in_memory_to_file = mock.Mock()
        assert md.use_local_zarr_jsons is False

        assert md.kerchunkify("/home/body/over/here.json") is md.local_kerchunk.return_value
        md.local_kerchunk.assert_called_once_with("/home/body/over/here.json", 0)
        md.remote_kerchunk.assert_not_called()
        md.zarr_json_in_memory_to_file.assert_not_called()

    @staticmethod
    def test_kerchunkify_remote(manager_class):
        md = manager_class()
        md.local_kerchunk = mock.Mock()
        md.remote_kerchunk = mock.Mock()
        md.zarr_json_in_memory_to_file = mock.Mock()
        assert md.use_local_zarr_jsons is False

        assert md.kerchunkify("s3://remote/over/here.json", 42) is md.remote_kerchunk.return_value
        md.local_kerchunk.assert_not_called()
        md.remote_kerchunk.assert_called_once_with("s3://remote/over/here.json", 42)
        md.zarr_json_in_memory_to_file.assert_not_called()

    @staticmethod
    def test_kerchunkify_local_use_local_zarr_jsons(manager_class):
        md = manager_class()
        md.local_kerchunk = mock.Mock()
        md.remote_kerchunk = mock.Mock()
        md.zarr_json_in_memory_to_file = mock.Mock()
        md.use_local_zarr_jsons = True
        md.protocol = "file"

        assert md.kerchunkify("/home/body/over/here.json") is md.local_kerchunk.return_value
        md.local_kerchunk.assert_called_once_with("/home/body/over/here.json", 0)
        md.remote_kerchunk.assert_not_called()
        md.zarr_json_in_memory_to_file.assert_called_once_with(
            md.local_kerchunk.return_value, "/home/body/over/here.json"
        )

    @staticmethod
    def test_kerchunkify_remote_use_local_zarr_jsons(manager_class):
        md = manager_class()
        md.local_kerchunk = mock.Mock()
        md.remote_kerchunk = mock.Mock()
        md.zarr_json_in_memory_to_file = mock.Mock()
        md.use_local_zarr_jsons = True
        md.protocol = "s3"

        assert (
            md.kerchunkify("s3://remote/over/here.json", local_file_path="/local/here.json")
            is md.remote_kerchunk.return_value
        )
        md.local_kerchunk.assert_not_called()
        md.remote_kerchunk.assert_called_once_with("s3://remote/over/here.json", 0)
        md.zarr_json_in_memory_to_file.assert_called_once_with(md.remote_kerchunk.return_value, "/local/here.json")

    @staticmethod
    def test_kerchunkify_remote_use_local_zarr_jsons_missing_local_file_path(manager_class):
        md = manager_class()
        md.local_kerchunk = mock.Mock()
        md.remote_kerchunk = mock.Mock()
        md.zarr_json_in_memory_to_file = mock.Mock()
        md.use_local_zarr_jsons = True
        md.protocol = "s3"

        with pytest.raises(ValueError):
            md.kerchunkify("s3://remote/over/here.json")

    @staticmethod
    def test_kerchunkify_local_use_local_zarr_jsons_kerchunk_returns_list(manager_class):
        md = manager_class()
        md.local_kerchunk = mock.Mock(return_value=["abacus", "bertrand", "chloroform"])
        md.remote_kerchunk = mock.Mock()
        md.zarr_json_in_memory_to_file = mock.Mock()
        md.use_local_zarr_jsons = True
        md.protocol = "file"

        assert md.kerchunkify("/home/body/over/here.json") is md.local_kerchunk.return_value
        md.local_kerchunk.assert_called_once_with("/home/body/over/here.json", 0)
        md.remote_kerchunk.assert_not_called()
        md.zarr_json_in_memory_to_file.assert_has_calls(
            [
                mock.call("abacus", "/home/body/over/here.json"),
                mock.call("bertrand", "/home/body/over/here.json"),
                mock.call("chloroform", "/home/body/over/here.json"),
            ]
        )

    @staticmethod
    def test_local_kerchunk_netcdf(manager_class, mocker):
        fsspec = mocker.patch("gridded_etl_tools.utils.zarr_methods.fsspec")
        fs = fsspec.filesystem.return_value
        infile = fs.open.return_value.__enter__.return_value

        SingleHdf5ToZarr = mocker.patch("gridded_etl_tools.utils.zarr_methods.SingleHdf5ToZarr")
        scanned_zarr_json = SingleHdf5ToZarr.return_value.translate.return_value

        md = manager_class()
        md.file_type = "NetCDF"
        assert md.local_kerchunk("/read/from/here") is scanned_zarr_json

        fsspec.filesystem.assert_called_once_with("file")
        SingleHdf5ToZarr.assert_called_once_with(h5f=infile, url="/read/from/here", inline_threshold=5000)
        SingleHdf5ToZarr.return_value.translate.assert_called_once_with()

    @staticmethod
    def test_local_kerchunk_grib(manager_class, mocker):
        scan_grib = mocker.patch("gridded_etl_tools.utils.zarr_methods.scan_grib")
        scanned_zarr_json = mock.Mock()
        scan_grib.return_value = [scanned_zarr_json]

        md = manager_class()
        md.file_type = "GRIB"
        md.grib_filter = "iamafilter"  # TODO: Validate data manager attributes to make sure grib_filter exists
        assert md.local_kerchunk("/read/from/here") is scanned_zarr_json

        scan_grib.assert_called_once_with(url="/read/from/here", filter="iamafilter", inline_threshold=20)

    @staticmethod
    def test_local_kerchunk_invalid_file_type(manager_class):
        md = manager_class()
        with pytest.raises(ValueError):
            md.local_kerchunk("/read/from/here")

    @staticmethod
    def test_local_kerchunk_netcdf_os_error(manager_class, mocker):
        fsspec = mocker.patch("gridded_etl_tools.utils.zarr_methods.fsspec")
        fs = fsspec.filesystem.return_value
        fs.open.return_value.__enter__.return_value

        SingleHdf5ToZarr = mocker.patch("gridded_etl_tools.utils.zarr_methods.SingleHdf5ToZarr")
        SingleHdf5ToZarr.side_effect = OSError

        md = manager_class()
        md.file_type = "NetCDF"
        with pytest.raises(ValueError):
            md.local_kerchunk("/read/from/here")

    @staticmethod
    def test_remote_kerchunk_netcdf(manager_class, mocker):
        s3fs = mocker.patch("gridded_etl_tools.utils.zarr_methods.s3fs")
        s3 = s3fs.S3FileSystem.return_value
        infile = s3.open.return_value.__enter__.return_value

        SingleHdf5ToZarr = mocker.patch("gridded_etl_tools.utils.zarr_methods.SingleHdf5ToZarr")
        SingleHdf5ToZarr.return_value.translate.return_value = scanned_zarr_json = {"hi": "mom!"}

        md = manager_class()
        md.zarr_jsons = []  # TODO: See note about code smell in remote_kerchunk method
        md.file_type = "NetCDF"

        assert md.remote_kerchunk("over/here") is scanned_zarr_json
        assert md.zarr_jsons == [{"hi": "mom!"}]

        s3fs.S3FileSystem.assert_called_once_with()
        s3.open.assert_called_once_with("over/here", anon=True, default_cache_type="readahead")

        SingleHdf5ToZarr.assert_called_once_with(h5f=infile, url="over/here")
        SingleHdf5ToZarr.return_value.translate.assert_called_once_with()

    @staticmethod
    def test_remote_kerchunk_grib(manager_class, mocker):
        scan_grib = mocker.patch("gridded_etl_tools.utils.zarr_methods.scan_grib")
        scanned_zarr_json = {"hi": "mom!"}
        scan_grib.return_value = [scanned_zarr_json]

        md = manager_class()
        md.zarr_jsons = []  # TODO: See note about code smell in remote_kerchunk method
        md.file_type = "GRIB"
        md.grib_filter = "iamafilter"  # TODO: Validate data manager attributes to make sure grib_filter exists

        assert md.remote_kerchunk("over/here") is scanned_zarr_json
        assert md.zarr_jsons == [{"hi": "mom!"}]

        scan_grib.assert_called_once_with(
            url="over/here",
            storage_options={"anon": True, "default_cache_type": "readahead"},
            filter="iamafilter",
            inline_threshold=20,
        )

    @staticmethod
    def test_remote_kerchunk_grib_non_zero_scan_indices(manager_class, mocker):
        scan_grib = mocker.patch("gridded_etl_tools.utils.zarr_methods.scan_grib")
        scanned_zarr_json = {"hi": "mom!"}
        scan_grib.return_value = {42: scanned_zarr_json}

        md = manager_class()
        md.zarr_jsons = []  # TODO: See note about code smell in remote_kerchunk method
        md.file_type = "GRIB"
        md.grib_filter = "iamafilter"  # TODO: Validate data manager attributes to make sure grib_filter exists

        assert md.remote_kerchunk("over/here", scan_indices=42) is scanned_zarr_json
        assert md.zarr_jsons == [{"hi": "mom!"}]

        scan_grib.assert_called_once_with(
            url="over/here",
            storage_options={"anon": True, "default_cache_type": "readahead"},
            filter="iamafilter",
            inline_threshold=20,
        )

    @staticmethod
    def test_remote_kerchunk_unexpected_file_type(manager_class, mocker):
        md = manager_class()
        md.file_type = "Some other weird format"

        with pytest.raises(ValueError):
            md.remote_kerchunk("over/here")

    @staticmethod
    def test_remote_kerchunk_grib_remote_scan_returns_list(manager_class, mocker):
        scan_grib = mocker.patch("gridded_etl_tools.utils.zarr_methods.scan_grib")
        scanned_zarr_json = [{"hi": "mom!"}]
        scan_grib.return_value = [scanned_zarr_json]

        md = manager_class()
        md.zarr_jsons = []  # TODO: See note about code smell in remote_kerchunk method
        md.file_type = "GRIB"
        md.grib_filter = "iamafilter"  # TODO: Validate data manager attributes to make sure grib_filter exists

        assert md.remote_kerchunk("over/here") is scanned_zarr_json
        assert md.zarr_jsons == [{"hi": "mom!"}]

        scan_grib.assert_called_once_with(
            url="over/here",
            storage_options={"anon": True, "default_cache_type": "readahead"},
            filter="iamafilter",
            inline_threshold=20,
        )

    @staticmethod
    def test_remote_kerchunk_grib_remote_scan_returns_unexpected_structure(manager_class, mocker):
        scan_grib = mocker.patch("gridded_etl_tools.utils.zarr_methods.scan_grib")
        scanned_zarr_json = {"hi", "mom!"}
        scan_grib.return_value = [scanned_zarr_json]

        md = manager_class()
        md.zarr_jsons = []  # TODO: See note about code smell in remote_kerchunk method
        md.file_type = "GRIB"
        md.grib_filter = "iamafilter"  # TODO: Validate data manager attributes to make sure grib_filter exists

        with pytest.raises(ValueError):
            md.remote_kerchunk("over/here")

        scan_grib.assert_called_once_with(
            url="over/here",
            storage_options={"anon": True, "default_cache_type": "readahead"},
            filter="iamafilter",
            inline_threshold=20,
        )

    @staticmethod
    def test_export_zarr_json_in_memory_to_file(manager_class, tmpdir):
        dm = manager_class()
        local_file_path = tmpdir / "output_zarr_json.json"
        json_data = {"hi": "mom!"}
        dm.zarr_json_in_memory_to_file(json_data, local_file_path=local_file_path)
        with open(local_file_path) as f:
            assert json.load(f) == json_data

    @staticmethod
    def test_export_zarr_json_in_memory_to_file_override_local_path(manager_class, tmpdir):
        json_data = {"hi": "mom!"}
        local_file_path = tmpdir / "output_zarr_json.json"

        def file_path_from_zarr_json_attrs(scanned_zarr_json, local_file_path):
            assert scanned_zarr_json == json_data
            assert local_file_path == local_file_path

            return tmpdir / "this_other_zarr.json"

        dm = manager_class()
        dm.file_path_from_zarr_json_attrs = file_path_from_zarr_json_attrs

        dm.zarr_json_in_memory_to_file(json_data, local_file_path=local_file_path)
        assert not os.path.exists(local_file_path)
        with open(tmpdir / "this_other_zarr.json") as f:
            assert json.load(f) == json_data

    @staticmethod
    def test_preprocess_kerchunk(manager_class, example_zarr_json):
        """
        Test that the preprocess_kerchunk method successfully changes the _FillValue attribute of all arrays
        """
        orig_fill_value = json.loads(example_zarr_json["refs"]["latitude/.zarray"])["fill_value"]

        # prepare a dataset manager and preprocess a Zarr JSON
        class MyManagerClass(manager_class):
            missing_value = -8888

        dm = MyManagerClass()

        pp_zarr_json = dm.preprocess_kerchunk(example_zarr_json["refs"])

        # populate before/after fill value variables
        modified_fill_value = int(json.loads(pp_zarr_json["latitude/.zarray"])["fill_value"])

        # test that None != -8888
        assert orig_fill_value != modified_fill_value
        assert modified_fill_value == -8888

    @staticmethod
    def test_parallel_subprocess_files(mocker, manager_class):
        subprocesses = []

        class DummyPopen:
            def __init__(self, args, waited=False, append=True):
                self.args = args
                self.waited = waited
                if append:
                    subprocesses.append(self)

            def wait(self):
                self.waited = True

            def __eq__(self, other):
                return isinstance(other, DummyPopen) and other.__dict__ == self.__dict__

            def __repr__(self):  # pragma NO COVER
                return f"DummyPopen({self.args}, waited={self.waited})"

        mocker.patch("gridded_etl_tools.utils.zarr_methods.Popen", DummyPopen)

        removed_files = []

        def remove(path):
            removed_files.append(path)

        os = mocker.patch("gridded_etl_tools.utils.zarr_methods.os")
        os.remove = remove

        N = 250
        dm = manager_class()
        input_files = [pathlib.Path(f"fido_{n:03d}.dog") for n in range(N)]

        dm.parallel_subprocess_files(input_files, ["convertpet", "--cat"], ".cat")

        expected = [
            DummyPopen(["convertpet", "--cat", f"fido_{n:03d}.dog", f"fido_{n:03d}.cat"], waited=True, append=False)
            for n in range(N)
        ]
        assert subprocesses == expected

        expected = list(map(str, input_files))
        assert removed_files == expected

    @staticmethod
    def test_parallel_subprocess_files_keep_originals(mocker, manager_class):
        subprocesses = []

        class DummyPopen:
            def __init__(self, args, waited=False, append=True):
                self.args = args
                self.waited = waited
                if append:
                    subprocesses.append(self)

            def wait(self):
                self.waited = True

            def __eq__(self, other):
                return isinstance(other, DummyPopen) and other.__dict__ == self.__dict__

            def __repr__(self):  # pragma NO COVER
                return f"DummyPopen({self.args}, waited={self.waited})"

        mocker.patch("gridded_etl_tools.utils.zarr_methods.Popen", DummyPopen)

        removed_files = []

        def remove(path):  # pragma NO COVER
            removed_files.append(path)

        os = mocker.patch("gridded_etl_tools.utils.zarr_methods.os")
        os.remove = remove

        N = 250
        dm = manager_class()
        dm.archive_original_files = mock.Mock()
        input_files = [pathlib.Path(f"fido_{n:03d}.dog") for n in range(N)]

        dm.parallel_subprocess_files(input_files, ["convertpet", "--cat"], ".cat", keep_originals=True)

        expected = [
            DummyPopen(["convertpet", "--cat", f"fido_{n:03d}.dog", f"fido_{n:03d}.cat"], waited=True, append=False)
            for n in range(N)
        ]
        assert subprocesses == expected
        assert removed_files == []
        dm.archive_original_files.assert_called_once_with(input_files)

    @staticmethod
    def test_parallel_subprocess_files_invert_file_order(mocker, manager_class):
        subprocesses = []

        class DummyPopen:
            def __init__(self, args, waited=False, append=True):
                self.args = args
                self.waited = waited
                if append:
                    subprocesses.append(self)

            def wait(self):
                self.waited = True

            def __eq__(self, other):
                return isinstance(other, DummyPopen) and other.__dict__ == self.__dict__

            def __repr__(self):  # pragma NO COVER
                return f"DummyPopen({self.args}, waited={self.waited})"

        mocker.patch("gridded_etl_tools.utils.zarr_methods.Popen", DummyPopen)

        removed_files = []

        def remove(path):
            removed_files.append(path)

        os = mocker.patch("gridded_etl_tools.utils.zarr_methods.os")
        os.remove = remove

        N = 250
        dm = manager_class()
        input_files = [pathlib.Path(f"fido_{n:03d}.dog") for n in range(N)]

        dm.parallel_subprocess_files(input_files, ["convertpet", "--cat"], ".cat", invert_file_order=True)

        expected = [
            DummyPopen(["convertpet", "--cat", f"fido_{n:03d}.cat", f"fido_{n:03d}.dog"], waited=True, append=False)
            for n in range(N)
        ]
        assert subprocesses == expected

        expected = list(map(str, input_files))
        assert removed_files == expected

    @staticmethod
    def test_convert_to_lowest_common_time_denom(manager_class):
        dm = manager_class()
        dm.parallel_subprocess_files = mock.Mock()

        dm.convert_to_lowest_common_time_denom(["a", "b", "c"])
        dm.parallel_subprocess_files.assert_called_once_with(
            ["a", "b", "c"], ["cdo", "-f", "nc4", "splitsel,1"], "", False
        )

    @staticmethod
    def test_convert_to_lowest_common_time_denom_no_files(manager_class):
        dm = manager_class()
        with pytest.raises(ValueError):
            dm.convert_to_lowest_common_time_denom([])

    @staticmethod
    def test_convert_to_lowest_common_time_denom_keep_originals(manager_class):
        dm = manager_class()
        dm.parallel_subprocess_files = mock.Mock()

        dm.convert_to_lowest_common_time_denom(["a", "b", "c"], keep_originals=True)
        dm.parallel_subprocess_files.assert_called_once_with(
            ["a", "b", "c"], ["cdo", "-f", "nc4", "splitsel,1"], "", True
        )

    @staticmethod
    def test_ncs_to_nc4s(manager_class, tmpdir):
        for fname in ["one.nc", "two.nc4", "three.foo", "four.nc", "five.nc"]:
            with open(tmpdir / fname, "w") as f:
                f.write("hi mom!")

        dm = manager_class(custom_input_path=tmpdir)
        dm.parallel_subprocess_files = mock.Mock()

        dm.ncs_to_nc4s()

        expected_files = sorted([tmpdir / fname for fname in ("one.nc", "four.nc", "five.nc")])
        dm.parallel_subprocess_files.assert_called_once_with(
            expected_files, ["nccopy", "-k", "netCDF-4 classic model"], ".nc4", False
        )

    @staticmethod
    def test_ncs_to_nc4s_no_files(manager_class, tmpdir):
        dm = manager_class(custom_input_path=tmpdir)
        with pytest.raises(FileNotFoundError):
            dm.ncs_to_nc4s()

    @staticmethod
    def test_ncs_to_nc4s_keep_originals(manager_class, tmpdir):
        for fname in ["one.nc", "two.nc4", "three.foo", "four.nc", "five.nc"]:
            with open(tmpdir / fname, "w") as f:
                f.write("hi mom!")

        dm = manager_class(custom_input_path=tmpdir)
        dm.parallel_subprocess_files = mock.Mock()

        dm.ncs_to_nc4s(True)

        expected_files = sorted([tmpdir / fname for fname in ("one.nc", "four.nc", "five.nc")])
        dm.parallel_subprocess_files.assert_called_once_with(
            expected_files, ["nccopy", "-k", "netCDF-4 classic model"], ".nc4", True
        )

    @staticmethod
    def test_archive_original_files(manager_class, tmpdir):
        tmpdir = pathlib.Path(tmpdir)
        input_dir = tmpdir / "input_files"
        input_dir.mkdir()
        orig_files = [input_dir / fname for fname in ("one.nc", "two.nc4", "three.foo", "four.nc", "five.nc")]
        for file in orig_files:
            with open(file, "w") as f:
                f.write("hi mom!")

        dm = manager_class()
        dm.archive_original_files(orig_files)

        assert all([not file.exists() for file in orig_files])

        moved_files = [
            tmpdir / "one_originals" / fname for fname in ("one.nc", "two.nc4", "three.foo", "four.nc", "five.nc")
        ]
        assert all([file.exists() for file in moved_files])


class fake_vmem(dict):
    """
    Fake a vmem object with 16gb total memory using a dict
    """

    def __init__(self):
        self.total = 2**34


class TestPublish:
    @staticmethod
    def test_parse_ipld_first_time(manager_class, mocker):
        LocalCluster = mocker.patch("gridded_etl_tools.utils.zarr_methods.LocalCluster")
        Client = mocker.patch("gridded_etl_tools.utils.zarr_methods.Client")
        nullcontext = mocker.patch("gridded_etl_tools.utils.zarr_methods.nullcontext")
        mocker.patch("psutil.virtual_memory", return_value=fake_vmem())

        dm = manager_class(rebuild_requested=False)
        dm.dataset_hash = "QmHiMom!"
        dm.dask_configuration = mock.Mock()
        dm.store = mock.Mock(spec=store.IPLD, has_existing=False)
        dm.update_zarr = mock.Mock()
        dm.write_initial_zarr = mock.Mock()

        dm.parse()

        LocalCluster.assert_called_once_with(
            processes=False,
            dashboard_address="127.0.0.1:8787",
            protocol="inproc://",
            threads_per_worker=2,
            n_workers=1,
        )

        dm.dask_configuration.assert_called_once_with()
        dm.update_zarr.assert_not_called()
        dm.write_initial_zarr.assert_called_once_with()

        Client.assert_not_called()
        nullcontext.assert_called_once_with()

    @staticmethod
    def test_parse_ipld_update(manager_class, mocker):
        LocalCluster = mocker.patch("gridded_etl_tools.utils.zarr_methods.LocalCluster")
        Client = mocker.patch("gridded_etl_tools.utils.zarr_methods.Client")
        nullcontext = mocker.patch("gridded_etl_tools.utils.zarr_methods.nullcontext")
        mocker.patch("psutil.virtual_memory", return_value=fake_vmem())

        dm = manager_class(rebuild_requested=False)
        dm.dask_configuration = mock.Mock()
        dm.store = mock.Mock(spec=store.IPLD, has_existing=True)
        dm.update_zarr = mock.Mock()
        dm.write_initial_zarr = mock.Mock()

        dm.parse()

        LocalCluster.assert_called_once_with(
            processes=False,
            dashboard_address="127.0.0.1:8787",
            protocol="inproc://",
            threads_per_worker=2,
            n_workers=1,
        )

        dm.dask_configuration.assert_called_once_with()
        dm.update_zarr.assert_called_once_with()
        dm.write_initial_zarr.assert_not_called()

        Client.assert_not_called()
        nullcontext.assert_called_once_with()

    @staticmethod
    def test_parse_not_ipld_rebuild(manager_class, mocker):
        LocalCluster = mocker.patch("gridded_etl_tools.utils.zarr_methods.LocalCluster")
        cluster = LocalCluster.return_value.__enter__.return_value
        Client = mocker.patch("gridded_etl_tools.utils.zarr_methods.Client")
        nullcontext = mocker.patch("gridded_etl_tools.utils.zarr_methods.nullcontext")
        mocker.patch("psutil.virtual_memory", return_value=fake_vmem())

        dm = manager_class(rebuild_requested=True, allow_overwrite=True)
        dm.dask_configuration = mock.Mock()
        dm.store = mock.Mock(spec=store.StoreInterface, has_existing=True)
        dm.update_zarr = mock.Mock()
        dm.write_initial_zarr = mock.Mock()

        dm.parse()

        LocalCluster.assert_called_once_with(
            processes=False,
            dashboard_address="127.0.0.1:8787",
            protocol="inproc://",
            threads_per_worker=2,
            n_workers=1,
        )

        dm.dask_configuration.assert_called_once_with()
        dm.update_zarr.assert_not_called()
        dm.write_initial_zarr.assert_called_once_with()

        Client.assert_called_once_with(cluster)
        nullcontext.assert_not_called()

    @staticmethod
    def test_parse_not_ipld_rebuild_but_overwrite_not_allowed(manager_class, mocker):
        LocalCluster = mocker.patch("gridded_etl_tools.utils.zarr_methods.LocalCluster")
        cluster = LocalCluster.return_value.__enter__.return_value
        Client = mocker.patch("gridded_etl_tools.utils.zarr_methods.Client")
        nullcontext = mocker.patch("gridded_etl_tools.utils.zarr_methods.nullcontext")
        mocker.patch("psutil.virtual_memory", return_value=fake_vmem())

        dm = manager_class(rebuild_requested=True, allow_overwrite=False)
        dm.dask_configuration = mock.Mock()
        dm.store = mock.Mock(spec=store.StoreInterface, has_existing=True)
        dm.update_zarr = mock.Mock()
        dm.write_initial_zarr = mock.Mock()

        with pytest.raises(RuntimeError):
            dm.parse()

        LocalCluster.assert_called_once_with(
            processes=False,
            dashboard_address="127.0.0.1:8787",
            protocol="inproc://",
            threads_per_worker=2,
            n_workers=1,
        )

        dm.dask_configuration.assert_called_once_with()
        dm.update_zarr.assert_not_called()
        dm.write_initial_zarr.assert_not_called()

        Client.assert_called_once_with(cluster)
        nullcontext.assert_not_called()

    @staticmethod
    def test_parse_ipld_update_ctrl_c(manager_class, mocker):
        LocalCluster = mocker.patch("gridded_etl_tools.utils.zarr_methods.LocalCluster")
        Client = mocker.patch("gridded_etl_tools.utils.zarr_methods.Client")
        nullcontext = mocker.patch("gridded_etl_tools.utils.zarr_methods.nullcontext")
        mocker.patch("psutil.virtual_memory", return_value=fake_vmem())

        dm = manager_class(rebuild_requested=False)
        dm.dask_configuration = mock.Mock()
        dm.store = mock.Mock(spec=store.IPLD, has_existing=True)
        dm.update_zarr = mock.Mock(side_effect=KeyboardInterrupt)
        dm.write_initial_zarr = mock.Mock()

        dm.parse()

        LocalCluster.assert_called_once_with(
            processes=False,
            dashboard_address="127.0.0.1:8787",
            protocol="inproc://",
            threads_per_worker=2,
            n_workers=1,
        )

        dm.dask_configuration.assert_called_once_with()
        dm.update_zarr.assert_called_once_with()
        dm.write_initial_zarr.assert_not_called()

        Client.assert_not_called()
        nullcontext.assert_called_once_with()

    @staticmethod
    def test_publish_metadata(manager_class):
        dm = manager_class()
        dm.metadata = {"hi": "mom!"}
        dm.time_dims = ["what", "is", "time", "really?"]
        dm.store = mock.Mock(spec=store.StoreInterface)
        dm.populate_metadata = mock.Mock()
        dm.set_key_dims = mock.Mock()
        dm.create_root_stac_catalog = mock.Mock()
        dm.create_stac_collection = mock.Mock()
        dm.create_stac_item = mock.Mock()
        current_zarr = dm.store.dataset.return_value

        dm.publish_metadata()

        dm.populate_metadata.assert_not_called()
        dm.set_key_dims.assert_not_called()
        dm.create_root_stac_catalog.assert_called_once_with()
        dm.create_stac_collection.assert_called_once_with(current_zarr)
        dm.create_stac_item.assert_called_once_with(current_zarr)

    @staticmethod
    def test_publish_metadata_no_current_zarr(manager_class):
        dm = manager_class()
        dm.metadata = {"hi": "mom!"}
        dm.time_dims = ["what", "is", "time", "really?"]
        dm.store = mock.Mock(spec=store.StoreInterface)
        dm.populate_metadata = mock.Mock()
        dm.set_key_dims = mock.Mock()
        dm.create_root_stac_catalog = mock.Mock()
        dm.create_stac_collection = mock.Mock()
        dm.create_stac_item = mock.Mock()
        dm.store.dataset.return_value = None

        with pytest.raises(RuntimeError):
            dm.publish_metadata()

        dm.populate_metadata.assert_not_called()
        dm.set_key_dims.assert_not_called()
        dm.create_root_stac_catalog.assert_not_called()
        dm.create_stac_collection.assert_not_called()
        dm.create_stac_item.assert_not_called()

    @staticmethod
    def test_publish_metadata_populate_metadata(manager_class):
        dm = manager_class()
        dm.time_dims = ["what", "is", "time", "really?"]
        dm.store = mock.Mock(spec=store.StoreInterface)
        dm.populate_metadata = mock.Mock()
        dm.set_key_dims = mock.Mock()
        dm.create_root_stac_catalog = mock.Mock()
        dm.create_stac_collection = mock.Mock()
        dm.create_stac_item = mock.Mock()
        current_zarr = dm.store.dataset.return_value

        dm.publish_metadata()

        dm.populate_metadata.assert_called_once_with()
        dm.set_key_dims.assert_not_called()
        dm.create_root_stac_catalog.assert_called_once_with()
        dm.create_stac_collection.assert_called_once_with(current_zarr)
        dm.create_stac_item.assert_called_once_with(current_zarr)

    @staticmethod
    def test_publish_metadata_set_key_dims(manager_class):
        dm = manager_class()
        dm.metadata = {"hi": "mom!"}
        dm.store = mock.Mock(spec=store.StoreInterface)
        dm.populate_metadata = mock.Mock()
        dm.set_key_dims = mock.Mock()
        dm.create_root_stac_catalog = mock.Mock()
        dm.create_stac_collection = mock.Mock()
        dm.create_stac_item = mock.Mock()
        current_zarr = dm.store.dataset.return_value

        dm.publish_metadata()

        dm.populate_metadata.assert_not_called()
        dm.set_key_dims.assert_called_once_with()
        dm.create_root_stac_catalog.assert_called_once_with()
        dm.create_stac_collection.assert_called_once_with(current_zarr)
        dm.create_stac_item.assert_called_once_with(current_zarr)

    @staticmethod
    def test_to_zarr_ipld(manager_class, mocker):
        dm = manager_class()
        dm.pre_parse_quality_check = mock.Mock()
        dm.move_post_parse_attrs_to_dict = mock.Mock()
        dm.store = mock.Mock(spec=store.IPLD)

        dataset = mock.Mock()
        dm.to_zarr(dataset, "foo", bar="baz")

        dataset.to_zarr.assert_called_once_with("foo", bar="baz")
        dm.pre_parse_quality_check.assert_called_once_with(dataset)
        dm.store.write_metadata_only.assert_not_called()
        dm.move_post_parse_attrs_to_dict.assert_not_called()

    @staticmethod
    def test_to_zarr_ipld_dry_run(manager_class, mocker):
        dm = manager_class()
        dm.pre_parse_quality_check = mock.Mock()
        dm.move_post_parse_attrs_to_dict = mock.Mock()
        dm.store = mock.Mock(spec=store.IPLD)
        dm.dry_run = True

        dataset = mock.Mock()
        dm.to_zarr(dataset, "foo", bar="baz")

        dataset.to_zarr.assert_not_called()
        dm.pre_parse_quality_check.assert_called_once_with(dataset)
        dm.store.write_metadata_only.assert_not_called()
        dm.move_post_parse_attrs_to_dict.assert_not_called()

    @staticmethod
    def test_to_zarr_not_ipld(manager_class, mocker):
        dm = manager_class()
        dm.pre_parse_quality_check = mock.Mock()
        dm.move_post_parse_attrs_to_dict = mock.Mock()
        dm.move_post_parse_attrs_to_dict.return_value = (post_dataset, post_parse_attrs) = (mock.Mock(), mock.Mock())
        dm.store = mock.Mock(spec=store.StoreInterface)

        dataset = mock.Mock()
        dataset.get.return_value = "is it?"
        dm.to_zarr(dataset, "foo", bar="baz")

        post_dataset.to_zarr.assert_called_once_with("foo", bar="baz")
        dataset.get.assert_called_once_with("update_is_append_only")
        dm.pre_parse_quality_check.assert_called_once_with(dataset)
        dm.store.write_metadata_only.assert_has_calls(
            [
                mock.call(update_attrs={"update_in_progress": True, "update_is_append_only": "is it?"}),
                mock.call(update_attrs=post_parse_attrs),
            ]
        )
        dm.move_post_parse_attrs_to_dict.assert_called_once_with(dataset=dataset)

    @staticmethod
    def test_to_zarr_integration(manager_class, fake_original_dataset, tmpdir):
        """
        Integration test that calls to `to_zarr` correctly run three times, updating relevant metadata fields to show a
        parse is underway.

        Test that metadata fields for date ranges, etc. are only populated to a datset *after* a successful parse
        """
        dm = manager_class()
        dm.update_attributes = ["date range", "update_previous_end_date", "another attribute"]
        pre_update_dict = {
            "date range": ["2000010100", "2020123123"],
            "update_date_range": ["202012293", "2020123123"],
            "update_previous_end_date": "2020123023",
            "update_in_progress": False,
            "attribute relevant to updates": 1,
            "another attribute": True,
        }
        post_update_dict = {
            "date range": ["2000010100", "2021010523"],
            "update_previous_end_date": "2020123123",
            "update_in_progress": False,
            "another attribute": True,
        }

        # Mock datasets
        dataset = copy.deepcopy(fake_original_dataset)
        dataset.attrs.update(**pre_update_dict)
        dm.custom_output_path = tmpdir / "to_zarr_dataset.zarr"
        dataset.to_zarr(dm.custom_output_path)  # write out local file to test updates on

        # Mock functions
        dm.pre_parse_quality_check = mock.Mock()

        # Tests
        for key in pre_update_dict.keys():
            assert dm.store.dataset().attrs[key] == pre_update_dict[key]

        dataset.attrs.update(**post_update_dict)
        dm.to_zarr(dataset, dm.store.mapper(), append_dim=dm.time_dim)

        for key in post_update_dict.keys():
            assert dm.store.dataset().attrs[key] == post_update_dict[key]

        dm.pre_parse_quality_check.assert_called_once_with(dataset)

    @staticmethod
    def test_move_post_parse_attrs_to_dict(manager_class, fake_original_dataset):
        dm = manager_class()
        dm.update_attributes = dm.update_attributes + ["some irrelevant attribute"]
        fake_original_dataset.attrs = {
            "date range": ["2000010100", "2020123123"],
            "update_date_range": ["202012293", "2020123123"],
            "update_previous_end_date": "2020123023",
            "update_in_progress": False,
        }

        dataset, update_attrs = dm.move_post_parse_attrs_to_dict(fake_original_dataset)
        assert update_attrs == {
            "update_in_progress": False,
            "date range": ["2000010100", "2020123123"],
            "update_previous_end_date": "2020123023",
        }
        assert dataset is not fake_original_dataset
        assert fake_original_dataset.attrs == {
            "date range": ["2000010100", "2020123123"],
            "update_date_range": ["202012293", "2020123123"],
            "update_previous_end_date": "2020123023",
            "update_in_progress": False,
        }
        assert dataset.attrs == {
            "update_date_range": ["202012293", "2020123123"],
            "update_in_progress": False,
        }

    @staticmethod
    def test_dask_configuration_ipld(manager_class, mocker):
        dask_config = {}

        def dask_config_set(config):
            dask_config.update(config)

        dask = mocker.patch("gridded_etl_tools.utils.zarr_methods.dask")
        dask.config.set = dask_config_set

        dm = manager_class()
        dm.store = mock.Mock(spec=store.IPLD)

        dm.dask_configuration()

        assert dask_config == {
            "distributed.scheduler.worker-saturation": dm.dask_scheduler_worker_saturation,
            "distributed.scheduler.worker-ttl": None,
            "distributed.worker.memory.target": dm.dask_worker_mem_target,
            "distributed.worker.memory.spill": dm.dask_worker_mem_spill,
            "distributed.worker.memory.pause": dm.dask_worker_mem_pause,
            "distributed.worker.memory.terminate": dm.dask_worker_mem_terminate,
            "scheduler": "threads",
        }

    @staticmethod
    def test_dask_configuration_not_ipld(manager_class, mocker):
        dask_config = {}

        def dask_config_set(config):
            dask_config.update(config)

        dask = mocker.patch("gridded_etl_tools.utils.zarr_methods.dask")
        dask.config.set = dask_config_set

        dm = manager_class()
        dm.store = mock.Mock(spec=store.StoreInterface)

        dm.dask_configuration()

        assert dask_config == {
            "distributed.scheduler.worker-saturation": dm.dask_scheduler_worker_saturation,
            "distributed.scheduler.worker-ttl": None,
            "distributed.worker.memory.target": dm.dask_worker_mem_target,
            "distributed.worker.memory.spill": dm.dask_worker_mem_spill,
            "distributed.worker.memory.pause": dm.dask_worker_mem_pause,
            "distributed.worker.memory.terminate": dm.dask_worker_mem_terminate,
        }

    @staticmethod
    def test_pre_initial_dataset(manager_class):
        dm = manager_class()
        dm.transformed_dataset = mock.Mock()
        dm.set_key_dims = mock.Mock()
        dm.set_zarr_metadata = mock.Mock()

        dataset1 = dm.transformed_dataset.return_value
        dataset2 = dataset1.transpose.return_value
        dataset3 = dataset2.chunk.return_value
        dataset4 = dm.set_zarr_metadata.return_value

        assert dm.pre_initial_dataset() is dataset4

        dm.transformed_dataset.assert_called_once_with()
        dm.set_key_dims.assert_called_once_with()
        dataset1.transpose.assert_called_once_with(*dm.standard_dims)
        dataset2.chunk.assert_called_once_with(dm.requested_dask_chunks)
        dm.set_zarr_metadata(dataset3)

    @staticmethod
    def test_transformed_dataset(manager_class):
        dm = manager_class()
        dm.zarr_json_to_dataset = mock.Mock()
        assert dm.transformed_dataset() is dm.zarr_json_to_dataset.return_value
        dm.zarr_json_to_dataset.assert_called_once_with()

    @staticmethod
    def test_zarr_hash_to_dataset(manager_class, mocker):
        xr = mocker.patch("gridded_etl_tools.utils.zarr_methods.xr")
        dataset = xr.open_zarr.return_value

        dm = manager_class()
        dm.store = mock.Mock(spec=store.IPLD)
        mapper = dm.store.mapper.return_value

        assert dm.zarr_hash_to_dataset("QmHiMom") is dataset

        dm.store.mapper.assert_called_once_with(set_root=False)
        mapper.set_root.assert_called_once_with("QmHiMom")
        xr.open_zarr.assert_called_once_with(mapper)

    @staticmethod
    def test_zarr_json_to_dataset(manager_class, mocker):
        xr = mocker.patch("gridded_etl_tools.utils.zarr_methods.xr")
        dataset = xr.open_dataset.return_value
        dm = manager_class()
        dm.postprocess_zarr = mock.Mock()
        dm.zarr_json_path = mock.Mock(return_value=pathlib.Path("/path/to/zarr.json"))

        assert dm.zarr_json_to_dataset() is dm.postprocess_zarr.return_value
        dm.zarr_json_path.assert_called_once_with()
        xr.open_dataset.assert_called_once_with(
            "reference://",
            engine="zarr",
            chunks={},
            backend_kwargs={
                "storage_options": {
                    "fo": "/path/to/zarr.json",
                    "remote_protocol": "handshake",
                    "skip_instance_cache": True,
                    "default_cache_type": "readahead",
                },
                "consolidated": False,
            },
            decode_times=True,
        )
        dm.postprocess_zarr.assert_called_once_with(dataset)

    @staticmethod
    def test_zarr_json_to_dataset_explicit_args(manager_class, mocker):
        xr = mocker.patch("gridded_etl_tools.utils.zarr_methods.xr")
        dataset = xr.open_dataset.return_value
        dm = manager_class()
        dm.postprocess_zarr = mock.Mock()
        dm.zarr_json_path = mock.Mock(return_value=pathlib.Path("/path/to/zarr.json"))

        assert dm.zarr_json_to_dataset("/path/to/different.json", False) is dm.postprocess_zarr.return_value
        dm.zarr_json_path.assert_not_called()
        xr.open_dataset.assert_called_once_with(
            "reference://",
            engine="zarr",
            chunks={},
            backend_kwargs={
                "storage_options": {
                    "fo": "/path/to/different.json",
                    "remote_protocol": "handshake",
                    "skip_instance_cache": True,
                    "default_cache_type": "readahead",
                },
                "consolidated": False,
            },
            decode_times=False,
        )
        dm.postprocess_zarr.assert_called_once_with(dataset)

    @staticmethod
    def test_postprocess_zarr(manager_class):
        dm = manager_class()
        dataset = object()
        assert dm.postprocess_zarr(dataset) is dataset

    @staticmethod
    def test_set_key_dims(manager_class):
        dm = manager_class()

        dm.set_key_dims()
        assert dm.standard_dims == ["time", "latitude", "longitude"]
        assert dm.time_dim == "time"

    @staticmethod
    def test_set_key_dims_hindcast(manager_class):
        dm = manager_class()
        dm.hindcast = True
        dm.ensemble = True
        dm.forecast = True

        dm.set_key_dims()
        assert dm.standard_dims == [
            "hindcast_reference_time",
            "forecast_reference_offset",
            "step",
            "ensemble",
            "latitude",
            "longitude",
        ]
        assert dm.time_dim == "hindcast_reference_time"

    @staticmethod
    def test_set_key_dims_ensemble(manager_class):
        dm = manager_class()
        dm.ensemble = True
        dm.forecast = True

        dm.set_key_dims()
        assert dm.standard_dims == [
            "forecast_reference_time",
            "step",
            "ensemble",
            "latitude",
            "longitude",
        ]
        assert dm.time_dim == "forecast_reference_time"

    @staticmethod
    def test_set_key_dims_forecast(manager_class):
        dm = manager_class()
        dm.forecast = True

        dm.set_key_dims()
        assert dm.standard_dims == [
            "forecast_reference_time",
            "step",
            "latitude",
            "longitude",
        ]
        assert dm.time_dim == "forecast_reference_time"

    @staticmethod
    def test_write_initial_zarr_ipld(manager_class):
        class DummyHash:
            def __str__(self):
                return "QmHiMom"

        dm = manager_class()
        dm.pre_initial_dataset = mock.Mock()
        dm.store = mock.Mock(spec=store.IPLD)
        dm.to_zarr = mock.Mock()
        dm.dataset_hash = None

        dataset = dm.pre_initial_dataset.return_value
        mapper = dm.store.mapper.return_value
        mapper.freeze.return_value = DummyHash()

        dm.write_initial_zarr()

        dm.pre_initial_dataset.assert_called_once_with()
        dm.store.mapper.assert_called_once_with(set_root=False)
        dm.to_zarr.assert_called_once_with(dataset, mapper, consolidated=True, mode="w")
        assert dm.dataset_hash == "QmHiMom"

    @staticmethod
    def test_write_initial_zarr_not_ipld(manager_class):
        class DummyHash:
            ...

        dm = manager_class()
        dm.pre_initial_dataset = mock.Mock()
        dm.store = mock.Mock(spec=store.StoreInterface)
        dm.to_zarr = mock.Mock()
        dm.dataset_hash = None

        dataset = dm.pre_initial_dataset.return_value
        mapper = dm.store.mapper.return_value
        mapper.freeze.return_value = DummyHash()

        dm.write_initial_zarr()

        dm.pre_initial_dataset.assert_called_once_with()
        dm.store.mapper.assert_called_once_with(set_root=False)
        dm.to_zarr.assert_called_once_with(dataset, mapper, consolidated=True, mode="w")
        assert dm.dataset_hash is None

    @staticmethod
    def test_update_zarr(manager_class):
        dm = manager_class()
        dm.store = mock.Mock(spec=store.StoreInterface)
        dm.transformed_dataset = mock.Mock()
        dm.set_key_dims = mock.Mock()
        dm.update_setup = mock.Mock()
        dm.update_parse_operations = mock.Mock()

        original_dataset = dm.store.dataset.return_value
        update_dataset = dm.transformed_dataset.return_value
        dm.update_setup.return_value = (insert_times, append_times) = (object(), object())
        dm.update_zarr()

        dm.store.dataset.assert_called_once_with()
        dm.transformed_dataset.assert_called_once_with()
        dm.set_key_dims.assert_called_once_with()
        dm.update_setup.assert_called_once_with(original_dataset, update_dataset)
        dm.update_parse_operations.assert_called_once_with(
            original_dataset, update_dataset, insert_times, append_times
        )

    @staticmethod
    def test_update_setup(manager_class, fake_original_dataset, fake_complex_update_dataset):
        dm = manager_class()
        insert_times, update_times = dm.update_setup(fake_original_dataset, fake_complex_update_dataset)
        assert insert_times == [
            numpy.datetime64("2021-10-10T00:00:00.000000000"),
        ] + list(
            numpy.arange(
                numpy.datetime64("2021-10-16T00:00:00.000000000"),
                numpy.datetime64("2021-10-24T00:00:00.000000000"),
                numpy.timedelta64(1, "[D]"),
            )
        ) + [
            numpy.datetime64("2021-11-11T00:00:00.000000000"),
            numpy.datetime64("2021-12-11T00:00:00.000000000"),
        ] + list(
            numpy.arange(
                numpy.datetime64("2021-12-25T00:00:00.000000000"),
                numpy.datetime64("2022-01-06T00:00:00.000000000"),
                numpy.timedelta64(1, "[D]"),
            )
        ) + [
            numpy.datetime64("2022-01-14T00:00:00.000000000"),
        ]
        assert update_times == list(
            numpy.arange(
                numpy.datetime64("2022-02-01T00:00:00.000000000"),
                numpy.datetime64("2022-03-09T00:00:00.000000000"),
                numpy.timedelta64(1, "[D]"),
            )
        )

    @staticmethod
    def test_update_setup_no_time_dimension(manager_class, fake_original_dataset, fake_complex_update_dataset):
        update_dataset = fake_complex_update_dataset.sel(time=[numpy.datetime64("2021-10-10T00:00:00.000000000")])
        update_dataset = update_dataset.squeeze()
        assert "time" not in update_dataset.dims
        dm = manager_class()
        insert_times, update_times = dm.update_setup(fake_original_dataset, update_dataset)
        assert insert_times == [
            numpy.datetime64("2021-10-10T00:00:00.000000000"),
        ]
        assert update_times == []

    @staticmethod
    def test_update_parse_operations(manager_class, fake_original_dataset):
        original_chunks = {dim: val_tuple[0] for dim, val_tuple in fake_original_dataset.chunks.items()}
        update_dataset = object()
        insert_times = []
        append_times = [object()]

        dm = manager_class()
        dm.update_quality_check = mock.Mock()
        dm.insert_into_dataset = mock.Mock()
        dm.append_to_dataset = mock.Mock()

        dm.update_parse_operations(fake_original_dataset, update_dataset, insert_times, append_times)

        dm.update_quality_check.assert_called_once_with(fake_original_dataset, insert_times, append_times)
        dm.insert_into_dataset.assert_not_called()
        dm.append_to_dataset.assert_called_once_with(update_dataset, append_times, original_chunks)

    @staticmethod
    def test_update_parse_operations_insert_but_overwrite_not_allowed(manager_class, fake_original_dataset):
        original_chunks = {dim: val_tuple[0] for dim, val_tuple in fake_original_dataset.chunks.items()}
        update_dataset = object()
        insert_times = [object()]
        append_times = [object()]

        dm = manager_class()
        dm.update_quality_check = mock.Mock()
        dm.insert_into_dataset = mock.Mock()
        dm.append_to_dataset = mock.Mock()

        dm.update_parse_operations(fake_original_dataset, update_dataset, insert_times, append_times)

        dm.update_quality_check.assert_called_once_with(fake_original_dataset, insert_times, append_times)
        dm.insert_into_dataset.assert_not_called()
        dm.append_to_dataset.assert_called_once_with(update_dataset, append_times, original_chunks)

    @staticmethod
    def test_update_parse_operations_insert(manager_class, fake_original_dataset):
        original_chunks = {dim: val_tuple[0] for dim, val_tuple in fake_original_dataset.chunks.items()}
        update_dataset = object()
        insert_times = [object()]
        append_times = []

        dm = manager_class()
        dm.overwrite_allowed = True
        dm.update_quality_check = mock.Mock()
        dm.insert_into_dataset = mock.Mock()
        dm.append_to_dataset = mock.Mock()

        dm.update_parse_operations(fake_original_dataset, update_dataset, insert_times, append_times)

        dm.update_quality_check.assert_called_once_with(fake_original_dataset, insert_times, append_times)
        dm.insert_into_dataset.assert_called_once_with(
            fake_original_dataset, update_dataset, insert_times, original_chunks
        )
        dm.append_to_dataset.assert_not_called()

    @staticmethod
    def test_insert_into_dataset_ipld(manager_class):
        dm = manager_class()
        dm.dataset_hash = "browns"
        dm.store = mock.Mock(spec=store.IPLD)
        dm.prep_update_dataset = mock.Mock()
        dm.calculate_update_time_ranges = mock.Mock()
        dm.to_zarr = mock.Mock()

        original_dataset = object()
        update_dataset = object()
        insert_times = object()
        original_chunks = object()

        slice1 = mock.Mock()
        slice2 = mock.Mock()
        insert_dataset = dm.prep_update_dataset.return_value = mock.MagicMock()
        insert_dataset.attrs = {}
        insert_dataset.sel.side_effect = [slice1, slice2]

        dm.calculate_update_time_ranges.return_value = (
            (("breakfast", "second breakfast"), ("dusk", "dawn")),
            (("the shire", "mordor"), ("vegas", "atlantic city")),
        )

        mapper = dm.store.mapper.return_value
        mapper.freeze.return_value = 42

        dm.insert_into_dataset(original_dataset, update_dataset, insert_times, original_chunks)

        dm.store.mapper.assert_called_once_with()
        dm.prep_update_dataset.assert_called_once_with(update_dataset, insert_times, original_chunks)
        dm.calculate_update_time_ranges.assert_called_once_with(original_dataset, insert_dataset)

        insert_dataset.sel.assert_has_calls(
            [mock.call(time=slice("breakfast", "second breakfast")), mock.call(time=slice("dusk", "dawn"))]
        )
        dm.to_zarr.assert_has_calls(
            [
                mock.call(slice1.drop.return_value, mapper, region={"time": slice("the shire", "mordor")}),
                mock.call(slice2.drop.return_value, mapper, region={"time": slice("vegas", "atlantic city")}),
            ]
        )

        assert insert_dataset.attrs == {"update_is_append_only": False}
        assert dm.dataset_hash == "42"

    @staticmethod
    def test_insert_into_dataset_not_ipld_dry_run(manager_class):
        dm = manager_class()
        dm.dataset_hash = "browns"
        dm.dry_run = True
        dm.store = mock.Mock(spec=store.StoreInterface)
        dm.prep_update_dataset = mock.Mock()
        dm.calculate_update_time_ranges = mock.Mock()
        dm.to_zarr = mock.Mock()

        original_dataset = object()
        update_dataset = object()
        insert_times = object()
        original_chunks = object()

        slice1 = mock.Mock()
        slice2 = mock.Mock()
        insert_dataset = dm.prep_update_dataset.return_value = mock.MagicMock()
        insert_dataset.attrs = {}
        insert_dataset.sel.side_effect = [slice1, slice2]

        dm.calculate_update_time_ranges.return_value = (
            (("breakfast", "second breakfast"), ("dusk", "dawn")),
            (("the shire", "mordor"), ("vegas", "atlantic city")),
        )

        mapper = dm.store.mapper.return_value
        mapper.freeze.return_value = 42

        dm.insert_into_dataset(original_dataset, update_dataset, insert_times, original_chunks)

        dm.store.mapper.assert_called_once_with()
        dm.prep_update_dataset.assert_called_once_with(update_dataset, insert_times, original_chunks)
        dm.calculate_update_time_ranges.assert_called_once_with(original_dataset, insert_dataset)

        insert_dataset.sel.assert_has_calls(
            [mock.call(time=slice("breakfast", "second breakfast")), mock.call(time=slice("dusk", "dawn"))]
        )
        dm.to_zarr.assert_has_calls(
            [
                mock.call(slice1.drop.return_value, mapper, region={"time": slice("the shire", "mordor")}),
                mock.call(slice2.drop.return_value, mapper, region={"time": slice("vegas", "atlantic city")}),
            ]
        )

        assert insert_dataset.attrs == {"update_is_append_only": False}
        assert dm.dataset_hash == "browns"

    @staticmethod
    def test_append_to_dataset_ipld(manager_class):
        dm = manager_class()
        dm.dataset_hash = "browns"
        dm.store = mock.Mock(spec=store.IPLD)
        dm.prep_update_dataset = mock.Mock()
        dm.calculate_update_time_ranges = mock.Mock()
        dm.to_zarr = mock.Mock()

        update_dataset = object()
        insert_times = object()
        original_chunks = object()

        append_dataset = dm.prep_update_dataset.return_value = mock.MagicMock()
        append_dataset.attrs = {}

        mapper = dm.store.mapper.return_value
        mapper.freeze.return_value = 42

        dm.append_to_dataset(update_dataset, insert_times, original_chunks)

        dm.store.mapper.assert_called_once_with()
        dm.prep_update_dataset.assert_called_once_with(update_dataset, insert_times, original_chunks)

        dm.to_zarr.assert_called_once_with(append_dataset, mapper, consolidated=True, append_dim="time")

        assert append_dataset.attrs == {"update_is_append_only": True}
        assert dm.dataset_hash == "42"

    @staticmethod
    def test_append_to_dataset_not_ipld_dry_run(manager_class):
        dm = manager_class()
        dm.dataset_hash = "browns"
        dm.dry_run = True
        dm.store = mock.Mock(spec=store.StoreInterface)
        dm.prep_update_dataset = mock.Mock()
        dm.calculate_update_time_ranges = mock.Mock()
        dm.to_zarr = mock.Mock()

        update_dataset = object()
        insert_times = object()
        original_chunks = object()

        append_dataset = dm.prep_update_dataset.return_value = mock.MagicMock()
        append_dataset.attrs = {}

        mapper = dm.store.mapper.return_value
        mapper.freeze.return_value = 42

        dm.append_to_dataset(update_dataset, insert_times, original_chunks)

        dm.store.mapper.assert_called_once_with()
        dm.prep_update_dataset.assert_called_once_with(update_dataset, insert_times, original_chunks)

        dm.to_zarr.assert_called_once_with(append_dataset, mapper, consolidated=True, append_dim="time")

        assert append_dataset.attrs == {"update_is_append_only": True}
        assert dm.dataset_hash == "browns"

    @staticmethod
    def test_prep_update_dataset(manager_class, fake_complex_update_dataset):
        # Give the transpose call in prep_update_dataset something to do
        dataset = fake_complex_update_dataset.transpose("longitude", "latitude", "time")
        chunks = {"time": 5}
        time_values = numpy.arange(
            numpy.datetime64("2022-02-01T00:00:00.000000000"),
            numpy.datetime64("2022-03-09T00:00:00.000000000"),
            numpy.timedelta64(1, "[D]"),
        )
        dm = manager_class()
        dm.set_zarr_metadata = lambda x: x

        assert len(dataset.time) > len(time_values)

        dataset = dm.prep_update_dataset(dataset, time_values, chunks)

        assert numpy.array_equal(dataset.time, time_values)
        dataset.chunks["time"][0] == 5
        dataset.chunks["latitude"][0] == 4
        dataset.chunks["longitude"][0] == 4

        # TODO: I'd like to be able to verify order of coordinates after transpose, but can't figure out how get
        # coordinate order programatically in xarray. Ideally we'd have something like:
        #
        #     assert dataset.coords_in_order == ["time", "latitude", "longitude"]

    @staticmethod
    def test_prep_update_dataset_no_time_dimension(manager_class, fake_complex_update_dataset):
        # Give the transpose call in prep_update_dataset something to do
        dataset = fake_complex_update_dataset.transpose("longitude", "latitude", "time")
        dataset = dataset.sel(time=[numpy.datetime64("2022-02-01T00:00:00.000000000")]).squeeze()
        chunks = {"time": 5}
        time_values = numpy.arange(
            numpy.datetime64("2022-02-01T00:00:00.000000000"),
            numpy.datetime64("2022-03-09T00:00:00.000000000"),
            numpy.timedelta64(1, "[D]"),
        )
        dm = manager_class()
        dm.set_zarr_metadata = lambda x: x

        assert "time" not in dataset.dims

        dataset = dm.prep_update_dataset(dataset, time_values, chunks)

        assert numpy.array_equal(dataset.time, time_values[:1])
        dataset.chunks["time"][0] == 5
        dataset.chunks["latitude"][0] == 4
        dataset.chunks["longitude"][0] == 4

        # TODO: I'd like to be able to verify order of coordinates after transpose, but can't figure out how get
        # coordinate order programatically in xarray. Ideally we'd have something like:
        #
        #     assert dataset.coords_in_order == ["time", "latitude", "longitude"]

    @staticmethod
    def test_calculate_update_time_ranges(
        manager_class,
        fake_original_dataset,
        fake_complex_update_dataset,
    ):
        """
        Test that the calculate_date_ranges function correctly prepares insert and append date ranges as anticipated
        """
        # prepare a dataset manager
        dm = manager_class()
        dm.set_key_dims()
        datetime_ranges, regions_indices = dm.calculate_update_time_ranges(
            fake_original_dataset, fake_complex_update_dataset
        )
        # Test that 7 distinct updates -- 6 inserts and 1 append -- have been prepared
        assert len(regions_indices) == 7
        # Test that all of the updates are of the expected sizes
        insert_range_sizes = []
        for region in regions_indices:
            index_range = region[1] - region[0]
            insert_range_sizes.append(index_range)
        assert insert_range_sizes == [1, 8, 1, 1, 12, 1, 1]
        # Test that the append is of the expected size
        append_update = datetime_ranges[-1]
        append_size = (append_update[-1] - append_update[0]).astype("timedelta64[D]")
        assert append_size == numpy.timedelta64(35, "D")
