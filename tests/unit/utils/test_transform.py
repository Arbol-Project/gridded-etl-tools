import json
import os
import pathlib

from unittest import mock

import pytest


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

    mocker.patch("gridded_etl_tools.utils.transform.Transform.kerchunkify", kerchunkify)
    mocker.patch("gridded_etl_tools.utils.transform.Transform.input_files", mock.Mock(return_value=files))

    return files


class TestTransform:
    @staticmethod
    def test_create_zarr_json(manager_class, tmp_path, mocker, input_files):
        mzz = mocker.patch("gridded_etl_tools.utils.transform.MultiZarrToZarr")
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
            postprocess=md.postprocess_kerchunk,
        )

        outfile = tmp_path / "datasets" / "merged_zarr_jsons" / "DummyManager_zarr.json"
        mzz.return_value.translate.assert_called_once_with(filename=outfile)

    @staticmethod
    def test_create_zarr_json_w_file_filter(manager_class, tmp_path, mocker, input_files):
        mzz = mocker.patch("gridded_etl_tools.utils.transform.MultiZarrToZarr")
        md = manager_class()
        md._root_directory = tmp_path

        md.create_zarr_json(file_filters=["three", "five"])

        mzz.assert_called_once_with(
            path=[str(list(input_files.keys())[i]) for i in [2, 4]],
            indicts=[list(input_files.values())[i] for i in [2, 4]],
            remote_protocol="handshake",
            remote_options={"anon": True},
            identical_dims=["x", "y"],
            concat_dims=["z", "zz"],
            preprocess=md.preprocess_kerchunk,
            postprocess=md.postprocess_kerchunk,
        )

        outfile = tmp_path / "datasets" / "merged_zarr_jsons" / "DummyManager_zarr.json"
        mzz.return_value.translate.assert_called_once_with(filename=outfile)

    @staticmethod
    def test_create_zarr_json_w_zarr_jsons_attribute(manager_class, tmp_path, mocker, input_files):
        mzz = mocker.patch("gridded_etl_tools.utils.transform.MultiZarrToZarr")
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
            postprocess=md.postprocess_kerchunk,
        )

        outfile = tmp_path / "datasets" / "merged_zarr_jsons" / "DummyManager_zarr.json"
        mzz.return_value.translate.assert_called_once_with(filename=outfile)

    @staticmethod
    def test_create_zarr_json_w_outfile_path(manager_class, tmp_path, mocker, input_files):
        mzz = mocker.patch("gridded_etl_tools.utils.transform.MultiZarrToZarr")
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
            postprocess=md.postprocess_kerchunk,
        )

        mzz.return_value.translate.assert_called_once_with(filename="put/it/here")

    @staticmethod
    def test_create_zarr_json_already_done(manager_class, tmp_path, mocker):
        folder = tmp_path / "datasets" / "merged_zarr_jsons"
        folder.mkdir(parents=True)
        outfile = folder / "DummyManager_zarr.json"
        open(outfile, "w").write("already done")

        mzz = mocker.patch("gridded_etl_tools.utils.transform.MultiZarrToZarr")
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

        mzz = mocker.patch("gridded_etl_tools.utils.transform.MultiZarrToZarr")
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
            postprocess=md.postprocess_kerchunk,
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
        fsspec = mocker.patch("gridded_etl_tools.utils.transform.fsspec")
        fs = fsspec.filesystem.return_value
        infile = fs.open.return_value.__enter__.return_value

        SingleHdf5ToZarr = mocker.patch("gridded_etl_tools.utils.transform.SingleHdf5ToZarr")
        scanned_zarr_json = SingleHdf5ToZarr.return_value.translate.return_value

        md = manager_class()
        md.file_type = "NetCDF"
        assert md.local_kerchunk("/read/from/here") is scanned_zarr_json

        fsspec.filesystem.assert_called_once_with("file")
        SingleHdf5ToZarr.assert_called_once_with(h5f=infile, url="/read/from/here", inline_threshold=5000)
        SingleHdf5ToZarr.return_value.translate.assert_called_once_with()

    @staticmethod
    def test_local_kerchunk_grib(manager_class, mocker):
        scan_grib = mocker.patch("gridded_etl_tools.utils.transform.scan_grib")
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
        fsspec = mocker.patch("gridded_etl_tools.utils.transform.fsspec")
        fs = fsspec.filesystem.return_value
        fs.open.return_value.__enter__.return_value

        SingleHdf5ToZarr = mocker.patch("gridded_etl_tools.utils.transform.SingleHdf5ToZarr")
        SingleHdf5ToZarr.side_effect = OSError

        md = manager_class()
        md.file_type = "NetCDF"
        with pytest.raises(ValueError):
            md.local_kerchunk("/read/from/here")

    @staticmethod
    def test_remote_kerchunk_netcdf(manager_class, mocker):
        s3fs = mocker.patch("gridded_etl_tools.utils.transform.s3fs")
        s3 = s3fs.S3FileSystem.return_value
        infile = s3.open.return_value.__enter__.return_value

        SingleHdf5ToZarr = mocker.patch("gridded_etl_tools.utils.transform.SingleHdf5ToZarr")
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
        scan_grib = mocker.patch("gridded_etl_tools.utils.transform.scan_grib")
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
        scan_grib = mocker.patch("gridded_etl_tools.utils.transform.scan_grib")
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
        scan_grib = mocker.patch("gridded_etl_tools.utils.transform.scan_grib")
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
        scan_grib = mocker.patch("gridded_etl_tools.utils.transform.scan_grib")
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
    def test_postprocess_kerchunk(manager_class):
        dm = manager_class()
        out_zarr = object()
        assert dm.postprocess_kerchunk(out_zarr) is out_zarr

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

        mocker.patch("gridded_etl_tools.utils.transform.Popen", DummyPopen)

        removed_files = []

        def remove(path):
            removed_files.append(path)

        os = mocker.patch("gridded_etl_tools.utils.transform.os")
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
    def test_parallel_subprocess_files_replacement_suffix(mocker, manager_class):
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

        mocker.patch("gridded_etl_tools.utils.transform.Popen", DummyPopen)

        removed_files = []

        def remove(path):  # pragma NO COVER
            removed_files.append(path)

        os = mocker.patch("gridded_etl_tools.utils.transform.os")
        os.remove = remove
        os.environ = {}

        N = 250
        dm = manager_class()
        dm.archive_original_files = mock.Mock()
        input_files = [pathlib.Path(f"fido_{n:03d}.dog") for n in range(N)]

        assert "CDO_FILE_SUFFIX" not in os.environ

        dm.parallel_subprocess_files(input_files, ["convertpet", "--cat"], ".cat")

        expected = [
            DummyPopen(["convertpet", "--cat", f"fido_{n:03d}.dog", f"fido_{n:03d}.cat"], waited=True, append=False)
            for n in range(N)
        ]
        assert subprocesses == expected
        assert "CDO_FILE_SUFFIX" not in os.environ

    @staticmethod
    def test_parallel_subprocess_files_replacement_suffix_cdo(mocker, manager_class):
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

        mocker.patch("gridded_etl_tools.utils.transform.Popen", DummyPopen)

        removed_files = []

        def remove(path):  # pragma NO COVER
            removed_files.append(path)

        os = mocker.patch("gridded_etl_tools.utils.transform.os")
        os.remove = remove
        os.environ = {}

        N = 250
        dm = manager_class()
        dm.archive_original_files = mock.Mock()
        input_files = [pathlib.Path(f"fido_{n:03d}.dog") for n in range(N)]

        assert "CDO_FILE_SUFFIX" not in os.environ

        dm.parallel_subprocess_files(input_files, ["cdo", "--cat"], ".nc4")

        expected = [
            DummyPopen(["cdo", "--cat", f"fido_{n:03d}.dog", f"fido_{n:03d}"], waited=True, append=False)
            for n in range(N)
        ]
        assert subprocesses == expected
        assert os.environ["CDO_FILE_SUFFIX"] == ".nc4"

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

        mocker.patch("gridded_etl_tools.utils.transform.Popen", DummyPopen)

        removed_files = []

        def remove(path):  # pragma NO COVER
            removed_files.append(path)

        os = mocker.patch("gridded_etl_tools.utils.transform.os")
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

        mocker.patch("gridded_etl_tools.utils.transform.Popen", DummyPopen)

        removed_files = []

        def remove(path):
            removed_files.append(path)

        os = mocker.patch("gridded_etl_tools.utils.transform.os")
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
            input_files=["a", "b", "c"],
            command_text=["cdo", "-f", "nc4c", "splitsel,1"],
            replacement_suffix=".nc4",
            keep_originals=False,
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
            input_files=["a", "b", "c"],
            command_text=["cdo", "-f", "nc4c", "splitsel,1"],
            replacement_suffix=".nc4",
            keep_originals=True,
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
            input_files=expected_files,
            command_text=["nccopy", "-k", "netCDF-4 classic model"],
            replacement_suffix=".nc4",
            keep_originals=False,
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
            input_files=expected_files,
            command_text=["nccopy", "-k", "netCDF-4 classic model"],
            replacement_suffix=".nc4",
            keep_originals=True,
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
